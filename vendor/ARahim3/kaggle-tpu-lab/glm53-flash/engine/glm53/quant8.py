"""int8 storage for the big NON-expert matrices (attention projections, shared experts, dense MLPs): per-channel
absmax scales, dequantized to bf16 at the top of each layer program (XLA fuses the convert into the matmul operand).
Frees ~1.1 GB of HBM per chip on v5e-8 (18 GB bf16 -> 9 GB). Routers stay fp32; norms/biases/1-D/small arrays untouched.

A quantized leaf is the dict {"q": int8 [in, out], "s": scale} with the scale laid out so it shards like the weight:
weights sharded on axis 0 get per-row scales [in, 1]; everything else per-column scales [1, out]."""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

MIN_SIZE = 1 << 20
SKIP_KEYS = {"router_w", "router_bias", "fn", "base", "scale", "gate_up", "down_", "embed", "lm_head"}
Q8_KEYS = frozenset({"q", "s"})


def is_q8(x):
    return isinstance(x, dict) and set(x.keys()) == Q8_KEYS


def eligible(key, a, spec):
    return (key not in SKIP_KEYS and getattr(a, "ndim", 0) == 2 and a.size >= MIN_SIZE
            and isinstance(spec, P) and a.dtype in (jnp.bfloat16, jnp.float32, np.dtype("float32"), np.dtype("bfloat16")))


def quantize_array(w, axis):
    """w [in, out] -> {"q": int8, "s": fp32}; axis 0 = per-row scales [in,1], axis 1 = per-column [1,out]."""
    w32 = jnp.asarray(w).astype(jnp.float32)
    amax = jnp.max(jnp.abs(w32), axis=1 - axis, keepdims=True)
    s = jnp.maximum(amax, 1e-12) / 127.0
    q = jnp.clip(jnp.round(w32 / s), -127, 127).astype(jnp.int8)
    return {"q": q, "s": s}


def quantize_array_host(w, axis, rows=8192):
    """`quantize_array` in NumPy on the host, chunked over rows: for the 155k x 4096 embedding / lm_head (a jnp
    version materialises a 2.5 GB f32 copy on chip 0 and OOMs a full HBM)."""
    w = np.asarray(w)
    n = w.shape[0]
    if axis == 1:                                            # per-column scales need the whole column: two passes
        amax = np.zeros((1, w.shape[1]), np.float32)
        for a in range(0, n, rows):
            amax = np.maximum(amax, np.abs(w[a:a + rows].astype(np.float32)).max(axis=0, keepdims=True))
        s = np.maximum(amax, 1e-12) / 127.0
        q = np.empty(w.shape, np.int8)
        for a in range(0, n, rows):
            q[a:a + rows] = np.clip(np.round(w[a:a + rows].astype(np.float32) / s), -127, 127).astype(np.int8)
        return {"q": q, "s": s}
    q = np.empty(w.shape, np.int8)
    s = np.empty((n, 1), np.float32)
    for a in range(0, n, rows):
        w32 = w[a:a + rows].astype(np.float32)
        sa = np.maximum(np.abs(w32).max(axis=1, keepdims=True), 1e-12) / 127.0
        s[a:a + rows] = sa
        q[a:a + rows] = np.clip(np.round(w32 / sa), -127, 127).astype(np.int8)
    return {"q": q, "s": s}


def scale_axis(spec):
    return 0 if (len(spec) >= 1 and spec[0] is not None) else 1


def quantize_layer(p, spec, fn=quantize_array):
    """Recursively quantize eligible leaves of one layer's params; returns (params, specs) with matching structure."""
    if isinstance(p, dict):
        out_p, out_s = {}, {}
        for k, v in p.items():
            sp = spec[k] if isinstance(spec, dict) else spec
            if isinstance(v, dict):
                out_p[k], out_s[k] = quantize_layer(v, sp, fn)
            elif eligible(k, v, sp):
                out_p[k] = fn(v, scale_axis(sp))
                out_s[k] = {"q": sp, "s": sp}
            else:
                out_p[k], out_s[k] = v, sp
        return out_p, out_s
    return p, spec


def expand_specs(p, spec):
    """Specs for params that are already quantized (q8 dict nodes where the spec is a single PartitionSpec)."""
    if is_q8(p) and isinstance(spec, P):
        return {"q": spec, "s": spec}
    if isinstance(p, dict) and isinstance(spec, dict):
        return {k: expand_specs(p[k], spec[k]) for k in p}
    if isinstance(p, list) and isinstance(spec, list):
        return [expand_specs(a, b) for a, b in zip(p, spec)]
    return spec


def dequant_tree(p, dtype=jnp.bfloat16):
    """Replace q8 nodes with dequantized arrays (inside a jitted program)."""
    if is_q8(p):
        return (p["q"].astype(jnp.float32) * p["s"]).astype(dtype)
    if isinstance(p, dict):
        return {k: dequant_tree(v, dtype) for k, v in p.items()}
    if isinstance(p, list):
        return [dequant_tree(v, dtype) for v in p]
    return p


def count_bytes(p):
    if is_q8(p):
        return p["q"].size + p["s"].size * 4
    if isinstance(p, dict):
        return sum(count_bytes(v) for v in p.values())
    if isinstance(p, list):
        return sum(count_bytes(v) for v in p)
    return getattr(p, "nbytes", 0)
