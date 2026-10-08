"""Read the real GLM-5.3-Flash safetensors checkpoint into the `glm53.model` param layout.

Checkpoint facts (verified 2026-09-05): fp8 e4m3 tensors carry `<name>_scale_inv` float32 with 128x128
blocks; KDA projections / conv / gates / indexer / kv_b / router / mHC are bf16 or fp32. Expert tensors
are per-expert (`mlp.experts.N.{gate,up,down}_proj.weight`). All Linear weights are torch [out, in].
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

import numpy as np

PREFIX = "model.language_model."
BLOCK = 128


@lru_cache(maxsize=None)
def weight_map(model_dir: str) -> dict[str, str]:
    """model_dir may be one directory or several joined by ':' (e.g. the 4 Kaggle dataset mounts); the index is
    taken from the first dir that has it and each shard is resolved to whichever dir contains it."""
    for d in model_dir.split(":"):
        p = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)["weight_map"]
    for d in model_dir.split(":"):                    # a single-file model (e.g. the DFlash drafter): no index
        p = os.path.join(d, "model.safetensors")
        if os.path.exists(p):
            with open(p, "rb") as f:
                hdr = json.loads(f.read(int.from_bytes(f.read(8), "little")))
            return {k: "model.safetensors" for k in hdr if k != "__metadata__"}
    raise FileNotFoundError("model.safetensors.index.json not found in " + model_dir)


def resolve_shard(model_dir: str, fname: str) -> str:
    for d in model_dir.split(":"):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{fname} not found in {model_dir}")


class ShardReader:
    """Caches open safetensors handles (torch framework, so fp8 loads natively). `model_dir` may be ':'-joined."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.wm = weight_map(model_dir)
        self._open: dict[str, object] = {}

    def _handle(self, fname):
        from safetensors import safe_open
        if fname not in self._open:
            self._open[fname] = safe_open(resolve_shard(self.model_dir, fname), framework="pt")
        return self._open[fname]

    def release(self, fname: str):
        """Close a shard and drop its pages from the page cache (page cache counts toward the Kaggle cgroup)."""
        self._open.pop(fname, None)
        try:
            fd = os.open(resolve_shard(self.model_dir, fname), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except Exception:  # noqa: BLE001
            pass

    def has(self, name: str) -> bool:
        return name in self.wm

    def raw(self, name: str):
        """torch tensor as stored (bf16 / fp8 / f32)."""
        return self._handle(self.wm[name]).get_tensor(name)

    def get(self, name: str, dtype=np.float32) -> np.ndarray:
        """Dequantized numpy array. fp8 tensors are multiplied by their blockwise scale_inv."""
        import torch
        t = self.raw(name)
        if t.dtype == torch.float8_e4m3fn:
            s = self.raw(name + "_scale_inv").float().numpy()
            return dequant_fp8(t.float().numpy(), s).astype(dtype)
        return t.float().numpy().astype(dtype)

    def get_fp8_raw(self, name: str):
        """(fp8 bytes as int8-viewed uint8 array [O,I], scale_inv f32 [O/128, I/128]) for host expert tables."""
        import torch
        t = self.raw(name)
        assert t.dtype == torch.float8_e4m3fn, name
        s = self.raw(name + "_scale_inv").float().numpy()
        return t.view(torch.uint8).numpy(), s


def dequant_fp8(w: np.ndarray, scale_inv: np.ndarray, block: int = BLOCK) -> np.ndarray:
    O, I = w.shape
    s = np.repeat(np.repeat(scale_inv, block, axis=0), block, axis=1)[:O, :I]
    return w.astype(np.float32) * s


def _lin(r: ShardReader, name: str, dtype):
    return np.ascontiguousarray(r.get(name, dtype).T)


def load_top(r: ShardReader, dtype=np.float32):
    return {
        "embed": r.get(PREFIX + "embed_tokens.weight", dtype),
        "norm": r.get(PREFIX + "norm.weight", dtype),
        "lm_head": _lin(r, "lm_head.weight", dtype),
    }


def load_hc(r: ShardReader, pfx: str, site: str, dtype):
    return {"fn": _lin(r, f"{pfx}hc_{site}_fn", dtype), "base": r.get(f"{pfx}hc_{site}_base"),
            "scale": r.get(f"{pfx}hc_{site}_scale")}


def load_kda(r: ShardReader, a: str, dtype):
    return {
        "q": _lin(r, a + "q_proj.weight", dtype), "k": _lin(r, a + "k_proj.weight", dtype),
        "v": _lin(r, a + "v_proj.weight", dtype),
        "conv_q": r.get(a + "q_conv1d.weight")[:, 0, :], "conv_k": r.get(a + "k_conv1d.weight")[:, 0, :],
        "conv_v": r.get(a + "v_conv1d.weight")[:, 0, :],
        "f_a": _lin(r, a + "f_a_proj.weight", dtype), "f_b": _lin(r, a + "f_b_proj.weight", dtype),
        "dt_bias": r.get(a + "dt_bias"), "A_log": r.get(a + "A_log"),
        "b": _lin(r, a + "b_proj.weight", dtype),
        "g_a": _lin(r, a + "g_a_proj.weight", dtype), "g_b": _lin(r, a + "g_b_proj.weight", dtype),
        "o_norm": r.get(a + "o_norm.weight"), "o": _lin(r, a + "o_proj.weight", dtype),
    }


def load_mla(r: ShardReader, a: str, dtype):
    p = {
        "q_a": _lin(r, a + "q_a_proj.weight", dtype), "q_a_norm": r.get(a + "q_a_layernorm.weight"),
        "q_b": _lin(r, a + "q_b_proj.weight", dtype),
        "kv_a": _lin(r, a + "kv_a_proj_with_mqa.weight", dtype), "kv_a_norm": r.get(a + "kv_a_layernorm.weight"),
        "kv_b": _lin(r, a + "kv_b_proj.weight", dtype), "o": _lin(r, a + "o_proj.weight", dtype),
    }
    ix = a + "indexer."
    if r.has(ix + "wk.weight"):
        p["indexer"] = {
            "wq_b": _lin(r, ix + "wq_b.weight", dtype), "wk": _lin(r, ix + "wk.weight", dtype),
            "k_norm_w": r.get(ix + "k_norm.weight"), "k_norm_b": r.get(ix + "k_norm.bias"),
            "weights_proj": _lin(r, ix + "weights_proj.weight", dtype),
            "ape": r.get(ix + "index_kpool_compress_ape"), "gate": _lin(r, ix + "index_kpool_compress_gate", dtype),
        }
    return p


def load_mlp(r: ShardReader, m: str, dtype):
    return {"gate": _lin(r, m + "gate_proj.weight", dtype), "up": _lin(r, m + "up_proj.weight", dtype),
            "down": _lin(r, m + "down_proj.weight", dtype)}


def load_moe_nonexpert(r: ShardReader, m: str, dtype):
    """Router + shared expert only. Routed experts are loaded separately (host tables)."""
    return {"router_w": _lin(r, m + "gate.weight", dtype), "router_bias": r.get(m + "gate.e_score_correction_bias"),
            "shared": load_mlp(r, m + "shared_experts.", dtype)}


def load_expert(r: ShardReader, layer: int, e: int, dtype=np.float32):
    """Dequantized (gate_up [D, 2mi], down [mi, D]) for one routed expert."""
    m = f"{PREFIX}layers.{layer}.mlp.experts.{e}."
    gate = _lin(r, m + "gate_proj.weight", dtype)
    up = _lin(r, m + "up_proj.weight", dtype)
    return np.concatenate([gate, up], axis=1), _lin(r, m + "down_proj.weight", dtype)


def load_layer(r: ShardReader, i: int, layer_type: str, mlp_type: str, dtype=np.float32, with_experts=False):
    pfx = f"{PREFIX}layers.{i}."
    p = {
        "ln1": r.get(pfx + "input_layernorm.weight"), "ln2": r.get(pfx + "post_attention_layernorm.weight"),
        "hc_attn": load_hc(r, pfx, "attn", dtype), "hc_ffn": load_hc(r, pfx, "ffn", dtype),
    }
    a = pfx + "self_attn."
    p["attn"] = load_kda(r, a, dtype) if layer_type == "linear_attention" else load_mla(r, a, dtype)
    m = pfx + "mlp."
    if mlp_type == "sparse":
        p["mlp"] = load_moe_nonexpert(r, m, dtype)
        if with_experts:
            n_exp = len({k for k in r.wm if k.startswith(m + "experts.") and k.endswith("gate_proj.weight")})
            gu, dn = zip(*(load_expert(r, i, e, dtype) for e in range(n_exp)))
            p["mlp"]["gate_up"] = np.stack(gu)
            p["mlp"]["down"] = np.stack(dn)
    else:
        p["mlp"] = load_mlp(r, m, dtype)
    return p


def load_mtp_layer(r: ShardReader, i: int, dtype=np.float32):
    """The MTP (NextN) layer: a plain pre-norm MLA (+ indexer) + MoE block without hyper-connections, fed with
    eh_proj(cat(enorm(embed(next token)), hnorm(previous hidden))); `head_norm` precedes the shared lm_head.
    Routed experts come from the GGUF (Q2_K / Q3_K planes), like every other sparse layer."""
    pfx = f"{PREFIX}layers.{i}."
    return {
        "ln1": r.get(pfx + "input_layernorm.weight"), "ln2": r.get(pfx + "post_attention_layernorm.weight"),
        "attn": load_mla(r, pfx + "self_attn.", dtype), "mlp": load_moe_nonexpert(r, pfx + "mlp.", dtype),
        "enorm": r.get(pfx + "enorm.weight"), "hnorm": r.get(pfx + "hnorm.weight"),
        "eh_proj": _lin(r, pfx + "eh_proj.weight", dtype), "head_norm": r.get(pfx + "shared_head.norm.weight"),
    }


def layer_shards(model_dir: str, i: int) -> set[str]:
    return {v for k, v in weight_map(model_dir).items() if re.search(rf"layers\.{i}\.", k)}


# ----------------------------------------------------------------------------- device-side fp8 dequant
def dequant_fp8_device(w_u8, scale_inv, out_dtype=None, block: int = BLOCK):
    """JAX: fp8-e4m3 bytes viewed as uint8 [..., O, I] + scale_inv [..., O/b, I/b] -> float [..., O, I].
    Used on TPU after streaming raw expert bytes from the host (halves PCIe traffic vs bf16)."""
    import jax
    import jax.numpy as jnp
    f8 = jax.lax.bitcast_convert_type(w_u8, jnp.float8_e4m3fn)
    w = f8.astype(jnp.float32)
    O, I = w.shape[-2:]
    s = jnp.repeat(jnp.repeat(scale_inv.astype(jnp.float32), block, axis=-2), block, axis=-1)[..., :O, :I]
    out = w * s
    return out if out_dtype is None else out.astype(out_dtype)


# ----------------------------------------------------------------------------- int4 (RTN, blockwise) pack/unpack
def quant_int4_blockwise(w: np.ndarray, block: int = BLOCK):
    """float [O, I] -> (packed uint8 [O, I//2] (low nibble = even column), scale f32 [O/b, I/b]); symmetric RTN
    to [-8, 7] per block. Requires O, I multiples of `block` and I even."""
    O, I = w.shape
    blk = w.reshape(O // block, block, I // block, block)
    amax = np.abs(blk).max(axis=(1, 3), keepdims=True)
    scale = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
    q = np.clip(np.round(blk / scale), -8, 7).astype(np.int8).reshape(O, I)
    u = (q & 0xF).astype(np.uint8)
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale.reshape(O // block, I // block)


def dequant_int4_device(packed_u8, scale, out_dtype=None, block: int = BLOCK):
    """JAX: packed [..., O, I//2] uint8 + scale [..., O/b, I/b] -> float [..., O, I]."""
    import jax.numpy as jnp
    lo = (packed_u8 & 0xF).astype(jnp.int8)
    hi = (packed_u8 >> 4).astype(jnp.int8)
    lo = jnp.where(lo > 7, lo - 16, lo)
    hi = jnp.where(hi > 7, hi - 16, hi)
    q = jnp.stack([lo, hi], axis=-1).reshape(packed_u8.shape[:-1] + (packed_u8.shape[-1] * 2,)).astype(jnp.float32)
    O, I = q.shape[-2:]
    s = jnp.repeat(jnp.repeat(scale.astype(jnp.float32), block, axis=-2), block, axis=-1)[..., :O, :I]
    out = q * s
    return out if out_dtype is None else out.astype(out_dtype)


def dequant_int4_numpy(packed_u8, scale, block: int = BLOCK):
    lo = (packed_u8 & 0xF).astype(np.int8); hi = (packed_u8 >> 4).astype(np.int8)
    lo[lo > 7] -= 16; hi[hi > 7] -= 16
    q = np.stack([lo, hi], -1).reshape(packed_u8.shape[:-1] + (packed_u8.shape[-1] * 2,)).astype(np.float32)
    O, I = q.shape[-2:]
    s = np.repeat(np.repeat(scale, block, axis=-2), block, axis=-1)[..., :O, :I]
    return q * s


# ----------------------------------------------------------------------------- mmap-free shard reader
class RawShardReader(ShardReader):
    """Reads tensors with explicit pread instead of safetensors' mmap. With mmap, every touched page of every open
    shard counts toward the process RSS (and the Kaggle cgroup), which is what pushed the 2026-09-06 full build over
    the memory guard and forced int4 for half the layers. Here only our own arrays occupy memory."""

    _DT = {"F32": np.float32, "F16": np.float16, "BF16": "bf16", "F8_E4M3": "fp8", "I8": np.int8, "U8": np.uint8,
           "I32": np.int32, "I64": np.int64}

    def _header(self, fname):
        if fname not in self._open:
            path = resolve_shard(self.model_dir, fname)
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                hdr = json.loads(f.read(n))
            hdr.pop("__metadata__", None)
            self._open[fname] = (path, 8 + n, hdr)
        return self._open[fname]

    def _read(self, name):
        path, base, hdr = self._header(self.wm[name])
        info = hdr[name]
        b0, b1 = info["data_offsets"]
        buf = np.empty(b1 - b0, np.uint8)
        fd = os.open(path, os.O_RDONLY)
        try:
            view = memoryview(buf)
            off = 0
            while off < len(buf):
                k = os.preadv(fd, [view[off:]], base + b0 + off)
                if k <= 0:
                    raise IOError(f"short read on {name}")
                off += k
        finally:
            os.close(fd)
        return buf, info["dtype"], tuple(info["shape"])

    def raw(self, name):  # torch tensor for API compatibility (rarely needed)
        import torch
        buf, dt, shape = self._read(name)
        t = torch.frombuffer(buf.tobytes(), dtype={"F32": torch.float32, "BF16": torch.bfloat16,
                                                   "F8_E4M3": torch.float8_e4m3fn}[dt])
        return t.reshape(shape)

    def get(self, name, dtype=np.float32):
        buf, dt, shape = self._read(name)
        if dt == "F32":
            return buf.view(np.float32).reshape(shape).astype(dtype, copy=False)
        if dt == "BF16":
            u16 = buf.view(np.uint16).astype(np.uint32) << 16
            return u16.view(np.float32).reshape(shape).astype(dtype, copy=False)
        if dt == "F8_E4M3":
            f = _fp8_to_f32(buf).reshape(shape)
            s = self.get(name + "_scale_inv")
            return dequant_fp8(f, s).astype(dtype, copy=False)
        raise ValueError(f"unsupported dtype {dt} for {name}")

    def get_fp8_raw(self, name):
        buf, dt, shape = self._read(name)
        assert dt == "F8_E4M3", (name, dt)
        return buf.reshape(shape), self.get(name + "_scale_inv")

    def release(self, fname):
        self._open.pop(fname, None)
        try:
            fd = os.open(resolve_shard(self.model_dir, fname), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except Exception:  # noqa: BLE001
            pass


_FP8_LUT = None


def _fp8_to_f32(u8: np.ndarray) -> np.ndarray:
    """e4m3fn bytes -> float32 via a 256-entry lookup table (pure numpy)."""
    global _FP8_LUT
    if _FP8_LUT is None:
        lut = np.empty(256, np.float32)
        for i in range(256):
            s = -1.0 if i & 0x80 else 1.0
            e = (i >> 3) & 0xF
            m = i & 0x7
            if e == 0:
                v = s * (m / 8.0) * 2.0 ** -6
            elif e == 15 and m == 7:
                v = np.nan
            else:
                v = s * (1 + m / 8.0) * 2.0 ** (e - 7)
            lut[i] = v
        _FP8_LUT = lut
    return _FP8_LUT[u8]
