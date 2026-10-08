"""Tensor-parallel (TP = #devices) engine for `glm53.model` with host-resident routed experts.

Sharding (mesh axis "x"):
  * attention heads (KDA 64, MLA 64) and MLP/expert intermediate columns are split across devices;
    every row-parallel matmul output is `psum`'d via `Cfg.tp_reduce`, so activations/streams stay replicated.
  * embed rows and lm_head rows (stored transposed, [V, D]: `lm_head_t`) are vocab-sharded.
  * routed experts are NOT on device: `HostExperts.fetch` gathers each device's slice of the selected experts on the
    host (NumPy) and ships it through `jax.pure_callback` inside the jitted `shard_map` program.

Everything runs on CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=8` for tests.
"""
from __future__ import annotations

import dataclasses
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm53 import model as M

try:
    from jax import shard_map
except ImportError:  # older jax
    from jax.experimental.shard_map import shard_map

AXIS = "x"
R = P()  # replicated


# ----------------------------------------------------------------------------- device-side sampling
SAMPLE_IMPL = "approx"          # "approx": chunked exact top-k candidates + bisection top-p (no full sorts); "sort": top_k + sort


def _candidates(z_local, temperature, top_p, n_cand, impl):
    """The candidate set of `device_sample`: (gidx [N, C] global ids, lp [N, C] f32 log-weights after temperature with
    -inf outside the top-p set, greedy [N] the global argmax, gi [N] its index in gidx, temperature [1 or N, 1])."""
    impl = SAMPLE_IMPL if impl is None else impl
    N, Vl = z_local.shape
    n = lax.axis_size(AXIS)
    kc = min(max(1, n_cand // n), Vl)
    z = z_local.astype(jnp.float32)
    if impl == "sort" or kc >= Vl:
        vals, idx = lax.top_k(z, kc)                                             # [N, kc] per chip
    else:
        # the exact chunked top-k (model.top_k): v6e 2026-10-05 (j151, [4, 19360] rows, k = 256) 74 us against 122 for
        # lax.approx_max_k (recall 0.99 + the argmax forced in) and 122 for the plain lax.top_k
        vals, idx = M.top_k(z, kc)
    gidx = idx.astype(jnp.int32) + lax.axis_index(AXIS).astype(jnp.int32) * Vl
    vals = lax.all_gather(vals, AXIS, axis=1, tiled=True)                        # [N, n*kc], replicated
    gidx = lax.all_gather(gidx, AXIS, axis=1, tiled=True)
    gi = jnp.argmax(vals, axis=1)
    greedy = jnp.take_along_axis(gidx, gi[:, None], axis=1)[:, 0]
    temperature = jnp.reshape(jnp.asarray(temperature, jnp.float32), (-1, 1))     # [] or [N] -> [1 or N, 1] per row
    top_p = jnp.reshape(jnp.asarray(top_p, jnp.float32), (-1, 1))
    t = jnp.where(temperature > 0, temperature, 1.0)
    lp = vals / t
    lp = lp - jnp.max(lp, axis=1, keepdims=True)
    p = jnp.exp(lp)
    p = p / jnp.sum(p, axis=1, keepdims=True)
    if impl == "sort":
        ps = -jnp.sort(-p, axis=1)                                               # descending
        cum = jnp.cumsum(ps, axis=1)
        n_keep = jnp.minimum(jnp.sum(cum <= top_p, axis=1) + 1, ps.shape[1])    # host rule: count(cum <= top_p) + 1
        thr = jnp.take_along_axis(ps, (n_keep - 1)[:, None], axis=1)
    else:                                                                        # largest t with mass(p >= t) >= top_p
        lo = jnp.zeros((N, 1), jnp.float32)
        hi = jnp.max(p, axis=1, keepdims=True)
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            ok = jnp.sum(jnp.where(p >= mid, p, 0.0), axis=1, keepdims=True) >= top_p
            lo = jnp.where(ok, mid, lo)
            hi = jnp.where(ok, hi, mid)
        thr = lo
    return gidx, jnp.where(p >= thr, lp, -jnp.inf), greedy, gi, temperature


def device_sample(z_local, temperature, top_p, key, n_cand=2048, impl=None):
    """Inside a shard_map program: one token per row of the logits, whose vocab columns are sharded over AXIS
    (`z_local` [N, V/n], any float dtype). Candidates = the top n_cand/n logits of every chip, all-gathered (2048 on
    8 chips, the same truncation as the host sampler's partial sort); temperature and top-p in f32 over the
    candidates, then a categorical draw with `key` (identical on every chip -> the same token everywhere).
    temperature <= 0 -> the argmax (exact over the whole vocab). impl "approx" (default): `lax.approx_max_k`
    picks the candidates (recall 0.99; the exact per-chip argmax is forced into the set, so greedy stays exact and
    the mode is never missing) and the top-p set {p >= t} is found by bisection on t (24 steps) instead of a sort —
    XLA sorts are slow on the TPU (the "sort" variant cost ~2 ms per token, as much as the host sampler).
    Returns int32 [N] token ids."""
    gidx, lp, greedy, _, temperature = _candidates(z_local, temperature, top_p, n_cand, impl)
    draw = jax.random.categorical(key, lp, axis=1)
    sampled = jnp.take_along_axis(gidx, draw[:, None], axis=1)[:, 0]
    return jnp.where(temperature[:, 0] > 0, sampled, greedy).astype(jnp.int32)


def device_probs(z_local, temperature, top_p, n_cand=2048, impl=None):
    """`device_sample`'s distribution instead of a draw: (gidx [N, C] global token ids, p [N, C] f32 their
    probabilities — the truncated softmax a draw would use; one-hot at the argmax for temperature <= 0). Tokens
    outside gidx have probability 0. For speculative acceptance (`spec_accept`)."""
    gidx, lp, _, gi, temperature = _candidates(z_local, temperature, top_p, n_cand, impl)
    p = jax.nn.softmax(lp, axis=1)
    onehot = (jnp.arange(gidx.shape[1])[None, :] == gi[:, None]).astype(jnp.float32)
    return gidx, jnp.where(temperature > 0, p, onehot)


def spec_accept(gidx, p, drafts, cand, q, key, stop_ids=()):
    """Speculative acceptance by rejection sampling (Leviathan et al. 2023), replicated arrays (no collectives).
    Row b verified the block [anchor, drafts[b]]: position t's target distribution (gidx, p: [B, T, C], `device_probs`)
    predicts drafts[b, t] (t < T-1) and the drafter drew drafts[b, t] from q[b, t] over its candidates cand[b, t]
    ([B, T-1, k]). Draft t is accepted with probability min(1, p(d)/q(d)); at the first rejection the emitted token
    is drawn from norm(max(0, p - q)), after T-1 acceptances from p of the last position — so every emitted token
    is distributed exactly as plain sampling from the target. A draft in `stop_ids` ends the run: if accepted it is
    the emitted (unfed) token. With one-hot p and q (temperature 0) this is "accepted iff equal to the argmax, then
    the argmax". -> (a [B] accepted drafts, bonus [B] the token emitted after them)."""
    B, T, C = p.shape
    d = drafts.astype(jnp.int32)
    k_u, k_r = jax.random.split(key)
    p_d = jnp.sum(jnp.where(gidx[:, :-1] == d[..., None], p[:, :-1], 0.0), -1)            # [B, T-1]
    q_d = jnp.sum(jnp.where(cand == d[..., None], q, 0.0), -1)
    u = jax.random.uniform(k_u, (B, T - 1), jnp.float32)
    acc = u * q_d < p_d
    ok = acc
    for sid in stop_ids:
        ok &= d != sid
    a = jnp.cumprod(ok.astype(jnp.int32), 1).sum(1)
    q_on = jnp.sum(jnp.where(gidx[:, :-1, :, None] == cand[:, :, None, :], q[:, :, None, :], 0.0), -1)   # [B, T-1, C]
    q_on = jnp.concatenate([q_on, jnp.zeros((B, 1, C), q_on.dtype)], 1)
    r = jnp.maximum(p - q_on, 0.0)
    r = jnp.where(jnp.sum(r, -1, keepdims=True) > 1e-12, r, p)                  # (no mass left: only on rounding)
    x = jnp.take_along_axis(gidx, jax.random.categorical(k_r, jnp.log(r), axis=-1)[..., None], -1)[..., 0]   # [B, T]
    x_a = jnp.take_along_axis(x, a[:, None], 1)[:, 0]
    dpad = jnp.concatenate([d, jnp.zeros((B, 1), jnp.int32)], 1)
    accpad = jnp.concatenate([acc, jnp.zeros((B, 1), bool)], 1)
    stopped = jnp.take_along_axis(accpad, a[:, None], 1)[:, 0]                   # an accepted stop draft ended the run
    return a, jnp.where(stopped, jnp.take_along_axis(dpad, a[:, None], 1)[:, 0], x_a).astype(jnp.int32)


class DeviceSampler:
    """Host-side handle of the device sampler: temperature, top_p and the PRNG key live on the chips as REPLICATED
    arrays (placed once per mesh by `bind`); the sampling programs return the next key, so a decode step moves no
    sampling state between host and device (per-step scalar transfers cost ~2 ms on the Kaggle TPU VM)."""

    def __init__(self, temperature=1.0, top_p=1.0, seed=None):
        self.seed = int(np.random.SeedSequence(seed).generate_state(1)[0]) if seed is None else int(seed)
        self._mesh = None
        self.set(temperature, top_p)

    def set(self, temperature, top_p):
        """Scalars (every row) or one value per row of the batch ([B] sequences, `decode_rows`); re-placed on the
        chips at the next `bind` only when they changed (a batch membership change, not every step)."""
        self.temperature = np.asarray(temperature, np.float32)
        self.top_p = np.asarray(top_p, np.float32)
        self._placed = None

    def bind(self, mesh):
        """-> (temperature, top_p, key) replicated on `mesh` (placed once per mesh / per `set`)."""
        sh = NamedSharding(mesh, P())
        if self._mesh is not mesh:
            self.key = jax.device_put(jax.random.PRNGKey(self.seed), sh)
            self._mesh, self._placed = mesh, None
        if self._placed is None:
            self.t = jax.device_put(self.temperature, sh)
            self.p = jax.device_put(self.top_p, sh)
            self._placed = True
        return self.t, self.p, self.key

    def advance(self, key):
        self.key = key


# ----------------------------------------------------------------------------- specs
def layer_specs(cfg: M.Cfg, i: int, p: dict) -> dict:
    col, row = P(None, AXIS), P(AXIS, None)
    s = {"ln1": R, "ln2": R,
         "hc_attn": {k: R for k in p["hc_attn"]}, "hc_ffn": {k: R for k in p["hc_ffn"]}}   # fn / fnT, base, scale
    if cfg.layer_types[i] == "linear_attention":
        s["attn"] = {"q": col, "k": col, "v": col, "conv_q": row, "conv_k": row, "conv_v": row,
                     "f_a": R, "f_b": col, "dt_bias": P(AXIS), "A_log": P(AXIS), "b": col,
                     "g_a": R, "g_b": col, "o_norm": R, "o": row}
    else:
        s["attn"] = {"q_a": R, "q_a_norm": R, "q_b": col, "kv_a": R, "kv_a_norm": R, "kv_b": col, "o": row}
        if "indexer" in p["attn"]:
            s["attn"]["indexer"] = {k: R for k in p["attn"]["indexer"]}
    if cfg.mlp_types[i] == "sparse":
        s["mlp"] = {"router_w": R, "router_bias": R, "shared": {"gate": col, "up": col, "down": row}}
        if "gate_up" in p["mlp"]:            # on-device experts (tests only): [E, D, 2, mi] / [E, mi, D]
            s["mlp"]["gate_up"] = P(None, None, None, AXIS)
            s["mlp"]["down"] = P(None, AXIS, None)
        for k in ("gate_q", "up_q", "down_q"):   # resident codebook-quantized experts, chip-major (glm53.resident)
            if k in p["mlp"]:                    # dict of planar arrays [n_dev, E*rows_p, R] (or one array, legacy)
                s["mlp"][k] = {pk: P(AXIS) for pk in p["mlp"][k]} if isinstance(p["mlp"][k], dict) else P(AXIS)
    else:
        s["mlp"] = {"gate": col, "up": col, "down": row}
    return s


def lm_head_t(w):
    """The lm_head as the engine stores it: [V, D] (vocab rows) instead of the checkpoint's [D, V]. The TPU dot wants
    the contracted dim minor in its weight, so a [D, V] weight was copied into that layout on EVERY call (the head
    and the draft programs: 50 us each per step for the 79 MB int8 shard, v6e 2026-10-05 j152); stored transposed
    and read with `dot_t` nothing is copied. Accepts an array or a q8 dict {q, s [V]} (left alone when q already has
    V rows)."""
    from glm53 import quant8 as Q8
    if Q8.is_q8(w):                                       # the base store's {q [D, V] int8, s [1, V]} (per-column scales)
        q, sc = np.asarray(w["q"]), np.asarray(w["s"])
        if q.shape[0] == sc.size and q.shape[0] != q.shape[1]:   # already [V, D] with s [V, 1]
            return {"q": q, "s": sc.reshape(-1, 1)}
        return {"q": np.ascontiguousarray(q.T), "s": np.ascontiguousarray(sc.reshape(-1, 1))}   # -> q [V, D], s [V, 1]
    return np.ascontiguousarray(np.asarray(w).T)


def dot_t(h, w_t, preferred_element_type=jnp.float32):
    """h [..., D] x w_t [N, D] -> [..., N] f32: the dot with its weight given transposed (the MXU's natural weight
    layout: no relayout copy of the weight)."""
    return lax.dot_general(h, w_t, (((h.ndim - 1,), (1,)), ((), ())), preferred_element_type=preferred_element_type)


def param_specs(cfg: M.Cfg, params: dict) -> dict:
    return {"embed": P(AXIS, None), "norm": R, "lm_head": P(AXIS, None),      # lm_head stored [V, D] (`lm_head_t`)
            "layers": [layer_specs(cfg, i, params["layers"][i]) for i in range(cfg.n_layers)]}


def cache_specs(cfg: M.Cfg, caches: list, seq_shard: int = 1) -> list:
    """KDA caches are head-sharded; MLA/indexer caches are replicated, or sequence-sharded (axis 1) when the
    engine was built with seq_shard (glm53.model cache layout notes)."""
    out = []
    mla = P(None, AXIS) if seq_shard > 1 else R
    for i, c in enumerate(caches):
        if cfg.layer_types[i] == "linear_attention":
            out.append({"conv": P(None, None, AXIS), "state": P(None, AXIS)})
        else:
            out.append({k: (mla if k in M.ROW_KEYS else R) for k in c})       # tail buffers are replicated
    return out


def local_cfg(cfg: M.Cfg, n: int, seq_shard: bool = False, q_block: int = 128, cache_q8: bool = False) -> M.Cfg:
    assert cfg.kda_heads % n == 0 and cfg.n_heads % n == 0 and cfg.inter % n == 0 and cfg.moe_inter % n == 0
    return dataclasses.replace(cfg, kda_heads=cfg.kda_heads // n, n_heads=cfg.n_heads // n, inter=cfg.inter // n,
                               moe_inter=cfg.moe_inter // n, tp_reduce=lambda y: lax.psum(y, AXIS), tp_axis=AXIS,
                               seq_shard=n if seq_shard else 1, q_block=q_block, cache_q8=cache_q8 or cfg.cache_q8)


# ----------------------------------------------------------------------------- host experts
class HostExperts:
    """Routed expert tables kept in host RAM, pre-sliced per device.

    Layout is CHIP-MAJOR: every table is [n_dev, E, ...] so a device's expert slice is one contiguous block
    (measured on Kaggle: [E, chip] layout + strided take = 912 ms/layer; chip-major + threaded memcpy = 7 ms).
    mode "float": tables[layer] = (gate_up [n_dev, E, D, 2*ml], down [n_dev, E, ml, D]) float arrays.
    mode "fp8":   tables[layer] = (gu_bytes [n_dev, E, D, 2*ml] uint8, gu_scale [n_dev, E, D/128, 2*ml/128] f32,
                                   dn_bytes [n_dev, E, ml, D] uint8,   dn_scale [n_dev, E, ml/128, D/128] f32)
                  — raw e4m3 bytes streamed to the device and dequantized there (`checkpoint.dequant_fp8_device`).
    """

    def __init__(self, tables: dict, mode="float", out_dtype=jnp.float32):
        """mode: "float" | "fp8" | "int4", or a dict {layer: mode} for mixed tables (e.g. fp8 until host RAM ran
        out, int4 after). int4 tables: (gu_packed [n,E,D,ml] u8, gu_scale, dn_packed [n,E,ml,D/2] u8, dn_scale)."""
        self.tables = tables
        self.modes = dict(mode) if isinstance(mode, dict) else {L: mode for L in tables}
        assert all(m in ("float", "fp8", "int4") for m in self.modes.values())
        self.out_dtype = out_dtype

    @property
    def mode(self):
        ms = set(self.modes.values())
        return ms.pop() if len(ms) == 1 else "mixed"

    def mode_for(self, layer):
        return self.modes[layer]

    @classmethod
    def from_dense(cls, params: dict, cfg: M.Cfg, n_dev: int, out_dtype=jnp.float32) -> "HostExperts":
        """Split on-device style tables gate_up [E,D,2mi], down [E,mi,D] into per-device slices (float mode)."""
        tables = {}
        ml = cfg.moe_inter // n_dev
        for i in range(cfg.n_layers):
            if cfg.mlp_types[i] != "sparse":
                continue
            gu = np.asarray(params["layers"][i]["mlp"]["gate_up"])           # [E, D, 2mi]
            dn = np.asarray(params["layers"][i]["mlp"]["down"])              # [E, mi, D]
            E, D, _ = gu.shape
            tables[i] = (chip_major(split_gate_up(gu, n_dev)), chip_major(dn.reshape(E, n_dev, ml, D)))
        return cls(tables, "float", out_dtype)

    def gather(self, layer: int, idx: np.ndarray, chip: int):
        return tuple(t[chip][idx] for t in self.tables[layer])

    def make_fetch(self, cfg_local: M.Cfg, n_tokens: int):
        """Returns fetch(layer, idx) usable inside shard_map (per-device callback)."""
        from glm53.checkpoint import BLOCK, dequant_fp8_device, dequant_int4_device
        k, D, ml = cfg_local.topk, cfg_local.hidden, cfg_local.moe_inter

        def shapes_for(mode):
            if mode == "float":
                return (jax.ShapeDtypeStruct((n_tokens, k, D, 2 * ml), self.out_dtype),
                        jax.ShapeDtypeStruct((n_tokens, k, ml, D), self.out_dtype))
            if mode == "int4":
                return (jax.ShapeDtypeStruct((n_tokens, k, D, ml), jnp.uint8),
                        jax.ShapeDtypeStruct((n_tokens, k, D // BLOCK, 2 * ml // BLOCK), jnp.float32),
                        jax.ShapeDtypeStruct((n_tokens, k, ml, D // 2), jnp.uint8),
                        jax.ShapeDtypeStruct((n_tokens, k, ml // BLOCK, D // BLOCK), jnp.float32))
            return (jax.ShapeDtypeStruct((n_tokens, k, D, 2 * ml), jnp.uint8),
                    jax.ShapeDtypeStruct((n_tokens, k, D // BLOCK, 2 * ml // BLOCK), jnp.float32),
                    jax.ShapeDtypeStruct((n_tokens, k, ml, D), jnp.uint8),
                    jax.ShapeDtypeStruct((n_tokens, k, ml // BLOCK, D // BLOCK), jnp.float32))

        def fetch(layer, idx):
            chip = lax.axis_index(AXIS)
            mode = self.mode_for(layer)
            out_shapes = shapes_for(mode)

            def host_fn(idx_np, chip_np):
                outs = self.gather(layer, np.asarray(idx_np), int(chip_np))
                return tuple(np.ascontiguousarray(o, dtype=np.dtype(sh.dtype)) for o, sh in zip(outs, out_shapes))
            outs = jax.pure_callback(host_fn, out_shapes, idx, chip)
            if mode == "float":
                return outs
            gu_b, gu_s, dn_b, dn_s = outs
            if mode == "int4":
                return (dequant_int4_device(gu_b, gu_s, self.out_dtype), dequant_int4_device(dn_b, dn_s, self.out_dtype))
            return (dequant_fp8_device(gu_b, gu_s, self.out_dtype), dequant_fp8_device(dn_b, dn_s, self.out_dtype))
        return fetch


def split_gate_up(gu: np.ndarray, n_dev: int) -> np.ndarray:
    """[E, D, 2mi] (gate|up) -> [E, n_dev, D, 2*ml] with each device holding (gate_slice|up_slice)."""
    E, D, two_mi = gu.shape
    ml = two_mi // 2 // n_dev
    return np.ascontiguousarray(gu.reshape(E, D, 2, n_dev, ml).transpose(0, 3, 1, 2, 4).reshape(E, n_dev, D, 2 * ml))


def split_gate_up_scales(gs: np.ndarray, n_dev: int) -> np.ndarray:
    """Same split for blockwise scales [E, D/128, 2mi/128] -> [E, n_dev, D/128, 2*ml/128] (ml multiple of 128)."""
    return split_gate_up(gs, n_dev)


def chip_major(t: np.ndarray) -> np.ndarray:
    """[E, n_dev, ...] -> contiguous [n_dev, E, ...]."""
    return np.ascontiguousarray(np.moveaxis(t, 1, 0))


# ----------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, cfg: M.Cfg, params: dict, host_experts: HostExperts | None = None, devices=None,
                 max_len: int = 4096, expert_fetch=None, int8_nonexpert: bool = False, seq_shard: bool = False,
                 q_block: int = 128, cache_q8: bool = False):
        devices = devices if devices is not None else jax.devices()
        cfg = dataclasses.replace(cfg, cache_q8=cache_q8 or cfg.cache_q8)     # int8 latent cache (half the HBM per set)
        self.max_len = max_len
        self.expert_fetch = expert_fetch      # e.g. glm53.resident.ResidentFetch (tables live inside params)
        self.n = len(devices)
        self.mesh = Mesh(np.array(devices), (AXIS,))
        self.cfg = cfg
        # seq_shard: MLA latent + indexer caches sharded over the chips (16.9 KB/token/chip replicated -> 2.1 KB;
        # needed beyond ~32k tokens); the capacity must be a multiple of kpool * n_dev
        assert not seq_shard or max_len % (cfg.idx_kpool * self.n) == 0, (max_len, cfg.idx_kpool, self.n)
        self.lcfg = local_cfg(cfg, self.n, seq_shard, q_block, cfg.cache_q8)
        self.host_experts = host_experts
        params = dict(params)
        params["lm_head"] = lm_head_t(params["lm_head"])
        if host_experts is not None:   # drop on-device expert tables
            params["layers"] = [{**L, "mlp": {k: v for k, v in L["mlp"].items() if k not in ("gate_up", "down")}}
                                if cfg.mlp_types[i] == "sparse" else L for i, L in enumerate(params["layers"])]
        else:
            params["layers"] = [{**L, "mlp": {**L["mlp"], "gate_up": np.asarray(L["mlp"]["gate_up"]).reshape(
                cfg.n_exp, cfg.hidden, 2, cfg.moe_inter)}} if cfg.mlp_types[i] == "sparse" and "gate_up" in L["mlp"]
                else L for i, L in enumerate(params["layers"])]
        if M.mhc_active():                    # the mHC kernels read the site projections transposed (same bytes)
            params["layers"] = [{**L, "hc_attn": M.mhc_params(L["hc_attn"]), "hc_ffn": M.mhc_params(L["hc_ffn"])}
                                for L in params["layers"]]
        self.specs = param_specs(cfg, params)
        from glm53 import quant8 as Q8
        if int8_nonexpert:                    # quantize the big non-expert matrices on the host before device_put
            layers, lspecs = [], []
            for L, sp in zip(params["layers"], self.specs["layers"]):
                qp, qs = Q8.quantize_layer(L, sp, lambda w, ax: jax.tree.map(np.asarray, Q8.quantize_array(w, ax)))
                layers.append(qp); lspecs.append(qs)
            params = {**params, "layers": layers}
            self.specs = {**self.specs, "layers": lspecs}
            if int8_nonexpert == "all":           # + the vocab-sharded embedding (per-row scales) and lm_head (per-column),
                q = Q8.quantize_array_host             # quantized on the host (a device pass OOMed at 262k)
                params = {**params, "embed": q(params["embed"], 0),
                          "lm_head": params["lm_head"] if Q8.is_q8(params["lm_head"]) else q(params["lm_head"], 0)}
        self.specs = Q8.expand_specs(params, self.specs)   # params already holding q8 nodes (hot-swapped on device)
        self.params = jax.tree.map(lambda a, s: jax.device_put(a, NamedSharding(self.mesh, s)), params, self.specs,
                                   is_leaf=lambda x: isinstance(x, P))
        self._prefill = {}
        self._decode = {}

    def _cache_spec(self, i):
        if self.cfg.layer_types[i] == "linear_attention":
            return {"conv": P(None, None, AXIS), "state": P(None, AXIS)}
        mla = P(None, AXIS) if self.lcfg.seq_shard > 1 else R
        spec = {"c": mla, "pk": mla, "tk": R, "tg": R} if "indexer" in self.params["layers"][i]["attn"] else {"c": mla}
        if self.cfg.cache_q8:
            spec["cs"] = mla
        return spec

    # ---- per-device program pieces
    def _embed(self, embed_local, tokens):
        from glm53 import quant8 as Q8
        chip = lax.axis_index(AXIS)
        q8 = Q8.is_q8(embed_local)
        table = embed_local["q"] if q8 else embed_local
        rows = table.shape[0]
        local = tokens - chip * rows
        ok = (local >= 0) & (local < rows)
        idx = jnp.clip(local, 0, rows - 1)
        e = jnp.take(table, idx, axis=0)
        if q8:                                              # dequantize only the gathered rows
            e = e.astype(jnp.float32) * jnp.take(embed_local["s"], idx, axis=0)
        return lax.psum(jnp.where(ok[..., None], e, 0), AXIS)

    def _logits_local(self, lm_head_local, h):
        """This chip's vocab columns of the logits [..., V/n] (the shard `_logits` all-gathers). The weight is stored
        [V/n, D] (`lm_head_t`) and read as the dot's transposed operand."""
        from glm53 import quant8 as Q8
        if Q8.is_q8(lm_head_local):                         # int8 rows: the convert fuses into the dot, scales after
            z = dot_t(h, lm_head_local["q"].astype(h.dtype)) * lm_head_local["s"].reshape(-1)    # s [V, 1] -> [V]
            return z.astype(h.dtype)
        return dot_t(h, lm_head_local).astype(h.dtype)

    def _logits(self, lm_head_local, h):
        return lax.all_gather(self._logits_local(lm_head_local, h), AXIS, axis=-1, tiled=True)

    PREFILL_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

    def _bucket(self, T):
        for b in self.PREFILL_BUCKETS:
            if b >= T and b <= self.max_len:
                return b
        return T

    def _program(self, params, tokens, caches, pos0, use_recurrent, n_tokens, length=None):
        # pos0 / length are traced int32 scalars; caches have fixed capacity self.max_len
        fetch = None if self.host_experts is None else self.host_experts.make_fetch(self.lcfg, n_tokens)
        if self.expert_fetch is not None:
            fetch = self.expert_fetch
        if self.host_experts is None:   # local on-device tables arrive as [E, D, 2, ml] -> [E, D, 2*ml]
            params = {**params, "layers": [
                {**L, "mlp": {**L["mlp"], "gate_up": L["mlp"]["gate_up"].reshape(L["mlp"]["gate_up"].shape[0],
                                                                                L["mlp"]["gate_up"].shape[1], -1)}}
                if "gate_up" in L["mlp"] else L for L in params["layers"]]}
        from glm53 import quant8 as Q8
        params = {**params, "layers": Q8.dequant_tree(params["layers"], self.lcfg.dtype)}
        emb = self._embed(params["embed"], tokens)
        h, new_caches = M.forward(params, tokens, self.lcfg, caches=caches, pos0=pos0, use_recurrent=use_recurrent,
                                  fetch=fetch, sparse="auto", embeds=emb, cap=self.max_len, length=length)
        last = h[:, -1] if length is None else lax.dynamic_index_in_dim(h, length - 1, axis=1, keepdims=False)
        return self._logits(params["lm_head"], last), new_caches

    def _build(self, B, T, caches, use_recurrent):
        # caches may be None (prefill) — we need their specs for in/out
        def prog(params, tokens, pos0, length, caches_in):
            return self._program(params, tokens, caches_in, pos0, use_recurrent, B * T, None if use_recurrent else length)

        cs = None if caches is None else cache_specs(self.cfg, caches, self.lcfg.seq_shard)
        # discover output cache structure by abstract evaluation is awkward under shard_map; build specs from a
        # dry structure: same layout as inputs, or generated for prefill
        cs_out = [self._cache_spec(i) for i in range(self.cfg.n_layers)] if cs is None else cs
        sm = shard_map(prog, mesh=self.mesh, in_specs=(self.specs, R, R, R, cs), out_specs=(R, cs_out), check_vma=False)
        return jax.jit(sm)

    def prefill(self, tokens: np.ndarray):
        B, T = tokens.shape
        assert T <= self.max_len
        Tp = self._bucket(T)
        padded = np.zeros((B, Tp), np.int32)
        padded[:, :T] = tokens
        key = (B, Tp)
        if key not in self._prefill:
            self._prefill[key] = self._build(B, Tp, None, False)
        logits, caches = self._prefill[key](self.params, jnp.asarray(padded), jnp.int32(0), jnp.int32(T), None)
        return logits, caches, T

    def decode(self, token: np.ndarray, caches, pos: int):
        B = token.shape[0]
        key = B
        if key not in self._decode:
            self._decode[key] = self._build(B, 1, caches, True)
        logits, caches = self._decode[key](self.params, jnp.asarray(token).reshape(B, 1), jnp.int32(pos), jnp.int32(1),
                                           caches)
        return logits, caches, pos + 1

    def generate(self, tokens: np.ndarray, max_new: int, eos=(), greedy=True):
        logits, caches, pos = self.prefill(tokens)
        out = []
        for _ in range(max_new):
            nxt = int(jnp.argmax(logits[0]))
            out.append(nxt)
            if nxt in eos:
                break
            logits, caches, pos = self.decode(np.array([nxt]), caches, pos)
        return out


# ----------------------------------------------------------------------------- segmented engine (no host callbacks)
class SegmentedEngine(Engine):
    """Same sharding as `Engine`, but each layer is its own jitted shard_map program and routed-expert weights are
    gathered on the host *between* programs and shipped with `device_put` (works on TPU runtimes where host
    callbacks inside jit are unavailable). Prefill and decode use the same per-layer programs.

    `transport(arrays: tuple[np.ndarray])` -> tuple of device arrays sharded P('x') on axis 0 (per-chip slices);
    default = numpy device_put. Override for pinned-host tables."""

    def __init__(self, cfg, params, host_experts: HostExperts, devices=None, transport=None, max_len: int = 4096):
        assert host_experts is not None
        super().__init__(cfg, params, host_experts, devices, max_len)
        self.transport = transport or self._default_transport
        self._progs = {}
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=32)
        from glm53.checkpoint import BLOCK
        self.BLOCK = BLOCK

    CHUNK_BYTES = 320 * 2 ** 20   # GB-sized puts run at ~5 GB/s, ~300 MB strided chunks at ~22 GB/s (measured)

    def _default_transport(self, arrays):
        """arrays: tuple of host [n_dev, Eh, ...] -> tuple of LISTS of device arrays (chunks along axis 1),
        each sharded P('x') on axis 0. Chunks are re-joined on device inside the post program."""
        sh = NamedSharding(self.mesh, P(AXIS))
        out = []
        for a in arrays:
            per_e = a[:, :1].nbytes
            step = max(1, self.CHUNK_BYTES // max(per_e, 1))
            out.append([jax.device_put(a[:, i:i + step], sh) for i in range(0, a.shape[1], step)])
        return tuple(out)

    def _dequant(self, outs, layer):
        from glm53.checkpoint import dequant_fp8_device, dequant_int4_device
        he = self.host_experts
        mode = he.mode_for(layer)
        if mode == "float":
            return outs
        gu_b, gu_s, dn_b, dn_s = outs
        f = dequant_int4_device if mode == "int4" else dequant_fp8_device
        return f(gu_b, gu_s, he.out_dtype), f(dn_b, dn_s, he.out_dtype)

    # ---- per-layer programs (built lazily per (layer, B, T, has_cache))
    def _prog_embed(self, B, T):
        key = ("embed", B, T)
        if key not in self._progs:
            def prog(params, tokens):
                x = self._embed(params["embed"], tokens).astype(self.lcfg.dtype)
                return jnp.broadcast_to(x[:, :, None, :], (B, T, self.cfg.hc, x.shape[-1]))
            self._progs[key] = jax.jit(shard_map(prog, mesh=self.mesh, in_specs=(self.specs, R), out_specs=R,
                                                 check_vma=False))
        return self._progs[key]

    def _prog_pre(self, i, B, T, has_cache, use_recurrent):
        key = ("pre", i, B, T, has_cache, use_recurrent)
        if key not in self._progs:
            lt, mt = self.cfg.layer_types[i], self.cfg.mlp_types[i]
            cs = self._cache_spec(i)
            def prog(params, streams, pos0, length, cache):
                p = params["layers"][i]
                streams, h, post, comb, new_cache, idx, w = M.layer_pre(
                    p, streams, self.lcfg, lt, mt, cache, pos0, use_recurrent, "auto", self.max_len,
                    None if use_recurrent else length)
                if idx is None:
                    idx = jnp.zeros((B * T, 1), jnp.int32); w = jnp.zeros((B * T, 1), jnp.float32)
                return streams, h, post, comb, new_cache, idx, w
            in_specs = (self.specs, R, R, R, cs if has_cache else None)
            out_specs = (R, R, R, R, cs, R, R)
            self._progs[key] = jax.jit(shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs,
                                                 check_vma=False))
        return self._progs[key]

    def _prog_post(self, i, B, T, Eh=None):
        """Eh: None (dense layer) or (Ep, (n_chunks per table...)) — both fix the program's input structure."""
        key = ("post", i, B, T, Eh)
        if key not in self._progs:
            mt = self.cfg.mlp_types[i]
            n_in = (2 if self.host_experts.mode_for(i) == "float" else 4) if mt == "sparse" else 0
            def prog(params, streams, h, post, comb, w, idx_local, *outs):
                p = params["layers"][i]
                if mt == "sparse":
                    # each table arrives as a list of chunks [1, e_i, ...]; join along the expert axis
                    outs = tuple(jnp.concatenate([c[0] for c in chunks], axis=0) for chunks in outs)
                    gu_w, dn_w = self._dequant(outs, i)
                    return M.layer_post(p, streams, h, post, comb, self.lcfg, mt, gu_w, dn_w, w, idx_local)
                return M.layer_post(p, streams, h, post, comb, self.lcfg, mt)
            in_specs = (self.specs, R, R, R, R, R, R) + (tuple([P(AXIS)] * nc for nc in Eh[1]) if mt == "sparse" else ())
            self._progs[key] = jax.jit(shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=R,
                                                 check_vma=False))
        return self._progs[key]

    def _prog_head(self, B, T):
        key = ("head", B, T)
        if key not in self._progs:
            def prog(params, streams, length):
                h = M.rmsnorm(streams.mean(2).astype(self.lcfg.dtype), params["norm"], self.cfg.eps)
                last = lax.dynamic_index_in_dim(h, length - 1, axis=1, keepdims=False)
                return self._logits(params["lm_head"], last)
            self._progs[key] = jax.jit(shard_map(prog, mesh=self.mesh, in_specs=(self.specs, R, R), out_specs=R,
                                                 check_vma=False))
        return self._progs[key]

    # ---- host side
    BUCKETS = (8, 16, 32, 64, 128, 192, 288)

    def _gather_unique(self, layer, idx_np):
        """Unique experts hit by idx [N,k] -> (host arrays [n_dev, Eh_pad, ...], idx_local [N,k] int32).
        Eh is padded to a bucket so the post program compiles once per bucket; padding experts are unused."""
        tables = self.host_experts.tables[layer]          # chip-major [n_dev, E, ...]
        E = tables[0].shape[1]
        uniq, inv = np.unique(idx_np, return_inverse=True)
        Eh = len(uniq)
        if Eh >= 0.75 * E:                                 # prefill: ship the whole table, no gather at all
            return tuple(tables), idx_np.astype(np.int32), E
        Ep = next(b for b in self.BUCKETS if b >= Eh) if Eh <= self.BUCKETS[-1] else Eh
        outs = [np.empty((self.n, Ep) + t.shape[2:], t.dtype) for t in tables]
        for o in outs:
            o[:, Eh:] = 0                                  # padding experts must not hold NaN-decodable garbage

        def task(j, c, i):                                 # one contiguous memcpy per (table, chip, expert)
            outs[j][c, i] = tables[j][c, uniq[i]]

        list(self._pool.map(lambda a: task(*a), [(j, c, i) for j in range(len(tables)) for c in range(self.n)
                                                  for i in range(Eh)]))
        return tuple(outs), inv.reshape(idx_np.shape).astype(np.int32), Ep

    def _gather_and_put(self, layer, idx_np):
        """Per-chip pipeline: each chip's slices are memcpy'd then immediately device_put (8 chips in parallel),
        so transfer of chip c overlaps the gather of the others. Returns device arrays sharded P('x')."""
        tables = self.host_experts.tables[layer]          # chip-major [n_dev, E, ...]
        N, k = idx_np.shape
        flat = idx_np.reshape(-1)
        devs = list(self.mesh.devices.flat)
        sh = NamedSharding(self.mesh, P(AXIS))

        def chip_task(c):
            pieces = []
            for t in tables:
                buf = np.empty((1, N, k) + t.shape[2:], t.dtype)
                dst = buf.reshape((N * k,) + t.shape[2:])
                src = t[c]
                for i in range(N * k):
                    dst[i] = src[flat[i]]
                pieces.append(jax.device_put(buf, devs[c]))
            return pieces

        per_chip = list(self._pool.map(chip_task, range(self.n)))
        out = []
        for j, t in enumerate(tables):
            shape = (self.n, N, k) + t.shape[2:]
            out.append(jax.make_array_from_single_device_arrays(shape, sh, [per_chip[c][j] for c in range(self.n)]))
        return tuple(out)

    def _run(self, tokens, caches, pos0, use_recurrent, length=None):
        B, T = tokens.shape
        length = T if length is None else length
        tokens = jnp.asarray(tokens)
        streams = self._prog_embed(B, T)(self.params, tokens)
        new_caches = []
        self.stats = {"gather_s": 0.0, "transfer_s": 0.0}
        for i in range(self.cfg.n_layers):
            has_cache = caches is not None
            pre = self._prog_pre(i, B, T, has_cache, use_recurrent)
            streams, h, post, comb, nc, idx, w = pre(self.params, streams, jnp.int32(pos0), jnp.int32(length),
                                                     caches[i] if has_cache else None)
            new_caches.append(nc)
            if self.cfg.mlp_types[i] == "sparse":
                t0 = time.perf_counter()
                idx_np = np.asarray(idx)
                arrays, idx_local, Ep = self._gather_unique(i, idx_np)
                t1 = time.perf_counter()
                dev = self.transport(arrays)
                post_prog = self._prog_post(i, B, T, (Ep, tuple(len(c) for c in dev)))
                streams = post_prog(self.params, streams, h, post, comb, w, jnp.asarray(idx_local), *dev)
                streams.block_until_ready()
                t2 = time.perf_counter()
                self.stats["gather_s"] += t1 - t0
                self.stats["transfer_s"] += t2 - t1
            else:
                post_prog = self._prog_post(i, B, T)
                streams = post_prog(self.params, streams, h, post, comb, w, idx)
        logits = self._prog_head(B, T)(self.params, streams, jnp.int32(length))
        return logits, new_caches

    def prefill(self, tokens: np.ndarray):
        tokens = np.asarray(tokens)
        B, T = tokens.shape
        Tp = self._bucket(T)
        padded = np.zeros((B, Tp), np.int32)
        padded[:, :T] = tokens
        logits, caches = self._run(padded, None, 0, False, length=T)
        return logits, caches, T

    def decode(self, token: np.ndarray, caches, pos: int):
        B = token.shape[0]
        logits, caches = self._run(np.asarray(token).reshape(B, 1), caches, pos, True)
        return logits, caches, pos + 1
