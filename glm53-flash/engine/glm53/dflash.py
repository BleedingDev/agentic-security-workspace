"""DFlash 2 drafter for GLM-5.3-Flash (block-diffusion speculative decoding) in pure JAX.

Reference: z-lab/dflash `dflash/model.py` (MIT; `DFlash2DraftModel`, vendored as tests/dflash_ref.py), checkpoint
`incoai/GLM-5.3-Flash-DFlash2` (5 Qwen3-style layers, hidden 4096, 32 q / 8 kv heads of 128, MLP 12288, block 8,
sliding window 2048). The drafter predicts the block-1 tokens after an anchor in ONE forward: the block is the
target's embedding rows of [anchor, MASK x (block-1)]; every layer attends bidirectionally to the block and to the K/V
of the committed context positions within the window. The context comes from the target's own hidden states:
feats = the mean of the 4 mHC streams after layers `target_layers`, concatenated -> fc -> hidden_norm, and that one
tensor goes through every layer's k/v projections (no separate target projections). DFlash 2 adds a grouped dynamic
causal conv (2 taps over the block) around each attention and MLP sublayer, and a selector that walks one path
through the top-16 candidates of every draft position.

Parameter layout ([in, out], y = x @ W, as glm53.model):
  fc [nF*D, D], hidden_norm [D], norm [D]
  layers[i]: ln1, ln2 [D]; q [D, H*hd], k, v [D, KV*hd], o [H*hd, D], q_norm, k_norm [hd];
             gate, up [D, I], down [I, D]; conv_a, conv_m: {base [2, K, D], proj [D, 2*K*G]}
  sel: pred, succ [V, r] (codebooks), h [D, r]

Context cache ("ring"): k, v [L, B, W, KV, hd] = post-k_norm, post-RoPE K and V of context position p in slot p % W
(exact: a block query never sees further back than W - 1 positions), p [B, W] int32 = the position each slot holds
+ 1 (0 = empty): a slot is read only when that position is inside the window, so a position that never reached the
ring is a hole, not stale context (the engine writes every fed position: prefill, the verify's accepted positions,
batched decode steps). Positions are absolute; the drafter has its own RoPE (theta 10000) although GLM is NoPE.

Tensor parallel inside shard_map (axis engine.AXIS): q/k/v heads and MLP columns split over the chips (one KV head per
chip on 8 chips; the ring is sharded with them), o and down row-parallel (psum), fc and the conv kernel projections
column-parallel (all-gather), the codebooks, the embedding and the lm_head ([V, D] rows) vocab-sharded like the target's.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from glm53 import aot
from glm53 import model as M
from glm53 import quant8 as Q8
from glm53.engine import AXIS, R, shard_map


@dataclasses.dataclass(frozen=True)
class DCfg:
    hidden: int
    heads: int
    kv_heads: int
    hd: int
    inter: int
    n_layers: int
    eps: float
    theta: float
    window: int
    block: int
    mask_id: int
    target_layers: tuple[int, ...]
    vocab: int
    sel_rank: int
    sel_topk: int
    conv_k: int
    conv_group: int
    out_mult: float = 1.0
    softcap: float | None = None
    emb_scale: float = 1.0
    dtype: Any = jnp.float32
    tp_axis: Any = None               # shard_map axis name (None = one device, no collectives)

    def reduce(self, y):
        return y if self.tp_axis is None else lax.psum(y, self.tp_axis)

    def gather(self, y):
        return y if self.tp_axis is None else lax.all_gather(y, self.tp_axis, axis=-1, tiled=True)

    @classmethod
    def from_hf(cls, c: dict, dtype=jnp.float32) -> "DCfg":
        """`c` = the drafter's config.json (a Qwen3 config with a `dflash_config` dict)."""
        dc = c.get("dflash_config", {})

        def val(name, default=None):          # dflash_config first, then the top level (the reference's _draft_value)
            return dc.get(name, c.get(name, default))
        assert not c.get("is_causal", False) and set(c["layer_types"]) == {"sliding_attention"}, "bidirectional SWA only"
        rope = c.get("rope_parameters") or {"rope_theta": c.get("rope_theta", 10000.0), "rope_type": "default"}
        assert rope.get("rope_type", "default") == "default", rope
        return cls(hidden=c["hidden_size"], heads=c["num_attention_heads"], kv_heads=c["num_key_value_heads"],
                   hd=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"], inter=c["intermediate_size"],
                   n_layers=c["num_hidden_layers"], eps=c["rms_norm_eps"], theta=float(rope["rope_theta"]),
                   window=c["sliding_window"], block=int(val("block_size", 16)), mask_id=int(val("mask_token_id")),
                   target_layers=tuple(val("target_layer_ids")), vocab=c["vocab_size"],
                   sel_rank=int(val("selector_rank")), sel_topk=int(val("selector_top_k")),
                   conv_k=int(val("conv_kernel_size")), conv_group=int(val("conv_group_size")),
                   out_mult=float(val("output_multiplier", 1.0)), softcap=val("final_logit_softcapping"),
                   emb_scale=float(val("input_embedding_scale", 1.0)), dtype=dtype)


# ----------------------------------------------------------------------------- weights
def load_params(model_dir: str, dtype=np.float32):
    """incoai/GLM-5.3-Flash-DFlash2 (config.json + model.safetensors) -> (DCfg, params as NumPy arrays)."""
    from glm53 import checkpoint as C
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = DCfg.from_hf(json.load(f))
    r = C.RawShardReader(model_dir)

    def lin(name):                                            # torch Linear [out, in] -> [in, out]
        return np.ascontiguousarray(r.get(name + ".weight", dtype).T)

    def conv(pfx):
        return {"base": r.get(pfx + ".base_kernel", dtype), "proj": lin(pfx + ".kernel_projection")}
    layers = []
    for i in range(cfg.n_layers):
        a, m = f"layers.{i}.self_attn.", f"layers.{i}.mlp."
        layers.append({"ln1": r.get(f"layers.{i}.input_layernorm.weight", dtype),
                       "ln2": r.get(f"layers.{i}.post_attention_layernorm.weight", dtype),
                       "q": lin(a + "q_proj"), "k": lin(a + "k_proj"), "v": lin(a + "v_proj"), "o": lin(a + "o_proj"),
                       "q_norm": r.get(a + "q_norm.weight", dtype), "k_norm": r.get(a + "k_norm.weight", dtype),
                       "gate": lin(m + "gate_proj"), "up": lin(m + "up_proj"), "down": lin(m + "down_proj"),
                       "conv_a": conv(f"layers.{i}.attention_conv"), "conv_m": conv(f"layers.{i}.mlp_conv")})
    s = "candidate_selector."
    params = {"fc": lin("fc"), "hidden_norm": r.get("hidden_norm.weight", dtype), "norm": r.get("norm.weight", dtype),
              "layers": layers,
              "sel": {"pred": r.get(s + "predecessor_codebook", dtype), "succ": r.get(s + "successor_codebook", dtype),
                      "h": lin(s + "hidden_projection")}}
    return cfg, params


def param_specs(cfg: DCfg) -> dict:
    col, row = P(None, AXIS), P(AXIS, None)
    conv = {"base": R, "proj": col}
    layer = {"ln1": R, "ln2": R, "q": col, "k": col, "v": col, "o": row, "q_norm": R, "k_norm": R,
             "gate": col, "up": col, "down": row, "conv_a": conv, "conv_m": conv}
    return {"fc": col, "hidden_norm": R, "norm": R, "layers": [layer] * cfg.n_layers,
            "sel": {"pred": row, "succ": row, "h": R}}


RING_SPEC = P(None, None, None, AXIS)             # [L, B, W, KV, hd]: KV heads over the chips
RING_SPECS = {"k": RING_SPEC, "v": RING_SPEC, "p": R}


# ----------------------------------------------------------------------------- the model
def rope(x, pos, theta):
    """Rotate-half RoPE (Qwen3): x [B, T, h, hd], pos [B, T] absolute positions; angles in f32."""
    hd = x.shape[-1]
    inv = 1.0 / (theta ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))
    ang = pos.astype(jnp.float32)[..., None] * inv
    ang = jnp.concatenate([ang, ang], -1)[:, :, None, :]
    xf = x.astype(jnp.float32)
    rot = jnp.concatenate([-xf[..., hd // 2:], xf[..., :hd // 2]], -1)
    return (xf * jnp.cos(ang) + rot * jnp.sin(ang)).astype(x.dtype)


def dyn_conv(x, dyn, base, group):
    """Grouped dynamic causal conv over the block: out_t = sum_o (base[o] + dyn_t[o]) * x_{t-o}, x before the block's
    first position = 0. x [B, T, D], dyn [B, T, K, G] (one kernel per group of `group` channels), base [K, D]."""
    out = jnp.zeros_like(x)
    for o in range(base.shape[0]):
        xs = x if o == 0 else jnp.pad(x[:, :-o], ((0, 0), (o, 0), (0, 0)))
        out = out + base[o].astype(x.dtype) * xs + jnp.repeat(dyn[:, :, o], group, axis=-1) * xs
    return out


def conv_kernels(p, x, cfg: DCfg):
    """The sublayer's (normed) input [B, T, D] -> dynamic kernels [B, T, 2, K, G]: [:, :, 0] for the input conv
    (`prepare`), [:, :, 1] for the output conv (`finish`)."""
    d = cfg.gather(x @ p["proj"])
    return d.reshape(x.shape[:2] + (2, cfg.conv_k, cfg.hidden // cfg.conv_group))


def _rows(feats):
    """(B, N) of target features given as one array or as the per-layer list."""
    return (feats[0] if isinstance(feats, (list, tuple)) else feats).shape[:2]


def fuse_features(params, feats, cfg: DCfg):
    """Target features [B, N, nF*D] (or the list of nF [B, N, D] per target layer, as the engine captures them) ->
    the drafter's context hidden hidden_norm(fc(feats)) [B, N, D]."""
    if isinstance(feats, (list, tuple)):
        feats = jnp.concatenate(feats, -1)
    return M.rmsnorm(cfg.gather(feats.astype(cfg.dtype) @ params["fc"]), params["hidden_norm"], cfg.eps)


def context_kv(params, ctx, pos, cfg: DCfg):
    """ctx [B, N, D] (fuse_features) at positions pos [B, N] -> (k, v) [L, B, N, kv_local, hd] for every layer."""
    B, N, _ = ctx.shape
    ks, vs = [], []
    for p in params["layers"]:
        k = M.rmsnorm((ctx @ p["k"]).reshape(B, N, -1, cfg.hd), p["k_norm"], cfg.eps)
        ks.append(rope(k, pos, cfg.theta))
        vs.append((ctx @ p["v"]).reshape(B, N, -1, cfg.hd))
    return jnp.stack(ks), jnp.stack(vs)


def ring_write(ring, k, v, pos0, length):
    """Write k, v [L, B, N, kvl, hd] of positions pos0 + i (i < length; pos0, length [B]) into slots position % W.
    Rows more than W before the end are skipped (a later row owns their slot)."""
    W = ring["k"].shape[2]
    B, N = k.shape[1], k.shape[2]
    i = jnp.arange(N)[None, :]
    keep = (i < length[:, None]) & (i >= length[:, None] - W)
    slot = jnp.where(keep, jnp.mod(pos0[:, None] + i, W), W)        # W = out of range -> dropped
    b = jnp.arange(B)[:, None]
    return {"k": ring["k"].at[:, b, slot].set(k.astype(ring["k"].dtype), mode="drop"),
            "v": ring["v"].at[:, b, slot].set(v.astype(ring["v"].dtype), mode="drop"),
            "p": ring["p"].at[b, slot].set(pos0[:, None] + i + 1, mode="drop")}


def ingest(params, ring, feats, pos0, length, cfg: DCfg):
    """Context features of positions pos0 + i, i < length -> written into the ring (prefill pieces, accepted tokens)."""
    N = _rows(feats)[1]
    pos = pos0[:, None] + jnp.arange(N)[None, :]
    k, v = context_kv(params, fuse_features(params, feats, cfg), pos, cfg)
    return ring_write(ring, k, v, pos0, length)


def block_attention(p, x, ring_k, ring_v, ring_p, start, cfg: DCfg):
    """x [B, T, D] (the block after norm + conv) at positions start + t; ring_k/v [B, W, kvl, hd] this layer's context
    (positions < start), ring_p [B, W] their positions + 1 — or lists of B per-row arrays (batch 1 each: every stream
    its own ring; the projections stay batched). Bidirectional inside the block, window |q - k| < cfg.window to the
    context. -> [B, T, D]."""
    B, T, _ = x.shape
    hd = cfg.hd
    pos = start[:, None] + jnp.arange(T)[None, :]
    q = rope(M.rmsnorm((x @ p["q"]).reshape(B, T, -1, hd), p["q_norm"], cfg.eps), pos, cfg.theta)
    k = rope(M.rmsnorm((x @ p["k"]).reshape(B, T, -1, hd), p["k_norm"], cfg.eps), pos, cfg.theta)
    v = (x @ p["v"]).reshape(B, T, -1, hd)
    if isinstance(ring_k, (list, tuple)):
        o = jnp.concatenate([_block_attend(q[b:b + 1], k[b:b + 1], v[b:b + 1], ring_k[b], ring_v[b], ring_p[b],
                                           pos[b:b + 1], cfg) for b in range(B)], 0)
    else:
        o = _block_attend(q, k, v, ring_k, ring_v, ring_p, pos, cfg)
    return cfg.reduce(o @ p["o"])


def _block_attend(q, k, v, ring_k, ring_v, ring_p, pos, cfg: DCfg):
    """The block's queries against [the ring's context, the block] -> [B, T, H_local * hd] (before o)."""
    B, T = q.shape[:2]
    hd = cfg.hd
    kvl = k.shape[2]
    keys = jnp.concatenate([ring_k.astype(k.dtype), k], 1)
    vals = jnp.concatenate([ring_v.astype(v.dtype), v], 1)
    kpos = ring_p - 1                                                    # -1: an empty slot
    ctx_ok = (kpos[:, None, :] >= 0) & (pos[:, :, None] - kpos[:, None, :] < cfg.window)      # [B, T, W]
    mask = jnp.concatenate([ctx_ok, jnp.ones((B, T, T), bool)], -1)
    qg = q.reshape(B, T, kvl, -1, hd)                                    # q head g*rep + r reads kv head g
    s = jnp.einsum("btgrd,bsgd->bgrts", qg, keys, preferred_element_type=jnp.float32) * hd ** -0.5
    s = jnp.where(mask[:, None, None], s, -jnp.inf)
    a = jax.nn.softmax(s, -1).astype(vals.dtype)
    return jnp.einsum("bgrts,bsgd->btgrd", a, vals).reshape(B, T, -1)


def draft_hidden(params, emb, ring, start, cfg: DCfg):
    """emb [B, T, D] = embedding rows of [anchor, MASK...] (x emb_scale) at positions start + t -> final-norm hidden
    [B, T, D] (the reference's `DFlashDraftModel.forward` over the block). `ring`: one ring of batch B or a list of B
    per-row rings."""
    x = emb.astype(cfg.dtype)
    g = cfg.conv_group
    rows = isinstance(ring, (list, tuple))
    for li, p in enumerate(params["layers"]):
        ca, cm = p["conv_a"], p["conv_m"]
        h = M.rmsnorm(x, p["ln1"], cfg.eps)
        dk = conv_kernels(ca, h, cfg)
        if rows:
            rk, rv, rp = [r["k"][li] for r in ring], [r["v"][li] for r in ring], [r["p"] for r in ring]
        else:
            rk, rv, rp = ring["k"][li], ring["v"][li], ring["p"]
        a = block_attention(p, dyn_conv(h, dk[:, :, 0], ca["base"][0], g), rk, rv, rp, start, cfg)
        x = x + dyn_conv(a, dk[:, :, 1], ca["base"][1], g)
        h = M.rmsnorm(x, p["ln2"], cfg.eps)
        dk = conv_kernels(cm, h, cfg)
        h = dyn_conv(h, dk[:, :, 0], cm["base"][0], g)
        m = cfg.reduce((M.silu(h @ p["gate"]) * (h @ p["up"])) @ p["down"])
        x = x + dyn_conv(m, dk[:, :, 1], cm["base"][1], g)
    return M.rmsnorm(x, params["norm"], cfg.eps)


# ----------------------------------------------------------------------------- vocab-sharded pieces
def vocab_rows(table_local, ids, cfg: DCfg):
    """Rows `ids` (any shape) of a table whose rows are split over the chips in consecutive blocks (the target's
    embedding, the selector's codebooks); an int8 q8 table is dequantized row by row (Engine._embed for any table)."""
    q8 = Q8.is_q8(table_local)
    t = table_local["q"] if q8 else table_local
    rows = t.shape[0]
    local = ids if cfg.tp_axis is None else ids - lax.axis_index(cfg.tp_axis) * rows
    idx = jnp.clip(local, 0, rows - 1)
    e = jnp.take(t, idx, axis=0)
    if q8:
        e = e.astype(jnp.float32) * jnp.take(table_local["s"], idx, axis=0)
    if cfg.tp_axis is None:
        return e
    return lax.psum(jnp.where(((local >= 0) & (local < rows))[..., None], e, 0), cfg.tp_axis)


def logits_local(lm_head_local, h, cfg: DCfg):
    """This chip's vocab columns of the draft logits (the target's lm_head, bf16 or int8 q8), f32, with the
    drafter's output multiplier and soft cap."""
    from glm53.engine import dot_t                     # the lm_head is stored [V, D] (engine.lm_head_t)
    if Q8.is_q8(lm_head_local):
        z = dot_t(h, lm_head_local["q"].astype(h.dtype)) * lm_head_local["s"].reshape(-1)
    else:
        z = dot_t(h, lm_head_local)
    z = z * cfg.out_mult
    if cfg.softcap:
        z = jnp.tanh(z / cfg.softcap) * cfg.softcap
    return z


def topk_vocab(z_local, k, cfg: DCfg):
    """Exact top-k over vocab-sharded logits: (values, global ids) [..., k], descending (each chip's top-k,
    all-gathered, then the top-k of those)."""
    v, i = M.top_k(z_local, k)                     # (chunked: one XLA sort over the vocab shard cost ~0.17 ms)
    if cfg.tp_axis is None:
        return v, i
    i = i + lax.axis_index(cfg.tp_axis) * z_local.shape[-1]
    v, i = cfg.gather(v), cfg.gather(i)
    v, j = lax.top_k(v, k)
    return v, jnp.take_along_axis(i, j, -1)


def select(sel, hidden, z_local, anchor, cfg: DCfg, temperature=None, key=None):
    """DFlash 2 candidate selector over the draft positions. hidden [B, T1, D], z_local their logits (this chip's
    vocab columns), anchor [B]. score_t(a, b) = logit_t(b) + <P[a] * (h_t @ H), S[b]> for the top-k candidates b of
    position t, a = the token chosen at t-1 (the anchor at t = 0); greedy argmax, or (temperature given: a float or a
    traced scalar / [B]) a draw from softmax(score / temperature) with `key`, the argmax where temperature <= 0 (q is
    then one-hot). -> (path [B, T1], candidates [B, T1, k], q [B, T1, k] or None, scores [B, T1, k])."""
    unary, cand = topk_vocab(z_local, cfg.sel_topk, cfg)
    B, T1, k = cand.shape
    hs = jnp.dot(hidden, sel["h"], preferred_element_type=jnp.float32)
    ids = jnp.concatenate([anchor[:, None], cand.reshape(B, -1)], 1)
    pr = vocab_rows(sel["pred"], ids, cfg).astype(jnp.float32)          # P rows of the anchor and every candidate
    pa, pc = pr[:, 0], pr[:, 1:].reshape(B, T1, k, -1)
    sc = vocab_rows(sel["succ"], cand, cfg).astype(jnp.float32)          # [B, T1, k, r]
    path, qs, scores = [], [], []
    if temperature is not None:              # a host float or a traced scalar / [B] (rows at <= 0 pick the argmax)
        tt = jnp.reshape(jnp.asarray(temperature, jnp.float32), (-1, 1))
    for t in range(T1):
        s = unary[:, t] + jnp.einsum("br,bkr->bk", pa * hs[:, t], sc[:, t])
        if temperature is None:
            j = jnp.argmax(s, -1)
        else:
            key, sub = jax.random.split(key)
            z = s / jnp.where(tt > 0, tt, 1.0)
            g = jnp.argmax(s, -1)
            j = jnp.where(tt[:, 0] > 0, jax.random.categorical(sub, z, -1), g)
            qs.append(jnp.where(tt > 0, jax.nn.softmax(z, -1), jax.nn.one_hot(g, k, dtype=jnp.float32)))
        path.append(jnp.take_along_axis(cand[:, t], j[:, None], -1)[:, 0])
        pa = jnp.take_along_axis(pc[:, t], j[:, None, None], 1)[:, 0]
        scores.append(s)
    return jnp.stack(path, 1), cand, (jnp.stack(qs, 1) if qs else None), jnp.stack(scores, 1)


def draft(params, ring, feats, pos0, n_new, anchor, embed, lm_head, cfg: DCfg, T=None, temperature=None, key=None):
    """One draft: write the n_new new context positions' features (feats [B, N, nF*D] at pos0 + i, i < n_new) into
    the ring, then draft the block [anchor, MASK x (T-1)] at position start = pos0 + n_new. `ring` = one ring of batch
    B, or a list of B per-row rings (batch 1 each; returned as a list).
    -> (path [B, T-1], candidates, q, scores, ring, hidden [B, T, D])."""
    T = cfg.block if T is None else T
    if isinstance(ring, (list, tuple)):              # per-row rings: the context K/V batched, written row by row
        N = _rows(feats)[1]
        k, v = context_kv(params, fuse_features(params, feats, cfg), pos0[:, None] + jnp.arange(N)[None, :], cfg)
        ring = [ring_write(r, k[:, b:b + 1], v[:, b:b + 1], pos0[b:b + 1], n_new[b:b + 1]) for b, r in enumerate(ring)]
    else:
        ring = ingest(params, ring, feats, pos0, n_new, cfg)
    start = pos0 + n_new
    B = anchor.shape[0]
    ids = jnp.concatenate([anchor[:, None], jnp.full((B, T - 1), cfg.mask_id, anchor.dtype)], 1)
    emb = vocab_rows(embed, ids, cfg).astype(cfg.dtype) * cfg.emb_scale
    h = draft_hidden(params, emb, ring, start, cfg)
    z = logits_local(lm_head, h[:, 1:], cfg)
    path, cand, q, scores = select(params["sel"], h[:, 1:], z, anchor, cfg, temperature, key)
    return path, cand, q, scores, ring, h


# ----------------------------------------------------------------------------- on the chips
class Drafter:
    """The drafter's params on a mesh (axis engine.AXIS) and its jitted programs. `embed_spec` / `lm_head_spec` =
    the target's (engine `specs["embed"]`, `specs["lm_head"]`: vocab-sharded, possibly q8 dicts); the programs take
    the target's arrays as arguments (the lm_head [V, D], `engine.lm_head_t`). int8=True stores the big matrices (layers, fc) as q8 (glm53.quant8); the
    codebooks stay in their dtype. Ring buffers are donated to `ingest` / `draft` (never reuse a ring passed in).
    temp_scale: a sampled draft draws its path at temp_scale x the given temperature (q is that distribution)."""

    def __init__(self, cfg: DCfg, params, mesh, embed_spec=P(AXIS, None), lm_head_spec=P(AXIS, None), int8=False,
                 temp_scale=1.0):
        assert cfg.kv_heads % mesh.size == 0 and cfg.vocab % mesh.size == 0, (cfg.kv_heads, cfg.vocab, mesh.size)
        self.cfg, self.mesh = cfg, mesh
        self.lcfg = dataclasses.replace(cfg, tp_axis=AXIS)
        specs = param_specs(cfg)
        if int8:
            layers, lspecs = [], []
            for L, sp in zip(params["layers"], specs["layers"]):
                qp, qs = Q8.quantize_layer(L, sp, lambda w, ax: Q8.quantize_array_host(w, ax))
                layers.append(qp); lspecs.append(qs)
            params = {**params, "layers": layers, "fc": Q8.quantize_array_host(params["fc"], 1)}
            specs = {**specs, "layers": lspecs, "fc": {"q": specs["fc"], "s": specs["fc"]}}
        self.specs = specs
        self.params = jax.tree.map(lambda a, s: jax.device_put(a, NamedSharding(mesh, s)), params, specs,
                                   is_leaf=lambda x: isinstance(x, P))
        self.embed_spec, self.lm_head_spec = embed_spec, lm_head_spec
        self.temp_scale = float(temp_scale)      # the sampled path is drawn at temp_scale x the given temperature
        self._progs = {}

    def nbytes_per_chip(self):
        return sum(s.data.nbytes for a in jax.tree.leaves(self.params) for s in a.addressable_shards[:1])

    def alloc_ring(self, B):
        shape = (self.cfg.n_layers, B, self.cfg.window, self.cfg.kv_heads, self.cfg.hd)
        z = jax.jit(lambda: jnp.zeros(shape, self.cfg.dtype), out_shardings=NamedSharding(self.mesh, RING_SPEC))
        zp = jax.jit(lambda: jnp.zeros((B, self.cfg.window), jnp.int32), out_shardings=NamedSharding(self.mesh, R))
        return {"k": z(), "v": z(), "p": zp()}

    def _prog_ingest(self, B, N):
        key = ("ingest", B, N)
        if key not in self._progs:
            def prog(params, ring, feats, pos0, length):
                return ingest(Q8.dequant_tree(params, self.cfg.dtype), ring, feats, pos0, length, self.lcfg)
            rs = RING_SPECS
            sm = shard_map(prog, mesh=self.mesh, in_specs=(self.specs, rs, R, R, R), out_specs=rs, check_vma=False)
            self._progs[key] = aot.jit(key, sm, donate_argnums=(1,))
        return self._progs[key]

    def _prog_draft(self, B, N, T, sample, feat_index=None, rows=False):
        """rows=True: B per-row rings (a list, each batch 1, donated) instead of one ring of batch B."""
        key = ("draft", B, N, T, sample) + ((feat_index,) if feat_index is not None else ()) + (("rows",) if rows else ())
        if key not in self._progs:
            def prog(params, ring, feats, pos0, n_new, anchor, embed, lm_head, *samp):
                if feat_index is not None:      # stacked per layer group (the engine's verify): pick the target layers
                    feats = [f[j] for f, js in zip(feats, feat_index) for j in js]
                params = Q8.dequant_tree(params, self.cfg.dtype)
                temp, k = samp if sample else (None, None)
                if sample:                      # the engine passes the sampler's key, which its acceptance also splits
                    k = jax.random.fold_in(k, 0xDF)
                    temp = temp * self.temp_scale   # (a sharper q than the target's accepts more: v6e 2026-10-04)
                path, cand, q, scores, ring, _ = draft(params, ring, feats, pos0, n_new, anchor, embed, lm_head,
                                                       self.lcfg, T, temp, k)
                out = {"path": path, "cand": cand, "scores": scores, "rings" if rows else "ring": ring,
                       "seq": jnp.concatenate([anchor[:, None].astype(path.dtype), path], 1)}
                return {**out, "q": q} if sample else out
            rs = [RING_SPECS] * B if rows else RING_SPECS
            in_specs = (self.specs, rs, R, R, R, R, self.embed_spec, self.lm_head_spec) + ((R, R) if sample else ())
            out_specs = {"path": R, "cand": R, "scores": R, "rings" if rows else "ring": rs, "seq": R,
                         **({"q": R} if sample else {})}
            sm = shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
            self._progs[key] = aot.jit(key, sm, donate_argnums=(1,))
        return self._progs[key]

    def ingest(self, ring, feats, pos0, length):
        """feats [B, N, nF*D] (or the per-layer list) of positions pos0 + i, i < length (pos0, length [B] int32)
        -> new ring."""
        B, N = _rows(feats)
        return self._prog_ingest(B, N)(self.params, ring, feats, jnp.asarray(pos0, jnp.int32),
                                       jnp.asarray(length, jnp.int32))

    def draft(self, ring, feats, pos0, n_new, anchor, embed, lm_head, T=None, temperature=None, key=None,
              feat_index=None):
        """-> dict: path [B, T-1] (the drafted tokens), seq [B, T] (anchor + path: the verify's input), cand / scores
        [B, T-1, k], ring (the new one), q [B, T-1, k] when a temperature is given. pos0, n_new, anchor [B] int32 (host
        values or device arrays). `feat_index`: feats is a list of stacked [n_g, B, N, D] arrays and feat_index[g] the
        target positions to take from each (the engine's per-group capture)."""
        B, N = _rows(feats) if feat_index is None else feats[0].shape[1:3]
        T = self.cfg.block if T is None else T
        args = (self.params, ring, feats, jnp.asarray(pos0, jnp.int32), jnp.asarray(n_new, jnp.int32),
                jnp.asarray(anchor, jnp.int32), embed, lm_head)
        if temperature is not None:              # (a device scalar, e.g. DeviceSampler.bind's, passes through)
            args += (temperature if isinstance(temperature, jax.Array) else jnp.float32(temperature), key)
        return self._prog_draft(B, N, T, temperature is not None, feat_index)(*args)

    def draft_rows(self, rings, feats, pos0, n_new, anchor, embed, lm_head, T=None, temperature=None, key=None,
                   feat_index=None):
        """`draft` for B streams that each keep their own ring (`rings`: a list of B rings of batch 1, donated; the
        engine's per-stream cache sets) -> the same dict with "rings" (the new list) instead of "ring"."""
        B = len(rings)
        N = (_rows(feats) if feat_index is None else feats[0].shape[1:3])[1]
        T = self.cfg.block if T is None else T
        args = (self.params, list(rings), feats, jnp.asarray(pos0, jnp.int32), jnp.asarray(n_new, jnp.int32),
                jnp.asarray(anchor, jnp.int32), embed, lm_head)
        if temperature is not None:
            args += (temperature if isinstance(temperature, jax.Array) else jnp.float32(temperature), key)
        return self._prog_draft(B, N, T, temperature is not None, feat_index, rows=True)(*args)
