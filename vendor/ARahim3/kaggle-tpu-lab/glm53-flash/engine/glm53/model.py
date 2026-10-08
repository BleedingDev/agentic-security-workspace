"""Glm5Next text model in pure JAX (functional, params = nested dict of arrays).

Reference: transformers `modeling_glm5_next.py` (v5.16). Every block is written so that
prefill (full sequence) and single-token decode share the same math; caches are explicit
pytrees threaded through the functions.

Parameter layout (all Linear weights stored as [in, out], i.e. y = x @ W):
  embed [V, D], norm [D], lm_head [D, V]
  layers[i]:
    ln1 [D], ln2 [D]
    hc_attn / hc_ffn: fn [hc*D, (2+hc)*hc], base [(2+hc)*hc], scale [3]
    attn (KDA):  q,k,v [D, H*hd]; conv_q, conv_k, conv_v [H*hd, K]; f_a [D, hd]; f_b [hd, H*hd];
                 dt_bias [H*hd]; A_log [H]; b [D, H]; g_a [D, hd]; g_b [hd, H*hd]; o_norm [hd]; o [H*hd, D]
    attn (MLA):  q_a [D, qr]; q_a_norm [qr]; q_b [qr, H*qk]; kv_a [D, kvr]; kv_a_norm [kvr];
                 kv_b [kvr, H*(qk+v)]; o [H*v, D]; indexer {...} (unused in v0)
    mlp (dense): gate [D, I], up [D, I], down [I, D]
    mlp (moe):   router_w [D, E], router_bias [E], gate_up [E, D, 2*mi], down [E, mi, D], shared {gate, up, down}
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import numpy as np
import jax.numpy as jnp
from jax import lax

Params = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Cfg:
    hidden: int
    vocab: int
    n_layers: int
    layer_types: tuple[str, ...]      # "linear_attention" | "deepseek_sparse_attention"
    mlp_types: tuple[str, ...]        # "dense" | "sparse"
    eps: float
    # KDA
    kda_heads: int
    kda_hd: int
    conv_k: int
    gate_lb: float | None
    # MLA (NoPE)
    n_heads: int
    q_lora: int
    kv_lora: int
    qk_nope: int
    v_hd: int
    # indexer
    idx_heads: int
    idx_hd: int
    idx_topk: int
    idx_kpool: int
    # mHC
    hc: int
    hc_iters: int
    hc_eps: float
    # MLP / MoE
    inter: int
    moe_inter: int
    n_exp: int
    topk: int
    n_shared: int
    scaling: float
    norm_topk: bool
    swiglu_limit: float
    # compute dtype for matmuls (fp32 for CPU tests, bf16 on TPU)
    dtype: Any = jnp.float32
    # tensor-parallel reduction applied after every row-parallel matmul (None = single device)
    tp_reduce: Any = None
    tp_axis: Any = None               # shard_map axis name (collectives inside the model)
    seq_shard: int = 1                # MLA/indexer caches sharded over this many chips (interleaved by k-pool)
    q_block: int = 128                # queries per attention block (bounds the per-query temporaries)
    union_pools: int = 512            # per-block key-dedup budget (local pools); over budget -> per-query gathers; 0 = always per-query
    per_query_max: int = 4            # query blocks of up to this many queries skip the dedup (decode, short verifies)
    topk_local: int = 128             # first-stage local top-k per chip (exact: falls back to the full K when a chip's list was cut short)
    cache_cap: int = 0                # set inside mla_attention: global cache capacity
    cache_q8: bool = False            # MLA latent cache stored as int8 rows + a per-token f32 scale ("cs"): half the HBM
    pool_gather: int = 0              # per-query key gathers by whole k-pools ((kp, kvr) windows, one per selected pool,
                                      # instead of a row per token; the scales as 128-lane rows + a lane select) from this
                                      # many queries per block (1 = decode too; 0 = never). OFF: in a micro-benchmark a
                                      # T = 4 verify's layer gather fell from 16.7 + 50.8 us (token rows + element-
                                      # gathered scales) to 11.4 + 9.7 (j148), but inside the programs XLA re-lays the
                                      # whole per-layer cache out for the window gather ({2,0,1:T(1,128)}, a copy per
                                      # call: j150 2026-10-05, plain B = 3 +0.6 ms per step, the verify's gain eaten)
    select_buckets: tuple = ()        # indexer selection (sharded cache): the local top-k over the smallest of these
                                      # pool counts holding every pool visible from the block's positions, chosen by a
                                      # lax.switch. OFF: the conditional per MLA layer cost more than the sorts it saved
                                      # (v6e 2026-10-05, j143: one-stream DFlash step +0.5 ms with (512, 2048))

    def reduce(self, y):
        return y if self.tp_reduce is None else self.tp_reduce(y)

    @classmethod
    def from_hf(cls, c, dtype=jnp.float32) -> "Cfg":
        return cls(
            hidden=c.hidden_size, vocab=c.vocab_size, n_layers=c.num_hidden_layers,
            layer_types=tuple(c.layer_types), mlp_types=tuple(c.mlp_layer_types), eps=c.rms_norm_eps,
            kda_heads=c.linear_num_heads, kda_hd=c.linear_head_dim, conv_k=c.linear_conv_kernel_dim,
            gate_lb=c.linear_lower_bound,
            n_heads=c.num_attention_heads, q_lora=c.q_lora_rank, kv_lora=c.kv_lora_rank,
            qk_nope=c.qk_nope_head_dim, v_hd=c.v_head_dim,
            idx_heads=c.index_n_heads, idx_hd=c.index_head_dim, idx_topk=c.index_topk, idx_kpool=c.index_kpool,
            hc=c.hc_mult, hc_iters=c.hc_sinkhorn_iters, hc_eps=c.hc_eps,
            inter=c.intermediate_size, moe_inter=c.moe_intermediate_size, n_exp=c.n_routed_experts,
            topk=c.num_experts_per_tok, n_shared=c.n_shared_experts, scaling=c.routed_scaling_factor,
            norm_topk=c.norm_topk_prob, swiglu_limit=c.swiglu_limit, dtype=dtype,
        )


# ----------------------------------------------------------------------------- basics
def rmsnorm(x, w, eps):
    xf = x.astype(jnp.float32)
    xf = xf * lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + eps)
    return (w.astype(jnp.float32) * xf).astype(x.dtype)


def unweighted_rmsnorm_f32(x, eps):
    return x * lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps)


def l2norm(x, eps=1e-6):
    return x / jnp.sqrt(jnp.sum(x * x, -1, keepdims=True) + eps)


def silu(x):
    return x * jax.nn.sigmoid(x)


def swiglu_clamped(gate, up, limit):
    gate = jnp.minimum(gate, limit)
    up = jnp.clip(up, -limit, limit)
    return silu(gate) * up


def mlp_partial(p, x, cfg: Cfg):
    """Column/row-parallel MLP without the final TP reduction."""
    return swiglu_clamped(x @ p["gate"], x @ p["up"], cfg.swiglu_limit) @ p["down"]


def mlp(p, x, cfg: Cfg):
    return cfg.reduce(mlp_partial(p, x, cfg))


# ----------------------------------------------------------------------------- mHC
def hc_pre(p, streams, cfg: Cfg):
    """streams [B,T,hc,D] -> (post [B,T,hc], comb [B,T,hc,hc], collapsed [B,T,D])."""
    hc = cfg.hc
    B, T = streams.shape[:2]
    flat = unweighted_rmsnorm_f32(streams.reshape(B, T, -1).astype(jnp.float32), cfg.eps)
    if "fn" in p:
        mix = flat @ p["fn"].astype(jnp.float32)                   # [B,T,(2+hc)*hc]
    else:                                                          # stored transposed for the kernels (mhc_params)
        mix = jnp.einsum("btk,mk->btm", flat, p["fnT"].astype(jnp.float32))
    pre_w, post_w, comb_w = mix[..., :hc], mix[..., hc:2 * hc], mix[..., 2 * hc:]
    base, scale = p["base"].astype(jnp.float32), p["scale"].astype(jnp.float32)
    pre = jax.nn.sigmoid(pre_w * scale[0] + base[:hc]) + cfg.hc_eps
    post = 2.0 * jax.nn.sigmoid(post_w * scale[1] + base[hc:2 * hc])
    comb_logits = comb_w.reshape(B, T, hc, hc) * scale[2] + base[2 * hc:].reshape(hc, hc)
    comb = jax.nn.softmax(comb_logits, -1) + cfg.hc_eps
    # Sinkhorn on an hc x hc matrix. Row/column sums are written as explicit slice adds instead of reductions so the
    # whole (2*iters-1)-step chain stays one elementwise XLA fusion: with reductions every step is its own tiny kernel
    # (~40 launches per layer per token ≈ 10 ms/token on v5e at decode).
    rsum = lambda c: sum(c[..., :, j:j + 1] for j in range(hc))          # [B,T,hc,1]
    csum = lambda c: sum(c[..., j:j + 1, :] for j in range(hc))          # [B,T,1,hc]
    comb = comb / (csum(comb) + cfg.hc_eps)
    for _ in range(cfg.hc_iters - 1):
        comb = comb / (rsum(comb) + cfg.hc_eps)
        comb = comb / (csum(comb) + cfg.hc_eps)
    collapsed = jnp.einsum("bth,bthd->btd", pre, streams.astype(jnp.float32)).astype(streams.dtype)
    return post, comb, collapsed


def mhc_params(p):
    """An mHC site's params as the kernels read them: fn [hc*D, M] stored transposed, fnT [M, hc*D] (a [., 24] minor
    dim would be lane-padded in VMEM; the HBM bytes are the same). hc_pre's XLA path reads either."""
    if "fn" not in p:
        return p
    fn = p["fn"]
    fnT = np.ascontiguousarray(np.asarray(fn).T) if isinstance(fn, np.ndarray) else jnp.transpose(fn)
    return {"fnT": fnT, **{k: v for k, v in p.items() if k != "fn"}}


def hc_post(post, comb, y, residual):
    """new_streams[h] = post[h] * y + sum_h' comb[h', h] * residual[h']  (matches torch code)."""
    dt = residual.dtype
    return post.astype(dt)[..., None] * y[..., None, :] + jnp.einsum("bthk,bthd->btkd", comb.astype(dt), residual)


# ----------------------------------------------------------------------------- KDA
def causal_conv1d(x, w, state=None, length=None, out_dtype=None):
    """Depthwise causal conv over time. x [B,T,C], w [C,K], state [B,K-1,C] or None (zeros).
    `length` (traced int, <= T): only the first `length` tokens are real; the new state is taken from them.
    Returns (y [B,T,C] in x's dtype (or out_dtype), new_state [B,K-1,C])."""
    B, T, C = x.shape
    K = w.shape[-1]
    if state is None:
        state = jnp.zeros((B, K - 1, C), x.dtype)
    xp = jnp.concatenate([state.astype(x.dtype), x], axis=1)        # [B, T+K-1, C]
    y = jnp.zeros((B, T, C), jnp.float32)
    for j in range(K):
        y = y + xp[:, j:j + T].astype(jnp.float32) * w[:, j].astype(jnp.float32)
    if length is None:
        new_state = xp[:, -(K - 1):]
    else:
        new_state = lax.dynamic_slice_in_dim(xp, length, K - 1, axis=1)   # inputs at t in [length-K+1, length-1]
    return y.astype(out_dtype or x.dtype), new_state


def kda_gates(p, x, cfg: Cfg):
    """Forget gate g [B,T,H,hd] (log-decay, <= 0) and beta [B,T,H] in fp32."""
    B, T, _ = x.shape
    f = (x @ p["f_a"]) @ p["f_b"]
    g = f.astype(jnp.float32) + p["dt_bias"].astype(jnp.float32)
    g = g.reshape(B, T, cfg.kda_heads, cfg.kda_hd)
    decay_rate = jnp.exp(p["A_log"].astype(jnp.float32))[None, None, :, None]
    if cfg.gate_lb is not None:
        g = cfg.gate_lb * jax.nn.sigmoid(decay_rate * g)
    else:
        g = -decay_rate * jax.nn.softplus(g)
    beta = jax.nn.sigmoid((x @ p["b"]).astype(jnp.float32))
    return g, beta


def kda_chunked(q, k, v, g, beta, state0=None, chunk=64):
    """Chunked KDA (HF `chunk_kimi_delta_attention`). q,k,v [B,T,H,d]; g [B,T,H,d]; beta [B,T,H].
    Returns out [B,T,H,d] (fp32) and final state [B,H,dk,dv] (fp32).
    Memory is O(one chunk) in the per-channel decay `exp(g_i - g_j)` [c,c,d]: it is computed per chunk inside a
    `lax.map` (intra-chunk term) and again inside the state scan, never for all chunks at once (that tensor was
    262 KB/token/chip and capped a single prefill call at ~2048 tokens)."""
    B, T, H, dk = k.shape
    dv = v.shape[-1]
    f32 = jnp.float32
    q, k, v, g = [jnp.swapaxes(t.astype(f32), 1, 2) for t in (q, k, v, g)]   # [B,H,T,*]
    beta = jnp.swapaxes(beta.astype(f32), 1, 2)                              # [B,H,T]
    q = l2norm(q) * (dk ** -0.5)
    k = l2norm(k)
    pad = (-T) % chunk
    Tp = T + pad
    if pad:
        q, k, v, g = [jnp.pad(t, ((0, 0), (0, 0), (0, pad), (0, 0))) for t in (q, k, v, g)]
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad)))
    n = Tp // chunk
    q, k, v, g = [t.reshape(B, H, n, chunk, -1) for t in (q, k, v, g)]
    beta = beta.reshape(B, H, n, chunk)
    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]
    g = jnp.cumsum(g, axis=-2)                                             # per-channel cumsum in chunk
    ii = jnp.arange(chunk)
    strict_lower = ii[:, None] > ii[None, :]
    lower_incl = ii[:, None] >= ii[None, :]
    eye = jnp.eye(chunk, dtype=f32)

    def chunk_decay(g_i):
        # decay[i,j,d] = exp(g_i[d] - g_j[d]) for i >= j (<= 1); the upper triangle is never used, clamp it so no
        # inf/NaN is ever produced (g spans down to -320 per chunk)
        return jnp.exp(jnp.minimum(g_i[..., :, None, :] - g_i[..., None, :, :], 0.0))   # [B,H,c,c,d]

    def intra(xs):
        k_i, g_i, kb_i = xs                                                  # [B,H,c,d]
        a = -jnp.einsum("bhid,bhjd,bhijd->bhij", kb_i, k_i, chunk_decay(g_i))
        return jnp.where(strict_lower, a, 0.0)

    per_chunk = lambda *ts: tuple(jnp.moveaxis(t, 2, 0) for t in ts)      # [n,B,H,c,*]
    attn = jnp.moveaxis(lax.map(intra, per_chunk(k, g, k_beta)), 0, 2)     # [B,H,n,c,c]
    # HF: forward substitution computing (I - attn)^{-1} with unit diagonal, i.e. inverse of (I + M)
    L = eye - attn                                                          # lower unit-triangular
    attn = jnp.vectorize(lambda m: jax.scipy.linalg.solve_triangular(m, eye, lower=True, unit_diagonal=True),
                         signature="(c,c)->(c,c)")(L)
    u = attn @ v_beta                                                       # [B,H,n,c,dv]
    w = attn @ (k_beta * jnp.exp(g))                                        # [B,H,n,c,dk]
    S0 = jnp.zeros((B, H, dk, dv), f32) if state0 is None else state0.astype(f32)

    def step(S, xs):
        q_i, k_i, g_i, u_i, w_i = xs                                          # leading dims [B,H,...]
        attn_inter = (q_i * jnp.exp(g_i)) @ S
        attn_intra = jnp.einsum("bhid,bhjd,bhijd->bhij", q_i, k_i, chunk_decay(g_i))
        attn_intra = jnp.where(lower_incl, attn_intra, 0.0)
        v_new = u_i - w_i @ S
        out = attn_inter + attn_intra @ v_new
        g_last = g_i[..., -1:, :]
        S = S * jnp.exp(g_last).swapaxes(-1, -2) + jnp.swapaxes(k_i * jnp.exp(g_last - g_i), -1, -2) @ v_new
        return S, out

    S, out = lax.scan(step, S0, per_chunk(q, k, g, u, w))
    out = jnp.moveaxis(out, 0, 2).reshape(B, H, Tp, dv)[:, :, :T]
    return jnp.swapaxes(out, 1, 2), S


KDA_UNROLL = 0             # kda_recurrent unrolls its token loop for 2..KDA_UNROLL tokens instead of a lax.scan (a
                           # while loop: 0.19 ms per one-stream DFlash step on the v6e). OFF: unrolled, the step's
                           # device time ROSE 13.16 -> 13.63 ms (j150 2026-10-05: loop fusions +0.25, all-reduces
                           # +0.15, the while gone) — the scan body is the better-fused form. Results agree to an ulp.


def kda_state_step(S, k_i, v_i, g_i, b_i):
    """The KDA state update of one token (f32; k_i l2-normed): S [B,H,dk,dv], k/v/g [B,H,d], b [B,H]."""
    S = S * jnp.exp(g_i)[..., None]
    kv_mem = jnp.einsum("bhkv,bhk->bhv", S, k_i)
    delta = (v_i - kv_mem) * b_i[..., None]
    return S + k_i[..., None] * delta[..., None, :]


def kda_recurrent(q, k, v, g, beta, state, hist=False):
    """One (or a few) tokens, recurrent form. Shapes as kda_chunked; state [B,H,dk,dv] fp32.
    hist=True also returns the state after every token ([T,B,H,dk,dv]; speculative-decoding rollback); hist="kin"
    instead returns every token's state-update inputs [T,B,H,4,d] f32 (k, v, g, beta broadcast over d: `kda_replay`
    recomputes the state after any prefix from them, 4/dv of the states' bytes)."""
    f32 = jnp.float32
    B, T, H, dk = k.shape
    q = l2norm(q.astype(f32)) * (dk ** -0.5)
    k = l2norm(k.astype(f32))
    v, g, beta = v.astype(f32), g.astype(f32), beta.astype(f32)
    keep = hist is True

    def step(S, xs):
        q_i, k_i, v_i, g_i, b_i = xs                                          # [B,H,d] / [B,H]
        S = kda_state_step(S, k_i, v_i, g_i, b_i)
        o = jnp.einsum("bhkv,bhk->bhv", S, q_i)
        return S, ((o, S) if keep else o)

    xs = tuple(jnp.moveaxis(t, 1, 0) for t in (q, k, v, g, beta))
    if 2 <= T <= KDA_UNROLL:              # a verify of a few tokens: straight-line code, no while loop (T = 1: XLA
        # inlines the one-iteration scan already; its bits are the plain decode fingerprint's)
        S, outs = state.astype(f32), []
        for t in range(T):
            S, o = step(S, tuple(x[t] for x in xs))
            outs.append(o)
        out = jax.tree.map(lambda *a: jnp.stack(a), *outs) if T > 1 else jax.tree.map(lambda a: a[None], outs[0])
    else:
        S, out = lax.scan(step, state.astype(f32), xs)
    if keep:
        out, states = out
        return jnp.moveaxis(out, 0, 1), S, states
    if hist == "kin":
        kin = jnp.stack([k, v, g, jnp.broadcast_to(beta[..., None], k.shape)], axis=3)     # [B,T,H,4,d]
        return jnp.moveaxis(out, 0, 1), S, jnp.moveaxis(kin, 1, 0)
    return jnp.moveaxis(out, 0, 1), S


def kda_replay(state, kin, n_keep):
    """The KDA state after the first n_keep (traced, 0..T) tokens of a verify, from the state before it [B,H,dk,dv]
    and the tokens' update inputs kin [T,B,H,4,d] (`kda_recurrent(hist="kin")`): the same step, unrolled and masked."""
    S = state.astype(jnp.float32)
    for t in range(kin.shape[0]):
        S = jnp.where(t < n_keep, kda_state_step(S, kin[t, :, :, 0], kin[t, :, :, 1], kin[t, :, :, 2],
                                                 kin[t, :, :, 3, 0]), S)
    return S


def kda_project(p, x, cfg: Cfg):
    """Position-free, batched part of a KDA layer: x [B,T,D] (input-normed) -> dict(qkv [B,T,3C] before the conv,
    conv_w [3C,K], g [B,T,H,hd], beta [B,T,H], gate [B,T,H,hd])."""
    B, T, _ = x.shape
    H, hd = cfg.kda_heads, cfg.kda_hd
    qkv = jnp.concatenate([x @ p["q"], x @ p["k"], x @ p["v"]], -1)            # [B,T,3C]
    conv_w = jnp.concatenate([p["conv_q"], p["conv_k"], p["conv_v"]], 0)        # [3C,K]
    g, beta = kda_gates(p, x, cfg)
    gate = ((x @ p["g_a"]) @ p["g_b"]).reshape(B, T, H, hd)
    return {"qkv": qkv, "conv_w": conv_w, "g": g, "beta": beta, "gate": gate}


def kda_core(proj, cfg: Cfg, cache=None, use_recurrent=False, length=None, hist=False):
    """Conv + KDA recurrence of `proj` (kda_project; any B in lockstep, ONE cache) -> (core [B,T,H,hd] f32,
    new_cache). cache: None or {"conv": [B,K-1,3C], "state": [B,H,dk,dv]}. `length`: number of real tokens (rest
    is padding: beta=0, g=0 so the state ignores them). hist=True (recurrent, T > 1: speculative verify) adds
    "state_hist" [T+1,B,H,dk,dv] and "conv_hist" [T+1,B,K-1,3C] = the cache after 0..T of the tokens (entry 0 =
    before them), for `rollback_states`; hist="kin": "kin_hist" [T,B,H,4,d] (the tokens' state-update inputs) instead
    of "state_hist", and the returned "state" is the one BEFORE the tokens (the rollback replays the accepted ones:
    `kda_replay`)."""
    qkv_raw, conv_w, g, beta = proj["qkv"], proj["conv_w"], proj["g"], proj["beta"]
    B, T, _ = qkv_raw.shape
    H, hd = cfg.kda_heads, cfg.kda_hd
    conv_state = None if cache is None else cache["conv"]
    qkv, new_conv = causal_conv1d(qkv_raw, conv_w, conv_state, length)
    qkv = silu(qkv)
    C = H * hd
    q, k, v = qkv[..., :C], qkv[..., C:2 * C], qkv[..., 2 * C:]
    q, k, v = [t.reshape(B, T, H, hd) for t in (q, k, v)]
    if length is not None:
        valid = jnp.arange(T)[None, :] < length                                    # [1,T]
        beta = jnp.where(valid[..., None], beta, 0.0)
        g = jnp.where(valid[..., None, None], g, 0.0)
    state = None if cache is None else cache["state"]
    extra = {}
    if use_recurrent:
        assert state is not None
        if hist:
            core, new_state, states = kda_recurrent(q, k, v, g, beta, state, hist=hist)
            K = conv_w.shape[-1]
            full = jnp.concatenate([conv_state.astype(qkv_raw.dtype), qkv_raw], axis=1)       # [B,K-1+T,3C]
            # entry t = the conv state after t tokens; entry 0 (before the block) lets a rollback to n_keep = 0 undo
            # a verify that was dispatched ahead of its tokens (`ResidentLayerEngine.dflash_step`)
            extra = {"conv_hist": jnp.stack([full[:, t:t + K - 1] for t in range(T + 1)])}
            if hist == "kin":
                extra["kin_hist"] = states                     # [T,B,H,4,d]: `kda_replay` masks by n_keep
                new_state = state                              # (replayed to the accepted prefix by the rollback)
            else:
                extra["state_hist"] = jnp.concatenate([state.astype(states.dtype)[None], states])   # [T+1,B,H,dk,dv]
        else:
            core, new_state = kda_recurrent(q, k, v, g, beta, state)
    else:
        core, new_state = kda_chunked(q, k, v, g, beta, state)
    return core, {"conv": new_conv, "state": new_state, **extra}


def kda_out(p, core, gate, cfg: Cfg, dtype):
    """Gated RMSNorm over hd (fp32, sigmoid gate) + output projection (TP-reduced): -> y [B,T,D]."""
    B, T, H, hd = gate.shape
    o = core.astype(jnp.float32)
    o = o * lax.rsqrt(jnp.mean(o * o, -1, keepdims=True) + cfg.eps) * p["o_norm"].astype(jnp.float32)
    o = o * jax.nn.sigmoid(gate.astype(jnp.float32))
    return cfg.reduce(o.astype(dtype).reshape(B, T, H * hd) @ p["o"])


KDA_KERNEL = False         # the KDA layers' recurrent core of decode / short-verify calls (at most KDA_MAX_TOKENS tokens
KDA_MAX_TOKENS = 8         # per row; silu, the norms, the gates, the delta-rule recurrence, the gated output norm) as
KDA_INTERPRET = False      # glm53.pallas_kda kernels on the TPU (elsewhere only with KDA_INTERPRET); the conv taps and
                           # the projections stay XLA. Read at trace time. hist=True (the state form) stays XLA.
                           # OFF: v6e 2026-10-05 (j143 / j150) the kernel costs ~9 us per layer call (96 lane-to-
                           # sublane transposes of [128, 128] per call: three per token and head) against ~10 us for
                           # XLA's chain (the while loop + one fusion), and the conv / gate glue left outside grew:
                           # one-stream DFlash step 13.42 -> 13.75 ms, plain B = 3 14.06 -> 14.98. Its exact-f32
                           # contractions also moved the logits by ~2 % of the max (XLA's einsums round to bf16 there).


def kda_kernel_active(n_tokens):
    """Whether the KDA core kernel serves a recurrent call of n_tokens per row on this backend."""
    return bool(KDA_KERNEL) and n_tokens <= KDA_MAX_TOKENS and (KDA_INTERPRET or jax.default_backend() == "tpu")


def _kda_attention_kernel(p, x, cfg: Cfg, cache, hist):
    """kda_attention's recurrent step as the glm53.pallas_kda kernel: x [B,T,D], cache {"conv": [B,K-1,3C], "state":
    [B,H,dk,dv]} -> (y [B,T,D], new cache (+ "kin_hist" [T,B,H,4,d] and "conv_hist" [T+1,B,K-1,3C] with hist)). The
    conv runs in f32 into the kernel (XLA's fused chain keeps f32 there too)."""
    from glm53 import pallas_kda as PK
    B, T, _ = x.shape
    H, hd = cfg.kda_heads, cfg.kda_hd
    qkv = jnp.concatenate([x @ p["q"], x @ p["k"], x @ p["v"]], -1)                                 # [B,T,3C]
    conv_w = jnp.concatenate([p["conv_q"], p["conv_k"], p["conv_v"]], 0)
    xc, new_conv = causal_conv1d(qkv, conv_w, cache["conv"], None, out_dtype=jnp.float32)
    f = (x @ p["f_a"]) @ p["f_b"]
    gate = (x @ p["g_a"]) @ p["g_b"]
    y, new_state, kin = PK.kda_core_out(xc, f, x @ p["b"], gate, cache["state"], p["dt_bias"], p["A_log"], p["o_norm"],
                                        hd=hd, eps=cfg.eps, gate_lb=cfg.gate_lb, hist=bool(hist), interpret=KDA_INTERPRET)
    new_cache = {"conv": new_conv, "state": new_state}
    if hist:
        K = conv_w.shape[-1]
        full = jnp.concatenate([cache["conv"].astype(qkv.dtype), qkv], axis=1)
        new_cache["kin_hist"] = jnp.moveaxis(kin, 1, 0)
        new_cache["conv_hist"] = jnp.stack([full[:, t:t + K - 1] for t in range(T + 1)])
    return cfg.reduce(y.astype(x.dtype).reshape(B, T, H * hd) @ p["o"]), new_cache


def kda_attention(p, x, cfg: Cfg, cache=None, use_recurrent=False, length=None, hist=False):
    """x [B,T,D] (already input-normed). cache: None or {"conv": [B,K-1,3C], "state": [B,H,dk,dv]} (one cache, all
    rows in lockstep). Returns (y [B,T,D], new_cache); see kda_core for `length` / `hist`."""
    if use_recurrent and cache is not None and length is None and hist is not True and kda_kernel_active(x.shape[1]):
        return _kda_attention_kernel(p, x, cfg, cache, hist)
    proj = kda_project(p, x, cfg)
    core, new_cache = kda_core(proj, cfg, cache, use_recurrent, length, hist)
    return kda_out(p, core, proj["gate"], cfg, x.dtype), new_cache


def kda_attention_rows(p, x, cfg: Cfg, caches, hist=False):
    """Recurrent (decode) step of B INDEPENDENT streams: x [B,T,D], caches[b] = row b's own cache (batch dim 1).
    The projections and the output run batched, the conv/recurrence per row -> (y [B,T,D], [new cache per row]);
    hist=True (T > 1: a speculative verify of every row) adds each row's per-token histories (kda_core)."""
    if hist is not True and kda_kernel_active(x.shape[1]):              # the kernel per row (its own cache / state)
        ys, new = [], []
        for b, c in enumerate(caches):
            y, nc = _kda_attention_kernel(p, x[b:b + 1], cfg, c, hist)
            ys.append(y); new.append(nc)
        return jnp.concatenate(ys, 0), new
    proj = kda_project(p, x, cfg)
    cores, new = [], []
    for b, c in enumerate(caches):
        pr = {k: (v if k == "conv_w" else v[b:b + 1]) for k, v in proj.items()}
        core, nc = kda_core(pr, cfg, c, True, None, hist)
        cores.append(core); new.append(nc)
    return kda_out(p, jnp.concatenate(cores, 0), proj["gate"], cfg, x.dtype), new


def layernorm(x, w, b, eps):
    xf = x.astype(jnp.float32)
    mu = xf.mean(-1, keepdims=True)
    var = jnp.mean((xf - mu) ** 2, -1, keepdims=True)
    return ((xf - mu) * lax.rsqrt(var + eps) * w.astype(jnp.float32) + b.astype(jnp.float32)).astype(x.dtype)


# ----------------------------------------------------------------------------- DSA indexer (k-pool)
# Cache layouts. Replicated (cfg.seq_shard == 1): slot t of the MLA/indexer caches holds position t.
# Sequence-sharded over n = cfg.seq_shard chips, interleaved by k-pool: position t belongs to pool p = t // kp, lives
# on chip p % n at local slot (p // n) * kp + t % kp. Every chip therefore holds an equal share of complete pools, a
# prefill piece scatters T/n rows per chip, and the indexer scores its own pools with all heads.
def _pos_to_slot(t, kp, n):
    """Global position(s) -> (owner chip, local slot)."""
    pool = t // kp
    return pool % n, (pool // n) * kp + t % kp


def _slot_to_pos(s, kp, n, chip):
    return ((s // kp) * n + chip) * kp + s % kp


def cache_write(cache, new, pos0, cfg: Cfg, chip=None, valid=None):
    """Write new [B,T,R] at global positions pos0.. into cache [B,S_local,R] (see layouts above). `valid` [T] bool
    (optional) keeps the cache untouched for rows that are False."""
    new = new.astype(cache.dtype)
    n, kp = cfg.seq_shard, cfg.idx_kpool
    if n == 1 and valid is None:
        return lax.dynamic_update_slice_in_dim(cache, new, pos0, axis=1)
    B, T = new.shape[:2]
    S_local = cache.shape[1]
    if n == 1:
        mine, slot = jnp.ones((T,), bool), pos0 + jnp.arange(T)
    else:
        owner, slot = _pos_to_slot(pos0 + jnp.arange(T), kp, n)
        mine = owner == chip
    if valid is not None:
        mine = mine & valid
    if T <= 2:                                     # decode: read-select-write the row(s), no scatter
        for t in range(T):
            cur = lax.dynamic_slice_in_dim(cache, slot[t], 1, axis=1)
            cache = lax.dynamic_update_slice_in_dim(cache, jnp.where(mine[t], new[:, t:t + 1], cur), slot[t], axis=1)
        return cache
    # one scatter: rows of other chips (or invalid rows) go to distinct out-of-bounds slots, which XLA drops
    slot = jnp.where(mine, slot, S_local + jnp.arange(T))
    return cache.at[:, slot].set(new, mode="drop", unique_indices=True)


def mla_cache_shapes(cfg: Cfg, B, S, has_indexer, n=1):
    """Global cache shapes of one MLA layer for capacity S (per chip: the sequence/pool axis divided by n):
    "c" latent [B,S,kvr] (+ "cs" [B,S] per-token scales when cfg.cache_q8: "c" then holds the int8 rows packed four
    to an int32 word, [B,S,kvr/4] — `pack_latent`); with an indexer "pk"
    pooled indexer keys [B,S/kp,ihd] (one row per complete k-pool) and the raw keys/gates of the incomplete tail
    pool "tk"/"tg" [B,kp,ihd] (replicated, tiny). Dtypes: `cache_dtype`."""
    kp = cfg.idx_kpool
    out = {"c": (B, S // n, cfg.kv_lora // 4 if cfg.cache_q8 else cfg.kv_lora)}
    if cfg.cache_q8:
        out["cs"] = (B, S // n)
    if has_indexer:
        out.update(pk=(B, S // kp // n, cfg.idx_hd), tk=(B, kp, cfg.idx_hd), tg=(B, kp, cfg.idx_hd))
    return out


ROW_KEYS = ("c", "cs", "pk")         # positional cache entries (rows per token / per pool); the rest are tail buffers


def cache_dtype(cfg: Cfg, key, dtype):
    """dtype of one MLA cache entry: int8 latents (packed in int32 words) + f32 scales under cache_q8, else the model
    dtype."""
    if cfg.cache_q8 and key == "c":
        return jnp.int32
    if key == "cs":
        return jnp.float32
    return dtype


def pack_latent(q):
    """int8 rows [..., kvr] -> int32 words [..., kvr/4] (the q8 latent cache's storage): word j holds the bytes of
    elements j, j + kvr/4, j + kvr/2, j + 3 kvr/4 (byte k = element j + k kvr/4). An int8 cache has (32,128) tiles
    while XLA writes and reads single rows in (8,128) tiles, so every program that touched it copied the whole
    per-layer cache (16.8 MB per chip at 262k) in and out (v6e xprof 2026-10-04: ~1.7 ms per DFlash step); 32-bit
    words have one natural tiling, and the strided byte order lets `unpack_latent` rebuild a row from four 128-lane
    blocks with shifts (a bitcast to int8 needed a relayout of every gathered row). Values are unchanged."""
    w = q.shape[-1] // 4
    b = q.reshape(q.shape[:-1] + (4, w)).astype(jnp.int32) & 0xFF
    return b[..., 0, :] | (b[..., 1, :] << 8) | (b[..., 2, :] << 16) | (b[..., 3, :] << 24)


def unpack_latent(w):
    """int32 words [..., kvr/4] (`pack_latent`) -> the int8 values [..., kvr], as int32 (sign-extended bytes)."""
    return jnp.concatenate([(w << (24 - 8 * k)) >> 24 for k in range(4)], -1)


def latent_dequant(q, s, dtype):
    """int8 latent rows [..., kvr] (or their int32 packing, `pack_latent`) x per-row scales [...] -> dtype."""
    if q.dtype == jnp.int32:
        q = unpack_latent(q)
    return (q.astype(jnp.float32) * s.astype(jnp.float32)[..., None]).astype(dtype)


def gather_scalars(a, idx):
    """a [B, S] (per-row latent scales), idx [B, Tq, W] in-range int32 -> a[b, idx] [B, Tq, W]: the 128-value rows
    holding them (a row gather, like the latent rows' own) and a lane select — exact. An element gather of scalars is
    slower on the TPU (v6e 2026-10-04, jobs/j127, gather + dequant per MLA layer: Tq = 1 34 -> 27 us, Tq = 4 67 -> 35
    us in isolation; but in the T = 4 verify the 4 MB of rows per layer made the step's other fusions slower: used
    for single-query steps only)."""
    B, S = a.shape
    if S % 128:
        return jnp.take_along_axis(a[:, None], idx, axis=2, mode="promise_in_bounds")
    rows = jnp.take_along_axis(a.reshape(B, 1, S // 128, 128), (idx // 128)[..., None], axis=2,
                               mode="promise_in_bounds")                                     # [B, Tq, W, 128]
    lane = jnp.arange(128, dtype=idx.dtype) == (idx % 128)[..., None]
    return jnp.sum(jnp.where(lane, rows, jnp.zeros((), rows.dtype)), -1)


def write_latent(cache, new_c, pos0, cfg: Cfg, chip=None):
    """New latents [B,T,kvr] into the cache at pos0 -> the updated {"c"} (or {"c", "cs"}: per-token absmax int8)."""
    if not cfg.cache_q8:
        return {"c": cache_write(cache["c"], new_c, pos0, cfg, chip)}
    x = new_c.astype(jnp.float32)
    s = jnp.maximum(jnp.max(jnp.abs(x), axis=-1), 1e-12) / 127.0                  # [B,T]
    q = jnp.clip(jnp.round(x / s[..., None]), -127, 127).astype(jnp.int8)
    return {"c": cache_write(cache["c"], pack_latent(q), pos0, cfg, chip), "cs": cache_write(cache["cs"], s, pos0, cfg, chip)}


def pool_new_keys(ix, tail_k, tail_g, new_k, new_g, pos0, cfg: Cfg, length, hist=False):
    """k-pool compression of the NEW tokens (the first `length` of new_k/new_g, at positions pos0..) together with
    the raw keys of the incomplete pool they may extend (tail_k/tail_g [B,kp,ihd]: row i = position (pos0//kp)*kp + i
    for i < pos0 % kp). -> (pooled [B,NP,ihd] f32, complete [NP] bool, pool0 = global id of pooled[0], new tails)."""
    B, T, ihd = new_k.shape
    kp = cfg.idx_kpool
    f32 = jnp.float32
    a = pos0 % kp                                                             # tokens of the tail pool already seen
    NP = (T + 2 * kp - 2) // kp + 1                                           # pools covering a + T rows (+1 spare)
    L = NP * kp

    def fill(tail, new):
        buf = jnp.zeros((B, L, ihd), f32).at[:, :kp].set(tail.astype(f32))
        return lax.dynamic_update_slice_in_dim(buf, new.astype(f32), a, axis=1)

    kb, gb = fill(tail_k, new_k), fill(tail_g, new_g)
    present = jnp.arange(L) < a + length                                      # [L] rows that hold real tokens
    logits = jnp.where(present[None, :, None], gb + jnp.tile(ix["ape"].astype(f32), (L // kp, 1))[None], -jnp.inf)
    probs = jnp.nan_to_num(jax.nn.softmax(logits.reshape(B, NP, kp, ihd), axis=2))
    pooled = (probs * kb.reshape(B, NP, kp, ihd)).sum(2)                      # [B,NP,ihd]
    complete = present.reshape(NP, kp).all(-1)
    start = ((a + length) // kp) * kp                                         # first row of the (incomplete) last pool
    tail_k2 = lax.dynamic_slice_in_dim(kb, start, kp, axis=1)
    tail_g2 = lax.dynamic_slice_in_dim(gb, start, kp, axis=1)
    if hist:                                                                  # tails after 0..T of the tokens
        st = [((a + t) // kp) * kp for t in range(T + 1)]                     # (entry 0 = before them)
        th_k = jnp.stack([lax.dynamic_slice_in_dim(kb, s_, kp, axis=1) for s_ in st])
        th_g = jnp.stack([lax.dynamic_slice_in_dim(gb, s_, kp, axis=1) for s_ in st])
        return pooled, complete, pos0 // kp, tail_k2, tail_g2, (th_k, th_g)
    return pooled, complete, pos0 // kp, tail_k2, tail_g2


def indexer_pool(ix, keys, gates, cfg: Cfg):
    """k-pool compression of ALL cached keys (query independent). keys/gates [B,S_local,ihd] ->
    (pool_keys [B,P_local,ihd] f32, pool_valid [P_local] bool). Pools straddling the end of a replicated cache
    whose size is not a multiple of kp are invalid (model-level calls with cap=None)."""
    B, S, ihd = keys.shape
    kp = cfg.idx_kpool
    P = -(-S // kp)
    pad = P * kp - S
    kpad = jnp.pad(keys, ((0, 0), (0, pad), (0, 0))).reshape(B, P, kp, ihd).astype(jnp.float32)
    gpad = jnp.pad(gates, ((0, 0), (0, pad), (0, 0))).reshape(B, P, kp, ihd).astype(jnp.float32)
    tok_valid = jnp.arange(P * kp).reshape(P, kp) < S                             # [P,kp]
    logits = jnp.where(tok_valid[None, :, :, None], gpad + ix["ape"].astype(jnp.float32)[None, None], -jnp.inf)
    probs = jnp.nan_to_num(jax.nn.softmax(logits, axis=2))
    return (probs * kpad).sum(2), tok_valid.all(-1)


def top_k(x, k, chunk=None):
    """`lax.top_k` over the last axis with a chunked first stage: the top k of every chunk of `chunk` values (default
    max(4k, 256)), then the top k of those candidates. Same values and ids as lax.top_k (the k largest are among their
    chunks' k largest; ties keep lower indices first), but much cheaper on the TPU, where a top_k over a long axis is
    one full sort (v6e 2026-10-04, jobs/j122: 8192 values k = 512: 37 -> 18 us; 4 x 8192 k = 128: 37 -> 30 us;
    3 x 19360 k = 16: 462 -> 26 us). Float inputs (the last chunk is padded with -inf)."""
    n = x.shape[-1]
    c = chunk or max(4 * k, 256)
    if n <= 2 * c or 2 * k > c:
        return lax.top_k(x, k)
    pad = (-n) % c
    if pad:
        x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)], constant_values=-jnp.inf)
    xc = x.reshape(x.shape[:-1] + (-1, c))
    v, i = lax.top_k(xc, k)                                                           # [..., n/c, k]
    i = i + (jnp.arange(xc.shape[-2], dtype=i.dtype) * c)[:, None]
    v, i = v.reshape(x.shape[:-1] + (-1,)), i.reshape(x.shape[:-1] + (-1,))
    v, j = lax.top_k(v, k)
    return v, jnp.take_along_axis(i, j, -1)


def indexer_select(scores, pool_valid, qpos, cfg: Cfg, chip=None):
    """scores [B,Tq,P_local] f32 (this chip's pools); qpos [Tq] (or [B,Tq]) global query positions -> (pools [B,Tq,K] global pool
    ids, valid [B,Tq,K]): the top-K pools per query (two-stage top-k across chips when the cache is sharded: local
    top-K, all-gather the candidates, global top-K)."""
    B, Tq, P_local = scores.shape
    n, kp = cfg.seq_shard, cfg.idx_kpool
    lo = jnp.finfo(jnp.float32).min
    j = jnp.arange(P_local)
    gpool = j * n + (chip if n > 1 else 0)                                            # global pool ids [P_local]
    pool_end = gpool * kp + kp - 1
    if qpos.ndim == 1:
        visible = (pool_end[None, :] <= qpos[:, None]) & pool_valid[None, :]          # [Tq,P_local]
        scores = jnp.where(visible[None], scores, lo)
    else:                                                                             # qpos [B,Tq]: positions per row
        scores = jnp.where((pool_end <= qpos[..., None]) & pool_valid, scores, lo)
    K = min(cfg.idx_topk // kp, P_local)
    if n == 1:
        vals, sel = top_k(scores, K)
        return sel, vals > lo
    off = chip

    def two_stage(k_local, L=P_local):
        """local top-k_local -> all-gather -> global top-K; also returns whether any chip's list was cut short of
        the global threshold (then the result may be inexact). L = the local pools that can be visible (static):
        only the first L are scored (the rest sit beyond every query's position)."""
        if k_local >= L:                           # every candidate pool goes to the global stage: no local sort
            v, j = scores[..., :L], jnp.broadcast_to(jnp.arange(L, dtype=jnp.int32), (B, Tq, L))
        else:
            v, j = top_k(scores[..., :L], k_local)                                    # [B,Tq,k_local]
        j = j * n + off
        both = lax.all_gather(jnp.stack([lax.bitcast_convert_type(v, jnp.int32), j]), cfg.tp_axis, axis=-1,
                              tiled=True)                                             # one collective, not two
        va, ca = lax.bitcast_convert_type(both[0], jnp.float32), both[1]              # [B,Tq,n*k_local]
        kk = min(cfg.idx_topk // kp, n * k_local)
        neg, ca = lax.sort((-va, ca), dimension=-1, num_keys=1)                       # descending by score, ids carried
        vs, ss = -neg[..., :kk], ca[..., :kk]
        thr = vs[..., -1:]                                                            # weakest globally selected score
        cut = (k_local < K) & jnp.any(v[..., -1:] > thr)                              # a chip had more above the bar
        cut = lax.pmax(cut.astype(jnp.int32), cfg.tp_axis) > 0
        return vs, ss, cut

    k_fast = min(cfg.topk_local, K) if Tq > 1 else K       # a single query (decode) gains nothing from the fallback dance

    def select(L):
        vs, ss, cut = two_stage(k_fast, L)
        if k_fast < K:
            vs, ss = lax.cond(cut, lambda _: two_stage(K, L)[:2], lambda _: (vs, ss), None)
        return vs, ss

    # the local pools a query at position q can see are the first (q + 1) // (n * kp) + 1 (interleaved layout): the
    # sorts run over that bucket of the P_local scores, not all of them (a 262k cache scores 8,192 pools per chip at
    # any context; the TPU-v5e paper's "blocks sized for the max context" cost). Every chip takes the same branch.
    Ls = [L for L in cfg.select_buckets if L < P_local] + [P_local]
    if len(Ls) == 1:
        vs, ss = select(P_local)
    else:
        nv = (jnp.max(qpos) + 1) // (n * kp) + 1
        idx = sum((nv > L).astype(jnp.int32) for L in Ls[:-1])
        vs, ss = lax.switch(idx, [functools.partial(select, L) for L in Ls])
    return ss, vs > lo


def select_tokens(pools, valid, qpos, cfg: Cfg):
    """Selected pools -> global token indices [B,Tq,W] int32 (-1 = invalid), W = idx_topk + kp - 1 padded to a
    multiple of 16 (a tile-aligned slot count keeps the gathered keys free of relayout copies): the pools' tokens plus
    the incomplete tail pool of the visible prefix (positions 0..qpos)."""
    B, Tq, K = pools.shape
    kp, S = cfg.idx_kpool, cfg.cache_cap
    tok = (pools[..., None] * kp + jnp.arange(kp)).reshape(B, Tq, K * kp)
    tok = jnp.where(jnp.repeat(valid, kp, axis=-1), tok, -1)
    tok = jnp.pad(tok, ((0, 0), (0, 0), (0, cfg.idx_topk - K * kp)), constant_values=-1)
    vis_count = qpos + 1
    tail_count = vis_count % kp
    tail_start = vis_count - tail_count
    off = jnp.arange(kp - 1)
    tail = tail_start[:, None] + off[None, :]                                         # [Tq,kp-1]
    tail_ok = (off[None, :] < tail_count[:, None]) & (tail < S)
    tail = jnp.where(tail_ok, tail, -1)
    tok = jnp.concatenate([tok, jnp.broadcast_to(tail[None], (B, Tq, kp - 1))], -1)
    W = tok.shape[-1]
    align = 32 if cfg.cache_q8 else 16                                              # int8 rows tile by 32 sublanes
    return jnp.pad(tok, ((0, 0), (0, 0), (0, (-W) % align)), constant_values=-1).astype(jnp.int32)


def union_pools(pools, valid, qpos, cfg: Cfg, chip=None):
    """Per-block key deduplication. pools/valid [B,Tq,K] global pool ids -> (member [B,Tq,U] bool, upool [U] local
    pool ids of the block's union (sentinel P_local past the end), count): the union over the block's queries of the
    selected pools that live on this chip (plus each query's incomplete tail pool), compacted into U slots."""
    B, Tq, K = pools.shape
    n, kp = cfg.seq_shard, cfg.idx_kpool
    P_local = -(-(cfg.cache_cap // n) // kp)                                           # pools per chip (last may be partial)
    U = min(cfg.union_pools, P_local)
    tail_pool = jnp.broadcast_to((qpos // kp)[None, :, None], (B, Tq, 1))
    tail_on = jnp.broadcast_to(((qpos + 1) % kp != 0)[None, :, None], (B, Tq, 1))
    cand = jnp.concatenate([pools, tail_pool], -1)                                    # [B,Tq,K+1] global pools
    ok = jnp.concatenate([valid, tail_on], -1)
    if n > 1:
        ok = ok & (cand % n == chip)
        cand = cand // n
    local = jnp.where(ok, cand, P_local)                                              # sentinel = P_local
    srt = jnp.sort(local.reshape(-1))
    first = jnp.concatenate([jnp.ones((1,), jnp.bool_), srt[1:] != srt[:-1]]) & (srt < P_local)
    count = first.sum()
    # the U smallest unique pool ids (top-k of the negated keys; a scatter compaction costs more on TPU)
    keys = jnp.where(first, -srt, -P_local)
    if keys.shape[0] < U:
        keys = jnp.pad(keys, (0, U - keys.shape[0]), constant_values=-P_local)
    upool = -lax.top_k(keys, U)[0]                                                    # ascending union, sentinel P_local
    member = (local[..., :, None] == upool[None, None, None, :]).any(-2) & (upool < P_local)[None, None]   # [B,Tq,U]
    return member, upool, count, U


def _attend(q, keys, mask, cfg: Cfg, per_query_keys):
    """Softmax attention of q [B,Tq,Hq,r] over keys ([B,Tq,W,r] if per_query_keys else [B,S,r]) restricted to
    `mask` ([B,Tq,W] or [Tq,S]). With a sharded cache Hq = all heads and every chip holds a slice of the keys:
    partial softmax with the global max, then reduce-scatter over heads -> [B,Tq,H_local,r]."""
    lo = jnp.finfo(jnp.float32).min
    scale = cfg.qk_nope ** -0.5
    if per_query_keys:
        s = jnp.einsum("bthr,btwr->bhtw", q, keys).astype(jnp.float32) * scale
        s = jnp.where(mask[:, None], s, lo)
    else:
        s = jnp.einsum("bthr,bsr->bhts", q, keys).astype(jnp.float32) * scale
        mask = mask if mask.ndim == 3 else mask[None]                                 # [B or 1, Tq, S]
        s = jnp.where(mask[:, None], s, lo)
    pv = "bhtw,btwr->bthr" if per_query_keys else "bhts,bsr->bthr"
    if cfg.seq_shard == 1:
        probs = jax.nn.softmax(s, -1).astype(keys.dtype)
        return jnp.einsum(pv, probs, keys)
    m = lax.pmax(s.max(-1), cfg.tp_axis)                                              # [B,Hq,Tq] global max
    e = jnp.exp(s - m[..., None])                                                     # 0 where masked / no local keys
    l = jnp.sum(e, -1)                                                                # [B,Hq,Tq]
    o = jnp.einsum(pv, e.astype(keys.dtype), keys, preferred_element_type=jnp.float32)   # [B,Tq,Hq,r]
    o = lax.psum_scatter(o, cfg.tp_axis, scatter_dimension=2, tiled=True)             # [B,Tq,H_local,r]
    l = lax.psum_scatter(l, cfg.tp_axis, scatter_dimension=1, tiled=True)             # [B,H_local,Tq]
    return (o / jnp.swapaxes(l, 1, 2)[..., None]).astype(keys.dtype)


# ----------------------------------------------------------------------------- MLA (NoPE)
def mla_project(p, x, cfg: Cfg):
    """Position-free, batched part of an MLA layer: x [B,T,D] -> dict(q_lat [B,T,H,kvr] absorbed queries, new_c
    [B,T,kvr] new latents; with an indexer also nk / ng [B,T,ihd] (its keys / gates), qi [B,T,IH,ihd] f32 and
    wts [B,T,IH] f32 (its queries / head weights))."""
    B, T, _ = x.shape
    H, qk, dv, kvr = cfg.n_heads, cfg.qk_nope, cfg.v_hd, cfg.kv_lora
    # T > 1 (a verify): XLA folds the reshape into a dot batched over heads and copies the dequantized q_b / wq_b into
    # that layout on every call; a barrier keeps the plain 2-D product (v6e 2026-10-04: DFlash step -0.17 ms)
    bar = (lambda a: a) if T == 1 else lax.optimization_barrier
    q_resid = rmsnorm(x @ p["q_a"], p["q_a_norm"], cfg.eps)
    q = bar(q_resid @ p["q_b"]).reshape(B, T, H, qk)
    new_c = rmsnorm(x @ p["kv_a"], p["kv_a_norm"], cfg.eps)                         # [B,T,kvr]
    kv_b = p["kv_b"].reshape(kvr, H, qk + dv)
    Wk = kv_b[:, :, :qk]
    out = {"q_lat": jnp.einsum("bthq,rhq->bthr", q, Wk), "new_c": new_c}            # absorbed query [B,T,H,kvr]
    if "indexer" in p:
        ix = p["indexer"]
        IH, ihd = cfg.idx_heads, cfg.idx_hd
        out["nk"] = layernorm(x @ ix["wk"], ix["k_norm_w"], ix["k_norm_b"], 1e-6)
        out["ng"] = x @ ix["gate"]
        out["qi"] = bar(q_resid @ ix["wq_b"]).reshape(B, T, IH, ihd).astype(jnp.float32)
        out["wts"] = (x @ ix["weights_proj"]).astype(jnp.float32) * (IH ** -0.5)     # [B,T,IH]
    return out


def _mla_write(p, proj, cfg: Cfg, cache, pos0, chip, length, hist):
    """`mla_core`'s cache update (cfg.cache_cap set) -> new cache: the new latents at pos0; with an indexer its pooled
    keys and tails (+ the tails after each token with hist)."""
    new_cache = write_latent(cache, proj["new_c"], pos0, cfg, chip)
    if "indexer" in p:
        ix = p["indexer"]
        res = pool_new_keys(ix, cache["tk"], cache["tg"], proj["nk"], proj["ng"], pos0, cfg, length, hist)
        pooled, complete, pool0, tk2, tg2 = res[:5]
        pcfg = dataclasses.replace(cfg, idx_kpool=1)                             # pool-level slots: pool % n, pool // n
        new_cache["pk"] = cache_write(cache["pk"], pooled, pool0, pcfg, chip, valid=complete)
        new_cache["tk"], new_cache["tg"] = tk2.astype(cache["tk"].dtype), tg2.astype(cache["tg"].dtype)
        if hist:
            new_cache["tk_hist"] = res[5][0].astype(cache["tk"].dtype)
            new_cache["tg_hist"] = res[5][1].astype(cache["tg"].dtype)
    return new_cache


def _indexer_scores(qi, wts, pool_keys, cfg: Cfg):
    """The indexer's score of every (query, local pool): qi [B,Tq,IH,ihd], wts [B,Tq,IH], pool_keys [B,P,ihd] f32
    -> [B,Tq,P]."""
    sc = jax.nn.relu(jnp.einsum("bthd,bpd->bthp", qi, pool_keys) * (cfg.idx_hd ** -0.5))
    return jnp.einsum("bth,bthp->btp", wts, sc)


def _per_query_keys(c_all, cs_all, pools, pvalid, qpos, cfg: Cfg, chip, dtype):
    """The latents of every query's selected tokens that live on this chip (cfg.cache_cap set): (keys [B,Tq,W,kvr],
    valid [B,Tq,W])."""
    n, kp = cfg.seq_shard, cfg.idx_kpool
    tok = select_tokens(pools, pvalid, qpos, cfg)                                  # [B,Tq,W] global positions
    valid = (tok >= 0) & (tok <= qpos[None, :, None])
    if n > 1:
        owner, slot = _pos_to_slot(jnp.maximum(tok, 0), kp, n)
        valid = valid & (owner == chip)
    else:
        slot = tok
    slot = jnp.where(valid, slot, 0)
    c_sel = jnp.take_along_axis(c_all[:, None], slot[..., None], axis=2, mode="promise_in_bounds")
    if cfg.cache_q8:
        # the scales [B,Tq,W]: 128-value rows + a lane select for one query (decode, `gather_scalars`); by pools for
        # a verify (`_token_scales`: the 4 x 2080 element gathers were 50.8 us per MLA layer, 4 x 520 pool rows 9.7)
        if slot.shape[1] == 1:
            s_sel = gather_scalars(cs_all, slot)
        else:
            s_sel = _token_scales(cs_all, pools, pvalid, qpos, cfg, chip, slot.shape[2])
        c_sel = latent_dequant(c_sel, s_sel, dtype)
    return c_sel, valid


def _token_scales(cs_all, pools, pvalid, qpos, cfg: Cfg, chip, W):
    """The q8 scales of `select_tokens`' slots, [B,Tq,W] in its order (the K pools' tokens pool-major, zeros up to
    idx_topk, the tail pool's first kp-1 tokens, zeros up to W), gathered by POOLS: `_pool_scales` over the K + 1
    pools reads the 128-value row holding each pool's kp consecutive scales and selects them by lane — 4x fewer
    gather rows than the tokens' scales one by one (v6e 2026-10-05, j148: 9.7 vs 50.8 us per MLA layer of a T = 4
    verify). Invalid or foreign pools read local pool 0 (their slots are masked by `valid`)."""
    B, Tq, K = pools.shape
    n, kp = cfg.seq_shard, cfg.idx_kpool
    tail = jnp.broadcast_to((qpos // kp)[None, :, None], (B, Tq, 1))                   # the query's incomplete pool
    gpool = jnp.concatenate([pools, tail], -1)                                           # [B,Tq,K+1] global pool ids
    ok = jnp.concatenate([pvalid, jnp.ones((B, Tq, 1), bool)], -1)
    if n > 1:
        ok = ok & (gpool % n == chip)
        gpool = gpool // n
    sc = _pool_scales(cs_all, jnp.where(ok, gpool, 0), kp)                               # [B,Tq,K+1,kp]
    parts = [sc[:, :, :K].reshape(B, Tq, K * kp)]
    if K * kp < cfg.idx_topk:
        parts.append(jnp.zeros((B, Tq, cfg.idx_topk - K * kp), sc.dtype))
    parts.append(sc[:, :, K, :kp - 1])
    s = jnp.concatenate(parts, -1)
    return jnp.pad(s, ((0, 0), (0, 0), (0, W - s.shape[-1])))


POOL_GATHER_WINDOWS = True    # `_per_query_pool_keys` gathers (kp, kvr) windows (lax.gather) instead of rows of a
                              # pool-major reshape of the cache (read at trace time)


def _per_query_pool_keys(c_all, cs_all, pools, pvalid, qpos, cfg: Cfg, chip, dtype):
    """`_per_query_keys` with the latents gathered by whole k-pools: one gathered row of kp tokens per selected pool
    (plus the query's incomplete tail pool) instead of one row per token — the same keys and per-token validity, in
    pool-major order, from 4x fewer gather rows (a TPU gather pays per row, not per byte). The q8 scales come as the
    128-value rows holding each pool's kp consecutive scales plus a lane select (`gather_scalars`). -> (keys
    [B,Tq,W,kvr], valid [B,Tq,W]) with W = (K + 1) * kp padded to a multiple of 32."""
    B, Tq, K = pools.shape
    n, kp = cfg.seq_shard, cfg.idx_kpool
    S_local, R = c_all.shape[1], c_all.shape[2]
    P_local = -(-S_local // kp)
    tail = (qpos // kp)[None, :, None]                                                # [1,Tq,1] the query's own pool
    tail_on = ((qpos + 1) % kp != 0)[None, :, None]                                   # incomplete: not in pk, its tokens <= qpos
    gpool = jnp.concatenate([pools, jnp.broadcast_to(tail, (B, Tq, 1))], -1)          # [B,Tq,K+1] global pool ids
    ok = jnp.concatenate([pvalid, jnp.broadcast_to(tail_on, (B, Tq, 1))], -1)
    if n > 1:
        ok = ok & (gpool % n == chip)
        local = gpool // n
    else:
        local = gpool
    local = jnp.where(ok, local, 0)
    tok = gpool[..., None] * kp + jnp.arange(kp, dtype=gpool.dtype)                   # [B,Tq,K+1,kp] global positions
    valid = (ok[..., None] & (tok <= qpos[None, :, None, None])).reshape(B, Tq, (K + 1) * kp)
    if POOL_GATHER_WINDOWS and S_local % kp == 0:
        # (kp, R) windows straight from the token-major cache: no pool-major reshape (a relayout copy of the whole
        # per-layer cache per call on the TPU)
        bidx = jnp.broadcast_to(jnp.arange(B, dtype=jnp.int32)[:, None, None], local.shape)
        start = jnp.stack([bidx, local * kp], -1)                                                    # [B,Tq,K+1,2]
        dn = lax.GatherDimensionNumbers(offset_dims=(3, 4), collapsed_slice_dims=(0,), start_index_map=(0, 1))
        rows = lax.gather(c_all, start, dn, slice_sizes=(1, kp, R), mode="promise_in_bounds")        # [B,Tq,K+1,kp,R]
        c_sel = rows.reshape(B, Tq, (K + 1) * kp, R)
        if cfg.cache_q8:                     # (a (1, kp) window gather of the scales is 6x slower than this, j148)
            c_sel = latent_dequant(c_sel, _pool_scales(cs_all, local, kp).reshape(B, Tq, (K + 1) * kp), dtype)
        W = c_sel.shape[2]
        pad = (-W) % 32
        if pad:
            c_sel = jnp.pad(c_sel, ((0, 0), (0, 0), (0, pad), (0, 0)))
            valid = jnp.pad(valid, ((0, 0), (0, 0), (0, pad)))
        return c_sel, valid
    c_pools = jnp.pad(c_all, ((0, 0), (0, P_local * kp - S_local), (0, 0))) if S_local % kp else c_all
    c_view = c_pools.reshape(B, P_local, kp * R)
    rows = jnp.take_along_axis(c_view[:, None], local[..., None], axis=2, mode="promise_in_bounds")   # [B,Tq,K+1,kp*R]
    c_sel = rows.reshape(B, Tq, (K + 1) * kp, R)
    if cfg.cache_q8:
        c_sel = latent_dequant(c_sel, _pool_scales(cs_all, local, kp).reshape(B, Tq, (K + 1) * kp), dtype)
    W = c_sel.shape[2]
    pad = (-W) % 32
    if pad:
        c_sel = jnp.pad(c_sel, ((0, 0), (0, 0), (0, pad), (0, 0)))
        valid = jnp.pad(valid, ((0, 0), (0, 0), (0, pad)))
    return c_sel, valid


def _pool_scales(cs_all, local, kp):
    """The kp consecutive per-token scales of every pool in `local` [B,Tq,P] from cs_all [B,S_local] -> [B,Tq,P,kp]:
    the 128-value rows holding them (a row gather) and a lane select per position (`gather_scalars`' way; v6e
    2026-10-05, j148: 9.7 us per MLA layer at 4 x 520 pools vs 50.8 for the element gather of the 4 x 2080 scalars)."""
    B, S_local = cs_all.shape
    Tq, P = local.shape[1:]
    if S_local % 128 or 128 % kp:
        slot = (local[..., None] * kp + jnp.arange(kp)).reshape(B, Tq, P * kp)
        return jnp.take_along_axis(cs_all[:, None], jnp.minimum(slot, S_local - 1), axis=2,
                                   mode="promise_in_bounds").reshape(B, Tq, P, kp)
    srows = jnp.take_along_axis(cs_all.reshape(B, 1, S_local // 128, 128), (local * kp // 128)[..., None], axis=2,
                                mode="promise_in_bounds")                                              # [B,Tq,P,128]
    lane = jnp.arange(128, dtype=jnp.int32)
    base = (local * kp % 128)[..., None]
    return jnp.stack([jnp.sum(jnp.where(lane == base + t, srows, jnp.zeros((), srows.dtype)), -1) for t in range(kp)], -1)


def _query_keys(c_all, cs_all, pools, pvalid, qpos, cfg: Cfg, chip, dtype):
    """The per-query key gather of a block: by pools (`_per_query_pool_keys`) from cfg.pool_gather queries, else by
    tokens (`_per_query_keys`)."""
    f = _per_query_pool_keys if cfg.pool_gather and pools.shape[1] >= cfg.pool_gather else _per_query_keys
    return f(c_all, cs_all, pools, pvalid, qpos, cfg, chip, dtype)


def _attend_rows(q, keys, masks, cfg: Cfg):
    """`_attend` with per-query keys for B rows whose keys stay separate (keys[b] [1,Tq,W,r], masks[b] [1,Tq,W]; no
    concatenation of the gathered latents): the scores and products per row, the cross-chip max and the two
    reduce-scatters once for all rows. Per row the same operations as `_attend`."""
    lo = jnp.finfo(jnp.float32).min
    scale = cfg.qk_nope ** -0.5
    pv = "bhtw,btwr->bthr"
    ss = [jnp.where(m[:, None], jnp.einsum("bthr,btwr->bhtw", q[b:b + 1], k).astype(jnp.float32) * scale, lo)
          for b, (k, m) in enumerate(zip(keys, masks))]
    if cfg.seq_shard == 1:
        return jnp.concatenate([jnp.einsum(pv, jax.nn.softmax(s, -1).astype(k.dtype), k) for s, k in zip(ss, keys)], 0)
    m = lax.pmax(jnp.concatenate([s.max(-1) for s in ss], 0), cfg.tp_axis)          # [B,Hq,Tq] global max
    es = [jnp.exp(s - m[b:b + 1, ..., None]) for b, s in enumerate(ss)]
    l = jnp.concatenate([jnp.sum(e, -1) for e in es], 0)                             # [B,Hq,Tq]
    o = jnp.concatenate([jnp.einsum(pv, e.astype(k.dtype), k, preferred_element_type=jnp.float32)
                         for e, k in zip(es, keys)], 0)                              # [B,Tq,Hq,r]
    o = lax.psum_scatter(o, cfg.tp_axis, scatter_dimension=2, tiled=True)
    l = lax.psum_scatter(l, cfg.tp_axis, scatter_dimension=1, tiled=True)
    return (o / jnp.swapaxes(l, 1, 2)[..., None]).astype(keys[0].dtype)


ROWS_SHARED_MIN = 3        # mla_core_rows shares the collectives from this many queries over all rows (B*T; 0 = never;
                           # read at trace). v6e 2026-10-04 (j135, same engine, shared vs per row): plain decode B=2
                           # 12.53 vs 12.19 ms, B=3 14.90 vs 15.67; DFlash step of 2 / 3 streams 22.86 / 32.5 vs
                           # 23.47 / 32.8


def mla_core_rows(p, proj, cfg: Cfg, caches, pos, sparse="auto", cap=None, hist=False):
    """`mla_core` for B INDEPENDENT rows (caches[b] = row b's own cache, batch 1, at position pos[b]: traced scalars)
    -> (o_lat [B,T,H_local,kvr], [new cache per row]). On the DSA path of decode and short verifies (sparse, at most
    per_query_max queries) every row updates and gathers from its own cache, but the query all-gather, the indexer's
    selection (top-k, the candidates' all-gather) and the softmax combine across the chips run once for all rows:
    latency-bound collectives and sorts (~10-20 us each, about six per row and MLA layer; v6e 2026-10-04). Otherwise
    (one row, fewer than `ROWS_SHARED_MIN` queries in all: two decode rows ran faster apart) `mla_core` per row."""
    B = len(caches)
    q_lat = proj["q_lat"]
    T = q_lat.shape[1]
    n = cfg.seq_shard
    S = caches[0]["c"].shape[1] * n
    use_sparse = sparse == "always" or (sparse == "auto" and S > cfg.idx_topk)
    if (B == 1 or not ROWS_SHARED_MIN or B * T < ROWS_SHARED_MIN
            or not (use_sparse and "indexer" in p and T <= min(cfg.per_query_max, cfg.q_block))):
        outs, new = [], []
        for b, (c, pb) in enumerate(zip(caches, pos)):
            o_b, nc = mla_core(p, {k: v[b:b + 1] for k, v in proj.items()}, cfg, c, pb, sparse, cap, None, hist)
            outs.append(o_b); new.append(nc)
        return jnp.concatenate(outs, 0), new
    chip = lax.axis_index(cfg.tp_axis) if n > 1 else None
    cfg = dataclasses.replace(cfg, cache_cap=S)
    dtype = proj["new_c"].dtype
    new, scores, qpos = [], [], []
    for b, (c, pb) in enumerate(zip(caches, pos)):
        pr = {k: v[b:b + 1] for k, v in proj.items()}
        nc = _mla_write(p, pr, cfg, c, pb, chip, T, hist)
        new.append(nc)
        scores.append(_indexer_scores(pr["qi"], pr["wts"], nc["pk"].astype(jnp.float32), cfg))
        qpos.append(pb + jnp.arange(T))
    qpos = jnp.stack(qpos)                                                          # [B,T]
    pools, pvalid = indexer_select(jnp.concatenate(scores, 0), jnp.ones((new[0]["pk"].shape[1],), bool), qpos, cfg,
                                   chip)                                            # [B,T,K]
    kv = [_query_keys(nc["c"], nc.get("cs"), pools[b:b + 1], pvalid[b:b + 1], qpos[b], cfg, chip, dtype)
          for b, nc in enumerate(new)]
    q = lax.all_gather(q_lat, cfg.tp_axis, axis=2, tiled=True) if n > 1 else q_lat   # all heads [B,T,H*n,kvr]
    return _attend_rows(q, [k for k, _ in kv], [v for _, v in kv], cfg), new


def mla_core(p, proj, cfg: Cfg, cache=None, pos0=0, sparse=None, cap=None, length=None, hist=False):
    """Position-dependent part of an MLA layer for ONE cache (all rows in lockstep at pos0): cache update, DSA
    selection and attention -> (o_lat [B,T,H_local,kvr], new_cache). Arguments as in `mla_attention`."""
    q_lat, new_c = proj["q_lat"], proj["new_c"]
    B, T, H, kvr = q_lat.shape
    n, kp = cfg.seq_shard, cfg.idx_kpool
    chip = lax.axis_index(cfg.tp_axis) if n > 1 else None
    has_ix = "indexer" in p
    q8 = cfg.cache_q8
    if cache is None:
        S = T if cap is None else cap
        assert S % (kp * n) == 0 or n == 1, (S, kp, n)
        S_alloc = -(-S // kp) * kp if n == 1 else S                                # a whole number of pools
        cache = {k: jnp.zeros(sh, cache_dtype(cfg, k, new_c.dtype)) for k, sh in mla_cache_shapes(cfg, B, S_alloc, has_ix, n).items()}
        if n == 1 and S_alloc != S:
            cache["c"] = cache["c"][:, :S]
            if q8:
                cache["cs"] = cache["cs"][:, :S]
    S = cache["c"].shape[1] * n                                                    # global capacity
    cfg = dataclasses.replace(cfg, cache_cap=S)
    new_cache = _mla_write(p, proj, cfg, cache, pos0, chip, T if length is None else length, hist)
    c_all = new_cache["c"]                                                         # [B,S_local,kvr] (int8 under q8)
    cs_all = new_cache.get("cs")                                                   # [B,S_local] scales under q8
    dtype = new_c.dtype
    qpos = pos0 + jnp.arange(T)
    use_sparse = sparse == "always" or (sparse == "auto" and S > cfg.idx_topk)
    if use_sparse:
        pool_keys = new_cache["pk"].astype(jnp.float32)                            # [B,P_local,ihd] cached pooled keys
        pool_valid = jnp.ones((pool_keys.shape[1],), bool)                         # only complete pools are written;
        qi, wts = proj["qi"], proj["wts"]                                          # later ones are masked by position
    if n > 1 and not use_sparse:
        kpos = _slot_to_pos(jnp.arange(c_all.shape[1]), kp, n, chip)               # global position per local slot
    elif not use_sparse:
        kpos = jnp.arange(S)
    if use_sparse:                                                                 # pool-major view for the key-dedup path (one relayout per call)
        S_local = c_all.shape[1]
        P_local = -(-S_local // kp)
        c_pools = jnp.pad(c_all, ((0, 0), (0, P_local * kp - S_local), (0, 0))) if S_local % kp else c_all
        c_view = c_pools.reshape(B, P_local, kp * c_all.shape[-1])                   # (kvr/4 int32 words under q8)
        if q8:
            cs_pools = jnp.pad(cs_all, ((0, 0), (0, P_local * kp - S_local))) if S_local % kp else cs_all
            cs_view = cs_pools.reshape(B, P_local, kp)

    def per_query_attend(q_b, pools, pvalid, qpos_b):
        """Per-query gather of the selected keys that live on this chip (the pre-dedup path; decode + fallback)."""
        c_sel, valid = _query_keys(c_all, cs_all, pools, pvalid, qpos_b, cfg, chip, dtype)
        return _attend(q_b, c_sel, valid, cfg, True)

    def block(q_lat_b, qi_b, wts_b, qpos_b):
        """One block of queries -> o_lat [B,Tq,H_local,kvr]."""
        if n > 1:
            q_lat_b = lax.all_gather(q_lat_b, cfg.tp_axis, axis=2, tiled=True)      # all heads [B,Tq,H*n,kvr]
        if use_sparse:
            sc = _indexer_scores(qi_b, wts_b, pool_keys, cfg)                       # [B,Tq,P_local]
            pools, pvalid = indexer_select(sc, pool_valid, qpos_b, cfg, chip)      # [B,Tq,K] global pools
            if qpos_b.shape[0] <= cfg.per_query_max or cfg.union_pools == 0:       # decode / spec verify / dedup off
                return per_query_attend(q_lat_b, pools, pvalid, qpos_b)             # (no union bookkeeping, no conds)
            member, upool, count, U = union_pools(pools, pvalid, qpos_b, cfg, chip)
            over = count > U
            if n > 1:
                over = lax.pmax(over.astype(jnp.int32), cfg.tp_axis) > 0          # every chip takes the same branch
            def dedup(args):
                q_b, member, upool = args
                idx = jnp.minimum(upool, P_local - 1)
                keys = c_view.at[:, idx].get(mode="promise_in_bounds")                # [B,U,kp*kvr]
                keys = keys.reshape(B, U * kp, c_all.shape[-1])                     # the union's tokens, pool-major
                if q8:
                    keys = latent_dequant(keys, cs_view.at[:, idx].get(mode="promise_in_bounds").reshape(B, U * kp), dtype)
                slots = (upool[:, None] * kp + jnp.arange(kp)[None, :]).reshape(-1)
                kpos_u = _slot_to_pos(slots, kp, n, chip) if n > 1 else slots
                mask = jnp.repeat(member, kp, axis=-1) & (kpos_u[None, None, :] <= qpos_b[None, :, None])
                return _attend(q_b, keys, mask, cfg, False)

            per_query = lambda args: per_query_attend(args[0], pools, pvalid, qpos_b)
            return lax.cond(over, per_query, dedup, (q_lat_b, member, upool))
        c_dense = latent_dequant(c_all, cs_all, dtype) if q8 else c_all
        return _attend(q_lat_b, c_dense, kpos[None, :] <= qpos_b[:, None], cfg, False)

    Tq = min(cfg.q_block, T)
    nb = -(-T // Tq)
    if nb == 1:
        o_lat = block(q_lat, qi if use_sparse else None, wts if use_sparse else None, qpos)
    else:
        Tpad = nb * Tq - T
        padq = lambda a: jnp.pad(a, ((0, 0), (0, Tpad)) + ((0, 0),) * (a.ndim - 2)) if Tpad else a
        blk = lambda a: jnp.moveaxis(padq(a).reshape((B, nb, Tq) + a.shape[2:]), 1, 0)   # [nb,B,Tq,...]
        qpos_blk = jnp.pad(qpos, (0, Tpad)).reshape(nb, Tq)
        xs = (blk(q_lat), blk(qi) if use_sparse else None, blk(wts) if use_sparse else None, qpos_blk)
        o_lat = lax.map(lambda t: block(*t), xs)                                    # [nb,B,Tq,H,kvr]
        o_lat = jnp.moveaxis(o_lat, 0, 1).reshape(B, nb * Tq, H, kvr)[:, :T]
    return o_lat, new_cache


def mla_out(p, o_lat, cfg: Cfg):
    """o_lat [B,T,H_local,kvr] -> value up-projection + output projection (TP-reduced) [B,T,D]."""
    B, T, H, kvr = o_lat.shape
    qk, dv = cfg.qk_nope, cfg.v_hd
    Wv = p["kv_b"].reshape(kvr, cfg.n_heads, qk + dv)[:, :, qk:]
    o = jnp.einsum("bthr,rhv->bthv", o_lat, Wv).reshape(B, T, H * dv)
    return cfg.reduce(o @ p["o"])


def mla_attention(p, x, cfg: Cfg, cache=None, pos0=0, sparse=None, cap=None, length=None, hist=False):
    """Causal NoPE MLA with absorbed kv_b and a FIXED-CAPACITY latent cache (no recompiles as the sequence grows).
    x [B,T,D]. cache: None (prefill; allocates `cap` slots, or exactly T when cap is None) or
    {"c": [B,S_local,kvr], "pk": [B,P_local,ihd] pooled indexer keys, "tk"/"tg": [B,kp,ihd] tail}; new tokens are written at pos0 (may be
    traced). Slots beyond the current query position hold stale/zero data and are always masked by causality.
    sparse: None -> dense causal; "auto" -> DSA indexer when the (global) capacity > idx_topk; "always".
    Queries are processed in blocks of cfg.q_block (lax.map) so the per-query temporaries (indexer scores over all
    pools, the gathered top-k keys) never scale with the prompt length. cfg.seq_shard > 1: cache sharded over the
    chips (see the layout notes above), queries all-gathered over heads, partial softmax combined across chips.
    = mla_project (batched, position-free) -> mla_core (one cache, all rows at pos0) -> mla_out."""
    proj = mla_project(p, x, cfg)
    o_lat, new_cache = mla_core(p, proj, cfg, cache, pos0, sparse, cap, length, hist)
    return mla_out(p, o_lat, cfg), new_cache


def mla_attention_rows(p, x, cfg: Cfg, caches, pos, sparse="auto", cap=None, hist=False):
    """Decode step of B INDEPENDENT streams: x [B,T,D]; caches[b] = row b's own cache (batch dim 1) at position
    pos[b] (traced scalars). Projections and the output run batched, the cache updates and key gathers per row, the
    DSA selection's and the attention's collectives batched (`mla_core_rows`) -> (y [B,T,D], [new cache per row]);
    hist=True (T > 1, a verify per row): the pool-tail histories."""
    proj = mla_project(p, x, cfg)
    o_lat, new = mla_core_rows(p, proj, cfg, caches, pos, sparse, cap, hist)
    return mla_out(p, o_lat, cfg), new


# ----------------------------------------------------------------------------- MoE
def moe_route(p, x, cfg: Cfg):
    """x [N,D] -> (topk_idx [N,k] int32, topk_w [N,k] fp32). n_group == 1 (no group masking)."""
    logits = x.astype(jnp.float32) @ p["router_w"].astype(jnp.float32)
    scores = jax.nn.sigmoid(logits)
    choice = scores + p["router_bias"].astype(jnp.float32)
    _, idx = lax.top_k(choice, cfg.topk)
    # one-hot pick instead of take_along_axis: XLA's generic gather costs ~2.5 ms per call on TPU v5e
    w = jnp.sum(jax.nn.one_hot(idx, scores.shape[-1], dtype=scores.dtype) * scores[:, None, :], axis=-1)
    if cfg.norm_topk:
        w = w / (w.sum(-1, keepdims=True) + 1e-20)
    return idx.astype(jnp.int32), w * cfg.scaling


def moe_apply(x, gate_up_w, down_w, w, limit):
    """x [N,D]; gate_up_w [N,k,D,2mi]; down_w [N,k,mi,D]; w [N,k] -> [N,D]."""
    gu = jnp.einsum("nd,nkdm->nkm", x, gate_up_w)
    mi = gu.shape[-1] // 2
    h = swiglu_clamped(gu[..., :mi], gu[..., mi:], limit)
    y = jnp.einsum("nkm,nkmd->nkd", h, down_w)
    return jnp.einsum("nkd,nk->nd", y.astype(jnp.float32), w).astype(x.dtype)


def moe_apply_grouped(x, gu_w, dn_w, idx_local, w, limit, chunk=256):
    """Grouped experts: x [N,D]; gu_w [Eh,D,2mi]; dn_w [Eh,mi,D]; idx_local [N,k] (ids into Eh, -1 = none);
    w [N,k]. Every token is run through every gathered expert (dense, masked) — Eh/k× extra FLOPs but no per-token
    weight copies, so prefill fits HBM. Token chunks of `chunk` bound the [N,Eh,mi] intermediate."""
    N, D = x.shape
    Eh = gu_w.shape[0]
    mi = dn_w.shape[1]
    # routing weights as a dense [N, Eh] matrix
    onehot = jax.nn.one_hot(jnp.maximum(idx_local, 0), Eh, dtype=jnp.float32) * (idx_local >= 0)[..., None]
    rw = jnp.einsum("nk,nke->ne", w.astype(jnp.float32), onehot)            # [N,Eh]
    outs = []
    for s0 in range(0, N, chunk):
        xs, rws = x[s0:s0 + chunk], rw[s0:s0 + chunk]
        gu = jnp.einsum("nd,edm->nem", xs, gu_w)                              # [n,Eh,2mi]
        h = swiglu_clamped(gu[..., :mi], gu[..., mi:], limit)
        h = jnp.where(rws[..., None] > 0, h.astype(jnp.float32) * rws[..., None], 0.0).astype(x.dtype)   # fold routing weights in
        outs.append(jnp.einsum("nem,emd->nd", h, dn_w))
    return jnp.concatenate(outs, 0) if len(outs) > 1 else outs[0]


def moe_dense(p, x, cfg: Cfg, fetch=None, layer=0):
    """MoE block for x [B,T,D]. `fetch(layer, idx)` returns (gate_up_w [N,k,D,2mi], down_w [N,k,mi,D]) for
    idx [N,k]; default = gather from the on-device tables p["gate_up"] [E,D,2mi], p["down"] [E,mi,D].
    Under TP the expert/shared outputs are partial sums; one reduction covers both."""
    B, T, D = x.shape
    xf = x.reshape(-1, D)
    idx, w = moe_route(p, xf, cfg)
    if fetch is not None and hasattr(fetch, "apply"):      # resident quantized experts (glm53.resident)
        y = fetch.apply(p, xf, idx, w, layer, cfg.swiglu_limit, seq_tokens=T).astype(xf.dtype)
    else:
        if fetch is None:
            gu_w, dn_w = p["gate_up"][idx], p["down"][idx]
        else:
            gu_w, dn_w = fetch(layer, idx)
        y = moe_apply(xf, gu_w, dn_w, w, cfg.swiglu_limit)
    y = cfg.reduce(y + mlp_partial(p["shared"], xf, cfg))
    return y.reshape(B, T, D)


# ----------------------------------------------------------------------------- layer / model
MHC_KERNEL = "bf16"        # "bf16" / "f32": the mHC sites of decode and short-verify layer calls (at most
MHC_MAX_TOKENS = 16        # MHC_MAX_TOKENS = B*T tokens) run as glm53.pallas_mhc kernels with that precision for the
MHC_INTERPRET = False      # two small contractions, on the TPU (elsewhere only with MHC_INTERPRET); None = XLA's
                           # chain. Read at trace time. v6e 2026-10-04 (j139, j136b): plain decode B=1 10.47 -> 9.65 ms,
                           # B=3 15.28 -> 14.14, DFlash 3 streams 30.99 -> 29.68, one-stream DFlash device 14.29 -> 13.56


def mhc_active():
    """Whether the mHC kernels serve decode / short-verify layer calls on this backend."""
    return bool(MHC_KERNEL) and (MHC_INTERPRET or jax.default_backend() == "tpu")


def _mhc(streams):
    B, T = streams.shape[:2]
    return MHC_KERNEL if mhc_active() and B * T <= MHC_MAX_TOKENS else None


def decoder_layer(p, streams, cfg: Cfg, ltype, mtype, cache=None, pos0=0, use_recurrent=False, fetch=None,
                  sparse="auto", layer=0, cap=None, length=None, hist=False):
    kern = _mhc(streams)
    post, comb, h = _site_pre(p["hc_attn"], p["ln1"], streams, cfg, kern)
    if ltype == "linear_attention":
        h, new_cache = kda_attention(p["attn"], h, cfg, cache, use_recurrent, length, hist)
    else:
        h, new_cache = mla_attention(p["attn"], h, cfg, cache, pos0, sparse, cap, length, hist)
    streams, post, comb, h = _site_post_pre(post, comb, h, streams, p["hc_ffn"], p["ln2"], cfg, kern)
    if mtype == "sparse":
        h = moe_dense(p["mlp"], h, cfg, fetch, layer)
    else:
        h = mlp(p["mlp"], h, cfg)
    return _site_post(post, comb, h, streams, kern), new_cache


def _site_pre(p_hc, ln, streams, cfg: Cfg, kern):
    """A layer's first mHC site: (post, comb, normed block input). With `kern` (the pallas_mhc path) comb is flat."""
    if kern:
        from glm53 import pallas_mhc as PMH
        return PMH.mhc_pre(streams, p_hc, ln, cfg, kern, MHC_INTERPRET)
    post, comb, h = hc_pre(p_hc, streams, cfg)
    return post, comb, rmsnorm(h, ln, cfg.eps)


def _site_post_pre(post, comb, y, streams, p_hc, ln, cfg: Cfg, kern):
    """The middle site: the attention block's residual update, then the MLP block's pre."""
    if kern:
        from glm53 import pallas_mhc as PMH
        return PMH.mhc_post_pre(post, comb, y, streams, p_hc, ln, cfg, kern, MHC_INTERPRET)
    streams = hc_post(post, comb, y, streams)
    return (streams,) + _site_pre(p_hc, ln, streams, cfg, None)


def _site_post(post, comb, y, streams, kern):
    if kern:
        from glm53 import pallas_mhc as PMH
        return PMH.mhc_post(post, comb, y, streams, MHC_INTERPRET)
    return hc_post(post, comb, y, streams)


def decoder_layer_rows(p, streams, cfg: Cfg, ltype, mtype, caches, pos, fetch=None, sparse="auto", layer=0, cap=None,
                       hist=False):
    """Recurrent step of B INDEPENDENT streams: streams [B,T,hc,D] (T = 1: decode; T > 1 with hist=True: every row
    verifies its own T-token block); caches[b] = row b's own cache (batch dim 1), pos[b] its position (traced
    scalar). Attention runs per row on its own cache, everything else (mHC, projections, MoE: T consecutive tokens
    per row) batched over the rows. Returns (streams, [new cache per row], with the *_hist entries when hist)."""
    kern = _mhc(streams)
    post, comb, h = _site_pre(p["hc_attn"], p["ln1"], streams, cfg, kern)
    if ltype == "linear_attention":
        h, new = kda_attention_rows(p["attn"], h, cfg, caches, hist)
    else:
        h, new = mla_attention_rows(p["attn"], h, cfg, caches, pos, sparse, cap, hist)
    streams, post, comb, h = _site_post_pre(post, comb, h, streams, p["hc_ffn"], p["ln2"], cfg, kern)
    h = moe_dense(p["mlp"], h, cfg, fetch, layer) if mtype == "sparse" else mlp(p["mlp"], h, cfg)
    return _site_post(post, comb, h, streams, kern), new


def layer_pre(p, streams, cfg: Cfg, ltype, mtype, cache=None, pos0=0, use_recurrent=False, sparse="auto", cap=None,
              length=None):
    """First half of a decoder layer up to (and including) MoE routing. Returns
    (streams_after_attn, h [B,T,D] normed MLP input, post, comb, new_cache, idx [N,k] or None, w [N,k] or None)."""
    post, comb, h = hc_pre(p["hc_attn"], streams, cfg)
    h = rmsnorm(h, p["ln1"], cfg.eps)
    if ltype == "linear_attention":
        h, new_cache = kda_attention(p["attn"], h, cfg, cache, use_recurrent, length)
    else:
        h, new_cache = mla_attention(p["attn"], h, cfg, cache, pos0, sparse, cap, length)
    streams = hc_post(post, comb, h, streams)
    post, comb, h = hc_pre(p["hc_ffn"], streams, cfg)
    h = rmsnorm(h, p["ln2"], cfg.eps)
    idx = w = None
    if mtype == "sparse":
        idx, w = moe_route(p["mlp"], h.reshape(-1, h.shape[-1]), cfg)
    return streams, h, post, comb, new_cache, idx, w


def layer_post(p, streams, h, post, comb, cfg: Cfg, mtype, gu_w=None, dn_w=None, w=None, idx_local=None):
    """Second half: MLP / experts (weights already gathered for this layer's routing) + residual update.
    If idx_local is given the experts are grouped ([Eh,...] tables + local ids), else per-token ([N,k,...])."""
    B, T, D = h.shape
    if mtype == "sparse":
        xf = h.reshape(-1, D)
        if idx_local is not None:
            y = moe_apply_grouped(xf, gu_w, dn_w, idx_local, w, cfg.swiglu_limit)
        else:
            y = moe_apply(xf, gu_w, dn_w, w, cfg.swiglu_limit)
        y = cfg.reduce(y + mlp_partial(p["mlp"]["shared"], xf, cfg)).reshape(B, T, D)
    else:
        y = mlp(p["mlp"], h, cfg)
    return hc_post(post, comb, y, streams)


HIST_KEYS = {"state_hist": "state", "conv_hist": "conv", "tk_hist": "tk", "tg_hist": "tg", "kin_hist": "state"}


def split_hist(cache):
    """cache dict with *_hist entries -> (cache without them, hist dict or None)."""
    hist = {k: cache[k] for k in cache if k in HIST_KEYS}
    return {k: v for k, v in cache.items() if k not in HIST_KEYS}, (hist or None)


def rollback_states(hists, n_keep, dtypes, cur=None):
    """Speculative verify of T tokens produced `hists` (per layer: the recurrent/tail state after 0..T of the tokens
    (entry 0 = before them), or None); return per layer the small state entries as after the first `n_keep` (traced,
    0..T; 0 undoes the whole verify) tokens (the positional cache rows beyond are overwritten later). `dtypes`: per
    layer {key: dtype} of the cache entries. A "kin_hist" (the KDA tokens' update inputs, T entries) replays the
    accepted tokens from cur[layer]["state"] (the state the verify left unchanged)."""
    out = []
    for i, (h, dt) in enumerate(zip(hists, dtypes)):
        if not h:
            out.append(None); continue
        o = {}
        for hk, k in HIST_KEYS.items():
            if hk not in h:
                continue
            if hk == "kin_hist":
                o[k] = kda_replay(cur[i][k], h[hk], n_keep).astype(dt[k])
            else:
                o[k] = lax.dynamic_index_in_dim(h[hk], n_keep, axis=0, keepdims=False).astype(dt[k])
        out.append(o)
    return out


def mtp_layer(p, embeds, h_prev, cfg: Cfg, cache=None, pos0=0, fetch=None, sparse="auto", cap=None, length=None,
              layer=None):
    """GLM-5 MTP (NextN) block (vLLM `glm5next/nvidia/mtp.py`): x = eh_proj(cat(rmsnorm(embeds, enorm),
    rmsnorm(h_prev, hnorm))) — embeds [B,T,D] of the token AFTER each position (zero at position 0), h_prev [B,T,D]
    the main model's post-norm hidden at that position — then one plain pre-norm MLA + MoE block (no
    hyper-connections) and the shared-head norm. Returns (h [B,T,D] for the shared lm_head, new attention cache)."""
    x = jnp.concatenate([rmsnorm(embeds.astype(cfg.dtype), p["enorm"], cfg.eps),
                         rmsnorm(h_prev.astype(cfg.dtype), p["hnorm"], cfg.eps)], -1) @ p["eh_proj"]
    h = rmsnorm(x, p["ln1"], cfg.eps)
    a, new_cache = mla_attention(p["attn"], h, cfg, cache, pos0, sparse, cap, length)
    x = x + a
    h = rmsnorm(x, p["ln2"], cfg.eps)
    x = x + moe_dense(p["mlp"], h, cfg, fetch, layer)
    return rmsnorm(x, p["head_norm"], cfg.eps), new_cache


def forward(params, tokens, cfg: Cfg, caches=None, pos0=0, use_recurrent=False, fetch=None, sparse="auto",
            embeds=None, cap=None, length=None):
    """tokens [B,T] -> (hidden [B,T,D] after final norm, new_caches list). Prefill: caches=None.
    `embeds` [B,T,D] overrides the embedding lookup (used by the TP engine with a vocab-sharded table)."""
    x = params["embed"][tokens].astype(cfg.dtype) if embeds is None else embeds.astype(cfg.dtype)
    streams = jnp.broadcast_to(x[:, :, None, :], x.shape[:2] + (cfg.hc, x.shape[-1]))
    new_caches = []
    for i in range(cfg.n_layers):
        c = None if caches is None else caches[i]
        streams, nc = decoder_layer(params["layers"][i], streams, cfg, cfg.layer_types[i], cfg.mlp_types[i],
                                    c, pos0, use_recurrent, fetch, sparse, layer=i, cap=cap, length=length)
        new_caches.append(nc)
    h = rmsnorm(streams.mean(2).astype(cfg.dtype), params["norm"], cfg.eps)
    return h, new_caches


def logits(params, h):
    return h @ params["lm_head"]
