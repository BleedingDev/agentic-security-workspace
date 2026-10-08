"""Pallas TPU kernel for a KDA layer's recurrent core at decode / short-verify shapes (B rows x T tokens, T <= 8).

After the conv taps (XLA, f32) it does what model.kda_core + model.kda_out do up to the output projection, in ONE
kernel per layer call instead of ~10 small fusions and a while loop: silu, the per-head l2 norms of q and k, the forget
gate g and beta from their projections, the T-token delta-rule recurrence on the f32 state [H, dk, dv] and the gated
RMS norm of the outputs. hist=True: also the tokens' update inputs ("kin", [B, T, H, 4, d]: k, v, g, beta) for the
rollback replay, and the state comes back as BEFORE the block (model.kda_core's hist="kin" contract).

Layout: everything is one grid step per row with whole arrays in VMEM (the state is 4 x 64 KB per chip); token t's
vectors are taken out of the [T, W] inputs by a masked sublane reduction (exact), a per-head vector becomes a matrix
with constant rows by broadcast + transpose ([d, d] f32: what the recurrence multiplies the state by), so no sublane
slicing of packed dtypes is needed. The arithmetic follows model.py op by op in f32; the sums of the two state
contractions run in a different order than XLA's reductions, so results agree to rounding, not bit for bit. On the TPU
the head dim must be a multiple of 128 (lane-aligned head slices); in interpret mode any shape works."""
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _row(x, t):
    """Row t of a small [T, W] f32 value as [1, W] (exact: one nonzero term per column)."""
    sub = lax.broadcasted_iota(jnp.int32, x.shape, 0)
    return jnp.sum(jnp.where(sub == t, x, 0.0), axis=0, keepdims=True)


def _lane(x, j):
    """Lane j of a small [N, W] f32 value as [N, 1]."""
    lane = lax.broadcasted_iota(jnp.int32, x.shape, 1)
    return jnp.sum(jnp.where(lane == j, x, 0.0), axis=1, keepdims=True)


def _colmat(r, n):
    """[1, n] row -> [n, n] with M[i, j] = r[i] (row i constant): the matrix the state is scaled / contracted with."""
    return jnp.transpose(jnp.broadcast_to(r, (n, n)))


def _rows(*rows):
    """[1, n] f32 rows -> [len(rows), n]."""
    n = rows[0].shape[1]
    sub = lax.broadcasted_iota(jnp.int32, (len(rows), n), 0)
    out = jnp.zeros((len(rows), n), jnp.float32)
    for i, r in enumerate(rows):
        out = jnp.where(sub == i, jnp.broadcast_to(r, (len(rows), n)), out)
    return out


def _softplus(x):
    return jnp.maximum(x, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(x)))      # (= jax.nn.softplus's logaddexp(x, 0))


def _kernel(x_ref, f_ref, b_ref, gate_ref, s_ref, dtb_ref, on_ref, alog_ref, y_ref, ns_ref, *kin_refs, H, hd, T, eps,
            gate_lb, hist):
    f32 = jnp.float32
    C = H * hd
    x = x_ref[...].astype(f32)                       # [T, 3C]: the conv output (q | k | v), before silu
    x = x * jax.nn.sigmoid(x)                        # silu
    f = f_ref[...].astype(f32) + dtb_ref[...]        # [T, C] forget-gate pre-activations + dt_bias
    bp = b_ref[...].astype(f32)                      # [T, H] beta pre-activations
    gt = gate_ref[...].astype(f32)                   # [T, C] output-gate pre-activations
    onorm = on_ref[...]                              # [1, hd]
    states = [s_ref[h] for h in range(H)]            # [hd, hd] f32 each (rows = dk)
    sub = lax.broadcasted_iota(jnp.int32, (T, C), 0)
    y = jnp.zeros((T, C), f32)
    for t in range(T):
        xt, ft, bt, gtt = _row(x, t), _row(f, t), _row(bp, t), _row(gt, t)
        outs = []
        for h in range(H):
            lo, hi = h * hd, (h + 1) * hd
            q, k, v = xt[:, lo:hi], xt[:, C + lo:C + hi], xt[:, 2 * C + lo:2 * C + hi]
            q = q / jnp.sqrt(jnp.sum(q * q, axis=1, keepdims=True) + 1e-6) * (hd ** -0.5)      # model.l2norm
            k = k / jnp.sqrt(jnp.sum(k * k, axis=1, keepdims=True) + 1e-6)
            dr = jnp.exp(alog_ref[h])                                                           # decay rate
            gh = ft[:, lo:hi]
            g = gate_lb * jax.nn.sigmoid(dr * gh) if gate_lb is not None else -dr * _softplus(gh)   # model.kda_gates
            beta = jax.nn.sigmoid(_lane(bt, h))                                                 # [1, 1]
            kmat = _colmat(k, hd)
            S = states[h] * _colmat(jnp.exp(g), hd)                                             # S[k, v] *= exp(g[k])
            kv = jnp.sum(S * kmat, axis=0, keepdims=True)                                       # k^T S  [1, dv]
            delta = (v - kv) * beta
            S = S + kmat * delta                                                                # + k (x) delta
            states[h] = S
            o = jnp.sum(S * _colmat(q, hd), axis=0, keepdims=True)                              # q^T S
            o = o * lax.rsqrt(jnp.mean(o * o, axis=1, keepdims=True) + eps) * onorm             # model.kda_out
            o = o * jax.nn.sigmoid(gtt[:, lo:hi])
            outs.append(o)
            if hist:
                kin_refs[0][t, h] = _rows(k, v, g, jnp.broadcast_to(beta, (1, hd)))
        row = outs[0] if H == 1 else jnp.concatenate(outs, axis=1)                             # [1, C]
        y = jnp.where(sub == t, jnp.broadcast_to(row, (T, C)), y)
    y_ref[...] = y.astype(y_ref.dtype)
    for h in range(H):
        ns_ref[h] = s_ref[h] if hist else states[h]  # hist: the state before the block (the rollback replays)


@functools.partial(jax.jit, static_argnames=("hd", "eps", "gate_lb", "hist", "interpret"))
def kda_core_out(x, f, bpre, gate, state, dt_bias, A_log, o_norm, *, hd, eps, gate_lb, hist=False, interpret=False):
    """x [B, T, 3C] (the conv output, any float dtype), f [B, T, C] = (x @ f_a) @ f_b, bpre [B, T, H] = x @ b, gate
    [B, T, C] = (x @ g_a) @ g_b, state [B, H, hd, hd] f32, dt_bias [C], A_log [H], o_norm [hd] -> (y [B, T, C] in
    gate's dtype = the gated, normed outputs before the o projection, new state [B, H, hd, hd] f32 (hist: the input
    state), kin [B, T, H, 4, hd] f32 or None)."""
    B, T, C3 = x.shape
    C = C3 // 3
    H = C // hd
    f32 = jnp.float32
    kern = functools.partial(_kernel, H=H, hd=hd, T=T, eps=eps, gate_lb=gate_lb, hist=hist)

    def row(*dims):
        return pl.BlockSpec((None,) + dims, lambda b: (b,) + (0,) * len(dims))
    in_specs = [row(T, C3), row(T, C), row(T, H), row(T, C), row(H, hd, hd),
                pl.BlockSpec((1, C), lambda b: (0, 0)), pl.BlockSpec((1, hd), lambda b: (0, 0)),
                pl.BlockSpec(memory_space=pltpu.SMEM)]
    out_shape = [jax.ShapeDtypeStruct((B, T, C), gate.dtype), jax.ShapeDtypeStruct((B, H, hd, hd), f32)]
    out_specs = [row(T, C), row(H, hd, hd)]
    if hist:
        out_shape.append(jax.ShapeDtypeStruct((B, T, H, 4, hd), f32))
        out_specs.append(row(T, H, 4, hd))
    outs = pl.pallas_call(kern, grid=(B,), out_shape=tuple(out_shape), in_specs=in_specs, out_specs=tuple(out_specs),
                          interpret=interpret)(
        x, f, bpre.astype(f32), gate, state.astype(f32), dt_bias.astype(f32).reshape(1, C),
        o_norm.astype(f32).reshape(1, hd), A_log.astype(f32))
    return outs[0], outs[1], (outs[2] if hist else None)
