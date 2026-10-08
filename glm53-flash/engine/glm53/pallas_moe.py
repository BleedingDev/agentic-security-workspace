"""Pallas TPU kernel: fused codebook-dequant + matvec of routed experts straight from the planar HBM tables.

`moe_matvec(planes, qtype, nblk, idx, X)` computes, for every expert slot i (one (token, expert) pair):
    out[i, r] = sum_n  W_{idx[i]}[r, n] * x_i[n]              (W dequantized on the fly, never written anywhere)
where X[i, w, c] = x_i[input(w, c)] is the per-word activation matrix of glm53.planes (see `planes.pm_x`).

Grid = expert slots; the expert id comes from a scalar-prefetch operand so each grid step DMAs exactly the planes
of the chosen expert (double-buffered by Pallas). Inside, one (8, 128) vreg of the qs plane = 8 packed words x 128
matrix rows; the codebook index of every group is looked up with in-vreg lane gathers (two codes per int32 table
word: one gather per 128-word table row + a select tree over the high index bits + a shift for the half), the values
come out of the code with shifts (IQ2_S: one int multiply builds the f32 bits), the sign bits are XORed into the f32
sign, and the values are multiplied-accumulated on the VPU against the activation broadcast along lanes:
acc[s, r] += v[s, r] * X[8b + s, c]. The lane broadcasts of X are computed once per expert slot into VMEM. The
sublane sum of the accumulator over all words is the output row block. `moe_matvec_gu` does gate and up of the same
slots in one kernel (one grid step per slot, the broadcasts shared). Everything is 32-bit (no packed dtypes), loads
are (n, 128) windows at 8-aligned sublane offsets, so the kernels also run under `interpret=True` on CPU for the tests.

Why this shape (speed plan §4/§12, measured on the v6e 2026-10-03 with jobs/j65): the decode kernel is bound by
cross-lane ops (each lane gather or lane broadcast ~1.5 ns per vreg) and then by VALU ops per weight; one gather per
128 table entries (8 for IQ2_S's 1024-entry grid) + a lane broadcast per weight vreg cost 15.7 us per expert, the
pair table + hoisted broadcasts + fused gate/up + pre-encoded values + XOR sign + one grid step per down slot 8.4 us,
with identical outputs. Mosaic in libtpu 0.0.42 has no multi-dimension in-vreg gather ("Zero or multiple gather
dimensions"), so a one-op 1024-entry lookup is not available.

Codebook tables: `code_table(qtype)` (see there).
"""
import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from glm53 import iqquant as Q
from glm53 import planes as PL
from glm53 import model as M

LANES = 128
SUB = 8


SIGN = 0x80000000
IQ2S_F32 = (0x41000000, 0x640000)          # f32 bits of the IQ2_S levels 8, 25, 43 = base + u * step, u = 0, 2, 3


@functools.lru_cache(None)
def code_table(qtype):
    """int32 [n // 256, 128]: the grid, TWO entries per int32 (entry 2w in the low 16 bits of word w, 2w + 1 in the
    high 16; word w at [w // 128, w % 128]). One field per weight k of the entry: IQ2_S stores its level lv in
    {0, 1, 2} as u = 0, 2, 3 in bits 2k..2k+1 (so the value's f32 bits are IQ2S_F32[0] + u * IQ2S_F32[1]), IQ3_S
    stores the value 2 lv + 1 in bits 4k..4k+3."""
    grid, levels = Q.GRIDS[qtype], Q.LEVELS[qtype]
    lv = np.searchsorted(levels, grid).astype(np.int64)                 # [n, w]
    assert np.array_equal(levels[lv], grid)
    if qtype == "IQ2_S":
        assert np.array_equal(levels, [8, 25, 43])
        f, fw = np.array([0, 2, 3])[lv], 2
    elif qtype == "IQ3_S":
        assert np.array_equal(levels, 2 * np.arange(8) + 1)
        f, fw = 2 * lv + 1, 4
    else:
        raise ValueError(qtype)
    code = np.zeros(grid.shape[0], np.int64)
    for k in range(grid.shape[1]):
        code |= f[:, k] << (fw * k)
    return (code[0::2] | (code[1::2] << 16)).astype(np.uint32).view(np.int32).reshape(-1, LANES)


@functools.lru_cache(None)
def kv16_table():
    t = np.zeros((1, LANES), np.float32)
    t[0, :16] = Q.KVALUES_IQ4NL
    return t


def _f16_bits_to_f32(h):
    """u32 holding f16 bits (low 16) -> f32 (normal + subnormal + zero; no inf/nan expected)."""
    h = h & 0xFFFF
    s = (h >> 15) & 1
    e = (h >> 10) & 31
    m = h & 1023
    bits = (s << 31) | ((e + 112) << 23) | (m << 13)
    normal = lax.bitcast_convert_type(bits.astype(jnp.uint32), jnp.float32)
    sub = m.astype(jnp.int32).astype(jnp.float32) * (2.0 ** -24)
    sub = jnp.where(s == 1, -sub, sub)
    return jnp.where(e == 0, sub, normal)


def _rows8(src, reps):
    """src (n, 128) with n * reps == 8 -> (8, 128) whose sublane s is src[s // reps]."""
    n = src.shape[0]
    if n == 1:
        return jnp.broadcast_to(src, (SUB, LANES))
    row = lax.broadcasted_iota(jnp.int32, (SUB, LANES), 0)
    out = jnp.broadcast_to(src[n - 1:n], (SUB, LANES))
    for r in range(n - 2, -1, -1):
        out = jnp.where(row < (r + 1) * reps, jnp.broadcast_to(src[r:r + 1], (SUB, LANES)), out)
    return out


def _sub_iota():
    return lax.broadcasted_iota(jnp.uint32, (SUB, LANES), 0)


_GATHER_DN = lax.GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(1,), start_index_map=(1,),
                                        operand_batching_dims=(0,), start_indices_batching_dims=(0,))


def _take(t, i):
    """In-vreg lane gather t[s, i[s, l]] (Mosaic's dynamic_gather; indices are in range, so no negative-index fix-up
    as jnp.take_along_axis adds)."""
    return lax.gather(t, i[..., None], _GATHER_DN, (1, 1), mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS)


def _lookup(tb, q, j, hw, hb):
    """Code of grid entry idx = (byte j of q) | (high index bits << 8), from `code_table`'s rows `tb` (each (8, 128),
    one table row broadcast to all sublanes): table word = idx >> 1 (low 7 bits = bits 1..7 of the byte, row = the high
    index bits, bit hb.. of hw), half = bit 0 of the byte. Returns int32 (8, 128) with the code in its low 16 bits (the
    high bits are the other entry or a sign extension; the decoders never read them)."""
    vals = [_take(t, ((q >> (8 * j + 1)) & 127).astype(jnp.int32)) for t in tb]
    bit = 0
    while len(vals) > 1:
        sel = (hw & (1 << (hb + bit))) != 0
        vals = [jnp.where(sel, vals[2 * m + 1], vals[2 * m]) for m in range(len(vals) // 2)]
        bit += 1
    half = ((q << 4) if j == 0 else (q >> (8 * j - 4))) & 16
    return vals[0] >> half.astype(jnp.int32)


def _signed(v, word, bit):
    """f32 v negated where bit `bit` of the u32 `word` is set (XOR into the f32 sign bit: shl, and, xor)."""
    m = (word << (31 - bit)) & jnp.uint32(SIGN)
    return lax.bitcast_convert_type(lax.bitcast_convert_type(v, jnp.uint32) ^ m, jnp.float32)


# ------------------------------------------------------------------------------------------ per-format word decoders
# `rows(name, r0, n)` returns rows r0..r0+n-1 of that plane for the current expert as an (n, 128) value (static lanes).
# Each decoder returns a list of (scale (8,128) f32, [(c, value (8,128) f32), ...]) for the 8 words 8b..8b+7.
def _decode_iq2_s(b, rows, tb):
    q = rows("qs", 8 * b, 8)
    sgw = rows("sg", 8 * b, 8)
    sub = _sub_iota()
    shift8 = (sub & 3) * 8
    qhw = (_rows8(rows("qh", 2 * b, 2), 4) >> shift8) & 255                 # qh byte of word 8b+s
    scw = (_rows8(rows("sc", 2 * b, 2), 4) >> shift8) & 255                 # sc byte of word 8b+s
    dw = rows("d", b // 2, 1) >> (16 * (b % 2))                             # block b for all 8 words
    dv = jnp.broadcast_to(_f16_bits_to_f32(dw), (SUB, LANES))
    scale = [dv * (0.5 + ((scw >> (4 * h)) & 15).astype(jnp.int32).astype(jnp.float32)) * 0.25 for h in range(2)]
    groups = [(scale[0], []), (scale[1], [])]
    base, step = IQ2S_F32
    for j in range(4):
        code = _lookup(tb, q, j, qhw, 2 * j)
        for k in range(8):
            bits = (code & (3 << (2 * k))) * (step >> (2 * k)) + base                    # u << 2k times step >> 2k
            v = lax.bitcast_convert_type(bits, jnp.float32)
            groups[j // 2][1].append((8 * j + k, _signed(v, sgw, 8 * j + k)))
    return groups


def _decode_iq3_s(b, rows, tb):
    q = rows("qs", 8 * b, 8)
    sub = _sub_iota()
    sgh = _rows8(rows("sg", 4 * b, 4), 2) >> ((sub & 1) * 16)              # the 16 sign bits of word 8b+s (low half)
    qhn = (_rows8(rows("qh", b, 1), 8) >> (sub * 4)) & 15                    # 4 high bits of word 8b+s
    scn = (_rows8(rows("sc", b // 2, 1), 8) >> ((4 * (b % 2) + (sub >> 1)) * 4)) & 15
    dw = rows("d", b // 4, 1) >> (16 * ((b // 2) % 2))
    dv = jnp.broadcast_to(_f16_bits_to_f32(dw), (SUB, LANES))
    scale = dv * (1.0 + 2.0 * scn.astype(jnp.int32).astype(jnp.float32))
    items = []
    for j in range(4):
        code = _lookup(tb, q, j, qhn, j)
        for k in range(4):
            v = ((code >> (4 * k)) & 15 if k else code & 15).astype(jnp.float32)     # 2 lv + 1
            items.append((4 * j + k, _signed(v, sgh, 4 * j + k)))
    return [(scale, items)]


def _decode_iq4_xs(b, rows, tb):
    q = rows("qs", 8 * b, 8)
    sub = _sub_iota()
    i = (2 * b + (sub >> 2)) % 8                                             # sub-block of word 8b+s within its block
    slw = (_rows8(rows("sl", b // 4, 1), 8) >> (4 * i)) & 15
    shw = (_rows8(rows("sh", b // 4, 1), 8) >> (2 * i)) & 3
    dw = rows("d", b // 8, 1) >> (16 * ((b // 4) % 2))
    dv = jnp.broadcast_to(_f16_bits_to_f32(dw), (SUB, LANES))
    scale = dv * ((slw | (shw << 4)).astype(jnp.int32) - 32).astype(jnp.float32)
    kv = tb[0]                                                               # (8,128) f32, entries at lanes 0..15
    items = []
    for pos in range(2):
        for j in range(4):
            nib = ((q >> (8 * j + 4 * pos)) & 15).astype(jnp.int32)
            items.append((4 * pos + j, jnp.take_along_axis(kv, nib, axis=1)))
    return [(scale, items)]


def _kq_scales(rows, b, sub):
    """K-quant scale bytes for the 8 words of vreg b (half n = b % 2 of block b // 2): per shift s an (8,128) u32 with
    the scale byte `is` = 8 n + 2 s + (sublane >= 4) of the block's 16, plus the block's d word (1,128)."""
    blk, n = b // 2, b % 2
    scw = rows("sc", 4 * blk, 4)                                                 # (4,128): the 16 scale bytes
    hi = (sub >= 4).astype(jnp.uint32)
    out = []
    for s_ in range(4):
        r = 2 * n + s_ // 2
        row = jnp.broadcast_to(scw[r:r + 1], (SUB, LANES))
        out.append((row >> (8 * (2 * (s_ % 2) + hi))) & 255)
    return out, rows("d", blk, 1)


def _decode_q2_k(b, rows, tb):
    q = rows("qs", 8 * b, 8)
    sub = _sub_iota()
    sc, dw = _kq_scales(rows, b, sub)
    d = jnp.broadcast_to(_f16_bits_to_f32(dw), (SUB, LANES))
    dmin = jnp.broadcast_to(_f16_bits_to_f32(dw >> 16), (SUB, LANES))
    items = []
    for s_ in range(4):
        dl = d * (sc[s_] & 15).astype(jnp.int32).astype(jnp.float32)
        ml = dmin * (sc[s_] >> 4).astype(jnp.int32).astype(jnp.float32)
        for j in range(4):
            v = ((q >> (8 * j + 2 * s_)) & 3).astype(jnp.int32).astype(jnp.float32)
            items.append((4 * s_ + j, dl * v - ml))
    return [(jnp.ones((SUB, LANES), jnp.float32), items)]


def _decode_q3_k(b, rows, tb):
    q = rows("qs", 8 * b, 8)
    sub = _sub_iota()
    blk, n = b // 2, b % 2
    hm = rows("hm", 8 * blk, 8)                                                  # (8,128): u32 q = hmask bytes 4q..4q+3
    sc, dw = _kq_scales(rows, b, sub)
    d = jnp.broadcast_to(_f16_bits_to_f32(dw), (SUB, LANES))
    items = []
    for s_ in range(4):
        dl = d * ((sc[s_] & 255).astype(jnp.int32) - 32).astype(jnp.float32)
        for j in range(4):
            v = ((q >> (8 * j + 2 * s_)) & 3).astype(jnp.int32)
            hb = ((hm >> (8 * j + 4 * n + s_)) & 1).astype(jnp.int32)
            items.append((4 * s_ + j, dl * (v - 4 + 4 * hb).astype(jnp.float32)))
    return [(jnp.ones((SUB, LANES), jnp.float32), items)]


DECODE = {"IQ2_S": _decode_iq2_s, "IQ3_S": _decode_iq3_s, "IQ4_XS": _decode_iq4_xs, "Q2_K": _decode_q2_k,
          "Q3_K": _decode_q3_k}


def _shared(*static):
    """jax.jit with these static arguments around a kernel entry point: every call site with the same configuration and
    shapes shares ONE trace and lowering (XLA inlines the call; same HLO). Without it each call site re-traced and
    re-lowered its kernel in Python — v6e 2026-10-04, jobs/j125: ~3.6 s per MoE layer per program for the per-slot
    kernels and ~25 s for the grouped ones (845 s for the five T = 4 verify programs), while the XLA/Mosaic compile
    of 1 or 4 identical kernels took the same time."""
    return lambda f: jax.jit(f, static_argnames=static)


def _table_arrays(qtype):
    if qtype == "IQ4_XS":
        return jnp.asarray(kv16_table())
    if qtype in ("Q2_K", "Q3_K"):
        return jnp.zeros((1, LANES), jnp.int32)                              # no codebook
    return jnp.asarray(code_table(qtype))


def _pick_rows(block, off, r0, n):
    """block (8, 128) value; rows off+r0 .. off+r0+n-1 where `off` is a traced scalar in [0, 8): select tree
    (Mosaic has no unaligned dynamic sublane loads)."""
    outs = []
    for r in range(n):
        want = off + r0 + r
        v = block[0:1]
        for q in range(1, SUB):
            v = jnp.where(want == q, block[q:q + 1], v)
        outs.append(v)
    return outs[0] if n == 1 else jnp.concatenate(outs, axis=0)


# ------------------------------------------------------------------------------------------------------- the kernel
def _table_rows(tbl_ref, qtype):
    """The decoders' `tb`: each table row broadcast to all sublanes (IQ4_XS: its one 16-entry row)."""
    n = 1 if qtype == "IQ4_XS" else tbl_ref.shape[0]
    return [jnp.broadcast_to(tbl_ref[r:r + 1, :], (SUB, LANES)) for r in range(n)]


def _fill_bx(x_ref, bx_ref, nb, C):
    """bx[b*C + c] = X[8b:8b+8, c] broadcast along lanes, once per expert slot (instead of once per weight vreg)."""
    for b in range(nb):
        xb = x_ref[SUB * b:SUB * (b + 1), :]
        for c in range(C):
            bx_ref[b * C + c] = jnp.broadcast_to(xb[:, c:c + 1], (SUB, LANES))


def _matvec_lanes(decode, rows, tb, bx_ref, nb, C):
    """sum_n W[r, n] x[n] for the 128 rows of one lane vreg -> (1, 128) f32."""
    out = jnp.zeros((SUB, LANES), jnp.float32)
    for b in range(nb):
        for scale, items in decode(b, rows, tb):
            acc = jnp.zeros((SUB, LANES), jnp.float32)
            for c, val in items:
                acc = acc + val * bx_ref[b * C + c]
            out = out + acc * scale
    return jnp.sum(out, axis=0, keepdims=True)


def _rows_fn(prefs, offs, lanes):
    """rows(plane, r0, n) of the current expert at `lanes` (planes with < 8 rows per expert: picked from their 8-row
    block with selects on the scalar-prefetched offset)."""
    def rows(k, r0, n):
        if offs[k] is None:
            return prefs[k][r0:r0 + n, lanes]
        return _pick_rows(prefs[k][0:SUB, lanes], offs[k], r0, n)
    return rows


def _offsets(keys, rows_p, e):
    return {k: (lax.rem(e * rows_p[k], SUB) if rows_p[k] % SUB else None) for k in keys}


@_shared("qtype", "nblk", "interpret", "lane_block")
def moe_matvec(planes, qtype, nblk, idx, X, *, interpret=False, lane_block=None):
    """planes: {k: u32 [E*rows_p, R] or [1, E*rows_p, R]}; idx int32 [Nk]; X f32 [Nk, W, C] -> f32 [Nk, R].

    Grid = (expert slot, lane block of `lane_block` matrix rows, default all rows up to 4096: one grid step per slot,
    every lane vreg unrolled; 2026-10-03 on the v6e the down matrix took 2.99 us per expert this way vs 3.35 with two
    steps and 6.0 with a fori_loop over the lane vregs); every ref access uses static indices (Mosaic rejects dynamic
    sublane/lane offsets that are not provably tile aligned): planes with fewer than 8 rows per expert are fetched as
    their 8-row block and the expert's rows are picked with selects on the scalar-prefetched offset."""
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    W = PL.WORDS_PER_BLOCK[qtype] * nblk
    C = PL.PLANES_PER_WORD[qtype]
    first = planes[keys[0]]
    lead = first.ndim == 3
    R = first.shape[-1]
    Nk = idx.shape[0]
    assert X.shape == (Nk, W, C), (X.shape, (Nk, W, C))
    assert R % LANES == 0 and W % SUB == 0, (R, W)
    LB = lane_block or min(R, 4096)
    assert R % LB == 0 and LB % LANES == 0, (R, LB)
    n_lb = R // LB
    n_vreg = LB // LANES
    nb = W // SUB
    tbl = _table_arrays(qtype)
    blocks = {k: (rows_p[k] if rows_p[k] % SUB == 0 else SUB) for k in keys}
    decode = DECODE[qtype]

    def kernel(idx_ref, *refs):
        plane_refs = dict(zip(keys, refs[:len(keys)]))
        tbl_ref, x_ref, o_ref, bx_ref = refs[len(keys):]
        offs = _offsets(keys, rows_p, idx_ref[pl.program_id(0)])
        tb = _table_rows(tbl_ref, qtype)
        pl.when(pl.program_id(1) == 0)(lambda: _fill_bx(x_ref, bx_ref, nb, C))
        for v in range(n_vreg):
            lanes = slice(v * LANES, (v + 1) * LANES)
            o_ref[:, lanes] = _matvec_lanes(decode, _rows_fn(plane_refs, offs, lanes), tb, bx_ref, nb, C)

    def plane_spec(k):
        B = blocks[k]
        if lead:
            return pl.BlockSpec((None, B, LB), lambda i, cb, idx_ref: (0, (idx_ref[i] * rows_p[k]) // B, cb))
        return pl.BlockSpec((B, LB), lambda i, cb, idx_ref: ((idx_ref[i] * rows_p[k]) // B, cb))

    in_specs = [plane_spec(k) for k in keys]
    in_specs += [pl.BlockSpec(tbl.shape, lambda i, cb, idx_ref: (0, 0)),
                 pl.BlockSpec((None, W, C), lambda i, cb, idx_ref: (i, 0, 0))]
    out_spec = pl.BlockSpec((None, 1, LB), lambda i, cb, idx_ref: (i, 0, cb))
    grid_spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(Nk, n_lb), in_specs=in_specs,
                                             out_specs=out_spec,
                                             scratch_shapes=[pltpu.VMEM((nb * C, SUB, LANES), jnp.float32)])
    fn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((Nk, 1, R), jnp.float32),
                        interpret=interpret,
                        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary")))
    args = [planes[k] for k in keys] + [tbl, X]
    return fn(idx, *args).reshape(Nk, R)


@_shared("qtype", "nblk", "interpret", "k")
def moe_matvec_gu(planes_g, planes_u, qtype, nblk, idx, X, *, interpret=False, k=1):
    """gate and up of the same expert slots in ONE kernel: planes_{g,u} as in `moe_matvec` (same qtype and shape,
    R <= 1024 rows: all lane vregs unrolled), X f32 [Nk / k, W, C] = one row per TOKEN, whose k consecutive slots
    share it -> (g, u) f32 [Nk, R] each. One grid step per slot; the lane broadcasts of X are made once per token
    (the first of its k slots; the scratch persists across the sequential grid) and serve both matrices (2026-10-03 on
    the v6e: 5.47 us per expert for gate + up vs 2 x 3.43 as two hoisted kernels and 2 x 4.75 shipped)."""
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    W = PL.WORDS_PER_BLOCK[qtype] * nblk
    C = PL.PLANES_PER_WORD[qtype]
    first = planes_g[keys[0]]
    lead = first.ndim == 3
    R = first.shape[-1]
    Nk = idx.shape[0]
    assert Nk % k == 0 and X.shape == (Nk // k, W, C), (X.shape, (Nk // k, W, C))
    assert R % LANES == 0 and R <= 1024 and W % SUB == 0, (R, W)
    assert all(planes_u[kk].shape == planes_g[kk].shape for kk in keys)
    n_vreg = R // LANES
    nb = W // SUB
    nk = len(keys)
    tbl = _table_arrays(qtype)
    blocks = {k: (rows_p[k] if rows_p[k] % SUB == 0 else SUB) for k in keys}
    decode = DECODE[qtype]

    def kernel(idx_ref, *refs):
        g_refs, u_refs = dict(zip(keys, refs[:nk])), dict(zip(keys, refs[nk:2 * nk]))
        tbl_ref, x_ref, o_ref, bx_ref = refs[2 * nk:]
        offs = _offsets(keys, rows_p, idx_ref[pl.program_id(0)])
        tb = _table_rows(tbl_ref, qtype)
        if k == 1:
            _fill_bx(x_ref, bx_ref, nb, C)
        else:                                      # the token's first slot fills, its other k-1 slots reuse
            pl.when(pl.program_id(0) % k == 0)(lambda: _fill_bx(x_ref, bx_ref, nb, C))
        for m, prefs in enumerate((g_refs, u_refs)):
            for v in range(n_vreg):
                lanes = slice(v * LANES, (v + 1) * LANES)
                o_ref[m:m + 1, lanes] = _matvec_lanes(decode, _rows_fn(prefs, offs, lanes), tb, bx_ref, nb, C)

    def plane_spec(k):
        B = blocks[k]
        if lead:
            return pl.BlockSpec((None, B, R), lambda i, idx_ref: (0, (idx_ref[i] * rows_p[k]) // B, 0))
        return pl.BlockSpec((B, R), lambda i, idx_ref: ((idx_ref[i] * rows_p[k]) // B, 0))

    in_specs = [plane_spec(kk) for kk in keys] * 2
    in_specs += [pl.BlockSpec(tbl.shape, lambda i, idx_ref: (0, 0)),
                 pl.BlockSpec((None, W, C), lambda i, idx_ref: (i // k, 0, 0))]
    grid_spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(Nk,), in_specs=in_specs,
                                             out_specs=pl.BlockSpec((None, 2, R), lambda i, idx_ref: (i, 0, 0)),
                                             scratch_shapes=[pltpu.VMEM((nb * C, SUB, LANES), jnp.float32)])
    fn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((Nk, 2, R), jnp.float32),
                        interpret=interpret, compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)))
    out = fn(idx, *[planes_g[k] for k in keys], *[planes_u[k] for k in keys], tbl, X)
    return out[:, 0], out[:, 1]


# ---------------------------------------------------------------- grouped slots (a multi-token step of one sequence)
# A verify of T tokens of ONE sequence routes S = T*k slots, and neighbouring tokens share experts (v6e j64: ~24
# distinct of 32 slots at T = 4). These variants take one grid step per DISTINCT expert: its weights are decoded once
# and multiplied into the accumulators of every slot routed to it, each in exactly the per-slot kernel's order
# (bit-identical results). The step runs the body of the smallest size class >= its slot count (the classes are
# unrolled bodies of 1..tm accumulators; a group below its class repeats its last slot, which recomputes and rewrites
# the same values), so a single-token group pays one MAC per weight, not tm. Activations and outputs stay whole in
# VMEM (dynamic slot indices, no reordering outside the kernel); gate+up broadcasts every token's activation once for
# the whole call. Groups past n_groups have count 0: they repeat the last expert id (no new plane DMA) and skip compute.
# (The first version, 2026-10-04, ran every group at tm = T with per-group broadcasts: 21.9 vs 17.3 ms per DFlash step.)
def group_plan(idx):
    """idx int32 [T, k] (a token's k experts distinct) -> eids int32 [S] (the distinct experts in first-slot order,
    then the last one repeated), cnt int32 [S] (slots per group, 0 past the distinct count), slots int32 [S * T]
    (row g = the slots of group g in slot order; entries past cnt repeat the group's last slot). Exact integer ops on
    S x S compares (no sort, no gather)."""
    T, k = idx.shape
    S = T * k
    flat = idx.reshape(-1)
    eq = flat[:, None] == flat[None, :]
    earlier = jnp.tril(eq, -1)
    first = ~jnp.any(earlier, axis=1)                                   # the first slot of each expert
    gid = jnp.cumsum(first.astype(jnp.int32)) - 1
    g_of = jnp.sum(jnp.where(eq & first[None, :], gid[None, :], 0), axis=1)     # [S] the group of every slot
    rank = jnp.sum(earlier, axis=1, dtype=jnp.int32)                    # [S] its place in the group
    n_groups = jnp.sum(first, dtype=jnp.int32)
    ga = jnp.arange(S, dtype=jnp.int32)
    member = g_of[None, :] == ga[:, None]                               # [G, S]
    cnt = jnp.sum(member, axis=1, dtype=jnp.int32)
    eids = jnp.sum(jnp.where(member & first[None, :], flat[None, :], 0), axis=1)
    last = jnp.sum(jnp.where(ga == n_groups - 1, eids, 0))
    eids = jnp.where(ga < n_groups, eids, last)
    want = jnp.minimum(jnp.arange(T, dtype=jnp.int32)[None, :], jnp.maximum(cnt - 1, 0)[:, None])     # [G, T]
    hit = member[:, None, :] & (rank[None, None, :] == want[:, :, None])                              # [G, T, S]
    slots = jnp.sum(jnp.where(hit, ga[None, None, :], 0), axis=2)
    return eids.astype(jnp.int32), cnt, slots.reshape(-1).astype(jnp.int32)


def _class_bodies(classes, cnt, body):
    """Run body(c) for the size class c that holds `cnt` slots (classes ascending; cnt 0 runs nothing)."""
    lo = 1
    for c in classes:
        pl.when((cnt >= lo) & (cnt <= c))(functools.partial(body, c))
        lo = c + 1


def _fill_bx_at(x_ref, xi, bx_ref, base, nb, C):
    """`_fill_bx` for row xi of a whole-array X ref into bx_ref[base:base + nb*C]."""
    for b in range(nb):
        xb = x_ref[xi, SUB * b:SUB * (b + 1), :]
        for c in range(C):
            bx_ref[base + (b * C + c)] = jnp.broadcast_to(xb[:, c:c + 1], (SUB, LANES))


def _matvec_lanes_grouped(decode, rows, tb, bx_ref, bases, nb, C):
    """`_matvec_lanes` for the slots sharing the expert: each weight vreg decoded once, one accumulator per slot whose
    activation broadcasts start at bx_ref[bases[j]] -> len(bases) x (1, 128), each in `_matvec_lanes`' order."""
    n = len(bases)
    outs = [jnp.zeros((SUB, LANES), jnp.float32) for _ in range(n)]
    for b in range(nb):
        for scale, items in decode(b, rows, tb):
            accs = [jnp.zeros((SUB, LANES), jnp.float32) for _ in range(n)]
            for c, val in items:
                for j in range(n):
                    accs[j] = accs[j] + val * bx_ref[bases[j] + (b * C + c)]
            for j in range(n):
                outs[j] = outs[j] + accs[j] * scale
    return [jnp.sum(o, axis=0, keepdims=True) for o in outs]


def _decode_store(decode, rows, tb, w_ref, s_ref, wi, si, nb):
    """Decode one lane vreg's weights once into VMEM: values to w_ref[wi:], scales to s_ref[si:], in `_matvec_lanes`'
    order -> (wi, si) after them and the static plan [(b, [c, ...]) per scale] for `_mac_stored`."""
    plan = []
    for b in range(nb):
        for scale, items in decode(b, rows, tb):
            s_ref[si] = scale
            si += 1
            for c, val in items:
                w_ref[wi] = val
                wi += 1
            plan.append((b, [c for c, _ in items]))
    return wi, si, plan


def _mac_stored(w_ref, s_ref, wi, si, plan, bx_ref, base, C):
    """`_matvec_lanes` of one slot from the values `_decode_store` wrote (same products and sums, same order)."""
    out = jnp.zeros((SUB, LANES), jnp.float32)
    for b, cs in plan:
        acc = jnp.zeros((SUB, LANES), jnp.float32)
        for c in cs:
            acc = acc + w_ref[wi] * bx_ref[base + (b * C + c)]
            wi += 1
        out = out + acc * s_ref[si]
        si += 1
    return jnp.sum(out, axis=0, keepdims=True)


def _n_values(qtype, nb):
    """(values, scales) that DECODE[qtype] produces for one lane vreg of nb word blocks (an abstract trace)."""
    counts = []

    def f():
        z = lambda k, r0, n: jnp.zeros((n, LANES), jnp.uint32)                             # noqa: E731
        tbl = _table_arrays(qtype)
        tb = [jnp.zeros((SUB, LANES), tbl.dtype)] * (1 if qtype == "IQ4_XS" else tbl.shape[0])
        out = []
        for b in range(nb):
            for scale, items in DECODE[qtype](b, z, tb):
                counts.append(len(items))
                out.append(scale)
        return out
    jax.eval_shape(f)
    return sum(counts), len(counts)


def _n_steps(cnt, G, dynamic):
    return jnp.sum(cnt > 0, dtype=jnp.int32) if dynamic else G


def _grouped_geometry(planes, qtype, nblk, eids, cnt, slots, classes, loop):
    """Shapes of a grouped call; classes = the fused (unrolled) body sizes: without `loop` up to tm (default 1..tm),
    with `loop` the listed ones below tm (default (1,)) and larger counts take the decode-once loop."""
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    G = eids.shape[0]
    assert cnt.shape == (G,) and slots.shape[0] % G == 0, (eids.shape, cnt.shape, slots.shape)
    tm = slots.shape[0] // G
    classes = tuple(classes) or ((1,) if loop else tuple(range(1, tm)))
    classes = tuple(sorted(set(c for c in classes if c < tm))) + (() if loop else (tm,))
    first = planes[keys[0]]
    return (keys, rows_p, PL.WORDS_PER_BLOCK[qtype] * nblk, PL.PLANES_PER_WORD[qtype], first.ndim == 3,
            first.shape[-1], G, tm, classes, {k: (rows_p[k] if rows_p[k] % SUB == 0 else SUB) for k in keys})


@_shared("qtype", "nblk", "k", "classes", "interpret", "dynamic_grid", "loop", "vmem_mb")
def moe_matvec_gu_grouped(planes_g, planes_u, qtype, nblk, eids, cnt, slots, X, *, k, classes=(), interpret=False,
                          dynamic_grid=False, loop=False, vmem_mb=32):
    """`moe_matvec_gu` over grouped slots (`group_plan`): X f32 [T, W, C] one row per TOKEN (slot s is token s // k)
    -> (g, u) f32 [S, R] each, in slot order. classes: the unrolled body sizes below tm (= T, always one); () = all.
    dynamic_grid: one grid step per distinct expert (a traced grid bound) instead of G with empty steps at the end.
    loop: groups above the fused classes decode the expert once into VMEM and run one MAC pass per slot from it (a
    fori_loop over the group's slots: the kernel's code no longer grows with the class sizes; compile time).
    vmem_mb: the scoped VMEM limit (every token's activation broadcasts stay in VMEM: 2 MB per token at D = 4096)."""
    keys, rows_p, W, C, lead, R, G, tm, classes, blocks = _grouped_geometry(planes_g, qtype, nblk, eids, cnt, slots,
                                                                            classes, loop)
    T = X.shape[0]
    S = T * k
    assert X.shape == (T, W, C) and tm <= T, (X.shape, (T, W, C), tm)
    assert R % LANES == 0 and R <= 1024 and W % SUB == 0, (R, W)
    assert all(planes_u[kk].shape == planes_g[kk].shape for kk in keys)
    n_vreg = R // LANES
    nb = W // SUB
    nk = len(keys)
    tbl = _table_arrays(qtype)
    decode = DECODE[qtype]

    nv, ns = _n_values(qtype, nb)

    def kernel(eid_ref, cnt_ref, slot_ref, *refs):
        g_refs, u_refs = dict(zip(keys, refs[:nk])), dict(zip(keys, refs[nk:2 * nk]))
        tbl_ref, x_ref, o_ref, bx_ref = refs[2 * nk:2 * nk + 4]
        g = pl.program_id(0)                       # (scalars read outside pl.when: interpret mode cannot inside)
        offs = _offsets(keys, rows_p, eid_ref[g])
        n = cnt_ref[g]
        sl = [slot_ref[g * tm + j] for j in range(tm)]
        bases = [(s // k) * (nb * C) for s in sl]

        @pl.when(g == 0)
        def _():                                   # every token's broadcasts, once for the call
            for t in range(T):
                _fill_bx_at(x_ref, t, bx_ref, t * nb * C, nb, C)

        def body(c):
            tb = _table_rows(tbl_ref, qtype)
            for m, prefs in enumerate((g_refs, u_refs)):
                for v in range(n_vreg):
                    lanes = slice(v * LANES, (v + 1) * LANES)
                    res = _matvec_lanes_grouped(decode, _rows_fn(prefs, offs, lanes), tb, bx_ref, bases[:c], nb, C)
                    for j in range(c):
                        o_ref[sl[j], m:m + 1, lanes] = res[j]
        _class_bodies(classes, n, body)

        def decode_once_loop():
            w_ref, s_ref = refs[2 * nk + 4:]
            tb = _table_rows(tbl_ref, qtype)
            plans, wi, si = [], 0, 0
            for m, prefs in enumerate((g_refs, u_refs)):
                for v in range(n_vreg):
                    wi0, si0 = wi, si
                    wi, si, plan = _decode_store(decode, _rows_fn(prefs, offs, slice(v * LANES, (v + 1) * LANES)),
                                                 tb, w_ref, s_ref, wi, si, nb)
                    plans.append((m, v, wi0, si0, plan))

            def one_slot(j, carry):
                s_ = slot_ref[g * tm + j]
                base = (s_ // k) * (nb * C)
                for m, v, wi0, si0, plan in plans:
                    o_ref[s_, m:m + 1, v * LANES:(v + 1) * LANES] = _mac_stored(w_ref, s_ref, wi0, si0, plan, bx_ref,
                                                                                 base, C)
                return carry
            lax.fori_loop(0, n, one_slot, 0)
        if loop:
            pl.when(n > (classes[-1] if classes else 0))(decode_once_loop)

    def plane_spec(kk):
        B = blocks[kk]
        if lead:
            return pl.BlockSpec((None, B, R), lambda i, e, c, s: (0, (e[i] * rows_p[kk]) // B, 0))
        return pl.BlockSpec((B, R), lambda i, e, c, s: ((e[i] * rows_p[kk]) // B, 0))

    in_specs = [plane_spec(kk) for kk in keys] * 2
    in_specs += [pl.BlockSpec(tbl.shape, lambda i, e, c, s: (0, 0)),
                 pl.BlockSpec((T, W, C), lambda i, e, c, s: (0, 0, 0))]
    grid_spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=3, grid=(_n_steps(cnt, G, dynamic_grid),),
                                             in_specs=in_specs,
                                             out_specs=pl.BlockSpec((S, 2, R), lambda i, e, c, s: (0, 0, 0)),
                                             scratch_shapes=[pltpu.VMEM((T * nb * C, SUB, LANES), jnp.float32)] + (
                                                 [pltpu.VMEM((2 * n_vreg * nv, SUB, LANES), jnp.float32),
                                                  pltpu.VMEM((2 * n_vreg * ns, SUB, LANES), jnp.float32)] if loop else []))
    fn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((S, 2, R), jnp.float32),
                        interpret=interpret,
                        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",),
                                                             vmem_limit_bytes=vmem_mb * 2 ** 20))
    out = fn(eids, cnt, slots, *[planes_g[kk] for kk in keys], *[planes_u[kk] for kk in keys], tbl, X)
    return out[:, 0], out[:, 1]


@_shared("qtype", "nblk", "classes", "interpret", "dynamic_grid", "loop", "vmem_mb")
def moe_matvec_grouped(planes, qtype, nblk, eids, cnt, slots, X, *, classes=(), interpret=False, dynamic_grid=False,
                       loop=False, vmem_mb=32):
    """`moe_matvec` over grouped slots (`group_plan`): X f32 [S, W, C] one row per SLOT -> f32 [S, R] in slot order
    (all R rows in one grid step per group; the group's slots are broadcast at the start of its step; `loop`: see
    `moe_matvec_gu_grouped`, each slot's broadcasts made in its pass)."""
    keys, rows_p, W, C, lead, R, G, tm, classes, blocks = _grouped_geometry(planes, qtype, nblk, eids, cnt, slots,
                                                                            classes, loop)
    S = X.shape[0]
    assert X.shape == (S, W, C), (X.shape, (S, W, C))
    assert R % LANES == 0 and R <= 4096 and W % SUB == 0, (R, W)
    n_vreg = R // LANES
    nb = W // SUB
    tbl = _table_arrays(qtype)
    decode = DECODE[qtype]

    nv, ns = _n_values(qtype, nb)

    def kernel(eid_ref, cnt_ref, slot_ref, *refs):
        plane_refs = dict(zip(keys, refs[:len(keys)]))
        tbl_ref, x_ref, o_ref, bx_ref = refs[len(keys):len(keys) + 4]
        g = pl.program_id(0)
        offs = _offsets(keys, rows_p, eid_ref[g])
        n = cnt_ref[g]
        sl = [slot_ref[g * tm + j] for j in range(tm)]

        def body(c):
            for j in range(c):
                _fill_bx_at(x_ref, sl[j], bx_ref, j * nb * C, nb, C)
            tb = _table_rows(tbl_ref, qtype)
            bases = [j * nb * C for j in range(c)]
            for v in range(n_vreg):
                lanes = slice(v * LANES, (v + 1) * LANES)
                res = _matvec_lanes_grouped(decode, _rows_fn(plane_refs, offs, lanes), tb, bx_ref, bases, nb, C)
                for j in range(c):
                    o_ref[sl[j], :, lanes] = res[j]
        _class_bodies(classes, n, body)

        def decode_once_loop():
            w_ref, s_ref = refs[len(keys) + 4:]
            tb = _table_rows(tbl_ref, qtype)
            plans, wi, si = [], 0, 0
            for v in range(n_vreg):
                wi0, si0 = wi, si
                wi, si, plan = _decode_store(decode, _rows_fn(plane_refs, offs, slice(v * LANES, (v + 1) * LANES)),
                                             tb, w_ref, s_ref, wi, si, nb)
                plans.append((v, wi0, si0, plan))

            def one_slot(j, carry):
                s_ = slot_ref[g * tm + j]
                _fill_bx_at(x_ref, s_, bx_ref, 0, nb, C)
                for v, wi0, si0, plan in plans:
                    o_ref[s_, :, v * LANES:(v + 1) * LANES] = _mac_stored(w_ref, s_ref, wi0, si0, plan, bx_ref, 0, C)
                return carry
            lax.fori_loop(0, n, one_slot, 0)
        if loop:
            pl.when(n > (classes[-1] if classes else 0))(decode_once_loop)

    def plane_spec(kk):
        B = blocks[kk]
        if lead:
            return pl.BlockSpec((None, B, R), lambda i, e, c, s: (0, (e[i] * rows_p[kk]) // B, 0))
        return pl.BlockSpec((B, R), lambda i, e, c, s: ((e[i] * rows_p[kk]) // B, 0))

    in_specs = [plane_spec(kk) for kk in keys]
    in_specs += [pl.BlockSpec(tbl.shape, lambda i, e, c, s: (0, 0)),
                 pl.BlockSpec((S, W, C), lambda i, e, c, s: (0, 0, 0))]
    grid_spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=3, grid=(_n_steps(cnt, G, dynamic_grid),),
                                             in_specs=in_specs,
                                             out_specs=pl.BlockSpec((S, 1, R), lambda i, e, c, s: (0, 0, 0)),
                                             scratch_shapes=[pltpu.VMEM((tm * nb * C, SUB, LANES), jnp.float32)] + (
                                                 [pltpu.VMEM((n_vreg * nv, SUB, LANES), jnp.float32),
                                                  pltpu.VMEM((n_vreg * ns, SUB, LANES), jnp.float32)] if loop else []))
    fn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((S, 1, R), jnp.float32),
                        interpret=interpret,
                        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",),
                                                             vmem_limit_bytes=vmem_mb * 2 ** 20))
    return fn(eids, cnt, slots, *[planes[kk] for kk in keys], tbl, X).reshape(S, R)


def moe_matvec_ref(planes, qtype, nblk, idx, X):
    """Pure-JAX reference (gathers the experts, dequantizes with glm53.planes, einsum)."""
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    sel = {}
    for k in keys:
        a = planes[k]
        a = a[0] if a.ndim == 3 else a
        R = a.shape[-1]
        a = a.reshape(-1, rows_p[k], R)
        sel[k] = jnp.take(a, idx, axis=0)                                    # [Nk, rows_p, R]
    V = PL.dequant_planes(sel, qtype, nblk)                                 # [Nk, C*W, R]
    x_pm = jnp.swapaxes(X, 1, 2).reshape(X.shape[0], -1)                   # [Nk, C*W]
    return jnp.einsum("ni,nir->nr", x_pm, V, precision="highest")


# ------------------------------------------------------------------------------------------ prefill (sweep) kernels
# Dequantize each expert ONCE into (K, 128) bf16 tiles of W^T in VMEM and MXU-matmul all T tokens against them.
# Tile rows are ordered (word block b, value plane c, word s): input(8b+s, c) -> row (b*C + c)*8 + s; `pm_mxu`
# permutes activations to that order. Expert slots come from scalar-prefetched `ids` (unique active experts first,
# inactive slots repeat the last active id so their planes are never re-DMA'd) and `n_active` gates the compute.
def pm_mxu(qtype, x):
    """x [..., n] natural -> [..., n] in the kernel tile order (b, c, s)."""
    X = PL.pm_x(qtype, x)                                                    # [..., W, C]
    lead = X.shape[:-2]
    W, C = X.shape[-2:]
    y = X.reshape(lead + (W // SUB, SUB, C))
    y = jnp.swapaxes(y, -1, -2)                                              # [..., W/8, C, 8]
    return y.reshape(lead + (W * C,))


def _tiles(decode, rows, tb, b0, nb, dtype=jnp.bfloat16):
    """(nb*C*8, 128) tile of scaled W^T rows for word blocks b0..b0+nb-1 at the current lanes, in `dtype`."""
    parts = []
    for b in range(b0, b0 + nb):
        items = []
        for scale, its in decode(b, rows, tb):
            items += [(c, v * scale) for c, v in its]
        items.sort(key=lambda t: t[0])
        parts += [v for _, v in items]
    return jnp.concatenate(parts, axis=0).astype(dtype)


def _dot(a, b):
    prec = lax.Precision.HIGHEST if a.dtype == jnp.float32 else None
    return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=prec)


def active_slots(idx, E):
    """idx int32 [N, k] -> (ids int32 [E], n_active int32): unique routed experts first (ascending), the rest of the
    slots repeat the last active id."""
    hit = (jax.nn.one_hot(idx.reshape(-1), E, dtype=jnp.int32).sum(0) > 0)          # [E] bool
    order = jnp.argsort(jnp.where(hit, 0, 1) * E + jnp.arange(E))                     # active ids first, ascending
    n_active = hit.sum().astype(jnp.int32)
    last = order[jnp.maximum(n_active - 1, 0)]
    ids = jnp.where(jnp.arange(E) < n_active, order, last).astype(jnp.int32)
    return ids, n_active


def _k_group(qtype):
    """word blocks per MXU tile so that K >= 128."""
    return max(1, 128 // (PL.PLANES_PER_WORD[qtype] * SUB))


def _probe_tiles(probe, tiles_fn, K, dtype):
    """Cost-split probes: 'mxu' replaces the dequantized tile by a constant (matmul cost only), 'deq' keeps the
    dequant but returns None so the caller skips the dot (dequant cost only)."""
    if probe == "mxu":
        return jnp.full((K, LANES), 0.01, dtype)
    return tiles_fn()


def moe_sweep_gateup(planes_g, planes_u, qtype, nblk, ids, n_active, xk, limit, *, interpret=False, lane_block=None,
                     t_block=512, vmem_mb=48, probe=None):
    """planes_{g,u}: {k: u32 [1, E*rows_p, R]} (same qtype); ids [E] int32; n_active int32 scalar; xk [T, n_in] in
    pm_mxu order (bf16 on TPU; f32 -> f32 tiles + HIGHEST-precision dots for the CPU tests) ->
    h [E, T, R] (xk.dtype) = swiglu_clamped(x @ gate_e^T, x @ up_e^T) for active slots, 0 elsewhere.
    `probe` (timing only, wrong numbers): 'mxu' = constant tiles, 'deq' = dequant without the matmul."""
    dtype = xk.dtype
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    W = PL.WORDS_PER_BLOCK[qtype] * nblk
    C = PL.PLANES_PER_WORD[qtype]
    E = ids.shape[0]
    T, n_in = xk.shape
    assert n_in == nblk * PL.QK, (n_in, nblk)
    first = planes_g[keys[0]]
    lead = first.ndim == 3
    R = first.shape[-1]
    LB = min(lane_block or 1024, R)
    TB = min(T, t_block)
    assert T % TB == 0 and R % LB == 0
    n_lb, n_vreg, n_tb = R // LB, LB // LANES, T // TB
    nbg = _k_group(qtype)
    K = nbg * C * SUB
    tbl = _table_arrays(qtype)
    blocks = {k: (rows_p[k] if rows_p[k] % SUB == 0 else SUB) for k in keys}
    decode = DECODE[qtype]
    nk = len(keys)

    def kernel(ids_ref, nact_ref, *refs):
        g_refs = dict(zip(keys, refs[:nk]))
        u_refs = dict(zip(keys, refs[nk:2 * nk]))
        tbl_ref, x_ref, o_ref, acc_g, acc_u = refs[2 * nk:]
        e = pl.program_id(1)
        eid = ids_ref[e]
        offs = {k: (lax.rem(eid * rows_p[k], SUB) if rows_p[k] % SUB else None) for k in keys}
        tb = _table_rows(tbl_ref, qtype)

        @pl.when(e >= nact_ref[0])
        def _inactive():
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)

        @pl.when(e < nact_ref[0])
        def _active():
            for v in range(n_vreg):
                lanes = slice(v * LANES, (v + 1) * LANES)

                def rows_of(prefs, lanes=lanes):
                    def rows(k, r0, n):
                        if offs[k] is None:
                            return prefs[k][r0:r0 + n, lanes]
                        return _pick_rows(prefs[k][0:SUB, lanes], offs[k], r0, n)
                    return rows

                rg, ru = rows_of(g_refs), rows_of(u_refs)
                ag = jnp.zeros((TB, LANES), jnp.float32)
                au = jnp.zeros((TB, LANES), jnp.float32)
                for b0 in range(0, W // SUB, nbg):
                    k0 = b0 * C * SUB
                    xs = x_ref[:, k0:k0 + K]                                              # (TB, K) bf16
                    tg = _probe_tiles(probe, lambda: _tiles(decode, rg, tb, b0, nbg, dtype), K, dtype)
                    tu = _probe_tiles(probe, lambda: _tiles(decode, ru, tb, b0, nbg, dtype), K, dtype)
                    if probe == "deq":
                        ag = ag + jnp.sum(tg, axis=0, keepdims=True).astype(jnp.float32)
                        au = au + jnp.sum(tu, axis=0, keepdims=True).astype(jnp.float32)
                        continue
                    ag = ag + _dot(xs, tg)
                    au = au + _dot(xs, tu)
                acc_g[:, lanes] = ag
                acc_u[:, lanes] = au
            o_ref[...] = M.swiglu_clamped(acc_g[...], acc_u[...], limit).astype(o_ref.dtype)

    def plane_spec(k):
        B = blocks[k]
        if lead:
            return pl.BlockSpec((None, B, LB), lambda t, e, cb, ids_ref, n_ref: (0, (ids_ref[e] * rows_p[k]) // B, cb))
        return pl.BlockSpec((B, LB), lambda t, e, cb, ids_ref, n_ref: ((ids_ref[e] * rows_p[k]) // B, cb))

    in_specs = [plane_spec(k) for k in keys] * 2
    in_specs += [pl.BlockSpec(tbl.shape, lambda t, e, cb, ids_ref, n_ref: (0, 0)),
                 pl.BlockSpec((TB, n_in), lambda t, e, cb, ids_ref, n_ref: (t, 0))]
    out_spec = pl.BlockSpec((None, TB, LB), lambda t, e, cb, ids_ref, n_ref: (e, t, cb))
    grid_spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2, grid=(n_tb, E, n_lb), in_specs=in_specs,
                                             out_specs=out_spec,
                                             scratch_shapes=[pltpu.VMEM((TB, LB), jnp.float32)] * 2)
    fn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((E, T, R), dtype),
                        interpret=interpret,
                        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",) * 3,
                                                             vmem_limit_bytes=vmem_mb << 20))
    args = [planes_g[k] for k in keys] + [planes_u[k] for k in keys] + [tbl, xk]
    return fn(ids, n_active.reshape(1), *args)


def moe_sweep_down(planes, qtype, nblk, ids, n_active, hk, *, interpret=False, lane_block=None, t_block=512,
                   vmem_mb=48, probe=None):
    """planes {k: u32 [1, E*rows_p, R]}; hk [E, T, n_in] (pm_mxu order, routing weights folded in; bf16 or f32) ->
    y f32 [T, R] = sum over active slots of hk[e] @ down_e^T."""
    dtype = hk.dtype
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    W = PL.WORDS_PER_BLOCK[qtype] * nblk
    C = PL.PLANES_PER_WORD[qtype]
    E, T, n_in = hk.shape
    assert n_in == nblk * PL.QK and ids.shape == (E,)
    first = planes[keys[0]]
    lead = first.ndim == 3
    R = first.shape[-1]
    LB = min(lane_block or 1024, R)
    TB = min(T, t_block)
    assert T % TB == 0 and R % LB == 0
    n_lb, n_vreg, n_tb = R // LB, LB // LANES, T // TB
    nbg = _k_group(qtype)
    K = nbg * C * SUB
    tbl = _table_arrays(qtype)
    blocks = {k: (rows_p[k] if rows_p[k] % SUB == 0 else SUB) for k in keys}
    decode = DECODE[qtype]
    nk = len(keys)

    def kernel(ids_ref, nact_ref, *refs):
        prefs = dict(zip(keys, refs[:nk]))
        tbl_ref, h_ref, o_ref = refs[nk:]
        e = pl.program_id(2)
        eid = ids_ref[e]
        offs = {k: (lax.rem(eid * rows_p[k], SUB) if rows_p[k] % SUB else None) for k in keys}
        tb = _table_rows(tbl_ref, qtype)

        @pl.when(e == 0)
        def _init():
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)

        @pl.when(e < nact_ref[0])
        def _active():
            for v in range(n_vreg):
                lanes = slice(v * LANES, (v + 1) * LANES)

                def rows(k, r0, n, lanes=lanes):
                    if offs[k] is None:
                        return prefs[k][r0:r0 + n, lanes]
                    return _pick_rows(prefs[k][0:SUB, lanes], offs[k], r0, n)

                acc = o_ref[:, lanes]
                for b0 in range(0, W // SUB, nbg):
                    k0 = b0 * C * SUB
                    td = _probe_tiles(probe, lambda: _tiles(decode, rows, tb, b0, nbg, dtype), K, dtype)
                    if probe == "deq":
                        acc = acc + jnp.sum(td, axis=0, keepdims=True).astype(jnp.float32)
                        continue
                    acc = acc + _dot(h_ref[:, k0:k0 + K], td)
                o_ref[:, lanes] = acc

    def plane_spec(k):
        B = blocks[k]
        if lead:
            return pl.BlockSpec((None, B, LB), lambda t, cb, e, ids_ref, n_ref: (0, (ids_ref[e] * rows_p[k]) // B, cb))
        return pl.BlockSpec((B, LB), lambda t, cb, e, ids_ref, n_ref: ((ids_ref[e] * rows_p[k]) // B, cb))

    in_specs = [plane_spec(k) for k in keys]
    in_specs += [pl.BlockSpec(tbl.shape, lambda t, cb, e, ids_ref, n_ref: (0, 0)),
                 pl.BlockSpec((None, TB, n_in), lambda t, cb, e, ids_ref, n_ref: (e, t, 0))]
    out_spec = pl.BlockSpec((TB, LB), lambda t, cb, e, ids_ref, n_ref: (t, cb))
    grid_spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2, grid=(n_tb, n_lb, E), in_specs=in_specs,
                                             out_specs=out_spec)
    fn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((T, R), jnp.float32),
                        interpret=interpret,
                        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",) * 3,
                                                             vmem_limit_bytes=vmem_mb << 20))
    return fn(ids, n_active.reshape(1), *[planes[k] for k in keys], tbl, hk)


# ------------------------------------------------------------------------------------- grouped (ragged) prefill GEMM
# The dense sweep multiplies every 512-token chunk against every active expert (masked): on real text ~all 288 experts
# are active, so ~36x the useful MXU work, each expert dequantized once per chunk, and an [E, T, ml] intermediate.
# Here the (token, expert) slots are SORTED BY EXPERT into blocks of `tm` rows (each expert's rows start at a block
# boundary, zero-padded); the kernels walk the blocks in order, DMA + dequantize an expert's planes ONCE (when the
# block's expert changes; the dequantized W^T tiles stay in VMEM scratch) and multiply only that expert's rows.
def ragged_plan(idx, w, E, tm):
    """Slot layout for the grouped kernels. idx int32 [T, k], w f32 [T, k] -> dict with
    blk_expert int32 [G] (expert of row block g; G = ceil(T*k / tm) + min(E, T*k) blocks of `tm` rows — every routed
    expert needs ceil(count / tm) <= count / tm + 1 blocks, and at most min(E, T*k) experts are routed — blocks >=
    n_blocks are padding and repeat the last expert), n_blocks int32 [1], tok_row int32 [Rp] (token of every sorted
    row, -1 = padding), w_row f32 [Rp] (its routing weight), row_slot int32 [T, k] (the row of every (token, slot)).
    (A verify of 8 tokens: 2 + 64 blocks instead of 2 + 288 — each padding block is a grid step.)"""
    T, k = idx.shape
    S = T * k
    G = -(-S // tm) + min(E, S)
    Rp = G * tm
    flat = idx.reshape(-1).astype(jnp.int32)
    order = jnp.argsort(flat)
    se = flat[order]
    counts = jnp.sum(jax.nn.one_hot(flat, E, dtype=jnp.int32), axis=0)                    # [E]
    padded = -(-counts // tm) * tm
    pstart = jnp.cumsum(padded) - padded
    ustart = jnp.cumsum(counts) - counts
    row_sorted = pstart[se] + (jnp.arange(S, dtype=jnp.int32) - ustart[se])              # row of sorted slot j
    tok_row = jnp.full((Rp,), -1, jnp.int32).at[row_sorted].set((order // k).astype(jnp.int32))
    w_row = jnp.zeros((Rp,), jnp.float32).at[row_sorted].set(w.reshape(-1)[order].astype(jnp.float32))
    row_slot = jnp.zeros((S,), jnp.int32).at[order].set(row_sorted).reshape(T, k)
    nb = padded // tm
    bstart = jnp.cumsum(nb) - nb
    n_blocks = jnp.sum(nb).astype(jnp.int32)
    g = jnp.arange(G, dtype=jnp.int32)
    be = jnp.sum((g[:, None] >= (bstart + nb)[None, :]).astype(jnp.int32), axis=1)        # experts ended at or before g
    be = jnp.minimum(be, E - 1)
    last = be[jnp.maximum(n_blocks - 1, 0)]
    be = jnp.where(g < n_blocks, be, last).astype(jnp.int32)
    return {"blk_expert": be, "n_blocks": n_blocks.reshape(1), "tok_row": tok_row, "w_row": w_row, "row_slot": row_slot}


def _ragged_call(kernel, planes_list, keys, rows_p, blocks, tbl, act, out_dtype, R, LB, tm, G, scratch, interpret, vmem_mb,
                 be, n_blocks):
    """Shared pallas_call plumbing of the two ragged kernels: grid (lane block, row block)."""
    lead = planes_list[0][keys[0]].ndim == 3
    n_lb = R // LB
    Rp, n_in = act.shape

    def plane_spec(k):
        B = blocks[k]
        if lead:
            return pl.BlockSpec((None, B, LB), lambda cb, g, be_ref, nb_ref: (0, (be_ref[g] * rows_p[k]) // B, cb))
        return pl.BlockSpec((B, LB), lambda cb, g, be_ref, nb_ref: ((be_ref[g] * rows_p[k]) // B, cb))

    in_specs = [plane_spec(k) for k in keys] * len(planes_list)
    in_specs += [pl.BlockSpec(tbl.shape, lambda cb, g, be_ref, nb_ref: (0, 0)),
                 pl.BlockSpec((tm, n_in), lambda cb, g, be_ref, nb_ref: (jnp.minimum(g, nb_ref[0] - 1), 0))]
    out_spec = pl.BlockSpec((tm, LB), lambda cb, g, be_ref, nb_ref: (g, cb))
    grid_spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2, grid=(n_lb, G), in_specs=in_specs,
                                             out_specs=out_spec, scratch_shapes=scratch)
    fn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((Rp, R), out_dtype),
                        interpret=interpret,
                        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"),
                                                             vmem_limit_bytes=vmem_mb << 20))
    args = [p[k] for p in planes_list for k in keys] + [tbl, act]
    return fn(be, n_blocks, *args)


@_shared("qtype", "nblk", "limit", "tm", "interpret", "lane_block", "vmem_mb")
def moe_ragged_gateup(planes_g, planes_u, qtype, nblk, blk_expert, n_blocks, xs, limit, *, tm=32, interpret=False,
                      lane_block=None, vmem_mb=48):
    """Grouped prefill GEMM, gate + up. xs [Rp, n_in] in pm_mxu order, rows sorted by expert in blocks of `tm` rows
    (block g = rows g*tm.., expert blk_expert[g]; blocks >= n_blocks are padding) -> h [Rp, R] (xs.dtype) =
    swiglu_clamped(x @ gate_e^T, x @ up_e^T) per row with e = its block's expert; padding blocks are zero."""
    dtype = xs.dtype
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    W = PL.WORDS_PER_BLOCK[qtype] * nblk
    C = PL.PLANES_PER_WORD[qtype]
    Rp, n_in = xs.shape
    assert n_in == nblk * PL.QK and n_in == W * C and Rp % tm == 0, (n_in, nblk, W, C, Rp, tm)
    G = Rp // tm
    assert blk_expert.shape == (G,), (blk_expert.shape, G)
    R = planes_g[keys[0]].shape[-1]
    LB = min(lane_block or 1024, R)
    assert R % LB == 0
    n_vreg = LB // LANES
    nbg = _k_group(qtype)
    K = nbg * C * SUB
    tbl = _table_arrays(qtype)
    blocks = {k: (rows_p[k] if rows_p[k] % SUB == 0 else SUB) for k in keys}
    decode = DECODE[qtype]
    nk = len(keys)

    def kernel(be_ref, nb_ref, *refs):
        g_refs = dict(zip(keys, refs[:nk]))
        u_refs = dict(zip(keys, refs[nk:2 * nk]))
        tbl_ref, x_ref, o_ref, wg, wu = refs[2 * nk:]
        g = pl.program_id(1)
        eid = be_ref[g]
        prev = be_ref[jnp.maximum(g - 1, 0)]
        offs = {k: (lax.rem(eid * rows_p[k], SUB) if rows_p[k] % SUB else None) for k in keys}
        tb = _table_rows(tbl_ref, qtype)

        @pl.when((g == 0) | (eid != prev))
        def _decode():                                     # this expert's W^T tiles for the lane block, once
            for v in range(n_vreg):
                lanes = slice(v * LANES, (v + 1) * LANES)

                def rows_of(prefs, lanes=lanes):
                    def rows(k, r0, n):
                        if offs[k] is None:
                            return prefs[k][r0:r0 + n, lanes]
                        return _pick_rows(prefs[k][0:SUB, lanes], offs[k], r0, n)
                    return rows

                rg, ru = rows_of(g_refs), rows_of(u_refs)
                for b0 in range(0, W // SUB, nbg):
                    k0 = b0 * C * SUB
                    wg[k0:k0 + K, lanes] = _tiles(decode, rg, tb, b0, nbg, dtype)
                    wu[k0:k0 + K, lanes] = _tiles(decode, ru, tb, b0, nbg, dtype)

        @pl.when(g < nb_ref[0])
        def _rows():
            x = x_ref[...]                                                                  # (tm, n_in)
            o_ref[...] = M.swiglu_clamped(_dot(x, wg[...]), _dot(x, wu[...]), limit).astype(o_ref.dtype)

        @pl.when(g >= nb_ref[0])
        def _pad():
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)

    scratch = [pltpu.VMEM((n_in, LB), dtype)] * 2
    return _ragged_call(kernel, [planes_g, planes_u], keys, rows_p, blocks, tbl, xs, dtype, R, LB, tm, G, scratch,
                        interpret, vmem_mb, blk_expert, n_blocks)


@_shared("qtype", "nblk", "tm", "interpret", "lane_block", "vmem_mb", "out_dtype")
def moe_ragged_down(planes, qtype, nblk, blk_expert, n_blocks, hs, *, tm=32, interpret=False, lane_block=None,
                    vmem_mb=48, out_dtype=None):
    """Grouped prefill GEMM, down. hs [Rp, n_in] (pm_mxu order, sorted rows as in `moe_ragged_gateup`) ->
    y [Rp, R] (out_dtype, default hs.dtype) = h @ down_e^T per row; padding blocks are zero."""
    dtype = hs.dtype
    out_dtype = out_dtype or dtype
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    W = PL.WORDS_PER_BLOCK[qtype] * nblk
    C = PL.PLANES_PER_WORD[qtype]
    Rp, n_in = hs.shape
    assert n_in == nblk * PL.QK and n_in == W * C and Rp % tm == 0, (n_in, nblk, W, C, Rp, tm)
    G = Rp // tm
    assert blk_expert.shape == (G,), (blk_expert.shape, G)
    R = planes[keys[0]].shape[-1]
    LB = min(lane_block or 1024, R)
    assert R % LB == 0
    n_vreg = LB // LANES
    nbg = _k_group(qtype)
    K = nbg * C * SUB
    tbl = _table_arrays(qtype)
    blocks = {k: (rows_p[k] if rows_p[k] % SUB == 0 else SUB) for k in keys}
    decode = DECODE[qtype]
    nk = len(keys)

    def kernel(be_ref, nb_ref, *refs):
        prefs = dict(zip(keys, refs[:nk]))
        tbl_ref, h_ref, o_ref, wd = refs[nk:]
        g = pl.program_id(1)
        eid = be_ref[g]
        prev = be_ref[jnp.maximum(g - 1, 0)]
        offs = {k: (lax.rem(eid * rows_p[k], SUB) if rows_p[k] % SUB else None) for k in keys}
        tb = _table_rows(tbl_ref, qtype)

        @pl.when((g == 0) | (eid != prev))
        def _decode():
            for v in range(n_vreg):
                lanes = slice(v * LANES, (v + 1) * LANES)

                def rows(k, r0, n, lanes=lanes):
                    if offs[k] is None:
                        return prefs[k][r0:r0 + n, lanes]
                    return _pick_rows(prefs[k][0:SUB, lanes], offs[k], r0, n)

                for b0 in range(0, W // SUB, nbg):
                    k0 = b0 * C * SUB
                    wd[k0:k0 + K, lanes] = _tiles(decode, rows, tb, b0, nbg, dtype)

        @pl.when(g < nb_ref[0])
        def _rows():
            o_ref[...] = _dot(h_ref[...], wd[...]).astype(o_ref.dtype)

        @pl.when(g >= nb_ref[0])
        def _pad():
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)

    scratch = [pltpu.VMEM((n_in, LB), dtype)]
    return _ragged_call(kernel, [planes], keys, rows_p, blocks, tbl, hs, out_dtype, R, LB, tm, G, scratch, interpret,
                        vmem_mb, blk_expert, n_blocks)
