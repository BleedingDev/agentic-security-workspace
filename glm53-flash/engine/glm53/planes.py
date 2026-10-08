"""Planar ("lane = matrix row") storage of codebook-quantized expert matrices for the resident engine.

The raw GGUF block layout (82/110/136 bytes per 256 weights) makes every field a byte slice at an odd offset, which
costs XLA one relayout copy + one "header parsing" fusion per field per step and is unusable from a Pallas kernel.
Here each block field becomes its own dense 32-bit plane, transposed so that the matrix ROW is the lane (minor) axis
and the 4-byte "word" of packed codes is the sublane axis:

    word w of a row = 4 consecutive qs bytes = 4 codebook groups (4w .. 4w+3); byte j of the word is group 4w+j and
    each group holds GROUP_W[qtype] weights (positions k). So one (8, 128) vreg of the qs plane covers 8 words x 128
    rows and the weight it describes at (byte j, position k) is input index INPUT(w, j, k) of those 128 rows.

Per expert the planes hold exactly the bytes of the GGUF blocks (no expansion) except `d`, which is stored as pairs
of f16 bit patterns in one u32 per two blocks. All planes of one table are E-merged: shape [E * rows_p, R] so that
every array is (8, 128)-tile dense whatever rows_p is (a [E, 1, 4096] array would pad 1 -> 8 sublanes = 8x waste).

Formats (W = words per row, C = value planes per word = bytes x positions, `input(w, c)`):
  IQ2_S : qs u32 [W, R]; sg u32 [W, R] (sign byte j of word w = group 4w+j, bit k = position k);
          qh u32 [W/4, R] (byte w%4 of u32 w//4: bits 2j..2j+1 = high index bits of group 4w+j);
          sc u32 [W/4, R] (byte w%4 of u32 w//4: nibble h = scale of groups 4w+2h, 4w+2h+1);
          d  u32 [ceil(nblk/2), R] (f16 bits of block b at half b%2 of u32 b//2; block b = w // 8).
          W = 8*nblk, C = 32, input = 32w + 8j + k.
  IQ3_S : qs u32 [W, R]; sg u32 [W/2, R] (half w%2 of u32 w//2: nibble j = group 4w+j, bit k = position k);
          qh u32 [W/8, R] (bits 4(w%8)+j of u32 w//8 = high bit of group 4w+j);
          sc u32 [W/16, R] (nibble (w//2)%8 of u32 w//16 = scale of words 2n, 2n+1); d as above, block b = w // 16.
          W = 16*nblk, C = 16, input = 16w + 4j + k.
  IQ4_XS: qs u32 [W, R] (byte j of word w: low nibble = position 0, high = position 1); sl u32 [nblk, R] (nibble i =
          low scale bits of sub-block i); sh u32 [nblk, R] (bits 2i..2i+1 = high scale bits); d as above.
          W = 32*nblk, C = 8 (c = 4*pos + j), input = 32*(w//4) + 16*pos + 4*(w%4) + j.

The XLA dequant returns each expert as [C*W, R] with input index c*W + w ("pm order"); `pm_x` permutes activations
to match: x_pm[c*W + w] = x[input(w, c)], and the per-word activation matrix used by the Pallas kernel is
X[w, c] = x[input(w, c)] (same data as x_pm reshaped [C, W] and transposed).
"""
import numpy as np
import jax
import jax.numpy as jnp

from glm53 import iqquant as Q

QK = Q.QK_K
U32 = np.dtype("<u4")

WORDS_PER_BLOCK = {"IQ2_S": 8, "IQ3_S": 16, "IQ4_XS": 32, "Q2_K": 16, "Q3_K": 16}
PLANES_PER_WORD = {"IQ2_S": 32, "IQ3_S": 16, "IQ4_XS": 8, "Q2_K": 16, "Q3_K": 16}   # C
PLANE_KEYS = {"IQ2_S": ("qs", "sg", "qh", "sc", "d"), "IQ3_S": ("qs", "sg", "qh", "sc", "d"),
              "IQ4_XS": ("qs", "sl", "sh", "d"), "Q2_K": ("qs", "sc", "d"), "Q3_K": ("qs", "hm", "sc", "d")}
# K-quants (the MTP layer's experts; no codebook): word w of a row = 4 qs bytes; block b = w // 16, half n = (w % 16)
# // 8, q = w % 8; byte j of the word is ggml byte l = 4 q + j of half n and holds 4 two-bit values (shift s):
#   input(w, c) = 256 b + 128 n + 32 s + 4 q + j with c = 4 s + j (C = 16).
#   Q2_K: sc u32 [4 nblk, R] (the block's 16 scale/min bytes: byte is = 8 n + 2 s + (q >= 4), low nibble scale,
#         high nibble min); d u32 [nblk, R] (f16 d in the low half, f16 dmin in the high half).
#   Q3_K: hm u32 [8 nblk, R] (u32 q of block b = hmask bytes 4 q .. 4 q + 3; bit 4 n + s of byte l = high bit);
#         sc u32 [4 nblk, R] (the 16 six-bit scales, pre-unpacked to bytes, same `is` indexing); d u32 [nblk, R].


def plane_rows(qtype, nblk):
    """rows_p per expert of every plane."""
    W = WORDS_PER_BLOCK[qtype] * nblk
    dd = (nblk + 1) // 2
    if qtype == "IQ2_S":
        return {"qs": W, "sg": W, "qh": W // 4, "sc": W // 4, "d": dd}
    if qtype == "IQ3_S":
        return {"qs": W, "sg": W // 2, "qh": W // 8, "sc": W // 16, "d": dd}
    if qtype == "IQ4_XS":
        return {"qs": W, "sl": nblk, "sh": nblk, "d": dd}
    if qtype == "Q2_K":
        return {"qs": W, "sc": 4 * nblk, "d": nblk}
    if qtype == "Q3_K":
        return {"qs": W, "hm": 8 * nblk, "sc": 4 * nblk, "d": nblk}
    raise ValueError(qtype)


def input_index(qtype, w, c):
    """Natural input index of value plane c of word w (numpy-vectorised over w, c)."""
    if qtype == "IQ2_S":
        return 32 * w + c                       # c = 8j + k
    if qtype == "IQ3_S":
        return 16 * w + c                       # c = 4j + k
    if qtype == "IQ4_XS":
        pos, j = c // 4, c % 4
        return 32 * (w // 4) + 16 * pos + 4 * (w % 4) + j
    if qtype in ("Q2_K", "Q3_K"):
        s_, j = c // 4, c % 4
        return 256 * (w // 16) + 128 * ((w % 16) // 8) + 32 * s_ + 4 * (w % 8) + j
    raise ValueError(qtype)


def pm_perm(qtype, n_inputs):
    """x_pm = x[perm]: perm[c*W + w] = input(w, c)."""
    W = n_inputs * WORDS_PER_BLOCK[qtype] // QK
    C = PLANES_PER_WORD[qtype]
    c, w = np.meshgrid(np.arange(C), np.arange(W), indexing="ij")
    return input_index(qtype, w, c).reshape(-1).astype(np.int32)


def pm_x(qtype, x):
    """x [..., n] natural -> X [..., W, C] with X[w, c] = x[input(w, c)] (reshape/transpose only, no gather)."""
    n = x.shape[-1]
    lead = x.shape[:-1]
    if qtype == "IQ2_S":
        return x.reshape(lead + (n // 32, 32))
    if qtype == "IQ3_S":
        return x.reshape(lead + (n // 16, 16))
    if qtype == "IQ4_XS":                                     # input = 32*sub + 16*pos + 4*wq + j -> [sub, pos, wq, j]
        y = x.reshape(lead + (n // 32, 2, 4, 4))
        y = jnp.swapaxes(y, -3, -2)                           # [sub, wq, pos, j]
        return y.reshape(lead + (n // 8, 8))                  # w = 4*sub + wq, c = 4*pos + j
    if qtype in ("Q2_K", "Q3_K"):                             # input = 256b + 128n + 32s + 4q + j -> [b, n, s, q, j]
        y = x.reshape(lead + (n // 256, 2, 4, 8, 4))
        y = jnp.swapaxes(y, -3, -2)                           # [b, n, q, s, j]
        return y.reshape(lead + (n // 16, 16))                # w = 16b + 8n + q, c = 4s + j
    raise ValueError(qtype)


def pm_flat(qtype, x):
    """x [..., n] natural -> x_pm [..., n] with x_pm[c*W + w] = x[input(w, c)]."""
    X = pm_x(qtype, x)
    return jnp.swapaxes(X, -1, -2).reshape(x.shape)


# ------------------------------------------------------------------------------------------------ numpy packing
def _u32(a):
    """[E, R, nblk, n] u8 -> [E, R, nblk*n/4] u32 (little-endian words, group order preserved)."""
    E, R = a.shape[:2]
    a = np.ascontiguousarray(a.reshape(E, R, -1))
    assert a.shape[-1] % 4 == 0, a.shape
    return a.view(U32)


def _merge(a):
    """[E, R, P] -> [E*P, R] (planes are stored expert-major, rows_p sublanes per expert, matrix rows on lanes)."""
    E, R, P = a.shape
    return np.ascontiguousarray(a.transpose(0, 2, 1)).reshape(E * P, R)


def pack_planes(t, qtype):
    """t uint8 [E, R, nblk, block_bytes] (one chip's slice of one expert table) -> {plane: u32 [E*rows_p, R]}."""
    E, R, nblk, bb = t.shape
    assert bb == Q.BLOCK_BYTES[qtype], (bb, qtype)
    f = lambda a, b: t[..., a:b]
    if qtype in ("Q2_K", "Q3_K"):
        if qtype == "Q2_K":
            out = {"qs": _u32(f(16, 80)), "sc": _u32(f(0, 16)), "d": _u32(f(80, 84))}
        else:
            sc16 = Q.q3k_scales(f(96, 108))                                              # [E, R, nblk, 16] u8
            d = np.concatenate([f(108, 110), np.zeros(t.shape[:3] + (2,), np.uint8)], -1)
            out = {"qs": _u32(f(32, 96)), "hm": _u32(f(0, 32)), "sc": _u32(sc16), "d": _u32(d)}
        rows = plane_rows(qtype, nblk)
        out = {k: _merge(out[k]) for k in PLANE_KEYS[qtype]}
        for k, a in out.items():
            assert a.shape == (E * rows[k], R), (k, a.shape, rows[k])
        return out
    d = t[..., 0:2].reshape(E, R, nblk * 2)
    if nblk % 2:
        d = np.concatenate([d, np.zeros((E, R, 2), np.uint8)], -1)
    out = {"d": d.copy().view(U32)}
    if qtype == "IQ2_S":
        out.update(qs=_u32(f(2, 34)), sg=_u32(f(34, 66)), qh=_u32(f(66, 74)), sc=_u32(f(74, 82)))
    elif qtype == "IQ3_S":
        out.update(qs=_u32(f(2, 66)), qh=_u32(f(66, 74)), sg=_u32(f(74, 106)), sc=_u32(f(106, 110)))
    elif qtype == "IQ4_XS":
        sh = np.concatenate([f(2, 4), np.zeros(t.shape[:3] + (2,), np.uint8)], -1)
        out.update(sh=_u32(sh), sl=_u32(f(4, 8)), qs=_u32(f(8, 136)))
    else:
        raise ValueError(qtype)
    rows = plane_rows(qtype, nblk)
    out = {k: _merge(out[k]) for k in PLANE_KEYS[qtype]}
    for k, a in out.items():
        assert a.shape == (E * rows[k], R), (k, a.shape, rows[k])
    return out


def planes_bytes(planes):
    return sum(int(a.nbytes) if hasattr(a, "nbytes") else int(np.prod(a.shape)) * 4 for a in planes.values())


# ------------------------------------------------------------------------------------------------ XLA dequant
def _f16_pairs(d, nblk):
    """d u32 [E, ceil(nblk/2), R] -> f32 [E, nblk, R] block scales."""
    E, _, R = d.shape
    halves = jnp.stack([d & 0xFFFF, d >> 16], axis=2).reshape(E, -1, R)[:, :nblk]        # [E, nblk, R] u32
    return jax.lax.bitcast_convert_type(halves.astype(jnp.uint16), jnp.float16).astype(jnp.float32)


def _bytes(a, n=4, bits=8):
    """u32 [E, P, R] -> [E, P*n, R] sub-fields of `bits` bits, field m of word p at index p*n + m."""
    E, P, R = a.shape
    mask = (1 << bits) - 1
    return jnp.stack([(a >> (bits * m)) & mask for m in range(n)], axis=2).reshape(E, P * n, R)


def _sgn(bits, k):
    return 1.0 - 2.0 * ((bits >> k) & 1).astype(jnp.float32)


def dequant_planes(planes, qtype, nblk):
    """planes {k: u32 [E, rows_p, R]} -> f32 [E, C*W, R] (pm order along axis 1)."""
    if qtype == "IQ2_S":
        return _deq_iq2_s(planes, nblk)
    if qtype == "IQ3_S":
        return _deq_iq3_s(planes, nblk)
    if qtype == "IQ4_XS":
        return _deq_iq4_xs(planes, nblk)
    if qtype in ("Q2_K", "Q3_K"):
        return _deq_kq(planes, qtype, nblk)
    raise ValueError(qtype)


def _f16_lo(u):
    return jax.lax.bitcast_convert_type((u & 0xFFFF).astype(jnp.uint16), jnp.float16).astype(jnp.float32)


def _kq_scale_index(W):
    """Scale byte of word w (into the block-major 16-byte scale lists): is = 8 n + 2 s + (q >= 4), block-offset 16 b."""
    w = np.arange(W)
    base = 16 * (w // 16) + 8 * ((w % 16) // 8) + ((w % 8) >= 4)
    return [base + 2 * s_ for s_ in range(4)]                                       # per shift s


def _deq_kq(p, qtype, nblk):
    qs = p["qs"].astype(jnp.int32)                                              # [E, W, R], W = 16 nblk
    E, W, R = qs.shape
    scb = _bytes(p["sc"]).astype(jnp.int32)                                     # [E, 16 nblk, R] scale bytes (`is` order)
    idx = _kq_scale_index(W)
    d = jnp.repeat(_f16_lo(p["d"]), 16, axis=1)                                 # [E, W, R]
    if qtype == "Q2_K":
        dmin = jnp.repeat(_f16_lo(p["d"] >> 16), 16, axis=1)
    else:
        hmb = _bytes(p["hm"]).astype(jnp.int32)                                 # [E, 32 nblk, R] hmask byte l of block b
        w = np.arange(W)
        hidx = [32 * (w // 16) + 4 * (w % 8) + j for j in range(4)]              # byte l = 4 q + j
        nhalf = (w % 16) // 8
    outs = []
    for c in range(16):
        s_, j = c // 4, c % 4
        v = ((qs >> (8 * j + 2 * s_)) & 3).astype(jnp.float32)
        sc = jnp.take(scb, jnp.asarray(idx[s_]), axis=1)                        # [E, W, R]
        if qtype == "Q2_K":
            outs.append(d * (sc & 15).astype(jnp.float32) * v - dmin * (sc >> 4).astype(jnp.float32))
        else:
            hb = (jnp.take(hmb, jnp.asarray(hidx[j]), axis=1) >> jnp.asarray(4 * nhalf + s_)[None, :, None]) & 1
            outs.append(d * (sc - 32).astype(jnp.float32) * (v - 4.0 * (1 - hb).astype(jnp.float32)))
    return jnp.stack(outs, axis=1).reshape(E, 16 * W, R)


def _deq_iq2_s(p, nblk):
    qs, sg = p["qs"].astype(jnp.int32), p["sg"].astype(jnp.int32)               # [E, W, R]
    E, W, R = qs.shape
    qhb = _bytes(p["qh"]).astype(jnp.int32)                                     # [E, W, R] qh byte of word w
    scb = _bytes(p["sc"]).astype(jnp.int32)                                     # [E, W, R]
    d = jnp.repeat(_f16_pairs(p["d"], nblk), 8, axis=1)                         # [E, W, R] block = w // 8
    scale = [d * (0.5 + ((scb >> (4 * h)) & 15).astype(jnp.float32)) * 0.25 for h in range(2)]
    outs = []
    for j in range(4):
        idx = ((qs >> (8 * j)) & 255) | (((qhb >> (2 * j)) & 3) << 8)
        sgn = (sg >> (8 * j)) & 255
        vals = Q._tree_pos("IQ2_S", idx)                                        # 8 int8 [E, W, R]
        outs += [scale[j // 2] * v.astype(jnp.float32) * _sgn(sgn, k) for k, v in enumerate(vals)]
    return jnp.stack(outs, axis=1).reshape(E, 32 * W, R)


def _deq_iq3_s(p, nblk):
    qs = p["qs"].astype(jnp.int32)                                              # [E, W, R]
    E, W, R = qs.shape
    sgh = _bytes(p["sg"], 2, 16).astype(jnp.int32)                              # [E, W, R] 16 sign bits of word w
    qhn = _bytes(p["qh"], 8, 4).astype(jnp.int32)                               # [E, W, R] 4 high bits of word w
    scn = jnp.repeat(_bytes(p["sc"], 8, 4), 2, axis=1).astype(jnp.float32)      # [E, W, R] scale nibble of word w
    d = jnp.repeat(_f16_pairs(p["d"], nblk), 16, axis=1)                        # [E, W, R] block = w // 16
    scale = d * (1.0 + 2.0 * scn)
    outs = []
    for j in range(4):
        idx = ((qs >> (8 * j)) & 255) | (((qhn >> j) & 1) << 8)
        sgn = (sgh >> (4 * j)) & 15
        vals = Q._tree_pos("IQ3_S", idx)                                        # 4 int8 [E, W, R]
        outs += [scale * v.astype(jnp.float32) * _sgn(sgn, k) for k, v in enumerate(vals)]
    return jnp.stack(outs, axis=1).reshape(E, 16 * W, R)


def _deq_iq4_xs(p, nblk):
    qs = p["qs"].astype(jnp.int32)                                              # [E, W, R], W = 32*nblk
    E, W, R = qs.shape
    sl = _bytes(p["sl"], 8, 4).astype(jnp.int32)                                # [E, 8*nblk, R] sub-block i
    sh = _bytes(p["sh"], 8, 2).astype(jnp.int32)                                # [E, 8*nblk, R]
    d = jnp.repeat(_f16_pairs(p["d"], nblk), 8, axis=1)                         # [E, 8*nblk, R]
    scale = jnp.repeat(d * ((sl | (sh << 4)) - 32).astype(jnp.float32), 4, axis=1)   # [E, W, R] sub-block = w // 4
    outs = []
    for pos in range(2):
        for j in range(4):
            nib = ((qs >> (8 * j + 4 * pos)) & 15)
            outs.append(scale * Q._kv16(nib))
    return jnp.stack(outs, axis=1).reshape(E, 8 * W, R)


def dequant_natural(planes, qtype, nblk):
    """Test helper: [E, C*W, R] pm order -> [E, R, n_inputs] natural order."""
    v = dequant_planes(planes, qtype, nblk)
    E, _, R = v.shape
    n = nblk * QK
    inv = np.argsort(pm_perm(qtype, n))
    return jnp.take(jnp.swapaxes(v, 1, 2), jnp.asarray(inv), axis=-1)
