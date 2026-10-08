"""Dequantization of llama.cpp "IQ" codebook formats (as used by Unsloth's UD-IQ3_XXS / UD-Q2_K_XL GLM-5.3-Flash GGUFs)
written in jax.numpy so the same code runs on CPU (tests vs gguf-py) and on TPU (inside the resident engine).

Block = 256 weights (QK_K). Inputs are the raw GGUF block bytes, uint8 [..., block_bytes]; output float [..., 256].
Layouts (bytes): IQ2_S 82 = d2|qs32|signs32|qh8|scales8 · IQ3_S 110 = d2|qs64|qh8|signs32|scales4 ·
IQ4_XS 136 = d2|sh2|sl4|qs128 · IQ2_XS 74 = d2|qs64(u16)|scales8 · IQ3_XXS 98 = d2|qs64|scales32(u32).
The grid lookup is `jnp.take(grid, idx)`; `LOOKUP` can be swapped for a faster TPU implementation.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

QK_K = 256
BLOCK_BYTES = {"IQ2_S": 82, "IQ3_S": 110, "IQ4_XS": 136, "IQ2_XS": 74, "IQ3_XXS": 98, "Q2_K": 84, "Q3_K": 110}
from glm53 import iq_grids_data as _gd
_g = _gd.load()
GRIDS = {k: np.asarray(_g[k]) for k in ("IQ2_S", "IQ3_S", "IQ2_XS", "IQ3_XXS")}   # int8 [entries, 8 or 4]
KSIGNS = np.asarray(_g["ksigns"])                                                  # uint8 [128]
KVALUES_IQ4NL = np.asarray(_g["kvalues_iq4nl"])                                    # int8 [16]


def _u16(b):   # uint8 [..., 2k] -> uint16 [..., k] (little endian)
    return jax.lax.bitcast_convert_type(b.reshape(b.shape[:-1] + (-1, 2)), jnp.uint16)


def _u32(b):
    return jax.lax.bitcast_convert_type(b.reshape(b.shape[:-1] + (-1, 4)), jnp.uint32)


def _f16(b):   # uint8 [..., 2] -> float32 [...]
    return jax.lax.bitcast_convert_type(b, jnp.float16).astype(jnp.float32)


def _bits8(b):   # uint8 [..., n] -> [..., n, 8] bit k of each byte (LSB first)
    return (b[..., None] >> jnp.arange(8, dtype=jnp.uint8)) & 1


def _nibbles(b):  # uint8 [..., n] -> [..., 2n] low nibble first
    return jnp.stack([b & 0x0F, b >> 4], -1).reshape(b.shape[:-1] + (-1,))


# ---------------------------------------------------------------------------------------------------- grid lookup
# XLA's gather is ~0.1 G lookups/s on v5e (measured) — useless. The "tree" implementation is gather-free: every grid
# entry is encoded as level indices (2 or 3 bits per value), bit-sliced into 32-entry uint32 words, and each output bit
# is fetched with a binary tree of `where`s over the word index (n_words-1 selects) + one variable shift.
LOOKUP_IMPL = "tree"          # "take" (jnp.take) or "tree"
LEVELS = {"IQ2_S": np.array([8, 25, 43], np.int8), "IQ2_XS": np.array([8, 25, 43], np.int8),
          "IQ3_S": np.array([1, 3, 5, 7, 9, 11, 13, 15], np.int8),
          "IQ3_XXS": np.array([4, 12, 20, 28, 36, 44, 52, 62], np.int8)}


class _TreeLUT:
    def __init__(self, name):
        grid, levels = GRIDS[name], LEVELS[name]
        self.levels = levels
        self.w = grid.shape[1]
        self.lb = int(np.ceil(np.log2(len(levels))))                       # bits per level index (2 or 3)
        n = grid.shape[0]
        lv = np.searchsorted(levels, grid)                                 # [n, w] level indices
        assert np.array_equal(levels[lv], grid)
        code = np.zeros(n, np.int64)
        for k in range(self.w):
            code |= lv[:, k].astype(np.int64) << (self.lb * k)
        self.nbits = self.lb * self.w
        self.n_words = n // 32
        # words[b][j] = 32 bits (entries 32j..32j+31) of output bit b
        self.words = np.zeros((self.nbits, self.n_words), np.uint32)
        for b in range(self.nbits):
            bits = ((code >> b) & 1).astype(np.uint32)
            self.words[b] = (bits.reshape(self.n_words, 32) << np.arange(32, dtype=np.uint32)).sum(1, dtype=np.uint32)

    def __call__(self, idx):
        wi = idx >> 5
        sh = (idx & 31).astype(jnp.uint32)
        bits = [((wi >> k) & 1).astype(bool) for k in range(int(np.log2(self.n_words)))]
        code = jnp.zeros(idx.shape, jnp.uint32)
        for b in range(self.nbits):
            nodes = [jnp.uint32(int(x)) for x in self.words[b]]
            for k, bit in enumerate(bits):                                 # select tree over word index bits
                nodes = [jnp.where(bit, nodes[2 * j + 1], nodes[2 * j]) for j in range(len(nodes) // 2)]
            word = nodes[0] if len(nodes) == 1 else jnp.broadcast_to(jnp.uint32(int(self.words[b][0])), idx.shape)
            code = code | (((word >> sh) & 1) << b)
        mask = (1 << self.lb) - 1
        lv = jnp.stack([((code >> (self.lb * k)) & mask).astype(jnp.int32) for k in range(self.w)], -1)   # [..., w]
        levels = self.levels
        out = jnp.full(lv.shape, int(levels[0]), jnp.int8)
        for i in range(1, len(levels)):
            out = jnp.where(lv == i, jnp.int8(int(levels[i])), out)
        return out


_TREES = {}


def lookup(name, idx):
    """name: grid name; idx int32 [...] -> int8 [..., w] grid values."""
    if LOOKUP_IMPL == "take":
        return jnp.take(jnp.asarray(GRIDS[name]), idx, axis=0)
    if name not in _TREES:
        _TREES[name] = _TreeLUT(name)
    return _TREES[name](idx)


def _signs_from_bits(signs_u8):   # uint8 [..., 32] -> float32 [..., 256] of ±1
    s = _bits8(signs_u8).reshape(signs_u8.shape[:-1] + (QK_K,))
    return 1.0 - 2.0 * s.astype(jnp.float32)


def dequant_iq2_s(blk):
    d = _f16(blk[..., 0:2])                                   # [...]
    qs = blk[..., 2:34].astype(jnp.int32)                     # [..., 32] low 8 bits of grid index (32 groups of 8)
    signs = _signs_from_bits(blk[..., 34:66])                 # [..., 256]
    qh = blk[..., 66:74]                                      # [..., 8] 2 bits per group, 4 groups per byte
    sc = _nibbles(blk[..., 74:82]).astype(jnp.float32)        # [..., 16] one 4-bit scale per 16 weights
    hi = jnp.stack([(qh >> (2 * k)) & 3 for k in range(4)], -1).reshape(qh.shape[:-1] + (32,)).astype(jnp.int32)
    idx = qs | (hi << 8)                                      # [..., 32]
    g = lookup("IQ2_S", idx).astype(jnp.float32).reshape(idx.shape[:-1] + (16, 16))  # 2 groups per scale
    db = d[..., None, None] * (0.5 + sc)[..., None] * 0.25
    return (db * g).reshape(idx.shape[:-1] + (QK_K,)) * signs


def dequant_iq3_s(blk):
    d = _f16(blk[..., 0:2])
    qs = blk[..., 2:66].astype(jnp.int32)                     # [..., 64] low 8 bits (64 groups of 4)
    qh = _bits8(blk[..., 66:74]).reshape(blk.shape[:-1] + (64,)).astype(jnp.int32)   # high bit per group
    signs = _signs_from_bits(blk[..., 74:106])
    sc = _nibbles(blk[..., 106:110]).astype(jnp.float32)      # [..., 8] one 4-bit scale per 32 weights
    idx = qs | (qh << 8)
    g = lookup("IQ3_S", idx).astype(jnp.float32).reshape(idx.shape[:-1] + (8, 32))   # 8 groups of 4 per scale
    db = d[..., None, None] * (1.0 + 2.0 * sc)[..., None]
    return (db * g).reshape(idx.shape[:-1] + (QK_K,)) * signs


def dequant_iq4_xs(blk):
    d = _f16(blk[..., 0:2])
    sh = _u16(blk[..., 2:4])[..., 0]                          # [...] 2 bits per 32-block (8 blocks)
    sl = _nibbles(blk[..., 4:8]).astype(jnp.int32)            # [..., 8]
    qs = blk[..., 8:136]                                      # [..., 128] two 4-bit values per byte, low nibble first
    shb = jnp.stack([(sh >> (2 * k)) & 3 for k in range(8)], -1).astype(jnp.int32)   # [..., 8]
    scales = (sl | (shb << 4)) - 32                           # [..., 8]
    q = qs.reshape(qs.shape[:-1] + (8, 16))                   # 8 sub-blocks of 16 bytes = 32 weights each
    q = jnp.concatenate([q & 0x0F, q >> 4], -1).astype(jnp.int32)   # [..., 8, 32]: first 16 low nibbles then 16 high
    v = _kv16(q)
    out = d[..., None, None] * scales.astype(jnp.float32)[..., None] * v
    return out.reshape(qs.shape[:-1] + (QK_K,))


def dequant_iq2_xs(blk):
    d = _f16(blk[..., 0:2])
    qs = _u16(blk[..., 2:66]).astype(jnp.int32)               # [..., 32]: 9-bit grid index | 7-bit sign index << 9
    sc = _nibbles(blk[..., 66:74]).astype(jnp.float32)        # [..., 16]
    sidx = qs >> 9
    sbits = jnp.take(jnp.asarray(KSIGNS), sidx, axis=0)       # uint8 [..., 32]
    signs = 1.0 - 2.0 * _bits8(sbits).astype(jnp.float32)     # [..., 32, 8]
    g = lookup("IQ2_XS", qs & 511).astype(jnp.float32) # [..., 32, 8]
    db = (d[..., None] * (0.5 + sc) * 0.25)                    # [..., 16]
    out = (g * signs).reshape(qs.shape[:-1] + (16, 16)) * db[..., None]
    return out.reshape(qs.shape[:-1] + (QK_K,))


def dequant_iq3_xxs(blk):
    d = _f16(blk[..., 0:2])
    qs = blk[..., 2:66].astype(jnp.int32)                     # [..., 64] 8-bit grid index (64 groups of 4)
    sc = _u32(blk[..., 66:98])                                 # [..., 8]: 4 x 7-bit sign idx | 4-bit scale << 28
    db = d[..., None] * (0.5 + (sc >> 28).astype(jnp.float32)) * 0.5     # [..., 8] one scale per 32 weights
    sidx = jnp.stack([(sc >> s) & 0x7F for s in (0, 7, 14, 21)], -1).astype(jnp.int32)   # [..., 8, 4]
    sbits = jnp.take(jnp.asarray(KSIGNS), sidx, axis=0)                                 # [..., 8, 4] uint8
    signs = 1.0 - 2.0 * _bits8(sbits).astype(jnp.float32)                               # [..., 8, 4, 8]
    g = lookup("IQ3_XXS", qs).astype(jnp.float32).reshape(qs.shape[:-1] + (8, 4, 8))   # 8 x (4 groups x 4... see below)
    # each 32-weight sub-block = 8 groups of 4 = the [8 groups] x [4 values]; regroup to match sign layout (4 sign-bytes x 8)
    g = g.reshape(qs.shape[:-1] + (8, 32)); signs = signs.reshape(qs.shape[:-1] + (8, 32))
    return (db[..., None] * g * signs).reshape(qs.shape[:-1] + (QK_K,))


# ------------------------------------------------------------------------------------ K-quants (the MTP layer's experts)
def _f16_at(blk, off):
    """f16 at byte offset `off` of every block -> f32 [..., nblk]."""
    u = blk[..., off].astype(jnp.uint32) | (blk[..., off + 1].astype(jnp.uint32) << 8)
    return jax.lax.bitcast_convert_type(u.astype(jnp.uint16), jnp.float16).astype(jnp.float32)


def _kq_layout(q):
    """q u8 [..., nblk, 64] -> list of 8 (n, s) groups of 32 2-bit values [..., nblk, 32] in ggml order
    (weight index 128 n + 32 s + l; byte q[32 n + l] shifted by 2 s)."""
    q = q.astype(jnp.int32)
    return [((q[..., 32 * n:32 * (n + 1)] >> (2 * s)) & 3) for n in range(2) for s in range(4)]


def q3k_scales(sc12):
    """Q3_K 12 packed bytes -> 16 six-bit scales (uint8 [..., 16]) in `is` order (ggml `dequantize_row_q3_K`)."""
    sc = np.asarray(sc12).astype(np.uint32)
    aux = [sc[..., 4 * i] | (sc[..., 4 * i + 1] << 8) | (sc[..., 4 * i + 2] << 16) | (sc[..., 4 * i + 3] << 24) for i in range(3)]
    k1, k2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
    tmp = aux[2]
    a2 = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4)
    a3 = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4)
    a0 = (aux[0] & k2) | (((tmp >> 0) & k1) << 4)
    a1 = (aux[1] & k2) | (((tmp >> 2) & k1) << 4)
    words = np.stack([a0, a1, a2, a3], axis=-1)                                   # [..., 4] u32
    return np.stack([(words >> (8 * b)) & 255 for b in range(4)], axis=-1).reshape(words.shape[:-1] + (16,)).astype(np.uint8)


def dequant_q2_k(blk):
    """blk uint8 [..., nblk, 84] -> float32 [..., nblk, 256] (natural order)."""
    sc = blk[..., 0:16].astype(jnp.int32)
    d, dmin = _f16_at(blk, 80), _f16_at(blk, 82)
    outs = []
    for g, v in enumerate(_kq_layout(blk[..., 16:80])):                          # g = 4 n + s
        for h in range(2):                                                        # l < 16 / l >= 16
            b = sc[..., 2 * g + h]
            dl, ml = d * (b & 15).astype(jnp.float32), dmin * (b >> 4).astype(jnp.float32)
            outs.append(dl[..., None] * v[..., 16 * h:16 * (h + 1)].astype(jnp.float32) - ml[..., None])
    return jnp.concatenate(outs, axis=-1)


def dequant_q3_k(blk):
    """blk uint8 [..., nblk, 110] -> float32 [..., nblk, 256] (natural order)."""
    hm = blk[..., 0:32].astype(jnp.int32)
    sc = jnp.asarray(q3k_scales(np.asarray(blk[..., 96:108]))).astype(jnp.int32) - 32
    d = _f16_at(blk, 108)
    outs = []
    for g, v in enumerate(_kq_layout(blk[..., 32:96])):
        hbit = (hm >> g) & 1                                                      # bit 4 n + s of hmask byte l
        val = v - 4 * (1 - hbit)
        for h in range(2):
            dl = d * sc[..., 2 * g + h].astype(jnp.float32)
            outs.append(dl[..., None] * val[..., 16 * h:16 * (h + 1)].astype(jnp.float32))
    return jnp.concatenate(outs, axis=-1)


DEQUANT = {"IQ2_S": dequant_iq2_s, "IQ3_S": dequant_iq3_s, "IQ4_XS": dequant_iq4_xs,
           "IQ2_XS": dequant_iq2_xs, "IQ3_XXS": dequant_iq3_xxs, "Q2_K": dequant_q2_k, "Q3_K": dequant_q3_k}


def dequant_rows(raw, qtype, ne0):
    """raw uint8 [nbytes] of a row-major GGUF tensor slice with ne0 weights per row -> float32 [rows, ne0]."""
    bb = BLOCK_BYTES[qtype]
    blk = jnp.asarray(raw).reshape(-1, ne0 // QK_K, bb)
    return DEQUANT[qtype](blk).reshape(blk.shape[0], ne0)


# ------------------------------------------------------------------------------------ position-major ("pm") dequant
# TPU pads the last two dims of every materialised array to (8,128) tiles, so [..., groups, 8] int8 intermediates cost
# 16x their size. Instead each codebook *position* is produced as its own lane-dense [..., n_groups] array and the row is
# emitted in position-major order: flat index = pos * G + group, G = groups per row (all blocks). The contraction dim of a
# matmul does not care about order as long as the activation is permuted the same way: x_pm = x[..., perm_pm(...)].
GROUP_W = {"IQ2_S": 8, "IQ2_XS": 8, "IQ3_S": 4, "IQ3_XXS": 4, "IQ4_XS": 2}   # positions per group (IQ4_XS: 2 nibbles)


def perm_pm(qtype, n_blocks):
    """Natural index of each position-major slot: x_pm = x[..., perm]; out_natural = out_pm[..., inv]."""
    w = GROUP_W[qtype]
    G = n_blocks * QK_K // w
    slots = np.arange(n_blocks * QK_K)
    if qtype == "IQ4_XS":   # weight d = sub*32 + pos*16 + j ; group g = sub*16 + j ; pos in {0 (low nibble), 1 (high)}
        pos, g = slots // G, slots % G
        return (32 * (g // 16) + 16 * pos + (g % 16)).astype(np.int32)
    pos, g = slots // G, slots % G                         # weight d = g*w + pos
    return (g * w + pos).astype(np.int32)


def _tree_pos(name, idx):
    """Codebook lookup returning one lane-dense int8 array per position: list of w arrays shaped like idx."""
    lut = _TREES.setdefault(name, _TreeLUT(name))
    wi = idx >> 5
    sh = (idx & 31).astype(jnp.uint32)
    bits = [((wi >> k) & 1).astype(bool) for k in range(int(np.log2(lut.n_words)))]
    code = jnp.zeros(idx.shape, jnp.uint32)
    for b in range(lut.nbits):
        nodes = [jnp.uint32(int(x)) for x in lut.words[b]]
        for bit in bits:
            nodes = [jnp.where(bit, nodes[2 * j + 1], nodes[2 * j]) for j in range(len(nodes) // 2)]
        code = code | (((nodes[0] >> sh) & 1) << b)
    mask = (1 << lut.lb) - 1
    outs = []
    for k in range(lut.w):
        lv = ((code >> (lut.lb * k)) & mask).astype(jnp.int32)
        v = jnp.full(lv.shape, int(lut.levels[0]), jnp.int8)
        for i in range(1, len(lut.levels)):
            v = jnp.where(lv == i, jnp.int8(int(lut.levels[i])), v)
        outs.append(v)
    return outs


def _rowflat(a):   # [..., nblk, m] -> [..., nblk*m]
    return a.reshape(a.shape[:-2] + (a.shape[-2] * a.shape[-1],))


def dequant_pm_iq2_s(blk):
    """blk uint8 [..., nblk, 82] -> float32 [..., 8 * nblk*32] position-major."""
    d = _f16(blk[..., 0:2])                                          # [..., nblk]
    qs = _rowflat(blk[..., 2:34].astype(jnp.int32))                  # [..., G]  G = nblk*32
    qh = blk[..., 66:74]
    hi = _rowflat(jnp.stack([(qh >> (2 * k)) & 3 for k in range(4)], -1).reshape(qh.shape[:-1] + (32,)).astype(jnp.int32))
    idx = qs | (hi << 8)                                             # [..., G]
    sc = _rowflat(_nibbles(blk[..., 74:82]).astype(jnp.float32))     # [..., nblk*16]  one per 2 groups
    scale = _rowflat(jnp.repeat(d, 16, axis=-1).reshape(d.shape + (16,))) * (0.5 + sc) * 0.25   # [..., nblk*16]
    scale = jnp.repeat(scale, 2, axis=-1)                            # [..., G]
    signs = _rowflat(blk[..., 34:66])                                # [..., G] one byte per group
    vals = _tree_pos("IQ2_S", idx)
    outs = [scale * v.astype(jnp.float32) * (1.0 - 2.0 * ((signs >> k) & 1).astype(jnp.float32)) for k, v in enumerate(vals)]
    return jnp.concatenate(outs, -1)


def dequant_pm_iq3_s(blk):
    """blk uint8 [..., nblk, 110] -> float32 [..., 4 * nblk*64] position-major."""
    d = _f16(blk[..., 0:2])
    qs = _rowflat(blk[..., 2:66].astype(jnp.int32))                  # [..., G]  G = nblk*64
    qh = _rowflat(_bits8(blk[..., 66:74]).reshape(blk.shape[:-1] + (64,)).astype(jnp.int32))
    idx = qs | (qh << 8)
    sc = _rowflat(_nibbles(blk[..., 106:110]).astype(jnp.float32))   # [..., nblk*8] one per 8 groups
    scale = _rowflat(jnp.repeat(d, 8, axis=-1).reshape(d.shape + (8,))) * (1.0 + 2.0 * sc)
    scale = jnp.repeat(scale, 8, axis=-1)                            # [..., G]
    sb = _rowflat(blk[..., 74:106])                                  # [..., nblk*32] sign bytes: byte j = groups 2j, 2j+1
    s_even = jnp.repeat(sb, 2, axis=-1)                              # [..., G] byte for each group
    shift = jnp.tile(jnp.array([0, 4], jnp.uint8), sb.shape[-1])     # group parity picks the nibble
    vals = _tree_pos("IQ3_S", idx)
    outs = [scale * v.astype(jnp.float32) * (1.0 - 2.0 * ((s_even >> (shift + k)) & 1).astype(jnp.float32))
            for k, v in enumerate(vals)]
    return jnp.concatenate(outs, -1)


def dequant_pm_iq4_xs(blk):
    """blk uint8 [..., nblk, 136] -> float32 [..., 2 * nblk*128] position-major (pos = nibble)."""
    d = _f16(blk[..., 0:2])
    sh = _u16(blk[..., 2:4])[..., 0]
    sl = _nibbles(blk[..., 4:8]).astype(jnp.int32)                   # [..., nblk, 8]
    shb = jnp.stack([(sh >> (2 * k)) & 3 for k in range(8)], -1).astype(jnp.int32)
    scales = ((sl | (shb << 4)) - 32).astype(jnp.float32) * d[..., None]          # [..., nblk, 8] per 32 weights
    scale = _rowflat(jnp.repeat(scales, 16, axis=-1))                # [..., nblk*128] per group (16 groups per sub-block)
    qs = _rowflat(blk[..., 8:136])                                   # [..., G] G = nblk*128
    lo = _kv16(qs & 0x0F)
    hi = _kv16(qs >> 4)
    return jnp.concatenate([scale * lo, scale * hi], -1)


def _kv16(nib):
    """16-entry IQ4_NL value table without a gather (XLA's generic gather ran at 0.11 G lookups/s = 74 ms per
    IQ4_XS layer): 4-bit nibble -> float32 via a 4-level select tree (15 selects)."""
    kv = [float(v) for v in KVALUES_IQ4NL]
    bits = [((nib >> k) & 1).astype(bool) for k in range(4)]
    nodes = [jnp.float32(v) for v in kv]
    for bit in bits:
        nodes = [jnp.where(bit, nodes[2 * j + 1], nodes[2 * j]) for j in range(len(nodes) // 2)]
    return nodes[0]


DEQUANT_PM = {"IQ2_S": dequant_pm_iq2_s, "IQ3_S": dequant_pm_iq3_s, "IQ4_XS": dequant_pm_iq4_xs}


def pm_permute(qtype, x):
    """x[..., n] natural order -> position-major order along the last axis, via reshapes/transposes only
    (XLA's generic gather costs ~2.5 ms per call on v5e, a transpose is free). Equals jnp.take(x, perm_pm(...), -1)."""
    n = x.shape[-1]
    lead = x.shape[:-1]
    if qtype == "IQ4_XS":                      # d = sub*32 + pos*16 + j  ->  [pos, sub, j]
        y = x.reshape(lead + (n // 32, 2, 16))
        return jnp.swapaxes(y, -3, -2).reshape(lead + (n,))
    w = GROUP_W[qtype]                          # d = g*w + pos  ->  [pos, g]
    y = x.reshape(lead + (n // w, w))
    return jnp.swapaxes(y, -2, -1).reshape(lead + (n,))
