"""Planar expert tables (glm53.planes) and the fused Pallas dequant-matvec kernel (glm53.pallas_moe, interpret mode)."""
import json
import os
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from glm53 import iqquant as Q
from glm53 import planes as PL
from glm53 import pallas_moe as K

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))


def random_blocks(rng, E, R, nblk, qt, scale=0.02):
    bb = Q.BLOCK_BYTES[qt]
    t = rng.integers(0, 256, (E, R, nblk, bb), dtype=np.uint8)
    d = rng.uniform(0.5 * scale, scale, (E, R, nblk)).astype(np.float16)
    off = {"Q2_K": 80, "Q3_K": 108}.get(qt, 0)                      # K-quants keep d (and dmin) at the block end
    t[..., off:off + 2] = np.frombuffer(d.tobytes(), np.uint8).reshape(E, R, nblk, 2)
    if qt == "Q2_K":
        dm = rng.uniform(0.05 * scale, 0.2 * scale, (E, R, nblk)).astype(np.float16)
        t[..., 82:84] = np.frombuffer(dm.tobytes(), np.uint8).reshape(E, R, nblk, 2)
    return t


@pytest.mark.parametrize("qt", ["Q2_K", "Q3_K"])
def test_kquant_matches_gguf_py(qt):
    """The K-quant reference dequant and the planar XLA dequant are bit-exact vs gguf-py."""
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType as G
    rng = np.random.default_rng(5)
    E, R, nblk = 2, 8, 3
    t = random_blocks(rng, E, R, nblk, qt, scale=0.05)
    ref = dequantize(t.reshape(E * R, nblk * Q.BLOCK_BYTES[qt]), getattr(G, qt)).reshape(E, R, nblk * 256)
    ours = np.asarray(Q.DEQUANT[qt](jnp.asarray(t))).reshape(E, R, nblk * 256)
    np.testing.assert_array_equal(ours, ref)
    planes, rows = as_planes(t, qt)
    nat = np.asarray(PL.dequant_natural({k: a.reshape(E, rows[k], R) for k, a in planes.items()}, qt, nblk))
    np.testing.assert_array_equal(nat, ref)


def as_planes(t, qt, lead=False):
    nblk = t.shape[2]
    rows = PL.plane_rows(qt, nblk)
    p = {k: jnp.asarray(a) for k, a in PL.pack_planes(t, qt).items()}
    return {k: (a[None] if lead else a) for k, a in p.items()}, rows


@pytest.mark.parametrize("qt,R,nblk", [("IQ2_S", 16, 16), ("IQ3_S", 16, 16), ("IQ3_S", 24, 1), ("IQ4_XS", 24, 1),
                                        ("IQ4_XS", 8, 2), ("IQ2_S", 8, 3), ("Q2_K", 8, 3), ("Q3_K", 16, 2)])
def test_pack_and_xla_dequant_exact(qt, R, nblk):
    rng = np.random.default_rng(0)
    E = 3
    t = random_blocks(rng, E, R, nblk, qt)
    ref = np.asarray(Q.DEQUANT[qt](jnp.asarray(t))).reshape(E, R, nblk * 256)
    planes, rows = as_planes(t, qt)
    for k, a in planes.items():
        assert a.shape == (E * rows[k], R) and a.dtype == jnp.uint32
    pj = {k: a.reshape(E, rows[k], R) for k, a in planes.items()}
    out = np.asarray(PL.dequant_natural(pj, qt, nblk))
    assert np.array_equal(out, ref)


@pytest.mark.parametrize("qt", ["IQ2_S", "IQ3_S", "IQ4_XS"])
def test_real_samples_exact(qt):
    s = json.load(open(os.path.join(HERE, "data", "gguf_samples.json")))[qt]
    raw = np.fromfile(os.path.join(ROOT, s["file"]), np.uint8)
    nblk, R = s["ne0"] // 256, s["rows"]
    t = raw.reshape(1, R, nblk, Q.BLOCK_BYTES[qt])
    ref = np.asarray(Q.DEQUANT[qt](jnp.asarray(t))).reshape(1, R, s["ne0"])
    planes, rows = as_planes(t, qt)
    pj = {k: a.reshape(1, rows[k], R) for k, a in planes.items()}
    assert np.array_equal(np.asarray(PL.dequant_natural(pj, qt, nblk)), ref)


@pytest.mark.parametrize("qt,n", [("IQ2_S", 4096), ("IQ3_S", 256), ("IQ4_XS", 256), ("IQ3_S", 4096), ("Q2_K", 4096),
                                  ("Q3_K", 256)])
def test_pm_permutations(qt, n):
    x = jnp.arange(n, dtype=jnp.float32)
    perm = PL.pm_perm(qt, n)
    assert sorted(perm.tolist()) == list(range(n))
    assert np.array_equal(np.asarray(PL.pm_flat(qt, x)), np.asarray(x)[perm])
    W, C = PL.WORDS_PER_BLOCK[qt] * n // 256, PL.PLANES_PER_WORD[qt]
    X = np.asarray(PL.pm_x(qt, x))
    assert X.shape == (W, C) and np.array_equal(X.T.reshape(-1), np.asarray(x)[perm])


@pytest.mark.parametrize("qt,R,nblk,lead", [("IQ2_S", 256, 4, True), ("IQ3_S", 256, 2, False), ("IQ3_S", 512, 1, True),
                                             ("Q2_K", 256, 2, True), ("Q3_K", 256, 1, False),
                                             ("IQ4_XS", 512, 1, True), ("IQ2_S", 128, 1, True),
                                             ("IQ3_S", 2048, 1, True), ("IQ3_S", 8192, 1, True)])
def test_kernel_interpret_matches_dense(qt, R, nblk, lead):
    rng = np.random.default_rng(1)
    E, Nk = 6, 5
    t = random_blocks(rng, E, R, nblk, qt)
    planes, _ = as_planes(t, qt, lead)
    idx = jnp.asarray(rng.integers(0, E, Nk, dtype=np.int32))
    n_in = nblk * 256
    x = jnp.asarray(rng.standard_normal((Nk, n_in)).astype(np.float32))
    X = PL.pm_x(qt, x)
    dense = np.asarray(Q.DEQUANT[qt](jnp.asarray(t))).reshape(E, R, n_in)
    want = np.stack([dense[int(idx[i])] @ np.asarray(x[i]) for i in range(Nk)])
    ref = np.asarray(K.moe_matvec_ref(planes, qt, nblk, idx, X))
    np.testing.assert_allclose(ref, want, rtol=1e-4, atol=1e-4)
    out = np.asarray(K.moe_matvec(planes, qt, nblk, idx, X, interpret=True))
    np.testing.assert_allclose(out, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("qt", ["IQ2_S", "IQ3_S"])
def test_code_table_decodes_grid(qt):
    """Every grid entry of the pair-packed, pre-encoded `code_table` gives back the grid's values the way the decoders
    read it (IQ2_S: f32 bits base + u * step from the 2-bit field; IQ3_S: the 4-bit field)."""
    grid = np.asarray(Q.GRIDS[qt])
    words = K.code_table(qt).reshape(-1)                                     # int32
    e = np.arange(grid.shape[0])
    code = words[e >> 1].astype(np.int64) >> (16 * (e & 1))                 # arithmetic shift, as in _lookup
    for k in range(grid.shape[1]):
        if qt == "IQ2_S":
            base, step = K.IQ2S_F32
            bits = ((code & (3 << (2 * k))) * (step >> (2 * k)) + base).astype(np.uint32)
            v = bits.view(np.float32)
        else:
            v = (code >> (4 * k)) & 15
        assert np.array_equal(v, grid[:, k]), (qt, k)


@pytest.mark.parametrize("qt,R,nblk,lead", [("IQ2_S", 256, 2, True), ("IQ3_S", 128, 2, False)])
def test_gateup_kernel_interpret_matches_dense(qt, R, nblk, lead):
    rng = np.random.default_rng(6)
    E, Nk = 6, 5
    tg, tu = random_blocks(rng, E, R, nblk, qt), random_blocks(rng, E, R, nblk, qt)
    pg, _ = as_planes(tg, qt, lead)
    pu, _ = as_planes(tu, qt, lead)
    idx = jnp.asarray(rng.integers(0, E, Nk, dtype=np.int32))
    n_in = nblk * 256
    x = jnp.asarray(rng.standard_normal((Nk, n_in)).astype(np.float32))
    g, u = K.moe_matvec_gu(pg, pu, qt, nblk, idx, PL.pm_x(qt, x), interpret=True)
    for t, out in ((tg, g), (tu, u)):
        dense = np.asarray(Q.DEQUANT[qt](jnp.asarray(t))).reshape(E, R, n_in)
        want = np.stack([dense[int(idx[i])] @ np.asarray(x[i]) for i in range(Nk)])
        np.testing.assert_allclose(np.asarray(out), want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("qt_gu,nblk_gu", [("IQ2_S", 2), ("IQ3_S", 1)])
def test_sweep_gateup_interpret(qt_gu, nblk_gu):
    from glm53 import model as M
    rng = np.random.default_rng(2)
    E, T, R = 12, 16, 128
    n_in = nblk_gu * 256
    tg, tu = random_blocks(rng, E, R, nblk_gu, qt_gu), random_blocks(rng, E, R, nblk_gu, qt_gu)
    pg, _ = as_planes(tg, qt_gu, True)
    pu, _ = as_planes(tu, qt_gu, True)
    x = rng.standard_normal((T, n_in)).astype(np.float32)
    idx = jnp.asarray(rng.integers(0, 6, (T, 2), dtype=np.int32))          # experts 6..11 never routed
    ids, n_act = K.active_slots(idx, E)
    assert int(n_act) <= 6 and np.all(np.asarray(ids)[int(n_act):] == np.asarray(ids)[int(n_act) - 1])
    xk = K.pm_mxu(qt_gu, jnp.asarray(x)).astype(jnp.bfloat16)
    h = np.asarray(K.moe_sweep_gateup(pg, pu, qt_gu, nblk_gu, ids, n_act, xk, 10.0, interpret=True, t_block=8).astype(jnp.float32))
    bf = lambda a: np.asarray(jnp.asarray(a).astype(jnp.bfloat16).astype(jnp.float32))
    dg = bf(np.asarray(Q.DEQUANT[qt_gu](jnp.asarray(tg))).reshape(E, R, n_in))
    du = bf(np.asarray(Q.DEQUANT[qt_gu](jnp.asarray(tu))).reshape(E, R, n_in))
    xb = bf(x)
    want = np.zeros((E, T, R), np.float32)
    for sl in range(int(n_act)):
        e = int(ids[sl])
        want[sl] = np.asarray(M.swiglu_clamped(jnp.asarray(xb @ dg[e].T), jnp.asarray(xb @ du[e].T), 10.0))
    np.testing.assert_allclose(h, want, rtol=2e-2, atol=2e-2 * np.abs(want).max())
    assert np.all(h[int(n_act):] == 0)


@pytest.mark.parametrize("qt_dn", ["IQ3_S", "IQ4_XS"])
def test_sweep_down_interpret(qt_dn):
    rng = np.random.default_rng(3)
    E, T, R, ml = 12, 16, 256, 256
    td = random_blocks(rng, E, R, 1, qt_dn)
    pd, _ = as_planes(td, qt_dn, True)
    hs = rng.standard_normal((E, T, ml)).astype(np.float32)
    idx = jnp.asarray(rng.integers(0, 5, (T, 2), dtype=np.int32))
    ids, n_act = K.active_slots(idx, E)
    hk = K.pm_mxu(qt_dn, jnp.asarray(hs)).astype(jnp.bfloat16)
    y = np.asarray(K.moe_sweep_down(pd, qt_dn, 1, ids, n_act, hk, interpret=True, t_block=8))
    bf = lambda a: np.asarray(jnp.asarray(a).astype(jnp.bfloat16).astype(jnp.float32))
    dd = bf(np.asarray(Q.DEQUANT[qt_dn](jnp.asarray(td))).reshape(E, R, ml))
    hb = bf(hs)
    want = np.zeros((T, R), np.float32)
    for sl in range(int(n_act)):
        want += hb[sl] @ dd[int(ids[sl])].T
    np.testing.assert_allclose(y, want, rtol=1e-4, atol=1e-4 * np.abs(want).max())
