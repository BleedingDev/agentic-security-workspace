import json, os, numpy as np, pytest
import jax.numpy as jnp
import gguf.quants as gq
from gguf import GGMLQuantizationType as T
from glm53 import iqquant

HERE = os.path.dirname(__file__)
SAMPLES = json.load(open(os.path.join(HERE, "data", "gguf_samples.json")))


@pytest.mark.parametrize("qtype", ["IQ2_S", "IQ3_S", "IQ4_XS"])
def test_real_samples_match_gguf_py(qtype):
    s = SAMPLES[qtype]
    raw = np.fromfile(os.path.join(os.path.dirname(HERE), "..", s["file"]), dtype=np.uint8)
    ref = gq.dequantize(raw, getattr(T, qtype)).reshape(s["rows"], s["ne0"])
    out = np.asarray(iqquant.dequant_rows(raw, qtype, s["ne0"]))
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-7)
    assert np.abs(ref).max() > 0


@pytest.mark.parametrize("qtype", ["IQ2_S", "IQ3_S", "IQ4_XS", "IQ2_XS", "IQ3_XXS"])
def test_random_blocks_match_gguf_py(qtype):
    rng = np.random.default_rng(1)
    bb = iqquant.BLOCK_BYTES[qtype]
    raw = rng.integers(0, 256, 64 * bb, dtype=np.uint8)
    raw = raw.reshape(64, bb)
    raw[:, 0:2] = np.frombuffer(rng.uniform(1e-3, 2e-2, 64).astype(np.float16).tobytes(), np.uint8).reshape(64, 2)
    raw = raw.reshape(-1)
    ref = gq.dequantize(raw, getattr(T, qtype)).reshape(-1, 256 * 4)
    out = np.asarray(iqquant.dequant_rows(raw, qtype, 256 * 4))
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("name", ["IQ2_S", "IQ3_S", "IQ2_XS", "IQ3_XXS"])
def test_tree_lookup_equals_take(name):
    rng = np.random.default_rng(3)
    n = iqquant.GRIDS[name].shape[0]
    idx = rng.integers(0, n, (7, 1000), dtype=np.int32)
    idx[0, :n] = np.arange(n)[:1000]
    ref = np.asarray(iqquant.GRIDS[name])[idx]
    out = np.asarray(iqquant._TreeLUT(name)(np.asarray(idx)))
    assert np.array_equal(out, ref)


def test_dequant_same_under_both_impls():
    rng = np.random.default_rng(4)
    raw = rng.integers(0, 256, 16 * 82, dtype=np.uint8)
    old = iqquant.LOOKUP_IMPL
    try:
        iqquant.LOOKUP_IMPL = "take"; a = np.asarray(iqquant.dequant_rows(raw, "IQ2_S", 1024))
        iqquant.LOOKUP_IMPL = "tree"; b = np.asarray(iqquant.dequant_rows(raw, "IQ2_S", 1024))
    finally:
        iqquant.LOOKUP_IMPL = old
    assert np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize("qtype", ["IQ2_S", "IQ3_S", "IQ4_XS"])
def test_position_major_dequant(qtype):
    rng = np.random.default_rng(5)
    bb, nblk, rows = iqquant.BLOCK_BYTES[qtype], 3, 5
    raw = rng.integers(0, 256, (rows, nblk, bb), dtype=np.uint8)
    raw[..., 0:2] = np.frombuffer(rng.uniform(1e-3, 2e-2, rows * nblk).astype(np.float16).tobytes(), np.uint8).reshape(rows, nblk, 2)
    ref = np.asarray(iqquant.DEQUANT[qtype](jnp.asarray(raw))).reshape(rows, nblk * 256)
    pm = np.asarray(iqquant.DEQUANT_PM[qtype](jnp.asarray(raw)))
    perm = iqquant.perm_pm(qtype, nblk)
    assert pm.shape == (rows, nblk * 256) and sorted(perm.tolist()) == list(range(nblk * 256))
    np.testing.assert_allclose(pm, ref[:, perm], rtol=1e-6, atol=1e-7)
    # matvec invariance: x . ref == x[perm] . pm
    x = rng.standard_normal(nblk * 256).astype(np.float32)
    np.testing.assert_allclose(pm @ x[perm], ref @ x, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("qtype", ["IQ2_S", "IQ3_S", "IQ4_XS", "IQ2_XS", "IQ3_XXS"])
def test_pm_permute_matches_perm(qtype):
    rng = np.random.default_rng(6)
    x = rng.standard_normal((3, 3 * 256)).astype(np.float32)
    ref = x[:, iqquant.perm_pm(qtype, 3)]
    out = np.asarray(iqquant.pm_permute(qtype, jnp.asarray(x)))
    assert np.array_equal(out, ref)
