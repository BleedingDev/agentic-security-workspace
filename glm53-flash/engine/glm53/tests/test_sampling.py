"""Device-side sampler (`engine.device_sample`, vocab columns sharded over 8 CPU devices): temperature 0 is the
global argmax; temperature / top-p sampling reproduces the truncated softmax (frequencies over many rows, the host
sampler's top-p rule); tokens outside the nucleus or outside the per-chip candidate set are never drawn."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402

from glm53.engine import AXIS, R, shard_map, device_sample, DeviceSampler  # noqa: E402


def run(z, temperature, top_p, seed, n_cand=2048, impl="approx"):
    mesh = Mesh(np.array(jax.devices()), (AXIS,))

    def prog(zl, t, p, key):
        return device_sample(zl, t, p, key, n_cand, impl)
    f = jax.jit(shard_map(prog, mesh=mesh, in_specs=(P(None, AXIS), R, R, R), out_specs=R, check_vma=False))
    return np.asarray(f(jnp.asarray(z, jnp.float32), *DeviceSampler(temperature, top_p, seed=seed).bind(mesh)))


def host_probs(z, temperature, top_p):
    """The server's host rule: softmax(z / T) over the candidates, keep count(cum <= top_p) + 1 of them."""
    p = np.exp((z - z.max()) / temperature); p /= p.sum()
    order = np.argsort(-p); cum = np.cumsum(p[order])
    keep = order[: max(1, int((cum <= top_p).sum()) + 1)]
    q = np.zeros_like(p); q[keep] = p[keep]
    return q / q.sum()


@pytest.mark.parametrize("impl", ["approx", "sort"])
def test_greedy_is_global_argmax(impl):
    rng = np.random.default_rng(0)
    z = rng.standard_normal((37, 8 * 96)).astype(np.float32) * 3
    ids = run(z, 0.0, 1.0, seed=0, impl=impl)
    np.testing.assert_array_equal(ids, z.argmax(-1))
    ids = run(z, 0.0, 0.3, seed=5, impl=impl)              # top_p is irrelevant at temperature 0
    np.testing.assert_array_equal(ids, z.argmax(-1))
    z = rng.standard_normal((200, 8 * 4096)).astype(np.float32) * 3     # real-size rows: 256 candidates per chip
    z[np.arange(200), rng.integers(0, 8 * 4096, 200)] += 40
    ids = run(z, 0.0, 1.0, seed=0, n_cand=2048, impl=impl)
    np.testing.assert_array_equal(ids, z.argmax(-1))


@pytest.mark.parametrize("impl", ["approx", "sort"])
@pytest.mark.parametrize("temperature,top_p", [(0.8, 1.0), (1.0, 0.6), (1.3, 0.9)])
def test_sampling_matches_truncated_softmax(temperature, top_p, impl):
    rng = np.random.default_rng(1)
    V, N = 16, 40000                                       # 2 vocab columns per chip; N identical rows = N draws
    z = (rng.standard_normal(V) * 1.5).astype(np.float32)
    ids = run(np.broadcast_to(z, (N, V)).copy(), temperature, top_p, seed=3, impl=impl)
    q = host_probs(z, temperature, top_p)
    freq = np.bincount(ids, minlength=V) / N
    assert np.all(freq[q == 0] == 0), "a token outside the nucleus was drawn"
    assert 0.5 * np.abs(freq - q).sum() < 0.03, (freq, q)


def test_candidate_truncation_and_key_determinism():
    rng = np.random.default_rng(2)
    V, N = 8 * 64, 4000
    row = rng.standard_normal(V).astype(np.float32) * 0.1
    big = rng.choice(64, 20, replace=False)                # 20 large logits, all on chip 0 (columns 0..63)
    row[big] += 40 + 3 * rng.standard_normal(20).astype(np.float32)
    z = np.broadcast_to(row, (N, V)).copy()
    ids = run(z, 1.0, 1.0, seed=4, n_cand=64)              # 8 candidates per chip: only chip 0's top 8 can be drawn
    top8 = set(np.argsort(-row[:64])[:8].tolist())
    assert set(ids.tolist()) <= top8, set(ids.tolist()) - top8
    assert len(set(ids.tolist())) > 1
    assert np.array_equal(run(z[:5], 1.0, 1.0, seed=4), run(z[:5], 1.0, 1.0, seed=4))
    assert not np.array_equal(run(z[:50], 1.0, 1.0, seed=4), run(z[:50], 1.0, 1.0, seed=9))


def test_approx_candidates_carry_the_mass():
    """Real-size rows (8 x 19360 columns, 256 candidates per chip, approx_max_k): at temperature 1 the drawn tokens
    fall in the exact top 64 essentially always, and the exact argmax is always a candidate (greedy checked above)."""
    rng = np.random.default_rng(5)
    V, N = 8 * 19360, 512
    z = (rng.standard_normal((N, V)) * 2).astype(np.float32)
    hot = rng.integers(0, V, (N, 8))
    np.put_along_axis(z, hot, z.max() + rng.uniform(4, 8, (N, 8)).astype(np.float32), axis=1)
    ids = run(z, 1.0, 0.95, seed=6, impl="approx")
    top64 = np.argsort(-z, axis=1)[:, :64]
    assert np.mean([ids[i] in top64[i] for i in range(N)]) > 0.99
