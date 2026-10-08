"""Speculative acceptance by rejection sampling (`engine.spec_accept`, used by the DFlash verify's head): with drafts
drawn from the drafter's q, every emitted token is distributed as the target's p (the first token, the token after
an accepted draft, the bonus after a fully accepted block); one-hot p and q (temperature 0) give "accepted iff equal
to the argmax, then the argmax"; a drafted stop token ends the run and is the emitted token when accepted."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from glm53.engine import spec_accept  # noqa: E402

V, K = 12, 4


def dist(rng, zeros=0):
    p = rng.random(V) ** 2
    p[rng.choice(V, zeros, replace=False)] = 0              # a top-p cut
    return p / p.sum()


def setup(rng, N, T, p_rows, cand_rows, q_rows):
    """N identical rows; drafts drawn per row from q over the candidates."""
    gidx = np.broadcast_to(np.arange(V, dtype=np.int32), (N, T, V))
    p = np.broadcast_to(np.stack(p_rows), (N, T, V)).astype(np.float32)
    cand = np.broadcast_to(np.stack(cand_rows), (N, T - 1, K)).astype(np.int32)
    q = np.broadcast_to(np.stack(q_rows), (N, T - 1, K)).astype(np.float32)
    d = np.stack([np.stack([rng.choice(cand_rows[t], p=q_rows[t]) for t in range(T - 1)]) for _ in range(N)])
    return jnp.asarray(gidx), jnp.asarray(p), jnp.asarray(d, jnp.int32), jnp.asarray(cand), jnp.asarray(q)


def tv(samples, p):
    return 0.5 * np.abs(np.bincount(samples, minlength=V) / len(samples) - p).sum()


def test_emitted_tokens_follow_the_target():
    rng = np.random.default_rng(0)
    N, T = 200000, 3
    ps = [dist(rng, 3), dist(rng, 5), dist(rng, 0)]
    cands = [rng.choice(V, K, replace=False) for _ in range(T - 1)]
    cands[0][0] = int(np.argmax(ps[0] == 0))               # a candidate the target never emits (p = 0)
    qs = [rng.dirichlet(np.ones(K)) for _ in range(T - 1)]
    gidx, p, d, cand, q = setup(rng, N, T, ps, cands, qs)
    a, bonus = jax.jit(spec_accept)(gidx, p, d, cand, q, jax.random.PRNGKey(7))
    a, bonus, d = np.asarray(a), np.asarray(bonus), np.asarray(d)
    emitted = [np.concatenate([d[i, :a[i]], [bonus[i]]]) for i in range(N)]
    first = np.array([e[0] for e in emitted])
    assert tv(first, ps[0]) < 0.008, (np.bincount(first, minlength=V) / N, ps[0])
    second = np.array([e[1] for e in emitted if len(e) > 1])          # after an accepted first draft
    assert len(second) > 20000 and tv(second, ps[1]) < 0.02
    last = bonus[a == T - 1]                                          # the bonus after a fully accepted block
    assert len(last) > 4000 and tv(last, ps[2]) < 0.04
    # acceptance = sum(min(p, q)) per position: the first draft's rate
    qfull = np.zeros(V); qfull[cands[0]] = qs[0]
    np.testing.assert_allclose((a >= 1).mean(), np.minimum(ps[0], qfull).sum(), atol=0.01)


def test_one_hot_is_greedy():
    rng = np.random.default_rng(1)
    N, T = 64, 5
    g = rng.integers(0, V, (N, T))                                    # the target's argmax per position
    p = jax.nn.one_hot(g, V, dtype=jnp.float32)
    gidx = jnp.broadcast_to(jnp.arange(V, dtype=jnp.int32), (N, T, V))
    d = np.where(rng.random((N, T - 1)) < 0.7, g[:, :-1], (g[:, :-1] + 1) % V)
    cand = np.stack([d, (d + 3) % V, (d + 5) % V, (d + 7) % V], -1)  # the drafter chose d: q one-hot on it
    q = jax.nn.one_hot(np.zeros((N, T - 1), int), K, dtype=jnp.float32)
    a, bonus = spec_accept(gidx, p, jnp.asarray(d), jnp.asarray(cand), q, jax.random.PRNGKey(0))
    ok = d == g[:, :-1]
    a_ref = np.cumprod(ok, 1).sum(1)
    np.testing.assert_array_equal(np.asarray(a), a_ref)
    np.testing.assert_array_equal(np.asarray(bonus), g[np.arange(N), a_ref])
    # drafts that are not the drafter's own choice (overrides): still exact
    cand2 = (cand + 1) % V
    a2, b2 = spec_accept(gidx, p, jnp.asarray(d), jnp.asarray(cand2), q, jax.random.PRNGKey(1))
    np.testing.assert_array_equal(np.asarray(a2), a_ref)
    np.testing.assert_array_equal(np.asarray(b2), g[np.arange(N), a_ref])


def test_stop_draft_ends_the_run():
    N, T, stop = 8, 4, 5
    g = np.array([[1, stop, 2, 3]] * N)                               # the target wants the stop token at position 1
    p = jax.nn.one_hot(g, V, dtype=jnp.float32)
    gidx = jnp.broadcast_to(jnp.arange(V, dtype=jnp.int32), (N, T, V))
    d = g[:, :-1].copy()
    cand = np.stack([d, (d + 1) % V, (d + 2) % V, (d + 3) % V], -1)
    q = jax.nn.one_hot(np.zeros((N, T - 1), int), K, dtype=jnp.float32)
    a, bonus = spec_accept(gidx, p, jnp.asarray(d), jnp.asarray(cand), q, jax.random.PRNGKey(0), stop_ids=(stop,))
    np.testing.assert_array_equal(np.asarray(a), 1)                   # the stop draft is not fed ...
    np.testing.assert_array_equal(np.asarray(bonus), stop)            # ... but emitted
