"""Grouped expert slots (`ResidentFetch._apply_kernel_grouped`: a verify of T tokens of ONE sequence decodes each
distinct expert once for all its slots): `group_plan` against a NumPy grouping; the grouped kernels equal the per-slot
kernels (`_apply_kernel`) in interpret mode, f32, for routings that share experts between tokens and routings that do
not, T = 2..4, two quant-type pairs, several size-class sets, with and without the dynamic grid and the decode-once
loop; `apply` takes the grouped path for a multi-token step of one sequence and the per-slot path for independent
rows; B rows of T tokens (the batched DFlash verify) grouped per row or over all slots equal the per-slot kernels,
and the per-row mode equals each row's own grouped call bit for bit."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from glm53 import pallas_moe as K  # noqa: E402
from glm53.resident import ResidentFetch  # noqa: E402
from glm53.tests.test_resident_cpu import N_DEV, random_tables  # noqa: E402


def routing(rng, T, E, k, shared):
    """[T, k] expert ids, distinct within a token; `shared` of them drawn from a small pool all tokens use."""
    pool = rng.choice(E, max(shared, 1) + 2, replace=False)
    rows = []
    for _ in range(T):
        a = rng.choice(pool, shared, replace=False) if shared else np.zeros(0, int)
        rest = rng.choice(np.setdiff1d(np.arange(E), a), k - shared, replace=False)
        rows.append(rng.permutation(np.concatenate([a, rest])))
    return np.stack(rows).astype(np.int32)


@pytest.mark.parametrize("T,shared", [(2, 3), (4, 2), (4, 0), (3, 4), (4, 8)])
def test_group_plan(T, shared):
    rng = np.random.default_rng(T + 7 * shared)
    k = 8
    idx = routing(rng, T, 40, k, shared)
    eids, cnt, slots = (np.asarray(a) for a in K.group_plan(jnp.asarray(idx)))
    flat = idx.reshape(-1)
    S = T * k
    order = list(dict.fromkeys(flat.tolist()))                         # distinct experts, first-slot order
    G = len(order)
    assert eids.shape == (S,) and cnt.shape == (S,) and slots.shape == (S * T,)
    assert eids[:G].tolist() == order and (eids[G:] == order[-1]).all()
    slots = slots.reshape(S, T)
    for g, e in enumerate(order):
        mine = [s for s in range(S) if flat[s] == e]
        assert cnt[g] == len(mine)
        assert slots[g, :len(mine)].tolist() == mine and (slots[g, len(mine):] == mine[-1]).all()
    assert (cnt[G:] == 0).all() and cnt.sum() == S


@pytest.mark.parametrize("gu,dn", [("IQ2_S", "IQ3_S"), ("IQ3_S", "IQ4_XS")])
@pytest.mark.parametrize("T,shared,classes,dyn,loop", [(2, 3, (), False, False), (4, 2, (), False, False),
                                                       (4, 0, (), False, False), (3, 4, (), False, False),
                                                       (4, 3, (1, 2), False, False), (4, 4, (1,), False, False),
                                                       (4, 2, (), True, False), (3, 4, (1,), True, False),
                                                       (4, 3, (), False, True), (4, 4, (4,), True, True),
                                                       (3, 2, (1, 2), False, True)])
def test_grouped_equals_per_slot(gu, dn, T, shared, classes, dyn, loop):
    rng = np.random.default_rng(10 * T + shared)
    E, D, mi, k = 12, 256, 2048, 4
    packed, _ = random_tables(rng, E, D, mi, gu, dn)
    p = {key: {pk: jnp.asarray(a[0:1]) for pk, a in packed[key].items()} for key in packed}   # chip 0's planes
    fetch = ResidentFetch({0: {"gate_q": gu, "up_q": gu, "down_q": dn}}, D, mi // N_DEV, out_dtype=jnp.float32,
                          interpret=True)
    fetch.group_min_seq, fetch.group_classes = 2, classes
    fetch.group_dynamic_grid, fetch.group_loop = dyn, loop
    idx = jnp.asarray(routing(rng, T, E, k, shared))
    x = jnp.asarray(rng.standard_normal((T, D)).astype(np.float32))
    w = jnp.asarray(rng.random((T, k)).astype(np.float32))
    a = np.asarray(fetch._apply_kernel(x, idx, w, 0, p["gate_q"], p["up_q"], p["down_q"], 10.0))
    b = np.asarray(fetch._apply_kernel_grouped(x, idx, w, 0, p["gate_q"], p["up_q"], p["down_q"], 10.0))
    np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-5 * np.abs(a).max())
    via = np.asarray(fetch.apply(p, x, idx, w, 0, 10.0, seq_tokens=T))          # dispatch: the grouped path
    np.testing.assert_array_equal(via, b)
    rows = np.asarray(fetch.apply(p, x, idx, w, 0, 10.0, seq_tokens=1))         # independent rows: per slot
    np.testing.assert_array_equal(rows, a)


@pytest.mark.parametrize("mode", ["row", "all"])
@pytest.mark.parametrize("B,T,shared", [(2, 4, 2), (3, 4, 0), (3, 2, 3)])
def test_rows_grouped(mode, B, T, shared):
    rng = np.random.default_rng(100 * B + 10 * T + shared)
    E, D, mi, k = 12, 256, 2048, 4
    gu, dn = "IQ2_S", "IQ3_S"
    packed, _ = random_tables(rng, E, D, mi, gu, dn)
    p = {key: {pk: jnp.asarray(a[0:1]) for pk, a in packed[key].items()} for key in packed}
    fetch = ResidentFetch({0: {"gate_q": gu, "up_q": gu, "down_q": dn}}, D, mi // N_DEV, out_dtype=jnp.float32,
                          interpret=True)
    fetch.group_rows = mode
    idx = jnp.asarray(np.concatenate([routing(rng, T, E, k, shared) for _ in range(B)]))   # rows share experts too
    x = jnp.asarray(rng.standard_normal((B * T, D)).astype(np.float32))
    w = jnp.asarray(rng.random((B * T, k)).astype(np.float32))
    a = np.asarray(fetch._apply_kernel(x, idx, w, 0, p["gate_q"], p["up_q"], p["down_q"], 10.0))
    via = np.asarray(fetch.apply(p, x, idx, w, 0, 10.0, seq_tokens=T))           # dispatch: B rows of T tokens
    np.testing.assert_allclose(via, a, rtol=1e-5, atol=1e-5 * np.abs(a).max())
    if mode == "row":
        for r in range(B):
            sl = slice(r * T, (r + 1) * T)
            one = np.asarray(fetch.apply(p, x[sl], idx[sl], w[sl], 0, 10.0, seq_tokens=T))
            np.testing.assert_array_equal(via[sl], one)
