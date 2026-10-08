"""Exact rewrites of slow TPU ops: `model.top_k` (chunked first stage) returns exactly lax.top_k's values and ids at the
engine's shapes (indexer decode / verify, drafter vocab shard), with ties and with the indexer's masked (finfo.min)
scores; `model.gather_scalars` (row gather + lane select) equals the element gather of the latent scales."""
import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from glm53 import model as M  # noqa: E402


@pytest.mark.parametrize("shape,k", [((1, 4, 8192), 128), ((1, 1, 8192), 512), ((1, 3, 19360), 16),
                                     ((2, 5, 1000), 16), ((3, 8193), 64), ((4, 300), 8)])
@pytest.mark.parametrize("mode", ["random", "ties", "masked"])
def test_top_k_equals_lax(shape, k, mode):
    rng = np.random.default_rng(len(shape) * 1000 + k)
    x = rng.standard_normal(shape).astype(np.float32)
    if mode == "ties":
        x = np.round(x, 1)
    if mode == "masked":                                   # most pools invisible: fewer than k real scores
        x[..., rng.random(shape[-1]) < 0.995] = np.finfo(np.float32).min
    a, b = lax.top_k(jnp.asarray(x), k), M.top_k(jnp.asarray(x), k)
    np.testing.assert_array_equal(np.asarray(b[0]), np.asarray(a[0]))
    np.testing.assert_array_equal(np.asarray(b[1]), np.asarray(a[1]))


@pytest.mark.parametrize("B,S,Tq,W", [(1, 32768, 4, 2080), (2, 1024, 1, 96), (1, 1000, 3, 40)])
def test_gather_scalars(B, S, Tq, W):
    rng = np.random.default_rng(S + W)
    a = rng.standard_normal((B, S)).astype(np.float32)
    idx = rng.integers(0, S, (B, Tq, W)).astype(np.int32)
    ref = np.take_along_axis(a[:, None], idx, axis=2)
    np.testing.assert_array_equal(np.asarray(M.gather_scalars(jnp.asarray(a), jnp.asarray(idx))), ref)


@pytest.mark.parametrize("Tq", [1, 4])
def test_indexer_select_buckets(Tq):
    """The bucketed indexer selection (`Cfg.select_buckets`: the local top-k over the first 512 / 2048 / all local
    pools, chosen from the block's positions) returns exactly the unbucketed selection on the sharded layout (8 CPU
    devices), at positions in every bucket, with random and tied scores."""
    from jax.sharding import Mesh, PartitionSpec as P
    from glm53.engine import AXIS, shard_map
    assert jax.device_count() == 8, jax.devices()
    n, kp, P_local = 8, 4, 4096
    cfg0 = M.Cfg(hidden=8, vocab=8, n_layers=1, layer_types=("deepseek_sparse_attention",), mlp_types=("dense",), eps=1e-5,
                 kda_heads=1, kda_hd=8, conv_k=2, gate_lb=None, n_heads=1, q_lora=8, kv_lora=8, qk_nope=8, v_hd=8,
                 idx_heads=1, idx_hd=8, idx_topk=2048, idx_kpool=kp, hc=1, hc_iters=1, hc_eps=0.0, inter=8, moe_inter=8,
                 n_exp=1, topk=1, n_shared=0, scaling=1.0, norm_topk=False, swiglu_limit=1.0, tp_axis=AXIS,
                 seq_shard=n, cache_cap=P_local * kp * n)
    mesh = Mesh(np.array(jax.devices()), (AXIS,))
    rng = np.random.default_rng(Tq)

    def run(cfg, scores, qpos):
        def f(sc, qp):
            chip = jax.lax.axis_index(AXIS)
            pools, valid = M.indexer_select(sc, jnp.ones((P_local,), bool), qp, cfg, chip)
            return pools, valid
        g = jax.jit(shard_map(f, mesh=mesh, in_specs=(P(AXIS), P()), out_specs=(P(), P()), check_vma=False))
        pools, valid = g(scores, qpos)
        return np.asarray(pools), np.asarray(valid)

    for q0 in (3, 2000, 16000, 17000, 70000, 300000):
        qpos = jnp.asarray(np.arange(q0, q0 + Tq, dtype=np.int32))
        scores = rng.standard_normal((n, Tq, P_local)).astype(np.float32)
        scores = np.round(scores, 2)                                   # ties
        scores = jnp.asarray(scores.reshape(n, Tq, P_local)).reshape(n * 1, Tq, P_local)   # [n*B, Tq, P] over chips
        pa, va = run(dataclasses.replace(cfg0, select_buckets=(512, 2048)), scores, qpos)
        pb, vb = run(cfg0, scores, qpos)
        np.testing.assert_array_equal(va, vb)
        # the selected pool SETS per query are equal (ties may order equal scores differently across the paths)
        for t in range(Tq):
            sa, sb = sorted(pa[0, t][va[0, t]].tolist()), sorted(pb[0, t][vb[0, t]].tolist())
            assert sa == sb, (q0, t, len(sa), len(sb))
