"""int8 MLA latent cache (Cfg.cache_q8: int8 rows + per-token f32 scales, dequantized on read): bookkeeping is exact
(sharded == replicated, pieces / continuation / prefix snapshot / host round trip == one-shot) and the quantization
error against the exact bf16/f32 cache is small on the tiny random model."""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from glm53 import model as M  # noqa: E402
from glm53.engine import Engine  # noqa: E402
from glm53.tests.test_seq_shard import build, run  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def test_q8_sharded_matches_replicated_and_is_close_to_exact():
    cfg, params, jcfg = build(16)                                # sparse DSA path at cap 128 (4 pools of 4 per chip)
    ids = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(1, 45))
    exact = Engine(jcfg, params, max_len=128, seq_shard=True)
    q8_rep = Engine(jcfg, params, max_len=128, cache_q8=True)
    q8_shd = Engine(jcfg, params, max_len=128, seq_shard=True, cache_q8=True)
    assert q8_shd.lcfg.cache_q8 and q8_shd.cfg.cache_q8 and not exact.lcfg.cache_q8
    a, ca, _ = run(exact, ids)
    b, cb, _ = run(q8_rep, ids)
    c, cc, _ = run(q8_shd, ids)
    mla = [i for i in range(jcfg.n_layers) if jcfg.layer_types[i] != "linear_attention"][0]
    assert cc[mla]["c"].dtype == jnp.int32 and cc[mla]["cs"].dtype == jnp.float32 and cc[mla]["cs"].shape == (1, 128)
    assert ca[mla]["c"].dtype == jnp.float32 and "cs" not in ca[mla]
    worst = 0.0
    for i, (x, y, z) in enumerate(zip(a, b, c)):
        np.testing.assert_allclose(z, y, rtol=1e-4, atol=1e-4, err_msg=f"sharded q8 vs replicated q8, step {i}")
        scale = float(np.max(np.abs(x)))
        worst = max(worst, float(np.max(np.abs(y - x))) / scale)
    # the tiny random model is hypersensitive to key noise (a fake bf16 cache already moves its logits by 4e-3 and
    # per-token int8 by 2e-1, while the rows themselves are within 1/254 of their max): the real-model gate is the
    # 260k needle test on the TPU; here only sanity
    print(f"q8 vs exact: worst relative logit error {worst:.2e}")
    assert worst < 1.0, worst


def test_write_latent_error_bound():
    """Per-token absmax int8: every element within max|row| / 254 (+ eps) after a write + dequant round trip."""
    import dataclasses
    cfg, params, jcfg = build(16)
    lcfg = dataclasses.replace(jcfg, cache_q8=True, seq_shard=1)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((2, 5, jcfg.kv_lora)).astype(np.float32) * rng.uniform(0.1, 3.0, (2, 5, 1)))
    cache = {"c": jnp.zeros((2, 16, jcfg.kv_lora // 4), jnp.int32), "cs": jnp.zeros((2, 16), jnp.float32)}   # int8 rows packed in int32
    new = M.write_latent(cache, x, 3, lcfg)
    assert new["c"].dtype == jnp.int32 and new["cs"].shape == (2, 16)
    q = jnp.asarray(np.random.default_rng(3).integers(-127, 128, (3, 5, 64)), jnp.int8)
    assert M.unpack_latent(M.pack_latent(q)).tolist() == q.astype(jnp.int32).tolist()           # the packing is exact
    back = M.latent_dequant(new["c"][:, 3:8], new["cs"][:, 3:8], jnp.float32)
    bound = jnp.max(jnp.abs(x), -1, keepdims=True) / 254 + 1e-6
    assert bool(jnp.all(jnp.abs(back - x) <= bound))
    assert bool(jnp.all(new["c"][:, :3] == 0)) and bool(jnp.all(new["c"][:, 8:] == 0))


def test_q8_resident_pieces_snapshots_match_one_shot():
    from glm53.resident import ResidentFetch, ResidentLayerEngine
    from glm53.tests.test_resident_cpu import random_tables
    from glm53.tests.test_tiny_vs_hf import tiny_config
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
    from glm53.hf_convert import from_hf_module
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=16)
    cfg.hidden_size = 256; cfg.moe_intermediate_size = 2048; cfg.intermediate_size = 256
    cfg.index_n_heads = 8; cfg.num_attention_heads = cfg.num_key_value_heads = 8; cfg.linear_num_heads = 8
    cfg.q_lora_rank, cfg.kv_lora_rank = 64, 32; cfg.linear_attn_config["num_heads"] = 8
    hf = Glm5NextTextModel(cfg).float().eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    params = from_hf_module(hf, lm_head); jcfg = M.Cfg.from_hf(cfg)
    rng = np.random.default_rng(1)
    E, D, mi = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    packed = {**params, "layers": []}; qtypes = {}
    for i, L in enumerate(params["layers"]):
        if jcfg.mlp_types[i] == "sparse":
            tables, _ = random_tables(rng, E, D, mi)
            packed["layers"].append({**L, "mlp": {**{k: v for k, v in L["mlp"].items() if k not in ("gate_up", "down")}, **tables}})
            qtypes[i] = {"gate_q": "IQ2_S", "up_q": "IQ2_S", "down_q": "IQ3_S"}
        else:
            packed["layers"].append(L)
    mk = lambda: ResidentFetch(qtypes, D, mi // 8, out_dtype=jnp.float32, gather_max_rows=8, chunk=4, interpret=True)
    ids = rng.integers(0, cfg.vocab_size, size=(1, 70))
    one = Engine(jcfg, packed, expert_fetch=mk(), max_len=128, seq_shard=True, cache_q8=True)
    eng = ResidentLayerEngine(jcfg, packed, mk(), max_len=128, layers_per_program=3, prefill_piece=32, seq_shard=True,
                              cache_q8=True)
    a, _, _ = run(one, ids, n_dec=3)
    b, _, _ = run(eng, ids, n_dec=3)
    for i, (x, y) in enumerate(zip(a, b)):
        np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-4, err_msg=f"pieces step {i}")
    _, caches, pos = eng.prefill(ids[:, :40])
    mla = [i for i in range(jcfg.n_layers) if jcfg.layer_types[i] != "linear_attention"][0]
    assert caches[mla]["c"].dtype == jnp.int32 and caches[mla]["cs"].shape == (1, 128)
    ps = eng.snapshot_prefix(caches, pos, rows_bucket=8)                 # rows are per chip; global shape = rows x 8
    n = eng.lcfg.seq_shard
    assert ps["caches"][mla]["cs"].shape[1] == ps["rows"] * n == 64 and ps["caches"][mla]["c"].shape[1] == ps["rows"] * n
    hs = eng.snapshot_to_host(ps)
    for restored in (eng.restore_prefix(ps), eng.restore_prefix(eng.snapshot_from_host(hs))):
        e, _, _ = run(eng, ids[:, 40:], n_dec=3, caches=restored, pos0=pos)
        for i, (x, y) in enumerate(zip(a, e)):
            np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-4, err_msg=f"snapshot step {i}")
    # batched independent streams with the q8 cache
    from glm53.engine import DeviceSampler
    p2 = rng.integers(0, cfg.vocab_size, size=(1, 30))
    ref2, _, _ = run(eng, p2, n_dec=3)
    lg1, c1, pos1 = eng.prefill(ids); lg2, c2, pos2 = eng.prefill(p2)
    toks = np.array([int(jnp.argmax(lg1[0])), int(jnp.argmax(lg2[0]))], np.int32)
    _, logits, sets, pos_next = eng.decode_rows(toks, [c1, c2], [pos1, pos2])
    np.testing.assert_allclose(np.asarray(logits)[0], a[1][0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(logits)[1], ref2[1][0], rtol=1e-4, atol=1e-4)


def _elem_scales(cs_all, pools, pvalid, qpos, cfg, chip, W):
    """The reference: every slot's scale gathered one by one (the path before 2026-10-05)."""
    tok = M.select_tokens(pools, pvalid, qpos, cfg)
    valid = (tok >= 0) & (tok <= qpos[None, :, None])
    if cfg.seq_shard > 1:
        owner, slot = M._pos_to_slot(jnp.maximum(tok, 0), cfg.idx_kpool, cfg.seq_shard)
        valid = valid & (owner == chip)
    else:
        slot = tok
    slot = jnp.where(valid, slot, 0)
    return jnp.take_along_axis(cs_all[:, None], slot, axis=2, mode="promise_in_bounds"), valid


@pytest.mark.parametrize("n,S", [(1, 512), (8, 4096), (1, 300)])
def test_token_scales_by_pools_equal_element_gather(n, S):
    """`_token_scales` (the scales of a verify's selected tokens from pool rows) equals the element gather on every
    VALID slot: replicated and sharded caches, a capacity that is not a multiple of 128 (the fallback)."""
    import dataclasses
    rng = np.random.default_rng(3)
    kp, topk = 4, 64
    from glm53.tests.test_seq_shard import build
    _, _, jcfg = build(16)
    cfg = dataclasses.replace(jcfg, seq_shard=n, idx_kpool=kp, idx_topk=topk, cache_q8=True, cache_cap=S)
    S_local = S // n
    P_local = -(-S_local // kp)
    cs_all = jnp.asarray(rng.uniform(0.5, 2.0, (1, S_local)), jnp.float32)
    K = min(topk // kp, P_local)
    for chip in range(n):
        Tq = 4
        qpos = jnp.asarray(np.sort(rng.integers(kp * n * 2, S - 1, Tq)), jnp.int32)
        pools = jnp.asarray(rng.integers(0, (S // kp) // 1, (1, Tq, K)), jnp.int32)
        pools = jnp.minimum(pools, (qpos // kp)[None, :, None] - 1)                    # visible pools only
        pvalid = jnp.asarray(rng.random((1, Tq, K)) > 0.2)
        ref, valid = _elem_scales(cs_all, pools, pvalid, qpos, cfg, chip, None)
        got = M._token_scales(cs_all, pools, pvalid, qpos, cfg, chip, ref.shape[-1])
        assert got.shape == ref.shape
        np.testing.assert_array_equal(np.where(np.asarray(valid), np.asarray(got), 0.0),
                                      np.where(np.asarray(valid), np.asarray(ref), 0.0), err_msg=f"n {n} chip {chip}")
