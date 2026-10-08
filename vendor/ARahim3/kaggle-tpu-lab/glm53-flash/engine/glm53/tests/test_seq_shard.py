"""Sequence-sharded MLA/indexer caches (Engine(seq_shard=True)) vs the replicated caches, chunked prefill pieces and
prefix continuation (ResidentLayerEngine) vs one-shot prefill. 8 virtual CPU devices, fp32, tiny random model."""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel  # noqa: E402

from glm53 import model as M  # noqa: E402
from glm53.engine import Engine  # noqa: E402
from glm53.hf_convert import from_hf_module  # noqa: E402
from glm53.tests.test_tiny_vs_hf import tiny_config  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def build(index_topk=16):
    assert jax.device_count() == 8, jax.devices()
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=index_topk)
    cfg.index_n_heads = 8
    cfg.num_attention_heads = cfg.num_key_value_heads = 8
    cfg.linear_num_heads = 8
    cfg.intermediate_size = 128
    hf = Glm5NextTextModel(cfg).float().eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    with torch.no_grad():
        for L in hf.layers:
            if hasattr(L.mlp, "gate"):
                L.mlp.gate.e_score_correction_bias.normal_(0, 0.1)
            if hasattr(L.self_attn, "indexer") and L.self_attn.indexer is not None:
                L.self_attn.indexer.index_kpool_compress_ape.normal_(0, 0.5)
                L.self_attn.indexer.index_kpool_compress_gate.normal_(0, 0.05)
    return cfg, from_hf_module(hf, lm_head), M.Cfg.from_hf(cfg)


def run(eng, ids, n_dec=4, caches=None, pos0=0):
    logits, caches, pos = eng.prefill(ids, caches, pos0) if caches is not None else eng.prefill(ids)
    outs = [np.asarray(logits)]
    tok = np.array([int(jnp.argmax(logits[0]))])
    for _ in range(n_dec):
        logits, caches, pos = eng.decode(tok, caches, pos)
        outs.append(np.asarray(logits))
        tok = np.array([int(jnp.argmax(logits[0]))])
    return outs, caches, pos


@pytest.mark.parametrize("q_block,index_topk,union,topk_local", [(128, 16, 512, 128), (16, 16, 512, 128), (128, 64, 512, 128),
                                                                 (16, 16, 1, 128), (16, 16, 512, 1), (16, 16, 0, 128)])
def test_sharded_cache_matches_replicated(q_block, index_topk, union, topk_local):
    """index_topk=16 -> sparse DSA path (two-stage top-k across chips; 4 pools of 4 tokens per chip at cap 128);
    index_topk=64 with cap 32 -> dense sharded path (partial-softmax combine over the chips' key slices);
    union=1 -> every block overflows the dedup budget and takes the per-query fallback (global decision)."""
    import dataclasses
    cfg, params, jcfg = build(index_topk)
    # topk_local=1 with 4 pools per chip: the first-stage list is cut short -> the exact fallback path must run
    jcfg = dataclasses.replace(jcfg, union_pools=union, topk_local=topk_local)
    cap = 128 if index_topk == 16 else 32
    ids = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(1, 45 if cap == 128 else 20))
    ref = Engine(jcfg, params, max_len=cap, q_block=q_block)
    shd = Engine(jcfg, params, max_len=cap, q_block=q_block, seq_shard=True)
    assert shd.lcfg.seq_shard == 8 and ref.lcfg.seq_shard == 1
    a, ca, _ = run(ref, ids)
    b, cb, _ = run(shd, ids)
    for i, (x, y) in enumerate(zip(a, b)):
        np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-4, err_msg=f"step {i}")
    # the sharded MLA cache really is split: each chip holds cap/8 slots
    mla = [i for i in range(jcfg.n_layers) if jcfg.layer_types[i] != "linear_attention"][0]
    assert cb[mla]["c"].shape[1] == cap and cb[mla]["c"].sharding.shard_shape(cb[mla]["c"].shape)[1] == cap // 8
    assert ca[mla]["c"].sharding.shard_shape(ca[mla]["c"].shape)[1] == cap


@pytest.mark.parametrize("seq_shard", [False, True])
def test_chunked_prefill_and_continue(seq_shard):
    """ResidentLayerEngine pieces of 32 tokens (3 pieces for T=70, the last padded) == one-shot Engine prefill;
    prefill of 40 tokens then a 30-token continuation == the same."""
    from glm53.resident import ResidentLayerEngine
    from glm53.tests.test_resident_cpu import random_tables
    from glm53.resident import ResidentFetch
    cfg, params, jcfg = build(16)
    # resident experts need 256-wide dims; rebuild the tiny model at the resident sizes
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=16)
    cfg.hidden_size = 256; cfg.moe_intermediate_size = 2048; cfg.intermediate_size = 256
    cfg.index_n_heads = 8; cfg.num_attention_heads = cfg.num_key_value_heads = 8; cfg.linear_num_heads = 8
    cfg.q_lora_rank, cfg.kv_lora_rank = 64, 32; cfg.linear_attn_config["num_heads"] = 8
    hf = Glm5NextTextModel(cfg).float().eval()
    with torch.no_grad():
        for L in hf.layers:
            if hasattr(L.self_attn, "indexer") and L.self_attn.indexer is not None:
                L.self_attn.indexer.index_kpool_compress_ape.normal_(0, 0.5)
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    params = from_hf_module(hf, lm_head)
    jcfg = M.Cfg.from_hf(cfg)
    rng = np.random.default_rng(1)
    E, D, mi = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    packed_params = {**params, "layers": []}; qtypes = {}
    for i, L in enumerate(params["layers"]):
        if jcfg.mlp_types[i] == "sparse":
            packed, _ = random_tables(rng, E, D, mi)
            packed_params["layers"].append({**L, "mlp": {**{k: v for k, v in L["mlp"].items() if k not in ("gate_up", "down")}, **packed}})
            qtypes[i] = {"gate_q": "IQ2_S", "up_q": "IQ2_S", "down_q": "IQ3_S"}
        else:
            packed_params["layers"].append(L)
    mk = lambda: ResidentFetch(qtypes, D, mi // 8, out_dtype=jnp.float32, gather_max_rows=8, chunk=4, interpret=True)
    ids = rng.integers(0, cfg.vocab_size, size=(1, 70))
    one = Engine(jcfg, packed_params, expert_fetch=mk(), max_len=128, seq_shard=seq_shard)
    eng = ResidentLayerEngine(jcfg, packed_params, mk(), max_len=128, layers_per_program=3, prefill_piece=32,
                              seq_shard=seq_shard)
    a, _, _ = run(one, ids, n_dec=3)
    b, _, _ = run(eng, ids, n_dec=3)
    for i, (x, y) in enumerate(zip(a, b)):
        np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-4, err_msg=f"pieces step {i}")
    # prefix continuation: 40 tokens, snapshot, continue with the remaining 30 (unaligned pos0 = 40 -> scatter path)
    _, caches, pos = eng.prefill(ids[:, :40])
    snap = eng.copy_caches(caches)
    c, _, _ = run(eng, ids[:, 40:], n_dec=3, caches=caches, pos0=pos)
    for i, (x, y) in enumerate(zip(a, c)):
        np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-4, err_msg=f"continue step {i}")
    # the snapshot is untouched by the continuation and can be continued again
    d, _, _ = run(eng, ids[:, 40:], n_dec=3, caches=snap, pos0=pos)
    for i, (x, y) in enumerate(zip(a, d)):
        np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-4, err_msg=f"snapshot step {i}")
    # compact prefix snapshot (used cache rows only) restored into fresh full-capacity caches, continued twice
    _, caches, pos = eng.prefill(ids[:, :40])
    ps = eng.snapshot_prefix(caches, pos)
    mla = [i for i in range(jcfg.n_layers) if jcfg.layer_types[i] != "linear_attention"][0]
    assert ps["caches"][mla]["c"].shape[1] < caches[mla]["c"].shape[1]
    for _ in range(2):
        e, _, _ = run(eng, ids[:, 40:], n_dec=3, caches=eng.restore_prefix(ps), pos0=pos)
        for i, (x, y) in enumerate(zip(a, e)):
            np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-4, err_msg=f"prefix snapshot step {i}")
    # bucketed row count + host round trip (the server parks evicted contexts in host memory)
    pb = eng.snapshot_prefix(caches, pos, rows_bucket=16)
    assert pb["rows"] % 16 == 0 and pb["rows"] >= ps["rows"]
    hs = eng.snapshot_to_host(pb)
    del pb
    from glm53.resident import _HostShards
    leaves = jax.tree.leaves(hs["caches"], is_leaf=lambda x: isinstance(x, _HostShards))
    assert hs["bytes"] > 0 and all(isinstance(x, _HostShards) and all(isinstance(y, np.ndarray) for y in x.shards) for x in leaves)
    for _ in range(2):
        e, _, _ = run(eng, ids[:, 40:], n_dec=3, caches=eng.restore_prefix(eng.snapshot_from_host(hs)), pos0=pos)
        for i, (x, y) in enumerate(zip(a, e)):
            np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-4, err_msg=f"host snapshot step {i}")
