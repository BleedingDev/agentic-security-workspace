"""Tiny random Glm5Next: JAX (glm53.model) vs transformers reference, CPU fp32."""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel  # noqa: E402

from glm53 import model as M  # noqa: E402
from glm53.hf_convert import from_hf_module  # noqa: E402

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")


def tiny_config(index_topk=64):
    return Glm5NextTextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
        num_hidden_layers=6,
        layer_types=["linear_attention", "linear_attention", "deepseek_sparse_attention",
                     "linear_attention", "deepseek_sparse_attention", "linear_attention"],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse", "sparse", "sparse"],
        indexer_types=["full"] * 6,
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1, n_group=1, topk_group=1,
        norm_topk_prob=True, routed_scaling_factor=2.5, scoring_func="sigmoid", topk_method="noaux_tc",
        num_attention_heads=4, num_key_value_heads=4, q_lora_rank=48, kv_lora_rank=32,
        qk_nope_head_dim=16, qk_rope_head_dim=0, v_head_dim=16, head_dim=0, mla_use_nope=True,
        index_n_heads=2, index_head_dim=16, index_topk=index_topk, index_kpool=4,
        index_kpool_always_select_tail=True, index_kpool_compress=True,
        linear_attn_config={"num_heads": 4, "head_dim": 16, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, mhc=True,
        rms_norm_eps=1e-5, swiglu_limit=10.0, max_position_embeddings=4096, first_k_dense_replace=1,
        dtype=torch.float32, tie_word_embeddings=False, moe_router_dtype="float32", pad_token_id=0,
    )


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = tiny_config()
    hf = Glm5NextTextModel(cfg).float().eval()
    # make the KDA forget gate non-trivial (init sets A_log=0 → decay_rate 1, fine) and the router biased
    with torch.no_grad():
        for L in hf.layers:
            if hasattr(L.mlp, "gate"):
                L.mlp.gate.e_score_correction_bias.normal_(0, 0.1)
    params = jax.tree.map(jnp.asarray, from_hf_module(hf))
    jcfg = M.Cfg.from_hf(cfg)
    return cfg, hf, params, jcfg


def hf_hidden(hf, ids):
    with torch.no_grad():
        return hf(input_ids=torch.tensor(ids), use_cache=False).last_hidden_state.numpy()


def test_prefill_matches_hf(setup):
    cfg, hf, params, jcfg = setup
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab_size, size=(2, 37))
    ref = hf_hidden(hf, ids)
    out, _ = M.forward(params, jnp.asarray(ids), jcfg)
    out = np.asarray(out)
    err = np.abs(out - ref).max()
    rel = err / (np.abs(ref).max() + 1e-6)
    print("prefill max abs err", err, "rel", rel)
    assert rel < 2e-3, (err, rel)


def test_chunk_boundary(setup):
    """Sequence longer than one KDA chunk (64) exercises the inter-chunk state path."""
    cfg, hf, params, jcfg = setup
    rng = np.random.default_rng(1)
    ids = rng.integers(0, cfg.vocab_size, size=(1, 130))
    # index_topk=64 in HF would make attention sparse beyond 64 tokens; rebuild HF with a big topk for this test
    ref = hf_hidden(hf, ids[:, :64])
    out, _ = M.forward(params, jnp.asarray(ids[:, :64]), jcfg)
    assert np.abs(np.asarray(out) - ref).max() / np.abs(ref).max() < 2e-3
    # dense check across chunk boundary needs topk >= 130
    big = tiny_config(index_topk=256)
    torch.manual_seed(0)
    hf2 = Glm5NextTextModel(big).float().eval()
    hf2.load_state_dict(hf.state_dict())
    ref2 = hf_hidden(hf2, ids)
    out2, _ = M.forward(params, jnp.asarray(ids), M.Cfg.from_hf(big))
    rel = np.abs(np.asarray(out2) - ref2).max() / np.abs(ref2).max()
    print("130-token rel err", rel)
    assert rel < 2e-3


def test_decode_matches_prefill(setup):
    """prefill(T-1) + recurrent decode(1) == prefill(T)."""
    cfg, hf, params, jcfg = setup
    rng = np.random.default_rng(2)
    T = 20
    ids = rng.integers(0, cfg.vocab_size, size=(1, T))
    full, _ = M.forward(params, jnp.asarray(ids), jcfg)
    h, caches = M.forward(params, jnp.asarray(ids[:, :T - 1]), jcfg, cap=T + 4)
    step, _ = M.forward(params, jnp.asarray(ids[:, T - 1:]), jcfg, caches=caches, pos0=T - 1, use_recurrent=True)
    rel = np.abs(np.asarray(step[:, -1]) - np.asarray(full[:, -1])).max() / np.abs(np.asarray(full[:, -1])).max()
    print("decode rel err", rel)
    assert rel < 2e-3
    # and a 3-token continuation through the chunked path with caches (mid-sequence prefill)
    h, caches = M.forward(params, jnp.asarray(ids[:, :T - 3]), jcfg, cap=T)
    step3, _ = M.forward(params, jnp.asarray(ids[:, T - 3:]), jcfg, caches=caches, pos0=T - 3, use_recurrent=False)
    rel3 = np.abs(np.asarray(step3) - np.asarray(full[:, T - 3:])).max() / np.abs(np.asarray(full[:, T - 3:])).max()
    print("3-token continuation rel err", rel3)
    assert rel3 < 2e-3


def test_sparse_indexer_matches_hf(setup):
    """index_topk=16 (4 pools) with 45 tokens -> DSA selects a strict subset; compare vs HF, prefill and decode."""
    cfg, hf, params, jcfg = setup
    # 8 indexer heads: with 2 heads many pools tie at exactly 0 after ReLU and torch/JAX top-k break ties
    # differently (verified 2026-09-05); the model math is identical.
    small = tiny_config(index_topk=16)
    small.index_n_heads = 8
    torch.manual_seed(0)
    hf2 = Glm5NextTextModel(small).float().eval()
    with torch.no_grad():  # give the indexer non-trivial pooling weights
        for L in hf2.layers:
            if hasattr(L.self_attn, "indexer") and L.self_attn.indexer is not None:
                L.self_attn.indexer.index_kpool_compress_ape.normal_(0, 0.5)
                L.self_attn.indexer.index_kpool_compress_gate.normal_(0, 0.05)
    params2 = jax.tree.map(jnp.asarray, from_hf_module(hf2))
    jcfg2 = M.Cfg.from_hf(small)
    rng = np.random.default_rng(3)
    T = 45
    ids = rng.integers(0, cfg.vocab_size, size=(1, T))
    ref = hf_hidden(hf2, ids)
    out, _ = M.forward(params2, jnp.asarray(ids), jcfg2, sparse="always")
    rel = np.abs(np.asarray(out) - ref).max() / np.abs(ref).max()
    print("sparse prefill rel err", rel)
    assert rel < 2e-3
    # small query blocks (ragged last block) and a union budget of 1 pool (every block overflows -> per-query gathers)
    import dataclasses
    for kw in ({"q_block": 7}, {"union_pools": 1}, {"q_block": 16, "union_pools": 1}):
        out2, _ = M.forward(params2, jnp.asarray(ids), dataclasses.replace(jcfg2, **kw), sparse="always")
        rel = np.abs(np.asarray(out2) - ref).max() / np.abs(ref).max()
        assert rel < 2e-3, (kw, rel)
    # sanity: dense attention must NOT match (otherwise the test isn't exercising sparsity)
    dense, _ = M.forward(params2, jnp.asarray(ids), jcfg2, sparse=None)
    assert np.abs(np.asarray(dense) - ref).max() / np.abs(ref).max() > 1e-3
    # decode consistency in the sparse regime
    h, caches = M.forward(params2, jnp.asarray(ids[:, :T - 1]), jcfg2, sparse="always", cap=T + 8)
    step, _ = M.forward(params2, jnp.asarray(ids[:, T - 1:]), jcfg2, caches=caches, pos0=T - 1,
                        use_recurrent=True, sparse="always")
    rel2 = np.abs(np.asarray(step[:, -1]) - ref[:, -1]).max() / np.abs(ref[:, -1]).max()
    print("sparse decode rel err", rel2)
    assert rel2 < 2e-3
