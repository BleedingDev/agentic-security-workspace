"""SegmentedEngine (per-layer programs, host gather between them) vs reference, float + int4 modes, CPU x8."""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel  # noqa: E402

from glm53 import model as M  # noqa: E402
from glm53.engine import HostExperts, SegmentedEngine  # noqa: E402
from glm53.hf_convert import from_hf_module  # noqa: E402
from glm53.tests.test_tiny_vs_hf import tiny_config  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def test_segmented_matches_reference():
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=16)
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
    params = from_hf_module(hf, lm_head)
    jcfg = M.Cfg.from_hf(cfg)
    rng = np.random.default_rng(0)
    T = 30
    ids = rng.integers(0, cfg.vocab_size, size=(1, T))
    pj = jax.tree.map(jnp.asarray, params)
    ref_a = np.asarray(M.logits(pj, M.forward(pj, jnp.asarray(ids[:, :T - 1]), jcfg)[0][:, -1]))
    ref_b = np.asarray(M.logits(pj, M.forward(pj, jnp.asarray(ids), jcfg)[0][:, -1]))
    eng = SegmentedEngine(jcfg, params, HostExperts.from_dense(params, jcfg, 8))
    logits, caches, pos = eng.prefill(ids[:, :T - 1])
    rel = np.abs(np.asarray(logits) - ref_a).max() / np.abs(ref_a).max()
    print("segmented prefill rel err", rel, eng.stats)
    assert rel < 2e-3
    logits2, caches, pos = eng.decode(ids[:, T - 1], caches, pos)
    rel2 = np.abs(np.asarray(logits2) - ref_b).max() / np.abs(ref_b).max()
    print("segmented decode rel err", rel2, eng.stats)
    assert rel2 < 2e-3
    toks = eng.generate(ids[:, :5], 3)
    assert len(toks) == 3
