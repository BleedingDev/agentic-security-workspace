"""TP engine on 8 virtual CPU devices vs the unsharded reference forward (tiny random model)."""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel  # noqa: E402

from glm53 import model as M  # noqa: E402
from glm53.engine import Engine, HostExperts  # noqa: E402
from glm53.hf_convert import from_hf_module  # noqa: E402
from glm53.tests.test_tiny_vs_hf import tiny_config  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


@pytest.fixture(scope="module")
def tiny():
    assert jax.device_count() == 8, jax.devices()
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=16)
    cfg.index_n_heads = 8
    # TP=8 needs head/column counts divisible by 8
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
    return cfg, params, jcfg


def ref_logits(params, jcfg, ids):
    p = jax.tree.map(jnp.asarray, params)
    h, _ = M.forward(p, jnp.asarray(ids), jcfg, sparse="auto")
    return np.asarray(M.logits(p, h[:, -1]))


@pytest.mark.parametrize("host", [False, True])
def test_engine_prefill_and_decode(tiny, host):
    cfg, params, jcfg = tiny
    rng = np.random.default_rng(0)
    T = 30
    ids = rng.integers(0, cfg.vocab_size, size=(1, T))
    he = HostExperts.from_dense(params, jcfg, 8) if host else None
    eng = Engine(jcfg, params, he, devices=jax.devices())
    # prefill T-1 then decode 1  vs  reference full forward on T
    logits, caches, pos = eng.prefill(ids[:, :T - 1])
    ref = ref_logits(params, jcfg, ids[:, :T - 1])
    rel = np.abs(np.asarray(logits) - ref).max() / np.abs(ref).max()
    print(f"[host={host}] prefill logits rel err", rel)
    assert rel < 2e-3
    logits2, caches, pos = eng.decode(ids[:, T - 1], caches, pos)
    ref2 = ref_logits(params, jcfg, ids)
    rel2 = np.abs(np.asarray(logits2) - ref2).max() / np.abs(ref2).max()
    print(f"[host={host}] decode logits rel err", rel2)
    assert rel2 < 2e-3
    # greedy generation runs end to end
    toks = eng.generate(ids[:, :5], 4)
    assert len(toks) == 4
