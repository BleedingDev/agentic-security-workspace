"""The base store (`glm53.basestore`): a tiny resident engine built with int8 non-expert weights is dumped (expert
tables left out) and rebuilt from the store next to the same tables — identical logits; plus the tree round trip
(bf16 as uint16, lists, non-array leaves)."""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel  # noqa: E402

from glm53 import basestore as BS  # noqa: E402
from glm53 import model as M  # noqa: E402
from glm53.hf_convert import from_hf_module  # noqa: E402
from glm53.resident import ResidentFetch, ResidentLayerEngine  # noqa: E402
from glm53.tests.test_resident_cpu import random_tables  # noqa: E402
from glm53.tests.test_tiny_vs_hf import tiny_config  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def test_tree_round_trip(tmp_path):
    import ml_dtypes
    rng = np.random.default_rng(0)
    tree = {"a": rng.standard_normal((4, 6)).astype(ml_dtypes.bfloat16), "b": {"q": rng.integers(-127, 127, (8, 8)).astype(np.int8),
            "s": rng.random((1, 8)).astype(np.float32)}, "blocks": [{"w": np.arange(3, dtype=np.int32)}, {"w": np.arange(5, dtype=np.int32)}],
            "n_heads": 16, "name": "x"}
    BS.save_tree(tmp_path / "t.npz", tree)
    back = BS.load_tree(tmp_path / "t.npz")
    assert back["a"].dtype == ml_dtypes.bfloat16 and np.array_equal(back["a"].view(np.uint16), tree["a"].view(np.uint16))
    assert np.array_equal(back["b"]["q"], tree["b"]["q"]) and np.array_equal(back["b"]["s"], tree["b"]["s"])
    assert isinstance(back["blocks"], list) and len(back["blocks"]) == 2 and np.array_equal(back["blocks"][1]["w"], np.arange(5))
    assert back["n_heads"] == 16 and back["name"] == "x"


def test_engine_rebuilt_from_store(tmp_path):
    rng = np.random.default_rng(3)
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=16)
    cfg.hidden_size = 256; cfg.moe_intermediate_size = 2048; cfg.intermediate_size = 256
    cfg.index_n_heads = 8; cfg.num_attention_heads = cfg.num_key_value_heads = 8; cfg.linear_num_heads = 8
    cfg.q_lora_rank, cfg.kv_lora_rank = 64, 32; cfg.linear_attn_config["num_heads"] = 8
    hf = Glm5NextTextModel(cfg).float().eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    params = from_hf_module(hf, lm_head)
    jcfg = M.Cfg.from_hf(cfg)
    E, D, mi = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    packed = {**params, "layers": []}; qtypes = {}; tables_of = {}
    for i, L in enumerate(params["layers"]):
        if jcfg.mlp_types[i] == "sparse":
            tables, _ = random_tables(rng, E, D, mi)
            tables_of[i] = tables
            packed["layers"].append({**L, "mlp": {**{k: v for k, v in L["mlp"].items() if k not in ("gate_up", "down")}, **tables}})
            qtypes[i] = {"gate_q": "IQ2_S", "up_q": "IQ2_S", "down_q": "IQ3_S"}
        else:
            packed["layers"].append(L)
    fetch = ResidentFetch(qtypes, D, mi // 8, out_dtype=jnp.float32, gather_max_rows=64, chunk=4, interpret=True)
    kw = dict(max_len=128, layers_per_program=3, prefill_piece=32, seq_shard=True)
    eng1 = ResidentLayerEngine(jcfg, packed, fetch, int8_nonexpert="all", **kw)
    # dump the engine's own parameters (device arrays, int8 nodes included; expert tables skipped)
    BS.dump(str(tmp_path), eng1.params, vision=None, hf_dir=None, meta={"test": True}, log=lambda *a: None)
    store = BS.load(str(tmp_path))
    assert store["manifest"]["n_layers"] == jcfg.n_layers and store["manifest"]["test"] is True
    for i, L in enumerate(store["layers"]):
        assert not any(k in L["mlp"] for k in BS.EXPERT_KEYS)
    params2 = {**store["top"], "layers": [{**L, "mlp": {**L["mlp"], **tables_of.get(i, {})}} for i, L in enumerate(store["layers"])]}
    eng2 = ResidentLayerEngine(jcfg, params2, fetch, int8_nonexpert=False, **kw)
    prompt = rng.integers(0, cfg.vocab_size, size=(1, 40)).astype(np.int32)
    lg1, c1, p1 = eng1.prefill(prompt)
    lg2, c2, p2 = eng2.prefill(prompt)
    assert np.array_equal(np.asarray(lg1), np.asarray(lg2)), "prefill logits differ after the store round trip"
    nxt = np.array([int(np.argmax(np.asarray(lg1)[0]))])
    d1, _, _ = eng1.decode(nxt, c1, p1)
    d2, _, _ = eng2.decode(nxt, c2, p2)
    assert np.array_equal(np.asarray(d1), np.asarray(d2)), "decode logits differ after the store round trip"
