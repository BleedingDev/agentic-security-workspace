"""Batched decode of INDEPENDENT streams (`ResidentLayerEngine.decode_rows`): streams prefilled separately, sitting at
different positions, decoded together for several steps, with a stream admitted and one removed mid-way — every
row must equal its own single-stream decode (sharded and replicated caches, tiny random resident model, 8 virtual
CPU devices, fp32)."""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel  # noqa: E402

from glm53 import model as M  # noqa: E402
from glm53.engine import DeviceSampler  # noqa: E402
from glm53.hf_convert import from_hf_module  # noqa: E402
from glm53.resident import ResidentFetch, ResidentLayerEngine  # noqa: E402
from glm53.tests.test_resident_cpu import random_tables  # noqa: E402
from glm53.tests.test_tiny_vs_hf import tiny_config  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def build_engine(seq_shard, rng, max_len=128):
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
    E, D, mi = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    packed = {**params, "layers": []}; qtypes = {}
    for i, L in enumerate(params["layers"]):
        if jcfg.mlp_types[i] == "sparse":
            tables, _ = random_tables(rng, E, D, mi)
            packed["layers"].append({**L, "mlp": {**{k: v for k, v in L["mlp"].items() if k not in ("gate_up", "down")}, **tables}})
            qtypes[i] = {"gate_q": "IQ2_S", "up_q": "IQ2_S", "down_q": "IQ3_S"}
        else:
            packed["layers"].append(L)
    fetch = ResidentFetch(qtypes, D, mi // 8, out_dtype=jnp.float32, gather_max_rows=64, chunk=4, interpret=True)
    eng = ResidentLayerEngine(jcfg, packed, fetch, max_len=max_len, layers_per_program=3, prefill_piece=32,
                              seq_shard=seq_shard)
    return eng, cfg.vocab_size


def single_stream(eng, ids, n_dec):
    """Reference: prefill + n_dec greedy decode steps of one stream -> (logits per step, tokens fed per step)."""
    logits, caches, pos = eng.prefill(ids)
    outs, toks = [np.asarray(logits)[0]], [int(jnp.argmax(logits[0]))]
    for _ in range(n_dec):
        logits, caches, pos = eng.decode(np.array([toks[-1]]), caches, pos)
        outs.append(np.asarray(logits)[0]); toks.append(int(jnp.argmax(logits[0])))
    return outs, toks


@pytest.mark.parametrize("seq_shard", [True, False])
def test_decode_rows_matches_single_streams(seq_shard):
    rng = np.random.default_rng(3)
    eng, V = build_engine(seq_shard, rng)
    lens = {"a": 45, "b": 30, "c": 61}                          # different lengths -> different positions per row
    prompts = {k: rng.integers(0, V, size=(1, n)) for k, n in lens.items()}
    N = 6
    ref = {k: single_stream(eng, p, N) for k, p in prompts.items()}

    # streams a and b decode together from step 0; c is admitted after 2 steps; a is removed after 4 steps
    live = {}
    for k in ("a", "b"):
        lg, cc, pos = eng.prefill(prompts[k])
        live[k] = {"caches": cc, "pos": pos, "tok": int(jnp.argmax(lg[0])), "step": 0}
        np.testing.assert_allclose(np.asarray(lg)[0], ref[k][0][0], rtol=1e-4, atol=1e-4)
    for step in range(N):
        if step == 2:
            lg, cc, pos = eng.prefill(prompts["c"])
            live["c"] = {"caches": cc, "pos": pos, "tok": int(jnp.argmax(lg[0])), "step": 0}
        if step == 4:
            del live["a"]
        keys = list(live)
        toks = np.array([live[k]["tok"] for k in keys], np.int32)
        pos = eng.device_positions([live[k]["pos"] for k in keys])
        ids, logits, sets, pos_next = eng.decode_rows(toks, [live[k]["caches"] for k in keys], pos)
        assert ids is None
        logits, pos_next = np.asarray(logits), np.asarray(pos_next)
        for r, k in enumerate(keys):
            s = live[k]["step"] + 1
            assert toks[r] == ref[k][1][s - 1]
            np.testing.assert_allclose(logits[r], ref[k][0][s], rtol=1e-4, atol=1e-4, err_msg=f"stream {k} step {s}")
            assert pos_next[r] == live[k]["pos"] + 1
            live[k].update(caches=sets[r], pos=int(pos_next[r]), tok=int(np.argmax(logits[r])), step=s)
    assert {k: v["step"] for k, v in live.items()} == {"b": 6, "c": 4}


def test_decode_rows_device_sampler_and_device_state():
    """Greedy through the sampled head program at B=2 == the single-stream tokens; token ids and positions can be
    carried as device arrays between steps (nothing crosses to the host but what the caller reads)."""
    rng = np.random.default_rng(4)
    eng, V = build_engine(True, rng)
    prompts = [rng.integers(0, V, size=(1, n)) for n in (20, 33)]
    N = 4
    ref = [single_stream(eng, p, N) for p in prompts]
    sets, pos, toks = [], [], []
    for p in prompts:
        lg, cc, pp = eng.prefill(p)
        sets.append(cc); pos.append(pp); toks.append(int(jnp.argmax(lg[0])))
    pos = eng.device_positions(pos)
    toks = jnp.asarray(toks, jnp.int32)
    samp = DeviceSampler(0.0, 1.0, seed=0)
    got = [np.asarray(toks)]
    for _ in range(N):
        toks, logits, sets, pos = eng.decode_rows(toks, sets, pos, samp)
        assert isinstance(toks, jax.Array) and isinstance(pos, jax.Array)
        got.append(np.asarray(toks))
    got = np.stack(got, 1)                                    # [2, N+1]
    for r in range(2):
        assert got[r].tolist() == ref[r][1], (r, got[r].tolist(), ref[r][1])
    assert np.asarray(pos).tolist() == [20 + N, 33 + N]
