"""Resident codebook-quantized experts (gather path for decode, sweep path for prefill) vs the dense on-device
Engine fed with the dequantized tables. CPU x8 virtual devices, float32."""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel  # noqa: E402

from glm53 import iqquant as Q  # noqa: E402
from glm53 import model as M  # noqa: E402
from glm53.engine import Engine  # noqa: E402
from glm53.hf_convert import from_hf_module  # noqa: E402
from glm53.resident import ResidentFetch, hbm_bytes, pack_chip_planes  # noqa: E402
from glm53.tests.test_tiny_vs_hf import tiny_config  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
N_DEV = 8


def random_tables(rng, E, D, mi, gu_type="IQ2_S", dn_type="IQ3_S"):
    """Random (valid) packed blocks in the resident layout + the dequantized dense tables they represent."""
    ml = mi // N_DEV
    bbg, bbd = Q.BLOCK_BYTES[gu_type], Q.BLOCK_BYTES[dn_type]

    def blocks(shape, bb, scale):
        t = rng.integers(0, 256, shape + (bb,), dtype=np.uint8)
        d = rng.uniform(0.5 * scale, scale, shape).astype(np.float16)
        t[..., 0:2] = np.frombuffer(d.tobytes(), np.uint8).reshape(shape + (2,))
        return t

    gate_q = blocks((N_DEV, E, ml, D // 256), bbg, 0.02)
    up_q = blocks((N_DEV, E, ml, D // 256), bbg, 0.02)
    down_q = blocks((N_DEV, E, D, ml // 256), bbd, 0.02)
    deq = lambda t, ty: np.asarray(Q.DEQUANT[ty](jnp.asarray(t))).reshape(t.shape[:-2] + (-1,))
    g, u, d = deq(gate_q, gu_type), deq(up_q, gu_type), deq(down_q, dn_type)     # [n,E,ml,D] x2, [n,E,D,ml]
    gate = np.concatenate(list(g), axis=1)            # [E, mi, D]
    up = np.concatenate(list(u), axis=1)
    down = np.concatenate(list(d), axis=2)            # [E, D, mi]
    gate_up = np.concatenate([gate.transpose(0, 2, 1), up.transpose(0, 2, 1)], axis=2)   # [E, D, 2mi]
    dense = {"gate_up": gate_up.astype(np.float32), "down": down.transpose(0, 2, 1).astype(np.float32)}
    packed = {"gate_q": pack_chip_planes(gate_q, gu_type), "up_q": pack_chip_planes(up_q, gu_type),
              "down_q": pack_chip_planes(down_q, dn_type)}
    return packed, dense


def test_resident_matches_dense():
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=16)
    cfg.hidden_size = 256                  # quant blocks are 256 wide
    cfg.moe_intermediate_size = 2048       # ml = 256 per chip -> one down block per chip
    cfg.intermediate_size = 256
    cfg.index_n_heads = 8
    cfg.num_attention_heads = cfg.num_key_value_heads = 8
    cfg.linear_num_heads = 8
    cfg.q_lora_rank, cfg.kv_lora_rank = 64, 32
    cfg.linear_attn_config["num_heads"] = 8
    hf = Glm5NextTextModel(cfg).float().eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    params = from_hf_module(hf, lm_head)
    jcfg = M.Cfg.from_hf(cfg)
    rng = np.random.default_rng(0)
    E, D, mi = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    dense_params = {**params, "layers": []}
    packed_params = {**params, "layers": []}
    qtypes = {}
    for i, L in enumerate(params["layers"]):
        if jcfg.mlp_types[i] == "sparse":
            packed, dense = random_tables(rng, E, D, mi)
            dense_params["layers"].append({**L, "mlp": {**{k: v for k, v in L["mlp"].items() if k not in ("gate_up", "down")}, **dense}})
            packed_params["layers"].append({**L, "mlp": {**{k: v for k, v in L["mlp"].items() if k not in ("gate_up", "down")}, **packed}})
            qtypes[i] = {"gate_q": "IQ2_S", "up_q": "IQ2_S", "down_q": "IQ3_S"}
        else:
            dense_params["layers"].append(L); packed_params["layers"].append(L)
    # planar layout keeps every block byte; `d` (f16 per block) is padded to a whole u32 per pair of blocks
    ml = mi // N_DEV
    assert hbm_bytes(packed) == E * (2 * ml * (D // 256) * 82 + D * (ml // 256) * 110) + E * (2 * ml + D) * 2 * ((D // 256) % 2)
    ids = rng.integers(0, cfg.vocab_size, size=(1, 20))
    ref = Engine(jcfg, dense_params)
    res = Engine(jcfg, packed_params, expert_fetch=ResidentFetch(qtypes, D, mi // N_DEV, out_dtype=jnp.float32, gather_max_rows=8, chunk=4, interpret=True))
    # prefill (sweep path: N*k = 20*2 > 8) then 3 decode steps (gather path: N*k = 2)
    la, ca, pa = ref.prefill(ids); lb, cb, pb = res.prefill(ids)
    lb0 = np.asarray(lb)
    np.testing.assert_allclose(lb0, np.asarray(la), rtol=1e-4, atol=1e-4)
    tok = np.array([int(jnp.argmax(la[0]))])
    for _ in range(3):
        la, ca, pa = ref.decode(tok, ca, pa); lb, cb, pb = res.decode(tok, cb, pb)
        np.testing.assert_allclose(np.asarray(lb), np.asarray(la), rtol=1e-4, atol=1e-4)
        tok = np.array([int(jnp.argmax(la[0]))])
    # both paths on the same shapes must agree with each other too (gather vs sweep for a small prefill)
    res2 = Engine(jcfg, packed_params, expert_fetch=ResidentFetch(qtypes, D, mi // N_DEV, out_dtype=jnp.float32, gather_max_rows=1000, chunk=8, interpret=True))
    lc, _, _ = res2.prefill(ids)
    np.testing.assert_allclose(np.asarray(lc), lb0, rtol=1e-4, atol=1e-4)
    # the XLA (no Pallas) decode path on the same planes: gather of 20*2 rows via dynamic slices + XLA dequant
    res3 = Engine(jcfg, packed_params, expert_fetch=ResidentFetch(qtypes, D, mi // N_DEV, out_dtype=jnp.float32, gather_max_rows=1000, chunk=8, use_pallas=False))
    ld, _, _ = res3.prefill(ids)
    np.testing.assert_allclose(np.asarray(ld), lb0, rtol=1e-4, atol=1e-4)


def test_layer_engine_matches_single_program():
    """ResidentLayerEngine (one program per layer kind) == Engine (one program) on the same packed tables."""
    from glm53.resident import ResidentLayerEngine
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=16)
    cfg.hidden_size = 256; cfg.moe_intermediate_size = 2048; cfg.intermediate_size = 256
    cfg.index_n_heads = 8; cfg.num_attention_heads = cfg.num_key_value_heads = 8; cfg.linear_num_heads = 8
    cfg.q_lora_rank, cfg.kv_lora_rank = 64, 32; cfg.linear_attn_config["num_heads"] = 8
    hf = Glm5NextTextModel(cfg).float().eval()
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
    ids = rng.integers(0, cfg.vocab_size, size=(1, 20))
    mk = lambda: ResidentFetch(qtypes, D, mi // N_DEV, out_dtype=jnp.float32, gather_max_rows=8, chunk=4, interpret=True)
    a = Engine(jcfg, packed_params, expert_fetch=mk())
    b = ResidentLayerEngine(jcfg, packed_params, mk(), layers_per_program=1)
    c = ResidentLayerEngine(jcfg, packed_params, mk(), layers_per_program=4, layers_per_program_prefill=2)
    la, ca, pa = a.prefill(ids); lb, cb, pb = b.prefill(ids); lc, cc, pc = c.prefill(ids)
    np.testing.assert_allclose(np.asarray(lb), np.asarray(la), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(lc), np.asarray(la), rtol=1e-4, atol=1e-4)
    assert b.launches == jcfg.n_layers and c.launches == 3
    tok = np.array([int(jnp.argmax(la[0]))])
    for _ in range(3):
        la, ca, pa = a.decode(tok, ca, pa); lb, cb, pb = b.decode(tok, cb, pb); lc, cc, pc = c.decode(tok, cc, pc)
        np.testing.assert_allclose(np.asarray(lb), np.asarray(la), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(lc), np.asarray(la), rtol=1e-4, atol=1e-4)
        tok = np.array([int(jnp.argmax(la[0]))])
    kinds = {k[1] for k in b._progs if k[0] == "group"}
    assert len(kinds) <= 4     # (KDA|MLA) x (dense|sparse) for the tiny config


def test_int8_nonexpert_close():
    """int8 per-channel non-expert weights: logits close to bf16/fp32 weights and same argmax."""
    from glm53.resident import ResidentLayerEngine
    from glm53 import quant8 as Q8
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=16)
    cfg.hidden_size = 256; cfg.moe_intermediate_size = 2048; cfg.intermediate_size = 4096   # dense MLP big enough to quantize
    cfg.index_n_heads = 8; cfg.num_attention_heads = cfg.num_key_value_heads = 8; cfg.linear_num_heads = 8
    cfg.q_lora_rank, cfg.kv_lora_rank = 64, 32; cfg.linear_attn_config["num_heads"] = 8
    hf = Glm5NextTextModel(cfg).float().eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    params = from_hf_module(hf, lm_head)
    jcfg = M.Cfg.from_hf(cfg)
    rng = np.random.default_rng(2)
    E, D, mi = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    pp = {**params, "layers": []}; qtypes = {}
    for i, L in enumerate(params["layers"]):
        if jcfg.mlp_types[i] == "sparse":
            packed, _ = random_tables(rng, E, D, mi)
            pp["layers"].append({**L, "mlp": {**{k: v for k, v in L["mlp"].items() if k not in ("gate_up", "down")}, **packed}})
            qtypes[i] = {"gate_q": "IQ2_S", "up_q": "IQ2_S", "down_q": "IQ3_S"}
        else:
            pp["layers"].append(L)
    ids = rng.integers(0, cfg.vocab_size, size=(1, 12))
    mk = lambda: ResidentFetch(qtypes, D, mi // N_DEV, out_dtype=jnp.float32, gather_max_rows=8, chunk=4, interpret=True)
    a = ResidentLayerEngine(jcfg, pp, mk())
    b = ResidentLayerEngine(jcfg, pp, mk(), int8_nonexpert=True)
    nq = sum(1 for L in b.params["layers"] for v in jax.tree.leaves(L, is_leaf=Q8.is_q8) if Q8.is_q8(v))
    assert nq >= 1, "nothing was quantized"
    assert Q8.count_bytes(b.params["layers"]) < Q8.count_bytes(a.params["layers"]) - 3 * (1 << 20)   # 3 dense-MLP matrices shrank 4x
    la, ca, pa = a.prefill(ids); lb, cb, pb = b.prefill(ids)
    la, lb = np.asarray(la), np.asarray(lb)
    assert np.argmax(la) == np.argmax(lb)
    assert np.abs(la - lb).max() < 0.10 * np.abs(la).max()
    # "all": the embedding table (per-row scales, dequantized per gathered row) and lm_head (per-column) too
    c = ResidentLayerEngine(jcfg, pp, mk(), int8_nonexpert="all")
    assert Q8.is_q8(c.params["embed"]) and Q8.is_q8(c.params["lm_head"])
    lc, cc, pc = c.prefill(ids); lc = np.asarray(lc)
    assert np.abs(la - lc).max() < 0.10 * np.abs(la).max()        # (the random tiny model's top-1 is a near-tie)
    tokn = np.array([int(np.argmax(la))])
    ld, _, _ = a.decode(tokn, ca, pa); le, _, _ = c.decode(tokn, cc, pc)
    ld, le = np.asarray(ld), np.asarray(le)
    assert np.abs(ld - le).max() < 0.10 * np.abs(ld).max()


@pytest.mark.parametrize("dn_type,combine,tm", [("IQ3_S", "gather", 8), ("IQ3_S", "matmul", 8), ("IQ4_XS", "gather", 16)])
def test_ragged_sweep_matches_dense(dn_type, combine, tm):
    """The grouped (ragged) prefill GEMM path == the dense reference on the same packed tables (interpret mode)."""
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=16)
    cfg.hidden_size = 256; cfg.moe_intermediate_size = 2048; cfg.intermediate_size = 256
    cfg.index_n_heads = 8; cfg.num_attention_heads = cfg.num_key_value_heads = 8; cfg.linear_num_heads = 8
    cfg.q_lora_rank, cfg.kv_lora_rank = 64, 32; cfg.linear_attn_config["num_heads"] = 8
    hf = Glm5NextTextModel(cfg).float().eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    params = from_hf_module(hf, lm_head)
    jcfg = M.Cfg.from_hf(cfg)
    rng = np.random.default_rng(3)
    E, D, mi = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    dense_params = {**params, "layers": []}; packed_params = {**params, "layers": []}; qtypes = {}
    for i, L in enumerate(params["layers"]):
        if jcfg.mlp_types[i] == "sparse":
            packed, dense = random_tables(rng, E, D, mi, dn_type=dn_type)
            dense_params["layers"].append({**L, "mlp": {**{k: v for k, v in L["mlp"].items() if k not in ("gate_up", "down")}, **dense}})
            packed_params["layers"].append({**L, "mlp": {**{k: v for k, v in L["mlp"].items() if k not in ("gate_up", "down")}, **packed}})
            qtypes[i] = {"gate_q": "IQ2_S", "up_q": "IQ2_S", "down_q": dn_type}
        else:
            dense_params["layers"].append(L); packed_params["layers"].append(L)
    n_tok = 24 if tm == 16 else 20                       # T*k must be a multiple of tm (k = 2 in the tiny config)
    ids = rng.integers(0, cfg.vocab_size, size=(1, n_tok))
    ref = Engine(jcfg, dense_params)
    rag = Engine(jcfg, packed_params, expert_fetch=ResidentFetch(qtypes, D, mi // N_DEV, out_dtype=jnp.float32, gather_max_rows=8,
                                                                  chunk=4, interpret=True, sweep_mode="ragged", tm=tm, combine=combine))
    la, _, _ = ref.prefill(ids); lb, _, _ = rag.prefill(ids)
    np.testing.assert_allclose(np.asarray(lb), np.asarray(la), rtol=1e-4, atol=1e-4)


def test_prefill_embedding_override():
    """prefill(embeds=) replaces token embeddings: overriding positions with the table rows of other tokens gives the
    logits of prefilling those tokens, across piece boundaries and in a continued context."""
    from glm53.resident import ResidentLayerEngine
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=16)
    cfg.hidden_size = 256; cfg.moe_intermediate_size = 2048; cfg.intermediate_size = 256
    cfg.index_n_heads = 8; cfg.num_attention_heads = cfg.num_key_value_heads = 8; cfg.linear_num_heads = 8
    cfg.q_lora_rank, cfg.kv_lora_rank = 64, 32; cfg.linear_attn_config["num_heads"] = 8
    hf = Glm5NextTextModel(cfg).float().eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    params = from_hf_module(hf, lm_head)
    jcfg = M.Cfg.from_hf(cfg)
    rng = np.random.default_rng(3)
    E, D, mi = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    packed_params = {**params, "layers": []}; qtypes = {}
    for i, L in enumerate(params["layers"]):
        if jcfg.mlp_types[i] == "sparse":
            packed, _ = random_tables(rng, E, D, mi)
            packed_params["layers"].append({**L, "mlp": {**{k: v for k, v in L["mlp"].items() if k not in ("gate_up", "down")}, **packed}})
            qtypes[i] = {"gate_q": "IQ2_S", "up_q": "IQ2_S", "down_q": "IQ3_S"}
        else:
            packed_params["layers"].append(L)
    fetch = ResidentFetch(qtypes, D, mi // N_DEV, out_dtype=jnp.float32, gather_max_rows=8, chunk=4, interpret=True)
    eng = ResidentLayerEngine(jcfg, packed_params, fetch, max_len=128, layers_per_program=3, prefill_piece=32, seq_shard=True)
    table = np.asarray(params["embed"])
    ids = rng.integers(0, cfg.vocab_size, size=(1, 50))
    alt = rng.integers(0, cfg.vocab_size, size=(1, 50))
    idx = np.array([3, 31, 32, 45])                                     # straddles the 32-token piece boundary
    mixed = ids.copy(); mixed[0, idx] = alt[0, idx]
    ref, cref, pref = eng.prefill(mixed)
    out, cout, pout = eng.prefill(ids, embeds=(idx, table[alt[0, idx]]))
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-4, atol=1e-4)
    # the caches agree too: continue both contexts with the same tokens
    more = rng.integers(0, cfg.vocab_size, size=(1, 7))
    l1, _, _ = eng.prefill(more, cref, pref)
    l2, _, _ = eng.prefill(more, cout, pout)
    np.testing.assert_allclose(np.asarray(l2), np.asarray(l1), rtol=1e-4, atol=1e-4)
    # override in a continued context (positions relative to the new tokens)
    _, c1, p1 = eng.prefill(ids[:, :20]); _, c2, p2 = eng.prefill(ids[:, :20])
    l1, _, _ = eng.prefill(mixed[:, 20:], c1, p1)
    l2, _, _ = eng.prefill(ids[:, 20:], c2, p2, embeds=(idx - 20, table[alt[0, idx]]))
    np.testing.assert_allclose(np.asarray(l2), np.asarray(l1), rtol=1e-4, atol=1e-4)


def test_lm_head_t_forms():
    """`engine.lm_head_t` stores the lm_head [V, D] with [V, 1] scales from every form it is given: the checkpoint's
    [D, V] array, the base store's pre-quantized {q [D, V], s [1, V]} (the TPU kernel's path: a build failed on the
    unconverted scales, 2026-10-05), and an already converted dict (idempotent)."""
    from glm53.engine import lm_head_t
    from glm53 import quant8 as Q8
    D, V = 16, 64
    w = np.random.default_rng(0).standard_normal((D, V)).astype(np.float32)
    assert lm_head_t(w).shape == (V, D)
    t = lm_head_t(Q8.quantize_array_host(w, 1))
    ref = Q8.quantize_array_host(np.ascontiguousarray(w.T), 0)
    assert t["q"].shape == (V, D) and t["s"].shape == (V, 1)
    np.testing.assert_array_equal(t["q"], ref["q"]); np.testing.assert_allclose(t["s"], ref["s"])
    t2 = lm_head_t(t)
    assert t2["q"].shape == (V, D) and t2["s"].shape == (V, 1)
