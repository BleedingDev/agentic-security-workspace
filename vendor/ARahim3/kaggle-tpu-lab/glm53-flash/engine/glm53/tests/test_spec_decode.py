"""Speculative decoding with an MTP (NextN) layer on the tiny model (ResidentLayerEngine, 8 CPU devices, interpret-mode
Pallas, Q2_K / Q3_K expert planes for the MTP layer): greedy spec decoding must reproduce plain greedy decoding
token for token — with random MTP weights (drafts mostly rejected: the rollback path) and with oracle drafts
(all accepted: the acceptance path), for k = 1 and k = 2; the MTP cache survives a snapshot round trip."""
import copy
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel  # noqa: E402

from glm53 import iqquant as Q  # noqa: E402
from glm53 import model as M  # noqa: E402
from glm53 import planes as PL  # noqa: E402
from glm53.hf_convert import from_hf_module  # noqa: E402
from glm53.engine import DeviceSampler  # noqa: E402
from glm53.resident import ResidentFetch, ResidentLayerEngine, pack_chip_planes  # noqa: E402
from glm53.tests.test_resident_cpu import random_tables  # noqa: E402
from glm53.tests.test_tiny_vs_hf import tiny_config  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
N_DEV = 8


def kq_tables(rng, E, D, mi):
    """Random Q2_K (gate/up) + Q3_K (down) planes in the resident layout, like random_tables for the IQ formats."""
    ml = mi // N_DEV
    out = {}
    for key, qt, rows, nblk in (("gate_q", "Q2_K", ml, D // 256), ("up_q", "Q2_K", ml, D // 256), ("down_q", "Q3_K", D, ml // 256)):
        bb = Q.BLOCK_BYTES[qt]
        t = rng.integers(0, 256, (N_DEV, E, rows, nblk, bb), dtype=np.uint8)
        off = 80 if qt == "Q2_K" else 108
        d = rng.uniform(0.005, 0.02, (N_DEV, E, rows, nblk)).astype(np.float16)
        t[..., off:off + 2] = np.frombuffer(d.tobytes(), np.uint8).reshape(N_DEV, E, rows, nblk, 2)
        if qt == "Q2_K":
            dm = rng.uniform(0.001, 0.004, (N_DEV, E, rows, nblk)).astype(np.float16)
            t[..., 82:84] = np.frombuffer(dm.tobytes(), np.uint8).reshape(N_DEV, E, rows, nblk, 2)
        out[key] = pack_chip_planes(t, qt)
    return out, {"gate_q": "Q2_K", "up_q": "Q2_K", "down_q": "Q3_K"}


def build(seq_shard, mtp=True):
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
    rng = np.random.default_rng(7)
    E, D, mi = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    packed_params = {**params, "layers": []}; qtypes = {}
    for i, L in enumerate(params["layers"]):
        if jcfg.mlp_types[i] == "sparse":
            packed, _ = random_tables(rng, E, D, mi)
            packed_params["layers"].append({**L, "mlp": {**{k: v for k, v in L["mlp"].items() if k not in ("gate_up", "down")}, **packed}})
            qtypes[i] = {"gate_q": "IQ2_S", "up_q": "IQ2_S", "down_q": "IQ3_S"}
        else:
            packed_params["layers"].append(L)
    fetch = ResidentFetch(qtypes, D, mi // N_DEV, out_dtype=jnp.float32, gather_max_rows=64, chunk=4, interpret=True, tm=8)
    eng = ResidentLayerEngine(jcfg, packed_params, fetch, max_len=128, layers_per_program=3, prefill_piece=32, seq_shard=seq_shard)
    if not mtp:
        return eng, cfg, rng
    # the MTP layer: the attention of an MLA+indexer layer, a router, random K-quant experts, random projections
    mla = [i for i in range(jcfg.n_layers) if jcfg.layer_types[i] != "linear_attention" and jcfg.mlp_types[i] == "sparse"][0]
    src = params["layers"][mla]
    tables, kq = kq_tables(rng, E, D, mi)
    mtp = {"ln1": np.ones(D, np.float32), "ln2": np.ones(D, np.float32), "attn": copy.deepcopy(src["attn"]),
           "mlp": {**{k: v for k, v in src["mlp"].items() if k not in ("gate_up", "down")}, **tables},
           "enorm": np.ones(D, np.float32), "hnorm": np.ones(D, np.float32),
           "eh_proj": (rng.standard_normal((2 * D, D)) * 0.05).astype(np.float32), "head_norm": np.ones(D, np.float32)}
    eng.set_mtp(mtp, kq, k=1)
    return eng, cfg, rng


def greedy(eng, ids, n):
    logits, caches, pos = eng.prefill(ids)
    out = [int(jnp.argmax(logits[0]))]
    for _ in range(n - 1):
        logits, caches, pos = eng.decode(np.array([out[-1]]), caches, pos)
        out.append(int(jnp.argmax(logits[0])))
    return out


def spec(eng, ids, n, k, oracle=None, sampler=None):
    logits, caches, pos = eng.prefill(ids)
    out = [int(jnp.argmax(logits[0]))]
    accepted = 0
    while len(out) < n:
        ov = None if oracle is None else oracle[len(out):len(out) + k]
        if ov is not None and len(ov) < k:
            ov = list(ov) + [0] * (k - len(ov))
        emitted, lg, caches, pos = eng.spec_decode(np.array([out[-1]]), caches, pos, k=k, draft_override=ov,
                                                   sampler=sampler)
        accepted += len(emitted) - 1
        out.extend(emitted)
    return out[:n], accepted, caches, pos


@pytest.mark.parametrize("seq_shard", [False, True])
def test_spec_decode_matches_greedy(seq_shard):
    eng, cfg, rng = build(seq_shard)
    ids = rng.integers(0, cfg.vocab_size, size=(1, 45))
    ref = greedy(eng, ids, 12)
    for k in (1, 2):
        out, acc, _, _ = spec(eng, ids, 12, k)                                   # random MTP: rollback path
        assert out == ref, (k, out, ref)
        out, acc, caches, pos = spec(eng, ids, 12, k, oracle=ref)                # oracle drafts: acceptance path
        assert out == ref and acc > 0, (k, out, ref, acc)
        assert pos >= 45 + 11 and len(caches) == eng.cfg.n_layers + 1       # (oracle drafts may overshoot n)
    # device-side sampling: temperature 0 and a degenerate nucleus (top_p ~ 0 keeps the argmax only) both reproduce
    # plain greedy, through decode_sample and through the spec verify (ids sampled in the head program)
    for samp in (DeviceSampler(0.0, 1.0, seed=1), DeviceSampler(1.0, 1e-6, seed=2)):
        logits, caches, pos = eng.prefill(ids)
        out = [int(jnp.argmax(logits[0]))]
        while len(out) < 12:
            nid, logits, caches, pos = eng.decode_sample(np.array([out[-1]]), caches, pos, samp)
            out.append(int(nid[0]))
        assert out == ref, (samp.temperature, out, ref)
        for k in (1, 2):
            out, _, _, _ = spec(eng, ids, 12, k, sampler=samp)
            assert out == ref, (samp.temperature, k, out, ref)
            out, acc, _, _ = spec(eng, ids, 12, k, oracle=ref, sampler=samp)
            assert out == ref and acc > 0, (samp.temperature, k, out, ref, acc)
    # the MTP cache entry survives a compact snapshot round trip and the context continues identically
    logits, caches, pos = eng.prefill(ids[:, :40])
    snap = eng.snapshot_to_host(eng.snapshot_prefix(caches, pos, rows_bucket=16))
    restored = eng.restore_prefix(eng.snapshot_from_host(snap))
    l1, c1, p1 = eng.prefill(ids[:, 40:], caches, pos)
    l2, c2, p2 = eng.prefill(ids[:, 40:], restored, pos)
    np.testing.assert_allclose(np.asarray(l2), np.asarray(l1), rtol=1e-4, atol=1e-4)
    e1, _, _, _ = eng.spec_decode(np.array([int(jnp.argmax(l1[0]))]), c1, p1, k=1)
    e2, _, _, _ = eng.spec_decode(np.array([int(jnp.argmax(l2[0]))]), c2, p2, k=1)
    assert e1 == e2


def test_ragged_verify_matches_greedy():
    """Phase 2: the verify's MoE through the ragged (grouped) kernels — every routed expert dequantized once for all
    T positions — still reproduces plain greedy decoding (rollback and acceptance paths, verifies of 2..8 positions,
    with the attention's per-query cut-off at 4 and at 8 queries); no multi-token step falls back to the per-slot
    kernels."""
    import dataclasses
    eng, cfg, rng = build(False)
    ids = rng.integers(0, cfg.vocab_size, size=(1, 45))
    ref = greedy(eng, ids, 12)
    f = eng.expert_fetch
    slot = f._apply_kernel

    def only_single(x, *a):
        assert x.shape[0] == 1, f"a {x.shape[0]}-token step took the per-slot kernels"
        return slot(x, *a)
    rms0 = f.ragged_min_seq
    f._apply_kernel, f.ragged_min_seq = only_single, 2
    l0 = eng.lcfg
    try:
        for pq in (4, 8):
            eng.lcfg, eng._progs = dataclasses.replace(l0, per_query_max=pq), {}
            for k in (1, 2):
                out, _, _, _ = spec(eng, ids, 12, k)                             # random MTP: rollback
                assert out == ref, (pq, k, out, ref)
            for k in (2, 7):
                out, acc, _, _ = spec(eng, ids, 12, k, oracle=ref)              # oracle drafts: acceptance, T = k+1
                assert out == ref and acc > 0, (pq, k, out, ref, acc)
    finally:
        f._apply_kernel, f.ragged_min_seq, eng.lcfg = slot, rms0, l0
