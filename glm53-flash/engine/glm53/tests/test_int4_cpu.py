"""int4 RTN pack/unpack round trip + int4 streaming mode of the engine (CPU, 8 virtual devices)."""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel  # noqa: E402

from glm53 import model as M  # noqa: E402
from glm53.checkpoint import dequant_int4_device, dequant_int4_numpy, quant_int4_blockwise  # noqa: E402
from glm53.engine import Engine, HostExperts, chip_major, split_gate_up  # noqa: E402
from glm53.hf_convert import from_hf_module  # noqa: E402
from glm53.tests.test_tiny_vs_hf import tiny_config  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def test_int4_roundtrip():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((256, 384)).astype(np.float32) * 0.02
    p, s = quant_int4_blockwise(w)
    assert p.shape == (256, 192) and s.shape == (2, 3)
    d_np = dequant_int4_numpy(p, s)
    d_dev = np.asarray(dequant_int4_device(jnp.asarray(p), jnp.asarray(s)))
    assert np.abs(d_np - d_dev).max() == 0
    rel_rms = np.sqrt(((d_np - w) ** 2).mean()) / np.sqrt((w ** 2).mean())
    print("int4 RTN rel rms err", rel_rms)
    assert rel_rms < 0.2   # ~0.165 expected for Gaussian, symmetric RTN, 128x128 blocks


def test_int4_stream_mode():
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=16)
    cfg.index_n_heads = 8
    cfg.num_attention_heads = cfg.num_key_value_heads = 8
    cfg.linear_num_heads = 8
    cfg.intermediate_size = 128
    cfg.hidden_size = 256
    cfg.moe_intermediate_size = 1024
    hf = Glm5NextTextModel(cfg).float().eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    params = from_hf_module(hf, lm_head)
    jcfg = M.Cfg.from_hf(cfg)
    n_dev, D, mi, E = 8, cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_routed_experts
    ml = mi // n_dev
    int4_tables = {}
    for i in range(cfg.num_hidden_layers):
        if cfg.mlp_layer_types[i] != "sparse":
            continue
        gu = params["layers"][i]["mlp"]["gate_up"]      # [E, D, 2mi]
        dn = params["layers"][i]["mlp"]["down"]         # [E, mi, D]
        gu_p, gu_s, gu_deq = [], [], []
        for e in range(E):
            p_, s_ = quant_int4_blockwise(gu[e]); gu_p.append(p_); gu_s.append(s_); gu_deq.append(dequant_int4_numpy(p_, s_))
        dn_p, dn_s, dn_deq = [], [], []
        for e in range(E):
            p_, s_ = quant_int4_blockwise(dn[e]); dn_p.append(p_); dn_s.append(s_); dn_deq.append(dequant_int4_numpy(p_, s_))
        # reference params use the dequantized weights so the comparison isolates the streaming mechanics
        params["layers"][i]["mlp"]["gate_up"] = np.stack(gu_deq)
        params["layers"][i]["mlp"]["down"] = np.stack(dn_deq)
        # packed gate_up columns: [E, D, mi] bytes = (gate|up) halves of mi/2 bytes each -> per device split
        # must split *unpacked column* space: split then repack. Simplest: split the dequantized float table, requantize
        # per device slice (identical numbers since blocks are 128-aligned: ml = 128 = one block).
        gu_split = split_gate_up(np.stack(gu_deq), n_dev)                      # [E, n, D, 2*ml]
        dn_split = np.stack(dn_deq).reshape(E, n_dev, ml, D)
        gp = np.empty((E, n_dev, D, ml), np.uint8); gs = np.empty((E, n_dev, D // 128, 2 * ml // 128), np.float32)
        dp = np.empty((E, n_dev, ml, D // 2), np.uint8); ds = np.empty((E, n_dev, ml // 128, D // 128), np.float32)
        for e in range(E):
            for c in range(n_dev):
                gp[e, c], gs[e, c] = quant_int4_blockwise(gu_split[e, c])
                dp[e, c], ds[e, c] = quant_int4_blockwise(dn_split[e, c])
        int4_tables[i] = tuple(chip_major(x) for x in (gp, gs, dp, ds))
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab_size, size=(1, 12))
    eng = Engine(jcfg, params, HostExperts(int4_tables, "int4"))
    l4, _, _ = eng.prefill(ids)
    # mixed: first sparse layer float, the rest int4 (what the full build produced when host RAM ran out)
    from glm53.engine import SegmentedEngine
    sparse_layers = sorted(int4_tables)
    mixed = dict(int4_tables)
    L0 = sparse_layers[0]
    mixed[L0] = (chip_major(split_gate_up(params["layers"][L0]["mlp"]["gate_up"], n_dev)),
                 chip_major(params["layers"][L0]["mlp"]["down"].reshape(E, n_dev, ml, D)))
    modes = {L: ("float" if L == L0 else "int4") for L in sparse_layers}
    eng_m = SegmentedEngine(jcfg, params, HostExperts(mixed, modes))
    lm, _, _ = eng_m.prefill(ids)
    relm = np.abs(np.asarray(lm) - np.asarray(l4)).max() / np.abs(np.asarray(l4)).max()
    print("mixed-mode segmented vs int4 rel err", relm)
    assert relm < 2e-3
    pj = jax.tree.map(jnp.asarray, params)
    ref = M.logits(pj, M.forward(pj, jnp.asarray(ids), jcfg)[0][:, -1])
    rel = np.abs(np.asarray(l4) - np.asarray(ref)).max() / np.abs(np.asarray(ref)).max()
    print("int4 stream rel err", rel)
    assert rel < 2e-3
