"""fp8 streaming mode of the TP engine on CPU: random e4m3 expert bytes + 128x128 scales, compared with the
float-mode engine fed the numpy-dequantized tables."""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel  # noqa: E402

from glm53 import model as M  # noqa: E402
from glm53.checkpoint import dequant_fp8  # noqa: E402
from glm53.engine import Engine, HostExperts, chip_major, split_gate_up  # noqa: E402
from glm53.hf_convert import from_hf_module  # noqa: E402
from glm53.tests.test_tiny_vs_hf import tiny_config  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def test_fp8_stream_mode():
    torch.manual_seed(0)
    cfg = tiny_config(index_topk=16)
    cfg.index_n_heads = 8
    cfg.num_attention_heads = cfg.num_key_value_heads = 8
    cfg.linear_num_heads = 8
    cfg.intermediate_size = 128
    cfg.hidden_size = 256            # D multiple of 128 for the block scales
    cfg.moe_intermediate_size = 1024  # ml = 128 per device
    hf = Glm5NextTextModel(cfg).float().eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    params = from_hf_module(hf, lm_head)
    jcfg = M.Cfg.from_hf(cfg)
    n_dev = 8
    rng = np.random.default_rng(0)
    float_tables, fp8_tables = {}, {}
    D, mi, E = cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_routed_experts
    for i in range(cfg.num_hidden_layers):
        if cfg.mlp_layer_types[i] != "sparse":
            continue
        # random fp8 bytes (avoid NaN pattern 0x7f/0xff) + random scales
        gu_b = rng.integers(0, 256, size=(E, D, 2 * mi), dtype=np.uint8)
        dn_b = rng.integers(0, 256, size=(E, mi, D), dtype=np.uint8)
        gu_b[(gu_b & 0x7F) == 0x7F] = 0
        dn_b[(dn_b & 0x7F) == 0x7F] = 0
        gu_s = rng.uniform(0.001, 0.01, size=(E, D // 128, 2 * mi // 128)).astype(np.float32)
        dn_s = rng.uniform(0.001, 0.01, size=(E, mi // 128, D // 128)).astype(np.float32)
        f8 = lambda b: np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(b), jnp.float8_e4m3fn).astype(jnp.float32))
        gu_f = np.stack([dequant_fp8(f8(gu_b[e]), gu_s[e]) for e in range(E)])
        dn_f = np.stack([dequant_fp8(f8(dn_b[e]), dn_s[e]) for e in range(E)])
        params["layers"][i]["mlp"]["gate_up"] = gu_f
        params["layers"][i]["mlp"]["down"] = dn_f
        float_tables[i] = (chip_major(split_gate_up(gu_f, n_dev)), chip_major(dn_f.reshape(E, n_dev, mi // n_dev, D)))
        fp8_tables[i] = (chip_major(split_gate_up(gu_b, n_dev)), chip_major(split_gate_up(gu_s, n_dev)),
                         chip_major(dn_b.reshape(E, n_dev, mi // n_dev, D)),
                         chip_major(dn_s.reshape(E, n_dev, mi // n_dev // 128, D // 128)))
    ids = rng.integers(0, cfg.vocab_size, size=(1, 12))
    eng_f = Engine(jcfg, params, HostExperts(float_tables, "float"))
    eng_8 = Engine(jcfg, params, HostExperts(fp8_tables, "fp8"))
    lf, _, _ = eng_f.prefill(ids)
    l8, _, _ = eng_8.prefill(ids)
    ref = M.logits(jax.tree.map(jnp.asarray, params), M.forward(jax.tree.map(jnp.asarray, params), jnp.asarray(ids), jcfg)[0][:, -1])
    for name, l in (("float", lf), ("fp8", l8)):
        rel = np.abs(np.asarray(l) - np.asarray(ref)).max() / np.abs(np.asarray(ref)).max()
        print(name, "rel err", rel)
        assert rel < 2e-3
