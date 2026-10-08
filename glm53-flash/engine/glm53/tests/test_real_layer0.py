"""Real-checkpoint layer 0 (KDA + dense MLP + mHC, fp8 MLP) : glm53 loader+model vs HF decoder layer, CPU fp32.

Needs shard 2 + index + config locally:  MODEL_DIR must contain model.safetensors.index.json, config.json and
model-00002-of-00062.safetensors (hf_hub_download into the same snapshot dir).  Skips otherwise.
"""
import glob
import os

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from glm53 import checkpoint as C  # noqa: E402
from glm53 import model as M  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def find_model_dir():
    env = os.environ.get("GLM_MODEL_DIR")
    cands = [env] if env else []
    cands += glob.glob(os.path.expanduser("~/.cache/glm_hf/models--zai-org--GLM-5.3-Flash/snapshots/*"))
    for d in cands:
        if d and os.path.exists(os.path.join(d, "model-00002-of-00062.safetensors")) and \
                os.path.exists(os.path.join(d, "model.safetensors.index.json")):
            return d
    return None


MODEL_DIR = find_model_dir()
pytestmark = pytest.mark.skipif(MODEL_DIR is None, reason="real shard 2 not downloaded")


def hf_layer0(text_cfg, r: C.ShardReader):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
    layer = Glm5NextTextDecoderLayer(text_cfg, 0).float().eval()
    pfx = C.PREFIX + "layers.0."
    sd = {}
    for name in r.wm:
        if not name.startswith(pfx) or name.endswith("_scale_inv"):
            continue
        key = name[len(pfx):]
        key = key.replace("self_attn.f_a_proj.", "self_attn.forget_gate.f_a_proj.")
        key = key.replace("self_attn.f_b_proj.", "self_attn.forget_gate.f_b_proj.")
        key = key.replace("self_attn.dt_bias", "self_attn.forget_gate.dt_bias")
        key = key.replace("self_attn.A_log", "self_attn.forget_gate.A_log")
        for site, tgt in (("attn", "attn_hc"), ("ffn", "ffn_hc")):
            for part in ("fn", "base", "scale"):
                key = key.replace(f"hc_{site}_{part}", f"{tgt}.{part}")
        if "conv1d" in key:
            continue
        sd[key] = torch.from_numpy(r.get(name))
    a = pfx + "self_attn."
    sd["self_attn.conv1d.weight"] = torch.cat([torch.from_numpy(r.get(a + f"{c}_conv1d.weight")) for c in "qkv"], 0)
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    return layer


def test_layer0_real_weights():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    cfg = Glm5NextConfig.from_pretrained(MODEL_DIR)
    tc = cfg.text_config
    tc.dtype = torch.float32
    r = C.ShardReader(MODEL_DIR)
    hf = hf_layer0(tc, r)
    p = jax.tree.map(jnp.asarray, C.load_layer(r, 0, tc.layer_types[0], tc.mlp_layer_types[0]))
    jcfg = M.Cfg.from_hf(tc)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 12, tc.hc_mult, tc.hidden_size)) * 0.5).astype(np.float32)
    with torch.no_grad():
        ref, _ = hf(torch.from_numpy(x), attention_mask=None, position_ids=None, past_key_values=None)
    ref = ref.numpy()
    out, _ = M.decoder_layer(p, jnp.asarray(x), jcfg, tc.layer_types[0], tc.mlp_layer_types[0])
    out = np.asarray(out)
    rel = np.abs(out - ref).max() / np.abs(ref).max()
    print("layer0 real-weights rel err", rel, "ref scale", np.abs(ref).max())
    assert rel < 1e-3
