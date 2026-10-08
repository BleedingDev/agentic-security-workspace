"""RawShardReader (pread, no mmap) vs safetensors safe_open on a synthetic shard with fp8/bf16/f32 tensors."""
import json, os
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from glm53 import checkpoint as C


def test_raw_reader_matches_safetensors(tmp_path):
    torch.manual_seed(0)
    w8 = (torch.randn(256, 384) * 0.05).to(torch.float8_e4m3fn)
    scale = torch.rand(2, 3) * 0.01 + 0.001
    tensors = {"model.language_model.layers.0.mlp.gate_proj.weight": w8,
               "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv": scale,
               "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(64, 128).to(torch.bfloat16),
               "model.language_model.layers.0.self_attn.A_log": torch.randn(8)}
    save_file(tensors, str(tmp_path / "model-00001-of-00001.safetensors"))
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}},
              open(tmp_path / "model.safetensors.index.json", "w"))
    C.weight_map.cache_clear()
    ref = C.ShardReader(str(tmp_path))
    raw = C.RawShardReader(str(tmp_path))
    for k in tensors:
        if k.endswith("_scale_inv"):
            continue
        a, b = ref.get(k), raw.get(k)
        assert a.shape == b.shape and np.abs(a - b).max() == 0, k
    u8a, sa = ref.get_fp8_raw("model.language_model.layers.0.mlp.gate_proj.weight")
    u8b, sb = raw.get_fp8_raw("model.language_model.layers.0.mlp.gate_proj.weight")
    assert np.array_equal(u8a, u8b) and np.array_equal(sa, sb)
