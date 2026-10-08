"""Convert Glm5Next weights to the JAX param layout of `glm53.model`.

Two sources:
  * `from_hf_module(model)`: a transformers `Glm5NextTextModel` (used for tiny random-init tests).
  * `dequant_fp8(w, scale_inv, block=128)`: helper for real checkpoints (fp8 e4m3 blockwise).
"""
from __future__ import annotations

import numpy as np


def _np(t):
    import torch
    if t.dtype == torch.bfloat16:
        return t.detach().float().cpu().numpy()
    return t.detach().cpu().numpy()


def _lin(t):
    """torch Linear weight [out, in] -> JAX [in, out]."""
    return np.ascontiguousarray(_np(t).T)


def convert_hc(mod):
    return {"fn": _lin(mod.fn), "base": _np(mod.base), "scale": _np(mod.scale)}


def convert_kda(a):
    C = a.qkv_dim
    conv_w = _np(a.conv1d.weight)[:, 0, :]                     # [3C, K]
    return {
        "q": _lin(a.q_proj.weight), "k": _lin(a.k_proj.weight), "v": _lin(a.v_proj.weight),
        "conv_q": conv_w[:C], "conv_k": conv_w[C:2 * C], "conv_v": conv_w[2 * C:],
        "f_a": _lin(a.forget_gate.f_a_proj.weight), "f_b": _lin(a.forget_gate.f_b_proj.weight),
        "dt_bias": _np(a.forget_gate.dt_bias), "A_log": _np(a.forget_gate.A_log),
        "b": _lin(a.b_proj.weight),
        "g_a": _lin(a.g_a_proj.weight), "g_b": _lin(a.g_b_proj.weight),
        "o_norm": _np(a.o_norm.weight), "o": _lin(a.o_proj.weight),
    }


def convert_mla(a):
    p = {
        "q_a": _lin(a.q_a_proj.weight), "q_a_norm": _np(a.q_a_layernorm.weight), "q_b": _lin(a.q_b_proj.weight),
        "kv_a": _lin(a.kv_a_proj_with_mqa.weight), "kv_a_norm": _np(a.kv_a_layernorm.weight),
        "kv_b": _lin(a.kv_b_proj.weight), "o": _lin(a.o_proj.weight),
    }
    if a.indexer is not None:
        ix = a.indexer
        p["indexer"] = {
            "wq_b": _lin(ix.wq_b.weight), "wk": _lin(ix.wk.weight),
            "k_norm_w": _np(ix.k_norm.weight), "k_norm_b": _np(ix.k_norm.bias),
            "weights_proj": _lin(ix.weights_proj.weight),
            "ape": _np(ix.index_kpool_compress_ape), "gate": _lin(ix.index_kpool_compress_gate),
        }
    return p


def convert_mlp(m):
    return {"gate": _lin(m.gate_proj.weight), "up": _lin(m.up_proj.weight), "down": _lin(m.down_proj.weight)}


def convert_moe(m):
    gu = _np(m.experts.gate_up_proj)                              # [E, 2mi, D]
    dn = _np(m.experts.down_proj)                                 # [E, D, mi]
    return {
        "router_w": _lin(m.gate.weight), "router_bias": _np(m.gate.e_score_correction_bias),
        "gate_up": np.ascontiguousarray(np.transpose(gu, (0, 2, 1))),   # [E, D, 2mi]
        "down": np.ascontiguousarray(np.transpose(dn, (0, 2, 1))),      # [E, mi, D]
        "shared": convert_mlp(m.shared_experts),
    }


def from_hf_module(model, lm_head=None):
    """`model`: transformers Glm5NextTextModel. Returns nested dict of numpy arrays."""
    layers = []
    for L in model.layers:
        p = {
            "ln1": _np(L.input_layernorm.weight), "ln2": _np(L.post_attention_layernorm.weight),
            "hc_attn": convert_hc(L.attn_hc), "hc_ffn": convert_hc(L.ffn_hc),
        }
        p["attn"] = convert_kda(L.self_attn) if L.block_type == "linear_attention" else convert_mla(L.self_attn)
        p["mlp"] = convert_moe(L.mlp) if hasattr(L.mlp, "experts") else convert_mlp(L.mlp)
        layers.append(p)
    params = {"embed": _np(model.embed_tokens.weight), "norm": _np(model.norm.weight), "layers": layers}
    if lm_head is not None:
        params["lm_head"] = _lin(lm_head.weight)
    return params


def dequant_fp8(w_fp8: np.ndarray, scale_inv: np.ndarray, block: int = 128) -> np.ndarray:
    """fp8 e4m3 weight [O, I] (as float32/np) with per-block scales [ceil(O/b), ceil(I/b)] -> float32 [O, I]."""
    O, I = w_fp8.shape
    so, si = scale_inv.shape
    s = np.repeat(np.repeat(scale_inv.astype(np.float32), block, axis=0), block, axis=1)[:O, :I]
    assert s.shape == (O, I), (s.shape, O, I, so, si)
    return w_fp8.astype(np.float32) * s
