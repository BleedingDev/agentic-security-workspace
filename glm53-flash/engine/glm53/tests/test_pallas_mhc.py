"""The mHC site kernels (glm53.pallas_mhc, interpret mode on CPU) against model.hc_post / hc_pre / rmsnorm."""
import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from glm53 import model as M  # noqa: E402
from glm53 import pallas_mhc as PM  # noqa: E402


@dataclasses.dataclass(frozen=True)
class _Cfg:
    hc: int = 4
    hc_iters: int = 20
    hc_eps: float = 1e-6
    eps: float = 1e-6


def _site(rng, B, T, D, dtype, hc=4):
    M_ = (2 + hc) * hc
    streams = jnp.asarray(rng.standard_normal((B, T, hc, D)), dtype)
    y = jnp.asarray(rng.standard_normal((B, T, D)), dtype)
    p = {"fn": jnp.asarray(rng.standard_normal((hc * D, M_)) * 0.02, dtype),
         "base": jnp.asarray(rng.standard_normal(M_) * 0.5, jnp.float32),
         "scale": jnp.asarray(rng.uniform(0.5, 2.0, 3), jnp.float32)}
    ln = jnp.asarray(rng.uniform(0.5, 1.5, D), dtype)
    post = jnp.asarray(2.0 / (1 + np.exp(-rng.standard_normal((B, T, hc)))), jnp.float32)
    comb = jnp.asarray(rng.uniform(0.0, 0.5, (B, T, hc, hc)), jnp.float32)
    return streams, y, p, ln, post, comb


def _ref_pre(streams, p, ln, cfg):
    post, comb, h = M.hc_pre(p, streams, cfg)
    return post, comb, M.rmsnorm(h, ln, cfg.eps)


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)


@pytest.mark.parametrize("B,T,D", [(1, 1, 256), (1, 4, 256), (3, 4, 512), (1, 1, 4096)])
def test_pre_f32(B, T, D):
    cfg = _Cfg()
    streams, y, p, ln, post, comb = _site(np.random.default_rng(0), B, T, D, jnp.float32)
    po, co, h = PM.mhc_pre(streams, p, ln, cfg, prec="f32", interpret=True)
    rpo, rco, rh = _ref_pre(streams, p, ln, cfg)
    co = co.reshape(rco.shape)
    assert _rel(po, rpo) < 1e-5 and _rel(co, rco) < 1e-5 and _rel(h, rh) < 1e-5, (_rel(po, rpo), _rel(co, rco),
                                                                                    _rel(h, rh))


@pytest.mark.parametrize("B,T,D", [(1, 1, 256), (1, 4, 256), (3, 4, 512)])
def test_post_pre_f32(B, T, D):
    cfg = _Cfg()
    streams, y, p, ln, post, comb = _site(np.random.default_rng(1), B, T, D, jnp.float32)
    ns, po, co, h = PM.mhc_post_pre(post, comb.reshape(B, T, -1), y, streams, p, ln, cfg, prec="f32", interpret=True)
    rns = M.hc_post(post, comb, y, streams)
    rpo, rco, rh = _ref_pre(rns, p, ln, cfg)
    for got, ref in ((ns, rns), (po, rpo), (co.reshape(rco.shape), rco), (h, rh)):
        assert _rel(got, ref) < 1e-5, _rel(got, ref)


@pytest.mark.parametrize("prec", ["bf16", "f32"])
def test_post_pre_bf16_model(prec):
    """bf16 model dtype (the TPU's): streams and h to bf16 rounding of the reference; the residual update itself
    rounds where the reference rounds, so almost every element is identical."""
    cfg = _Cfg()
    streams, y, p, ln, post, comb = _site(np.random.default_rng(2), 1, 4, 512, jnp.bfloat16)
    ns, po, co, h = PM.mhc_post_pre(post, comb.reshape(1, 4, -1), y, streams, p, ln, cfg, prec=prec, interpret=True,
                                    opts=())                                           # model.py's rounding
    co = co.reshape(1, 4, 4, 4)
    rns = M.hc_post(post, comb, y, streams)
    rpo, rco, rh = _ref_pre(rns, p, ln, cfg)
    same = np.mean(np.asarray(ns, np.float32) == np.asarray(rns, np.float32))
    assert same > 0.99, same
    tol = 2e-2 if prec == "bf16" else 1e-2
    for got, ref in ((po, rpo), (co, rco), (h, rh)):
        assert _rel(got, ref) < tol, _rel(got, ref)


def test_fnT_param():
    cfg = _Cfg()
    streams, y, p, ln, post, comb = _site(np.random.default_rng(3), 1, 2, 256, jnp.float32)
    a = PM.mhc_pre(streams, p, ln, cfg, prec="f32", interpret=True)
    b = PM.mhc_pre(streams, {**p, "fnT": p["fn"].T}, ln, cfg, prec="f32", interpret=True)
    for x, z in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(z))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_post(dtype):
    streams, y, p, ln, post, comb = _site(np.random.default_rng(4), 3, 1, 256, dtype)
    ns = PM.mhc_post(post, comb.reshape(3, 1, -1), y, streams, interpret=True)
    rns = M.hc_post(post, comb, y, streams)
    if dtype == jnp.float32:
        assert _rel(ns, rns) < 1e-6, _rel(ns, rns)
    else:
        assert np.mean(np.asarray(ns, np.float32) == np.asarray(rns, np.float32)) > 0.99


@pytest.mark.parametrize("prec", ["f32", "bf16"])
def test_model_forward_switch(prec, monkeypatch):
    """The tiny model's forward with the mHC sites as kernels (model.MHC_KERNEL) = the XLA forward (f32 model dtype:
    "f32" to f32 rounding; "bf16" rounds the two small contractions' operands to bf16)."""
    torch = pytest.importorskip("torch")
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
    from glm53.hf_convert import from_hf_module
    from glm53.tests.test_tiny_vs_hf import tiny_config
    torch.manual_seed(0)
    c = tiny_config(index_topk=16)
    hf = Glm5NextTextModel(c).float().eval()
    params = jax.tree.map(jnp.asarray, from_hf_module(hf, torch.nn.Linear(c.hidden_size, c.vocab_size, bias=False)))
    cfg = M.Cfg.from_hf(c)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, c.vocab_size, size=(1, 6)))
    ref = np.asarray(M.logits(params, M.forward(params, ids, cfg, sparse="auto")[0][:, -1]))
    monkeypatch.setattr(M, "MHC_KERNEL", prec)
    monkeypatch.setattr(M, "MHC_INTERPRET", True)
    got = np.asarray(M.logits(params, M.forward(params, ids, cfg, sparse="auto")[0][:, -1]))
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < (1e-4 if prec == "f32" else 3e-2), rel


def test_engine_decode_rows_switch(monkeypatch):
    """Through the engine (shard_map over 8 CPU devices, resident experts): prefill (XLA mHC: pieces of 32 tokens)
    then batched decode of 2 streams with the mHC kernels = the same steps with XLA's mHC (f32, "f32" kernels)."""
    from glm53.tests.test_batched_rows import build_engine
    rng = np.random.default_rng(0)
    eng, V = build_engine(1, rng)
    ids = [rng.integers(0, V, size=(1, n)) for n in (9, 14)]

    def run():
        sets, toks, pos = [], [], []
        for x in ids:
            lg, cc, p = eng.prefill(x)
            sets.append(cc); toks.append(int(jnp.argmax(lg[0]))); pos.append(p)
        outs = []
        pd = eng.device_positions(pos)
        t = np.array(toks, np.int32)
        for _ in range(3):
            t, lg, sets, pd = eng.decode_rows(t, sets, pd, DeviceSampler([0.0, 0.0], [1.0, 1.0], seed=0))
            outs.append(np.asarray(lg))
        return outs

    from glm53.engine import DeviceSampler
    ref = run()
    monkeypatch.setattr(M, "MHC_KERNEL", "f32")
    monkeypatch.setattr(M, "MHC_INTERPRET", True)
    eng._progs.clear()
    got = run()
    for a, b in zip(got, ref):
        assert np.abs(a - b).max() / np.abs(b).max() < 1e-4


def test_default_rounding_close():
    """The default (OPTS: rounded once, as XLA's TPU fusions) stays within bf16 rounding of model.py's per-op rounding."""
    cfg = _Cfg()
    streams, y, p, ln, post, comb = _site(np.random.default_rng(5), 1, 4, 512, jnp.bfloat16)
    a = PM.mhc_post_pre(post, comb.reshape(1, 4, -1), y, streams, p, ln, cfg, interpret=True)
    b = PM.mhc_post_pre(post, comb.reshape(1, 4, -1), y, streams, p, ln, cfg, interpret=True, opts=())
    for x, z in zip(a, b):
        assert _rel(x, z) < 2e-2, _rel(x, z)
