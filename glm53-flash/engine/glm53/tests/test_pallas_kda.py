"""The KDA core kernel (glm53.pallas_kda, interpret mode on CPU) against model.kda_attention's XLA path: the recurrent
step of 1 and 4 tokens, the "kin" histories, lockstep rows and independent rows; then through the engine (prefill, plain
batched decode, one-stream DFlash with its rollback replay) against the XLA path."""
import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from glm53 import model as M  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


@dataclasses.dataclass(frozen=True)
class _Cfg:
    kda_heads: int = 2
    kda_hd: int = 128
    conv_k: int = 4
    gate_lb: float | None = None
    eps: float = 1e-5
    tp_reduce: object = None

    def reduce(self, y):
        return y


def _layer(rng, D, cfg):
    H, hd, K = cfg.kda_heads, cfg.kda_hd, cfg.conv_k
    C = H * hd

    def lin(i, o, s=1.0):
        return jnp.asarray(rng.standard_normal((i, o)) * s / i ** 0.5, jnp.float32)
    return {"q": lin(D, C), "k": lin(D, C), "v": lin(D, C), "o": lin(C, D),
            "conv_q": jnp.asarray(rng.standard_normal((C, K)) * 0.3, jnp.float32),
            "conv_k": jnp.asarray(rng.standard_normal((C, K)) * 0.3, jnp.float32),
            "conv_v": jnp.asarray(rng.standard_normal((C, K)) * 0.3, jnp.float32),
            "f_a": lin(D, 32), "f_b": lin(32, C, 3.0), "dt_bias": jnp.asarray(rng.standard_normal(C), jnp.float32),
            "A_log": jnp.asarray(rng.uniform(-2.0, 0.5, H), jnp.float32),
            "b": lin(D, H, 2.0), "g_a": lin(D, 32), "g_b": lin(32, C, 2.0),
            "o_norm": jnp.asarray(rng.uniform(0.5, 1.5, hd), jnp.float32)}


def _cache(rng, B, cfg):
    H, hd, K = cfg.kda_heads, cfg.kda_hd, cfg.conv_k
    return {"conv": jnp.asarray(rng.standard_normal((B, K - 1, 3 * H * hd)), jnp.float32),
            "state": jnp.asarray(rng.standard_normal((B, H, hd, hd)) * 0.1, jnp.float32)}


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)


def _run(p, x, cfg, cache, hist, kernel, monkeypatch):
    monkeypatch.setattr(M, "KDA_KERNEL", kernel)
    monkeypatch.setattr(M, "KDA_INTERPRET", True)
    return M.kda_attention(p, x, cfg, cache, True, None, hist)


@pytest.mark.parametrize("B,T,gate_lb", [(1, 1, None), (1, 4, None), (2, 3, None), (1, 4, -0.5)])
def test_step_matches_xla(B, T, gate_lb, monkeypatch):
    cfg = _Cfg(gate_lb=gate_lb)
    rng = np.random.default_rng(0)
    D = 64
    p, cache = _layer(rng, D, cfg), _cache(rng, B, cfg)
    x = jnp.asarray(rng.standard_normal((B, T, D)), jnp.float32)
    y, nc = _run(p, x, cfg, cache, False, True, monkeypatch)
    ry, rnc = _run(p, x, cfg, cache, False, False, monkeypatch)
    assert _rel(y, ry) < 1e-5, _rel(y, ry)
    assert _rel(nc["state"], rnc["state"]) < 1e-5 and _rel(nc["conv"], rnc["conv"]) < 1e-6


@pytest.mark.parametrize("T", [1, 4])
def test_kin_histories_match_xla(T, monkeypatch):
    """hist="kin": the kernel's update inputs and conv histories equal the XLA path's and the state stays the
    pre-block one; replaying every token (model.kda_replay) gives the plain step's state."""
    cfg = _Cfg()
    rng = np.random.default_rng(1)
    D = 64
    p, cache = _layer(rng, D, cfg), _cache(rng, 1, cfg)
    x = jnp.asarray(rng.standard_normal((1, T, D)), jnp.float32)
    y, nc = _run(p, x, cfg, cache, "kin", True, monkeypatch)
    ry, rnc = _run(p, x, cfg, cache, "kin", False, monkeypatch)
    assert _rel(y, ry) < 1e-5
    for k in ("kin_hist", "conv_hist", "state", "conv"):
        assert _rel(nc[k], rnc[k]) < 1e-5, (k, _rel(nc[k], rnc[k]))
    assert nc["kin_hist"].shape == (T, 1, cfg.kda_heads, 4, cfg.kda_hd) and nc["conv_hist"].shape[0] == T + 1
    np.testing.assert_array_equal(np.asarray(nc["state"]), np.asarray(cache["state"]))
    _, plain = _run(p, x, cfg, cache, False, False, monkeypatch)
    replayed = M.kda_replay(cache["state"], nc["kin_hist"], T)
    assert _rel(replayed, plain["state"]) < 1e-5


def test_rows_match_xla(monkeypatch):
    """Independent rows (kda_attention_rows): every row's output and cache equal the XLA path's."""
    cfg = _Cfg()
    rng = np.random.default_rng(2)
    D, B, T = 64, 3, 4
    p = _layer(rng, D, cfg)
    caches = [_cache(rng, 1, cfg) for _ in range(B)]
    x = jnp.asarray(rng.standard_normal((B, T, D)), jnp.float32)
    monkeypatch.setattr(M, "KDA_INTERPRET", True)
    monkeypatch.setattr(M, "KDA_KERNEL", True)
    y, new = M.kda_attention_rows(p, x, cfg, caches, hist="kin")
    monkeypatch.setattr(M, "KDA_KERNEL", False)
    ry, rnew = M.kda_attention_rows(p, x, cfg, caches, hist="kin")
    assert _rel(y, ry) < 1e-5
    for a, b in zip(new, rnew):
        for k in a:
            assert _rel(a[k], b[k]) < 1e-5, k


def test_engine_paths_switch(monkeypatch):
    """Through the engine (8 CPU devices, resident experts, the tiny model): plain batched decode of 2 streams and a
    one-stream DFlash run (verify with kin histories, the deferred rollback's replay) with the KDA kernel = the same
    steps with XLA's KDA (f32 model dtype)."""
    from glm53.engine import DeviceSampler
    from glm53.tests.test_dflash_engine import random_drafter
    from glm53.tests.test_spec_decode import build, greedy
    eng, cfg, rng = build(False, mtp=False)
    dcfg, dp = random_drafter(rng, cfg.hidden_size, cfg.vocab_size)
    eng.set_dflash(dp, dcfg)
    ids = rng.integers(0, cfg.vocab_size - 1, size=(1, 45))
    seqs = [ids[:, :45], ids[:, 5:38]]

    def rows():
        sets, toks, pos = [], [], []
        for x in seqs:
            lg, cc, p = eng.prefill(x)
            sets.append(cc); toks.append(int(jnp.argmax(lg[0]))); pos.append(p)
        outs = []
        pd = eng.device_positions(pos)
        t = np.array(toks, np.int32)
        for _ in range(3):
            t, lg, sets, pd = eng.decode_rows(t, sets, pd, DeviceSampler([0.0, 0.0], [1.0, 1.0], seed=0))
            outs.append(np.asarray(lg))
        return outs

    def dflash(n=12):
        logits, caches, pos = eng.prefill(ids)
        out = [int(jnp.argmax(logits[0]))]
        st = eng.dflash_start(np.array([out[0]]), caches, pos, T=4)
        while len(out) < n:
            emitted, st = eng.dflash_step(st)
            out.extend(emitted)
        eng.dflash_end()
        return out
    ref_rows, ref_greedy = rows(), greedy(eng, ids, 14)
    ref_df = dflash()
    assert ref_df[:12] == ref_greedy[:12]
    monkeypatch.setattr(M, "KDA_KERNEL", True)
    monkeypatch.setattr(M, "KDA_INTERPRET", True)
    eng._progs.clear()
    got = rows()
    for a, b in zip(got, ref_rows):
        assert _rel(a, b) < 1e-4, _rel(a, b)
    assert dflash()[:12] == ref_greedy[:12]
