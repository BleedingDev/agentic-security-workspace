"""DFlash 2 drafter (glm53.dflash) vs the PyTorch reference (z-lab/dflash `dflash/model.py`, vendored as
tests/dflash_ref.py), CPU f32: a tiny random drafter on 1 and 8 devices over draft steps that wrap the sliding-window
ring (greedy and sampled selector, two streams at different positions); the real incoai/GLM-5.3-Flash-DFlash2
weights when GLM53_DFLASH_DIR points to a download of it (config.json + model.safetensors)."""
import dataclasses
import json
import os

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from glm53 import dflash as D  # noqa: E402
from glm53.engine import AXIS  # noqa: E402
from glm53.tests import dflash_ref as REF  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")

CODEBOOKS = ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook")


def tiny_config():
    return dict(architectures=["DFlash2DraftModel"], model_type="qwen3", hidden_size=64, num_attention_heads=16,
                num_key_value_heads=8, head_dim=8, intermediate_size=128, num_hidden_layers=2, rms_norm_eps=1e-5,
                vocab_size=512, use_sliding_window=True, sliding_window=16, max_window_layers=2, is_causal=False,
                layer_types=["sliding_attention"] * 2, max_position_embeddings=4096, attention_bias=False,
                rope_parameters={"rope_theta": 10000.0, "rope_type": "default"}, num_target_layers=6,
                tie_word_embeddings=False,
                dflash_config={"block_size": 8, "mask_token_id": 500, "target_layer_ids": [1, 3, 4],
                               "selector_rank": 8, "selector_top_k": 4, "conv_kernel_size": 2, "conv_group_size": 4})


def ref_model(c):
    from transformers import Qwen3Config
    return REF.DFlash2DraftModel(Qwen3Config(**c)).float().eval()


def ref_steps(m, E, W, steps, temperature=0.0, paths=None):
    """The draft half of `dflash_generate` per step: steps = [(feats [n, nF*D] of the n newly committed positions,
    anchor id)]; the drafter's cache is cropped back to `start` after each step. With `paths` (one per step) the
    selector scores are taken along those paths instead of the reference's own choice (to check sampled q rows).
    -> per step (hidden [T, D], candidates [T-1, k], path [T-1], scores [T-1, k])."""
    T = m.block_size
    head = torch.nn.Linear(W.shape[1], W.shape[0], bias=False)
    head.weight = torch.nn.Parameter(W, requires_grad=False)
    sel = m.candidate_selector
    cache, start, out = REF._make_cache(m.config), 0, []
    for si, (feats, anchor) in enumerate(steps):
        n = feats.shape[0]
        start += n
        ids = torch.tensor([[anchor] + [m.mask_token_id] * (T - 1)])
        with torch.no_grad():
            h = m(target_hidden=torch.as_tensor(feats)[None], noise_embedding=torch.nn.functional.embedding(ids, E),
                  position_ids=torch.arange(start - n, start + T)[None], past_key_values=cache, use_cache=True)
            REF._crop_to(cache, start)
            hid = h[:, 1:]
            logits = m.compute_logits(hid, head)
            path, cand, _ = sel.select(hid, logits, ids[:, 0], temperature)
            if paths is not None:
                path = torch.as_tensor(np.asarray(paths[si]))[None]
            unary = logits.gather(-1, cand)
            hp = sel.hidden_projection(hid)
            pred, scores = ids[:, 0], []
            for t in range(T - 1):
                scores.append(unary[:, t] + torch.einsum("br,bkr->bk", sel.predecessor_codebook(pred) * hp[:, t],
                                                         sel.successor_codebook(cand[:, t])))
                pred = path[:, t]
        out.append((h[0].numpy(), cand[0].numpy(), path[0].numpy(), torch.stack(scores, 1)[0].numpy()))
    return out


def by_id(cand, scores):
    """Candidates and their scores ordered by token id (the reference's top-k is unsorted)."""
    o = np.argsort(cand, -1)
    return np.take_along_axis(cand, o, -1), np.take_along_axis(scores, o, -1)


def check_step(mine, ref, tol=2e-4, hidden=None):
    cand, path, scores = mine
    h_ref, cand_ref, path_ref, scores_ref = ref
    c0, s0 = by_id(np.asarray(cand), np.asarray(scores))
    c1, s1 = by_id(cand_ref, scores_ref)
    np.testing.assert_array_equal(c0, c1)
    np.testing.assert_allclose(s0, s1, rtol=tol, atol=tol * np.abs(s1).max())
    np.testing.assert_array_equal(np.asarray(path), path_ref)
    if hidden is not None:
        np.testing.assert_allclose(np.asarray(hidden), h_ref, rtol=tol, atol=tol * np.abs(h_ref).max())


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A random tiny DFlash 2 drafter saved like the HF checkpoint, loaded back with glm53.dflash.load_params."""
    c = tiny_config()
    torch.manual_seed(0)
    m = ref_model(c)
    with torch.no_grad():
        for name, p in m.named_parameters():         # base_kernel is torch.empty in the reference: set everything
            if name.endswith("norm.weight") or "layernorm" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif p.ndim == 2 and "codebook" not in name:
                p.copy_(torch.randn_like(p) / p.shape[1] ** 0.5)
            else:
                p.copy_(0.5 * torch.randn_like(p))
    d = tmp_path_factory.mktemp("dflash_tiny")
    from safetensors.torch import save_file
    sd = {(k[:-len(".weight")] if k[:-len(".weight")] in CODEBOOKS else k): v.contiguous()
          for k, v in m.state_dict().items()}
    save_file(sd, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps(c))
    cfg, params = D.load_params(str(d))
    V, Dm = c["vocab_size"], c["hidden_size"]
    E = torch.randn(V, Dm)                                  # the target's embedding and lm_head (random here)
    W = torch.randn(V, Dm) / Dm ** 0.5                     # [V, D]: the layout the engine stores the lm_head in
    return m, cfg, params, E, W


def make_steps(rng, cfg, ns, prompt):
    F = len(cfg.target_layers) * cfg.hidden
    return [(rng.standard_normal((n, F)).astype(np.float32), int(rng.integers(0, cfg.mask_id)))
            for n in [prompt] + list(ns)]


def run_mine(drafter, cfg, embed, lm_head, streams, temperature=None, key=None):
    """streams: per stream a step list (all the same length). The prompt step goes through `ingest`, later steps
    through `draft`'s fused ingest (n_new rows padded to the block). -> per step (cand, path, scores[, q]) [B, ...]."""
    B, T, F = len(streams), cfg.block, len(cfg.target_layers) * cfg.hidden
    ring = drafter.alloc_ring(B)
    n0 = max(s[0][0].shape[0] for s in streams)
    pre = np.zeros((B, n0, F), np.float32)
    for b, s in enumerate(streams):
        pre[b, :s[0][0].shape[0]] = s[0][0]
    ring = drafter.ingest(ring, jnp.asarray(pre), np.zeros(B), [s[0][0].shape[0] for s in streams])
    start = np.array([s[0][0].shape[0] for s in streams])
    out = []
    for i in range(len(streams[0])):
        feats = np.zeros((B, T, F), np.float32)
        n_new = np.zeros(B, np.int32)
        if i > 0:
            for b, s in enumerate(streams):
                n_new[b] = s[i][0].shape[0]
                feats[b, :n_new[b]] = s[i][0]
        anchor = [s[i][1] for s in streams]
        r = drafter.draft(ring, jnp.asarray(feats), start, n_new, anchor, embed, lm_head, temperature=temperature,
                          key=None if key is None else jax.random.fold_in(key, i))
        ring = r["ring"]
        out.append((r["cand"], r["path"], r["scores"]) + ((r["q"],) if "q" in r else ()))
        start = start + n_new
    return out


@pytest.mark.parametrize("n_dev", [1, 8])
def test_tiny_matches_reference(tiny, n_dev):
    """Two streams at different positions, steps of 1..8 new positions, the 16-slot ring wrapped several times."""
    m, cfg, params, E, W = tiny
    rng = np.random.default_rng(1)
    s0 = make_steps(rng, cfg, [3, 8, 1, 6, 8, 2, 8], prompt=21)
    s1 = make_steps(rng, cfg, [8, 8, 5, 1, 4, 8, 7], prompt=6)
    drafter = D.Drafter(cfg, params, Mesh(np.array(jax.devices()[:n_dev]), (AXIS,)))
    mine = run_mine(drafter, cfg, jnp.asarray(E.numpy()), jnp.asarray(W.numpy()), [s0, s1])
    for b, s in enumerate((s0, s1)):
        for i, ref in enumerate(ref_steps(m, E, W, s)):
            check_step(tuple(x[b] for x in mine[i][:3]), ref)


def test_tiny_hidden_and_sampled_q(tiny):
    """Single device, the module-level `draft` (returns the block's hidden) vs the reference's hidden; sampled
    selector: q rows = softmax(reference scores along the sampled path / temperature)."""
    m, cfg, params, E, W = tiny
    rng = np.random.default_rng(2)
    steps = make_steps(rng, cfg, [8, 2, 8, 5], prompt=19)
    p = jax.tree.map(jnp.asarray, params)
    embed, lm_head = jnp.asarray(E.numpy()), jnp.asarray(W.numpy())
    F = len(cfg.target_layers) * cfg.hidden
    ring = {"k": jnp.zeros((cfg.n_layers, 1, cfg.window, cfg.kv_heads, cfg.hd)),
            "v": jnp.zeros((cfg.n_layers, 1, cfg.window, cfg.kv_heads, cfg.hd)), "p": jnp.zeros((1, cfg.window), jnp.int32)}
    fn = jax.jit(lambda ring, f, p0, n, a: D.draft(p, ring, f, p0, n, a, embed, lm_head, cfg))
    start = 0
    refs = ref_steps(m, E, W, steps)
    for i, (feats, anchor) in enumerate(steps):
        n = feats.shape[0]
        f = np.zeros((1, max(n, cfg.block), F), np.float32)
        f[0, :n] = feats
        path, cand, _, scores, ring, h = fn(ring, jnp.asarray(f), jnp.array([start]), jnp.array([n]),
                                            jnp.array([anchor]))
        start += n
        check_step((cand[0], path[0], scores[0]), refs[i], hidden=h[0])
    drafter = D.Drafter(cfg, params, Mesh(np.array(jax.devices()), (AXIS,)))
    temp = 0.7
    mine = run_mine(drafter, cfg, embed, lm_head, [steps], temperature=temp, key=jax.random.PRNGKey(3))
    refs = ref_steps(m, E, W, steps, paths=[np.asarray(x[1][0]) for x in mine])
    for (cand, path, scores, q), ref in zip(mine, refs):
        c0, q0 = by_id(np.asarray(cand[0]), np.asarray(q[0]))
        c1, s1 = by_id(ref[1], ref[3])
        np.testing.assert_array_equal(c0, c1)
        z = s1 / temp
        q1 = np.exp(z - z.max(-1, keepdims=True))
        np.testing.assert_allclose(q0, q1 / q1.sum(-1, keepdims=True), rtol=1e-3, atol=1e-5)
        assert all(int(path[0, t]) in set(np.asarray(cand[0, t]).tolist()) for t in range(cfg.block - 1))


REAL = os.environ.get("GLM53_DFLASH_DIR")


@pytest.mark.skipif(not REAL, reason="GLM53_DFLASH_DIR (incoai/GLM-5.3-Flash-DFlash2 download) not set")
@pytest.mark.parametrize("window", [None, 24])
def test_real_weights_match_reference(window):
    """The real drafter, 8 devices (the TPU layout), f32 on both sides, random target features / embedding / lm_head;
    window=24 also shrinks the sliding window on both sides so the ring wraps with the real weights."""
    torch.manual_seed(0)
    cfg, params = D.load_params(REAL)
    with open(os.path.join(REAL, "config.json")) as f:
        c = json.load(f)
    if window:
        cfg = dataclasses.replace(cfg, window=window)
        c["sliding_window"] = window
    m = ref_model(c)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(REAL, "model.safetensors"))
    m.load_state_dict({(k + ".weight" if k in CODEBOOKS else k): v for k, v in sd.items()}, strict=True)
    del sd
    drafter = D.Drafter(cfg, params, Mesh(np.array(jax.devices()), (AXIS,)))
    del params
    E = torch.randn(cfg.vocab, cfg.hidden) * 0.02           # GLM-like embedding scale; reused as the lm_head
    En = E.numpy()
    rng = np.random.default_rng(4)
    steps = make_steps(rng, cfg, [8, 3, 8, 5], prompt=40)
    mine = run_mine(drafter, cfg, jnp.asarray(En), jnp.asarray(En), [steps])
    refs = ref_steps(m, E, E, steps)
    for i, ref in enumerate(refs):
        check_step(tuple(x[0] for x in mine[i][:3]), ref, tol=1e-3)


def test_tiny_int8_plumbing(tiny):
    """int8=True (q8 layers and fc, dequantized inside the programs) = the f32 drafter run on the same weights
    dequantized on the host."""
    from glm53 import quant8 as Q8
    m, cfg, params, E, W = tiny
    rng = np.random.default_rng(5)
    steps = make_steps(rng, cfg, [8, 3, 8], prompt=20)
    mesh = Mesh(np.array(jax.devices()), (AXIS,))
    deq = lambda t: jax.tree.map(lambda x: np.asarray(Q8.dequant_tree(x, jnp.float32)) if Q8.is_q8(x) else x,  # noqa: E731
                                 t, is_leaf=Q8.is_q8)
    q8 = D.Drafter(cfg, params, mesh, int8=True)
    f32 = D.Drafter(cfg, {**params, "layers": deq(jax.device_get(q8.params["layers"])),
                          "fc": deq(jax.device_get(q8.params["fc"]))}, mesh)
    embed, lm_head = jnp.asarray(E.numpy()), jnp.asarray(W.numpy())
    for a, b in zip(run_mine(q8, cfg, embed, lm_head, [steps]), run_mine(f32, cfg, embed, lm_head, [steps])):
        np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))
        np.testing.assert_allclose(np.asarray(a[2]), np.asarray(b[2]), rtol=1e-5, atol=1e-5)
