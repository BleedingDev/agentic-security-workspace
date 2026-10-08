"""DFlash 2 in the engine (ResidentLayerEngine, tiny model, 8 CPU devices, interpret-mode Pallas, a random tiny
drafter whose target layers sit in the middle and at the end of the 3-layer decode groups): the features the verify
captures equal the prefill path's; greedy DFlash decoding reproduces plain greedy decoding token for token (random
drafts: the rollback path; oracle drafts with errors: partial acceptance, several features per ring write; blocks
of 8 and 4; the acceptance on the chips, the next draft queued before the host reads the tokens); the ring after
those steps equals the ring a fresh prefill of the same committed tokens builds; the ring survives a compact snapshot
round trip."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from glm53 import dflash as D  # noqa: E402
from glm53.tests.test_spec_decode import build, greedy  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def random_drafter(rng, hidden, vocab, layers=(1, 2, 4)):
    cfg = D.DCfg(hidden=hidden, heads=16, kv_heads=8, hd=16, inter=256, n_layers=2, eps=1e-5, theta=10000.0, window=16,
                 block=8, mask_id=vocab - 1, target_layers=layers, vocab=vocab, sel_rank=8, sel_topk=4, conv_k=2,
                 conv_group=16)

    def lin(i, o):
        return (rng.standard_normal((i, o)) / i ** 0.5).astype(np.float32)

    def norm(n):
        return (1 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    def conv():
        return {"base": (0.5 * rng.standard_normal((2, 2, hidden))).astype(np.float32),
                "proj": lin(hidden, 4 * hidden // cfg.conv_group)}
    qd, kvd = cfg.heads * cfg.hd, cfg.kv_heads * cfg.hd
    layer = lambda: {"ln1": norm(hidden), "ln2": norm(hidden), "q": lin(hidden, qd), "k": lin(hidden, kvd),  # noqa: E731
                     "v": lin(hidden, kvd), "o": lin(qd, hidden), "q_norm": norm(cfg.hd), "k_norm": norm(cfg.hd),
                     "gate": lin(hidden, cfg.inter), "up": lin(hidden, cfg.inter), "down": lin(cfg.inter, hidden),
                     "conv_a": conv(), "conv_m": conv()}
    params = {"fc": lin(len(layers) * hidden, hidden), "hidden_norm": norm(hidden), "norm": norm(hidden),
              "layers": [layer() for _ in range(cfg.n_layers)],
              "sel": {"pred": (0.5 * rng.standard_normal((vocab, 8))).astype(np.float32),
                      "succ": (0.5 * rng.standard_normal((vocab, 8))).astype(np.float32), "h": lin(hidden, 8)}}
    return cfg, params


@pytest.fixture(scope="module")
def setup():
    eng, cfg, rng = build(False, mtp=False)
    dcfg, dp = random_drafter(rng, cfg.hidden_size, cfg.vocab_size)
    eng.set_dflash(dp, dcfg)
    ids = rng.integers(0, cfg.vocab_size - 1, size=(1, 45))
    return eng, cfg, ids, greedy(eng, ids, 14)


def run(eng, ids, n, T=None, oracle=None):
    logits, caches, pos = eng.prefill(ids)
    out = [int(jnp.argmax(logits[0]))]
    st = eng.dflash_start(np.array([out[0]]), caches, pos, T=T)
    accepted, T = 0, st["T"]
    while len(out) < n:
        ov = None if oracle is None else (list(oracle[len(out):len(out) + T - 1]) + [0] * T)[:T - 1]
        emitted, st = eng.dflash_step(st, draft_override=ov)
        accepted += len(emitted) - 1
        out.extend(emitted)
    return out, accepted, st


def test_capture_matches_prefill_path(setup):
    """Layers 1 and 4 are in the middle of the decode groups [0,1,2], [3,4,5] (the program returns their stream mean),
    layer 2 ends one (the mean of the group's output); prefill runs one layer per program."""
    eng, cfg, ids, _ = setup
    n = eng.cfg.n_layers
    _, caches, pos = eng.prefill(ids[:, :40])
    stacked, fidx = eng._run_verify(jnp.asarray(ids[:, 40:45]), eng.copy_caches(caches[:n]), 40, capture=True)[5]
    ver = [f[j] for f, js in zip(stacked, fidx) for j in js]          # per group stacked -> per target layer
    padded = np.zeros((1, 32), np.int32)
    padded[:, :5] = ids[:, 40:45]
    pre = eng._run(padded, eng.copy_caches(caches[:n]), 40, False, length=5, want_logits=False, capture=True)[2]
    assert len(ver) == len(pre) == 3
    for a, b in zip(ver, pre):                     # the verify's features are padded to the drafter's block (8)
        a, b = np.asarray(a), np.asarray(b)[:, :5]
        assert a.shape[1] == eng.dflash.cfg.block and not a[:, 5:].any()
        np.testing.assert_allclose(a[:, :5], b, rtol=1e-4, atol=1e-4 * np.abs(b).max())
    assert not np.allclose(np.asarray(ver[0]), np.asarray(ver[1]), rtol=0.1)


def test_dflash_matches_greedy(setup):
    eng, cfg, ids, ref = setup
    for T in (8, 4):
        out, acc, _ = run(eng, ids, 14, T)                                         # random drafts: rollbacks
        assert out[:14] == ref, (T, out, ref)
        oracle = [t if i % 5 != 3 else (t + 1) % (cfg.vocab_size - 1) for i, t in enumerate(ref)]
        out, acc, st = run(eng, ids, 14, T, oracle=oracle)                          # partial acceptance
        assert out[:14] == ref and acc > 0, (T, out, ref, acc)
    # after a step the ring holds every committed position (the queued draft wrote the accepted ones) = the ring a
    # fresh prefill of the committed tokens builds; the device position = the host's
    n = eng.cfg.n_layers
    p = st["pos_h"]
    assert int(np.asarray(st["pos"])) == p
    toks = np.concatenate([ids[0], np.asarray(out[:p - ids.shape[1]])])[None]
    _, fresh, _ = eng.prefill(toks)
    np.testing.assert_array_equal(np.asarray(st["caches"][n]["p"]), np.asarray(fresh[n]["p"]))
    for k in ("k", "v"):
        a, b = np.asarray(st["caches"][n][k]), np.asarray(fresh[n][k])
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4 * np.abs(b).max())


def test_ring_survives_snapshot(setup):
    eng, cfg, ids, _ = setup
    logits, caches, pos = eng.prefill(ids[:, :40])
    snap = eng.snapshot_to_host(eng.snapshot_prefix(caches, pos, rows_bucket=16))
    restored = eng.restore_prefix(eng.snapshot_from_host(snap))
    l1, c1, p1 = eng.prefill(ids[:, 40:], caches, pos)
    l2, c2, p2 = eng.prefill(ids[:, 40:], restored, pos)
    e1, s1 = eng.dflash_step(eng.dflash_start(np.array([int(jnp.argmax(l1[0]))]), c1, p1))
    e2, s2 = eng.dflash_step(eng.dflash_start(np.array([int(jnp.argmax(l2[0]))]), c2, p2))
    assert e1 == e2
    n = eng.cfg.n_layers
    np.testing.assert_allclose(np.asarray(s1["caches"][n]["k"]), np.asarray(s2["caches"][n]["k"]), rtol=1e-5, atol=1e-6)


def test_stop_token_ends_the_run(setup):
    """A drafted stop token ends the accepted run on the chips and is emitted unfed: with oracle drafts the output
    stops exactly at greedy's first stop token, and the position counts only the tokens before it as fed."""
    eng, cfg, ids, ref = setup
    stop = ref[6]
    first = ref.index(stop)
    logits, caches, pos = eng.prefill(ids)
    out = [int(jnp.argmax(logits[0]))]
    st = eng.dflash_start(np.array([out[0]]), caches, pos)
    while stop not in out:
        T = st["T"]
        emitted, st = eng.dflash_step(st, stop_ids=(stop,),
                                      draft_override=(list(ref[len(out):len(out) + T - 1]) + [0] * T)[:T - 1])
        out.extend(emitted)
    assert out == ref[:first + 1], (out, ref)
    assert st["pos_h"] == ids.shape[1] + first and int(np.asarray(st["pos"])) == st["pos_h"]


def small_entries(cache, pos, kp):
    """The small state entries a rollback restores: KDA state / conv whole, the pool tails' committed rows only (rows
    from pos % kp on hold stale keys that `pool_new_keys` masks: don't-care)."""
    out = {k: np.asarray(cache[k]) for k in ("state", "conv") if k in cache}
    if pos % kp:
        out.update({k: np.asarray(cache[k])[:, :pos % kp] for k in ("tk", "tg") if k in cache})
    return out


def test_run_ahead_undone_when_the_stream_stops(setup):
    """With `run_ahead` every step dispatches the next block's verify before the host reads this block's tokens; when
    the stream leaves DFlash that verify is undone (`dflash_settle`: a rollback to n_keep = 0). The
    settled caches' small state entries (KDA states / conv states, pool tails) equal a fresh prefill's of the committed
    tokens, the device position = the host's, and plain greedy decoding continues exactly as the reference."""
    eng, cfg, ids, _ = setup
    ref = greedy(eng, ids, 20)
    eng.run_ahead = True
    try:
        out, _, st = run(eng, ids, 8, T=4)                  # random drafts, no override: the verify ahead stays
    finally:
        del eng.run_ahead                                   # (back to the class default)
    assert st.get("ahead") is not None and "pend" not in st
    st = eng.dflash_settle(st)
    eng.dflash_end()
    assert st.get("ahead") is None and st["pend"] is None
    assert int(np.asarray(st["pos"])) == st["pos_h"]
    fed = out[:st["pos_h"] - ids.shape[1]]
    assert fed == out[:-1]
    _, fresh, _ = eng.prefill(np.concatenate([ids[0], np.asarray(fed)])[None])
    kp = eng.cfg.idx_kpool
    for i in range(eng.cfg.n_layers):
        for k, b in small_entries(fresh[i], st["pos_h"], kp).items():
            a = small_entries(st["caches"][i], st["pos_h"], kp)[k]
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3 * max(np.abs(b).max(), 1e-6), err_msg=f"layer {i} {k}")
    caches, pos, tok = st["caches"], st["pos_h"], out[-1]
    for _ in range(4):
        lg, caches, pos = eng.decode(np.array([tok]), caches, pos)
        tok = int(jnp.argmax(lg[0])); out.append(tok)
    assert out == ref[:len(out)], (out, ref)
    assert eng.dflash_settle(st) is st or eng.dflash_settle(st)["pend"] is None     # idempotent


def test_decode_rows_fills_the_ring(setup):
    """Batched plain steps (decode_rows, two streams at different positions) write every stream's positions into its
    own drafter ring: afterwards each ring equals the ring a fresh prefill of that stream's tokens builds (no holes
    for a stream that goes back to DFlash)."""
    eng, cfg, ids, _ = setup
    n = eng.cfg.n_layers
    seqs = [list(ids[0, :45]), list(ids[0, 5:38])]
    toks, sets, pos = [], [], []
    for s in seqs:
        lg, cc, p = eng.prefill(np.asarray(s)[None])
        toks.append(int(jnp.argmax(lg[0]))); sets.append(cc); pos.append(p)
    for _ in range(6):
        _, logits, sets, _ = eng.decode_rows(np.array(toks), sets, pos)
        for r in range(2):
            seqs[r].append(toks[r])
        toks = [int(x) for x in np.asarray(jnp.argmax(logits, -1))]
        pos = [p + 1 for p in pos]
    for r in range(2):
        _, fresh, _ = eng.prefill(np.asarray(seqs[r])[None])
        np.testing.assert_array_equal(np.asarray(sets[r][n]["p"]), np.asarray(fresh[n]["p"]))
        for k in ("k", "v"):
            a, b = np.asarray(sets[r][n][k]), np.asarray(fresh[n][k])
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3 * np.abs(b).max())


def run_rows(eng, prompts, n, T=4, oracles=None, stop=()):
    """DFlash for all prompts at once (`dflash_start_rows` / `dflash_step_rows`) -> (tokens per row, last state)."""
    toks, sets, pos = [], [], []
    for p in prompts:
        lg, cc, ps = eng.prefill(np.asarray(p)[None])
        toks.append(int(jnp.argmax(lg[0]))); sets.append(cc); pos.append(ps)
    outs = [[t] for t in toks]
    st = eng.dflash_start_rows(np.array(toks), sets, pos, T=T)
    while min(len(o) for o in outs) < n:
        ov = None if oracles is None else [(list(o_[len(o):len(o) + T - 1]) + [0] * T)[:T - 1]
                                           for o_, o in zip(oracles, outs)]
        em, st = eng.dflash_step_rows(st, stop_ids=stop, draft_override=ov)
        for o, e in zip(outs, em):
            assert not any(t in stop for t in e[:-1]), (e, stop)       # a stop token is only ever the last one
            o.extend(e)
    eng.dflash_end()
    return outs, st


def test_dflash_rows_match_greedy(setup):
    """Three streams at different positions in one batched DFlash step: every row emits exactly its own greedy
    tokens (random drafts: per-row rollbacks; oracle drafts with errors at different places per row: different
    accepted lengths in the same step), each row's device position = the host's, and each row's ring = the ring a
    fresh prefill of that stream's committed tokens builds."""
    eng, cfg, ids, _ = setup
    n = eng.cfg.n_layers
    prompts = [list(ids[0, :45]), list(ids[0, 5:38]), list(ids[0, 12:30])]
    refs = [greedy(eng, np.asarray(p)[None], 14) for p in prompts]
    for T in (4, 8):
        outs, _ = run_rows(eng, prompts, 14, T)
        assert [o[:14] for o in outs] == refs, (T, outs, refs)
        oracles = [[t if i % (3 + r) != 2 else (t + 1) % (cfg.vocab_size - 1) for i, t in enumerate(ref)]
                   for r, ref in enumerate(refs)]
        outs, st = run_rows(eng, prompts, 14, T, oracles=oracles)
        assert [o[:14] for o in outs] == refs, (T, outs, refs)
    assert [int(x) for x in np.asarray(st["pos"])] == st["pos_h"]
    for r, p in enumerate(prompts):
        committed = st["pos_h"][r] - len(p)
        _, fresh, _ = eng.prefill(np.asarray(p + outs[r][:committed])[None])
        np.testing.assert_array_equal(np.asarray(st["rings"][r]["p"]), np.asarray(fresh[n]["p"]))
        for k in ("k", "v"):
            a, b = np.asarray(st["rings"][r][k]), np.asarray(fresh[n][k])
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4 * np.abs(b).max())


def test_dflash_rows_stop_token(setup):
    """A drafted stop token ends that row's accepted run (it is emitted last, unfed: `run_rows` checks every step);
    the row goes on from it next step like plain decoding, and the other rows are not cut."""
    eng, cfg, ids, _ = setup
    prompts = [list(ids[0, :45]), list(ids[0, 5:38])]
    refs = [greedy(eng, np.asarray(p)[None], 14) for p in prompts]
    outs, st = run_rows(eng, prompts, 14, 4, oracles=refs, stop=(refs[0][5],))
    assert [o[:14] for o in outs] == refs, (outs, refs)


def test_dflash_rows_run_ahead_undone(setup):
    """`test_run_ahead_undone_when_the_stream_stops` for the batched state: the verify ahead of two streams is undone by
    `dflash_settle_rows`, every row's small state entries equal a fresh prefill's of its committed tokens, and plain
    batched decoding (`decode_rows`) continues every row exactly as its own greedy reference."""
    eng, cfg, ids, _ = setup
    prompts = [list(ids[0, :45]), list(ids[0, 5:38])]
    refs = [greedy(eng, np.asarray(p)[None], 20) for p in prompts]
    eng.run_ahead = True
    try:
        outs, st = run_rows(eng, prompts, 8, 4)              # random drafts: the verify ahead stays
    finally:
        del eng.run_ahead
    assert st.get("ahead") is not None and "pend" not in st
    st = eng.dflash_settle_rows(st)
    assert st.get("ahead") is None and st["pend"] is None
    assert [int(x) for x in np.asarray(st["pos"])] == st["pos_h"]
    for b, p in enumerate(prompts):
        fed = outs[b][:st["pos_h"][b] - len(p)]
        assert fed == outs[b][:-1]
        _, fresh, _ = eng.prefill(np.asarray(p + fed)[None])
        for i in range(eng.cfg.n_layers):
            for k, ref in small_entries(fresh[i], st["pos_h"][b], eng.cfg.idx_kpool).items():
                a = small_entries(st["sets"][b][i], st["pos_h"][b], eng.cfg.idx_kpool)[k]
                np.testing.assert_allclose(a, ref, rtol=1e-3, atol=1e-3 * max(np.abs(ref).max(), 1e-6), err_msg=f"row {b} layer {i} {k}")
    sets = [st["sets"][b] + [st["rings"][b]] for b in range(2)]
    toks, pos = np.array([o[-1] for o in outs], np.int32), list(st["pos_h"])
    for _ in range(4):
        _, logits, sets, _ = eng.decode_rows(toks, sets, pos)
        toks = np.asarray(jnp.argmax(logits, -1)).astype(np.int32)
        for b in range(2):
            outs[b].append(int(toks[b]))
        pos = [p + 1 for p in pos]
    assert all(outs[b] == refs[b][:len(outs[b])] for b in range(2)), (outs, refs)


def test_dflash_rows_seq_shard():
    """The served layout (latent cache sharded over the chips by position: the attention's collectives per row):
    three streams in one batched DFlash step still emit exactly their own greedy tokens."""
    eng, cfg, rng = build(True, mtp=False)
    dcfg, dp = random_drafter(rng, cfg.hidden_size, cfg.vocab_size)
    eng.set_dflash(dp, dcfg)
    ids = rng.integers(0, cfg.vocab_size - 1, size=(1, 45))
    prompts = [list(ids[0, :45]), list(ids[0, 5:38]), list(ids[0, 12:30])]
    refs = [greedy(eng, np.asarray(p)[None], 12) for p in prompts]
    outs, _ = run_rows(eng, prompts, 12, 4)
    assert [o[:12] for o in outs] == refs, (outs, refs)
    oracles = [[t if i % (3 + r) != 2 else (t + 1) % (cfg.vocab_size - 1) for i, t in enumerate(ref)]
               for r, ref in enumerate(refs)]
    outs, _ = run_rows(eng, prompts, 12, 4, oracles=oracles)
    assert [o[:12] for o in outs] == refs, (outs, refs)
