"""AOT export of the engine's programs (`glm53.aot`; tiny resident engine + a random tiny drafter, 8 CPU devices,
interpret-mode Pallas): a first engine traces, exports and runs every program through its export; a second engine of
the same build finds every file and traces nothing; both produce what the plain `jax.jit` engine produces (prefill
logits, batched plain decode, speculative steps for one stream and for two, the settle, a snapshot round trip). A
changed setting or program key is a different file."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flatbuffers")              # the serialization of an export needs it
import jax.numpy as jnp  # noqa: E402

from glm53 import aot  # noqa: E402
from glm53 import model as M  # noqa: E402
from glm53.engine import DeviceSampler  # noqa: E402
from glm53.tests.test_dflash_engine import random_drafter  # noqa: E402
from glm53.tests.test_spec_decode import build  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def make():
    eng, cfg, rng = build(False, mtp=False)
    dcfg, dp = random_drafter(rng, cfg.hidden_size, cfg.vocab_size)
    eng.set_dflash(dp, dcfg)
    return eng, rng.integers(0, cfg.vocab_size - 1, size=(1, 45))


def sequence(eng, ids):
    """Every kind of program the kernel's warm-up builds, on fixed inputs -> host values to compare."""
    out = {}
    logits, caches, pos = eng.prefill(ids)
    out["prefill"] = np.asarray(logits)
    first = int(jnp.argmax(logits[0]))
    # batched plain decode of two streams, sampled on the device at temperature 0
    sets = [caches, eng.prefill(ids[:, :30])[1]]
    samp = DeviceSampler([0.0, 0.0], [1.0, 1.0], seed=0)
    tok, p, rows = np.array([first, 3], np.int32), [pos, 30], []
    for _ in range(3):
        tok, lg, sets, p = eng.decode_rows(tok, sets, p, samp)
        rows.append(np.asarray(tok))
    out["rows"], out["rows_logits"] = np.stack(rows), np.asarray(lg)
    # one stream, speculative (greedy), then the stream's end
    _, caches, pos = eng.prefill(ids)
    st = eng.dflash_start(np.array([first]), caches, pos, T=4)
    em = []
    for _ in range(3):
        e, st = eng.dflash_step(st)
        em.append(e)
    st = eng.dflash_settle(st)
    eng.dflash_end()
    out["dflash"] = em
    out["state"] = np.asarray(st["caches"][0]["state"])
    # two streams, speculative, and their end
    sets = [eng.prefill(ids)[1], eng.prefill(ids[:, :30])[1]]
    st = eng.dflash_start_rows([first, 3], sets, [pos, 30], T=4)
    em = []
    for _ in range(2):
        e, st = eng.dflash_step_rows(st)
        em.append(e)
    st = eng.dflash_settle_rows(st)
    eng.dflash_end()
    out["dflash_rows"] = em
    # a compact snapshot and its restore
    _, caches, pos = eng.prefill(ids[:, :40])
    restored = eng.restore_prefix(eng.snapshot_from_host(eng.snapshot_to_host(eng.snapshot_prefix(caches, pos, rows_bucket=16))))
    out["restored"] = np.asarray(eng.prefill(ids[:, 40:], restored, pos)[0])
    return out


def same(a, b):
    assert a["dflash"] == b["dflash"] and a["dflash_rows"] == b["dflash_rows"], (a["dflash"], b["dflash"])
    np.testing.assert_array_equal(a["rows"], b["rows"])
    for k in ("prefill", "rows_logits", "state", "restored"):
        np.testing.assert_allclose(a[k], b[k], rtol=1e-5, atol=1e-5 * np.abs(b[k]).max())


@pytest.fixture()
def store(tmp_path):
    saved = dict(vars(aot.STORE))
    yield tmp_path / "exported"
    vars(aot.STORE).update(saved)


def test_export_then_load(store):
    eng, ids = make()
    ref = sequence(eng, ids)                                   # plain jax.jit programs
    n_progs = len(eng._progs) + len(eng.dflash._progs)

    eng, ids = make()
    st = aot.configure(store, context=lambda: aot.engine_context(eng), log=lambda *a: None)
    st.stats = dict.fromkeys(st.stats, 0)
    first = sequence(eng, ids)                                 # traced, exported, run through the export
    assert st.stats["failed"] == 0 and st.stats["traced"] == 0 and st.stats["loaded"] == 0, st.stats
    n_files = len(list(store.glob("*" + aot.EXT)))
    assert n_files == st.stats["exported"] >= n_progs, (n_files, st.stats, n_progs)
    assert len(eng._progs) + len(eng.dflash._progs) == n_progs
    same(first, ref)

    eng, ids = make()                                          # a new engine of the same build: nothing is traced
    st.stats = dict.fromkeys(st.stats, 0)
    second = sequence(eng, ids)
    assert st.stats["loaded"] == n_files and st.stats["exported"] == st.stats["traced"] == st.stats["failed"] == 0, st.stats
    assert len(list(store.glob("*" + aot.EXT))) == n_files
    same(second, ref)
    for k in ("prefill", "rows_logits", "state", "restored"):  # the same module both times
        np.testing.assert_array_equal(first[k], second[k])

    # a setting the programs close over changes the files' names: nothing of the old build is picked up
    st.stats = dict.fromkeys(st.stats, 0)
    old = M.ROWS_SHARED_MIN
    try:
        M.ROWS_SHARED_MIN = old + 1
        eng._progs.clear()
        eng.prefill(ids)
    finally:
        M.ROWS_SHARED_MIN = old
    assert st.stats["loaded"] == 0 and st.stats["exported"] > 0, st.stats


def test_names_do_not_depend_on_import_order(store):
    """A module first imported while a program is traced (pallas_mhc on the TPU) must not change the names of the
    programs resolved after it: `configure` imports every module whose settings are part of a name."""
    import sys
    saved = {m: sys.modules.pop(m) for m in ("glm53.pallas_mhc", "glm53.pallas_kda") if m in sys.modules}
    try:
        aot.configure(store, log=lambda *a: None)
        assert all(m in sys.modules for m in ("glm53.pallas_mhc", "glm53.pallas_kda"))
        before = aot._stable(aot._flags())
        import glm53.pallas_mhc  # noqa: F401
        assert aot._stable(aot._flags()) == before
    finally:
        sys.modules.update(saved)


def test_off_is_plain_jit():
    assert not aot.STORE.active
    f = aot.jit(("k",), lambda x: x + 1)
    assert not isinstance(f, aot._Program) and int(f(jnp.int32(1))) == 2


def test_second_signature_falls_back(store):
    st = aot.configure(store, log=lambda *a: None)
    st.stats = dict.fromkeys(st.stats, 0)
    f = aot.jit(("k2",), lambda x: x * 2)
    assert np.asarray(f(jnp.arange(4))).tolist() == [0, 2, 4, 6] and f.origin == "exported"
    assert np.asarray(f(jnp.arange(3))).tolist() == [0, 2, 4] and f.origin == "traced"
    g = aot.jit(("k2",), lambda x: x * 2)
    assert np.asarray(g(jnp.arange(4))).tolist() == [0, 2, 4, 6] and g.origin == "loaded"
