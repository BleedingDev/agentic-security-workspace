"""Ahead-of-time export of the engine's programs (`jax.export`).

The first call of a program pays three things: JAX traces the Python (the model code and every Pallas kernel body),
lowers the trace to StableHLO, and XLA compiles that. The persistent compile cache removes the third. This module
removes the first two: a program that had to be traced is exported and written to disk (StableHLO with the Mosaic
kernels as `tpu_custom_call`s; the weights are arguments, so a file holds code only), and a later process
deserializes the file and calls it (`Exported.call`), whose lowering is one module merge.

`aot.jit(key, fn, **jit_kw)` stands in for `jax.jit` at the engine's program sites. Until `configure` names a
directory it IS `jax.jit`. With one, the program is resolved at its first call:

  * a file for (code, versions, engine configuration, program key, argument signature) exists -> `jit(Exported.call)`;
  * otherwise the function is traced once for the export, the file is written, and the program that runs is the
    export read back — the module a later run rebuilds from the file, so both runs ask the compile cache for the
    same executable;
  * an export that fails for any reason leaves the plain `jax.jit` program (logged once per program).

A file's name hashes everything the trace depends on besides the function's own arguments: the package's source, the
jax / jaxlib versions, the device kind and jax's tracing flags (`configure`), the engine configuration the programs
close over (`engine_context`), the program's key and the shapes, dtypes and shardings of its arguments. A file from
any other build is simply not found.
"""
import dataclasses
import hashlib
import importlib
import os
import sys
import time

import numpy as np
import jax

EXT = ".jaxexp"
MODULES = ("model", "engine", "resident", "dflash", "pallas_moe", "pallas_mhc", "pallas_kda", "quant8", "planes",
           "iqquant", "iq_grids_data", "vision", "aot")       # the code a program's trace can reach


class _Store:
    def __init__(self):
        self.write_dir = None          # traced programs are exported here (None: never export)
        self.read_dirs = ()            # searched in order for a program's file
        self.tag = ""                  # code + versions + device kind
        self.context = None            # () -> str: the configuration the programs close over
        self.log = print
        self.stats = {"loaded": 0, "exported": 0, "traced": 0, "failed": 0, "load_s": 0.0, "export_s": 0.0,
                      "first_loaded_s": 0.0, "first_exported_s": 0.0, "first_traced_s": 0.0}   # first calls, by origin

    @property
    def active(self):
        return bool(self.write_dir or self.read_dirs)


STORE = _Store()


def _stable(x):
    """A repr without addresses or array contents (configuration values only)."""
    if x is None or isinstance(x, (bool, int, float, str, bytes)):
        return repr(x)
    if isinstance(x, (tuple, list)):
        return "[" + ",".join(_stable(v) for v in x) + "]"
    if isinstance(x, (set, frozenset)):
        return "{" + ",".join(sorted(_stable(v) for v in x)) + "}"
    if isinstance(x, dict):
        items = sorted(x.items(), key=lambda kv: repr(kv[0]))
        return "{" + ",".join(f"{_stable(k)}:{_stable(v)}" for k, v in items) + "}"
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return type(x).__name__ + _stable({f.name: getattr(x, f.name) for f in dataclasses.fields(x)})
    if isinstance(x, jax.sharding.PartitionSpec):
        return repr(x)
    try:
        return "dtype:" + np.dtype(x).name
    except Exception:  # noqa: BLE001
        pass
    if callable(x):
        return f"{getattr(x, '__module__', '')}.{getattr(x, '__qualname__', type(x).__name__)}"
    return f"<{type(x).__name__}>"


def _flags():
    """The modules' upper-case settings (MHC_KERNEL, KDA_KERNEL, SAMPLE_IMPL, ...): options a build may switch."""
    out = {}
    for m in MODULES:
        mod = sys.modules.get("glm53." + m)             # (`configure` imported them all: see there)
        for k, v in (vars(mod).items() if mod is not None else ()):
            if k.isupper() and (v is None or isinstance(v, (bool, int, float, str, tuple))):
                out[f"{m}.{k}"] = v
    return out


def engine_context(eng):
    """What `eng`'s programs close over besides their own key and arguments, as a string: the model and drafter
    configurations, the capacity, the layer grouping, the expert kernels' options and the modules' settings."""
    f, d = eng.expert_fetch, getattr(eng, "dflash", None)
    simple = lambda o: {k: v for k, v in vars(o).items() if not k.startswith("_")}   # noqa: E731
    return _stable({
        "cfg": eng.cfg, "lcfg": eng.lcfg, "max_len": eng.max_len, "n": eng.n, "axes": tuple(eng.mesh.axis_names),
        "lpp": getattr(eng, "lpp", None), "piece": getattr(eng, "prefill_piece", None), "class": type(eng).__name__,
        "fetch": None if f is None else (type(f).__name__, simple(f)),
        "dflash": None if d is None else (d.cfg, d.lcfg, d.temp_scale, d.embed_spec, d.lm_head_spec),
        "mtp": None if getattr(eng, "mtp", None) is None else getattr(eng, "mtp_k", None),
        "flags": _flags()})


def configure(write_dir=None, read_dirs=(), context=None, log=None):
    """Turn the export on: traced programs are written to `write_dir`, and a program's file is looked up there and
    in `read_dirs` (e.g. the serve dataset's `exported/`). `context` () -> str names the engine configuration
    (`lambda: aot.engine_context(eng)`; evaluated when a program is resolved, so the engine may be built later).
    Without the `flatbuffers` package (jax serializes an export with it) the export stays off."""
    if log is not None:
        STORE.log = log
    try:
        import flatbuffers  # noqa: F401
    except ImportError:
        STORE.write_dir, STORE.read_dirs = None, ()
        STORE.log("   aot: no flatbuffers package (pip install flatbuffers) -> programs are traced as usual")
        return STORE
    h = hashlib.sha256()
    for m in MODULES:                                   # the package's source as it is on disk now
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), m + ".py"), "rb") as fh:
            h.update(fh.read())
        # Import every module now. Some are first imported while a program is traced (pallas_mhc: by the first decode
        # trace), and their settings are part of the files' names (`_flags`): a run that loads its programs traces
        # nothing, so without this it would name the programs after that point differently from the run that
        # exported them (v6e 2026-10-06: 98 of 99 loaded, the first program after the import exported again).
        try:
            importlib.import_module("glm53." + m)
        except Exception as e:  # noqa: BLE001
            STORE.log(f"   aot: glm53.{m} not importable ({type(e).__name__}: {str(e)[:120]}); its settings are left out")
    import jaxlib
    dev = jax.devices()[0]
    flags = [str(getattr(jax.config, k, "")) for k in ("jax_enable_x64", "jax_default_matmul_precision",
                                                       "jax_use_shardy_partitioner")]      # what a trace reads
    STORE.tag = "|".join((h.hexdigest(), jax.__version__, jaxlib.__version__, dev.platform,
                          str(getattr(dev, "device_kind", "")), str(jax.device_count()), *flags))
    STORE.write_dir = str(write_dir) if write_dir else None
    dirs = ([STORE.write_dir] if STORE.write_dir else []) + [str(d) for d in read_dirs if d]
    STORE.read_dirs = tuple(dict.fromkeys(dirs))
    STORE.context = context
    if STORE.write_dir:
        os.makedirs(STORE.write_dir, exist_ok=True)
    return STORE


def _signature(args):
    """Shapes, dtypes and shardings of a call's arguments (the pytree structure included)."""
    leaves, tree = jax.tree.flatten(args)

    def one(a):
        sh = getattr(a, "sharding", None)
        where = "host" if sh is None else str(getattr(sh, "spec", type(sh).__name__))
        return f"{tuple(np.shape(a))}:{getattr(a, 'dtype', type(a).__name__)}:{where}:{getattr(a, '_committed', '')}"
    return str(tree) + "|" + ";".join(one(a) for a in leaves)


def _label(key):
    k = key
    while isinstance(k, tuple) and k:
        k = k[0]
    return "".join(c if c.isalnum() or c == "_" else "_" for c in str(k))[:32] or "prog"


def _call(exp):
    """`exp.call` under the exported function's own name (the jit program keeps the name it had: logs, profiles)."""
    def call(*args):
        return exp.call(*args)
    call.__name__ = call.__qualname__ = exp.fun_name
    return call


class _Program:
    """A program resolved at its first call (see the module docstring); afterwards one extra Python call."""

    def __init__(self, fn, key, jit_kw):
        self._fn, self._key, self._kw = fn, key, jit_kw
        self._f = self._sig = None
        self.origin = None                          # "loaded" | "exported" | "traced" once resolved

    def __call__(self, *args):
        f = self._f
        if f is None:
            return self._first(args)
        try:
            return f(*args)
        except Exception:
            # jax.jit retraces for a new argument signature; an export holds one. The error of a mismatch is raised
            # while the call is traced (nothing ran, no donated buffer consumed): fall back to the plain program.
            try:
                same = self._sig is None or _signature(args) == self._sig
            except Exception:  # noqa: BLE001
                same = True
            if same:
                raise
            STORE.log(f"   aot: {_label(self._key)} called with a second signature -> traced")
            self._plain()
            return self._f(*args)

    def _plain(self):
        STORE.stats["traced"] += 1
        self._f, self._sig, self.origin = jax.jit(self._fn, **self._kw), None, "traced"

    def _first(self, args):
        """Resolve and make the first call (the lowering and the compile, or the compile cache's read, happen here).
        An export that cannot be called after all (another platform or device count, a compile error) is replaced by
        the plain program, once."""
        f = self._resolve(args)
        t = time.time()
        try:
            out = f(*args)
        except Exception as e:  # noqa: BLE001
            if self.origin == "traced":
                raise
            STORE.stats["failed"] += 1
            STORE.log(f"   aot: {_label(self._key)} ({self.origin}) failed at its first call ({type(e).__name__}: "
                      f"{str(e)[:300]}) -> traced")
            self._plain()
            f, out = self._f, self._f(*args)
        STORE.stats[f"first_{self.origin}_s"] += time.time() - t
        self._f = f
        return out

    def _cache_size(self):
        return 0 if self._f is None else self._f._cache_size()

    def _resolve(self, args):
        from jax import export as jex
        st = STORE
        sig = _signature(args)
        ctx = st.context() if st.context is not None else ""
        h = hashlib.sha256("\n".join((st.tag, ctx, _stable(self._key), _stable(self._kw), sig)).encode()).hexdigest()
        name = f"{_label(self._key)}-{h[:24]}{EXT}"
        for d in st.read_dirs:
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                continue
            try:
                t = time.time()
                with open(p, "rb") as fh:
                    exp = jex.deserialize(bytearray(fh.read()))
                f = jax.jit(_call(exp), **self._kw)
                st.stats["loaded"] += 1
                st.stats["load_s"] += time.time() - t
                self._sig, self.origin = sig, "loaded"
                return f
            except Exception as e:  # noqa: BLE001
                st.log(f"   aot: {p} unreadable ({type(e).__name__}: {str(e)[:200]}) -> traced")
                break
        jf = jax.jit(self._fn, **self._kw)
        if st.write_dir:
            try:
                t = time.time()
                blob = jex.export(jf)(*args).serialize()
                tmp = os.path.join(st.write_dir, f".{name}.{os.getpid()}.tmp")
                with open(tmp, "wb") as fh:
                    fh.write(blob)
                os.replace(tmp, os.path.join(st.write_dir, name))
                f = jax.jit(_call(jex.deserialize(bytearray(blob))), **self._kw)
                st.stats["exported"] += 1
                st.stats["export_s"] += time.time() - t
                self._sig, self.origin = sig, "exported"
                return f
            except Exception as e:  # noqa: BLE001
                st.stats["failed"] += 1
                st.log(f"   aot: export of {name} failed ({type(e).__name__}: {str(e)[:400]}) -> traced")
        st.stats["traced"] += 1
        self.origin = "traced"
        return jf


def jit(key, fn, **jit_kw):
    """`jax.jit(fn, **jit_kw)` for a program site; `key` identifies the program within its engine (the `_progs` key)."""
    if not STORE.active:
        return jax.jit(fn, **jit_kw)
    return _Program(fn, key, jit_kw)


def summary():
    s = STORE.stats
    return (f"{s['loaded']} programs loaded from exports (read {s['load_s']:.0f}s, first calls "
            f"{s['first_loaded_s']:.0f}s), {s['exported']} traced and exported (export {s['export_s']:.0f}s, first "
            f"calls {s['first_exported_s']:.0f}s), {s['traced']} traced only (first calls {s['first_traced_s']:.0f}s), "
            f"{s['failed']} export failures")
