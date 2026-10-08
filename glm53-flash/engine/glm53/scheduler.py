"""Continuous batching for `ResidentLayerEngine.decode_rows`: independent streams decoded together, admitted between
steps, prefilled piece by piece with decode steps of the running streams in between, each keeping its own cache set.

One engine thread (`Scheduler.run`) owns every JAX call. Client threads `submit` a `Request` and wait on its
`done` event (tokens arrive through `on_token`, called on the engine thread). Cache sets on the chips = the active
streams' + finished contexts kept LIVE for the next turn (LRU), at most `max_sets` in total; beyond that, finished
contexts are parked in host memory (`SnapStore`) and resumed when a later prompt extends them. Between steps only the
sampled ids cross to the host: token ids, positions and the sampler state stay on the device.

DFlash 2 (`spec`, with a drafter installed on the engine: `ResidentLayerEngine.set_dflash`): a stream that runs ALONE
decodes speculatively (`dflash_start` / `dflash_step`: a block of T-1 drafts verified per step); the block size is
picked per stream and step from an EMA of the tokens per step (`DFlashPolicy`). Several active streams decode together:
either speculatively in ONE batched step (`dflash_start_rows` / `dflash_step_rows`: every stream verifies its own block
of the smallest size on its own caches, accepted per row) when the policy's measured step costs (`rows`: B -> ms of a
B-row DFlash step and of a plain batched step) and the streams' EMAs say it yields more tokens per ms, or in the
batched plain steps (`decode_rows`, which also write every stream's positions into its own drafter ring, so a stream
that goes back to DFlash drafts with its full context). A DFlash state owns its streams' caches and hands them back
when the membership or the mode changes. With temperature > 0 the drafts are accepted by rejection sampling against
the target's distribution (every token distributed as plain sampling).
"""
import collections
import threading
import time

import numpy as np
import jax
import jax.numpy as jnp

from glm53.engine import DeviceSampler


def match_prefix(live_ids, pos, prompt, quiet=False):
    """Default matcher: (k, fed) when the context (live_ids[:pos]) is a prefix of `prompt`: prompt[k:] must still be
    prefilled and `fed` = the ids the engine will have seen after that; (0, None) otherwise."""
    if pos == 0 or len(prompt) < pos:
        return 0, None
    a = np.asarray(live_ids[:pos]); b = np.asarray(prompt[:pos])
    if a.shape != b.shape or not np.array_equal(a, b):
        return 0, None
    return pos, list(live_ids[:pos]) + list(prompt[pos:])


class Request:
    """A generation request. `prompt`: token ids (the server's signature ids: image tokens negative, `imgs` maps
    them; the scheduler's `feed` turns slices into real ids + embedding overrides). Results: `out` (emitted tokens,
    the stop token included when hit), `stop_reason` ("stop" | "length" | "cancelled" | "error"), timings."""

    def __init__(self, prompt, max_new, temperature=1.0, top_p=1.0, imgs=None, on_token=None, rid=None, on_done=None,
                 budget=None):
        self.prompt = [int(t) for t in prompt]
        self.max_new, self.temperature, self.top_p = int(max_new), float(temperature), float(top_p)
        self.imgs, self.on_token, self.rid, self.on_done = imgs, on_token, rid, on_done
        self.budget = budget            # (n, end_id, forced_ids): unless `end_id` was emitted within the first n output
                                        # tokens, the next tokens are forced to `forced_ids` (a thinking budget)
        self.out = []
        self.done = threading.Event()
        self.error = None
        self.cancelled = False
        self.stop_reason = None
        self.reused = 0
        self.prefill_s = 0.0
        self.t_submit, self.t_first, self.t_end = time.time(), None, None

    def cancel(self):
        self.cancelled = True

    @property
    def decode_s(self):
        return (self.t_end or time.time()) - (self.t_first or self.t_submit)


class _Stream:
    """An admitted request: its own cache set, its position, the ids fed so far and the last (unfed) token."""

    def __init__(self, req, caches, pos, fed, tok, logits):
        self.req, self.caches, self.pos, self.fed, self.tok = req, caches, pos, list(fed), tok
        self.seen = []                                   # tokens fed beyond `fed`
        self.logits = logits                             # logits [1,V] after the last fed token (predict `tok`)
        self.max_new = req.max_new
        self.open = req.budget is not None               # the budgeted block (thinking) is still open
        self.force = []                                  # tokens to feed instead of the sampled ones (budget reached)
        self.spec = None                                 # DFlash state (engine.dflash_start) while decoding alone
        self.sampler = None                              # its own device sampler in DFlash mode
        self.tau = None                                  # DFlashPolicy's per-stream EMAs


class SnapStore:
    """Host-memory LRU of compact context snapshots keyed by their token ids (see `ResidentLayerEngine.snapshot_prefix`).
    `park` moves a context off the chips, `lookup` finds the entry a prompt extends (the matcher handles dropped
    thinking blocks and boundary drift when the server passes its own), `restore` brings it back into fresh
    full-capacity caches. A parked conversation keeps its two latest turn boundaries; older strict prefixes are
    dropped unless pinned (pinned = a conversation's system section, the branch point new sessions start from)."""

    def __init__(self, eng, max_bytes, rows_bucket=1024, match_len=match_prefix, log=print, state=None):
        self.eng, self.max_bytes, self.rows_bucket = eng, max_bytes, rows_bucket
        self.match_len, self.log = match_len, log
        self.state = state if state is not None else {}
        self.entries = collections.OrderedDict()                     # tuple(ids) -> entry, oldest first
        self.bytes = 0

    def park(self, ids, caches, pos, logits, pinned=False):
        """Snapshot `caches` (a context of `pos` tokens = `ids`) into host memory; the caches stay valid."""
        key = tuple(int(t) for t in ids)
        if key in self.entries:
            e = self.entries[key]
            e["pinned"] = e["pinned"] or pinned
            if logits is not None and e["logits"] is None:
                e["logits"] = np.asarray(logits)
            self.entries.move_to_end(key)
            return e
        t = time.time()
        snap = self.eng.snapshot_prefix(caches, pos, rows_bucket=self.rows_bucket)
        host = self.eng.snapshot_to_host(snap)
        del snap
        e = {"ids": list(key), "n": pos, "snap": host, "logits": None if logits is None else np.asarray(logits),
             "bytes": host["bytes"], "pinned": pinned}
        if not pinned:                                                # supersede our own earlier turns but the latest
            older = sorted((k for k, o in self.entries.items() if not o["pinned"] and len(k) < len(key) and key[:len(k)] == k),
                           key=len)
            for k in older[:-1]:
                self._drop(k)
        self.entries[key] = e
        self.bytes += e["bytes"]
        while self.bytes > self.max_bytes and len(self.entries) > 1:
            victims = [k for k, o in self.entries.items() if not o["pinned"]] or list(self.entries)
            self._drop(victims[0])
        st = self.state
        st["snap_parks" if not pinned else "snap_pins"] = st.get("snap_parks" if not pinned else "snap_pins", 0) + 1
        st["snap_entries"], st["snap_bytes"] = len(self.entries), self.bytes
        st["snap_s"] = st.get("snap_s", 0.0) + time.time() - t
        self.log(f"parked {pos} tokens ({e['bytes'] / 1e6:.0f} MB, {'pinned' if pinned else 'lru'}) in {time.time() - t:.2f}s; "
                 f"store {len(self.entries)} entries, {self.bytes / 1e9:.2f} GB")
        return e

    def _drop(self, key):
        e = self.entries.pop(key)
        self.bytes -= e["bytes"]

    def lookup(self, prompt):
        """-> (k, fed, entry) for the entry that covers most of `prompt`, or (0, None, None)."""
        best = (0, None, None)
        for key, e in list(self.entries.items()):
            if e["n"] <= best[0]:
                continue
            k, fed = self.match_len(e["ids"], e["n"], prompt, quiet=True)
            if k > best[0] and not (k == len(prompt) and e["logits"] is None):
                best = (k, fed, e)
        if best[2] is not None:
            self.entries.move_to_end(tuple(best[2]["ids"]))
        return best

    def restore(self, e):
        t = time.time()
        caches = self.eng.restore_prefix(self.eng.snapshot_from_host(e["snap"]))
        self.state["snap_hits"] = self.state.get("snap_hits", 0) + 1
        self.state["snap_s"] = self.state.get("snap_s", 0.0) + time.time() - t
        return caches


def _argmax_first(logits_row, temperature, top_p):
    return int(np.argmax(logits_row))


class DFlashPolicy:
    """Block size per DFlash step: `cost` = {T: ms per step} (measured, e.g. v6e 2026-10-03: {4: 21.2, 8: 27.3});
    the stream's EMA of tokens per step for each T picks the T with the least ms per token. A step of T also tells
    what a smaller T' would have accepted (min(tokens, T')), and a run cut before T what a larger one would have
    (the same tokens); only a fully accepted block leaves the larger T unknown — a T without news for `probe` steps
    is tried once. `prior` = the EMAs' start values. `rows` = {B: (ms of a DFlash step of B streams at the smallest
    block, ms of a plain batched step of B)}: B streams decode speculatively together when the sum of their EMAs per
    DFlash ms beats B tokens per plain ms (`pick_rows`); no entry for B = always plain. Plain steps measure no
    acceptance, so the choice is sticky: a batched DFlash state, once started (by the costs or by a probe after
    `probe` plain steps), runs at least `hold` steps and ends only below `stay` x the plain rate (at temperature 1.0 the
    EMAs of prose sit near the break-even: without it one dip kept a pair of streams plain, v6e 2026-10-04 j131)."""

    def __init__(self, cost, prior=None, alpha=0.15, probe=32, rows=None, hold=8, stay=0.9):
        self.cost = {int(t): float(c) for t, c in cost.items()}
        self.prior = prior or {t: 1.0 + 0.5 * (t - 1) ** 0.5 for t in self.cost}
        self.alpha, self.probe = alpha, probe
        self.rows = {int(b): (float(c[0]), float(c[1])) for b, c in (rows or {}).items()}
        self.hold, self.stay = hold, stay

    @property
    def rows_T(self):
        """The block size of batched DFlash steps (one for all rows: the smallest compiled)."""
        return min(self.cost)

    def pick_rows(self, streams, margin=1.0):
        """True when B streams decoding speculatively together (each `start`ed) yield more tokens per ms than
        margin x plain."""
        c = self.rows.get(len(streams))
        if c is None:
            return False
        return sum(s.tau["ema"][self.rows_T] for s in streams) / c[0] > margin * len(streams) / c[1]

    def start(self, s):
        s.tau = {"ema": dict(self.prior), "last": {t: 0 for t in self.cost}, "n": 0}

    def pick(self, s):
        st = s.tau
        for t, last in st["last"].items():
            if st["n"] - last >= self.probe:
                return t
        return min(self.cost, key=lambda t: self.cost[t] / st["ema"][t])

    def update(self, s, T, n_tokens):
        st = s.tau
        st["n"] += 1
        for t in self.cost:
            if t <= T or n_tokens < T:                 # a run cut before T ends there for any larger block too
                st["ema"][t] += self.alpha * (min(n_tokens, t) - st["ema"][t])
                st["last"][t] = st["n"]


class Scheduler:
    """See the module docstring. `feed(req, a, b) -> (ids [n] int32, embeds or None)` renders req.prompt[a:b] for
    the engine (the server attaches its image data to the request); `match_len(live_ids, pos, prompt, quiet=False) -> (k, fed)`; `system_end(prompt)` = length of
    the system section worth pinning (0 = none); `first_sample(logits_row, temperature, top_p) -> int` samples the
    first token of a request on the host (the rest come from the device sampler); `spec` = a `DFlashPolicy` to
    decode a stream that runs alone with the engine's DFlash drafter (None = never)."""

    def __init__(self, eng, stop_ids, max_streams=4, max_sets=None, feed=None, match_len=match_prefix,
                 system_end=None, first_sample=_argmax_first, snaps=None, log=print, state=None,
                 base_min=512, snap_min=256, piece=None, seed=None, min_free_gb=0.65, max_wait_s=90.0, spec=None):
        self.eng, self.stop_ids = eng, set(int(t) for t in stop_ids)
        self.max_streams = max(1, int(max_streams))
        self.max_sets = max(self.max_streams, int(max_sets if max_sets is not None else max_streams + 1))
        self.feed = feed or (lambda req, a, b: (np.asarray(req.prompt[a:b], np.int32), None))
        self.match_len, self.system_end = match_len, (system_end or (lambda prompt: 0))
        self.first_sample = first_sample
        self.snaps = snaps if snaps is not None else SnapStore(eng, 0, 1, match_len, log, state)
        self.log, self.state = log, (state if state is not None else {})
        self.base_min, self.snap_min = base_min, snap_min
        self.piece = piece or eng.prefill_piece
        self.min_free_gb = min_free_gb                  # HBM an admission needs (free_gb: a set, prefill temporaries, programs); 0 = off
        self.max_wait_s = max_wait_s                    # a request queued longer than this fails ("queue_timeout")
        self.live = collections.OrderedDict()            # finished contexts on the chips (LRU): key -> ctx dict
        self._live_n = 0
        self.active = []
        self.pending = collections.deque()
        self.cond = threading.Condition()
        self.sampler = DeviceSampler(1.0, 1.0, seed=seed)
        self._members = None                             # ids of the streams the device state below belongs to
        self._toks = self._pos = None
        self._stop = False
        self._paused, self._idle = False, threading.Event()
        self._waits = 0
        self._t_step = None
        self.thread = None
        self.steps = 0
        self.spec = spec if spec is not None and getattr(eng, "dflash", None) is not None else None
        self.seed = seed
        self._buf_cache = None                           # (sets on the chips, live buffer bytes on chip 0): free_gb
        self._grave = None                               # the last batched step's consumed caches (deferred release)
        self._rows = None                                # DFlash of several streams: {"streams", "st", "sampler"}
        self._rows_skip = 0                              # plain batched steps since the policy last chose DFlash rows
        self._rows_hold = 0                              # steps the running batched DFlash state still runs regardless

    # ---- client side
    def submit(self, req: Request):
        if len(req.prompt) + 2 > self.eng.max_len:
            self._fail(req, ValueError(f"prompt of {len(req.prompt)} tokens exceeds the context capacity {self.eng.max_len}"))
            return req
        with self.cond:
            self.pending.append(req)
            self.cond.notify()
        return req

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True, name="glm-scheduler")
        self.thread.start()
        return self.thread

    def stop(self):
        with self.cond:
            self._stop = True
            self.cond.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=60)
        for ctx in self.live.values():
            self._free_set(ctx["caches"])
        for s in self.active:
            self._spec_end(s)
            self._free_set(s.caches)
        self.live.clear(); self.active.clear(); self._toks = self._pos = None

    @staticmethod
    def _free_set(caches):
        """Release a dropped cache set's HBM now (`Array.delete`), whatever else still references the arrays: a
        dropped set kept alive by a stray reference (seen once after a concurrent burst, 2026-09-13) cost a stream
        slot for the rest of the session — refcounts alone are not a guarantee."""
        for x in jax.tree.leaves(caches):
            try:
                x.delete()
            except Exception:  # noqa: BLE001
                pass

    @property
    def n_sets(self):
        return len(self.active) + len(self.live)

    def free_gb(self):
        """HBM an admission can count on, chip 0, in GB: bytes_limit minus the live device buffers (None when the backend
        reports no memory statistics: CPU tests). bytes_in_use is NOT the measure: it also holds the runtime's cache of
        loaded programs, which the runtime evicts when a buffer needs the room and reloads at a program's next run
        (2026-10-03, v6e under the v5e budget: five 262k sets allocated where bytes_limit - bytes_in_use promised two,
        and 0.3-0.6 GB of in-use were such programs). `min_free_gb` therefore covers a new set, one prefill piece's
        temporaries AND the programs a decode step and a prefill piece need at once. The buffer sum (global arrays,
        ~6 ms) is cached while the number of cache sets on the chips does not change."""
        st = self.eng.mesh.devices.flat[0].memory_stats() or {}
        if "bytes_limit" not in st:
            return None
        key = self.n_sets
        if self._buf_cache is None or self._buf_cache[0] != key:
            n = len(self.eng.mesh.devices.flat)
            tot = 0
            for a in jax.live_arrays():
                try:
                    if len(a.sharding.device_set) == n:
                        tot += a.nbytes if a.sharding.is_fully_replicated else a.nbytes // n
                except Exception:  # noqa: BLE001  (deleted arrays)
                    pass
            self._buf_cache = (key, tot)
        return (st["bytes_limit"] - self._buf_cache[1]) / 1e9

    def in_use_free_gb(self):
        """bytes_limit - bytes_in_use on chip 0 (GB; the loaded programs count as used): for logs."""
        st = self.eng.mesh.devices.flat[0].memory_stats() or {}
        return round((st["bytes_limit"] - st["bytes_in_use"]) / 1e9, 3) if "bytes_limit" in st else None

    def _headroom(self):
        """True when an admission can allocate a cache set and run its prefill: parks live contexts (LRU) until the
        set budget and the HBM margin allow it; False when only active streams hold the chips (wait for one)."""
        self._make_room()
        f = self.free_gb()
        return f is None or not self.min_free_gb or f >= self.min_free_gb

    # ---- engine thread
    def pause(self, timeout=60.0):
        """Stop the engine thread between steps (running streams stall, nothing is admitted) until `resume`; returns
        once the thread is idle. For probe jobs that need the chips."""
        with self.cond:
            self._paused = True
            self.cond.notify_all()
        return self._idle.wait(timeout)

    def resume(self):
        with self.cond:
            self._paused = False
            self.cond.notify_all()

    def run(self):
        while not self._stop:
            with self.cond:
                while not self._stop and (self._paused or (not self.pending and not self.active)):
                    if self._paused:
                        self._idle.set()
                    self._t_step = None
                    self.cond.wait(timeout=1.0)
                self._idle.clear()
                if self._stop:
                    break
                req = None
                while self.pending and self.max_wait_s and time.time() - self.pending[0].t_submit > self.max_wait_s:
                    old = self.pending.popleft()                        # queued too long: fail it (the client can retry)
                    self._fail(old, TimeoutError(f"queued for {time.time() - old.t_submit:.0f}s without a free cache set"), "queue_timeout")
                if self.pending and len(self.active) < self.max_streams and self._headroom():
                    req = self.pending.popleft()
                elif self.pending and not self.active:                  # nothing running and still no room
                    if self._waits % 10 == 0:
                        self.log(f"admission waits: {len(self.live)} live contexts, free HBM {self.free_gb():.3f} GB "
                                 f"(in-use figure {self.in_use_free_gb()} GB)" if self.free_gb() is not None else
                                 f"admission waits: {len(self.live)} live contexts")
                    self._waits += 1
                    self.cond.wait(timeout=1.0)
            if req is not None:
                if req.cancelled:
                    self._fail(req, None, "cancelled"); continue
                try:
                    self._admit(req)
                except Exception as e:  # noqa: BLE001
                    import gc, traceback
                    self.log("admission failed:", traceback.format_exc()[-1500:])
                    self._fail(req, RuntimeError(f"{type(e).__name__}: {str(e)[:800]}"))   # no traceback: its frames
                    del e                                                                    # would pin the half-built caches
                    gc.collect()
                continue
            if self.active:
                try:
                    self._step()
                except Exception as e:  # noqa: BLE001
                    import gc, traceback
                    self.log("decode step failed:", traceback.format_exc()[-1500:])
                    err = RuntimeError(f"{type(e).__name__}: {str(e)[:800]}")
                    del e
                    for s in list(self.active):
                        s.req.error = err
                        self._finish(s, "error", keep=False)
                    self.active = []; self._members = None; self._toks = self._pos = None
                    gc.collect()

    def _fail(self, req, error, reason="error"):
        """Finish a request that never became a stream (capacity error, admission failure, cancelled while queued)."""
        req.error, req.stop_reason, req.t_end = error, reason, time.time()
        if req.on_done is not None:
            try:
                req.on_done()
            except Exception:  # noqa: BLE001
                pass
        req.done.set()

    def _make_room(self):
        """Park LRU live contexts until a cache set can be allocated within the set budget AND the HBM margin
        (`min_free_gb`, prefill temporaries); a parked set is freed (donated caches)."""
        while self.live:                                                 # (no gc.collect() here: a full collection
            f = self.free_gb()                                           #  costs ~14 s in a process with hundreds of
            if self.n_sets < self.max_sets and (f is None or not self.min_free_gb or f >= self.min_free_gb):
                break                                                    #  compiled programs; refcounts free the set)
            key, ctx = self.live.popitem(last=False)
            if ctx["pos"] >= self.snap_min:
                self.snaps.park(ctx["ids"], ctx["caches"], ctx["pos"], ctx["logits"])
            self._free_set(ctx["caches"])
            del ctx

    def _prefill(self, req, a, b, caches, pos):
        """prompt[a:b] into `caches` at `pos`, one piece at a time with a decode step of the running streams between
        pieces (chunked admission). Returns (logits, caches, pos)."""
        logits = None
        for x in range(a, b, self.piece):
            y = min(b, x + self.piece)
            ids, emb = self.feed(req, x, y)
            ids = np.asarray(ids, np.int32).reshape(1, -1)
            logits, caches, pos = self.eng.prefill(ids, caches, pos, embeds=emb)
            if y < b and self.active:
                self._step()
        return logits, caches, pos

    def _admit(self, req):
        prompt = req.prompt
        t0 = time.time()
        st = self.state
        # 1. a finished context still on the chips that the prompt extends
        best = (0, None, None)
        for key, c in self.live.items():
            k, fed = self.match_len(c["ids"], c["pos"], prompt, quiet=True)
            if k > best[0] and not (k == len(prompt) and c["logits"] is None):
                best = (k, fed, key)
        c = None                                                         # (keep no reference to a context _make_room may drop)
        if best[2] is None and self.live:
            self.log(f"no live context matched ({len(self.live)} live, {len(self.snaps.entries)} parked)")
        if best[2] is not None:
            k, fed, key = best
            ctx = self.live.pop(key)
            st["prefix_hits"] = st.get("prefix_hits", 0) + 1
            st["prefix_tokens_reused"] = st.get("prefix_tokens_reused", 0) + k
            if k == len(prompt):
                logits, caches, pos = ctx["logits"], ctx["caches"], ctx["pos"]
            else:
                logits, caches, pos = self._prefill(req, k, len(prompt), ctx["caches"], ctx["pos"])
            req.reused = k
        else:
            self._make_room()
            k, fed, entry = self.snaps.lookup(prompt)
            if k > 0:                                                    # 2. a parked context the prompt extends
                st["prefix_tokens_reused"] = st.get("prefix_tokens_reused", 0) + k
                caches = self.snaps.restore(entry)
                if k == len(prompt):
                    logits, pos = entry["logits"], entry["n"]
                else:
                    logits, caches, pos = self._prefill(req, k, len(prompt), caches, entry["n"])
                req.reused = k
            else:                                                        # 3. from scratch (pin the system section)
                fed = list(prompt)
                n_sys = self.system_end(prompt)
                if n_sys >= self.base_min and n_sys < len(prompt):
                    _, caches, pos = self._prefill(req, 0, n_sys, None, 0)
                    self.snaps.park(prompt[:n_sys], caches, pos, None, pinned=True)
                    logits, caches, pos = self._prefill(req, n_sys, len(prompt), caches, pos)
                else:
                    logits, caches, pos = self._prefill(req, 0, len(prompt), None, 0)
        jax.block_until_ready(logits)
        self._t_step = None
        req.prefill_s = time.time() - t0
        st["prefill_s"] = st.get("prefill_s", 0.0) + req.prefill_s
        tok = self.first_sample(np.asarray(logits)[0], req.temperature, req.top_p)
        s = _Stream(req, caches, pos, fed, tok, logits)
        s.max_new = max(1, min(req.max_new, self.eng.max_len - pos - 1))
        req.t_first = time.time()
        self.active.append(s); self._members = None
        self._emit(s, tok)

    def _emit(self, s, tok):
        req = s.req
        req.out.append(tok)
        if s.open and tok == req.budget[1]:
            s.open = False
        if req.on_token is not None and not req.cancelled:
            try:
                req.on_token(tok)
            except Exception as e:  # noqa: BLE001
                self.log(f"on_token failed ({e!r}): cancelling {req.rid}")
                req.cancelled = True
        if tok in self.stop_ids:
            self._finish(s, "stop")
        elif len(req.out) >= s.max_new or s.pos + 1 >= self.eng.max_len:
            self._finish(s, "length")
        elif req.cancelled:
            self._finish(s, "cancelled")

    def _finish(self, s, reason, keep=True):
        """The stream's context becomes a live (on-chip) context: ids = everything the engine has seen (the last
        emitted token is never fed), logits = the prediction after them."""
        req = s.req
        self._spec_end(s)
        if s in self.active:
            self.active.remove(s); self._members = None
        if not keep:
            self._free_set(s.caches)
        if keep:
            self._live_n += 1
            lg = s.logits[0][s.logits[1]:s.logits[1] + 1] if isinstance(s.logits, tuple) else s.logits
            self.live[self._live_n] = {"ids": s.fed + s.seen, "caches": s.caches, "pos": s.pos, "logits": lg}
        req.stop_reason = req.stop_reason or reason
        req.t_end = time.time()
        self.state["tokens"] = self.state.get("tokens", 0) + len(req.out)
        if req.on_done is not None:
            try:
                req.on_done()
            except Exception as e:  # noqa: BLE001
                self.log(f"on_done failed ({e!r}) for {req.rid}")
        req.done.set()

    def _step(self):
        """One batched decode step of the active streams."""
        for s in list(self.active):
            if s.req.cancelled:
                self._finish(s, "cancelled")
        act = list(self.active)                                         # (streams finishing below leave self.active)
        if not act:
            return
        if len(act) == 1 and self._spec_ok(act[0]):
            self._rows_end()
            return self._spec_step(act[0])
        if len(act) > 1 and self._rows_ok(act):
            return self._rows_step(act)
        self._rows_end()
        for s in act:                                                   # batched again: DFlash states hand back
            self._spec_end(s)
        B = len(act)
        members = tuple(id(s) for s in act)
        if members != self._members:                                    # membership changed: re-place the small state
            self._pos = self.eng.device_positions([s.pos for s in act])
            self._toks = jnp.asarray([s.tok for s in act], jnp.int32)
            self.sampler.set([s.req.temperature for s in act], [s.req.top_p for s in act])
            self._members = members
        t0 = time.perf_counter()
        ids, logits, sets, pos_next = self.eng.decode_rows(self._toks, [s.caches for s in act], self._pos, self.sampler)
        self._grave = None                     # the last step's consumed caches: released while the device runs this one
        ids_h = np.asarray(ids)                                         # the one device -> host sync per step
        forced = False
        for r, s in enumerate(act):                                     # a budget reached: feed its forced tokens
            if s.open and not s.force and len(s.req.out) >= s.req.budget[0]:
                s.force = list(s.req.budget[2])
            if s.force:
                if not forced:
                    ids_h, forced = np.array(ids_h), True
                ids_h[r] = s.force.pop(0)
        if forced:
            ids = jnp.asarray(ids_h, jnp.int32)
        t1 = time.perf_counter()
        self._toks, self._pos = ids, pos_next
        self.steps += 1
        st = self.state
        st["steps"] = self.steps
        st["step_tokens"] = st.get("step_tokens", 0) + B
        st["decode_s"] = st.get("decode_s", 0.0) + (t1 - t0)          # inside the engine (dispatch + device + sync)
        if self._t_step is not None:                                    # back-to-back steps: the loop's own overhead
            st["step_s"] = st.get("step_s", 0.0) + (t1 - self._t_step)
            st["steps_busy"] = st.get("steps_busy", 0) + 1
        self._t_step = t1
        self._grave = [s.caches for s in act]           # (dropping ~126 arrays per set costs ~1.5 ms of host time)
        for r, s in enumerate(act):                                     # every row takes its new caches first
            s.caches, s.pos = sets[r], s.pos + 1
            s.seen.append(s.tok)
            s.tok = int(ids_h[r])
            s.logits = (logits, r)                                      # sliced only when the stream finishes
        for s in act:
            self._emit(s, s.tok)

    # ---- DFlash (a stream running alone)
    def _spec_ok(self, s):
        """DFlash for this stream now: a policy and a drafter, no forced tokens pending, and a whole block of room
        before the thinking budget, max_new and the context capacity (the tail decodes plainly)."""
        if self.spec is None or s.force:
            return False
        T, req = max(self.spec.cost), s.req
        if s.open and len(req.out) + T >= req.budget[0]:
            return False
        return len(req.out) + T < s.max_new and s.pos + T + 1 < self.eng.max_len

    def _spec_end(self, s):
        """Leave DFlash mode: the stream takes its caches (the drafter's ring included) and position back; the queued
        draft is dropped (it only wrote committed positions into the ring). A stream of a batched DFlash state ends
        that state for all its streams."""
        if self._rows is not None and any(x is s for x in self._rows["streams"]):
            self._rows_end()
        if s.spec is None:
            return
        st = self.eng.dflash_settle(s.spec)                            # the last step's rollback was deferred
        s.caches, s.pos, s.spec = st["caches"], st["pos_h"], None
        self.eng.dflash_end()
        self._members = None

    def _spec_step(self, s):
        req, st = s.req, self.state
        if s.spec is None:
            if s.tau is None:
                self.spec.start(s)
            if s.sampler is None:
                s.sampler = DeviceSampler(req.temperature, req.top_p, seed=self.seed)
            s.spec = self.eng.dflash_start(np.array([s.tok]), s.caches, s.pos, T=self.spec.pick(s), sampler=s.sampler)
            s.caches = None
            self._members = None
        T = s.spec["T"]
        t0 = time.perf_counter()                                        # the next block's T is chosen before this
        emitted, s.spec = self.eng.dflash_step(s.spec, s.sampler, tuple(sorted(self.stop_ids)),   # step's tokens are
                                               next_T=self.spec.pick(s))                         # known (it is queued)
        self.spec.update(s, T, len(emitted))
        t1 = time.perf_counter()
        self.steps += 1
        st["steps"] = self.steps
        st["step_tokens"] = st.get("step_tokens", 0) + len(emitted)
        st["spec_steps"] = st.get("spec_steps", 0) + 1
        st["spec_tokens"] = st.get("spec_tokens", 0) + len(emitted)
        st["decode_s"] = st.get("decode_s", 0.0) + (t1 - t0)
        if self._t_step is not None:
            st["step_s"] = st.get("step_s", 0.0) + (t1 - self._t_step)
            st["steps_busy"] = st.get("steps_busy", 0) + 1
        self._t_step = t1
        s.seen.extend([s.tok] + emitted[:-1])                           # the anchor and the accepted drafts were fed
        s.tok, s.pos, s.logits = emitted[-1], s.spec["pos_h"], None
        for tok in emitted:
            if s not in self.active:                                    # finished mid-block (length / cancelled)
                break
            self._emit(s, tok)

    # ---- DFlash for several streams at once (one batched verify per step)
    def _rows_ok(self, act):
        """Batched DFlash for these streams now: every one could run DFlash alone (`_spec_ok`), the policy has costs for
        this B, and their EMAs say it pays — or `probe` plain batched steps passed without (to measure again). A
        running state stays for its `hold` steps, then while it pays `stay` x plain (`DFlashPolicy`)."""
        pol = self.spec
        if pol is None or len(act) not in pol.rows or not all(self._spec_ok(s) for s in act):
            return False
        for s in act:
            if s.tau is None:
                pol.start(s)
        if self._rows is not None and [id(x) for x in self._rows["streams"]] == [id(x) for x in act]:
            self._rows_hold -= 1                       # (the running state)
            ok = self._rows_hold >= 0 or pol.pick_rows(act, pol.stay)
        else:
            ok = pol.pick_rows(act) or self._rows_skip >= pol.probe
            self._rows_hold = pol.hold
        self._rows_skip = 0 if ok else self._rows_skip + 1
        return ok

    def _rows_end(self):
        """Leave batched DFlash: every stream of the state takes its caches (ring included) and position back."""
        R = self._rows
        if R is None:
            return
        st = self.eng.dflash_settle_rows(R["st"])                         # the last step's rollbacks were deferred
        for b, s in enumerate(R["streams"]):
            s.caches, s.pos = st["sets"][b] + [st["rings"][b]], st["pos_h"][b]
        self._rows = None
        self.eng.dflash_end()
        self._members = None

    def _rows_step(self, act):
        st = self.state
        if self._rows is None or [id(x) for x in self._rows["streams"]] != [id(x) for x in act]:
            self._rows_end()                                            # membership changed: a new state
            for s in act:
                self._spec_end(s)
            samp = DeviceSampler([s.req.temperature for s in act], [s.req.top_p for s in act], seed=self.seed)
            rs = self.eng.dflash_start_rows([s.tok for s in act], [s.caches for s in act], [s.pos for s in act],
                                            T=self.spec.rows_T, sampler=samp)
            for s in act:
                s.caches = None
            self._rows = {"streams": list(act), "st": rs, "sampler": samp}
            self._members = None
            st["rows_starts"] = st.get("rows_starts", 0) + 1
        R = self._rows
        T = R["st"]["T"]
        t0 = time.perf_counter()
        emitted, R["st"] = self.eng.dflash_step_rows(R["st"], R["sampler"], tuple(sorted(self.stop_ids)))
        t1 = time.perf_counter()
        n_tok = sum(len(e) for e in emitted)
        self.steps += 1
        st["steps"] = self.steps
        st["step_tokens"] = st.get("step_tokens", 0) + n_tok
        st["rows_steps"] = st.get("rows_steps", 0) + 1
        st["rows_tokens"] = st.get("rows_tokens", 0) + n_tok
        st["decode_s"] = st.get("decode_s", 0.0) + (t1 - t0)
        if self._t_step is not None:
            st["step_s"] = st.get("step_s", 0.0) + (t1 - self._t_step)
            st["steps_busy"] = st.get("steps_busy", 0) + 1
        self._t_step = t1
        for b, s in enumerate(act):                     # every row's bookkeeping first (a finish ends the state)
            em = emitted[b]
            self.spec.update(s, T, len(em))
            s.seen.extend([s.tok] + em[:-1])
            s.tok, s.pos, s.logits = em[-1], R["st"]["pos_h"][b], None
        for b, s in enumerate(act):
            for tok in emitted[b]:
                if s not in self.active:                                # finished mid-block (length / cancelled)
                    break
                self._emit(s, tok)

    # ---- warm-up
    def warmup(self, max_B=None, token=0):
        """Compile the batched decode programs for every batch size up to max_streams (zero caches, position 0)."""
        max_B = max_B or self.max_streams
        for B in range(1, max_B + 1):
            t = time.time()
            sets = [self.eng.alloc_caches(1) for _ in range(B)]
            samp = DeviceSampler([1.0] * B, [0.95] * B, seed=0)
            ids, lg, sets, pos = self.eng.decode_rows(np.full((B,), token, np.int32), sets, [0] * B, samp)
            jax.block_until_ready(ids)
            del sets
            self.log(f"warm-up batched decode B={B}: {time.time() - t:.0f}s")
