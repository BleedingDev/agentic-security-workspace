"""HBM-resident codebook-quantized experts (Unsloth UD-IQ*/Q2_K_XL GGUF) for the fully-jitted `Engine`.

Storage per sparse layer (chip-major, `P(AXIS)` on axis 0 hands each chip its slice; TP over moe_inter): every table
(gate_q, up_q, down_q) is a dict of PLANAR arrays `{plane: u32 [n_dev, E * rows_p, R]}` (glm53.planes): the GGUF block
fields split into dense 32-bit planes with the matrix row on the lane axis, so a Pallas kernel can dequantize an
expert in VMEM without byte slicing and XLA needs no relayout copies. gate/up: R = ml rows (this chip's slice of
moe_inter), inputs D; down: R = D rows, inputs ml.

Tables live in `params["layers"][i]["mlp"]`; `M.moe_dense` calls `ResidentFetch.apply`:
  decode (N*k <= gather_max_rows and fewer than `ragged_min_seq` tokens of one sequence): `glm53.pallas_moe`
  `moe_matvec_gu` + `moe_matvec` per (token, expert) slot (fused dequant + VPU matvec, the expert planes DMA'd by
  index), or the XLA path (dynamic slices + glm53.planes.dequant_planes + einsum) when use_pallas=False;
  prefill and multi-token verifies of one sequence: the grouped (ragged) kernels — every routed expert dequantized
  once, so the cost follows the DISTINCT experts (consecutive tokens share many; independent streams almost none).
Activations are permuted to the planes' "pm" input order (planes.pm_x / pm_flat); outputs are in natural row order.
"""
import dataclasses
import queue
import threading

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from glm53 import aot
from glm53 import iqquant as Q
from glm53 import model as M
from glm53 import planes as PL
from glm53 import pallas_moe as K

QK = Q.QK_K
KEYS = ("gate_q", "up_q", "down_q")


def pack_layer_from_gguf(gm, layer: int, n_dev: int, threads: int = 8):
    """Read one layer's routed experts from a `gguf_reader.GGUFModel` -> (tables dict, qtypes dict)."""
    out, qtypes = {}, {}
    for key, tname in (("gate_q", "gate"), ("up_q", "up"), ("down_q", "down")):
        name = f"blk.{layer}.ffn_{tname}_exps.weight"
        info = gm.info(name)
        ne0, ne1, E = info["dims"]                       # ne0 = input dim (blocks along it), ne1 = rows, E experts
        bb = Q.BLOCK_BYTES[info["type"]]
        raws = gm.read_experts(name, range(E), threads=threads)
        t = np.stack([np.frombuffer(r, np.uint8).reshape(ne1, ne0 // QK, bb) for r in raws])   # [E, rows, nblk, bb]
        if key == "down_q":                              # split blocks (the mi input dim) across chips
            t = t.reshape(E, ne1, n_dev, ne0 // QK // n_dev, bb).transpose(2, 0, 1, 3, 4)
        else:                                            # split rows (the mi output dim) across chips
            t = t.reshape(E, n_dev, ne1 // n_dev, ne0 // QK, bb).transpose(1, 0, 2, 3, 4)
        out[key] = pack_chip_planes(t, info["type"])
        qtypes[key] = info["type"]
    return out, qtypes


def pack_chip_planes(t, qtype, threads=None):
    """t uint8 [n_dev, E, rows, nblk, bb] -> {plane: u32 [n_dev, E*rows_p, rows]} (glm53.planes layout per chip).
    The chips' slices are packed in parallel threads (NumPy releases the GIL on the big slices; the packing, not the
    GGUF read, dominated the 9-minute build: the dataset mount serves ~650 MB/s to 8-16 readers, 2026-09-14)."""
    n = t.shape[0]
    threads = n if threads is None else max(1, int(threads))
    if threads > 1 and n > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(threads, n)) as ex:
            per_chip = list(ex.map(lambda c: PL.pack_planes(t[c], qtype), range(n)))
    else:
        per_chip = [PL.pack_planes(t[c], qtype) for c in range(n)]
    return {k: np.stack([pc[k] for pc in per_chip]) for k in per_chip[0]}


def hbm_bytes(tables: dict) -> int:
    """Bytes per chip of a layer's tables ({key: {plane: [n_dev, ...]}})."""
    return sum(int(a.nbytes) // int(a.shape[0]) for planes in tables.values() for a in planes.values())


class ResidentFetch:
    """Stored on the Engine; `M.moe_dense` calls `.apply(p, x, idx, w, layer, limit)` when present."""

    def __init__(self, qtypes: dict, D: int, ml: int, out_dtype=jnp.bfloat16, gather_max_rows: int = 64, chunk: int = 8,
                 use_pallas: bool = True, interpret: bool = False, sweep_mode: str = "ragged", tm: int = 32,
                 combine: str = "matmul"):
        self.qtypes = qtypes            # {layer: {"gate_q": "IQ2_S", "up_q": "IQ2_S", "down_q": "IQ3_S"}}
        self.rows = {"gate_q": ml, "up_q": ml, "down_q": D}     # logical rows per expert (this chip)
        self.nblk = {"gate_q": D // QK, "up_q": D // QK, "down_q": ml // QK}
        self.out_dtype = out_dtype
        self.gather_max_rows = gather_max_rows
        self.chunk = chunk
        self.use_pallas = use_pallas    # decode path: Pallas fused kernel (True) or XLA planes dequant + einsum
        self.interpret = interpret      # run the Pallas kernel in interpret mode (CPU tests)
        self.sweep_mode = sweep_mode    # prefill: "dense" = masked sweep over active experts, "ragged" = grouped GEMM
        self.tm = tm                    # ragged: rows per block (multiple of 16 for bf16)
        self.combine = combine          # ragged: per-token combine of slot outputs, "gather" (row gather) or "matmul"

    def _deq(self, layer, key, tbl, sel):
        """Dequantize the experts picked by `sel` from planes `tbl` -> [n, C*W, R] (pm order along inputs) in out_dtype."""
        qt = self.qtypes[layer][key]
        rows_p = PL.plane_rows(qt, self.nblk[key])
        picked = {k: sel(a, rows_p[k]).reshape(-1, rows_p[k], a.shape[-1]) for k, a in tbl.items()}
        return PL.dequant_planes(picked, qt, self.nblk[key]).astype(self.out_dtype)

    @staticmethod
    def _slice_experts(a, rows_p, flat):
        """a u32 [1, E*rows_p, R]; flat int32 [n] -> [n*rows_p, R] via n dynamic slices (DMA, never a generic gather
        and never a copy of the whole table)."""
        parts = [lax.dynamic_slice_in_dim(a, flat[j] * rows_p, rows_p, axis=1) for j in range(flat.shape[0])]
        return jnp.concatenate(parts, axis=1)[0]

    # Multi-token steps of ONE sequence (verifies): from `ragged_min_seq` tokens and up to `verify_max_rows` slots they
    # take the ragged kernels with these settings. v6e 2026-10-03 (j67-4c51, whole verify step): T=4 slot 18.2 / ragged
    # 17.9 ms, T=8 27.8 / 25.6, T=16 48.8 / 36.3 — below 8 tokens the per-slot decode kernels are as fast or faster.
    ragged_min_seq = 8
    verify_dtype = None        # activations/tiles (None = out_dtype; f32 = f32 tiles + HIGHEST dots: ~1.5x slower)
    verify_tm = 16             # rows per block (a verify has <= T rows per expert; 16 = the bf16 minimum)
    verify_lane_block = 4096   # lane block (the down kernel in one grid step per block: j66 -19 % MoE at T=8)
    verify_max_rows = 256
    group_min_seq = 2          # multi-token steps (2..ragged_min_seq-1 tokens of one sequence: the DFlash verify) decode
                               # each distinct expert once for all its slots (`_apply_kernel_grouped`, bit-identical).
                               # v6e 2026-10-04 (j126): DFlash step 15.95 -> 14.59 ms at T = 4 (with the chunked
                               # top-k). 0 = per slot.
    group_classes = ()         # unrolled body sizes below T (T itself is always one); () = 1..T
    group_dynamic_grid = True  # one kernel grid step per distinct expert (traced grid bound), no empty steps (j120b)
    group_loop = False         # groups above the fused classes: decode once into VMEM + a MAC loop over their slots
                               # (fixed code size; j120c: ~2 % slower than the classes at the same compile cost)
    group_rows = "all"         # a multi-token step of SEVERAL sequences (B rows x T tokens: the batched DFlash verify):
                               # "all" = one grouped call over every slot (an expert two rows share is decoded once;
                               # `group_rows_classes` + the decode-once loop), "row" = one grouped call per row (the
                               # one-row verify's kernels and shapes). v6e 2026-10-04 (j129, T = 4): DFlash step of
                               # 2 / 3 streams 24.1 / 34.7 ms per row, 23.4 / 32.7 all
    group_rows_classes = (1, 2, 3, 4)
    group_rows_vmem_mb = 64    # "all": every token's activation broadcasts in VMEM (2 MB each at D = 4096: 24 MB at 3 x 4)

    def apply(self, p, x, idx, w, layer, limit, seq_tokens=1):
        """x [N,D]; idx [N,k] int32 into E; w [N,k] fp32 -> partial MoE output [N,D] (TP-reduced by the caller).
        seq_tokens: consecutive tokens per sequence in x (a verify of T positions: T; B independent rows: 1)."""
        gate_q, up_q, down_q = p["gate_q"], p["up_q"], p["down_q"]         # {plane: [1, E*rows_p, R]} on this chip
        N, k = idx.shape
        if self.use_pallas and seq_tokens >= self.ragged_min_seq and N * k <= self.verify_max_rows:
            return self._apply_ragged_kernel(x, idx, w, layer, gate_q, up_q, down_q, limit, self.verify_dtype,
                                             self.verify_tm, self.verify_lane_block)
        if (self.use_pallas and self.group_min_seq and self.group_min_seq <= seq_tokens < self.ragged_min_seq
                and N > seq_tokens and N % seq_tokens == 0):         # B rows of T tokens (a batched verify)
            return self._apply_rows_grouped(x, idx, w, layer, gate_q, up_q, down_q, limit, seq_tokens)
        if N * k <= self.gather_max_rows:
            if self.use_pallas and self.group_min_seq and seq_tokens >= self.group_min_seq and seq_tokens == N:
                return self._apply_kernel_grouped(x, idx, w, layer, gate_q, up_q, down_q, limit)
            if self.use_pallas:
                return self._apply_kernel(x, idx, w, layer, gate_q, up_q, down_q, limit)
            return self._apply_gather(x, idx, w, layer, gate_q, up_q, down_q, limit)
        return self._apply_sweep(x, idx, w, layer, gate_q, up_q, down_q, limit)

    def _apply_kernel(self, x, idx, w, layer, gate_q, up_q, down_q, limit):
        N, k = idx.shape
        flat = idx.reshape(-1)
        qt_gu, qt_dn = self.qtypes[layer]["gate_q"], self.qtypes[layer]["down_q"]
        assert self.qtypes[layer]["up_q"] == qt_gu, self.qtypes[layer]
        Xt = PL.pm_x(qt_gu, x.astype(jnp.float32))                                      # [N, W, C]: per token
        g, u = K.moe_matvec_gu(gate_q, up_q, qt_gu, self.nblk["gate_q"], flat, Xt, interpret=self.interpret, k=k)
        h = M.swiglu_clamped(g, u, limit)                                               # [Nk, ml]
        y = K.moe_matvec(down_q, qt_dn, self.nblk["down_q"], flat, PL.pm_x(qt_dn, h),
                         interpret=self.interpret).reshape(N, k, -1)                    # [N, k, D]
        return jnp.einsum("nkd,nk->nd", y, w)

    def _apply_kernel_grouped(self, x, idx, w, layer, gate_q, up_q, down_q, limit, classes=None, loop=None, vmem_mb=32):
        """`_apply_kernel` for the T tokens of ONE sequence (a verify): the T*k slots are grouped by expert (a token's k
        experts are distinct, so a group holds at most T slots) and each distinct expert is decoded once for all its
        slots (`pallas_moe.group_plan`, `moe_matvec_gu_grouped` / `moe_matvec_grouped`: bit-identical to the per-slot
        kernels, outputs already in slot order). classes / loop override `group_classes` / `group_loop`."""
        N, k = idx.shape
        qt_gu, qt_dn = self.qtypes[layer]["gate_q"], self.qtypes[layer]["down_q"]
        classes = self.group_classes if classes is None else classes
        loop = self.group_loop if loop is None else loop
        eids, cnt, slots = K.group_plan(idx)
        g, u = K.moe_matvec_gu_grouped(gate_q, up_q, qt_gu, self.nblk["gate_q"], eids, cnt, slots,
                                       PL.pm_x(qt_gu, x.astype(jnp.float32)), k=k, classes=classes,
                                       interpret=self.interpret, dynamic_grid=self.group_dynamic_grid, loop=loop,
                                       vmem_mb=vmem_mb)
        h = M.swiglu_clamped(g, u, limit)                                               # [Nk, ml]
        y = K.moe_matvec_grouped(down_q, qt_dn, self.nblk["down_q"], eids, cnt, slots, PL.pm_x(qt_dn, h),
                                 classes=classes, interpret=self.interpret,
                                 dynamic_grid=self.group_dynamic_grid, loop=loop, vmem_mb=vmem_mb).reshape(N, k, -1)
        return jnp.einsum("nkd,nk->nd", y, w)

    def _apply_rows_grouped(self, x, idx, w, layer, gate_q, up_q, down_q, limit, T):
        """B independent rows of T consecutive tokens each (x [B*T, D], row-major): `group_rows` "row" = the grouped
        kernels once per row (neighbouring tokens of a row share experts, independent rows hardly do), "all" = one
        grouped call over all B*T*k slots (groups of up to B*T slots: the unrolled classes `group_rows_classes`, larger
        groups through the decode-once loop)."""
        N = x.shape[0]
        if self.group_rows == "all":
            return self._apply_kernel_grouped(x, idx, w, layer, gate_q, up_q, down_q, limit,
                                              classes=self.group_rows_classes, loop=True,
                                              vmem_mb=self.group_rows_vmem_mb)
        return jnp.concatenate([self._apply_kernel_grouped(x[r:r + T], idx[r:r + T], w[r:r + T], layer, gate_q, up_q,
                                                           down_q, limit) for r in range(0, N, T)], 0)

    def _apply_gather(self, x, idx, w, layer, gate_q, up_q, down_q, limit):
        """XLA decode path on the planar tables (no Pallas): per-expert dynamic slices + dequant + einsum."""
        N, k = idx.shape
        flat = idx.reshape(-1)
        qt_gu, qt_dn = self.qtypes[layer]["gate_q"], self.qtypes[layer]["down_q"]
        sel = lambda a, rows_p: self._slice_experts(a, rows_p, flat)
        g = self._deq(layer, "gate_q", gate_q, sel)                                     # [Nk, D_pm, ml]
        u = self._deq(layer, "up_q", up_q, sel)
        d = self._deq(layer, "down_q", down_q, sel)                                     # [Nk, ml_pm, D]
        xs = jnp.repeat(PL.pm_flat(qt_gu, x).astype(self.out_dtype), k, axis=0)         # [Nk, D_pm]
        h = M.swiglu_clamped(jnp.einsum("nd,ndm->nm", xs, g), jnp.einsum("nd,ndm->nm", xs, u), limit)   # [Nk, ml]
        h = PL.pm_flat(qt_dn, h).astype(d.dtype)
        y = jnp.einsum("nm,nmd->nd", h, d).astype(jnp.float32).reshape(N, k, -1)
        return jnp.einsum("nkd,nk->nd", y, w)

    def _apply_sweep(self, x, idx, w, layer, gate_q, up_q, down_q, limit):
        if self.use_pallas and self.sweep_mode == "ragged":
            return self._apply_ragged_kernel(x, idx, w, layer, gate_q, up_q, down_q, limit)
        if self.use_pallas:
            return self._apply_sweep_kernel(x, idx, w, layer, gate_q, up_q, down_q, limit)
        return self._apply_sweep_xla(x, idx, w, layer, gate_q, up_q, down_q, limit)

    def _apply_ragged_kernel(self, x, idx, w, layer, gate_q, up_q, down_q, limit, dtype=None, tm=None, lane_block=None):
        """Prefill through the grouped GEMM kernels: (token, expert) slots sorted by expert into `tm`-row blocks,
        each expert dequantized once and multiplied against its own rows only; the per-token combine gathers each
        token's k slot rows (or, `combine="matmul"`, multiplies by a [T, Rp] weight matrix). dtype (default out_dtype)
        = activations, VMEM tiles and outputs; f32 = f32 tiles + HIGHEST-precision dots (a verify close to the decode
        kernels' f32 numerics)."""
        qt_gu, qt_dn = self.qtypes[layer]["gate_q"], self.qtypes[layer]["down_q"]
        E = gate_q["qs"].shape[1] // PL.plane_rows(qt_gu, self.nblk["gate_q"])["qs"]
        T, k = idx.shape
        tm = tm or self.tm
        plan = K.ragged_plan(idx, w, E, tm)
        tok = plan["tok_row"]
        dtype = dtype or self.out_dtype
        xk = K.pm_mxu(qt_gu, x.astype(jnp.float32)).astype(dtype)                             # [T, D]
        xs = jnp.take(xk, jnp.maximum(tok, 0), axis=0, mode="clip")
        xs = jnp.where((tok >= 0)[:, None], xs, jnp.zeros((), xs.dtype))                       # [Rp, D]
        h = K.moe_ragged_gateup(gate_q, up_q, qt_gu, self.nblk["gate_q"], plan["blk_expert"], plan["n_blocks"], xs,
                                limit, tm=tm, interpret=self.interpret, lane_block=lane_block)   # [Rp, ml]
        hk = K.pm_mxu(qt_dn, h.astype(jnp.float32)).astype(dtype)
        y = K.moe_ragged_down(down_q, qt_dn, self.nblk["down_q"], plan["blk_expert"], plan["n_blocks"], hk,
                              tm=tm, interpret=self.interpret, lane_block=lane_block)            # [Rp, D]
        if self.combine == "matmul" and dtype != jnp.float32:     # f32 (verify): the slot path's combine below
            wc = (tok[None, :] == jnp.arange(T, dtype=jnp.int32)[:, None]).astype(jnp.float32) * plan["w_row"][None, :]
            return jnp.dot(wc.astype(y.dtype), y, preferred_element_type=jnp.float32)
        ys = jnp.take(y, plan["row_slot"].reshape(-1), axis=0, mode="clip").reshape(T, k, -1).astype(jnp.float32)
        return jnp.einsum("tkd,tk->td", ys, w.astype(jnp.float32))

    SWEEP_TOKENS = 512     # token chunk per kernel launch (bounds the [E, T, ml] intermediate to ~75 MB per chip)

    def _apply_sweep_kernel(self, x, idx, w, layer, gate_q, up_q, down_q, limit):
        """Prefill through the Pallas sweep kernels: every routed expert is dequantized once into VMEM tiles and MXU-
        multiplied against all tokens; experts no token routes to are skipped (scalar-prefetched active slots)."""
        qt_gu, qt_dn = self.qtypes[layer]["gate_q"], self.qtypes[layer]["down_q"]
        E = gate_q["qs"].shape[1] // PL.plane_rows(qt_gu, self.nblk["gate_q"])["qs"]
        N = x.shape[0]
        ids, n_act = K.active_slots(idx, E)
        rw = jnp.einsum("nk,nke->ne", w.astype(jnp.float32),
                        (idx[:, :, None] == ids[None, None, :]).astype(jnp.float32))          # [N, E] per slot
        outs = []
        for t0 in range(0, N, self.SWEEP_TOKENS):
            xs = x[t0:t0 + self.SWEEP_TOKENS]
            xk = K.pm_mxu(qt_gu, xs.astype(jnp.float32)).astype(self.out_dtype)                # [T, D]
            h = K.moe_sweep_gateup(gate_q, up_q, qt_gu, self.nblk["gate_q"], ids, n_act, xk, limit,
                                   interpret=self.interpret)                                    # [E, T, ml] bf16
            hw = h.astype(jnp.float32) * jnp.swapaxes(rw[t0:t0 + self.SWEEP_TOKENS], 0, 1)[:, :, None]
            hk = K.pm_mxu(qt_dn, hw).astype(self.out_dtype)                                     # [E, T, ml]
            outs.append(K.moe_sweep_down(down_q, qt_dn, self.nblk["down_q"], ids, n_act, hk, interpret=self.interpret))
        return outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=0)

    def _apply_sweep_xla(self, x, idx, w, layer, gate_q, up_q, down_q, limit):
        E = gate_q["qs"].shape[1] // PL.plane_rows(self.qtypes[layer]["gate_q"], self.nblk["gate_q"])["qs"]
        CH = self.chunk
        assert E % CH == 0, (E, CH)
        qt_gu, qt_dn = self.qtypes[layer]["gate_q"], self.qtypes[layer]["down_q"]
        onehot = jax.nn.one_hot(idx, E, dtype=jnp.float32)                  # [N,k,E]
        rw = jnp.einsum("nk,nke->ne", w.astype(jnp.float32), onehot)        # [N,E] routing weight (0 if unused)
        xg = PL.pm_flat(qt_gu, x).astype(self.out_dtype)                    # [N, D_pm]

        def body(acc, c):
            c0 = c * CH
            sel = lambda a, rows_p: lax.dynamic_slice_in_dim(a, c0 * rows_p, CH * rows_p, axis=1)[0]
            g = self._deq(layer, "gate_q", gate_q, sel)                                          # [CH, D_pm, ml]
            u = self._deq(layer, "up_q", up_q, sel)
            d = self._deq(layer, "down_q", down_q, sel)                                          # [CH, ml_pm, D]
            h = M.swiglu_clamped(jnp.einsum("nd,edm->nem", xg, g), jnp.einsum("nd,edm->nem", xg, u), limit)  # [N,CH,ml]
            rws = lax.dynamic_slice_in_dim(rw, c0, CH, 1)                                        # [N, CH]
            h = PL.pm_flat(qt_dn, h.astype(jnp.float32) * rws[..., None]).astype(d.dtype)
            return acc + jnp.einsum("nem,emd->nd", h, d).astype(jnp.float32), None

        acc, _ = lax.scan(body, jnp.zeros((xg.shape[0], x.shape[1]), jnp.float32), jnp.arange(E // CH))
        return acc


# ------------------------------------------------------------------------------------- per-layer-kind engine
from glm53.engine import Engine, AXIS, R, shard_map, cache_specs, device_sample, device_probs, spec_accept, DeviceSampler   # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P           # noqa: E402


class _Reaper(threading.Thread):
    """Drops objects on a helper thread. A step leaves ~300 jax arrays behind (consumed caches, histories, the
    previous state); dropping one costs ~11-20 us of host time on the 8-chip mesh (j107), so the main thread hands
    them over instead of paying between its dispatches (`ResidentLayerEngine.reap`)."""

    def __init__(self):
        super().__init__(daemon=True, name="glm53-reaper")
        self.q = queue.SimpleQueue()
        self.start()

    def run(self):
        while True:
            item = self.q.get()
            del item

    def drop(self, *objs):
        self.q.put(objs)


class _HostShards:
    """A sharded device array parked in host memory as its per-device shards (a pytree leaf)."""

    def __init__(self, shards, devices, shape, dtype, sharding):
        self.shards, self.devices, self.shape, self.dtype, self.sharding = shards, devices, shape, dtype, sharding
        self.nbytes = sum(int(x.nbytes) for x in shards)

    @classmethod
    def of(cls, a):
        sh = sorted(a.addressable_shards, key=lambda s: s.device.id)
        return cls([np.asarray(s.data) for s in sh], [s.device for s in sh], a.shape, a.dtype, a.sharding)

    def to_device(self):
        parts = [jax.device_put(x, d) for x, d in zip(self.shards, self.devices)]
        return jax.make_array_from_single_device_arrays(self.shape, self.sharding, parts)


class ResidentLayerEngine(Engine):
    """Resident experts with ONE jitted program per layer *kind* (attention type, MLP type, expert quant types,
    indexer presence) instead of one 45-layer program: the layer's params are program arguments, so ~6 programs cover
    all layers and the compile stays small (a single unrolled program with 42 codebook dequants got the XLA compiler
    SIGKILLed on the Kaggle host). Per step: embed program, 45 layer programs, head program — all device-resident,
    no host gathers.

    Prefill is CHUNKED: the caches are allocated up front (`alloc_caches`) and the prompt is fed in pieces of
    `prefill_piece` tokens (the last one padded to a bucket) through cache-carrying programs, so no temporary scales
    with the prompt length; `prefill(tokens, caches, pos0)` continues an existing context (prefix caching). Cache
    buffers are DONATED to the layer programs (in-place update; never reuse a cache object after passing it in —
    `copy_caches` for snapshots)."""

    def __init__(self, cfg, params, expert_fetch, devices=None, max_len: int = 4096, layers_per_program: int = 1,
                 layers_per_program_prefill: int = 1, int8_nonexpert: bool = False, seq_shard: bool = False,
                 q_block: int = 128, prefill_piece: int = 2048, cache_q8: bool = False):
        super().__init__(cfg, params, None, devices, max_len, expert_fetch=expert_fetch, int8_nonexpert=int8_nonexpert,
                         seq_shard=seq_shard, q_block=q_block, cache_q8=cache_q8)
        self._progs = {}
        self._allocs = {}
        self.launches = 0
        assert prefill_piece in self.PREFILL_BUCKETS, prefill_piece
        self.prefill_piece = min(prefill_piece, max_len)
        # consecutive layer groups; a group's program is keyed by the tuple of its layer kinds (dispatch ≈ 2 ms per
        # launch on the Kaggle TPU VM, so 45 single-layer launches cost ~100 ms/token). Prefill keeps 1 layer per
        # program by default: a 4-layer prefill program (expert sweep scans inside) did not finish compiling in 50 min.
        self.lpp = {"decode": layers_per_program, "prefill": layers_per_program_prefill}
        self.groups = {m: [list(range(a, min(a + k, cfg.n_layers))) for a in range(0, cfg.n_layers, k)]
                       for m, k in self.lpp.items()}

    def _kind(self, i):
        qt = tuple(sorted(self.expert_fetch.qtypes.get(i, {}).items())) if self.cfg.mlp_types[i] == "sparse" else ()
        return (self.cfg.layer_types[i], self.cfg.mlp_types[i], qt, "indexer" in self.params["layers"][i]["attn"])

    def _prog_embed(self, B, T, override=False):
        """Token embeddings broadcast to the hc residual streams [B,T,hc,D]; with override=True the extra arguments
        (vectors [B,T,D], mask [B,T] bool) replace the table rows where mask is set (image tokens: glm53.vision)."""
        key = ("embed", B, T, override)
        if key not in self._progs:
            def prog(embed, tokens, *ov):
                x = self._embed(embed, tokens).astype(self.lcfg.dtype)
                if override:
                    vec, mask = ov
                    x = jnp.where(mask[..., None], vec.astype(x.dtype), x)
                return jnp.broadcast_to(x[:, :, None, :], (B, T, self.cfg.hc, x.shape[-1]))
            in_specs = (self.specs["embed"], R) + ((R, R) if override else ())
            self._progs[key] = aot.jit(key, shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=R, check_vma=False))
        return self._progs[key]

    def _prog_head(self, B, T, sample=False):
        """Logits [B,V] of the last valid position; with sample=True also the device-sampled token ids [B]
        (extra args: temperature, top_p, key — `DeviceSampler.bind(mesh)`), returned as (ids, logits, next key)."""
        key = ("head", B, T, sample)
        if key not in self._progs:
            def prog(norm, lm_head, streams, length, *samp):
                h = M.rmsnorm(streams.mean(2).astype(self.lcfg.dtype), norm, self.cfg.eps)
                last = lax.dynamic_index_in_dim(h, length - 1, axis=1, keepdims=False)
                zl = self._logits_local(lm_head, last)
                logits = lax.all_gather(zl, AXIS, axis=-1, tiled=True)
                if not sample:
                    return logits
                temp, top_p, k = samp
                k1, k2 = jax.random.split(k)
                return device_sample(zl, temp, top_p, k1), logits, k2
            in_specs = (R, self.specs["lm_head"], R, R) + ((R, R, R) if sample else ())
            self._progs[key] = aot.jit(key, shard_map(prog, mesh=self.mesh, in_specs=in_specs,
                                                 out_specs=(R, R, R) if sample else R, check_vma=False))
        return self._progs[key]

    def _hist_spec(self, i, kin=False):
        """Sharding of the per-token histories a speculative verify step returns for layer i (None: no history);
        kin=True: the KDA layers' token update inputs instead of their states (`M.kda_recurrent(hist="kin")`)."""
        if self.cfg.layer_types[i] == "linear_attention":
            return {"kin_hist" if kin else "state_hist": P(None, None, AXIS), "conv_hist": P(None, None, None, AXIS)}
        if "indexer" in self.params["layers"][i]["attn"]:
            return {"tk_hist": R, "tg_hist": R}
        return None

    def _hist_layout(self, layers, kin=False):
        """How a group program returns the per-token histories of a verify: {hist key: positions j in the group that
        have it}, one array stacked over those layers per key (not one per layer: every array a program allocates for
        its outputs costs ~17.5 us of host time per dispatch on the 8-chip mesh, j101 2026-10-04)."""
        lay = {}
        for j, i in enumerate(layers):
            for k in (self._hist_spec(i, kin) or {}):
                lay.setdefault(k, []).append(j)
        return {k: tuple(v) for k, v in sorted(lay.items())}

    def _prog_group(self, layers, B, T, has_cache, use_recurrent, hist=False, capture=(), pend=None, embed=False):
        """capture = positions j in the group whose output's stream mean is returned as a last output, stacked
        [len(capture), B, T, D] (DFlash features); hist=True / "kin" returns the per-token histories stacked per key
        (`_hist_layout`; "kin": the KDA layers return their tokens' update inputs and leave their states as before the
        block, `M.kda_core`). pend = T of the previous verify whose rollback is still pending (`dflash_step`): the
        program then also takes that verify's histories of this group (DONATED: this verify's histories reuse their
        buffers) and its n_keep, and first rolls the small state entries of its layers back to the accepted prefix
        (the rollback program's own op, `M.rollback_states`), so no separate rollback program runs between the steps.
        embed=True: the streams argument is (embedding table, tokens [B,T]) and the program embeds them itself (the
        step's first group: no separate embed program and its output between the dispatches)."""
        kinds = tuple(self._kind(i) for i in layers)
        key = ("group", kinds, B, T, has_cache, use_recurrent, hist) + ((capture,) if capture else ()) + \
            ((("pend", pend),) if pend else ()) + (("embed",) if embed else ())
        if key not in self._progs:
            lts = [self.cfg.layer_types[i] for i in layers]
            mts = [self.cfg.mlp_types[i] for i in layers]
            css = [self._cache_spec(i) for i in layers]
            kin = hist == "kin"
            lay = self._hist_layout(layers, kin)
            hss = {k: P(None, *self._hist_spec(layers[js[0]], kin)[k]) for k, js in lay.items()}
            rep = list(layers)                     # representative layer ids (only used to look up expert qtypes)

            from glm53 import quant8 as Q8

            def prog(ps, streams, pos0, length, caches, hg=None, nk=None):
                new, hs, feats = [], [], []
                if embed:
                    streams = self._embed_streams(*streams)
                if pend:
                    caches = list(caches)
                    roll = M.rollback_states([{k: hg[k][js.index(j)] for k, js in lay.items() if j in js} or None
                                              for j in range(len(ps))], nk,
                                             [{k_: c[k_].dtype for k_ in c} for c in caches], caches)
                    caches = [c if r_ is None else {**c, **r_} for c, r_ in zip(caches, roll)]
                for j, p in enumerate(ps):
                    p = Q8.dequant_tree(p, self.lcfg.dtype)          # int8 non-expert matrices -> bf16 (fused into dots)
                    fetch = self.expert_fetch if mts[j] == "sparse" else None
                    streams, nc = M.decoder_layer(p, streams, self.lcfg, lts[j], mts[j],
                                                  caches[j] if has_cache else None, pos0, use_recurrent, fetch, "auto",
                                                  layer=rep[j], cap=self.max_len, length=None if use_recurrent else length,
                                                  hist=hist)
                    nc, h = M.split_hist(nc)
                    new.append(nc); hs.append(h)
                    if j in capture:
                        feats.append(self._stream_mean(streams))
                out = (streams, new)
                if hist:
                    out += ({k: jnp.stack([hs[j][k] for j in js]) for k, js in lay.items()},)
                return out + ((jnp.stack(feats),) if capture else ())
            in_specs = ([self.specs["layers"][i] for i in layers], (self.specs["embed"], R) if embed else R, R, R,
                        css if has_cache else None) + ((hss, R) if pend else ())
            out_specs = (R, css) + ((hss,) if hist else ()) + ((R,) if capture else ())
            sm = shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
            self._progs[key] = aot.jit(key, sm, donate_argnums=((4,) if has_cache else ()) + ((5,) if pend else ()))
        return self._progs[key]

    def _embed_streams(self, table, tokens):
        """Inside a program: tokens [B,T] -> the hc residual streams [B,T,hc,D] (what `_prog_embed` returns)."""
        x = self._embed(table, tokens).astype(self.lcfg.dtype)
        return jnp.broadcast_to(x[:, :, None, :], x.shape[:2] + (self.cfg.hc, x.shape[-1]))

    def _stream_mean(self, streams):
        """DFlash feature of a layer: the mean of its 4 output residual streams (SGLang's hc_contract) [B,T,D]; a
        verify shorter than the drafter's block is padded to the block (zero rows), so the next draft program takes
        the same shape after a verify of any T (one draft program per block size)."""
        m = streams.astype(jnp.float32).mean(2).astype(self.lcfg.dtype)
        T, blk = m.shape[1], self.dflash.cfg.block
        return jnp.pad(m, ((0, 0), (0, blk - T), (0, 0))) if T < blk else m

    def _prog_stream_mean(self, B, T):
        key = ("stream_mean", B, T)
        if key not in self._progs:
            self._progs[key] = aot.jit(key, shard_map(self._stream_mean, mesh=self.mesh, in_specs=(R,), out_specs=R,
                                                 check_vma=False))
        return self._progs[key]

    def _prog_hidden(self, B, T):
        """Final-norm hidden states of every position [B,T,D] (MTP prefill input)."""
        key = ("hidden", B, T)
        if key not in self._progs:
            def prog(norm, streams):
                return M.rmsnorm(streams.mean(2).astype(self.lcfg.dtype), norm, self.cfg.eps)
            self._progs[key] = aot.jit(key, shard_map(prog, mesh=self.mesh, in_specs=(R, R), out_specs=R, check_vma=False))
        return self._progs[key]

    def _prog_head_all(self, B, T, sample=False):
        """Logits [B,T,V] and hidden [B,T,D] of every position (speculative verify; T small); with sample=True
        (extra args as in `_prog_head`) also the device-sampled ids [B,T]: (ids, logits, hidden, next key)."""
        key = ("head_all", B, T, sample)
        if key not in self._progs:
            def prog(norm, lm_head, streams, *samp):
                h = M.rmsnorm(streams.mean(2).astype(self.lcfg.dtype), norm, self.cfg.eps)
                zl = self._logits_local(lm_head, h)
                logits = lax.all_gather(zl, AXIS, axis=-1, tiled=True)
                if not sample:
                    return logits, h
                temp, top_p, k = samp
                k1, k2 = jax.random.split(k)
                ids = device_sample(zl.reshape(B * T, -1), temp, top_p, k1)
                return ids.reshape(B, T), logits, h, k2
            in_specs = (R, self.specs["lm_head"], R) + ((R, R, R) if sample else ())
            self._progs[key] = aot.jit(key, shard_map(prog, mesh=self.mesh, in_specs=in_specs,
                                                 out_specs=(R, R, R, R) if sample else (R, R), check_vma=False))
        return self._progs[key]

    def _prog_rollback(self, B, T, kin=False):
        """the per-group stacked histories of a T-token verify (`_verify_layers`) + n_keep (0..T; 0 undoes the verify)
        + the current small state entries (`_hist_cur`, DONATED: the outputs reuse their buffers, so the dispatch
        allocates nothing) -> the small state entries (KDA state/conv, pool tails) as after the first n_keep tokens,
        per layer; spliced into the caches by the caller (the big positional arrays are not touched). kin=True: the
        histories hold the KDA tokens' update inputs and the KDA states are those before the block (replayed)."""
        key = ("rollback", B, T) + (("kin",) if kin else ())
        if key not in self._progs:
            groups = self.groups["decode"]
            lays = [self._hist_layout(g, kin) for g in groups]
            hgs = [{k: P(None, *self._hist_spec(g[js[0]], kin)[k]) for k, js in lay.items()} for g, lay in zip(groups, lays)]
            hss = [self._hist_spec(i, kin) for i in range(self.cfg.n_layers)]
            oss = [None if h is None else {M.HIST_KEYS[k]: P(*v[1:]) for k, v in h.items()} for h in hss]   # drop the T axis
            dts = [{M.HIST_KEYS[k]: self.cfg.dtype if M.HIST_KEYS[k] != "state" else jnp.float32 for k in h} if h else None
                   for h in hss]

            def prog(hg, n, cur):
                per = [None] * self.cfg.n_layers
                for g, lay, h in zip(groups, lays, hg):
                    for k, js in lay.items():
                        for s_, j in enumerate(js):
                            per[g[j]] = {**(per[g[j]] or {}), k: h[k][s_]}
                return M.rollback_states(per, n, dts, cur)
            sm = shard_map(prog, mesh=self.mesh, in_specs=(hgs, R, oss), out_specs=oss, check_vma=False)
            self._progs[key] = aot.jit(key, sm, donate_argnums=(2,))
        return self._progs[key]

    def _hist_cur(self, caches, kin=False):
        """The small state entries of a verify's output caches that `_prog_rollback` replaces (and takes donated)."""
        out = []
        for i, c in enumerate(caches[:self.cfg.n_layers]):
            h = self._hist_spec(i, kin)
            out.append(None if h is None else {M.HIST_KEYS[k]: c[M.HIST_KEYS[k]] for k in h})
        return out

    mtp = None                       # MTP (NextN) layer params on device, or None (see set_mtp)
    dflash = None                    # DFlash 2 drafter (glm53.dflash.Drafter), or None (see set_dflash)
    reap = True                      # DFlash steps drop the previous step's arrays on a helper thread (`_Reaper`):
                                     # v6e 2026-10-05 (j145) host work per step 12.0 -> 9.9 ms (one stream), 18.5 ->
                                     # 14.0 (3 streams); the wall time was device-bound either way
    run_ahead = False                # DFlash steps dispatch the NEXT block's verify before the host reads this block's
                                     # tokens (all its inputs are device arrays), so the device never waits for the
                                     # host's turnaround; a stream that stops undoes that verify (`dflash_settle`).
                                     # Off: measured on the v6e 2026-10-05 (j143 / j126 / j102 / j131) the step and the
                                     # served rates were unchanged (the queued draft already covers the turnaround and
                                     # the remaining wall - busy gap is ~20 us per program launch), while every stream
                                     # end wastes one verify and needs the undo program.

    def _drop(self, *objs):
        """Release `objs` (the previous step's arrays) now, on the reaper thread when `reap` is set."""
        if self.reap:
            if getattr(self, "_reaper", None) is None:
                self._reaper = _Reaper()
            self._reaper.drop(*objs)

    def _cache_spec(self, i):
        if i == self.cfg.n_layers and self.dflash is not None:                 # the drafter's context ring
            from glm53.dflash import RING_SPECS
            return dict(RING_SPECS)
        if i == self.cfg.n_layers:                                             # the MTP layer's cache entry
            mla = P(None, AXIS) if self.lcfg.seq_shard > 1 else R
            spec = {"c": mla, "pk": mla, "tk": R, "tg": R, "h": R}
            if self.cfg.cache_q8:
                spec["cs"] = mla
            return spec
        return super()._cache_spec(i)

    def _is_kda(self, i):
        return i < self.cfg.n_layers and self.cfg.layer_types[i] == "linear_attention"

    def alloc_caches(self, B):
        """Zero caches for a new context at the engine's capacity, allocated directly on the chips (+ the MTP
        layer's entry, index n_layers, when an MTP layer is installed: its attention cache and the carried final
        hidden state "h" [B,1,D] of the last processed position; or the DFlash drafter's context ring there)."""
        cfg = self.cfg
        out = []
        n_entries = cfg.n_layers + (1 if self.mtp is not None or self.dflash is not None else 0)
        for i in range(n_entries):
            spec = self._cache_spec(i)
            if i == cfg.n_layers and self.dflash is not None:
                dc = self.dflash.cfg
                shapes = {k: ((dc.n_layers, B, dc.window, dc.kv_heads, dc.hd), dc.dtype) for k in ("k", "v")}
                shapes["p"] = ((B, dc.window), jnp.int32)               # slot positions + 1 (0 = empty)
            elif i == cfg.n_layers:
                shapes = {k: (sh, M.cache_dtype(cfg, k, cfg.dtype)) for k, sh in M.mla_cache_shapes(cfg, B, self.max_len, True).items()}
                shapes["h"] = ((B, 1, cfg.hidden), cfg.dtype)
            elif cfg.layer_types[i] == "linear_attention":
                shapes = {"conv": ((B, cfg.conv_k - 1, 3 * cfg.kda_heads * cfg.kda_hd), cfg.dtype),
                          "state": ((B, cfg.kda_heads, cfg.kda_hd, cfg.kda_hd), jnp.float32)}
            else:
                has_ix = "indexer" in self.params["layers"][i]["attn"]
                shapes = {k: (sh, M.cache_dtype(cfg, k, cfg.dtype)) for k, sh in M.mla_cache_shapes(cfg, B, self.max_len, has_ix).items()}
            c = {}
            for k, (shape, dtype) in shapes.items():
                key = (shape, jnp.dtype(dtype), spec[k])
                if key not in self._allocs:
                    self._allocs[key] = jax.jit(lambda shape=shape, dtype=dtype: jnp.zeros(shape, dtype),
                                                out_shardings=NamedSharding(self.mesh, spec[k]))
                c[k] = self._allocs[key]()
            out.append(c)
        return out

    @staticmethod
    def copy_caches(caches):
        """Device-side copy (the programs update caches in place)."""
        return jax.tree.map(jnp.copy, caches)

    def _prefix_rows(self, n_tokens):
        """Local cache rows that positions < n_tokens can occupy (interleaved layout: 4-token pools round-robin)."""
        kp, n = self.cfg.idx_kpool, self.lcfg.seq_shard
        return -(-n_tokens // (kp * n)) * kp if n > 1 else n_tokens

    def snapshot_prefix(self, caches, n_tokens, rows_bucket=1):
        """Compact snapshot of a context of n_tokens (the KDA states and only the used MLA/indexer cache rows), e.g.
        a shared system prompt: ~2 KB/token/chip instead of a full cache set. Restore with `restore_prefix`.
        `rows_bucket` rounds the row count up (bounds the number of slice/restore programs compiled for arbitrary
        context lengths; the extra rows are never read back)."""
        rows = self._prefix_rows(n_tokens)
        cap = self.max_len // max(self.lcfg.seq_shard, 1)
        rows = min(-(-rows // rows_bucket) * rows_bucket, cap)
        key = ("snap", rows)
        if key not in self._progs:
            self._progs[key] = {}
        out = []
        for i, c in enumerate(caches):
            if self._is_kda(i):
                out.append(jax.tree.map(jnp.copy, c)); continue
            spec = self._cache_spec(i)
            snap = {}
            for k, a in c.items():
                if k not in M.ROW_KEYS:                                       # tail buffers: whole copies
                    snap[k] = jnp.copy(a); continue
                rk = rows // self.cfg.idx_kpool if k == "pk" else rows
                pk = (k, a.shape, str(a.dtype))
                if pk not in self._progs[key]:
                    sp = spec[k]
                    self._progs[key][pk] = aot.jit((key, pk), shard_map(lambda x, rk=rk: x[:, :rk], mesh=self.mesh, in_specs=(sp,),
                                                             out_specs=sp, check_vma=False))
                snap[k] = self._progs[key][pk](a)
            out.append(snap)
        return {"n_tokens": n_tokens, "rows": rows, "caches": out}

    def restore_prefix(self, snap, B=1):
        """Fresh caches at full capacity holding the snapshot's context (the snapshot stays valid)."""
        caches = self.alloc_caches(B)
        rows = snap.get("rows") or self._prefix_rows(snap["n_tokens"])
        key = ("restore", rows)
        if key not in self._progs:
            self._progs[key] = {}
        out = []
        for i, c in enumerate(caches):
            sc = snap["caches"][i]
            if self._is_kda(i):
                out.append(jax.tree.map(jnp.copy, sc)); continue
            spec = self._cache_spec(i)
            new = {}
            for k, a in c.items():
                if k not in M.ROW_KEYS:
                    new[k] = jnp.copy(sc[k]); continue
                pk = (k, a.shape, str(a.dtype))
                if pk not in self._progs[key]:
                    sp = spec[k]
                    self._progs[key][pk] = aot.jit((key, pk), shard_map(lambda x, p: lax.dynamic_update_slice_in_dim(x, p, 0, axis=1),
                                                             mesh=self.mesh, in_specs=(sp, sp), out_specs=sp, check_vma=False),
                                                   donate_argnums=(0,))
                new[k] = self._progs[key][pk](a, sc[k])
            out.append(new)
        return out

    @staticmethod
    def snapshot_to_host(snap):
        """Move a compact snapshot into host memory so it holds no HBM; a parked context costs ~17 KB/token on the
        host. Every array is kept as its per-device shards (no host-side assembly of the 8 shards, which ran at
        ~1.5 GB/s); `snapshot_from_host` brings it back for `restore_prefix`."""
        host = jax.tree.map(_HostShards.of, snap["caches"])
        n_bytes = sum(h.nbytes for h in jax.tree.leaves(host, is_leaf=lambda x: isinstance(x, _HostShards)))
        return {"n_tokens": snap["n_tokens"], "rows": snap["rows"], "caches": host, "bytes": n_bytes}

    @staticmethod
    def snapshot_from_host(hsnap):
        """Device snapshot (as `snapshot_prefix` returns it) from a host snapshot; the host copy stays valid."""
        caches = jax.tree.map(lambda h: h.to_device(), hsnap["caches"], is_leaf=lambda x: isinstance(x, _HostShards))
        return {"n_tokens": hsnap["n_tokens"], "rows": hsnap["rows"], "caches": caches}

    def _capture_plan(self, g, capture):
        """DFlash feature capture for layer group g: (positions in the middle of the group, whose program returns
        their stream mean; whether the group's last layer is captured: the mean of the group's output)."""
        if not capture:
            return (), False
        js = [j for j, i in enumerate(g) if i in self.dflash.cfg.target_layers]
        return tuple(j for j in js if j < len(g) - 1), bool(js) and js[-1] == len(g) - 1

    def _run_layers(self, tokens, caches, pos0, use_recurrent, length, override=None, capture=False):
        """embed + all layer groups -> (final residual streams, new caches[, DFlash features: the stream mean after
        each of the drafter's target layers, [B,T,D] each, with capture=True]). `override` = (vectors [B,T,D],
        mask [B,T]) replaces the token embeddings where mask is set."""
        B, T = tokens.shape
        if override is None:
            streams = self._prog_embed(B, T)(self.params["embed"], jnp.asarray(tokens))
        else:
            streams = self._prog_embed(B, T, True)(self.params["embed"], jnp.asarray(tokens), jnp.asarray(override[0]),
                                                   jnp.asarray(override[1]))
        new_caches, feats = [], []
        has_cache = caches is not None
        pos0, length = self.scalar(pos0), self.scalar(length)
        for g in self.groups["decode" if use_recurrent else "prefill"]:
            mid, last = self._capture_plan(g, capture)
            prog = self._prog_group(g, B, T, has_cache, use_recurrent, capture=mid)
            out = prog([self.params["layers"][i] for i in g], streams, pos0, length,
                       [caches[i] for i in g] if has_cache else None)
            streams, ncs = out[:2]
            new_caches.extend(ncs)
            if mid:
                feats.extend(out[2][j] for j in range(len(mid)))
            if last:
                feats.append(self._prog_stream_mean(B, T)(streams))
            self.launches += 1
        return (streams, new_caches, feats) if capture else (streams, new_caches)

    def _run(self, tokens, caches, pos0, use_recurrent, length=None, want_logits=True, want_streams=False, override=None,
             capture=False):
        B, T = tokens.shape
        length = T if length is None else length
        out = self._run_layers(tokens, caches, pos0, use_recurrent, length, override, capture)
        streams, new_caches = out[:2]
        logits = None
        if want_logits:
            logits = self._prog_head(B, T)(self.params["norm"], self.params["lm_head"], streams, self.scalar(length))
        return (logits, new_caches) + ((streams,) if want_streams else ()) + ((out[2],) if capture else ())

    def scalar(self, x):
        """A host int as a replicated int32 device scalar. A plain jnp.int32(x) is uncommitted, so every program it is
        passed to reshards it on the host (`cpp_pjit_shard_arg_fallback`, ~0.3 ms per argument and call: ~3.6 ms of a
        verify step's ~16 ms host time, j69 2026-10-03). A device array (e.g. a position computed on the chips)
        passes through."""
        if isinstance(x, jax.Array):
            return x
        x = int(x)
        if 0 <= x < 64:                                  # small constants (block sizes, lengths): placed once
            c = self.__dict__.setdefault("_small_scalars", {})
            if x not in c:
                c[x] = jax.device_put(np.int32(x), NamedSharding(self.mesh, P()))
            return c[x]
        return jax.device_put(np.int32(x), NamedSharding(self.mesh, P()))

    def decode_sample(self, token, caches, pos, sampler: DeviceSampler):
        """`decode` with the next token sampled on the device: returns (ids [B] int32 device array, logits [B,V],
        caches, pos + 1). Only the ids need to reach the host (one small transfer instead of the [B,V] logits)."""
        B = token.shape[0]
        n = self.cfg.n_layers
        extra = caches[n:] if len(caches) > n else []
        streams, new = self._run_layers(np.asarray(token).reshape(B, 1), caches[:n], pos, True, 1)
        if getattr(self, "_one", None) is None:          # a replicated constant (no per-step scalar transfer); lazy so
            self._one = jax.device_put(jnp.int32(1), NamedSharding(self.mesh, P()))   # live engines survive a reload
        ids, logits, nk = self._prog_head(B, 1, True)(self.params["norm"], self.params["lm_head"], streams, self._one,
                                                      *sampler.bind(self.mesh))
        sampler.advance(nk)
        return ids, logits, new + extra, pos + 1

    # ---- batched decode of independent streams (continuous batching): one cache set per stream, per-row positions
    def _prog_embed_rows(self, B):
        """(embed, tokens [B] int32, pos [B] int32) -> (streams [B,1,hc,D], pos + 1): the positions stay on the
        device (no per-step host transfer), the program advances them."""
        key = ("embed_rows", B)
        if key not in self._progs:
            def prog(embed, tokens, pos):
                x = self._embed(embed, tokens[:, None]).astype(self.lcfg.dtype)
                return jnp.broadcast_to(x[:, :, None, :], (B, 1, self.cfg.hc, x.shape[-1])), pos + 1
            self._progs[key] = aot.jit(key, shard_map(prog, mesh=self.mesh, in_specs=(self.specs["embed"], R, R),
                                                 out_specs=(R, R), check_vma=False))
        return self._progs[key]

    def _prog_group_rows(self, layers, B, capture=False):
        """Decode step of `layers` for B independent streams: (params, streams [B,1,hc,D], pos [B], caches) with
        caches[j][b] = layer j's cache of stream b (batch dim 1, donated) -> (streams, new caches[j][b][, the stream
        means after EACH layer stacked [len(layers), B, 1, D] with capture=True: DFlash features, one program per layer
        kinds as in the verify])."""
        kinds = tuple(self._kind(i) for i in layers)
        key = ("group_rows", kinds, B) + (("capture",) if capture else ())
        if key not in self._progs:
            lts = [self.cfg.layer_types[i] for i in layers]
            mts = [self.cfg.mlp_types[i] for i in layers]
            css = [[self._cache_spec(i)] * B for i in layers]
            rep = list(layers)
            from glm53 import quant8 as Q8

            def prog(ps, streams, pos, caches):
                pos_b = [pos[b] for b in range(B)]
                new, feats = [], []
                for j, p in enumerate(ps):
                    p = Q8.dequant_tree(p, self.lcfg.dtype)
                    fetch = self.expert_fetch if mts[j] == "sparse" else None
                    streams, ncs = M.decoder_layer_rows(p, streams, self.lcfg, lts[j], mts[j], caches[j], pos_b, fetch,
                                                        "auto", layer=rep[j], cap=self.max_len)
                    new.append(ncs)
                    if capture:
                        feats.append(streams.astype(jnp.float32).mean(2).astype(self.lcfg.dtype))
                return (streams, new) + ((jnp.stack(feats),) if capture else ())
            in_specs = ([self.specs["layers"][i] for i in layers], R, R, css)
            out_specs = (R, css) + ((R,) if capture else ())
            sm = shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
            self._progs[key] = aot.jit(key, sm, donate_argnums=(3,))
        return self._progs[key]

    def _prog_head_ring(self, B, sample, fidx):
        """`decode_rows`' head with a DFlash drafter installed: the head (`_prog_head(B, 1, sample)`: logits [B,V][, ids,
        next key]) + every row's drafter ring takes the step's position (its features captured after the drafter's
        target layers -> the drafter's context K/V, `dflash.ingest`): positions decoded in batched or plain steps are
        never holes in a stream's ring. (norm, lm_head, streams, drafter params, feats (the stacked stream means of the
        groups holding target layers, [len(g), B, 1, D]; `fidx` = the target positions in each), rings [B] (each batch
        1, donated), pos [B][, temperature, top_p, key]) -> (logits, rings) or (ids, logits, key, rings)."""
        key = ("head_ring", B, sample, fidx)
        if key not in self._progs:
            from glm53 import dflash as DF
            from glm53 import quant8 as Q8
            dr = self.dflash

            def prog(norm, lm_head, streams, dparams, feats, rings, pos, *samp):
                h = M.rmsnorm(streams.mean(2).astype(self.lcfg.dtype), norm, self.cfg.eps)[:, 0]
                zl = self._logits_local(lm_head, h)
                logits = lax.all_gather(zl, AXIS, axis=-1, tiled=True)
                dp = Q8.dequant_tree(dparams, dr.cfg.dtype)
                one = jnp.ones((1,), jnp.int32)
                fl = [f[j] for f, js in zip(feats, fidx) for j in js]
                new = [DF.ingest(dp, rings[b], [f[b:b + 1] for f in fl], pos[b:b + 1], one, dr.lcfg) for b in range(B)]
                if not sample:
                    return logits, new
                temp, top_p, k = samp
                k1, k2 = jax.random.split(k)
                return device_sample(zl, temp, top_p, k1), logits, k2, new
            rs = [dict(DF.RING_SPECS)] * B
            in_specs = (R, self.specs["lm_head"], R, dr.specs, [R] * len(fidx), rs, R) + ((R, R, R) if sample else ())
            out_specs = (R, R, R, rs) if sample else (R, rs)
            self._progs[key] = aot.jit(key, shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs,
                                                 check_vma=False), donate_argnums=(5,))
        return self._progs[key]

    def _prog_group_vrows(self, layers, B, T, embed=False, pend=None):
        """The batched DFlash verify of `layers`: B independent streams, each verifying its own T-token block (anchor +
        drafts) on its own cache set at its own position. (params, streams [B,T,hc,D], pos [B], caches[j][b] (batch 1,
        donated)) -> (streams, new caches[j][b], the per-token histories stacked per key over the group's layers with
        the rows on the batch axis ([n, T, B, ...], `_hist_layout`; the KDA layers return their tokens' update inputs
        and leave their states as before the block: the rollback replays the accepted tokens — 64 KB per layer and row
        instead of T states of 0.5 MB, which did not fit next to 3 streams at 262k), the stream means after EACH layer
        [len(layers), B, block, D] (DFlash features, padded to the drafter's block: one program per layer kinds)).
        pend = T of the previous batched verify whose rollback is still pending (`dflash_step_rows`): the program
        also takes that verify's histories of this group (rows on the batch axis; DONATED, this verify's reuse the
        buffers) and its n_keep [B], and first rolls every row's small state entries back to its own accepted prefix
        (`_prog_rollback_rows`'s op), so no rollback program runs between the steps."""
        kinds = tuple(self._kind(i) for i in layers)
        key = ("group_vrows", kinds, B, T) + (("embed",) if embed else ()) + ((("pend", pend),) if pend else ())
        if key not in self._progs:
            lts = [self.cfg.layer_types[i] for i in layers]
            mts = [self.cfg.mlp_types[i] for i in layers]
            css = [[self._cache_spec(i)] * B for i in layers]
            lay = self._hist_layout(layers, kin=True)
            hss = {k: P(None, *self._hist_spec(layers[js[0]], kin=True)[k]) for k, js in lay.items()}
            rep = list(layers)
            from glm53 import quant8 as Q8

            def prog(ps, streams, pos, caches, hg=None, nk=None):
                pos_b = [pos[b] for b in range(B)]
                new, hs, feats = [], [], []
                if embed:
                    streams = self._embed_streams(*streams)
                if pend:
                    caches = [list(c) for c in caches]
                    dts = [{k_: v.dtype for k_, v in caches[j][0].items()} for j in range(len(ps))]
                    for b in range(B):
                        per = [{k: hg[k][js.index(j)][:, b:b + 1] for k, js in lay.items() if j in js} or None
                               for j in range(len(ps))]
                        roll = M.rollback_states(per, nk[b], dts, [caches[j][b] for j in range(len(ps))])
                        for j, r_ in enumerate(roll):
                            if r_ is not None:
                                caches[j][b] = {**caches[j][b], **r_}
                for j, p in enumerate(ps):
                    p = Q8.dequant_tree(p, self.lcfg.dtype)
                    fetch = self.expert_fetch if mts[j] == "sparse" else None
                    streams, ncs = M.decoder_layer_rows(p, streams, self.lcfg, lts[j], mts[j], caches[j], pos_b, fetch,
                                                        "auto", layer=rep[j], cap=self.max_len, hist="kin")
                    split = [M.split_hist(c) for c in ncs]
                    new.append([c for c, _ in split])
                    h0 = split[0][1]
                    hs.append(None if h0 is None else {k: jnp.concatenate([h[k] for _, h in split], axis=1) for k in h0})
                    feats.append(self._stream_mean(streams))
                hist = {k: jnp.stack([hs[j][k] for j in js]) for k, js in lay.items()}
                return streams, new, hist, jnp.stack(feats)
            in_specs = ([self.specs["layers"][i] for i in layers], (self.specs["embed"], R) if embed else R, R, css) + \
                ((hss, R) if pend else ())
            sm = shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=(R, css, hss, R), check_vma=False)
            self._progs[key] = aot.jit(key, sm, donate_argnums=(3,) + ((4,) if pend else ()))
        return self._progs[key]

    def _pend_identity_rows(self, sets, T):
        """`_pend_identity` for B streams: histories that change nothing (zero KDA update inputs, the current conv
        states and pool tails repeated T times), n_keep = T for every row."""
        B = len(sets)
        key = ("pend_identity_rows", B, T)
        if key not in self._progs:
            groups = self.groups["decode"]
            lays = [self._hist_layout(g, kin=True) for g in groups]
            hgs = [{k: P(None, *self._hist_spec(g[js[0]], kin=True)[k]) for k, js in lay.items()}
                   for g, lay in zip(groups, lays)]
            cur_specs = [None if h is None else {M.HIST_KEYS[k]: P(*v[1:]) for k, v in h.items()}
                         for h in (self._hist_spec(i, kin=True) for i in range(self.cfg.n_layers))]
            hd = self.cfg.kda_hd

            def one(cur, k):
                if k == "kin_hist":
                    s = cur["state"]                                           # [1,H,dk,dv] -> kin [T,1,H,4,dk] f32
                    return jnp.zeros((T, 1, s.shape[1], 4, hd), jnp.float32)
                a = cur[M.HIST_KEYS[k]]
                return jnp.broadcast_to(a[None], (T + 1,) + a.shape)           # entries 0..T (`M.rollback_states`)

            def prog(curs):                                                     # curs[b][layer] -> per group {k: [n,T,B,...]}
                return [{k: jnp.stack([jnp.concatenate([one(curs[b][g[j]], k) for b in range(B)], axis=1) for j in js])
                         for k, js in lay.items()} for g, lay in zip(groups, lays)]
            self._progs[key] = aot.jit(key, shard_map(prog, mesh=self.mesh, in_specs=([cur_specs] * B,), out_specs=hgs,
                                                 check_vma=False))
        return self._progs[key]([self._hist_cur(c, kin=True) for c in sets]), self.device_positions([T] * B), T

    def dflash_settle_rows(self, st):
        """`dflash_settle` for a batched state: the verify dispatched ahead undone (every row back to before its block,
        n_keep = 0) or the pending rollback applied to every row's caches. Idempotent."""
        B = len(st["sets"])
        if st.get("ahead") is not None:
            hists, nk, T = st["ahead"][2], self.device_positions([0] * B), st["T"]
        elif st.get("pend"):
            hists, nk, T = st["pend"]
        else:
            return st
        rolled = self._prog_rollback_rows(B, T)(hists, nk, [self._hist_cur(c, kin=True) for c in st["sets"]])
        new = [[({**c, **x} if x else c) for c, x in zip(cs, rs)] for cs, rs in zip(st["sets"], rolled)]
        return {**{k: v for k, v in st.items() if k != "ahead"}, "sets": new, "pend": None}

    def _prog_rollback_rows(self, B, T):
        """`_prog_rollback` for B independent streams: (the per-group histories of `_verify_rows` (rows on the batch
        axis), n_keep [B], cur[b] = row b's current small state entries (`_hist_cur`, DONATED; the KDA states are
        those before the block)) -> per row the small state entries as after its own first n_keep[b] tokens (the KDA
        states by replaying them: `M.kda_replay`)."""
        key = ("rollback_rows", B, T)
        if key not in self._progs:
            groups = self.groups["decode"]
            lays = [self._hist_layout(g, kin=True) for g in groups]
            hgs = [{k: P(None, *self._hist_spec(g[js[0]], kin=True)[k]) for k, js in lay.items()}
                   for g, lay in zip(groups, lays)]
            hss = [self._hist_spec(i, kin=True) for i in range(self.cfg.n_layers)]
            oss = [None if h is None else {M.HIST_KEYS[k]: P(*v[1:]) for k, v in h.items()} for h in hss]
            dts = [{M.HIST_KEYS[k]: self.cfg.dtype if M.HIST_KEYS[k] != "state" else jnp.float32 for k in h} if h else None
                   for h in hss]

            def prog(hg, n, cur):
                out = []
                for b in range(B):
                    per = [None] * self.cfg.n_layers
                    for g, lay, h in zip(groups, lays, hg):
                        for k, js in lay.items():
                            for s_, j in enumerate(js):
                                per[g[j]] = {**(per[g[j]] or {}), k: h[k][s_][:, b:b + 1]}
                    out.append(M.rollback_states(per, n[b], dts, cur[b]))
                return out
            sm = shard_map(prog, mesh=self.mesh, in_specs=(hgs, R, [oss] * B), out_specs=[oss] * B, check_vma=False)
            self._progs[key] = aot.jit(key, sm, donate_argnums=(2,))
        return self._progs[key]

    def _verify_rows(self, tokens, sets, pos, pend=None):
        """embed + the decode layer groups of a batched verify (`_prog_group_vrows`): tokens [B,T] (row b = stream b's
        block), sets[b] = its n layer caches (consumed), pos [B] device positions -> (final streams, new sets, histories
        per group, DFlash features (stacked per group holding target layers, the target positions in each)). pend =
        (histories per group, n_keep [B], T) of the previous verify, whose rollback the groups apply first."""
        B, T = tokens.shape
        streams = (self.params["embed"], jnp.asarray(tokens))       # the first group program embeds
        new_sets = [[] for _ in range(B)]
        hists, feats, fidx = [], [], []
        tl = self.dflash.cfg.target_layers
        for gi, g in enumerate(self.groups["decode"]):
            out = self._prog_group_vrows(g, B, T, embed=gi == 0, pend=pend[2] if pend else None)(
                [self.params["layers"][i] for i in g], streams, pos, [[sets[b][i] for b in range(B)] for i in g],
                *((pend[0][gi], pend[1]) if pend else ()))
            streams, ncs, hs, f = out
            for j in range(len(g)):
                for b in range(B):
                    new_sets[b].append(ncs[j][b])
            hists.append(hs)
            js = tuple(j for j, i in enumerate(g) if i in tl)
            if js:
                feats.append(f); fidx.append(js)
            self.launches += 1
        return streams, new_sets, hists, (feats, tuple(fidx))

    def device_positions(self, pos):
        """Host positions (ints) -> one replicated int32 [B] device array (the form `decode_rows` carries)."""
        return jax.device_put(np.asarray(pos, np.int32).reshape(-1), NamedSharding(self.mesh, P()))

    def decode_rows(self, tokens, cache_sets, pos, sampler: DeviceSampler | None = None):
        """One decode step of B independent streams in one batch: `tokens` [B] (their last tokens; a device array
        from the previous step or host ints), `cache_sets[b]` = stream b's own cache list (as `prefill` /
        `restore_prefix` return them, batch dim 1; consumed), `pos` [B] int32 replicated device array
        (`device_positions`) or host ints. Returns (ids [B] device int32 or None without a sampler, logits [B,V],
        new cache sets, pos + 1 on the device). Streams may sit at any positions; the MoE/mHC/projections run
        batched, the attention per row. Compiles one program set per B. With a DFlash drafter installed (cache sets
        with its ring) the step also writes every row's position into that stream's ring (`_prog_head_ring`)."""
        B = len(cache_sets)
        n = self.cfg.n_layers
        assert all(len(c) >= n for c in cache_sets), "every stream needs a full cache set"
        if not isinstance(pos, jax.Array):
            pos = self.device_positions(pos)
        if not (isinstance(tokens, jax.Array) and tokens.shape == (B,) and tokens.dtype == jnp.int32):
            tokens = jnp.asarray(np.asarray(tokens, np.int32).reshape(B))   # (a device array from the last step passes through)
        streams, pos_next = self._prog_embed_rows(B)(self.params["embed"], tokens, pos)
        cap = self.dflash is not None and all(len(c) > n for c in cache_sets)
        new_sets = [[] for _ in range(B)]
        feats, fidx = [], []
        for g in self.groups["decode"]:
            prog = self._prog_group_rows(g, B, cap)
            out = prog([self.params["layers"][i] for i in g], streams, pos, [[cache_sets[b][i] for b in range(B)] for i in g])
            streams, ncs = out[:2]
            js = tuple(j for j, i in enumerate(g) if i in self.dflash.cfg.target_layers) if cap else ()
            if js:
                feats.append(out[2]); fidx.append(js)
            for j in range(len(g)):
                for b in range(B):
                    new_sets[b].append(ncs[j][b])
            self.launches += 1
        if getattr(self, "_one", None) is None:
            self._one = jax.device_put(jnp.int32(1), NamedSharding(self.mesh, P()))
        if cap:                                              # the drafter's rings take this step's positions
            args = (self.params["norm"], self.params["lm_head"], streams, self.dflash.params, feats,
                    [cache_sets[b][n] for b in range(B)], pos)
            if sampler is None:
                logits, rings = self._prog_head_ring(B, False, tuple(fidx))(*args)
                ids = None
            else:
                ids, logits, nk, rings = self._prog_head_ring(B, True, tuple(fidx))(*args, *sampler.bind(self.mesh))
                sampler.advance(nk)
            for b in range(B):
                new_sets[b].append(rings[b])
                new_sets[b].extend(cache_sets[b][n + 1:])
            return ids, logits, new_sets, pos_next
        for b in range(B):                                   # extra entries (an MTP cache) pass through untouched
            new_sets[b].extend(cache_sets[b][n:])
        if sampler is None:
            logits = self._prog_head(B, 1)(self.params["norm"], self.params["lm_head"], streams, self._one)
            return None, logits, new_sets, pos_next
        ids, logits, nk = self._prog_head(B, 1, True)(self.params["norm"], self.params["lm_head"], streams, self._one,
                                                      *sampler.bind(self.mesh))
        sampler.advance(nk)
        return ids, logits, new_sets, pos_next

    def prefill(self, tokens, caches=None, pos0=0, embeds=None):
        """Prefill `tokens` [B,T] into a fresh context (caches=None) or continue one at position pos0 (the caches
        are consumed). Returns (logits of the last token, caches, pos0 + T). With an MTP layer installed the MTP
        cache (entry n_layers) is prefilled too: MTP position p takes (token p+1, final hidden p), so a piece
        covers positions [pos0-1, pos0+L-1) (or [0, L-1) for the first piece) and the last hidden is carried.
        `embeds` = (idx [M] int, vec [M, D]) replaces the embeddings of tokens[0, idx] (B = 1; image tokens,
        glm53.vision) — the MTP layer's own embedding input keeps the table rows (drafts only; verify is exact).
        With a DFlash drafter installed, the pieces that hold the last window-1 positions also capture the target
        features, and the drafter's ring (entry n_layers) takes their K/V."""
        tokens = np.asarray(tokens)
        B, T = tokens.shape
        assert T >= 1 and pos0 + T <= self.max_len, (pos0, T, self.max_len)
        if caches is None:
            caches = self.alloc_caches(B)
        if embeds is not None:
            assert B == 1
            e_idx, e_vec = np.asarray(embeds[0], np.int64), np.asarray(embeds[1])
        logits = None
        n = self.cfg.n_layers
        main, mc = (caches[:n], caches[n]) if self.mtp is not None else (caches[:n], None)
        ring = caches[n] if self.dflash is not None else None
        for a in range(0, T, self.prefill_piece):
            seg = tokens[:, a:a + self.prefill_piece]
            L = seg.shape[1]
            Tp = self._bucket(L)
            padded = np.zeros((B, Tp), np.int32)
            padded[:, :L] = seg
            ov = None
            if embeds is not None:
                sel = (e_idx >= a) & (e_idx < a + L)
                if sel.any():
                    vec = np.zeros((B, Tp, e_vec.shape[-1]), e_vec.dtype)
                    mask = np.zeros((B, Tp), bool)
                    vec[0, e_idx[sel] - a] = e_vec[sel]
                    mask[0, e_idx[sel] - a] = True
                    ov = (vec, mask)
            if ring is not None:
                cap = a + L > T - self.dflash.cfg.window          # positions a block can still see
                out = self._run(padded, main, pos0 + a, False, length=L, want_logits=a + L >= T, override=ov,
                                capture=cap)
                logits, main = out[:2]
                if cap:
                    ring = self.dflash.ingest(ring, out[2], self.device_positions([pos0 + a] * B),
                                              self.device_positions([L] * B))
                continue
            if mc is None:
                logits, main = self._run(padded, main, pos0 + a, False, length=L, want_logits=a + L >= T, override=ov)
                continue
            logits, main, streams = self._run(padded, main, pos0 + a, False, length=L, want_logits=a + L >= T,
                                              want_streams=True, override=ov)
            hidden = self._prog_hidden(B, Tp)(self.params["norm"], streams)                 # [B,Tp,D]
            p0 = pos0 + a
            if p0 == 0:                                        # rows 0..L-2: token j+1 with hidden j
                mt = np.zeros((B, Tp), np.int32); mt[:, :L - 1] = seg[:, 1:]
                hp = jnp.concatenate([hidden[:, :Tp - 1], jnp.zeros_like(hidden[:, :1])], axis=1)
                start, length = 0, L - 1
            else:                                              # rows p0-1..p0+L-2: token p0+j with hidden p0+j-1
                mt = padded
                hp = jnp.concatenate([mc["h"], hidden[:, :Tp - 1]], axis=1)
                start, length = p0 - 1, L
            mcache = {k: v for k, v in mc.items() if k != "h"}
            if length > 0:
                _, _, mcache = self._prog_mtp(B, Tp, True)(self.mtp, self.params["embed"], self.params["lm_head"],
                                                           jnp.asarray(mt), hp, self.scalar(start), self.scalar(length), mcache)
            mc = {**mcache, "h": lax.dynamic_index_in_dim(hidden, L - 1, axis=1, keepdims=True)}
        return logits, main + [e for e in (mc, ring) if e is not None], pos0 + T

    # ---- speculative decoding with the MTP (NextN) layer
    def set_mtp(self, params, qtypes, k=1):
        """Install the MTP layer: `params` from checkpoint.load_mtp_layer (2-D matrices bf16) plus its expert planes
        (pack_layer_from_gguf) under mlp["gate_q"/"up_q"/"down_q"]; `qtypes` their formats; `k` drafts per step.
        The layer's attention cache + carried hidden state become cache entry n_layers (alloc_caches)."""
        col, row = P(None, AXIS), P(AXIS, None)
        attn = {"q_a": R, "q_a_norm": R, "q_b": col, "kv_a": R, "kv_a_norm": R, "kv_b": col, "o": row,
                "indexer": {kk: R for kk in params["attn"]["indexer"]}}
        mlp = {"router_w": R, "router_bias": R, "shared": {"gate": col, "up": col, "down": row}}
        for kk in ("gate_q", "up_q", "down_q"):
            mlp[kk] = {pk: P(AXIS) for pk in params["mlp"][kk]}
        spec = {"ln1": R, "ln2": R, "attn": attn, "mlp": mlp, "enorm": R, "hnorm": R, "eh_proj": R, "head_norm": R}
        self.expert_fetch.qtypes[self.cfg.n_layers] = qtypes
        self.mtp = jax.tree.map(lambda a, sp: a if hasattr(a, "sharding") and a.sharding == NamedSharding(self.mesh, sp)
                                else jax.device_put(a, NamedSharding(self.mesh, sp)), params, spec)
        self.mtp_spec, self.mtp_k = spec, k

    def _prog_mtp(self, B, T, has_cache=True):
        """(mtp params, embed, lm_head, tokens [B,T] = the token AFTER each MTP position, h_prev [B,T,D], pos0,
        length, cache) -> (logits of the last real position [B,V], hidden [B,T,D], new cache)."""
        key = ("mtp", B, T, has_cache)
        if key not in self._progs:
            from glm53 import quant8 as Q8
            cs = {k: v for k, v in self._cache_spec(self.cfg.n_layers).items() if k != "h"}

            def prog(mp, embed, lm_head, tokens, h_prev, pos0, length, cache):
                mp = Q8.dequant_tree(mp, self.lcfg.dtype)
                e = self._embed(embed, tokens).astype(self.lcfg.dtype)
                e = jnp.where((pos0 + jnp.arange(T))[None, :, None] == 0, jnp.zeros((), e.dtype), e)   # no position -1
                h, nc = M.mtp_layer(mp, e, h_prev, self.lcfg, cache, pos0, self.expert_fetch, "auto", self.max_len,
                                    length, layer=self.cfg.n_layers)
                last = lax.dynamic_index_in_dim(h, length - 1, axis=1, keepdims=False)
                return self._logits(lm_head, last), h, nc
            in_specs = (self.mtp_spec, self.specs["embed"], self.specs["lm_head"], R, R, R, R, cs)
            sm = shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=(R, R, cs), check_vma=False)
            self._progs[key] = aot.jit(key, sm, donate_argnums=(7,))
        return self._progs[key]

    def _verify_layers(self, tokens, caches, pos0, capture=False, pend=None):
        """embed + the decode layer groups of a recurrent T-token step with per-token histories -> (final streams,
        caches, histories (per group, stacked per key, the rows' "kin" form: `_prog_rollback(kin=True)` takes them),
        DFlash features when capture: (the stacked stream means [len(g), B, T, D] of the groups holding drafter target
        layers, the target positions in each) — the drafter's programs take that pair
        (`dflash.Drafter.draft(feat_index=...)`)). pend = (histories per group, n_keep, T) of the previous verify whose
        rollback the groups apply first (`dflash_step`)."""
        B, T = tokens.shape
        streams = (self.params["embed"], jnp.asarray(tokens))       # the first group program embeds (no embed program)
        new, hists, feats, fidx = [], [], [], []
        pos0, length = self.scalar(pos0), self.scalar(T)
        tl = self.dflash.cfg.target_layers if capture else ()
        for gi, g in enumerate(self.groups["decode"]):
            # with capture, every group program returns the stream mean after EACH of its layers: groups with the same
            # layer kinds then share one program whichever layer they capture (5 verify programs per T instead of 9)
            every = tuple(range(len(g))) if capture else ()
            prog = self._prog_group(g, B, T, True, True, hist="kin", capture=every, pend=pend[2] if pend else None,
                                    embed=gi == 0)
            out = prog([self.params["layers"][i] for i in g], streams, pos0, length, [caches[i] for i in g],
                       *((pend[0][gi], pend[1]) if pend else ()))
            streams, ncs, hs = out[:3]
            new.extend(ncs); hists.append(hs)
            js = tuple(j for j, i in enumerate(g) if i in tl)
            if js:
                feats.append(out[3]); fidx.append(js)
            self.launches += 1
        return streams, new, hists, (feats, tuple(fidx))

    def _run_verify(self, tokens, caches, pos0, sampler=None, capture=False):
        """Recurrent T-token step returning (logits [B,T,V], caches, per-layer histories, hidden [B,T,D], ids[,
        DFlash features: list of [B,T,D] per drafter target layer, with capture=True]): ids = the device-sampled token
        of every position [B,T] when a DeviceSampler is given, else None."""
        B, T = tokens.shape
        streams, new, hists, feats = self._verify_layers(tokens, caches, pos0, capture)
        extra = (feats,) if capture else ()
        if sampler is None:
            logits, hidden = self._prog_head_all(B, T)(self.params["norm"], self.params["lm_head"], streams)
            return (logits, new, hists, hidden, None) + extra
        ids, logits, hidden, nk = self._prog_head_all(B, T, True)(self.params["norm"], self.params["lm_head"], streams,
                                                                  *sampler.bind(self.mesh))
        sampler.advance(nk)
        return (logits, new, hists, hidden, ids) + extra

    def spec_decode(self, token, caches, pos, k=None, sampler=None, draft_override=None, stop_ids=()):
        """One speculative step from the committed token `token` [B] at position `pos` (B = 1): the MTP layer
        drafts k tokens, the main model verifies them in one (k+1)-token recurrent step, the longest accepted prefix
        is kept (recurrent states and pool tails rolled back to it). `sampler` decides the target's token per
        position: None = argmax, a host callable `sampler(logits_row) -> int`, or a `DeviceSampler` (sampled inside
        the head program; only the ids cross to the host). A draft is accepted iff it equals the target's own choice,
        which is exact for greedy and a valid, if conservative, scheme for sampling. A draft in `stop_ids` ends the
        accepted run (it is emitted as the last, unfed token). Returns (emitted tokens [1..k+1] — all but the last
        were fed to the model, logits of the last emitted position [B,V], caches, new position)."""
        k = self.mtp_k if k is None else k
        n = self.cfg.n_layers
        B = token.shape[0]
        assert B == 1 and self.mtp is not None
        mc = caches[n]
        mcache = {kk: v for kk, v in mc.items() if kk != "h"}
        hp = mc["h"]
        tok = jnp.asarray(np.asarray(token, np.int32).reshape(B, 1))
        drafts, tails = [], []                                  # the draft chain stays on the device (one host sync per step)
        for j in range(k):
            tails.append((mcache["tk"], mcache["tg"]))          # before draft j (donated: keep the objects, not copies)
            lg, hd, mcache = self._prog_mtp(B, 1, True)(self.mtp, self.params["embed"], self.params["lm_head"],
                                                        tok, hp, self.scalar(pos - 1 + j), self.scalar(1),
                                                        {**mcache, "tk": jnp.copy(mcache["tk"]), "tg": jnp.copy(mcache["tg"])})
            d = jnp.asarray([[int(draft_override[j])]], jnp.int32) if draft_override is not None else jnp.argmax(lg, axis=-1)[:, None].astype(jnp.int32)
            drafts.append(d); tok = d; hp = hd
        seq = jnp.concatenate([jnp.asarray(np.asarray(token, np.int32).reshape(B, 1))] + drafts, axis=1)   # [B,k+1]
        dev = isinstance(sampler, DeviceSampler)
        logits, new, hists, hidden, ids = self._run_verify(seq, caches[:n], pos, sampler if dev else None)
        seq_h = np.asarray(seq)[0]
        drafts_h = [int(x) for x in seq_h[1:]]
        if dev:                                                 # only k+1 ids cross to the host, not [k+1, V] logits
            preds = [int(x) for x in np.asarray(ids[0])]
        else:
            lg_h = np.asarray(logits[0])
            preds = [int(np.argmax(lg_h[j])) if sampler is None else int(sampler(lg_h[j])) for j in range(k + 1)]
        a = 0                                                   # a stop token is never fed: it ends the accepted run
        while a < k and preds[a] == drafts_h[a] and drafts_h[a] not in stop_ids:
            a += 1
        # the verify's KDA states are those before the block (the "kin" histories): replay the a + 1 accepted tokens
        # (every one of them when all drafts were accepted), the conv states and pool tails to after them
        rolled = self._prog_rollback(B, k + 1, kin=True)(hists, self.scalar(a + 1), self._hist_cur(new, kin=True))
        new = [({**c, **r} if r else c) for c, r in zip(new, rolled)]
        if a + 1 < k:                                           # drafts a+1.. were fed rejected tokens: tail as before draft a+1
            mcache = {**mcache, "tk": tails[a + 1][0], "tg": tails[a + 1][1]}
        emitted = drafts_h[:a] + [preds[a]]
        h_next = lax.dynamic_index_in_dim(hidden, a, axis=1, keepdims=True)
        return emitted, logits[:, a], new + [{**mcache, "h": h_next}], pos + a + 1

    # ---- speculative decoding with the DFlash 2 drafter (block diffusion, glm53.dflash)
    def set_dflash(self, params, dcfg, int8=False, temp_scale=0.7):
        """Install the DFlash 2 drafter (`glm53.dflash.load_params`) on the engine's mesh; it shares the target's
        embedding and lm_head. Its context ring becomes cache entry n_layers (`alloc_caches`), filled by `prefill`
        with the captured target features (the stream mean after each of dcfg.target_layers) and by `dflash_step`
        with the accepted positions'. int8: the drafter's big matrices as q8 (glm53.quant8). temp_scale: drafts are
        drawn at temp_scale x the request's temperature (rejection sampling accepts against the target's own; 0.5-0.7
        accepted ~1 % more than 1.0 and ~4 % more than greedy drafts at temperature 1.0, v6e 2026-10-04)."""
        from glm53 import dflash as DF
        assert self.mtp is None, "MTP and DFlash share cache entry n_layers"
        assert max(dcfg.target_layers) < self.cfg.n_layers, dcfg.target_layers
        self.dflash = DF.Drafter(dataclasses.replace(dcfg, dtype=self.cfg.dtype), params, self.mesh,
                                 self.specs["embed"], self.specs["lm_head"], int8=int8, temp_scale=temp_scale)

    def _dflash_zeros(self, B):
        """Zero DFlash features in the verify's layout (`_verify_layers`): for a draft with no new positions."""
        key = ("dflash_zeros", B)
        if key not in self._progs:
            tl, blk = self.dflash.cfg.target_layers, self.dflash.cfg.block
            shapes, fidx = [], []
            for g in self.groups["decode"]:
                js = tuple(j for j, i in enumerate(g) if i in tl)
                if js:
                    shapes.append((len(g), B, blk, self.cfg.hidden)); fidx.append(js)
            z = jax.jit(lambda: [jnp.zeros(sh, self.cfg.dtype) for sh in shapes],
                        out_shardings=[NamedSharding(self.mesh, R)] * len(shapes))()
            self._progs[key] = (z, tuple(fidx))
        return self._progs[key]

    def _prog_head_accept(self, B, T, stop_ids, per_row=False):
        """The DFlash verify's head with the acceptance on the chips: (norm, lm_head, streams, seq [B,T] = anchor +
        drafts, cand / q [B,T-1,k] = the drafter's candidates and its distribution over them, pos (), temperature,
        top_p, key) -> dict: emitted [B,T] (the accepted drafts, then the emitted token, -1 after), a [B] (accepted
        drafts), bonus [B] (the emitted token = the next anchor), n_keep () = a+1, pos_next () = pos + a + 1, pos_b [B] =
        pos, n_new_b [B] = a + 1, key. The acceptance is rejection sampling against the target's distribution after
        temperature and top-p (`engine.spec_accept`): every emitted token is distributed as plain sampling; at
        temperature 0 a draft is accepted iff it is the argmax (greedy). A draft in `stop_ids` ends the accepted run
        (B = 1: n_keep and pos_next are row 0's). per_row=True (B independent streams, `dflash_step_rows`): pos [B] and
        temperature / top_p per row ([B] or scalars); n_keep and pos_next [B]."""
        key = ("head_accept", B, T, tuple(stop_ids)) + (("rows",) if per_row else ())
        if key not in self._progs:
            def prog(norm, lm_head, streams, seq, cand, q, pos, temp, top_p, k):
                h = M.rmsnorm(streams.mean(2).astype(self.lcfg.dtype), norm, self.cfg.eps)
                k1, k2 = jax.random.split(k)
                rows = lambda v: jnp.broadcast_to(jnp.reshape(v, (-1, 1)), (B, T)).reshape(-1)   # noqa: E731
                gidx, p = device_probs(self._logits_local(lm_head, h).reshape(B * T, -1), rows(temp), rows(top_p))
                C = gidx.shape[-1]
                d = seq[:, 1:].astype(jnp.int32)
                a, bonus = spec_accept(gidx.reshape(B, T, C), p.reshape(B, T, C), d, cand, q, k1, stop_ids)
                j = jnp.arange(T)[None, :]
                dpad = jnp.concatenate([d, jnp.zeros((B, 1), jnp.int32)], 1)
                emitted = jnp.where(j < a[:, None], dpad, jnp.where(j == a[:, None], bonus[:, None], -1))
                if per_row:
                    return {"emitted": emitted, "a": a, "bonus": bonus, "n_keep": a + 1, "pos_next": pos + a + 1,
                            "pos_b": pos, "n_new_b": a + 1, "key": k2}
                return {"emitted": emitted, "a": a, "bonus": bonus, "n_keep": a[0] + 1, "pos_next": pos + a[0] + 1,
                        "pos_b": jnp.broadcast_to(pos, (B,)), "n_new_b": a + 1, "key": k2}
            outs = {k: R for k in ("emitted", "a", "bonus", "n_keep", "pos_next", "pos_b", "n_new_b", "key")}
            in_specs = (R, self.specs["lm_head"], R, R, R, R, R, R, R, R)
            self._progs[key] = aot.jit(key, shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=outs,
                                                 check_vma=False))
        return self._progs[key]

    def _dflash_draft(self, ring, feats, pos_b, n_new_b, anchor_b, T, temp, key):
        """One draft program (any temperature: the selector draws from its q, the argmax at temperature 0) ->
        (seq [B,T] = anchor + drafts, cand, q [B,T-1,k], the new ring)."""
        r = self.dflash.draft(ring, feats[0], pos_b, n_new_b, anchor_b, self.params["embed"], self.params["lm_head"], T=T,
                              temperature=temp, key=key, feat_index=feats[1])
        return r["seq"], r["cand"], r["q"], r["ring"]

    def dflash_start(self, token, caches, pos, T=None, pending=None, sampler: DeviceSampler | None = None):
        """Begin DFlash 2 decoding of one stream (B = 1) from the committed token `token` [B] at position `pos` (not yet
        fed; e.g. the token sampled from `prefill`'s logits): dispatches the first draft (drawn at the sampler's
        temperature; pass the same sampler to `dflash_step`). `pending` = the features of positions the ring has not
        seen yet (None right after `prefill`, which fills it). Returns the state that `dflash_step` advances:
        {"caches", "pos" (device scalar), "pos_h" (host int), "seq" (the drafted block, on the device), "cand", "q",
        "T"}."""
        n = self.cfg.n_layers
        T = self.dflash.cfg.block if T is None else T
        B = np.asarray(token).reshape(-1).shape[0]
        assert B == 1 and self.dflash is not None
        sampler = sampler if sampler is not None else self._dflash_greedy()
        temp, _, k = sampler.bind(self.mesh)
        feats, p0, n_new = pending if pending is not None else (self._dflash_zeros(B), pos, 0)
        seq, cand, q, ring = self._dflash_draft(caches[n], feats, self.device_positions([p0] * B),
                                                self.device_positions([n_new] * B),
                                                self.device_positions(np.asarray(token).reshape(B)), T, temp, k)
        return {"caches": caches[:n] + [ring], "pos": self.scalar(pos), "pos_h": int(pos), "seq": seq, "cand": cand,
                "q": q, "T": T}

    def dflash_step(self, st, sampler: DeviceSampler | None = None, stop_ids=(), draft_override=None, next_T=None):
        """One DFlash 2 step: the target verifies the drafted block [anchor, drafts] in one T-token step that also
        captures its features; the acceptance runs on the chips (the head program: rejection sampling of the drafts
        against the target's distribution, `sampler` None = greedy, where a draft is accepted iff it is the argmax; a
        draft in `stop_ids` ends the run and is emitted unfed); the KDA states and pool tails are rolled back to the
        accepted prefix; the NEXT draft (anchor = the emitted token, the accepted positions' features into the ring)
        and, with `run_ahead`, the verify of that next block are dispatched before the host reads this step's tokens,
        so the host's turnaround overlaps them (the draft only writes committed positions; the verify ahead is undone
        by `dflash_settle` when the stream stops: stopping after any step leaves a consistent state). `next_T` = the
        block size of the next draft (default: this one's). `draft_override` (T-1 ids, tests; exact only at
        temperature 0) replaces this step's drafts (a verify dispatched ahead is undone and run again). Returns
        (emitted tokens [1..T] — all but the last were fed to the model, the new state)."""
        n = self.cfg.n_layers
        T, B = st["T"], 1
        sampler = sampler if sampler is not None else self._dflash_greedy()
        grave = self.__dict__.pop("_grave", None)        # the previous step's dropped arrays (see the end)
        seq = st["seq"]
        ahead = st.get("ahead")                          # this block's verify, dispatched by the previous step
        if draft_override is not None:
            if ahead is not None:                        # (tests) the verify ahead saw the real drafts: undo it
                st, ahead = self.dflash_settle(st), None
            seq = jnp.concatenate([seq[:, :1], jnp.asarray(np.asarray(draft_override, np.int32).reshape(B, T - 1))], 1)
        caches = st["caches"]
        if ahead is None:
            pend = st.get("pend") or self._pend_identity(caches[:n], T)
            ahead = self._verify_layers(seq, caches[:n], st["pos"], capture=True, pend=pend)
        else:
            pend = None
        streams, new, hists, feats = ahead
        self._drop(grave, pend)
        del grave, pend, ahead                           # released while the device runs the verify
        temp, top_p, k = sampler.bind(self.mesh)
        r = self._prog_head_accept(B, T, tuple(stop_ids))(self.params["norm"], self.params["lm_head"], streams, seq,
                                                          st["cand"], st["q"], st["pos"], temp, top_p, k)
        sampler.advance(r["key"])
        # the rollback of this verify to its accepted prefix is deferred: the next verify's group programs apply it
        # first (or `dflash_settle`, when the stream leaves DFlash): no rollback program, its ~90 outputs and their
        # release between the steps (the one-stream step was host-bound, j141 2026-10-04). The draft stays a program
        # of its own, queued behind the head (merged into the head program, 2026-10-05 j143 / j150, the device idled
        # through the host's turnaround: +0.5 ms per step).
        next_T = T if next_T is None else next_T
        nseq, cand, q, ring = self._dflash_draft(caches[n], feats, r["pos_b"], r["n_new_b"], r["bonus"], next_T, temp,
                                                 r["key"])
        out = {"caches": new + [ring], "pend": (hists, r["n_keep"], T), "pos": r["pos_next"], "seq": nseq, "cand": cand,
               "q": q, "T": next_T}
        # run ahead: the next block's verify (its inputs — the drafted block, the position, the caches, this verify's
        # histories and n_keep — are all on the device) is queued behind the draft, so the device has ~a step of work
        # when the host wakes up below; only when this block and the next both fit the context
        if self.run_ahead and st["pos_h"] + T + next_T <= self.max_len:
            out["ahead"] = self._verify_layers(nseq, new, r["pos_next"], capture=True, pend=out.pop("pend"))
            out["caches"] = out["ahead"][1] + [ring]
        emitted = [int(x) for x in np.asarray(r["emitted"])[0] if x >= 0]     # the host waits here
        out["pos_h"] = st["pos_h"] + len(emitted)
        # Dropping a jax array costs ~11-20 us of host time on the 8-chip mesh (j107) and a step leaves ~300 behind (the
        # consumed caches, the pre-rollback states, histories, ...): keep them until the next step has dispatched its
        # head, so the release happens while the device works instead of between the steps (`dflash_end` frees them;
        # with `reap` the drop itself runs on the reaper thread).
        self._grave = (st, feats, streams, r)
        return emitted, out

    def _pend_identity(self, caches, T):
        """A pending rollback that changes nothing (the first step of a DFlash run): the rows' form for one stream
        (`_pend_identity_rows`: zero KDA update inputs, the current conv states and pool tails) with a scalar
        n_keep = T; the same group programs then serve every step."""
        hists, _, _ = self._pend_identity_rows([caches], T)
        return hists, self.scalar(T), T

    def dflash_settle(self, st):
        """The state with its deferred work resolved — a verify dispatched ahead of its tokens (`run_ahead`) undone,
        since its block is never accepted (the small state entries back to before it: n_keep = 0), or the last step's
        pending rollback applied. Call before using st["caches"] outside DFlash steps (a stream leaving DFlash mode,
        tests). Idempotent."""
        n = self.cfg.n_layers
        caches = st["caches"]
        if st.get("ahead") is not None:
            hists, nk, T = st["ahead"][2], self.scalar(0), st["T"]
        elif st.get("pend"):
            hists, nk, T = st["pend"]
        else:
            return st
        rolled = self._prog_rollback(1, T, kin=True)(hists, nk, self._hist_cur(caches[:n], kin=True))
        new = [({**c, **x} if x else c) for c, x in zip(caches[:n], rolled)]
        return {**{k: v for k, v in st.items() if k != "ahead"}, "caches": new + list(caches[n:]), "pend": None}

    def dflash_end(self):
        """Release what the last `dflash_step` kept for a deferred release (call when a stream leaves DFlash mode)."""
        self.__dict__.pop("_grave", None)

    # ---- DFlash 2 for B streams at once (continuous batching: each stream its own cache set, ring and position)
    def _dflash_draft_rows(self, rings, feats, pos_b, n_new_b, anchor_b, T, temp, key):
        r = self.dflash.draft_rows(rings, feats[0], pos_b, n_new_b, anchor_b, self.params["embed"],
                                   self.params["lm_head"], T=T, temperature=temp, key=key, feat_index=feats[1])
        return r["seq"], r["cand"], r["q"], r["rings"]

    def dflash_start_rows(self, tokens, cache_sets, pos, T=None, sampler: DeviceSampler | None = None):
        """Begin DFlash 2 decoding of B independent streams together (`decode_rows`' layout): cache_sets[b] = stream b's
        full cache set (its drafter ring included, holding every committed position: prefill, `decode_rows` and DFlash
        steps keep it so; consumed), tokens [B] = their committed, not yet fed tokens, pos [B] host ints. Dispatches
        the first draft of every row; `sampler` = one temperature / top_p per row (or scalars; None = greedy). Returns
        the state `dflash_step_rows` advances: {"sets" (each row's n layer caches), "rings", "pos" (device [B]),
        "pos_h" (host ints), "seq", "cand", "q", "T"}."""
        n = self.cfg.n_layers
        B = len(cache_sets)
        assert self.dflash is not None and all(len(c) > n for c in cache_sets)
        T = self.dflash.cfg.block if T is None else T
        sampler = sampler if sampler is not None else self._dflash_greedy()
        temp, _, k = sampler.bind(self.mesh)
        pos_d = self.device_positions(pos)
        seq, cand, q, rings = self._dflash_draft_rows([c[n] for c in cache_sets], self._dflash_zeros(B), pos_d,
                                                      self.device_positions([0] * B),
                                                      self.device_positions(np.asarray(tokens).reshape(B)), T, temp, k)
        return {"sets": [list(c[:n]) for c in cache_sets], "rings": rings, "pos": pos_d,
                "pos_h": [int(p) for p in pos], "seq": seq, "cand": cand, "q": q, "T": T}

    def dflash_step_rows(self, st, sampler: DeviceSampler | None = None, stop_ids=(), draft_override=None, next_T=None):
        """`dflash_step` for B streams at once: one batched verify (every row its own block on its own caches; the MoE
        groups each row's T tokens by expert), the acceptance per row on the chips, each row's states rolled back to
        its own accepted prefix, then the next draft of every row (its accepted positions into its ring) queued before
        the host reads the tokens. Rows are independent: at temperature 0 each row emits exactly what that stream's
        own greedy decoding would. draft_override [B, T-1] (tests). With `run_ahead` the next blocks' verify is
        dispatched too (undone by `dflash_settle_rows` when the state ends). Returns (emitted tokens per row, the new
        state)."""
        T, B = st["T"], len(st["sets"])
        sampler = sampler if sampler is not None else self._dflash_greedy()
        grave = self.__dict__.pop("_grave", None)        # the previous step's dropped arrays (as in dflash_step)
        seq = st["seq"]
        ahead = st.get("ahead")                          # this step's verify, dispatched by the previous step
        if draft_override is not None:
            if ahead is not None:                        # (tests) the verify ahead saw the real drafts: undo it
                st, ahead = self.dflash_settle_rows(st), None
            seq = jnp.concatenate([seq[:, :1], jnp.asarray(np.asarray(draft_override, np.int32).reshape(B, T - 1))], 1)
        if ahead is None:
            pend = st.get("pend") or self._pend_identity_rows(st["sets"], T)
            ahead = self._verify_rows(seq, st["sets"], st["pos"], pend)
        else:
            pend = None
        streams, new, hists, feats = ahead
        self._drop(grave, pend)
        del grave, pend, ahead
        temp, top_p, k = sampler.bind(self.mesh)
        r = self._prog_head_accept(B, T, tuple(stop_ids), per_row=True)(self.params["norm"], self.params["lm_head"],
                                                                      streams, seq, st["cand"], st["q"], st["pos"],
                                                                      temp, top_p, k)
        sampler.advance(r["key"])
        # the rollback of every row to its accepted prefix is deferred to the next verify's group programs (or
        # `dflash_settle_rows`), as in `dflash_step`; the draft is queued behind the head (see there)
        next_T = T if next_T is None else next_T
        nseq, cand, q, rings = self._dflash_draft_rows(st["rings"], feats, r["pos_b"], r["n_new_b"], r["bonus"], next_T,
                                                       temp, r["key"])
        out = {"sets": new, "rings": rings, "pos": r["pos_next"], "pend": (hists, r["n_keep"], T), "seq": nseq,
               "cand": cand, "q": q, "T": next_T}
        if self.run_ahead and max(st["pos_h"]) + T + next_T <= self.max_len:     # run ahead (see dflash_step)
            out["ahead"] = self._verify_rows(nseq, new, r["pos_next"], out.pop("pend"))
            out["sets"] = out["ahead"][1]
        emitted = [[int(x) for x in row if x >= 0] for row in np.asarray(r["emitted"])]   # the host waits here
        out["pos_h"] = [p + len(e) for p, e in zip(st["pos_h"], emitted)]
        self._grave = (st, feats, streams, r)
        return emitted, out

    def _dflash_greedy(self):
        if getattr(self, "_greedy_sampler", None) is None:
            self._greedy_sampler = DeviceSampler(0.0, 1.0, seed=0)
        return self._greedy_sampler

    def decode(self, token, caches, pos):
        B = token.shape[0]
        n = self.cfg.n_layers
        extra = caches[n:] if len(caches) > n else []
        logits, caches = self._run(np.asarray(token).reshape(B, 1), caches[:n], pos, True)
        return logits, caches + extra, pos + 1
