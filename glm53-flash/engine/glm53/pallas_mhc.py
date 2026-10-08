"""Pallas TPU kernels for the mHC sites of a decoder layer at decode / short-verify shapes (N = B*T tokens, a few to a
few dozen). One kernel per site instead of XLA's chain of small fusions (norm, the [N, hc*D] x [hc*D, (2+hc)*hc]
projection, sigmoids, softmax, the Sinkhorn iterations, the collapse, the block norm; plus the residual update of
the previous block at the middle site):

  mhc_pre(streams, p, ln_w, cfg)                      == model.hc_pre, then model.rmsnorm(h, ln_w)
  mhc_post_pre(post, comb, y, streams, p, ln_w, cfg)  == model.hc_post(post, comb, y, streams), then the above
  mhc_post(post, comb, y, streams)                    == model.hc_post (the end of a layer)

Shapes as in model.py: streams [B, T, hc, D] (model dtype), y and h [B, T, D], post [B, T, hc] f32; comb is FLAT,
[B, T, hc*hc] f32 with comb[h, k] (the weight of stream h in new stream k) at h*hc + k, so it moves between the
kernels of a layer without relayouts (mhc_pre / mhc_post_pre return it so, mhc_post_pre / mhc_post take it so). The arithmetic follows model.py op by op
(hc_post in the model dtype with f32 accumulation, hc_pre in f32); sums run in a different order than XLA's
reductions, so results agree to rounding, not bit for bit. `prec`: "bf16" rounds the operands of the two small
contractions (the mix projection, the collapse) to bf16 as XLA's default matmul precision does for f32 dots on the
TPU; "f32" keeps them exact (the reference model's f32 math).

The projection weight is read transposed, fnT [(2+hc)*hc, hc*D]: [hc*D, 24] would be padded to 128 lanes in VMEM
(and in HBM). `p` may hold "fnT" (made once at load) or "fn" (transposed here, one XLA transpose per call).
Everything is one grid step with whole arrays in VMEM (a decode site moves ~0.1 MB per token plus the 0.8 MB
weight). The 16 Sinkhorn entries are separate [N, 1] columns, so every step is the reference's own elementwise op."""
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _col(x, j):
    """Column j of a small [N, W] f32 value as [N, 1] (exact: one nonzero term per row)."""
    lane = lax.broadcasted_iota(jnp.int32, x.shape, 1)
    return jnp.sum(jnp.where(lane == j, x, 0.0), axis=1, keepdims=True)


def _assemble(cols, n_rows):
    """[N, 1] columns -> [N, len(cols)]."""
    W = len(cols)
    lane = lax.broadcasted_iota(jnp.int32, (n_rows, W), 1)
    out = jnp.zeros((n_rows, W), jnp.float32)
    for j, c in enumerate(cols):
        out = jnp.where(lane == j, jnp.broadcast_to(c, (n_rows, W)), out)
    return out


def _sigmoid(x):
    return jax.nn.sigmoid(x)


def _kernel(*refs, hc, D, iters, eps, hc_eps, with_post, prec, with_pre=True, opts=()):
    """opts (static): "once" = the residual update rounded once (f32 coefficients, products and sums: what XLA's
    fusions compute on the TPU, which keep f32 intermediates), "nocr" = the collapse not rounded to the model dtype
    before the norm (likewise), "recip" = Sinkhorn divisions as one reciprocal per row/column sum, "fnany" = the
    projection weight DMA'd by hand, overlapped with the residual update; timing stubs: "empty", "it1", "nodot"."""
    f32 = jnp.float32
    if "fnany" in opts:
        *refs, fn_vmem, sem = refs
    if not with_pre:
        pin_ref, cin_ref, y_ref, s_ref, ns_ref = refs
    elif with_post:
        pin_ref, cin_ref, y_ref, s_ref, fnT_ref, sc_ref, ln_ref, ns_ref, po_ref, co_ref, h_ref = refs
    else:
        s_ref, fnT_ref, sc_ref, ln_ref, po_ref, co_ref, h_ref = refs
    dt = s_ref.dtype
    N = s_ref.shape[0]
    M = (2 + hc) * hc
    if "fnany" in opts:
        cp = pltpu.make_async_copy(fnT_ref, fn_vmem, sem)
        cp.start()
        fnT_ref = fn_vmem
    if "empty" in opts:
        if with_post:
            ns_ref[...] = s_ref[...]
        if with_pre:
            if "fnany" in opts:
                cp.wait()
            po_ref[...] = jnp.zeros(po_ref.shape, f32)
            co_ref[...] = jnp.zeros(co_ref.shape, f32)
            h_ref[...] = s_ref[:, :D]
        return
    if "it1" in opts:
        iters = 1
    if with_post:
        # hc_post: new[k] = post[k] * y + sum_h comb[h, k] * streams[h], in the model dtype (f32 products and sums,
        # rounded where the reference's ops round: the product, the einsum, the add)
        y = y_ref[...].astype(f32)
        pin = pin_ref[...]
        cin = cin_ref[...]
        once = "once" in opts
        rnd = (lambda v: v) if once else (lambda v: v.astype(dt).astype(f32))   # noqa: E731
        pc = [rnd(_col(pin, k)) for k in range(hc)]
        cc = [rnd(_col(cin, j)) for j in range(hc * hc)]
        olds = [s_ref[:, h * D:(h + 1) * D].astype(f32) for h in range(hc)]
        xs = []
        for k in range(hc):
            a = rnd(pc[k] * y)
            e = cc[k] * olds[0]
            for h in range(1, hc):
                e = e + cc[h * hc + k] * olds[h]
            nk = (a + rnd(e)).astype(dt)
            ns_ref[:, k * D:(k + 1) * D] = nk
            xs.append(nk.astype(f32))
    else:
        xs = [s_ref[:, h * D:(h + 1) * D].astype(f32) for h in range(hc)]
    if not with_pre:
        return
    # hc_pre: unweighted RMS norm over all hc*D values of a token, the mix projection
    ss = jnp.sum(xs[0] * xs[0], axis=1, keepdims=True)
    for h in range(1, hc):
        ss = ss + jnp.sum(xs[h] * xs[h], axis=1, keepdims=True)
    r = lax.rsqrt(ss / (hc * D) + eps)
    if "fnany" in opts:
        cp.wait()
    mix = None
    for h in range(hc):
        fl = xs[h] * r
        w = fnT_ref[:, h * D:(h + 1) * D]
        if prec == "bf16":
            part = lax.dot_general(fl.astype(jnp.bfloat16), w.astype(jnp.bfloat16), (((1,), (1,)), ((), ())),
                                   preferred_element_type=f32)
        else:
            part = lax.dot_general(fl, w.astype(f32), (((1,), (1,)), ((), ())), preferred_element_type=f32,
                                   precision=lax.Precision.HIGHEST)
        mix = part if mix is None else mix + part                                     # [N, M]
    if "nodot" in opts:
        mix = jnp.zeros_like(mix)
    s0, s1, s2 = sc_ref[M], sc_ref[M + 1], sc_ref[M + 2]
    base = [sc_ref[j] for j in range(M)]
    m = [_col(mix, j) for j in range(M)]
    pre = [_sigmoid(m[j] * s0 + base[j]) + hc_eps for j in range(hc)]
    post = [2.0 * _sigmoid(m[hc + j] * s1 + base[hc + j]) for j in range(hc)]
    lg = [m[2 * hc + j] * s2 + base[2 * hc + j] for j in range(hc * hc)]              # [i * hc + j]
    c = []
    for i in range(hc):                                                               # softmax over j, + hc_eps
        row = lg[i * hc:(i + 1) * hc]
        mx = row[0]
        for v in row[1:]:
            mx = jnp.maximum(mx, v)
        ex = [jnp.exp(v - mx) for v in row]
        den = ex[0]
        for v in ex[1:]:
            den = den + v
        c += [v / den + hc_eps for v in ex]

    def csum(c, j):
        out = c[j]
        for i in range(1, hc):
            out = out + c[i * hc + j]
        return out

    def rsum(c, i):
        out = c[i * hc]
        for j in range(1, hc):
            out = out + c[i * hc + j]
        return out

    if "recip" in opts:
        div = lambda a, b: a * b                                                      # noqa: E731
        inv = lambda v: 1.0 / v                                                       # noqa: E731
    else:
        div = lambda a, b: a / b                                                      # noqa: E731
        inv = lambda v: v                                                             # noqa: E731
    cs = [inv(csum(c, j) + hc_eps) for j in range(hc)]
    c = [div(c[i * hc + j], cs[j]) for i in range(hc) for j in range(hc)]
    for _ in range(iters - 1):
        rs = [inv(rsum(c, i) + hc_eps) for i in range(hc)]
        c = [div(c[i * hc + j], rs[i]) for i in range(hc) for j in range(hc)]
        cs = [inv(csum(c, j) + hc_eps) for j in range(hc)]
        c = [div(c[i * hc + j], cs[j]) for i in range(hc) for j in range(hc)]
    po_ref[...] = _assemble(post, N)
    co_ref[...] = _assemble(c, N)
    # collapse with pre, then the block's RMS norm (weighted)
    if prec == "bf16":
        pr = [v.astype(jnp.bfloat16).astype(f32) for v in pre]
        col = pr[0] * xs[0].astype(jnp.bfloat16).astype(f32)
        for h in range(1, hc):
            col = col + pr[h] * xs[h].astype(jnp.bfloat16).astype(f32)
    else:
        col = pre[0] * xs[0]
        for h in range(1, hc):
            col = col + pre[h] * xs[h]
    cf = col if "nocr" in opts else col.astype(dt).astype(f32)
    r2 = lax.rsqrt(jnp.sum(cf * cf, axis=1, keepdims=True) / D + eps)
    h_ref[...] = (ln_ref[...].astype(f32) * (cf * r2)).astype(dt)


@functools.partial(jax.jit, static_argnames=("hc", "interpret"))
def _call_post(post, comb, y, streams, *, hc, interpret):
    N, HD = streams.shape
    D = HD // hc
    V = pltpu.VMEM
    kern = functools.partial(_kernel, hc=hc, D=D, iters=0, eps=0.0, hc_eps=0.0, with_post=True, prec="f32",
                             with_pre=False)
    return pl.pallas_call(
        kern, out_shape=jax.ShapeDtypeStruct((N, HD), streams.dtype),
        in_specs=[pl.BlockSpec(memory_space=V)] * 4, out_specs=pl.BlockSpec(memory_space=V),
        interpret=interpret,
    )(post, comb, y, streams)


def _scalars(p):
    return jnp.concatenate([p["base"].astype(jnp.float32).reshape(-1), p["scale"].astype(jnp.float32).reshape(-1)])


def _fnT(p):
    return p["fnT"] if "fnT" in p else p["fn"].T


@functools.partial(jax.jit, static_argnames=("hc", "iters", "eps", "hc_eps", "prec", "interpret", "opts"))
def _call_pre(streams, fnT, sc, ln_w, *, hc, iters, eps, hc_eps, prec, interpret, opts=()):
    N, HD = streams.shape
    D = HD // hc
    V = pltpu.VMEM
    kern = functools.partial(_kernel, hc=hc, D=D, iters=iters, eps=eps, hc_eps=hc_eps, with_post=False, prec=prec,
                             opts=opts)
    fspec, scratch = _fn_specs(opts, fnT)
    return pl.pallas_call(
        kern,
        out_shape=(jax.ShapeDtypeStruct((N, hc), jnp.float32), jax.ShapeDtypeStruct((N, hc * hc), jnp.float32),
                   jax.ShapeDtypeStruct((N, D), streams.dtype)),
        in_specs=[pl.BlockSpec(memory_space=V), fspec, pl.BlockSpec(memory_space=pltpu.SMEM),
                  pl.BlockSpec(memory_space=V)],
        out_specs=(pl.BlockSpec(memory_space=V),) * 3,
        scratch_shapes=scratch,
        interpret=interpret,
    )(streams, fnT, sc, ln_w.reshape(1, D))


def _fn_specs(opts, fnT):
    """(in_spec of the projection weight, scratch shapes)."""
    if "fnany" in opts:
        return pl.BlockSpec(memory_space=pl.ANY), [pltpu.VMEM(fnT.shape, fnT.dtype), pltpu.SemaphoreType.DMA(())]
    return pl.BlockSpec(memory_space=pltpu.VMEM), []


@functools.partial(jax.jit, static_argnames=("hc", "iters", "eps", "hc_eps", "prec", "interpret", "opts"))
def _call_post_pre(post, comb, y, streams, fnT, sc, ln_w, *, hc, iters, eps, hc_eps, prec, interpret, opts=()):
    N, HD = streams.shape
    D = HD // hc
    V = pltpu.VMEM
    kern = functools.partial(_kernel, hc=hc, D=D, iters=iters, eps=eps, hc_eps=hc_eps, with_post=True, prec=prec,
                             opts=opts)
    fspec, scratch = _fn_specs(opts, fnT)
    return pl.pallas_call(
        kern,
        out_shape=(jax.ShapeDtypeStruct((N, HD), streams.dtype), jax.ShapeDtypeStruct((N, hc), jnp.float32),
                   jax.ShapeDtypeStruct((N, hc * hc), jnp.float32), jax.ShapeDtypeStruct((N, D), streams.dtype)),
        in_specs=[pl.BlockSpec(memory_space=V)] * 4 + [fspec, pl.BlockSpec(memory_space=pltpu.SMEM),
                                                       pl.BlockSpec(memory_space=V)],
        out_specs=(pl.BlockSpec(memory_space=V),) * 4,
        scratch_shapes=scratch,
        interpret=interpret,
    )(post, comb, y, streams, fnT, sc, ln_w.reshape(1, D))


OPTS = ("once", "nocr")     # rounding as XLA's TPU fusions do (j140, v6e 2026-10-04: closest to XLA's chain)


def mhc_pre(streams, p, ln_w, cfg, prec="bf16", interpret=False, opts=OPTS):
    """streams [B, T, hc, D] -> (post [B, T, hc] f32, comb [B, T, hc*hc] f32 (flat: comb[h, k] at h*hc + k),
    h [B, T, D]): model.hc_pre + model.rmsnorm(h, ln_w)."""
    B, T, hc, D = streams.shape
    po, co, h = _call_pre(streams.reshape(B * T, hc * D), _fnT(p), _scalars(p), ln_w, hc=hc, iters=cfg.hc_iters,
                          eps=cfg.eps, hc_eps=cfg.hc_eps, prec=prec, interpret=interpret, opts=tuple(opts))
    return po.reshape(B, T, hc), co.reshape(B, T, hc * hc), h.reshape(B, T, D)


def mhc_post_pre(post, comb, y, streams, p, ln_w, cfg, prec="bf16", interpret=False, opts=OPTS):
    """model.hc_post(post, comb, y, streams), then mhc_pre of the result -> (streams, post, comb, h); comb flat
    [B, T, hc*hc] in and out (as mhc_pre returns it)."""
    B, T, hc, D = streams.shape
    N = B * T
    ns, po, co, h = _call_post_pre(post.reshape(N, hc), comb.reshape(N, hc * hc), y.reshape(N, D),
                                   streams.reshape(N, hc * D), _fnT(p), _scalars(p), ln_w, hc=hc, iters=cfg.hc_iters,
                                   eps=cfg.eps, hc_eps=cfg.hc_eps, prec=prec, interpret=interpret, opts=tuple(opts))
    return ns.reshape(B, T, hc, D), po.reshape(B, T, hc), co.reshape(B, T, hc * hc), h.reshape(B, T, D)


def mhc_post(post, comb, y, streams, interpret=False):
    """model.hc_post with the flat comb [B, T, hc*hc] -> streams [B, T, hc, D]."""
    B, T, hc, D = streams.shape
    N = B * T
    ns = _call_post(post.reshape(N, hc), comb.reshape(N, hc * hc), y.reshape(N, D), streams.reshape(N, hc * D), hc=hc,
                    interpret=interpret)
    return ns.reshape(B, T, hc, D)
