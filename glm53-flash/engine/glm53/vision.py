"""GLM-5.3-Flash vision tower (HF `Glm5NextVisionModel`, weights `model.visual.*`, bf16, 1.13 GB) in pure JAX, plus the
image preprocessing of `Glm5NextImageProcessor` (PIL): dynamic-resolution resize to a 28-pixel grid, 14x14 patches
duplicated over the temporal axis (2), 2x2 spatial merge -> one language-model token per 28x28 pixels.

Forward: patch embed (Conv3d == linear over (c, t, ph, pw) = 1176 inputs) -> 24 pre-norm blocks (RMSNorm; attention
with q/k RMSNorm over head_dim 64, 2-D rotary over (h, w) patch positions, full attention within each image; SwiGLU
MLP with the ±10 clamps, all with biases) -> post RMSNorm -> the 2x2 downsample conv (== linear over (c, i, j)) to
4096 -> merger (proj, LayerNorm, GELU, clamped SwiGLU to 10240 and back). Patches are laid out block-major (the 4
patches of a merge block consecutive), so consecutive groups of 4 form one output token; output tokens are the
merge blocks in row-major order, i.e. the order of the `<|image|>` tokens in the prompt.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp

PATCH, MERGE, TEMPORAL = 14, 2, 2
FACTOR = PATCH * MERGE                                                  # 28 pixels per token side
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)          # OPENAI_CLIP_MEAN / STD
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
EPS = 1e-5


# ----------------------------------------------------------------------------- preprocessing (port of the HF processor)
def smart_resize(height, width, min_tokens=16, max_tokens=8000, factor=FACTOR):
    """The HF `smart_resize` for one image (num_frames = temporal_factor = 2): the padded canvas (H, W), both multiples
    of 28, holding between min_tokens and max_tokens 28x28 tokens."""
    ppt = TEMPORAL * factor ** 2
    min_px, max_px = min_tokens * ppt, max_tokens * ppt
    align = lambda v: math.ceil(v / factor) * factor  # noqa: E731
    frames = TEMPORAL
    ah, aw = align(height), align(width)
    if frames * ah * aw < min_px:
        s = math.sqrt(min_px / (TEMPORAL * height * width))
        ah, aw = align(max(1, math.ceil(height * s))), align(max(1, math.ceil(width * s)))
    if frames * ah * aw > max_px:
        lo, hi = 1, height
        bh, bw = factor, factor
        while lo <= hi:
            ch = (lo + hi) // 2
            cw = max(1, math.floor(width * ch / height))
            H, W = align(ch), align(cw)
            if frames * H * W <= max_px:
                bh, bw = H, W
                lo = ch + 1
            else:
                hi = ch - 1
        ah, aw = bh, bw
    return ah, aw


def preprocess(img, min_tokens=16, max_tokens=8000):
    """PIL image -> (patches float32 [gh*gw, 3*2*14*14] in the model's block-major order, (1, gh, gw)). Mirrors the
    HF processor: RGB, bicubic resize of the content to fit the smart_resize canvas (never upscaled unless the image
    is below min_tokens), zero (black) padding right/bottom, rescale 1/255, CLIP mean/std, patchify."""
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    H, W = smart_resize(h, w, min_tokens, max_tokens)
    ppt = TEMPORAL * FACTOR ** 2
    s = min(H / h, W / w)
    if TEMPORAL * h * w >= ppt * min_tokens:
        s = min(1.0, s)
    ch, cw = max(1, min(H, math.floor(h * s))), max(1, min(W, math.floor(w * s)))
    if (ch, cw) != (h, w):
        img = img.resize((cw, ch), Image.BICUBIC)
    x = np.zeros((H, W, 3), np.uint8)
    x[:ch, :cw] = np.asarray(img, np.uint8)
    x = ((x.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)             # [3, H, W]
    gh, gw = H // PATCH, W // PATCH
    p = x.reshape(3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    p = p.transpose(1, 4, 2, 5, 0, 3, 6)                                              # [bh, bw, mi, mj, c, ph, pw]
    p = p.reshape(gh * gw, 3, 1, PATCH, PATCH)
    p = np.broadcast_to(p, (gh * gw, 3, TEMPORAL, PATCH, PATCH)).reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH)
    return np.ascontiguousarray(p), (1, gh, gw)


def n_tokens(grid):
    t, gh, gw = grid
    return t * gh * gw // (MERGE * MERGE)


def position_ids(grids):
    """[N, 2] (h, w) patch positions in the block-major order, for all images concatenated."""
    out = []
    for t, gh, gw in grids:
        hh, ww = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        shp = (gh // MERGE, MERGE, gw // MERGE, MERGE)
        hh = hh.reshape(shp).transpose(0, 2, 1, 3).reshape(-1)
        ww = ww.reshape(shp).transpose(0, 2, 1, 3).reshape(-1)
        out.append(np.tile(np.stack([hh, ww], -1), (t, 1)))
    return np.concatenate(out, 0).astype(np.int32)


def segment_ids(grids):
    """[N] image index of every patch (attention stays within an image)."""
    return np.concatenate([np.full(t * gh * gw, i, np.int32) for i, (t, gh, gw) in enumerate(grids)])


# ----------------------------------------------------------------------------- parameters
def from_state_dict(get, depth, hidden, dtype=np.float32):
    """`get(name)` -> np array for the HF names without the `model.visual.` prefix -> our params (matrices
    transposed to [in, out]; the two convs flattened to matrices)."""
    t = lambda n: np.ascontiguousarray(np.asarray(get(n), dtype).T)  # noqa: E731
    v = lambda n: np.asarray(get(n), dtype)  # noqa: E731
    blocks = []
    for i in range(depth):
        b = f"blocks.{i}."
        blocks.append({"n1": v(b + "norm1.weight"), "n2": v(b + "norm2.weight"),
                       "qkv": t(b + "attn.qkv.weight"), "qkv_b": v(b + "attn.qkv.bias"),
                       "proj": t(b + "attn.proj.weight"), "proj_b": v(b + "attn.proj.bias"),
                       "qn": v(b + "attn.q_norm.weight"), "kn": v(b + "attn.k_norm.weight"),
                       "gate": t(b + "mlp.gate_proj.weight"), "gate_b": v(b + "mlp.gate_proj.bias"),
                       "up": t(b + "mlp.up_proj.weight"), "up_b": v(b + "mlp.up_proj.bias"),
                       "down": t(b + "mlp.down_proj.weight"), "down_b": v(b + "mlp.down_proj.bias")})
    pe = np.asarray(get("patch_embed.proj.weight"), dtype)                              # [C, 3, 2, 14, 14]
    ds = np.asarray(get("downsample.weight"), dtype)                                    # [O, C, 2, 2]
    return {"patch": np.ascontiguousarray(pe.reshape(pe.shape[0], -1).T), "patch_b": v("patch_embed.proj.bias"),
            "blocks": blocks, "post_ln": v("post_layernorm.weight"),
            "ds": np.ascontiguousarray(ds.reshape(ds.shape[0], -1).T), "ds_b": v("downsample.bias"),
            "m_proj": t("merger.proj.weight"), "m_ln": v("merger.post_projection_norm.weight"),
            "m_ln_b": v("merger.post_projection_norm.bias"), "m_gate": t("merger.gate_proj.weight"),
            "m_up": t("merger.up_proj.weight"), "m_down": t("merger.down_proj.weight"),
            "n_heads": None}


def from_hf_module(m, dtype=np.float32):
    sd = {k: v.detach().float().cpu().numpy() for k, v in m.state_dict().items()}
    p = from_state_dict(sd.__getitem__, m.config.depth, m.config.hidden_size, dtype)
    p["n_heads"] = m.config.num_heads
    return p


def load_vision(reader, dtype=np.float32, depth=24, hidden=1024, n_heads=16):
    """From a checkpoint `ShardReader` (names `model.visual.*`, all in the last shard)."""
    p = from_state_dict(lambda n: reader.get("model.visual." + n, dtype), depth, hidden, dtype)
    p["n_heads"] = n_heads
    return p


# ----------------------------------------------------------------------------- forward
def _rms(x, w, eps=EPS):
    xf = x.astype(jnp.float32)
    return (xf * jax.lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + eps) * w.astype(jnp.float32)).astype(x.dtype)


def _rope(pos, head_dim, theta=10000.0):
    d = head_dim // 2                                                    # rotary dim per axis pair = 32 -> 16 freqs
    inv = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float32) / d))
    f = (pos.astype(np.float32)[:, :, None] * inv[None, None, :]).reshape(pos.shape[0], -1)   # [N, 32] (h freqs, w freqs)
    emb = np.concatenate([f, f], -1)                                     # [N, 64]
    return jnp.asarray(np.cos(emb)), jnp.asarray(np.sin(emb))


def _rot_half(x):
    a, b = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-b, a], -1)


def _attention(p, x, cos, sin, same, n_heads, dtype):
    N, C = x.shape
    hd = C // n_heads
    qkv = (jnp.dot(x, p["qkv"].astype(dtype), preferred_element_type=jnp.float32) + p["qkv_b"]).reshape(N, 3, n_heads, hd)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q, k = _rms(q, p["qn"]), _rms(k, p["kn"])
    c, s = cos[:, None, :], sin[:, None, :]
    q = q * c + _rot_half(q) * s
    k = k * c + _rot_half(k) * s
    sc = jnp.einsum("nhd,mhd->hnm", q, k) * (hd ** -0.5)
    sc = jnp.where(same[None], sc, -jnp.inf)
    a = jax.nn.softmax(sc, axis=-1)
    o = jnp.einsum("hnm,mhd->nhd", a, v).reshape(N, C).astype(dtype)
    return jnp.dot(o, p["proj"].astype(dtype), preferred_element_type=jnp.float32) + p["proj_b"]


def _swiglu(g, u, limit=10.0):
    g = jnp.minimum(g, limit)
    u = jnp.clip(u, -limit, limit)
    return jax.nn.silu(g) * u


def forward(p, patches, grids, dtype=jnp.float32):
    """patches [N, 1176] (all images concatenated), grids [(t, gh, gw), ...] -> [sum n_tokens, out_hidden] f32.
    Shapes are static per call (one compile per (N, layout)); pad N to a bucket for the TPU."""
    hd = p["blocks"][0]["qn"].shape[0]                                   # static shapes (works under jit; the
    pos, seg = position_ids(grids), segment_ids(grids)                    # n_heads entry may be a traced leaf)
    same = jnp.asarray(seg[:, None] == seg[None, :])
    x = jnp.dot(jnp.asarray(patches, dtype), p["patch"].astype(dtype), preferred_element_type=jnp.float32) + p["patch_b"]
    x = x.astype(dtype)
    n_heads = x.shape[-1] // hd
    cos, sin = _rope(pos, hd)
    for b in p["blocks"]:
        x = x + _attention(b, _rms(x, b["n1"]), cos, sin, same, n_heads, dtype).astype(dtype)
        h = _rms(x, b["n2"])
        g = jnp.dot(h, b["gate"].astype(dtype), preferred_element_type=jnp.float32) + b["gate_b"]
        u = jnp.dot(h, b["up"].astype(dtype), preferred_element_type=jnp.float32) + b["up_b"]
        x = x + (jnp.dot(_swiglu(g, u).astype(dtype), b["down"].astype(dtype), preferred_element_type=jnp.float32) + b["down_b"]).astype(dtype)
    x = _rms(x, p["post_ln"])
    N, C = x.shape
    x4 = x.reshape(N // 4, 2, 2, C).transpose(0, 3, 1, 2).reshape(N // 4, C * 4)   # (c, i, j) like the conv weight
    y = jnp.dot(x4, p["ds"].astype(dtype), preferred_element_type=jnp.float32) + p["ds_b"]
    y = jnp.dot(y.astype(dtype), p["m_proj"].astype(dtype), preferred_element_type=jnp.float32)
    mu = y.mean(-1, keepdims=True)
    var = ((y - mu) ** 2).mean(-1, keepdims=True)
    y = (y - mu) * jax.lax.rsqrt(var + EPS) * p["m_ln"] + p["m_ln_b"]
    y = jax.nn.gelu(y, approximate=False).astype(dtype)
    g = jnp.dot(y, p["m_gate"].astype(dtype), preferred_element_type=jnp.float32)
    u = jnp.dot(y, p["m_up"].astype(dtype), preferred_element_type=jnp.float32)
    return jnp.dot(_swiglu(g, u).astype(dtype), p["m_down"].astype(dtype), preferred_element_type=jnp.float32)


# ----------------------------------------------------------------------------- TP-sharded forward (8 chips)
def shard_specs(p):
    """PartitionSpecs for `to_sharded(p)`: attention heads over AXIS (qkv columns / proj rows regrouped per head),
    MLP and merger intermediate columns over AXIS (down rows), the downsample columns and m_proj rows over AXIS;
    small vectors replicated. ~140 MB per chip in bf16 for the real tower."""
    from jax.sharding import PartitionSpec as P
    from glm53.engine import AXIS
    R = P()
    blk = {"n1": R, "n2": R, "qkv": P(None, None, AXIS, None), "qkv_b": P(None, AXIS, None),
           "proj": P(AXIS, None, None), "proj_b": R, "qn": R, "kn": R,
           "gate": P(None, AXIS), "gate_b": P(AXIS), "up": P(None, AXIS), "up_b": P(AXIS), "down": P(AXIS, None), "down_b": R}
    blk = {k: P(None, *v) for k, v in blk.items()}                  # blocks are STACKED on a leading axis (lax.scan)
    return {"patch": R, "patch_b": R, "blocks": blk, "post_ln": R,
            "ds": P(None, AXIS), "ds_b": P(AXIS), "m_proj": P(AXIS, None), "m_ln": R, "m_ln_b": R,
            "m_gate": P(None, AXIS), "m_up": P(None, AXIS), "m_down": P(AXIS, None), "n_heads": None}


def to_sharded(p, mesh, dtype=jnp.bfloat16):
    """Regroup the head-sharded matrices ([in, 3, H, hd] qkv / [H, hd, out] proj) and place every array on the mesh."""
    from jax.sharding import NamedSharding
    H = p["n_heads"]
    q = dict(p)
    blocks = []
    for b in p["blocks"]:
        C = b["qkv"].shape[0]
        hd = C // H
        blocks.append({**b, "qkv": np.asarray(b["qkv"]).reshape(C, 3, H, hd), "qkv_b": np.asarray(b["qkv_b"]).reshape(3, H, hd),
                       "proj": np.asarray(b["proj"]).reshape(H, hd, C)})
    q["blocks"] = {k: np.stack([np.asarray(b[k]) for b in blocks]) for k in blocks[0]}   # [depth, ...] for lax.scan
    specs = shard_specs(p)

    def place(a, sp):
        if sp is None or a is None:
            return a
        a = np.asarray(a)
        a = a.astype(np.float32) if a.dtype == np.float32 else a
        return jax.device_put(jnp.asarray(a, dtype if a.ndim >= 2 else jnp.float32), NamedSharding(mesh, sp))
    out = jax.tree.map(place, q, specs, is_leaf=lambda x: x is None or not isinstance(x, (dict, list)))
    out["n_heads"] = H
    return out


def make_forward_sharded(p_sharded, mesh, dtype=jnp.bfloat16, q_chunk=1024):
    """-> f(patches [N,1176] np, grids tuple) -> [n_tokens, out] f32, one jit per (N, grids). Per chip: its heads
    (attention scores chunked over q_chunk queries), its MLP/merger columns; two psums per block. The 24 blocks run
    as a `lax.scan` over the stacked parameters (one block body per executable: TPU executables live in HBM, and
    five unrolled buckets held ~0.4 GB)."""
    from glm53.engine import AXIS, shard_map
    from jax import lax
    from jax.sharding import PartitionSpec as P
    R = P()
    n_heads = p_sharded["n_heads"]
    specs = shard_specs(p_sharded)
    del specs["n_heads"]
    pp = {k: v for k, v in p_sharded.items() if k != "n_heads"}
    n_dev = mesh.devices.size
    Hl = n_heads // n_dev
    cache = {}

    def build(N):
        """One program per patch count N: positions / segments arrive as arrays, so images of any grid share it."""
        def attn(b, x, cos, sin, same):
            C = x.shape[-1]
            hd = C // n_heads
            qkv = jnp.einsum("nc,cthd->nthd", x, b["qkv"].astype(dtype), preferred_element_type=jnp.float32) + b["qkv_b"]
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                                 # [N, Hl, hd]
            q, k = _rms(q, b["qn"]), _rms(k, b["kn"])
            c, s_ = cos[:, None, :], sin[:, None, :]
            q = q * c + _rot_half(q) * s_
            k = k * c + _rot_half(k) * s_
            outs = []
            for a in range(0, N, q_chunk):
                sc = jnp.einsum("nhd,mhd->hnm", q[a:a + q_chunk], k) * (hd ** -0.5)
                sc = jnp.where(same[a:a + q_chunk][None], sc, -jnp.inf)
                outs.append(jnp.einsum("hnm,mhd->nhd", jax.nn.softmax(sc, axis=-1), v))
            o = jnp.concatenate(outs, 0).astype(dtype)                                # [N, Hl, hd]
            y = jnp.einsum("nhd,hdc->nc", o, b["proj"].astype(dtype), preferred_element_type=jnp.float32)
            return lax.psum(y, AXIS) + b["proj_b"]

        def prog(p, patches, cos, sin, same):
            x = jnp.dot(patches.astype(dtype), p["patch"].astype(dtype), preferred_element_type=jnp.float32) + p["patch_b"]
            x = x.astype(dtype)

            def block(x, b):
                x = x + attn(b, _rms(x, b["n1"]), cos, sin, same).astype(dtype)
                h = _rms(x, b["n2"])
                g = jnp.dot(h, b["gate"].astype(dtype), preferred_element_type=jnp.float32) + b["gate_b"]
                u = jnp.dot(h, b["up"].astype(dtype), preferred_element_type=jnp.float32) + b["up_b"]
                y = jnp.dot(_swiglu(g, u).astype(dtype), b["down"].astype(dtype), preferred_element_type=jnp.float32)
                return x + (lax.psum(y, AXIS) + b["down_b"]).astype(dtype), None
            x, _ = lax.scan(block, x, p["blocks"])
            x = _rms(x, p["post_ln"])
            C = x.shape[-1]
            x4 = x.reshape(N // 4, 2, 2, C).transpose(0, 3, 1, 2).reshape(N // 4, C * 4)
            y = jnp.dot(x4, p["ds"].astype(dtype), preferred_element_type=jnp.float32) + p["ds_b"]   # local columns
            y = lax.psum(jnp.dot(y.astype(dtype), p["m_proj"].astype(dtype), preferred_element_type=jnp.float32), AXIS)
            mu = y.mean(-1, keepdims=True)
            var = ((y - mu) ** 2).mean(-1, keepdims=True)
            y = (y - mu) * lax.rsqrt(var + EPS) * p["m_ln"] + p["m_ln_b"]
            y = jax.nn.gelu(y, approximate=False).astype(dtype)
            g = jnp.dot(y, p["m_gate"].astype(dtype), preferred_element_type=jnp.float32)
            u = jnp.dot(y, p["m_up"].astype(dtype), preferred_element_type=jnp.float32)
            return lax.psum(jnp.dot(_swiglu(g, u).astype(dtype), p["m_down"].astype(dtype), preferred_element_type=jnp.float32), AXIS)
        return jax.jit(shard_map(prog, mesh=mesh, in_specs=(specs, R, R, R, R), out_specs=R, check_vma=False))

    hd = pp["blocks"]["qn"].shape[-1]

    def run(patches, grids):
        grids = tuple(tuple(int(v) for v in g) for g in grids)
        N = patches.shape[0]
        if N not in cache:
            cache[N] = build(N)
        pos, seg = position_ids(grids), segment_ids(grids)
        cos, sin = _rope(pos, hd)
        same = jnp.asarray(seg[:, None] == seg[None, :])
        return cache[N](pp, jnp.asarray(patches, dtype), cos, sin, same)
    return run
