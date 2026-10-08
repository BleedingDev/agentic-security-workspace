"""Base store: everything a serving build needs besides the routed experts — the non-expert parameters exactly as
the engine holds them (int8 nodes included), the vision tower and the tokenizer / config files — written once to a
directory (a Kaggle kernel's output, turned into a dataset) and read back in seconds instead of from the 265 GB FP8
checkpoint.

Layout: `manifest.json`, `top.npz` (embed / norm / lm_head), `layer_NN.npz` (one per layer, the expert tables left
out), `vision.npz`, and the tokenizer files. Trees are flattened to "a/b/0/c" keys (lists as numbered keys); bf16
arrays are stored as uint16 bit patterns with the dtype recorded; non-array leaves go into the manifest."""
import json
import os
import shutil

import numpy as np
import jax

EXPERT_KEYS = ("gate_q", "up_q", "down_q")                     # resident expert tables (glm53.resident), never stored
HF_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "special_tokens_map.json", "preprocessor_config.json", "video_preprocessor_config.json")


def flatten(tree, prefix=""):
    """-> {"a/b/0/c": leaf}; dicts and lists are containers, everything else is a leaf."""
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(flatten(v, f"{prefix}{k}/"))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            out.update(flatten(v, f"{prefix}{i}/"))
    else:
        out[prefix[:-1]] = tree
    return out


def unflatten(flat):
    """Inverse of `flatten`: a container whose keys are all integers becomes a list (in index order)."""
    root = {}
    for key, leaf in flat.items():
        parts = key.split("/")
        d = root
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = leaf

    def fix(d):
        if not isinstance(d, dict):
            return d
        d = {k: fix(v) for k, v in d.items()}
        if d and all(k.isdigit() for k in d):
            return [d[str(i)] for i in range(len(d))]
        return d
    return fix(root)


def _host(x):
    a = np.asarray(jax.device_get(x)) if hasattr(x, "shape") else x
    if getattr(a, "dtype", None) is not None and a.dtype.name == "bfloat16":
        return a.view(np.uint16), "bfloat16"
    return a, str(a.dtype)


def save_tree(path, tree):
    """One npz for a tree: arrays stored natively (bf16 as uint16), other leaves in the "__meta__" entry."""
    arrays, dtypes, others = {}, {}, {}
    for k, v in flatten(tree).items():
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            a, dt = _host(v)
            arrays[k] = a
            dtypes[k] = dt
        else:
            others[k] = v
    meta = json.dumps({"dtypes": dtypes, "others": others})
    np.savez(path, __meta__=np.frombuffer(meta.encode(), np.uint8), **arrays)


def load_tree(path):
    import ml_dtypes
    with np.load(path) as z:
        meta = json.loads(bytes(z["__meta__"]).decode())
        flat = {}
        for k, dt in meta["dtypes"].items():
            a = z[k]
            flat[k] = a.view(ml_dtypes.bfloat16) if dt == "bfloat16" else a
    flat.update(meta["others"])
    return unflatten(flat)


def dump(out_dir, params, vision=None, hf_dir=None, meta=None, log=print):
    """Write the store: `params` = the engine's parameter tree (device or host arrays; expert tables skipped),
    `vision` = the vision tower's host parameter tree (as `glm53.vision.load_vision` returns it, any dtype),
    `hf_dir` = where the tokenizer / config files are."""
    os.makedirs(out_dir, exist_ok=True)
    save_tree(os.path.join(out_dir, "top.npz"), {k: params[k] for k in ("embed", "norm", "lm_head")})
    n = len(params["layers"])
    for i, L in enumerate(params["layers"]):
        L = {**L, "mlp": {k: v for k, v in L["mlp"].items() if k not in EXPERT_KEYS}}
        save_tree(os.path.join(out_dir, f"layer_{i:02d}.npz"), L)
        if i % 10 == 0 or i == n - 1:
            log(f"   base store: layer {i + 1}/{n}")
    if vision is not None:
        save_tree(os.path.join(out_dir, "vision.npz"), vision)
    files = []
    if hf_dir:
        for f in HF_FILES:
            src = os.path.join(hf_dir, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out_dir, f)); files.append(f)
    manifest = {"n_layers": n, "vision": vision is not None, "files": files, **(meta or {})}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 1e9
    log(f"   base store written: {out_dir} ({size:.2f} GB, {n} layers, vision {vision is not None})")
    return manifest


def load(in_dir):
    """-> {"top": {embed, norm, lm_head}, "layers": [layer trees without expert tables], "vision": tree or None,
    "manifest": dict}. Host arrays; bf16 restored."""
    manifest = json.load(open(os.path.join(in_dir, "manifest.json")))
    top = load_tree(os.path.join(in_dir, "top.npz"))
    layers = [load_tree(os.path.join(in_dir, f"layer_{i:02d}.npz")) for i in range(manifest["n_layers"])]
    vision = load_tree(os.path.join(in_dir, "vision.npz")) if manifest.get("vision") else None
    return {"top": top, "layers": layers, "vision": vision, "manifest": manifest}
