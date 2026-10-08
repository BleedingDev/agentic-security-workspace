"""glm53.vision vs the HF `Glm5NextVisionModel` (random tiny weights, f32): merged image embeddings must match for one
and for two images of different sizes; preprocessing produces the HF patch layout and token count."""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel  # noqa: E402

from glm53 import vision as V  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


def tiny(num_heads=4):
    torch.manual_seed(0)
    cfg = Glm5NextVisionConfig(depth=2, hidden_size=64, num_heads=num_heads, intermediate_size=96, out_hidden_size=96,
                               projection_intermediate_size=128)
    m = Glm5NextVisionModel(cfg).float().eval()
    with torch.no_grad():                                   # non-trivial norms / biases
        for n, p in m.named_parameters():
            if p.ndim == 1:
                p.add_(torch.randn_like(p) * 0.2)
    return m, V.from_hf_module(m)


@pytest.mark.parametrize("grids", [[(1, 4, 6)], [(1, 4, 6), (1, 2, 2)]])
def test_forward_matches_hf(grids):
    m, p = tiny()
    rng = np.random.default_rng(1)
    N = sum(t * h * w for t, h, w in grids)
    patches = rng.standard_normal((N, 3 * 2 * 14 * 14)).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.tensor(patches), grid_thw=torch.tensor(grids)).pooler_output.numpy()
    out = np.asarray(V.forward(p, patches, grids))
    assert out.shape == ref.shape == (sum(V.n_tokens(g) for g in grids), 96)
    np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)


def test_preprocess_layout():
    from PIL import Image
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    patches, grid = V.preprocess(Image.fromarray(arr))
    assert grid == (1, 16, 22) and patches.shape == (352, 1176)        # 200x300 -> 224x308 canvas = 16x22 patches
    assert V.n_tokens(grid) == 88
    # patch 0 = top-left 14x14 block: channel-major, both temporal copies equal, values normalized
    x = ((arr.astype(np.float32) / 255 - V.MEAN) / V.STD).transpose(2, 0, 1)
    p0 = patches[0].reshape(3, 2, 14, 14)
    np.testing.assert_allclose(p0[:, 0], x[:, :14, :14], atol=1e-6)
    np.testing.assert_allclose(p0[:, 1], p0[:, 0])
    # block-major: patch 1 is (row 0, col 1), patch 2 is (row 1, col 0), patch 4 is block (0, 1) = (row 0, col 2)
    np.testing.assert_allclose(patches[1].reshape(3, 2, 14, 14)[:, 0], x[:, :14, 14:28], atol=1e-6)
    np.testing.assert_allclose(patches[2].reshape(3, 2, 14, 14)[:, 0], x[:, 14:28, :14], atol=1e-6)
    np.testing.assert_allclose(patches[4].reshape(3, 2, 14, 14)[:, 0], x[:, :14, 28:42], atol=1e-6)
    # a big image is capped by max_tokens; a tiny one is upscaled to min_tokens
    _, g = V.preprocess(Image.fromarray(np.zeros((3000, 4000, 3), np.uint8)), max_tokens=1024)
    assert V.n_tokens(g) <= 1024 and V.n_tokens(g) > 900
    _, g = V.preprocess(Image.fromarray(np.zeros((20, 20, 3), np.uint8)))
    assert V.n_tokens(g) >= 16


def test_smart_resize_matches_hf():
    try:
        from transformers.models.glm5_next.image_processing_glm5_next import smart_resize
    except Exception:  # noqa: BLE001
        pytest.skip("HF image processor needs torchvision")
    for h, w in [(200, 300), (1080, 1920), (20, 20), (4000, 3000), (28, 5000)]:
        assert V.smart_resize(h, w) == smart_resize(2, h, w), (h, w)
        assert V.smart_resize(h, w, max_tokens=1024) == smart_resize(2, h, w, max_pixels=1024), (h, w)


def test_sharded_forward_matches(monkeypatch):
    """The 8-way TP forward (heads / MLP columns sharded, q-chunked attention) == the plain forward, f32."""
    from jax.sharding import Mesh
    from glm53.engine import AXIS
    m, p = tiny(num_heads=8)                                   # heads must divide by the 8 devices (real: 16)
    mesh = Mesh(np.array(jax.devices()), (AXIS,))
    ps = V.to_sharded(p, mesh, dtype=jnp.float32)
    f = V.make_forward_sharded(ps, mesh, dtype=jnp.float32, q_chunk=8)
    rng = np.random.default_rng(4)
    grids = [(1, 4, 6), (1, 2, 2)]
    N = sum(t * h * w for t, h, w in grids)
    patches = rng.standard_normal((N, 3 * 2 * 14 * 14)).astype(np.float32)
    ref = np.asarray(V.forward(p, patches, grids))
    out = np.asarray(f(patches, grids))
    np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)
    out2 = np.asarray(f(patches, grids))                     # cached program
    np.testing.assert_array_equal(out2, out)
    grids2 = [(1, 2, 8), (1, 6, 2)]                          # same patch count, other grids: same program, right answer
    ref2 = np.asarray(V.forward(p, patches, grids2))
    np.testing.assert_allclose(np.asarray(f(patches, grids2)), ref2, rtol=2e-4, atol=2e-4)
