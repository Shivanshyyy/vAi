"""Built-in models: config presets (no GPU), MNIST end-to-end (weights ship
in-repo), and the YOLO pre/post-processing functions."""
import numpy as np
import pytest

import pyvai
from pyvai.models import (
    GPT2Config, Llama3Config, MNIST, MNISTModel,
    yolov8_preprocess, yolov8_postprocess,
)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def test_gpt2_config_124m():
    c = GPT2Config._124M()
    assert (c.vocab_size, c.max_trained_context, c.embed_dim, c.num_heads, c.num_layers) \
        == (50257, 1024, 768, 12, 12)
    assert c.name == "124M"


def test_llama3_config_3_2_1b():
    c = Llama3Config._3_2_1B()
    assert (c.vocab_size, c.embed_dim, c.num_heads, c.num_kv_heads,
            c.num_layers, c.ffn_hidden_dim) == (128256, 2048, 32, 8, 16, 8192)
    assert c.name == "3.2-1B"


# ---------------------------------------------------------------------------
# MNIST — the one model whose checkpoint ships in-repo
# ---------------------------------------------------------------------------

def _weights(data_dir):
    p = data_dir / "mnist" / "model.safetensors"
    if not p.exists():
        pytest.skip(f"missing weights: {p}")
    return str(p)


def _digit_image(data_dir, digit):
    Image = pytest.importorskip("PIL.Image")
    p = data_dir / "mnist" / f"{digit}.png"
    if not p.exists():
        pytest.skip(f"missing image: {p}")
    return np.asarray(Image.open(p).convert("L"), dtype=np.uint8)   # (28, 28)


@pytest.fixture(scope="module")
def mnist(data_dir):
    return MNIST().load_weights(_weights(data_dir))


@pytest.mark.parametrize("digit", [0, 9])
def test_wrapper_predicts_digit(mnist, data_dir, digit):
    assert mnist.predict(_digit_image(data_dir, digit)) == digit


def test_bound_matches_wrapper(mnist, data_dir):
    """MNISTModel (raw dict-in/out) and the MNIST wrapper agree."""
    img = _digit_image(data_dir, 9)

    bound = MNISTModel()
    bound.load_weight(_weights(data_dir))
    chw = (img.astype(np.float32) / 255.0)[np.newaxis, :, :]
    logits = bound({"in0": pyvai.Tensor.from_numpy(chw, dtype=pyvai.Dtype.F32)})["out0"].to_numpy()

    assert int(np.argmax(logits)) == mnist.predict(img) == 9


def test_load_weight_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        MNISTModel().load_weight("does-not-exist.safetensors")


# ---------------------------------------------------------------------------
# YOLO pre/post-processing (pure functions — no GPU or weights)
# ---------------------------------------------------------------------------

def test_preprocess_letterbox_shape_and_range():
    w = h = 4
    target = 32
    image = np.full(w * h * 3, 128, dtype=np.uint8)  # flat HWC RGB
    out = np.asarray(yolov8_preprocess(image, w, h, target), dtype=np.float32)
    assert out.shape == (3 * target * target,)      # CHW float
    assert out.min() >= 0.0 and out.max() <= 1.0    # normalized to [0, 1]


def test_postprocess_argmax_and_class_mapping():
    # [x1, y1, x2, y2, cls0..cls79] for one anchor; class 2 (car) is the argmax.
    data = np.zeros(84, dtype=np.float32)
    data[:4] = [10.0, 20.0, 30.0, 40.0]
    data[4 + 0], data[4 + 2] = 0.3, 0.8
    dets = yolov8_postprocess(data, 0.25, 0.45)
    assert len(dets) == 1
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == (10.0, 20.0, 30.0, 40.0)
    assert d.class_id == 2 and d.class_name == "car"
    assert abs(d.confidence - 0.8) < 1e-6


def test_inference_without_weights_raises(data_dir):
    """A forward before load_weights() dies in the driver, so the wrapper
    refuses it (the bound *Model classes stay unguarded)."""
    with pytest.raises(RuntimeError, match="load_weights"):
        MNIST().predict(np.zeros((28, 28), dtype=np.uint8))
