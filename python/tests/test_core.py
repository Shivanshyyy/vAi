"""Engine, Tensor round-trips, the readback guard, and safetensors loading."""
import gc
import json
import struct

import numpy as np
import pytest

import pyvai
from pyvai import nodes


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def test_init_idempotent():
    """Multiple calls must succeed (conftest already did one for autouse init)."""
    pyvai.init_vai_engine()
    pyvai.init_vai_engine()


# ---------------------------------------------------------------------------
# Tensor — numpy round-trips (converting and raw paths)
# ---------------------------------------------------------------------------

def test_from_numpy_roundtrip(rng):
    arr = rng.standard_normal((28, 28, 1)).astype(np.float32)
    t = pyvai.Tensor.from_numpy(arr)
    assert list(t.shape) == [28, 28, 1]
    np.testing.assert_array_equal(arr, t.to_numpy())


def test_permute_matches_numpy(rng):
    arr = rng.standard_normal((28, 28, 1)).astype(np.float32)
    t = pyvai.Tensor.from_numpy(arr).permute(2, 0, 1)
    assert list(t.shape) == [1, 28, 28]
    np.testing.assert_array_equal(np.transpose(arr, (2, 0, 1)), t.to_numpy())


def test_reshape_preserves_data():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = pyvai.Tensor.from_numpy(arr).reshape([4, 3])
    assert list(t.shape) == [4, 3]
    np.testing.assert_array_equal(arr.reshape(4, 3), t.to_numpy())


def test_from_data_get_data_raw_roundtrip():
    """Raw typed I/O: int32 ids survive a 4-byte slot without float conversion."""
    ids = np.array([0, 1, 50256, 12345], dtype=np.int32)
    t = pyvai.Tensor.from_data(ids)
    assert t.dtype == pyvai.Dtype.F32          # F32-sized slot, raw int32 bits
    np.testing.assert_array_equal(t.get_data("int32"), ids)
    np.testing.assert_array_equal(t.get_data("uint32"), ids.astype(np.uint32))


def test_from_data_itemsize_mismatch_raises():
    # int16 (2 bytes) into a default F32 (4-byte) slot must be rejected.
    with pytest.raises(ValueError, match="itemsize"):
        pyvai.Tensor.from_data(np.array([1, 2], dtype=np.int16))


# ---------------------------------------------------------------------------
# Readback guard — the two reads that used to abort now raise RuntimeError.
# Wrapper-level bookkeeping by NeuralNet.__call__; views of the same storage
# are unmarked and fall through to the core path (documented limit).
# ---------------------------------------------------------------------------

class _IdentityNet(pyvai.NeuralNet):
    def __init__(self):
        super().__init__()
        self.id = nodes.IdentityNode()
        self.input("in0") - self.id - self.output("out0")


def _x():
    return pyvai.Tensor.from_numpy(np.arange(4, dtype=np.float32))


def test_input_readable_after_forward():
    net = _IdentityNet()
    x = _x()
    net({"in0": x})
    np.testing.assert_array_equal(x.to_numpy(), np.arange(4, dtype=np.float32))


def test_stale_output_raises():
    net = _IdentityNet()
    out1 = net({"in0": _x()})["out0"]
    net({"in0": _x()})                          # clobbers out1's buffer
    with pytest.raises(RuntimeError, match="stale forward output"):
        out1.to_numpy()


def test_immediate_read_ok():
    net = _IdentityNet()
    out = net({"in0": _x()})["out0"]
    np.testing.assert_array_equal(out.to_numpy(), np.arange(4, dtype=np.float32))


def test_read_output_survives_next_forward():
    """An output read once is served from the synced cache afterwards."""
    net = _IdentityNet()
    out = net({"in0": _x()})["out0"]
    first = out.to_numpy()
    net({"in0": pyvai.Tensor.from_numpy(np.zeros(4, dtype=np.float32))})
    np.testing.assert_array_equal(out.to_numpy(), first)


# ---------------------------------------------------------------------------
# Safetensors
# ---------------------------------------------------------------------------

def _write_safetensors(path, tensors: dict[str, np.ndarray]) -> None:
    """Hand-rolled minimal safetensors writer — avoids depending on the
    `safetensors` package just to produce test fixtures."""
    tags = {np.dtype(np.float32): "F32", np.dtype(np.int64): "I64"}
    metadata: dict = {}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        size = arr.nbytes
        metadata[name] = {
            "dtype": tags[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + size],
        }
        blobs.append(arr.tobytes())
        offset += size
    header = json.dumps(metadata).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for b in blobs:
            f.write(b)


def test_load_safetensors_roundtrip(tmp_path, rng):
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((2,)).astype(np.float32)
    path = tmp_path / "w.safetensors"
    _write_safetensors(str(path), {"a": a, "b": b})

    store = pyvai.parse_weight(str(path))
    assert sorted(store.tensor_names()) == ["a", "b"]
    np.testing.assert_array_equal(store["a"].to_numpy(), a)
    np.testing.assert_array_equal(store["b"].to_numpy(), b)


def test_unsupported_checkpoint_dtype_raises(tmp_path):
    """Loading a non-float entry used to reach dtypeFromString's assert (debug
    abort / release fp32 mislabel); now it is a TypeError. Metadata access via
    ref() stays open."""
    path = tmp_path / "w.safetensors"
    _write_safetensors(str(path), {"idx": np.array([1, 2], dtype=np.int64)})

    store = pyvai.parse_weight(str(path))
    with pytest.raises(TypeError, match="I64"):
        _ = store["idx"]
    with pytest.raises(TypeError, match="I64"):
        _ = store.get("idx", pyvai.Dtype.F32)
    assert store.ref("idx").dtype == "I64"


def test_get_dtype_conversion(tmp_path, rng):
    """SafetensorsParser.get(key, dtype=...) converts at load time."""
    arr = rng.standard_normal((4, 4)).astype(np.float32)
    path = tmp_path / "w.safetensors"
    _write_safetensors(str(path), {"w": arr})

    store = pyvai.parse_weight(str(path))
    assert store.get("w").dtype == pyvai.Dtype.F32          # native
    t_fp16 = store.get("w", dtype=pyvai.Dtype.F16)
    assert t_fp16.dtype == pyvai.Dtype.F16
    np.testing.assert_allclose(t_fp16.to_numpy(), arr.astype(np.float16), atol=1e-3)


def test_getitem_missing_key_raises_key_error(tmp_path, rng):
    path = tmp_path / "w.safetensors"
    _write_safetensors(str(path), {"present": rng.standard_normal((2, 2)).astype(np.float32)})
    with pytest.raises(KeyError, match="missing"):
        _ = pyvai.parse_weight(str(path))["missing"]


def test_ref_keeps_parser_alive(tmp_path, rng):
    """The ref holds a raw parser pointer — keep_alive must pin the parser."""
    path = tmp_path / "w.safetensors"
    _write_safetensors(str(path), {"a": rng.standard_normal((5,)).astype(np.float32)})

    store = pyvai.parse_weight(str(path))
    ref = store.ref("a")
    del store
    gc.collect()
    assert list(ref.shape) == [5]       # parser still alive under the ref
