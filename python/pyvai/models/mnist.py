"""MNIST — high-level wrapper over the C++-bound classifier.

Upstream folded MNIST into a self-contained demo (`examples/mnist/main.cpp`),
so the model graph and weight loader are copied into the bindings
(`src/bindings/models/utils/mnist.h`), bound as `MNISTModel`.
This module wraps it into a `predict(image) -> int` classifier (normalize +
forward + argmax), mirroring the other built-in model wrappers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .._pyvai import MNISTModel, Tensor, Device, Dtype


__all__ = ["MNIST", "MNISTModel"]


class MNIST:
    """MNIST digit classifier: bound C++ `MNISTModel` + normalize/argmax."""

    def __init__(self, w_default_dtype: Dtype = Dtype.F16,
                 o_default_dtype: Dtype = Dtype.F32,
                 device: Device | None = None):
        self._weights_loaded = False
        self.model = MNISTModel(w_default_dtype=w_default_dtype,
                                 o_default_dtype=o_default_dtype, device=device)

    def load_weights(self, path: str | Path | None = None) -> "MNIST":
        """Load a safetensors checkpoint; None reads the canonical data/ path."""
        self.model.load_weight(None if path is None else str(path))
        self._weights_loaded = True
        return self

    def set_profile_tags(self) -> "MNIST":
        """Turn on per-node GPU profiling (tags may be set at any point)."""
        self.model.set_profile_tags()
        return self

    def logits(self, image: np.ndarray) -> np.ndarray:
        """Class logits for a 28x28 grayscale (H, W) uint8 image array."""
        if not self._weights_loaded:
            raise RuntimeError("MNIST: call load_weights(...) before inference")
        if image.shape != (28, 28):
            raise ValueError(f"MNIST expects a (28, 28) image, got {image.shape}")
        # Normalize to [0, 1] and add the channel axis -> (1, 28, 28) f32 CHW.
        chw = (np.ascontiguousarray(image, dtype=np.float32) / 255.0)[np.newaxis, :, :]
        return self.model({"in0": Tensor.from_numpy(chw, dtype=Dtype.F32)})["out0"].to_numpy()

    def predict(self, image: np.ndarray) -> int:
        """Predicted digit (argmax over logits) for a grayscale (H, W) uint8 image."""
        return int(np.argmax(self.logits(image)))

    def print_profile(self) -> None:
        self.model.print_profile()
