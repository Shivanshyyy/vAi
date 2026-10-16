"""YOLOv8 — high-level wrapper over the C++-bound model + pre/post-processing.

The detection network and weight loader live in `models/yolov8/` (C++); the
letterbox preprocess and NMS postprocess are demo-side code
(`examples/yolov8/processing.*`) reaching Python through the bindings. This
module wires them into a `detect(image) -> list[Detection]` pipeline — the
orchestration the C++ demo (`examples/yolov8/demo.cpp`) does in main().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .._pyvai import (
    YOLOv8Model, YOLOv8Tier, Detection, Tensor, Device, Dtype,
    yolov8_preprocess, yolov8_postprocess,
)


__all__ = ["YOLOv8", "YOLOv8Model", "YOLOv8Tier", "Detection"]


class YOLOv8:
    """YOLOv8: bound C++ `YOLOv8Model` + letterbox preprocess + NMS postprocess."""

    def __init__(self, tier: YOLOv8Tier = YOLOv8Tier.l, image_size: int = 640,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 w_default_dtype: Dtype = Dtype.F16,
                 o_default_dtype: Dtype = Dtype.F32,
                 device: Device | None = None):
        if o_default_dtype != Dtype.F32:
            raise ValueError(
                "YOLOv8: detect() feeds float32 tensors, so o_default_dtype "
                "must be Dtype.F32 (use YOLOv8Model directly for other hints)")
        self.image_size     = image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self._weights_loaded = False
        self.model = YOLOv8Model(tier, image_size, conf_threshold, iou_threshold,
                                  w_default_dtype=w_default_dtype,
                                  o_default_dtype=o_default_dtype, device=device)

    def load_weights(self, path: str | Path | None = None) -> "YOLOv8":
        """Load a safetensors checkpoint; None reads the canonical data/ path."""
        self.model.load_weight(None if path is None else str(path))
        self._weights_loaded = True
        return self

    def set_profile_tags(self) -> "YOLOv8":
        """Turn on per-node GPU profiling (tags may be set at any point)."""
        self.model.set_profile_tags()
        return self

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on an HWC uint8 RGB image array."""
        if not self._weights_loaded:
            raise RuntimeError("YOLOv8: call load_weights(...) before detect()")
        h, w = image.shape[:2]
        flat = np.ascontiguousarray(image, dtype=np.uint8).reshape(-1)

        # Letterbox + normalize (C++) -> CHW float ndarray (zero-copy return),
        # then run the network.
        chw = yolov8_preprocess(flat, w, h, self.image_size) \
            .reshape(3, self.image_size, self.image_size)
        out = self.model({"in0": Tensor.from_numpy(chw)})["out0"].to_numpy().reshape(-1)

        # Confidence filter + NMS (C++).
        return list(yolov8_postprocess(out, self.conf_threshold, self.iou_threshold, self.image_size))

    def print_profile(self) -> None:
        self.model.print_profile()
