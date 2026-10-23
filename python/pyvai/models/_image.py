"""What the wrappers take as an image: a file path or decoded pixels."""
from __future__ import annotations

import os

import numpy as np

from .._pyvai import ImageProcessor


def image_processor(image, channels: int = 3) -> ImageProcessor:
    """An `ImageProcessor` over `image`: a path (str / os.PathLike), or an HWC
    uint8 array of `channels` channels ([H, W] also reads as one channel;
    anything np.asarray reads as such an array)."""
    if isinstance(image, (str, os.PathLike)):
        return ImageProcessor(image, channels)
    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    got = arr.shape[2] if arr.ndim == 3 else 1 if arr.ndim == 2 else None
    if got != channels:
        want = "[H, W]" if channels == 1 else f"[H, W, {channels}]"
        raise ValueError(
            f"image_processor: expected an image path or an {want} uint8 array, "
            f"got shape {arr.shape}")
    return ImageProcessor(arr)
