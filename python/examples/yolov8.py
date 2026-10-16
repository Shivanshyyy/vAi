"""YOLOv8 object detection with pyvai's built-in models.

python python/examples/yolov8.py \\
    --weights_path data/yolov8/l/yolov8.safetensors \\
    --image_path   data/images/000000133000.jpg \\
    --tier l
"""
from __future__ import annotations

import os
import argparse

import numpy as np
from PIL import Image

import pyvai
from pyvai.models import YOLOv8, YOLOv8Tier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 detection with pyvai (built-in models).")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to <tier>/yolov8.safetensors.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to an input image (any size, RGB).")
    parser.add_argument("--tier", choices=list("nsmlx"), default="l",
                        help="Model size n/s/m/l/x.")
    parser.add_argument("--image_size", type=int, default=640,
                        help="Network input size.")
    parser.add_argument("--conf_threshold", type=float, default=0.25,
                        help="Confidence threshold.")
    parser.add_argument("--iou_threshold", type=float, default=0.45,
                        help="NMS IoU threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"weights not found: {args.weights_path}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"image not found: {args.image_path}")

    pyvai.init_vai_engine()

    # Switch w_default_dtype to Dtype.F32 to run weights in FP32. Tagging the
    # nodes (set_profile_tags) is what turns per-node GPU profiling on.
    model = YOLOv8(
        tier=getattr(YOLOv8Tier, args.tier),
        image_size=args.image_size,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        w_default_dtype=pyvai.Dtype.F16,
    )
    model.set_profile_tags()
    model.load_weights(args.weights_path)

    img = np.asarray(Image.open(args.image_path).convert("RGB"), dtype=np.uint8)
    print(f"Image: {img.shape[1]}x{img.shape[0]}")
    detections = model.detect(img)

    print(f"\n=== Detections ({len(detections)}) ===")
    for i, d in enumerate(detections):
        print(f"[{i}] {d.class_name} ({d.confidence * 100:.1f}%): "
              f"[{d.x1:.0f}, {d.y1:.0f}, {d.x2:.0f}, {d.y2:.0f}]")

    model.print_profile()
    return 0


if __name__ == "__main__":
    main()
