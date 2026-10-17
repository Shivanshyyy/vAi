"""YOLOX object detection with pyvai's built-in models.

python python/examples/yolox/demo.py \\
    --weights_dir  data/yolox/ckpts/large \\
    --image_path   data/yolox/dog.jpg \\
    --tier l
"""
from __future__ import annotations

import os
import argparse

import pyvai
from pyvai.models import YOLOX, YOLOXTier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOX detection with pyvai (built-in models).")
    parser.add_argument("--weights_dir", type=str, required=True,
                        help="Checkpoint folder (data/yolox/ckpts/<small|medium|large|xlarge>).")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to an input image (any size; JPEG, PNG, ...).")
    parser.add_argument("--tier", choices=list("smlx"), default="l",
                        help="Model size s/m/l/x.")
    parser.add_argument("--image_size", type=int, default=640,
                        help="Network input size.")
    parser.add_argument("--conf_threshold", type=float, default=0.25,
                        help="Confidence threshold.")
    parser.add_argument("--iou_threshold", type=float, default=0.45,
                        help="NMS IoU threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.weights_dir):
        raise FileNotFoundError(f"weights not found: {args.weights_dir}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"image not found: {args.image_path}")

    # w_default_dtype=Dtype.F32 runs the weights in FP32; tagging turns profiling on.
    model = YOLOX(
        tier=YOLOXTier.from_letter(args.tier),
        image_size=args.image_size,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        w_default_dtype=pyvai.Dtype.F16,
    )
    model.set_profile_tags()
    model.load_weights(args.weights_dir)

    print(f"Image: {args.image_path}")
    detections = model.detect(args.image_path)   # decoded and letterboxed in C++

    print(f"\n=== Detections ({len(detections)}) ===")
    for i, d in enumerate(detections):
        print(f"[{i}] {d.class_name} ({d.confidence * 100:.1f}%): "
              f"[{d.x1:.0f}, {d.y1:.0f}, {d.x2:.0f}, {d.y2:.0f}]")

    model.print_profile()
    return 0


if __name__ == "__main__":
    main()
