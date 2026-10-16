"""MNIST digit classification using pyvai's built-in models.

python python/examples/mnist.py \\
    --weights_path data/mnist/model.safetensors \\
    --image_path   data/mnist/9.png
"""

import os
import argparse

import numpy as np
from PIL import Image

import pyvai
from pyvai.models import MNIST


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST classification with pyvai (built-in models).")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to MNIST safetensors weights (keys: conv1/conv2/fc).")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to a 28x28 grayscale input image.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"weights not found: {args.weights_path}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"image not found: {args.image_path}")

    pyvai.init_vai_engine()

    # Switch w_default_dtype to Dtype.F32 to run weights in FP32. Tagging the
    # nodes (set_profile_tags) is what turns per-node GPU profiling on.
    model = MNIST(w_default_dtype=pyvai.Dtype.F16)
    model.set_profile_tags()
    model.load_weights(args.weights_path)

    img  = np.asarray(Image.open(args.image_path).convert("L"), dtype=np.uint8)   # (28, 28)
    pred = model.predict(img)

    print(f"input      : {img.shape} {img.dtype}")
    print(f"prediction : {pred}")

    model.print_profile()
    return 0


if __name__ == "__main__":
    main()
