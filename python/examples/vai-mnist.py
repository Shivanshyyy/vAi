"""MNIST with the low-level vAi graph builder (node DSL) — not the built-in models.

This mirrors how the C++ model programs build a network: subclass `pyvai.NeuralNet`,
instantiate nodes from `pyvai.nodes`, and wire them with the `-` operator
(the same DSL the C++ models use).
Compare with `examples/mnist.py`, which instead uses the high-level `MNIST`
wrapper from `pyvai.models` (the architecture defined in C++).

Because the graph is built here, weight key names are up to you — this demo
uses the `conv1 / conv2 / fc` scheme of the committed MNIST checkpoint, so it
runs out of the box:

    python python/examples/vai-mnist.py \\
        --weights_path data/mnist/model.safetensors \\
        --image_path   data/mnist/9.png
"""
from __future__ import annotations

import os
import argparse

import numpy as np
from PIL import Image

import pyvai
from pyvai import nodes


# fp16 weights with fp32 I/O (the C++ demo examples/mnist/main.cpp runs
# fp32 weights; use Dtype.F32 to reproduce its reference logits exactly).
WEIGHT_DTYPE = pyvai.Dtype.F16
INOUT_DTYPE  = pyvai.Dtype.F32


class MnistNet(pyvai.NeuralNet):
    """Hand-built MNIST convnet using the node DSL (mirrors examples/mnist/main.cpp)."""

    def __init__(self, profile: bool = False):
        super().__init__(w_default_dtype=WEIGHT_DTYPE)
        self.conv0   = nodes.ConvolutionNode(
            in_channels=1, out_channels=32, kernel_size=3, stride=1,
            epilogue=pyvai.Epilogue.relu)
        self.pool0   = nodes.MaxPool2dNode(kernel_size=2, stride=2)
        self.conv1   = nodes.ConvolutionNode(
            in_channels=32, out_channels=64, kernel_size=3, stride=1,
            epilogue=pyvai.Epilogue.relu)
        self.pool1   = nodes.MaxPool2dNode(kernel_size=2, stride=2)
        self.reshape = nodes.ReshapeNode([7 * 7 * 64])
        self.linear  = nodes.LinearNode(in_dim=7 * 7 * 64, out_dim=10, has_bias=True)

        self._build_graph()

        # Tagging the nodes is what turns per-node GPU profiling on — there is
        # no separate switch.
        if profile:
            self.conv0.profile_tag  = "conv0"
            self.pool0.profile_tag  = "maxpool0"
            self.conv1.profile_tag  = "conv1"
            self.pool1.profile_tag  = "maxpool1"
            self.linear.profile_tag = "linear"

    def _build_graph(self) -> None:
        """`-` connects the left node's output slot to the right one's input
        slot and yields the right node, so the chain keeps going."""
        (self.input("in0")
            - self.conv0 - self.pool0
            - self.conv1 - self.pool1
            - self.reshape - self.linear
         - self.output("out0"))


def load_weights(net: MnistNet, weights) -> None:
    net.conv0["weight"]  = weights.get("conv1.weight", dtype=WEIGHT_DTYPE)
    net.conv0["bias"]    = weights.get("conv1.bias",   dtype=WEIGHT_DTYPE)
    net.conv1["weight"]  = weights.get("conv2.weight", dtype=WEIGHT_DTYPE)
    net.conv1["bias"]    = weights.get("conv2.bias",   dtype=WEIGHT_DTYPE)
    net.linear["weight"] = weights.get("fc.weight",    dtype=WEIGHT_DTYPE)
    net.linear["bias"]   = weights.get("fc.bias",      dtype=WEIGHT_DTYPE)


def load_image_chw(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")             # grayscale 28x28
    arr = np.array(img, dtype=np.float32) / 255.0   # (28, 28)
    return arr[np.newaxis, :, :]                    # (1, 28, 28) — C, H, W


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST via the low-level vAi node DSL.")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="MNIST safetensors (keys: conv1/conv2/fc).")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to a 28x28 grayscale input image.")
    parser.add_argument("--profile", action="store_true",
                        help="Enable per-node GPU profiling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"weights not found: {args.weights_path}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"image not found: {args.image_path}")

    pyvai.init_vai_engine()

    net = MnistNet(profile=args.profile)
    # Node DSL mirrors the C++ model programs, so load weights the same way: index a
    # SafetensorsParser directly (see loadWeights<"jroot"> in examples/mnist/main.cpp).
    load_weights(net, pyvai.SafetensorsParser(args.weights_path))

    img    = load_image_chw(args.image_path)
    logits = net({"in0": pyvai.Tensor.from_numpy(img, dtype=INOUT_DTYPE)})["out0"].to_numpy()

    print(f"input  : {img.shape} {img.dtype}")
    for i, v in enumerate(logits):
        print(f"  data[{i}] = {v:.6f}")
    print(f"argmax : {int(np.argmax(logits))}")

    if args.profile:
        net.print_profile()
    return 0


if __name__ == "__main__":
    main()
