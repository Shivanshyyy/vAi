# pyvai

![status](https://img.shields.io/badge/status-experimental-orange)
![python](https://img.shields.io/badge/python-3.9--3.14-blue)
![backend](https://img.shields.io/badge/backend-Vulkan%201.3-red)
![license](https://img.shields.io/badge/license-MIT-blue)

<!-- Once published, swap the static badges above for dynamic ones:
![PyPI](https://img.shields.io/pypi/v/pyvai)
![Python versions](https://img.shields.io/pypi/pyversions/pyvai)
![CI](https://github.com/<org>/<repo>/actions/workflows/ci.yml/badge.svg)
-->

## Overview

Python bindings for the **vAi** Vulkan inference engine, run the built-in models, or assemble your own graphs from the node DSL.

> [!WARNING]
> **`pyvai` is experimental.** The API can change between commits, there is no PyPI release, and the build-from-source install stays tied to the directory it was built in. See [Status / Limitations](#status--limitations).

## Installation

No PyPI release, build from source.

### Requirements

**System**:

- **Vulkan** 1.3 loader / SDK, and `glslc` to compile the shaders
- **Toolchain**: CMake ≥ 3.16, Ninja, a C++23 compiler

**Python**: 3.9 – 3.14. pip supplies the packages:

- **Build**: `scikit-build-core` + `nanobind`, fetched automatically. (Python 3.9 pins nanobind to 2.x; the marker in `pyproject.toml` does that for you.)
- **Runtime**: `numpy` and `Pillow` (the examples' image I/O), installed with the package.

### Install (editable)

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv --python 3.12
uv pip install -e .
```

Or with conda:

```bash
conda create -n pyvai python=3.12 && conda activate pyvai
pip install -e .
```

> [!NOTE]
> If CMake stops at `Could NOT find Python (missing: Development.Module)`, the venv's base interpreter lacks the Python development headers. Install them (e.g. `sudo apt install python3.12-dev`) or pass `--managed-python` to `uv venv`.

### Docker

If you would rather work in a container, build the image from `docker/Dockerfile`. See [`docker/README.md`](../docker/README.md).

## Built-in Models

`pyvai.models` exposes the C++ models as-is (MNIST's graph is a copy kept in the bindings — upstream folded it into `examples/mnist/main.cpp`). Each entry has a bound raw model (`*Model`, dict-in/dict-out) and a high-level wrapper around it:

| Model | Wrapper | Bound classes |
|---|---|---|
| MNIST | `MNIST` | `MNISTModel` |
| GPT-2 | `GPT2` | `GPT2Model`, `GPT2Config` |
| Llama 3 | `Llama3` | `Llama3Model`, `Llama3Config` |
| YOLOv8 | `YOLOv8` | `YOLOv8Model`, `YOLOv8Tier`, `Detection` |

GPT-2 and Llama 3 size themselves from config presets named after the C++ factories (`GPT2Config._124M()`, `Llama3Config._3_2_1B()`, ...), and both read a HuggingFace `tokenizer.json` through the shared `pyvai.Tokenizer`.

## Run Examples

Each model below pairs its Python API with the matching runnable script in
[`python/examples/`](examples/).

### MNIST

```python
import numpy as np
import pyvai
from PIL import Image
from pyvai.models import MNIST

pyvai.init_vai_engine()
mnist = MNIST().load_weights("data/mnist/model.safetensors")

img = np.asarray(Image.open("digit.png").convert("L"), dtype=np.uint8)  # (28, 28)
print("digit:", mnist.predict(img))
```

```bash
python python/examples/mnist.py --weights_path data/mnist/model.safetensors --image_path digit.png
```

### GPT-2

```python
import pyvai
from pyvai.models import GPT2

pyvai.init_vai_engine()
gpt = GPT2()                                     # config=GPT2Config._124M() by default
gpt.load_weights("data/gpt2/124M/model.safetensors")
gpt.load_tokenizer("data/gpt2/tokenizer.json")   # HuggingFace tokenizer.json
text = gpt.generate("The future of artificial intelligence is", max_new_tokens=50)
print(text)
```

```bash
python python/examples/gpt2.py --weights_path data/gpt2/124M/model.safetensors --tokenizer_path data/gpt2/tokenizer.json
```

### Llama 3

```python
import pyvai
from pyvai.models import Llama3

pyvai.init_vai_engine()
llama = Llama3(max_runtime_context=4096)      # KV-cache budget; 0 = trained context (128K)
llama.load_weights("data/llama3/3.2-1B/model.safetensors")
llama.load_tokenizer("data/llama3/tokenizer.json")
text = llama.generate("The capital of France is",
                      stop_ids=[llama.tokenizer.piece_to_id("<|eot_id|>")])
print(text)
```

```bash
# One-shot completion
python python/examples/llama3.py --weights_path data/llama3/3.2-1B/model.safetensors --tokenizer_path data/llama3/tokenizer.json

# Multi-turn chat
python python/examples/llama3.py --weights_path data/llama3/3.2-1B/model.safetensors --tokenizer_path data/llama3/tokenizer.json --chat
```

### YOLOv8

```python
import numpy as np
import pyvai
from PIL import Image
from pyvai.models import YOLOv8, YOLOv8Tier

pyvai.init_vai_engine()
yolov8 = YOLOv8(tier=YOLOv8Tier.l)
yolov8.load_weights("data/yolov8/l/yolov8.safetensors")

img = np.asarray(Image.open("street.jpg").convert("RGB"), dtype=np.uint8)
for d in yolov8.detect(img):
    print(f"{d.class_name} {d.confidence*100:.1f}%  [{d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}]")
```

```bash
python python/examples/yolov8.py --weights_path data/yolov8/l/yolov8.safetensors --image_path street.jpg --tier l
```

Letterbox preprocessing and NMS postprocessing run in C++; only image decoding is done in Python (Pillow). Both are also exported as free functions (`pyvai.models.yolov8_preprocess` / `yolov8_postprocess`) for driving `YOLOv8Model` directly.

## Building Graphs

The graph builder is exposed as well: subclass `pyvai.NeuralNet`, instantiate nodes from `pyvai.nodes`, and wire them with the `-` operator (a mirror of vAi's C++ DSL) or with calls. You define the architecture and pick the weight key names yourself, where a built-in model fixes both in C++.

[`examples/vai-mnist.py`](examples/vai-mnist.py) is the worked example — the same MNIST inference as above, assembled from nodes instead.

## Status / Limitations

`pyvai` is **experimental**; expect breaking changes between commits.

- **Build-from-source only**: install with `pip install -e .`; the result is not relocatable (the engine loads its compiled shaders from the build directory) Distributable wheels are not available in this experimental release.
- **Vulkan only**: no CPU fallback; the device is chosen at engine init.
- **Linux-tested only**: the bindings build and run on Windows in principle, but nothing here has been verified on it yet.
- **Weights are user-provided** (MNIST excepted): pyvai ships model code, not weights, and expects the C++ models' key naming (e.g. MNIST `conv1/conv2/fc`; GPT-2/Llama 3 use the upstream HuggingFace checkpoints).

## License

[MIT](../LICENSE), same as the parent vAi repository.
