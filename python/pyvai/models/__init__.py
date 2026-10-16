"""pyvai built-in models.

Models come in two flavors; pick by how much orchestration you need.

**Task wrappers** (`MNIST`, `GPT2`, `Llama3`, `YOLOv8`) — high-level, end-to-end.
Each wraps a bound model with its pre/post-processing and returns a task result:
digit (MNIST), text (LLMs), detections (YOLO).

    gpt = GPT2()
    gpt.load_weights("gpt2.safetensors")
    gpt.load_tokenizer("tokenizer.json")
    print(gpt.generate("Hello", max_new_tokens=50))

**Bound models** (`MNISTModel`, `GPT2Model`, `Llama3Model`, `YOLOv8Model`) —
the C++ sample `Model` classes as-is. Each is a `NeuralNet`, called with dict-in/
dict-out `__call__`; you drive weights and pre/post-processing yourself.

    model = MNISTModel()
    model.load_weight("mnist.safetensors")
    logits = model({"in0": pyvai.Tensor.from_numpy(chw)})["out0"].to_numpy()
"""

# Task wrappers — bound model + pre/post-processing, returns a task result
from .mnist import MNIST
from .gpt2 import GPT2
from .llama3 import Llama3
from .yolov8 import YOLOv8

# Bound models — C++ sample `Model` classes as-is (NeuralNet subclasses)
from .mnist import MNISTModel
from .gpt2 import GPT2Model
from .llama3 import Llama3Model
from .yolov8 import YOLOv8Model

# Supporting types — configs / enums / detection record. GPT-2 and Llama 3 no
# longer carry a tokenizer apiece; both use the engine-wide `pyvai.Tokenizer`.
from .gpt2 import GPT2Config
from .llama3 import Llama3Config
from .yolov8 import YOLOv8Tier, Detection

# YOLO pre/post-processing as free functions, for driving YOLOv8Model directly.
from .._pyvai import yolov8_preprocess, yolov8_postprocess


__all__ = [
    # Task wrappers
    "MNIST",
    "GPT2",
    "Llama3",
    "YOLOv8",
    # Bound models
    "MNISTModel",
    "GPT2Model",
    "Llama3Model",
    "YOLOv8Model",
    # Supporting types
    "GPT2Config",
    "Llama3Config",
    "YOLOv8Tier",
    "Detection",
    # YOLO pre/post-processing
    "yolov8_preprocess",
    "yolov8_postprocess",
]
