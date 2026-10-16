from typing import Protocol

from ._pyvai import (
    init_vai_engine,
    get_vai_device,
    Device,
    Dtype,
    Tensor,
    Node,
    NodeSlot,
    NodeGroup,
    InputNode,
    OutputNode,
    NodeFlow,
    Epilogue,
    KVCacheCounter,
    NeuralNet,
    Tokenizer,
)
from ._pyvai import SafetensorsParser, SafetensorRef
from . import nodes
from . import models


class _WeightStore(Protocol):
    """Duck-typed interface returned by `parse_weight`. Concrete parsers
    (e.g. `_pyvai.SafetensorsParser`) satisfy this without subclassing."""
    def tensor_names(self) -> list[str]: ...
    def __getitem__(self, key: str) -> Tensor: ...


def parse_weight(path: str, format: str = "safetensors") -> _WeightStore:
    """Parse a weights file into an inspectable store.

    The returned store (e.g. a `SafetensorsParser`) exposes `tensor_names()`
    and `__getitem__(key) -> Tensor` for inspecting a file's tensors. To load
    weights into a model, pass the path straight to `model.load_weight(path)`.
    Add formats by extending the dispatch below.
    """
    if format == "safetensors":
        return SafetensorsParser(path)
    raise ValueError(f"parse_weight: unsupported format '{format}'")


# Single source of the package version — pyproject.toml reads it via regex.
__version__: str = "0.1.0"
__all__ = [
    "init_vai_engine",
    "get_vai_device",
    "Device",
    "Dtype",
    "Tensor",
    "Node",
    "NodeSlot",
    "NodeGroup",
    "InputNode",
    "OutputNode",
    "NodeFlow",
    "Epilogue",
    "KVCacheCounter",
    "NeuralNet",
    "Tokenizer",
    "SafetensorsParser",
    "SafetensorRef",
    "nodes",
    "models",
    "parse_weight",
]
