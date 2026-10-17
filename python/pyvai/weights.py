"""Weight files — one entry point over the engine's checkpoint parsers.

`Weights` opens one checkpoint file and hands out its tensors. The format
comes from the file's suffix or `format=`; safetensors is the only one, and
another plugs in through `_PARSERS`. A built-in model takes its checkpoint
folder through `model.load_weights(dir)` instead: `Weights` is for inspecting
a file or feeding a hand-built graph (see examples/mnist/main.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ._pyvai import Dtype, SafetensorsParser, Tensor


__all__ = ["Weights", "TensorInfo"]


@dataclass(frozen=True)
class TensorInfo:
    """Header metadata of one tensor in a weight file. `dtype` is the file's
    own tag ("F32", "F16", "BF16", ...), not a `pyvai.Dtype`: the engine may
    hold the tensor as another dtype (BF16 loads as F32)."""
    key: str
    shape: tuple[int, ...]
    dtype: str


# format -> bound parser class; each exposes tensor_names / __getitem__ / get / ref.
_PARSERS = {
    "safetensors": SafetensorsParser,
}

_SUFFIXES = {
    ".safetensors": "safetensors",
}


class Weights:
    """A weight file opened for reading: each `w[key]` / `w.get(key)` reads
    that tensor from the file on demand."""

    def __init__(self, path: str | Path, format: str | None = None):
        path = Path(path)
        if format is None:
            format = _SUFFIXES.get(path.suffix.lower())
            if format is None:
                raise ValueError(
                    f"Weights: cannot tell the format of '{path}' from its suffix "
                    f"(known: {', '.join(_SUFFIXES)}); pass format=")
        parser = _PARSERS.get(format)
        if parser is None:
            raise ValueError(
                f"Weights: unsupported format '{format}' "
                f"(known: {', '.join(_PARSERS)})")
        self._path = path
        self._format = format
        self._parser = parser(str(path))   # raises FileNotFoundError

    @property
    def path(self) -> Path:
        return self._path

    @property
    def format(self) -> str:
        return self._format

    def keys(self) -> list[str]:
        """Names of every tensor in the file, in the file's order."""
        return list(self._parser.tensor_names())

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._parser.tensor_names())

    def __contains__(self, key: object) -> bool:
        return key in self._parser.tensor_names()

    def __getitem__(self, key: str) -> Tensor:
        """Load `key` as a Tensor in the file's native dtype. KeyError for an
        unknown key; TypeError for a dtype the engine cannot hold."""
        return self._parser[key]

    def get(self, key: str, dtype: Dtype | None = None) -> Tensor:
        """Load `key`, converting to `dtype` at load time (None: the file's
        native dtype)."""
        return self._parser.get(key, dtype)

    def info(self, key: str) -> TensorInfo:
        """Shape and file dtype of `key` from the header, without reading the
        tensor data. KeyError for an unknown key."""
        ref = self._parser.ref(key)
        return TensorInfo(key=key, shape=ref.shape, dtype=ref.dtype)

    def __repr__(self) -> str:
        return f"Weights('{self._path}', format='{self._format}', num_tensors={len(self)})"
