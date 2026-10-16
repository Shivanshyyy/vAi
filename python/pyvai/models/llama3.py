"""Llama 3 — high-level wrapper over the C++-bound model + tokenizer.

The transformer graph (GQA + RoPE) lives in `models/llama3/model.h` (C++),
the weight loader in `models/llama3/meta-weights.h`, and the tokenizer is the
engine-wide `Tokenizer` (HuggingFace `tokenizer.json`). This module wires
them into a `generate()` loop — the orchestration the C++ demo
(`examples/llama3/demo.cpp`) does in main(). The model is KV-cached: prefill
once with the full prompt, then feed one token per step. Sampling runs
on-device (top-k), so each forward returns the next token id directly.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .._pyvai import Llama3Model, Llama3Config, Tokenizer, Tensor, Device, Dtype


__all__ = ["Llama3", "Llama3Model", "Llama3Config"]


class Llama3:
    """Llama 3: bound C++ `Llama3Model` + `Tokenizer` with a generate loop.

    `config` picks the checkpoint (default: `Llama3Config._3_2_1B()`).
    """

    def __init__(self, config: Llama3Config | None = None,
                 w_default_dtype: Dtype = Dtype.F16,
                 o_default_dtype: Dtype = Dtype.F32,
                 device: Device | None = None,
                 max_runtime_context: int = 0):
        self.config = config if config is not None else Llama3Config._3_2_1B()
        self.model = Llama3Model(self.config, w_default_dtype=w_default_dtype, o_default_dtype=o_default_dtype,
                                  device=device,
                                  max_runtime_context=max_runtime_context)
        self.tokenizer: Tokenizer | None = None
        self._weights_loaded = False

    def load_weights(self, path: str | Path | None = None) -> "Llama3":
        """Load a safetensors checkpoint; None reads the canonical data/ path."""
        self.model.load_weight(None if path is None else str(path))
        self._weights_loaded = True
        return self

    def load_tokenizer(self, tokenizer_json: str | Path) -> "Llama3":
        """Load the checkpoint's `tokenizer.json` (one file — no vocab/merges pair)."""
        self.tokenizer = Tokenizer(str(tokenizer_json))
        return self

    def set_profile_tags(self) -> "Llama3":
        """Turn on per-node GPU profiling (tags may be set at any point)."""
        self.model.set_profile_tags()
        return self

    @staticmethod
    def _pack(toks: list[int]) -> "Tensor":
        # Raw int32 token ids into a 4-byte slot (the embedding shader reads them
        # as uint32) — no float conversion. See Tensor.from_data / setData<T>.
        return Tensor.from_data(np.asarray(toks, dtype=np.int32))

    @staticmethod
    def _next_token(out: np.ndarray) -> int:
        # The model samples on-device (top-k); the output holds one raw uint32 id.
        return int(out.reshape(-1)[0])

    def generate(self, prompt: str, max_new_tokens: int = 50,
                 stop_ids: Iterable[int] | None = None) -> str:
        """Continue `prompt`, stopping after `max_new_tokens` or at a `stop_ids` token.

        Which ids end a turn is declared by the checkpoint's
        generation_config.json, not by the tokenizer, so resolve them yourself —
        e.g. `stop_ids=[llama.tokenizer.piece_to_id("<|eot_id|>")]`.
        """
        if self.tokenizer is None:
            raise RuntimeError("Llama3.generate: call load_tokenizer(...) first")
        if not self._weights_loaded:
            raise RuntimeError("Llama3.generate: call load_weights(...) first")

        stop = set(stop_ids or ())
        tokens = list(self.tokenizer.encode(prompt, add_special=True))
        if max_new_tokens < 1:
            return self.tokenizer.decode(tokens)
        self.model.reset_cache()
        self.model.invalidate_command_buffer()

        # Prefill the full prompt → first sampled token (KV counter auto-updates
        # inside Model.__call__).
        out = self.model({"inTokenIds": self._pack(tokens)})["nextTokenId"]
        tokens.append(self._next_token(out.get_data("uint32")))
        self.model.invalidate_command_buffer()   # prefill → decode boundary

        # Decode: feed only the last token, one new token per step.
        for _ in range(max_new_tokens - 1):
            if tokens[-1] in stop:
                break
            out = self.model({"inTokenIds": self._pack([tokens[-1]])})["nextTokenId"].get_data("uint32")
            tokens.append(self._next_token(out))

        return self.tokenizer.decode(tokens)

    def print_profile(self) -> None:
        self.model.print_profile()
