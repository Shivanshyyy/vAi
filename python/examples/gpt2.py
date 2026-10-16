"""GPT-2 text generation with pyvai's built-in models.

python python/examples/gpt2.py \\
    --weights_path   data/gpt2/124M/model.safetensors \\
    --tokenizer_path data/gpt2/tokenizer.json
"""
from __future__ import annotations

import os
import argparse

import pyvai
from pyvai.models import GPT2


# Same prompt the C++ demo runs (examples/gpt2/demo.cpp).
PROMPT = (
    "In recent decades, artificial intelligence has progressed from a niche "
    "academic curiosity to a force that quietly reshapes nearly every corner of "
    "modern life. Algorithms now translate languages in real time, diagnose diseases "
    "from medical images, draft emails on our behalf, and pilot vehicles through "
    "unpredictable city streets. What once required entire research teams can "
    "sometimes be done by a single model running on a laptop. Yet despite these "
    "remarkable advances, the deeper questions remain unresolved: how should we "
    "measure understanding, what responsibilities accompany increasingly autonomous "
    "systems, and what role will human judgment play in a world where machines "
    "write, decide, and predict at unprecedented speed and scale? The next chapter "
    "of this story is still being written, and the choices we make in the coming "
    "years may"
)


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-2 generation with pyvai (built-in models).")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to the HuggingFace GPT-2 model.safetensors.")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the HuggingFace tokenizer.json.")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Number of tokens to generate.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"weights not found: {args.weights_path}")
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"tokenizer not found: {args.tokenizer_path}")

    pyvai.init_vai_engine()

    # Switch w_default_dtype to Dtype.F32 to run weights in FP32. Tagging the
    # nodes (set_profile_tags) is what turns per-node GPU profiling on.
    gpt = GPT2(w_default_dtype=pyvai.Dtype.F16)
    gpt.set_profile_tags()
    gpt.load_weights(args.weights_path)
    gpt.load_tokenizer(args.tokenizer_path)

    print(f'Prompt: "{PROMPT}"')
    print(f"Generating {args.max_new_tokens} tokens...\n")
    text = gpt.generate(PROMPT, max_new_tokens=args.max_new_tokens)

    print("=== Generated ===")
    print(text)

    gpt.print_profile()
    return 0


if __name__ == "__main__":
    main()