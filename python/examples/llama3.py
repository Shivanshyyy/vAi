"""Llama 3.2 1B text generation with pyvai's built-in models.

python python/examples/llama3.py \\
    --weights_path   data/llama3/3.2-1B/model.safetensors \\
    --tokenizer_path data/llama3/tokenizer.json

Add --chat for a multi-turn session (mirrors examples/llama3/chat.cpp) instead
of the one-shot completion.
"""
from __future__ import annotations

import os
import argparse

import numpy as np

import pyvai
from pyvai.models import Llama3


# Same prompt the C++ demo runs (examples/llama3/demo.cpp), which keeps only
# the opening of the GPT-2 one.
PROMPT = (
    "In recent decades, artificial intelligence has progressed from a niche "
    "academic curiosity to a force that quietly reshapes nearly every corner of "
    "modern life. "
)

# --chat. The template is declared by "chat_template" in the checkpoint's
# tokenizer_config.json as a Jinja2 script; vAi does not run it, so the pieces
# are spelled out, and the end-of-turn ids by generation_config.json's
# "eos_token_id", which vAi does not read either.
BEGIN_OF_TEXT    = "<|begin_of_text|>"
USER_HEADER      = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
END_OF_TURN      = "<|eot_id|>"
END_OF_TURN_PIECES = ("<|end_of_text|>", "<|eom_id|>", "<|eot_id|>")


def parse_args():
    parser = argparse.ArgumentParser(description="Llama 3.2 1B generation with pyvai (built-in models).")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to the HuggingFace Llama 3 model.safetensors.")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the HuggingFace tokenizer.json.")
    parser.add_argument("--max_runtime_context", type=int, default=4096,
                        help="KV-cache budget in tokens (0: the 128K trained "
                             "context — several GB of cache; the C++ demos "
                             "also run at 4096).")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Number of tokens to generate.")
    parser.add_argument("--chat", action="store_true",
                        help="Multi-turn chat instead of the one-shot completion.")
    return parser.parse_args()


def run_chat(llama: Llama3) -> None:
    """Multi-turn session. The conversation lives in the KV cache, so each turn
    only hands over the text written since the last one."""
    tok, model = llama.tokenizer, llama.model

    eog = {tok.piece_to_id(p) for p in END_OF_TURN_PIECES}
    if -1 in eog:
        raise RuntimeError(f"tokenizer is missing one of {END_OF_TURN_PIECES}")

    print("\n=== Chat (blank line to quit) ===")
    first_turn = True
    context_full = False

    while not context_full:
        try:
            user_msg = input("\n> ")
        except EOFError:
            break
        if not user_msg:
            break

        # Generation stops *on* the end-of-turn token without consuming it, so
        # the previous reply is still open in the cache: close it here, so the
        # cache matches the training format.
        delta = ((BEGIN_OF_TEXT if first_turn else END_OF_TURN)
                 + USER_HEADER + user_msg + END_OF_TURN + ASSISTANT_HEADER)
        first_turn = False

        tokens = tok.encode(delta)   # the delta carries BOS itself on turn one
        if model.tokens_so_far + len(tokens) > model.max_runtime_context:
            context_full = True
            break

        # Prefill the turn; sampling runs on-device, so the forward already
        # returns the first reply token.
        print()
        out = model({"inTokenIds": pyvai.Tensor.from_data(np.asarray(tokens, dtype=np.int32))})
        nxt = int(out["nextTokenId"].get_data("uint32").reshape(-1)[0])
        model.invalidate_command_buffer()   # prefill -> decode boundary

        # Decode until end-of-turn, printing the reply as it arrives. Decoding
        # the whole reply each step (rather than one token) keeps multi-token
        # characters from being split mid-UTF-8.
        reply, shown = [], 0
        while nxt not in eog:
            if model.tokens_so_far + 1 > model.max_runtime_context:
                context_full = True
                break
            reply.append(nxt)
            text = tok.decode(reply)
            print(text[shown:], end="", flush=True)
            shown = len(text)
            out = model({"inTokenIds": pyvai.Tensor.from_data(np.asarray([nxt], dtype=np.int32))})
            nxt = int(out["nextTokenId"].get_data("uint32").reshape(-1)[0])
        print()

    if context_full:
        print(f"\n[context full: {model.tokens_so_far} of "
              f"{model.max_runtime_context} tokens used]")
              
    print("\n--- Chat ended ---")


def main():
    args = parse_args()
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"weights not found: {args.weights_path}")
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"tokenizer not found: {args.tokenizer_path}")

    pyvai.init_vai_engine()

    # Switch w_default_dtype to Dtype.F32 to run weights in FP32. Tagging the
    # nodes (set_profile_tags) is what turns per-node GPU profiling on.
    llama = Llama3(w_default_dtype=pyvai.Dtype.F16,
                   max_runtime_context=args.max_runtime_context)
    llama.set_profile_tags()
    llama.load_weights(args.weights_path)
    llama.load_tokenizer(args.tokenizer_path)

    if args.chat:
        run_chat(llama)
    else:
        print(f'Prompt: "{PROMPT}"')
        print(f"Generating {args.max_new_tokens} tokens...\n")
        text = llama.generate(PROMPT, max_new_tokens=args.max_new_tokens)

        print("=== Generated ===")
        print(text)

    llama.print_profile()
    return 0


if __name__ == "__main__":
    main()