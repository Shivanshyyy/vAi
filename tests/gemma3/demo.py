"""gemma3 layer dump (PyTorch side) — the reference the vAi dump is read against.

Runs google/gemma-3-4b-it from data/gemma3/4B through transformers with the
same image and the same prompt examples/gemma3/demo-4B.cpp uses, greedy, and
writes every layer boundary to <out>/<name>.npy. The names are the ones
tests/gemma3/dump-vai.cpp taps, so compare.py can pair them up.

fp32 by default: the point of the reference is to be the accurate side. --dtype
bfloat16 reproduces what demo-4B.py used to do, which is a useful second column
because bf16's own rounding is the yardstick for "does this gap matter".

Run this BEFORE the vAi dump: it writes pixel_values.npy (the image both sides
then share) and steps.ids.npy (the tokens both sides then decode).

    python tests/gemma3/dump_torch.py [--steps N] [--dtype float32|bfloat16]
                                      [--device cuda|cpu] [--out DIR]
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "data", "gemma3", "4B")
IMAGE_PATH = os.path.join(ROOT, "data", "gemma3", "image.jpg")
QUESTION = "Where is he looking?"


def save(out_dir, name, tensor):
    arr = tensor.detach().to(torch.float32).cpu().numpy()
    np.save(os.path.join(out_dir, name + ".npy"), np.ascontiguousarray(arr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "gemma3", "dumps", "torch"))
    ap.add_argument("--no-vision", action="store_true",
                    help="skip the 27 SigLIP layer taps (~0.85 GB of the dump)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    print(f"model  : {MODEL_PATH}\ndevice : {args.device}\ndtype  : {dtype}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=dtype, local_files_only=True).to(args.device).eval()

    image = Image.open(IMAGE_PATH).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": QUESTION}]}]
    batch = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt")

    ids = batch["input_ids"]
    print(f"prompt : {ids.shape[-1]} tokens")
    np.save(os.path.join(args.out, "prompt.ids.npy"), ids[0].to(torch.int32).numpy())
    # [1, 3, 896, 896] -> [3, 896, 896]: the layout the vAi side reads back.
    save(args.out, "pixel_values", batch["pixel_values"][0])

    # -----------------------------------------------------------------------
    # Taps. capture[] is filled by the hooks; `grab` says whether this forward's
    # values are the ones being written, so the decode steps do not overwrite
    # the prefill's.
    # -----------------------------------------------------------------------
    capture = {}
    state = {"grab": True}

    def hook(name, index=None, on_input=False):
        def fn(_module, inputs, output):
            if not state["grab"]:
                return
            value = inputs[0] if on_input else output
            if isinstance(value, (tuple, list)):
                value = value[index if index is not None else 0]
            capture[name] = value.detach()
        return fn

    base = model.model
    # transformers flattened SiglipVisionModel at some point: the tower is either
    # the module itself or the .vision_model inside it.
    vision = getattr(base.vision_tower, "vision_model", base.vision_tower)
    text = base.language_model
    proj = base.multi_modal_projector

    handles = [
        vision.embeddings.register_forward_hook(hook("vis.embed")),
        vision.post_layernorm.register_forward_hook(hook("vis.post_norm")),
        # The 4x4 average pool is inline in get_image_features, so it is caught
        # as what the soft-embedding norm receives.
        proj.mm_soft_emb_norm.register_forward_hook(hook("vis.pool", on_input=True)),
        proj.mm_soft_emb_norm.register_forward_hook(hook("vis.pool_norm")),
        proj.register_forward_hook(hook("vis.proj")),
        # Layer 0's input is the embedding the decoder actually starts from:
        # scaled by sqrt(D), with the image rows already merged in.
        text.layers[0].register_forward_hook(hook("txt.embed", on_input=True)),
        text.norm.register_forward_hook(hook("txt.final_norm")),
        model.lm_head.register_forward_hook(hook("logits")),
    ]
    if not args.no_vision:
        for i, layer in enumerate(vision.encoder.layers):
            handles.append(layer.register_forward_hook(hook(f"vis.{i:02d}")))
    for i, layer in enumerate(text.layers):
        handles.append(layer.register_forward_hook(hook(f"txt.{i:02d}")))

    dev_batch = {k: (v.to(args.device).to(dtype) if torch.is_floating_point(v) else v.to(args.device))
                 for k, v in batch.items()}

    # ----- image prefill ---------------------------------------------------
    with torch.inference_mode():
        out = model(**dev_batch, use_cache=True)

    for name, value in capture.items():
        save(args.out, name, value[0] if value.dim() == 3 else value)
    print(f"prefill: {len(capture)} taps written")

    logits = out.logits[0, -1].float()
    tok = int(logits.argmax())
    cache = out.past_key_values

    # ----- decode ----------------------------------------------------------
    # Free-running greedy: what this side picks is also what the vAi side is
    # teacher-forced on, so both consume the same tokens either way.
    stop_ids = set(processor.tokenizer.convert_tokens_to_ids(
        ["<eos>", "<end_of_turn>"]))
    fed, picks = [], []
    for step in range(args.steps):
        if tok in stop_ids:
            break
        fed.append(tok)
        state["grab"] = (step == 0)
        capture.clear()
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([[tok]], device=args.device),
                        past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[0, -1].float()
        save(args.out, f"step.{step:02d}.logits", logits)
        if step == 0:
            for name, value in capture.items():
                if name.startswith("txt.") or name == "logits":
                    save(args.out, "step00." + name, value[0] if value.dim() == 3 else value)
        tok = int(logits.argmax())
        picks.append(tok)

    for h in handles:
        h.remove()

    np.save(os.path.join(args.out, "steps.ids.npy"), np.array(fed, dtype=np.int32))
    np.save(os.path.join(args.out, "steps.picks.npy"), np.array(picks, dtype=np.int32))

    text_out = processor.decode(fed + [tok], skip_special_tokens=True)
    print(f"decode : {len(fed)} steps — {text_out!r}")
    print(f"wrote  : {args.out}")

    print("---------------------------")
    print(text_out)


if __name__ == "__main__":
    main()
