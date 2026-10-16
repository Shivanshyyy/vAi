# Gemma 3 4B — vAi against PyTorch, layer by layer

Runs the same image prompt through both stacks, taps every layer boundary, and
reports where the two part company. The point is to separate three things that
look alike from the outside:

- arithmetic that is simply *different* (fp16 vs fp32 rounding, accumulated)
- arithmetic that is *wrong* (a mismatched formula, a missing mask, a bad layout)
- a decision that *flipped* because two candidates were within the noise

## Files

| | |
|---|---|
| `dump_torch.py` | the reference: transformers hooks, one `.npy` per tap |
| `dump-vai.cpp` | the same run under vAi, tapped with extra graph outputs (`test-gemma3-dump`) |
| `compare.py` | pairs the two dumps up and prints the tables |
| `npy.h` | the `.npy` reader/writer the C++ side uses |

## Running it

Needs `data/gemma3/4B/` (the full HuggingFace snapshot) and a Python
environment with `torch`, `transformers` and `pillow`.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_TESTS=ON
cmake --build build --config Release

# 1. reference first — it writes the image and the tokens the vAi side reuses
python tests/gemma3/dump_torch.py --steps 8

# 2. vAi, from the repository root (the default paths are relative to it)
build/bin/test-gemma3-dump --steps 8

# 3. the tables
python tests/gemma3/compare.py
```

Both dumpers take `--no-vision` (drops the 27 SigLIP taps, ~0.85 GB per dump),
`--steps N` and `--out DIR`. `tests/gemma3/dumps/` is gitignored; a full pair is
about 1.5 GB.

### What the two sides are made to share

A layer comparison is only worth reading when the inputs are identical, so:

- **the image**: `dump_torch.py` writes `pixel_values.npy` and the vAi side reads
  it instead of preprocessing the JPEG itself. Without it the two would differ by
  the resize convention before the model even starts, and every row would measure
  that instead. (Run without the PyTorch dump present and the header says so.)
- **the prompt**: built independently on both sides and compared id by id — the
  first line of `compare.py` output.
- **the decode inputs**: `dump_torch.py` writes `steps.ids.npy`, the tokens its
  own greedy run consumed, and the vAi side is teacher-forced on them. A step's
  logit gap is then that step's own error, not the tail of an earlier one. Each
  side still records what it *would* have picked, which is what dates a flip.
- **sampling**: `topkSampling.config_.k = 1` on the vAi side, `do_sample=False`
  on the other. No draw enters the comparison.

## Reading the tables

`rel` is the one to watch: `rms(diff) / rms(reference)`, so it is scale-free
where the raw `max|d|` is not — this model's residual stream runs to ~3e5, and a
`max|d|` of 10,000 there is a smaller error than a `max|d|` of 1 in the vision
tower. `growth` divides a tap's `rel` by the previous tap's: ~1 means the layer
passed the error through, >1 means it made its own, <1 means a norm pulled it in.

The logits table is the one that decides anything. A logit difference only
matters where it moves the argmax, so each row carries both sides' top-1 and the
reference's own top1−top2 margin. A margin below the row's `max|d|` is a
coin-flip position, whoever wins it.

## Establishing what "close enough" means

Compare the reference against *itself* at a lower precision — `compare.py` takes
any two dump directories:

```bash
python tests/gemma3/dump_torch.py --dtype bfloat16 --out tests/gemma3/dumps/torch-bf16
python tests/gemma3/compare.py --vai tests/gemma3/dumps/torch-bf16 \
                               --torch tests/gemma3/dumps/torch
```

That is the band the reference implementation itself occupies at the dtype the
checkpoint ships in, and it is the right yardstick for vAi's numbers.

## What it found (Radeon 8060S, RADV Mesa 25.2.8)

Measured with vAi fp16 weights / fp32 activations against PyTorch fp32, and
PyTorch bf16 against PyTorch fp32 as the control:

| | vAi fp16 | torch bf16 (control) |
|---|---|---|
| vision tower, in → out | 2.6e-3 → 7.6e-2 | 2.7e-3 → 1.5e-1 |
| decoder (image prefill), in → out | 7.5e-2 → 4.2e-2 | 1.5e-1 → 6.1e-1 |
| decoder (decode step), in → out | 5.2e-4 → 1.7e-2 | 4.3e-3 → 2.8e-2 |
| decode logits, `rel` per step | 1.1e-2 … 4.7e-2 | 1.9e-2 … 6.4e-2 |
| argmax over 8 teacher-forced steps | all 8 agree | all 8 agree |

- **vAi is closer to fp32 than bf16 is, at every stage.** fp16 carries 10
  mantissa bits to bf16's 7, so this is what correct arithmetic looks like.
- **The error is made in the vision tower, not the decoder.** The decoder's own
  contribution is ~1.05x per layer; the image rows simply arrive carrying 7.5e-2.
  The text-only decode path, which never touches the tower, stays near 1e-3.
- **fp16 is not usable for this model's activations.** PyTorch in fp16 overflows
  at text layer 5 (the residual stream peaks at ~3e5 against fp16's 65504 ceiling)
  and every tap after it is NaN. vAi holds fp16 *weights* with an fp32 residual
  stream, which is why it runs at all — `ioDtypeHint = Dtype::Fp32` is load-bearing
  here, not a default worth changing.
- **The first generated token is decided by 0.74 logit**, against a `max|d|` of
  1.3. That position is a coin flip for any implementation, which is why two
  correct runs can still open with different words.
