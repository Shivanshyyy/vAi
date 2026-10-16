# Gemma 3 Benchmarks

## Summary

Runs Gemma 3 under vAi and llama.cpp on the same GPU and compares prefill / decode throughput.

- Checkpoint: the same `google/gemma-3-4b-it` checkpoint for both — vAi reads the safetensors, llama.cpp the [f16 model and multimodal-projector GGUF files](#gguf-conversion) converted from it.
- Precision: FP16 weights and KV cache for the results below.
- Input: the same deterministic 8-bit RGB image and a shared VQA prompt. vAi normalizes those RGB pixels to `[-1, 1]`; llama.cpp processes them through mtmd. Full input counts include 256 image tokens and chat-template tokens.
- Timing: host wall-clock for both frameworks. Prefill includes the vision encoder and LLM prefill, with throughput calculated from the full input token count. Decode excludes the prefill that seeds it.
- Conditions: as set out in the shared conditions of [benchmarks/README.md](../README.md#shared-conditions-vai-vs-llamacpp). Each length uses 3 warm-up prefills and 8 warm-up decode tokens, followed by 15 prefill samples and five 128-token decode runs. Result tables show median throughput.

> The original `google/gemma-3-4b-it` repository is gated and requires accepting the Gemma license and authenticating with Hugging Face.

## GGUF conversion

For the llama.cpp bench only. Begin once [installing llama.cpp](../README.md#installing-llamacpp) and the [GGUF conversion setup](../README.md#gguf-conversion) (`$CONVERT`, `pip install`) are done, and the Gemma 3 checkpoint (safetensors, configuration, tokenizer, and processor files) is in `data/gemma3/4B/`.

llama.cpp must be built and installed with multimodal support (`-DLLAMA_BUILD_MTMD=ON`) so that CMake can find both `llama` and `mtmd`.

```bash
mkdir -p data/gemma3/4B/GGUF
PYTHONPATH=$CONVERT python $CONVERT/convert_hf_to_gguf.py data/gemma3/4B --outtype f16 --outfile data/gemma3/4B/GGUF/model-f16.gguf
PYTHONPATH=$CONVERT python $CONVERT/convert_hf_to_gguf.py data/gemma3/4B --mmproj --outtype f16 --outfile data/gemma3/4B/GGUF/mmproj-f16.gguf
```

## Building and running

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON
cmake --build build --config Release

build/bin/bench-gemma3-vai --size 4B --precision fp16
build/bin/bench-gemma3-llamacpp --size 4B --precision fp16
```

*With no arguments a bench measures text lengths 128 256 512 1024 2048 4096; pass lengths to measure only those (e.g. `bench-gemma3-vai 512 2048`).*

Options:

- `--size <name>`: checkpoint folder under `data/gemma3/`; use `4B` for this benchmark (default). Both benches take the same name.
- `--precision fp16|fp32`: precision (default: `fp16`). vAi sets weight, compute, and KV-cache precision; llama.cpp selects the corresponding model/projector GGUF files and KV-cache precision.
- `--showText`: accepted for compatibility; rep 0's generated text for the last length is already printed by default.
- `--list-devices`: print the device list and exit (llamacpp bench).

Set `BENCH_VERBOSE=1` to retain llama.cpp and mtmd logs.

To verify input parity without running inference:

```bash
cmake --build build --target bench-gemma3-check-input
build/bin/bench-gemma3-check-input
```

This requires the 4B tokenizer and both FP16 GGUF files. It compares the actual
vAi prompt builder with llama.cpp/mtmd at all six default lengths, checking
text token IDs, the 256-token image block and its position, and bidirectional
image attention. It also checks the shared RGB input and vAi normalization.
For the benchmark logs below, the shared 128 / 256 / 512 / 1024 / 2048 runs have total prompt lengths 396 / 524 / 780 / 1292 / 2316.

The runners use top-k 40 sampling, but their random generators and numerical
kernels differ, so generated text need not match. Both time 128 decode steps;
the displayed text also includes the first token sampled during prefill.
llama.cpp prefills text/image/text chunks separately, while vAi uses one model
forward with an image attention block. Both include vision encoding and LLM
prefill in the measured interval and request logits only at the final token.

## Results

### NVIDIA GeForce RTX 5090

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 4B (FP16) | Ubuntu 20.04 | 580.380.320 | 1.4.312 | not recorded |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 128 | vAi | 3,245.68 | 129.47 |
|  | llama.cpp | 993.14 | 146.26 |
| 256 | vAi | 4,094.95 | 129.28 |
|  | llama.cpp | 1,277.73 | 145.98 |
| 512 | vAi | 5,530.30 | 129.50 |
|  | llama.cpp | 1,829.83 | 146.02 |
| 1024 | vAi | 7,938.64 | 129.50 |
|  | llama.cpp | 2,840.20 | 145.70 |
| 2048 | vAi | 10,975.87 | 128.98 |
|  | llama.cpp | 4,580.10 | 145.29 |

*Decode 128 tokens, 5 reps; prefill 15 reps; medians shown. GPU clock was not recorded in the benchmark log.*

### NVIDIA GeForce RTX 4090

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 4B (FP16) | Ubuntu 20.04 | 610.172.128 | 1.4.341 | not recorded |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 128 | vAi | 2,712.18 | 100.13 |
|  | llama.cpp | 666.96 | 94.61 |
| 256 | vAi | 3,290.96 | 99.87 |
|  | llama.cpp | 855.63 | 94.74 |
| 512 | vAi | 4,451.27 | 99.78 |
|  | llama.cpp | 1,294.05 | 94.82 |
| 1024 | vAi | 6,311.11 | 99.98 |
|  | llama.cpp | 1,949.27 | 93.22 |
| 2048 | vAi | 7,915.86 | 99.66 |
|  | llama.cpp | 2,949.38 | 92.85 |

*Decode 128 tokens, 5 reps; prefill 15 reps; medians shown. GPU clock was not recorded in the benchmark log.*

### AMD Radeon 8060S (Strix Halo)

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 4B (FP16) | Ubuntu 24.04.4 LTS | RADV Mesa 25.2.8 | 1.4.318 | auto (sclk 600/1100/2900 MHz; mclk 400/800/1000 MHz) |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 128 | vAi | 328.63 | 27.52 |
|  | llama.cpp | 324.67 | 25.14 |
| 256 | vAi | 367.63 | 27.35 |
|  | llama.cpp | 414.56 | 25.04 |
| 512 | vAi | 542.32 | 27.21 |
|  | llama.cpp | 565.38 | 24.96 |
| 1024 | vAi | 828.50 | 27.04 |
|  | llama.cpp | 781.57 | 24.84 |
| 2048 | vAi | 1,095.14 | 26.80 |
|  | llama.cpp | 1,048.20 | 24.59 |

*Decode 128 tokens, 5 reps; prefill 15 reps; medians shown. The benchmark log reports the AMDGPU clock mode as `auto`.*

### AMD Radeon 890M Graphics (Strix Point)

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 4B (FP16) | not recorded | RADV Mesa 26.0.8 | 1.4.335 | not recorded |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 16 | vAi | 66.78 | 5.17 |
|  | llama.cpp | 72.84 | 5.09 |
| 32 | vAi | 70.22 | 5.17 |
|  | llama.cpp | 76.53 | 4.33 |
| 64 | vAi | 77.40 | 5.17 |
|  | llama.cpp | 83.55 | 5.08 |
| 128 | vAi | 88.50 | 5.16 |
|  | llama.cpp | 93.92 | 5.07 |

*Decode 128 tokens, 5 reps; prefill 15 reps; medians shown. This run uses prompt lengths 16 / 32 / 64 / 128 rather than the 128–2048 sequence used by the other devices. OS and GPU clock were not recorded in the benchmark log.*

The `prompt` column is the requested text length (`request` in the benchmark output). For the 128 / 256 / 512 / 1024 / 2048 runs, full input token counts including image and chat-template tokens are 396 / 524 / 780 / 1292 / 2316 for both frameworks. For the Strix Point 16 / 32 / 64 / 128 run, the corresponding full input token counts are 284 / 300 / 332 / 396. Prefill throughput includes the vision encoder and LLM prefill and uses these full input token counts.
