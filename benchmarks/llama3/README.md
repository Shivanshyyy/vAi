# Llama 3 Benchmarks

## Summary

Runs Llama 3 under vAi and llama.cpp on the same GPU and compares prefill / decode throughput.

- Checkpoint: `meta-llama/Llama-3.2-1B-Instruct` for both — vAi reads the safetensors, llama.cpp the [f16 GGUF](#gguf-conversion) converted from it.
- Conditions: as set out in the shared conditions of [benchmarks/README.md](../README.md#shared-conditions-vai-vs-llamacpp).

> The original `meta-llama/*` repositories are gated and need approval plus a token. The [unsloth mirror](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) carries the same checkpoint and can be downloaded instead (the Llama 3.2 Community License still applies).

## GGUF conversion

For the llama.cpp bench only. Begin once [installing llama.cpp](../README.md#installing-llamacpp) and the [GGUF conversion setup](../README.md#gguf-conversion) (`$CONVERT`, `pip install`) are done, and the Llama 3 checkpoint (`data/llama3/tokenizer.json` + `data/llama3/3.2-1B/model.safetensors`) is in place.

```bash
cd data/llama3/3.2-1B
curl -LO https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/config.json
curl -LO https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/tokenizer_config.json
curl -LO https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/special_tokens_map.json
cp ../tokenizer.json .
cd ../../..
PYTHONPATH=$CONVERT python $CONVERT/convert_hf_to_gguf.py data/llama3/3.2-1B --outtype f16 --outfile data/llama3/3.2-1B/model-f16.gguf
```

## Building and running

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON
cmake --build build --config Release

build/bin/bench-llama3-vai
build/bin/bench-llama3-llamacpp
```

*With no arguments a bench measures every prompt length in the tables above (146 512 1024 2048); pass lengths to measure only those (e.g. `bench-llama3-vai 2048`).*

Options:

- `--size <name>`: which checkpoint to measure. The name is the folder under `data/llama3/` — `3.2-1B` (default) or `3.2-3B`. Both benches take the same name, so pass it to each to keep them on the same size.

  ```bash
  build/bin/bench-llama3-vai      --size 3.2-3B
  build/bin/bench-llama3-llamacpp --size 3.2-3B
  ```

- `--showText`: print rep 0's generated text for the last length after the table (off by default, so the output stays just the table)
- `--list-devices`: print the device list and exit (llamacpp bench)

## Results

### Llama 3.2-1B

#### AMD Radeon 890M (Strix Point)

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 3.2-1B | Ubuntu 26.04 LTS | RADV Mesa 26.0.3 | 1.4.335 | DPM `high` |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 1,475 | 16.53 |
|  | llama.cpp | 863.0 | 16.16 |
| 512 | vAi | 1,653 | 16.46 |
|  | llama.cpp | 1,290 | 16.09 |
| 1024 | vAi | 1,831 | 16.37 |
|  | llama.cpp | 1,311 | 16.04 |
| 2048 | vAi | 1,871 | 16.12 |
|  | llama.cpp | 1,299 | 15.84 |

#### AMD Radeon 8060S (Strix Halo)

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 3.2-1B | Ubuntu 24.04.4 LTS | RADV Mesa 25.2.8 | 1.4.318 | DPM `high` |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 3,697 | 81.94 |
|  | llama.cpp | 1,453 | 67.30 |
| 512 | vAi | 4,932 | 81.66 |
|  | llama.cpp | 2,975 | 66.83 |
| 1024 | vAi | 5,440 | 81.09 |
|  | llama.cpp | 3,593 | 67.26 |
| 2048 | vAi | 5,486 | 75.27 |
|  | llama.cpp | 3,894 | 65.98 |

#### Intel UHD Graphics (TGL GT1)

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 3.2-1B | Ubuntu 24.04.4 LTS | ANV Mesa 25.2.8 | 1.4.318 | not lockable |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 93.24 | 9.67 |
|  | llama.cpp | 92.30 | 9.45 |
| 512 | vAi | 123.1 | 9.63 |
|  | llama.cpp | 122.3 | 9.39 |
| 1024 | vAi | 119.5 | 9.57 |
|  | llama.cpp | 118.5 | 9.29 |
| 2048 | vAi | 111.2 | 9.43 |
|  | llama.cpp | 109.9 | 9.05 |

#### NVIDIA GeForce RTX 3060 Laptop

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 3.2-1B | Ubuntu 24.04.4 LTS | NVIDIA 595.84 | 1.4.329 | 1700 MHz |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 6,350 | 114.3 |
|  | llama.cpp | 6,096 | 113.8 |
| 512 | vAi | 8,761 | 115.3 |
|  | llama.cpp | 10,020 | 110.5 |
| 1024 | vAi | 9,253 | 113.2 |
|  | llama.cpp | 10,934 | 110.0 |
| 2048 | vAi | 9,135 | 111.8 |
|  | llama.cpp | 11,196 | 105.7 |

#### NVIDIA GeForce RTX 5070

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 3.2-1B | Ubuntu 24.04.4 LTS | NVIDIA 595.84 | 1.4.329 | 2700 MHz |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 15,759 | 213.9 |
|  | llama.cpp | 12,625 | 225.1 |
| 512 | vAi | 25,313 | 213.2 |
|  | llama.cpp | 29,671 | 221.5 |
| 1024 | vAi | 28,591 | 212.3 |
|  | llama.cpp | 30,211 | 220.1 |
| 2048 | vAi | 27,188 | 209.2 |
|  | llama.cpp | 31,198 | 215.8 |

#### NVIDIA GeForce RTX 4090

*Rented on vast.ai*

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 3.2-1B | Ubuntu 24.04 (vastai/base-image) | NVIDIA 595.284.320 | 1.4.329 | not lockable |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 22,699 | 313.2 |
|  | llama.cpp | 26,975 | 292.7 |
| 512 | vAi | 51,332 | 311.0 |
|  | llama.cpp | 50,166 | 290.6 |
| 1024 | vAi | 52,535 | 309.9 |
|  | llama.cpp | 50,713 | 288.6 |
| 2048 | vAi | 57,269 | 305.9 |
|  | llama.cpp | 43,456 | 282.8 |

#### NVIDIA GeForce RTX 5090

*Rented on vast.ai*

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 3.2-1B | Ubuntu 24.04 (vastai/base-image) | NVIDIA 595.232.192 | 1.4.329 | not lockable |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 33,620 | 455.5 |
|  | llama.cpp | 25,334 | 427.1 |
| 512 | vAi | 56,603 | 455.4 |
|  | llama.cpp | 66,053 | 424.0 |
| 1024 | vAi | 61,723 | 453.9 |
|  | llama.cpp | 55,646 | 425.6 |
| 2048 | vAi | 70,532 | 449.6 |
|  | llama.cpp | 52,877 | 420.2 |

### Llama 3.2-3B

#### AMD Radeon 8060S (Strix Halo)

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 3.2-3B | Ubuntu 24.04.4 LTS | RADV Mesa 25.2.8 | 1.4.318 | DPM `high` |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 1,362 | 33.97 |
|  | llama.cpp | 875.3 | 32.83 |
| 512 | vAi | 1,726 | 33.87 |
|  | llama.cpp | 1,609 | 32.63 |
| 1024 | vAi | 2,033 | 31.98 |
|  | llama.cpp | 1,694 | 32.26 |
| 2048 | vAi | 2,062 | 31.46 |
|  | llama.cpp | 1,705 | 31.01 |

#### NVIDIA GeForce RTX 5070

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 3.2-3B | Ubuntu 24.04.4 LTS | NVIDIA 595.84 | 1.4.329 | 2700 MHz |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 5,191 | 88.23 |
|  | llama.cpp | 6,800 | 89.45 |
| 512 | vAi | 9,091 | 87.74 |
|  | llama.cpp | 12,644 | 88.81 |
| 1024 | vAi | 9,308 | 87.07 |
|  | llama.cpp | 13,059 | 88.02 |
| 2048 | vAi | 10,035 | 85.71 |
|  | llama.cpp | 12,546 | 87.05 |

#### NVIDIA GeForce RTX 4090

*Rented on vast.ai*

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 3.2-3B | Ubuntu 24.04 (vastai/base-image) | NVIDIA 595.284.320 | 1.4.329 | not lockable |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 9,738 | 128.0 |
|  | llama.cpp | 11,535 | 125.5 |
| 512 | vAi | 17,219 | 128.0 |
|  | llama.cpp | 22,849 | 124.8 |
| 1024 | vAi | 19,649 | 127.1 |
|  | llama.cpp | 23,503 | 123.8 |
| 2048 | vAi | 17,964 | 125.2 |
|  | llama.cpp | 21,387 | 121.2 |

#### NVIDIA GeForce RTX 5090

*Rented on vast.ai*

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 3.2-3B | Ubuntu 24.04 (vastai/base-image) | NVIDIA 595.232.192 | 1.4.329 | not lockable |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 12,168 | 194.0 |
|  | llama.cpp | 14,315 | 187.5 |
| 512 | vAi | 21,013 | 194.0 |
|  | llama.cpp | 27,118 | 185.7 |
| 1024 | vAi | 23,087 | 196.7 |
|  | llama.cpp | 27,020 | 180.5 |
| 2048 | vAi | 23,948 | 194.3 |
|  | llama.cpp | 24,943 | 176.7 |
