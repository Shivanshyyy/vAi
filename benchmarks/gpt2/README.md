# GPT-2 Benchmarks

## Summary

Runs GPT-2 under vAi and llama.cpp on the same GPU and compares prefill / decode throughput.

- Checkpoint: the same HuggingFace checkpoint (`openai-community/gpt2`) for both — vAi reads the safetensors, llama.cpp the [f16 GGUF](#gguf-conversion) converted from it.
- Conditions: as set out in the shared conditions of [benchmarks/README.md](../README.md#shared-conditions-vai-vs-llamacpp).

## GGUF conversion

For the llama.cpp bench only. Begin once [installing llama.cpp](../README.md#installing-llamacpp) and the [GGUF conversion setup](../README.md#gguf-conversion) (`$CONVERT`, `pip install`) are done, and the GPT-2 checkpoint (`data/gpt2/tokenizer.json` + `data/gpt2/124M/model.safetensors`) is in place.

```bash
cd data/gpt2/124M
curl -LO https://huggingface.co/openai-community/gpt2/resolve/main/config.json
cp ../tokenizer.json .
cd ../../..
PYTHONPATH=$CONVERT python $CONVERT/convert_hf_to_gguf.py data/gpt2/124M --outtype f16 --outfile data/gpt2/124M/model-f16.gguf
```

## Building and running

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON
cmake --build build --config Release

build/bin/bench-gpt2-vai
build/bin/bench-gpt2-llamacpp
```

*With no arguments a bench measures every prompt length in the tables above (146 256 512); pass lengths to measure only those (e.g. `bench-gpt2-vai 512`).*

Options:

- `--size <name>`: which checkpoint to measure. The name is the folder under `data/gpt2/` — `124M` (default) · `355M` · `774M` · `1.5B`. Both benches take the same name, so pass it to each to keep them on the same size.

  ```bash
  build/bin/bench-gpt2-vai      --size 355M
  build/bin/bench-gpt2-llamacpp --size 355M
  ```
  
- `--showText`: print rep 0's generated text for the last length after the table (off by default, so the output stays just the table)
- `--list-devices`: print the device list and exit (llamacpp bench)

## Results

### AMD Radeon 890M (Strix Point)

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 124M | Ubuntu 26.04 LTS | RADV Mesa 26.0.3 | 1.4.335 | DPM `high` |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 7,694 | 137.3 |
|  | llama.cpp | 5,712 | 143.5 |
| 256 | vAi | 8,689 | 135.2 |
|  | llama.cpp | 7,413 | 139.9 |
| 512 | vAi | 10,899 | 130.5 |
|  | llama.cpp | 7,803 | 133.6 |

### AMD Radeon 8060S (Strix Halo)

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 124M | Ubuntu 24.04.4 LTS | RADV Mesa 25.2.8 | 1.4.318 | DPM `high` |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 33,432 | 502.5 |
|  | llama.cpp | 24,258 | 609.4 |
| 256 | vAi | 51,241 | 500.2 |
|  | llama.cpp | 36,447 | 603.1 |
| 512 | vAi | 50,777 | 500.9 |
|  | llama.cpp | 41,159 | 595.4 |

### Intel UHD Graphics (TGL GT1)

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 124M | Ubuntu 24.04.4 LTS | ANV Mesa 25.2.8 | 1.4.318 | not lockable |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 944.3 | 97.55 |
|  | llama.cpp | 843.4 | 84.29 |
| 256 | vAi | 1,232 | 96.61 |
|  | llama.cpp | 1,071 | 83.05 |
| 512 | vAi | 1,201 | 94.06 |
|  | llama.cpp | 1,065 | 80.90 |

### NVIDIA GeForce RTX 3060 Laptop

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 124M | Ubuntu 24.04.4 LTS | NVIDIA 595.84 | 1.4.329 | 1700 MHz |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 49,337 | 778.0 |
|  | llama.cpp | 34,787 | 614.1 |
| 256 | vAi | 57,060 | 772.6 |
|  | llama.cpp | 50,282 | 604.9 |
| 512 | vAi | 61,236 | 743.2 |
|  | llama.cpp | 50,910 | 592.5 |

### NVIDIA GeForce RTX 5070

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 124M | Ubuntu 24.04.4 LTS | NVIDIA 595.84 | 1.4.329 | 2700 MHz |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 85,240 | 1,340 |
|  | llama.cpp | 69,927 | 1,051 |
| 256 | vAi | 109,732 | 1,336 |
|  | llama.cpp | 103,042 | 1,060 |
| 512 | vAi | 154,785 | 1,319 |
|  | llama.cpp | 131,056 | 1,020 |

### NVIDIA GeForce RTX 4090

*Rented on vast.ai*

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 124M | Ubuntu 24.04 (vastai/base-image) | NVIDIA 595.284.320 | 1.4.329 | not lockable |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 130,956 | 1,514 |
|  | llama.cpp | 56,686 | 985.9 |
| 256 | vAi | 207,425 | 1,642 |
|  | llama.cpp | 113,774 | 1,004 |
| 512 | vAi | 269,911 | 1,631 |
|  | llama.cpp | 155,877 | 979.0 |

### NVIDIA GeForce RTX 5090

*Rented on vast.ai*

| Model size | OS | Driver | Vulkan | GPU clock |
|---|---|---|---|---|
| 124M | Ubuntu 24.04 (vastai/base-image) | NVIDIA 595.232.192 | 1.4.329 | not lockable |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 146 | vAi | 140,584 | 1,638 |
|  | llama.cpp | 75,363 | 949.3 |
| 256 | vAi | 214,617 | 1,765 |
|  | llama.cpp | 111,381 | 976.0 |
| 512 | vAi | 308,638 | 1,756 |
|  | llama.cpp | 153,821 | 940.5 |
