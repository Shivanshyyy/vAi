cmake_minimum_required(VERSION 3.16)

project(vai-tests)

# One folder per test, one executable per folder. To add a test: make a folder,
# drop its sources in, and append a block here.

# ---------------------------------------------------------------------------
# linear-nodes — LinearNode against the NaiveLinearNode reference, side by side
# ---------------------------------------------------------------------------
add_executable(test-linear-nodes
    linear-nodes/main.cpp
)

target_link_libraries(test-linear-nodes PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# linear-slab — linear-slab vs test-linear-slab2, the A/B geometry variant
# (a subgroup owns each slab instead of a workgroup). slab2 is not part of vAi,
# so it is built here and loaded by path through test-nodes.h.
# ---------------------------------------------------------------------------
set(LINEAR_SLAB_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/linear-slab)

# main.cpp verifies and benchmarks; main2.cpp benchmarks only. Swap which one
# is active — they both define main().
add_executable(test-linear-slab
    # linear-slab/main.cpp
    linear-slab/main2.cpp

    linear-slab/test-linear-slab.comp
    linear-slab/test-linear-slab2.comp
    linear-slab/test-linear-slab-v0.comp
    linear-slab/test-linear-slab-v2.comp
    linear-slab/test-linear-slab-v5.comp
    linear-slab/test-stream.comp
)

eva_auto_compile_shaders(test-linear-slab ${LINEAR_SLAB_SHADER_DIR})

target_link_libraries(test-linear-slab PRIVATE
    vAi
)

target_compile_definitions(test-linear-slab PRIVATE
    TEST_LINEAR_SLAB_SPV="${LINEAR_SLAB_SHADER_DIR}/test-linear-slab.comp.spv"
    TEST_LINEAR_SLAB2_SPV="${LINEAR_SLAB_SHADER_DIR}/test-linear-slab2.comp.spv"
    TEST_LINEAR_SLAB_V0_SPV="${LINEAR_SLAB_SHADER_DIR}/test-linear-slab-v0.comp.spv"
    TEST_LINEAR_SLAB_V2_SPV="${LINEAR_SLAB_SHADER_DIR}/test-linear-slab-v2.comp.spv"
    TEST_LINEAR_SLAB_V5_SPV="${LINEAR_SLAB_SHADER_DIR}/test-linear-slab-v5.comp.spv"
    TEST_STREAM_SPV="${LINEAR_SLAB_SHADER_DIR}/test-stream.comp.spv"
)

# ---------------------------------------------------------------------------
# vai-vs-native — what the framework costs on top of the kernel: the same
# ggml-style-matmul .spv driven by raw Vulkan and by a vAi graph. The raw path
# calls Vulkan directly, so it needs the loader on top of what vAi pulls in.
#
# The kernel is a benchmark reference, not part of vAi, so it lives with the
# test and is compiled here rather than by eva's vai/shaders sweep. Both paths
# load this one .spv, which is what makes the comparison host-side only.
# ---------------------------------------------------------------------------
find_package(Vulkan REQUIRED)

set(VAI_VS_NATIVE_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/vai-vs-native)

add_executable(test-vai-vs-native
    vai-vs-native/main.cpp
    vai-vs-native/on-native.cpp
    vai-vs-native/on-vai.cpp
    vai-vs-native/comparison-native.h
    vai-vs-native/test-linear-node.h

    vai-vs-native/ggml-style-matmul.comp
)

eva_auto_compile_shaders(test-vai-vs-native ${VAI_VS_NATIVE_SHADER_DIR})

target_include_directories(test-vai-vs-native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vai-vs-native
)

target_link_libraries(test-vai-vs-native PRIVATE
    vAi
    Vulkan::Vulkan
)

target_compile_definitions(test-vai-vs-native PRIVATE
    GGML_STYLE_MATMUL_SPV="${VAI_VS_NATIVE_SHADER_DIR}/ggml-style-matmul.comp.spv"
)

# ---------------------------------------------------------------------------
# attention — AttentionNode's per-(shape, device) paths (coopmat FA,
# scalar FA, GQA-packed decode, split-K, naive fallback) vs a host reference,
# plus a shape bench. The kernels are vAi's own, so no shader compile here.
# ---------------------------------------------------------------------------
add_executable(test-attention
    attention/main.cpp
)

target_link_libraries(test-attention PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# linear-cm — LinearNode's coopmat path (linear-cm) vs the NaiveLinearNode
# reference, plus the device-local-input benchmark (graph inputs are pinned,
# which distorts staging-bound GEMM timings).
# ---------------------------------------------------------------------------
add_executable(test-linear-cm
    linear-cm/main.cpp
)

target_link_libraries(test-linear-cm PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# test-robust-buffer-access — the same tile GEMM with the loaders' row-tail
# guards kept vs folded away behind VK_EXT_pipeline_robustness. vai::node::Linear
# derives that choice from the device, so the A/B needs a local copy of the
# node with the option forced; it lives beside the test and uses vAi's kernels.
# ---------------------------------------------------------------------------
# main.cpp is the A/B table over model shapes; main2.cpp sweeps M across a
# workgroup-tile boundary to split the two costs folding the guards away pays.
# Swap which one is active — they both define main().
add_executable(test-robust-buffer-access
    # test-robust-buffer-access/main.cpp
    test-robust-buffer-access/main2.cpp

    test-robust-buffer-access/bench-common.h
    test-robust-buffer-access/test-robust-linear-node.h
)

target_link_libraries(test-robust-buffer-access PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# tokenizer — Tokenizer vs HuggingFace tokenizers on the tokenizer.json files
# under data/, as fixed cases carrying the reference ids. Host-side only, so it
# is the one test here that needs no device.
# ---------------------------------------------------------------------------
add_executable(test-tokenizer
    tokenizer/main.cpp
)

target_link_libraries(test-tokenizer PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# llama3-decode-tps — decode throughput of the llama3 sample model, prefill
# excluded. The model owns its definition and its checkpoint loader, so this
# test and examples/llama3 can never end up on different weights.
# ---------------------------------------------------------------------------
add_executable(test-llama3-decode-tps
    llama3-decode-tps/main.cpp
)

target_link_libraries(test-llama3-decode-tps PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# vit-batching - pi0's vision tower over its three cameras in one pass against
# three passes of one. Same arithmetic either way, so the gap is dispatch size
# and the per-forward graph walk. Weights are random; only shapes matter.
# ---------------------------------------------------------------------------
add_executable(test-vit-batching
    vit-batching/main.cpp
)

target_link_libraries(test-vit-batching PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# chunked-prefill — the generated ids against the chunk size.
# ---------------------------------------------------------------------------
add_executable(test-chunked-prefill
    chunked-prefill/main.cpp
)

target_link_libraries(test-chunked-prefill PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# gemma3 — the Gemma 3 4B VLM against PyTorch, layer by layer and step by step.
# Dumps every layer boundary of one image prefill plus N decode steps; the
# PyTorch side (dump_torch.py) and the table (compare.py) live next to it. The
# model owns its definition and its checkpoint loader, so this test and
# examples/gemma3 can never end up on different weights.
# ---------------------------------------------------------------------------
add_executable(test-gemma3-dump
    gemma3/dump-vai.cpp
)

target_include_directories(test-gemma3-dump PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gemma3          # npy.h
)

target_link_libraries(test-gemma3-dump PRIVATE
    vAi
)

# ---------------------------------------------------------------------------
# concat-contiguous — Concat's contiguous case three ways: concat-1.comp (one
# thread per output element), concat-2.comp under contiguous=true, and buffer
# copies, timed and compared byte for byte. Both kernels live with the test.
# ---------------------------------------------------------------------------
set(CONCAT_CONTIGUOUS_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/concat-contiguous)

add_executable(test-concat-contiguous
    concat-contiguous/main.cpp

    concat-contiguous/concat-1.comp
    concat-contiguous/concat-2.comp
)

eva_auto_compile_shaders(test-concat-contiguous ${CONCAT_CONTIGUOUS_SHADER_DIR})

target_link_libraries(test-concat-contiguous PRIVATE
    vAi
)

target_compile_definitions(test-concat-contiguous PRIVATE
    CONCAT_1_SPV="${CONCAT_CONTIGUOUS_SHADER_DIR}/concat-1.comp.spv"
    CONCAT_2_SPV="${CONCAT_CONTIGUOUS_SHADER_DIR}/concat-2.comp.spv"
)
