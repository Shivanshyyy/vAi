#ifndef VAI_TEST_COMPARISON_NATIVE_H
#define VAI_TEST_COMPARISON_NATIVE_H

// Shared contract between the two benchmark paths of test-vai-vs-native.
// Both drive the same kernel (ggml-style-matmul) over the same shapes and
// return GPU-timestamp timings, so the caller can compare them directly:
//
//   on-native.cpp : raw Vulkan — device, pipeline and dispatch by hand
//   on-vai.cpp    : the vAi graph (Linear3Node) doing the same work
//
// Both load the .spv that eva compiled from src/shaders, so the SPIR-V is
// byte-identical between paths and only the surrounding host code differs.
#include <cstdint>

// O [M, N] = I [M, K] · W [N, K]ᵀ  (vai naming; M = batch, N = outDim, K = inDim)
struct BenchCase {
    uint32_t M, N, K;
};

struct BenchResult {
    double   avgMs;      // mean over the measured reps, GPU timestamps only
    char     tileClass;  // ggml warptile pick: 's' | 'm' | 'l'
    uint32_t gridX, gridY;
};

// Deterministic operand fill, shared so both paths multiply identical data.
inline float benchInitW(uint32_t n, uint32_t k) { return float(((n * 7 + k * 3) % 17) - 8) / 32.0f; }
inline float benchInitI(uint32_t m, uint32_t k) { return float(((m * 5 + k * 11) % 19) - 9) / 32.0f; }

// Raw-Vulkan path. init/shutdown bracket the whole run: the device, pipeline
// layout and query pool are built once and reused across cases.
void nativeInit(int gpuIndex);
BenchResult nativeRun(const BenchCase& c, int warmup, int reps);
void nativeShutdown();

// vAi path. Each call builds, warms up and measures one net; vaiInit only
// forces eva to create its device up front, so its banner does not land in
// the middle of the report.
void vaiInit();
BenchResult vaiRun(const BenchCase& c, int warmup, int reps);

#endif // VAI_TEST_COMPARISON_NATIVE_H
