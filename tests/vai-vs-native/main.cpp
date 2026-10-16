// test-vai-vs-native: what does the vAi framework cost on top of the kernel?
//
// The same ggml-style-matmul .spv is driven two ways over the same shapes —
// once by hand-written Vulkan (on-native.cpp), once through a vAi graph
// (on-vai.cpp) — and both report GPU-timestamp time for the dispatch alone.
// A ratio near 1.0 means the framework adds nothing the GPU can see.
//
//   --gpu <i>   pick a physical device for the raw path (default: first discrete)
//   --reps <n>  measured iterations per case

#include "comparison-native.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// K must be a multiple of BK (32): the kernel is built in its ALIGNED
// permutation and reads whole k-blocks.
const BenchCase CASES[] = {
    //   M     N     K
    {    1,  2048,  768 },   // decode-shaped: one token
    {  150,  2048,  768 },   // prefill-shaped
    {    1,   512,  768 },
    {  150,   512,  768 },
    {  150,  3072,  768 },   // ffn fc1
};

const int WARMUP = 3;

} // namespace


int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);

    int gpuIndex = -1;
    int reps     = 500;
    for (int i = 1; i < argc; i++) 
    {
        if      (!strcmp(argv[i], "--gpu")  && i + 1 < argc) gpuIndex = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps     = atoi(argv[++i]);
    }

    // Both devices up front, so neither backend's start-up chatter lands
    // inside the report.
    nativeInit(gpuIndex);
    vaiInit();

    printf("\n=== ggml-style-matmul: native Vulkan vs vAi (GPU avg ms over %d iters) ===\n", reps);
    printf("%6s %6s %6s | %4s %10s | %10s | %10s | %s\n",
           "M", "N", "K", "tile", "grid", "native(ms)", "vai(ms)", "vai/native");
    printf("--------------------------------------------------------------------------------\n");

    for (const BenchCase& c : CASES) 
    {
        if (c.K % 32 != 0) 
        {
            fprintf(stderr, "K=%u is not a multiple of 32, skipping\n", c.K);
            continue;
        }

        const BenchResult native = nativeRun(c, WARMUP, reps);
        const BenchResult vai    = vaiRun(c, WARMUP, reps);

        // The two paths must have picked the same warptile and grid, or they
        // are not running comparable work.
        const bool sameShape = native.tileClass == vai.tileClass
                            && native.gridX == vai.gridX && native.gridY == vai.gridY;

        char grid[16];
        snprintf(grid, sizeof(grid), "%ux%u", native.gridX, native.gridY);
        printf("%6u %6u %6u | %4c %10s | %10.4f | %10.4f | %9.3fx%s\n",
               c.M, c.N, c.K, native.tileClass, grid, native.avgMs, vai.avgMs,
               vai.avgMs / native.avgMs, sameShape ? "" : "  <- MISMATCH");

        if (!sameShape)
            fprintf(stderr, "  native: tile %c grid %ux%u | vai: tile %c grid %ux%u\n",
                    native.tileClass, native.gridX, native.gridY,
                    vai.tileClass, vai.gridX, vai.gridY);
    }

    nativeShutdown();
    return 0;
}
