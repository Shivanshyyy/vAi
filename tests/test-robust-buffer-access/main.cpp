// robustBufferAccess A/B on the Linear tile path: the identical GEMM
// kernel with the loaders' row-tail guards kept (robust off) vs folded away
// behind VK_EXT_pipeline_robustness (robust on) — the optimization
// linear.h applies whenever the device supports it.
//
// Both input placements are timed, because they are known to disagree
// (see the robust note in linear.h):
//  - pinned: the graph input feeds Linear directly (host-visible memory,
//    PCIe reads dominate the staging loads)
//  - device-local: an identity Activation in front, so Linear reads
//    a device-local activation — the in-network condition
//
// Outputs of the two variants are verified identical first; a mismatch means
// a robust OOB read leaked past the store masks. Non-zero exit on failure.
#include "bench-common.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace vai;
using namespace vai::node;


// One summary row: robust off vs on on both input placements.
// speedup > 0 means robust (guards folded) is faster.
void comparisonTest(uint32_t M, uint32_t K, uint32_t N,
                    int rounds, int repeats, double budgetMs)
{
    std::vector<float> wData = makeFloats(N, K, [&](uint32_t n, uint32_t k) { return 0.1 * rnd(n, k, 1); });
    Tensor input = Tensor({M, K}, Dtype::Fp32)
        .setFloats(makeFloats(M, K, [&](uint32_t m, uint32_t k) { return rnd(m, k, 2); }));

    std::vector<double> pinOff, pinOn, pinUp, pinCtl, devOff, devOn, devUp, devCtl;
    for (int r = 0; r < repeats; ++r)
    {
        // Each duel is built from scratch, so the repeats differ in everything a
        // single call holds fixed. The verdict is the median of the per-duel
        // ratios, not a ratio of medians: each ratio is formed inside one duel.
        auto p = duel<PinnedModel>(M, K, N, wData, input, rounds, budgetMs);
        pinOff.push_back(p[0]);
        pinOn .push_back(p[1]);
        pinUp .push_back((p[0] - p[1]) / p[0] * 100.0);
        pinCtl.push_back(std::abs(p[0] - p[2]) / p[0] * 100.0);

        auto d = duel<DeviceLocalModel>(M, K, N, wData, input, rounds, budgetMs);
        devOff.push_back(d[0]);
        devOn .push_back(d[1]);
        devUp .push_back((d[0] - d[1]) / d[0] * 100.0);
        devCtl.push_back(std::abs(d[0] - d[2]) / d[0] * 100.0);
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto lo = [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); };
    auto hi = [](const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); };

    printf(" %5u %5u %7u | %8.3f %8.3f %+6.1f%% [%+5.1f,%+6.1f] %5.1f%%"
                        " | %8.3f %8.3f %+6.1f%% [%+5.1f,%+6.1f] %5.1f%%\n",
        M, K, N,
        median(pinOff), median(pinOn), median(pinUp), lo(pinUp), hi(pinUp), hi(pinCtl),
        median(devOff), median(devOn), median(devUp), lo(devUp), hi(devUp), hi(devCtl));
}


int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep prints visible up to an assert/abort
    eva::Device device = getVaiDevice();

    if (!device.supportsPipelineRobustness())
    {
        printf("VK_EXT_pipeline_robustness not supported on this device - "
               "the robust variant cannot be built, nothing to compare.\n");
        return 0;
    }

    printf("\n=== robust on/off output identity (tile GEMM, fp32 I/W/O, fp16 acc) ===\n");
    int fails = 0;
    fails += !verifyTest("clean",       256, 768, 2048);   // no tails: guards never fire
    fails += !verifyTest("mn-tail",     130, 768, 2050);   // row guards fire on both axes
    fails += !verifyTest("unaligned-K", 256, 770, 2052);   // masked scalar loads + n-tail
    printf(fails == 0 ? "  => ALL PASS\n" : "  => %d case(s) FAILED\n", fails);
    if (fails != 0)
        return 1;   // don't bench a kernel that leaks OOB garbage

    // Rounds per duel (a multiple of three, so the slot rotation closes) and
    // independent duels per row. 0 = verify only, stress-looping the identity
    // check above.
    const int rounds = (argc > 1) ? std::max(0, atoi(argv[1])) : 9;
    if (argc > 1 && rounds == 0)
        return 0;
    const int repeats = (argc > 2) ? std::max(1, atoi(argv[2])) : 3;
    constexpr double kBudgetMs = 120.0;   // per sample, sizes the forward count

    printf("\n=== robust vs guarded, GPU ms, p25 of %d rounds, %d independent duels ===\n",
           rounds, repeats);
    printf("                    |                    pinned input          "
           "        |                 device-local input\n");
    printf("    M     K       N |  guarded   robust  speedup [ min,   max]   ctl "
           "|  guarded   robust  speedup [ min,   max]   ctl\n");
    printf("                    |   median   median  median  [ over duels]   max "
           "|   median   median  median  [ over duels]   max\n");

    // GPT-2 124M prefill shapes (embed 768)
    comparisonTest(150, 768,  768,   rounds, repeats, kBudgetMs);   // attn q/k/v/out
    comparisonTest(150, 768,  3072,  rounds, repeats, kBudgetMs);   // ffn fc1
    comparisonTest(150, 3072, 768,   rounds, repeats, kBudgetMs);   // ffn fc2
    comparisonTest(150, 768,  50257, rounds, repeats, kBudgetMs);   // lm head

    // Llama3.2-1B prefill shapes (embed 2048)
    comparisonTest(150, 2048, 2048,  rounds, repeats, kBudgetMs);   // attn q/out
    comparisonTest(150, 2048, 8192,  rounds, repeats, kBudgetMs);   // ffn gate/up
    comparisonTest(150, 8192, 2048,  rounds, repeats, kBudgetMs);   // ffn down

    // Tail shape: accumulator tiles cross the M and N edges, so the folded
    // guards are actually exercised (clean shapes never fire them).
    comparisonTest(130, 768,  2050,  rounds, repeats, kBudgetMs);

    return 0;
}
