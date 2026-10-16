// Why robust costs what it costs: sweep M across a workgroup-tile boundary and
// watch whether the A/B tracks the rows that fall outside it.
//
// Folding the guards away does two separate things, and main.cpp's shapes cannot
// tell them apart because every one of them sits on an M tail:
//
//  - the tail rows of the last M tile stop being skipped. With the guard the
//    lanes holding idx_m >= M are predicated off and issue no load at all; with
//    it folded they read out of bounds every k-block, for every N tile, and the
//    stores mask the result off afterwards. Cost scales with the wasted rows.
//  - every access in the pipeline is bounds-checked. Cost scales with all
//    accesses, whether or not any of them is out of bounds.
//
// So M % wgM splits the two:
//
//    M % wgM == 0   no wasted rows. Wasted-load cost is zero, per-access cost
//                   is not - whatever gap survives here is the second term.
//    M % wgM == 1   one useful row in the last tile, wgM - 1 wasted: the most
//                   the first term can be for that tile count.
//
// The sweep holds K and N fixed and moves M only, so the staged W volume per
// tile row is constant and the I side is what changes. wgM comes from the same
// heuristic the node uses, and is printed per row: it is picked from M, so it
// can move under the sweep, and a row where it did is a row to read carefully.
#include "bench-common.h"
#include <cstdlib>
#include <cstring>

// Rows the last M tile carries but does not use.
static uint32_t wastedRows(uint32_t M, uint32_t wgM)
{
    const uint32_t r = M % wgM;
    return r == 0 ? 0 : wgM - r;
}


// One sweep row: robust vs guarded at this M, with the tile geometry that made it.
static void sweepRow(uint32_t M, uint32_t K, uint32_t N, int rounds, int repeats, double budgetMs)
{
    const vai::node::Linear::PipeParamsTile t = vai::node::ggmlMulmmTileParams(M, N, K, vai::Dtype::Fp16, vai::Dtype::Fp16, getVaiDevice());

    std::vector<float> wData = makeFloats(N, K, [&](uint32_t n, uint32_t k) { return 0.1 * rnd(n, k, 1); });
    Tensor input = Tensor({M, K}, Dtype::Fp32)
        .setFloats(makeFloats(M, K, [&](uint32_t m, uint32_t k) { return rnd(m, k, 2); }));

    std::vector<double> off, on, up, ctl;
    for (int r = 0; r < repeats; ++r)
    {
        auto d = duel<DeviceLocalModel>(M, K, N, wData, input, rounds, budgetMs);
        off.push_back(d[0]);
        on .push_back(d[1]);
        up .push_back((d[0] - d[1]) / d[0] * 100.0);
        ctl.push_back(std::abs(d[0] - d[2]) / d[0] * 100.0);
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto lo = [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); };
    auto hi = [](const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); };

    const uint32_t wasted = wastedRows(M, t.wgM);
    const uint32_t staged = ((M + t.wgM - 1) / t.wgM) * t.wgM;   // rows the loaders walk

    printf(" %5u %5u %7u | %4u %4u | %4u  %5.1f%% | %8.3f %8.3f %+6.1f%% [%+5.1f,%+6.1f] %5.1f%%\n",
        M, K, N, t.wgM, t.wgN, wasted, 100.0 * wasted / double(staged),
        median(off), median(on), median(up), lo(up), hi(up), hi(ctl));
}


// What the driver compiled, for both variants of one shape. Intel exposes no
// internal representation, but the statistics carry instruction and memory
// message counts, which is what separates the two costs at the source:
//   robust folds guards away, so its instruction count should DROP. If it rises
//   instead, the driver is paying for the bounds checks in software - the only
//   way a check can cost anything once no access is out of bounds.
static void statsTest(uint32_t M, uint32_t K, uint32_t N)
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32 };

    auto statsOf = [&](bool robust) {
        test::LinearNode::Config opts = { .inDim = K, .outDim = N, .robust = robust,
                                           .captureStatistics = true, .accDtype = Dtype::Fp16 };
        PinnedModel m(opts, netCfg);
        m.lin["weight"] = Tensor({N, K}, Dtype::Fp32).setFloats(
            makeFloats(N, K, [&](uint32_t n, uint32_t k) { return 0.1 * rnd(n, k, 1); }));
        m({{"in0", Tensor({M, K}, Dtype::Fp32).setFloats(
            makeFloats(M, K, [&](uint32_t r, uint32_t c) { return rnd(r, c, 2); }))}})["out0"].getData();
        return m.lin.pipeline().statistics();
    };

    auto off = statsOf(false), on = statsOf(true);

    printf("\n=== compiled pipeline statistics, M=%u K=%u N=%u ===\n", M, K, N);
    if (off.empty())
    {
        printf("  driver exposes none (VK_KHR_pipeline_executable_properties absent or silent)\n");
        return;
    }

    printf("  %-40s %14s %14s\n", "statistic", "guarded", "robust");
    for (size_t i = 0; i < off.size() && i < on.size(); ++i)
        printf("  %-40s %14s %14s%s\n", off[i].name.c_str(),
               off[i].value.c_str(), on[i].value.c_str(),
               off[i].value == on[i].value ? "" : "   <-- differs");
}


int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    eva::Device device = getVaiDevice();

    if (!device.supportsPipelineRobustness())
    {
        printf("VK_EXT_pipeline_robustness not supported on this device - "
               "the robust variant cannot be built, nothing to compare.\n");
        return 0;
    }

    const int rounds  = (argc > 1) ? std::max(3, atoi(argv[1])) : 9;
    const int repeats = (argc > 2) ? std::max(1, atoi(argv[2])) : 3;
    constexpr double kBudgetMs = 120.0;

    printf("\n=== robust vs guarded across an M tile boundary, device-local input,\n"
           "    p25 of %d rounds, %d independent duels ===\n", rounds, repeats);
    printf("  If the gap tracks 'wasted' the cost is the out-of-bounds loads;\n"
           "  if it survives at wasted == 0 the cost is the per-access checking.\n\n");
    printf("    M     K       N |  wgM  wgN | wasted  share |  guarded   robust  speedup [ min,   max]   ctl\n");

    // Two (K, N) bases, one small and one where main.cpp measured the worst gap.
    // M walks a tile boundary: exact multiple, one row past it, and one short.
    for (auto kn : { std::pair<uint32_t, uint32_t>{768, 3072},
                     std::pair<uint32_t, uint32_t>{2048, 8192} })
    {
        const uint32_t K = kn.first, N = kn.second;
        const uint32_t wgM = vai::node::ggmlMulmmTileParams(150, N, K, vai::Dtype::Fp16, vai::Dtype::Fp16, device).wgM;

        for (uint32_t M : { wgM, wgM + 1, 2 * wgM, 2 * wgM - 1, 2 * wgM + 1 })
            sweepRow(M, K, N, rounds, repeats, kBudgetMs);
        printf("\n");
    }

    // Same question asked of the compiler instead of the clock.
    statsTest(256, 2048, 8192);

    return 0;
}
