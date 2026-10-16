#ifndef VAI_TESTS_ROBUST_BENCH_COMMON_H
#define VAI_TESTS_ROBUST_BENCH_COMMON_H

#include "test-robust-linear-node.h"
#include "vai/nodes/activation.h"
#include "eva-runtime.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace vai;
using namespace vai::node;


// One pool shared by every net in this suite (see models/tests/linear-cm):
// buffers come back on net destruction and the next net reuses them.
std::shared_ptr<BufferPool> sharedPool()
{
    static std::shared_ptr<BufferPool> pool =
        std::make_shared<BufferPool>(BufferPoolConfig{getVaiDevice(), 0});
    return pool;
}


// Build [rows, cols] fp32 data from a formula: data[r, c] = f(r, c)
template <typename F>
std::vector<float> makeFloats(uint32_t rows, uint32_t cols, F f)
{
    std::vector<float> v(size_t(rows) * cols);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < cols; ++c)
            v[size_t(r) * cols + c] = float(f(r, c));
    return v;
}

// Deterministic pseudo-random data in loosely GPT-2-ish ranges.
double rnd(uint32_t r, uint32_t c, uint32_t salt)
{
    uint32_t h = r * 2654435761u ^ c * 40503u ^ salt * 97u;
    h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
    return double(h % 20001) / 10000.0 - 1.0;   // [-1, 1]
}


/////////////////////////////////////////////////////////////////////////////////////////
// Models - pinned (in0 -> linear) and device-local (in0 -> identity -> linear)
/////////////////////////////////////////////////////////////////////////////////////////
struct PinnedModel : public NeuralNet {
    test::LinearNode lin;

    PinnedModel(test::LinearNode::Config config, NeuralNetConfig netConfig)
    : NeuralNet(netConfig)
    , lin(config)
    {
        input("in0") - lin - output("out0");
    }
};

struct DeviceLocalModel : public NeuralNet {
    Activation act;
    test::LinearNode lin;

    DeviceLocalModel(test::LinearNode::Config config, NeuralNetConfig netConfig)
    : NeuralNet(netConfig)
    , act({ .epilogue = Epilogue::None })
    , lin(config)
    {
        input("in0") - act - lin - output("out0");
    }
};


// Verify robust on/off produce the same output on one shape. Robust OOB tail
// reads may only feed lanes the stores mask off, so the outputs must match
// exactly; any drift means a guard was folded that actually mattered.
bool verifyTest(const char* label, uint32_t M, uint32_t K, uint32_t N)
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32 };

    std::vector<float> wData = makeFloats(N, K, [&](uint32_t n, uint32_t k) { return 0.1 * rnd(n, k, 1); });
    Tensor input = Tensor({M, K}, Dtype::Fp32)
        .setFloats(makeFloats(M, K, [&](uint32_t m, uint32_t k) { return rnd(m, k, 2); }));

    // One net at a time so only a single copy of the weights is resident.
    // The guarded variant runs twice: off2 vs off separates nondeterminism
    // (a difference with robustness out of the picture) from a robust leak.
    std::vector<float> off, off2, on;
    for (int pass = 0; pass < 3; ++pass)
    {
        const bool robust = (pass == 2);
        PinnedModel m({ .inDim = K, .outDim = N, .robust = robust, .accDtype = Dtype::Fp16 }, netCfg);
        m.lin["weight"] = Tensor({N, K}, Dtype::Fp32).setFloats(wData);

        // Keep the result tensor alive while its host copy is read:
        // getData()'s pointer lives inside the tensor, so reading it off the
        // dict temporary is a use-after-free once the statement ends.
        Tensor out = m({{"in0", input}})["out0"];
        const float* p = out.getData();
        (pass == 0 ? off : pass == 1 ? off2 : on).assign(p, p + size_t(M) * N);
    }

    size_t nondet = 0;
    for (size_t i = 0; i < off.size(); ++i)
        if (off[i] != off2[i]) ++nondet;
    if (nondet != 0)
        printf("         NOTE: two identical robust=off runs disagree at %zu elems - "
               "nondeterminism unrelated to the robust option\n", nondet);

    double maxAbs = 0.0;
    size_t diffs = 0;
    for (size_t i = 0; i < off.size(); ++i)
        if (off[i] != on[i]) ++diffs, maxAbs = std::max(maxAbs, std::fabs(double(off[i]) - on[i]));

    bool ok = (diffs == 0);
    printf("  [%s] %-12s M=%-3u K=%-4u N=%-6u  maxAbsDiff=%.2e (%zu/%zu elems)\n",
        ok ? "PASS" : "FAIL", label, M, K, N, maxAbs, diffs, off.size());

    // Where the leak lands, for the first few mismatches.
    int shown = 0;
    for (size_t i = 0; i < off.size() && shown < 5; ++i)
        if (off[i] != on[i])
            printf("         diff at m=%zu n=%zu: guarded=%g robust=%g\n",
                i / N, i % N, off[i], on[i]), ++shown;
    return ok;
}


// One net, kept alive across every sample it feeds: construction, the weight
// upload and the warm-up all land outside the timed region, and the buffers stay
// where the pool first put them. sample() returns the GPU timestamp average of
// `iters` forwards.
template <typename Model>
struct Harness
{
    Model m;
    Tensor input;
    int iters = 3;

    Harness(test::LinearNode::Config opts, NeuralNetConfig netCfg,
            const std::vector<float>& wData, Tensor in, double budgetMs)
    : m(opts, netCfg), input(in)
    {
        m.lin.setProfileTag("lin");
        m.lin["weight"] = Tensor({opts.outDim, opts.inDim}, Dtype::Fp32).setFloats(wData);

        for (int i = 0; i < 3; ++i)   // pipeline build, caches, clock ramp
            m({{"in0", input}})["out0"].getData();

        // One timed probe sizes the sample: enough forwards to amortize the
        // submit, few enough that a round stays short and rotation stays cheap.
        auto t0 = std::chrono::steady_clock::now();
        m({{"in0", input}})["out0"].getData();
        const double probeMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        iters = std::clamp(int(budgetMs / std::max(0.01, probeMs)), 3, 200);
    }

    double sample()
    {
        m.resetProfileRecords();
        for (int i = 0; i < iters; ++i)
            m({{"in0", input}})["out0"].getData();
        return m.getGpuTimeMs("lin");
    }
};


// One duel on one placement: guarded, robust, guarded again. The third entry
// differs from the first by nothing, so the gap between them is what the call
// can resolve — anything the pair cannot separate shows up there first.
//
// Two things borrowed from models/tests/linear-slab, for the same reasons:
//  - the slot order rotates every round, so over any multiple of three each
//    entry has run in every slot the same number of times. A fixed order lets
//    whatever the GPU does early in a round settle on whichever entry sits there.
//  - p25, not the minimum. Everything that perturbs a sample only adds time, so
//    the truth is at the low end, but the minimum is one sample and its variance
//    does not fall off with round count.
//
// Neither reaches past the call: all three entries live in nets built together
// and timed together, so whatever is fixed for that lifetime shifts them as one.
// That term only appears across independent duels, which is why the caller
// repeats this one from scratch.
template <typename Model>
std::array<double, 3> duel(uint32_t M, uint32_t K, uint32_t N,
                           const std::vector<float>& wData, Tensor input,
                           int rounds, double budgetMs)
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32 };

    auto opts = [&](bool robust) {
        return test::LinearNode::Config{ .inDim = K, .outDim = N,
                                          .robust = robust, .accDtype = Dtype::Fp16 };
    };

    Harness<Model> entry[3] = {
        { opts(false), netCfg, wData, input, budgetMs },
        { opts(true),  netCfg, wData, input, budgetMs },
        { opts(false), netCfg, wData, input, budgetMs },
    };

    std::vector<double> samples[3];
    for (int r = 0; r < rounds; ++r)
        for (int s = 0; s < 3; ++s)
        {
            const int i = (s + r) % 3;
            samples[i].push_back(entry[i].sample());
        }

    std::array<double, 3> p25{};
    for (int i = 0; i < 3; ++i)
    {
        std::sort(samples[i].begin(), samples[i].end());
        p25[i] = samples[i][samples[i].size() / 4];
    }
    return p25;
}


#endif // VAI_TESTS_ROBUST_BENCH_COMMON_H
