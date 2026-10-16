/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         A/B of the two slab kernels — linear-slab (a workgroup owns
               each slab) vs test-linear-slab2 (a subgroup owns each slab, so
               no shared memory and no barrier).
  Requirements none; test-linear-slab2.comp is built alongside this target and
               reached through test-nodes.h.

  Verify runs first and the benchmark is skipped if it fails. slab is checked
  against NaiveLinearNode, an independently written GEMM; slab2 is checked against
  slab, which that first check has already pinned down.
=============================================================================*/

#include "test-nodes.h"
#include "vai/nodes/linear.h"
#include "tests/test-naive-node.h"
#include "eva-runtime.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace vai;
using namespace vai::node;


// eva buffers are non-owning handles and vai::BufferPool never destroys what it
// holds, so every net that builds its own pool leaks its device memory for the
// rest of the process. One pool shared by every net in this suite turns that
// leak into a cache: a net's buffers come back on destruction and the next net
// reuses them. Without it the suite grows past 6 GB and aborts with
// ERROR_OUT_OF_DEVICE_MEMORY on the 1x2048x128256 lm-head shapes.
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


// Deterministic pseudo-random values in [-1, 1].
double rnd(uint32_t r, uint32_t c, uint32_t salt)
{
    uint32_t h = r * 2654435761u ^ c * 40503u ^ salt * 97u;
    h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
    return double(h % 20001) / 10000.0 - 1.0;
}


/////////////////////////////////////////////////////////////////////////////////////////
// SingleNodeModel - single-node graph: in0 -> node -> out0
/////////////////////////////////////////////////////////////////////////////////////////
template <typename NodeT>
struct SingleNodeModel : public NeuralNet {
    NodeT node;

    SingleNodeModel(
        typename NodeT::Config config,
        NeuralNetConfig netConfig = {})
    : NeuralNet(netConfig)
    , node(config)
    {
        input("in0") - node - output("out0");
    }
};


// Max relative error with the reference RMS as the denominator floor:
// near-zero outputs (catastrophic cancellation) are judged against the
// output scale instead of exploding a tiny absolute difference.
double maxRelErr(const float* ref, const float* got, size_t count)
{
    double sumSq = 0.0;
    for (size_t i = 0; i < count; ++i) sumSq += (double) ref[i] * ref[i];
    const double rms = std::max(1e-6, std::sqrt(sumSq / count));

    double maxRel = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        double denom = std::max(rms, (double) std::fabs(ref[i]));
        maxRel = std::max(maxRel, std::fabs((double) ref[i] - got[i]) / denom);
    }
    return maxRel;
}


// Run one node on the given data and return its output on the host. Scoped by
// the caller so only one net's weights are resident at a time; each net returns
// its buffers to the shared pool for the next one.
template <typename NodeT>
std::vector<float> runOne(typename NodeT::Config config, const NeuralNetConfig& netCfg,
                          const Tensor& input, uint32_t M, uint32_t N, uint32_t K,
                          const std::vector<float>& wData, const std::vector<float>& bData)
{
    SingleNodeModel<NodeT> m(config, netCfg);
    m.node["weight"] = Tensor({N, K}, config.wDtype).setFloats(wData);
    if (config.hasBias)
        m.node["bias"] = Tensor({N}, config.wDtype).setFloats(bData);

    Tensor out = m({{"in0", input}})["out0"];
    const float* p = out.getData();
    return std::vector<float>(p, p + size_t(M) * N);
}


// Verify one config: slab against the NaiveLinearNode reference, then slab2 against
// slab. slab2 only covers M <= 8, so it is skipped on the GEMM fallback case.
bool verifyTest(const char* label, uint32_t M, uint32_t K, uint32_t N,
                Dtype wDtype = Dtype::Fp32, bool hasBias = false,
                Epilogue ep = Epilogue::None)
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32 };

    std::vector<float> wData = makeFloats(N, K, [&](uint32_t n, uint32_t k) { return 0.1 * rnd(n, k, 1); });
    std::vector<float> bData = makeFloats(1, N, [&](uint32_t, uint32_t n) { return rnd(0, n, 3); });
    Tensor input = Tensor({M, K}, Dtype::Fp32)
        .setFloats(makeFloats(M, K, [&](uint32_t m, uint32_t k) { return rnd(m, k, 2); }));

    auto refOut = runOne<NaiveLinearNode>({ .inDim = K, .outDim = N, .hasBias = hasBias,
                                        .epilogue = ep, .wDtype = wDtype },
                                      netCfg, input, M, N, K, wData, bData);
    auto slabOut = runOne<Linear>({ .inDim = K, .outDim = N, .hasBias = hasBias,
                                        .epilogue = ep, .wDtype = wDtype },
                                      netCfg, input, M, N, K, wData, bData);

    // fp16 W — and the M > 8 GEMM fallback, which always stages fp16.
    const double tol = (wDtype == Dtype::Fp16 || M > 8) ? 1e-2 : 1e-4;
    const double slabErr = maxRelErr(refOut.data(), slabOut.data(), (size_t) M * N);
    bool ok = (slabErr <= tol);

    // slab2 reorders the K reduction (one subgroup instead of a whole
    // workgroup), so it is close to slab rather than bit-identical.
    double slab2Err = 0.0;
    if (M <= 8)
    {
        auto slab2Out = runOne<test::Slab2Node>({ .inDim = K, .outDim = N, .hasBias = hasBias,
                                                  .epilogue = ep, .wDtype = wDtype },
                                                netCfg, input, M, N, K, wData, bData);
        slab2Err = maxRelErr(slabOut.data(), slab2Out.data(), (size_t) M * N);
        ok = ok && (slab2Err <= tol);
    }

    char slab2Cell[16];
    if (M <= 8) snprintf(slab2Cell, sizeof(slab2Cell), "%.2e", slab2Err);
    else        snprintf(slab2Cell, sizeof(slab2Cell), "%8s", "n/a");

    printf("  [%s] %-13s M=%u K=%-4u N=%-6u w=%s bias=%d ep=%d  slab=%.2e slab2=%s\n",
        ok ? "PASS" : "FAIL", label, M, K, N,
        wDtype == Dtype::Fp16 ? "f16" : "f32", (int) hasBias, (int) ep, slabErr, slab2Cell);
    return ok;
}


// M switching on one Linear: 1 -> 3 -> 9 (GEMM fallback) -> 1, verifying
// against NaiveLinearNode at every step (pipeSlabM re-request across recordings).
bool sequentialMTest(uint32_t K, uint32_t N)
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32 };

    std::vector<float> wData = makeFloats(N, K, [&](uint32_t n, uint32_t k) { return 0.1 * rnd(n, k, 1); });

    SingleNodeModel<NaiveLinearNode> m1({ .inDim = K, .outDim = N, .hasBias = false }, netCfg);
    SingleNodeModel<Linear> m2({ .inDim = K, .outDim = N, .hasBias = false }, netCfg);
    m1.node["weight"] = Tensor({N, K}, Dtype::Fp32).setFloats(wData);
    m2.node["weight"] = Tensor({N, K}, Dtype::Fp32).setFloats(wData);

    bool ok = true;
    for (uint32_t M : { 1u, 3u, 9u, 1u })
    {
        Tensor input = Tensor({M, K}, Dtype::Fp32)
            .setFloats(makeFloats(M, K, [&](uint32_t m, uint32_t k) { return rnd(m, k, 2 + M); }));

        Tensor refOut = m1({{"in0", input}})["out0"];
        Tensor gotOut = m2({{"in0", input}})["out0"];

        double tol = (M > 8) ? 1e-2 : 1e-4;   // M = 9 runs the fp16-staging GEMM
        double err = maxRelErr(refOut.getData(), gotOut.getData(), (size_t) M * N);
        bool pass = (err <= tol);
        printf("  [%s] %-13s M=%u K=%-4u N=%-6u                    slab=%.2e\n",
            pass ? "PASS" : "FAIL", "sequential-M", M, K, N, err);
        ok = ok && pass;
    }
    return ok;
}


// Time slab against slab2 on one shape (GPU timestamps) and print one row.
void comparisonTest(uint32_t M, uint32_t K, uint32_t N, Dtype wDtype, int iters = 500)
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32 };

    std::vector<float> wData = makeFloats(N, K, [&](uint32_t n, uint32_t k) { return 0.1 * rnd(n, k, 1); });
    Tensor input = Tensor({M, K}, Dtype::Fp32)
        .setFloats(makeFloats(M, K, [&](uint32_t m, uint32_t k) { return rnd(m, k, 2); }));

    auto bench = [&](auto& model, const char* tag) -> double
    {
        model.node.setProfileTag(tag);
        model.node["weight"] = Tensor({N, K}, wDtype).setFloats(wData);

        for (int i = 0; i < 5; ++i)   // warm-up: pipeline build + caches
            model({{"in0", input}})["out0"].getData();
        model.resetProfileRecords();

        for (int i = 0; i < iters; ++i)
            model({{"in0", input}})["out0"].getData();

        return model.getGpuTimeMs(tag);
    };

    // Scoped one at a time so only a single copy of the weights (up to 1 GB at
    // the lm-head shapes) is resident.
    double ms1, ms2;
    {
        SingleNodeModel<Linear> m1({ .inDim = K, .outDim = N, .hasBias = false, .wDtype = wDtype }, netCfg);
        ms1 = bench(m1, "slab");
    }
    {
        SingleNodeModel<test::Slab2Node> m2({ .inDim = K, .outDim = N, .hasBias = false, .wDtype = wDtype }, netCfg);
        ms2 = bench(m2, "slab2");
    }

    // Weight bytes dominate a slab dispatch, so the achieved bandwidth is the
    // number that carries across shapes.
    const double bytes = double(N) * K * bytesPerElement(wDtype);
    printf(" %5u %5u %7u  %s | %9.3f %9.3f | %6.1f %6.1f | %+6.1f%%\n",
        M, K, N, wDtype == Dtype::Fp16 ? "f16" : "f32", ms1, ms2,
        bytes / (ms1 * 1e6), bytes / (ms2 * 1e6),
        (ms1 - ms2) / ms1 * 100.0);
}


int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep prints visible up to an assert/abort
    getVaiDevice();

    printf("\n=== Verify (slab vs NaiveLinearNode, then slab2 vs slab) ===\n");
    int fails = 0;

    // M(spec) variants + GEMM fallback
    fails += !verifyTest("m1",           1, 768, 512);
    fails += !verifyTest("m2",           2, 768, 512);
    fails += !verifyTest("m8",           8, 768, 512);
    fails += !verifyTest("fallback-M9",  9, 768, 512);

    // K variants: unaligned (masked tail), tiny (numIters edge)
    fails += !verifyTest("unaligned-K",  1, 770, 512);
    fails += !verifyTest("tiny-K",       1, 6,   512);

    // fp16 W (rowPair = 2): even N, and odd N (compile-time single-row tail)
    fails += !verifyTest("f16W-evenN",   1, 768, 512, Dtype::Fp16);
    fails += !verifyTest("f16W-oddN",    1, 768, 511, Dtype::Fp16);
    fails += !verifyTest("f16W-unal-odd", 3, 770, 511, Dtype::Fp16);

    // bias / epilogue fusion
    fails += !verifyTest("bias-gelu",    2, 768, 512, Dtype::Fp32, true, Epilogue::GeluTanh);
    fails += !verifyTest("f16W-b-gelu",  2, 768, 512, Dtype::Fp16, true, Epilogue::GeluTanh);

    // lmhead shape (N > 8192 -> SUBGROUP workgroup variant, odd N)
    fails += !verifyTest("lmhead-sg",    1, 768, 50257);
    fails += !verifyTest("f16W-lmhead",  1, 768, 50257, Dtype::Fp16);

    // LARGE workgroup variant (K >= 1024 && N <= 8192 on NVIDIA/Intel)
    fails += !verifyTest("large-wg",     1, 2048, 4096);
    fails += !verifyTest("f16W-lg-odd",  3, 2048, 4095, Dtype::Fp16);

    // llama3-1B lmhead: N/rowPair exceeds the 65535 spec-minimum workgroup
    // count on Intel -> z-split path there (straight X grid on desktop limits)
    fails += !verifyTest("lmhead-1b",    1, 2048, 128256);

    // pipeline re-request across recordings
    fails += !sequentialMTest(768, 512);

    printf(fails == 0 ? "  => ALL PASS\n" : "  => %d case(s) FAILED\n", fails);
    if (fails != 0)
        return 1;   // don't bench a broken kernel

    const int iters = 100;
    printf("\n=== slab vs slab2 (GPU avg ms over %d iters; weight bytes / ms as GB/s) ===\n", iters);
    printf("    M     K       N    w |      slab     slab2 |   GB/s   GB/s |   slab2\n");

    // Decode shapes at fp16, the dtype the samples actually run.
    for (Dtype wt : { Dtype::Fp16, Dtype::Fp32 })
    {
        comparisonTest(1, 768,  768,    wt, iters);   // gpt2 attn q/k/v/out
        comparisonTest(1, 768,  3072,   wt, iters);   // gpt2 ffn fc1
        comparisonTest(1, 3072, 768,    wt, iters);   // gpt2 ffn fc2
        comparisonTest(1, 768,  50257,  wt, iters);   // gpt2 lm head
        comparisonTest(1, 2048, 2048,   wt, iters);   // llama attn q/out
        comparisonTest(1, 2048, 512,    wt, iters);   // llama attn k/v (GQA)
        comparisonTest(1, 2048, 8192,   wt, iters);   // llama ffn gate/up
        comparisonTest(1, 8192, 2048,   wt, iters);   // llama ffn down
        comparisonTest(1, 2048, 128256, wt, iters);   // llama lm head
    }

    // Small-batch sweep (slab M coverage, M = 1..8)
    for (uint32_t M = 1; M <= 8; ++M)
        comparisonTest(M, 2048, 8192, Dtype::Fp16, iters);   // llama ffn gate/up

    return 0;
}
