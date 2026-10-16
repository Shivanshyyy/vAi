// vAi side of test-vai-vs-native: the same ggml-style-matmul kernel, but
// reached through the framework — test::LinearNode in a one-node NeuralNet,
// with eva owning the device, buffers, descriptor sets and pipeline cache.
//
// Timings come from the net's GPU profiler, which brackets the node's
// dispatch with timestamps exactly like the raw path does.

#include "comparison-native.h"

#include "test-linear-node.h"

#include <vector>

using namespace vai;
using namespace vai::node;

namespace {

// One-node graph: in0 -> test::LinearNode -> out0
struct MatmulModel : public NeuralNet {
    test::LinearNode node;

    MatmulModel(test::LinearNode::Config config, NeuralNetConfig netConfig)
    : NeuralNet(netConfig)
    , node(config)
    {
        input("in0") - node - output("out0");
    }
};

// The warptile class Linear3Node will land on, mirrored from
// ggmlMulmmTileParams' S/M/L thresholds so the report can show it.
char tileClassOf(uint32_t M, uint32_t N)
{
    if (M <= 32 || N <= 32) return 's';
    if (M <= 64 || N <= 64) return 'm';
    return 'l';
}

} // namespace


void vaiInit()
{
    getVaiDevice();   // eva creates its device (and prints its banner) here
}

BenchResult vaiRun(const BenchCase& c, int warmup, int reps)
{
    const uint32_t M = c.M, N = c.N, K = c.K;
    const char* tag = "ggml-style-matmul";

    NeuralNetConfig netCfg = { .wDefaultDtype = Dtype::Fp32 };

    MatmulModel model({ .inDim = K, .outDim = N }, netCfg);
    model.node.setProfileTag(tag);

    std::vector<float> wData((size_t) N * K), iData((size_t) M * K);
    for (uint32_t n = 0; n < N; n++)
        for (uint32_t k = 0; k < K; k++) wData[(size_t) n * K + k] = benchInitW(n, k);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t k = 0; k < K; k++) iData[(size_t) m * K + k] = benchInitI(m, k);

    model.node["weight"] = Tensor({N, K}, Dtype::Fp32).setFloats(std::move(wData));
    Tensor input = Tensor({M, K}, Dtype::Fp32).setFloats(std::move(iData));

    // Warm-up covers pipeline creation and the buffer-pool first touch, then
    // the running totals are cleared so only the measured reps count.
    for (int i = 0; i < warmup; i++)
        model({{"in0", input}})["out0"].getData();
    model.resetProfileRecords();

    for (int r = 0; r < reps; r++)
        model({{"in0", input}})["out0"].getData();

    const Linear::PipeParamsTile t = ggmlMulmmTileParams(M, N, K, vai::Dtype::Fp16, vai::Dtype::Fp16, getVaiDevice());
    return { model.getGpuTimeMs(tag), tileClassOf(M, N), CEIL_DIV(N, t.wgN), CEIL_DIV(M, t.wgM) };
}
