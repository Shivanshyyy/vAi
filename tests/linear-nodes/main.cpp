#include "vai/nodes/linear.h"
#include "tests/test-naive-node.h"
#include "eva-runtime.h"
#include <cstdio>
#include <string>
#include <vector>

using namespace vai;
using namespace vai::node;


// Print fp32 values as [rows, cols], one row per line
void printMatrix(const char* label, const float* data, uint32_t rows, uint32_t cols)
{
    printf("%s [%u, %u]:\n", label, rows, cols);
    for (uint32_t r = 0; r < rows; ++r)
    {
        printf("  ");
        for (uint32_t c = 0; c < cols; ++c)
            printf("% 8.1f", data[r * cols + c]);
        printf("\n");
    }
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


void testNode()
{
    printf("\n=== NaiveLinearNode ===\n");
    const uint32_t M = 2, K = 8, N = 4;
    Dtype wDtype = Dtype::Fp16;

    SingleNodeModel<NaiveLinearNode> model({
        .inDim = K,
        .outDim = N,
        .hasBias = false,
        .wDtype = wDtype
    });

    // in0 [2, 8]:
    //    0  1  2  3  4  5  6  7
    //   10 11 12 13 14 15 16 17
    Tensor input = Tensor({M, K}, Dtype::Fp32)
        .setFloats(makeFloats(M, K, [](uint32_t m, uint32_t k) { return 10 * m + k; }));

    // weight [4, 8]:
    //    1  1  1  1  1  1  1  1
    //    2  2  2  2  2  2  2  2
    //    3  3  3  3  3  3  3  3
    //    4  4  4  4  4  4  4  4
    model.node["weight"] = Tensor({N, K}, wDtype)
        .setFloats(makeFloats(N, K, [](uint32_t n, uint32_t k) { return n + 1; }));

    // Tensor output = model({{"in0", input}})["out0"];

    // Expected out0 (row sums 28, 108):
    //    28.0    56.0    84.0   112.0
    //   108.0   216.0   324.0   432.0
    // printMatrix("out0", output.getData(), M, N);
}


// Time NaiveLinearNode vs Linear on one shape (GPU timestamps) and print a
// single summary row.
void comparisonTest(uint32_t M, uint32_t K, uint32_t N, int iters = 500)
{
    NeuralNetConfig netCfg = { .wDefaultDtype = Dtype::Fp32 };

    // Deterministic pseudo-random data in loosely GPT-2-ish ranges.
    auto rnd = [](uint32_t r, uint32_t c, uint32_t salt) {
        uint32_t h = r * 2654435761u ^ c * 40503u ^ salt * 97u;
        h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
        return double(h % 20001) / 10000.0 - 1.0;   // [-1, 1]
    };
    std::vector<float> wData = makeFloats(N, K, [&](uint32_t n, uint32_t k) { return 0.1 * rnd(n, k, 1); });
    Tensor input = Tensor({M, K}, Dtype::Fp32)
        .setFloats(makeFloats(M, K, [&](uint32_t m, uint32_t k) { return rnd(m, k, 2); }));

    auto bench = [&](auto& model, const char* tag) -> double
    {
        model.node.setProfileTag(tag);
        model.node["weight"] = Tensor({N, K}, Dtype::Fp32).setFloats(wData);

        for (int i = 0; i < 5; ++i)   // warm-up: pipeline build + caches
            model({{"in0", input}})["out0"].getData();
        model.resetProfileRecords();

        for (int i = 0; i < iters; ++i)
            model({{"in0", input}})["out0"].getData();  

        return model.getGpuTimeMs(tag);
    };

    SingleNodeModel<NaiveLinearNode> m1({ .inDim = K, .outDim = N, .hasBias = false }, netCfg);
    double ms1 = bench(m1, "Linear(native)");

    SingleNodeModel<Linear> m2({ .inDim = K, .outDim = N, .hasBias = false, .accDtype = Dtype::Fp16 }, netCfg);
    double ms2 = bench(m2, "Linear(tiled)");

    SingleNodeModel<Linear> m3({ .inDim = K, .outDim = N, .hasBias = false, .splitSize = 256, .accDtype = Dtype::Fp16 }, netCfg);
    double ms3 = bench(m3, "Linear(tiled+split)");

    printf(" %5u %5u %7u | %9.3f %9.3f %9.3f\n",
        M, K, N, ms1, ms2, ms3);
}


int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep prints visible up to an assert/abort
    getVaiDevice();

    // testNode();

    const int iters = 100;
    printf("\n=== (fp32 I/W/O, fp16 acc, GPU avg ms over %d iters) ===\n", iters);
    printf("    M     K       N |   Linear(native)    Linear(tiled)   Linear(tiled+split)\n");

    comparisonTest(1,   768, 2048, iters);   // lm-head, decode
    comparisonTest(150, 768, 2048, iters);   // lm-head, prefill
    comparisonTest(1,   768, 512,  iters);
    comparisonTest(150, 768, 512,  iters);
    comparisonTest(150, 768, 3072, iters);   // ffn-ish
    // comparisonTest(2048, 2048, 2048, iters);   // ffn-ish

    return 0;
}
