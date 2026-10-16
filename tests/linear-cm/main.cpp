// Coopmat GEMM verification: Linear's M > 8 path (linear-cm on
// devices that pass the coopmat gates, linear-tile otherwise) against the
// independent NaiveLinearNode kernel on identical inputs / weights / bias.
// Non-zero exit on any failure.
//
// The gate-forced variants (non-square coopmat shape, shrunken shared-memory
// limit, scalar fallback on a coopmat device) need temporary code changes and
// are not part of this suite: pin the shape in ggmlCoopmatShape / shrink the
// budget in coopmatTileUsable / early-return from ggmlCoopmatShape, then
// re-run — every case below must still pass.
#include "vai/nodes/activation.h"
#include "vai/nodes/linear.h"
#include "tests/test-naive-node.h"
#include "eva-runtime.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vai;
using namespace vai::node;


// eva buffers are non-owning handles and vai::BufferPool never destroys what it
// holds, so every net that builds its own pool leaks its device memory for the
// rest of the process. One pool shared by every net in this suite turns that
// leak into a cache: a net's buffers come back on destruction and the next net
// reuses them.
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


// fp16 (IEEE binary16) to float — Tensor::getData is a raw typed view and
// does not convert, so fp16 outputs are widened here for comparison.
float halfToFloat(uint16_t h)
{
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;

    uint32_t bits;
    if (exp == 31)          bits = sign | 0x7F800000u | (man << 13);   // inf / nan
    else if (exp != 0)      bits = sign | ((exp + 112u) << 23) | (man << 13);
    else if (man == 0)      bits = sign;                               // +-0
    else
    {
        exp = 113u;   // subnormal: renormalize
        while ((man & 0x400u) == 0)
        {
            man <<= 1;
            exp--;
        }
        bits = sign | (exp << 23) | ((man & 0x3FFu) << 13);
    }

    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}


// A net's output copied to host floats.
std::vector<float> readFloats(Tensor out, size_t count, Dtype dtype)
{
    std::vector<float> v(count);
    if (dtype == Dtype::Fp16)
    {
        const uint16_t* p = out.getData<uint16_t>();
        for (size_t i = 0; i < count; ++i)
            v[i] = halfToFloat(p[i]);
    }
    else
    {
        const float* p = out.getData();
        std::copy(p, p + count, v.begin());
    }
    return v;
}


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


// The axes of one verify case beyond the shape (defaults = plain fp32 GEMM).
struct CaseOpts {
    Dtype wDtype   = Dtype::Fp32;
    Dtype ioDtype  = Dtype::Fp32;   // I and O together
    Dtype accDtype = Dtype::Fp32;
    bool hasBias   = false;
    Epilogue ep    = Epilogue::None;

    uint32_t splitSize = 0;
};


// Verify Linear vs NaiveLinearNode on one config (GPU outputs compared) and
// print a single summary row.
bool verifyTest(const char* label, uint32_t M, uint32_t K, uint32_t N, CaseOpts c = {})
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32, .oDefaultDtype = c.ioDtype };

    // Deterministic pseudo-random data in loosely GPT-2-ish ranges.
    auto rnd = [](uint32_t r, uint32_t c, uint32_t salt) {
        uint32_t h = r * 2654435761u ^ c * 40503u ^ salt * 97u;
        h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
        return double(h % 20001) / 10000.0 - 1.0;   // [-1, 1]
    };
    std::vector<float> wData = makeFloats(N, K, [&](uint32_t n, uint32_t k) { return 0.1 * rnd(n, k, 1); });
    std::vector<float> bData = makeFloats(1, N, [&](uint32_t, uint32_t n) { return rnd(0, n, 3); });
    Tensor input = Tensor({M, K}, c.ioDtype)
        .setFloats(makeFloats(M, K, [&](uint32_t m, uint32_t k) { return rnd(m, k, 2); }));

    // One net at a time, each result copied to the host before its net dies:
    // the reference net's buffers are back in the shared pool before the
    // Linear net asks for its own.
    std::vector<float> refOut, gotOut;
    {
        SingleNodeModel<NaiveLinearNode> m1({ .inDim = K, .outDim = N, .hasBias = c.hasBias,
                                         .epilogue = c.ep, .wDtype = c.wDtype,
                                         .oDtype = c.ioDtype }, netCfg);
        m1.node["weight"] = Tensor({N, K}, c.wDtype).setFloats(wData);
        if (c.hasBias)
            m1.node["bias"] = Tensor({N}, c.wDtype).setFloats(bData);

        refOut = readFloats(m1({{"in0", input}})["out0"], size_t(M) * N, c.ioDtype);
    }
    {
        SingleNodeModel<Linear> m2({ .inDim = K, .outDim = N, .hasBias = c.hasBias,
                                         .epilogue = c.ep, .wDtype = c.wDtype,
                                         .oDtype = c.ioDtype,
                                         .splitSize = c.splitSize, .accDtype = c.accDtype }, netCfg);
        m2.node["weight"] = Tensor({N, K}, c.wDtype).setFloats(wData);
        if (c.hasBias)
            m2.node["bias"] = Tensor({N}, c.wDtype).setFloats(bData);

        gotOut = readFloats(m2({{"in0", input}})["out0"], size_t(M) * N, c.ioDtype);
    }

    // The M > 8 GEMM always stages fp16 -> 1e-2. fp16 accumulation loses more
    // against the fp32-accumulating reference — the scalar kernel's plain fp16
    // adds sit near 3e-2 at K = 768 (coopmat accumulators do better), and the
    // suite must also pass where the scalar kernel is the fallback.
    double tol = (c.accDtype == Dtype::Fp16) ? 5e-2 : 1e-2;
    double err = maxRelErr(refOut.data(), gotOut.data(), (size_t) M * N);
    bool ok = (err <= tol);
    printf("  [%s] %-13s M=%-3u K=%-4u N=%-6u w=%s io=%s acc=%s bias=%d ep=%d split=%-3u  maxRel=%.2e\n",
        ok ? "PASS" : "FAIL", label, M, K, N,
        c.wDtype == Dtype::Fp16 ? "f16" : "f32",
        c.ioDtype == Dtype::Fp16 ? "f16" : "f32",
        c.accDtype == Dtype::Fp16 ? "f16" : "f32",
        (int) c.hasBias, (int) c.ep, c.splitSize, err);
    return ok;
}


// M switching on one Linear: 16 -> 256 -> 16 (tile / pipeline
// re-request across tiers) -> 8 (tile -> slab switch), verifying against
// NaiveLinearNode at every step.
bool sequentialMTest(uint32_t K, uint32_t N)
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32 };

    auto rnd = [](uint32_t r, uint32_t c, uint32_t salt) {
        uint32_t h = r * 2654435761u ^ c * 40503u ^ salt * 97u;
        h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
        return double(h % 20001) / 10000.0 - 1.0;   // [-1, 1]
    };
    std::vector<float> wData = makeFloats(N, K, [&](uint32_t n, uint32_t k) { return 0.1 * rnd(n, k, 1); });

    SingleNodeModel<NaiveLinearNode> m1({ .inDim = K, .outDim = N, .hasBias = false }, netCfg);
    SingleNodeModel<Linear> m2({ .inDim = K, .outDim = N, .hasBias = false }, netCfg);
    m1.node["weight"] = Tensor({N, K}, Dtype::Fp32).setFloats(wData);
    m2.node["weight"] = Tensor({N, K}, Dtype::Fp32).setFloats(wData);

    bool ok = true;
    for (uint32_t M : { 16u, 256u, 16u, 8u })
    {
        Tensor input = Tensor({M, K}, Dtype::Fp32)
            .setFloats(makeFloats(M, K, [&](uint32_t m, uint32_t k) { return rnd(m, k, 2 + M); }));

        Tensor refOut = m1({{"in0", input}})["out0"];
        Tensor gotOut = m2({{"in0", input}})["out0"];

        double tol = (M > 8) ? 1e-2 : 1e-4;   // M = 8 runs the slab path
        double err = maxRelErr(refOut.getData(), gotOut.getData(), (size_t) M * N);
        bool pass = (err <= tol);
        printf("  [%s] %-13s M=%-3u K=%-4u N=%-6u                                            maxRel=%.2e\n",
            pass ? "PASS" : "FAIL", "sequential-M", M, K, N, err);
        ok = ok && pass;
    }
    return ok;
}


// Time Linear fed by a device-local intermediate (an identity
// Activation between the graph input and Linear). Graph inputs are
// pinned (host-visible), and reading them across PCIe dominates the GEMM's
// staging loads — this is the in-network condition, where Linear reads a
// device-local activation.
void deviceInputTest(uint32_t M, uint32_t K, uint32_t N, int iters = 500)
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32 };

    auto rnd = [](uint32_t r, uint32_t c, uint32_t salt) {
        uint32_t h = r * 2654435761u ^ c * 40503u ^ salt * 97u;
        h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
        return double(h % 20001) / 10000.0 - 1.0;   // [-1, 1]
    };
    std::vector<float> wData = makeFloats(N, K, [&](uint32_t n, uint32_t k) { return 0.1 * rnd(n, k, 1); });
    Tensor input = Tensor({M, K}, Dtype::Fp32)
        .setFloats(makeFloats(M, K, [&](uint32_t m, uint32_t k) { return rnd(m, k, 2); }));

    struct Model : public NeuralNet {
        Activation act;
        Linear lin;

        Model(Linear::Config config, NeuralNetConfig netConfig)
        : NeuralNet(netConfig)
        , act({ .epilogue = Epilogue::None })
        , lin(config)
        {
            input("in0") - act - lin - output("out0");
        }
    };

    Model m({ .inDim = K, .outDim = N, .hasBias = false, .accDtype = Dtype::Fp16 }, netCfg);
    m.lin.setProfileTag("Linear(tiled)");
    m.lin["weight"] = Tensor({N, K}, Dtype::Fp32).setFloats(std::move(wData));

    for (int i = 0; i < 5; ++i)
        m({{"in0", input}})["out0"].getData();
    m.resetProfileRecords();

    for (int i = 0; i < iters; ++i)
        m({{"in0", input}})["out0"].getData();

    printf(" %5u %5u %7u | %9.3f  (device-local I)\n", M, K, N, m.getGpuTimeMs("Linear(tiled)"));
}


// Time NaiveLinearNode vs Linear on one shape (GPU timestamps) and print a
// single summary row.
void comparisonTest(uint32_t M, uint32_t K, uint32_t N, int iters = 500)
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32 };

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

    // Scoped one at a time so only a single copy of the weights is resident;
    // each net returns its buffers to the shared pool for the next one.
    double ms1, ms2, ms3;
    {
        SingleNodeModel<NaiveLinearNode> m1({ .inDim = K, .outDim = N, .hasBias = false }, netCfg);
        ms1 = bench(m1, "Linear(native)");
    }
    {
        SingleNodeModel<Linear> m2({ .inDim = K, .outDim = N, .hasBias = false, .accDtype = Dtype::Fp16 }, netCfg);
        ms2 = bench(m2, "Linear(tiled)");
    }
    {
        SingleNodeModel<Linear> m3({ .inDim = K, .outDim = N, .hasBias = false, .splitSize = 256, .accDtype = Dtype::Fp16 }, netCfg);
        ms3 = bench(m3, "Linear(tiled+split)");
    }

    printf(" %5u %5u %7u | %15.3f %15.3f %19.3f\n",
        M, K, N, ms1, ms2, ms3);
}


int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep prints visible up to an assert/abort
    getVaiDevice();

    printf("\n=== Coopmat GEMM verify (Linear M>8 path vs NaiveLinearNode reference) ===\n");
    int fails = 0;

    // Tile tiers (S / M / L warptiles)
    fails += !verifyTest("s-tile",       16,  768,  32);
    fails += !verifyTest("m-tile",       48,  768,  64);
    fails += !verifyTest("l-tile",       256, 768,  2048);

    // M / N tails: accumulator tiles crossing an edge take the masked
    // staged-store path
    fails += !verifyTest("m-tail",       130, 768,  2048);
    fails += !verifyTest("n-tail",       256, 768,  2052);
    fails += !verifyTest("mn-tail",      130, 768,  2050);

    // O-store stride misaligned (fp32: N % 4 != 0): whole in-bounds tiles
    // fall back from direct coopMatStore to the staged path
    fails += !verifyTest("ostore-unal",  256, 768,  2046);

    // Loader variants: K % wgK != 0 (masked scalar loads), K below one k-block
    fails += !verifyTest("unaligned-K",  256, 770,  2048);
    fails += !verifyTest("k-lt-wgK",     256, 16,   2048);

    // dtype axes: fp16 W, fp16 I/O (direct fp16 coopMatStore at N % 8 == 0,
    // staged when misaligned), fp16 accumulation (clamp; promoted to fp32
    // where the device shape lacks fp16 acc)
    fails += !verifyTest("f16-w",        256, 768,  2048, {.wDtype = Dtype::Fp16});
    fails += !verifyTest("f16-io",       256, 768,  2048, {.ioDtype = Dtype::Fp16});
    fails += !verifyTest("f16-io-unal",  256, 768,  2052, {.ioDtype = Dtype::Fp16});
    fails += !verifyTest("acc-f16",      256, 768,  2048, {.accDtype = Dtype::Fp16});

    // bias / epilogue ride the reduce pass
    fails += !verifyTest("bias-gelu",    256, 768,  2048, {.hasBias = true, .ep = Epilogue::GeluTanh});

    // split-K: z-dispatch partials + reduce; the tail variant misaligns the
    // split boundary against wgK
    fails += !verifyTest("split-k",      64,  8192, 512, {.splitSize = 256});
    fails += !verifyTest("split-k-tail", 64,  8190, 512, {.splitSize = 256});

    // tile / pipeline re-request across recordings, and the slab boundary
    fails += !sequentialMTest(768, 2048);

    printf(fails == 0 ? "  => ALL PASS\n" : "  => %d case(s) FAILED\n", fails);
    if (fails != 0)
        return 1;   // don't bench a broken kernel

    const int iters = 500;
    printf("\n=== (fp32 I/W/O, fp16 acc tile GEMM, GPU avg ms over %d iters) ===\n", iters);
    printf("    M     K       N |   Linear(native)   Linear(tiled)   Linear(tiled+split)\n");

    // GPT-2 124M prefill shapes (embed 768)
    comparisonTest(150, 768,  768,   iters);   // attn q/k/v/out
    comparisonTest(150, 768,  3072,  iters);   // ffn fc1
    comparisonTest(150, 3072, 768,   iters);   // ffn fc2
    comparisonTest(150, 768,  50257, iters);   // lm head

    // Llama3.2-1B prefill shapes (embed 2048)
    comparisonTest(150, 2048, 2048,  iters);   // attn q/out
    comparisonTest(150, 2048, 8192,  iters);   // ffn gate/up
    comparisonTest(150, 8192, 2048,  iters);   // ffn down

    // Batch sweep across the tile tiers (gpt2 ffn fc1)
    for (uint32_t M : { 16u, 32u, 64u, 128u, 256u })
        comparisonTest(M, 768, 3072, iters);

    // The rows above feed Linear the pinned (host-visible) graph input,
    // and that PCIe read path dominates staging-bound GEMM timings. These
    // repeat key shapes with a device-local input — the in-network condition.
    printf("\n=== Linear with a device-local input (identity node in front) ===\n");
    printf("    M     K       N |\n");
    deviceInputTest(150, 768,  3072,  iters);
    deviceInputTest(150, 2048, 2048,  iters);
    deviceInputTest(150, 2048, 8192,  iters);
    deviceInputTest(150, 8192, 2048,  iters);

    return 0;
}
