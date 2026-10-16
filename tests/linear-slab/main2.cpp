/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         test-linear-slab vs test-linear-slab2 at their default knobs,
               one row per decode shape.
  Requirements none; both shaders are built alongside this target and reached
               through test-nodes.h.

  Correctness lives in main.cpp. Nothing here checks a result, so a kernel that
  is fast because it is wrong will look good — run that one first after any
  shader change.
=============================================================================*/

#include "test-nodes.h"
#include "vai/nodes/linear.h"
#include "eva-runtime.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

using namespace vai;
using namespace vai::node;


// eva buffers are non-owning handles and vai::BufferPool never destroys what it
// holds, so every net that builds its own pool leaks its device memory for the
// rest of the process. One pool shared by every net turns that leak into a
// cache: a net's buffers come back on destruction and the next net reuses them.
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


// slab, slab-v5, slab-v2, slab-v0 and slab again on one shape. Returns their
// p25 GPU times in that order; the fifth is a control that differs from the
// first by nothing, so the gap between them is what the call can resolve.
//
// slab and v5 take whatever wgSize the heuristic picks, so they are the same
// geometry on every row. v2 and v0 stride their K by sgSize, so they are pinned
// to wgSize == sgSize; on the rows where the heuristic asks for the wider
// workgroup they are answering a different question — read those columns as
// "what the subgroup-only geometry costs", not as an A/B of the shader.
//
// Two things keep this readable at the few-percent level:
//
// - The slot order rotates every round, so over any multiple of five each
//   entry has run in every slot the same number of times. A fixed order lets
//   whatever the GPU does early in a round settle on whichever entry sits there.
// - p25, not the minimum. Everything that perturbs the measurement only adds
//   time, so the truth is at the low end — but the minimum is one sample and
//   its variance does not fall off with round count. p25 sits on the clean side
//   with a quarter of the samples behind it. Measured control gaps: minimum
//   0.02-3.9%, p25 0.09-1.3%.
//
// Neither reaches past the call. Both entries live in the same nets, on the
// same buffers, for as long as the call lasts, so anything that is fixed for
// that lifetime — where the pool put the buffers, what the driver JIT produced,
// what clock policy the device settled into — shifts the pair together and the
// control reads clean regardless. That term only appears across independent
// calls, which is why the caller repeats this one.
std::array<double, 5> duel(uint32_t M, uint32_t K, uint32_t N, uint32_t wgN, int rounds, int iters)
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32 };

    // One weight tensor for every net: a Tensor is a handle, so the buffer is
    // shared rather than copied, and every entry reads the same bytes.
    Tensor weight = Tensor({N, K}, Dtype::Fp16)
        .setFloats(makeFloats(N, K, [&](uint32_t n, uint32_t k) { return 0.1 * rnd(n, k, 1); }));
    Tensor input = Tensor({M, K}, Dtype::Fp32)
        .setFloats(makeFloats(M, K, [&](uint32_t m2, uint32_t k) { return rnd(m2, k, 2); }));

    std::vector<std::function<double()>> round;
    std::vector<std::function<void()>>   warm;
    std::vector<std::unique_ptr<SingleNodeModel<test::SlabNode>>>   a;
    std::vector<std::unique_ptr<SingleNodeModel<test::SlabV5Node>>> b;
    std::vector<std::unique_ptr<SingleNodeModel<test::SlabV2Node>>> c;
    std::vector<std::unique_ptr<SingleNodeModel<test::SlabV0Node>>> d;

    auto add = [&](auto& store, auto opts)
    {
        using ModelT = typename std::remove_reference_t<decltype(store)>::value_type::element_type;
        opts.inDim = K; opts.outDim = N; opts.wgN = wgN;
        store.push_back(std::make_unique<ModelT>(opts, netCfg));
        auto* m = store.back().get();
        m->node.setProfileTag("k");
        m->node["weight"] = weight;
        warm.push_back([m, &input] { (*m)({{"in0", input}})["out0"].getData(); });
        round.push_back([m, &input, iters] {
            m->resetProfileRecords();
            for (int i = 0; i < iters; ++i) (*m)({{"in0", input}})["out0"].getData();
            return m->getGpuTimeMs("k");
        });
    };

    const uint32_t sg = ggmlSlabSgSize(getVaiDevice());

    add(a, test::SlabNode::Config  {});                 // slab
    add(b, test::SlabV5Node::Config{});                 // slab-v5
    add(c, test::SlabV2Node::Config{ .wgSize = sg });   // slab-v2 — lane strides by sgSize
    add(d, test::SlabV0Node::Config{ .wgSize = sg });   // slab-v0 — same, and no shared array
    add(a, test::SlabNode::Config  {});                 // control

    const size_t n = round.size();
    for (int i = 0; i < 10; ++i)   // warm-up: pipeline build, caches, clock ramp
        for (auto& w : warm) w();

    std::vector<std::vector<double>> samples(n);
    for (int r = 0; r < rounds; ++r)
        for (size_t s = 0; s < n; ++s)
        {
            const size_t i = (s + size_t(r)) % n;
            samples[i].push_back(round[i]());
        }

    std::array<double, 5> p25{};
    for (size_t i = 0; i < n; ++i)
    {
        std::sort(samples[i].begin(), samples[i].end());
        p25[i] = samples[i][samples[i].size() / 4];
    }
    return p25;
}


// What the driver kept from compiling each kernel at the shape under test.
// These pipelines are built only to be asked; the benchmark's own are built
// without the capture flag, which the spec lets a driver compile differently.
// Names and units are the driver's, and the ISA lands next to the executable.
void dumpPipelineStats(uint32_t wgSize, uint32_t sgSize, uint32_t M, uint32_t wgN,
                       const std::vector<std::string>& spvPaths = {})
{
    auto show = [](const char* name, eva::ComputePipeline pipe)
    {
        printf("\n--- %s ---\n", name);
        for (const auto& x : pipe.executables())
            printf("  [exec] \"%s\" sgSize %u | %s\n",
                   x.name.c_str(), x.subgroupSize, x.description.c_str());

        for (const auto& s : pipe.statistics())
            printf("  %-38s %s\n", s.name.c_str(), s.value.c_str());

        for (const auto& ir : pipe.internalRepresentations())
        {
            std::string path = std::string("isa-") + name + "-" + ir.name + ".txt";
            for (char& c : path)
                if (c == ' ' || c == '/' || c == '\\') c = '_';
            if (FILE* f = fopen(path.c_str(), "wb"))
            {
                fwrite(ir.data.data(), 1, ir.data.size(), f);
                fclose(f);
            }
            printf("  [%s] %zu bytes -> %s\n", ir.name.c_str(), ir.data.size(), path.c_str());
        }

        // Leaving these resident would put three extra pipelines alongside the
        // benchmark's, which is exactly the kind of process state the timings
        // turned out to be sensitive to.
        pipe.destroy();
    };

    // constant_id 4 goes to every one of them: a shader that does not declare
    // K_PER_ITER as a specialization constant simply ignores the entry, which
    // is what lets one call site serve every variant.
    auto build = [&](const std::string& spv)
    {
        return getVaiDevice().createComputePipeline({
            .csStage = { eva::SpvBlob::readFrom(spv.c_str()), {
                eva::constant_id<0>(wgSize), eva::constant_id<1>(sgSize),
                eva::constant_id<2>(M),      eva::constant_id<3>(wgN),
                eva::constant_id<4>(4u),
                eva::constant_id<SPEC_ID_TYPE_O>(Dtype::Fp32),
                eva::constant_id<SPEC_ID_TYPE_I>(Dtype::Fp32),
                eva::constant_id<SPEC_ID_ALIGNED>(1u) }},
            .requiredSubgroupSize = sgSize,
            .captureStatistics = true,
        });
    };

    if (spvPaths.empty())
    {
        show("slab",     build(TEST_LINEAR_SLAB_SPV));
        show("slab-v5",  build(TEST_LINEAR_SLAB_V5_SPV));
        show("slab-v2",  build(TEST_LINEAR_SLAB_V2_SPV));
        show("slab-v0",  build(TEST_LINEAR_SLAB_V0_SPV));
        show("slab2",    build(TEST_LINEAR_SLAB2_SPV));
        return;
    }

    for (const auto& p : spvPaths)
    {
        const size_t slash = p.find_last_of("/\\");
        show((slash == std::string::npos ? p : p.substr(slash + 1)).c_str(), build(p));
    }
}


// --stats [spv...]  compile statistics and nothing else. Building and dropping
//                   those pipelines moves the device state enough to widen the
//                   control entry tenfold, so it never shares a process with a
//                   timing run.
// (no argument)     slab / slab-v5 / slab-v2 / slab-v0 across the decode shapes.
int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep prints visible up to an assert/abort
    getVaiDevice();

    const uint32_t sg = ggmlSlabSgSize(getVaiDevice());
    const std::vector<std::string> args(argv + 1, argv + argc);

    if (!args.empty() && args[0] == "--stats")
    {
        dumpPipelineStats(sg, sg, 1, 2, { args.begin() + 1, args.end() });
        return 0;
    }

    struct Shape { uint32_t K, N, wgN; };
    // The decode shapes. The wg column says which geometry the heuristic hands
    // slab and v5; where it equals sg, all four entries run the same geometry
    // and the columns are a straight A/B of the shader text.
    const Shape shapes[] = {
        {  768,    768, 2 },
        { 2048,    512, 2 },
        {  768,   3072, 2 },
        { 3072,    768, 2 },
        { 2048,   2048, 2 },
        { 8192,   2048, 2 },
        { 2048,   8192, 2 },
        {  768,  50257, 2 },   // gpt2 lm_head
        { 2048, 128256, 2 },   // llama3.2-1B lm_head
    };

    const int rounds  = 45;   // a multiple of the five entries, so rotation closes
    const int repeats = 5;    // independent duels — see below

    // Every measurement moves about the same weight bytes, so a round costs the
    // same wall time whatever the shape and no row is starved of samples or
    // left running for minutes.
    const double bytesPerMeasure = 64.0 * 1024 * 1024;

    printf("subgroup size: default %u, range [%u, %u]\n",
           getVaiDevice().subgroupSize(), getVaiDevice().minSubgroupSize(), getVaiDevice().maxSubgroupSize());

    // A duel is built from scratch for each repeat, so the repeats differ in
    // everything one duel holds fixed, and the spread of their verdicts is the
    // error the control cannot see. The verdict is the median of the per-repeat
    // ratios, not a ratio of medians: each ratio is formed inside one duel,
    // where the two entries shared a lifetime, so forming it first cancels
    // whatever that lifetime contributed.
    //
    // Read the spread column first. A verdict narrower than it is noise.
    printf("\n=== default knobs, p25 of %d rounds, %d independent repeats ===\n", rounds, repeats);
    printf("       K        N | wg | sg | iters |   slab ms |     v5 ms |     v2 ms |     v0 ms "
           "|     v5 vs slab     |     v2 vs slab     |     v0 vs slab     |   ctl\n");
    printf("                  |    |    |       |    median |    median |    median |    median "
           "|  median [min, max] |  median [min, max] |  median [min, max] |   max\n");

    for (const Shape& s : shapes)
    {
        const double bytes = double(s.N) * s.K * bytesPerElement(Dtype::Fp16);
        const int iters = std::clamp(int(bytesPerMeasure / bytes), 1, 64);

        std::vector<double> slab, v5, v2, v0, r5, r2, r0, ctl;
        for (int r = 0; r < repeats; ++r)
        {
            auto p = duel(1, s.K, s.N, s.wgN, rounds, iters);
            slab.push_back(p[0]);
            v5  .push_back(p[1]);
            v2  .push_back(p[2]);
            v0  .push_back(p[3]);
            r5  .push_back((p[0] - p[1]) / p[0] * 100.0);
            r2  .push_back((p[0] - p[2]) / p[0] * 100.0);
            r0  .push_back((p[0] - p[3]) / p[0] * 100.0);
            ctl .push_back(std::abs(p[0] - p[4]) / p[0] * 100.0);
        }

        auto median = [](std::vector<double> v)
        {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        auto ratio = [&](std::vector<double>& r)
        {
            printf("| %+5.1f%% [%+5.1f,%+5.1f] ",
                   median(r), *std::min_element(r.begin(), r.end()),
                              *std::max_element(r.begin(), r.end()));
        };

        printf("  %6u  %7u | %2u | %2u | %5d | %9.4f | %9.4f | %9.4f | %9.4f ",
            s.K, s.N, ggmlSlabWgSize(s.K, s.N, sg, getVaiDevice()), sg, iters,
            median(slab), median(v5), median(v2), median(v0));
        ratio(r5);
        ratio(r2);
        ratio(r0);
        printf("| %5.1f%%\n", *std::max_element(ctl.begin(), ctl.end()));
    }

    return 0;
}
