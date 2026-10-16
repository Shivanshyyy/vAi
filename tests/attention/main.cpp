// Attention verification: the path prepare() picks per (shape,
// device) — coopmat FA, scalar FA, GQA-packed decode, split-K, naive fallback
// — against a host reference on identical inputs, walking prefill -> decode
// sequences so the KV cache and the tuning re-requests are exercised too.
// Non-zero exit on any failure.
//
// Path selection is automatic, so which FA flavor runs depends on the device;
// every row prints the selected path(s). The gate-forced variants (scalar on
// a coopmat device, forced split counts, ...) need temporary code changes in
// ggmlFATuningParams / ggmlFASplitKCount and are not part of this suite — force
// the gate, then re-run: every case below must still pass.
//
// Input data lives on an fp16-exact grid (multiples of 1/512 in [-1, 1]), so
// the fp32 host mirror of the fp16 K/V cache — and fp16 I/O tensors — match
// the GPU values bit-for-bit; the compared error is accumulation order and
// f16-accumulation paths only, never input rounding.
#include "vai/nodes/attention.h"
#include "vai/groups/transformer.h"
#include "eva-runtime.h"
#include <algorithm>
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


// Deterministic pseudo-random data on the fp16-exact grid (see file header).
std::vector<float> makeData(size_t count, uint32_t salt)
{
    std::vector<float> v(count);
    for (size_t i = 0; i < count; ++i)
    {
        uint32_t h = uint32_t(i) * 2654435761u ^ salt * 97u;
        h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
        v[i] = float(h % 1025u) / 512.0f - 1.0f;   // {-1, -511/512, ..., 1}
    }
    return v;
}


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
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i]))
            return INFINITY;
        double denom = std::max(rms, (double) std::fabs(ref[i]));
        maxRel = std::max(maxRel, std::fabs((double) ref[i] - got[i]) / denom);
    }
    return maxRel;
}


/////////////////////////////////////////////////////////////////////////////////////////
// AttnModel - single-attention-node graph: q/k/v -> Attention -> out0.
// Owns the KV-cache counter the node (and the model driving it) shares —
// same wiring as the gpt2 / llama samples.
/////////////////////////////////////////////////////////////////////////////////////////
struct AttnModel : public NeuralNet {
    Attention node;
    eva::Buffer counter;
    eva::Buffer causalMask;

    uint32_t tokensSoFar = 0;   // host-side running count consumed by the KV cache
    int64_t fixedCached = -1;   // >= 0: pin the counter instead (steady-state bench)

    AttnModel(Attention::Config config, NeuralNetConfig netConfig = {},
              uint32_t causalBits = 1, vai::CausalBlockEnd blockEnd = {})
    : NeuralNet(netConfig)
    , node(config)
    {
        causalMask = vai::createCausalMask(device(), causalBits, blockEnd);
        node.setCausalMask(causalMask);

        input("q") - "Q"/node - output("out0");
        input("k") - "K"/node;
        input("v") - "V"/node;

        // Cacheless nodes bind the counter null and read 0 from it.
        if (config.kvCacheSize == 0)
            return;

        counter = vai::createKvCacheCounter(device());
        node.setKvCacheCounter(counter);
    }

    void operator()(const TensorDict& inputs, TensorRefDict outputs = {}, bool syncPrev = true)
    {
        if (syncPrev)
            sync();

        if (!counter)
            ;   // no cache: nothing to advance
        else if (fixedCached >= 0)
            *(uint32_t*)counter.map() = uint32_t(fixedCached);
        else
        {
            *(uint32_t*)counter.map() = tokensSoFar;
            tokensSoFar += (uint32_t)inputs.at("q").shape()[1];   // Q is [H, Tq, Dh]
        }

        NeuralNet::operator()(inputs, outputs, false);
    }
};


// Which path the last prepare() selected, for the summary rows.
std::string pathStr(const Attention& n)
{
    if (!n.usingFa()) return "naive";
    std::string s = n.usingCm() ? "fa-cm" : "fa";
    if (n.getGqaRatio() > 1)    s += "+pack" + std::to_string(n.getGqaRatio());
    if (n.getSplitKCount() > 1) s += "+split" + std::to_string(n.getSplitKCount());
    return s;
}


/////////////////////////////////////////////////////////////////////////////////////////
// CpuAttention - host mirror of the kernels' contract (see
// attention.comp): append new K/V to the cache, then causal
// attention where query token nq attends cache[0, cached) plus new tokens
// [0, nq]. GQA maps query head h to KV head h / G. Softmax in double.
/////////////////////////////////////////////////////////////////////////////////////////
struct CpuAttention {
    uint32_t H, Hkv, Dh, maxTokens;
    uint32_t cached = 0;
    uint32_t windowSize = 0;
    std::vector<uint32_t> blockEnd;
    uint32_t causalBits = 1;
    std::vector<float> kCache, vCache;   // [Hkv, maxTokens, Dh]

    CpuAttention(uint32_t H, uint32_t Hkv, uint32_t Dh, uint32_t maxTokens)
    : H(H), Hkv(Hkv), Dh(Dh), maxTokens(maxTokens)
    , kCache(size_t(Hkv) * maxTokens * Dh)
    , vCache(size_t(Hkv) * maxTokens * Dh)
    {}

    // q [H, Tq, Dh], k / v [Hkv, Tq, Dh] -> out [Tq, H, Dh] (token-major)
    std::vector<float> forward(const std::vector<float>& q, const std::vector<float>& k,
                               const std::vector<float>& v, uint32_t Tq)
    {
        const uint32_t G = H / Hkv;
        const double scale = 1.0 / std::sqrt(double(Dh));
        std::vector<float> out(size_t(Tq) * H * Dh);
        std::vector<double> w;

        for (uint32_t nq = 0; nq < Tq; ++nq)
            for (uint32_t h = 0; h < H; ++h)
            {
                const uint32_t kvh = h / G;
                auto segEnd = [&](uint32_t b) {
                    return b < blockEnd.size() ? std::min(blockEnd[b], Tq) : Tq;
                };
                const uint32_t segs = (uint32_t) blockEnd.size() + 1;
                uint32_t seg = 0, end = segEnd(0);
                for (uint32_t b = 1; b <= blockEnd.size(); ++b)
                    if (nq >= end) { seg = b; end = segEnd(b); }
                const uint32_t newEnd = ((causalBits >> (segs - 1 - seg)) & 1u) ? nq + 1 : end;
                const uint32_t nKv = cached + newEnd;
                const uint32_t qPos = cached + nq;
                const uint32_t firstKv = windowSize && qPos >= windowSize
                    ? qPos + 1 - windowSize : 0;
                const float* qRow = &q[(size_t(h) * Tq + nq) * Dh];

                auto kRow = [&](uint32_t j) {
                    return j < cached ? &kCache[(size_t(kvh) * maxTokens + j) * Dh]
                                      : &k[(size_t(kvh) * Tq + (j - cached)) * Dh];
                };
                auto vRow = [&](uint32_t j) {
                    return j < cached ? &vCache[(size_t(kvh) * maxTokens + j) * Dh]
                                      : &v[(size_t(kvh) * Tq + (j - cached)) * Dh];
                };

                w.assign(nKv, 0.0);
                double m = -1e30;
                for (uint32_t j = firstKv; j < nKv; ++j)
                {
                    const float* kr = kRow(j);
                    double dot = 0.0;
                    for (uint32_t d = 0; d < Dh; ++d)
                        dot += double(qRow[d]) * kr[d];
                    w[j] = dot * scale;
                    m = std::max(m, w[j]);
                }
                double l = 0.0;
                for (uint32_t j = firstKv; j < nKv; ++j)
                {
                    w[j] = std::exp(w[j] - m);
                    l += w[j];
                }

                float* oRow = &out[(size_t(nq) * H + h) * Dh];
                for (uint32_t d = 0; d < Dh; ++d)
                {
                    double acc = 0.0;
                    for (uint32_t j = firstKv; j < nKv; ++j)
                        acc += w[j] * vRow(j)[d];
                    oRow[d] = float(acc / l);
                }
            }

        if (maxTokens == 0)
            return out;   // nothing carried over, so there is nothing to append

        // Append the new tokens (values are fp16-exact, so the fp32 mirror
        // matches the fp16 cache bit-for-bit).
        for (uint32_t kvh = 0; kvh < Hkv; ++kvh)
            for (uint32_t t = 0; t < Tq; ++t)
                for (uint32_t d = 0; d < Dh; ++d)
                {
                    const size_t src = (size_t(kvh) * Tq + t) * Dh + d;
                    const size_t dst = (size_t(kvh) * maxTokens + cached + t) * Dh + d;
                    kCache[dst] = k[src];
                    vCache[dst] = v[src];
                }
        cached += Tq;
        return out;
    }
};


// "150+1x4" for {150, 1, 1, 1, 1}
std::string stepsStr(const std::vector<uint32_t>& steps)
{
    std::string s;
    for (size_t i = 0; i < steps.size(); )
    {
        size_t j = i;
        while (j < steps.size() && steps[j] == steps[i]) ++j;
        if (!s.empty()) s += "+";
        s += std::to_string(steps[i]);
        if (j - i > 1) s += "x" + std::to_string(j - i);
        i = j;
    }
    return s;
}


// The axes of one verify case beyond the shape (defaults = fp32 I/O, fp16 cache).
struct CaseOpts {
    Dtype kvCacheDtype     = Dtype::Fp16;   // K / V cache
    Dtype ioDtype     = Dtype::Fp32;   // Q / K / V inputs and O
    bool  expectNaive = false;         // the shape must be rejected by both FA tuners
    std::vector<uint32_t> blockEnd;
    uint32_t causalBits = 1;
    uint32_t windowSize = 0;
    bool resetCausalityAfterPrefill = false;
    uint32_t batch    = 0;   // > 0: pass Q/K/V as [batch, heads, Tq, Dh]
};


vai::CausalBlockEnd toBlockEnd(const std::vector<uint32_t>& v)
{
    vai::CausalBlockEnd a = {};
    VAI_ASSERT(v.size() < a.size());
    std::copy(v.begin(), v.end(), a.begin());
    return a;
}


// Run one prefill/decode sequence on one node, verifying every forward
// against the host reference, and print a single summary row.
bool verifyCase(const char* label, uint32_t H, uint32_t Hkv, uint32_t Dh,
                uint32_t maxTokens, const std::vector<uint32_t>& steps, CaseOpts c = {})
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32, .oDefaultDtype = c.ioDtype };
    AttnModel model({ .headDim = Dh, .numQHeads = H, .numKVHeads = Hkv,
                      .kvCacheSize = maxTokens,
                      .windowSize = c.windowSize,
                      .kvCacheDtype = c.kvCacheDtype }, netCfg,
                    c.causalBits, toBlockEnd(c.blockEnd));
    CpuAttention ref(H, Hkv, Dh, maxTokens);
    ref.windowSize = c.windowSize;
    ref.blockEnd = c.blockEnd;
    ref.causalBits = c.causalBits;

    double maxErr = 0.0;
    bool pathOk = true;
    std::string paths, lastPath;
    uint32_t salt = 1;

    for (uint32_t Tq : steps)
    {
        std::vector<float> q = makeData(size_t(H) * Tq * Dh, salt++);
        std::vector<float> k = makeData(size_t(Hkv) * Tq * Dh, salt++);
        std::vector<float> v = makeData(size_t(Hkv) * Tq * Dh, salt++);

        // A leading batch axis is the same bytes: B sweeps laid end to end, so
        // the host reference of B * H heads answers for it unchanged.
        const uint32_t B = c.batch ? c.batch : 1;
        auto shape = [&](uint32_t heads) {
            return c.batch ? std::vector<uint32_t>{B, heads / B, Tq, Dh}
                           : std::vector<uint32_t>{heads, Tq, Dh};
        };

        Tensor out;
        model({
            {"q", Tensor(shape(H), c.ioDtype).setFloats(q)},
            {"k", Tensor(shape(Hkv), c.ioDtype).setFloats(k)},
            {"v", Tensor(shape(Hkv), c.ioDtype).setFloats(v)}
        }, {{"out0", &out}});

        std::vector<float> gotOut = readFloats(out, size_t(Tq) * H * Dh, c.ioDtype);
        std::vector<float> refOut = ref.forward(q, k, v, Tq);
        maxErr = std::max(maxErr, maxRelErr(refOut.data(), gotOut.data(), refOut.size()));

        if (c.resetCausalityAfterPrefill)
        {
            model.sync();
            vai::writeCausalMask(model.causalMask, 1);
            ref.blockEnd.clear();
            ref.causalBits = 1;
        }

        if (c.expectNaive && model.node.usingFa()) pathOk = false;
        std::string p = pathStr(model.node);
        if (p != lastPath)
        {
            paths += (paths.empty() ? "" : ">") + p;
            lastPath = p;
        }
    }

    // fp32 accumulation lands near 1e-4; the budget covers the device-gated
    // f16-accumulation flavors (cm PV/O/S, scalar O regs) over deep caches.
    // fp16 O adds its own rounding on top.
    double tol = (c.ioDtype == Dtype::Fp16) ? 5e-2 : 3e-2;
    bool ok = (maxErr <= tol) && pathOk;
    char blocks[32] = "";
    if (!c.blockEnd.empty())
        snprintf(blocks, sizeof(blocks), " blocks=%zu", c.blockEnd.size() + 1);

    printf("  [%s] %-16s H=%-2u Hkv=%-2u Dh=%-3u io=%s kv=%s%s%s%s steps=%-10s path=%-18s maxRel=%.2e%s\n",
        ok ? "PASS" : "FAIL", label, H, Hkv, Dh,
        c.ioDtype == Dtype::Fp16 ? "f16" : "f32",
        c.kvCacheDtype == Dtype::Fp16 ? "f16" : "f32",
        c.causalBits ? "" : " noncausal", maxTokens ? "" : " nocache", blocks,
        stepsStr(steps).c_str(), paths.c_str(), maxErr,
        pathOk ? "" : " (expected naive)");
    return ok;
}


// Time one steady-state shape (GPU timestamps): the counter is pinned to
// `cached`, so every iteration appends over the same rows and sweeps the same
// cache depth. Prints a single summary row.
void benchCase(uint32_t H, uint32_t Hkv, uint32_t Dh, uint32_t maxTokens,
               uint32_t Tq, uint32_t cached, int iters)
{
    NeuralNetConfig netCfg = { .bufferPool = sharedPool(),
                               .wDefaultDtype = Dtype::Fp32 };

    AttnModel model({ .headDim = Dh, .numQHeads = H, .numKVHeads = Hkv,
                      .kvCacheSize = maxTokens }, netCfg);
    model.node.setProfileTag("attn");

    // Fill the cache rows the sweep will read — the pool hands back garbage,
    // and NaN/Inf bit patterns in the softmax would distort the timing.
    Tensor out;
    if (cached > 0)
    {
        model.fixedCached = 0;
        model({
            {"q", Tensor({H, cached, Dh}, Dtype::Fp32).setFloats(makeData(size_t(H) * cached * Dh, 4))},
            {"k", Tensor({Hkv, cached, Dh}, Dtype::Fp16).setFloats(makeData(size_t(Hkv) * cached * Dh, 5))},
            {"v", Tensor({Hkv, cached, Dh}, Dtype::Fp16).setFloats(makeData(size_t(Hkv) * cached * Dh, 6))}
        }, {{"out0", &out}});
        out.getData();
    }
    model.fixedCached = cached;

    Tensor qT = Tensor({H, Tq, Dh}, Dtype::Fp32).setFloats(makeData(size_t(H) * Tq * Dh, 1));
    Tensor kT = Tensor({Hkv, Tq, Dh}, Dtype::Fp16).setFloats(makeData(size_t(Hkv) * Tq * Dh, 2));
    Tensor vT = Tensor({Hkv, Tq, Dh}, Dtype::Fp16).setFloats(makeData(size_t(Hkv) * Tq * Dh, 3));

    for (int i = 0; i < 5; ++i)   // warm-up: pipeline build + caches
    {
        model({{"q", qT}, {"k", kT}, {"v", vT}}, {{"out0", &out}});
        out.getData();
    }
    model.resetProfileRecords();

    for (int i = 0; i < iters; ++i)
    {
        model({{"q", qT}, {"k", kT}, {"v", vT}}, {{"out0", &out}});
        out.getData();
    }

    printf(" %4u %4u %4u %5u %7u | %9.4f  %s\n",
        H, Hkv, Dh, Tq, cached, model.getGpuTimeMs("attn"), pathStr(model.node).c_str());
}


int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep prints visible up to an assert/abort
    getVaiDevice();

    printf("\n=== Attention verify (selected path vs host reference) ===\n");
    int fails = 0;

    // Model shapes, prefill -> decode (the flip re-requests the tuning)
    fails += !verifyCase("mha-gpt2",     12, 12, 64,  256,  {150, 1, 1, 1, 1});
    fails += !verifyCase("gqa-llama",    32, 8,  64,  256,  {150, 1, 1, 1, 1});

    // Small-batch decode rows: the Tq <= 8 GQA-packed tile gate
    fails += !verifyCase("gqa-smallrows",32, 8,  64,  128,  {64, 2, 4, 8});

    // Decode with no prefill: cached = 0 first step, nRows = 1 tuning
    fails += !verifyCase("decode-first", 8,  8,  64,  16,   {1, 1, 1, 1});

    // Head-dim axes: 128 / 256 (staging / occupancy-limiter branches),
    // 40 / 80 (Dh & 8: reduced rows, wgKv = 64)
    fails += !verifyCase("dh128",        8,  8,  128, 128,  {100, 1, 1});
    fails += !verifyCase("dh256",        4,  4,  256, 128,  {64, 1});
    fails += !verifyCase("gqa-dh256",    4,  1,  256, 128,  {64, 1, 1});
    fails += !verifyCase("dh40",         8,  8,  40,  128,  {100, 1, 1});
    fails += !verifyCase("dh80",         8,  8,  80,  128,  {100, 1});

    // Dh % 8 != 0 is rejected by both FA tuners on every device -> naive
    fails += !verifyCase("naive-dh44",   8,  8,  44,  64,   {32, 1}, {.expectNaive = true});

    // dtype axes
    fails += !verifyCase("f16-io",       32, 8,  64,  128,  {64, 1, 1}, {.ioDtype = Dtype::Fp16});
    fails += !verifyCase("f32-kv",       32, 8,  64,  128,  {64, 1, 1}, {.kvCacheDtype = Dtype::Fp32});

    // Split-K decode: few heads + deep cache underfill the grid
    fails += !verifyCase("split-k",      2,  2,  64,  1152, {1024, 1, 1});

    // a run of blocks in front of a dispatch: pi0's suffix is [state | 50 actions],
    // where the state may not see the actions that follow it
    fails += !verifyCase("blocks-pi0",    8, 1,  256, 867,  {816, 51},
                         {.blockEnd = {1}, .causalBits = 0});
    fails += !verifyCase("blocks-pi0-naive", 8, 8, 44, 128,  {64, 51},
                         {.expectNaive = true, .blockEnd = {1}, .causalBits = 0});
    // several equal blocks, each seeing the ones before it, then causal text
    fails += !verifyCase("blocks-images", 12, 12, 64,  256,  {40, 1, 1},
                         {.blockEnd = {8, 16, 24}, .causalBits = 0b0001});

    // Image blocks plus local attention, including unaligned tile boundaries,
    // windows shorter than the image, cached prefixes and causal decode reset.
    for (uint32_t dh : {44u, 64u, 256u})
    {
        CaseOpts c{.expectNaive = dh == 44, .blockEnd = {7, 39},
                   .causalBits = 0b101, .windowSize = 16,
                   .resetCausalityAfterPrefill = true};
        fails += !verifyCase("image-window", 8, 4, dh, 128, {65, 1, 1, 4}, c);
        c.resetCausalityAfterPrefill = false;
        fails += !verifyCase("image-cached", 8, 4, dh, 256, {65, 65}, c);
        c.blockEnd = {39}; c.causalBits = 0b01;
        fails += !verifyCase("image-first", 8, 4, dh, 128, {65}, c);
        c.blockEnd = {7}; c.causalBits = 0b10;
        fails += !verifyCase("image-last", 8, 4, dh, 128, {65}, c);
        c.blockEnd.clear(); c.causalBits = 0;
        fails += !verifyCase("image-only", 8, 4, dh, 128, {65}, c);
    }
    fails += !verifyCase("gemma4b-window", 8, 4, 256, 1536, {1100, 1, 1},
                         {.ioDtype = Dtype::Fp16, .blockEnd = {777, 1033},
                          .causalBits = 0b101, .windowSize = 1024,
                          .resetCausalityAfterPrefill = true});
    fails += !verifyCase("window-split", 2, 2, 64, 1152, {1024, 1, 1},
                         {.windowSize = 32});

    // causal, orthogonal to the cache: a dispatch also sees its newer tokens
    fails += !verifyCase("noncausal",    12, 12, 64,  256,  {150, 1, 1},   {.causalBits = 0});
    fails += !verifyCase("noncausal-gqa",32, 8,  64,  256,  {150, 8},      {.causalBits = 0});

    // No cache (kvCacheSize 0): no slots, and the counter binds null.
    // Repeated steps must stay independent, and the packed Tq == 1 rows are
    // where appendBeforeSweep would read a cache that does not exist.
    fails += !verifyCase("nocache-siglip",16, 16, 72, 0,    {256, 256},    {.causalBits = 0});
    fails += !verifyCase("nocache-causal",12, 12, 64, 0,    {150, 32, 1},  {});
    fails += !verifyCase("nocache-gqa",   32, 8,  64, 0,    {64, 8, 1},    {});
    fails += !verifyCase("nocache-naive",  8, 8,  44, 0,    {32, 1},       {.expectNaive = true});

    // A leading batch axis. The bytes are what the unbatched case already sent,
    // so any difference is the rank alone.
    fails += !verifyCase("rank4-b1",      12, 12, 64, 0,    {29, 1},       {.batch = 1});
    fails += !verifyCase("rank4-b1-gqa",  32, 8,  64, 0,    {64, 1},       {.batch = 1});
    fails += !verifyCase("rank4-b1-naive", 8, 8,  44, 0,    {32, 1},       {.expectNaive = true, .batch = 1});

    printf(fails == 0 ? "  => ALL PASS\n" : "  => %d case(s) FAILED\n", fails);
    if (fails != 0)
        return 1;   // don't bench a broken kernel

    if (argc > 1 && std::string(argv[1]) == "--verify-only")
        return 0;

    const int iters = 200;
    printf("\n=== bench (fp32 I/O, fp16 cache, GPU avg ms over %d iters) ===\n", iters);
    printf("    H  Hkv   Dh    Tq  cached |        ms  path\n");

    benchCase(12, 12, 64,  1024, 150, 0,    iters);   // gpt2 prefill
    benchCase(12, 12, 64,  1024, 1,   512,  iters);   // gpt2 decode
    benchCase(32, 8,  64,  4096, 150, 0,    iters);   // llama3.2-1B prefill
    benchCase(32, 8,  64,  4096, 1,   512,  iters);   // llama3.2-1B decode
    benchCase(32, 8,  64,  4096, 1,   2048, iters);   // llama3.2-1B long-context decode
    benchCase(8,  8,  128, 2048, 1,   1024, iters);   // big-head decode
    benchCase(32, 32, 256, 64,   1,   0,    iters * 5); // shallow-cache decode: 1-token
                                                        // sweep, the tile epilogue dominates

    return 0;
}
