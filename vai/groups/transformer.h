#ifndef VAI_GROUPS_TRANSFORMER_H
#define VAI_GROUPS_TRANSFORMER_H

#include "vai/groups/transformer-layer.h"
#include "vai/nodes/layer-norm.h"
#include "vai/nodes/rms-norm.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/slice.h"
#include <optional>
#include <functional>
#include <vector>

namespace vai {


using CausalBlockEnd = std::array<uint32_t, MAX_CAUSAL_BLOCKS - 1>;

// The host mirror of shaders/common/causal_mask.h. A model owns the buffer, as it
// owns the KV-cache counter, and hands it to every attention it masks: an
// attention without one does not record.
struct CausalMask {
    uint32_t blockCount;
    uint32_t bits;
    CausalBlockEnd blockEnd;
};

// bits marks each block causal, the first block in the highest of blockCount
// bits; blockEnd lists every end but the last, which is the prompt's, and
// stops at its first 0. Rewriting a mask the GPU is reading is a race: write
// it after sync().
inline void writeCausalMask(eva::Buffer buffer, uint32_t bits, const CausalBlockEnd& blockEnd = {})
{
    CausalMask mask{.blockCount = 1, .bits = bits, .blockEnd = blockEnd};
    while (mask.blockCount < MAX_CAUSAL_BLOCKS && blockEnd[mask.blockCount - 1] != 0)
    {
        VAI_ASSERT(mask.blockCount == 1 || blockEnd[mask.blockCount - 1] > blockEnd[mask.blockCount - 2]);
        ++mask.blockCount;
    }
    std::memcpy(buffer.map(), &mask, sizeof(mask));
}

inline eva::Buffer createCausalMask(eva::Device device, uint32_t bits, const CausalBlockEnd& blockEnd = {})
{
    eva::Buffer buffer = device.createBuffer({
        .size = sizeof(CausalMask),
        .usage = eva::BUFFER_USAGE::UNIFORM_BUFFER,
        .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT
    });
    writeCausalMask(buffer, bits, blockEnd);
    return buffer;
}

// The tokens the KV cache already holds, which attention and RoPE read. A model
// owns it and rewrites it after sync(), before each forward.
inline eva::Buffer createKvCacheCounter(eva::Device device)
{
    eva::Buffer buffer = device.createBuffer({
        .size = sizeof(uint32_t),
        .usage = eva::BUFFER_USAGE::UNIFORM_BUFFER,
        .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT
    });
    *(uint32_t*)buffer.map() = 0;
    return buffer;
}


// Runs on every layer's config, for the fields that vary down the stack; the two
// flags below are applied after it.
using TransformerPerLayer = std::function<void(uint32_t, vai::TransformerLayerConfig&)>;


// The fields of the two configs no set of weights decides, so a checkpoint entry
// leaves them out and the caller stamps them on with TransformerConfig::with().
struct StackOptions {
    bool allowBatch = false;
    vai::Dtype kvCacheDtype = vai::Dtype::Fp16;
    uint32_t kvCacheSize = 0;
    bool sliceLastRow = false;
    bool gatingLastOut = false;
};


// One stack: numLayers copies of layerCfg, whose embedDim and normEps serve the
// whole of it.
struct TransformerConfig {
    uint32_t numLayers;
    bool sliceLastRow = false;
    bool gatingLastOut = false;             // slightly faster when the prefill is chunked
    vai::TransformerLayerConfig layerCfg;
    vai::TransformerPerLayer perLayer;      // absent: every layer is layerCfg

    TransformerConfig with(const vai::StackOptions& opts) const
    {
        TransformerConfig cfg = *this;
        cfg.sliceLastRow = opts.sliceLastRow;
        cfg.gatingLastOut = opts.gatingLastOut;
        cfg.layerCfg.allowBatch = opts.allowBatch;
        cfg.layerCfg.kvCacheDtype = opts.kvCacheDtype;
        cfg.layerCfg.kvCacheSize = opts.kvCacheSize;
        return cfg;
    }
};


}   // namespace vai


namespace vai::group {


// The final norm is the one the layers normalize with.
template<TransformerType type> struct FinalNorm;
template<> struct FinalNorm<TransformerType::Classic>    { using Type = vai::node::LayerNorm; };
template<> struct FinalNorm<TransformerType::Default>    { using Type = vai::node::RmsNorm; };
template<> struct FinalNorm<TransformerType::Gemma3>     { using Type = vai::node::RmsNorm; };
template<> struct FinalNorm<TransformerType::Pi05Action> { using Type = vai::node::RmsNorm; };


/////////////////////////////////////////////////////////////////////////////////////////
// Transformer - a stack of layers and the norm that closes it
// Input:  in0  [T, embedDim], or [B, T, embedDim] with layerCfg.allowBatch
//         cond [6 * numLayers + 2, embedDim], Pi05Action only: each layer's six rows in
//              order, then scale and shift of the final norm
// Output: out0 the shape of in0, its T 1 under sliceLastRow
/////////////////////////////////////////////////////////////////////////////////////////
template<
    TransformerType type = TransformerType::Default,
    QkvLayout qkv = QkvLayout::Separate
>
struct Transformer : public vai::NodeGroup
{
    using Layer = TransformerLayer<type, qkv>;

    std::vector<Layer> layers;
    typename FinalNorm<type>::Type finalNorm;

    // Pi05Action only: the condition, cut into each layer's rows and the final norm's.
    std::optional<vai::node::Identity> cond;
    std::vector<vai::node::Slice> condSubTables;

    using Config = TransformerConfig;

    static typename FinalNorm<type>::Type::Config finalNormConfig(const Config& config)
    {
        if constexpr (type == TransformerType::Pi05Action)
            return {.eps = config.layerCfg.normEps, .hasShift = true, .coeffsAsInputs = true};
        else
            return {.eps = config.layerCfg.normEps};
    }

    Transformer(Config config)
    : finalNorm(finalNormConfig(config))
    {
        VAI_ASSERT(config.numLayers >= 1);

        layers.reserve(config.numLayers);
        for (uint32_t i = 0; i < config.numLayers; ++i)
        {
            const bool last = i == config.numLayers - 1;
            if (config.perLayer)
                config.perLayer(i, config.layerCfg);
            config.layerCfg.sliceLastRow = config.sliceLastRow && last;
            config.layerCfg.gatingLastOut = config.gatingLastOut && last;
            layers.emplace_back(config.layerCfg);
        }

        for (size_t i = 1; i < layers.size(); ++i)
            layers[i - 1] - layers[i];
        layers.back() - finalNorm;

        if constexpr (type == TransformerType::Pi05Action)
        {
            const uint32_t L = config.numLayers;

            cond.emplace();
            condSubTables.reserve(L + 2);
            for (uint32_t i = 0; i < L; ++i)
                condSubTables.emplace_back(vai::node::Slice::Config{.start = 6 * i, .count = 6});
            condSubTables.emplace_back(vai::node::Slice::Config{.start = 6 * L});
            condSubTables.emplace_back(vai::node::Slice::Config{.start = 6 * L + 1});

            for (uint32_t i = 0; i < L; ++i)
                *cond - condSubTables[i] - "cond"/layers[i];
            *cond - condSubTables[L]     - "scale"/finalNorm;
            *cond - condSubTables[L + 1] - "shift"/finalNorm;

            defineSlot("cond", cond->slot("in0"));
        }

        defineSlot("in0", layers[0].slot("in0"));
        defineSlot("out0", finalNorm.slot("out0"));
    }

    void setCausalMask(eva::Buffer buffer)
    {
        for (auto& layer : layers)
            layer.mhAttn.attn.setCausalMask(buffer);
    }

    void setKvCacheCounter(eva::Buffer buffer)
    {
        for (auto& layer : layers)
            layer.mhAttn.setKvCacheCounter(buffer);
    }

    // Closing it drops the last layer's FFN and everything past it, finalNorm included.
    void setGateOpen(bool open)
    {
        auto& gate = layers.back().gate;
        if (gate) gate->setOpen(open);
    }
};


}   // namespace vai::group

#endif // VAI_GROUPS_TRANSFORMER_H
