/*=============================================================================
  Written by   oneway@vaistrata.com,
               qkrtmddn0507@vaistrata.com
  Model        Gemma 3 (270M / 1B / 4B(VLM) / 12B(VLM))
=============================================================================*/

#ifndef GEMMA3_MODEL_H
#define GEMMA3_MODEL_H

#define TEST_INDEX_COPY

#include "models/config-base.h"
#include "groups/vision-tower.h"
#include "vai/nodes/linear.h"
#ifdef TEST_INDEX_COPY
    #include "vai/nodes/reshape.h"
    #include "vai/nodes/index-copy.h"
#else
    #include "vai/nodes/row-slice-scatter.h"
#endif
#include "vai/nodes/token-embedding.h"
#include "vai/nodes/topk-sampling.h"
#include "vai/groups/transformer.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>


namespace gemma3 {


using LanguageTransformer = vai::group::Transformer<
    vai::TransformerType::Gemma3
>;


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - Gemma 3 (270M / 1B: text only, 4B / 12B: image + text)
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;                    // checkpoint folder under data/
    LanguageModelConfig languageCfg;
    std::optional<VisionTowerConfig> visionCfg;

    static bool isGlobalLayer(uint32_t layer)
    {
        static constexpr uint32_t globalLayerPeriod = 6;
        return (layer + 1) % globalLayerPeriod == 0;
    }

    static ModelConfig _270M()
    {
        return {
            .variant = "270M",
            .languageCfg = {
                vai::TransformerConfig{
                    .numLayers = 18,
                    .layerCfg = {
                        .embedDim = 640,
                        .headDim = 256,
                        .numQHeads = 4,
                        .numKVHeads = 1,
                        .withQkNorm = true,
                        .ffnHiddenDim = 2048,
                        .ffnActivation = vai::Epilogue::GeluTanh,
                        .normEps = 1e-6f},
                    .perLayer = [](uint32_t layer, vai::TransformerLayerConfig& layerCfg)
                    {
                        const bool global = isGlobalLayer(layer);
                        layerCfg.windowSize = global ? 0 : 512;
                        layerCfg.ropeCfg = global ? rope(1000000.0f) : rope(10000.0f);
                    }},
                262144, // vocabSize
                32768   // maxTrainedContext
            }
        };
    }

    static ModelConfig _1B()
    {
        return {
            .variant = "1B",
            .languageCfg = {
                vai::TransformerConfig{
                    .numLayers = 26,
                    .layerCfg = {
                        .embedDim = 1152,
                        .headDim = 256,
                        .numQHeads = 4,
                        .numKVHeads = 1,
                        .withQkNorm = true,
                        .ffnHiddenDim = 6912,
                        .ffnActivation = vai::Epilogue::GeluTanh,
                        .normEps = 1e-6f},
                    .perLayer = [](uint32_t layer, vai::TransformerLayerConfig& layerCfg)
                    {
                        const bool global = isGlobalLayer(layer);
                        layerCfg.windowSize = global ? 0 : 512;
                        layerCfg.ropeCfg = global ? rope(1000000.0f) : rope(10000.0f);
                    }},
                262144, // vocabSize
                32768   // maxTrainedContext
            }
        };
    }

    static ModelConfig _4B()
    {
        const uint32_t textEmbedDim = 2560;

        return {
            .variant = "4B",
            .languageCfg = {
                vai::TransformerConfig{
                    .numLayers = 34,
                    .layerCfg = {
                        .embedDim = textEmbedDim,
                        .headDim = 256,
                        .numQHeads = 8,
                        .numKVHeads = 4,
                        .withQkNorm = true,
                        .ffnHiddenDim = 10240,
                        .ffnActivation = vai::Epilogue::GeluTanh,
                        .normEps = 1e-6f},
                    .perLayer = [](uint32_t layer, vai::TransformerLayerConfig& layerCfg)
                    {
                        const bool global = isGlobalLayer(layer);
                        layerCfg.windowSize = global ? 0 : 1024;
                        layerCfg.ropeCfg = global ? linearRope(1000000.0f) : rope(10000.0f);
                    }},
                262208,   // vocabSize
                131072   // maxTrainedContext
            },
            .visionCfg = VisionTowerConfig(textEmbedDim),
        };
    }

    static ModelConfig _12B()
    {
        const uint32_t textEmbedDim = 3840;

        return {
            .variant = "12B",
            .languageCfg = {
                vai::TransformerConfig{
                    .numLayers = 48,
                    .layerCfg = {
                        .embedDim = textEmbedDim,
                        .headDim = 256,
                        .numQHeads = 16,
                        .numKVHeads = 8,
                        .withQkNorm = true,
                        .ffnHiddenDim = 15360,
                        .ffnActivation = vai::Epilogue::GeluTanh,
                        .normEps = 1e-6f},
                    .perLayer = [](uint32_t layer, vai::TransformerLayerConfig& layerCfg)
                    {
                        const bool global = isGlobalLayer(layer);
                        layerCfg.windowSize = global ? 0 : 1024;
                        layerCfg.ropeCfg = global ? linearRope(1000000.0f) : rope(10000.0f);
                    }},
                262208,   // vocabSize
                131072   // maxTrainedContext
            },
            .visionCfg = VisionTowerConfig(textEmbedDim),
        };
    }

private:
    // Plain RoPE, so the two bands differ in the only field it reads.
    static vai::RopeConfig rope(float base)
    {
        return {
            .type = vai::RopeType::Default,
            .base = base
        };
    }

    static vai::RopeConfig linearRope(float base)
    {
        return {
            .type = vai::RopeType::Linear,
            .base = base,
            .factor = 8.0f
        };
    }
};


struct RuntimeConfig
{
    vai::Dtype kvCacheDtype = vai::Dtype::Fp16;
    uint32_t maxRuntimeContext = uint32_t(-1);
};


struct Model : public vai::NeuralNet
{
    ModelConfig modelCfg;
    RuntimeConfig runtimeCfg;
    uint32_t maxRuntimeContext;

    std::optional<VisionTower> visionTower;
#ifdef TEST_INDEX_COPY
    vai::node::Reshape imageRows;
    vai::node::IndexCopy imageCopy;
#else
    vai::node::RowSliceScatter imageScatter;
#endif

    vai::node::TokenEmbedding embedding;
    LanguageTransformer decoder;
    vai::node::Linear lmHead;
    vai::node::TopKSampling topkSampling;

    eva::Buffer kvCacheCounter;
    eva::Buffer causalMask;
    eva::Buffer bidirectionalMask;
#ifdef TEST_INDEX_COPY
    eva::Buffer imageIndex;
#endif

    Model(
        const vai::NeuralNetConfig& netCfg,
        const ModelConfig& modelCfg,
        const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , runtimeCfg(runtimeCfg)
    , maxRuntimeContext(std::min(runtimeCfg.maxRuntimeContext, modelCfg.languageCfg.maxTrainedContext))
#ifdef TEST_INDEX_COPY
    , imageRows({-1, int64_t(modelCfg.languageCfg.layerCfg.embedDim)})
#else
    , imageScatter({.sliceSize = modelCfg.visionCfg ? modelCfg.visionCfg->tokensPerImage : 0})
#endif
    , embedding({
        .vocabSize = modelCfg.languageCfg.vocabSize,
        .embedDim = modelCfg.languageCfg.layerCfg.embedDim,
        .scale = std::sqrt(float(modelCfg.languageCfg.layerCfg.embedDim))})
    , decoder(modelCfg.languageCfg.with({
        .kvCacheDtype = runtimeCfg.kvCacheDtype,
        .kvCacheSize = maxRuntimeContext,
        .sliceLastRow = true}))
    , lmHead({
        .inDim = modelCfg.languageCfg.layerCfg.embedDim,
        .outDim = modelCfg.languageCfg.vocabSize})
    {
        this->runtimeCfg.maxRuntimeContext = maxRuntimeContext;

        if (modelCfg.visionCfg)
            visionTower.emplace(*modelCfg.visionCfg);

#ifdef TEST_INDEX_COPY
        input("inTokenIds") - embedding             - "in0"/imageCopy;
        input("inImages") - visionTower - imageRows - "in1"/imageCopy;
        imageCopy - decoder - lmHead - topkSampling - output("nextTokenId");
#else
        input("inTokenIds") - embedding - "in0"/imageScatter;
        input("inImages") - visionTower - "in1"/imageScatter;
        imageScatter - decoder - lmHead - topkSampling - output("nextTokenId");
#endif

        kvCacheCounter = vai::createKvCacheCounter(device());
        decoder.setKvCacheCounter(kvCacheCounter);

        causalMask = vai::createCausalMask(device(), 1);
        decoder.setCausalMask(causalMask);

        if (visionTower)
        {
            bidirectionalMask = vai::createCausalMask(device(), 0);
            visionTower->encoder.setCausalMask(bidirectionalMask);
#ifdef TEST_INDEX_COPY
            imageIndex = device().createBuffer({
                .size = MAX_CAUSAL_BLOCKS * modelCfg.visionCfg->tokensPerImage * sizeof(uint32_t),
                .usage = eva::BUFFER_USAGE::STORAGE_BUFFER,
                .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT
            });
            imageCopy.setIndex(imageIndex);
#endif
        }
    }

    uint32_t tokensSoFar_ = 0;

    // Input: images [B, C, S, S] (or [C, S, S]); empty for text only. promptTokenIds [T].
    // Output: nextTokenId [1], the first generated token.
    // Images must match placeholder blocks in prompt order.
    void prefill(const vai::Tensor& images, const vai::Tensor& promptTokenIds, vai::Tensor* nextTokenId)
    {
        VAI_ASSERT(nextTokenId);

        const uint32_t total = promptTokenIds.shape()[0];
        const auto& shape = images.shape();
        const uint32_t imageCount = !images.validShape() ? 0 : shape.size() == 4 ? shape[0] : 1;

        if (imageCount > 0)
        {
            if (!modelCfg.visionCfg || !visionTower)
                throw std::runtime_error("image input is not supported by this Gemma 3 checkpoint");
            const auto& cfg = *modelCfg.visionCfg;
            VAI_ASSERT(shape.size() == 3 || shape.size() == 4);
            VAI_ASSERT(shape[shape.size() - 3] == cfg.imageChannels);
            VAI_ASSERT(shape[shape.size() - 2] == cfg.imageSize);
            VAI_ASSERT(shape.back() == cfg.imageSize);
            if (!promptTokenIds.hasHostData())
                throw std::runtime_error("image prefill requires host-visible prompt tokens");
        }

        // Image blocks use bidirectional attention; text blocks are causal.
        std::vector<uint32_t> imageStarts;
        vai::CausalBlockEnd blockEnd = {};
        uint32_t causalBits = 0;
        {
            const uint32_t tokensPerImage = modelCfg.visionCfg ? modelCfg.visionCfg->tokensPerImage : 0;
            const uint32_t imageTokenId = modelCfg.visionCfg ? modelCfg.visionCfg->imageTokenId : 0;
            const uint32_t* ids = modelCfg.visionCfg ? static_cast<const uint32_t*>(promptTokenIds.hostData()) : nullptr;
            uint32_t block = 0;
            uint32_t last = 0;

            auto appendBlock = [&](uint32_t end, bool causal)
            {
                if (MAX_CAUSAL_BLOCKS - 1 < block)
                    throw std::runtime_error("too many Gemma 3 image/text attention blocks");

                if (MAX_CAUSAL_BLOCKS - 2 < block && end < total)
                    throw std::runtime_error("too many Gemma 3 image/text attention blocks");

                causalBits = (causalBits << 1) | uint32_t(causal);
                if (end < total)
                    blockEnd[block] = end;
                ++block;
            };

            for (uint32_t pos = 0; ids && pos < total; )
            {
                if (ids[pos] != imageTokenId)
                {
                    ++pos;
                    continue;
                }

                if (imageCount <= imageStarts.size() ||
                    total - pos < tokensPerImage ||
                    !std::all_of(ids + pos, ids + pos + tokensPerImage,
                        [&](uint32_t id) { return id == imageTokenId; }))
                    throw std::runtime_error("Gemma 3 image placeholders do not match the image batch");

                if (last < pos)
                    appendBlock(pos, true);

                imageStarts.push_back(pos);
                pos += tokensPerImage;
                appendBlock(pos, false);
                last = pos;
            }

            if (imageStarts.size() != imageCount)
                throw std::runtime_error("Gemma 3 image placeholders do not match the image batch");

            if (last < total)
                appendBlock(total, true);
        }

        sync();  // The GPU may still be reading causalMask and imageIndex.
        vai::writeCausalMask(causalMask, causalBits, blockEnd);

#ifdef TEST_INDEX_COPY
        uint32_t* rows = imageIndex ? (uint32_t*)imageIndex.map() : nullptr;
        for (uint32_t start : imageStarts)
            for (uint32_t t = 0; t < modelCfg.visionCfg->tokensPerImage; ++t)
                *rows++ = start + t;
#else
        imageScatter.setRowStarts(std::move(imageStarts));
#endif

        step(images, promptTokenIds, nextTokenId);
    }

    // Input/output: inoutTokenId [1], previous token replaced by the next token.
    void decode(vai::Tensor* inoutTokenId)
    {
        VAI_ASSERT(inoutTokenId);
        step({}, *inoutTokenId, inoutTokenId);
    }

    void resetCache()
    {
        sync();  // The GPU may still be reading kvCacheCounter.
        tokensSoFar_ = 0;
        *(uint32_t*)kvCacheCounter.map() = 0;
    }

private:
    void step(const vai::Tensor& inImages, const vai::Tensor& inTokenIds, vai::Tensor* nextTokenId)
    {
        sync();  // The GPU may still be reading kvCacheCounter.

        *(uint32_t*)kvCacheCounter.map() = tokensSoFar_;
        tokensSoFar_ += inTokenIds.shape()[0];

        topkSampling.setUniform((float)std::rand() / ((float)RAND_MAX + 1.0f));

        vai::TensorRefDict outputs = {{"nextTokenId", nextTokenId}};

        vai::TensorDict inputs = {{"inTokenIds", inTokenIds}};
        if (inImages.validShape())
            inputs.emplace("inImages", inImages);

        (*this)(std::move(inputs), std::move(outputs), false);
    }
};


}

#endif
