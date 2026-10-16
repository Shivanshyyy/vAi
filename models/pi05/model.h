/*=============================================================================
  Written by   oneway@vaistrata.com
  Model        pi0.5 (3B VLA — PaliGemma 3B and a 300M action expert on adaRMS)
=============================================================================*/

#ifndef PI05_MODEL_H
#define PI05_MODEL_H

#include "models/config-base.h"
#include "vai/nodes/token-embedding.h"
#include "vai/groups/patch-embedding.h"
#include "vai/nodes/concat.h"
#include "vai/nodes/cyclic-table-reader.h"
#include "vai/groups/transformer.h"
#include "vai/nodes/linear.h"
#include <cmath>
#include <string>
#include <utility>


namespace pi05 {


using VisionTransformer = vai::group::Transformer<
    vai::TransformerType::Classic
>;


using GemmaTransformer = vai::group::Transformer<
    vai::TransformerType::Default
>;


using ActionTransformer = vai::group::Transformer<
    vai::TransformerType::Pi05Action
>;


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - pi0.5 (3B VLA) - PaliGemma 3B and a Gemma 300M action expert
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;                    // checkpoint folder under data/pi05/ckpts/
    uint32_t maxCameras = 3;                // image slots the policy was trained with
    uint32_t maxPromptTokens = 200;         // task and discretized state, as text
    uint32_t actionDim = 32;                // one action, and the state padded to the same width
    uint32_t actionHorizon = 50;            // actions one chunk carries
    float timeMinPeriod = 4e-3f;            // fastest sinusoid of the timestep embedding
    float timeMaxPeriod = 4.0f;             // ... and its slowest

    VisionModelConfig visionCfg = siglip::so400m14(224);
    LanguageModelConfig languageCfg = {
        gemma::ModelConfig::_2B().languageCfg,
        257152   // vocabSize: 256000 text, 1024 <loc*> and 128 <seg*>
    };
    vai::TransformerConfig actionCfg = gemma::ModelConfig::_300M().languageCfg;

    // Every camera the policy allows, the prompt and the action chunk: the
    // longest context the cache ever holds. The state rides in the prompt.
    uint32_t maxContext() const
    {
        return maxCameras * visionCfg.numPatches()
            + maxPromptTokens
            + actionHorizon;
    }

    // Rows of the action expert's condition: γ, β and gate for both norms of
    // every layer, then γ and β of its final norm.
    uint32_t condRows() const
    {
        return 6 * actionCfg.numLayers + 2;
    }

    static ModelConfig _3B()
    {
        return {
            .variant    = "3B",
        };
    }
};


struct RuntimeConfig
{
    uint32_t denoiseSteps = 10;
    vai::Dtype kvCacheDtype = vai::Dtype::Fp16;
};


struct Model : public vai::NeuralNet
{
    ModelConfig modelCfg;
    RuntimeConfig runtimeCfg;

    vai::group::PatchEmbedding patchEmbed;
    VisionTransformer visionEncoder;
    vai::node::Linear projector;
    vai::node::Reshape mergeCameras;

    vai::node::TokenEmbedding promptEmbed;
    vai::node::Concat catCamsPrompt;
    GemmaTransformer languageEncoder;

    vai::node::Linear actionProj;
    vai::node::CyclicTableReader condTable;
    ActionTransformer actionDecoder;
    vai::node::Linear velocityHead;

    eva::Buffer kvCacheCounter;
    eva::Buffer bidirectionalMask;

    Model(
        const vai::NeuralNetConfig& netCfg,
        const ModelConfig& modelCfg,
        const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , runtimeCfg(runtimeCfg)
    , patchEmbed({
        .imageChannels = modelCfg.visionCfg.imageChannels,
        .imageSize = modelCfg.visionCfg.imageSize,
        .patchSize = modelCfg.visionCfg.patchSize,
        .embedDim = modelCfg.visionCfg.layerCfg.embedDim,
        .allowBatch = true})
    , visionEncoder(modelCfg.visionCfg.with({
        .allowBatch = true,
        .kvCacheDtype = runtimeCfg.kvCacheDtype}))
    , projector({
        .inDim = modelCfg.visionCfg.layerCfg.embedDim,
        .outDim = modelCfg.languageCfg.layerCfg.embedDim,
        .hasBias = true})
    , mergeCameras({-1, modelCfg.languageCfg.layerCfg.embedDim})
    , promptEmbed({
        .vocabSize = modelCfg.languageCfg.vocabSize,
        .embedDim = modelCfg.languageCfg.layerCfg.embedDim,
        .scale = std::sqrt(float(modelCfg.languageCfg.layerCfg.embedDim))})
    , languageEncoder(modelCfg.languageCfg.with({
        .kvCacheDtype = runtimeCfg.kvCacheDtype,
        .kvCacheSize = modelCfg.maxContext(),
        .gatingLastOut = true}))  // Only the prefix KV cache is used.
    , actionProj({
        .inDim = modelCfg.actionDim,
        .outDim = modelCfg.actionCfg.layerCfg.embedDim,
        .hasBias = true})
    , condTable({.numRows = runtimeCfg.denoiseSteps})
    , actionDecoder(modelCfg.actionCfg.with({
        .kvCacheDtype = runtimeCfg.kvCacheDtype,
        .kvCacheSize = modelCfg.maxContext()}))
    , velocityHead({
        .inDim = modelCfg.actionCfg.layerCfg.embedDim,
        .outDim = modelCfg.actionDim,
        .hasBias = true,
        .acceptResidual = true,
        .scalarScale = -1.0f / runtimeCfg.denoiseSteps,
        .oDtype = vai::Dtype::Fp32})
    {
        input("inImages") - patchEmbed - visionEncoder - projector - mergeCameras - "in0"/catCamsPrompt;
        input("inPromptTokens")                                     - promptEmbed - "in1"/catCamsPrompt;
        catCamsPrompt - languageEncoder;
        languageEncoder.setGateOpen(false);

        input("inActions") -                         "residual"/velocityHead;
        input("inActions") - actionProj -       actionDecoder - velocityHead - output("outActions");
        input("inActions") - condTable - "cond"/actionDecoder;

        kvCacheCounter = vai::createKvCacheCounter(device());
        languageEncoder.setKvCacheCounter(kvCacheCounter);
        actionDecoder.setKvCacheCounter(kvCacheCounter);

        // The suffix sees the prefix and the whole chunk, so one mask serves all three.
        bidirectionalMask = vai::createCausalMask(device(), 0);
        visionEncoder.setCausalMask(bidirectionalMask);
        languageEncoder.setCausalMask(bidirectionalMask);
        actionDecoder.setCausalMask(bidirectionalMask);

        for (uint32_t i = 0; i < modelCfg.languageCfg.numLayers; ++i)
        {
            auto& encAttn = languageEncoder.layers[i].mhAttn.attn;
            auto& decAttn = actionDecoder.layers[i].mhAttn.attn;
            decAttn["K_cache"] = encAttn["K_cache"];
            decAttn["V_cache"] = encAttn["V_cache"];
        }
    }

    uint32_t prefixLength_ = 0;

    // Input: images [cameras, C, S, S], promptTokenIds [T] carrying task and state.
    void prefill(const vai::Tensor& images, const vai::Tensor& promptTokenIds)
    {
        const uint32_t cameras = images.shape()[0];
        VAI_ASSERT(cameras >= 1 && cameras <= modelCfg.maxCameras);
        VAI_ASSERT(promptTokenIds.shape()[0] <= modelCfg.maxPromptTokens);
        sync();  // The GPU may still be reading kvCacheCounter.

        prefixLength_ = cameras * modelCfg.visionCfg.numPatches() + promptTokenIds.shape()[0];
        *(uint32_t*)kvCacheCounter.map() = 0;
        condTable.rewind();

        (*this)({{"inImages", images}, {"inPromptTokens", promptTokenIds}}, {}, false);
    }

    // Input: action [actionHorizon, actionDim], initialized with noise.
    // Output: denoised actions in the same GPU tensor.
    void decode(vai::Tensor* action)
    {
        VAI_ASSERT(action);

        sync();  // The GPU may still be reading kvCacheCounter.
        *(uint32_t*)kvCacheCounter.map() = prefixLength_;

        for (uint32_t step = 0; step < runtimeCfg.denoiseSteps; ++step)
            (*this)({{"inActions", *action}}, {{"outActions", action}}, false);
    }
};


}

#endif
