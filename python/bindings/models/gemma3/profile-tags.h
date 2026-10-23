#ifndef PYVAI_MODELS_GEMMA3_PROFILE_TAGS_H
#define PYVAI_MODELS_GEMMA3_PROFILE_TAGS_H

// Verbatim copy of setProfileTags() from examples/gemma3/demo.cpp (the
// members it walks are not exposed to Python). Re-copy when the source changes.

#include "models/gemma3/model.h"
#include <cstdint>
#include <string>

namespace pyvai::examples {
namespace {

void setProfileTags(gemma3::Model& net)
{
    auto towerLin = [](const char* tower, uint32_t in, uint32_t out)
    {
        return std::string(tower) + ".Linear.[" + std::to_string(in) + "x"
             + std::to_string(out) + "]";
    };
    const auto& decoderCfg = net.modelCfg.languageCfg;
    const auto& layerCfg = decoderCfg.layerCfg;
    const uint32_t D    = layerCfg.embedDim;
    const uint32_t H    = layerCfg.numQHeads;
    const uint32_t Hkv  = layerCfg.numKVHeads;
    const uint32_t Dh   = layerCfg.headDim;
    const uint32_t Dq   = H * Dh;
    const uint32_t Dkv  = Hkv * Dh;
    const uint32_t Dff  = layerCfg.ffnHiddenDim;
    const uint32_t V    = net.modelCfg.languageCfg.vocabSize;

    // Vision tower
    if (net.visionTower)
    {
        const auto& v = *net.modelCfg.visionCfg;
        const auto& visionLayer = v.layerCfg;
        auto& vision = net.visionTower.value();

        for (auto& b : vision.encoder.layers)
        {
            b.norm1.setProfileTag("Vision.LayerNorm");
            b.mhAttn.toQ.setProfileTag(towerLin("Vision", visionLayer.embedDim, visionLayer.embedDim));
            b.mhAttn.toK.setProfileTag(towerLin("Vision", visionLayer.embedDim, visionLayer.embedDim));
            b.mhAttn.toV.setProfileTag(towerLin("Vision", visionLayer.embedDim, visionLayer.embedDim));
            b.mhAttn.attn.setProfileTag("Vision.Attn");
            b.mhAttn.toOut.setProfileTag(towerLin("Vision", visionLayer.embedDim, visionLayer.embedDim));
            b.norm2.setProfileTag("Vision.LayerNorm");
            b.ffn.fc1.setProfileTag(towerLin("Vision", visionLayer.embedDim, visionLayer.ffnHiddenDim));
            b.ffn.fc2.setProfileTag(towerLin("Vision", visionLayer.ffnHiddenDim, visionLayer.embedDim));
        }

        vision.patchEmbed.perPatches.setProfileTag("Vision.Permute");

        vision.patchEmbed.project.setProfileTag(towerLin("Vision",
            v.imageChannels * v.patchSize * v.patchSize, visionLayer.embedDim));

        vision.encoder.finalNorm.setProfileTag("Vision.LayerNorm");

        vision.pool.setProfileTag("Vision.Pool");
        vision.imageNorm.setProfileTag("Vision.RmsNorm");
        vision.projector.setProfileTag(
            towerLin("Vision", visionLayer.embedDim, layerCfg.embedDim));

#ifdef TEST_INDEX_COPY
        net.imageCopy.setProfileTag("Vision.IndexCopy");
#else
        net.imageScatter.setProfileTag("Vision.RowSliceScatter");
#endif
    }

    // Text decoder
    const std::string textPrefix = net.visionTower ? "Text." : "";

    auto textTag = [&textPrefix](const char* tag)
    {
        return textPrefix + tag;
    };

    auto textLin = [&textPrefix](uint32_t in, uint32_t out)
    {
        return textPrefix
             + "Linear.["
             + std::to_string(in)
             + "x"
             + std::to_string(out)
             + "]";
    };

    for (uint32_t i = 0; i < decoderCfg.numLayers; ++i)
    {
        auto& b = net.decoder.layers[i];
        b.preAttnNorm.setProfileTag(textTag("RmsNorm"));
        b.mhAttn.toQ.setProfileTag(textLin(D, Dq));
        b.mhAttn.toK.setProfileTag(textLin(D, Dkv));
        b.mhAttn.toV.setProfileTag(textLin(D, Dkv));
        b.mhAttn.normQ->setProfileTag(textTag("RmsNorm.QK"));
        b.mhAttn.normK->setProfileTag(textTag("RmsNorm.QK"));
        b.mhAttn.ropeQ->setProfileTag(textTag("Rope"));
        b.mhAttn.ropeK->setProfileTag(textTag("Rope"));

        b.mhAttn.attn.setProfileTag(textTag(gemma3::ModelConfig::isGlobalLayer(i)
                    ? "Attn.global" : "Attn.local"));

        b.mhAttn.toOut.setProfileTag(textLin(Dq, D));
        b.postAttnNorm.setProfileTag(textTag("RmsNorm"));
        b.preFfnNorm.setProfileTag(textTag("RmsNorm"));
        b.ffn.gateValue.setProfileTag(textLin(D, 2 * Dff));
        b.ffn.gatedMul.setProfileTag(textTag("GatedMul"));
        b.ffn.out.setProfileTag(textLin(Dff, D));
        b.postFfnNorm.setProfileTag(textTag("RmsNorm"));
    }

    if (net.visionTower) net.embedding.setProfileTag("Text.Embedding");
    net.decoder.finalNorm.setProfileTag(textTag("RmsNorm"));
    net.lmHead.setProfileTag(textLin(D, V));
    net.topkSampling.setProfileTag(textTag("Sampling"));
}

}   // namespace
}   // namespace pyvai::examples

#endif // PYVAI_MODELS_GEMMA3_PROFILE_TAGS_H
