#ifndef PYVAI_MODELS_LLAMA3_PROFILE_TAGS_H
#define PYVAI_MODELS_LLAMA3_PROFILE_TAGS_H

// Verbatim copy of setProfileTags() from examples/llama3/demo.cpp (the
// members it walks are not exposed to Python). Re-copy when the source changes.

#include "models/llama3/model.h"
#include <string>

namespace pyvai::examples {
namespace {

void setProfileTags(llama3::Model& net)
{
    const auto& layerCfg = net.modelCfg.languageCfg.layerCfg;
    const uint32_t D    = layerCfg.embedDim;
    const uint32_t Dkv  = layerCfg.headDim * layerCfg.numKVHeads;
    const uint32_t Dff  = layerCfg.ffnHiddenDim;
    const uint32_t V    = net.modelCfg.languageCfg.vocabSize;

    auto lin = [](uint32_t in, uint32_t out) {
        return "Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    for (uint32_t i = 0; i < net.modelCfg.languageCfg.numLayers; ++i)
    {
        auto& b = net.decoder.layers[i];
        b.norm1.setProfileTag("RmsNorm");
        b.mhAttn.toQ.setProfileTag(lin(D, D));
        b.mhAttn.toK.setProfileTag(lin(D, Dkv));
        b.mhAttn.toV.setProfileTag(lin(D, Dkv));
        b.mhAttn.ropeQ->setProfileTag("Rope");
        b.mhAttn.ropeK->setProfileTag("Rope");
        b.mhAttn.attn.setProfileTag("Attn");
        b.mhAttn.toOut.setProfileTag(lin(D, D));
        b.norm2.setProfileTag("RmsNorm");
        b.ffn.gateValue.setProfileTag(lin(D, 2 * Dff));
        b.ffn.gatedMul.setProfileTag("GatedMul");
        b.ffn.out.setProfileTag(lin(Dff, D));
    }
    net.decoder.finalNorm.setProfileTag("RmsNorm");
    net.lmHead.setProfileTag(lin(D, V));
    net.topkSampling.setProfileTag("Sampling");
}

}   // namespace
}   // namespace pyvai::examples

#endif // PYVAI_MODELS_LLAMA3_PROFILE_TAGS_H
