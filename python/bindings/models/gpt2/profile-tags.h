#ifndef PYVAI_MODELS_GPT2_PROFILE_TAGS_H
#define PYVAI_MODELS_GPT2_PROFILE_TAGS_H

// Verbatim copy of setProfileTags() from examples/gpt2/demo.cpp (the
// members it walks are not exposed to Python). Re-copy when the source changes.

#include "models/gpt2/model.h"
#include <string>

namespace pyvai::examples {
namespace {

void setProfileTags(gpt2::Model& net)
{
    const uint32_t D = net.modelCfg.languageCfg.layerCfg.embedDim;
    const uint32_t V = net.modelCfg.languageCfg.vocabSize;

    auto lin = [](uint32_t in, uint32_t out) {
        return "Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    for (uint32_t i = 0; i < net.modelCfg.languageCfg.numLayers; ++i)
    {
        auto& b = net.decoder.layers[i];
        b.norm1.setProfileTag("LayerNorm");
        b.mhAttn.toQKV.setProfileTag(lin(D, 3 * D));
        b.mhAttn.attn.setProfileTag("Attn");
        b.mhAttn.toOut.setProfileTag(lin(D, D));
        b.norm2.setProfileTag("LayerNorm");
        b.ffn.fc1.setProfileTag(lin(D, 4 * D));
        b.ffn.fc2.setProfileTag(lin(4 * D, D));
    }
    net.decoder.finalNorm.setProfileTag("LayerNorm");
    net.lmHead.setProfileTag(lin(D, V));
    net.topkSampling.setProfileTag("Sampling");
}

}   // namespace
}   // namespace pyvai::examples

#endif // PYVAI_MODELS_GPT2_PROFILE_TAGS_H
