#ifndef PYVAI_UTILS_LLAMA3_H
#define PYVAI_UTILS_LLAMA3_H

// TODO: temporary copy — binding this is not possible right now. The model's
// setProfileTags() is a free function in its demo program
// (examples/llama3/demo.cpp), and the model members it walks are not exposed
// to Python, so the body below is copied verbatim instead. Re-copy it when
// the source changes; replace this file once a bindable path exists.
// (The weight loader that used to be copied here now ships upstream as
// models/llama3/meta-weights.h — the bindings call it directly.)

#include "models/llama3/model.h"

#include <string>

namespace pyvai::utils {
namespace {

using namespace vai;

void setProfileTags(llama3::Model& net)
{
    const auto& layerCfg = net.modelCfg.languageCfg.layerCfg;
    const uint32_t D    = layerCfg.embedDim;
    const uint32_t Dh   = layerCfg.headDim;
    const uint32_t H    = D / Dh;                      // numQHeads is derived, not configured
    const uint32_t Hkv  = layerCfg.numKVHeads;         // GQA
    const uint32_t Dkv  = Hkv * Dh;                    // K/V projection
    const uint32_t Dff  = layerCfg.ffnHiddenDim;
    const uint32_t V    = net.modelCfg.languageCfg.vocabSize;

    auto lin = [](uint32_t in, uint32_t out) {
        return "Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    for (uint32_t i = 0; i < net.modelCfg.languageCfg.numLayers; ++i)
    {
        auto& b = net.blocks[i];
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
    net.finalNorm.setProfileTag("RmsNorm");
    net.lmHead.setProfileTag(lin(D, V));
    net.topkSampling.setProfileTag("Sampling");
}

}   // namespace
}   // namespace pyvai::utils

#endif  // PYVAI_UTILS_LLAMA3_H
