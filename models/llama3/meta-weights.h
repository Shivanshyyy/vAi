/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each size reads data/llama3/ckpts/<ModelConfig::variant>/model.safetensors,
               taken from the matching Instruct repo on HuggingFace:
                 3.2-1B  meta-llama/Llama-3.2-1B-Instruct
                      or unsloth/Llama-3.2-1B-Instruct
                 3.2-3B  meta-llama/Llama-3.2-3B-Instruct
                      or unsloth/Llama-3.2-3B-Instruct
=============================================================================*/

#ifndef LLAMA3_META_WEIGHTS_H
#define LLAMA3_META_WEIGHTS_H

#include "model.h"
#include "vai/engine/weights.h"
#include <string>


namespace llama3::meta {

inline void loadWeights(Model& model, const vai::Checkpoint& ckpt)
{
    // Each tensor is loaded at the dtype declared by its owning node.
    auto loadLin = [&](auto& lin, const std::string& wKey)
    {
        // Llama3 linear layers have hasBias=false — only weight needs loading.
        vai::Dtype dt = lin.config_.wDtype;
        lin["weight"] = vai::Tensor(ckpt[wKey], dt);
    };
    auto loadRMSNorm = [&](auto& rn, const std::string& wKey)
    {
        vai::Dtype dt = rn.config_.wDtype;
        rn["scale"] = vai::Tensor(ckpt[wKey], dt);
    };

    // Token embedding
    vai::Dtype edt = model.embedding.config_.wDtype;
    model.embedding["weight"] = vai::Tensor(ckpt["model.embed_tokens.weight"], edt);

    // Layer layers
    for (uint32_t i = 0; i < model.modelCfg.languageCfg.numLayers; ++i)
    {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        auto& layer = model.decoder.layers[i];

        // RMSNorms (params loaded at the node's wDtype)
        loadRMSNorm(layer.norm1, prefix + "input_layernorm.weight");
        loadRMSNorm(layer.norm2, prefix + "post_attention_layernorm.weight");

        // Self attention projections (no bias in Llama3)
        loadLin(layer.mhAttn.toQ,   prefix + "self_attn.q_proj.weight");
        loadLin(layer.mhAttn.toK,   prefix + "self_attn.k_proj.weight");
        loadLin(layer.mhAttn.toV,   prefix + "self_attn.v_proj.weight");
        loadLin(layer.mhAttn.toOut, prefix + "self_attn.o_proj.weight");

        // FFN projections (SwiGLU). gate and value are one projection here, so
        // their rows are stacked — gate first, the half GatedMul activates.
        {
            vai::Dtype dt = layer.ffn.gateValue.config_.wDtype;
            layer.ffn.gateValue["weight"] =
                vai::concat({ckpt[prefix + "mlp.gate_proj.weight"],
                             ckpt[prefix + "mlp.up_proj.weight"]}, dt);
        }
        loadLin(layer.ffn.out, prefix + "mlp.down_proj.weight");
    }

    // Final RMSNorm
    loadRMSNorm(model.decoder.finalNorm, "model.norm.weight");

    // LM Head. Llama 3.2 ties it to the token embedding, so the checkpoint
    // carries no lm_head tensor of its own.
    model.lmHead["weight"] = model.embedding["weight"];
}

} // namespace llama3::meta


// loadWeights<"meta">(model) — composes the path in Checkpoint and loads it.
template<>
inline void loadWeights<"meta", llama3::Model>(llama3::Model& model)
{
    std::string dir = std::string(VAI_DATA_DIR "/llama3/ckpts/") + model.modelCfg.variant;
    llama3::meta::loadWeights(model, vai::Checkpoint(dir));
}

#endif // LLAMA3_META_WEIGHTS_H
