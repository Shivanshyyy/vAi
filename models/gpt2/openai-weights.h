/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each size reads data/gpt2/ckpts/<ModelConfig::variant>/model.safetensors,
               taken from the matching repo on HuggingFace:
                 124M   openai-community/gpt2
                 355M   openai-community/gpt2-medium
                 774M   openai-community/gpt2-large
                 1.5B   openai-community/gpt2-xl
=============================================================================*/

#ifndef GPT2_OPENAI_WEIGHTS_H
#define GPT2_OPENAI_WEIGHTS_H

#include "model.h"
#include "vai/engine/weights.h"
#include <string>


namespace gpt2::openai {

// HuggingFace GPT-2's checkpoint stores every linear transposed (PyTorch
// Conv1D: [in, out]); vAi wants [out, in], so loadLin below transposes each.
// The fused c_attn projection loads as-is — the model keeps Q, K and V fused.
inline void loadWeights(Model& model, const vai::Checkpoint& ckpt)
{
    auto loadLin = [&](vai::node::Linear& lin, const std::string& key)
    {
        vai::Dtype dt = lin.config_.wDtype;
        lin["weight"] = vai::Tensor(ckpt[key + ".weight"], dt).permute(1, 0);
        lin["bias"]   = vai::Tensor(ckpt[key + ".bias"],   dt);
    };

    auto loadLN = [&](auto& ln, const std::string& key)
    {
        vai::Dtype dt = ln.config_.wDtype;
        ln["gamma"] = vai::Tensor(ckpt[key + ".weight"], dt);
        ln["beta"]  = vai::Tensor(ckpt[key + ".bias"],   dt);
    };

    // Token and position embeddings are already [count, embedDim].
    vai::Dtype edt = model.embedding.config_.wDtype;
    model.embedding["weight"] = vai::Tensor(ckpt["wte.weight"], edt);
    model.embedding["posWeight"]   = vai::Tensor(ckpt["wpe.weight"], edt);

    // Layer layers
    for (uint32_t i = 0; i < model.modelCfg.languageCfg.numLayers; ++i)
    {
        std::string prefix = "h." + std::to_string(i) + ".";
        auto& layer = model.decoder.layers[i];

        loadLN(layer.norm1, prefix + "ln_1");
        loadLN(layer.norm2, prefix + "ln_2");

        loadLin(layer.mhAttn.toQKV, prefix + "attn.c_attn");
        loadLin(layer.mhAttn.toOut, prefix + "attn.c_proj");

        loadLin(layer.ffn.fc1, prefix + "mlp.c_fc");
        loadLin(layer.ffn.fc2, prefix + "mlp.c_proj");
    }

    // Final layer norm
    loadLN(model.decoder.finalNorm, "ln_f");

    // LM Head (weight tying with token embeddings)
    model.lmHead["weight"] = model.embedding["weight"];
}

} // namespace gpt2::openai


// loadWeights<"openai">(model) — composes the path in Checkpoint and loads it.
template<>
inline void loadWeights<"openai", gpt2::Model>(gpt2::Model& model)
{
    std::string dir = std::string(VAI_DATA_DIR "/gpt2/ckpts/") + model.modelCfg.variant;
    gpt2::openai::loadWeights(model, vai::Checkpoint(dir));
}

#endif // GPT2_OPENAI_WEIGHTS_H
