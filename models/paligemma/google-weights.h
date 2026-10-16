/*=============================================================================
  Written by   oneway@vaistrata.com
  Checkpoint   Each resolution reads every *.safetensors file in
               data/paligemma/ckpts/<ModelConfig::variant>/, taken from the matching
               mix repo on HuggingFace:
                 3B-224  google/paligemma-3b-mix-224
                 3B-448  google/paligemma-3b-mix-448
=============================================================================*/

#ifndef PALIGEMMA_GOOGLE_WEIGHTS_H
#define PALIGEMMA_GOOGLE_WEIGHTS_H

#include "model.h"
#include "vai/engine/weights.h"
#include <string>
#include <vector>


namespace paligemma::google {

/////////////////////////////////////////////////////////////////////////////////////////
// Weight loading - maps HuggingFace safetensors key names to model members
/////////////////////////////////////////////////////////////////////////////////////////
inline void loadWeights(Model& model, const vai::Checkpoint& ckpt)
{
    // Each tensor is loaded at the dtype declared by its owning node.
    auto loadLin = [&](auto& lin, const std::string& prefix)
    {
        vai::Dtype dt = lin.config_.wDtype;
        lin["weight"] = vai::Tensor(ckpt[prefix + ".weight"], dt);
        if (lin.config_.hasBias)
            lin["bias"] = vai::Tensor(ckpt[prefix + ".bias"], dt);
    };
    auto loadLayerNorm = [&](auto& ln, const std::string& prefix)
    {
        vai::Dtype dt = ln.config_.wDtype;
        ln["gamma"] = vai::Tensor(ckpt[prefix + ".weight"], dt);
        ln["beta"]  = vai::Tensor(ckpt[prefix + ".bias"], dt);
    };
    // Gemma stores the RMS scale centered on zero. The kernel scales by scale as
    // it is, so the +1 is folded in here — once, at load.
    auto loadRMSNorm = [&](auto& rn, const std::string& wKey)
    {
        vai::Tensor centered(ckpt[wKey], vai::Dtype::Fp32);
        const float* src = static_cast<const float*>(centered.hostData());

        std::vector<float> scale(src, src + centered.numElements());
        for (float& value : scale)
            value += 1.0f;

        rn["scale"] = vai::Tensor(centered.shape(), rn.config_.wDtype).setFloats(std::move(scale));
    };

    // Vision tower (SigLIP)
    {
        const auto& modelCfg = model.modelCfg;
        const std::string root = "vision_tower.vision_model.";

        // The patch projection is stored as a [D, C, P, P] kernel, whose trailing axes
        // are the (c, kh, kw) order a patch row carries — so it loads as a matrix.
        auto& patch = model.patchEmbed;
        vai::Dtype patchDt = patch.project.config_.wDtype;
        patch.project["weight"] =
            vai::Tensor(ckpt[root + "embeddings.patch_embedding.weight"], patchDt)
                .reshape({modelCfg.visionCfg.layerCfg.embedDim,
                          modelCfg.visionCfg.imageChannels
                              * modelCfg.visionCfg.patchSize * modelCfg.visionCfg.patchSize});
        patch.project["bias"] = vai::sum(
            ckpt[root + "embeddings.position_embedding.weight"],
            ckpt[root + "embeddings.patch_embedding.bias"], patchDt);

        for (uint32_t i = 0; i < modelCfg.visionCfg.numLayers; ++i)
        {
            std::string prefix = root + "encoder.layers." + std::to_string(i) + ".";
            auto& layer = model.encoder.layers[i];

            loadLayerNorm(layer.norm1, prefix + "layer_norm1");
            loadLayerNorm(layer.norm2, prefix + "layer_norm2");

            loadLin(layer.mhAttn.toQ,   prefix + "self_attn.q_proj");
            loadLin(layer.mhAttn.toK,   prefix + "self_attn.k_proj");
            loadLin(layer.mhAttn.toV,   prefix + "self_attn.v_proj");
            loadLin(layer.mhAttn.toOut, prefix + "self_attn.out_proj");

            loadLin(layer.ffn.fc1, prefix + "mlp.fc1");
            loadLin(layer.ffn.fc2, prefix + "mlp.fc2");
        }

        loadLayerNorm(model.encoder.finalNorm, root + "post_layernorm");
    }

    // Multimodal projector
    loadLin(model.projector, "multi_modal_projector.linear");

    // Token embedding. The export pads the table to 257216 rows; the 64 past the
    // vocabulary were never trained, so they never reach the tied head that would
    // otherwise sample them.
    vai::Dtype edt = model.embedding.config_.wDtype;
    model.embedding["weight"] = vai::slice(
        ckpt["language_model.model.embed_tokens.weight"],
        0, model.modelCfg.languageCfg.vocabSize, edt);

    // Layer layers
    for (uint32_t i = 0; i < model.modelCfg.languageCfg.numLayers; ++i)
    {
        std::string prefix = "language_model.model.layers." + std::to_string(i) + ".";
        auto& layer = model.decoder.layers[i];

        // RMSNorms (params loaded at the node's wDtype)
        loadRMSNorm(layer.norm1, prefix + "input_layernorm.weight");
        loadRMSNorm(layer.norm2, prefix + "post_attention_layernorm.weight");

        // Self attention projections (no bias in Gemma)
        loadLin(layer.mhAttn.toQ,   prefix + "self_attn.q_proj");
        loadLin(layer.mhAttn.toK,   prefix + "self_attn.k_proj");
        loadLin(layer.mhAttn.toV,   prefix + "self_attn.v_proj");
        loadLin(layer.mhAttn.toOut, prefix + "self_attn.o_proj");

        // FFN projections (GeGLU). gate and value are one projection here, so
        // their rows are stacked — gate first, the half GatedMul activates.
        {
            vai::Dtype dt = layer.ffn.gateValue.config_.wDtype;
            layer.ffn.gateValue["weight"] = vai::concat(
                {ckpt[prefix + "mlp.gate_proj.weight"],
                 ckpt[prefix + "mlp.up_proj.weight"]}, dt);
        }
        loadLin(layer.ffn.out, prefix + "mlp.down_proj");
    }

    // Final RMSNorm
    loadRMSNorm(model.decoder.finalNorm, "language_model.model.norm.weight");

    // LM Head. Gemma ties it to the token embedding, so the checkpoint carries
    // no lm_head tensor of its own. The embedding's sqrt(D) scale lives in the
    // node rather than in the weights, which is what keeps the tie sound.
    model.lmHead["weight"] = model.embedding["weight"];
}

} // namespace paligemma::google


// loadWeights<"google">(model) — composes the path in Checkpoint and loads it.
template<>
inline void loadWeights<"google", paligemma::Model>(paligemma::Model& model)
{
    std::string dir = std::string(VAI_DATA_DIR "/paligemma/ckpts/") + model.modelCfg.variant;
    paligemma::google::loadWeights(model, vai::Checkpoint(dir));
}

#endif // PALIGEMMA_GOOGLE_WEIGHTS_H
