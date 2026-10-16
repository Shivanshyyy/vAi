/*=============================================================================
  Written by   qkrtmddn0507@vaistrata.com

  Checkpoint   Each size reads every *.safetensors in data/gemma3/ckpts/<ModelConfig::variant>/,
               taken from the matching Instruct repo on HuggingFace:

               Text-only models:
                 270M  google/gemma-3-270m-it
                    or unsloth/gemma-3-270m-it
                 1B    google/gemma-3-1b-it
                    or unsloth/gemma-3-1b-it

               Vision-language models:
                 4B    google/gemma-3-4b-it
                 12B   google/gemma-3-12b-it
=============================================================================*/

#ifndef GEMMA3_GOOGLE_WEIGHTS_H
#define GEMMA3_GOOGLE_WEIGHTS_H

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "model.h"
#include "vai/engine/weights.h"


namespace gemma3::google {

// -----------------------------------------------------------------------------
// Weight loader
// -----------------------------------------------------------------------------

inline void loadWeights(Model& model, const vai::Checkpoint& parser)
{
    const bool hasVision = model.modelCfg.visionCfg.has_value();

    // Text-only checkpoints use:
    //   model.*
    //
    // Gemma 3 VLM checkpoints use:
    //   language_model.model.*
    const std::string textRoot = hasVision ? "language_model.model." : "model.";

    // Each tensor is loaded at the dtype declared by its owning node.
    auto loadLin = [&](auto& lin, const std::string& wKey)
    {
        // Gemma3 text linear layers have hasBias=false.
        vai::Dtype dt = lin.config_.wDtype;
        lin["weight"] = vai::Tensor(parser[wKey], dt);
    };

    // Gemma stores the RMS scale centered on zero. The kernel scales by scale as
    // it is, so the +1 is folded in here — once, at load.
    auto loadRMSNorm = [&](auto& rn, const std::string& wKey)
    {
        vai::Tensor centered(parser[wKey], vai::Dtype::Fp32);

        const float* src = static_cast<const float*>(centered.hostData());

        std::vector<float> scale(src, src + centered.numElements());

        for (float& value : scale) value += 1.0f;

        rn["scale"] =
            vai::Tensor(centered.shape(), rn.config_.wDtype).setFloats(std::move(scale));
    };


    // -------------------------------------------------------------------------
    // Vision tower + multimodal projector
    //
    // Only Gemma 3 VLM checkpoints have these tensors.
    // -------------------------------------------------------------------------

    if (hasVision)
    {
        const VisionTowerConfig& cfg = *model.modelCfg.visionCfg;
        auto& vision = model.visionTower.value();

        auto loadLayerNorm = [&](auto& ln, const std::string& prefix)
        {
            vai::Dtype dt = ln.config_.wDtype;

            ln["gamma"] = vai::Tensor(parser[prefix + ".weight"], dt);
            ln["beta"] = vai::Tensor(parser[prefix + ".bias"], dt);
        };

        auto loadBiasedLin = [&](auto& lin, const std::string& prefix)
        {
            vai::Dtype dt = lin.config_.wDtype;

            lin["weight"] = vai::Tensor(parser[prefix + ".weight"], dt);
            lin["bias"] = vai::Tensor(parser[prefix + ".bias"], dt);
        };

        const std::string visionRoot = "vision_tower.vision_model.";

        // Patch embedding
        {
            auto& patch = vision.patchEmbed;

            vai::Dtype dt = patch.project.config_.wDtype;

            patch.project["weight"] = vai::Tensor(parser[visionRoot + "embeddings.patch_embedding.weight"], dt)
                    .reshape({cfg.layerCfg.embedDim,
                              cfg.imageChannels * cfg.patchSize * cfg.patchSize});

            patch.project["bias"] = vai::sum(
                parser[visionRoot + "embeddings.position_embedding.weight"],
                parser[visionRoot + "embeddings.patch_embedding.bias"], dt);
        }

        // Vision encoder layers
        for (uint32_t i = 0; i < cfg.numLayers; ++i)
        {
            const std::string prefix =
                visionRoot
                + "encoder.layers."
                + std::to_string(i)
                + ".";

            auto& layer = vision.encoder.layers[i];

            loadLayerNorm(layer.norm1, prefix + "layer_norm1");
            loadLayerNorm(layer.norm2, prefix + "layer_norm2");
            loadBiasedLin(layer.mhAttn.toQ, prefix + "self_attn.q_proj");
            loadBiasedLin(layer.mhAttn.toK, prefix + "self_attn.k_proj");
            loadBiasedLin(layer.mhAttn.toV, prefix + "self_attn.v_proj");
            loadBiasedLin(layer.mhAttn.toOut, prefix + "self_attn.out_proj");
            loadBiasedLin(layer.ffn.fc1, prefix + "mlp.fc1");
            loadBiasedLin(layer.ffn.fc2, prefix + "mlp.fc2");
        }

        // Final vision LayerNorm
        loadLayerNorm(vision.encoder.finalNorm, visionRoot + "post_layernorm");

        // Multimodal projector input norm
        loadRMSNorm(vision.imageNorm,"multi_modal_projector.mm_soft_emb_norm.weight");
        // Multimodal projector
        {
            vai::Dtype dt = vision.projector.config_.wDtype;
            vision.projector["weight"] = vai::Tensor(parser["multi_modal_projector." "mm_input_projection_weight"],dt).permute(1, 0);
        }
    }


    // -------------------------------------------------------------------------
    // Language model
    //
    // Shared by text-only and VLM Gemma 3 checkpoints.
    // Only the checkpoint prefix differs.
    // -------------------------------------------------------------------------

    // Token embedding
    {
        vai::Dtype dt = model.embedding.config_.wDtype;
        model.embedding["weight"] = vai::Tensor(parser[textRoot + "embed_tokens.weight"], dt);
    }

    // Decoder layers
    for (uint32_t i = 0; i < model.modelCfg.languageCfg.numLayers; ++i)
    {
        const std::string prefix = textRoot + "layers." + std::to_string(i) + ".";
        auto& layer = model.decoder.layers[i];

        // RMSNorms
        loadRMSNorm(layer.preAttnNorm, prefix + "input_layernorm.weight");
        loadRMSNorm(layer.postAttnNorm, prefix + "post_attention_layernorm.weight");
        loadRMSNorm(layer.preFfnNorm, prefix + "pre_feedforward_layernorm.weight");
        loadRMSNorm(layer.postFfnNorm, prefix + "post_feedforward_layernorm.weight");

        // Self attention projections
        loadLin(layer.mhAttn.toQ, prefix + "self_attn.q_proj.weight");
        loadLin(layer.mhAttn.toK, prefix + "self_attn.k_proj.weight");
        loadLin(layer.mhAttn.toV, prefix + "self_attn.v_proj.weight");
        loadLin(layer.mhAttn.toOut, prefix + "self_attn.o_proj.weight");

        // Per-head Q/K norms
        loadRMSNorm(*layer.mhAttn.normQ, prefix + "self_attn.q_norm.weight");
        loadRMSNorm(*layer.mhAttn.normK, prefix + "self_attn.k_norm.weight");

        // FFN projections (GeGLU).
        //
        // gate and value are represented as one projection internally, so the
        // checkpoint tensors are stacked row-wise: gate first, then up/value.
        {
            vai::Dtype dt = layer.ffn.gateValue.config_.wDtype;

            layer.ffn.gateValue["weight"] = vai::Tensor(parser[prefix + "mlp.gate_proj.weight"], dt)
                    .concat(vai::Tensor(parser[prefix + "mlp.up_proj.weight"],dt));
        }

        loadLin(layer.ffn.out, prefix + "mlp.down_proj.weight");
    }

    // Final RMSNorm
    loadRMSNorm(model.decoder.finalNorm, textRoot + "norm.weight");

    // LM Head.
    //
    // Gemma 3 ties it to the token embedding, so the checkpoint carries no
    // separate lm_head tensor.
    model.lmHead["weight"] = model.embedding["weight"];
}

} // namespace gemma3::google


// loadWeights<"google">(model)
//
// Text-only models:
//   data/gemma3/<name>/model.safetensors
//
// VLM models:
//   all *.safetensors under data/gemma3/<name>/
template<>
inline void loadWeights<"google", gemma3::Model>(gemma3::Model& model)
{
    const std::string dir = std::string(VAI_DATA_DIR "/gemma3/ckpts/")
        + model.modelCfg.variant;

    gemma3::google::loadWeights(model, vai::Checkpoint(dir));
}

#endif // GEMMA3_GOOGLE_WEIGHTS_H
