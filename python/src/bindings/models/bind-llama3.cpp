#include "pyvai-internal.h"

#include "bind-models.h"
#include "utils/llama3.h"

#include "models/llama3/model.h"
#include "models/llama3/meta-weights.h"


namespace pyvai {

void bindLlama3(nb::module_& m)
{
    // Preset names mirror the C++ ModelConfig factories (_3_2_1B() ...);
    // `name` is read-only — a static string naming the checkpoint folder
    // under data/.
    nb::class_<llama3::ModelConfig>(m, "Llama3Config")
        .def_static("_3_2_1B", &llama3::ModelConfig::_3_2_1B)
        .def_static("_3_2_3B", &llama3::ModelConfig::_3_2_3B)
        .def_prop_ro("name",
            [](const llama3::ModelConfig& self) -> std::string { return self.name; })
        .def_rw("vocab_size",          &llama3::ModelConfig::vocabSize)
        .def_rw("max_trained_context", &llama3::ModelConfig::maxTrainedContext)
        .def_rw("embed_dim",           &llama3::ModelConfig::embedDim)
        .def_rw("ffn_hidden_dim",      &llama3::ModelConfig::ffnHiddenDim)
        .def_rw("num_q_heads",           &llama3::ModelConfig::numQHeads)
        .def_rw("num_kv_heads",        &llama3::ModelConfig::numKVHeads)
        .def_rw("num_layers",          &llama3::ModelConfig::numLayers)
        // RoPE frequency knobs (ModelConfig.ropeConfig) for differently-scaled
        // checkpoints; head_dim is filled in per attention by the model.
        .def_prop_rw("rope_base",
            [](const llama3::ModelConfig& c) { return c.ropeConfig.base; },
            [](llama3::ModelConfig& c, float v) { c.ropeConfig.base = v; })
        .def_prop_rw("rope_factor",
            [](const llama3::ModelConfig& c) { return c.ropeConfig.factor; },
            [](llama3::ModelConfig& c, float v) { c.ropeConfig.factor = v; })
        .def_prop_rw("rope_low_freq_factor",
            [](const llama3::ModelConfig& c) { return c.ropeConfig.lowFreqFactor; },
            [](llama3::ModelConfig& c, float v) { c.ropeConfig.lowFreqFactor = v; })
        .def_prop_rw("rope_high_freq_factor",
            [](const llama3::ModelConfig& c) { return c.ropeConfig.highFreqFactor; },
            [](llama3::ModelConfig& c, float v) { c.ropeConfig.highFreqFactor = v; })
        .def_prop_rw("rope_original_max_pos",
            [](const llama3::ModelConfig& c) { return c.ropeConfig.originalMaxPos; },
            [](llama3::ModelConfig& c, uint32_t v) { c.ropeConfig.originalMaxPos = v; });

    nb::class_<llama3::Model, vai::NeuralNet>(m, "Llama3Model")
        .def(
            "__init__",
            [](llama3::Model* self,
               const llama3::ModelConfig& config,
               vai::Dtype w_default_dtype,
               vai::Dtype o_default_dtype,
               std::optional<eva::Device> device,
               uint32_t max_runtime_context)
            {
                // Core asserts (abort) reachable straight from these args.
                if (w_default_dtype == vai::Dtype::None || o_default_dtype == vai::Dtype::None)
                    throw nb::value_error("Llama3Model: dtypes must be F32 or F16, not NONE");
                if (config.numQHeads == 0 || config.embedDim % config.numQHeads != 0)
                    throw nb::value_error("Llama3Model: embed_dim must be divisible by num_q_heads");
                if (config.numKVHeads > 0 &&
                    (config.numKVHeads > config.numQHeads ||
                     config.numQHeads % config.numKVHeads != 0))
                    throw nb::value_error("Llama3Model: num_kv_heads must divide num_q_heads");
                if (max_runtime_context > config.maxTrainedContext)
                    throw nb::value_error(
                        "Llama3Model: max_runtime_context exceeds config.max_trained_context");
                if (config.ropeConfig.lowFreqFactor == config.ropeConfig.highFreqFactor)
                    throw nb::value_error(
                        "Llama3Model: rope_low_freq_factor and rope_high_freq_factor must differ");
                vai::NeuralNetConfig cfg;
                cfg.device       = device.value_or(eva::Device{});
                cfg.wDefaultDtype = w_default_dtype;
                cfg.oDefaultDtype = o_default_dtype;
                new (self) llama3::Model(cfg, config, {max_runtime_context});
            },
            nb::arg("config")              = llama3::ModelConfig::_3_2_1B(),
            nb::arg("w_default_dtype")       = vai::Dtype::Fp16,
            nb::arg("o_default_dtype")       = vai::Dtype::Fp32,
            nb::arg("device")              = nb::none(),
            nb::arg("max_runtime_context") = 0u,
            "Build a Llama 3 model. `max_runtime_context` bounds the KV-cache "
            "depth (0: the checkpoint's max_trained_context — 128K for Llama 3, "
            "so set a real budget for interactive use)."
        )
        .def(
            "load_weight",
            [](llama3::Model& self, std::optional<std::string> path)
            {
                // Default: the path loadWeights<"meta"> composes
                // (models/llama3/meta-weights.h) — data/llama3/<name>/model.safetensors.
                std::string p = path.value_or(
                    std::string(VAI_DATA_DIR "/llama3/") + self.modelCfg.name
                    + "/model.safetensors");
                ensureFileExists("load_weight", p);
                // Pure C++ from here — parse + upload can take seconds.
                nb::gil_scoped_release rel;
                llama3::meta::loadWeights(self, vai::SafetensorsParser(p.c_str()));
            },
            nb::arg("path") = nb::none(),
            "Load weights from a safetensors file path. With no path, reads "
            "the canonical data/llama3/<name>/model.safetensors."
        )
        .def(
            "set_profile_tags",
            [](llama3::Model& self) { utils::setProfileTags(self); },
            "Tag every node for per-node GPU profiling. Tagging is what turns "
            "profiling on; tags may be set or changed at any point."
        )
        .def("reset_cache",
             [](llama3::Model& self) {
                 nb::gil_scoped_release rel;   // waits the in-flight forward
                 self.resetCache();
             },
             "Rewind the KV cache to an empty context.")

        // The KV cache is not rewound between forwards, and nothing stops a
        // caller from running past the budget (the core does not check).
        .def_prop_ro("tokens_so_far",
            [](llama3::Model& self) { return self.tokensSoFar_; },
            "Tokens the KV cache currently holds; reset_cache() zeroes it.")
        .def_prop_ro("max_runtime_context",
            [](llama3::Model& self) { return self.runtimeCfg.maxRuntimeContext; },
            "The KV-cache budget, resolved: the ctor's max_runtime_context, or "
            "the checkpoint's max_trained_context when that was 0.");
}

}   // namespace pyvai
