#include "pyvai-internal.h"

#include "bind-models.h"
#include "utils/gpt2.h"

#include "models/gpt2/model.h"
#include "models/gpt2/openai-weights.h"


namespace pyvai {

void bindGPT2(nb::module_& m)
{
    // Preset names mirror the C++ ModelConfig factories (_124M() ...); `name`
    // is read-only — it is a static string naming the checkpoint folder under
    // data/, and the presets are the way to get a coherent config.
    nb::class_<gpt2::ModelConfig>(m, "GPT2Config")
        .def_static("_124M", &gpt2::ModelConfig::_124M)
        .def_static("_355M", &gpt2::ModelConfig::_355M)
        .def_static("_774M", &gpt2::ModelConfig::_774M)
        .def_static("_1_5B", &gpt2::ModelConfig::_1_5B)
        .def_prop_ro("name",
            [](const gpt2::ModelConfig& self) -> std::string { return self.name; })
        .def_rw("vocab_size",          &gpt2::ModelConfig::vocabSize)
        .def_rw("max_trained_context", &gpt2::ModelConfig::maxTrainedContext)
        .def_rw("embed_dim",           &gpt2::ModelConfig::embedDim)
        .def_rw("num_q_heads",           &gpt2::ModelConfig::numQHeads)
        .def_rw("num_layers",          &gpt2::ModelConfig::numLayers);

    nb::class_<gpt2::Model, vai::NeuralNet>(m, "GPT2Model")
        .def(
            "__init__",
            [](gpt2::Model* self,
               const gpt2::ModelConfig& config,
               vai::Dtype w_default_dtype,
               vai::Dtype o_default_dtype,
               std::optional<eva::Device> device,
               uint32_t max_runtime_context)
            {
                // Core asserts (abort) reachable straight from these args.
                if (w_default_dtype == vai::Dtype::None || o_default_dtype == vai::Dtype::None)
                    throw nb::value_error("GPT2Model: dtypes must be F32 or F16, not NONE");
                if (config.numQHeads == 0 || config.embedDim % config.numQHeads != 0)
                    throw nb::value_error("GPT2Model: embed_dim must be divisible by num_q_heads");
                if (max_runtime_context > config.maxTrainedContext)
                    throw nb::value_error(
                        "GPT2Model: max_runtime_context exceeds config.max_trained_context");
                vai::NeuralNetConfig cfg;
                cfg.device       = device.value_or(eva::Device{});
                cfg.wDefaultDtype = w_default_dtype;
                cfg.oDefaultDtype = o_default_dtype;
                new (self) gpt2::Model(cfg, config, {max_runtime_context});
            },
            nb::arg("config")              = gpt2::ModelConfig::_124M(),
            nb::arg("w_default_dtype")       = vai::Dtype::Fp16,
            nb::arg("o_default_dtype")       = vai::Dtype::Fp32,
            nb::arg("device")              = nb::none(),
            nb::arg("max_runtime_context") = 0u,
            "Build a GPT-2 model. `max_runtime_context` bounds the KV-cache "
            "depth (0: the checkpoint's max_trained_context)."
        )
        .def(
            "load_weight",
            [](gpt2::Model& self, std::optional<std::string> path)
            {
                // Default: the path loadWeights<"openai"> composes
                // (models/gpt2/openai-weights.h) — data/gpt2/<name>/model.safetensors.
                std::string p = path.value_or(
                    std::string(VAI_DATA_DIR "/gpt2/") + self.modelCfg.name
                    + "/model.safetensors");
                ensureFileExists("load_weight", p);
                // Pure C++ from here — parse + upload can take seconds.
                nb::gil_scoped_release rel;
                gpt2::openai::loadWeights(self, vai::SafetensorsParser(p.c_str()));
            },
            nb::arg("path") = nb::none(),
            "Load the upstream HuggingFace GPT-2 checkpoint (the fused c_attn "
            "projection loads into the fused QKV linear). With no path, reads "
            "the canonical data/gpt2/<name>/model.safetensors."
        )
        .def(
            "set_profile_tags",
            [](gpt2::Model& self) { utils::setProfileTags(self); },
            "Tag every node for per-node GPU profiling. Tagging is what turns "
            "profiling on; tags may be set or changed at any point."
        )
        .def("reset_cache",
             [](gpt2::Model& self) {
                 nb::gil_scoped_release rel;   // waits the in-flight forward
                 self.resetCache();
             },
             "Rewind the KV cache to an empty context.")

        // The KV cache is not rewound between forwards, and nothing stops a
        // caller from running past the budget (the core does not check).
        .def_prop_ro("tokens_so_far",
            [](gpt2::Model& self) { return self.tokensSoFar_; },
            "Tokens the KV cache currently holds; reset_cache() zeroes it.")
        .def_prop_ro("max_runtime_context",
            [](gpt2::Model& self) { return self.runtimeCfg.maxRuntimeContext; },
            "The KV-cache budget, resolved: the ctor's max_runtime_context, or "
            "the checkpoint's max_trained_context when that was 0.");
}

}   // namespace pyvai
