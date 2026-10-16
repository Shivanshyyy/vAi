#include "pyvai-internal.h"

#include "bind-models.h"
#include "utils/mnist.h"   // carries the model definition (see its header comment)


namespace pyvai {

void bindMnist(nb::module_& m)
{
    nb::class_<mnist::Model, vai::NeuralNet>(m, "MNISTModel")
        .def(
            "__init__",
            [](mnist::Model* self,
               vai::Dtype w_default_dtype,
               vai::Dtype o_default_dtype,
               std::optional<eva::Device> device)
            {
                // Core asserts (abort) reachable straight from these args.
                if (w_default_dtype == vai::Dtype::None || o_default_dtype == vai::Dtype::None)
                    throw nb::value_error("MNISTModel: dtypes must be F32 or F16, not NONE");
                // ConvolutionNode's dtype rule: fp16 activations need fp16 weights.
                if (w_default_dtype != vai::Dtype::Fp16 && o_default_dtype == vai::Dtype::Fp16)
                    throw nb::value_error(
                        "MNISTModel: o_default_dtype=F16 requires w_default_dtype=F16");
                vai::NeuralNetConfig cfg;
                cfg.device        = device.value_or(eva::Device{});
                cfg.wDefaultDtype = w_default_dtype;
                cfg.oDefaultDtype = o_default_dtype;
                new (self) mnist::Model(cfg);
            },
            nb::arg("w_default_dtype") = vai::Dtype::Fp16,
            nb::arg("o_default_dtype") = vai::Dtype::Fp32,
            nb::arg("device")          = nb::none(),
            "Build the MNIST model. Default weights are F16; the C++ demo "
            "(examples/mnist/main.cpp) runs F32 — pass w_default_dtype=Dtype.F32 "
            "to reproduce its reference logits exactly."
        )
        .def(
            "load_weight",
            [](mnist::Model& self, std::optional<std::string> path)
            {
                // Default: the committed checkpoint, same path the demo's
                // loadWeights<"jroot"> composes (examples/mnist/main.cpp).
                std::string p = path.value_or(VAI_DATA_DIR "/mnist/model.safetensors");
                ensureFileExists("load_weight", p);
                // Pure C++ from here — parse + upload can take seconds.
                nb::gil_scoped_release rel;
                utils::loadWeight(self, vai::SafetensorsParser(p.c_str()));
            },
            nb::arg("path") = nb::none(),
            "Load weights from a safetensors file path (default: the committed "
            "data/mnist checkpoint). Expects the checkpoint's key naming "
            "(conv1 / conv2 / fc)."
        )
        .def(
            "set_profile_tags",
            [](mnist::Model& self) { utils::setProfileTags(self); },
            "Tag every node for per-node GPU profiling. Tagging is what turns "
            "profiling on; tags may be set or changed at any point."
        );
}

}   // namespace pyvai
