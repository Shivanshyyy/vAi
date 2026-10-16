#include "bind-nodes.h"

#include "vai/nodes/topk-sampling.h"

#include <cstdint>
#include <optional>

namespace pyvai {

void bindTopKSampling(nb::module_& m)
{
    nb::class_<vai::node::TopKSampling, vai::Node>(m, "TopKSamplingNode")
        .def(
            "__init__",
            [](vai::node::TopKSampling* self,
               uint32_t k,
               float temperature)
            {
                requireCurrentNet("TopKSamplingNode");
                // core ctor asserts (aborts) outside [1, maxK]
                if (k < 1 || k > vai::node::TopKSampling::maxK)
                    throw nb::value_error("TopKSamplingNode: k must be in [1, 128]");
                new (self) vai::node::TopKSampling(vai::node::TopKSampling::Config{
                    .k           = k,
                    .temperature = temperature,
                });
            },
            nb::arg("k")           = 40u,
            nb::arg("temperature") = 1.0f,
            "Top-k sampling over the logits (k <= 128, the kernel's cap). The "
            "output holds raw uint32 indices — read it with "
            "Tensor.get_data(\"uint32\")."
        )
        .def(
            "set_uniform",
            &vai::node::TopKSampling::setUniform,
            nb::arg("u"),
            "Feed this forward's uniform random draw, in [0, 1). The node does "
            "not generate randomness itself: call this before every forward, "
            "or every step samples identically."
        );
}

}   // namespace pyvai
