#include "bind-nodes.h"

#include "vai/nodes/batch-norm.h"

#include <cstdint>
#include <optional>

namespace pyvai {

void bindBatchNorm(nb::module_& m)
{
    nb::class_<vai::node::BatchNorm, vai::Node>(m, "BatchNormNode")
        .def(
            "__init__",
            [](vai::node::BatchNorm* self,
               uint32_t channels,
               float eps,
               std::optional<vai::Dtype> w_dtype,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("BatchNormNode");
                new (self) vai::node::BatchNorm(vai::node::BatchNorm::Config{
                    .channels = channels,
                    .eps      = eps,
                    .wDtype   = w_dtype.value_or(vai::Dtype::None),
                    .oDtype   = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("channels"),
            nb::arg("eps")     = 1e-5f,
            nb::arg("w_dtype") = nb::none(),
            nb::arg("o_dtype") = nb::none(),
            "Inference-time batch norm. Takes gamma / beta / mean / variance "
            "slots, all at w_dtype."
        );
}

}   // namespace pyvai
