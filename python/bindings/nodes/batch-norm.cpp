#include "python/bindings/nodes/nodes.h"
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
            "Inference-time batch norm over [channels, H, W]: out0 = gamma[c] * "
            "(in0 - mean[c]) * rsqrt(variance[c] + eps) + beta[c]. The `gamma`, "
            "`beta`, `mean` and `variance` slots are each [channels] at w_dtype."
        );
}

}   // namespace pyvai
