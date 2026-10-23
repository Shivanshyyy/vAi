#include "python/bindings/nodes/nodes.h"
#include "vai/nodes/layer-norm.h"
#include <optional>

namespace pyvai {

void bindLayerNorm(nb::module_& m)
{
    nb::class_<vai::node::LayerNorm, vai::Node>(m, "LayerNormNode")
        .def(
            "__init__",
            [](vai::node::LayerNorm* self,
               float eps, bool has_beta,
               std::optional<vai::Dtype> w_dtype,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("LayerNormNode");
                new (self) vai::node::LayerNorm(vai::node::LayerNorm::Config{
                    .eps     = eps,
                    .hasBeta = has_beta,
                    .wDtype  = w_dtype.value_or(vai::Dtype::Fp32),
                    .oDtype  = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("eps")      = 1e-5f,
            nb::arg("has_beta") = true,
            nb::arg("w_dtype")  = nb::none(),
            nb::arg("o_dtype")  = nb::none(),
            "Normalize an [..., feature] input (rank >= 2) over its last axis: "
            "(x - mean) * rsqrt(var + eps) * gamma + beta. has_beta=False drops the "
            "`beta` slot. Row statistics accumulate in fp32 whatever the I/O dtypes."
        );
}

}   // namespace pyvai
