#include "python/bindings/nodes/nodes.h"
#include "vai/nodes/rms-norm.h"
#include <optional>

namespace pyvai {

void bindRmsNorm(nb::module_& m)
{
    nb::class_<vai::node::RmsNorm, vai::Node>(m, "RmsNormNode")
        .def(
            "__init__",
            [](vai::node::RmsNorm* self,
               float eps,
               bool has_scale, bool has_shift, bool coeffs_as_inputs,
               std::optional<vai::Dtype> w_dtype,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("RmsNormNode");
                new (self) vai::node::RmsNorm(vai::node::RmsNorm::Config{
                    .eps            = eps,
                    .hasScale       = has_scale,
                    .hasShift       = has_shift,
                    .coeffsAsInputs = coeffs_as_inputs,
                    .wDtype         = w_dtype.value_or(vai::Dtype::Fp32),   // the core's default
                    .oDtype         = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("eps")              = 1e-5f,
            nb::arg("has_scale")        = true,
            nb::arg("has_shift")        = false,
            nb::arg("coeffs_as_inputs") = false,
            nb::arg("w_dtype")          = nb::none(),
            nb::arg("o_dtype")          = nb::none(),
            "RMS normalization of [..., feature] over the feature axis: x * "
            "rsqrt(mean(x^2) + eps) * scale + shift. has_scale / has_shift open the "
            "`scale` / `shift` slots [feature]: parameters at w_dtype (None = F32), "
            "or inputs computed upstream with coeffs_as_inputs=True. Row statistics "
            "accumulate in fp32 whatever the I/O dtypes."
        );
}

}   // namespace pyvai
