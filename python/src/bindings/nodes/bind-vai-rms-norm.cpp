#include "bind-nodes.h"

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
               std::optional<vai::Dtype> w_dtype,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("RmsNormNode");
                new (self) vai::node::RmsNorm(vai::node::RmsNorm::Config{
                    .eps    = eps,
                    .wDtype = w_dtype.value_or(vai::Dtype::Fp32),
                    .oDtype = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("eps")     = 1e-5f,
            nb::arg("w_dtype") = nb::none(),
            nb::arg("o_dtype") = nb::none(),
            "RMS normalization. Takes a `scale` slot (gain); no bias."
        );
}

}   // namespace pyvai
