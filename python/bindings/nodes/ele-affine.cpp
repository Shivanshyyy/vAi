#include "python/bindings/nodes/nodes.h"
#include "vai/nodes/ele-affine.h"
#include <optional>

namespace pyvai {

void bindEleAffine(nb::module_& m)
{
    nb::class_<vai::node::EleAffine, vai::Node>(m, "EleAffineNode")
        .def(
            "__init__",
            [](vai::node::EleAffine* self,
               bool has_scale, bool has_shift,
               bool scale_as_input, bool shift_as_input,
               std::optional<vai::Dtype> w_dtype,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("EleAffineNode");
                if (!has_scale && !has_shift)
                    throw nb::value_error(
                        "EleAffineNode: has_scale and has_shift cannot both be False");
                new (self) vai::node::EleAffine(vai::node::EleAffine::Config{
                    .hasScale     = has_scale,
                    .hasShift     = has_shift,
                    .scaleAsInput = scale_as_input,
                    .shiftAsInput = shift_as_input,
                    .wDtype       = w_dtype.value_or(vai::Dtype::None),
                    .oDtype       = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("has_scale")      = true,
            nb::arg("has_shift")      = true,
            nb::arg("scale_as_input") = true,
            nb::arg("shift_as_input") = true,
            nb::arg("w_dtype")        = nb::none(),
            nb::arg("o_dtype")        = nb::none(),
            "Element-wise out0 = scale * in0 + shift. has_scale / has_shift open the "
            "`scale` / `shift` slots: inputs, or parameters at w_dtype with "
            "scale_as_input / shift_as_input False. Each broadcasts to in0's shape "
            "right-aligned, as long as the axes it keeps are adjacent ([D] or [1, D] "
            "over [T, D], [C, 1, 1] over [C, H, W]). A gated residual x + g * y is "
            "in0=y, scale=g, shift=x."
        );
}

}   // namespace pyvai
