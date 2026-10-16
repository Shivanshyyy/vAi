#include "bind-nodes.h"

#include "vai/nodes/gated-mul.h"

#include <optional>

namespace pyvai {

void bindGatedMul(nb::module_& m)
{
    nb::class_<vai::node::GatedMul, vai::Node>(m, "GatedMulNode")
        .def(
            "__init__",
            [](vai::node::GatedMul* self,
               vai::Epilogue activation,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("GatedMulNode");
                new (self) vai::node::GatedMul(vai::node::GatedMul::Config{
                    .activation = activation,
                    .oDtype     = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("activation") = vai::Epilogue::Silu,
            nb::arg("o_dtype")    = nb::none(),
            "Gated FFN tail: act(gate) * up, where a single input carries the "
            "gate and up halves split on the trailing axis (in0 [..., 2H] -> "
            "out0 [..., H]). Pairs with one LinearNode producing both "
            "projections at once."
        );
}

}   // namespace pyvai
