#include "bind-nodes.h"

#include "vai/nodes/mul.h"

#include <optional>

namespace pyvai {

void bindMul(nb::module_& m)
{
    nb::class_<vai::node::Mul, vai::Node>(m, "MulNode")
        .def(
            "__init__",
            [](vai::node::Mul* self,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("MulNode");
                new (self) vai::node::Mul(vai::node::Mul::Config{
                    .oDtype = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("o_dtype") = nb::none(),
            "Element-wise product of in0 and in1 (both at in0's dtype)."
        );
}

}   // namespace pyvai
