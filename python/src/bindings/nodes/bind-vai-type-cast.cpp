#include "bind-nodes.h"

#include "vai/nodes/type-cast.h"

#include <optional>

namespace pyvai {

void bindTypeCast(nb::module_& m)
{
    nb::class_<vai::node::TypeCast, vai::Node>(m, "TypeCastNode")
        .def(
            "__init__",
            [](vai::node::TypeCast* self, vai::Dtype o_dtype)
            {
                requireCurrentNet("TypeCastNode");
                new (self) vai::node::TypeCast(vai::node::TypeCast::Config{
                    .oDtype = o_dtype,
                });
            },
            nb::arg("o_dtype"),
            "Element-wise dtype cast of in0 to o_dtype (any fp32 / fp16 combination). "
             "When in0 is already at o_dtype it is a no-op and out0 aliases in0's buffer.");
}

}   // namespace pyvai
