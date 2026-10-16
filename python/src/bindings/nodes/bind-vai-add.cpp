#include "bind-nodes.h"

#include "vai/nodes/add.h"

#include <optional>

namespace pyvai {

void bindAdd(nb::module_& m)
{
    nb::class_<vai::node::Add, vai::Node>(m, "AddNode")
        .def(
            "__init__",
            [](vai::node::Add* self,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("AddNode");
                new (self) vai::node::Add(vai::node::Add::Config{
                    .oDtype = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("o_dtype") = nb::none(),
            "Element-wise in0 + in1, the residual add. Both inputs must have "
            "identical shapes and dtypes."
        );
}

}   // namespace pyvai
