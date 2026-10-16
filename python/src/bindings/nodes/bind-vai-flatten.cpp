#include "bind-nodes.h"

#include "vai/nodes/flatten.h"

namespace pyvai {

void bindFlatten(nb::module_& m)
{
    nb::class_<vai::node::Flatten, vai::Node>(m, "FlattenNode")
        .def(
            "__init__",
            [](vai::node::Flatten* self)
            {
                requireCurrentNet("FlattenNode");
                new (self) vai::node::Flatten();
            },
            "Collapse the input to a single axis, keeping element order. "
            "out0 aliases in0's buffer.");
}

}   // namespace pyvai
