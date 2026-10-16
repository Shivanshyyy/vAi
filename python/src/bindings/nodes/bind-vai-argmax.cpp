#include "bind-nodes.h"

#include "vai/nodes/argmax.h"

#include <optional>

namespace pyvai {

void bindArgmax(nb::module_& m)
{
    nb::class_<vai::node::Argmax, vai::Node>(m, "ArgmaxNode")
        .def(
            "__init__",
            [](vai::node::Argmax* self)
            {
                requireCurrentNet("ArgmaxNode");
                new (self) vai::node::Argmax(vai::node::Argmax::Config{
                });
            },
            "Greedy pick over 2D [M, N] logits of either dtype. The "
            "output holds raw uint32 indices, so read it with "
            "Tensor.get_data(\"uint32\") rather than to_numpy()."
        );
}

}   // namespace pyvai
