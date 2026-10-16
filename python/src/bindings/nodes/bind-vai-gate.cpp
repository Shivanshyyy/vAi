#include "bind-nodes.h"

#include "vai/nodes/gate.h"

namespace pyvai {

void bindGate(nb::module_& m)
{
    nb::class_<vai::node::Gate, vai::Node>(m, "GateNode")
        .def(
            "__init__",
            [](vai::node::Gate* self)
            {
                requireCurrentNet("GateNode");
                new (self) vai::node::Gate();
            },
            "Conditional branch: in0 is republished on both `then` and `else`, "
             "which alias its buffer. One branch is recorded per command buffer, "
             "and every node reached only by the closed branch is dropped — it "
             "neither prepares nor runs. Leaving `else` unwired therefore drops "
             "the whole downstream while the gate is closed. Weights still upload "
             "in uploadModelParams(), which ignores gates, so a branch closed on the "
             "first forward has its weights on the device anyway. The branches "
             "need not agree on shape or dtype.")

        .def_prop_rw(
            "open",
            [](const vai::node::Gate& self) { return self.isOpen(); },
            [](vai::node::Gate& self, bool open) { self.setOpen(open); },
            "Which branch the next recording takes: True is `then`, False is "
            "`else`. Flipping it re-records the command buffer, so set it while "
            "the net is idle — on_forward_begin, or any point after the previous "
            "forward has completed."
        );
}

}   // namespace pyvai
