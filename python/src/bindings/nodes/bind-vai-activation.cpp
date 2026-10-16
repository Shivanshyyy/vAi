#include "bind-nodes.h"

#include "vai/nodes/activation.h"

#include <optional>

namespace pyvai {

void bindActivation(nb::module_& m)
{
    // `epilogue` is deliberately REQUIRED although the C++ header defaults it
    // to Epilogue::None — a standalone activation with "none" is a silent
    // pass-through, far more likely a bug than an intent from Python.
    nb::class_<vai::node::Activation, vai::Node>(m, "ActivationNode")
        .def(
            "__init__",
            [](vai::node::Activation* self, vai::Epilogue epilogue, bool in_place,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("ActivationNode");
                new (self) vai::node::Activation(vai::node::Activation::Config{
                    .epilogue = epilogue,
                    .inPlace  = in_place,
                    .oDtype   = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("epilogue"),
            nb::arg("in_place") = false,
            nb::arg("o_dtype")  = nb::none(),
            "Element-wise epilogue over in0. in_place=True aliases out0 onto in0's "
            "buffer and requires o_dtype to be in0's (or none); otherwise out0 gets its own storage."
        );
}

}   // namespace pyvai
