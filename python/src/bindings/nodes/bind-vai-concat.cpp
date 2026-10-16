#include "bind-nodes.h"

#include "vai/nodes/concat.h"

#include <cstdint>
#include <optional>

namespace pyvai {

void bindConcat(nb::module_& m)
{
    nb::class_<vai::node::Concat, vai::Node>(m, "ConcatNode")
        .def(
            "__init__",
            [](vai::node::Concat* self, int axis, uint32_t input_count,
               bool allow_partial_inputs,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("ConcatNode");
                if (input_count < 2)   // core ctor asserts (aborts) below 2
                    throw nb::value_error("ConcatNode: input_count must be >= 2");
                new (self) vai::node::Concat(vai::node::Concat::Config{
                    .axis                = axis,
                    .inputCount          = input_count,
                    .allowPartialInputs  = allow_partial_inputs,
                    .oDtype              = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("axis") = 0,
            nb::arg("input_count") = 2u,
            nb::arg("allow_partial_inputs") = false,
            nb::arg("o_dtype")     = nb::none(),
            "Concatenate in0..inN along `axis` (negative counts from the end). "
            "Inputs land as buffer copies when nothing but size-1 axes sit ahead "
            "of `axis` and no dtype conversion is asked for — out0 then aliases a "
            "lone input — and through a gather kernel otherwise, which caps "
            "input_count at 8. allow_partial_inputs lets a recording bring only "
            "some of them."
        );
}

}   // namespace pyvai
