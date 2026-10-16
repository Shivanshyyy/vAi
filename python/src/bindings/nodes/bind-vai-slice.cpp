#include "bind-nodes.h"

#include "vai/nodes/slice.h"

#include <cstdint>
#include <optional>

namespace pyvai {

void bindSlice(nb::module_& m)
{
    nb::class_<vai::node::Slice, vai::Node>(m, "SliceNode")
        .def(
            "__init__",
            [](vai::node::Slice* self, int start, uint32_t count, uint32_t axis,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("SliceNode");
                if (count < 1)   // core ctor asserts (aborts) otherwise
                    throw nb::value_error("SliceNode: count must be >= 1");
                new (self) vai::node::Slice(vai::node::Slice::Config{
                    .axis   = axis,
                    .start  = start,
                    .count  = count,
                    .oDtype = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("start"),
            nb::arg("count")   = 1u,
            nb::arg("axis")    = 0u,
            nb::arg("o_dtype") = nb::none(),
            "Take `count` consecutive indices from `axis`, starting at `start`. "
            "A negative start counts from the end, so SliceNode(-1) is the last "
            "index — the usual decode-step slice. On axis 0 the run is contiguous: "
            "out0 aliases it when the offset meets device alignment and is one copy "
            "otherwise. A deeper axis leaves it strided and a gather kernel reads it."
        );
}

}   // namespace pyvai
