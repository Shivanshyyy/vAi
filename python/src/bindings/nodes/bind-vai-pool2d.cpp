#include "bind-nodes.h"

#include "vai/nodes/pool2d.h"

#include <cstdint>
#include <optional>
#include <string>

namespace pyvai {

template <vai::PoolMode mode>
static void bindPool2dMode(nb::module_& m, const char* name, const char* doc)
{
    using Node = vai::node::Pool2d<mode>;

    nb::class_<Node, vai::Node>(m, name)
        .def(
            "__init__",
            [name](Node* self,
                   uint32_t kernel_size, uint32_t stride, uint32_t padding,
                   bool ceil_mode, bool count_include_pad, bool channels_last,
                   std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet(name);
                if (stride < 1)   // core divides by stride at prepare
                    throw nb::value_error((std::string(name) + ": stride must be >= 1").c_str());
                new (self) Node(typename Node::Config{
                    .kernelSize      = kernel_size,
                    .stride          = stride,
                    .padding         = padding,
                    .ceilMode        = ceil_mode,
                    .countIncludePad = count_include_pad,
                    .channelsLast    = channels_last,
                    .oDtype          = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("kernel_size"),
            nb::arg("stride"),
            nb::arg("padding") = 0u,
            nb::arg("ceil_mode") = false,
            nb::arg("count_include_pad") = true,
            nb::arg("channels_last") = false,
            nb::arg("o_dtype") = nb::none(),
            doc
        );
}

void bindPool2d(nb::module_& m)
{
    bindPool2dMode<vai::PoolMode::Max>(m, "MaxPool2dNode",
        "One square window on both spatial axes of [..., C, H, W], or [..., H, W, C] "
        "with channels_last: Ho = (H + 2*padding - kernel_size) / stride + 1, rounded "
        "up with ceil_mode, and Wo follows from W the same way. Samples falling "
        "outside the input are ignored.");
    bindPool2dMode<vai::PoolMode::Avg>(m, "AvgPool2dNode",
        "One square window on both spatial axes of [..., C, H, W], or [..., H, W, C] "
        "with channels_last: Ho = (H + 2*padding - kernel_size) / stride + 1, rounded "
        "up with ceil_mode, and Wo follows from W the same way. Padding adds zeros; "
        "with count_include_pad it also counts in the divisor, as PyTorch does.");
}

}   // namespace pyvai
