#include "bind-nodes.h"

#include "vai/nodes/convolution.h"

#include <cstdint>
#include <optional>

namespace pyvai {

void bindConvolution(nb::module_& m)
{
    nb::class_<vai::node::Convolution, vai::Node>(m, "ConvolutionNode")
        .def(
            "__init__",
            [](vai::node::Convolution* self,
               uint32_t in_channels, uint32_t out_channels,
               uint32_t kernel_size, uint32_t stride,
               bool has_bias, bool has_scale,
               vai::Epilogue epilogue,
               std::optional<vai::Dtype> w_dtype,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("ConvolutionNode");
                // The core ctor asserts (aborts) on each of these.
                if (in_channels == 0 || out_channels == 0)
                    throw nb::value_error("ConvolutionNode: channels must be >= 1");
                if (kernel_size == 0 || kernel_size % 2 == 0)
                    throw nb::value_error("ConvolutionNode: kernel_size must be odd");
                if (stride == 0)
                    throw nb::value_error("ConvolutionNode: stride must be >= 1");
                if (has_bias == has_batch_norm)
                    throw nb::value_error(
                        "ConvolutionNode: exactly one of has_bias / has_batch_norm "
                        "(BN fusion replaces the bias)");
                new (self) vai::node::Convolution(vai::node::Convolution::Config{
                    .inChannels  = in_channels,
                    .outChannels = out_channels,
                    .kernelSize = kernel_size,
                    .stride      = stride,
                    .hasBias     = has_bias,
                    .hasScale    = has_scale,
                    .epilogue    = epilogue,
                    .wDtype      = w_dtype.value_or(vai::Dtype::None),
                    .oDtype      = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("in_channels"),
            nb::arg("out_channels"),
            nb::arg("kernel_size"),
            nb::arg("stride"),
            nb::arg("has_bias") = true,
            nb::arg("has_scale") = false,
            nb::arg("epilogue") = vai::Epilogue::None,
            nb::arg("w_dtype")  = nb::none(),
            nb::arg("o_dtype")  = nb::none(),
            "has_bias and has_scale independently enable channel-wise bias and "
            "scale slots. The loader supplies prepared coefficients; BN is not "
            "computed by the node. fp16 activations require fp16 weights."
        )
        .def(
            "enable_residual_add",
            &vai::node::Convolution::enableResidualAdd,
            "Fuse a residual add into the conv by adding a `residual` input "
            "slot ([out_channels, Ho, Wo]). Only available on 3x3 stride-1 "
            "scaled convs with in_channels == out_channels; returns False "
            "(and changes nothing) otherwise. Call before wiring the slot."
        );
}

}   // namespace pyvai
