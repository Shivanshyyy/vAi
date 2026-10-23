#include "python/bindings/nodes/nodes.h"
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
                if (in_channels == 0 || out_channels == 0)
                    throw nb::value_error("ConvolutionNode: channels must be >= 1");
                if (kernel_size == 0 || kernel_size % 2 == 0)
                    throw nb::value_error("ConvolutionNode: kernel_size must be odd");
                if (stride == 0)
                    throw nb::value_error("ConvolutionNode: stride must be >= 1");
                // The core resolves None to the net's defaults, then asserts that
                // fp32 weights never write an fp16 output.
                const vai::NeuralNetConfig& net = vai::NeuralNet::current()->config();
                if (w_dtype.value_or(net.wDefaultDtype) != vai::Dtype::Fp16
                    && o_dtype.value_or(net.oDefaultDtype) == vai::Dtype::Fp16)
                    throw nb::value_error(
                        "ConvolutionNode: fp32 weights cannot write an fp16 output "
                        "(w_dtype / o_dtype, or the net's defaults)");
                new (self) vai::node::Convolution(vai::node::Convolution::Config{
                    .inChannels  = in_channels,
                    .outChannels = out_channels,
                    .kernelSize  = kernel_size,
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
            nb::arg("has_bias")  = true,
            nb::arg("has_scale") = false,
            nb::arg("epilogue")  = vai::Epilogue::None,
            nb::arg("w_dtype")   = nb::none(),
            nb::arg("o_dtype")   = nb::none(),
            "has_bias / has_scale add `bias` / `scale` slots [out_channels], applied "
            "as out0 = scale * conv + bias; a batch norm folds into them. fp16 "
            "activations require fp16 weights, and so does an fp16 output."
        )
        .def(
            "enable_residual_add",
            &vai::node::Convolution::enableResidualAdd,
            "Fuse a residual add through a new `residual` input slot [out_channels, "
            "Ho, Wo]; call before wiring it. Returns False and changes nothing unless "
            "the conv is 3x3, stride 1 and scaled with in_channels == out_channels."
        );
}

}   // namespace pyvai
