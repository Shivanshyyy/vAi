#include "bind-nodes.h"

#include "vai/nodes/linear.h"

#include <cstdint>
#include <optional>

namespace pyvai {

void bindLinear(nb::module_& m)
{
    nb::enum_<vai::ResidualOrder>(m, "ResidualOrder")
        .value("after_epilogue",  vai::ResidualOrder::AfterEpilogue)
        .value("before_epilogue", vai::ResidualOrder::BeforeEpilogue);

    nb::class_<vai::node::Linear, vai::Node>(m, "LinearNode")
        .def(
            "__init__",
            [](vai::node::Linear* self,
               uint32_t in_dim, uint32_t out_dim,
               bool has_bias,
               bool accept_residual, vai::ResidualOrder residual_order,
               vai::Epilogue epilogue,
               std::optional<vai::Dtype> w_dtype,
               std::optional<vai::Dtype> o_dtype,
               uint32_t split_size,
               std::optional<vai::Dtype> acc_dtype)
            {
                requireCurrentNet("LinearNode");
                if (split_size > 0 && split_size < 32)   // core asserts (aborts) on the tile path
                    throw nb::value_error("LinearNode: split_size must be 0 or >= 32");
                new (self) vai::node::Linear(vai::node::Linear::Config{
                    .inDim          = in_dim,
                    .outDim         = out_dim,
                    .hasBias        = has_bias,
                    .acceptResidual = accept_residual,
                    .residualOrder  = residual_order,
                    .epilogue       = epilogue,
                    .wDtype         = w_dtype.value_or(vai::Dtype::None),
                    .oDtype         = o_dtype.value_or(vai::Dtype::None),
                    .splitSize      = split_size,
                    .accDtype       = acc_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("in_dim"),
            nb::arg("out_dim"),
            nb::arg("has_bias")        = false,
            nb::arg("accept_residual") = false,
            nb::arg("residual_order")  = vai::ResidualOrder::AfterEpilogue,
            nb::arg("epilogue")        = vai::Epilogue::None,
            nb::arg("w_dtype")         = nb::none(),
            nb::arg("o_dtype")         = nb::none(),
            nb::arg("split_size")      = 0u,
            nb::arg("acc_dtype")       = nb::none(),
            "Dense layer. The bias (parameter slot \"bias\") is [M, out_dim] or rows "
            "that repeat over M, such as [out_dim]; it is added "
            "before the epilogue. accept_residual opens input slot \"residual\", added "
            "after the epilogue or before it per residual_order. Leaving a dtype None takes the net's default "
            "(w_default_dtype for weights/accumulator, in0's dtype for the output). "
            "split_size is the K length per split-K span (0 picks the kernel's "
            "own heuristic, otherwise >= 32); it only affects batched shapes "
            "(M > 8) — decode-step slabs ignore it."
        );
}

}   // namespace pyvai
