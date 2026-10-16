#include "bind-nodes.h"
#include "bind-kvcache-counter.h"

#include "vai/nodes/rope.h"

#include <cstdint>
#include <optional>

namespace pyvai {

void bindRope(nb::module_& m)
{
    nb::class_<vai::node::Rope, vai::Node>(m, "RopeNode")
        .def(
            "__init__",
            [](vai::node::Rope* self,
               uint32_t head_dim,
               float base,
               float factor,
               float low_freq_factor,
               float high_freq_factor,
               uint32_t original_max_pos,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("RopeNode");
                if (head_dim < 2 || head_dim % 2 != 0)   // core ctor asserts (aborts)
                    throw nb::value_error("RopeNode: head_dim must be even and >= 2");
                new (self) vai::node::Rope(vai::node::Rope::Config{
                    .cfg = {
                        // The scaling knobs below are the Llama3 ones, and factor
                        // defaults to 1, which is how this binding spells plain RoPE.
                        .type            = vai::RopeType::Llama3,
                        .base            = base,
                        .factor          = factor,
                        .lowFreqFactor   = low_freq_factor,
                        .highFreqFactor  = high_freq_factor,
                        .originalMaxPos  = original_max_pos,
                    },
                    .headDim         = head_dim,
                    .oDtype          = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("head_dim"),
            nb::arg("base")             = 10000.0f,
            nb::arg("factor")           = 1.0f,
            nb::arg("low_freq_factor")  = 1.0f,
            nb::arg("high_freq_factor") = 4.0f,
            nb::arg("original_max_pos") = 8192u,
            nb::arg("o_dtype")          = nb::none(),
            "Rotary position embedding. head_dim must be even and >= 2. factor=1 "
            "disables frequency scaling; the remaining knobs only matter when "
            "it does not."
        )
        .def(
            "set_kv_cache_counter",
            [](vai::node::Rope& self, KVCacheCounter& counter) {
                self.setKvCacheCounter(counter.buf);
            },
            nb::arg("counter"),
            // The counter's dtor destroys its buffer — pin it to the node.
            nb::keep_alive<1, 2>(),
            "Bind the KV-cache token counter that supplies this forward's "
            "starting position. Required before the first forward."
        );
}

}   // namespace pyvai
