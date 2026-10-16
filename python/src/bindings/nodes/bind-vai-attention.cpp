#include "bind-nodes.h"
#include "bind-kvcache-counter.h"

#include "vai/nodes/attention.h"
#include "vai/groups/transformer.h"

#include <cstdint>
#include <optional>

namespace pyvai {

void bindAttention(nb::module_& m)
{
    nb::class_<vai::node::Attention, vai::Node>(m, "AttentionNode")
        .def(
            "__init__",
            [](vai::node::Attention* self,
               uint32_t num_q_heads, uint32_t head_dim,
               uint32_t kv_cache_size, uint32_t num_kv_heads, bool causal,
               std::optional<vai::Dtype> kv_cache_dtype,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("AttentionNode");
                // core ctor assert (abort): this throws instead
                if (num_kv_heads > 0 &&
                    (num_kv_heads > num_q_heads || num_q_heads % num_kv_heads != 0))
                    throw nb::value_error(
                        "AttentionNode: num_kv_heads must divide num_q_heads");
                if (!kv_cache_dtype)
                    throw nb::value_error("AttentionNode: kv_cache_dtype is required");
                new (self) vai::node::Attention(
                    vai::node::Attention::Config{
                        .headDim     = head_dim,
                        .numQHeads   = num_q_heads,
                        .numKVHeads  = num_kv_heads,
                        .kvCacheSize = kv_cache_size,
                        .kvCacheDtype = *kv_cache_dtype,
                        .oDtype      = o_dtype.value_or(vai::Dtype::None),
                    });
                // The node keeps the mask alive.
                self->setCausalMask(vai::createCausalMask(self->device(), causal ? 1u : 0u));
            },
            nb::arg("num_q_heads"),
            nb::arg("head_dim"),
            nb::arg("kv_cache_size") = 0u,   // 0 = no cache
            nb::arg("num_kv_heads")  = 0u,
            nb::arg("causal")        = true,
            nb::arg("kv_cache_dtype") = nb::none(),   // K / V cache; required
            nb::arg("o_dtype")       = nb::none(),
            "Attention over separate Q/K/V inputs. out0 is [Tq, H, head_dim] — "
            "token-major, so merging heads is a plain reshape. causal=False "
            "lets a query see the newer tokens of its own dispatch. "
            "kv_cache_size is how many tokens the KV cache holds and needs a "
            "set_kv_cache_counter() before the first forward; 0 keeps no "
            "cache and needs neither."
        )
        .def(
            "set_kv_cache_counter",
            [](vai::node::Attention& self, KVCacheCounter& counter) {
                self.setKvCacheCounter(counter.buf);
            },
            nb::arg("counter"),
            // The counter's dtor destroys its buffer — pin it to the node.
            nb::keep_alive<1, 2>(),
            "REQUIRED before the first forward when kv_cache_size > 0 — prepare() "
            "reads the counter then (a missing counter is a crash, not an "
            "error). Rejected when kv_cache_size is 0."
        );
}

}   // namespace pyvai
