#include "bind-nodes.h"
#include "bind-kvcache-counter.h"

#include "vai/nodes/token-embedding.h"

#include <cstdint>
#include <optional>

namespace pyvai {

void bindTokenEmbedding(nb::module_& m)
{
    nb::class_<vai::node::TokenEmbedding, vai::Node>(m, "TokenEmbeddingNode")
        .def(
            "__init__",
            [](vai::node::TokenEmbedding* self,
               uint32_t vocab_size, uint32_t embed_dim,
               bool pos_embedding, uint32_t num_positions,
               std::optional<vai::Dtype> w_dtype,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("TokenEmbeddingNode");
                // required coupling: the posWeight slot is checked against
                // num_positions at forward (core asserts, aborting, on mismatch)
                if (pos_embedding && num_positions == 0)
                    throw nb::value_error(
                        "TokenEmbeddingNode: pos_embedding=True requires num_positions");
                new (self) vai::node::TokenEmbedding(vai::node::TokenEmbedding::Config{
                    .vocabSize    = vocab_size,
                    .embedDim     = embed_dim,
                    .posEmbedding = pos_embedding,
                    .numPositions = num_positions,
                    .wDtype       = w_dtype.value_or(vai::Dtype::None),
                    .oDtype       = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("vocab_size"),
            nb::arg("embed_dim"),
            nb::arg("pos_embedding") = false,
            nb::arg("num_positions") = 0u,
            nb::arg("w_dtype")       = nb::none(),
            nb::arg("o_dtype")       = nb::none(),   // lookup only: no input dtype
            "Token embedding, optionally fused with a positional one. in0 is uint32 "
            "token IDs [Tq]; out0 is [Tq, embed_dim]. pos_embedding=True adds "
            "pos_w[cached_count + n], which needs num_positions, the `posWeight` "
            "slot and set_kv_cache_counter(); with False none of the three exist."
        )
        .def(
            "set_kv_cache_counter",
            [](vai::node::TokenEmbedding& self, KVCacheCounter& counter) {
                self.setKvCacheCounter(counter.buf);
            },
            nb::arg("counter"),
            // The counter's dtor destroys its buffer — pin it to the node.
            nb::keep_alive<1, 2>()
        );
}

}   // namespace pyvai
