#include "pyvai-internal.h"

#include "bind-nodes.h"
#include "bind-kvcache-counter.h"

#include "vai/nodes/node-common.h"

namespace pyvai {

// Thin aggregator. Each node is bound in its own bindings/nodes/bind-vai-<node>.cpp
// (filename mirrors the engine header vai/nodes/vai-<node>.h); this just registers
// the shared enum and calls the per-node entry points in a deterministic order.
void bindNodes(nb::module_& m)
{
    // vai::Epilogue is shared by many nodes as an argument type; bind it first so
    // the enum is registered before any node that references it.
    nb::enum_<vai::Epilogue>(m, "Epilogue")
        .value("none",      vai::Epilogue::None)
        .value("relu",      vai::Epilogue::Relu)
        .value("gelu",      vai::Epilogue::Gelu)
        .value("gelu_tanh", vai::Epilogue::GeluTanh)
        .value("silu",      vai::Epilogue::Silu)
        .value("sigmoid",   vai::Epilogue::Sigmoid);

    // KVCacheCounter is consumed by TokenEmbeddingNode / RopeNode /
    // AttentionNode[Fused] as an argument type, so register it before
    // any of those nodes bind.
    bindKVCacheCounter(m);

    bindIdentity(m);
    bindGate(m);
    bindReshape(m);
    bindFlatten(m);
    bindConcat(m);
    bindPermute(m);
    bindSlice(m);
    bindTypeCast(m);

    bindAdd(m);
    bindMul(m);
    bindGatedMul(m);
    bindActivation(m);

    bindConvolution(m);
    bindPool2d(m);
    bindNearestUpsample(m);

    bindLinear(m);
    bindLayerNorm(m);
    bindRmsNorm(m);
    bindBatchNorm(m);

    bindTokenEmbedding(m);
    bindRope(m);
    bindAttention(m);

    bindArgmax(m);
    bindTopKSampling(m);
}

}   // namespace pyvai
