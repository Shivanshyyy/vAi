#include "python/bindings/module.h"
#include "python/bindings/nodes/nodes.h"
#include "python/bindings/nodes/kvcache-counter.h"
#include "vai/nodes/node-common.h"

namespace pyvai {

// Call order must match the list in nodes.h.
void bindNodes(nb::module_& m)
{
    // Epilogue and KVCacheCounter are argument types of nodes below: register them first.
    nb::enum_<vai::Epilogue>(m, "Epilogue")
        .value("NONE",      vai::Epilogue::None)
        .value("RELU",      vai::Epilogue::Relu)
        .value("GELU",      vai::Epilogue::Gelu,     "Exact, through erf.")
        .value("GELU_TANH", vai::Epilogue::GeluTanh, "The tanh approximation (gelu_pytorch_tanh).")
        .value("SILU",      vai::Epilogue::Silu)
        .value("SIGMOID",   vai::Epilogue::Sigmoid);

    bindKVCacheCounter(m);

    bindIdentity(m);
    bindGate(m);
    bindReshape(m);
    bindFlatten(m);
    bindConcat(m);
    bindPermute(m);
    bindSlice(m);
    bindRowSliceScatter(m);
    bindIndexCopy(m);
    bindTypeCast(m);

    bindAdd(m);
    bindMul(m);
    bindGatedMul(m);
    bindEleAffine(m);
    bindActivation(m);

    bindConvolution(m);
    bindPool2d(m);
    bindNearestUpsample(m);

    bindLinear(m);
    bindLayerNorm(m);
    bindRmsNorm(m);
    bindBatchNorm(m);

    bindTokenEmbedding(m);
    bindTimeEmbedding(m);
    bindCyclicTableReader(m);
    bindRope(m);
    bindAttention(m);

    bindArgmax(m);
    bindTopKSampling(m);
}

}   // namespace pyvai
