#ifndef PYVAI_BIND_NODES_H
#define PYVAI_BIND_NODES_H

#include "binding-core.h"

#include "vai/engine/neural-net.h"

namespace pyvai {

// Node() latches the thread-local current net; null latch = segfault later.
inline void requireCurrentNet(const char* node)
{
    if (!vai::NeuralNet::current())
        throw std::runtime_error(
            std::string(node) + ": no active NeuralNet on this thread — "
            "construct nodes inside a NeuralNet (its __init__ makes it current)");
}

// One entry per engine node header (vai/nodes/vai-<node>.h). Keep this list and
// bindNodes()'s call order in step with that directory: a node missing here is
// simply unreachable from Python, and nothing in the build will say so.

// Shape / plumbing
void bindIdentity(nb::module_&);
void bindGate(nb::module_&);
void bindReshape(nb::module_&);
void bindFlatten(nb::module_&);
void bindConcat(nb::module_&);
void bindPermute(nb::module_&);
void bindSlice(nb::module_&);
void bindTypeCast(nb::module_&);

// Element-wise
void bindAdd(nb::module_&);
void bindMul(nb::module_&);
void bindGatedMul(nb::module_&);
void bindActivation(nb::module_&);

// Vision
void bindConvolution(nb::module_&);
void bindPool2d(nb::module_&);
void bindNearestUpsample(nb::module_&);

// Linear / normalization
void bindLinear(nb::module_&);
void bindLayerNorm(nb::module_&);
void bindRmsNorm(nb::module_&);
void bindBatchNorm(nb::module_&);

// Transformer
void bindEmbedding(nb::module_&);
void bindRope(nb::module_&);
void bindAttention(nb::module_&);

// Output heads
void bindArgmax(nb::module_&);
void bindTopKSampling(nb::module_&);

}   // namespace pyvai

#endif  // PYVAI_BIND_NODES_H
