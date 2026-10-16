#ifndef VAI_TESTS_NAIVE_NODE_H
#define VAI_TESTS_NAIVE_NODE_H

#include "vai/nodes/node-common.h"

namespace vai {

/////////////////////////////////////////////////////////////////////////////////////////
// NaiveLinearNode - one thread per output element, no tiling, no staging
// Input:  in0 [..., inDim]      weight [outDim, inDim]    bias [outDim]
// Output: out0 [..., outDim] = epilogue(in0 · weightT + bias)
//
// The reference Linear's tuned paths (slab / tile / coopmat) are checked
// against: written independently of them, so a shared misreading of the kernel
// geometry cannot make both agree on a wrong answer. Not part of vAi - it lives
// with the tests because nothing ships it.
/////////////////////////////////////////////////////////////////////////////////////////
class NaiveLinearNode : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct Config {
        uint32_t inDim;
        uint32_t outDim;

        bool hasBias = true;
        vai::Epilogue epilogue = vai::Epilogue::None;
        vai::Dtype wDtype = vai::Dtype::None;
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    NaiveLinearNode(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;
};


inline NaiveLinearNode::NaiveLinearNode(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("weight", NodeSlot::persistent);
    if (config_.hasBias) addSlot("bias", NodeSlot::persistent);

    config_.wDtype = (config_.wDtype == Dtype::None) ? wDefaultDtype() : config_.wDtype;
    config_.oDtype = (config_.oDtype == Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void NaiveLinearNode::prepare()
{
    const Tensor& inTen = (*this)["in0"];
    const Tensor& wTen  = (*this)["weight"];

    const Dtype iDtype = inTen.dtype();

    pipe = pipelinePool().request("linear-naive", {
        eva::constant_id<0>(config_.hasBias ? 1u : 0u),       // hasBias
        eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),     // typeO
        eva::constant_id<SPEC_ID_TYPE_I>(iDtype),             // typeI
        eva::constant_id<SPEC_ID_TYPE_W>(config_.wDtype),     // typeW (W & bias)
        eva::constant_id<SPEC_ID_ALIGNED>(config_.inDim % 4 == 0 ? 1u : 0u),   // aligned
        eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t) config_.epilogue)        // epilogue
    });

    if (!descSet)
        descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    const Tensor* bTen  = findSlot("bias") ? &(*this)["bias"] : nullptr;

    VAI_ASSERT(inTen.shape().back() == config_.inDim);
    VAI_ASSERT(wTen.dtype() == config_.wDtype);
    VAI_ASSERT(wTen.isShapeOf(config_.outDim, config_.inDim));
    VAI_ASSERT(!bTen || bTen->dtype() == config_.wDtype);
    VAI_ASSERT(!bTen || bTen->isShapeOf(config_.outDim));

    auto outShape = inTen.shape();
    outShape.back() = config_.outDim;
    (*this)["out0"] = Tensor(outShape, config_.oDtype);
}

inline void NaiveLinearNode::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();
    auto wBuf   = (*this)["weight"].buffer();
    auto bBuf   = findSlot("bias") ? (*this)["bias"].buffer() : eva::Buffer();

    // Single set: binding 0 = O(out), 1 = I(in), 2 = W(weight), 3 = bias.
    descSet.write({ outBuf, inBuf, wBuf, bBuf });

    uint32_t M = vai::outerSize((*this)["in0"].shape());

    uint32_t K = config_.inDim;
    uint32_t N = config_.outDim;
    uint32_t constants[] = { M, K, N };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(N, M)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

} // namespace vai

#endif // VAI_TESTS_NAIVE_NODE_H
