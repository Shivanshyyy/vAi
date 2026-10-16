#ifndef VAI_NODES_BATCH_NORM_H
#define VAI_NODES_BATCH_NORM_H

#include "vai/nodes/node-common.h"

namespace vai::node {


/*=============================================================================
  BatchNorm - batch normalization (inference mode)
  Input:  in0 [C, H, W]
  Output: γ[c] * (in0 - μ[c]) * rsqrt(σ²[c] + eps) + β[c]

  gamma, beta, mean, and variance are one-dimensional tensors of length C. This is
  inference-only normalization; stored statistics are not updated.
=============================================================================*/
class BatchNorm : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        vai::Dtype iDtype = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        uint32_t channels;
        float eps = 1e-5f;
        vai::Dtype wDtype = vai::Dtype::None;
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    BatchNorm(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


inline BatchNorm::BatchNorm(Config config)
: config_(config)
{
    addSlot("out0",     NodeSlot::output);
    addSlot("in0",      NodeSlot::input);
    addSlot("gamma",    NodeSlot::persistent);
    addSlot("beta",     NodeSlot::persistent);
    addSlot("mean",     NodeSlot::persistent);
    addSlot("variance", NodeSlot::persistent);

    config_.wDtype = (config_.wDtype == Dtype::None) ? wDefaultDtype() : config_.wDtype;
    config_.oDtype = (config_.oDtype == Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void BatchNorm::prepare()
{
    uint32_t C = config_.channels;

    const Tensor& inTen = (*this)["in0"];

    const Dtype iDtype = inTen.dtype();

    const PipeParams p = { .iDtype = iDtype };
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("batchnorm", {
            eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),   // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype),           // typeI
            eva::constant_id<SPEC_ID_TYPE_W>(config_.wDtype)    // typeW (γ/β/μ/σ²)
        });
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }
    const Tensor& gTen  = (*this)["gamma"];
    const Tensor& bTen  = (*this)["beta"];
    const Tensor& mTen  = (*this)["mean"];
    const Tensor& vTen  = (*this)["variance"];

    VAI_ASSERT(inTen.isShapeOf(C, -1, -1));
    VAI_ASSERT(gTen.dtype() == config_.wDtype);
    VAI_ASSERT(gTen.isShapeOf(C));
    VAI_ASSERT(bTen.dtype() == config_.wDtype);
    VAI_ASSERT(bTen.isShapeOf(C));
    VAI_ASSERT(mTen.dtype() == config_.wDtype);
    VAI_ASSERT(mTen.isShapeOf(C));
    VAI_ASSERT(vTen.dtype() == config_.wDtype);
    VAI_ASSERT(vTen.isShapeOf(C));

    (*this)["out0"] = Tensor(inTen.shape(), config_.oDtype);
}

inline void BatchNorm::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();
    auto gBuf   = (*this)["gamma"].buffer();
    auto bBuf   = (*this)["beta"].buffer();
    auto mBuf   = (*this)["mean"].buffer();
    auto vBuf   = (*this)["variance"].buffer();

    descSet.write({ outBuf, inBuf, gBuf, bBuf, mBuf, vBuf });

    const auto& shape = (*this)["in0"].shape();
    uint32_t HW = shape[1] * shape[2];
    uint32_t N  = config_.channels * HW;

    struct {
        uint32_t C, HW;
        float eps;
    } constants = { config_.channels, HW, config_.eps };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(N)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_BATCH_NORM_H
