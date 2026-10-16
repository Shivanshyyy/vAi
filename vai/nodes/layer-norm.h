#ifndef VAI_NODES_LAYER_NORM_H
#define VAI_NODES_LAYER_NORM_H

#include "vai/nodes/node-common.h"

namespace vai::node {

/*=============================================================================
  LayerNorm - layer normalization over the feature axis, single kernel
  Input : activations [..., D]
  Output: normalized  [..., D]

  O[i, j] = (I[i, j] - mean(I[i, :])) * rsqrt(var(I[i, :]) + eps) * γ[j] + β[j]

  One workgroup per row (layer-norm); the row statistics come from a
  shifted 1-pass reduction (Chan–Golub–LeVeque) accumulated in fp32
  regardless of I/O dtypes. Two spec-selected variants share the kernel:

  - reduction: subgroupAdd two-level when the device has subgroup arithmetic
    (subgroup size pinned to max — Intel varies it otherwise), shared-memory
    tree fallback everywhere else.
  - loads: vec4 when I and O are fp32 and featSize % 4 == 0, scalar otherwise.

  Geometry / reduction / vec4 are selected per (featSize, device) in
  lnTuning(); the workgroup size depends on featSize, so the pipeline is
  requested in prepare() once the shape is known, and a tuning change
  re-requests it from the pool (no-op while the tuning stays put).
=============================================================================*/
class LayerNorm : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct PipeParams {
        uint32_t   wgSize   = 0;      // threads per row, ~4 elements each
        uint32_t   sgSize   = 0;      // forced subgroup size (0 = don't force; Intel)
        bool       sgReduce = false;  // subgroupAdd two-level reduction instead of the shared tree
        bool       aligned  = false;  // vec4 I/O loads (fp32, featSize % 4 == 0)
        vai::Dtype iDtype   = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        float eps = 1e-5f;
        bool hasBeta = true;
        vai::Dtype wDtype = vai::Dtype::Fp32;   // gamma / beta; the statistic is fp32 regardless
        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype
    } config_;

    LayerNorm(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


// Workgroup geometry / kernel variants from device queries + shape:
// wgSize = pow2ceil(featSize / 4) in [sg, min(16 * sg, 1024)], subgroupAdd
// when supported (subgroup size pinned — Intel varies it otherwise).
inline LayerNorm::PipeParams lnTuning(uint32_t featSize, Dtype iDtype, Dtype oDtype, eva::Device device)
{
    const uint32_t sg = device.maxSubgroupSize();

    // TODO (eva): maxComputeWorkGroupInvocations limit query — until then
    //             cap at 1024 (common desktop value; spec guarantees 128)
    const uint32_t cap = std::min(16u * sg, 1024u);
    uint32_t wg = sg;
    while (wg * 4u < featSize && wg < cap) wg *= 2u;        // pow2ceil(featSize / 4), [sg, cap]

    LayerNorm::PipeParams t;
    t.wgSize   = wg;
    t.sgReduce = device.supportsSubgroupArithmetic();
    t.aligned  = (iDtype == Dtype::Fp32 && oDtype == Dtype::Fp32 && featSize % 4u == 0u);
    t.sgSize   = t.sgReduce ? sg : 0u;
    t.iDtype   = iDtype;
    return t;
}


inline LayerNorm::LayerNorm(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("gamma", NodeSlot::persistent);
    if (config_.hasBeta) addSlot("beta", NodeSlot::persistent);

    config_.wDtype = (config_.wDtype == Dtype::None) ? wDefaultDtype() : config_.wDtype;
    config_.oDtype = (config_.oDtype == Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void LayerNorm::prepare()
{
    const Tensor& inTen = (*this)["in0"];
    const Tensor& gTen  = (*this)["gamma"];
    const Tensor* bTen  = findSlot("beta") ? &(*this)["beta"] : nullptr;

    uint32_t batchSize = outerSize(inTen.shape());
    uint32_t featSize  = inTen.shape().back();

    VAI_ASSERT(inTen.shape().size() >= 2);
    // run() folds rows over grid x/y; x*y covers any int batch on spec-minimum
    // (65535 per axis) devices, so this is defense against a broken query.
    VAI_ASSERT(uint64_t(batchSize) <= uint64_t(device().maxComputeWorkGroupCount()[0])
                                      * device().maxComputeWorkGroupCount()[1]);
    VAI_ASSERT(gTen.dtype() == config_.wDtype);
    VAI_ASSERT(gTen.isShapeOf(featSize));
    VAI_ASSERT(!bTen || bTen->dtype() == config_.wDtype);
    VAI_ASSERT(!bTen || bTen->isShapeOf(featSize));

    const Dtype iDtype = inTen.dtype();

    const PipeParams p = lnTuning(featSize, iDtype, config_.oDtype, device());
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("layer-norm", {
            eva::constant_id<0>(p.wgSize),                      // wgSize
            eva::constant_id<1>(p.sgReduce ? 1u : 0u),          // sgReduce
            eva::constant_id<2>(config_.hasBeta ? 1u : 0u),     // hasBeta
            eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),   // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype),           // typeI
            eva::constant_id<SPEC_ID_TYPE_W>(config_.wDtype),   // typeW (γ & β)
            eva::constant_id<SPEC_ID_ALIGNED>(p.aligned ? 1u : 0u)  // aligned (vec4 loads)
        }, { .requiredSubgroupSize = p.sgSize });
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = Tensor(inTen.shape(), config_.oDtype);
}

inline void LayerNorm::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();
    auto gBuf   = (*this)["gamma"].buffer();
    auto bBuf   = findSlot("beta") ? (*this)["beta"].buffer() : eva::Buffer();

    descSet.write({ outBuf, inBuf, gBuf, bBuf });

    const auto& shape = (*this)["in0"].shape();
    uint32_t batchSize = outerSize(shape);
    uint32_t featSize  = shape.back();

    // One workgroup per row, rows folded over grid x/y — x alone caps at 65535
    // on some devices (Intel), below a long prefill's row count. gy stays 1
    // (gx = batchSize, today's geometry) until the batch outgrows the x limit.
    const uint32_t limX = device().maxComputeWorkGroupCount()[0];
    const uint32_t gy   = std::max(1u, CEIL_DIV(batchSize, limX));
    const uint32_t gx   = CEIL_DIV(batchSize, gy);

    struct {
        uint32_t batchSize, featSize;
        float eps;
        uint32_t gridX;
    } constants = { batchSize, featSize, config_.eps, gx };

    cmdBuff
        .bindPipeline(pipe)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({descSet})
        .dispatch(gx, gy)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}

}   // namespace vai::node

#endif // VAI_NODES_LAYER_NORM_H
