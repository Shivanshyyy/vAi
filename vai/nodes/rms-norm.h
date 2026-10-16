#ifndef VAI_NODES_RMS_NORM_H
#define VAI_NODES_RMS_NORM_H

#include "vai/nodes/node-common.h"

namespace vai::node {


/*=============================================================================
  RmsNorm - RMS normalization over the feature axis, single kernel
  Input : in0    activations [..., D]
  Output: out0   normalized  [..., D]
  scale / shift   [D] (or [1, D] as an input), present per Config::hasScale /
                 hasShift; parameters, or inputs computed upstream under
                 Config::coeffsAsInputs (e.g. pi0.5's adaRMS)

  O[i, j] = I[i, j] * rsqrt(mean(I[i, :]^2) + eps) (* scale[j]) (+ shift[j])

  The row statistic is a 1-pass sum of squares accumulated in fp32 regardless
  of I/O dtypes. The workgroup size depends on D, so the pipeline is requested in
  prepare() once the shape is known.
=============================================================================*/
class RmsNorm : public vai::Node
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
        vai::Dtype wDtype   = vai::Dtype::None;

        bool operator==(const PipeParams&) const = default;
    };

    struct Config {
        float eps = 1e-5f;
        bool hasScale = true;
        bool hasShift = false;
        bool coeffsAsInputs = false;   // scale / shift are inputs, not parameters
        vai::Dtype wDtype = vai::Dtype::Fp32;           // persistent scale / shift; None: wDefaultDtype
        vai::Dtype oDtype = vai::Dtype::None;           // None: oDefaultDtype
    } config_;

    RmsNorm(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

private:
    PipeParams pipeParams;
};


// Workgroup geometry / kernel variants from device queries + shape:
// wgSize = pow2ceil(featSize / 4) in [sg, min(16 * sg, 1024)], subgroupAdd
// when supported (subgroup size pinned — Intel varies it otherwise).
inline RmsNorm::PipeParams rmsTuning(uint32_t featSize, Dtype iDtype, Dtype wDtype, Dtype oDtype, eva::Device device)
{
    const uint32_t sg = device.maxSubgroupSize();

    // TODO (eva): maxComputeWorkGroupInvocations limit query — until then
    //             cap at 1024 (common desktop value; spec guarantees 128)
    const uint32_t cap = std::min(16u * sg, 1024u);
    uint32_t wg = sg;
    while (wg * 4u < featSize && wg < cap) wg *= 2u;        // pow2ceil(featSize / 4), [sg, cap]

    RmsNorm::PipeParams t;
    t.wgSize   = wg;
    t.sgReduce = device.supportsSubgroupArithmetic();
    t.aligned  = (iDtype == Dtype::Fp32 && oDtype == Dtype::Fp32 && featSize % 4u == 0u);
    t.sgSize   = t.sgReduce ? sg : 0u;
    t.iDtype   = iDtype;
    t.wDtype   = wDtype;
    return t;
}


inline RmsNorm::RmsNorm(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    const NodeSlot::Type wSlot = config_.coeffsAsInputs ? NodeSlot::input : NodeSlot::persistent;
    if (config_.hasScale) addSlot("scale", wSlot);
    if (config_.hasShift)  addSlot("shift",  wSlot);

    config_.wDtype = (config_.wDtype == Dtype::None) ? wDefaultDtype() : config_.wDtype;
    config_.oDtype = (config_.oDtype == Dtype::None) ? oDefaultDtype() : config_.oDtype;
}

inline void RmsNorm::prepare()
{
    const Tensor& inTen = (*this)["in0"];
    const Tensor* scaleTen  = config_.hasScale ? &(*this)["scale"] : nullptr;
    const Tensor* shiftTen  = config_.hasShift  ? &(*this)["shift"]  : nullptr;

    uint32_t batchSize = outerSize(inTen.shape());
    uint32_t featSize = inTen.shape().back();

    VAI_ASSERT(inTen.shape().size() >= 2);
    // run() folds rows over grid x/y; x*y covers any int batch on spec-minimum
    // (65535 per axis) devices, so this is defense against a broken query.
    VAI_ASSERT(uint64_t(batchSize) <= uint64_t(device().maxComputeWorkGroupCount()[0])
                                                     * device().maxComputeWorkGroupCount()[1]);
    // Input scale / shift come at whatever dtype their producer wrote, sharing typeW.
    const Dtype wDtype = !config_.coeffsAsInputs ? config_.wDtype
                       : scaleTen ? scaleTen->dtype() : shiftTen ? shiftTen->dtype() : config_.wDtype;
    for (const Tensor* t : {scaleTen, shiftTen})
    {
        if (!t) continue;
        VAI_ASSERT(t->dtype() == wDtype);
        VAI_ASSERT(t->shape().back() == featSize && t->numElements() == featSize);
    }

    const Dtype iDtype = inTen.dtype();

    const PipeParams p = rmsTuning(featSize, iDtype, wDtype, config_.oDtype, device());
    if (!pipe || p != pipeParams)
    {
        pipeParams = p;
        pipe = pipelinePool().request("rms-norm", {
            eva::constant_id<0>(p.wgSize),                       // wgSize
            eva::constant_id<1>(p.sgReduce ? 1u : 0u),           // sgReduce
            eva::constant_id<2>(config_.hasScale ? 1u : 0u),     // hasScale
            eva::constant_id<3>(config_.hasShift ? 1u : 0u),     // hasShift
            eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),    // typeO
            eva::constant_id<SPEC_ID_TYPE_I>(iDtype),            // typeI
            eva::constant_id<SPEC_ID_TYPE_W>(wDtype),            // typeW (scale & shift)
            eva::constant_id<SPEC_ID_ALIGNED>(p.aligned ? 1u : 0u)  // aligned (vec4 loads)
        }, { .requiredSubgroupSize = p.sgSize });
        if (!descSet)
            descSet = gVaiDestSetPool(pipe.descSetLayout(0));
    }

    (*this)["out0"] = Tensor(inTen.shape(), config_.oDtype);
}

inline void RmsNorm::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = (*this)["in0"].buffer();
    auto scaleBuf   = config_.hasScale ? (*this)["scale"].buffer() : eva::Buffer();
    auto shiftBuf   = config_.hasShift  ? (*this)["shift"].buffer()  : eva::Buffer();

    descSet.write({ outBuf, inBuf, scaleBuf, shiftBuf });

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

#endif // VAI_NODES_RMS_NORM_H
