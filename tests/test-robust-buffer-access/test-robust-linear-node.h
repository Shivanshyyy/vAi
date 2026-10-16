#ifndef VAI_TEST_ROBUST_LINEAR_NODE_H
#define VAI_TEST_ROBUST_LINEAR_NODE_H

// vai::node::Linear's M > 8 tile path with the robustBufferAccess choice forced
// by an option instead of derived from the device (linear.h always picks
// supportsPipelineRobustness()). It exists only so this benchmark can run the
// identical kernel both ways in one process — loaders' row-tail guards kept
// (robust off) vs folded away behind per-pipeline robustness (robust on).
//
// Trimmed against vai::node::Linear: no bias / epilogue / split-K (they ride the
// reduce pass, which has no robust specialization) and no M <= 8 slab path
// (same — the robust spec constant only exists in the tile / cm kernels).
#include "vai/nodes/node-common.h"
#include "vai/nodes/linear.h"   // PipeParamsTile + ggmlMulmmTileParams + ggmlCoopmatShape

namespace test {


class LinearNode : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct Config {
        uint32_t inDim;
        uint32_t outDim;

        bool robust = false;   // the A/B axis: guard folding + per-pipeline robustness
        bool captureStatistics = false;   // read the compiled pipeline back (main2.cpp)

        vai::Dtype wDtype   = vai::Dtype::None;
        vai::Dtype oDtype   = vai::Dtype::None;   // None: oDefaultDtype
        vai::Dtype accDtype = vai::Dtype::None;
    } config_;

    // The compiled pipeline, for reading statistics back off it (main2.cpp).
    eva::ComputePipeline pipeline() const { return pipe; }

    LinearNode(Config config)
    : config_(config)
    {
        addSlot("out0", vai::NodeSlot::output);
        addSlot("in0", vai::NodeSlot::input);
        addSlot("weight", vai::NodeSlot::persistent);

        config_.wDtype   = (config_.wDtype   == vai::Dtype::None) ? wDefaultDtype() : config_.wDtype;
        config_.accDtype = (config_.accDtype == vai::Dtype::None) ? wDefaultDtype() : config_.accDtype;

        config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
    }

    void prepare() override
    {
        const vai::Tensor& inTen = (*this)["in0"];
        const vai::Tensor& wTen  = (*this)["weight"];

        uint32_t N = config_.outDim;
        uint32_t K = config_.inDim;
        const vai::Dtype iDtype = inTen.dtype();
        VAI_ASSERT(inTen.shape().back() == K);
        VAI_ASSERT(wTen.dtype() == config_.wDtype);
        VAI_ASSERT(wTen.isShapeOf(N, K));

        const auto& inShape = inTen.shape();
        uint32_t M = vai::outerSize(inShape);
        VAI_ASSERT(M > 8 && "robust A/B covers the tile path only (M <= 8 runs the slab kernel)");

        // robust on needs VK_EXT_pipeline_robustness — without it the folded
        // guards would read genuinely out of bounds.
        const bool robust = config_.robust;
        VAI_ASSERT(!robust || device().supportsPipelineRobustness());

        vai::node::Linear::PipeParamsTile t = vai::node::ggmlMulmmTileParams(M, N, K, vai::Dtype::Fp16, vai::Dtype::Fp16, device());
        if (!pipe || !(t == tileParams))
        {
            tileParams = t;

            const bool aligned = (K % t.wgK == 0);   // no split-K: splitSize = K

            // Loader constraints, as in vai::node::Linear::prepare().
            const uint32_t el = aligned ? 8u : 2u;
            VAI_ASSERT(t.wgK % 8 == 0);
            VAI_ASSERT((t.wgSize * el) % t.wgK == 0);
            VAI_ASSERT(t.wgM % (t.wgSize * el / t.wgK) == 0 && t.wgN % (t.wgSize * el / t.wgK) == 0);

            if (t.tK != 0)   // coopmat kernel
            {
                const vai::Dtype acc = (config_.accDtype == vai::Dtype::Fp16
                                        && !vai::node::ggmlCoopmatShape(device()).accF16)
                                     ? vai::Dtype::Fp32 : config_.accDtype;

                pipe = pipelinePool().request("linear-cm", {
                    eva::constant_id<0>(t.wgSize),                       // wgSize (= local_size_x)
                    eva::constant_id<1>(t.sgSize),                       // sgSize (= device subgroup size)
                    eva::constant_id<2>(t.wgM),                          // wgM
                    eva::constant_id<3>(t.wgN),                          // wgN
                    eva::constant_id<4>(t.wgK),                          // wgK
                    eva::constant_id<5>(t.sgM),                          // sgM
                    eva::constant_id<6>(t.sgN),                          // sgN
                    eva::constant_id<7>(t.tM),                           // tM
                    eva::constant_id<8>(t.tN),                           // tN
                    eva::constant_id<9>(t.tK),                           // tK
                    eva::constant_id<11>(acc),                           // accType
                    eva::constant_id<12>(robust ? 1u : 0u),              // robustBufferAccess
                    eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),    // typeO
                    eva::constant_id<SPEC_ID_TYPE_I>(iDtype),            // typeI
                    eva::constant_id<SPEC_ID_TYPE_W>(config_.wDtype),    // typeW
                    eva::constant_id<SPEC_ID_ALIGNED>(aligned ? 1u : 0u) // aligned
                }, {
                    .robustBufferAccess = robust,
                    .captureStatistics = config_.captureStatistics,
                });
            }
            else   // scalar tile kernel
            {
                VAI_ASSERT(t.tN % 2 == 0);   // the kernel accumulates W-row pairs

                pipe = pipelinePool().request("linear-tile", {
                    eva::constant_id<0>(t.wgSize),                       // wgSize (= local_size_x)
                    eva::constant_id<1>(t.sgSize),                       // sgSize
                    eva::constant_id<2>(t.wgM),                          // wgM
                    eva::constant_id<3>(t.wgN),                          // wgN
                    eva::constant_id<4>(t.wgK),                          // wgK
                    eva::constant_id<5>(t.sgM),                          // sgM
                    eva::constant_id<6>(t.sgN),                          // sgN
                    eva::constant_id<7>(t.tM),                           // tM
                    eva::constant_id<8>(t.tN),                           // tN
                    eva::constant_id<9>(t.sgpMIter),                     // sgpMIter
                    eva::constant_id<10>(t.sgpNIter),                    // sgpNIter
                    eva::constant_id<11>(config_.accDtype),              // accType
                    eva::constant_id<12>(robust ? 1u : 0u),              // robustBufferAccess
                    eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),    // typeO
                    eva::constant_id<SPEC_ID_TYPE_I>(iDtype),            // typeI
                    eva::constant_id<SPEC_ID_TYPE_W>(config_.wDtype),    // typeW
                    eva::constant_id<SPEC_ID_ALIGNED>(aligned ? 1u : 0u) // aligned
                }, {
                    .robustBufferAccess = robust,
                    .captureStatistics = config_.captureStatistics,
                });
            }

            if (!descSet)
                descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
        }

        auto outShape = inShape;
        outShape.back() = N;
        (*this)["out0"] = vai::Tensor(outShape, config_.oDtype);
    }

    void run(eva::CommandBuffer cmdBuff) override
    {
        auto outBuf = (*this)["out0"].buffer();
        auto inBuf  = (*this)["in0"].buffer();
        auto wBuf   = (*this)["weight"].buffer();

        uint32_t M = vai::outerSize((*this)["in0"].shape());

        uint32_t N = config_.outDim;
        uint32_t K = config_.inDim;
        uint32_t constants[] = { M, N, K, K };   // splitSize = K (no split)

        descSet.write({ outBuf, inBuf, wBuf });
        cmdBuff
            .bindPipeline(pipe)
            .bindDescSets({descSet})
            .setPushConstants(0, sizeof(constants), &constants)
            .dispatch(CEIL_DIV(N, tileParams.wgN), CEIL_DIV(M, tileParams.wgM), 1)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
    }

private:
    vai::node::Linear::PipeParamsTile tileParams;   // tile params of the current pipeline (valid once pipe is set)
};


} // namespace test

#endif // VAI_TEST_ROBUST_LINEAR_NODE_H
