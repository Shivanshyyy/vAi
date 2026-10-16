#ifndef VAI_TEST_LINEAR_NODE_H
#define VAI_TEST_LINEAR_NODE_H

// A vAi node wrapping the verbatim ggml mul_mm kernel (ggml-style-matmul.comp —
// aligned_fp16 permutation, fp32 A/B/D fixed). It exists only so the framework
// path of this benchmark runs the exact kernel the raw-Vulkan path runs; the
// framework itself does not ship it.
//
// Upstream keeps the row-tail loads unmasked, so the pipeline is created with
// per-pipeline robustBufferAccess instead.
#include "vai/nodes/node-common.h"
#include "vai/nodes/linear.h"   // PipeParamsTile + ggmlMulmmTileParams

#ifndef GGML_STYLE_MATMUL_SPV
#error "GGML_STYLE_MATMUL_SPV must point at the compiled ggml-style-matmul.comp.spv"
#endif

namespace test {


class LinearNode : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct Config {
        uint32_t inDim;
        uint32_t outDim;
    } config_;

    LinearNode(Config config)
    : config_(config)
    {
        addSlot("out0", vai::NodeSlot::output);
        addSlot("in0", vai::NodeSlot::input);
        addSlot("weight", vai::NodeSlot::persistent);
    }

    void prepare() override
    {
        const vai::Tensor& inTen = (*this)["in0"];
        const vai::Tensor& wTen  = (*this)["weight"];

        uint32_t N = config_.outDim;
        uint32_t K = config_.inDim;
        VAI_ASSERT(inTen.dtype() == vai::Dtype::Fp32);   // the shader fixes fp32 A/B/D
        VAI_ASSERT(inTen.shape().back() == K);
        VAI_ASSERT(wTen.dtype() == vai::Dtype::Fp32);
        VAI_ASSERT(wTen.isShapeOf(N, K));
        VAI_ASSERT(K % 32 == 0);                         // BK = 32, ALIGNED permutation

        const auto& inShape = inTen.shape();
        uint32_t M = vai::outerSize(inShape);

        vai::node::Linear::PipeParamsTile t = vai::node::ggmlMulmmTileParams(M, N, K, vai::Dtype::Fp16, vai::Dtype::Fp16, device());
        if (!pipe || !(t == tileParams))
        {
            tileParams = t;

            // ggml letters are vai's with M/N swapped: BM/WM/TM run along
            // vai N (weight rows), BN/WN/TN along vai M (batch).
            eva::SpecializationConstant spec = {
                eva::constant_id<0>(t.wgSize),      // BLOCK_SIZE
                eva::constant_id<1>(t.wgN),         // BM
                eva::constant_id<2>(t.wgM),         // BN
                eva::constant_id<4>(t.sgN),         // WM
                eva::constant_id<5>(t.sgM),         // WN
                eva::constant_id<6>(t.sgpNIter),    // WMITER
                eva::constant_id<7>(t.tN),          // TM
                eva::constant_id<8>(t.tM),          // TN
                eva::constant_id<9>(1u),            // TK (coopmat only)
                eva::constant_id<10>(t.sgSize)      // WARP
            };

            pipe = device().createComputePipeline({
                .csStage = { eva::SpvBlob::readFrom(GGML_STYLE_MATMUL_SPV), std::move(spec) },
                .robustBufferAccess = true,   // ggml runs with robustBufferAccess enabled
            });

            if (!descSet)
                descSet = vai::gVaiDestSetPool(pipe.descSetLayout(0));
        }

        auto outShape = inShape;
        outShape.back() = N;
        (*this)["out0"] = vai::Tensor(outShape, vai::Dtype::Fp32);
    }

    void run(eva::CommandBuffer cmdBuff) override
    {
        auto outBuf = (*this)["out0"].buffer();
        auto inBuf  = (*this)["in0"].buffer();
        auto wBuf   = (*this)["weight"].buffer();
        descSet.write({ wBuf, inBuf, outBuf });   // ggml bindings: 0 = A(W), 1 = B(I), 2 = D(O)

        uint32_t M = vai::outerSize((*this)["in0"].shape());

        uint32_t N = config_.outDim;
        uint32_t K = config_.inDim;

        // ggml push constants; single 2D GEMM: packed strides, no batching,
        // no k-split (k_split = K -> ik = 0).
        uint32_t constants[] = {
            N, M, K,        // p.M, p.N, p.K   (ggml M = vai N)
            K, K, N,        // stride_a, stride_b, stride_d
            0, 0, 0,        // batch_stride_a/b/d
            0, 1, K,        // base_work_group_z, num_batches, k_split
            1, 1, 1, 1      // ne02, ne12, broadcast2, broadcast3
        };

        cmdBuff
            .bindPipeline(pipe)
            .bindDescSets({descSet})
            .setPushConstants(0, sizeof(constants), &constants)
            .dispatch(CEIL_DIV(N, tileParams.wgN), CEIL_DIV(M, tileParams.wgM), 1);
    }

private:
    vai::node::Linear::PipeParamsTile tileParams;   // tile params of the current pipeline (valid once pipe is set)
};


} // namespace test

#endif // VAI_TEST_LINEAR_NODE_H
