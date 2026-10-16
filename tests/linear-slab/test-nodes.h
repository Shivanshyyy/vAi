/*=============================================================================
  Written by   oneway@vaistrata.com
  Nodes        SlabNode (test-linear-slab) · Slab2Node (test-linear-slab2)
               SlabV5Node · SlabV2Node · SlabV0Node (test-linear-slab-v5/-v2/-v0)

  Five nodes for A/B-ing the slab kernels at M <= 8. Every shader is cut down to
  the axis under test — fp16 weights, no bias, no epilogue — so none is a vAi
  shader and all are read by .spv path. Linear drives the shipping
  linear-slab instead and decides every parameter itself; these expose the
  parameters as knobs and do nothing else, so a sweep can set them.

  The three V nodes name what their shader still carries of test-linear-slab:
  V5 keeps the fold and the workgroup-wide lane stride, V2 keeps the fold only,
  V0 keeps neither. So V5 answers for every shape, and V2 / V0 only for
  wgSize == sgSize.

  They are deliberately kept separate rather than sharing a base. They agree
  on the buffers and the push block, but not on what the specialization
  constants mean — wgSize alone already differs (see each class), and only
  SlabNode and SlabV5Node have K_PER_ITER.

  Config leaves a knob at 0 to mean "this kernel's own default".
=============================================================================*/

#ifndef TEST_NODES_H
#define TEST_NODES_H

#include "vai/nodes/node-common.h"
#include "vai/nodes/linear.h"   // ggmlSlabSgSize / ggmlSlabWgSize

#ifndef TEST_LINEAR_SLAB_SPV
#error "TEST_LINEAR_SLAB_SPV must point at the compiled test-linear-slab.comp.spv"
#endif
#ifndef TEST_LINEAR_SLAB2_SPV
#error "TEST_LINEAR_SLAB2_SPV must point at the compiled test-linear-slab2.comp.spv"
#endif
#ifndef TEST_LINEAR_SLAB_V0_SPV
#error "TEST_LINEAR_SLAB_V0_SPV must point at the compiled test-linear-slab-v0.comp.spv"
#endif
#ifndef TEST_LINEAR_SLAB_V2_SPV
#error "TEST_LINEAR_SLAB_V2_SPV must point at the compiled test-linear-slab-v2.comp.spv"
#endif
#ifndef TEST_LINEAR_SLAB_V5_SPV
#error "TEST_LINEAR_SLAB_V5_SPV must point at the compiled test-linear-slab-v5.comp.spv"
#endif
#ifndef TEST_STREAM_SPV
#error "TEST_STREAM_SPV must point at the compiled test-stream.comp.spv"
#endif

namespace test {


/////////////////////////////////////////////////////////////////////////////////////////
// StreamNode - test-stream, the read-bandwidth roofline
//
// Reads the weight tensor and nothing else, so its rate is the ceiling the slab
// kernels are measured against. It takes the same slots and the same profile
// path as they do, so the two numbers are comparable without adjustment.
//
// wgCount is the grid: with a grid-stride loop the buffer is covered whatever
// it is, so it is a knob for finding the configuration that reaches furthest
// rather than a correctness constraint.
/////////////////////////////////////////////////////////////////////////////////////////
class StreamNode : public vai::Node
{
    eva::ComputePipeline pipe_;
    eva::DescriptorSet descSet_;
    uint32_t wgSize_  = 0;
    uint32_t wgCount_ = 0;

public:
    struct Config {
        uint32_t inDim;     // K — only used to size the tensors, never read as a shape
        uint32_t outDim;    // N

        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype

        uint32_t wgSize  = 0;   // threads per workgroup;  0: 64
        uint32_t wgCount = 0;   // workgroups;             0: 1024
    } config_;

    StreamNode(Config config)
    : config_(config)
    {
        addSlot("out0", vai::NodeSlot::output);
        addSlot("in0", vai::NodeSlot::input);
        addSlot("weight", vai::NodeSlot::persistent);

        config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
    }

    void prepare() override
    {
        const uint32_t N = config_.outDim;
        const uint32_t K = config_.inDim;
        VAI_ASSERT((*this)["weight"].isShapeOf(N, K));
        VAI_ASSERT((*this)["weight"].dtype() == vai::Dtype::Fp16);
        VAI_ASSERT((size_t(N) * K) % 4 == 0);   // read as f16vec4

        wgSize_  = config_.wgSize  ? config_.wgSize  : 64u;
        wgCount_ = config_.wgCount ? config_.wgCount : 1024u;

        if (!pipe_)
        {
            pipe_ = device().createComputePipeline({
                .csStage = { eva::SpvBlob::readFrom(TEST_STREAM_SPV), {
                    eva::constant_id<0>(wgSize_)
                }},
            });
            descSet_ = vai::gVaiDestSetPool(pipe_.descSetLayout(0));
        }

        auto outShape = (*this)["in0"].shape();
        outShape.back() = N;
        (*this)["out0"] = vai::Tensor(outShape, config_.oDtype);
    }

    void run(eva::CommandBuffer cmdBuff) override
    {
        auto outBuf = (*this)["out0"].buffer();
        uint32_t count = uint32_t(size_t(config_.outDim) * config_.inDim / 4);

        descSet_.write({ outBuf, (*this)["weight"].buffer() });
        cmdBuff
            .bindPipeline(pipe_)
            .bindDescSets({descSet_})
            .setPushConstants(0, sizeof(count), &count)
            .dispatch(wgCount_, 1, 1)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
    }

    uint32_t wgSize()  const { return wgSize_; }
    uint32_t wgCount() const { return wgCount_; }
};


class TestNode : public vai::Node
{
    std::string computeSpvName;
    // scMap : scId -> scName, scValue
    std::map<uint32_t, std::pair<std::string, uint32_t>> scMap;
};

/////////////////////////////////////////////////////////////////////////////////////////
// SlabNode - test-linear-slab, the shipping geometry
//
// A workgroup owns one slab: its wgSize threads split that slab's K and the
// per-subgroup partials are folded through shared memory. So wgSize is "how
// many threads share one output", and the grid is one workgroup per slab.
/////////////////////////////////////////////////////////////////////////////////////////
class SlabNode : public vai::Node
{
    eva::ComputePipeline pipe_;
    eva::DescriptorSet descSet_;
    uint32_t pipeM_   = 0;   // M baked into pipe_ (0: not built yet)
    uint32_t wgSize_  = 0;
    uint32_t sgSize_  = 0;
    uint32_t wgN_     = 0;
    uint32_t kPerIter_ = 0;

public:
    struct Config {
        uint32_t inDim;
        uint32_t outDim;

        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype

        uint32_t wgSize   = 0;   // threads sharing one slab; 0: ggmlSlabWgSize()
        uint32_t sgSize   = 0;   // pinned subgroup size;     0: ggmlSlabSgSize()
        uint32_t wgN      = 0;   // N positions per slab;     0: 2 (fp16 W pairs them)
        uint32_t kPerIter = 0;   // k per thread per pass;    0: 4. Multiple of 4

    } config_;

    SlabNode(Config config)
    : config_(config)
    {
        addSlot("out0", vai::NodeSlot::output);
        addSlot("in0", vai::NodeSlot::input);
        addSlot("weight", vai::NodeSlot::persistent);

        config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
    }

    void prepare() override
    {
        const vai::Tensor& inTen = (*this)["in0"];
        const uint32_t N = config_.outDim;
        const uint32_t K = config_.inDim;
        const vai::Dtype iDtype = inTen.dtype();
        VAI_ASSERT(inTen.shape().back() == K);
        VAI_ASSERT((*this)["weight"].isShapeOf(N, K));
        VAI_ASSERT((*this)["weight"].dtype() == vai::Dtype::Fp16);   // the shader fixes fp16 W

        const auto& inShape = inTen.shape();
        uint32_t M = vai::outerSize(inShape);
        VAI_ASSERT(M <= 8);

        sgSize_   = config_.sgSize ? config_.sgSize : vai::node::ggmlSlabSgSize(device());
        wgSize_   = config_.wgSize ? config_.wgSize
                                    : vai::node::ggmlSlabWgSize(K, N, sgSize_, device());
        wgN_      = config_.wgN      ? config_.wgN      : 2u;
        kPerIter_ = config_.kPerIter ? config_.kPerIter : 4u;
        VAI_ASSERT(kPerIter_ % 4 == 0);   // a chunk is read as kPerIter / 4 vec4s

        const bool aligned = (K % kPerIter_ == 0);

        if (!pipe_ || pipeM_ != M)
        {
            pipeM_ = M;

            pipe_ = device().createComputePipeline({
                .csStage = { eva::SpvBlob::readFrom(TEST_LINEAR_SLAB_SPV), {
                    eva::constant_id<0>(wgSize_),                            // wgSize (= local_size_x)
                    eva::constant_id<1>(sgSize_),                            // sgSize
                    eva::constant_id<2>(M),                                  // M
                    eva::constant_id<3>(wgN_),                               // wgN
                    eva::constant_id<4>(kPerIter_),                          // K_PER_ITER
                    eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),        // typeO
                    eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                // typeI
                    eva::constant_id<SPEC_ID_ALIGNED>(aligned ? 1u : 0u)     // aligned
                }},
                .requiredSubgroupSize = sgSize_,
            });

            if (!descSet_)
                descSet_ = vai::gVaiDestSetPool(pipe_.descSetLayout(0));
        }

        auto outShape = inShape;
        outShape.back() = N;
        (*this)["out0"] = vai::Tensor(outShape, config_.oDtype);
    }

    void run(eva::CommandBuffer cmdBuff) override
    {
        auto outBuf = (*this)["out0"].buffer();
        const uint32_t N = config_.outDim;
        uint32_t constants[] = { config_.inDim, N };

        // One workgroup per slab. An N past the X-grid limit folds into z; the
        // kernel re-linearizes with the z term.
        uint32_t gx = CEIL_DIV(N, wgN_);
        uint32_t gz = 1;
        if (gx > device().maxComputeWorkGroupCount()[0])
        {
            gz = 64;
            gx = CEIL_DIV(gx, gz);
        }

        descSet_.write({ outBuf, (*this)["in0"].buffer(), (*this)["weight"].buffer() });
        cmdBuff
            .bindPipeline(pipe_)
            .bindDescSets({descSet_})
            .setPushConstants(0, sizeof(constants), &constants)
            .dispatch(gx, 1, gz)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
    }

    uint32_t wgSize()   const { return wgSize_; }
    uint32_t sgSize()   const { return sgSize_; }
    uint32_t wgN()      const { return wgN_; }
    uint32_t kPerIter() const { return kPerIter_; }
};


/////////////////////////////////////////////////////////////////////////////////////////
// Slab2Node - test-linear-slab2, the A/B geometry variant
//
// A subgroup owns one slab and finishes it alone — no shared memory, no
// barrier. So wgSize is "how many slabs a workgroup covers" (wgSize / sgSize of
// them), not how many threads share one, and the grid must be divided by that
// count. Defaults to one subgroup per workgroup, the geometry the kernel was
// written for; a wider wgSize is a knob for packing several slabs per launch.
/////////////////////////////////////////////////////////////////////////////////////////
class Slab2Node : public vai::Node
{
    eva::ComputePipeline pipe_;
    eva::DescriptorSet descSet_;
    uint32_t pipeM_  = 0;   // M baked into pipe_ (0: not built yet)
    uint32_t wgSize_ = 0;
    uint32_t sgSize_ = 0;
    uint32_t wgN_    = 0;

public:
    struct Config {
        uint32_t inDim;
        uint32_t outDim;

        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype

        uint32_t wgSize = 0;   // threads per workgroup;  0: sgSize (one slab per workgroup)
        uint32_t sgSize = 0;   // pinned subgroup size;   0: ggmlSlabSgSize()
        uint32_t wgN    = 0;   // N positions per slab;   0: 2 (fp16 W pairs them)
    } config_;

    Slab2Node(Config config)
    : config_(config)
    {
        addSlot("out0", vai::NodeSlot::output);
        addSlot("in0", vai::NodeSlot::input);
        addSlot("weight", vai::NodeSlot::persistent);

        config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
    }

    void prepare() override
    {
        const vai::Tensor& inTen = (*this)["in0"];
        const uint32_t N = config_.outDim;
        const uint32_t K = config_.inDim;
        const vai::Dtype iDtype = inTen.dtype();
        VAI_ASSERT(inTen.shape().back() == K);
        VAI_ASSERT((*this)["weight"].isShapeOf(N, K));
        VAI_ASSERT((*this)["weight"].dtype() == vai::Dtype::Fp16);   // the shader fixes fp16 W

        const auto& inShape = inTen.shape();
        uint32_t M = vai::outerSize(inShape);
        VAI_ASSERT(M <= 8);

        sgSize_ = config_.sgSize ? config_.sgSize : vai::node::ggmlSlabSgSize(device());
        wgSize_ = config_.wgSize ? config_.wgSize : sgSize_;
        wgN_    = config_.wgN    ? config_.wgN    : 2u;
        VAI_ASSERT(wgSize_ % sgSize_ == 0);   // whole subgroups, or a slab has no owner

        const bool aligned = (K % 4 == 0);   // K_PER_ITER = 4

        if (!pipe_ || pipeM_ != M)
        {
            pipeM_ = M;

            pipe_ = device().createComputePipeline({
                .csStage = { eva::SpvBlob::readFrom(TEST_LINEAR_SLAB2_SPV), {
                    eva::constant_id<0>(wgSize_),                            // wgSize (= local_size_x)
                    eva::constant_id<1>(sgSize_),                            // sgSize
                    eva::constant_id<2>(M),                                  // M
                    eva::constant_id<3>(wgN_),                               // wgN
                    eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),        // typeO
                    eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                // typeI
                    eva::constant_id<SPEC_ID_ALIGNED>(aligned ? 1u : 0u)     // aligned
                }},
                .requiredSubgroupSize = sgSize_,
            });

            if (!descSet_)
                descSet_ = vai::gVaiDestSetPool(pipe_.descSetLayout(0));
        }

        auto outShape = inShape;
        outShape.back() = N;
        (*this)["out0"] = vai::Tensor(outShape, config_.oDtype);
    }

    void run(eva::CommandBuffer cmdBuff) override
    {
        auto outBuf = (*this)["out0"].buffer();
        const uint32_t N = config_.outDim;
        uint32_t constants[] = { config_.inDim, N };

        // A workgroup covers wgSize / sgSize slabs, so the slab count is divided
        // by that — dispatching one workgroup per slab would leave all but the
        // first subgroup of each launching only to exit.
        uint32_t gx = CEIL_DIV(CEIL_DIV(N, wgN_), wgSize_ / sgSize_);
        uint32_t gz = 1;
        if (gx > device().maxComputeWorkGroupCount()[0])
        {
            gz = 64;
            gx = CEIL_DIV(gx, gz);
        }

        descSet_.write({ outBuf, (*this)["in0"].buffer(), (*this)["weight"].buffer() });
        cmdBuff
            .bindPipeline(pipe_)
            .bindDescSets({descSet_})
            .setPushConstants(0, sizeof(constants), &constants)
            .dispatch(gx, 1, gz)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
    }

    uint32_t wgSize() const { return wgSize_; }
    uint32_t sgSize() const { return sgSize_; }
    uint32_t wgN()    const { return wgN_; }
};


/////////////////////////////////////////////////////////////////////////////////////////
// SlabV0Node - test-linear-slab-v0. SlabV2Node's twin down to the defaults; the
// two shaders differ only in whether the shared array is declared, so the pair
// prices that declaration and nothing else.
/////////////////////////////////////////////////////////////////////////////////////////
class SlabV0Node : public vai::Node
{
    eva::ComputePipeline pipe_;
    eva::DescriptorSet descSet_;
    uint32_t pipeM_   = 0;   // M baked into pipe_ (0: not built yet)
    uint32_t wgSize_  = 0;
    uint32_t sgSize_  = 0;
    uint32_t wgN_     = 0;

public:
    struct Config {
        uint32_t inDim;
        uint32_t outDim;

        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype

        uint32_t wgSize = 0;   // threads sharing one slab; 0: ggmlSlabWgSize()
        uint32_t sgSize = 0;   // pinned subgroup size;     0: ggmlSlabSgSize()
        uint32_t wgN    = 0;   // N positions per slab;     0: 2 (fp16 W pairs them)

        const char* spv = TEST_LINEAR_SLAB_V0_SPV;   // see SlabV2Node::Config::spv
    } config_;

    SlabV0Node(Config config)
    : config_(config)
    {
        addSlot("out0", vai::NodeSlot::output);
        addSlot("in0", vai::NodeSlot::input);
        addSlot("weight", vai::NodeSlot::persistent);

        config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
    }

    void prepare() override
    {
        const vai::Tensor& inTen = (*this)["in0"];
        const uint32_t N = config_.outDim;
        const uint32_t K = config_.inDim;
        const vai::Dtype iDtype = inTen.dtype();
        VAI_ASSERT(inTen.shape().back() == K);
        VAI_ASSERT((*this)["weight"].isShapeOf(N, K));
        VAI_ASSERT((*this)["weight"].dtype() == vai::Dtype::Fp16);   // the shader fixes fp16 W

        const auto& inShape = inTen.shape();
        uint32_t M = vai::outerSize(inShape);
        VAI_ASSERT(M <= 8);

        sgSize_ = config_.sgSize ? config_.sgSize : vai::node::ggmlSlabSgSize(device());
        wgSize_ = config_.wgSize ? config_.wgSize
                                  : vai::node::ggmlSlabWgSize(K, N, sgSize_, device());
        wgN_    = config_.wgN    ? config_.wgN    : 2u;
        VAI_ASSERT(wgSize_ == sgSize_);   // lane strides by sgSize: one subgroup owns a slab

        const bool aligned = (K % 4 == 0);   // K_PER_ITER = 4

        if (!pipe_ || pipeM_ != M)
        {
            pipeM_ = M;

            pipe_ = device().createComputePipeline({
                .csStage = { eva::SpvBlob::readFrom(config_.spv), {
                    eva::constant_id<0>(wgSize_),                            // wgSize (= local_size_x)
                    eva::constant_id<1>(sgSize_),                            // sgSize
                    eva::constant_id<2>(M),                                  // M
                    eva::constant_id<3>(wgN_),                               // wgN
                    eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),        // typeO
                    eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                // typeI
                    eva::constant_id<SPEC_ID_ALIGNED>(aligned ? 1u : 0u)     // aligned
                }},
                .requiredSubgroupSize = sgSize_,
            });

            if (!descSet_)
                descSet_ = vai::gVaiDestSetPool(pipe_.descSetLayout(0));
        }

        auto outShape = inShape;
        outShape.back() = N;
        (*this)["out0"] = vai::Tensor(outShape, config_.oDtype);
    }

    void run(eva::CommandBuffer cmdBuff) override
    {
        auto outBuf = (*this)["out0"].buffer();
        const uint32_t N = config_.outDim;
        uint32_t constants[] = { config_.inDim, N };

        // One workgroup per slab. An N past the X-grid limit folds into z; the
        // kernel re-linearizes with the z term.
        uint32_t gx = CEIL_DIV(N, wgN_);
        uint32_t gz = 1;
        if (gx > device().maxComputeWorkGroupCount()[0])
        {
            gz = 64;
            gx = CEIL_DIV(gx, gz);
        }

        descSet_.write({ outBuf, (*this)["in0"].buffer(), (*this)["weight"].buffer() });
        cmdBuff
            .bindPipeline(pipe_)
            .bindDescSets({descSet_})
            .setPushConstants(0, sizeof(constants), &constants)
            .dispatch(gx, 1, gz)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
    }

    uint32_t wgSize() const { return wgSize_; }
    uint32_t sgSize() const { return sgSize_; }
    uint32_t wgN()    const { return wgN_; }
};


/////////////////////////////////////////////////////////////////////////////////////////
// SlabV2Node - test-linear-slab-v2, SlabNode's grid with the shader rewritten
// (see its header). Same config and same defaults as SlabNode, so a duel
// between the two isolates the shader text and nothing else — but that shader's
// lane strides its K by sgSize, so wgSize must come out equal to sgSize. Shapes
// whose heuristic asks for the wider workgroup are out of its scope and assert
// here rather than compute the wrong thing quietly.
/////////////////////////////////////////////////////////////////////////////////////////
class SlabV2Node : public vai::Node
{
    eva::ComputePipeline pipe_;
    eva::DescriptorSet descSet_;
    uint32_t pipeM_   = 0;   // M baked into pipe_ (0: not built yet)
    uint32_t wgSize_  = 0;
    uint32_t sgSize_  = 0;
    uint32_t wgN_     = 0;

public:
    struct Config {
        uint32_t inDim;
        uint32_t outDim;

        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype

        uint32_t wgSize = 0;   // threads sharing one slab; 0: ggmlSlabWgSize()
        uint32_t sgSize = 0;   // pinned subgroup size;     0: ggmlSlabSgSize()
        uint32_t wgN    = 0;   // N positions per slab;     0: 2 (fp16 W pairs them)

        // Any .spv that honours this node's contract — same bindings, same push
        // block, same specialization ids. Anything built off test-linear-slab-v2
        // qualifies, which is how a one-line shader edit gets A/B'd without a
        // second node.
        const char* spv = TEST_LINEAR_SLAB_V2_SPV;
    } config_;

    SlabV2Node(Config config)
    : config_(config)
    {
        addSlot("out0", vai::NodeSlot::output);
        addSlot("in0", vai::NodeSlot::input);
        addSlot("weight", vai::NodeSlot::persistent);

        config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
    }

    void prepare() override
    {
        const vai::Tensor& inTen = (*this)["in0"];
        const uint32_t N = config_.outDim;
        const uint32_t K = config_.inDim;
        const vai::Dtype iDtype = inTen.dtype();
        VAI_ASSERT(inTen.shape().back() == K);
        VAI_ASSERT((*this)["weight"].isShapeOf(N, K));
        VAI_ASSERT((*this)["weight"].dtype() == vai::Dtype::Fp16);   // the shader fixes fp16 W

        const auto& inShape = inTen.shape();
        uint32_t M = vai::outerSize(inShape);
        VAI_ASSERT(M <= 8);

        sgSize_ = config_.sgSize ? config_.sgSize : vai::node::ggmlSlabSgSize(device());
        wgSize_ = config_.wgSize ? config_.wgSize
                                  : vai::node::ggmlSlabWgSize(K, N, sgSize_, device());
        wgN_    = config_.wgN    ? config_.wgN    : 2u;
        VAI_ASSERT(wgSize_ == sgSize_);   // no fold in this shader: one subgroup owns a slab

        const bool aligned = (K % 4 == 0);   // K_PER_ITER = 4

        if (!pipe_ || pipeM_ != M)
        {
            pipeM_ = M;

            pipe_ = device().createComputePipeline({
                .csStage = { eva::SpvBlob::readFrom(config_.spv), {
                    eva::constant_id<0>(wgSize_),                            // wgSize (= local_size_x)
                    eva::constant_id<1>(sgSize_),                            // sgSize
                    eva::constant_id<2>(M),                                  // M
                    eva::constant_id<3>(wgN_),                               // wgN
                    eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),        // typeO
                    eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                // typeI
                    eva::constant_id<SPEC_ID_ALIGNED>(aligned ? 1u : 0u)     // aligned
                }},
                .requiredSubgroupSize = sgSize_,
            });

            if (!descSet_)
                descSet_ = vai::gVaiDestSetPool(pipe_.descSetLayout(0));
        }

        auto outShape = inShape;
        outShape.back() = N;
        (*this)["out0"] = vai::Tensor(outShape, config_.oDtype);
    }

    void run(eva::CommandBuffer cmdBuff) override
    {
        auto outBuf = (*this)["out0"].buffer();
        const uint32_t N = config_.outDim;
        uint32_t constants[] = { config_.inDim, N };

        // One workgroup per slab. An N past the X-grid limit folds into z; the
        // kernel re-linearizes with the z term.
        uint32_t gx = CEIL_DIV(N, wgN_);
        uint32_t gz = 1;
        if (gx > device().maxComputeWorkGroupCount()[0])
        {
            gz = 64;
            gx = CEIL_DIV(gx, gz);
        }

        descSet_.write({ outBuf, (*this)["in0"].buffer(), (*this)["weight"].buffer() });
        cmdBuff
            .bindPipeline(pipe_)
            .bindDescSets({descSet_})
            .setPushConstants(0, sizeof(constants), &constants)
            .dispatch(gx, 1, gz)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
    }

    uint32_t wgSize() const { return wgSize_; }
    uint32_t sgSize() const { return sgSize_; }
    uint32_t wgN()    const { return wgN_; }
};


/////////////////////////////////////////////////////////////////////////////////////////
// SlabV5Node - test-linear-slab-v5, SlabNode with three loop changes and the
// fold left in place. Host side is SlabNode's, unchanged, including the wgSize
// the heuristic picks: this is the candidate patch for the shipping kernel, so
// it has to answer for every shape SlabNode does, not just subgroup-sized ones.
/////////////////////////////////////////////////////////////////////////////////////////
class SlabV5Node : public vai::Node
{
    eva::ComputePipeline pipe_;
    eva::DescriptorSet descSet_;
    uint32_t pipeM_   = 0;   // M baked into pipe_ (0: not built yet)
    uint32_t wgSize_  = 0;
    uint32_t sgSize_  = 0;
    uint32_t wgN_     = 0;
    uint32_t kPerIter_ = 0;

public:
    struct Config {
        uint32_t inDim;
        uint32_t outDim;

        vai::Dtype oDtype = vai::Dtype::None;   // None: oDefaultDtype

        uint32_t wgSize   = 0;   // threads sharing one slab; 0: ggmlSlabWgSize()
        uint32_t sgSize   = 0;   // pinned subgroup size;     0: ggmlSlabSgSize()
        uint32_t wgN      = 0;   // N positions per slab;     0: 2 (fp16 W pairs them)
        uint32_t kPerIter = 0;   // k per thread per pass;    0: 4. Multiple of 4

    } config_;

    SlabV5Node(Config config)
    : config_(config)
    {
        addSlot("out0", vai::NodeSlot::output);
        addSlot("in0", vai::NodeSlot::input);
        addSlot("weight", vai::NodeSlot::persistent);

        config_.oDtype = (config_.oDtype == vai::Dtype::None) ? oDefaultDtype() : config_.oDtype;
    }

    void prepare() override
    {
        const vai::Tensor& inTen = (*this)["in0"];
        const uint32_t N = config_.outDim;
        const uint32_t K = config_.inDim;
        const vai::Dtype iDtype = inTen.dtype();
        VAI_ASSERT(inTen.shape().back() == K);
        VAI_ASSERT((*this)["weight"].isShapeOf(N, K));
        VAI_ASSERT((*this)["weight"].dtype() == vai::Dtype::Fp16);   // the shader fixes fp16 W

        const auto& inShape = inTen.shape();
        uint32_t M = vai::outerSize(inShape);
        VAI_ASSERT(M <= 8);

        sgSize_   = config_.sgSize ? config_.sgSize : vai::node::ggmlSlabSgSize(device());
        wgSize_   = config_.wgSize ? config_.wgSize
                                    : vai::node::ggmlSlabWgSize(K, N, sgSize_, device());
        wgN_      = config_.wgN      ? config_.wgN      : 2u;
        kPerIter_ = config_.kPerIter ? config_.kPerIter : 4u;
        VAI_ASSERT(kPerIter_ % 4 == 0);   // a chunk is read as kPerIter / 4 vec4s

        const bool aligned = (K % kPerIter_ == 0);

        if (!pipe_ || pipeM_ != M)
        {
            pipeM_ = M;

            pipe_ = device().createComputePipeline({
                .csStage = { eva::SpvBlob::readFrom(TEST_LINEAR_SLAB_V5_SPV), {
                    eva::constant_id<0>(wgSize_),                            // wgSize (= local_size_x)
                    eva::constant_id<1>(sgSize_),                            // sgSize
                    eva::constant_id<2>(M),                                  // M
                    eva::constant_id<3>(wgN_),                               // wgN
                    eva::constant_id<4>(kPerIter_),                          // K_PER_ITER
                    eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),        // typeO
                    eva::constant_id<SPEC_ID_TYPE_I>(iDtype),                // typeI
                    eva::constant_id<SPEC_ID_ALIGNED>(aligned ? 1u : 0u)     // aligned
                }},
                .requiredSubgroupSize = sgSize_,
            });

            if (!descSet_)
                descSet_ = vai::gVaiDestSetPool(pipe_.descSetLayout(0));
        }

        auto outShape = inShape;
        outShape.back() = N;
        (*this)["out0"] = vai::Tensor(outShape, config_.oDtype);
    }

    void run(eva::CommandBuffer cmdBuff) override
    {
        auto outBuf = (*this)["out0"].buffer();
        const uint32_t N = config_.outDim;
        uint32_t constants[] = { config_.inDim, N };

        // One workgroup per slab. An N past the X-grid limit folds into z; the
        // kernel re-linearizes with the z term.
        uint32_t gx = CEIL_DIV(N, wgN_);
        uint32_t gz = 1;
        if (gx > device().maxComputeWorkGroupCount()[0])
        {
            gz = 64;
            gx = CEIL_DIV(gx, gz);
        }

        descSet_.write({ outBuf, (*this)["in0"].buffer(), (*this)["weight"].buffer() });
        cmdBuff
            .bindPipeline(pipe_)
            .bindDescSets({descSet_})
            .setPushConstants(0, sizeof(constants), &constants)
            .dispatch(gx, 1, gz)
            .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
    }

    uint32_t wgSize()   const { return wgSize_; }
    uint32_t sgSize()   const { return sgSize_; }
    uint32_t wgN()      const { return wgN_; }
    uint32_t kPerIter() const { return kPerIter_; }
};


} // namespace test

#endif // TEST_NODES_H
