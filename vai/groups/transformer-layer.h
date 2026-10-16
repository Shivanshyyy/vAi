#ifndef VAI_GROUPS_TRANSFORMER_LAYER_H
#define VAI_GROUPS_TRANSFORMER_LAYER_H

#include "vai/groups/multi-head-attention.h"
#include "vai/groups/gated-feed-forward.h"
#include "vai/groups/feed-forward.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/rms-norm.h"
#include "vai/nodes/layer-norm.h"
#include "vai/nodes/gate.h"
#include "vai/nodes/slice.h"
#include "vai/nodes/add.h"
#include "vai/nodes/ele-affine.h"
#include <array>
#include <optional>

namespace vai {


// Where a checkpoint puts its normalizations and its residual adds. The two
// branches (attention, FFN) always exist; only their frame differs, so the
// layout is a type rather than a runtime option.
enum class TransformerType {
    Classic,      // pre-LayerNorm per branch, a plain MLP, the residual folded in
    Default,      // pre-RmsNorm per branch, a gated MLP, the residual folded in
    Gemma3,       // pre- AND post-RmsNorm per branch, the residual added outside them
    Pi05Action,   // Default with a condition: each norm's scale, shift and the residual's gate per step
};


// Every layout reads the same layer, so one option set serves them all. A
// layout that has no use for a field ignores it (Default never normalizes its
// branch output, Gemma3 is always causal).
struct TransformerLayerConfig {
    bool allowBatch = false;     // true: a leading batch axis, kept the whole way
    uint32_t embedDim;

    uint32_t headDim;
    uint32_t numQHeads;
    uint32_t numKVHeads;         // fewer than numQHeads for GQA
    vai::Dtype kvCacheDtype;     // K / V cache
    uint32_t kvCacheSize = 0;    // tokens the KV cache holds; 0 = no cache
    uint32_t windowSize = 0;     // sliding-window span; 0 = attend the whole prefix
    bool qkvBias = false;        // bias on the attention's Q / K / V projections
    bool outBias = false;        // ... and on its output projection
    bool withQkNorm = false;     // per-head RMS norm on Q and K, before RoPE
    std::optional<vai::RopeConfig> ropeCfg;   // absent: no position encoding

    uint32_t ffnHiddenDim;
    vai::Epilogue ffnActivation; // the FFN's activation, its GLU variant when gated
    bool ffnBias = false;        // bias on the FFN's projections (Classic only)

    float normEps = 1e-5f;       // every norm the layer owns, QK included
    bool sliceLastRow = false;   // carry only the last row past the attention
    bool gatingLastOut = false;  // a closed gate drops the FFN and everything downstream
};


}   // namespace vai


namespace vai::group {


// A layout the primary template's static_assert does not admit needs a
// specialization of its own. The QKV layout is the attention's business, so it
// passes straight through to it.
template<
    TransformerType layout = TransformerType::Default,
    QkvLayout qkv = QkvLayout::Separate
> struct TransformerLayer;


/////////////////////////////////////////////////////////////////////////////////////////
// TransformerLayer<Classic>, TransformerLayer<Default> - pre-norm layer, both
// residuals folded into projections
// Input: in0 [Tq, embedDim], or [B, Tq, embedDim] with allowBatch
// Output: out0 the shape of in0, its Tq 1 under sliceLastRow
//
// out0 = ffn(norm2(h)) + h,  h = mhAttn(norm1(in0)) + in0
//
// Neither add is its own node: the branch's last projection takes the residual
// on its "residual" (e.g. Llama3, PaliGemma's decoder).
//
// Classic is the same frame with the two choices that predate RMS norm and gated
// MLPs: LayerNorm on each branch's input, and one hidden projection rather than
// a gate and a value (e.g. GPT-2).
//
// sliceLastRow drops every row but the last once the attention has run, so the
// FFN and the head that follow it do one row's work. Only the last layer of a
// prefill wants it: attention needs the whole prefix, the rest does not.
/////////////////////////////////////////////////////////////////////////////////////////
template<TransformerType layout> struct BranchNorm;
template<> struct BranchNorm<TransformerType::Classic> { using Type = vai::node::LayerNorm; };
template<> struct BranchNorm<TransformerType::Default> { using Type = vai::node::RmsNorm; };

template<TransformerType layout> struct BranchFfn;
template<> struct BranchFfn<TransformerType::Classic> { using Type = FeedForward; };
template<> struct BranchFfn<TransformerType::Default> { using Type = GatedFeedForward; };


template<TransformerType layout, QkvLayout qkv>
struct TransformerLayer : public vai::NodeGroup
{
    static_assert(layout == TransformerType::Classic || layout == TransformerType::Default);

    using Config = TransformerLayerConfig;

    vai::node::Identity input;
    typename BranchNorm<layout>::Type norm1;
    MultiHeadAttention<qkv> mhAttn;
    std::optional<vai::node::Slice> lastRow;
    std::optional<vai::node::Gate> gate;
    typename BranchNorm<layout>::Type norm2;
    typename BranchFfn<layout>::Type ffn;

    static auto ffnConfig(const Config& config)
    {
        if constexpr (layout == TransformerType::Classic)
            return FeedForward::Config{
                .inDim = config.embedDim,
                .hiddenDim = config.ffnHiddenDim,
                .outDim = config.embedDim,
                .activation = config.ffnActivation,
                .hasBias = config.ffnBias,
                .acceptResidual = true};
        else
            return GatedFeedForward::Config{
                .inDim = config.embedDim,
                .hiddenDim = config.ffnHiddenDim,
                .outDim = config.embedDim,
                .activation = config.ffnActivation,
                .acceptResidual = true};
    }

    TransformerLayer(Config config)
    : norm1({.eps = config.normEps})
    , mhAttn({
        .allowBatch = config.allowBatch,
        .embedDim = config.embedDim,
        .headDim = config.headDim,
        .numQHeads = config.numQHeads,
        .numKVHeads = config.numKVHeads,
        .kvCacheDtype = config.kvCacheDtype,
        .kvCacheSize = config.kvCacheSize,
        .windowSize = config.windowSize,
        .qkvBias = config.qkvBias,
        .outBias = config.outBias,
        .qkNormEps = config.withQkNorm ? config.normEps : 0.0f,
        .ropeCfg = config.ropeCfg,
        .acceptResidual = true})
    , norm2({.eps = config.normEps})
    , ffn(ffnConfig(config))
    {
        if (config.sliceLastRow)
            lastRow.emplace(vai::node::Slice::Config{
                .axis = config.allowBatch ? 1u : 0u, .start = -1});
        if (config.gatingLastOut)
            gate.emplace();

        input - norm1 -            mhAttn - lastRow - gate/"then" - norm2 -            ffn;
        input         - "residual"/mhAttn - lastRow - gate/"then"         - "residual"/ffn;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", ffn.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// TransformerLayer<Gemma3> - each branch normalized on both ends, residuals outside
// Input: in0 [Tq, embedDim], or [B, Tq, embedDim] with allowBatch
// Output: out0 the shape of in0, its Tq 1 under sliceLastRow
//
// out0 = postFfnNorm(ffn(preFfnNorm(h))) + h,
//     h = postAttnNorm(mhAttn(preAttnNorm(in0))) + in0
//
// The post-norms sit on the branch output, before it rejoins the residual, so
// the residual stream itself is never rescaled. That is also why the adds are
// their own nodes here: a projection that folded the add in would add to a
// value the norm has not seen yet.
/////////////////////////////////////////////////////////////////////////////////////////
template<QkvLayout qkv>
struct TransformerLayer<TransformerType::Gemma3, qkv> : public vai::NodeGroup
{
    using Config = TransformerLayerConfig;

    vai::node::Identity input;
    vai::node::RmsNorm preAttnNorm;
    MultiHeadAttention<qkv> mhAttn;
    vai::node::RmsNorm postAttnNorm;
    vai::node::Add add1;
    std::optional<vai::node::Slice> lastRow;
    std::optional<vai::node::Gate> gate;
    vai::node::RmsNorm preFfnNorm;
    GatedFeedForward ffn;
    vai::node::RmsNorm postFfnNorm;
    vai::node::Add add2;

    TransformerLayer(Config config)
    : preAttnNorm({.eps = config.normEps})
    , mhAttn({
        .allowBatch = config.allowBatch,
        .embedDim = config.embedDim,
        .headDim = config.headDim,
        .numQHeads = config.numQHeads,
        .numKVHeads = config.numKVHeads,
        .kvCacheDtype = config.kvCacheDtype,
        .kvCacheSize = config.kvCacheSize,
        .windowSize = config.windowSize,
        .qkvBias = config.qkvBias,
        .outBias = config.outBias,
        .qkNormEps = config.withQkNorm ? config.normEps : 0.0f,
        .ropeCfg = config.ropeCfg})
    , postAttnNorm({.eps = config.normEps})
    , preFfnNorm({.eps = config.normEps})
    , ffn({
        .inDim = config.embedDim,
        .hiddenDim = config.ffnHiddenDim,
        .outDim = config.embedDim,
        .activation = config.ffnActivation})
    , postFfnNorm({.eps = config.normEps})
    {
        if (config.sliceLastRow)
            lastRow.emplace(vai::node::Slice::Config{
                .axis = config.allowBatch ? 1u : 0u, .start = -1});
        if (config.gatingLastOut)
            gate.emplace();

        input - preAttnNorm - mhAttn - postAttnNorm - "in0"/add1 - lastRow - gate/"then" - preFfnNorm - ffn - postFfnNorm - "in0"/add2;
        input                                       - "in1"/add1 - lastRow - gate/"then"                                  - "in1"/add2;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", add2.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// TransformerLayer<Pi05Action> - Default's layer, its norms and residuals set by a condition
// Input: in0  [Tq, embedDim], or [B, Tq, embedDim] with allowBatch
//        cond [6, embedDim]   scale1 shift1 g1 scale2 shift2 g2, one row each, shared by every token
// Output: out0 the shape of in0, its Tq 1 under sliceLastRow
//
// out0 = h + g2 * ffn(norm2(h)),  h = in0 + g1 * mhAttn(norm1(in0))
// norm(x) = rms(x) * scale + shift
//
// scale, shift and g come from a condition rather than the checkpoint (e.g. pi0.5's
// flow-matching timestep), so the norms read them as inputs and each gated
// residual is an EleAffine (g * branch + x): a projection that folded the
// residual in would add before the gate has scaled the branch.
/////////////////////////////////////////////////////////////////////////////////////////
template<QkvLayout qkv>
struct TransformerLayer<TransformerType::Pi05Action, qkv> : public vai::NodeGroup
{
    using Config = TransformerLayerConfig;

    vai::node::Identity input;
    vai::node::Identity cond;
    std::array<vai::node::Slice, 6> condRows;   // [scale1], [shift1], [g1], [scale2], [shift2], [g2]
    vai::node::RmsNorm norm1;
    MultiHeadAttention<qkv> mhAttn;
    vai::node::EleAffine fma1;
    std::optional<vai::node::Slice> lastRow;
    std::optional<vai::node::Gate> gate;
    vai::node::RmsNorm norm2;
    GatedFeedForward ffn;
    vai::node::EleAffine fma2;

    TransformerLayer(Config config)
    : condRows{{
        {{.start = 0}}, {{.start = 1}}, {{.start = 2}},
        {{.start = 3}}, {{.start = 4}}, {{.start = 5}}}}
    , norm1({.eps = config.normEps, .hasShift = true, .coeffsAsInputs = true})
    , mhAttn({
        .allowBatch = config.allowBatch,
        .embedDim = config.embedDim,
        .headDim = config.headDim,
        .numQHeads = config.numQHeads,
        .numKVHeads = config.numKVHeads,
        .kvCacheDtype = config.kvCacheDtype,
        .kvCacheSize = config.kvCacheSize,
        .windowSize = config.windowSize,
        .qkvBias = config.qkvBias,
        .outBias = config.outBias,
        .qkNormEps = config.withQkNorm ? config.normEps : 0.0f,
        .ropeCfg = config.ropeCfg})
    , norm2({.eps = config.normEps, .hasShift = true, .coeffsAsInputs = true})
    , ffn({
        .inDim = config.embedDim,
        .hiddenDim = config.ffnHiddenDim,
        .outDim = config.embedDim,
        .activation = config.ffnActivation})
    {
        if (config.sliceLastRow)
            lastRow.emplace(vai::node::Slice::Config{
                .axis = config.allowBatch ? 1u : 0u, .start = -1});
        if (config.gatingLastOut)
            gate.emplace();

        cond - condRows[0] - "scale"/norm1;
        cond - condRows[1] - "shift"/norm1;
        cond - condRows[2] - "scale"/fma1;
        cond - condRows[3] - "scale"/norm2;
        cond - condRows[4] - "shift"/norm2;
        cond - condRows[5] - "scale"/fma2;

        input - norm1 - mhAttn - "in0"/fma1   - lastRow - gate/"then" - norm2 - ffn - "in0"/fma2;
        input                  - "shift"/fma1 - lastRow - gate/"then"               - "shift"/fma2;

        defineSlot("in0", input.slot("in0"));
        defineSlot("cond", cond.slot("in0"));
        defineSlot("out0", fma2.slot("out0"));
    }
};


}   // namespace vai::group

#endif // VAI_GROUPS_TRANSFORMER_LAYER_H
