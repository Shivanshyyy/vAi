#ifndef PYVAI_UTILS_MNIST_H
#define PYVAI_UTILS_MNIST_H

// TODO: temporary copy — binding these is not possible right now. Upstream
// folded the whole MNIST example into examples/mnist/main.cpp (model
// definition, loadWeights<"jroot"> and setProfileTags in one file), so there
// is no models/mnist/model.h to include any more: the model structs and the
// two helpers below are copied verbatim from that file. Re-copy them when the
// source changes; replace this file once a bindable path exists.

#include "vai/nodes/convolution.h"    // ConvolutionNode
#include "vai/nodes/pool2d.h"    // MaxPool2dNode
#include "vai/nodes/flatten.h"        // FlattenNode
#include "vai/nodes/linear.h"         // LinearNode
#include "vai/engine/safetensors-parser.h"

#include <string>
#include <vector>


namespace mnist {

/////////////////////////////////////////////////////////////////////////////////////////
// ConvBlock - Conv2D (ReLU) followed by 2x2 max pooling
// Input : in0  [C, H, W]
// Output: out0 [F, H/2, W/2]
/////////////////////////////////////////////////////////////////////////////////////////
struct ConvBlock : public vai::NodeGroup {
    vai::node::Convolution conv;
    vai::node::MaxPool2d maxpool;

    ConvBlock(uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize)
    : conv({
        .inChannels = inChannels,
        .outChannels = outChannels,
        .kernelSize = kernelSize,
        .stride = 1,
        .hasBias = true,
        .epilogue = vai::Epilogue::Relu
    })
    , maxpool({
        .kernelSize = 2,
        .stride = 2
    })
    {
        conv - maxpool;
        defineSlot("in0", conv.slot("in0"));
        defineSlot("out0", maxpool.slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// ConvSequence - chain of ConvBlocks (channels[i] -> channels[i + 1])
// Input : in0  [channels.front(), H, W]
// Output: out0 [channels.back(), H / 2^n, W / 2^n]
/////////////////////////////////////////////////////////////////////////////////////////
struct ConvSequence : public vai::NodeGroup {
    std::vector<ConvBlock> blocks;

    ConvSequence(std::vector<uint32_t> channels, uint32_t kernelSize)
    {
        const uint32_t nBlocks = channels.size() - 1;
        VAI_ASSERT(nBlocks >= 1);

        blocks.reserve(nBlocks);
        for (uint32_t i = 0; i < nBlocks; ++i)
        {
            blocks.emplace_back(channels[i], channels[i + 1], kernelSize);
            if (i > 0)
                blocks[i - 1] - blocks[i];
        }

        defineSlot("in0", blocks[0].slot("in0"));
        defineSlot("out0", blocks[nBlocks - 1].slot("out0"));
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// Model - MNIST classifier: 2x ConvBlock -> flatten -> Linear(10)
// Input : [1, 28, 28]
// Output: [10] class logits
/////////////////////////////////////////////////////////////////////////////////////////
struct Model : public vai::NeuralNet {
    ConvSequence convX2;
    vai::node::Flatten flatten;
    vai::node::Linear linear;

    Model(const vai::NeuralNetConfig& netCfg)
    : NeuralNet(netCfg)
    , convX2({1, 32, 64}, 3)
    , flatten()
    , linear({.inDim = 7 * 7 * 64, .outDim = 10, .hasBias = true})
    {
        input("in0") - convX2 - flatten - linear - output("out0");
    }
};

} // namespace mnist


namespace pyvai::utils {
namespace {

using namespace vai;

void loadWeight(mnist::Model& net, const SafetensorsParser& parser)
{
    // Each tensor is loaded at the dtype declared by its owning node.
    // ConvolutionNode and LinearNode share the same config_.wDtype + weight/bias interface.
    auto load = [&](auto& node, const std::string& wKey, const std::string& bKey)
    {
        Dtype dt = node.config_.wDtype;
        node["weight"] = Tensor(parser[wKey], dt);
        node["bias"]   = Tensor(parser[bKey], dt);
    };

    // Conv blocks (the safetensors file uses 1-indexed conv names)
    load(net.convX2.blocks[0].conv, "conv1.weight", "conv1.bias");  // (C1, C0, K, K)
    load(net.convX2.blocks[1].conv, "conv2.weight", "conv2.bias");  // (C2, C1, K, K)

    // Fully-connected classification head
    load(net.linear, "fc.weight", "fc.bias");                       // (10, C2 * H/4 * W/4)
}

void setProfileTags(mnist::Model& net)
{
    // Hierarchical path tags: '.' builds a tree following the member-access
    // path; each block is kept as its own entry.
    for (uint32_t i = 0; i < net.convX2.blocks.size(); ++i)
    {
        auto& b = net.convX2.blocks[i];
        std::string p = "convX2.block" + std::to_string(i) + ".";
        b.conv.setProfileTag(p + "conv");
        b.maxpool.setProfileTag(p + "maxpool");
    }
    net.linear.setProfileTag("linear");
}

}   // namespace
}   // namespace pyvai::utils

#endif  // PYVAI_UTILS_MNIST_H
