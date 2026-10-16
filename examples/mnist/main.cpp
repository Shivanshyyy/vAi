/*=============================================================================
  Written by   oneway@vaistrata.com
               jroot@vaistrata.com
  Task         MNIST — classifies a digit image, profiles the evaluation.
               The whole example lives in this file: model definition,
               checkpoint loading and the run.
  Checkpoint   data/mnist/ckpts/model.safetensors — trained by jroooot, ships with the repo.
  Assets       data/mnist/0.png
                 ships with the repo
=============================================================================*/

#include "vai/nodes/convolution.h" 
#include "vai/nodes/pool2d.h" 
#include "vai/nodes/flatten.h"     
#include "vai/nodes/linear.h"      
#include "vai/engine/weights.h"
#include "vai/utils/image-processor.h"
#include <cstdio>
#include <cstring>
#include <string>

// #define PROFILE_ON

#ifdef PROFILE_ON
    #include "vai/utils/time-checker.h"
#endif

using namespace vai;


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


// This checkpoint has no upstream distributor, so the tag names the contributor
// who trained it.
template<>
void loadWeights<"jroot", mnist::Model>(mnist::Model& model)
{
    auto ckpt = vai::Checkpoint(VAI_DATA_DIR "/mnist/ckpts");

    // Each tensor is loaded at the dtype declared by its owning node.
    // Convolution and Linear share the same config_.wDtype + weight/bias interface.
    auto load = [&](auto& node, const std::string& wKey, const std::string& bKey)
    {
        vai::Dtype dt = node.config_.wDtype;
        node["weight"] = vai::Tensor(ckpt[wKey], dt);
        node["bias"]   = vai::Tensor(ckpt[bKey], dt);
    };

    // Conv blocks
    load(model.convX2.blocks[0].conv, "conv1.weight", "conv1.bias");  // (C1, C0, K, K)
    load(model.convX2.blocks[1].conv, "conv2.weight", "conv2.bias");  // (C2, C1, K, K)

    // Fully-connected classification head
    load(model.linear, "fc.weight", "fc.bias");                       // (10, C2 * H/4 * W/4)
}


#ifdef PROFILE_ON
    // Tag nodes for profiling by name. Same-name nodes are accumulated together;
    // unnamed nodes are excluded from per-name results.
    void setProfileTags(mnist::Model& model)
    {
        // Hierarchical path tags: '.' builds a tree following the member-access
        // path; each block is kept as its own entry.
        for (uint32_t i = 0; i < model.convX2.blocks.size(); ++i)
        {
            auto& b = model.convX2.blocks[i];
            std::string p = "convX2.block" + std::to_string(i) + ".";
            b.conv.setProfileTag(p + "conv");
            b.maxpool.setProfileTag(p + "maxpool");
        }
        model.linear.setProfileTag("linear");
    }
#endif


int main()
{
    // Load the 28x28 gray image and normalize it to [0, 1]
    std::vector<float> inputData = ImageProcessor(VAI_DATA_DIR "/mnist/0.png", 1)
        .normalize(0.0f, 1.0f)
        .chw();

    // Load model. Switch wDefaultDtype to Dtype::Fp32 to run weights in FP32.
    mnist::Model model({
        .wDefaultDtype = Dtype::Fp32,
    });

    // Load weights from the "jroot" checkpoint.
    loadWeights<"jroot">(model); 

#ifdef PROFILE_ON
    // Tag nodes for profiling
    setProfileTags(model);
#endif

    Tensor input = Tensor({1, 28, 28}, Dtype::Fp32).setFloats(std::move(inputData));
    Tensor output;
    
#ifdef PROFILE_ON
    // Warm-up run (graph setup) — excluded from timing and profiling.
    model({{"in0", input}}, {{"out0", &output}});
    model.resetProfileRecords();

    // Run the model multiple times to get a stable profile. 
    
    {
        uint32_t iter = 100;
        TimeChecker timer("(vAi) MNIST evaluation: {} iterations", iter);

        for (uint32_t i = 0; i < iter; ++i)
        {
            model({{"in0", input}}, {{"out0", &output}});
        }
    }

    // Print the profile results
    model.printProfile();

#else
    // Run the model and get the output logits
    model({{"in0", input}}, {{"out0", &output}});
#endif

    // The network output is fp32.
    const float* data = output.getData();

    // Print the logits for each class
    printf("\n=== Output logits ===\n");
    for (int i = 0; i < 10; ++i)
        printf("data[%d] = %f\n", i, data[i]);

    return 0;
}


/*
Result:
data[0] = 23.026661
data[1] = -24.376106
data[2] = -1.872803
data[3] = -15.056368
data[4] = -14.481293
data[5] = -13.030799
data[6] = -0.130337
data[7] = -14.873937
data[8] = 1.507278
data[9] = -2.433280
*/

