#ifndef YOLOX_GROUPS_BOTTLENECK_H
#define YOLOX_GROUPS_BOTTLENECK_H

#include "common.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/add.h"


namespace yolox {


/////////////////////////////////////////////////////////////////////////////////////////
// Bottleneck - YOLOX bottleneck with an optional residual connection
// Input : in0  [Ci, H, W]
// Output: out0 [Co, H, W]
/////////////////////////////////////////////////////////////////////////////////////////
struct Bottleneck : public vai::NodeGroup
{
    vai::node::Identity input;
    vai::node::Convolution conv1;
    vai::node::Convolution conv2;
    vai::node::Add add;

    Bottleneck(
        uint32_t inChannels,
        uint32_t outChannels,
        bool shortcut = true,
        float expansion = 0.5f)
    : conv1(baseConv(
        inChannels,
        static_cast<uint32_t>(outChannels * expansion),
        1u,
        1u))
    , conv2(baseConv(
        static_cast<uint32_t>(outChannels * expansion),
        outChannels,
        3u,
        1u))
    {
        const bool fusedResidual = shortcut && inChannels == outChannels
            && conv2.enableResidualAdd();

        if (fusedResidual)
        {
            input - conv1 - conv2;
            input - "residual"/conv2;
            defineSlot("in0", input.slot("in0"));
            defineSlot("out0", conv2.slot("out0"));
        }
        else
        {
            input - conv1 - conv2;
            defineSlot("in0", input.slot("in0"));

            if (shortcut && inChannels == outChannels)
            {
                conv2 - "in0"/add;
                input - "in1"/add;
                defineSlot("out0", add.slot("out0"));
            }
            else
            {
                defineSlot("out0", conv2.slot("out0"));
            }
        }
    }
};


}   // namespace yolox

#endif // YOLOX_GROUPS_BOTTLENECK_H
