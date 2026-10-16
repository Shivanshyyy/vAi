#ifndef YOLOX_GROUPS_FOCUS_MODULE_H
#define YOLOX_GROUPS_FOCUS_MODULE_H

#include "common.h"
#include "../nodes/focus-conv.h"


namespace yolox {


/////////////////////////////////////////////////////////////////////////////////////////
// FocusModule - Focus folded into Conv-BN-SiLU (no intermediate Focus tensor)
// Input : in0  [Ci, H, W]
// Output: out0 [Co, H/2, W/2]
/////////////////////////////////////////////////////////////////////////////////////////
struct FocusModule : public vai::NodeGroup
{
    FocusConvNode conv;

    FocusModule(uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize)
    : conv(inChannels, outChannels, kernelSize)
    {
        defineSlot("in0", conv.slot("in0"));
        defineSlot("out0", conv.slot("out0"));
    }
};


}   // namespace yolox

#endif // YOLOX_GROUPS_FOCUS_MODULE_H
