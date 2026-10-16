#ifndef YOLOX_GROUPS_COMMON_H
#define YOLOX_GROUPS_COMMON_H

#include "vai/nodes/convolution.h"
#include <cstdint>


namespace yolox {


/////////////////////////////////////////////////////////////////////////////////////////
// Shared by every yolox group: the conv builders the checkpoint is packed for,
// and the two numbers one tier fixes.
/////////////////////////////////////////////////////////////////////////////////////////
inline vai::node::Convolution::Config baseConv(
    uint32_t inChannels,
    uint32_t outChannels,
    uint32_t kernelSize,
    uint32_t stride)
{
    return {
        .inChannels = inChannels,
        .outChannels = outChannels,
        .kernelSize = kernelSize,
        .stride = stride,
        .hasBias = true,
        .hasScale = true,
        .epilogue = vai::Epilogue::Silu,
    };
}

inline vai::node::Convolution::Config predictionConv(
    uint32_t inChannels,
    uint32_t outChannels)
{
    return {
        .inChannels = inChannels,
        .outChannels = outChannels,
        .kernelSize = 1,
        .stride = 1,
    };
}


struct ModelBaseConfig {
    uint32_t baseChannels;
    uint32_t baseDepth;
};


}   // namespace yolox

#endif // YOLOX_GROUPS_COMMON_H
