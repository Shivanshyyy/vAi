// =============================================================================
// causal_mask.h - the attention blocks a stack sets on the host, read from a
// uniform buffer the host rewrites without re-recording.
//
// The block ends are prompt rows; a dispatch clips them to its own Tq, and its
// last block ends there. Bit (count - 1 - k) marks block k causal, so the first
// block holds the highest bit.
//
// Include after the push constants, which carry Tq, with CAUSAL_MASK_BINDING
// defined and GL_EXT_scalar_block_layout required for the std430 block.
// =============================================================================
#ifndef VAI_COMMON_CAUSAL_MASK_H
#define VAI_COMMON_CAUSAL_MASK_H

layout(set = 0, binding = CAUSAL_MASK_BINDING, std430) uniform CAUSAL_MASK {
    uint causalBlockCount;
    uint causalBits;
    uint causalBlockEnd[MAX_CAUSAL_BLOCKS - 1];
};

// The new tokens a query at `t` may read: past itself only when its block is
// causal, otherwise to the end of its block.
uint newTokenEnd(uint t)
{
    uint k = 0u, end = Tq;
    for (; k + 1u < causalBlockCount; ++k)
    {
        uint e = min(causalBlockEnd[k], Tq);
        if (t < e) { end = e; break; }
    }
    return ((causalBits >> (causalBlockCount - 1u - k)) & 1u) != 0u ? t + 1u : end;
}

#endif // VAI_COMMON_CAUSAL_MASK_H
