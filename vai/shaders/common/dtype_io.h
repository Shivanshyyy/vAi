// =============================================================================
// dtype_io.h - read a fp16/fp32 PARAMETER buffer (weight/bias/...) stored as
// raw uint[] words.
//
// Activations (in/out, KV cache, ...) are always fp32 and use plain float[]
// buffers directly. Only read-only parameters are packed into uint[]; a `type`
// spec constant selects how each word is unpacked:
//     DTYPE_F32 -> 1 element  per word  (uintBitsToFloat)
//     DTYPE_F16 -> 2 elements per word  (unpackHalf2x16)
//
// Reads are element-indexed and race-free. Parameter buffers are pool-allocated
// as a multiple of 4 bytes, so reading the final word is in-bounds even for an
// odd fp16 element count. (un)packHalf2x16 is core GLSL 4.2 -- no extensions.
// =============================================================================
#ifndef VAI_DTYPE_IO_H
#define VAI_DTYPE_IO_H

#include "spec_constants.h"   // DTYPE_F32 / DTYPE_F16

// Read element `e` (absolute element index) from a uint[] parameter buffer as float.
#define READ_F(buf, e, type)                              \
    ((type) == DTYPE_F32 ? uintBitsToFloat((buf)[(e)])           \
                  : unpackHalf2x16((buf)[(e) >> 1])[(e) & 1u])

// -----------------------------------------------------------------------------
// Word-owned WRITE helpers. Currently unused: activations are fp32 float[] and
// parameters are read-only, so nothing word-packs its output today. Kept for a
// future fp16-output / packed-write path (one thread owns a 32-bit word and
// stores both lanes once -> race-free, but needs the inner dimension even).
// -----------------------------------------------------------------------------

// Elements packed per 32-bit word for a given type (1 for fp32, 2 for fp16).
#define ELEMS_PER_WORD(type) ((type) == DTYPE_F32 ? 1u : 2u)

// Word index that holds element `e`.
#define WORD_OF(e, type) ((type) == DTYPE_F32 ? (e) : ((e) >> 1))

// Word-owned store: a thread owns output word `wIdx` and writes it once.
//   fp32 -> store v0; fp16 -> pack (v0, v1) into the word.
// v0/v1 must be pre-computed (pure ALU here, no memory loads), so this is
// race-free and safe regardless of which lane is in range. Compute v1 only for
// fp16 + in-range (a control-flow `if`), leaving it 0.0 otherwise.
#define WRITE_WORD(buf, wIdx, v0, v1, type)                       \
    ((buf)[(wIdx)] = (type) == DTYPE_F32 ? floatBitsToUint(v0)           \
                                  : packHalf2x16(vec2((v0), (v1))))

#endif // VAI_DTYPE_IO_H
