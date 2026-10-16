# 제공 노드들

빌트인 노드 27종. 개념은 [노드(Node)](03-usage/node.md)를, 새 노드를 만드는 법은 [커스텀 노드와 커널](07-contributing-nodes.md)을 참조하세요.

## 목록

| 노드 | 요약 | 커널 |
|---|---|---|
| [Activation](#activation) | Applies an element-wise activation. | `activation.comp` |
| [Add](#add) | element-wise addition for residual connections | `add.comp` |
| [Argmax](#argmax) | Row-wise argmax / greedy sampling | `argmax.comp` |
| [Attention](#attention) | fused KV cache append + causal attention (GQA), split between three kernels | `attention-fa-cm.comp`, `attention-fa.comp`, `attention-fa-reduce.comp`, `attention.comp` |
| [AttentionFused](#attentionfused) | fused-QKV KV cache update + causal scaled-dot-product attention | `attention-fused.comp` |
| [BatchNorm](#batchnorm) | batch normalization (inference mode) | `batchnorm.comp` |
| [Concat](#concat) | join inputs along one axis | `concat.comp` |
| [Convolution](#convolution) | 2D convolution with device-selected direct, implicit, or Winograd plans | `conv-k1s1-gemm-cm.comp`, `conv-k1s1-gemm.comp`, `conv-k3s2-gemm-cm.comp`, `conv-k3s1-implicit-gemm-cm.comp`, `conv-k3s1-implicit-gemm.comp`, `conv-implicit-gemm.comp`, `conv-implicit-gemm-cm.comp`, `conv-k3s2-gemm.comp`, `cm-transform.comp`, `conv-k3s1-winograd-filter-transform.comp`, `conv-k3s1-winograd-input.comp`, `conv-k3s1-winograd-output.comp`, `bn-reduce.comp`, `conv-k3s1-winograd-gemm.comp`, `conv-k3s1-winograd-gemm-cm.comp` |
| [Flatten](#flatten) | Flatten tensor to 1D | — |
| [Gate](#gate) | conditional branch (no computation) | — |
| [GatedMul](#gatedmul) | element-wise multiply with one operand activated (GLU gating) | `gated-mul.comp` |
| [Identity](#identity) | Pass-through node (no computation) | — |
| [LayerNorm](#layernorm) | layer normalization over the feature axis, single kernel | `layer-norm.comp` |
| [Linear](#linear) | O = I · Wᵀ, split between two output decompositions | `linear-slab.comp`, `linear-cm.comp`, `linear-tile.comp`, `linear-reduce.comp` |
| [Mul](#mul) | Element-wise multiplication | `mul.comp` |
| [NearestUpsample](#nearestupsample) | Replicates CHW pixels by an integer scale factor. | `upsample.comp` |
| [Permute](#permute) | Permutes the axes of a tensor of any rank. | `permute.comp` |
| [Pool2d](#pool2d) | 2D pooling with a square kernel (MaxPool2d, AvgPool2d) | `pool2d.comp` |
| [Reshape](#reshape) | Reshape tensor to specified dimensions | — |
| [RmsNorm](#rmsnorm) | RMS normalization over the feature axis, single kernel | `rms-norm.comp` |
| [Rope](#rope) | rotary position embedding (half-split rotation) | `rope.comp` |
| [Slice](#slice) | Take a run of consecutive indices along one axis | `slice.comp` |
| [TimeEmbedding](#timeembedding) | sinusoidal embedding of a timestep the node counts down itself | `time-embedding.comp` |
| [TokenEmbedding](#tokenembedding) | token embedding, optionally fused with positional embedding | `token-embedding.comp` |
| [TopKSampling](#topksampling) | row-wise top-k sampling, one kernel run as two passes | `topk-sampling.comp` |
| [TypeCast](#typecast) | Casts each tensor element to another dtype. | `cast.comp` |

---

## Activation

Applies an element-wise activation.

소스: [`vai/nodes/activation.h`](../../vai/nodes/activation.h)

**시그니처**

- 입력: in0 [any shape]
- 출력: epilogue(in0); optionally in-place

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `epilogue` | `vai::Epilogue` | `vai::Epilogue::None` |  |
| `inPlace` | `bool` | `false` |  |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

The inPlace option aliases the output to the input and requires matching input
and output dtypes. Otherwise, the output uses separate storage.

**커널**

- `vai/shaders/activation.comp`

---

## Add

element-wise addition for residual connections

소스: [`vai/nodes/add.h`](../../vai/nodes/add.h)

**시그니처**

- 입력: in0, in1 [same shape]
- 출력: in0 + in1

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `in1` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` | in0 and in1 (shared) |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

Inputs must have identical shapes and dtypes.

**커널**

- `vai/shaders/add.comp`

---

## Argmax

Row-wise argmax / greedy sampling

소스: [`vai/nodes/argmax.h`](../../vai/nodes/argmax.h)

**시그니처**

- 입력: in0 [M, N] (fp32 logits)
- 출력: out0 [M] (one raw uint32 index per row, in 4-byte slots)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` | logits dtype (output is raw uint32 indices) |

**동작 및 제약**

Each row is reduced independently. out0[m] is the index of the largest value in
in0[m, :]. Output elements are raw uint32 values, not fp32 numeric values.

**커널**

- `vai/shaders/argmax.comp`

---

## Attention

fused KV cache append + causal attention (GQA), split between three kernels

소스: [`vai/nodes/attention.h`](../../vai/nodes/attention.h)

**시그니처**

- 입력: Q [..., Hq, Tq, headDim], K / V [..., Hkv, Tq, headDim]
- 출력: out0 attention [..., Tq, Hq, headDim]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `K` | input |  |
| `Q` | input |  |
| `V` | input |  |
| `out0` | output |  |
| `K_cache` | persistent |  |
| `V_cache` | persistent |  |
| `partialML` | temporary |  |
| `partialO` | temporary |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `headDim` | `uint32_t` | — |  |
| `numQHeads` | `uint32_t` | — |  |
| `numKVHeads` | `uint32_t` | `0` | for GQA (Hkv < numQHeads); 0 → Hkv = numQHeads |
| `kvCacheSize` | `uint32_t` | `0` | tokens the cache holds; 0 = no cache at all |
| `windowSize` | `uint32_t` | `0` | sliding-window span (0 = attend the whole prefix) |
| `qDtype` | `vai::Dtype` | `vai::Dtype::None` | Q |
| `kvDtype` | `vai::Dtype` | `vai::Dtype::None` | K / V, input and cache alike |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

Leading axes are more sequences swept in one dispatch; they ride the grid,
so every count below stays per-sequence.

Q, K, V arrive as three separate buffers (so callers can transform Q/K but
not V in between — e.g. RoPE). Supports GQA (numKVHeads < numQHeads).

- coopmat FA (attention-fa-cm): subgroup-MMA flash attention,
fixed 16x64 tile; prefill and GQA-packed decode on cm-capable devices.
- scalar FA (attention-fa): flash attention with per-(shape,
vendor) tuned tiles; everything the cm gates reject.
- naive (attention): the unconditional fallback — unfamiliar
devices degrade instead of crashing. See ggmlFATuningParams().

The path and tuning are selected per (shape, device) in prepare(); a tuning
change re-requests the pipeline from the pool. Every path leaves the KV
cache in the identical state (fused append), so the graph meaning does not
depend on which one runs.

Sliding window (windowSize > 0): a query at position p attends
[p - windowSize + 1, p] (e.g. Gemma3's local layers). The cache still holds every
token — the window narrows the sweep, not the storage.

kvCacheSize = 0 is orthogonal to both: it drops the cache itself, so a
dispatch attends its own tokens and nothing else (an encoder, e.g. SigLIP).
Nothing is allocated and no counter is wired — the cache and counter
descriptors go null, and the 0 a null counter reads back is the whole trick.

Split-K (decode): when the grid would underfill the GPU, the cache sweep is
split into kCount segments accumulating partials (partialO/partialML) that
a second pass (attention-fa-reduce) folds with the online-
softmax merge. kCount is the dispatch grid, so it holds for the whole
recording and is derived from occupancy alone; where the segments fall
follows the cache, which grows under a recording, so the kernel derives that
from kvCacheCount per forward.

**커널**

- `vai/shaders/attention-fa-cm.comp`
- `vai/shaders/attention-fa.comp`
- `vai/shaders/attention-fa-reduce.comp`
- `vai/shaders/attention.comp`

---

## AttentionFused

fused-QKV KV cache update + causal scaled-dot-product attention

소스: [`vai/nodes/attention-fused.h`](../../vai/nodes/attention-fused.h)

**시그니처**

- 입력: in0 QKV [Hq + 2*Hkv, Tq, headDim] (Q=[0,Hq) K=[Hq,Hq+Hkv) V=[Hq+Hkv,Hq+2*Hkv))
- 출력: out0 attention [Hq, Tq, headDim]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |
| `K_cache` | persistent |  |
| `V_cache` | persistent |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `headDim` | `uint32_t` | — |  |
| `numQHeads` | `uint32_t` | — |  |
| `numKVHeads` | `uint32_t` | `0` | for GQA (Hkv < numQHeads); 0 → Hkv = numQHeads |
| `kvCacheSize` | `uint32_t` | — |  |
| `cacheDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` | fused QKV |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

Reads Q / K / V directly from the one fused buffer via per-block offsets — no
pre-split copy. Supports GQA (numKVHeads < numQHeads).

**커널**

- `vai/shaders/attention-fused.comp`

---

## BatchNorm

batch normalization (inference mode)

소스: [`vai/nodes/batch-norm.h`](../../vai/nodes/batch-norm.h)

**시그니처**

- 입력: in0 [C, H, W]
- 출력: γ[c] * (in0 - μ[c]) * rsqrt(σ²[c] + eps) + β[c]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |
| `beta` | persistent |  |
| `gamma` | persistent |  |
| `mean` | persistent |  |
| `variance` | persistent |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `channels` | `uint32_t` | — |  |
| `eps` | `float` | `1e-5f` |  |
| `wDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

gamma, beta, mean, and variance are one-dimensional tensors of length C. This is
inference-only normalization; stored statistics are not updated.

**커널**

- `vai/shaders/batchnorm.comp`

---

## Concat

join inputs along one axis

소스: [`vai/nodes/concat.h`](../../vai/nodes/concat.h)

**시그니처**

- 입력: in0 .. inN [same shape except along the concat axis]
- 출력: out0 the inputs joined along axis, in input order

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0, in1, …` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `axis` | `int` | `0` | negative counts from the end |
| `inputCount` | `uint32_t` | `2` |  |
| `allowPartialInputs` | `bool` | `false` |  |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` | in0..inN (shared) |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

All inputs must have identical dtypes, ranks, and dimensions except along axis. A
negative axis counts from the end.

What the axes ahead of the cut multiply to decides the cost, not the axis number. At
1 each input is one contiguous run of the output, so buffer copies place them and no
shader runs — axis 0 is always this case, and so is a deeper axis with nothing but
size-1 axes in front. One input arriving is then the whole output and out0 aliases it.
Past 1 the runs interleave and a gather kernel reads them, which is where the 8-input
limit and the dtype conversion live.

allowPartialInputs lets a recording bring only some of them; the rest are not joined.
Off, every input is required, so a recording that cannot supply one drops this node.

**커널**

- `vai/shaders/concat.comp`

---

## Convolution

2D convolution with device-selected direct, implicit, or Winograd plans

소스: [`vai/nodes/convolution.h`](../../vai/nodes/convolution.h)

**시그니처**

- 입력: in0 [Ci, Hi, Wi], optional residual [Co, Ho, Wo]
- 파라미터: weight [Co, Ci, K, K] and bias [Co], or BatchNorm affine inputs [Co]
- 출력: out0 [Co, ceil(Hi / stride), ceil(Wi / stride)]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `residual` | input |  |
| `out0` | output |  |
| `bias` | persistent |  |
| `scale` | persistent |  |
| `weight` | persistent |  |
| `split` | temporary |  |
| `winM` | temporary |  |
| `winV` | temporary |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `inChannels` | `uint32_t` | — |  |
| `outChannels` | `uint32_t` | — |  |
| `kernelSize` | `uint32_t` | — |  |
| `stride` | `uint32_t` | — |  |
| `hasBias` | `bool` | `true` |  |
| `hasScale` | `bool` | `false` |  |
| `epilogue` | `vai::Epilogue` | `vai::Epilogue::None` |  |
| `wDtype` | `vai::Dtype` | `vai::Dtype::None` | conv weight |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

Exactly one of bias or BatchNorm fusion must be enabled. Odd kernels use symmetric
same-padding, so each spatial output size is ceil(inputSize / stride). Residual
fusion is available only for BatchNorm K=3, stride=1, and equal input/output channels.

**커널**

- `vai/shaders/conv-k1s1-gemm-cm.comp`
- `vai/shaders/conv-k1s1-gemm.comp`
- `vai/shaders/conv-k3s2-gemm-cm.comp`
- `vai/shaders/conv-k3s1-implicit-gemm-cm.comp`
- `vai/shaders/conv-k3s1-implicit-gemm.comp`
- `vai/shaders/conv-implicit-gemm.comp`
- `vai/shaders/conv-implicit-gemm-cm.comp`
- `vai/shaders/conv-k3s2-gemm.comp`
- `vai/shaders/cm-transform.comp`
- `vai/shaders/conv-k3s1-winograd-filter-transform.comp`
- `vai/shaders/conv-k3s1-winograd-input.comp`
- `vai/shaders/conv-k3s1-winograd-output.comp`
- `vai/shaders/bn-reduce.comp`
- `vai/shaders/conv-k3s1-winograd-gemm.comp`
- `vai/shaders/conv-k3s1-winograd-gemm-cm.comp`

---

## Flatten

Flatten tensor to 1D

소스: [`vai/nodes/flatten.h`](../../vai/nodes/flatten.h)

**시그니처**

- 입력: in0 [any shape]
- 출력: out0 [N] where N = total number of elements (shares buffer with input)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**동작 및 제약**

elements is the input's total element count. The output aliases the input buffer.

---

## Gate

conditional branch (no computation)

소스: [`vai/nodes/gate.h`](../../vai/nodes/gate.h)

**시그니처**

- 입력: in0 [any shape]
- 출력: then [same shape], else [same shape] — both alias in0's buffer

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `else` | output |  |
| `then` | output |  |

**동작 및 제약**

One branch is recorded per command buffer. The engine drops every node that
only the closed branch reaches, so those nodes neither prepare() nor run();
a node both branches reach (the merge point and everything past it) is kept.
Leaving "else" unwired makes a closed gate drop the whole downstream.

The condition is host state, so flipping it re-records. Weights are uploaded
by uploadModelParams(), which ignores gates — a branch closed on the first forward
still gets its weights on the device.

Two producers may share one input slot only where a gate's branches merge;
sortNodes() rejects any other fan-in. The branches need not agree on shape or
dtype: a flip re-records, so every recording derives the merge point and its
downstream from whichever branch survived it.

---

## GatedMul

element-wise multiply with one operand activated (GLU gating)

소스: [`vai/nodes/gated-mul.h`](../../vai/nodes/gated-mul.h)

**시그니처**

- 입력: in0 [..., 2H]
- 출력: out0 [..., H] = activation(in0[..., j]) * in0[..., H + j]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `activation` | `vai::Epilogue` | `vai::Epilogue::Silu` |  |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

The two halves arrive in ONE buffer: the projections feeding them are a
single Linear over the concatenated weights, so the leading half is whatever
that concatenation put first. It is the activated one.

activation names the GLU variant — silu: SwiGLU, gelu: GEGLU, relu: ReGLU,
none: bilinear.

**커널**

- `vai/shaders/gated-mul.comp`

---

## Identity

Pass-through node (no computation)

소스: [`vai/nodes/identity.h`](../../vai/nodes/identity.h)

**시그니처**

- 입력: in0 [any shape]
- 출력: out0 [same shape] (shares buffer with input)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**동작 및 제약**

The output aliases the input buffer; no computation or copy occurs.

---

## LayerNorm

layer normalization over the feature axis, single kernel

소스: [`vai/nodes/layer-norm.h`](../../vai/nodes/layer-norm.h)

**시그니처**

- 입력: activations [..., D]
- 출력: normalized [..., D]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |
| `beta` | persistent | 옵션에 따라 존재 |
| `gamma` | persistent |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `eps` | `float` | `1e-5f` |  |
| `hasBeta` | `bool` | `true` |  |
| `wDtype` | `vai::Dtype` | `vai::Dtype::Fp32` | gamma / beta; the statistic is fp32 regardless |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

O[i, j] = (I[i, j] - mean(I[i, :])) * rsqrt(var(I[i, :]) + eps) * γ[j] + β[j]

One workgroup per row (layer-norm); the row statistics come from a
shifted 1-pass reduction (Chan–Golub–LeVeque) accumulated in fp32
regardless of I/O dtypes. Two spec-selected variants share the kernel:

- reduction: subgroupAdd two-level when the device has subgroup arithmetic
(subgroup size pinned to max — Intel varies it otherwise), shared-memory
tree fallback everywhere else.
- loads: vec4 when I and O are fp32 and featSize % 4 == 0, scalar otherwise.

Geometry / reduction / vec4 are selected per (featSize, device) in
lnTuning(); the workgroup size depends on featSize, so the pipeline is
requested in prepare() once the shape is known, and a tuning change
re-requests it from the pool (no-op while the tuning stays put).

**커널**

- `vai/shaders/layer-norm.comp`

---

## Linear

O = I · Wᵀ, split between two output decompositions

소스: [`vai/nodes/linear.h`](../../vai/nodes/linear.h)

**시그니처**

- 입력: [..., inDim]
- 출력: [..., outDim]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `residual` | input | 옵션에 따라 존재 |
| `out0` | output |  |
| `bias` | persistent | 옵션에 따라 존재 |
| `bias2d` | persistent | 옵션에 따라 존재 |
| `weight` | persistent |  |
| `partial` | temporary |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `inDim` | `uint32_t` | — |  |
| `outDim` | `uint32_t` | — |  |
| `hasBias` | `bool` | `false` |  |
| `hasBias2D` | `bool` | `false` | parameter "bias2d" [M, N], one bias per position |
| `acceptResidual` | `bool` | `false` | input "residual" [M, N], a following Add folded in |
| `epilogue` | `vai::Epilogue` | `vai::Epilogue::None` |  |
| `wDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `splitSize` | `uint32_t` | `0` | split-K: K length per split (0: ggml auto heuristic) |
| `accDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

vai naming: O [M, N], I [M, K] (M = flattened batch), W [N, K]
(N = outDim, K = inDim).

- slab (M <= 8, linear-slab): a workgroup owns wgN whole columns of O,
so K is split across its threads and reduced; bias / epilogue are fused
in-kernel. See useSlabPath().
- tile (M > 8): a workgroup owns a wgM x wgN tile. Two kernels share the
frame, chosen per device in ggmlMulmmTileParams(): scalar
(linear-tile, each thread accumulates its own outputs) and
cooperative-matrix (linear-cm, subgroup MMA tiles). Everything
below applies to both:

The tile params are selected per output shape in prepare(); when a shape
change selects a different tile, the pipeline is re-requested from the pool.

Split-K (splitSize option; 0 selects the ggml heuristic per shape)
accumulates partials P[splitCount, M, N] (oDtype) in a temporary buffer
and folds them in a second pass (linear-reduce); bias / epilogue / the
addend live in that reduce pass, so requesting them without splitting still
runs the reduce as a plain epilogue pass (splitCount = 1).

One addend may be folded into the result, after the epilogue so a fused
activation stays inside the projection. Its source is exclusive: acceptResidual
opens the "residual" input and folds a following Add in, hasBias2D opens the
"bias2d" parameter for a bias that varies per row as well as per column. The
source also picks its dtype, iDtype for the activation and wDtype for the
parameter, which the kernels take as hasAddend 1 and 2.

**커널**

- `vai/shaders/linear-slab.comp`
- `vai/shaders/linear-cm.comp`
- `vai/shaders/linear-tile.comp`
- `vai/shaders/linear-reduce.comp`

---

## Mul

Element-wise multiplication

소스: [`vai/nodes/mul.h`](../../vai/nodes/mul.h)

**시그니처**

- 입력: in0 [any shape], in1 [same shape]
- 출력: out0 [same shape] = in0 * in1

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `in1` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` | in0 and in1 (shared) |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

Inputs must have identical shapes and dtypes.

**커널**

- `vai/shaders/mul.comp`

---

## NearestUpsample

Replicates CHW pixels by an integer scale factor.

소스: [`vai/nodes/nearest-upsample.h`](../../vai/nodes/nearest-upsample.h)

**시그니처**

- 입력: in0 [C, H, W]
- 출력: out0 [C, H*scale, W*scale]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

scaleFactor must be greater than one. Each input pixel is replicated without
interpolation.

**커널**

- `vai/shaders/upsample.comp`

---

## Permute

Permutes the axes of a tensor of any rank.

소스: [`vai/nodes/permute.h`](../../vai/nodes/permute.h)

**시그니처**

- 입력: in0 [D0, .., Dr-1], r = perms.size()
- 출력: axes reordered by perms; reshape-only fast path when layout is unchanged

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

Output dimension i is input dimension perms[i], and perms must be a permutation of
{0, .., r-1} with r at most PERMUTE_MAX_RANK. When the permutation preserves memory
order and the dtype is unchanged, the output aliases the input; otherwise it uses new
contiguous storage.

The rank is a specialization constant, so the pool holds one pipeline per rank.

**커널**

- `vai/shaders/permute.comp`

---

## Pool2d

2D pooling with a square kernel (MaxPool2d, AvgPool2d)

소스: [`vai/nodes/pool2d.h`](../../vai/nodes/pool2d.h)

**시그니처**

- 입력: in0 [..., C, H, W], or [..., H, W, C] with channelsLast
- 출력: [..., C, Ho, Wo], or [..., Ho, Wo, C] where Ho = (H + 2*padding - kernelSize) / stride + 1, rounded up with ceilMode

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `kernelSize` | `uint32_t` | — |  |
| `stride` | `uint32_t` | — |  |
| `padding` | `uint32_t` | `0` |  |
| `ceilMode` | `bool` | `false` | a last window that would start in the right padding is dropped |
| `countIncludePad` | `bool` | `true` | Avg: padding counts in the divisor, as PyTorch's default |
| `channelsLast` | `bool` | `false` |  |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

channelsLast reads a token grid as it is laid out (rows of embedDim, e.g. a
ViT's patches viewed as [G, G, D]), so no transpose is needed around the pool.

**커널**

- `vai/shaders/pool2d.comp`

---

## Reshape

Reshape tensor to specified dimensions

소스: [`vai/nodes/reshape.h`](../../vai/nodes/reshape.h)

**시그니처**

- 입력: in0 [any shape]
- 출력: out0 [dims] (shares buffer with input)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**동작 및 제약**

A dim is a size, 0 for the input's dimension at that same position, or -1 for
whatever the element count leaves over, at most once (= ONNX). 0 carries a size only
the caller's data knows without the graph naming it. The resolved dims must multiply
to the input's element count. The output aliases the input buffer.

---

## RmsNorm

RMS normalization over the feature axis, single kernel

소스: [`vai/nodes/rms-norm.h`](../../vai/nodes/rms-norm.h)

**시그니처**

- 입력: in0 activations [..., D]
- 파라미터: scale [D]
- 출력: out0 normalized [..., D]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |
| `scale` | persistent |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `eps` | `float` | `1e-5f` |  |
| `wDtype` | `vai::Dtype` | `vai::Dtype::Fp32` | scale; the statistic is fp32 regardless |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

O[i, j] = I[i, j] * rsqrt(mean(I[i, :]^2) + eps) * scale[j]

The row statistic is a 1-pass sum of squares accumulated in fp32 regardless
of I/O dtypes. The workgroup size depends on D, so the pipeline is requested in
prepare() once the shape is known.

**커널**

- `vai/shaders/rms-norm.comp`

---

## Rope

rotary position embedding (half-split rotation)

소스: [`vai/nodes/rope.h`](../../vai/nodes/rope.h)

**시그니처**

- 입력: in0 [.., H, Tq, Dh] (Q or K, Dh even)
- 파라미터: freqs [Dh/2] per-dimension angular frequency, built by the constructor
- 출력: out0 [.., H, Tq, Dh]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |
| `freqs` | persistent |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `cfg` | `RopeConfig` | — | frequencies; describe the whole model |
| `headDim` | `uint32_t` | — | Dh, must be even |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

The rotation depends on the token position alone, so anything ahead of the head
axis is one more sweep and folds into the head count.

Positions continue from setKvCacheCounter()'s buffer, so a KV-cache decode
step rotates by the absolute position rather than by its offset in this forward.

cfg.type picks how the frequencies are scaled: Default leaves them alone,
Linear divides every one by factor, and Llama3 divides only the long
wavelengths, interpolating the band between the two frequency bounds.

**커널**

- `vai/shaders/rope.comp`

---

## Slice

Take a run of consecutive indices along one axis

소스: [`vai/nodes/slice.h`](../../vai/nodes/slice.h)

**시그니처**

- 입력: in0 [d0, .., d(r-1)] (rank >= 2)
- 출력: out0 the same shape with d(axis) replaced by count

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `axis` | `uint32_t` | `0` | the axis to cut; 0 takes a contiguous block |
| `start` | `int64_t` | — | first index on the axis; negative counts from the end |
| `count` | `uint32_t` | `1` | how many consecutive indices to take |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

A negative start counts from the end, with -1 selecting the last index.

What the axes ahead of the cut multiply to decides the cost, not the axis number.
At 1 the run is one contiguous block, so out0 aliases it when the offset satisfies
device alignment and is one buffer copy otherwise — axis 0 is always this case, and
so is a deeper axis with nothing but size-1 axes in front. Past 1 the run is strided,
count indices every d(axis), and a gather kernel reads it out. The shape decides the
path; the caller writes the same node either way.

**커널**

- `vai/shaders/slice.comp`

---

## TimeEmbedding

sinusoidal embedding of a timestep the node counts down itself

소스: [`vai/nodes/time-embedding.h`](../../vai/nodes/time-embedding.h)

**시그니처**

- 입력: in0 [.., rows, *] only its shape is read: one embedding per row
- 출력: out0 [.., rows, embedDim]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `embedDim` | `uint32_t` | — | even; sin in the first half, cos in the second |
| `minPeriod` | `float` | — | the fastest sinusoid, in units of t |
| `maxPeriod` | `float` | — | ... and the slowest |
| `numSteps` | `uint32_t` | — | runs before t is back at 1 |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

out0[r, i]        = sin(2*pi * t / p_i)
out0[r, half + i] = cos(2*pi * t / p_i),   p_i = minPeriod * (maxPeriod / minPeriod)^(i / (half - 1))

Run k after resetTime() embeds t = 1 - (k % numSteps) / numSteps, so a
flow-matching loop of numSteps steps walks t from 1 down with no host write.
resetTime() starts it over (e.g. pi0 after a prefill); the count wraps after
2^32 workgroups, so a caller that never resets does so within that.

**커널**

- `vai/shaders/time-embedding.comp`

---

## TokenEmbedding

token embedding, optionally fused with positional embedding

소스: [`vai/nodes/token-embedding.h`](../../vai/nodes/token-embedding.h)

**시그니처**

- 입력: in0 token IDs [..., T] (uint32)
- 출력: out0 embedding [..., T, D] (fp32)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |
| `posWeight` | persistent | 옵션에 따라 존재 |
| `weight` | persistent |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `vocabSize` | `uint32_t` | — |  |
| `embedDim` | `uint32_t` | — |  |
| `posTableSize` | `uint32_t` | `0` | learned positional table rows; 0 = token only |
| `scale` | `float` | `1.0f` | multiplies the looked-up embedding |
| `wDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

Leading axes are more sequences swept in one dispatch, the way a Linear takes its
leading axes as more rows, so the output keeps the input's rank.

posTableSize == 0 : O[n] = scale * W[ids[n]]                       (e.g. Llama3 — RoPE)
posTableSize  > 0 : O[n] = scale * W[ids[n]] + posW[cachedCount + n]  (e.g. GPT-2)

Only Gemma uses scale (sqrt(D)). It stays out of the weights because a tied LM head
reads the same table. posTableSize > 0 adds a "posWeight" [posTableSize, D] slot
and needs setKvCacheCounter() to supply cachedCount; positions restart at each
sequence, which the kernel only has to work out when there is more than one.

**커널**

- `vai/shaders/token-embedding.comp`

---

## TopKSampling

row-wise top-k sampling, one kernel run as two passes

소스: [`vai/nodes/topk-sampling.h`](../../vai/nodes/topk-sampling.h)

**시그니처**

- 입력: logits [.., N]
- 출력: out0 [..] (one raw uint32 sampled index per row, in 4-byte slots)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |
| `cands` | temporary |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `k` | `uint32_t` | `40` |  |
| `temperature` | `float` | `1.0f` |  |
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` | logits dtype (output is raw uint32 indices) |

**동작 및 제약**

Every axis ahead of N is a row, the way a Linear takes its leading axes, and they
come back on the output where they went in.

Keeps the k largest logits of each row, applies a temperature softmax over
them, and samples by inverse CDF with a host-provided uniform refreshed
every forward (setUniform()).

Two dispatches of one kernel: G workgroups each select the top-k of one
ceil(N/G)-logit slice — exact while the slice stays register-resident — then
one workgroup merges the G*k candidates and samples. Small rows (G == 1)
skip the second dispatch.

Geometry (workgroup size, register list length), the reduction variant and G
are selected per (shape, k, device) in topkTuning(); the workgroup size is a
spec constant, so the pipeline is requested in prepare() once the shape is
known, and a tuning change re-requests it from the pool (no-op while the
tuning stays put).

**커널**

- `vai/shaders/topk-sampling.comp`

---

## TypeCast

Casts each tensor element to another dtype.

소스: [`vai/nodes/type-cast.h`](../../vai/nodes/type-cast.h)

**시그니처**

- 입력: in0 [any shape] (iDtype)
- 출력: out0 [same shape] (oDtype)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `iDtype` | `vai::Dtype` | `vai::Dtype::None` |  |
| `oDtype` | `vai::Dtype` | `vai::Dtype::None` |  |

**동작 및 제약**

Both dtypes default to the net's hint, so naming only oDtype casts whatever
the producer left. Matching dtypes make out0 the input tensor itself: no
pipeline, no dispatch.

**커널**

- `vai/shaders/cast.comp`

---
