# Gemma3 image/window assertion regression (2026-10-01)

## Reproduction and fixes

Debug GDB stopped at `Attention::updateCausality` with `windowSize=1024`,
`bits=5`, `blockEnd={5,261,0,0,0,0,0}`. The call came from
`Model::prefill`, through `Transformer::updateCausality`, during the first
warm-up. The new image/window test also aborted on the original assertion.

All three attention shaders already combine a window lower bound with a
per-block upper bound. Removed the obsolete prohibition on this combination;
kept the increasing-block-boundary validation. No shader arithmetic changed.

End-to-end execution then exposed two input-lifetime issues in the demo:
`std::move(inputs)` consumed the dictionary on the first warm-up, and Tensor
copies share host storage that is cleared during upload. Preserve the dictionary
and restore prompt IDs using `inTokens.setData(tokens.data())` before each
prefill. The preceding `resetCache()` synchronizes outstanding work.

## Validation

Environment: Linux, NVIDIA GeForce RTX 5080, driver 595.336.0, Debug build
(assertions enabled), Gemma3 4B fp16 checkpoint.

| Run | Result | Maximum normalized error |
| --- | --- | --- |
| Default path (coopmat FA, scalar decode, naive fallback) | 42/42 PASS | 0.00348 |
| Scalar FA forced using a temporary header; naive fallback retained | 42/42 PASS | 0.000487 |
| Gemma3 demo, 2 warm-ups, timed prefill and generation | exit 0, Generation complete | not a logit comparison |

Each suite includes 17 new cases: image blocks in the middle/start/end or alone,
unaligned boundaries, a 16-token window, cached prefixes, resetting to causal
decode, Gemma3 head dimensions with 1100 tokens and a 1024-token window, and
split-K decode with a short window. Non-finite values now fail comparison.
The CPU reference uses double-precision softmax. Error is max absolute difference
divided by max(reference RMS, absolute reference element), not token accuracy.
Existing tolerances remain 0.03 for fp32 output and 0.05 for fp16 output.

The demo produced a complete answer beginning:
“Based on the image, he seems to be looking slightly upwards and to his left.”
This checks successful execution, not semantic correctness or equality to
HuggingFace logits. Other GPUs and a full PyTorch layer comparison were not run.
`git diff --check` passed.

## Reproduce

The existing `tests/CMakeLists.txt` contains uncommented prose and duplicate
target definitions, so the attention executable was compiled directly against
the existing Debug libraries without changing that unrelated file:

```bash
cmake --build build --target demo-gemma3 --parallel 2
g++ -g -std=gnu++23 -I. -Ivai -Iexternal/eva/src \
  -DSHADER_OUTPUT_DIR=\"$PWD/build/shaders\" -DUSE_PYTORCH_CONVENTION \
  tests/attention/main.cpp build/lib/libvAi.a build/Debug/lib/libeva.a \
  /usr/lib/x86_64-linux-gnu/libvulkan.so build/Debug/lib/libspirv-reflect-static.a \
  -o /tmp/test-attention
/tmp/test-attention --verify-only
build/bin/demo-gemma3 --size 4B --precision fp16
```

Select the RTX 5080 at the device prompt. For the scalar check, copy
`vai/nodes/attention.h` into a temporary `include/nodes` directory, insert
`return {};` at the start of `ggmlFACmTuningParams`, and place that temporary
include directory first on the test compile command. This override was only
used for validation and is not part of the production change.

Session logs and GDB output are in `/tmp/gemma3-debug/`.
