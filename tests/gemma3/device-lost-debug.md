# Gemma3 fp32 device-lost debugging (2026-09-22)

## Environment and reproduction

- AMD Radeon Graphics / RADV GFX1151, Mesa 25.2.8.
- Kernel: 7.0.0-30-generic; existing Debug CMake build.
- `test-gemma3-dump`, fp32 weights/KV, 896x896 demo image, 273 prompt tokens.
- Demo preprocessing and free-running decode; no shared PyTorch pixels or forced IDs.

```bash
printf '0\n' | build/bin/test-gemma3-dump --steps 64 --out /tmp/gemma3-debug/recheck
```

The original binary repeatedly produced a compute-ring timeout and queue reset.
The visible failure at `eva-runtime.cpp:2053` was a later readback submit, not
necessarily the command that caused the hang. The first saved `vis.patches.npy`
in failed runs differed from successful reference runs and must not be trusted.

## Direct GPU evidence

`RADV_DEBUG=hang,noumr` generated:

`/home/vaistrata/radv_dumps_713965_2026.09.22_00.33.02/`

The last reached trace point was **2101**. The next dispatch had:

- Push constants: M=273, N=20480, K=2560, splitSize=2560.
- Dispatch: (640, 2, 1), workgroup size 128, wave64.
- SPIR-V SHA-256: `8eebf52eeeb7744146c9603d5825707d0161a04eed62c64e7e52f0581af4f702`.
- This hash exactly matches `build/shaders/linear-cm.comp.spv`.
- Shader statistics: 256 VGPRs, 128 SGPRs, 43,520 bytes LDS, no spills.

This is the text decoder MLP's combined gate/value projection
(`layer.ffn.gateValue`): 2560 input channels -> 2 x 10240 output channels.
Eleven earlier N=20480 dispatches precede this one in the captured trace,
placing this occurrence at decoder layer index 11 (the twelfth layer).
The selected cooperative-matrix tile was wgM=256, wgN=32, wgK=64,
sgM=256, sgN=16.

This identifies an observed hanging kernel/path. It does not prove whether the
underlying defect is shader lowering, RADV scheduling, firmware, or hardware
resource handling. High register/LDS usage is a diagnostic observation, not
proof of the low-level cause.

## Controlled experiments

| Experiment | Result |
| --- | --- |
| Original full prefill | Repeated GPU timeout/reset |
| Submit/wait after each of 1493 nodes | Passed; sum of host-measured node intervals ~1.61 s |
| Submit in groups of 100 / 500 / 1000 nodes | Passed in diagnostic runs |
| One full legacy submit | Failed; not specific to vkQueueSubmit2 |
| Full memory barrier before each node | Failed |
| Compute-to-transfer barrier before readback | Failed |
| Wait for full forward / producer before readback | Some passes, later failure; not a fix |
| RADV_DEBUG=noibs | Failed |
| Reduce fp32 CM row tile 256 -> 128 | One complete 64-step pass, repeat failed; not a fix |
| Use scalar fp32 Linear tiles for fp32 input AND weight | Three consecutive complete 64-step passes |

Timing-sensitive successes initially implicated readback or tile size. Repeated
runs disproved those fixes; their instrumentation and code changes were removed.

## Retained change

In `vai/nodes/linear.h`, `ggmlMulmmTileParams()` only considers the current
fp16 cooperative-matrix kernel when `stageF32` is false. The existing scalar
fallback stages and accumulates fp32 when both input and weight are fp32.

Besides avoiding the observed hang path, this prevents this Linear tile path
from silently rounding fp32 operands to fp16. Lower-precision operands retain
the existing selection behavior. This is a precision-based rule across devices,
not a device-name-specific workaround. Performance can differ from the CM path;
no cross-device performance claim is made.

Attention still has its own mixed-precision tuning; this change does not claim
that the entire model now performs every operation in strict fp32.

## Validation and remaining limits

Final runs:

- `/tmp/gemma3-debug/scalar32-a.log`
- `/tmp/gemma3-debug/scalar32-b.log`
- `/tmp/gemma3-debug/scalar32-c.log`

Each completed prefill and 64 decode steps, wrote 171 tapped arrays plus three
ID arrays, and exited successfully. All 174 arrays per run were checked for
NaN/Inf (none). All 64 selected tokens matched between the three runs.
No additional kernel compute-ring timeout was observed during those runs.

Prefill logits matched between runs. Decode logits began differing at step 46;
maximum absolute differences relative to run A were 0.0379791 (B) and 0.0348625
(C), without changing selected tokens. This residual numerical reproducibility
issue is not resolved by the device-lost fix. Outputs also differ numerically
from the old CM path because that path rounded operands to fp16.

The model was not compared with PyTorch during this debugging session.
Temporary node/batch/readback instrumentation was removed, and existing user
changes were preserved. Build target `test-gemma3-dump` and diff whitespace
checks for the changed files passed.

RADV diagnostic procedure: https://docs.mesa3d.org/drivers/amd/hang-debugging.html
