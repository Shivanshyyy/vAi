# Gemma3 FP32 comparison analysis (2026-09-22)

Measured against the existing `tests/gemma3/dumps/torch_32` files. Original user dumps were preserved.

## Confirmed causes

1. Image input mismatch: reconstructed PyTorch patches vs original vAi `vis.patches` had RMSE 0.0136056235 and max difference 0.2189613. Original vAi patch embedding/logits/picks exactly matched the earlier demo-preprocessing, free-running debug run. Prompt IDs matched, but image pixels did not. The custom resize uses endpoint-aligned bilinear coordinates; the processor uses its own bilinear resize path. These are not equivalent pixel pipelines.
2. Autoregressive input divergence: first differing pick at step 9, first differing fed token at step 10. The PyTorch top-two margin at the first flip was 0.0878. Later comparisons include different histories and KV states, not just implementation error.
3. Remaining attention mixed precision: fp32 buffers still select cooperative-matrix attention with fp16 internals, or scalar FA with fp16 staging/accumulation. Disabling CM and setting stagingF16=false, accF16=false in a temporary diagnostic build greatly reduced the residual error.

Original `txt.embed` differences were confined to the 256 image-token rows: max difference 9.437216. The other 17 text-token rows matched exactly.

## Controlled runs

- Original: existing `tests/gemma3/dumps/vai`, unchanged.
- Shared: same production binary, explicit absolute `--torch` path; same pixels and all 64 PyTorch-fed tokens. Output `/tmp/gemma3-fp32-analysis/shared`.
- Attention FP32: same shared inputs, temporary scalar fp32 attention build. Output `/tmp/gemma3-fp32-analysis/attention-fp32`.

Relative RMS error, as a percentage (100 * RMSE/reference RMS):

| Tap | Original | Shared inputs | Shared + FP32 attention |
| --- | ---: | ---: | ---: |
| vis.embed | 0.930891% | 2.38825e-05% | 2.38825e-05% |
| vis.00 | 1.41638% | 0.00872659% | 7.68623e-05% |
| vis.26 | 17.731% | 0.520308% | 0.00819794% |
| vis.post_norm | 30.7905% | 0.618995% | 0.00606659% |
| vis.proj | 16.3963% | 0.264802% | 0.00294113% |
| txt.embed | 16.0953% | 0.259942% | 0.00288714% |
| txt.32 | 22.3293% | 1.2102% | 0.0116195% |
| txt.33 | 2.07909% | 0.204399% | 0.00127636% |
| txt.final_norm | 2.54762% | 0.234259% | 0.00168899% |
| logits | 1.6142% | 0.155266% | 0.00117687% |
| step.09.logits | 3.98119% | 0.286981% | 0.00210872% |
| step.10.logits | 74.0317% | 0.227762% | 0.00184465% |

Both controlled runs matched all 64 PyTorch picks under teacher forcing. This is not a claim about a separate free-running 64-step run. Maximum relative error across the 64 decode logits was 0.619916% for shared inputs, and 0.00495514% for the FP32-attention experiment.

## Interpretation limits

- Reductions and different reduction trees, FMA, softmax/exp and GELU implementations are plausible sources of the small remaining FP32 differences. They have not been individually isolated.
- `txt.32` measures 273 rows whereas `txt.33` measures only the final row. Their growth ratio cannot show that the last layer fixed the error.
- Relative error growth is not an isolated per-layer error measurement; errors from preceding layers and changed inputs propagate.
- Decode step 0 still uses the prefill KV cache, so it is not an isolated text-only test.
- The npy writer casts tensors to float32; the file dtype alone does not establish compute dtype. Current `dump_torch.py` defaults to float16, so future reference runs should explicitly use `--dtype float32`. The current reference outputs are not simply fp16 values widened to float32.
- The earlier Linear fp32 scalar-path fix remains in place. The temporary attention modifications were restored; these strict-attention numbers do not describe the default binary.

## Re-run with shared inputs

Run from the repository root:

```bash
build/bin/test-gemma3-dump --fp32 --steps 64 \
  --torch /home/vaistrata/Workspace/Gemma3/vAi-internal/tests/gemma3/dumps/torch_32 \
  --out /tmp/gemma3-fp32-analysis/shared-new
python tests/gemma3/compare.py \
  --vai /tmp/gemma3-fp32-analysis/shared-new \
  --torch tests/gemma3/dumps/torch_32
```

Diagnostic logs, full metrics JSON, and the temporary attention patch are in `/tmp/gemma3-fp32-analysis/`.
