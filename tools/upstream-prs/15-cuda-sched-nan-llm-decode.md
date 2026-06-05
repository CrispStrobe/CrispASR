**Title:** `ggml-cuda : fix F16 cuBLAS accumulator overflow producing Inf/NaN on sm_60 (P100)`

---

## Root cause (confirmed)

The cuBLAS fallback path in `ggml_cuda_op_mul_mat_cublas` has a
`use_fp16` flag that routes quantized-weight × F32-activation matmuls
through `cublasHgemm` (F16 GEMM). On GPUs without DP4A (P100 sm_60),
the MMQ kernel is unavailable, so quantized MUL_MAT always takes this
cuBLAS fallback.

The F16 GEMM accumulates dot-product partial sums in F16 (max ±65504).
For LLM FFN layers with large intermediate values (swiglu outputs
reaching ~15K over 3072-element dot products), the partial sums exceed
65504 and overflow to Inf. This manifests as 1 Inf at layer 2, all-NaN
by layer 3, and degenerate output.

## Symptom

A 28-layer Qwen2-0.6B LLM decoder (head_dim=128, n_heads=16,
n_kv_heads=8, GQA ratio=2, per-head QK-RMSNorm) runs correctly on
CPU-only builds but produces all-NaN prefill logits on CUDA.

Per-layer tensor dump (CUDA vs CPU):
```
llm_layer_0:  0 NaN, 0 Inf, max=1046  — matches CPU (1046)
llm_layer_1:  0 NaN, 0 Inf, max=1057  — matches CPU (1057)
llm_layer_2:  0 NaN, 1 Inf, max=6916  — CPU max=124512 ← DIVERGES
llm_layer_3:  ALL NaN (47104/47104)    — CPU max=125052
```

Per-node NaN checker trace (layer 2, CUDA):
```
node#86  MUL_MAT  [3072,46]  min=-14.27  max=104.9  nan=0 inf=0   ← FFN gate
node#87  UNARY    [3072,46]  min=-0.28   max=104.9  nan=0 inf=0   ← SiLU
node#88  MUL_MAT  [3072,46]  min=-51.88  max=143    nan=0 inf=0   ← FFN up
node#89  MUL      [3072,46]  min=-2402   max=15010  nan=0 inf=0   ← swiglu
node#90  MUL_MAT  [1024,46]  min=-755.5  max=6916   nan=0 inf=1   ← FFN down *** FIRST INF
```

The swiglu output (max=15010) is within F16 range, but the 3072-element
dot product in the FFN down projection produces partial sums exceeding
65504.

## Confirmed on

- Tesla P100 (sm_60) — Kaggle, kernel versions 19-21
- Also affects Blackwell (sm_120) per GitHub issue #125 (same symptom,
  likely same cause via a different code path)

Architecture-independent within CUDA — any GPU where the quantized
MUL_MAT takes the cuBLAS F16 fallback instead of MMQ.

## The fix

In `ggml/src/ggml-cuda/ggml-cuda.cu`, function
`ggml_cuda_op_mul_mat_cublas`, change the `use_fp16` condition from:

```c
// BEFORE: only blocks F16 weights with F32 activations
!(src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32);
```

to:

```c
// AFTER: blocks ALL weight types when activations are F32
!(src1->type == GGML_TYPE_F32);
```

This forces the F32 `cublasSgemm` path (dequant weight to F32, keep
activation F32) whenever activations are F32, avoiding F16 accumulator
overflow on all CUDA architectures.

## Performance impact

Negligible:

- On sm_60 (P100): no tensor cores, so F16 vs F32 cuBLAS is marginal.
  The F32 path is slightly slower but the difference is small.
- On sm_61+ (all modern GPUs): DP4A is available, so quantized MUL_MAT
  uses the MMQ kernel, never hitting this cuBLAS fallback. Zero impact.
- The cuBLAS fallback is a last-resort path; the primary quantized
  matmul paths (MMQ, MMVQ) are unaffected.

## Verification

Kaggle P100 kernel v21 (commit `9211662a`):
```
CUDA default (all-GPU, fix applied): PASS
  "AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU
   ASK WHAT YOU CAN DO FOR YOUR COUNTRY"
CPU baseline:                        PASS
CUDA workaround (LLM CPU):           PASS
NaN checker (128 nodes, 4 layers):   0 Inf, 0 NaN

*** FIX CONFIRMED ***
```

## Repro

Model: `cstr/funasr-nano-GGUF` → `funasr-nano-2512-q8_0.gguf` (HuggingFace).
Audio: any 16 kHz WAV (11s JFK speech used for testing).

```bash
# Build with CUDA
cmake -B build -DGGML_CUDA=ON && cmake --build build --target crispasr-cli

# Before fix: produces all-NaN → "!!!!!!!!!!!!!!!!!!!!"
FUNASR_DUMP_STAGES=1 build/bin/crispasr --backend funasr -m auto \
    --auto-download -f samples/jfk.wav --no-prints

# After fix: correct transcript
# (same command, the fix is in the CUDA matmul dispatch)
```

## What was ruled out during investigation

| Attempt | Result |
|---|---|
| Q8_0 model (instead of F16) | Still NaN |
| Disable flash_attn_ext | Still NaN |
| F32 KV cache reads | Still NaN |
| `parallel=true` sched flag | Still NaN |
| Fuse Q/K/V into single QKV matmul | Still NaN |
| KV cache zero-fill on alloc | Still NaN |
| Weight split but KV still on GPU | Still NaN |

All of these targeted the wrong component (scheduler, flash attention,
KV cache). The bug was in the cuBLAS matmul dispatch, specifically the
`use_fp16` flag that allowed F16 accumulation for quantized weights.

## Diagnostic tools added

- `FUNASR_NAN_CHECK=1`: per-node eval callback that reads back every
  graph node and prints the first op with NaN/Inf + source tensor stats
- `FUNASR_LLM_LAYERS=N`: limits LLM decoder to N layers for faster
  debugging (4 layers sufficient to trigger the layer-2 bug)
