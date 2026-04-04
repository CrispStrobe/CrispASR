# Optimization Roadmap: cohere-whisper.cpp

**Original baseline**: ~5 min for 4s audio on 4 CPU cores (~75× slower than real-time).
**Target**: Match ONNX int8 / Rust / PyTorch F16 speeds (~0.3–1× real-time on CPU).

---

## Status

| Priority | Optimization | Status | Notes |
|----------|-------------|--------|-------|
| 1 | FFT for STFT | **DONE** | FFTW3f (see below) |
| 2A | OpenBLAS GEMM | **DONE (intermediate)** | `cblas_sgemm` in `ct_linear`; will be superseded by ggml port |
| 3 | Cross KV caching | **DONE** | `cohere_precompute_cross_kv()` called once per utterance |
| 2B | ggml compute graph port | TODO (next major step) | Supersedes 2A; unlocks F16, GPU, quantization |
| 4 | F16 matmul | blocked on 2B | Free once on ggml graph |
| 5 | Quantized GEMM (Q8/Q4) | blocked on 2B | Re-export GGUF with quant weights |
| 6 | GPU backend (Metal/CUDA) | blocked on 2B | Zero code change once on ggml graph |
| 7 | Streaming / chunked | independent | Long-audio support |
| 8 | Batched encoder (conv STFT) | independent | GEMM-based STFT |

---

## Priority 1 — STFT: FFTW3f (DONE)

**Was**: O(n_fft²) ≈ 512² = 262,144 ops/frame direct DFT.
**Now**: `fftwf_plan_dft_r2c_1d` in `cohere_compute_features` — O(n_fft·log n_fft) ≈ 4,608 ops/frame.
**Speedup**: ~57×.

**Why FFTW3f and not whisper.cpp's own `fft()`**:
- whisper.cpp has a hand-rolled recursive Cooley-Tukey (whisper.cpp:3060) with a precomputed
  sin/cos table. It is O(N log N) but scalar — no SIMD.
- FFTW3f uses AVX/AVX2/SSE2 automatically. It is the better wheel and is already a system
  package (`libfftw3f-dev`).
- Downside: external dependency, so upstreaming to mainline whisper.cpp would require making
  FFTW3 optional (as BLAS already is). For this fork it is fine.

CMake: `find_library(FFTW3F_LIB fftw3f)` + link in `src/CMakeLists.txt`.

---

## Priority 2A — OpenBLAS GEMM (DONE, intermediate)

**Was**: Triple-nested scalar loops in `ct_linear` (OMP-parallel over output dim only).
**Now**: `cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, n_out, n_in, ...)` + bias loop.
**Speedup**: ~10–30×.

**Layout**: `in` (T×n_in), `w` (n_out×n_in) → `out = in @ w^T` (T×n_out), then add bias.

**Why this is an intermediate step**:
ggml already has `ggml/src/ggml-blas/ggml-blas.cpp` which does this exact same call (line 141,
206), plus:
- Falls back to AVX2/NEON kernels when BLAS is not built (`-DGGML_BLAS=OFF`)
- Routes to CUDA/Metal/ROCm when enabled — zero code changes
- Supports F16 weight storage via `ggml_mul_mat` (halves memory bandwidth)
- Enables quantized GEMM (Q4_K, Q8_0)

The current `cblas_sgemm` hard-wires OpenBLAS only. It will be removed when the ggml
compute graph port (Priority 2B) is done.

CMake: `find_package(BLAS)` + link in `src/CMakeLists.txt`.

---

## Priority 3 — Cross KV caching (DONE)

**Was**: `cohere_decode_step` recomputed CK and CV from `enc_out` on every autoregressive step:
```cpp
auto CK = ct_linear(enc_out, d, T_enc, cross_k_w, d, cross_k_b);  // 8 layers × 2
auto CV = ct_linear(enc_out, d, T_enc, cross_v_w, d, cross_v_b);
```
For n_steps=20, T_enc=53: `20 × 8 × 2 × T_enc × d²` redundant FLOPs ≈ 1.8 × 10¹² FLOP saved.

**Now**: `cohere_precompute_cross_kv()` runs once after encoding, stores results in
`cohere_context::cross_kv_k/v` (mirrors `whisper_kv_cache kv_cross` in whisper.cpp).
`cohere_decode_step` signature no longer takes `enc_out` pointer.

This is the same design pattern as `kv_cross` in whisper.cpp (whisper.cpp:858, 2303–2338).

---

## Priority 2B — ggml compute graph port (next major step)

Replace all imperative `ct_linear` + `ct_layer_norm` calls with a proper ggml compute graph.
This is the unlock for everything else.

**Key graph nodes needed**:
```
ggml_mul_mat       → GEMM (replaces ct_linear, cblas_sgemm goes away)
ggml_add           → bias addition
ggml_norm          → layer normalization (replaces ct_layer_norm)
ggml_silu / ggml_relu → activations
ggml_conv_1d       → 1D depthwise/pointwise conv (Conformer convolution module)
ggml_soft_max      → attention softmax
ggml_rope          → rotary embeddings (if needed)
```

The encoder alone has ~48 × (6 GEMM + 3 layer_norm + 3 conv1d + softmax) ≈ 600 graph nodes.

**How to approach**:
1. Port the decoder first (8 layers, simpler, d=1024) — validate against current output
2. Port the encoder (48 layers, d=1280, conv subsampling) — validate
3. Remove `ct_linear`, `ct_layer_norm`, OpenBLAS dependency
4. Enable `-DGGML_BLAS=ON` for the OpenBLAS path via ggml's own infrastructure

**Reference**: `whisper_encode_internal` and `whisper_decode_internal` in whisper.cpp are the
patterns to follow. The cross KV cache can stay as a vector<float> or move to ggml tensors.

---

## Priority 4 — F16 matmul (blocked on 2B)

Once on the ggml compute graph, `ggml_mul_mat` with F16 weight tensors automatically uses
AVX2 F16→F32 conversion, halving memory reads for the 672 weight matrices.

The GGUF already stores weights as F16. The current `ct_to_f32` discards this advantage.
After the ggml port, keep tensors F16 and let `ggml_mul_mat` handle the conversion.

---

## Priority 5 — Quantized GEMM (Q8_0 / Q4_K_M) (blocked on 2B)

Re-export GGUF with quantized encoder/decoder weight matrices:
```bash
./quantize cohere-transcribe.gguf cohere-transcribe-q8.gguf Q8_0
./quantize cohere-transcribe.gguf cohere-transcribe-q4km.gguf Q4_K_M
```

Expected sizes and quality:
- F16: ~2.5 GB, baseline
- Q8_0: ~1.3 GB, ~1–2% WER degradation
- Q4_K_M: ~700 MB, ~3–5% WER degradation

---

## Priority 6 — GPU backend (Metal / CUDA) (blocked on 2B)

After ggml compute graph port, GPU dispatch is nearly free:
```cmake
-DGGML_METAL=ON   # Apple Silicon: M1 Pro → ~0.1× real-time for 4s audio
-DGGML_CUDA=ON    # NVIDIA: RTX 3090/A100 → ~0.05× real-time
```

---

## Priority 7 — Streaming / chunked processing (independent)

For long audio:
- Chunk into overlapping segments (e.g., 30s with 2s overlap)
- Reuse KV cache across chunks (timestamp-aware)
- Producer-consumer pipeline: encoder + decoder overlap

---

## Priority 8 — Batched STFT via Conv1d GEMM (independent)

Replace the per-frame FFTW3f call with a single GEMM over all frames:
precompute a (n_fft/2+1, n_fft) real/imaginary filter matrix and apply as one batched matmul.
This is what the ONNX model does. Integrates naturally into the ggml graph.

---

## Bottleneck Analysis (measured, 11s JFK audio, 4 threads)

**Baseline estimate** (original scalar code): ~825s for 11s audio (scaled from 5min/4s).

| Component | Before | After P1+P2A+P3 (measured) | Next target |
|-----------|--------|---------------------------|-------------|
| STFT | ~4 min | ~3–5 s (FFTW3f) | keep |
| ct_to_f32 per-inference | 0 | ~30–40 s (new bottleneck, 3.8 GB scalar F16→F32) | P2A-cache fix |
| Encoder GEMM (48 layers) | ~8 min | ~30–40 s (OpenBLAS) | ggml F16 |
| Encoder attn scalar loops | ~3 min | ~15–20 s (not yet BLAS-ized) | ggml |
| Decoder cross-KV | ~45 s | ~0 s (pre-cached) | done |
| Decoder self-attn | ~30 s | ~5–10 s (OpenBLAS) | ggml |
| Memory alloc/sys overhead | ~0 | ~20–30 s (high sys time) | pre-alloc buffers |
| **Total (measured)** | **~825 s** | **104 s** | **→ ~20 s target** |

**Measured speedup**: 825s → 100s = **~8.3× total** (latest: lazy F32 cache)

| Session | Wall | User | Sys | Note |
|---------|------|------|-----|------|
| Baseline (scalar) | ~825s | — | — | — |
| P1+P2A+P3 (FFTW3f+OpenBLAS+cross-KV) | 104s | 262s | 107s | OpenBLAS verified linked |
| + BLAS attention | 105s | 135s | 106s | cblas_sgemm in ct_rel_pos_mha |
| + lazy F32 cache | 100s | 79s | 67s | user time /4 threads ≈ 20s actual CPU |
| + EncScratch + AVX2 F16C | **32s** | 32s | 22s | **~3.1× over prev, ~26× total** |

**Remaining bottleneck (33s wall, ~25× total):**
Almost entirely encoder BLAS (~25s single-thread). Multi-threading gives only ~1.77×
for T=138 encoder matrices — thread overhead dominates for small M. The cblas_sgemm + F32
path has hit its ceiling. Next gains require P2B (ggml graph port).

**Next steps in priority order:**
1. **P2B — ggml compute graph**: `ggml_mul_mat` with F16 weights → 2× memory BW + GPU dispatch
2. Quantization (Q8_0): ~2× speedup on top of F16 (blocked on P2B)
3. GPU (CUDA/Metal): ~10-20× over current (blocked on P2B)

After P2B (ggml graph + F16):
- F16 matmul: ~2× over F32 OpenBLAS → ~15–20 s total
- GPU (CUDA/Metal): ~20–100× over CPU → **real-time or better**

---

## Estimated Speedup Summary (revised)

| Optimization | Component | Speedup | Status |
|-------------|-----------|---------|--------|
| FFTW3f for STFT | Feature extraction | 50–60× | DONE |
| OpenBLAS GEMM | All ct_linear calls | 10–20× | DONE (intermediate) |
| Cross KV caching | Decoder | 2–10× | DONE |
| F32 weight cache (lazy) | Weight loading | ~3.3× (user 262s→79s) | DONE |
| EncScratch + AVX2 F16C on-the-fly | Memory churn + conversion | **3.1× (100s→32s)** | DONE |
| BLAS-ize attn score loops | Encoder attn | ~2× | TODO |
| ggml compute graph | All | enables F16/GPU/quant | TODO (P2B) |
| F16 matmul | Encoder/decoder | 2× | blocked on ggml |
| Q8_0 quant | Encoder/decoder | 1.5–2× | blocked on ggml |
| GPU (CUDA/Metal) | All | 20–100× | blocked on ggml |

**Measured (P1+P2A+P3, 11s audio)**: 825s → 104s = **~8×**
**Measured (P1+P2A+P3+scratch+F16C)**: 825s → 32s = **~26× total**
**After decoder F16C fix**: est. ~20–25s
**With ggml + F16 (P2B)**: est. ~10–15s → approaching real-time on CPU
**With GPU**: real-time easily achievable

---

## Comparison with Reference Implementations

| Implementation | Speed (11s audio) | Notes |
|----------------|------------------|-------|
| ONNX int8 CPU | ~1–2s | int8 encoder + F32 decoder |
| PyTorch F16 GPU | ~0.3s | A100 / RTX 3090 |
| Rust (cpu, no simd) | ~90–180s | pure Rust, no BLAS |
| **Ours (baseline)** | **~825s** | naive DFT + scalar GEMM |
| **Ours (P1+P2A+P3, measured)** | **104s** | FFTW3f + OpenBLAS + cross KV |
| **Ours (lazy F32 cache, measured)** | **100s** | user 79s, sys 67s (page-fault storm) |
| **Ours (+ EncScratch + AVX2 F16C, measured)** | **32s** | user 32s, sys 22s — **26× total** |
| **Ours (+ decoder F16C, est.)** | **~20–25s** | |
| **Ours (ggml + F16, est.)** | **~10–15s** | CPU target |
| **Ours (ggml + GPU, est.)** | **~0.3–1s** | stretch goal |
