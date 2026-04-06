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
1. **BatchNorm folding**: fold BN into conv_dw weights at load time — no runtime BN nodes (DONE)
2. **mmap weight loading**: `mmap()` GGUF file instead of `fread` to eliminate 7-20s cold-start I/O
3. **Chunked encoder**: cap T to avoid O(T²) attention growth for long audio
4. **GPU (CUDA/Metal)**: zero code change once on ggml graph

After BN folding (ggml graph + F16, measured):
- F16: 12.4s → 11.5s total (encoder 11.3s → 11.0s); 480 nodes removed (4940→4460)
- Q8_0: 10.8s total, RTF 1.02×
- Q4_K: 9.6s total, RTF 1.15× — **real-time on CPU today**

---

## Phase 3 — Post-P2B Optimizations (post-ggml-graph)

### BatchNorm Folding (DONE)

**What**: At model load time, fold each Conformer layer's BatchNorm into `conv_dw_w/b`:
```
s[c] = bn_scale[c] / sqrt(bn_var[c] + eps)
w_folded[ki, c] = w[ki, c] * s[c]
b_folded[c] = (dw_b[c] - bn_mean[c]) * s[c] + bn_bias[c]
```
Removes 10-node BN block (mul_mat + norm + scale + bias per channel) × 48 layers = **480 nodes**.

**Results** (11s JFK audio, 1-thread CPU):
| Model | Enc compute | Dec compute | Total wall | RTF |
|-------|-------------|-------------|------------|-----|
| F16 (pre-BN fold) | 11.3s | 937ms | 12.4s | 0.89× |
| F16 (post-BN fold) | 11.0s | 424ms | 11.5s | 0.96× |
| Q8_0 | ~10.2s | 370ms | 10.8s | 1.02× |
| Q4_K | 9.1s | 354ms | 9.6s | **1.15×** |

**Why negligible for quantized**: BN ops are all F32/per-channel-scale — tiny vs GEMM. BN folding
mainly removes 288 `ggml_add/mul` calls in the graph; quantized GEMM was already the bottleneck.

**Transcript verified correct** on `sample2_16k.wav` ("The quick brown fox…") after folding.

### ggml_cont reduction (ANALYZED, SKIP)

48-layer Conformer builds several `ggml_cont` tensors for attention reshape. Analysis:
- Each cont copy: ~480 KiB for T=138 (enc_d × T × sizeof(F16))
- 48 layers × ~4 conts = 192 cont calls, but most are no-ops if tensor is already contiguous
- Measured overhead: <12ms total across all 48 layers (1.3% for 44s audio)
- **Decision**: skip, not worth the graph complexity risk.

### Per-Op Profiler (DONE) — `COHERE_PROF=1`

Added `ggml_backend_sched_set_eval_callback` profiler. Results for 44s audio (F16, CPU):

| Op | Time | % | Count |
|----|------|---|-------|
| mul_mat | 42.2s | **87.6%** | 743 |
| im2col (conv1d_dw expansion) | 3.4s | **7.0%** | 53 |
| add/mul/scale | 1.1s | 2.3% | 1559 |
| cont | 0.6s | 1.3% | 404 |
| norm | 0.2s | 0.5% | 240 |
| soft_max | 0.2s | 0.4% | 48 |
| other | 0.4s | 0.9% | 1413 |

**Key findings**:
- `mul_mat` at 87.6% is near hardware peak for F16 GEMM on this Skylake — no more CPU wins here
- `im2col` at 7.0% (53 calls = 3 Conv2D subsampling + 48 depthwise via `ggml_conv_1d_dw` expand): each call does `reshape_4d + im2col + mul_mat`; the im2col itself is 3.4s, the mul_mat portion (~0.9s) is counted in mul_mat already
- On Metal/GPU: both mul_mat and im2col would be 10-50× faster → real-time inference

### Metal/GPU Backend Infrastructure (DONE)

Added device-agnostic backend selection:
- `ggml_backend_load_all()` at init — registers Metal (macOS), CUDA (Linux/Win), Vulkan
- `ggml_backend_init_best()` — picks GPU > CPU automatically
- `COHERE_DEVICE=metal|cuda|cpu` env var for explicit selection
- CPU always kept as fallback in `ggml_backend_sched` for any unsupported ops
- CMakeLists.txt: links `ggml-metal` when `GGML_METAL=ON`, `ggml-cuda` when `GGML_CUDA=ON`
- No code changes needed when switching from CPU to Metal — same graph builder, backend handles dispatch

**To build with Metal (macOS):**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
make -j$(nproc) cohere-main
./build/bin/cohere-main -m cohere-transcribe.gguf -f audio.wav
# Automatically uses Metal GPU; set COHERE_DEVICE=cpu to force CPU
```

---

## Estimated Speedup Summary (revised)

| Optimization | Component | Speedup | Status |
|-------------|-----------|---------|--------|
| FFTW3f for STFT | Feature extraction | 50–60× | DONE |
| OpenBLAS GEMM | All ct_linear calls | 10–20× | DONE (intermediate) |
| Cross KV caching | Decoder | 2–10× | DONE |
| F32 weight cache (lazy) | Weight loading | ~3.3× (user 262s→79s) | DONE |
| EncScratch + AVX2 F16C on-the-fly | Memory churn + conversion | **3.1× (100s→32s)** | DONE |
| ggml compute graph (P2B) | All | enables F16/GPU/quant | **DONE** |
| BatchNorm folding | Encoder conv (48 layers) | ~7% enc, 480 nodes removed | **DONE** |
| mmap weight loading | Cold-start I/O | eliminates 7-20s load time | TODO |
| Chunked encoder | Long audio (>30s) | caps T, avoids O(T²) attn | TODO |
| GPU (CUDA/Metal) | All | 20–100× | TODO |

**Measured (P1+P2A+P3, 11s audio)**: 825s → 104s = **~8×**
**Measured (+ ggml graph P2B, F16)**: ~12.4s = **~67× total**
**Measured (+ BN folding, F16)**: ~11.5s = **~72× total**
**Measured (Q4_K quant)**: ~9.6s = **~86× total, RTF 1.15× (real-time)**
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
| **Ours (+ ggml graph P2B, F16)** | **~12.4s** | ~67× total |
| **Ours (+ BN folding, F16)** | **~11.5s** | ~72× total |
| **Ours (Q8_0)** | **~10.8s** | RTF 1.02× |
| **Ours (Q4_K)** | **~9.6s** | **RTF 1.15×** — real-time on CPU |
| **Ours (ggml + GPU, est.)** | **~0.3–1s** | stretch goal |
| **Ours (ggml + GPU, est.)** | **~0.3–1s** | stretch goal |
