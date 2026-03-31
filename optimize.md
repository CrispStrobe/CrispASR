# Optimization Roadmap: cohere-whisper.cpp

**Current state**: Correct transcription, ~5 min for 4s audio on 4 CPU cores (~75× slower than real-time).
**Target**: Match ONNX int8 / Rust / PyTorch F16 speeds (~0.3–1× real-time on CPU).

---

## Bottleneck Analysis

### Where time is spent (rough breakdown for 4s audio, 4 threads)

| Component | Est. time | Notes |
|-----------|-----------|-------|
| STFT (DFT loop) | ~1–2 min | O(T·K·N) direct DFT, should be O(T·K·logN) FFT |
| Encoder GEMM (48 layers) | ~3 min | naive loops, no BLAS |
| Decoder (8 layers × 9 prompt) | ~10 s | small d=1024, manageable |

The **encoder is the dominant cost** due to 48 Conformer layers × d=1280 × ffn=5120.

---

## Priority 1 — Replace naive DFT with real FFT (~60× STFT speedup)

**Current**: `cohere_compute_features` computes DFT by direct summation:
```cpp
for k in 0..n_freqs:
    for n in 0..n_fft:
        re += frame[n] * cos(2π k n / n_fft)
```
This is O(n_fft²) ≈ 512² = 262,144 ops per frame.

**Fix**: Use `kiss_fft` (already bundled in ggml) or compute via real FFT.
Cost with FFT: O(n_fft · log2(n_fft)) = 512 × 9 ≈ 4,608 ops per frame — **57× faster**.

Implementation: replace the double loop with `ggml_fft_r2c` or add `kiss_fft` call.
Alternatively: precompute DFT filter weights as a convolution matrix and use a single GEMM.

```cpp
// Replace inner DFT loop with:
#include <kiss_fft/kiss_fftr.h>
kiss_fftr_cfg cfg = kiss_fftr_alloc(n_fft, 0, NULL, NULL);
std::vector<kiss_fft_scalar> in(n_fft);
std::vector<kiss_fft_cpx> out(n_fft/2+1);
// copy windowed frame into in[], call kiss_fftr
kiss_fftr(cfg, in.data(), out.data());
// power[k] = out[k].r*out[k].r + out[k].i*out[k].i
```

---

## Priority 2 — Replace naive GEMM with BLAS / GGML compute graph (~10–30× speedup)

**Current**: `ct_linear` and all attention matmuls use triple-nested loops.

### Option A: OpenBLAS / MKL (quick win)

Add a BLAS call in `ct_linear`:
```cpp
#include <cblas.h>
// out (n_out × T) = W (n_out × n_in) × in (n_in × T)
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n_out, T, n_in,
            1.0f, w, n_in, in, T, 0.0f, out.data(), T);
```
OpenBLAS uses AVX2/AVX-512 kernels, typically 10–30× faster than scalar loops.

CMake change:
```cmake
find_package(BLAS REQUIRED)
target_link_libraries(cohere ${BLAS_LIBRARIES})
```

### Option B: Port to ggml compute graph (best long-term)

Replace imperative `ct_linear` + `ct_layer_norm` calls with `ggml_mul_mat`, `ggml_norm`, etc.
This enables:
- Automatic F16 matmul via `ggml_mul_mat` (2× memory bandwidth vs F32)
- Metal/CUDA backend without code changes
- Quantized GEMM (Q4_K, Q8_0) for further compression

Key graph nodes needed:
```
ggml_mul_mat       → GEMM (weight × input)
ggml_add           → bias addition
ggml_norm          → layer normalization
ggml_silu / ggml_relu → activations
ggml_conv_1d       → 1D depthwise/pointwise conv
ggml_soft_max      → attention softmax
```

The encoder alone has ~48 × (6 GEMM + 3 layer_norm + 3 conv1d + softmax) ≈ 600 ops.
With ggml graph: can be fused, scheduled, and dispatched to GPU.

---

## Priority 3 — Pre-project cross KV once per utterance

**Current**: On every decode step, `cohere_decode_step` recomputes all 8 layers' cross K and V from `enc_out`:
```cpp
auto CK = ct_linear(enc_out, d, T_enc, cross_k_w.data(), d, cross_k_b.data());
auto CV = ct_linear(enc_out, d, T_enc, cross_v_w.data(), d, cross_v_b.data());
```
For n_dec_steps decode steps: `n_dec_steps × 8 × 2 × T_enc × d²` redundant FLOPs.

**Fix**: Pre-compute all 8 layers' cross K/V once in `cohere_encode()`:
```cpp
// In cohere_context: store precomputed cross KV
std::vector<std::vector<float>> cross_k_cache;  // [n_dec_layers][T_enc × d]
std::vector<std::vector<float>> cross_v_cache;
```
Then `cohere_decode_step` reads from cache instead of recomputing.

This saves `n_decode_steps × 8 × 2 × T_enc × d²` FLOP — for a 20-token transcript with T_enc=53:
`20 × 8 × 2 × 53 × 1024² ≈ 1.8 × 10¹² FLOP` saved.

---

## Priority 4 — F16 matmul (GGML native)

Once on the ggml compute graph, `ggml_mul_mat` automatically uses F16 weight × F32 input
with AVX2 F16→F32 conversion, giving ~2× memory bandwidth improvement over F32×F32.

The weight matrices are already stored as F16 in the GGUF file — the current `ct_to_f32`
immediately converts them, discarding the bandwidth benefit. Keeping them F16 until
the multiply (via `ggml_mul_mat`) would halve memory reads for the 672 F16 weight matrices.

---

## Priority 5 — Quantized GEMM (Q8_0 / Q4_K_M)

Re-export GGUF with quantized encoder/decoder weight matrices:
```bash
# Using llama.cpp quantize tool (once ported to ggml graph):
./quantize cohere-transcribe.gguf cohere-transcribe-q8.gguf Q8_0
./quantize cohere-transcribe.gguf cohere-transcribe-q4km.gguf Q4_K_M
```

Expected sizes:
- F16: ~2.5 GB
- Q8_0: ~1.3 GB, ~1–2% WER degradation
- Q4_K_M: ~700 MB, ~3–5% WER degradation

---

## Priority 6 — GPU backend (Metal / CUDA)

After ggml compute graph port, GPU dispatch is nearly free:

**Metal (Apple Silicon)**:
```cmake
-DGGML_METAL=ON
```
M1 Pro with 16 GB unified memory: expected ~0.1× real-time for 4s audio.

**CUDA**:
```cmake
-DGGML_CUDA=ON
```
RTX 3090 / A100: expected ~0.05× real-time.

The encoder's large FFN (5120-wide, 48 layers) is ideal for GPU: highly parallelizable,
memory-bandwidth-bound with large batches.

---

## Priority 7 — Streaming / chunked processing

For long audio:
- Chunk audio into overlapping segments (e.g., 30s chunks with 2s overlap)
- Reuse KV cache across chunks (timestamp-aware)
- Run encoder and decoder in a producer-consumer pipeline

This matches how the Rust `inference.rs` handles long audio.

---

## Priority 8 — Batched encoder with larger n_fft

The current O(n_fft²) STFT can also be replaced with a Conv1d approach:
precompute a (n_fft/2+1, n_fft) real/imaginary filter matrix and apply as a
single GEMM over all frames simultaneously. This is exactly what the ONNX model does.

---

## Estimated Speedup Summary

| Optimization | Component | Speedup | Effort |
|-------------|-----------|---------|--------|
| FFT for STFT | Feature extraction | 50–60× | Low |
| OpenBLAS GEMM | All matmuls | 10–20× | Low |
| ggml compute graph | All | 2–5× extra | Medium |
| Cross KV caching | Decoder | 2–10× | Low |
| F16 matmul | Encoder/decoder | 2× | Medium (requires ggml) |
| Q8_0 quant | Encoder/decoder | 1.5–2× | Medium |
| GPU (CUDA/Metal) | All | 20–100× | High |

**Combined (FFT + OpenBLAS + cross KV cache)**: ~100–300× → under 2s for 4s audio → real-time.

---

## Comparison with Reference Implementations

| Implementation | Speed (4s audio) | Notes |
|----------------|-----------------|-------|
| ONNX int8 CPU | ~0.5–1s | int8 encoder + F32 decoder |
| PyTorch F16 GPU | ~0.1s | A100 / RTX 3090 |
| Rust (cpu, no simd) | ~30–60s | pure Rust, no BLAS |
| **Ours (current)** | **~300s** | naive DFT + scalar GEMM |
| **Ours (FFT + BLAS)** | **~2–5s** | target |
| **Ours (ggml + GPU)** | **~0.1–0.5s** | stretch goal |
