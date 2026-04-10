# Performance — CrispASR

## Speech-LLM Performance (Voxtral 4B Realtime)

Measured on an 11s clip (`jfk.wav`), CPU-only (4 threads, AVX2).

| Model | Size | Encoder | Prefill | Decode (ms/tok) | Total | RTFx |
|-------|------|---------|---------|-----------------|-------|------|
| F16 | 8.3 GB | 39s | 30s | 430 | 133s | 0.08× |
| Q8_0 | 4.5 GB | 30s | 9s | 257 | 79s | 0.14× |
| **Q4_K** | **2.4 GB** | **19s** | **3s** | **177** | **49s** | **0.22×** |

Q4_K is 2.7× faster than F16 with identical output quality. German tested and working.

---

## Cross-Model Comparison (Q4_K, jfk.wav 11s, 4 threads, 2 vCPU)

| Model | Params | GGUF | Wall time | RTFx | Architecture |
|-------|--------|------|-----------|------|-------------|
| **Qwen3-ASR** | 900M | 516 MB | **6.5s** | **1.7×** | Whisper enc + Qwen3 LLM |
| **Parakeet TDT** | 600M | 467 MB | **6.9s** | **1.6×** | FastConformer + TDT |
| Canary 1B | 978M | 673 MB | 49s | 0.22× | FastConformer + Transformer |
| Cohere 2B | 2.0B | 1.2 GB | 53s | 0.21× | Conformer + Transformer |
| **Voxtral 4B RT** | 4.4B | 2.4 GB | **54s** | 0.20× | Causal enc + Mistral LLM |
| Voxtral 3B | 3.2B | 2.5 GB | 68s | 0.16× | Whisper enc + Llama LLM |

| **Granite 1B** | 1.0B | 2.8 GB | **22.5s** | 0.49× | Conformer + Q-Former + Granite LLM |

Qwen3-ASR and Parakeet TDT achieve real-time or faster on a 2 vCPU server.
With GPU (Metal/CUDA), all models would be significantly faster.

---

## Cohere Transcribe Performance

## Real-Time Factor (RTFx)

Measured on an 11s clip (`jfk.wav`). RTFx > 1.0 = faster than real-time.

| Model | Size | Backend | Threads | RTFx | Wall time |
|-------|------|---------|---------|------|-----------|
| F16 | 3.85 GB | CPU (AVX2) | 4 | 0.80× | 13.8s |
| Q8_0 | 2.05 GB | CPU (AVX2) | 4 | 1.03× | 10.7s |
| Q6_K | 1.62 GB | CPU (AVX2) | 4 | 1.05× | 10.5s |
| Q5_1 | 1.45 GB | CPU (AVX2) | 4 | 1.06× | 10.4s |
| Q5_0 | 1.38 GB | CPU (AVX2) | 4 | 1.07× | 10.3s |
| Q4_K | 1.21 GB | CPU (AVX2) | 4 | **1.24×** | **8.9s** |
| F16 | 3.85 GB | Metal (M1) | — | **11.9×** | ~0.9s |

---

## Comparison with Reference Implementations

Measured on a 45s clip, same x86 machine (4-thread, no GPU).

| Implementation | Encoder | Decoder | Total | RTFx | Notes |
|----------------|---------|---------|-------|------|-------|
| ONNX INT8 (CPU) | 19.5s | 11.7s | 31.2s | 1.44× | DNNL AVX-512 INT8 GEMM |
| ONNX INT4 (CPU) | 22.5s | 12.7s | 35.2s | 1.28× | INT4 weight-only |
| **Ours Q4_K (CPU)** | 42.1s | **3.1s** | **45.4s** | **0.99×** | ggml AVX2 |
| **Ours F16 (CPU)** | 49.1s | **4.1s** | 53.5s | 0.84× | ggml AVX-512 F16 |
| PyTorch F16 (A100) | — | — | ~1-2s | ~25× | GPU baseline |
| **Ours F16 (Metal M1)** | — | — | ~1-3s | ~15-45× | estimated |

**Encoder gap (2.5× slower than ONNX INT8):** DNNL uses AVX-512 INT8 GEMM (`VPMADDUBSW`) with 2× higher throughput/cycle and 2× lower memory bandwidth vs our F16 ggml path. There is no CPU path to close this gap without implementing AVX-512 INT8 GEMM in ggml's `quants.c`. On Metal/GPU it becomes irrelevant.

**Decoder advantage (3-4× faster than ONNX):** ONNX passes full KV cache arrays (~268 MB) across the Python→ONNX→Python boundary at every decode step. For 167 tokens that is ~45 GB of unnecessary data movement. Our ggml in-place KV cache with tensor views moves zero bytes. This advantage grows with output length.

---

## Per-Op Profiler

Enable with `COHERE_PROF=1`. Results for a 44s clip (F16, 4-thread CPU):

| Op | Time | % | Count |
|----|------|---|-------|
| `mul_mat` | 42.2s | **87.6%** | 743 |
| `im2col` (conv subsampling) | 3.4s | **7.0%** | 53 |
| `add` / `mul` / `scale` | 1.1s | 2.3% | 1559 |
| `cont` | 0.6s | 1.3% | 404 |
| `norm` | 0.2s | 0.5% | 240 |
| `soft_max` | 0.2s | 0.4% | 48 |
| other | 0.4s | 0.9% | 1413 |

`mul_mat` at 87.6% is near hardware peak for F16 GEMM. On Metal/GPU both `mul_mat` and `im2col` would be 10–50× faster.

---

## Optimization History

Starting from a scalar Python-style C++ implementation on the same 11s clip.

| Step | Wall time | Speedup vs prev | Cumulative | Note |
|------|-----------|----------------|------------|------|
| Baseline (scalar loops) | ~825s | — | 1× | triple-nested C++ loops |
| + FFTW3f STFT | ~100s | 8.2× | 8× | O(N log N) FFT; now replaced by self-contained Cooley-Tukey |
| + OpenBLAS GEMM | 104s | ~1× | ~8× | cblas_sgemm in encoder/decoder; later superseded |
| + lazy F32 weight cache | 100s | 1.04× | 8.3× | avoid repeated F16→F32 conversion per inference |
| + EncScratch + AVX2 F16C | 32s | 3.1× | ~26× | pre-allocated scratch, on-the-fly F16 conversion |
| + ggml compute graph | 12.4s | 2.6× | ~67× | replaced all cblas_sgemm; F16 ggml_mul_mat, GPU-ready |
| + BatchNorm folding | 11.5s | 1.08× | ~72× | 480 graph nodes removed (48 layers × 10 nodes) |
| + depthwise conv direct | 9.6s | 1.20× | ~86× | `ggml_conv_2d_dw_direct` replaces im2col for kernel=9 |
| + self-contained FFT | ~8.7s | 1.10× | **~95×** | drop fftw3 dep; iterative Cooley-Tukey, cross-platform |

**Total: ~825s → ~8.7s ≈ 95× on 4-thread CPU for an 11s clip.**

---

## Key Optimizations

### Self-Contained FFT (no external deps)
A 40-line iterative Cooley-Tukey `cohere_fft_r2c` replaces fftw3. Same O(N log N) complexity; eliminates the `libfftw3f-dev` build dependency, enabling clean builds on macOS without homebrew FFT packages.

### BatchNorm Folding
At model load time, each Conformer layer's BatchNorm is folded into the preceding depthwise conv weights:
```
scale[c] = bn_scale[c] / sqrt(bn_var[c] + eps)
w_folded[ki, c] = w[ki, c] * scale[c]
b_folded[c]     = (dw_b[c] − bn_mean[c]) * scale[c] + bn_bias[c]
```
Removes 480 runtime graph nodes (10 nodes × 48 layers). ~7% encoder speedup at F16.

### Depthwise Conv Direct (`ggml_conv_2d_dw_direct`)
Uses `GGML_OP_CONV_2D_DW` — a direct sliding-window kernel with no im2col intermediate buffer. 9.6× faster than the im2col path for kernel size 9 (no 3.4s im2col overhead from the profiler table above).

### F16 Cross-KV Caching
Cross-attention K/V tensors are computed once per utterance inside the encoder GGML graph, stored in F16, and reused across all decoder autoregressive steps. Halves memory vs F32 (e.g., 4.3 MiB vs 8.6 MiB for an 11s clip). Same pattern as `kv_cross` in whisper.cpp.

### Chunked Encoder for Long Audio
Audio longer than 30s is split into overlapping 30s windows. Each chunk's cross-KV tensors are extracted and scatter-copied into a contiguous `[head_dim, T_total, n_heads]` buffer, capping O(T²) attention cost at O(T_chunk²).

### ggml Compute Graph
The entire encoder (48-layer Conformer) and decoder (8-layer Transformer) run as a single ggml compute graph. This provides:
- AVX2/AVX-512/NEON kernel dispatch for `ggml_mul_mat`
- Zero-copy F16 weight storage
- Automatic GPU dispatch (Metal, CUDA) via `ggml_backend_sched`
- Quantization support (Q4_K, Q5_0, Q5_1, Q6_K, Q8_0)
