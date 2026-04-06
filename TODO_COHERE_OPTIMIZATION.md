# Cohere Transcribe Optimization Plan

This document tracks the progress of porting `cohere-whisper.cpp` to a full `ggml` compute graph (Priority 2B).

## Progress Tracker

- [x] **Phase 0: Infrastructure Refactoring**
    - [x] Update `cohere_model` to manage `ggml_backend_buffer` and `ggml_context` for weights properly.
    - [x] Update `cohere_context` to include `ggml_backend`, `ggml_backend_sched`, and memory allocators.
    - [x] Implement robust GGUF tensor loading into the backend buffers.
- [x] **Phase 1: Decoder Graph Port**
    - [x] Implement `cohere_build_graph_decoder`.
    - [x] Support KV caching within the graph.
    - [x] Resolve `GGML_ASSERT(ggml_can_repeat(b, a))` broadcasting issues.
    - [x] Fix `ggml_mul_mat` dimension mismatches and transpose assertions.
    - [x] Verify numerical consistency and successful iterative decoding.
- [x] **Phase 2: Encoder Graph Port**
    - [x] Implement `cohere_build_graph_encoder` (48 Conformer layers).
    - [x] Port Conv2D subsampling to `ggml_conv_2d`.
    - [x] Implement Conformer relative-position attention (relative shift) in `ggml`.
    - [x] Implement Conformer convolution module.
- [x] **Phase 3: Integration & Cleanup**
    - [x] Remove `cblas_sgemm` and `ct_linear` dependencies (Ported to graph).
    - [x] Remove manual F32 weight caching.
    - [x] Clean up redundant feature extraction and conversion logic.
- [x] **Phase 4: Advanced Features**
    - [x] Quantization tool (Q8_0, Q4_K, etc.).
    - [ ] GPU Backend support (CUDA/Metal).
    - [ ] Flash Attention for decoder.
- [x] **Phase 3B: Post-graph micro-optimizations**
    - [x] BatchNorm folding: fold BN stats into conv_dw weights at load time (480 nodes removed, ~7% F16 enc speedup, Q4_K → RTF 1.15×).
    - [x] mmap weight loading: replace fread+vector into mmap to eliminate heap churn (done; load time = disk I/O, not software overhead).
    - [x] Per-op profiler: `COHERE_PROF=1` — eval_callback shows mul_mat=87.6%, im2col=7.0% for 44s audio.
    - [x] Metal/GPU backend: `ggml_backend_load_all()` + `ggml_backend_init_best()` + CPU fallback in sched; `COHERE_DEVICE=metal|cuda|cpu`; CMake: `GGML_METAL=ON` / `GGML_CUDA=ON`.
    - [x] Chunked encoder: process long audio in 30s windows to cap O(T²) attention cost.
    - [x] Fixed-shape KV views + mask: decoder uses max_ctx-sized K/V views with additive attention mask so gallocr plan is reused across autoregressive steps (dec sched alloc: 0.65 → 0.24 ms/step, 63% reduction).

## Current Status
- Decoder: **Graph implementation functional and verified**.
- Encoder: **Graph implementation functional and verified**. 
- Full Pipeline: **Verified correct output on sample audio.**
- BatchNorm folding: **Done and verified** — 4940→4460 nodes, F16 RTF 0.96×, Q4_K RTF 1.15× (real-time).
- Chunked encoder (30s windows): **Done and verified** — 89s audio Q4_K 4-thread: RTF 1.07× vs 1.26× full-audio (16% speedup). Threads=1: 0.35×, threads=2: 0.66×, threads=4: 1.07×.
- Fixed-shape decoder KV views: **Done and verified** — dec sched alloc 0.65 → 0.24 ms/step.

## Phase 5: Technical Learnings & Pitfalls (CRITICAL)

### 1. GGML Tensor Layouts & Operations
- **Conv1D Depthwise:** `ggml_conv_1d_dw` expects input layout `[T, C]` (ne0=T, ne1=C). If your hidden state is `[C, T]`, you **must** `ggml_transpose` before and `ggml_permute(1, 2, 0, 3)` after to restore layout.
- **Broadcast Bias:** To add a bias `[C]` to a tensor `[C, T]`, use `ggml_add` with a reshaped bias `[C, 1, 1, 1]`.
- **Softmax Scaling:** Conformer self-attention **requires** scaling by `1/sqrt(head_dim)` BEFORE softmax. Skipping this leads to saturated attention and broken output.

### 2. Audio Preprocessing Consistency
- **Mel Layout:** The subsampling Conv2D layers expect mel features in **time-major** layout `[T, n_mels]` (stored as `[1, n_mels, T]` for 2D conv). Storing them in mel-major layout `[n_mels, T]` will result in garbage encoder output.
- **Normalization:** Ensure per-feature normalization matches the reference (PyTorch/ONNX) exactly, including the biased/unbiased variance calculation.

### 4. Chunked Encoder Cross-KV Assembly
- **Per-chunk extraction, not standalone graph:** A standalone GGML cross-KV graph (taking concatenated enc_out as input) CANNOT be run through the same `ggml_backend_sched` that was used for the encoder graph. The `gallocr` buffer reset reuses the same virtual buffer; its allocated addresses overlap with input tensors, causing enc_in to be overwritten during computation.
- **Fix:** Extract ck/cv directly from each chunk's encoder graph (same proven path as single-chunk), copy to CPU vectors, then scatter-copy into the final `[head_dim, T_total, n_heads]` (K) and `[T_total, head_dim, n_heads]` (V) layouts before uploading to the backend buffer.
- **K scatter:** For head h, chunk c contributes a contiguous block at `k_full[h × T_total × hd + T_so_far × hd]`.
- **V scatter:** For head h and dim d, chunk c contributes a contiguous slice at `v_full[h × hd × T_total + d × T_total + T_so_far]`.

### 3. GGML Infrastructure
- **GGUF Memory Bug:** `gguf_init_from_file` with `no_alloc=true` has a known edge case where it might not calculate enough overhead for the `ggml_context` if many tensors are present. Adding a safety margin (`+10` descriptors) to the allocation in `gguf.cpp` fixes this.
- **KV Cache Reset:** Use `ggml_backend_buffer_clear` instead of `memset` when the buffer is allocated on a non-CPU backend.
- **Constants:** Tensors created in a `no_alloc=false` context (like persistent constants) use `ctx->data` directly; those on backends require `ggml_backend_tensor_set`. Mixed usage must be handled carefully.
