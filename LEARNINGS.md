# CrispASR — Technical learnings

Distilled from months of porting eight ASR architectures into one ggml
codebase. Nothing here is breaking news; everything here is something
we'd have saved days if we'd known up front.

If a lesson is still "live" (affects current work), it's linked from
`TODO.md`. If it's historical (a bug we already fixed), it's linked from
`HISTORY.md`.

---

## ggml / inference engine

### RoPE mode mapping: ALWAYS `NEOX` for modern models

The single most expensive bug in this project was shipping Granite with
`GGML_ROPE_TYPE_NORMAL` (mode=0) when HF models use `rotate_half`-style
RoPE. The two modes pair different dimension indices:

- `GGML_ROPE_TYPE_NEOX` (mode=2) pairs `(i, i+d/2)` — matches HF
  `rotate_half`. **This is what Llama, Mistral, Qwen, Granite, Gemma,
  GPT-NeoX, and basically every modern LLM uses.**
- `GGML_ROPE_TYPE_NORMAL` (mode=0) pairs adjacent dims `(0,1), (2,3)…`
  Very few models use this. If you can't find a citation for it in the
  model's reference code, you probably don't want it.

Signature of the bug: the model loads, runs, and generates fluent-looking
text — but it's garbage. Byte-level detail preservation at the layer
boundaries hides it for the first few layers; by layer 40 the hidden
state is in the wrong basis and the LM head picks nonsense tokens. The
giveaway is that the Python reference transcript is perfect and the
ggml transcript is fluent but wrong. Always diff against the reference
at each layer boundary.

### Flash attention tensor layout

`ggml_flash_attn_ext(Q, K, V, mask, scale, max_bias, logit_softcap)`
expects Q, K, V in `[head_dim, T, n_heads]` layout with their final
dimension stride 1. If you've computed Q/K/V as `[d_model, T]` from a
`ggml_mul_mat`, you need three steps to get there:

1. `ggml_reshape_3d(_, hd, n_heads, T)` — expose the head dim
2. `ggml_permute(_, 0, 2, 1, 3)` — swap `n_heads` and `T`
3. `ggml_cont(_, …)` — flash-attn requires contiguous memory

Skipping the `ggml_cont` causes a silent shape error downstream. The
output comes back as `[head_dim, n_heads, T, 1]` and you need a
`ggml_reshape_2d(_, hd * n_heads, T)` to collapse it back into `[d, T]`
for the output projection.

### GQA native support vs explicit expansion

`ggml_flash_attn_ext` natively handles GQA when `n_kv_heads < n_heads`
and the K/V tensors have the right shape — it broadcasts each KV head
across `n_heads / n_kv_heads` query heads internally. BUT the K/V
tensors must be laid out as `[head_dim, T, n_kv_heads]`, not
`[head_dim, T, n_heads]`.

If you manually expand KV via `ggml_repeat_4d` before calling flash-attn,
you get a more memory-hungry but more forgiving path that works with
either layout. All three of voxtral, voxtral4b, qwen3, and granite LLM
blocks do the explicit expand for simplicity.

### `ggml_backend_sched` lifetime

Two common patterns, with very different performance:

- **Create once, reset between calls.** Create the scheduler at model
  init with the worst-case graph size (whichever of your stages is
  largest — usually the LLM prefill), and call `ggml_backend_sched_reset`
  between compute calls. Near-zero per-call overhead.
- **Recreate every call.** This is what qwen3/voxtral currently do
  because their graph sizes differ between stages (conv, encoder, LLM
  prefill, LLM decode step). Cheap in absolute terms but adds ~5-15 ms
  per call, which matters for the single-token decode loop.

Fix: compute the max graph node count once at init by building the
largest graph variant and measuring its node count, then create a
single scheduler with that budget and `reset` between stages. See
`TODO.md` under "Per-model follow-ups → qwen3 / voxtral".

### Flash attention on prefill AND decode

The LLM-based backends all use `ggml_flash_attn_ext` for prefill. Using
it for the single-token decode step too (not just prefill) halves the
decode-time graph size and runs ~2× faster on CPU. Qwen3 and voxtral
already do this. Check any new backend's per-token wall time to
confirm it's taking this path.

### In-place recursive FFTs are const-unsafe

voxtral / voxtral4b / qwen3 ship a recursive radix-2 Cooley-Tukey FFT
that treats its input buffer as 4× scratch space during recursion.
These can't be called through a `const float *` function pointer —
they modify memory past their nominal input length. When integrating
with `core_mel::FftR2C` (which has a const-input contract), wrap the
FFT with a thread-local scratch copy:

```cpp
static void model_fft_wrapper(const float * in, int N, float * out) {
    static thread_local std::vector<float> scratch_in;
    static thread_local std::vector<float> scratch_out;
    if ((int)scratch_in.size()  < 4 * N) scratch_in.assign((size_t)4 * N, 0);
    if ((int)scratch_out.size() < 8 * N) scratch_out.assign((size_t)8 * N, 0);
    std::memcpy(scratch_in.data(), in, (size_t)N * sizeof(float));
    model_fft(scratch_in.data(), N, scratch_out.data());
    std::memcpy(out, scratch_out.data(), (size_t)(2 * N) * sizeof(float));
}
```

One allocation per thread, zero per-call heap churn.

---

## Mel spectrograms

### Two algorithm clusters, not one

Nine model files in `src/` had nine different mel implementations.
They fall into exactly two clusters, distinguished by log base and
normalisation scheme. Knowing this upfront would have collapsed the
refactor into one parameterised function.

| Cluster | Log | Normalisation | Output layout | Used by |
|---|---|---|---|---|
| **NeMo** | `ln` | per-mel z-score | `(T, n_mels)` | parakeet, canary, canary_ctc, cohere |
| **HF / Whisper** | `log10` | global clip `(max(x, max(x)-8) + 4) / 4` | `(n_mels, T)` | whisper, qwen3, voxtral, voxtral4b, granite |

Sub-variants you'll hit once per cluster:
- `log_guard_mode`: NeMo uses `log(x + eps)`, HF uses `log(max(x, eps))`.
  Numerically close but not identical.
- `matmul_precision`: NeMo uses `float` accumulator, HF uses `double`.
  This matters for bit-exact regression against PyTorch reference.
- `fb_layout`: NeMo stores the filterbank as `[n_mels, n_freqs]`, HF
  stores it as `[n_freqs, n_mels]`. Transposed.
- `drop_last_frame`: HF drops the last STFT frame; NeMo keeps it.
- `drop_first_frame_if_odd`: voxtral4b needs even T for a stride-2 conv.
- `pad_to_T`: voxtral 3B pads to 3000 frames (= 30s) AFTER log, BEFORE
  normalisation, using `log(eps)` as the pad value so padded frames
  don't skew the global-clip max.
- `stacked_frames`: granite's output is `(160, T/2)` = two 80-mel
  frames zipped along channels. (Still inline — see TODO.md.)

See `src/core/mel.h` for the parameterised version.

### Cohere's cohere_fft_r2c + pre-emphasis

Cohere is the one NeMo-cluster model that doesn't fit the others
cleanly: it applies a `samples[i] = samples[i] - 0.97 * samples[i-1]`
pre-emphasis filter before the STFT. Easy to handle — do the pre-
emphasis in the model wrapper, then call `core_mel::compute` on the
pre-emphasised signal.

Cohere also uses `cblas_sgemm` for the power→mel matmul. When we
migrated to the manual accumulator in `core_mel`, the summation order
changes slightly and one SRT timestamp shifted by 80 ms (one encoder
frame). The transcript text is bit-identical. If bit-exact BLAS
output becomes a hard requirement, a BLAS-backed matmul path can be
added to `core_mel` behind a feature flag.

---

## Quantisation and memory

### Q4_K is the production default

Across every model we've benchmarked, Q4_K has been the sweet spot:

- **parakeet**: F16 9.3s → Q4_K 5.3s (1.75× faster, 0.97× realtime CPU, quality identical)
- **canary**: F16 13.0s → Q4_K 6.5s (2.0× faster, 1.19× realtime CPU)
- **cohere**: F16 27.6s → Q4_K 14.8s (1.87× faster, 2.72× slower than realtime)
- **qwen3-asr**: Q4_K 6.5s on jfk.wav (1.7× realtime)
- **voxtral 3B**: 70s total, 242 ms/token (3B is heavy on CPU)
- **voxtral 4B Realtime**: F16 133s → Q4_K 49s (2.7× faster, 0.22× realtime CPU)
- **granite 1B**: Q4_K 22.5s on jfk.wav (0.49× realtime)

Q5_0, Q6_K, Q8_0 are marginal improvements on smaller models but don't
close the gap to Q4_K in wall-clock tests. F16 is 2-3× slower than
Q4_K on CPU with no measurable quality improvement for ASR.

### Baked mel filterbank, baked Hann window

Every model's GGUF stores the mel filterbank and Hann window as regular
F32 tensors, not as arrays of numbers in the GGUF metadata. The
`core_mel::compute` function reads them via `ggml_backend_tensor_get`
at inference time. Pros: same precision as the Python reference, no
numerical drift from Slaney reconstruction in C++; cons: a couple hundred
KB of extra weight bytes. Worth it.

### F16 KV cache is non-negotiable for LLM backends

Qwen3/voxtral/voxtral4b/granite LLM KV caches are all F16. Cohere's
self-attention KV is still F32 (historical, see TODO.md for the planned
upgrade). Halves GPU memory and bandwidth with no observable quality
loss in ASR workloads.

---

## CPU vs ONNX vs PyTorch baselines

### Where the time goes (Cohere, 11s clip, 8-thread CPU)

Representative profile from the Q4_K path:

| Op | % of time |
|---|---:|
| `mul_mat` | 87.6% |
| `im2col` (conv subsampling) | 7.0% |
| Everything else | 5.4% |

`mul_mat` at 87.6% is near hardware peak for F16 GEMM. Any optimisation
that doesn't move the `mul_mat` number is noise.

### Where ONNX beats ggml on x86 (and doesn't on Metal)

Measured on a 44s clip, x86 4-thread CPU, quantised:

| Implementation | Encoder | Decoder | Total | RTFx | Notes |
|---|---|---|---|---|---|
| ONNX INT8 (CPU) | 19.5s | 11.7s | 31.2s | 1.44× | DNNL AVX-512 INT8 GEMM |
| ONNX INT4 (CPU) | 22.5s | 12.7s | 35.2s | 1.28× | INT4 weight-only |
| **ggml Q4_K (CPU)** | 42.1s | **3.1s** | 45.4s | 0.99× | ggml AVX2 |
| ggml F16 (CPU) | 49.1s | 4.1s | 53.5s | 0.84× | ggml AVX-512 F16 |
| PyTorch F16 (A100 GPU) | — | — | ~1-2s | ~25× | baseline |

Two observations:

1. **ONNX is ~2× faster in the encoder** on x86 CPUs with AVX-VNNI, because
   DNNL uses `vpdpbusd` for INT8 GEMM and ggml's `vec_dot_q8_0_q8_0`
   still uses `pmaddubsw`/`pmaddwd`. There is no CPU path to close this
   gap without implementing AVX-512 INT8 GEMM in ggml's `quants.c`.
   Tracked in `UPSTREAM.md`.

2. **ggml is 3-4× faster in the decoder.** ONNX passes the full KV cache
   (~268 MB) across the Python→ONNX→Python boundary on every decode
   step. For 167 tokens that's ~45 GB of unnecessary data movement.
   Our ggml in-place KV cache with tensor views moves zero bytes. This
   advantage grows with output length.

On Metal or CUDA, the encoder gap closes entirely: our ggml graphs
already use ops that have GPU kernels (`ggml_mul_mat`,
`ggml_conv_2d_dw_direct`, `ggml_flash_attn_ext`). An M1 Metal run of
the same Cohere clip hits ~11.9× realtime compared to 1.24× Q4_K CPU.

### Python and Rust libtorch are both ~25-30× realtime

Both `transformers` and `cohere_transcribe_rs` (tch crate) go through
libtorch CPU F32 and land at ~160s for a 5.4s clip. There is no easy
win on the Rust side without switching backends.

---

## Audio format lessons

### miniaudio + stb_vorbis handle the common cases

Out of the box, every ASR runtime in this repo accepts WAV / FLAC / MP3
/ OGG Vorbis at any bit depth, any sample rate (auto-resampled to
16 kHz), mono or stereo (auto-mixed to mono). No external dependencies.
The two embedded single-header decoders (`miniaudio`, `stb_vorbis`) are
enough for 95% of real-world ASR pipelines.

### `WHISPER_FFMPEG=ON` only helps bare Opus

Upstream whisper.cpp's `examples/ffmpeg-transcode.cpp` has known bugs
on mp4-family containers: `.m4a` crashes with `munmap_chunk(): invalid
pointer` on the first audio chunk read, and `.webm` (Opus-in-WebM)
hangs indefinitely after the libavformat headers are parsed. Both use
the same `av_read_frame` + `avcodec_send_packet` loop.

Bare-codec `.opus` files work cleanly in the FFmpeg build. So the
practical advice is: enable `WHISPER_FFMPEG=ON` only if you need
in-process `.opus` decoding. For everything else, pre-convert:

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le -y /tmp/audio.wav
```

This is the universally safe path and identical to what the in-process
path would produce if it worked. Documented in `UPSTREAM.md` with a
minimal reproducer.

---

## Language handling

### Auto-detect can silently code-switch

Parakeet's auto-language-ID works well for clean speech but drifts into
English on German clips with technical vocabulary or proper nouns. A
90-second German clip about "Industrial Forschung" and "Technische
Universität" came back with "Industrial Forschung" and "Tech Technische
University" in the transcript. **This is not a chunking issue — VAD-based
segmentation gives the same code-switching.** The encoder classifies the
clip correctly but the decoder drops into English mid-stream on lexical
hints.

Lessons:
1. For production use on a known language, always prefer a model with
   an explicit language flag. Canary's `-sl de -tl de` is the fix — the
   decoder is forced into German by the task-token prefix and cannot
   code-switch.
2. Auto-detect models are better for mixed-language pipelines where the
   language isn't known.
3. Test with vocabulary-heavy, non-English clips before shipping. Clean
   short phrases pass every test you give them.

### Canary's prompt prefix is the mechanism, not magic

Canary's "explicit language" feature is implemented as a task-token
prefix in the decoder prompt, before the audio encoder output. Specifically:

```
<|startofcontext|>[source_lang][target_lang]<|transcribe|>[punctuation]
```

When `source_lang != target_lang`, the task token is `<|translate|>`
instead of `<|transcribe|>`. This is how canary does speech translation
(DE→EN, EN→FR, etc.) in the same model.

---

## Model architecture comparisons

### Voxtral: CrispASR standalone vs llama.cpp mtmd vs max-lt wrapper

Three independent C++ implementations of Voxtral-Mini-3B exist. We
compared them head-to-head and the conclusion was important enough to
preserve.

| | **CrispASR** | max-lt/voxtral-cpp | llama.cpp mtmd |
|---|---|---|---|
| Model files | 1 GGUF | 2 (model + mmproj) | 2 (model + mmproj) |
| Tokenizer | Embedded Tekken blob | llama.cpp native | llama.cpp native |
| LLM forward | Hand-written ggml | llama.cpp core | llama.cpp core |
| [BEGIN_AUDIO] bug | ✔ not affected | needs patch | needs manual fix |
| 30s truncation | ✔ not affected | affected | affected |
| Diff-tested vs PyTorch | ✔ every stage | ✗ | ✗ |
| Lines of model code | ~1300 | ~100 wrapper | 0 (all in llama.cpp) |
| GPU support | ✗ (CPU-only now) | ✔ via llama.cpp | ✔ via llama.cpp |

The llama.cpp `mtmd` multimodal subsystem has two known bugs affecting
Voxtral specifically ([#17868](https://github.com/ggml-org/llama.cpp/issues/17868),
[#18419](https://github.com/ggml-org/llama.cpp/issues/18419)) that
were ignored by maintainers, and a community member reports worse
accuracy in llama.cpp than in transformers/vLLM at the same precision.
Ollama dropped llama.cpp specifically for multimodal due to
instability.

**Recommendation:** keep CrispASR as its own standalone ggml runtime
for ASR. It is diff-tested against PyTorch at every architectural
boundary (LLM cosine sim 0.999973, top-5 5/5 match on identical inputs),
which is the confidence our users need. Do NOT rewrite it on top of
mtmd. When we want GPU, use ggml's Metal/CUDA backends directly on our
existing graph builders — `ggml_flash_attn_ext` already has GPU kernels.
The main work is wiring up `ggml_backend_metal_init()` /
`ggml_backend_cuda_init()` as alternatives to the CPU backend (~50 LOC).

---

## Regression testing discipline

Every migration commit in `src/core/` includes a `md5sum`-level
regression test against `samples/jfk.wav`. The discipline:

1. Run the current binary, capture output + auxiliary outputs (SRT/VTT/JSON)
2. Make the change
3. Rebuild
4. Re-run, compare with `md5sum` and `diff`
5. If bit-identical, commit. If not, investigate.

Two cases where bit-identity is not achievable:

1. **Cohere mel migration.** CBLAS sgemm → manual accumulator changes
   the float summation order, shifting one SRT boundary by 80 ms (one
   encoder frame). Transcript text is byte-identical. Accepted.
2. **Whisper code path.** Untouched by `src/core/` refactors; bit-
   identical against upstream `whisper-cli` is the gate.

The few FFNs / attention blocks where ggml graph op ordering matters
have all come back bit-identical so far. Flash attention results
depend on the order Q/K/V were committed to the graph, but as long as
the helper emits them in the same order the inline code did, you get
bit-identical output.

---

## Specific bugs that cost us a day each

These are each preserved in `HISTORY.md` with full context. Summary form:

1. **Granite RoPE mode (NEOX vs NORMAL).** Model loaded, ran, produced
   fluent nonsense. Fix: one enum value.
2. **Voxtral 4B realtime audio padding.** `32*1280 + 17*1280 + 1280*(right_align)`
   left and right pads are non-negotiable. Skipping the right pad
   silently breaks the encoder graph reshape.
3. **Voxtral 4B Realtime audio_length_per_tok=8.** 3B uses 4 (one audio
   frame per 4 Whisper frames); 4B uses 8. Wrong value → audio-to-token
   alignment off by 2× and transcript drifts.
4. **Cohere F32 self-attention KV.** Still not fixed; costs 2× GPU
   memory. Tracked in TODO.
5. **Qwen3 windowed attention.** Chunked self-attention via `cu_seqlens`
   with window size ~104 positions. Standard full self-attention
   produces wrong output. This is the trickiest part of the qwen3 port.
6. **Hann window centering in Granite mel.** The window must be
   symmetrically zero-padded to n_fft; off-by-one on the centering shifts
   the power spectrum peak and breaks downstream everything.
7. **Q-Former layer norm target.** BLIP-2 projector LN applies to the
   query tokens, not the encoder output. Wrong tensor → garbage projector
   output → garbage LLM input → garbage transcript.
8. **Silero LID: five compounding bugs.** The native port of Silero's
   95-language classifier went through Swedish → Mongolian → Bashkir →
   Khmer → Chinese → Punjabi → English on jfk.wav, each fix changing
   the top prediction. Root causes, in order of severity:
   (a) **Front-end padding.** ONNX uses constant zero-pad 160/side on
       audio; we used reflection-pad 320 on the left. The padding type
       and amount are buried in a Pad node with a dynamically-computed
       pad vector from a chain of 15 ONNX ops.
   (b) **Stride-2 output size.** Conv1d(T, k=1, s=2) output is
       `(T-1)/2+1`, not `T/2`. Off-by-one cascades through 4 stride-2
       stages (1101→551→276→138→69) — wrong value drops 1 frame per
       stage, silently shifting the feature alignment.
   (c) **QKV split order.** ONNX slices QKV as K[0:D], Q[D:2D],
       V[2D:3D]. We assumed Q,K,V order. The only way to discover this
       is to dump the Slice node inputs and compare the split boundaries.
   (d) **Missing ReLU after stride-1 projections.** Stages 4-7 use
       stride-1 Conv1x1→ReLU for dim change (128→192). The ReLU is
       easy to miss since the stride-2 stages already had it.
   (e) **Missing tanh in attention pooling.** ONNX does dot→Tanh→
       Softmax; we did dot→Softmax. The Tanh compresses the score
       range, which completely changes the attention distribution.
   **Lesson:** When porting an unfamiliar ONNX model, dump intermediates
   at every graph boundary and diff against the native code BEFORE
   debugging individual ops. The bug is almost never where you expect.

---

## Quantization

### Small models with conv-heavy architectures resist quantization

The Silero LID model (16 MB F32, 507 tensors) was tested with Q8_0 and
Q5_0 quantization. Both broke accuracy completely (French/Shona instead
of English). The model's parameters are mostly small Conv1d kernels
(dw_conv [5,1,C], pw_conv [1,C,C]) where C ∈ {128, 161, 192}. These
tensors have very few elements per row (1-5), making block quantization
destructive. Only the transformer QKV/out/FFN projections and classifiers
(34 of 507 tensors) have enough elements per row to quantize safely, but
that saves only 3-5 MB — not worth the accuracy loss.

**Rule of thumb:** If a model's parameter count is dominated by Conv1d
kernels with small spatial dimensions (k ≤ 5) and few channels (C < 256),
ship it F32. The 16 MB F32 Silero LID model is smaller than a single
layer of most ASR encoders — quantization is pointless.

---

## Methodical debugging of ported models against ground truth

This is the single most important workflow in the project. Every model
port that "almost works" but produces wrong output will eat days unless
you follow this process systematically.

### The protocol

1. **Get a reference implementation that provably works.** Either the
   original Python/ONNX model (preferred — run via onnxruntime), or a
   known-good C++ implementation. If ONNX: add all internal nodes as
   graph outputs and run with intermediate capture.

2. **Dump intermediates at every graph boundary.** Not just input/output
   — dump after EVERY stage: normalization, projection, attention,
   FFN, residual add. Save as `.npy` files with clear names.

3. **Compare C++ vs reference at each stage, starting from the INPUT.**
   Don't start debugging the attention if the input is already wrong.
   Print first 8-16 values of each tensor at frame t=0. The divergence
   point tells you exactly which operation is broken.

4. **When you find the divergence point, check these in order:**
   - **Tensor layout/transpose** — ggml uses ne[0]-fastest (column-major).
     A `[T, H]` row-major C array becomes `[H, T]` in ggml (ne[0]=H).
   - **Weight shapes** — GGUF stores shapes in ggml ne-order. A numpy
     `(1024, 4096)` weight becomes ggml ne `[1024, 4096]`. For
     `ggml_mul_mat(W, x)` = W^T @ x, we need `W.ne[0] == x.ne[0]`.
   - **Padding type and amount** — zero vs reflect vs replicate. ONNX
     Pad nodes encode padding as a dynamically-computed vector from
     chains of 10+ ops. Always dump the actual padded tensor.
   - **Activation functions** — missing ReLU, tanh, GELU. These are
     easy to miss when tracing the ONNX graph manually.
   - **Operation order** — pre-norm vs post-norm, attention before or
     after stride-2, QKV split order.
   - **Formula details** — stride-2 output is `(T-1)/2+1` not `T/2`.
     Reflection padding `pad[i] = data[pad_size - i]` not `data[i]`.
     Scale factor in attention: 1/sqrt(head_dim) not 1/sqrt(d_model).

5. **For ggml graph debugging specifically:**
   - The `ggml_backend_sched` may not correctly associate model weight
     tensors with their backend buffer. Test with `ggml_backend_alloc`
     instead of the scheduler for isolation.
   - Mark tensors with `ggml_set_name()` and read them with
     `ggml_backend_tensor_get()` BEFORE calling `ggml_backend_sched_free()`.
   - F16 weight tensors in ggml_mul_mat work correctly in single mini-
     graphs (as in `ggml_linear_f32()`) but may misbehave in large
     graphs where the scheduler manages buffer allocation.
   - When in doubt, build a 1-layer graph first and verify it matches
     the manual path before scaling to all layers.

6. **Never trust "close enough".** If the first frame's values differ
   by more than 1e-4 from the reference, there's a bug. Float32
   accumulation order can cause ~1e-5 drift per operation, so after
   24 transformer layers you might see ~1e-3 drift — but a 0.1
   difference at layer 0 means a structural bug.

### Common traps

- **ONNX QKV split order is not always Q,K,V.** Silero LID uses K,Q,V.
  The only way to know is to dump the Slice node boundaries.
- **ONNX padding is computed dynamically.** Don't assume reflect/zero
  from the model architecture — dump the Pad node's padding vector.
- **ONNX Reshape+Transpose chains for multi-head attention** can
  interleave heads differently than simple offset slicing. Always
  dump the post-reshape tensors to verify head layout.
- **ggml_norm normalizes over ne[0].** Make sure ne[0] is the feature
  dimension, not the time dimension.

---

## ggml graph allocation: gallocr vs compute_with_ctx

### gallocr/sched corrupt external weight tensors

When a ggml graph references tensors from an external context (e.g. model
weights loaded via `core_gguf::load_weights`), `ggml_gallocr_alloc_graph`
and `ggml_backend_sched` reallocate buffers for these tensors, overwriting
their data with uninitialized memory. This was confirmed by a minimal
single-op test:

- `ggml_graph_compute_with_ctx` (no allocator): **correct** — directly
  accesses `tensor->data` pointers, which point to the loaded GGUF data.
- `ggml_gallocr_alloc_graph` + `ggml_backend_graph_compute`: **wrong** —
  the allocator sees the external tensors as "unallocated" despite having
  valid `->data` and `->buffer` pointers, and allocates new buffers over
  them.

The `ggml_gallocr_is_allocated()` function at ggml-alloc.c:591 checks
`t->data != NULL || t->buffer != NULL`, which should catch external
tensors. But the two-phase reserve+alloc flow apparently doesn't preserve
this across the reserve step.

**Workaround:** Use `ggml_graph_compute_with_ctx` with `no_alloc=false`
for the graph context. All intermediate tensors get memory from the
context pool, and external weight tensors are referenced via their
existing `->data` pointers. Downside: no memory reuse between layers —
each intermediate stays alive for the entire graph (~80 MB/layer for
wav2vec2-large with 549 frames).

### ggml 2D tensor layout and transpose

ggml stores 2D tensor `[ne[0], ne[1]]` as `data[i0 + i1 * ne[0]]`.
A tensor with `ne[0]=V, ne[1]=T` has element `(v, t)` at `data[v + t*V]`.
This is the SAME memory layout as a C row-major array `float arr[T][V]`
where `arr[t][v] = data[t*V + v]`. So **no transpose needed** when
converting between ggml `[V, T]` and C `[T, V]` row-major — they're
the same bytes. The earlier wav2vec2 code had wrong transposes at THREE
places (input, layer readback, LM head input) that shuffled data into
garbage. The fix was to remove ALL transposes and use `memcpy` /
`std::copy` directly. **When in doubt, don't transpose.**

### Layer-by-layer graph execution as a gallocr workaround

When `ggml_gallocr` corrupts external weight tensors, building one
graph per transformer layer with `ggml_graph_compute_with_ctx` and
`no_alloc=false` is a viable workaround. Each layer graph uses ~80 MB
(for wav2vec2-large with T=549, H=1024, 16 heads) and is freed after
use, so total RSS stays at ~800 MB instead of 3+ GB. The hidden state
is copied in/out of each layer graph via `memcpy`. Per-layer max_diff
vs the manual reference path is < 0.005 (float32 accumulation noise).

This is slower than a single-graph approach (24 context alloc/free
cycles + 24 graph plans) but produces correct results and uses much
less memory. Good enough for CPU; for GPU acceleration, fixing gallocr
to skip pre-allocated tensors is the proper solution.

---

## Performance: what faster-whisper / insanely-fast-whisper do

Analysed SYSTRAN/faster-whisper and Vaibhavs10/insanely-fast-whisper
(April 2026). Key techniques and applicability to ggml:

**Already have in CrispASR:**
- Quantization (Q4_K/Q5_0/Q8_0) — fundamental to ggml
- Flash attention (ggml_flash_attn_ext) — used by whisper backend
- VAD pre-filtering (Silero) — skips silence before transcription
- Multi-file parallelism (n_processors)

**Could add (GPU-dependent, large impact):**
- **Batched encoder** — process N audio chunks simultaneously on GPU.
  Faster-whisper's `BatchedInferencePipeline` with batch_size=8 gives
  3-5x speedup. Requires GPU (batch doesn't help much on CPU since
  we already use all cores per chunk).
- **Speculative decoding** — use a small "draft" model to predict
  tokens, verify with the large model. 2-4x speedup for autoregressive
  LLM backends (granite, voxtral, qwen3). Needs two models loaded.

**Could add (CPU-friendly, moderate impact):**
- **Pipelined mel+encode** — while LLM decodes chunk N, compute mel
  for chunk N+1 in a background thread. ~15-20% speedup for LLM
  backends on multi-core CPUs.
- **Encoder output caching** — for repeated queries on the same audio
  (e.g. trying different languages), cache the encoder output and only
  re-run the decoder. Already implicit in whisper's architecture.

**Not applicable:**
- CTranslate2's CUDA kernels — ggml has its own CUDA backend
- BetterTransformer API — PyTorch-specific
- fp16 compute — ggml already does F16 matmul natively

**Bottom line:** On CPU, we're already within 2x of the theoretical
limit (2.2x realtime for parakeet on jfk.wav). The big wins are
GPU-specific: batched encoder (5x) and speculative decoding (2-4x).

**Implemented optimizations (April 2026):**
- Parallel VAD slice transcription (thread pool with separate backend
  instances — helps on GPU where each instance uses a separate stream)
- Full-graph ggml_backend_sched path for wav2vec2 with explicit weight
  tensor assignment via `ggml_backend_sched_set_tensor_backend` — GPU-
  ready single-graph dispatch for all 24 transformer layers
- Buffer reuse across layers (saves 24×80MB alloc/free cycles)
- Server-mode audio cache (instant response on repeated queries)
- Realtime speed reporting per file
- All model weights loaded to GPU when ggml_backend_init_best() picks
  a GPU backend (already built into core_gguf::load_weights)

**Key discovery:** `ggml_backend_sched_set_tensor_backend()` prevents
the scheduler from reallocating external weight tensors. This was the
missing piece for making the full-graph path work with model weights
on a separate buffer. Without it, gallocr corrupts external tensors.

### Windows fseek overflow: the silent >2 GB file killer

On Windows (MSVC), `long` is 32-bit even on x86_64. `fseek(fp, (long)offset, SEEK_SET)`
silently wraps around at 2^31 = 2.1 GB. For GGUF files larger than
this (voxtral4b Q4_K = 2.35 GB, Q8_0 = 4.4 GB), tensors stored past
the 2 GB boundary get read from the wrong file offset, resulting in
"missing tensor" errors or corrupt data.

The fix: `_fseeki64()` on Windows, `fseeko()` on POSIX. Also add
native Windows mmap (`CreateFileMapping` + `MapViewOfFile`) to bypass
the fseek path entirely.

**Lesson:** `fseek(fp, (long)x, ...)` is a bug on Windows for any file
that might exceed 2 GB. Always use platform-specific 64-bit seek. This
is a classic portability trap that doesn't manifest on Linux/macOS
(where `long` is 64-bit on LP64).

---

## VAD integration and long audio (April 2026)

### whisper VAD returns centiseconds, not seconds

The `whisper_vad_segments_get_segment_t0/t1()` functions return
timestamps in **centiseconds** (e.g. 29.0 = 0.29 seconds), not
seconds. Our initial integration multiplied by `sample_rate` directly,
producing sample indices 100× too large. Every segment fell past the
end of the audio, causing "no speech detected" for every file.

**Lesson:** Always check the units of external API return values. The
whisper.cpp VAD API stores `start`/`end` via `samples_to_cs()` (line
5676) and the internal code divides by 100.0 for display (line 6914).
The getter functions return the raw centisecond values.

### Short VAD segments break ASR quality

Silero VAD can produce very short segments (0.35s) for speech with
brief pauses. These are too short for most ASR encoders to produce
reliable output. On jfk.wav (11s), the VAD split into 5 segments of
0.35-2.4s each, causing parakeet to produce garbled output.

**Fix:** Post-merge adjacent VAD segments: combine if gap < 1s or if
the accumulated segment is shorter than 3s. This produces 2 merged
segments instead of 5 tiny ones, with correct transcription.

### VAD stitching matches whisper.cpp quality

whisper.cpp stitches VAD segments into one contiguous buffer with 0.1s
silence gaps, builds a mapping table, processes as one audio stream,
then remaps timestamps. This is fundamentally better than independent
per-slice processing because the decoder sees continuous audio context.

We now do the same for non-whisper backends: stitch → single
`transcribe()` call → remap. Tested on 89s and 227s audio — no
boundary artifacts, correct timestamps throughout.

### Backend-specific audio length limits

| Backend | Mel length | Hard limit? | Notes |
|---|---|---|---|
| whisper | 3000 frames (30s) | Yes | Pads to exactly 3000 frames |
| voxtral 3B | 3000 frames (30s) | Yes | `T_mel = 3000` hardcoded |
| voxtral4b | variable | No | Causal encoder, streams |
| qwen3 | variable | No | Chunked conv subsampler |
| parakeet | variable | No | O(T²) attention, ~5min practical limit |
| canary | variable | No | O(T²) attention, ~5min practical limit |
| cohere | variable | No | O(T²) attention, ~5min practical limit |
| granite | variable | No | Block-local attention (ctx=200), any length |

For whisper and voxtral 3B, 30s chunking is mandatory. For the rest,
longer chunks work but hit O(T²) memory walls. VAD stitching helps by
removing silence (shorter effective audio), and the max-chunk split
prevents OOM on very long continuous speech.

### Qwen3 forced aligner leading-silence issue

The qwen3 forced aligner assigns timestamps starting from 0 even when
audio has leading silence. On the user's 227s JavaScript tutorial with
~3s of leading silence, the first word was stamped at 0.24s instead
of ~3.2s. With VAD stitching, the silence is removed before alignment,
fixing the issue.

**Lesson:** The forced aligner only works well when the audio starts
with speech. Always use VAD to trim silence before alignment.

### Qwen3 forced aligner monotonicity

The reference implementation (`qwen3_forced_aligner.py`) has a
`fix_timestamp()` function using longest-increasing-subsequence (LIS)
to correct non-monotonic timestamps. We use a simpler forward clamp
(each timestamp >= previous). This handles most cases but may miss
complex inversions. Parakeet's native TDT timestamps are always
better when available.

### CrispASR vs voxtral.c: 3.8× faster on CPU

Direct same-hardware comparison (Xeon 4-core, no GPU) on jfk.wav:
- voxtral.c (OpenBLAS): 11m 0s (encoder 220s, decoder 2660ms/step)
- CrispASR (ggml): 2m 52s
- Speedup: 3.8×, attributable to ggml's optimised matmul kernels

### Susurrus architecture insights

Susurrus (CrispStrobe's Python ASR tool) uses:
- `vad_filter=True` hardcoded in faster-whisper (always on)
- 25-minute chunks with 2s overlap for voxtral local
- GPU memory explicitly freed between chunks (`torch.cuda.empty_cache`)
- Generator-based segment yielding (streaming/incremental)

**Lesson:** VAD should be the default, not an opt-in. 30s chunks are
too conservative for most models; 5-10 minutes is practical for
variable-length backends on 16GB VRAM.

### wav2vec2-base: post-norm vs pre-norm (the silent architecture trap)

wav2vec2-base models (`do_stable_layer_norm=False`) use **post-norm**
transformer layers: `attention → residual_add → LayerNorm → FFN →
residual_add → LayerNorm`. wav2vec2-large models
(`do_stable_layer_norm=True`) use **pre-norm**: `LayerNorm → attention →
residual_add → LayerNorm → FFN → residual_add`.

Our initial implementation only had pre-norm (matching the large XLSR
model we first ported). Running a base model through pre-norm produces
all-identical outputs at every time position — the encoder loses
positional information and the CTC decoder outputs the same character
(argmax=24 = "b") at every frame.

**Symptoms:** Output is a single character repeated, or empty text.
All positions have the same argmax.

**Root cause debugging protocol:**
1. Get HF reference intermediates (CNN out, feature projection, encoder
   out, logits argmax) — these are ground truth.
2. Add debug fprintf to C++ at each stage boundary.
3. Compare stage by stage — CNN matched, feature projection matched,
   but logits diverged completely.
4. The argmax pattern `[24,24,24,24,...]` (all same) immediately points
   to an encoder bug that collapses positional information.
5. Check `do_stable_layer_norm` in the HF config — it controls the
   norm ordering and is the first thing to verify when porting a new
   wav2vec2 variant.

**Second bug:** CTC blank token. `config.pad_token_id=1` (BOS) in base
models, but CTC greedy decoding must skip vocab index 0 (`<pad>`) which
is the actual CTC blank. The converter now hardcodes blank=0.

**Lesson:** When porting a model architecture, always check for
configuration flags that change the graph topology (norm ordering,
activation type, bias presence). These are silent — the model loads
and runs without errors, but produces garbage. A debug copy of the
forward pass (`wav2vec2-ggml-debug.cpp`) with fprintf at each stage
boundary is kept for future model variant debugging.

---

## CLI ↔ library DRY refactor (April 2026)

v0.4.4–v0.4.8 moved every non-presentation CLI concern into `src/`
behind the shared C-ABI. Below are the lessons from the five-release
cycle — things worth remembering the next time a helper turns out to
be shared across more consumers than its location suggests.

### File names are claims; check them periodically

`examples/cli/crispasr_dart_helpers.cpp` started as Dart-only in
0.2.0 but by 0.4.0 it was the common FFI surface consumed by the CLI,
Dart, Python, and Rust. The file name was a documentation bug for
four releases. The first move (`crispasr_c_api.cpp` + updated header
comment) was pure churn and should have been done earlier. An
occasional pass over file/function names vs actual callers is worth
doing.

### Header basename clashes surface late

`src/crispasr_vad.h` and `examples/cli/crispasr_vad.h` coexisted
without error until the CLI source that `#include "crispasr_vad.h"`
happened to compile against the `src/` version (because the whisper
target is `target_include_directories(... PUBLIC .)`) — producing a
cryptic type mismatch with the CLI's `whisper_params` usage. Renaming
the CLI headers to `*_cli.h` (vad/diarize/lid/model_mgr/aligner) is
the clean fix; guards like `-I` ordering are fragile.

### Function-name collisions are worse than symbol collisions

Both `src/crispasr_lid.cpp` and `examples/cli/crispasr_lid.cpp`
defined `crispasr_detect_language(...)` as non-member C++ functions.
Different argument types → different mangled names → the linker is
happy. But any caller looking at `crispasr_detect_language(samples,
n, params)` has no idea which one it's getting. The safer pattern is
to suffix all CLI-shim symbols (`crispasr_detect_language_cli`,
`crispasr_apply_diarize`, etc.) so the call sites themselves signal
which layer they belong to.

### Backwards-compat aliases for renamed C-ABI symbols

When we renamed `crispasr_dart_helpers_version()` to
`crispasr_c_api_version()`, 0.4.x-era binaries already existed that
probed the old name. The library now exports both — the new function
is canonical, the old one is a 2-line thunk that calls it. A TODO in
source marks the removal after the next major version. The Dart
smoke test asserts **both** resolve and return the same value, so
we can't accidentally drop the alias early.

### POD ABI structs: be explicit about padding

`crispasr_vad_abi_opts` is `float + 5×int32` = 24 bytes. Clean on
64-bit with no padding. `crispasr_diarize_seg_abi` would have been
`int64 + int64 + int32` = 20 bytes with 4 bytes of trailing padding
on 64-bit — so we added an explicit `int32_t _pad` and documented
the 24-byte size so Dart/Python/Rust bindings can allocate the
struct by hand. Always check `sizeof` on both 32- and 64-bit
platforms when promoting a struct to the ABI.

### Policy stays in the CLI; algorithms go to the library

Every CLI shim follows the same pattern:
- **CLI-only (stays in `examples/cli/*_cli.{h,cpp}`)**: auto-download
  from `~/.cache/crispasr`, `isatty()` / TTY prompts,
  `sherpa-onnx` subprocess spawn, CLI-specific types like
  `whisper_params` / `crispasr_segment` / `crispasr_word`.
- **Library (goes to `src/*.{h,cpp}`)**: the actual algorithm —
  Silero VAD + stitching, diarize methods, whisper encode for LID,
  canary-CTC Viterbi, the model registry table, the WinHTTP/curl
  download helper.

This line is obvious in hindsight but we kept crossing it early on.
Rule of thumb: if a wrapper consumer (Python / Rust / Flutter app)
could want the function too, it belongs in the library.

### Rust CStr + static buffers for string-returning C-ABI

For the registry lookup (`crispasr_registry_lookup_abi`) we went
with caller-allocated output buffers (`out_filename`, `out_url`,
`out_size` as `char* + int cap`) rather than returning an opaque
handle with accessors. Reasons:
- Single call rather than 5 round-trips to Python/Dart
- No lifetime management for the wrappers
- Registry strings are small (URL up to ~256 chars), so a fixed
  2 KB stack buffer is fine
- Easy to detect "buffer too small" (return code 2) and retry

For `crispasr_align_words_abi` we did the opposite — one result can
contain hundreds of words, each a variable-length string, so we
kept the `session_result`-style handle + accessors pattern. Choice
of pattern depends on how bounded the output is.

### A Dart smoke test that only checks `lib.lookup()` catches 90% of binding drift

`flutter/crispasr/test/bindings_smoke_test.dart` just resolves every
C-ABI symbol by name. It takes 50 ms to run, needs no audio, and
catches: symbol rename typos, missing `CA_EXPORT`, new backend
dropping a target from `target_link_libraries(whisper PUBLIC ...)`,
and stale `.so`/`.dylib` on the test machine. Ran it after every
release in this cycle; caught one typo that would've shipped.

### Rust FFI: C++ exceptions abort the process

The old `CrispASR` Rust API (wrapping `whisper_full()` directly) crashes
with "Rust cannot catch foreign exceptions" because whisper.cpp's C++
code can throw exceptions (ggml assertion failures, `std::bad_alloc`).
Rust's `extern "C"` FFI boundary treats C++ exceptions as undefined
behavior — they unwind through Rust stack frames and trigger `abort()`.

The `Session` API works because `crispasr_session_transcribe()` is a
C-ABI wrapper implemented in C++ that catches exceptions internally
and returns error codes. The old `whisper_full()` path has no such
wrapper.

**Lesson:** All C-ABI functions exposed to Rust/Dart/Python must wrap
their body in `try { ... } catch (...) { return error_code; }`. The
Session API does this by design. The legacy whisper-direct functions
(`whisper_full`, `whisper_init_from_file_with_params`) do not. We mark
the old Rust `CrispASR` struct as deprecated in favor of `Session`.

### split-on-punct proportional fallback: the silent accuracy killer

When `--split-on-punct` was used without `-ml N`, the display segment
builder checked `seg.words.empty() || max_len == 0` and took the
proportional interpolation path — even when the backend (parakeet)
had produced accurate word-level timestamps. The proportional path
estimates sentence boundaries by character position ratio, which can
be off by 1+ seconds.

**Symptoms:** Sentence start/end times don't match the actual speech.
A sentence ending with "code." at 6.3s shows as ending at 7.3s.

**Root cause:** `max_len == 0` (the default when `-ml` isn't passed)
was treated as "no word packing" even though `split_on_punct` DOES
need word-level timestamps for accurate splitting.

**Fix:** `(max_len == 0 && !split_on_punct)` — only skip word packing
when neither max_len nor split_on_punct is requested.

**Second bug in the same path:** The flush happened AFTER updating
`cur.t1 = w.t1`, so the flushed sentence included the NEXT word's
end time. Moved flush to before the update.

**Lesson:** When two features interact (max_len + split_on_punct),
test all four combinations: (0,false), (0,true), (N,false), (N,true).
The (0,true) case was never tested and silently degraded accuracy.

### GLM-ASR-Nano: partial RoPE is non-negotiable

GLM-ASR-Nano uses `partial_rotary_factor = 0.5`, meaning RoPE is
applied to only the first half of each attention head's dimensions
(32 out of 64). Applying full RoPE (to all 64 dims) produces encoder
outputs that are ~30% off from the reference — close enough to load
and run, but too divergent for correct transcription.

**Implementation:** Split Q/K tensors along head_dim via `ggml_view_3d`,
apply `ggml_rope_ext` to the first-half view, concatenate back with
`ggml_concat`. This can't use `encoder_self_attn()` (which assumes
full RoPE), so the attention is implemented inline.

**Lesson:** Always check `partial_rotary_factor` in the config before
using RoPE helpers. If it's not 1.0, split-apply-concat is required.
The same pattern appears in Gemma, Phi, and other recent architectures.

### GLM-ASR-Nano: stride-2 conv length is floor(T/2), not ceil(T/2)

GLM-ASR's encoder stem uses `ggml_conv_1d` with `k=3, s=2, p=1`, then
immediately reshapes the result to `(T_enc, d)`. I initially used
`T_enc = (T_mel + 1) / 2`, which matches the textbook convolution size
formula for this kernel setup, but not the actual ggml tensor layout in
the unbatched `(T, C)` path used here.

On odd `T_mel`, ggml produced `floor(T_mel / 2)` frames, so the reshape
asked for one frame too many and hit:

`GGML_ASSERT(ggml_nelements(a) == ne0*ne1)`

This showed up immediately on real GLM-ASR inference with odd-length mel
sequences, while even-length samples hid the bug.

**Fix:** Use `T_enc = T_mel / 2` consistently in both the encoder graph
builder and the output-shape calculation in `glm_asr_run_encoder()`.

**Lesson:** For ggml conv outputs, trust the runtime tensor shape or a
known-good in-repo precedent over the paper formula, especially when the
input is using an implicit unbatched layout.

### FFT size must be power of 2 for radix-2

`core_mel::compute()` calls `fft(data, n_fft, output)` where `n_fft`
may not be a power of 2 (whisper uses 400). A radix-2 Cooley-Tukey
FFT requires power-of-2 input — passing 400 corrupts memory via
bit-reversal permutation on a non-power-of-2 array.

**Fix:** Zero-pad to the next power of 2 (400→512) inside the FFT
function, then truncate the output back to N bins.

### KV cache: no_alloc=true is mandatory for scheduler

The `ggml_backend_sched` requires all tensor contexts referenced in
the graph to have `no_alloc=true`. Creating the KV cache context with
`no_alloc=false` + `ggml_backend_alloc_ctx_tensors()` causes an
assertion failure in `ggml_backend_sched_alloc_graph`.

**Fix:** Use `no_alloc=true` context + manual `ggml_backend_alloc_buffer`
+ `ggml_backend_tensor_alloc` (matching voxtral's pattern). Also call
`ggml_backend_sched_set_tensor_backend` for KV tensors before graph
allocation.

---

## Windows / MSVC portability (April 2026)

### `M_PI` is not defined on MSVC by default

`<cmath>` under MSVC does not expose `M_PI` unless `_USE_MATH_DEFINES`
is `#define`d *before* the header is included. POSIX toolchains
(glibc, libc++ on macOS) leak it through by default, so code that
relies on `M_PI` builds cleanly on Linux and macOS and then fails on
Windows with:

```
error C2065: 'M_PI': undeclared identifier
```

This bit `src/glm_asr.cpp` (the Cooley-Tukey FFT butterfly uses
`-2 * M_PI / len`) — the rest of the codebase had already standardised
on `core/mel.h`'s FFT helpers, which avoid `M_PI` internally, so the
issue was invisible until glm-asr landed its own inline FFT.

**Fix pattern, applied at the very top of any TU that uses `M_PI`:**

```cpp
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
```

The redundant `#ifndef` guard covers the case where someone further
down the include graph has already pulled in `<cmath>` before the
define (harmless on POSIX; a no-op on MSVC where the guard fires).

**Lesson:** Every new `src/` TU that touches trigonometry on its own
(rather than going through `core/mel.h`) needs this three-line
preamble. Consider banning direct `M_PI` use in code review — pulling
FFT/trig through `core_mel` is portable for free.

### Vulkan first-run latency: the 13-second "Vulkan is slow" illusion

Initial measurement on a hybrid-GPU laptop (Intel Iris Xe + NVIDIA
RTX A1000) reported Vulkan at **0.5× realtime** on parakeet Q4_K /
jfk.wav vs CUDA at 10× RT — a 20× gap that looked like Vulkan being
hopeless for ASR. Diagnosis was initially directed at device
selection (maybe it's on the Intel iGPU?). **Wrong.**

`ggml_backend_init_best()` already prefers `GGML_BACKEND_DEVICE_TYPE_GPU`
over `_IGPU` — the NVIDIA dGPU was correctly selected. The real cost
was **first-run pipeline compilation**: SPIR-V → native GPU ISA for
~50-100 compute pipelines happens lazily on first dispatch, and
ggml-vulkan passed `VK_NULL_HANDLE` to every
`device->device.createComputePipeline(…)` call, meaning **no
VkPipelineCache was used at all**. Subsequent runs only appeared
"fast" because the NVIDIA driver has its own per-shader disk cache
(`%LOCALAPPDATA%\NVIDIA\GLCache`) that catches the miss one level
down. Wipe that cache and every Vulkan run is 13+ seconds again,
permanently.

**Fix** (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`): added a persistent
`vk::PipelineCache` on `vk_device_struct`, keyed by
`vendor:device:driverVersion`, stored under `$LOCALAPPDATA\ggml\vulkan_pipeline_cache\`
(Windows) / `$XDG_CACHE_HOME/ggml/vulkan_pipeline_cache/` (Linux) /
`~/Library/Caches/ggml/vulkan_pipeline_cache/` (macOS). Loaded at
device init, passed to every `createComputePipeline` call, flushed
to disk every 4 new pipelines (counter on the device struct).

Flushing in the destructor alone is not sufficient on Windows: we
call `_Exit(0)` from the CLI (see "Process exit hang" memory entry)
to sidestep a Vulkan static-destructor stall, which also bypasses
`~vk_device_struct()`. Periodic save inside
`ggml_pipeline_request_descriptor_sets` / pipeline-creation covers
this without any new public API.

**Results** (parakeet Q4_K / jfk.wav, 11 s audio, same laptop):

| Scenario | Transcribe time | RTFx |
|---|---:|---:|
| Vulkan cold (no caches) | 13.69 s | 0.8× |
| Vulkan, only our ggml cache warm (NVIDIA GLCache wiped) | **1.34 s** | **8.2×** |
| Vulkan, both caches warm | **0.64 s** | **17.1×** |
| CUDA baseline | 1.21 s | 9.1× |

Warm Vulkan now **beats** CUDA on this laptop (0.64 s vs 1.21 s —
likely because NV_coopmat2 matmul kernels in ggml-vulkan are
better-tuned for this shape than the CUDA path's `cublasGemmEx`
call), and cold-run latency is now a one-time cost per install
rather than per run. Disable via `GGML_VK_DISABLE_PIPELINE_CACHE=1`;
inspect with `GGML_VK_PIPELINE_CACHE_DEBUG=1`.

**Lessons:**

1. When benchmarking GPU backends, **always run the target path
   twice** and report both cold and warm numbers. "Vulkan is 20×
   slower" was a first-run artifact that would have survived code
   review unchanged if we'd trusted the single measurement.
2. Shader native-compilation caching is **not** a driver-only
   concern. Every Vulkan application that loads the same shaders
   repeatedly should pass a `VkPipelineCache` to
   `vkCreateComputePipelines` / `vkCreateGraphicsPipelines` and
   persist it across runs. ggml-vulkan didn't, upstream — our
   patch should probably be submitted.
3. `_Exit()` bypasses destructors. Any caching scheme that only
   flushes in a destructor will silently lose its work on Windows
   builds that call `_Exit`. Periodic incremental save from the
   hot path (throttled) is a simple workaround that doesn't need
   new shutdown hooks.

### Issue #12 (prebuilt binary: silent exit after "using cached")

Reported against a prebuilt release binary on Windows 11 / Intel i3
with no NVIDIA GPU. User sees `crispasr: using cached …` and then
the process returns to the shell prompt — no `parakeet: vocab=…`,
no error, no crash dialog.

**Could not reproduce** at HEAD with a fresh local build on Windows
11 (with NVIDIA GPU). The same `parakeet-tdt-0.6b-v3-q4_k` command
transcribes correctly. Deliberately-corrupt cache files (1 KB
truncated, empty) all produce **loud** errors:

```
gguf_init_from_file_ptr: failed to read key-value pairs
core_gguf: failed to open '…' for metadata read
parakeet: failed to load '…'
crispasr[parakeet]: failed to load model '…'
crispasr: error: failed to initialise backend 'parakeet'
```

with exit code 13. So a partial-download cache file is not the cause.

**The giveaway in the reporter's log:** no `ggml_cuda_init: …` line,
which we always print at startup as long as `ggml-cuda.dll` loads
successfully (regardless of `--no-gpu`). On a machine with no
NVIDIA driver installed, `ggml-cuda.dll` depends transitively on
`cudart64_*.dll` / `cublas64_*.dll`. If those are missing, Windows
fails the DLL load. The backend registry might still swallow the
error and let the exe run on CPU — but depending on the loader
state, a later *deferred-bind* resolve can exit the process with
code `0xc0000135` / `STATUS_DLL_NOT_FOUND` with no stderr output at
all. That matches the reporter's symptom.

**Remediations to consider (none shipped yet):**
1. Ship a **CPU-only** prebuilt alongside the CUDA build for users
   without NVIDIA drivers. `build-windows.bat -DGGML_CUDA=OFF`
   produces a binary with no CUDA dependency.
2. Delay-load `ggml-cuda.dll` via `/DELAYLOAD:ggml-cuda.dll` + a
   `__HrLoadAllImportsForDll` guard, so a missing runtime falls
   back to CPU instead of exiting the process.
3. At startup, call `SetErrorMode(SEM_FAILCRITICALERRORS)` and log
   `GetLastError()` on any DLL resolve failure so the user sees
   *why* the process stopped.
4. Add a `--diagnose` subcommand that prints loaded backends,
   device list, and cache dir — one-line "is my install broken"
   check for end-users.

**Lesson:** A Windows process can exit **completely silently**
when a delay-loaded or transitively-required DLL is missing. Any
"prints one line then disappears" bug report on Windows should
first be diagnosed by (a) checking Event Viewer →
`Application` for a `Faulting module name` crash log, and (b)
running the binary against `Dependencies.exe` or
`dumpbin /dependents` to find the missing import. The codebase
itself is usually fine.

## Kyutai STT: causal padding, interleaved RoPE, and codec-based ASR

### Causal (left-only) padding in conv1d

moshi/Mimi uses `StreamingConv1d` which prepends
`pad_left = kernel_size - stride` zeros to the LEFT before conv1d with
padding=0. Standard symmetric padding produces completely wrong Mimi
encoder output — the SEANet outputs are numerically different and the
RVQ codes cascade to garbage.

**Fix:** `ggml_pad_ext(x, pad_left, 0, 0, 0, 0, 0, 0, 0)` before
`ggml_conv_1d(weight, x, stride, 0, 1)`. After this fix, SEANet output
was bit-perfect vs the official Python Mimi encoder.

### Interleaved vs NEOX RoPE

Kyutai models use **interleaved** RoPE (`[r0,i0,r1,i1,...]`), which is
`GGML_ROPE_TYPE_NORMAL = 0`. Not the NEOX layout (`[r0,r1,...,i0,i1,...]`)
used by Llama/Mistral/Qwen. Using the wrong RoPE type makes the encoder
transformer output diverge (max diff 0.07) and the LM produce garbage.

**Lesson:** Always check `rope.interleave` in the Python source. The two
layouts are **not** compatible — there's no graceful degradation, just
completely wrong output.

### Initial token IDs

The STT LM uses `text_card` (8000) as the initial text token and `card`
(2048) as the initial audio token — NOT the padding ID (3). These are
"start-of-sequence" tokens at the end of the vocabulary. The moshi.cpp
code: `text_initial_token_id = config.text_card; initial_token_id = config.card`.

### Stage-by-stage diff protocol (applied)

1. SEANet: bit-perfect after causal padding fix (max diff = 0.000000)
2. Encoder transformer: bit-perfect after RoPE + causal mask fix
3. RVQ codes: 99.3% match (100% codebook-0, FP residual drift on rest)
4. LM: correct "And so, my fellow Americans..." after all fixes

The causal padding bug was invisible at the architecture level — the
shapes were correct, the model ran without errors, but every single
output value was wrong. Only the diff-test protocol caught it.

## FireRedASR: Conformer encoder debugging (April 2026)

### Internal residual in ConformerFeedForward

The `ConformerFeedForward` module has a **hidden internal residual**:
```python
def forward(self, x):
    residual = x
    output = self.net(x)
    output = output + residual  # ← internal!
    return output
```

The Conformer block's macaron residual `0.5*x + 0.5*ffn(x)` expands to:
`0.5*x + 0.5*(net(x) + x) = x + 0.5*net(x)`.

My code was computing `0.5*x + 0.5*net(x)` — missing the `0.5*x` that
comes from the internal residual. The fix changed FFN1 from matching
at 0.3 error to matching at 0.0003.

**Lesson:** Always check `forward()` of ALL modules, not just the
top-level block. Hidden residual connections are easy to miss when
reading the block-level code `out = 0.5*x + 0.5*ffn(x)`.

### Relative positional encoding index formula

The `_rel_shift` operation maps:
`shifted[h, tq, tk] = original[h, tq, T-1-tq+tk]`

NOT `original[h, tq, tq-tk+T-1]` (the sign of `tq-tk` is flipped).
Verified with a T=5 example: `shifted[0,0] = original[0,4]`,
`shifted[0,1] = original[0,5]`, `shifted[1,0] = original[1,3]`.

### Positional encoding center extraction

`RelPositionalEncoding.forward()` extracts the CENTER of the PE table:
`pe[:, Tmax//2 - T + 1 : Tmax//2 + T]` where Tmax=9999.

Taking the FIRST positions (pe[0:2T-1]) gives completely wrong values
and causes the position attention to produce garbage.

### ggml reshape is column-major

`ggml_reshape_2d([T1, T2] → [T2, T1])` reinterprets the same flat
data with ne[0] as the fast dimension. This is NOT the same as
Python's `view(T2, T1)` which reinterprets with the LAST dimension
fastest (row-major). For the `_rel_shift` operation, this means ggml
reshape cannot be used — need CPU-side computation or transposing.

### Hybrid ggml/CPU encoder for relative position attention

When a model requires an operation that ggml can't express natively
(like rel_shift's row-major reshape), split the computation:
- **ggml** for all matrix multiplications (FFN, projections, conv)
- **CPU** only for the unsupported operation (attention scoring)

For FireRedASR: 2 ggml graphs per layer (pre-attention + post-attention)
with CPU attention scoring in between. This gave **20x speedup**
(323s → 16s) over the full-CPU approach, because ggml handles the
O(T*d²) matmuls while CPU only does the O(T²*d) attention scoring.

### Depthwise conv padding: causal vs symmetric

Streaming models (Mimi/Kyutai) use **causal** (left-only) padding:
`pad_left = kernel_size - stride`.

Non-streaming models (FireRedASR Conformer) use **symmetric** padding:
`pad = (kernel_size - 1) / 2` on each side.

Using the wrong padding gives completely wrong conv outputs but no
error — the shapes are the same. Always check the PyTorch Conv1d's
`padding` attribute to determine which type.

### Stage-by-stage protocol results (FireRedASR)

All 6 bugs were found by comparing at each sub-module boundary:

1. FFN1 diverged at residual: found hidden internal residual in
   `ConformerFeedForward.forward()` — `ffn(x) = net(x) + x`
2. MHSA diverged: content-only matched perfectly, position component
   was wrong → found rel_shift formula was inverted (`T-1-tq+tk` not
   `tq-tk+T-1`) AND PE was extracted from wrong offset (first vs center)
3. Conv diverged: found padding was causal (32+0) instead of symmetric
   (16+16) by checking `depthwise_conv.padding` attribute
4. Each fix was verified to bring the sub-module output within 0.002
   of the reference before proceeding to the next

Without the stage-by-stage protocol, these bugs would have been
invisible — the model runs without errors in all cases, just produces
wrong text.

## FireRedVAD: FSMN Conv1d replication

### Manual conv index arithmetic vs PyTorch Conv1d

The FSMN uses lookback and lookahead depthwise Conv1d with specific
padding and dilation. Manually computing `x[t - n*stride]` does NOT
match PyTorch's `Conv1d(padding=P, dilation=D)` because:

1. Conv1d pads BOTH sides, then applies the kernel
2. The output is then trimmed/sliced in the FSMN code
3. Manual indexing skips the padding step entirely

**Fix:** Replicate the EXACT Conv1d operation — pad the input, apply
kernel with stride/dilation, then apply the same trim/slice as Python:
- Lookback: `conv[:,:,:-(N1-1)*S1]` (trim right)
- Lookahead: `F.pad(conv[:,:,N2*S2:], (0, S2))` (skip left, pad right)

### int16 vs float32 fbank scaling

Kaldi-based models (FireRedVAD, FireRedASR) train on int16 audio
input to `kaldi_native_fbank`. The log-mel features differ by a
constant `2*log(32768) ≈ 20.79` vs float32 (-1..1) input. The CMVN
absorbs this offset, but if CMVN was trained on int16 features and
you feed float32 features, the normalization is wrong.

**Fix:** Scale float32 input by 32768 before fbank computation:
`frame[i] = pcm[i] * 32768.0f`

### Decoder n_head mismatch (FireRedLID)

The FireRedLID decoder uses 8 attention heads (`layer_n_head=8`) but
the encoder uses 20 heads (`n_head=20`). The C++ code used the
encoder's n_head for the decoder, producing random language predictions
instead of "en" for English audio.

**Lesson:** Encoder and decoder may have DIFFERENT n_head values. Always
store them separately in the GGUF metadata and read both.
After fix: LID correctly identifies English on JFK audio.

### GGML_NATIVE=ON on CI runners silently ships AVX-512 to AVX2-only laptops

v0.4.10 Windows prebuilts (CPU / CUDA / Vulkan) all silently exited
with code 0 and no stderr output on a consumer AVX2 laptop CPU —
reproducing issue #12's "using cached → nothing" symptom exactly.

**Root cause**: ggml's `GGML_NATIVE` CMake option defaults to `ON`
unless cross-compiling. On the GitHub Actions `windows-latest`
runner (Azure Standard_D4_v3 or similar, typically with AVX-512),
`GGML_NATIVE=ON` detects the host CPU and emits AVX-512 / AVX10
instructions into `ggml-cpu.dll`. The binary then ships to users on
any x86-64 machine and the first AVX-512 instruction triggers
`STATUS_ILLEGAL_INSTRUCTION` (0xc000001d). On Windows, the exception
handler silently terminates the process — **exit code 0, no stderr,
no event-log entry that a casual user would find.**

**Isolation protocol** (used here to pin the bug):
1. Suspect `ggml-cpu.dll` because it's the only binary whose SIMD
   level changes with host-CPU autodetection.
2. Confirm with a file-size diff between a locally-built (known-good)
   DLL and the CI-built one: **42 KB larger on CI** (823 KB vs 780 KB).
   ~42 KB is the right order of magnitude for additional
   VEX-512-encoded instructions across a matmul + cpy kernel set.
3. Swap *only* the CI `ggml-cpu.dll` for the local one in the
   downloaded zip → the whole pipeline works. Put it back → silent exit.

**Fix** (release.yml, every Windows job): pass
`-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON` to
cmake. AVX2 is the right compat baseline — every x86-64 CPU shipped
since ~2013 (Intel Haswell / AMD Excavator) supports it. Users on
older CPUs or those wanting AVX-512 native kernels should build
from source.

**Alternative (not shipped here)**: set
`GGML_CPU_ALL_VARIANTS=ON GGML_BACKEND_DL=ON BUILD_SHARED_LIBS=ON` —
ggml builds one `ggml-cpu-<arch>.dll` per ISA level (x64, sse42,
sandybridge, haswell, skylakex, cannonlake, cascadelake, icelake,
cooperlake, zen4) and dispatches at runtime. Proper solution, but
adds ~10 DLLs to the package and requires `BUILD_SHARED_LIBS=ON`
which conflicts with our static-CPU prebuilt. Worth revisiting for
the CUDA / Vulkan variants since they're already shared-libs.

**Lessons** (in decreasing order of load-bearingness):

1. **Never ship CI-built binaries with `GGML_NATIVE=ON`.** The CI
   runner's CPU is *not* a representative target CPU. Always pin an
   explicit SIMD baseline for release artifacts. This is the #1
   "prebuilt works on my machine but nobody else's" footgun in
   ggml-based projects.

2. Silent SIGILL on Windows looks identical to "the program does
   nothing" — exit 0, no console output, no crash dialog (unless WER
   is configured to show them). It's not until you attach a debugger
   or check `Event Viewer → Windows Logs → Application` for the
   `Faulting module name: ggml-cpu.dll` entry that the real cause
   becomes visible. **Assume silent-exit on Windows is an illegal
   instruction until proven otherwise.**

3. File-size diffs between CI and local builds of the same DLL are
   a *very* strong signal. Same commit + same CMake flags should
   produce byte-sized-identical outputs (modulo timestamps, which
   shouldn't change size). A +42 KB difference in `ggml-cpu.dll`
   was the only clue we had, and it turned out to be the whole
   story.

### CUDA `cublas64_XX.dll` imports `cublasLt64_XX.dll` transitively

v0.4.10 CUDA prebuilt trimmed cublasLt64_12.dll (474 MB) on the
reasoning that `ggml-cuda.dll`'s own PE import table doesn't list
it — only `cublas64_12.dll`, `cudart64_12.dll`, and driver-loader
DLLs (`nvcuda.dll`). The reasoning was wrong: **`cublas64_12.dll`
itself imports `cublasLt64_12.dll`** (verified via PE import scan).
Without cublasLt, Windows fails `crispasr.exe` at load time with
`STATUS_DLL_NOT_FOUND` — same silent-exit symptom as the SIGILL
case above.

Upstream ggml explicitly notes in `ggml-cuda/CMakeLists.txt`:

> As of 12.3.1 CUDA Toolkit for Windows does not offer a static
> cublas library

so there's no side-stepping this via `GGML_STATIC` on Windows.
The cublasLt cost is unavoidable unless you're willing to replace
all ggml's `cublasGemmEx` calls with hand-written CUDA kernels.

**Lesson**: when triaging a Windows "my exe silently exits" bug,
check **transitive** DLL imports, not just the binary you control.
`dumpbin /dependents` / PE import parsing only shows first-order
imports — you need to walk the chain recursively. On this project
the chain was `crispasr.exe → ggml-cuda.dll → cublas64 → cublasLt`.

### Quantized weight dequantization (read_f32_vec)

The hybrid ggml/CPU encoder reads weights into CPU float vectors via
`read_f32_vec`. The original only handled F16→F32. Quantized models
(Q8_0, Q4_K_M, etc.) passed raw quantized bytes to float arrays →
garbage or crash.

**Fix:** Use `ggml_get_type_traits(t->type)->to_float` to dequantize
any type. Also apply to the conv2d subsampling lambda.

### Conv1d kernel=1 stored as 3D blocks quantization

Pointwise Conv1d weights `[out, in, 1]` stored as 3D tensors in GGUF
have `ne[0]=1`, failing the quantizer's row-alignment check (1 % 256 ≠ 0).
~30% of model weights were left unquantized.

**Fix:** Squeeze the kernel dimension in the converter (`t.squeeze()`
when shape has a `1` and name contains `pointwise_conv`). Makes them 2D
`[out, in]` → quantizer can process normally. Saves ~40% at Q2_K.

**Architecture-specific:** Only apply for `firered` architecture. Other
models' 3D conv weights may be actual spatial kernels.

### LID decoder decode length

FireRedLID only needs 1 decode step — the first token after SOS is
the language code. Running full beam search (300 steps, beam=3) wastes
~50x compute. Detect LID models by `odim <= 256` and set `max_len=2`,
`beam_size=1`.

### LID output mapping

The LID model outputs multi-token sequences for dialect languages
(e.g., "zh" then "mandarin" for Mandarin Chinese). Taking only the
first non-special token gives the ISO 639-1 code.

### Layer pruning for LID

Tested removing encoder layers to shrink the LID model. Only keeping
the last 4 of 16 layers (12-15) works for a single English sample,
but fails on multilingual test (0% accuracy). SLERP merging of adjacent
layers also fails. The Conformer encoder layers are too specialized
for simple pruning — unlike Whisper Turbo's decoder-only pruning.

### Q2_K too aggressive for similar languages

Q2_K quantization causes confusion between similar languages
(de→cy, hi→pa, es→gl). Q4_K maintains accuracy. For LID,
Q4_K (544 MB) is the practical minimum; Q2_K (350 MB) is unreliable.

### ECAPA-TDNN LID: fbank mismatch produces "nn" for everything

SpeechBrain's `lang-id-voxlingua107-ecapa` (Apache-2.0, 43 MB, 107 langs)
was trained with `torchaudio.compliance.kaldi.fbank`. Replacing this with
a simple mel fbank (Hamming window, no Kaldi preprocessing) causes the
model to predict "nn" (Norwegian Nynorsk) for ALL inputs — English, Thai,
German, even the model's own Thai test file.

Tested fbank variants that all fail:
- Simple Hamming+FFT (our C++ default)
- Kaldi-style with preemphasis+Povey window (manual Python)
- `kaldi_native_fbank` library (proper Kaldi C++ implementation)

All produce "nn" with ~0.1 confidence = near-random. The model
requires **exact** `torchaudio.compliance.kaldi.fbank` preprocessing.
Our dev machine has a broken torchaudio (missing CUDA libs), preventing
verification.

Note: "nn" as a default/wrong prediction was also seen in early
FireRedLID debugging — may be a common failure mode when fbank
features are in the wrong distribution (the model learned to map
out-of-distribution features to a specific class).

**Status:** ECAPA-TDNN is WIP. Infrastructure built (converter, runtime,
CLI/API integration). Accuracy blocked on fbank compatibility.
Path forward: test on machine with working torchaudio, or use ONNX
export (Xenova/ecapa-voxlingua107 may exist).

### Qwen Omni vs Qwen3-ASR: not worth implementing separately

Qwen2.5-Omni (3B/7B) and Qwen3-Omni (30B MoE) are multimodal models
(audio+vision+text+speech generation). For pure ASR:

- Much larger than Qwen3-ASR (0.6B/1.7B) with no accuracy advantage
- Split GGUF architecture (mmproj + LLM) — incompatible with our monolithic GGUF
- Already supported by llama.cpp's libmtmd
- Thinker-Talker architecture adds complexity with no ASR benefit

**Recommendation:** Stick with Qwen3-ASR for ASR. Omni models are
for multimodal use cases (speech generation, vision, etc.).

### SpeechBrain Conv1d uses reflect padding, not zero padding

SpeechBrain's `Conv1d` wrapper defaults to `padding_mode='reflect'`,
not zero padding. This causes the conv1d output to differ
dramatically at sequence boundaries. For the ECAPA block0 with k=5:
- Zero pad: `out[co, 0] = 45.07` (uses two zero-padded frames)
- Reflect pad: `out[co, 0] = 76.93` (uses reflected input frames)

The 70% difference at the first frame propagates through the network.
After fixing this, block0 output matches Python reference to <0.01.

**Lesson:** Always check the `padding_mode` attribute of Conv1d wrappers.
SpeechBrain, TorchAudio, and PyTorch all have different defaults.

### SpeechBrain skip_transpose flag is critical

SpeechBrain's `Conv1d` and `BatchNorm1d` both have a `skip_transpose`
flag that controls whether they transpose `[N, C, T] ↔ [N, T, C]`
before/after the underlying PyTorch operation. The ECAPA-TDNN model
uses `skip_transpose=True` for both conv and BN, meaning:
- Conv1d operates on `[N, C, T]` (standard temporal convolution)
- BatchNorm1d normalizes over channels (standard)

Without knowing this, one might transpose the input, causing the
conv to operate over channels instead of time (completely wrong).

### ECAPA-TDNN SE-Res2Net debugging status

Block0 (TDNNBlock) output matches Python after fbank + reflect pad fixes.
SE-Res2Net blocks (1-3) still produce different output. Possible causes:
- Res2Net sub-band cumulative connection ordering
- Dilation handling in reflect-padded dilated conv
- SE block global average pooling implementation
- Residual connection arithmetic

The model architecture is complex (8-way channel split, sequential
processing with cumulative additions, squeeze-excitation attention).
Each sub-component needs stage-by-stage comparison.

### Facebook OmniASR-CTC-300M architecture

fairseq2-based, NOT HuggingFace Transformers:
- 7-layer CNN feature extractor: Conv1d(1→512, k=10) + 6× Conv1d(512→512, k=3)
  with LayerNorm + GELU (wav2vec2 pattern, ~320x downsampling)
- Linear(512→1024) dimension projection
- 24 Transformer encoder layers: d=1024, 16 heads, FFN=4096
- Final projection: Linear(1024→9812) CTC head
- SentencePiece tokenizer (9812 tokens)
- 325M params, ~1.3 GB F32
- Input: raw 16kHz PCM (no mel features)
- Apache-2.0, 1600+ languages

### ECAPA-TDNN SE/tdnn2 ordering bug

SpeechBrain's `SERes2NetBlock.forward` processes in order:
  `tdnn1 → res2net → tdnn2 → SE → residual`

Our initial implementation had SE before tdnn2:
  `tdnn1 → res2net → SE → tdnn2 → residual`

The SE block's squeeze (global average pool) operates on the tdnn2
output, not the res2net output. With the wrong order, the SE scale
was computed from the wrong features, causing completely different
final outputs (mean=0.009 in Python vs mean=-0.133 in C++).

**Lesson:** When implementing a complex block with multiple sub-modules,
always verify the execution order from the Python forward() source.
The intuitive order (SE after the "main" processing) was wrong —
SpeechBrain applies a post-projection (tdnn2) before squeeze-excitation.

### ECAPA-TDNN: 43 MB model achieves ~100% on 12-language TTS benchmark

The SpeechBrain ECAPA-TDNN (21M params, 43 MB F16) correctly identifies
all 12 test languages (en, de, fr, es, ja, zh, ko, ru, ar, hi, pt, it)
with p ≥ 0.96 confidence on edge-tts generated samples.

This is dramatically better than FireRedLID (544 MB Q4_K, 83% accuracy)
for common languages, and 13x smaller. For the 25 extra languages
(Chinese dialects) that FireRedLID covers, it remains the only option.

### ggml tensor layout for conv1d input

ggml uses column-major storage. A 2D tensor `[T, C]` has `ne[0]=T, ne[1]=C`.
The flat data layout is `data[c * T + t]` — channels change SLOWER than time.

This is the SAME as our CPU layout `x[c * T + t]`. So when passing CPU arrays
to ggml tensors, NO transpose is needed — just copy directly.

The confusion arises because `ggml_conv_1d(kernel [K,IC,OC], input [T,IC])`
produces output `[T_out, OC]`, and the flat layout of input `data[ic * T + t]`
puts consecutive time steps of the same channel together — which IS what
conv1d processes along.

**Lesson:** For ggml 2D tensors, `ne[0]` is the fast-changing (innermost)
dimension. For `[T, C]`: time changes fastest, channels slowest.
CPU row-major `x[c * T + t]` and ggml column-major `data[c * T + t]`
are the SAME thing — both index as `slower_dim * faster_size + faster_dim`.

### ggml_pad_reflect_1d exists

ggml has `ggml_pad_reflect_1d(ctx, tensor, pad_left, pad_right)` for
reflect padding. Use this instead of ggml_conv_1d's built-in zero padding
when the model expects reflect padding (SpeechBrain default).

### OpenMP for CPU-only models

Adding `#pragma omp parallel for` to the outer loop of conv1d (over output
channels) and batchnorm1d (over channels) gives ~2x speedup on 4 threads
for ECAPA-TDNN. The CMakeLists needs explicit OpenMP linkage:
```cmake
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ecapa-lid PUBLIC OpenMP::OpenMP_CXX)
endif()
```

### CRITICAL: ggml column-major layout = C-style row-major for 2D arrays

This is the most important ggml lesson we keep re-learning:

**ggml 2D tensor `[A, B]`** means `ne[0]=A, ne[1]=B`. The flat data
layout is `data[b * A + a]` — `ne[0]` changes fastest (column-major).

**C/C++ 2D array `x[B][A]`** or `x[b * A + a]` — also has `A` changing
fastest (row-major).

**THEY ARE THE SAME LAYOUT.** For a tensor representing `[C, T]` where
C is channels and T is time:
- ggml: `ne[0]=C, ne[1]=T`, data at `data[t * C + c]` — C fastest
- C++: `x[c * T + t]` — WAIT, this has T fastest, not C!

**This is where it gets confusing.** When we store data as `x[c * T + t]`
in C++, this is a `[C, T]` array where T is the inner (fastest) dimension.
In ggml, this SAME layout corresponds to `ne[0]=T, ne[1]=C` — because
ggml's ne[0] is the fastest dimension.

**Rule of thumb:**
- If C++ stores as `x[outer * inner_size + inner]`
- Then ggml tensor should have `ne[0]=inner_size, ne[1]=outer_size`
- The flat data bytes are identical — just copy, don't transpose!

**For `ggml_conv_1d(kernel [K,IC,OC], input [T,IC])` → `[T_out,OC]`:**
- Input ne[0]=T, ne[1]=IC → flat: `data[ic * T + t]`
- Our C++ `x[c * T + t]` stores channel c at `data[c * T + t]`
- SAME layout → just copy x to ggml tensor directly

**For `ggml_mul_mat(a [C_in,C_out], b [C_in,T])` → `[C_out,T]`:**
- Requires `a.ne[0] == b.ne[0]` (both = C_in)
- Input `b` must be `[C_in, T]` with ne[0]=C_in
- If input is from conv1d `[T, C]` with ne[0]=T, transpose first

**For reading ggml output to C++ array:**
- ggml tensor `[T, C]` (ne[0]=T, ne[1]=C): `data[c * T + t]`
- C++ wants `x[c * T + t]`
- SAME layout → just copy, no transpose!

This caused bugs 3 times in ECAPA-TDNN:
1. Input: incorrectly transposed before feeding to ggml_conv_1d
2. MFA output: incorrectly treated as row-major when reading to CPU
3. build_conv1d_k1: unnecessary transpose of already-correct data

### OmniASR-CTC-300M: first working GGUF conversion

Successfully converted facebook/omniASR-CTC-300M to GGUF (0.65 GB F16,
423 tensors). Model loads, ggml graph computes in 7.7s for 11s audio.
But CTC decode returns empty (all blanks).

Architecture:
- 7-layer CNN: Conv1d strides [5,2,2,2,2,2,2] = 320x downsampling
- Linear(512→1024) projection
- 24 Transformer encoder layers (pre-norm, 16 heads, FFN=4096, GELU)
- CTC head: Linear(1024→9812) with SentencePiece tokenizer

Key: this is fairseq2-based (not HuggingFace), so tensor names differ
from standard wav2vec2. The converter shortens names to fit 64-char GGUF
limit. CNN strides stored as array in GGUF metadata.

The CTC blank = pad_id = 1 (SentencePiece <pad>).

### OmniASR-CTC: three critical findings

1. **Input normalization required**: wav2vec2 models expect `layer_norm(waveform)`
   — zero mean, unit variance. Without this, the model outputs mostly blanks.

2. **CTC blank = token 0 (<s>)**: In fairseq2, the BOS token serves as CTC blank.
   NOT token 1 (<pad>) which is the HuggingFace convention.
   The official code just removes consecutive duplicates + skip_special_tokens.

3. **Pos conv padding**: fairseq2 uses `padding = K // 2` (=64 for K=128),
   not `(K-1) // 2` (=63). The extra padding element gives correct same-padding
   for even kernel sizes. Without this, the pos encoding is misaligned by 1 frame.

4. **No language conditioning for CTC**: confirmed from official repo comment
   "It is ignored when performing inference with CTC." The CTC model is
   fully language-agnostic across 1600+ languages.

### OmniASR audio length limit

Official docs: "Currently only audio files shorter than 40 seconds are
accepted for inference." Models trained on ≤30s segments. For longer
audio, use VAD segmentation to split into chunks.

Our implementation doesn't enforce this limit — it will run on longer
audio but quality degrades. The CNN downsampling (320x) means 40s of
16kHz audio = 2000 frames through the transformer, which is within
typical attention window limits.

### fairseq2n native extension

fairseq2's Python package requires `fairseq2n` C++ extension which is
compiled for specific Python/CUDA combos. Not available for Python 3.13
or CPU-only setups via pip. Our manual forward pass serves as reference.

### ECAPA-TDNN: two model variants (VoxLingua107 vs CommonLanguage)

SpeechBrain has two ECAPA-TDNN LID models with different hyperparameters:

| | VoxLingua107 | CommonLanguage |
|---|---|---|
| n_mels | 60 | 80 |
| lin_neurons | 256 | 192 |
| Classifier | DNN (BN→Linear→BN→LeakyReLU→Linear) | Cosine (normalize(emb) @ normalize(weight)) |
| Labels | ISO codes (en, de, ...) | Full names (English, German, ...) |
| Languages | 107 | 45 |

The converter auto-detects these from `hyperparams.yaml` and `classifier.ckpt`
structure, storing `ecapa.cls_type` (0=DNN, 1=cosine) and `ecapa.lin_neurons`
in the GGUF metadata.

**Cosine classifier**: `F.linear(F.normalize(emb), F.normalize(weight))` — each
class output is the cosine similarity between the normalized embedding and the
normalized class weight vector. Scores are in [-1, 1], not softmax probabilities.

### ECAPA-TDNN: quantization destroys accuracy

ECAPA-TDNN cannot be meaningfully quantized. Even Q8_0 produces all-wrong
predictions (always returns "ms" regardless of input). Root causes:

1. **Small conv1d kernels**: The res2net conv weights are `[128, 128, 3]` (49K elements).
   Q8_0 block size 32 doesn't divide K=3, so ggml skips them — but the tdnn1/tdnn2
   weights `[1024, 1024]` ARE quantized, which corrupts the embedding.
2. **ggml_conv_1d + quantized weights**: The conv1d op may not properly dequantize
   weight tensors during computation, producing garbage output.
3. **Cosine classifier sensitivity**: Even small perturbations in the 192-dim
   embedding space flip the argmax due to narrow angular margins between classes.

**Conclusion**: Ship ECAPA-TDNN as F16 only. At 40-43 MB it's small enough
that quantization savings (14 MB Q4_K) aren't worth the accuracy loss.

### OmniASR-CTC: two GGUF formats (fairseq2 vs HF-native)

The fairseq2-converted GGUF (`omniasr-ctc-300m.gguf`) uses tensor names like:
- `cnn.0.ln.weight`, `enc.0.attn_ln.weight`, `enc.0.attn.q_proj.weight`
- `enc.0.ffn.up.weight`, `enc_ln.weight`, `ctc.weight`, `proj.weight`

The HF-native conversion (aadel4/omniASR-CTC-300M-v2) uses wav2vec2 names:
- `cnn.0.norm.weight`, `enc.0.ln1.weight`, `enc.0.attn.q.weight`
- `enc.0.ffn.fc1.weight`, `lm_head.weight`, `feat_proj.weight`

Our omniasr runtime expects the fairseq2 format. The HF-native model can
potentially be used with the existing wav2vec2 backend instead.

### OmniASR-LLM: decoder architecture and language conditioning

The OmniASR-LLM variant adds a 12-layer LLaMA decoder (d=4096, 8 heads,
head_dim=512, SwiGLU FFN with d_ffn=2816). The encoder is identical to CTC.

**Decoder input sequence** (from `create_default_syntax` in model.py):
```
[audio_embeddings...] [lid_marker] [lang_embedding] [BOS] [generated_tokens...]
```

**Special tokens** (from `Wav2Vec2LlamaSpecialTokens`):
- `lid_marker` = vocab_size (9812) — extra entry in text_frontend embedding
- Language ID = index in supported_langs list + 1 (factory.py adds +1, index 0 = no-language)
- BOS = 0, EOS = 2, PAD = 1

**Language ID mapping** (from `factory.py`):
```python
lang_mapping = {row["lang"].lower(): row["index"] + 1 for row in parquet_table}
```
Key indices: eng_Latn=414, deu_Latn=365

**RoPE**: fairseq2 uses interleaved pairing `(x[2i], x[2i+1])` — this maps to
`GGML_ROPE_TYPE_NORMAL` (mode 0), NOT NEOX (mode 2). This differs from most
HuggingFace LLMs which use `rotate_half` (NEOX). Getting this wrong produces
fluent but wrong-language output (Greek in our case).

**v1 vs v2**: Always use v2 models (`omniASR_LLM_300M_v2`). The v2 uses a
different tokenizer (`omniASR_tokenizer_written_v2`, 10288 tokens vs 9812)
and is the only variant that reliably transcribes challenging English audio.
v2 checkpoints available at `dl.fbaipublicfiles.com/mms/omniASR-LLM-300M-v2.pt`.

**Critical bug found**: The LLM converter shortened
`encoder_frontend.post_extract_layer_norm` to `post_extract_ln` but the runtime
code looked for the long name, got nullptr, and silently skipped the LayerNorm.
This caused the projection output to diverge (cos=0.77 vs reference) making
all downstream output garbage. Fix: try both short and long tensor names.

**Before fix**: "it sounded to one and that was a particular pillow..."
**After fix**: "and so my palamericas is not what your country can do for you..."

The reference dump protocol (dump intermediates at each stage, compare cosine
similarity) caught the bug immediately. CNN output was cos=0.999999, but
proj_out diverged to cos=0.767. The fix brought all stages to cos>0.9999.

### OmniASR-LLM: quantization requires skipping bridging tensors

Quantizing the OmniASR-LLM decoder with Q4_K/Q8_0 causes immediate EOS
output (0 generated tokens). Even Q8_0 is broken. Root cause: four
bridging tensors between encoder and decoder are precision-critical:

- `enc_proj.weight` (1024 -> 4096 projection)
- `lm_head.weight` (4096 -> 10288 vocabulary logits)
- `tok_emb.weight` (10289 token embeddings)
- `lang_emb.weight` (1694 language embeddings)

**Fix**: Skip these tensors during quantization (keep as F16). Added to
crispasr-quantize skip rules. With this fix, Q4_K (1.1 GB) produces
identical output to F16 (3.1 GB) — 3x size reduction with no quality loss.

### OmniASR: scheduler must include a CPU fallback backend

OmniASR initialized its ggml scheduler with only the "best" backend:

```cpp
ctx->sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, ...);
```

That worked until ggml tightened `ggml_backend_sched_new()` and started
asserting that the last backend in the scheduler list is CPU when a GPU
backend is present. On CUDA builds this crashed immediately during
`omniasr_init_from_file()` with:

`GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU)`

**Fix:** Mirror the working pattern used by the other backends: keep a
separate `backend_cpu`, append it to the scheduler backend list when the
main backend is not CPU, and free it separately on shutdown.

### OmniASR: `-ng` has to be plumbed into backend selection explicitly

The OmniASR CLI adapter ignored `whisper_params.use_gpu`, so `-ng` and
`--gpu-backend cpu` still called `ggml_backend_init_best()` and tried to
construct a GPU-first backend stack. After the scheduler fix this no
longer crashed, but the flag semantics were still wrong.

**Fix:** Add `use_gpu` to `omniasr_context_params` and set it from the
CLI adapter, treating `--gpu-backend cpu` the same as `-ng`.

**Lesson:** For the non-whisper backends, GPU selection is not automatic
just because the top-level CLI parsed the flag. Each backend adapter has
to propagate that intent into its own backend picker.
## 2026-04-22 - No-gpu mode must gate `ggml_backend_load_all()`

- Symptom: `-ng --gpu-backend cpu` could still print `ggml_cuda_init: found ...` even after backend-specific code paths stopped using `ggml_backend_init_best()`.
- Root cause: `examples/cli/cli.cpp` called `ggml_backend_load_all()` before parsing CLI flags, and `src/cohere.cpp` also loaded all backends unconditionally. That dynamic registration path probes CUDA as soon as the CUDA backend is loaded.
- Fix: parse CLI args first, then call `ggml_backend_load_all()` only when `params.use_gpu` is true and `params.gpu_backend != "cpu"`. Any backend with its own unconditional `ggml_backend_load_all()` must apply the same guard.
- Result: CPU-forced runs stop triggering global CUDA discovery just to select a CPU backend.

## 2026-04-23 - FireRed decoder optimization triage

### What actually helped

On the FireRed AED decoder, the wins came from moving **large, reused**
decoder matmuls onto ggml/GPU and from removing unnecessary beam work:

- Copy-on-write beam KV history instead of deep-copying `sa_k` / `sa_v`
  on every beam fork
- Dedicated greedy path for `beam_size == 1`
- Removing unused log-softmax bookkeeping from the greedy path
- Moving cross-attention encoder-side `K/V` precompute onto ggml/GPU
- Moving final decoder vocab projection onto ggml/GPU

Measured on `issue19-5s.wav` (`-t 8 -l en`) on the RTX A1000 laptop:

- Original baseline, `-bs 1`: `26.86s`
- Current best, `-bs 1`: `8.68s`
- Original baseline, `-bs 3`: `29.59s`
- Current best, `-bs 3`: `19.02s`

So the current FireRed decoder is about:

- `3.1x` faster for greedy decode
- `1.56x` faster for beam size 3

### What did not help

Several intuitive CPU-side micro-optimizations were regressions and were
reverted:

- Per-call scratch-buffer reuse inside the decoder loop
- Streaming logsumexp / top-k rewrite for the vocab projection
- Parallel `gemv` helper for small decoder vector-by-matrix products
- Per-call ggml graphs for small decoder MLP projections

The common pattern: **small per-step graphs or tiny parallel regions lose
to their own launch/alloc/scheduling overhead**. The decoder only speeds
up when the moved work is both substantial and reused.

### FireRed decoder strategy going forward

The remaining useful path is **larger persistent decoder subgraphs**, not
more loop-level CPU tuning. In particular:

1. Keep shared heavy decoder work on ggml/GPU (`K/V` precompute, logits).
2. Avoid one-graph-per-small-matmul designs.
3. Next meaningful step is a reused greedy decoder subgraph per layer or
   per token step, not isolated ggml calls for single projections.

### Data2Vec / HuBERT: three architecture traps when reusing wav2vec2 backend

Data2Vec and HuBERT share wav2vec2's CNN frontend + transformer encoder +
CTC head, but differ in three subtle ways that each cause complete failure
if wrong:

1. **Multi-layer positional convolution**: Data2Vec has 5 layers of
   `Conv1d(K=19, groups=16) + LayerNorm(no_affine) + GELU`, not 1 layer
   like wav2vec2. Only storing the first layer's weights causes the pos_conv
   output to diverge entirely (cos=0.08). Fix: store all N layers in GGUF
   as `pos_conv.{i}.weight/bias` and run them sequentially in C++.

2. **Global encoder LN placement**: Data2Vec applies the global encoder
   LayerNorm **BEFORE** the transformer layers, then uses post-norm inside
   each layer. wav2vec2 and HuBERT apply the global LN **AFTER** all layers.
   This is unique to Data2Vec and requires a separate flag
   (`global_ln_before_encoder=1`). Getting it wrong amplifies logits ~46x.

3. **Post-norm despite LayerNorm CNN**: Data2Vec uses LayerNorm in ALL CNN
   layers (like HuBERT) but uses **post-norm** in the encoder (unlike HuBERT
   which uses pre-norm). The encoder layer does `attn→add→LN→FFN+add→LN`.
   Setting `do_stable_layer_norm=1` (pre-norm) produces complete garbage.

Each bug was caught by systematic stage-by-stage diff against Python ref:
- CNN output: cos=0.999997 (correct from the start)
- feat_proj: cos=0.999968 (correct)
- pos_conv layers 0-4: cos>0.999961 (after multi-layer fix)
- after_global_ln: cos=0.999946 (after LN placement fix)
- **logits: cos=0.999972** (after post-norm fix)
- C++ decode matches Python exactly: "AND SO A MY FELLOW AMERICANS..."

### VibeVoice-ASR-1.5B: σ-VAE + Qwen2 hybrid architecture

VibeVoice uses a novel pipeline: two parallel σ-VAE CNN encoders (acoustic +
semantic) → linear connectors → Qwen2-1.5B autoregressive decoder.

**Key architecture findings:**
1. **Encoders are ConvNeXt-style**: 7 stages of `downsample_conv → N × Block1D`.
   Block1D = `RMSNorm → depthwise_conv → gamma_scale → residual + RMSNorm → FFN → gamma_scale → residual`.
2. **Depthwise conv via ggml_conv_1d_dw**: works but forces F16 im2col
   internally, causing cumulative precision loss (cos=0.7 after 29 blocks;
   Python F16 gives cos=0.999, so it's ggml-specific).
3. **Causal padding**: `padding_total = (K-1)*dilation - (stride-1)`, NOT `K-1`.
   Plus `get_extra_padding_for_conv1d` for stride alignment on the right side.
4. **Connectors**: simple `FC1 → RMSNorm → FC2` (NO activation, no SiLU).
5. **Scaling factors**: `speech_scaling_factor/speech_bias_factor` are for the
   base TTS model, NOT the ASR variant. ASR uses raw features directly.
6. **σ-VAE sampling**: ASR calls `.sample(dist_type='gaussian')` which adds
   noise. For deterministic C++ inference, using `.mean` (mode) is fine.
7. **Prompt template**: Qwen2 chat format with `<|object_ref_start|>` as
   speech_start, `<|box_start|>` as speech_pad, `<|object_ref_end|>` as
   speech_end. These repurpose existing Qwen2 special tokens.
8. **Qwen2 Q/K bias**: unlike most LLMs, Qwen2 has bias on Q and K projections
   (but not V and O). The `core_attn::kv_self_attn` helper doesn't support
   per-projection biases — needs inline attention implementation.

**For decoding (tokens → text)**: no tiktoken/BPE library needed. Just embed the
151665-token Qwen2 vocab as `tokenizer.ggml.tokens` in the GGUF and do
`vocab[token_id]` lookup. BPE merge rules are only needed for encoding
(text → tokens), which we don't do for ASR inference.

**ggml precision issue**: `ggml_conv_1d_dw` (line 4494 in ggml.c) creates
im2col with `GGML_TYPE_F16` regardless of input type. Through 29 ConvNeXt
blocks, each with a depthwise conv, this accumulates precision loss. Fix options:
1. CPU depthwise conv (simple loop, avoids im2col entirely)
2. Modify ggml to use F32 im2col when input is F32
3. Accept lower precision and rely on LM decoder robustness

### VibeVoice decoder: systematic debugging status

The Qwen2 decoder consistently outputs `<|vision_pad|>` (token 151654)
regardless of input. Confirmed NOT an encoder issue — injecting Python
reference features (cos=1.0) produces the same wrong output.

**Verified correct:**
- Embedding tensor values match Python checkpoint
- Prompt template matches processor output (143 tokens + assistant prefix)
- LM head uses tied weights (lm.tok_emb.weight)
- Embedding layout: data[token_id * d_lm + dim] (ggml column-major)

**Likely causes (in order):**
1. **RoPE theta**: Qwen2 uses theta=1000000.0 (not 10000). Our code sets this
   but the actual ggml_rope_ext call might interpret it differently.
2. **GQA native mode**: with n_heads=12, n_kv_heads=2, GQA ratio=6:1.
   The flash_attn_ext native GQA mode might handle this wrong for Qwen2.
3. **Causal mask**: the mask construction might be wrong for the prefix-fill case.
4. **Q/K bias interaction with RoPE**: bias is added before RoPE, which is correct
   in Python but might interact differently with ggml_rope_ext.

**Critical discovery**: the standalone Qwen2 decoder (loaded from VibeVoice
weights) produces `<|vision_pad|>` even in Python with correct features.
The full VibeVoice forward pass (`model.generate` with internal encode_speech)
is required — the speech features get special handling inside the model's
forward method that a standalone Qwen2 forward doesn't replicate.

**Model variant concern**: `microsoft/VibeVoice-1.5B` might not be the primary
ASR model. The documented ASR model is `microsoft/VibeVoice-ASR` (7B) or
`microsoft/VibeVoice-ASR-HF`. The 1.5B variant produced garbage on 2s of
mostly-silent audio. Need to verify with actual speech content before further
C++ debugging.

**CRITICAL**: `microsoft/VibeVoice-1.5B` is a **TTS model**, NOT ASR!
The HF model card explicitly says "Use to generate any text transcript"
is OUT OF SCOPE. We were using the WRONG model variant.

The correct ASR model is `microsoft/VibeVoice-ASR` (7B):
- Architecture: `VibeVoiceForASRTraining`
- Decoder: d=3584, 28 layers, 28 heads, 4 KV heads (bigger than 1.5B TTS)
- Same encoder: vae_dim=64, ratios=[8,5,5,4,2,2]
- Vocab: 152064 (slightly different from 1.5B's 151936)

Our C++ pipeline (encoder + connectors + Qwen2 decoder) has the right
architecture — just needs the correct 7B ASR weights. The converter
handles different decoder dimensions automatically.

---

## FireRedPunc / fullstop-punc — BERT punctuation restoration (April 2026)

### Architecture
Two punctuation models implemented as post-processors:

| Model | Base | Layers | d_model | Heads | Vocab | Labels | Tokenizer |
|---|---|---|---|---|---|---|---|
| FireRedPunc | BERT (LERT) | 12 | 768 | 12 | 21,128 | 5 (space/，/。/？/！) | WordPiece |
| fullstop-punc | XLM-RoBERTa-large | 24 | 1024 | 16 | 250,002 | 6 (space/./,/?/-/:) | SentencePiece |

Both are token classifiers: BERT/RoBERTa encoder → Linear(d, n_classes).
ggml graph uses `ggml_flash_attn_ext` for multi-head attention.

### Bugs found and fixed

**1. Missing SEP token (critical)**
BERT and RoBERTa both expect `[CLS] tokens [SEP]` as input. Our code
only prepended CLS (`seq_len = N + 1`), never appending SEP. This caused
completely wrong logits — the model was trained with SEP and its absence
shifted the attention patterns.

Fix: `seq_len = N + 2`, `ids[N+1] = SEP_id` (102 for BERT, 2 for RoBERTa).

Symptom: logits ~1-2 points off from reference, commas placed on wrong
words. Python F16 still predicted correctly — ruling out precision as
the cause. The diff-testing methodology (stage-by-stage comparison with
Python reference) quickly identified this: embeddings matched perfectly
(cos>0.999) but final logits diverged, pointing to a systematic error in
the self-attention computation that only manifests with a wrong sequence
structure.

**2. RoBERTa position ID offset**
RoBERTa position embeddings have `padding_idx=1`. Position 0 is for
`<pad>`, position 1 is zeroed out (the padding index), and actual content
starts at position 2. Our code used `pos_ids = [0, 1, 2, ...]` (BERT
convention) instead of `pos_ids = [2, 3, 4, ...]` (RoBERTa convention).

Fix: `pos[i] = i + padding_idx + 1` when `is_sentencepiece = true`.

Symptom: logits completely wrong (class 0 predicted for all tokens).
Diagnosed by comparing embedding output at position 15 — the values
were off because wrong position embeddings were summed.

**3. SentencePiece subtoken counting mismatch**
The text reconstruction code re-tokenizes each word to count how many
subtokens it consumed, mapping prediction indices back to words.
For SentencePiece, words are prefixed with `▁` (U+2581), not `##`.
The code was using WordPiece `##`-prefix matching for SentencePiece
tokens, causing wrong subtoken counts and shifted punctuation placement.

Fix: Separate SentencePiece path that prefixes with `▁` and does
greedy longest-match in the SentencePiece vocab.

Symptom: comma placed on "can" instead of "you" — the subtoken count
for "americans" (split into ["▁american", "s"] = 2 tokens) was counted
as 1 with the WordPiece path, shifting all subsequent predictions by 1.

**4. Chinese full-width punctuation for English text**
FireRedPunc was trained on Chinese data and outputs full-width marks
(`，` `。` `？` `！`) even for English input.

Fix: Auto-detect Latin script (count Latin vs CJK characters), map
full-width to ASCII when Latin dominates. Simple 4-replacement post-step.

### Methodology lesson reinforced

The user correctly pushed back when I blamed "F16 precision loss" for wrong
punctuation placement. The actual bug (missing SEP token) was a computation
error, not a precision issue. **Python F16 still predicted correctly** —
this ruled out precision as the root cause.

The diff-testing protocol worked exactly as designed:
1. Dump Python reference (logits, embeddings, per-layer outputs)
2. Dump C++ intermediates at the same positions
3. Compare cosine similarity at each stage
4. Embeddings matched (cos>0.999) → bug is after embeddings
5. Final logits diverged (cos~0.93) → systematic error in transformer
6. Traced to missing SEP token in input construction

Key principle: **when Python F16 works but C++ F16 doesn't, it's NOT a
precision issue.** Look for structural bugs (wrong input construction,
missing tokens, wrong tensor shapes).

### Quantization notes

| Model | F16 | Q8_0 | Q4_K | Accuracy |
|---|---|---|---|---|
| FireRedPunc | 195 MB | 104 MB | 56 MB | Q8_0 = F16 exact; Q4_K drops some commas |
| fullstop-punc | 1.6 GB | 572 MB | 254 MB | All quants identical on JFK test |

FireRedPunc Q4_K is more sensitive because BERT-base (12L, d=768) has
less redundancy than XLM-RoBERTa-large (24L, d=1024). Recommend Q8_0
for FireRedPunc, Q4_K for fullstop-punc.

### Progressive SRT output (issue #24)

Non-whisper backends buffered all segments before printing. Added
`--flush-after N` flag: when N=1, each SRT entry is flushed to stdout
as soon as its VAD slice finishes transcription. Post-processing (punc
model, punctuation stripping) runs per-slice.

Limitation: diarization needs full segment context — skip when
`--flush-after` is set. Word-level alignment (`-am`) works per-slice.

### Session API expansion

Added 5 missing backends to the C-ABI session API: glm-asr, kyutai-stt,
firered-asr, moonshine, omniasr. Pattern: `#ifdef CA_HAVE_*` guards in
`crispasr_c_api.cpp` for open/transcribe/close. All backends now
reachable from Python (`crispasr.Session`), Rust (`crispasr::Session`),
and Dart (`CrispasrSession`).
