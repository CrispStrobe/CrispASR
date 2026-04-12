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
