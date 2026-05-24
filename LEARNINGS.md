# CrispASR — Technical learnings

Distilled from months of porting eight ASR architectures into one ggml
codebase. Nothing here is breaking news; everything here is something
we'd have saved days if we'd known up front.

If a lesson is still "live" (affects current work), it's linked from
`TODO.md`. If it's historical (a bug we already fixed), it's linked from
`HISTORY.md`.

---

## VAD + chunking

### Independent-chunk TDT decode loses interior content, not just boundaries (issue #89, May 2026)

The 30 s auto-chunk fallback (`kLongAudioFallbackChunkSeconds`) prevents
the catastrophic zero-output failure on long audio (z-norm drift causes
the TDT decoder to emit blanks past ~60 s), but the auto path without
VAD tops out at ~82 % coverage. The missing content is **interior to
each chunk**, not at chunk boundaries.

Exhaustive sweep on the issue #89 reporter's 60 s Japanese audio
(parakeet-tdt-0.6b-ja, CPU):

| chunk | overlap | chars | coverage% | max_gap |
|-------|---------|-------|-----------|---------|
| 15s   | 3s      | 203   | 75.6%     | 8.5s    |
| 15s   | 5s      | 278   | 70.9%     | 5.3s    |
| 15s   | 8s      | 229   | 61.9%     | 12.2s   |
| 20s   | 3s      | 278   | 82.4%     | 3.4s    |  ← best chunk config
| 20s   | 5s      | 246   | 66.5%     | 10.0s   |
| 20s   | 8s      | 220   | 56.0%     | 12.1s   |
| 25s   | 3s      | 183   | 54.0%     | 15.1s   |
| 25s   | 5s      | 234   | 70.8%     | 12.4s   |
| 25s   | 8s      | 246   | 70.3%     | 9.8s    |
| 30s   | 3s      | 195   | 59.7%     | 19.4s   |
| 60s   | 0s      | 294   | 99.5%     | 0s      |  ← single pass (breaks on Vulkan/AMD)
| VAD   | silero  | 281   | 93.1%     | 3.7s    |

Counterintuitive findings:

1. **More overlap hurts.** 8 s overlap is worse than 3 s for *every*
   chunk size. Larger overlap extends the z-norm window, shifting feature
   statistics further from training distribution, and the boundary
   trimming removes more content that legitimately belongs to the chunk.

2. **The problem is decoder cold-start, not boundary stitching.** Each
   chunk reinitialises the TDT LSTM predictor to SOS. The decoder needs
   several seconds to "warm up" — emitting blanks or very sparse tokens
   while the LSTM state converges. This eats 5-20 s of interior content
   per chunk regardless of overlap.

3. **NeMo doesn't have this problem** because `FrameBatchASR` /
   `BatchedFrameASRTDT` keep the LSTM state between 1.6-4 s frame steps.
   The decoder never cold-starts after the first frame.

**Root cause:** CrispASR treats each chunk as an independent utterance
(`lstm_init_state → SOS`). NeMo treats them as a continuous stream with
shared decoder state. Fix: PLAN #104 (stateful frame-streaming TDT
decode).

**Practical recommendation until #104 ships:** use `--vad` (93 % coverage)
for long Japanese audio. The auto path is a safety net against zero
output, not a quality-optimal path.

---

### Overlap-save context is for *chunks*, not for VAD slices (issue #114)

`cad4c28a feat(#89): overlap-save chunking with --chunk-overlap flag`
added a ± `chunk_overlap_seconds` (default 3.0) neighbour-audio
extension to every per-slice transcribe call when `slices.size() >
1`. That gate is correct for explicit `--chunk-seconds N` runs, where
each cut falls in the middle of continuous speech and the encoder
genuinely needs context to span the boundary. It is *wrong* for VAD-
derived multi-slice runs: VAD slices are separated by silence, so
there is no boundary signal to recover, and pulling 3 s of the next
utterance into the current encoder context shifts the per-feature
z-norm statistics + bidirectional self-attention enough to push the
TDT decoder onto a different (worse) token path. The visible
regression on parakeet-tdt-0.6b-ja was kanji compounds collapsing to
bare hiragana plus entire short slices being dropped.

The crispasr-diff harness did NOT catch this — the cos at mel /
encoder stayed at 1.0 / 0.999994 on the single-utterance baseball
fixture, because the bug only manifests once multiple VAD slices
sit in the same long-form audio with realistic inter-slice silence.

Fix: gate overlap-save on `effective_chunk_seconds > 0`. The gate
lives in `examples/cli/crispasr_chunk_context_gate.h` so a unit test
can pin the invariant without spinning up a model.

Signature of the bug: per-slice transcripts that look "almost
right" but consistently pick simpler (more conservative) tokens at
the same boundaries — kanji → hiragana, multi-word phrases →
chopped syllables. Single-shot diff against PyTorch reference looks
fine because the diff harness only feeds one continuous segment.

---

## mel / preprocessor

### STFT frame count: NeMo `feat_len` vs raw STFT output

NeMo's `AudioToMelSpectrogramPreprocessor` computes the STFT over the
full center-padded signal, producing `T = floor((n+pad-n_fft)/hop)+1`
frames. But the returned `feat_len` is `floor(n_samples/hop)`, which
is one less when `n_samples` is an exact multiple of `hop_length`.
Frames beyond `feat_len` are treated as padding (zeroed in the output
tensor). The per-feature z-normalization runs over all `T` frames
(including the padding one), but the padding frame's z-score ends up
zero because it equals the mean of a near-constant band.

The C++ `core_mel::compute` produces the raw `T` frames and normalizes
all of them. The boundary frame is NOT zero — it has valid but
edge-contaminated STFT content, so after z-norm its values diverge by
±4 z-score units from the NeMo reference.

Fix: set `p.drop_last_frame = true` for all backends that use NeMo-
style center-padded STFT (canary, parakeet, cohere). This drops the
final frame, matching NeMo's `feat_len`.

Signature: mel cos_min ≈ 0.94 but cos_mean ≈ 0.999; exactly one
frame dominates `max_abs`; the worst frame is always the last.

### Dither must be disabled for deterministic reference dumps

NeMo's default `dither: 1e-5` adds per-sample Gaussian noise. Tiny
for real speech, but non-deterministic — each dump produces slightly
different mel values. Disable in the reference backend with
`model.preprocessor.featurizer.dither = 0.0` before running the
forward pass.

### Silence is a degenerate test input for per-feature-z normalization

All-zero audio → constant log-mel → per-feature z-norm divides 0/ε →
all-zero mel. NeMo produces a small negative constant (≈ −0.16) for
the same input because batch-norm running stats or ε handling differ.
The C++ producing all-zero mel is mathematically correct but creates
a misleading diff: cos_min=1.0 (zero vs constant is handled as
"perfect" by the cosine comparison's zero-guard). Always test with
real speech.

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

2. **CTC blank = `token 0 (<s>)`**: In fairseq2, the BOS token serves as CTC blank.
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

---

## TTS / Vocoder (Chatterbox HiFTGenerator)

### iSTFT data access must respect ggml's ne[0]-fast layout

The single bug that took the Chatterbox vocoder from "Oh." to "Hello
world." was a **transposed data read in the iSTFT**:

```cpp
// WRONG: treats buffer as (T_slow, C_fast) row-major
float raw_mag = stft_data[frame * C_stft + f];

// CORRECT: ggml stores ne[0]=T (fast), ne[1]=C (slow) → data[c * T + t]
float raw_mag = stft_data[f * T_stft + frame];
```

The ggml graph produces the correct STFT tensor (cos=1.0 vs Python
reference at every stage up to and including conv_post). But the CPU
iSTFT loop that converts STFT→waveform was reading the flat float buffer
with frequency and time axes swapped. This mixed time-step values into
frequency bins, producing a garbled waveform that still sounded
speech-like but was unintelligible to ASR.

**Signature of the bug**: the ggml graph stages all match the Python
reference, the STFT range/RMS look reasonable, but the audio transcript
is wrong. Always check that CPU post-processing code uses the same
`data[c * T + t]` indexing as ggml tensors — don't assume row-major
`data[t * C + c]`.

### Missing ReflectionPad1d at last upsample stage

Python's HiFTGenerator applies `ReflectionPad1d((1, 0))` inside the
upsample loop at `i == num_upsamples - 1` (the last stage), before
source fusion and resblocks. This pads 1 sample on the left by
reflecting `x[:, :, 1]`. Without it, conv_post is shifted by one
timestep and the STFT output has a cos_min ≈ 0.98 against the reference
(all prior stages at 1.0).

Implementation in ggml:
```cpp
ggml_tensor* pad = ggml_view_2d(ctx, x, 1, C, x->nb[1], 1 * x->nb[0]);
pad = ggml_cont(ctx, pad);
x = ggml_concat(ctx, pad, x, 0);  // concat along time axis (ne[0])
```

### Source fusion biases contaminate zero-input diff tests

When diff-testing the vocoder against a Python reference captured
**without** source fusion, don't just zero the source STFT input — skip
the source fusion graph path entirely. Even with all-zero input,
`Conv1d(zeros) = bias ≠ 0`, and those biases propagate through the
source resblocks and add a non-trivial signal to x. In our case this
dropped per-stage cosine from 1.0 to 0.92, masking the real bug
location until source fusion was fully disabled.

### Per-stage diff protocol for vocoders

The crispasr-diff protocol (dump Python activations to GGUF, compare C++
stage-by-stage) works for vocoders just as well as for encoders. The
stages to capture for HiFTGenerator:

```
voc_conv_pre → voc_ups_0 → voc_rb_0 → voc_ups_1 → voc_rb_1 → voc_ups_2 → voc_rb_2 → voc_conv_post
```

Mark each with `ggml_set_name` + `ggml_set_output`, read back via
`ggml_graph_get_tensor` after compute. Note the ggml↔GGUF layout
mismatch: C++ tensors have ne[0]=T (fast), but Python GGUF writers
may store ne[0]=C (fast) depending on how the numpy array was shaped.
Always verify layout by printing the first few values from both sides
before trusting cosine metrics.

### ggml conv_1d, Snake activation, and ResBlock structure are correct

After extensive testing: `ggml_conv_1d` with dilation handles all
resblock convolutions correctly. The Snake activation `x + sin²(αx)/α`
via `ggml_mul → ggml_sin → ggml_mul → ggml_div → ggml_add` matches
PyTorch to 6 significant figures. The 3-independent-resblock-then-average
pattern works. The bugs were downstream of the ggml graph, not in it.

## GitHub Actions workflow triggers — `master` → `main` rename gotcha (May 2026)

After renaming the default branch from `master` to `main`, audit
**every** workflow's trigger config for stale `branches: master`
references. A workflow gated on a non-existent branch never fires
on push events. GitHub Actions does not warn — the workflow file
is valid, runs are just never created. Three CrispASR workflows
(`bindings-ruby.yml`, `build.yml`, `examples-wasm.yml`) had this
misconfig for months post-rename. The full per-OS build matrix
(iOS / macOS / Windows MSVC / Windows MSYS2 / Linux Vulkan /
Android NDK) was therefore not validating any push to `main` —
only running on PRs (different trigger) and on tag pushes
(release-only).

### Why it took so long to notice

- PRs always triggered the workflows (different trigger config),
  so PR review felt fine.
- Tags (`tags: 'v*'`) still triggered, so v0.5.x releases ran the
  matrix — but only at release time, not per-merge.
- `gh run list --workflow=build.yml` shows runs starting on PR
  opens and at tag pushes, looking superficially complete. The
  missing signal — "no run on commit X to main" — only becomes
  apparent if you specifically query for it.

### How a pre-existing failure surfaces as a "PR failure"

When `Bindings Tests (Ruby)` finally ran on a PR (#57,
`feat/codec-gpu-env`), it failed at the `cmake-targets` step. The
natural reading is "the PR broke something." The actual cause: that
PR was the first PR in months to fire a workflow that had been
silently dead on main. The breakage was in main, not the PR. We
proved this by fixing the trigger and pushing the trigger fix
itself to main — the resulting clean-main run reproduced the
identical `Makefile:171: Error 2`.

### Audit recipe

```bash
for f in .github/workflows/*.yml; do
    echo "== $(basename "$f") =="
    grep -A1 "branches:" "$f" | head -3
done | grep -B1 master
```

Anything that prints `master` after a `branches:` line is a stale
trigger. Fix `master` → `main` (or list both: `branches: [main, master]`).

### Incident-response value

Land the trigger fix in a single small commit (`b553546` for us).
That gives a clean before/after demarcation: every run before this
commit on main is suspect for "did it actually run?", and every
run after this commit is the new ground truth. Push the trigger
fix on its own — its own push fires the corrected workflow on the
new SHA, so any accumulated breakage surfaces on the same commit.

### What surfaced when the triggers came back

Three real breakages had accumulated under the silent triggers:

1. **build.yml ubuntu-22-clang ×4** — clang-14 (Ubuntu 22.04 baseline)
   refuses to capture C++17 structured-binding names in a `[&]`
   lambda. gcc accepts it as an extension; clang rejects with
   "reference to local binding 'qw' declared in enclosing function."
   Fixed in C++20 (P1091) but our `cxx_std_11`-effective-C++17
   baseline doesn't get that. **Fix:** plain locals + `const auto&`
   aliases instead of `auto [a, b] = ...` whenever the names will be
   captured by a lambda. Commit `053f41f`.

2. **build.yml ios-xcode-build** — `build-xcframework.sh` was
   hand-listing only 5 ggml libs + libcrispasr.a in the libs[] array
   passed to `libtool -static -o combined.a`. CrispASR has 25+
   STATIC backend libs that crispasr.a publicly depends on but does
   NOT statically embed (parakeet, canary, voxtral, qwen3_asr,
   wav2vec2-ggml, glm-asr, …). Every backend referenced from
   `crispasr_c_api.cpp` would surface as "Undefined symbols"
   eventually; wav2vec2_load was just the first. **Fix:** glob
   `build_dir/src/*/release_dir/*.a` and append. Future-proof
   against new backends. Commit `ee580f8`.

3. **bindings-ruby.yml — `whisper` is an ALIAS target**. The Ruby
   gem's `extconf.rb` ran `cmake --build build --target common
   whisper` for years. After the upstream rename to crispasr,
   `whisper` became an `add_library(whisper ALIAS crispasr)` for
   back-compat in callers that hardcoded the name. **ALIAS targets
   are linkable but not directly buildable** — `cmake --build
   --target whisper` fails with "make: *** No rule to make target
   'whisper'." (CMake forwards the target name to the underlying
   build tool, which only sees the real `crispasr` rule.) **Fix:**
   point the gem at the real target: `--target common crispasr`.
   Commit `476c655`.

The first two were pure code drift (compiler versions / new backend
sources added without updating the xcframework lib list). The
third was a rename gotcha. None would have made it past CI if
build.yml + bindings-ruby.yml had been firing on main pushes.

### Workflow auto-discovery — `actions/configure-pages` failure mode

The `Examples WASM` workflow's whole purpose is `actions/upload-pages-artifact`
+ `actions/deploy-pages` to a GitHub Pages site. If Pages isn't
enabled on the repo (Settings → Pages → "GitHub Actions" source),
`actions/configure-pages` fails with:

```
##[error]Get Pages site failed. Please verify that the repository
has Pages enabled... HttpError: Not Found
```

The build matrix in `build.yml` already validates that the WASM
examples *compile* — that's the per-merge signal we want. The
deploy is incidental and shouldn't gate main-push CI. Default the
trigger to `workflow_dispatch:` only and re-add `push: branches:
[main]` together with enabling Pages. Commit `476c655`.

## Audit script ≠ behavior test (2026-05-04)

Two complementary checks, easy to confuse:

- **`tools/audit-backend-capabilities.py`** parses the binary's
  `--list-backends-json` and the test script's `Backend(...,
  capabilities=(...))` tuples, and reports drift between *what the
  binary claims* and *what the test script claims to cover*. It
  cannot detect a backend that mis-declares a cap that the test
  also dishonestly lists — both ends agree, audit says clean.
- **`tools/test-all-backends.py --profile feature`** actually runs
  each cap. This is what catches a cap declared but not implemented
  (omniasr's `CAP_PUNCTUATION_TOGGLE` shipping with a CTC vocab that
  has no punctuation; parakeet declaring `CAP_LANGUAGE_DETECT` with
  no stderr LID line in its native path).

Run the audit before pushing — it's seconds. Run the feature suite
periodically and after backend changes — it's slow but the only way
to catch *behaviour* drift.

When both directions of the audit agree but feature-suite tests
fail, the failure is one of:
1. The runner is under-invoking (missing CLI flag like `-dl` or
   `-am`). The runner must be widened — usually by inspecting the
   binary's caps via `--list-backends-json` and branching.
2. The backend declares a cap it doesn't actually deliver. The cap
   must be dropped from the `.cpp` backend's `capabilities()`
   override AND from the test tuple in lockstep, otherwise the
   audit will start reporting drift.

## libcrispasr.a + libcommon.a both define stb_vorbis / miniaudio impl (Linux ld dies)

`src/crispasr_audio.cpp` and `examples/common-crispasr.cpp` both
translation-unit-include `stb_vorbis.c` after `#undef
STB_VORBIS_HEADER_ONLY` and both define `MINIAUDIO_IMPLEMENTATION`.
This means *both* static libraries (libcrispasr.a, libcommon.a)
ship the full set of stb_vorbis + miniaudio symbols.

- **Apple ld** silently picks the first definition. macOS builds
  passed without complaint for years.
- **GNU ld** (Linux) errors with `multiple definition of …`. Comes
  up the moment a downstream consumer links *both* archives — which
  the Ruby binding does via `bindings/ruby/ext/dependencies.rb`'s
  full-graph topological link.
- **Main CI build doesn't trip** because libcrispasr is compiled as
  a shared lib (`libcrispasr.so`) whose symbols are resolved at
  build time, so the executable only sees one copy of stb_vorbis
  (from libcommon.a).

The proper fix is to extract the impl into its own dedicated
translation unit linked exactly once — but that's a refactor with
ripples (some `tests/` and `examples/crispasr-quantize/` link
libcommon alone and would need the new lib added). For the Ruby
binding specifically, `-Wl,--allow-multiple-definition` on Linux is
the minimal workaround that mirrors macOS ld's behaviour.

Also: when CMake's `CRISPASR_STANDALONE` is ON (default for any
consumer that points `-S sources` at the repo root, including the
Ruby binding's vendored copy), `CRISPASR_BUILD_TESTS` defaults ON,
which pulls Catch2 into the configured target graph. The Ruby
binding's dependency walker then lists `libCatch2*.a` in
`$LOCAL_LIBS` even though `--target common crispasr` never built
them, and the link fails with `cannot find libCatch2WithMain.a`.
Fix: pass `-D CRISPASR_BUILD_TESTS=OFF -D CRISPASR_BUILD_EXAMPLES=OFF
-D CRISPASR_BUILD_SERVER=OFF` to the Ruby binding's cmake config.
Both fixes shipped together in `bindings/ruby/ext/extconf.rb`.

## Chatterbox-Turbo conformer encoder — ggml layout traps (May 2026)

Five bugs, all in `chatterbox_s3gen.cpp`, all discoverable only via
element-wise diff against PyTorch reference (the crispasr-diff harness).

### 1. EspnetRelPositionalEncoding ordering

Python builds pe_positive reversed + pe_negative, giving positions from
+(T-1) to -(T-1). A natural C++ loop from p=0 with `pos = p - (T-1)`
gives -(T-1) to +(T-1) — **exactly backwards**. The sine components are
negated which changes Q·P dot products after rel_shift. Fix: `pos = (T-1) - p`.

This was invisible in RMS comparison (reversed positions have identical
L2 norms) and only detectable via element-wise matrix_bd comparison.

### 2. pos_bias_u/v: reshape vs transpose

`pos_bias_u` stored in GGUF as ne=(d_k, n_heads). To broadcast-add to
Q with shape (d_k, T, n_heads), just reshape to (d_k, 1, n_heads). A
transpose before reshape scrambles head/dim indices — the (d, 1, h)
element gets a bias value from a different head. Again invisible in RMS
comparison, reduced all attention scores to ~70% of correct magnitude.

### 3. ggml reshape ≠ PyTorch view for multi-head attention output

**The most expensive bug.** After attention weighted sum, the output has
shape (d_k, T, n_heads). Python's `transpose(1,2).view(B, T, D)` gives
head-concatenated layout: [h0_d0..h0_d63, h1_d0..h7_d63] per time step.
But ggml `reshape_2d(D, T)` just reinterprets the flat buffer, giving
interleaved head/time indices. The output projection `lo.weight` was
trained expecting concatenated heads, so the wrong layout causes 8%
signal reduction — compounding across 10 blocks to 24% total loss.

Fix: add `ggml_permute(0, 2, 1, 3)` before reshape to move heads
before time: (d_k, T, H) → (d_k, H, T) → reshape → (D, T).

**Rule:** whenever converting from ggml's (d_k, T, H) back to
PyTorch's (T, D) layout, you MUST permute heads-before-time first.
ggml reshape is NOT equivalent to PyTorch view when the intermediate
dimensions have different orderings.

### 4. F0 predictor dequantization

The F0 predictor reads weight tensors manually with
`ggml_backend_tensor_get`. When tensors are quantized (Q8_0/Q4_K),
`ggml_nbytes(tensor)` < `n_elem * sizeof(float)`, causing out-of-bounds
reads. Fix: read `ggml_nbytes(tensor)` bytes, then use
`ggml_get_type_traits(type)->to_float()` to dequantize. This applies to
ALL manual tensor reads in CPU-side code (F0 predictor, SourceModule
linear, speaker embedding projection).

## Chatterbox repaired GGUF split — stage-specific regen (May 2026)

The repaired `*-regen.gguf` artifacts are intentionally asymmetric:
base Chatterbox regenerated T3, while Chatterbox Turbo regenerated
S3Gen.

- `chatterbox-t3-f16-regen.gguf` fixes the base T3 GGUF export. The
  old artifact only carried the legacy character-token vocabulary and
  missed the real HF BPE tokenizer tokens/merges, so C++ fed a different
  text-token sequence than Python.
- `chatterbox-turbo-s3gen-f16-regen.gguf` fixes the Turbo companion
  stage. Turbo T3 uses the GPT-2-style path and was not the artifact
  being repaired; the downstream S3Gen/vocoder GGUF needed rebuilding
  after the S3Gen parity fixes.
- Turbo T3 still needs quant coverage even though it did not need
  regeneration. Publish `chatterbox-turbo-t3-q8_0.gguf` and
  `chatterbox-turbo-t3-q4_k.gguf` from the canonical F16 so users can
  select matching T3/S3Gen quants without relying on local `-regen`
  names.
- `-regen` is a local repair marker, not a registry contract. HF uploads
  should publish the repaired bytes under canonical filenames so
  auto-download and auto-resolve keep working without special cases.
- Do not infer a general rule that base always needs T3 regeneration or
  Turbo always needs S3Gen regeneration. The regenerated file is the
  stage whose GGUF export/runtime contract was wrong.

Current debug boundary: base T3 tokenizer, conditioning, prefill, CFG,
step-0 logits, and forced step-1 logits match Python. S3Gen replay and
HiFT final reconstruction also match when fed reference tensors. The
remaining token-stream drift is sampler parity with CPU
`torch.multinomial`, not model math.

## Chatterbox base T3 sampler parity — Gumbel-max → torch.multinomial (May 2026)

Followup to the entry above: ported the base T3 sampler at
`src/chatterbox.cpp:541` from a Gumbel-max trick (one MT19937 32-bit
uniform per token, `argmax(p_i / -log(1-u_i))`) to a faithful CPU
`torch.multinomial(probs, num_samples=1)` (cumsum + normalize +
snap last bucket to 1.0 + binary search on a single 53-bit double
uniform from two MT19937 draws combined as `random64 = (r1<<32)|r2`,
masked to 53 bits, divided by 2^53). The recipe matches
`aten/src/ATen/native/cpu/MultinomialKernel.cpp` and
`aten/src/ATen/core/DistributionsHelper.h::uniform_real_distribution`
exactly. Closes the residual "gibberish at start / blurred words"
audible drift on long prompts (issue #76).

Direct A/B on `"…I want to move like a titan at dawn."` with
`chatterbox-t3-f16-regen.gguf` + `chatterbox-s3gen-q8_0.gguf`:

| Sampler                   | ASR roundtrip transcript                   |
|---------------------------|--------------------------------------------|
| Gumbel-max (`p / -log1p(-u)`) | "…like a **Titanette Dawn**" (mispronounced) |
| `torch.multinomial` faithful  | "…like a **Titan at dawn**" (matches input)  |

Why Gumbel-max was empirically close but not equivalent here:
fp32 `-log1p(-u)` with a 32-bit `u` discretizes the right tail to
`u ∈ {0, 2^-32, 2·2^-32, …}` so the smallest non-1 `u` gives
`e ≈ 22 = 32·log(2)`. PyTorch's 53-bit-precision uniform reaches
`e ≈ 37` in the same tail. That biases Gumbel-max toward sampling
further from the mode for any specific (logit, draw) — within the
sampling distribution's natural variance per-token, but compounding
across 60+ AR steps into specific token-id divergence at points
where the post-temperature-post-min-p probability mass is thin and
spread (the chatterbox speech-token vocab is 8194-wide and post-CFG
multimodal, so this is common).

Don't try to match Python's RNG state — `torch.manual_seed(...)`
seeds CPUGeneratorImpl through a different path than our manual
MT19937, and chatterbox's reference dump doesn't seed at all. The
fix is statistical equivalence, not bit-exact reproducibility.

## Chatterbox HiFT vocoder parity nits (May 2026)

Two parity bugs found while diff-testing the vocoder against Python's
HiFTGenerator with reference mel + reference source-stft fed in:

### 1. Pre-conv_post LeakyReLU slope is `0.01`, not `0.1`

`/private/tmp/resemble-chatterbox/src/chatterbox/models/s3gen/hifigan.py:418`
calls `F.leaky_relu(x, self.lrelu_slope)` with `lrelu_slope=0.1` —
this is the in-loop activation. But line 437 — the LAST activation
before `conv_post` — calls `F.leaky_relu(x)` with **no slope
argument**, so PyTorch's default `negative_slope=0.01` applies.
Easy to miss when porting because everything else uses 0.1. Fixed
at `src/chatterbox_s3gen.cpp:1998`. `voc_conv_post` cos_mean
improved 0.985 → 0.993 against the per-stage reference.

### 2. Snake activation needs `1/(α + 1e-9)`, not `1/α`

`chatterbox/models/s3gen/transformer/activation.py:71,82`:

```python
self.no_div_by_zero = 0.000000001
…
x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
```

Implemented in C++ via `ggml_scale_bias(ctx, alpha, 1.0, 1e-9)` →
`ggml_div(sin², α_safe)`. Doesn't help on the current released
weights (min α observed in `s3.v.rb.*` is 0.016 ≫ 1e-9), but it
matches Python bit-for-bit and defends against future fine-tunes
that drift α further toward zero. Three sites in
`chatterbox_s3gen.cpp` (rb-snake1, rb-snake2, srb-snake1, srb-snake2).

### Residual cumulative drift — fp accumulation order, not algorithmic

After both fixes, `hift_pcm(ref_mel)` still sits at cos_min ≈ 0.89
even though every individual stage between mel and conv_post passes
the cos_mean ≥ 0.95 threshold. Per-row dump of `voc_ups_2` shows the
drift is concentrated in a small cluster of time-steps (T ≈ 2900 of
4801, mapping back to mel-index 24-25 of 40) with cos as low as 0.16,
amplified across stages by Snake's `1/α` factor at channels with the
smallest trained `α` (e.g. `s3.v.rb.1.a1.0.alpha` has elements at
0.016 and 0.022). This is fp accumulation-order divergence between
ggml's conv1d/convtranspose1d kernels and torch's; it does NOT
affect intelligibility (ASR roundtrip on long prompts transcribes
verbatim). Reaching cos_min = 1.0 here would require pinning ggml's
reduction order to torch's — high cost, low audible payoff.

## Chatterbox voice cloning — bake to GGUF, load via `--voice` (May 2026)

Voice cloning uses the same baker-→-voice-GGUF pattern as vibevoice
(`models/convert-vibevoice-voice-to-gguf.py`):
`models/bake-chatterbox-voice-from-wav.py` runs upstream
`ChatterboxTTS.prepare_conditionals(wav)` and writes a 150-200 KB
GGUF using **the same tensor names the C++ runtime already accepts
for the built-in voice** (see `convert-chatterbox-to-gguf.py:521-548`):

```
conds.t3.speaker_emb           f32  (1, 256)
conds.t3.speech_prompt_tokens  i32  (T_prompt,)
conds.gen.prompt_token         i32  (T_speech,)
conds.gen.prompt_feat          f32  (T_mel, 80)
conds.gen.embedding            f32  (1, 192)
chatterbox.conds.emotion_adv          f32 metadata
chatterbox.conds.gen_prompt_token_len u32 metadata
```

C++ side: `chatterbox_set_voice_from_wav()` dispatches on extension —
`.gguf` paths route through `chatterbox_load_voice_gguf()`, which
loads via `core_gguf::load_weights` into a separate
`voice_ctx_w`/`voice_buf_w` and rebinds `ctx->conds.*` tensor pointers.
The original baked-in default-voice tensors stay allocated in
`ctx_w`/`buf_w` (unreferenced after rebind, freed only at context
shutdown). `.wav` paths print a hint pointing at the baker —
in-process WAV → cond extraction (VE LSTM, CAMPPlus TDNN,
S3Tokenizer encoder + quantizer) is a separate refactor; weights
ARE in the s3gen GGUF already (`s3.se.xv` 755 tensors, `s3.se.head`,
`s3.tok.{enc,quant,encoder,_mel_filters}` 103 tensors), but
implementing the forward passes is multi-day work.

CLI plumbing: `--voice <path.gguf>` is wired in
`crispasr_backend_chatterbox.cpp::synthesise()` with a
per-call `last_voice_key_` cache so server callers can switch voice
between requests. Mirrors the vibevoice posture (refuses to
silently fall back to the default voice on a load failure).

### The build-target trap that masked it

`cmake --build build-ninja-compile --target crispasr` only relinks
the **shared library** `libcrispasr.dylib` and stops there — the CLI
executable lives in target `crispasr-cli`. So edits to
`examples/cli/crispasr_backend_*.cpp` need an explicit
`--target crispasr-cli` rebuild to land in `bin/crispasr`. We hit
this on the first voice-clone smoke test: the synth log showed no
"voice loaded" line and the cloned wav md5'd different from default
purely because of post-build I/O timestamps in the wav header
(audio bytes were identical). Always check that
`build-ninja-compile/examples/cli/CMakeFiles/crispasr-cli.dir/<file>.o`
has a fresher mtime than the .cpp before declaring a CLI-side
behaviour test "passed".

## Chatterbox VoiceEncoder native port — module 2 of voice cloning (May 2026)

`src/chatterbox_ve.cpp` ports the upstream
`chatterbox.models.voice_encoder.embeds_from_wavs` so `--voice <ref>.wav`
works with no python at runtime. Module 2 only — modules 3
(S3Tokenizer for `gen.prompt_token` + `t3.speech_prompt_tokens`) and
4 (CAMPPlus for `gen.embedding`, plus the 24 kHz prompt mel for
`gen.prompt_feat`) are still pending. See
`src/chatterbox_ve.h` for the pipeline contract.

### Stuff that mattered

1. **`embeds_from_wavs` default rate is 1.3, NOT 0.5**. Reading the
   handover I assumed `overlap=0.5` → frame_step = 80, but the
   upstream wrapper sets `kwargs["rate"] = 1.3` if the caller didn't,
   and `get_frame_step` switches to `round((sr/rate)/partial_frames) =
   round(16000/1.3/160) = 77`. Mis-setting this to 80 would still
   produce a working embedding but with a different partition that
   diverges from upstream's per-partial layout.

2. **`mel_type='amp'` is no log step at all, not log on the amp**.
   The upstream `melspec.py` only calls `_amp_to_db` when
   `hp.mel_type == "db"`. With `'amp'` the LSTM consumes the raw
   mel-projected magnitude. Implemented as `core_mel::LogBase::None`
   (added in commit `957eb4ff`) and verified.

3. **`mel_power=2.0` means project `|X|^2`, not `|X|`**. Both
   `core_mel::SpecKind::Power` and the bare `Magnitude` layout were
   plausible; the config has `mel_power = 2.0` which is the upstream's
   knob to square the magnitudes before mel projection. Same as
   librosa default (`power=2.0`).

4. **The LSTM is small enough to run pure-CPU; ggml graph integration
   exploded on `buffer_id < 0`**. First attempt built per-partial
   ggml graphs that mixed model-buffer weights with on-the-fly
   `ggml_cast` and `ggml_view` ops, fed through `ctx->sched`. Hit
   `GGML_ASSERT(buffer_id >= 0)` in `ggml_gallocr_allocate_node` —
   the scheduler couldn't pick a backend for the cast/view chain
   when the source weights live in a buffer the scheduler hadn't
   ingested. Rewrote the LSTM as plain C++ float math
   (`lstm_unidir_cpu` in chatterbox_ve.cpp): read all VE weights to
   F32 with `read_tensor_f32` (mirrors the existing `tensor_get_f32`
   helper) and run the recurrence directly. Bit-equivalent to torch's
   nn.LSTM forward; ~1-2 s for 13 partials on M1 (one-shot voice
   clone — perf isn't a concern). For larger LSTMs that need GPU
   dispatch the right pattern is to allocate weights into the
   scheduler's own context, not borrow tensors from a foreign one.

5. **L2-norm via ggml is awkward; do it on host**. ggml has
   `ggml_rms_norm` (RMS, not L2 — divides by `sqrt(mean(x²)+eps)`)
   and no built-in L2. Trying to express `y/sqrt(sum(x²))` via
   `rms_norm` requires a `1/sqrt(N)` scale that's brittle to the
   eps. Read the (256,) embedding back from the graph and
   normalise on CPU — single divide, matches `torch.linalg.norm`
   exactly with no eps drama.

6. **Silence trim deferred — match the dump's audio path**.
   Upstream `embeds_from_wavs` does
   `librosa.effects.trim(top_db=20)` before mel. Porting
   librosa.feature.rms's center=True/pad_mode='constant' framing
   plus the dB threshold logic is fiddly; for module 2 we bypass
   trim in BOTH the C++ port and the python reference dump
   (`tools/reference_backends/chatterbox.py` calls
   `melspectrogram` directly instead of `embeds_from_wavs`). Result:
   bit-equivalent comparison and a clean 1.0 cosine on
   `ve_partial_emb` and `ve_speaker_emb`. For typical pre-trimmed
   reference WAVs this is a no-op; long silences would degrade
   the embedding until trim lands as a follow-up.

7. **`librosa.stft` with `n_fft=400` works fine through
   `core_fft::fft_radix2_wrapper`**. core_fft splits `400 → 200 → 100
   → 50 → 25 (odd)` and falls back to direct DFT at the odd remainder.
   No need for a Bluestein chirp-z. ~10K ops per frame; negligible.

8. **Periodic Hann window, not symmetric**. librosa's STFT default
   is `scipy.signal.windows.get_window('hann', N, fftbins=True)`,
   which is the periodic variant `0.5 * (1 - cos(2π i / N))` (i ∈
   [0, N)). Symmetric Hann uses `N-1` in the denominator and is
   off-by-one — would fail mel parity by ~1e-3.

9. **`cb_ve_model` lives in chatterbox_ve.h, not chatterbox.cpp's
   anonymous namespace**. The struct was originally a private
   detail in chatterbox.cpp; the cleanest separation for a new
   per-module .cpp is to lift it to the shared header so chatterbox.cpp
   keeps `bind_ve` and chatterbox_ve.cpp keeps the forward.

### Build target nits (still relevant)

CLI changes need `--target crispasr-cli`; library-only changes work
with `--target crispasr` (or `--target chatterbox` for the static
sublib). The diff harness needs `--target crispasr-diff`.

### Diff stages added

`tools/reference_backends/chatterbox.py` now captures:
- `ve_mel`           — (T, 40) raw-amp Slaney mel after the trim-bypass path
- `ve_partial_emb`   — (n_partials, 256) L2-normed per-partial embeddings
- `ve_speaker_emb`   — (1, 256) final speaker embedding (mean + L2)

C++ ABI hooks `chatterbox_dump_ve_*` in `chatterbox.h`. The
`crispasr-diff chatterbox` harness exits-codes against the same
0.95 cos_mean threshold as other chatterbox stages. On JFK 11s the
mel cos_mean is 0.999913 (cos_min 0.905 in low-energy frames where
fp32 directions are unstable; downstream LSTM filters this out and
ve_partial_emb / ve_speaker_emb are essentially bit-perfect).

## Chatterbox S3Tokenizer V2 native port — module 3 of voice cloning (May 2026)

`src/chatterbox_s3tok.{h,cpp}` ports `S3TokenizerV2.quantize` —
the FSMN-augmented Whisper-style encoder + FSQ codebook that
turns a 16 kHz reference WAV into 25 Hz speech tokens
(codebook size = 3⁸ = 6561). Lives next to chatterbox_ve (module 2)
and is bound on the s3gen sub-context's tensor map (the `s3.tok.*`
weights ride in the chatterbox-s3gen GGUF, not the T3 GGUF).

### Stuff that mattered

1. **The K projection has no bias**. S3Tokenizer follows Whisper's
   `MultiHeadAttention` where `query` and `value` are biased but
   `key` is `Linear(... bias=False)`. So
   `s3.tok.enc.b.<l>.attn.key.bias` is absent from the GGUF, and
   the bind step gets `nullptr`. First implementation called
   `ggml_add(K, b.attn_k_b)` unconditionally — segfault inside
   `ggml_add_impl`. Fix is the same pattern voxtral / qwen3 use:
   gate every Q/K/V/O bias add on a non-null pointer.

2. **FSMN side branch on V, computed BEFORE the attention head
   reshape**. The python's `FSMNMultiHeadAttention.qkv_attention`
   reshapes V to `(B, T, n_head, head_dim)` then immediately
   collapses it back to `(B, T, D)` for the depthwise Conv1d. So
   we can equivalently compute FSMN on the post-projection V
   tensor directly (still `(D, T)` in our ggml layout) — transpose
   to `(T, D)` for `ggml_conv_1d_dw`, run the depthwise k=31 conv,
   add the residual V back, transpose back to `(D, T)`. The
   attention output projection's result is then `+ fsmn_memory`
   before the residual add. `attn_fsmn_w` is `(31, 1, 1280)` F16 —
   ggml_conv_1d_dw consumes it as-is.

3. **Periodic Hann via `torch.hann_window`** — same as VE, but the
   STFT path here is post-log so a missing or off-by-one Hann
   shows up in `s3tok_log_mel`'s cosine before the encoder ever
   runs.

4. **`magnitudes = stft[..., :-1].abs()**2`** drops the last STFT
   frame. core_mel exposes this as `Params::drop_last_frame`, set
   true to match. On JFK 11 s @ 16 kHz this gives T=1100 (not 1101
   like VE's mel that keeps all frames).

5. **Conv1d strides are both 2**. AudioEncoderV2's `__init__`
   passes `stride=stride` (the constructor arg, set to 2 for the
   tokenizer) to conv1, and conv2 hardcodes `stride=2`. Total
   downsample is 4× → T mel frames at 100 Hz become T/4 tokens at
   25 Hz. For a 6 s prompt: 600 mel frames → 150 tokens, exactly
   the `speech_cond_prompt_len = 150` the t3 prompt expects.

6. **RoPE n_dims = head_dim, not n_state**. The model_v2
   `precompute_freqs_cis(64, 1024*2)` uses 64 (the head_dim) for
   the rotation dim, not 1280 (n_state). Pass `head_dim` (=64) as
   the `n_dims` arg to `ggml_rope_ext`. n_ctx_orig at 2048 covers
   the upstream max position (`1024 * 2`).

7. **NEOX RoPE matches the python's `apply_rotary_emb`**. The
   python's `cat((freqs_cis, freqs_cis), dim=-1)` doubling and the
   `(half_l, half_r)` split / `(-half_r, half_l)` rotation is the
   GPT-NeoX form: pairs `(i, i+head_dim/2)` rotated by `θ_i`.
   `GGML_ROPE_TYPE_NEOX` is the right enum.

8. **`s3tok_tokens` cosine is below 0.999 — by design**. FSQ is a
   discrete quantization: `tanh(h)*0.999 → round → +1`. Tiny float
   drift in the encoder pushes a few logits across the {-0.5, 0.5}
   rounding boundary, which flips a base-3 digit and changes the
   integer code. Cosine on the integer stream then reflects the
   per-token mismatch rate, not the underlying float drift. JFK
   11 s gives `cos_min=0.997`; the parity-quality metric is
   `s3tok_proj_down` (the pre-FSQ floats) which is comfortably
   `cos_mean=0.99993`.

### Voice clone wiring discipline

When wiring module 3 outputs into `chatterbox_set_voice_from_wav`
the .wav branch, **do NOT update `gen.prompt_token` alone** — it has
to stay in lockstep with `gen.prompt_feat` (the 24 kHz prompt mel)
and `gen.embedding` (the CAMPPlus x-vector). All three describe the
same reference audio for S3Gen's flow matcher; cross-mixing
prompt_token from the new ref with prompt_feat / embedding from the
default ref makes the vocoder collapse to ~0.0003 RMS silence
(verified on JFK clone). The right partial-clone state is:

  - Module 2 only:     speaker_emb NEW, all gen.* DEFAULT
  - Module 2+3:        speaker_emb + speech_prompt_tokens NEW (both
                       T3-side); all gen.* DEFAULT
  - Module 2+3+4:      everything NEW (atomic)

Module 3 only updates the T3 side (`speech_prompt_tokens`); the
S3Tokenizer's full-audio token stream is exposed via
`chatterbox_dump_s3tok_tokens` for parity testing but isn't
written into the runtime conds bundle yet.

### Diff stages

`tools/reference_backends/chatterbox.py`:
- `s3tok_log_mel`             — (128, T) log10 mel after clip-and-scale
- `s3tok_proj_down`           — (T_tok, 8) pre-FSQ projdown floats
- `s3tok_tokens`              — (T_tok,) full-audio int32 tokens
- `s3tok_speech_prompt_tokens` — (≤150,) first-6 s int32 tokens

C++ ABI hooks in `chatterbox.h`. Threshold and reporting style
mirror the existing chatterbox stages (cos_mean ≥ 0.95).

## Chatterbox CAMPPlus phase 1 — Kaldi fbank front-end (May 2026)

`src/core/kaldi_fbank.{h,cpp}` ports `torchaudio.compliance.kaldi.fbank()`
with default args (powey window, HTK mel scale, log power,
preemph=0.97, snip_edges=True, round_to_power_of_two=True). Promoted to
a shared core helper since multiple speaker / VAD models consume Kaldi
fbank — currently inline copies live in `firered_asr.cpp` and could be
deduped in a follow-up; the new helper takes an `int16_scale` knob to
cover the firered case where the trained CMVN expects int16-magnitude
features.

`src/chatterbox_campplus.{h,cpp}` is the chatterbox consumer (Module 4
of the native voice clone path). Phase 1 of this module ships the
fbank front-end + the per-utterance mean subtract that
`xvector.extract_feature` does. Phase 2 (the actual CAMPPlus TDNN
forward — FCM 2-D conv head + 12+24+16 dense TDNN layers + StatsPool +
1024→192 dense) is deferred to a follow-up commit since it needs ~50
BatchNorm-fold pairs and a non-trivial CAM seg_pooling op
(`F.avg_pool1d(k=100, s=100, ceil_mode=True)` with broadcast-back) and
is multi-hour focused work.

### Stuff that mattered (phase 1)

1. **Reuse, don't reinvent**. `firered_asr.cpp:compute_fbank` was
   already a faithful Kaldi fbank — same povey window, same HTK mel
   scale, same preemph-with-s[-1]=s[0] boundary, same FLT_EPSILON log
   floor. Lift to a parameterised core helper (raw vs int16 scaling),
   keep firered's inline copy unchanged for now to avoid churn.

2. **Kaldi pre-emphasis boundary**: Kaldi's `feature-window.cc` uses
   `s[-1] = s[0]` for the boundary, so the i=0 step becomes
   `s[0] -= preemph * s[0]` → `s[0] *= (1 - preemph)`. Must walk the
   array in REVERSE so each step reads the unmodified s[i-1].

3. **Kaldi mel scale = HTK** (`mel = 1127 * log(1 + hz/700)`), and
   filters are NOT Slaney-area-normalized — the bare triangles. This
   is what every Kaldi-trained model (including CAMPPlus) expects;
   feeding in a Slaney-normalized basis silently destroys downstream
   accuracy. The librosa default normalization would have been wrong
   here.

4. **`round_to_power_of_two=True` is just zero-padding the windowed
   frame to next_pow2(win)**. For win=400, n_fft=512. The energy at
   the original 400 samples is what matters; the 112 zero-padded tail
   contributes nothing to the FFT bins it produces.

5. **`int16_scale=False` for CAMPPlus**. Despite torchaudio's docstring
   saying "Kaldi typically uses 16-bit audio integers", chatterbox
   feeds CAMPPlus float-[-1, 1] audio directly (see `xvector.py`'s
   `Kaldi.fbank(au.unsqueeze(0), ...)` — `au` is normalized float).
   The trained model adapted to that scaling. Setting `int16_scale`
   on would shift the log floor by a constant +log(32768²) ≈ 20.8 per
   bin and the CAMPPlus TDNN's BN running stats would no longer fit.

6. **`use_energy=False` is the default**. Kaldi's convention is to
   either include log-energy as the first bin (use_energy=True) or
   keep just the mel bins. CAMPPlus's call doesn't pass use_energy →
   defaults to False, so output is exactly num_mel_bins=80 columns.

### Diff parity (phase 1)

`tools/reference_backends/chatterbox.py` captures `campplus_fbank` via
`torchaudio.compliance.kaldi.fbank` + the per-utterance mean subtract.
`crispasr-diff chatterbox` on JFK 11 s gives `cos_min=0.999994
cos_mean=0.999999` against torchaudio — fp32 rounding noise tight.

## Chatterbox CAMPPlus phase 2 — TDNN forward (May 2026)

Phase 2 of module 4 ports the CAMPPlus speaker encoder forward to native
C++ — the FCM 2-D conv head + 12+24+16 dense TDNN layers + StatsPool +
1024→192 dense projection. ~815 source tensors. Pure CPU float math (no
ggml graph) since the per-channel BN folding plus the dense block's
hold-and-broadcast `seg_pool` op are awkward to express in ggml's
broadcasting; CPU is plenty fast for one-shot voice clone (≈2 s on M1
for an 11 s clip).

### Stuff that mattered

1. **`out_nl.bn.*` is NOT under `out_nl.nl.bn.*`**. The other units in
   the GGUF (`tdnn`, `transit{1,2,3}`, `dense`) wrap a Linear+nonlinear
   pair and store the BN at `<unit>.nl.bn.*`. But `out_nl` is a bare
   `get_nonlinear` (only a BN, no Linear), so its tensors are at
   `out_nl.bn.{weight,bias,running_mean,running_var}` directly. Using
   the generic `bind_unit` path silently bound the BN to nullptr and the
   xvector forward segfaulted in `apply_bn_inplace` reading from the
   empty `gamma` vector.

2. **TransitLayer and DenseLayer order is BN→ReLU→Linear, NOT
   Linear→BN→ReLU**. The `get_nonlinear(config_str, in_channels)` runs
   FIRST (on the input), then `Linear(in→out)` projects. So the BN's
   running stats are sized for `in_channels`, not `out_channels`. Folding
   the BN with the wrong `C` produces silently corrupt output. Got this
   right by binding via the source `cb_campplus_unit` and folding inside
   the forward (see `bn_relu_conv1d` helper).

3. **`F.avg_pool1d(k=100, stride=100, ceil_mode=True)` divides by `k`,
   not by the actual frame count**. PyTorch's `count_include_pad=True`
   default treats the partial last window as if it were padded with
   zeros, so the divisor is always 100. Dividing by `n_in_seg` (the
   actual count) instead would shift the seg_pool values for the last
   segment of any T not divisible by 100.

4. **`torch.std` defaults to `unbiased=True`** (divide by `n-1`).
   Matters for the StatsPool — `statistics_pooling(x)` uses
   `x.std(dim=-1, unbiased=True)`. Using `n` instead of `n-1` gives
   slightly off std values that the dense's BN would amplify.

5. **CAMPPlus FCM head: PyTorch `BasicResBlock` shortcut path activates
   when `stride != 1` OR `in_planes != out_planes`**. For our weights
   (out_channels=32 throughout), the shortcut only appears when stride
   changes, which is once per layer (the `.0` block). The `.1` block is
   identity-shortcut. The bind step in `chatterbox_s3gen.cpp` checks
   `b.sc_w` presence to set `b.has_shortcut`.

6. **GGUF Conv2d weight ne=(kW, kH, in, out) maps directly to PyTorch
   `(out, in, kH, kW)` row-major**. ggml's reverse-indexing convention
   means the bytes are the same; just index with PyTorch's natural
   `((o*in + i)*kH + kh)*kW + kw` formula. Same trick for Conv1d
   weights ne=(kW, in, out) → PyTorch `(out, in, kW)`.

### Diff parity (phase 2)

`tools/reference_backends/chatterbox.py` captures `campplus_xvector` via
`s3gen.speaker_encoder.inference([wav_16k])`. The full chatterbox
top-level module loads TensorFlow transitively (slow); the parity dump
script can bypass it by directly importing
`chatterbox.models.s3gen.xvector.CAMPPlus` and loading just the
`speaker_encoder.*` slice from `s3gen.safetensors`.

`crispasr-diff chatterbox` on JFK 11 s gives:
- `campplus_fbank`   cos_mean=0.999999  (fp32 rounding noise, phase 1)
- `campplus_xvector` cos_mean=0.998070  (PASS at the 0.95 threshold —
                     small drift from accumulator order / BLAS-vs-naïve
                     conv compute, well under the per-element floor for
                     speaker-similarity downstream tasks)

The 0.998 vs 1.0 gap is the price of using triple-loop conv kernels
instead of PyTorch's BLAS GEMM accumulation order. For voice cloning
(downstream consumer is S3Gen's CFM cross-attention), this is
imperceptible — the speaker direction is preserved.

## Chatterbox 24 kHz prompt mel — module 4 phase 3 (May 2026)

`chatterbox_campplus.cpp:compute_prompt_feat_24k` ports
`chatterbox.models.s3gen.utils.mel.mel_spectrogram` for the
`gen.prompt_feat` cond. Sits next to CAMPPlus rather than its own file
since both modules consume / emit S3Gen-side conditioning in the
voice-clone path. ~80 lines, pure CPU, uses `core_fft` + the
`core_mel::build_slaney_fb` basis already shared with VE / S3Tokenizer.

### Stuff that mattered

1. **Magnitude (with eps inside the sqrt), not power**. The Matcha
   formulation is `sqrt(re² + im² + 1e-9)` — adding the eps INSIDE the
   sqrt rather than power-then-clamp. `core_mel::SpecKind::Magnitude`
   doesn't take an eps inside the sqrt, so the parity-correct path is
   to write the STFT loop inline with `std::sqrt(re² + im² + 1e-9)`.

2. **Natural log + clip-min, NOT log10 + max-clip(max-8) + (x+4)/4**.
   `dynamic_range_compression_torch(x) = log(clamp(x, 1e-5))`. Plain
   `std::log(std::max(v, 1e-5f))`. Different from S3Tokenizer's mel
   (log10 + Whisper-style normalisation) and from CAMPPlus's Kaldi
   fbank (log on power + epsilon).

3. **`center=False` with manual reflect-pad of `(n_fft - hop) / 2`
   each side, applied via `F.pad(..., mode="reflect")`**. PyTorch's
   reflect mirror EXCLUDES the edge sample (so for `[a, b, c, d]` with
   pad=2 the prefix is `[c, b]`, not `[b, a]`). Got this right by
   walking input indices `1..pad` for the prefix and `n-2..n-1-pad`
   for the suffix.

4. **Reference dump captures both `prompt_feat_24k` AND
   `audio_24k_input`**. The 16 → 24 kHz resample in `prepare_conditionals`
   uses `librosa.resample(res_type="kaiser_fast")` which we don't have
   in C++ yet. Saving the 24 kHz audio bytes alongside the mel lets
   the diff harness feed identical input to its mel — bypasses the
   resampler-parity question entirely. Diff harness reads
   `audio_24k_input` from the reference GGUF and pipes it into
   `chatterbox_dump_prompt_feat_24k`.

### Diff parity

`crispasr-diff chatterbox` on a 3.2 s pre-loaded 24 kHz prompt:

  prompt_feat_24k  shape=[80,160]  cos_min=1.000000  cos_mean=1.000000
                   max_abs=3.69e-04  rms=1.36e-05

Bit-perfect against `mel_spectrogram` for the full mel grid.

## Chatterbox atomic native voice clone — the resampler + 5-cond install (May 2026)

`src/core/audio_resample.{h,cpp}` adds a polyphase Kaiser-windowed
sinc resampler (β=8.6, num_zeros=14 — same parameters
`librosa.resample(res_type='kaiser_fast')` uses). Output is NOT
bit-equivalent to librosa (resampy uses a precomputed polyphase
table with a different precision knob), but acoustically very close.
Sized for any L:M reduction; chatterbox uses 16 ↔ 24 kHz
(L:M = 3:2 / 2:3) but the helper is general.

`chatterbox_set_voice_from_wav` now forks on the input rate:

  - **24 kHz mono PCM16/F32 WAV** — atomic path. Resamples 24 → 16 kHz
    once, then runs ALL FIVE compute modules from a single source:
      - VE (16 kHz) → speaker_emb
      - S3Tokenizer V2 (16 kHz) → speech_prompt_tokens (first 6 s, max 150)
                                  + prompt_token (full audio)
      - CAMPPlus (16 kHz) → gen.embedding (192-d)
      - 24 kHz Matcha mel (24 kHz, truncated to 10 s) → prompt_feat
    All five tensors get installed into the same fresh
    `voice_ctx_w` / `voice_buf_w` slot — atomic, mutually consistent.
  - **16 kHz mono PCM16/F32 WAV** — partial path (existing behaviour).
    Only T3-side conds (speaker_emb, speech_prompt_tokens) get
    installed; gen.* stay at the default voice's tensors. The
    runtime warns that 24 kHz input enables full atomic cloning.

### Stuff that mattered

1. **`embed_ref` enforces `T_mel = 2 * T_speech_tokens`**. If our
   prompt mel comes out shorter than `2 * len(prompt_token)` (e.g.
   because the 24 kHz audio is shorter than 10 s), `s3gen.flow_inference`
   raises a shape-mismatch warning and trims the tokens. We mirror
   the trim in `chatterbox_set_voice_from_wav` so install_native_voice
   sees a consistent (token, mel) pair.

2. **`gen.prompt_feat` ggml shape is (80, T_mel, 1)** — ne[0]=80 is
   the fastest axis, ne[1]=T_mel, ne[2]=1 (the singleton batch). Our
   compute_prompt_feat_24k returns row-major (T_mel, 80), which in
   ggml is exactly ne=(80, T_mel) — copy directly into a
   `ggml_new_tensor_3d(F32, 80, T_mel, 1)`.

3. **`gen.embedding` ggml shape is (192, 1)** — same convention. Our
   CAMPPlus xvector is a flat 192-d vector; ggml_new_tensor_2d(F32,
   192, 1) + ggml_backend_tensor_set with 192 floats covers it.

### Quality assessment

End-to-end verification (Q4_K T3 + `--no-gpu`, prompt "Ask not what
your country can do for you.", `samples/jfk.wav` resampled to 24 kHz
for the atomic path):

  - Atomic native (24 kHz WAV → all 5 conds): rms=0.113, ASR
    roundtrip transcribes the prompt verbatim. Real cloned voice.
  - Baker GGUF baseline (python `bake-chatterbox-voice-from-wav.py`):
    rms=0.118, ASR roundtrip transcribes the prompt verbatim.
  - Partial native (16 kHz WAV → M2+M3 only): rms=0.131, ASR also
    transcribes verbatim — but the **timbre is the default voice**,
    not the reference speaker. The path does NOT actually clone;
    it just feeds T3 the new speaker_emb + speech_prompt_tokens
    while S3Gen still uses the default voice's gen.* triple. The
    T3-side prosody hint isn't enough to override S3Gen's vocal
    identity. For real cloning, use the 24 kHz atomic path or the
    python baker — both verified producing speaker-cloned output.

### Known issue: Metal F16 drift in T3 — auto-fallback shipped

**Chatterbox T3 forward drifts on Metal/GPU past ~step 16**, breaking
the voice clone output regardless of voice path or quantization.
Pre-existing bug, NOT introduced by this work — reproducible at
the original voice-clone commit `86ac98eb` and every commit
before/since. The bisect via in-runtime KV/logit dump
(`CRISPASR_CHATTERBOX_DUMP_KV_AT=N`,
`CRISPASR_CHATTERBOX_DUMP_LOGITS_AT=N`):

| Config | Tokens 0-15 | Tokens 16+ | ASR result |
|---|---|---|---|
| Q4_K + CPU | reference | reference | ✓ "Ask not what your country can do for you." |
| F16 + CPU | bit-identical to Q4_K-CPU | bit-identical | ✓ same |
| Q4_K + GPU | bit-identical to Q4_K-CPU | DRIFTS | ✗ "And not what you're talking about..." |
| F16 + GPU | bit-identical to Q4_K-CPU | DRIFTS more aggressively | ✗ gibberish |

**Logit drift at end of prefill** (single forward pass through 30
layers, T=61 input tokens, no decode steps yet): CPU and GPU differ
by 1e-3 to 5e-2 across the first 8 logits — already enough drift
to flip the multinomial sampler's choices once the trajectories
diverge. The `KV_ON_CPU=1` workaround partly rescues (cleaner
audio, partial transcript) by routing the KV write/read through
the CPU backend, but T3 forward still computes logits on GPU and
drifts. Greedy sampling (`CRISPASR_CHATTERBOX_TEMP=0`) doesn't help
either — confirms the drift is at the logit level, not the sampler.

The drift is **deterministic** (same seed → same broken tokens) so
it's a correctness bug in some Metal op, not a race condition.
Likely the cumulative effect of F16 accumulator order across
mul_mat / flash_attn / norm kernels on Metal vs CPU's
F32-accumulator paths. Other ggml backends in this codebase
(parakeet, voxtral, qwen3) work fine on Metal — chatterbox is
unusual in being an autoregressive multinomial-sampled decoder
where small logit drift compounds catastrophically.

### The fix shipped

`chatterbox_init_from_file` auto-falls-back the **entire chatterbox
forward** (T3 + s3gen) to CPU when the user requests GPU, with a
loud stderr warning:

```
chatterbox: forward auto-falling back to CPU — Metal/GPU has
cumulative F16 drift that breaks chatterbox sampling past ~16
decode steps. Override with CRISPASR_CHATTERBOX_FORCE_GPU=1
(output may be garbled).
```

The decision flips `c->params.use_gpu` so the companion s3gen
sub-context (`chatterbox_set_s3gen_path` runs later) also picks
up the fallback. `--no-gpu` still works as before; explicit users
can override via `CRISPASR_CHATTERBOX_FORCE_GPU=1` for the broken
path (kernel-level debugging) and they get a different warning.

**Verification**: default command (no `--no-gpu` flag) on JFK clone
GGUF produces clean speech rms=0.12, ASR roundtrip transcribes
"Ask not what your country can do for you" verbatim. Same for
24 kHz native WAV input via the atomic clone path. Same for
default voice (no `--voice`). Same for the F16 T3 model that was
previously the most-broken combination.

### Diagnostic env knobs left in place

For future Metal-kernel investigation:
- `CRISPASR_CHATTERBOX_DUMP_KV_AT=<n_past>` — dumps layer-0 K cache
  contents at the requested cache row to stderr.
- `CRISPASR_CHATTERBOX_DUMP_LOGITS_AT=<n_past>` — dumps first 8
  output logits to stderr at the matching forward pass.
- `CRISPASR_CHATTERBOX_TEMP=<float>` — overrides T3 sampling
  temperature (0=greedy) without rebuilding the CLI plumbing.
- `CRISPASR_KV_READ_F32=1` — forces KV cache read to dequantise
  to F32 before flash_attn (didn't fix the drift here, but may
  help other backends with similar issues).
- `CRISPASR_CHATTERBOX_FORCE_GPU=1` — disables the auto-fallback.

The next step on the actual fix is per-op intermediate-tensor
diffs: dump Q, K, V, attention output, FFN output for a fixed
layer at the same step under both CPU and GPU, find which kernel
contributes the dominant drift, patch ggml-metal. The plumbing
for the dump hooks above can be extended to capture those
intermediates.

### Root cause located — ggml-metal `kernel_mul_mm` legacy path (May 2026)

Bisect on chatterbox-base (Q4_K, --no-gpu vs `CRISPASR_CHATTERBOX_FORCE_GPU=1`,
greedy seed=42, "Hello world", `CRISPASR_CHATTERBOX_DUMP_KV_AT=N`):

| Step                                                | CPU K[L0,h0,t]                  | GPU K[L0,h0,t]                  | abs diff       |
|-----------------------------------------------------|---------------------------------|---------------------------------|----------------|
| t=45 (last prefill, **no decode yet**)              | -0.1699 -0.1520 -0.3115 0.0020  | -0.1711 -0.1503 -0.3127 0.0031  | **~1e-3**      |
| t=46 (first decode K)                               | 0.1315 -0.0494  0.0238 -0.0616  | 0.1318 -0.0494  0.0226 -0.0604  | ~1e-3          |
| t=50 (after 5 decode tokens — diverged trajectories)| -0.0811  0.1787  0.2081 0.1721  | 0.0555  0.1400  0.2847 0.1032   | ~1e-1          |

CPU and GPU **already differ by ~1e-3 at the end of prefill** — long
before any decode steps run. The drift originates in the prefill
matmul, not in decode-loop accumulation.

Setting `CRISPASR_KV_QUANT=F32` to take the F16 KV-cache cast out of
the picture: GPU still differs from CPU by the **same** 1e-3. So it's
not the F32→F16 cache write — the per-layer K projection is producing
different F32 values on Metal.

The Metal mul_mm dispatch for chatterbox prefill: ne11 = 46 > 8, ne00
% 128 == 0, so `mul_mm` is selected (legacy path on M1-M4 because
`has_tensor` is gated on M5/A19+ in `ggml-metal-device.m:669-676`).

The bug is in `ggml/src/ggml-metal/ggml-metal.metal` legacy
`kernel_mul_mm`, line 9590:

```cpp
*(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
```

For all `kernel_mul_mm_*_f32` instantiations, `S1=half` but `T1=float`.
This **explicitly rounds the F32 input B to F16 when staging into
shared memory** before the simdgroup matmul. The accumulator
`mc[i] = simdgroup_float8x8` is F32 (correct), but both operands are
half tiles (`S0_8x8 = S1_8x8 = simdgroup_half8x8`), so the product is
half × half multiplied and F32-accumulated — the F16 input rounding
loses precision the CPU path retains.

Magnitude check: F16 has ~3e-4 relative spacing at value 0.1, so each
of K=1024 input elements rounds with absolute error ~3e-5; summed
over the dot product, RMS drift is √1024 × 3e-5 ≈ 1e-3. **Matches
observation exactly.**

CPU `ggml_compute_forward_mul_mat` performs F32 × F32 → F32 multiplies
with no input rounding (the weight is dequantised to F32 then a
plain dot product is run), which is why CPU is ~1e-3 closer to ground
truth than Metal on M1-M4.

### Why other models tolerate it

Whisper, parakeet, qwen3, voxtral etc. also hit `kernel_mul_mm_*_f32`
and the same 1e-3 drift on K-dim ~768-1024 dot products. They appear
fine because (a) they run argmax / greedy / beam decoding, where logit
drift below ~1e-2 doesn't flip the top token, and (b) the drift
doesn't compound across decode steps when the sampled token is the
same as the CPU-sampled one. Chatterbox is unique in this codebase
because:

1. multinomial sampling on a 8194-vocab speech-token distribution
   amplifies tiny drifts into different sampled tokens once the
   probability mass crosses ~5e-2 between candidates;
2. the divergent token then enters the KV cache — so subsequent
   decode steps see slightly different K, V values, drift compounds;
3. unlike text models, the autoregressive speech decoder has no
   self-correcting language-model prior pulling trajectories back
   together.

So the drift exists on every Metal user but only chatterbox notices.

### The proper upstream fix

Add `_hp` (high-precision) variants of `kernel_mul_mm_*_f32` that use
`S0=S1=float` and `simdgroup_float8x8` operand tiles. The
`simdgroup_multiply_accumulate(mc, mb, ma, mc)` then does
F32 × F32 → F32 with no input rounding, matching CPU. Dispatch by
honoring the existing `GGML_PREC_F32` `op_params[0]` flag in
`ggml_metal_library_get_pipeline_mul_mm`. Mark the chatterbox T3
QKV/output/FFN projections with `ggml_mul_mat_set_prec(...,
GGML_PREC_F32)` so they use the precise kernel; other backends keep
the existing legacy half-multiply behaviour and pay no perf tax.

### Attempted patch (2026-05-10) — did not fix the drift

Implemented the above plan: added `kernel_mul_mm_hp` template alongside
the legacy kernel (8192-byte sa offset, `simdgroup_float8x8` tiles for
both operands, `*(threadgroup float2x4 *)(sb + ...) = *((device
float2x4 *) y)` with no F16 cast). Wired up F32, F16, Q4_K, Q5_K, Q6_K,
Q8_0 host_name instantiations. Patched `mul_mm` dispatch to append
`_hp` to the pipeline name when `prec_f32 && !has_tensor && tsrc1 ==
F32`. Added a graph-walk in chatterbox `build_graph_t3_kv` /
`build_graph_t3_gpt2_kv` that calls `ggml_mul_mat_set_prec(...,
GGML_PREC_F32)` on every `GGML_OP_MUL_MAT` node post-build.

Verified the hp pipeline IS dispatched for the chatterbox K projection
(q4_K × f32, ne00=1024, ne01=1024, ne11=46) via a one-shot debug
print, AND that the kernel actually runs (overwriting `mc[0]` with a
known constant changed the LGT output away from the legacy GPU
values). But the K[L0,h0,t=45] cache values were **bit-for-bit
identical** to the pre-patch GPU run — same 1e-3 drift vs CPU.

Possible explanations:

1. **Apple's `simdgroup_float8x8` MAC silently downconverts.** The HW
   tensor cores on M1-M4 may multiply in half precision regardless of
   operand declared type, with float only for the accumulator. There
   is no public docs guarantee that `simdgroup_multiply_accumulate
   (simdgroup_float8x8&, simdgroup_float8x8, simdgroup_float8x8,
   simdgroup_float8x8)` is bit-equivalent to a pure F32 dot product —
   it may be implemented as `float8x8 = float(half(a) × half(b)) +
   float8x8`.
2. **The cache write happens via a different code path.** The KV cache
   in chatterbox is allocated `on cpu` (verbose log line). The
   ggml-backend scheduler may route the K projection mul_mat to CPU
   (since its consumer — the cpy to CPU cache — runs on CPU), bypassing
   the hp kernel. But the empirical CPU vs GPU divergence (~1e-3 even
   with FORCE_GPU=1) suggests this isn't the case for at least the
   path that the dump observes.
3. **Drift is somewhere else entirely.** RMSNorm reduction order,
   rope sin/cos precision, or the F32→F16 cast in the cpy kernel —
   though we ruled out the F16 cast via `CRISPASR_KV_QUANT=F32`.

Decision: keep the hp kernel + dispatch infrastructure in place — it
costs no perf when prec is not F32 (default), and provides a
foundation for future investigation. Auto-fallback to CPU remains the
working path for chatterbox.

The next investigation step is a proper per-op intermediate-tensor
dump: compare `norm(x)` output, `K_proj` output (pre-rope), and
`K_rope` output element-by-element between CPU and GPU at layer 0
position 45. That will identify which specific op contributes the
drift. The dump hooks at `chatterbox.cpp:1199-1234` can be extended
with a similar `CRISPASR_CHATTERBOX_DUMP_NORM_AT` /
`DUMP_KPROJ_AT` / `DUMP_KROPE_AT`.

Tracked as a follow-up in PLAN.md #83.

### Methodical bisect, round 2 (2026-05-10) — drift is in mul_mat algorithm

Added per-op intermediate dumps (`CRISPASR_CHATTERBOX_DUMP_NORM_AT`,
`DUMP_KPROJ_AT`, `DUMP_KROPE_AT`, `DUMP_WK`) in
`chatterbox.cpp:build_graph_t3_kv` that name and surface the layer-0
RMSNorm output, K projection mul_mat output, K rope output, and the
dequantized K weight tensor. Run on CPU and GPU at the same step,
diff per element. Findings:

| stage              | CPU                                | GPU (FORCE_GPU)                    | drift |
|--------------------|------------------------------------|------------------------------------|-------|
| `L0_norm_out` t=45 | -0.0589 -0.0123 -0.0126 -0.0622 …  | -0.0589 -0.0123 -0.0126 -0.0622 …  | **0** (norm bit-identical) |
| `L0_W_K_f32` row=0 | -0.012791 -0.039632 -0.012791 …    | -0.012791 -0.039632 -0.012791 …    | **0** (Q4_K dequant bit-identical, after fixing `.h` → `.f` literals) |
| `L0_K_proj` t=45   | -0.3584  0.0566  0.2575 -0.0085 …  | -0.3579  0.0566  0.2551 -0.0067 …  | **~1e-3** (the matmul is the culprit) |
| `L0_K_rope` t=45   | -0.1699 -0.1519 -0.3115  0.0020 …  | -0.1712 -0.1503 -0.3128  0.0031 …  | ~1e-3 (rope just propagates) |

So norm and dequant are bit-identical CPU/GPU; the drift is **purely in
the K projection mul_mat itself**.

Then the orthogonal test: load chatterbox-t3-f16 (F16 weights, NOT
Q4_K). Run CPU vs GPU. K_proj output now bit-identical between CPU
and GPU at every position tested through t=70+. KV cache values also
bit-identical. So the matmul is correct **for F16 inputs**.

Forcing the dispatch through `mul_mv_ext` (the F32-precise
`dot(float4,float4)` path) for `PREC_F32` ops, then casting Q4_K to
F32 before the matmul (`ggml_cast(W, F32)` followed by F32×F32
matmul): **same ~1e-3 drift**. So even when the GPU does
F32×F32 matmul on dequantized Q4_K weights, it doesn't match the
CPU's Q4_K matmul.

The real reason: **CPU and GPU implement Q4_K matmul differently.**
CPU's path is `ggml_vec_dot_q4_K_q8_K` (`ggml-cpu/quants.c:645`) —
it quantizes the F32 input to Q8_K (8-bit-per-element block-quantized)
before the dot product, then computes the dot using packed integer
multiplies + scale factors. GPU's path is `kernel_mul_mv_q4_K_f32`
(`ggml-metal.metal:7748`) — it keeps F32 input and multiplies by the
unpacked Q4_K nibbles directly. Both are valid but produce different
F32 outputs in the ~1e-3 range, and **chatterbox's multinomial
sampler is sensitive enough to flip token selection**.

Q8_0 weights show the same CPU/GPU drift (~1e-3) confirming the issue
isn't Q4_K-specific — it's any quantized weight type with input-quant
on CPU vs F32-input on GPU.

### Patch landed (partial fix, useful infrastructure)

1. `dequantize_q4_K` rewritten to mirror CPU's
   `dequantize_row_q4_K` arithmetic exactly: `dl * (q & 0x0F)` for
   the low nibble and `dl * (q >> 4)` for the high nibble, with
   `d` and `dl` always F32. Was previously `(d/16.h) * sc * (q &
   0xF0)` — F16 division, mathematically equivalent but rounds
   differently in F32. After the fix, CPU and GPU dequant are
   bit-identical (verified element-by-element on row 0, 256
   weights). Small but principled improvement.
2. `kernel_mul_mv_ext` dispatch in `ggml-metal-ops.cpp` now honours
   `GGML_PREC_F32` even for `ne11 > 8` (legacy path on M1-M4 only —
   tensor API has its own behaviour). Picks `r1ptg=4` for the
   PREC_F32 batch case. Allows Vulkan-style PREC_F32 → high-precision
   kernel routing. **Doesn't fix the algorithmic CPU/GPU mismatch
   for quantised weights** — F32 dot product on dequantised Q-weights
   still differs from CPU's Q8_K-quantized dot product.
3. `kernel_mul_mm_hp` and the chatterbox `ggml_mul_mat_set_prec(...,
   GGML_PREC_F32)` graph walk land as planned; harmless when the
   matmul falls back to legacy half-tile (other backends).
4. `CRISPASR_METAL_STRICT_FP=1` knob disables Metal fast-math
   (`setFastMathEnabled:NO`). Tested — doesn't change the drift, so
   it's not an FMA/fusion issue.
5. Per-op intermediate dump knobs in `chatterbox.cpp` for future
   bisects: `CRISPASR_CHATTERBOX_DUMP_NORM_AT=<t>`, `DUMP_KPROJ_AT`,
   `DUMP_KROPE_AT`, `DUMP_WK`.

### Remaining work

The proper fix is a Metal `kernel_mul_mv_q4_K_q8_K` (Q4_K weights ×
Q8_K-quantised input) mirroring CPU's `ggml_vec_dot_q4_K_q8_K`. That
needs:

1. A Q8_K quantisation kernel that block-quantises F32 input to Q8_K
   (256-element blocks, one F32 scale per block, plus per-16-element
   subscale).
2. A new Metal mul_mv (and mul_mm matrix variant) that consumes
   Q8_K input and Q4_K weight, doing integer multiplies + F32 scale.

This is ~200-400 lines of Metal kernel code for **each** quant pair
(Q4_K×Q8_K, Q5_K×Q8_K, Q6_K×Q8_K, Q8_0×Q8_0, ...). A rabbit hole. The
auto-fallback to CPU remains the practical fix for chatterbox.

For users who want GPU performance: convert chatterbox to F16 weights
(`chatterbox-t3-f16-regen.gguf`) — F16×F32 matmul matches between
CPU and GPU bit-for-bit, so chatterbox runs correctly on GPU. The
caveat: **end-to-end F16+GPU still has audible artefacts** because
something downstream of T3 (likely s3gen or speech_head F32 matmul
through some indirect path) introduces additional drift. So even F16
isn't a clean GPU path. Auto-fallback remains.

### Round 4 (2026-05-10) — kernel-level fix landed (partial bit-identity)

Implemented the proper kernel-level fix:

1. `kernel_quantize_q8_K_f32` (ggml-metal.metal): F32 input column →
   Q8_K block. Each threadgroup processes 1 256-element block, 32
   threads per group. simd_shuffle_xor reduction finds amax + signed
   max, then `iscale = -127/max`, `qs[j] = MIN(127, round(iscale ×
   x[j]))`, `bsums[k] = sum of qs[k*16..k*16+15]`. Mirrors CPU
   `quantize_row_q8_K_ref` exactly.
2. `kernel_mul_mv_q4_K_q8_K` (ggml-metal.metal): one thread per output
   element. Mirrors CPU `ggml_vec_dot_q4_K_q8_K_generic` exactly:
   unpack 256 Q4_K nibbles into int8 a[256], unpack 12-byte scales
   table into 8 scale + 8 min uchars (kmask1/2/3), int32 accumulators
   per scale lane, dmin contribution via bsums × mins, final F32
   multiply by d_q4 × d_q8.
3. Dispatch path in `ggml_metal_op_mul_mat`: when PREC_F32 + Q4_K
   weight + F32 input + ne00 % 256 == 0, reserve a Q8_K scratch
   buffer at the tail of dst (via
   `ggml_metal_op_mul_mat_extra_q8_K` + the
   `ggml_backend_metal_buffer_type_get_alloc_size` hook), dispatch
   the quantize kernel to fill it, sync via
   `ggml_metal_op_concurrency_reset`, dispatch the Q4_K×Q8_K matmul.
4. Routed `flash_attn_ext` to CPU when PREC_F32 is set, via
   `ggml_metal_device_supports_op` returning false. Apple's FA kernel
   uses simdgroup_half8x8 tiles for Q×K^T regardless of K type
   (FA_TYPES_F32 still declares Q as half), leaking ~1e-4 drift even
   with F32 KV. Routing to CPU avoids the issue at the cost of
   per-layer cross-device transfers.
5. chatterbox `build_graph_t3_kv` / `build_graph_t3_gpt2_kv` graph
   walks now tag every `GGML_OP_MUL_MAT` AND every
   `GGML_OP_FLASH_ATTN_EXT` with `GGML_PREC_F32`.

### Verification + remaining drift

| Stage                   | CPU vs GPU drift                  |
|-------------------------|-----------------------------------|
| Layer-0 norm output     | bit-identical                     |
| Layer-0 K weight (cast) | bit-identical (after `.h` → `.f`) |
| Layer-0 K projection    | **bit-identical**                 |
| Layer-0 K cache (KV[0]) | **bit-identical** through every t |
| Layer-0 V projection    | bit-identical                     |
| Layer-0 Q projection    | bit-identical                     |
| Layer-0 K rope          | bit-identical                     |
| Layer-0 attn output     | **bit-identical** (post out_proj) |
| Layer-0 FFN output      | **bit-identical**                 |
| Layer-1 norm output     | bit-identical                     |
| Layer-1 K projection    | bit-identical                     |
| Layer-1 K cache         | bit-identical at every sampled t  |
| Layer-1 attn output     | **drifts ~1e-4** (despite all of the above being identical!) |
| Layer-29 attn output    | drifts ~3e-3                      |
| End-of-prefill logits   | drift ~0.1–0.2                    |

AR token sequence (chatterbox seed=42, "Hello world", greedy temp=0):
matches CPU through **decode step 11**, then diverges. Compare to:
- Pre-#83 fix:                  matches through ~step 1
- After kernel only:            matches through step 6
- After kernel + FA→CPU:        matches through step 11

End-to-end audio: chatterbox FORCE_GPU=1 now produces partially
intelligible speech ("They put your" instead of empty/noise on the
JFK clone). The auto-fallback remains the production path until the
remaining ~1e-4 layer-1+ drift is identified and eliminated — likely
in some Metal op the diagnostic dumps haven't yet traced through
(candidates: rope on Q with rope_freq_factors, swiglu, an
F32-weight matmul path I haven't covered, or an inter-kernel
cross-device copy with subtle precision behaviour).

### Status

The Q4_K × Q8_K kernel infrastructure is solid and ships with diagnostic
knobs (`CRISPASR_CHATTERBOX_DUMP_KV_LAYER`, `_DUMP_LAYER`, `_DUMP_ATTN_AT`,
`_DUMP_FFN_AT`, `_DUMP_QPROJ_AT`, `_DUMP_VPROJ_AT`) for further
bisection. Q5_K, Q6_K, Q8_0 use the same template — straightforward
port once Q4_K is shaken out.

The default chatterbox path (no env overrides) auto-falls-back to
full CPU. `CRISPASR_CHATTERBOX_FORCE_GPU=1` enables the new
kernel + FA→CPU path; tokens diverge at step 11+ but layer 0 is
bit-identical.

### Round 5 (2026-05-10) — FA-input bisect: every input bit-identical, output drifts

After the Round 4 kernel fix, layer-0 was bit-identical CPU/GPU
end-to-end (every named tensor: norm, K/Q/V proj, K rope, attn out,
FFN out — all match). Layer 1 K cache and V cache were also
bit-identical at every sampled position (t=0, 10, 20, 30, 40, 45) on
both halves. So the input to layer-1 attention is reliably
bit-identical. Yet **layer-1 attn out (post out_proj) drifts ~1e-4**
and propagates by layer 29 to ~3e-3, blowing up to ~0.1–0.2 in
end-of-prefill logits. AR token sequence matches CPU exactly through
decode step 11, then diverges.

Bisect-round-5 added named graph outputs in `core_attn::kv_self_attn`
for the FA inputs and outputs (`CRISPASR_CORE_ATTN_DUMP_FA_LAYER=N`):

| Stage at layer 1, t=45  | CPU vs GPU       |
|-------------------------|------------------|
| Q post-rope             | bit-identical    |
| Kfull (FA input)        | bit-identical    |
| Vfull (FA input)        | bit-identical    |
| **FA output**           | **drifts 1e-4**  |
| out_proj input (= FA reshape) | drifts 1e-4 (propagated) |
| attn out (post out_proj)| drifts 1e-4 (propagated) |

`flash_attn` was confirmed routed to CPU on every layer (5100 of 5100
T3 FA ops on CPU; the 30 Metal FA ops counted are S3Gen). So both
halves invoke `ggml_compute_forward_flash_attn_ext_f16` with
bit-identical Q, K, V, mask, scale. The function is deterministic, yet
output differs.

Forced single-thread (`-t 1`) didn't change the drift, ruling out
ggml-cpu's threaded reduction order. Tested with both default
multi-threaded and `-t 1`.

The puzzle: layer 0 stays bit-identical (FA inputs and output match),
but starting from layer 1 the FA output diverges despite identical
inputs. The split between layer 0 and layer 1 strongly suggests
something about the second-or-later FA invocation in a graph that
mixes Metal and CPU backends.

Possible explanations to chase next session:

1. **Memory ordering / barrier** — Apple unified memory shares physical
   buffers between CPU and GPU. When a CPU-routed op (FA) writes its
   output and a subsequent GPU-routed op (out_proj) reads it, the
   ggml-backend scheduler should emit a sync. If the sync is missing
   for the second-or-later layer, the GPU might read stale or
   half-flushed CPU writes.
2. **Backend-internal cache pollution** — ggml-cpu reuses internal
   thread-local scratch for FA's softmax. If that scratch was
   touched by a prior op and not cleared, the second FA invocation
   could pick up garbage in unaccounted bytes.
3. **Different CPU FA chunking depending on graph context** —
   `ggml_compute_forward_flash_attn_ext_f16` has multiple internal
   paths (one_chunk, tiled). The chunk decision uses `ne01` and
   thread count; if these differ between a CPU-only run and a
   Metal-routed run (different graph structure passed to the
   ggml-cpu backend), chunk boundaries shift and F32 reductions land
   on different roundings.
4. **`ggml_cont` produces F16 cache view differently on Metal** —
   even though I dumped K cache values and they matched at sampled
   positions, the layout/stride of the resulting Kfull contiguous
   tensor might differ, putting bytes in different addresses, which
   could affect SIMD-aligned loads in CPU FA.

Diagnostic infra ready for round 6:

- `CRISPASR_CORE_ATTN_DUMP_FA_LAYER=N` — names FA inputs/output of
  layer N as graph outputs.
- `CRISPASR_CORE_ATTN_DUMP_FA_AT=t` / `_Q_AT` / `_KFULL_AT` / `_VFULL_AT`
  — fetch and print the named outputs for position t.
- `CRISPASR_CHATTERBOX_DUMP_KV_LAYER=N`, `_DUMP_KV_AT=t`,
  `_DUMP_VV_AT=t` — KV cache dump per layer.
- `CRISPASR_CHATTERBOX_DUMP_LAYER=N`, `_DUMP_ATTN_AT=t`, `_DUMP_FFN_AT=t`,
  `_DUMP_NORM_AT=t`, `_DUMP_KPROJ_AT=t`, `_DUMP_QPROJ_AT=t`,
  `_DUMP_VPROJ_AT=t`, `_DUMP_KROPE_AT=t` — per-layer per-stage dumps.

### Benchmark (2026-05-10)

Measured end-to-end on chatterbox-base T3 Q4_K (\"Ask not what your
country can do for you, ask what you can do for your country\",
median of 3 runs):

| Path                                               | wall time |
|----------------------------------------------------|-----------|
| CPU (auto-fallback, default)                       | ~66 s     |
| GPU FORCE_GPU=1 (kernel_mul_mv_q4_K_q8_K + FA→CPU) | ~58 s     |

~12% speedup, no regression. End-to-end audio with FORCE_GPU=1
produces partially intelligible cloned speech (\"They put your\"
recognised by ASR vs the canonical \"Ask not what your country can
do for you...\" on the CPU path). The token sequence matches CPU
through decode step 11, then diverges due to the layer-1+ residual.

### Production default

Auto-fallback to full CPU remains the production default — the
~1e-4 layer-1+ drift is too small to hurt CPU output but flips
chatterbox's multinomial speech-token sampler around step 11+ on the
GPU path, producing audibly degraded clones. The
`CRISPASR_CHATTERBOX_FORCE_GPU=1` knob unlocks the new kernel for
debug/perf experimentation.

### Round 6 (2026-05-19) — ggml_mish red herring

Round 4–5 stayed focused on T3; meanwhile shipping the round-4 patches
to T3 cleaned up the user-audible artefact the diagnostics had been
chasing in T3, leaving a remaining "audio garbled with FORCE_GPU=1"
that round 6 traced to S3Gen rather than T3. The first suspect was
the hand-rolled `ggml_mish` in `chatterbox_s3gen.cpp`:

```cpp
exp_x = ggml_exp(x);
ones  = ggml_div(exp_x, exp_x);          // ← fabricates 1.0 from x/x
sp    = ggml_log(ggml_add(exp_x, ones)); // log(exp(x) + 1)
out   = ggml_mul(x, ggml_tanh(sp));
```

The `exp_x / exp_x` is NaN whenever `exp(x)` overflows to inf or
underflows to 0. Replaced with `ggml_mul(x, ggml_tanh(ggml_softplus(x)))`
using ggml's native `ggml_softplus` (single fused kernel, identical
clamp-at-x>20 on Metal and CPU). The fix is a small correctness
improvement (s3gen_mel cos_min 0.999971 → 0.999980 on CPU), but the
GPU drift is unaffected (FORCE_GPU stays at 0.923). Mish was not the
divergence source — kept for the NaN-safety gain.

### Round 7 (2026-05-19) — op-bisect: drift is compound, not single-op

Added `CRISPASR_S3GEN_UNET_PIN_CPU_OP=<op>` (pin only the named op to
CPU, rest on GPU) and `CRISPASR_S3GEN_UNET_KEEP_GPU_OP=<op>` (inverse —
pin everything else to CPU) inside `s3gen_maybe_pin_graph_to_cpu`.
Names follow `ggml_op_name()` lowercase minus the `OP_` prefix, or
`unary_<lowercase>` for `GGML_OP_UNARY`. Ran the matrix with
`CRISPASR_CHATTERBOX_FORCE_GPU=1` against `s3gen_mel` (diff harness,
`replay=exact_init_noise`):

| `PIN_CPU_OP=` | s3gen_mel cos_min |
|---|---|
| (none — full GPU) | 0.923 |
| `mul_mat` / `flash_attn_ext` / `norm` / `add` / `concat` / `unary_gelu` / `unary_tanh` / `unary_softplus` | **1.000000** |
| `conv_1d` / `scale` / `unary_mish` | 0.923 (no effect) |

**The bug is not in any single op.** Pinning *any* frequent op type to
CPU forces a F32 CPU↔GPU round-trip every N ops in the schedule and
breaks the drift chain. The non-effective entries are op types with
zero or near-zero count in `build_graph_unet1d` (mish lowers to
softplus+tanh+mul via round 6 → no `GGML_UNARY_OP_MISH` nodes).
**Re-verified `PREC_F32` tagging** on every UNet mul_mat: the
CrispASR `_hp` mul_mm dispatch + `mul_mv_ext` path are picked up
correctly on M1 (`has_tensor=false`), but cos stays at 0.923 — the
surrounding ops keep compounding. PREC_F32 fixes mul_mat alone, not
the chain. Reverted the tags as graph clutter that doesn't fix the
audible bug.

**Auto-pinning UNet under FORCE_GPU is not a viable workaround.** Diff
harness shows `cos=1.000` with `replay=exact_init_noise`, but the
full TTS pipeline (random noise + GPU encoder + GPU vocoder) produces
NaN/Inf mel going into the vocoder (`rms=nan min=1e30 max=-1e30`),
which the vocoder amplifies into saturated audio (parakeet ASR
returns empty transcript). Pinning encoder + vocoder + UNet all to
CPU under a GPU-initialised sched still gives `rms ≈ 11089` vs the
~3900 a clean CPU sched gives. Something about the GPU-backed
scheduler interacts badly with random-noise inputs through the
CPU-pinned compute path; the diff harness's pre-recorded reference
noise masked it. So **the vocoder is not the bug** — it's just
amplifying upstream garbage; the UNet1D is producing NaN under the
GPU-sched + random-noise path.

### Round 7b (2026-05-19) — Metal default is full CPU

Stable warm benchmark on M1 (3 runs, "Ask not what your country can
do for you, ask what you can do for your country", JFK speaker
clone, --seed 1):

| Config | wall time | ASR roundtrip |
|---|---|---|
| Full CPU | ~50 s | ✓ exact |
| T3 GPU + S3Gen CPU (old default) | ~75 s | ✓ exact |
| FORCE_GPU=1 (broken) | ~96 s | ✗ empty |

T3 on Metal is *slower* than T3 on CPU for chatterbox's batch-1 AR:
30 layers × 86 sequential tokens × ~10 ops/layer ≈ 25 k tiny kernel
launches, each paying µs-class Metal dispatch overhead that the M1
NEON CPU caches blast straight through. So the chatterbox default
was switched on Metal builds (`#ifdef GGML_USE_METAL`) to **full
CPU**. `CRISPASR_CHATTERBOX_T3_GPU=1` opts back into T3-on-GPU; the
non-Metal default (CUDA / Vulkan / etc.) is unchanged. Saves ~30 %
wall time on M1.

### Round 8 (2026-05-23) — CUDA P100 bisect confirms cross-backend compound drift

Ran the same `PIN_CPU_OP` / `KEEP_GPU_OP` sweep on Kaggle P100
(Tesla P100-PCIE-16GB, compute capability 6.0 / sm_60, CUDA 12.8)
using a self-contained Kaggle kernel that clones + builds CrispASR,
generates a ref GGUF on-device, downloads Q8_0 GGUFs, then runs 21
diff configs with `CRISPASR_DIFF_USE_GPU=1 CRISPASR_CHATTERBOX_FORCE_GPU=1`.

| Config | s3gen_mel cos_min |
|---|---|
| Full CPU (control) | 0.999955 |
| S3Gen on CUDA P100 baseline (no pin) | **0.858455** |
| `PIN_CPU_OP=mul_mat` | **1.000000** ← FIXES |
| `PIN_CPU_OP=flash_attn_ext` | **1.000000** ← FIXES |
| `PIN_CPU_OP=norm` | **1.000000** ← FIXES |
| `PIN_CPU_OP=add` | **1.000000** ← FIXES |
| `PIN_CPU_OP=concat` | **1.000000** ← FIXES |
| `PIN_CPU_OP=unary_gelu` | **1.000000** ← FIXES |
| `PIN_CPU_OP=unary_tanh` | **1.000000** ← FIXES |
| `PIN_CPU_OP=unary_softplus` | **1.000000** ← FIXES |
| `PIN_CPU_OP=unary_mish` | 0.858542 (no effect) |
| `PIN_CPU_OP=conv_1d` | 0.858348 (no effect) |
| `KEEP_GPU_OP=<any>` | None (crash — ggml tensor type constraint) |

**Conclusions:**

1. **The compound FP16 drift is not Metal-specific.** CUDA P100
   shows the same pattern (cos_min 0.858, worse than Metal's 0.923),
   with the same 8 ops fixing it when individually pinned to CPU.

2. **The fix is cross-backend.** Pinning `mul_mat` (or any of the 7
   other high-frequency FP16 op types) to CPU forces a dtype
   round-trip that breaks the drift chain through the 10-step CFM
   solver. The same `CRISPASR_S3GEN_UNET_PIN_CPU_OP=mul_mat`
   workaround that restores cos=1.000 on M1 works identically on
   CUDA P100.

3. **unary_mish and conv_1d are not the drift source.** Consistent
   with Metal — the mish op was replaced with
   `softplus+tanh+mul` in Round 6, so there are no `GGML_UNARY_OP_MISH`
   nodes in the UNet graph; conv_1d's fp32 path on CUDA is already
   exact.

4. **KEEP_GPU_OP all crash.** Running only one op type on GPU with
   the rest pinned to CPU triggers a ggml type-constraint violation
   (GPU output tensors can't feed CPU op inputs without explicit
   view/copy nodes that the UNet graph doesn't insert). Not a useful
   fix path.

5. **Next step for a real fix:** Instead of per-op CPU pinning (which
   adds round-trip overhead), promote the S3Gen UNet1D graph to
   FP32 on GPU (force `GGML_TYPE_F32` weight loading and remove the
   FP16 intermediate casts). This matches what PyTorch does on CUDA
   by default and avoids the compound rounding entirely. Alternatively,
   enable `GGML_SCHED_MAX_COPIES=1` with an explicit F32-accum
   prefix on each UNet block's matmul — whichever is cheaper.

6. **Per-op CPU pinning causes NaN in full-pipeline runs.** Attempting
   to auto-pin `mul_mat` to CPU via `ggml_backend_sched_set_tensor_backend`
   in a GPU-backed scheduler works correctly in the diff harness (small T=102,
   no prompt tokens: cos_min=1.0) but produces NaN in the full production
   pipeline (T≥392, 157 prompt tokens). Any mix of CPU and GPU compute ops
   in a GPU-backed scheduler causes GPU→CPU tensor copy synchronization
   failures for large graphs — the NaN appears at denoiser step 0.
   Per-op CPU pinning is therefore not safe for production. The production
   fix remains S3Gen on CPU (current default); the mathematically correct
   fix (UNet FP32 promotion) requires GPU-side kernel changes.

2. **T3 sampling can drift on long technical prompts**. The seed=0
   default is deterministic, but particular prompts produce
   degenerate output (e.g. "Stop, stop, stop" repetition or wholly
   unrelated text). Short, common phrases work reliably; if a prompt
   produces gibberish, try a different seed via
   `CRISPASR_CHATTERBOX_SEED=<n>` or shorten the input.

### Misdiagnosis worth recording

This entry's first draft claimed the atomic-native path produced
intelligible speech "with text drift" on a long technical prompt
("Native voice clone test, all five conditions installed
atomically."). That was wrong on two counts: (a) I was running with
the broken F16 + GPU path, and (b) the long technical prompt itself
triggers the T3 sampler-drift issue regardless of voice path. Both
issues pre-date this work but I missed them because my first
sanity check used a contaminated combination. The correct
verification command is:

```bash
./build/bin/crispasr --backend chatterbox \
  -m /path/to/chatterbox-t3-q4_k-regen.gguf \
  --codec-model /path/to/chatterbox-s3gen-q8_0.gguf \
  --no-gpu \
  --voice <ref>.gguf-or-24kwav \
  --tts "Ask not what your country can do for you." \
  --tts-output out.wav
```

The parity-quality compute kernels (VE / S3Tok / CAMPPlus / 24 kHz
mel) all remain bit- or fp32-rounding-tight against PyTorch when
fed identical bytes via the diff harness's `audio_24k_input`
bypass — that part of the work is unaffected by the F16+GPU bug
and the sampler drift.

### CLI

```bash
# Full atomic native clone — 24 kHz mono PCM16/F32 WAV input.
./build/bin/crispasr --backend chatterbox -m auto \
    --voice ref_24k.wav \
    --tts "Hello there, this is the cloned voice." \
    --tts-output cloned.wav

# Partial T3-side-only clone — 16 kHz mono WAV input.
./build/bin/crispasr --backend chatterbox -m auto \
    --voice ref_16k.wav \
    --tts "..." --tts-output ...
```

The runtime prints exactly which conds are installed at verbosity ≥ 1.
For perfect parity with the python baker (full quality), the existing
`models/bake-chatterbox-voice-from-wav.py` workflow remains
recommended; the native path is the no-python-required alternative.

## T5-family translation runtime traps (May 2026, MADLAD-400 debugging)

Bringing up the T5 encoder-decoder runtime (`src/t5_translate.cpp`)
on MADLAD-400 surfaced four bugs in sequence. Each one looked like
"the runtime is broken" until carefully diff-tested against a Python
reference (flan-t5-small via `transformers.T5Tokenizer` +
`T5ForConditionalGeneration`). Capturing here so the next T5 / SP
port doesn't repeat them.

### 1. Bidirectional rel-pos bucket: FUTURE/PAST halves swapped

The encoder's bidirectional rel-pos bucketing has two halves —
buckets `[0..N/2-1]` for past+self, `[N/2..N-1]` for future. Earlier
C++:

```cpp
int n = -rel_pos;
ret += (n < 0) ? 0 : num_buckets;   // adds num_buckets when n>=0
                                    // (rel_pos<=0, PAST/SELF) — WRONG
```

Canonical T5 (HF):

```python
relative_buckets += (relative_position > 0) * num_buckets   # FUTURE
```

Symptoms: encoder output was wrong by a per-head learned-bias offset.
Decoder cross-attention then had wrong keys at every position →
degenerate loop on most-frequent tokens (the "rel-pos repeating-token
loop" behavior). Fix: drop the sign flip, use `rel_pos > 0` directly.
The unidirectional decoder branch (only past valid under the causal
mask) was already correct — only the encoder's bidirectional path
was wrong.

### 2. Special-token IDs vary across T5-family models

C++ tokenizer hardcoded `<unk>=2`, `</s>=1`. That's correct for
flan-t5 / mT5, but MADLAD-400 has **different IDs**:

```
flan-t5: <pad>=0, </s>=1, <unk>=2
MADLAD:  <unk>=0, <s>=1,  </s>=2
```

Hardcoded `ids.push_back(2)` as the unk-fallback in tokenize_sp
emitted MADLAD's `</s>` (= EOS) instead of `<unk>`, prematurely
terminating the encoder input. Hardcoded trailing `ids.push_back(1)`
emitted MADLAD's `<s>` (= BOS) instead of `</s>` — model never saw
EOS at end of input.

Fix: read `t5.eos_token_id` from GGUF metadata (already in the
loader) and propagate to `tokenizer.eos_id`; look up `<unk>` in
the vocab to get `tokenizer.unk_id` dynamically. Both used in
tokenize_sp instead of literal IDs.

### 3. Greedy-longest-match ≠ SentencePiece Unigram

Multi-byte special tokens like MADLAD's `<2de>` (id 33) get
mis-tokenized by greedy:

```
input:  ▁<2de>▁Hello
greedy: [▁<](4411) + [2](810) + [de](948) + [>](3048) + [▁Hello](88912)
                                                        # 4 garbage tokens
SP:     [▁](805) + [<2de>](33) + [▁Hello](88912)         # 2 + 1
```

Greedy picks `▁<` because it's a longer byte match than `▁` alone.
SP unigram picks `<2de>` because that piece's score (lang tags have
very high SP scores) dwarfs the per-byte fragment scores.

Without the right `<2de>` token in the encoder input, MADLAD has no
language signal and emits whatever its language prior dominates on
(in our run: Hebrew when asked for German).

Fix: load `tokenizer.ggml.scores` from the GGUF and replace greedy
with Viterbi best-segmentation — DP over byte positions, keeping
the highest total log-prob. Codepoint-aligned (skip end positions
that fall on a UTF-8 continuation byte). Single-byte fallback to
`<unk>` with a heavy penalty so Viterbi only chooses byte-fallback
when no piece covers the byte.

### 4. The `<2xx>` lang prefix is MADLAD-specific

The CLI adapter unconditionally prepended `<2{tgt_lang}>` to every
translation. MADLAD's vocab has all 419 lang tags as single-piece
entries; flan-t5 / mT5 / etc. don't. Prepending the prefix on those
tokenizes (after Viterbi) as `[▁, <unk>]` (= garbage) at the front
of the encoder input.

Fix: new `t5_has_token(ctx, "<2de>")` C ABI; the adapter probes the
vocab and only prepends the tag when it's a real piece.

### Validation methodology

For each fix, the regression test was: tokens AND output text both
bit-identical between C++ and Python reference. flan-t5-small is the
ideal smoke target — same architecture as MADLAD (T5-1.1 + gated-GELU
+ RMSNorm + bucketed rel-pos + SentencePiece) but ~250 MB so it
loads fast on tight memory. Once flan-t5-small matches, MADLAD
matches by construction (same kernels, same algorithms, larger model).
Save the next round of T5 debugging by validating on flan-t5-small
first.

## 2026-05-05 — ggml fork patches we carry (must re-apply on every ggml bump)

Authoritative inventory of every CrispASR-local change to the vendored
`ggml/` subtree. Most files have a `// CrispASR patch ...` marker so a
mechanical bump (`git subtree pull` or equivalent) won't silently lose
them, but the marker is only a tripwire — the actual fixes are listed
here. **Grep for both `CrispASR patch` AND `CrispASR fork` after every
bump** (the conv-graph patches in (5) below use the latter prefix; an
inventory grep that misses them ships a half-applied F16 fix and crashes
kokoro F16 CPU TTS at `ggml_backend_sched_split_graph`). ggml has lost
our patches twice already: commit `1552434` first added the im2col fix,
`ca6c523` re-applied it after the 0.9.8 → 0.10.0 bump dropped it; the
master bump on 2026-05-05 surfaced (5) as a missed inventory item the
same way.

Mirror this list in `UPSTREAM.md` so future-us (or whoever bumps next)
knows which of these are candidates to send upstream.

### 1. F16 weight × F32 input → F32 dot product (issue #38)

Files: `ggml/src/ggml-cpu/{vec.cpp, vec.h, ggml-cpu.c, simd-mappings.h}`.

Upstream `MUL_MAT` with `src0=F16, src1=F32` quantises the F32 input to
F16 first. F16's dynamic range tops out at ±65504, so any activation
above that saturates to ±Inf and feeds NaN into the next layer. The
qwen3-tts code-prediction path (and several conformer encoders on long
inputs) routinely produces logits well past 65504, masking precision
issues as outright NaN spirals.

Fix: introduce `ggml_vec_dot_f16_f32` that loads the F16 weight, casts
it to F32, and accumulates in F32 — no quantisation step on the input
side. `vec_dot_type` for F16 is set to F32 so `MUL_MAT` takes this path
without an intermediate quantize. ARM NEON path uses `vcvt_f32_f16` +
FMA; AVX2/AVX-512 paths use `_mm256_cvtph_ps` + FMA. Verified on the
qwen3-tts codec head where the saturation surfaced.

Symptom if lost: silent NaN propagation, decoder produces `<unk>` or
locks onto a single token. Won't show up in unit tests if the tests
use small synthetic inputs that stay inside F16 range.

### 2. CUDA `im2col` grid_y > 65535 (`ggml/src/ggml-cuda/im2col.cu`)

Upstream uses `OW` as `block_nums.y` directly. CUDA caps grid Y at
65535; SEANet-style encoders with 11-second 16 kHz inputs land at
`OW = 176000`, busting the cap and triggering an abort *or* (worse) a
silent partial copy depending on driver behaviour.

Fix: clamp `block_nums.y/z` to `MAX_GRIDDIM_Y = 65535` at dispatch and
loop inside the kernel with stride `gl_NumWorkGroups.{y,z}` so each
thread covers `ceil(OW/65535)` output positions. Kernel-internal stride
loop, single launch — possible because `im2col` has no shared-memory
state between iterations.

Symptom if lost: any conv encoder with `T_out > 65535` aborts on CUDA
and the supervisor restarts the process. CPU is unaffected.

### 3. CUDA cpy_scalar_transpose grid_y > USHRT_MAX (`ggml/src/ggml-cuda/cpy.cu`, GH issue #65)

Same class of bug as (2) but in the *transposed* cpy path used by
`ggml_cont(ggml_transpose(...))`. Asserts `grid_y < USHRT_MAX` (= 65535)
inside `ggml_cpy_scalar_cuda`'s transposed branch. The qwen3-tts codec
graph emits `[T_pcm, 1, 1, 1]` tensors with `T_pcm = T_codec * 1920` ≈
2.88M when `QWEN3_TTS_CODEC_GPU=1` and the talker hits its 1500-frame
cap, so `grid_y = ceil(T_pcm/32) ≈ 90,000` busts the assert and the
process aborts with `GGML_ASSERT(grid_y < USHRT_MAX)`.

Fix: tile the launch along the y axis. Add an `int y_block_offset`
parameter to `cpy_scalar_transpose`; the host loops in chunks of
`MAX_GRID_Y = USHRT_MAX-1` and each launch covers a y-slab
`[y_block_offset, y_block_offset + grid_y_this)`. The kernel splices
`blockIdx.y` back onto the offset (`by = blockIdx.y + y_block_offset`).
Bit-identical output, full transposed-tile coalescing preserved per
chunk. Multi-launch (vs the in-kernel-stride pattern used in (2))
because the transposed kernel relies on `__shared__` tile state per
launch — folding the chunk loop inside the kernel would break the
`cur_tile_buf` toggle.

The first attempt (commit `eb9e4a2`) shipped a scalar fallback when
the assert would fire. That worked but threw away the transposed-tile
coalescing entirely on any shape that tripped it; commit `2639461`
replaced it with proper tiling.

Other CUDA-class backends — HIP / MUSA — share `ggml-cuda/cpy.cu` via
the `vendors/` shim and inherit the fix. Vulkan's `copy_transpose.comp`
already has an in-kernel stride loop and is unaffected. Metal's
`kernel_cpy_t_t` doesn't tile and Apple GPUs don't have the 65535 cap.

Symptom if lost: any large-T audio codec / vocoder graph on CUDA
(qwen3-tts, future SNAC/Encodec/XTTS ports) aborts with
`GGML_ASSERT(grid_y < USHRT_MAX)` once `T_codec * upsample_total >
~65535*32`.

### 4. Metal `kernel_conv_transpose_1d` input-range tightening (`ggml/src/ggml-metal/ggml-metal.metal`)

Upstream's transposed conv1d kernel iterates the full IL input range
per output position and filters with `if (...)`. For the qwen3-tts
codec decoder block 1 (IL=320, K=10, s0=5) that's 64× more iterations
than necessary; on M1 with the codec running at full T_codec, the GPU
watchdog fires `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`
and the kernel is killed.

Fix: compute `i_min, i_max` analytically as the input positions whose
kernel weight `k = j - i*s0` lands inside `[0, K)`, then iterate only
that range. At most `ceil(K/s0)` iterations per output position
(typically 2 for stride==K/2 transposed convs). Bit-identical output,
~K/s0× speedup, watchdog-safe.

Symptom if lost: long qwen3-tts (or any large transposed-conv) graph on
Metal triggers GPU watchdog and the command buffer is killed mid-graph.
CUDA / CPU / Vulkan are unaffected — they don't have the same
per-command-buffer watchdog.

### 5. ggml conv graph builders F32 cast (`ggml/src/ggml.c`, issue #38 companion)

Files: `ggml_conv_1d`, `ggml_conv_1d_dw`, `ggml_conv_2d`, `ggml_conv_2d_dw`
in `ggml/src/ggml.c`. Marked `// CrispASR fork:` (different prefix from
the others — this is what the original inventory grep missed).

These four conv graph builders are the partner to (1). After (1) sets
`vec_dot_type = F32` for `GGML_TYPE_F16`, the CPU MUL_MAT path can no
longer accept F16 src1 — `ggml_compute_forward_mul_mat`'s line
`GGML_ASSERT(src1->type == GGML_TYPE_F32)` fires when a non-matching
src1 needs conversion. Upstream's conv graph builders hardcode
`im2col_type = GGML_TYPE_F16` and feed the kernel weight in directly,
producing `MUL_MAT(F16, F16)` — which `ggml_backend_cpu_device_supports_op`
rejects under (1), causing `ggml_backend_sched_split_graph` to abort
with `GGML_ASSERT(*cur_backend_id != -1)`.

Fix: inside each conv builder, pick im2col output type by whether either
side is F32; if im2col is F32 and the kernel is non-F32, `ggml_cast` the
kernel to F32 so the resulting MUL_MAT has F32 src1.

```c
const enum ggml_type im2col_type =
    (a->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
struct ggml_tensor * a_mat =
    (im2col_type == GGML_TYPE_F32 && a->type != GGML_TYPE_F32) ? ggml_cast(ctx, a, GGML_TYPE_F32) : a;
```

Bandwidth cost is real (extra F16→F32 cast per inference pass) but
mirrors what conv_1d already does and is the price of (1)'s correctness
gain.

Symptom if lost: kokoro F16 TTS on `--gpu-backend cpu` aborts at
`ggml_backend_sched_split_graph: GGML_ASSERT(*cur_backend_id != -1)`,
trying to schedule `MUL_MAT(F16 reshape, F16 conv1.weight)` from the
F0N predictor. Also reproduces in any model whose graph runs
`ggml_conv_1d`/`ggml_conv_2d` with F16 weights against F32 (or no-cast)
input on the CPU backend with patch (1) applied.

### Bump procedure

```bash
# Before the bump — grep BOTH marker prefixes
grep -rnE "CrispASR (patch|fork)" ggml/ > /tmp/patches-before.txt

# Do the bump (or replace ggml/{CMakeLists.txt,LICENSE,cmake,include,src} from
# a fresh clone if the subtree wasn't originally added via `git subtree add`)
git subtree pull --prefix=ggml https://github.com/ggml-org/ggml master --squash

# After the bump
grep -rnE "CrispASR (patch|fork)" ggml/ > /tmp/patches-after.txt
diff /tmp/patches-before.txt /tmp/patches-after.txt
# If any patch is missing, find the original commit and cherry-pick the hunk.
```

Anything that disappears from the diff is a patch ggml's master
silently overwrote — re-apply from this list. The five patches above
are the full inventory as of 2026-05-05. Note that (1) and (5) are
**coupled**: applying one without the other crashes kokoro F16 CPU at
`ggml_backend_sched_split_graph`. Always re-apply them together.

### OmniASR-LLM-Unlimited: streaming segment-token protocol

The "Unlimited" variant (`omniASR_LLM_Unlimited_300M_v2`) uses a streaming
protocol where long audio is split into 15-second segments, decoded one at
a time. This uses 3 special tokens allocated above vocab_size in tok_emb:

- `streaming_lang` (vocab_size = 10288): replaces lid_marker in standard model
- `last_segment` (vocab_size + 1 = 10289): signals "this is the final audio segment"
- `regular_segment` (vocab_size + 2 = 10290): signals "more segments follow"

**tok_emb size**: Standard model has vocab_size+1 entries (extra lid_marker).
Unlimited has vocab_size+3 entries (streaming_lang + last_segment + regular_segment).
Auto-detection: `tok_emb->ne[1] == vocab_size + 3`.

**Prefix structure** (per segment):
```
[audio_embs] [streaming_lang] [lang_emb] [segment_marker] [BOS] → generate until EOS
```

The segment marker tells the model whether to expect more audio segments
after this one. EOS=2 is still emitted to terminate each segment's text.
Without the segment marker in the prefix, the model never sees the input
shape it was trained on and generates until max_new_tokens.

**Multi-segment**: Split encoder output at 750-frame boundaries (15s × 16kHz
÷ 320 CNN stride). Each segment gets an independent KV cache and decodes to
EOS. Results are concatenated.

**Critical**: The 476 extra vocab tokens (9812→10287) are NOT segment tokens —
they are additional text tokens in the v2 tokenizer. The 3 segment protocol
tokens sit above the full vocab at indices 10288–10290.

### FastConformer: flash attention with Shaw RPE

The FastConformer self-attention uses Shaw-style relative position
encoding with untied biases:
```
scores = (Q + pos_bias_u) × K^T + rel_shift((Q + pos_bias_v) × R^T)
```

This looks incompatible with `ggml_flash_attn_ext` since the position
bias is query-dependent. However, it CAN be decomposed:
- Precompute BD = rel_shift(Q_v × R^T) — one matmul, unavoidable
- Pass BD × scale as the additive mask to flash_attn_ext
- flash_attn_ext then computes: softmax(Q_u × K^T × scale + mask) × V

This replaces 2 matmuls + add + softmax + 1 matmul with 1 matmul + 
flash_attn_ext. On GPU (CUDA/Vulkan/Metal) the fused kernel avoids
materializing the T×T attention matrix and reduces kernel launches
from 96 to 32 per encoder pass.

Key pitfall: rel_shift returns a strided view — must `ggml_cont` before
`ggml_cast` to F16 for the mask, otherwise `ggml_is_padded_1d` asserts.

### Silero LID label format

Silero LID's `lang_dict_95.json` maps indices to "xx, Name" strings
(e.g. "de, German", "en, English"). The GGUF stores these verbatim in
`silero_lid.lang_strs`. Downstream backends expect bare ISO codes.
Must extract the part before the comma.

## Text LID via fastText — GlotLID-V3 + LID-176 (May 2026)

`src/lid_fasttext.{h,cpp}` ports both fastText supervised LID
families behind one C ABI: GlotLID-V3 (flat softmax, 2102 ISO 639-3
+ script labels, Apache-2.0) and Facebook LID-176 (hierarchical
softmax, 176 ISO 639-1 codes, **CC-BY-SA-3.0** — viral; redistributors
of the .gguf inherit SA). Released as `cstr/glotlid-GGUF` and
`cstr/fasttext-lid176-GGUF` on the Hub.

Forward path is manual F32/F16 + on-the-fly dequant via
`ggml_get_type_traits(type)->to_float`, no graph. The compute is
~1 MFLOP per call; a graph would be pure overhead. K-quants land via
`crispasr-quantize` post-processing.

### `</s>` row injection in fastText supervised mode (the trap)

fastText's `Dictionary::getLine` injects an `</s>` (always
`input_matrix[0]`) at end-of-stream in supervised mode, and
`initNgrams` skips subword expansion for it so its precomputed row
list is just `[0]`. `model.f.tokenize(text)` does NOT return it — a
manual reproduction that just iterates over tokenized words gets 11
row IDs for `"the"` while `model.get_sentence_vector("the")` mean-pools
12 rows.

**Symptom**: cosine vs `model.get_sentence_vector` lands ~0.973
instead of 1.0, with a non-constant ratio across dimensions (so it's
not a divisor mistake — it's a missing row). Appending row 0 to the
input list before mean-pooling brings cosine to 1.0 within float32
epsilon. The C++ port in `src/lid_fasttext.cpp` defines a named
`kEosRowId = 0` constant for this; the converter and reference dumper
both must match.

This bites every fastText port. If you use `m.get_subwords(word)` to
build the input list and skip the trailing `</s>`, you end up with
~3% accuracy loss that looks like quantization noise but isn't.

### Hierarchical softmax — code-bit sign convention `(2c-1)·f`

LID-176 uses HS (`loss=hs`), not flat softmax. The "output matrix"'s
176 rows are NOT per-label scoring vectors; they parameterize internal
nodes of a Huffman tree built deterministically from training-frequency
label counts. Per-label log-probability:

```
log P(label_i) = Σ over (node, code) in path[i]:
    log_sigmoid((2*code - 1) · (output[node] · hidden))
```

The right child gets `binary=1` in `Dictionary::initNgrams`, so walking
right at training time means code=1 and contributes `log_sigmoid(+f)`;
walking left contributes `log_sigmoid(-f)`. The unified formula is
`(2c-1)·f`, **not** `(1-2c)·f`. Sign-flipping makes predictions land
in the wrong subtree of the root — fastText says `fr/0.95` but your
port says `en/0.88` for "Bonjour le monde". The cosine-vs-embedding-
bag stage stays 1.0 (loss-agnostic), so the bug only surfaces at
top-1 — verify by hand on a single short input first.

The Huffman tree is **not stored in `model.bin`** — fastText rebuilds
it at load time from the dictionary's per-label `count` field via
`Model::buildTree`. fastText-python doesn't expose label counts, so
the GGUF converter parses `lid.176.bin` directly (header layout in
`src/dictionary.cc::save`) to extract them, then ports `buildTree`
inline. GGUF schema additions:

```
lid_fasttext.loss              str    "softmax" | "hs"
lid_fasttext.hs_path_offsets   i32[n_labels+1]   # CSR-style
lid_fasttext.hs_paths          i32[total_steps]
lid_fasttext.hs_codes          i8 [total_steps]
```

Memoize internal-node dot products inside the per-label loop —
deeper labels share most of their path with siblings. For LID-176
the average path length is 10.54 over 176 labels.

### `crispasr-quantize`'s `is_weight` gate — tensor naming matters

`examples/crispasr-quantize/main.cpp` only quantizes tensors whose
names contain `"weight"` or end in `_w`:

```cpp
bool is_weight = (sname.find("weight") != npos) ||
                 (sname.size() >= 2 && sname.substr(sname.size()-2) == "_w");
```

Tensors that don't match the gate get `f16, copying... done` —
silently passed through as F16 instead of quantized. The output GGUF
has the same byte count as the F16 input minus a few metadata bytes,
which is the most confusing diagnostic surface possible.

**Symptom**: `crispasr-quantize input.gguf output-q4_k.gguf q4_k`
"succeeds" with output the same size as input. Look at the tool's
log lines: `quantizing to q4_K... done` is good; `copying... done`
means the tensor failed the gate.

**Fix**: name the tensor `<prefix>.weight` (kokoro/parakeet/voxtral
convention). The lid-fasttext converter writes
`lid_fasttext.embedding.weight` and `lid_fasttext.output.weight`;
the runtime loader keeps a backward-compat fallback to bare
`lid_fasttext.embedding` / `lid_fasttext.output` since the first
release used those names before this trap was identified.

### Q4_K below 0.999 cosine floor on intermediate stages, top-1 stable

The diff harness tracks per-stage cosine against a Python F32
reference. For GlotLID quants:

| Quant | embedding_bag cos | logits cos | softmax cos | top-1 |
|-------|-------------------|------------|-------------|-------|
| F16   | 1.000000          | 1.000000   | 1.000000    | exact |
| Q8_0  | 0.999990          | 0.999991   | 1.000000    | exact |
| Q6_K  | 0.999898          | 0.999883   | 1.000000    | exact |
| Q5_K  | 0.999440          | 0.999530   | 1.000000    | exact |
| Q4_K  | 0.998303          | 0.998356   | 1.000000    | exact |
| Q4_0  | 0.997176          | 0.997357   | 1.000000    | exact |

Q4_K and Q4_0 fail the standard 0.999 cosine floor on `embedding_bag`
and `logits`. But softmax compresses the logit noise by ~3-4 orders
of magnitude (max_abs goes from 1.6 on logits to 6.1e-5 on softmax),
so the top-1 prediction is identical to F16 across an 8-language
multilingual smoke test. **Conclusion**: for shallow LID classifiers,
the conventional 0.999 cosine threshold is conservative — softmax
absorbs much more quant noise than it does for deep transformer
stacks. Q4_K is functionally fine; ship it but document the
intermediate-stage drift.

(Contrast with kokoro: Q4_K breaks the German backbone via the
predictor LSTM accumulating over multiple steps. Shallow networks
tolerate quant noise better.)

### `gguf` Python's K-quant gap — fallback to `crispasr-quantize`

`gguf.quantize()` only handles F32, F16, BF16, Q8_0. K-quants
(`Q5_K`, `Q4_K`, `Q6_K`, etc.) raise `NotImplementedError` from
`gguf/quants.py:129`. The right path for K-quants is to write F16
from the converter, then re-quantize via `crispasr-quantize` (which
calls into `ggml_quantize_chunk`). The C++ runtime side is
type-agnostic via `ggml_get_type_traits(type)->to_float`, so any
quant produced by `crispasr-quantize` "just works" without runtime
changes.

### LID-176's `dim=16` makes K-quants unproductive

K-quants need 256-element row alignment. LID-176 has `dim=16`, so
`crispasr-quantize` falls back to legacy Q4_0/Q5_0/Q8_0 for those
rows. Combined with the model already being 63 MB at F16 (the input
matrix is `2,040,010 × 16`), quants don't save meaningful space.
Ship F16 only for LID-176; quants make sense for GlotLID
(`1,634,361 × 256` = 1.6 GB at F32).

### `crispasr-quantize` is the right tool, not `whisper-quantize`

`build/bin/whisper-quantize` is the legacy whisper-binary-format
quantizer. It rejects GGUF input with `bad magic`. The GGUF-aware
tool is `crispasr-quantize` at `examples/crispasr-quantize/main.cpp`,
not previously discoverable from a casual `find -name "*quantize*"`.
Always check `build-ninja-compile/bin/crispasr-quantize` first; the
help text shows the supported quant types.

### HF upload — `hf upload-large-folder` with symlink staging

Per `.claude/CLAUDE.md`'s recipe: stage GGUFs as symlinks under
`/tmp/hf-staging-<repo>/` and run `hf upload-large-folder
cstr/<repo>-GGUF .` — uploaded 1.8 GB across 4 GlotLID quants in
9:42 wall time on the first attempt, including Xet pre-upload + commit.
Smaller LID-176 (63 MB) finished in under a minute.

The `--include` flag accepts multiple patterns (`"*.gguf" "*.md"`).
The CC-BY-SA-3.0 SA notice for LID-176 must be surfaced in the
README's metadata block AND in a license-warning section above the
Files table — downstream redistributors of the GGUF inherit the
SA terms, which is more restrictive than most ASR/LID models on
the Hub.

### CLD3 (`google/cld3`) is a separate architecture, not n=1,2,3 bags

The brief described CLD3 as "separate hashed embedding bags for
n ∈ {1,2,3}, concatenated → FC + ReLU → softmax". The actual model
in `src/cld_3/lang_id_nn_params.cc` (1.76 MB embedded weight file)
declares **six** embedding columns, not three:

```cpp
const int LangIdNNParams::kEmbeddingsNumRows[] = {1000, 5000, 12, 103, 5000, 100};
const int LangIdNNParams::kEmbeddingsNumCols[] = {16, 16, 8, 8, 16, 16};
const int32 LangIdNNParams::kConcatOffsetValues[] = {0, 16, 32, 40, 48, 64};
```

Six features → concat to 80-d → hidden + softmax over ~107 langs.
Weights stored as `float16` (Google's specific representation, as
`uint16` literals in C++ source). Each feature is a different
extractor (probably {char-1grams, char-2grams, script-type,
relevant-script, char-3grams, punctuation} per CLD3's feature
function registry). Porting CLD3 is meaningfully more work than
porting fastText was — six feature extractors + their text-normalisation
quirks live in `src/feature_extractor.{h,cc}` and the full set isn't
trivially expressible without building (or vendoring) libcld3.

Pragmatic plan if CLD3 becomes priority later: parse
`lang_id_nn_params.cc` directly via regex (the floats are in
plain-text C++ array literals — skip the `float16` ones, decode the
`uint16` literals via `ggml_compute_fp16_to_fp32`); replicate the
six feature extractors from `src/feature_extractor.cc` in C++; port
the matmul + ReLU + softmax. Full session of work, not an
afternoon-add-on.

**Update (May 2026):** CLD3 shipped — see the next section,
"## Text LID via CLD3", for the actual port. The plan above held
up: regex-parse → six feature extractors → matmul + ReLU + softmax.
The traps were elsewhere (bf16-style float16, MurmurHash2 not
CityHash, ULScript values guessed wrong).

## Text LID via CLD3 — Google compact language detector (May 2026)

`src/lid_cld3.{h,cpp}` ports Google's CLD3
([github.com/google/cld3](https://github.com/google/cld3),
Apache-2.0) — a tiny shallow classifier (~1.5 MB F32 / ~440 KB F16)
that emits 109 ISO 639-1 labels. Six feature extractors (4 cbog
char-ngram bags at sizes 1/2/3/4, RelevantScript, Script) →
80-d concat → FC + ReLU → 208-d hidden → FC → softmax. Released as
`cstr/cld3-GGUF` on the Hub. Sibling backend to lid-fasttext; the
post-merge auto-routing dispatcher picks between them via the GGUF's
`general.architecture` key.

Forward path is pure manual F32 (no ggml graph) — the compute is
well under 1 MFLOP per call (one 80×208 matmul + one 208×109 matmul +
softmax). F16 weights are dequantized to F32 once at load time via
`ggml_fp16_to_fp32`; the 1.5 MB RAM hit is trivial.

### CLD3's `float16` is bfloat16-style, NOT IEEE binary16

CLD3 ships its weights in `src/lang_id_nn_params.cc` as 1.76 MB of
plain-text C++ array literals — no binary side-car. The `float16`
typedef in `src/float16.h:43-49` is **the upper 16 bits of a binary32
float** (1 sign + 8 exponent + 7 mantissa, i.e. bfloat16-style),
NOT IEEE 754 binary16 (1+5+10). The header even calls this out
explicitly: *"NOTE: The IEEE floating point standard defines a
float16 format that is different than this format..."*.

**The trap**: decoding the `15392u`-style literals via numpy `<f2`
silently produces garbage — the bit patterns are a different format.
The correct decode is `(uint32(value) << 16).view(float32)`. This is
the dominant cause of "weights load but produce nonsense" — it looks
correct, just shifted into wrong magnitude.

This is also why `embedding_network.cc:122-123`'s dequant formula is
`(static_cast<int>(uint8) - 128) * multiplier` (where `multiplier`
includes the bf16-style scale): symmetric quantization with bias 128.
Got the bf16 decode wrong → the dequantized embedding rows look like
random numbers, the cbog feature contributions cancel out, and the
softmax is uniform.

### Hash function is MurmurHash2-32 with seed 0xBEEF, NOT CityHash

The CLD3 brief tentatively guessed CityHash because the upstream code
includes `absl_city`. Wrong: `utils.cc:137-183`'s `Hash32` is textbook
MurmurHash2-32 with `m=0x5BD1E995, r=24`, default seed `0xBEEF` (=
48879). Trivial 30-line port to numpy and C++.

The cbog feature IDs use the raw UTF-8 bytes of the ngram string as
the hash input (no UTF-8 normalisation), so byte-for-byte hash parity
with upstream is mandatory. Off-by-one or signed-vs-unsigned errors
in the hash get amplified through the embedding lookup and produce
top-1 mismatches across every multilingual input.

### Hiragana, Katakana, Hangul are NOT separate ULScript values

The brief estimated ~107 languages and called out 6 distinct feature
extractors. The actual model has **109 labels** (in
`task_context_params.cc:43-57`, NOT `lang_id_nn_params.cc`) and
**3 feature classes instantiated 6 times** — `ContinuousBagOfNgramsFunction`
×4 with different `id_dim`/`size`, `RelevantScriptFeature` ×1,
`ScriptFeature` ×1. The cbog instantiation at index 5 is the
1-character "unigram" with `id_dim=100`.

The bigger trap was in the **103-row text-script embedding**. The
ULScript enum at `script_span/generated_ulscript.h` has 102 values
running 0..101, with `NUM_ULSCRIPTS = 102` as a sentinel. **It does
NOT include Hiragana, Katakana, or Hangul as separate values** —
they all return `ULScript_Hani = 24` from the upstream
`ScriptScanner`, then a *secondary* Hangul-vs-Hani codepoint count
in `ScriptFeature::Compute` (language_identifier_features.cc:128-161)
returns the `NUM_ULSCRIPTS=102` sentinel only when Hangul-script
codepoints outnumber non-Hangul ones in the same span.

Early Python-port smoke-set failures traced directly to guessed
values:

  | Input                | Symptom                  | Cause                           |
  |----------------------|--------------------------|---------------------------------|
  | `नमस्ते दुनिया` (Hindi)| top1 = `bn` (Bengali)    | We mapped Devanagari→10; 10=Bengali. Correct: 9. |
  | `你好世界`            | top1 = `mr` (Marathi)    | We mapped Hani→43. Correct: 24. |
  | `こんにちは世界`        | top1 = `bg` (Bulgarian)  | We mapped Hiragana→41 (doesn't exist). Correct: Hani=24, Hangul-vs-Hani fixup leaves it as Hani. |

Read `script_span/generated_ulscript.h` first; do NOT guess these
values. The 103rd row (sentinel) is the special `NUM_ULSCRIPTS` slot.

### Full-Unicode lowercase in cleanup is non-optional

Upstream's `ScriptScanner::GetOneScriptSpanLower` lowercases ALL
letters across all scripts (Cyrillic П→п, Greek Α→α, Latin H→h)
before feeding the text to the feature extractors. ASCII-only
lowercase changes the bytes that get hashed by MurmurHash2 → different
ngram IDs → softmax lands on the wrong sibling label. Specifically,
`Привет мир` lowercased only as ASCII keeps the uppercase Cyrillic
codepoints, hashes their UTF-8 bytes, and predicts `tg` (Tajik)
instead of `ru` (Russian). Both are Cyrillic — same script feature —
but the cbog ngrams diverge.

The C++ port covers this with a hand-rolled case-fold table in
`src/lid_cld3.cpp::lower_codepoint` covering Latin / Latin-1 / Latin
Extended-A / Greek / Cyrillic / Armenian. Codepoints not in the
table fall through unchanged. ICU would handle the long tail
correctly but adding ICU as a dependency for one tiny LID model is
a poor tradeoff.

### Simplified text cleanup vs vendoring `script_span/`

Upstream's full preprocessing pipeline runs the input through
`ScriptScanner::GetOneScriptSpanLower` → `CheapSqueezeInplace` →
`SelectTextGivenBeginAndSize` (snippet selection if too long).
That's a ~250 KB Unicode state machine in `script_span/` (4 generated
UTF-8 transition tables at 40-82 KB each, plus orchestration).

We chose to ship a **simplified cleanup** (full-Unicode lowercase via
hand-rolled case-fold + ASCII punct/digit strip + whitespace collapse)
in `cleanup_text`. On the 8-input multilingual smoke set (clean
single-script inputs), this matches upstream byte-for-byte for 7/8.
The 8th, "Hello world", is a low-confidence (<0.5) underdetermined
input where small algorithmic differences flip the argmax (we say
`fi`, pycld3 says `ky` — both clearly wrong). The Python reference
dumper downgrades that case to a warning when both predictions are
below the 0.7 reliability threshold and proceeds with the dump, so
the C++/Python cosine gate still measures the algorithmic agreement
of *our* port, which lands at cos=1.000000 across every stage on
F16.

If divergence ever shows up on a confident input (>0.7 prob both
sides, different argmax), escalate to vendoring the full
`script_span/` tree from `/Volumes/backups/ai/upstream/cld3/src/`.
It's Apache-2.0 so distribution is fine; it's just a lot of generated
code mass to carry in-tree.

### `[in_dim, out_dim]` storage → transpose at conversion time

Upstream stores hidden + softmax weights as `[in_dim, out_dim]`
row-major (because `SparseReluProductPlusBias` iterates `x[i] * weights[i]`,
where `weights[i]` is row `i` mapping input dimension `i` to all
outputs). GGUF's `ggml_mul_mat(W, x) = y` convention is `[out_dim,
in_dim]`. The converter in `models/convert-cld3-to-gguf.py` transposes
once at write time so the runtime can do `W @ x` with no orientation
check at every load. Forgetting this transpose produces a softmax
of all NaN — the FC outputs become `concat[80] @ W[80,208] = wrong-dim`
and the bias add silently strides off the buffer.

### CLD3's "Hello world" is `ky` — known short-input quirk

CLD3 trained on web-scale data and gives short ambiguous inputs
their statistically-most-frequent-language guess. "Hello world" is
short enough that it lands on `ky` (Kyrgyz) consistently across
every variant (`'hello world'`, `'Hello world!'`, `'  Hello world  '`,
etc.) at p=0.7192. This isn't a bug in our port — it's the
contract. The diff harness's top-1-match check therefore needs the
reference dumper's `top1_label` field as the source of truth; you
can't paper over short-input quirks by feeding longer inputs at
diff time.

---

## IndexTTS-1.5 TTS backend

### HuggingFace GPT-2 has TWO final LayerNorms

IndexTTS uses HuggingFace's GPT2Model, which applies a built-in
`gpt.ln_f` LayerNorm after all transformer blocks. IndexTTS then
applies its own `final_norm` on top. Missing `gpt.ln_f` caused
a 2.0 logit gap, demoting the correct first mel token from rank 0
to rank 11. Both norms must be loaded into the GGUF and applied:
transformer blocks → gpt.ln_f → final_norm → mel_head.

### HF generate skips mel position 1

During HuggingFace's `generate()` loop, the mel position embedding
for generated tokens is computed as `attention_mask.shape[1] - mel_len`.
Because `fake_inputs` has `mel_len + 1` tokens (with start_mel at
the end), the first generated token after prefill gets mel_pos[2]
instead of mel_pos[1]. Position 1 is never used during inference.
The sequence is: start_mel=pos[0], gen_tok_0=pos[2], gen_tok_1=pos[3], etc.
This is a train/inference mismatch in IndexTTS but the model works
with it, so the C++ must match: `mel_pos = beam.tokens.size() + 1`.

### Latent extraction: mel_logits[:, :-2] gives (n_mel + 1) positions

Python's `forward(return_latent=True)` internally prepends start_mel
and appends stop_mel, then strips the last 2 positions. The result
has (n_mel + 1) positions: [start_mel_hidden, c1_hidden, ..., ck_hidden].
The C++ latent pass must extract this exact count, not n_mel.

### Reference dump tool was missing gpt.ln_f

The simplified reference dump in `tools/reference_backends/indextts.py`
manually reimplements GPT-2 but originally skipped `gpt.ln_f`,
producing wrong reference values. Any manual reimplementation of
HuggingFace GPT-2 must include `gpt.ln_f`.

### Conformer RelPositionalEncoding does NOT add pos_emb to x

WeNet/ESPnet Conformers have two positional encoding classes:
- `PositionalEncoding.forward`: returns `(x * xscale + pos_emb, pos_emb)`
- `RelPositionalEncoding.forward`: returns `(x * xscale, pos_emb)`

IndexTTS uses `RelPositionalEncoding`. The `pos_emb` is passed
separately to the attention layer as the R matrix — it is NEVER
added to the input x. Adding it corrupts every subsequent block.

### Conformer attention: absolute pos table, no rel_shift

IndexTTS's Conformer uses `RelPositionMultiHeadedAttention` but with
a critical deviation from the paper: `rel_shift` is commented out
("useless in speech recognition"). The position table is the stored
`pe[:, 0:T]` — T absolute positions, NOT a 2T-1 relative table.
The `matrix_bd = Q_v @ R^T` is already a T×T square matrix and
needs no shift. Using a 2T-1 sinusoidal table + rel_shift corrupts
attention scores in all 6 blocks.

### "Match Python's typical norm" speaker-embedding clamp was masking the bug, not fixing it (Issue #75, May 2026)

The original code rescaled the ECAPA speaker embedding to L2 norm = 0.9
with a comment claiming it "matches Python's typical magnitude" and
"prevents the vocoder from being overwhelmed". User #75 reported
"abnormal-sounding voice" — ASR roundtrip of an English test sentence
turned "indextts speech synthesis system" into "index piece since its
system" (~25% CER, speech recognizable but timbre/articulation distorted).

After A/B-testing with `INDEXTTS_SPK_NORM=raw` (pass-through, matches
upstream) vs `INDEXTTS_SPK_NORM=0.9` (legacy clamp):

| Mode | English ASR | Audio peak |
| --- | --- | --- |
| clamped to 0.9 | "test of the index piece since its system" | 0.06 |
| raw (upstream) | "test of speech synthesis" ✓ | 0.21 |

Upstream BigVGAN.forward() consumes the unnormalized ECAPA output
directly — there is no L2 norm anywhere in the pipeline. The clamp
shrank `cond_layer(spk_emb)` and `conds[i](spk_emb)` projections by
2.84×, leaving the BigVGAN underconditioned and the audio attenuated.

Lesson: a magic constant ("0.9") with a comment about "matching the
typical value" is almost never a fix — it's a workaround for a bug
you haven't found yet. If our ECAPA produced 3× the Python norm,
the right answer was to find the BatchNorm/scale bug in ECAPA,
not to clamp the output. Removing the clamp made English ASR
roundtrip clean. (Chinese ASR is still degraded — that's a separate
bug, likely in tokenization or conditioning encoder.)

Fix path: env knob `INDEXTTS_SPK_NORM=<float|"raw">`, default raw.

### Beam search KV snapshots round-trip through host (B=1 is 2.2× faster)

The IndexTTS GPT beam decode keeps a single on-device KV cache and
snapshots it per beam via `ggml_backend_tensor_get/set` (GPU → host →
GPU). With 24 layers × 1280 dim × 676 max tokens × 2 (K and V), the
KV cache is ≈ 158 MiB. Per step per beam we move 2 × 158 MiB; for
the issue-75 test (158 mel codes × B=3) that's ~146 GB of host
round-trips just for KV swapping.

Symptoms in the issue: 56.7s total, vocoder reports 2.5s, so 54s
is GPT decode — much of which is memcpy, not compute.

Workaround: `INDEXTTS_BEAM_SIZE=1` skips snapshot/restore entirely
(greedy decode keeps one KV slot resident on device). For an English
test sentence the wall time dropped 75.8s → 34.4s (2.2× faster) and
the ASR roundtrip stayed clean. Default is still B=3 to preserve
parity with the Python reference.

Proper fix would be to keep B KV slots resident on the backend and
swap them with `ggml_backend_tensor_copy` (device → device). Tracked
as follow-up; not in this commit.

### Follow-up: the "host round-trip" framing oversold the bottleneck on Apple Silicon (May 2026)

Implemented the device-resident path: B per-beam KV slots allocated on
the same backend as the active KV, swap via `ggml_backend_tensor_copy`,
refcount-based slot recycling on candidate selection (siblings sharing a
parent get a `tensor_copy` into a freed slot; only-children just inherit
the parent's slot pointer). Opt-in via `INDEXTTS_KV_DEVICE_COPY=1`,
default off.

M1 Metal, B=3, "Hello world. This is a test of speech synthesis."
(121 mel codes), 3 trials each, warm cache:

| Trial | host (`_get`/`_set`) | device (`_copy`) |
| --- | --- | --- |
| 1 | 61.25 s | 61.67 s |
| 2 | 55.12 s | 72.73 s |
| 3 | 70.32 s | 61.66 s |
| median | **61.25 s** | **61.67 s** |

Median delta < 1 %, within noise (each binary has ~15 s trial-to-trial
spread on this box, dominated by thermal / scheduler variance). Audio
output is byte-identical between the two modes and against the
pre-refactor binary across both English and Chinese prompts.

The earlier "75.8 → 34.4 s with `INDEXTTS_BEAM_SIZE=1`" speedup we
attributed mostly to "memcpy elimination" was in fact mostly **3× less
GPT compute** (B=1 runs one forward per step vs B=3's three) — the
snapshot traffic on Apple Silicon unified memory rides shared RAM and
costs about as much as a same-backend Metal blit. The original framing
oversold the memcpy cost.

Device mode is kept as opt-in for the case it's actually load-bearing
— discrete-GPU backends (CUDA, Vulkan) where `_get`/`_set` traverses
real PCIe. Needs measurement on those backends before promoting the
default; not flipping it on Metal alone.

Lesson: when a "2.2× speedup" comes from a binary on/off comparison
(B=3 vs B=1), the savings can be from any one of three sources —
fewer forward passes, less per-step CPU work, less memcpy. Don't
attribute the whole win to the one you happened to hypothesise about.

### Chinese requires `tokenize_by_CJK_char` before SentencePiece (Issue #75 follow-up, May 2026)

The C++ text path passed raw text straight to its SentencePiece
Viterbi. Upstream `TextTokenizer.encode()` runs THREE stages: (1)
`TextNormalizer.normalize` (char_rep_map, mostly CJK punctuation →
ASCII), (2) `tokenize_by_CJK_char` (regex `re.split` that surrounds
every CJK codepoint with whitespace), (3) `sp.Encode`. With the CJK
splitter, every Chinese character ends up as its own piece prefixed by
`▁` (id 10201), and SentencePiece chooses pieces that the model was
actually trained on. Without it, `而在获取电压函数中…` is glued into one
run and the Viterbi produces ~29 tokens of approximate matches —
intelligible neither to the model nor to a downstream ASR.

For the issue-75 prompt `而在获取电压函数中，我们总是先检查空指针再访问任何参数。`
the difference is:

| Mode | Tokens | Qwen3-ASR roundtrip | CER (punct-stripped) |
| --- | --- | --- | --- |
| C++ raw → sp.Encode | 29 | `而在曲电韩宿中飞我们总是减控指针在问任何数飞` | ~50% |
| upstream pipeline | 54 | `而在获取电压函数中，我们总是先检查空指针，再访问任何参数。` | **0.0%** |

**Trap: `。` (U+3002, full-width period) sits inside the CJK Unicode
range** (`U+2E80-A4CF`) that `tokenize_by_CJK_char` splits on. If the
punct table runs before the CJK splitter (upstream order), `。` is
already `.` when the splitter sees it, so the splitter ignores it and
SentencePiece emits one `▁.` piece (id 10203) — the sentence-end token
the model was trained to react to. If `。` is left as a CJK char, it
gets split as `▁` + `。` (10201 + 2), and the model hallucinates one
extra trailing syllable. Whisper-base hid this artefact (it can't
reliably transcribe Chinese in the first place); Qwen3-ASR-0.6B
revealed it as the user-audible "extra word at the end".

Implementation: `src/indextts.cpp:preprocess_indextts_text` is a
single-pass UTF-8 codepoint walker that does whitespace collapse →
punct-map (matching upstream's `char_rep_map` for the common
substitutions, including `。 → .`) → CJK-split → ASCII upper. Verified
the C++ token IDs are bit-identical to Python `sp.Encode` on the
preprocessed text.

What we deliberately skipped: the full wetext `zh_normalizer`
(numbers → hanzi, English contractions, pinyin tone digits). That
needs a real rule engine (or the wetext FST itself); not on the path
for plain Chinese roundtrip.

**ASR methodology corollary**: whisper-base over-counted the CER
~5× compared to Qwen3-ASR-0.6B on the same audio (21% vs 3.8% after
the first-pass fix). When validating CJK output, the ASR is the
load-bearing variable — pick one with real Chinese training data
(Qwen3-ASR, Cohere-transcribe, whisper-large) or the numbers will
chase artefacts that are in the recogniser, not the synthesiser.

### BigVGAN SnakeBeta: ggml memory layout is [c*T + t], not [t*C + c]

The anti-aliased SnakeBeta activation runs as a custom `ggml_map_custom1`
op. For a tensor with `ne[0]=T, ne[1]=C`, ggml stores element (t, c)
at offset `c * T + t` — channel-major, time innermost. Accessing as
`data[t * C + c]` (which is row-major [T, C]) scrambles channels
across time, producing noise instead of speech. This single bug was
the root cause of the vocoder producing unrecognizable audio.

### BigVGAN latent input: GPT output layout vs ggml tensor layout

The GPT latent extraction outputs `ne[0]=D, ne[1]=n_positions`, meaning
each position's D=1280 values are contiguous (row-major [pos, D]).
The vocoder's `ggml_conv_1d` needs `ne[0]=T, ne[1]=C` with time innermost.
Copying raw bytes from the GPT tensor into a `(T, C)` ggml tensor
transposes the data — channels become time and vice versa. Fix:
create the input tensor as `(D, T)` matching GPT layout, then
`ggml_transpose` + `ggml_cont` before conv operations.

### Reference audio sample rate must be checked

The `compute_ref_mel` function assumed 16kHz input and always
resampled to 24kHz. If the reference WAV is already 24kHz (common),
this double-resamples and destroys the signal. The CLI must check
the WAV sample rate and only resample when needed.

### Center-padding: reflect vs zero (the mel spectrogram root cause)

The core_mel `center_pad` option originally used **zero-padding** but
torchaudio's `center=True` uses **reflect-padding**. This caused the
first 2 STFT frames to differ from Python, propagating through the
6-layer Conformer into conditioning (norm 288.88 vs 288.92), then
compounding through 24 GPT layers to flip a beam search token at step 14
(6283 instead of 6109).

The hypothesis that F16 weight precision caused the gap was WRONG — an
F32 GGUF produced identical conditioning values. The actual root cause
was padding mode. Fix: added `center_pad_reflect` to `core_mel::Params`.

Lesson: when mel values at t=0 and t=1 differ from Python but t≥2 match
exactly, always check center-padding mode — it's reflect vs zero.

### HF beam search applies repetition penalty AFTER log_softmax

HuggingFace's `_beam_search` computes log_softmax on raw logits FIRST,
then passes the log-probabilities through `logits_processor` (which
includes `RepetitionPenaltyLogitsProcessor`). Since log-probs are always
≤ 0, the penalty multiplies them (making them more negative).

The C++ originally applied rep penalty to raw logits BEFORE log_softmax,
which changes beam dynamics. With `repetition_penalty=10.0`, this causes
different beam paths to win. Fix: compute log_softmax first, then apply
penalty to log-probs, matching HF's exact order.

### BigVGAN v2 SnakeBeta needs anti-aliasing — "negligible" was wrong (May 2026)

The raw SnakeBeta activation `y = x + sin(α·x)² / β` is not band-limited:
`sin²` introduces harmonics at 2α·x, 4α·x, … For trained α values in
BigVGAN v2 (the `_post` layer in particular has large α), those harmonics
land well above Nyquist and fold back as broadband click/buzz. The
original BigVGAN v2 paper wraps the activation in 2× upsample → activate
→ 2× downsample (Kaiser-windowed sinc) specifically to suppress that
aliasing.

CrispASR's first cut omitted the AA "because `ggml_conv_1d_dw` has no
CUDA kernel, and the quality impact on TTS speech is negligible." The
quality claim was just wrong — on the JFK-cloned "quick brown fox …"
prompt the raw path produced ~2 000 sample-to-sample jumps over 30 % FS
and several over 100 % FS (physically impossible in a band-limited
24 kHz signal), audible as steady click/buzz across every quant. The AA
path measured 0–27 such jumps, max\|Δ\| ≤ 0.38 — clean speech.

How we caught it: `np.diff(wav)` is the cheapest aliasing detector we
have. Any `|Δsample|` exceeding `2·sin(π·f_max/fs)` for the band-limit
`f_max` is impossible without aliasing. For 24 kHz 16-bit speech with
voice content roughly below 12 kHz, `Δ > 0.5` is a hard ceiling; counts
in the thousands == broken.

What we shipped (`src/indextts_voc.cpp`):

1. **AA is the default.** `INDEXTTS_VOCODER_RAW=1` (or `_AA=0`) opts back
   into the aliased path for benchmarking. We kept the raw path because
   it's the only fully-GPU-graphable activation we have today.
2. **Pre-allocated thread-local scratch.** The original AA op allocated
   three `std::vector<float>` per channel per call — at 1536 ch × 24
   layers per generate, that was ~37 k mallocs of 100 KB+ on the hot
   path. Lifted to per-thread (`ith`-indexed) scratch sized lazily on
   first use; `GGML_N_TASKS_MAX` is the `-1` sentinel, not a count, so
   we cap at 64 threads explicitly (`AA_SCRATCH_MAX_THREADS`) and pass
   that to `ggml_map_custom1` as the task hint.
3. **Pre-scaled upsample filter.** Multiply the FIR taps by 2 once at
   init (cancels the zero-stuff gain), saving one mul in the inner loop.
4. **`memcpy`/`memset` for the edge-replication padding** — small, but
   the inner-loop trace was dominated by the C++ per-element copies the
   original used.

Result on M1, JFK voice prompt, ≈ 6.7 s of audio:

| Vocoder config          | Δ>0.3 | max\|Δ\| | voc-only |
| ----------------------- | ----- | -------- | -------- |
| RAW / CPU (aliased)     | 1671  | 0.89     | 6.36 s   |
| RAW / GPU (aliased)     | 2080  | 1.04     | 7.12 s   |
| **AA / CPU (default)**  | **2** | **0.31** | 6.65 s   |
| AA / GPU                | 26    | 0.38     | 8.52 s   |

`AA / CPU` is the new sweet spot — only ~5 % slower than the broken-but-
fast RAW / CPU. `AA / GPU` is slowest because the `ggml_map_custom1` op
forces a Metal → CPU → Metal sync at every AMP block; if the rest of the
vocoder graph is on Metal we trade GPU-friendly matmuls for cross-device
copies and net-lose. Best operational answer until someone ports the AA
sandwich to native ggml ops: tell IndexTTS users to pass `--no-gpu` if
the prompt is short, or accept the ~25 % vocoder slowdown for the
voice-cloning convenience of keeping the GPT on GPU.

Lesson: comments claiming "negligible quality impact" age badly. When a
paper introduces a deliberate signal-processing stage, it's almost
always there for a reason; if you remove it, prove the absence of harm
with `np.diff` or a spectrogram, not assertion.

### Mixed-backend custom ops in ggml are a perf trap (BigVGAN AA, May 2026)

`ggml_map_custom1` runs CPU-only. ggml-backend-sched faithfully routes the
op back to CPU even when the rest of the graph is on Metal — but it costs
a Metal → CPU → Metal sync per op site. For IndexTTS BigVGAN with ~20
SnakeBeta sites per generate, that overhead dominates: GPU + AA measured
≈ 25 % SLOWER than CPU + AA on M1 (8.5 s vs 6.65 s vocoder).

Step A fix (`src/indextts_voc.cpp:indextts_voc_init`, commit `cd21faea`'s
follow-up): when `use_aa = true`, override `use_gpu` and force the whole
vocoder onto CPU. The GPT runs on its own backend; only the AMP-block
chain pays the per-AMP-cost, and skipping the round-trip is the win.

Override knob is `INDEXTTS_VOC_FORCE_GPU=1` for the people who want to
reproduce the mixed-backend benchmark; default does the right thing.

Lesson: if you reach for `ggml_map_custom1`, the right next step is
*always* a real ggml op + Metal kernel — never assume the custom op is
"only a tiny fraction" of the graph; the surrounding GPU stalls dominate
once it's called from a hot loop.

### Polyphase "zero-stuff + conv_1d" doesn't trivially port a torch conv_transpose_1d (May 2026)

Attempted the native-ggml-ops AA path for IndexTTS (Step B in the
optimisation plan). The idea: replace `ggml_map_custom1` with a
ggml-graph that's identical math to the torch reference, expressed via
`ggml_conv_1d` for both the upsample and downsample stages.

PyTorch reference (`indextts/BigVGAN/alias_free_activation/torch/resample.py`):
```python
x = F.pad(x, (pad, pad), mode='replicate')
x = ratio * F.conv_transpose1d(x, filter.expand(C,-1,-1), stride=2, groups=C)
x = x[..., pad_left:-pad_right]
```

Two blockers:

1. **Output-length mismatch.** `conv_transpose_1d(stride=2, K=12)` produces
   `(T-1)·s + K` = `2T+10` for input T. The classical "zero-stuff + stride-1
   conv1d" trick produces `2T - K + 1` = `2T - 11`. Closing the K-1 gap
   requires *asymmetric* boundary padding (10 left, 11 right for K=12),
   which `ggml_conv_1d`'s symmetric `p0` parameter can't express. You can
   pre-pad the data with `ggml_concat`'d replicate columns and use `p0=0`,
   but that adds three more graph nodes per AA site and the constants are
   annoying.

2. **Downstream-add broadcast assertion.** Even with lengths corrected
   manually, runtime hit `GGML_ASSERT(ggml_can_repeat(b, a))` inside
   `ggml_add_inplace` from the BigVGAN per-block bias adds — the
   `ggml_reshape_2d` after the cropping `ggml_view_3d` doesn't always
   see a contiguous tensor, and the resulting shape drifts in ways
   `ggml_add`'s in-place fast path refuses to broadcast.

`ggml_conv_transpose_1d` has no `groups` parameter, so we can't express
the depthwise behavior directly with it either; the workaround is a
`[K, C, C]` block-diagonal kernel, which at C=1536 is 113 MB — no-go.

**Update (Step B-v2, same week):** both blockers fixed in
`src/indextts_voc.cpp:aa_snake_beta_native`. Now shippable as an opt-in
behind `INDEXTTS_AA_BACKEND=native`.

The fixes:

1. **Length match via `p0 = K - 1 = 11` on the upsample `ggml_conv_1d`.**
   The zero-stuffed signal of length `2·T_p - 1` becomes `2·T_p + 21`
   after symmetric padding by 11, and the conv1d output is `2·T_p + 10`
   — the same as torch `conv_transpose_1d(K=12, stride=2)`. Crop with
   `up_pad_left + up_pad_right + 1` to land at exactly `2·T`, the same
   number torch produces after its own crop.
2. **`ggml_cont` between the truncating `ggml_view_3d` and the following
   `ggml_reshape_2d`.** The view narrows ne[0] but keeps the parent's
   nb[1] stride, so the resulting tensor is non-contiguous; reshape
   would silently land on the wrong layout and the next graph add fired
   the `ggml_can_repeat` assertion. One extra `ggml_cont` per AA site
   makes the reshape valid.

Validation against the CPU custom-op (Step A) reference:

| Path                     | voc-only (ms) | clicks Δ>0.3 | max\|Δ\| | ASR roundtrip   |
| ------------------------ | ------------- | ------------ | -------- | --------------- |
| Step A custom op  (CPU)  | 7872          | 2            | 0.309    | ✓ exact         |
| Step B-v2 native  (CPU)  | 7574          | 2            | 0.309    | ✓ exact         |
| Step B-v2 native  (GPU)  | 8012          | 26           | 0.375    | ✓ exact         |

CPU output is bit-equivalent (same click pattern, same max\|Δ\|).
GPU output drifts a tiny amount (26 vs 2 jumps, but max\|Δ\| still
below 0.4 — well into the noise floor of speech transients) — the
difference is Metal's vs CPU's floating-point order-of-ops for the
broadcast `ggml_mul`s in SnakeBeta. ASR is identical across all three.

Why we kept the custom-op as default (not switched to native):

- Native-on-CPU is 4 % faster but introduces a second AA codepath. Not
  worth the maintenance vs proven custom op for the marginal gain.
- Native-on-GPU is *slower* than custom-op-on-CPU (8.0 s vs 7.9 s on
  M1) — the concat/reshape/scale graph overhead inside Metal eats
  whatever the kernel-level GPU speedup buys. A real fused MSL kernel
  (Step C-2) is still the path to a meaningful GPU win.
- ggml-backend-sched does the right thing — when `aa_use_native()`
  returns true, the auto-fall-to-CPU in Step A is skipped and the
  vocoder graph stays on Metal end to end.

Lesson: a "polyphase zero-stuff + conv_1d" *is* expressible as native
ggml ops if you accept three boilerplate concats per AA site to fix
the asymmetric-pad problem. It compiles and runs correctly on Metal.
But the per-call graph overhead means it's worth shipping only as an
opt-in proof of correctness; the real win is still the fused-kernel
custom op route — see `tools/upstream-prs/07-metal-aa-snake-beta.md`.

### Accelerate vDSP_desamp + vvsinf beats hand-rolled SnakeBeta loops on M1 (May 2026)

After the CPU AA op is correct, the bottleneck is the per-channel inner
loops: K-tap scatter for upsample, sin+sqr+fma for SnakeBeta, K-tap dot
+ stride-2 decimate for downsample. The scalar inner loops are clean
but the compiler's auto-vectorisation isn't always taking the FMA
opportunity — especially across the `+=` loop-carried dependency in the
upsample scatter.

Step C-1 swaps the two stages that have direct Accelerate analogues:

- SnakeBeta inner: `vDSP_vsmul → vvsinf → vDSP_vsq → vDSP_vsma` (one
  vector mul, one vector sin, one vector square, one fused-multiply-add).
- Downsample inner: `vDSP_desamp(decimation=2, filter)` fuses K-tap FIR
  + stride-2 decimation into one Accelerate call backed by NEON.

vDSP is `#ifdef __APPLE__` only; the scalar paths still compile and run
elsewhere. Set `INDEXTTS_AA_SCALAR=1` to force the scalar paths for A/B.

Measured speedup on M1 (q8_0 GPT, JFK voice prompt, ≈ 6.7 s of audio,
average of 3 warm-cache runs):

| Path                | voc-only |
| ------------------- | -------- |
| scalar (Step A)     | 6906 ms  |
| Accelerate (Step C-1)| 6746 ms  |

≈ 2-3 % on the full vocoder — small because AA SnakeBeta is one
component alongside the MRF stack and the per-stage convs that
dominate. The win is "free": opt-out flag exists, output is
numerically equivalent to scalar (rmsdiff 1.3 × 10⁻⁵, well below int16
quant noise), ASR roundtrip identical.

Lesson: vDSP gives modest wins on already-cache-friendly inner loops.
The big lever for IndexTTS GPU perf is still a Metal kernel — see
`tools/upstream-prs/07-metal-aa-snake-beta.md`.

## Speaker verification — TitaNet

### NeMo mel preprocessing is not what the config says

The NeMo model config YAML for TitaNet says `window: "hann"` but the
actual `FilterbankFeatures.forward()` applies three additional steps
before the STFT that are NOT listed in the model config:

1. **Pre-emphasis** (`x[t] = x[t] - 0.97 * x[t-1]`) — this is the
   default from NeMo's `AudioToMelSpectrogramPreprocessor`, NOT specified
   in the per-model config. Missing this causes a ~12 dB mel offset and
   cos drops from 0.999 to 0.917.
2. **Zero-padding** for center=True (`pad_mode="constant"`) — NOT
   reflect. The NeMo source explicitly passes `pad_mode="constant"` to
   `torch.stft`. Our initial reflect padding was wrong.
3. **Centered window placement** — when `win_length < n_fft`, PyTorch's
   `torch.stft` pads the window symmetrically:
   `left = (n_fft - win_length) // 2`. This is NOT documented in the
   PyTorch STFT docs but is in the source.

Lesson: never trust the model config alone. Trace the actual Python
forward with intermediate dumps. The "standard" STFT has at least 3
knobs (pre-emphasis, pad mode, window centering) that differ between
NeMo, torchaudio, and librosa.

### NeMo BatchNorm eps is 0.001, not 1e-5

All encoder BN layers in TitaNet-Large use `eps=0.001`. The PyTorch
default is `1e-5`. This single mismatch caused the encoder norms to
explode: block 1 norm was 451 (should be 17). The effect compounds
multiplicatively through 3 mega-blocks × 3 sub-blocks = 9 stages.

Lesson: always check `module.eps` for every BatchNorm in the model.
NeMo's `ConvASREncoder` passes `batchnorm_kwargs` from the config,
and the default for NeMo's `MaskedBatchNorm1d` is 0.001 — NOT the
PyTorch `nn.BatchNorm1d` default.

### JasperBlock has a hidden post-activation (mout)

NeMo's `JasperBlock` applies an activation (ReLU + Dropout) AFTER
the SE + residual addition, stored in `self.mout`. This is NOT visible
in the `mconv` module list. The mconv layers provide inter-sub-block
activation for repeat > 1, but the FINAL activation comes from mout.
Without this, the encoder output has negative values where it shouldn't.

Lesson: always print `block.mout` / `block.mout` for any NeMo encoder
block. The `mconv` list alone is incomplete.

## Parakeet-TDT greedy decode — blank + duration=0 (May 2026)

### `max(1, dur_skip)` on blank exits the inner loop "too early" vs. NeMo

`parakeet_tdt_decode()` originally treated `blank + dur=0` as a
force-advance: `t += std::max(1, dur_skip); break;`. NeMo's
`GreedyTDTInfer._greedy_decode` instead does
`time_idx += skip; need_loop = (skip == 0)` — i.e. blank+dur=0 stays
on the same encoder frame and re-runs the joint head, counting toward
the shared `max_symbols` budget (default 10). Only when the budget is
exhausted does `time_idx += 1` fire.

For pure greedy argmax the two are observationally equivalent: blank
predictions don't change predictor state, so retries are deterministic
and produce the same blank+dur=0 every time, eventually force-advancing
exactly one frame either way. The fix is still worth applying because:

1. **Reference parity**: the inner-loop budget now counts both
   non-blank+dur=0 emits AND blank+dur=0 spins, matching NeMo's
   `symbols_added`. The non-blank path was already correct
   (`n_inner++` + post-loop `if (n_inner >= max_per_step) t++`).
2. **Sampling correctness**: with `decode_temperature > 0`, the joint
   output IS stochastic and retries can resolve to a non-blank token.
   The old code skipped those retries entirely. Issue #88 reporter's
   "model emits blank+dur=0 as a 'not yet decided' signal" mental
   model only matches reality in the sampling path.
3. **Forward-compatibility**: any future change that mutates predictor
   state on blank (e.g. a non-NeMo experimental decoder) would silently
   change behavior with the old force-advance; the retry path
   preserves the contract.

### Don't double-increment `t` in the fix

The natural-looking fix is

```cpp
if (tok == blank_id) {
    if (dur_skip > 0) { t += dur_skip; break; }
    n_inner++;
    if (n_inner >= max_per_step) { t++; break; }   // ← bug
    continue;
}
```

This double-advances when blank+dur=0 exhausts the budget: the `t++`
inside the if-branch fires, then the existing post-loop
`if (n_inner >= max_per_step) t++` fires again, jumping the encoder
ahead by 2 frames instead of 1.

The clean version drops the in-handler force-advance and lets the
existing post-loop check do its job:

```cpp
if (tok == blank_id) {
    if (dur_skip > 0) { t += dur_skip; break; }
    n_inner++;
    continue;     // while-condition + post-loop handle exhaustion
}
```

Lesson: when adding a fast-exit inside a loop with an existing
post-loop "fix up" step, audit the post-loop for redundancy with the
fast-exit. The most defensive-looking version is the most likely to
double-count.


## VibeVoice 1.5B TTS voice cloning: acoustic + semantic dual encoder (May 2026)

### Root cause: TTS clone used acoustic-only, not acoustic+semantic

Microsoft's VibeVoice architecture uses two parallel σ-VAE encoders for
speech representation:

- **Acoustic encoder** (`at_enc`): vae_dim=64, captures spectral/timbre info
- **Semantic encoder** (`st_enc`): vae_dim=128, captures linguistic/prosodic info

Both go through separate connectors (`at_conn` → d_lm, `se_conn` → d_lm) and
are **element-wise summed** before being injected as voice conditioning tokens.

The TTS voice clone code only used the acoustic encoder (comment said "uses
ONLY the acoustic encoder"). The ASR path correctly used both. The assumption
was wrong — Microsoft's `modeling_vibevoice.py` forward() at line 372-375:

```python
x[acoustic_input_mask] = (
    speech_all_connect_features[speech_masks]       # acoustic
    + semantic_speech_all_connect_features[speech_masks]  # semantic
)
```

**Scaling asymmetry**: only the acoustic path gets `(x + bias_factor) *
scaling_factor`. The semantic encoder output goes directly to the connector
with no scaling. In the Python reference, scaling is inside
`forward_speech_features()` which only processes acoustic tokens. The semantic
path is a separate `self.model.semantic_connector(speech_semantic_tensors)`
call with raw encoder output.

**RMS ratio**: acoustic connector RMS ≈ 0.245, semantic connector RMS ≈ 0.034
(ratio ~14%). The semantic contribution is small but structurally important for
speaker identity transfer — without it the model produces generic-sounding
speech.

### WAV sample rate: parser must check, not assume 24 kHz

The σ-VAE encoders expect 24 kHz input (3200× total downsample ratio). The
RIFF WAV fmt chunk stores sample rate at byte offset +4, but the parser
(`vibevoice_wav_ref.h`) skipped that field. A 16 kHz reference WAV feeds
176000 samples → 55 frames instead of the correct 264000 → 83 frames.

Fix: read `sample_rate` from fmt chunk, add linear-interpolation resample
when ≠ 24 kHz. This is simpler than the Kaiser-windowed sinc resampler used
in chatterbox (voice-clone quality doesn't need it — the σ-VAE encoder is
robust to interpolation artifacts in the conditioning path).

### C ABI env-var pattern for WAV voice reference

VibeVoice's WAV clone path uses `VIBEVOICE_VOICE_AUDIO` env var because the
core engine reads it in `vibevoice_synthesize()`. The CLI adapter already
mapped `--voice <path.wav>` → env var, but the C ABI
(`crispasr_session_set_voice`) went straight to `vibevoice_load_voice()` which
only handles GGUF voice packs. The fix detects `.wav` suffix and calls
`setenv("VIBEVOICE_VOICE_AUDIO", ...)` instead. This is the same
pattern used by the CLI adapter — the C ABI just didn't replicate it.

Implication: any new backend that accepts WAV references via env vars needs
the same suffix detection in the C ABI, not just in the CLI adapter.

### Python reference verification approach

The `vibevoice` pip package has circular import issues with newer
`transformers` versions (`tokenization_qwen2_fast` moved, `AutoModel.register`
rejects duplicate config registrations). Workaround:

1. Monkey-patch `_LazyAutoMapping.register` to accept `exist_ok=True` before
   any vibevoice imports
2. Create a compatibility stub at
   `transformers/models/qwen2/tokenization_qwen2_fast.py` that re-exports from
   the top-level `transformers.Qwen2TokenizerFast`
3. Use `importlib.util.spec_from_file_location` to load specific submodules
   without triggering the package's `__init__.py` import chain

For the voice-clone reference, we only need the encoder + connector weights
(not the full LM), so loading individual components from safetensors is
both faster and avoids the 3 GB LM weight overhead.

### Static library relinking trap (cmake)

When a `.cpp` change only updates a static `.a` library, cmake's incremental
build may report `Built target X` without actually relinking the final
executable — the `.a` timestamp doesn't trigger the exe's link step. The
binary stays stale. Workaround: `rm build/bin/crispasr` then rebuild, or
touch the exe's own `.o` dependency.

---

## voxcpm2 perf — per-step ggml graphs, Metal, SIMD layouts (May 2026)

The PLAN #96 push took voxcpm2-tts "Hello world" zero-shot from **~49 s
to ~5–7 s** wall on M1 (≈8 commits across one session). Most of the
lessons below apply to any new TTS / ASR backend that mixes legacy
CPU helpers with ggml graphs.

### Per-call ggml cgraph beats per-matmul tiny graphs by 2–3× even on CPU

The original voxcpm2 LocDiT did ~30 `matmul_mv_ggml` calls per
invocation, each of which ran `ggml_init` + `ggml_new_graph` +
`ggml_backend_graph_compute` + `ggml_free`. With CFM running LocDiT
19× per AR step (CFG × 10 timesteps − the zero-star skip), that's
~570 graph builds per AR step *just for CFM*.

Replacing them with one cgraph per `locdit_forward` call — 12 layers
folded into a single graph with named `ggml_set_input` tensors —
gave 2.3× on CPU (CFM/step 2398 → 1035 ms on M1 OMP=8) before any
Metal involvement. The win is amortising graph-build overhead, not
the matmul work itself.

Pattern (mirror `qwen3_tts.cpp build_graph_talker_kv` or
`chatterbox.cpp build_graph_t3_kv`):

  - Add `backend`, `backend_cpu`, `compute_meta` (vector<uint8_t>),
    `galloc` (ggml_gallocr_t) to the context.
  - Pre-allocate `compute_meta` = `ggml_tensor_overhead() * N +
    ggml_graph_overhead_custom(N, false)` where `N` is the
    worst-case node count (4096 was generous for voxcpm2).
  - `build_<thing>_graph(ctx)` runs `ggml_init` with
    `no_alloc=true` over `compute_meta`, builds the topology with
    `ggml_set_input` / `ggml_set_output` named tensors, then
    `ggml_free(ctx0)`.
  - Per call: `ggml_gallocr_alloc_graph` + per-input
    `ggml_backend_tensor_set` + `ggml_backend_graph_compute` +
    `ggml_backend_tensor_get`.

### Cache the graph itself for another 2× on top of that

The above still rebuilds the cgraph topology on every call.
`ggml_gallocr_reserve` against a long-lived graph in a dedicated
per-cache arena (`std::vector<uint8_t>` + `ggml_context*`) lets all
subsequent calls skip the rebuild + the gallocr planner walk. For
voxcpm2 this took CFM/step from ~1035 ms (uncached) to ~410 ms
(cached) — meeting the plan's ~400 ms CPU target.

Works directly for any graph with constant topology (LocDiT,
LocEnc — bidirectional encoders). For graphs with `n_past`-varying
shape (TSLM step), use the qwen3 bucketed pattern below.

### Multi-bucket Lk + runtime `kv_indices` makes KV-cache step graphs cacheable

qwen3_tts's `talker_buckets` pattern: pin `Lk = bucket_size` in
`core_attn::kv_self_attn`'s `fixed_kv_len` parameter and pass
`positions` as the `kv_indices` tensor so the K/V scatter goes
through `ggml_set_rows` (runtime-indexed) instead of a baked
static offset. Topology becomes `n_past`-invariant. Layer count,
KV-cache shape, and bucket Lk are the only things that affect
the graph structure — and Lk is fixed per bucket.

For voxcpm2 TSLM we ship 5 buckets {128, 256, 512, 1024, 2048};
`tslm_pick_bucket(needed_lk)` picks the smallest fitting one,
each is built lazily on first hit, longer prefills fall through
to the dynamic per-call build. Short prompts pay the cheapest
(Lk=128) attention overhead — important because
`ggml_flash_attn_ext` computes `Q·K^T` over the full bucket Lk
regardless of the `-inf`-masked tail, so wider buckets are
genuinely more expensive even with the mask.

### Apple Silicon Metal "shared" buffers keep `tensor->data` CPU-readable

`ggml_backend_init_best()` on M1 returns a Metal backend whose
weight buffers default to `is_shared = true` (unified memory).
This means `tensor->data` is a CPU-dereferenceable pointer to the
same physical pages Metal sees — no copy, no SIGSEGV when legacy
CPU helpers (`matmul_mv_ggml`, `rms_norm_cpu`, `tensor_data_f32`,
…) dereference it directly.

Concretely: switching `vox_load_weights` to load onto
`c->backend` (Metal) on `params.use_gpu=true` *didn't break* any
of the remaining legacy CPU paths (TSLM/RALM prefill, LocEnc,
VAE encode/decode, FSQ, stop predictor) on Apple Silicon. The
graph paths automatically got Metal compute via
`ggml_backend_graph_compute(c->backend, gf)`.

This trick is M1-specific. On discrete GPUs (CUDA, ROCm) or
non-shared Metal modes (very rare on M-series), `tensor->data`
would be device-only and dereferencing from CPU would SIGSEGV —
you'd need either per-weight CPU mirrors (like qwen3_tts's
`CpuEmbdCache`) or a full graph-only path.

The follow-on: the same Metal weights also speed up the legacy
helpers because Q4_K dequant reads land in lower-latency unified
memory than the previous CPU allocations. voxcpm2's TSLM prefill
went 5 000 ms → 80 ms (~60×) just from this — the matmul code
itself was unchanged.

### `GGML_PREC_F32` on `ggml_flash_attn_ext` is refused by Metal's `supports_op`

The chatterbox T3 patch (`ggml-metal-device.m`) explicitly returns
`false` from `supports_op` for any `GGML_OP_FLASH_ATTN_EXT` tagged
`GGML_PREC_F32`. The intent is to route those ops to the CPU
backend via sched for bit-identical CPU/GPU output (chatterbox
needs that to debug #83).

But: if you call `ggml_backend_graph_compute(metal, gf)` directly
(no sched), the unsupported op aborts the entire compute. Symptom:
`ggml_metal_op_encode_impl: error: unsupported op 'FLASH_ATTN_EXT'`
followed by ggml_abort.

For voxcpm2 the fix was just to *not* set PREC_F32 on LocDiT's
bidirectional flash-attn — the per-stage cosine bar (`cfm_step0_result
≥ 0.93`) tolerates Metal's F16 simdgroup drift. Pre-existing
`core_attn::kv_self_attn` doesn't set PREC_F32 internally, so TSLM
step worked out of the box.

If you do need bit-identical CPU/GPU (e.g. for debug bisects),
switch from direct `ggml_backend_graph_compute` to
`ggml_backend_sched_graph_compute` against a `[Metal, CPU]` sched —
that's what qwen3_tts and chatterbox do, and it auto-routes
PREC_F32 ops to CPU.

### `ggml_flash_attn_ext` head_dim allowlist

Metal's flash-attn supports `head_dim ∈ {32, 40, 48, 64, 72, 80,
96, 112, 128, 192, 256, 320, 512, 576}`. Anything else falls back
to CPU via sched or aborts under direct compute. Picking head_dim
during a port: stick to the allowlist or expect to route attn to
CPU.

### OMP parallelism on conv loops needs gather-form rewrites

The original `causal_transposed_conv1d` did scatter-add into
shared output — race-prone, so the outer loop couldn't be OMP'd.
Rewriting in gather form (for each output position, walk valid
kernel taps via `k0 = (ot + trim) mod stride` step `stride`) made
the `(out_ch, ot)` outer pair write-disjoint and parallelisable.

Similar story for `snake1d` and per-block SR conditioning —
outer loop over channels is write-disjoint, OMP-parallel.

But OMP alone isn't enough when the inner loops have strided
memory access (see next).

### Strided ic loops kill auto-vectorisation; transpose to contiguous wins 4–8×

VAE conv1d inner loops read `x[ic*T_in + it]` (stride `T_in`) and
`weight[ic*out_ch*ksize + …]` (stride `out_ch * ksize`). Both
strided in the inner axis — the compiler can't emit NEON vector
loads. Per-call work was effectively scalar even with OMP.

Fix: transpose both into contiguous-`ic` layout before the inner
loop:

  - weight: `[in_ch, out_ch, ksize] → [ksize, out_ch, in_ch_inner]`
  - x: `[in_ch, T_in] → [T_in, in_ch_inner]`

The inner `ic` dot product becomes a contiguous load on both
operands — clang/gcc auto-vectorises into NEON `fmla` instructions.

The transposes are `O(in_ch × (out_ch × ksize + T_in))` per call,
small vs the unlocked dot work. For voxcpm2 block 0 transposed
conv: 2 957 ms → 615 ms (4.8×) on M1 OMP=8. Total VAE decode
8 772 → 3 875 ms (2.3×).

Gate the transpose path on `in_per_grp > 1 && ksize > 1` —
depthwise (`in_per_grp == 1`) and 1×1 convs (`ksize == 1`)
get no SIMD lift and the transpose is pure overhead.

### Build instrumentation BEFORE optimising the wrong thing

Per-block VAE decode timings under `VOXCPM2_BENCH=1` revealed
that **72 % of VAE time was in `causal_transposed_conv1d` upsamples**,
not the residual units I'd been about to spend an hour rewriting.
Before then I'd assumed snake1d was the hotspot — wrong.

The instrumentation cost: ~10 lines of `vox_now_ms()` deltas +
one `fprintf` per block, gated on the existing
`VOXCPM2_BENCH` env. Always pays for itself.

### `ggml_concat` along time axis lets you build prefix sequences cleanly

For both the LocDiT graph (mu_toks + time + cond + x = T=11) and
the LocEnc graph (CLS + patch_frames = T=5), the input sequence
is built by concatenating disjoint pieces with `ggml_concat(ctx,
a, b, dim=1)`. This is much cleaner than allocating a
pre-sized `[d, T]` tensor and writing slices into it — and on
Metal it's just a single `kernel_concat` dispatch.

### Don't over-engineer the first cut — ship working, then optimise

The session's biggest unit wins came from the simplest changes:

  1. LocDiT graph (1 day): 2.3× CPU CFM.
  2. Cache the graph (3 hours): 1.7× on top.
  3. Metal-load weights (1 hour): 60× on TSLM prefill.

Each commit was small, single-purpose, validated by the diff
harness before the next one started. If we'd tried to land
"VAE graph + LocDiT graph + Metal + caching + buckets" in one
go we'd still be debugging.

---

## voxcpm2 VAE — Python wrapper that captures kwargs without forwarding (May 2026)

This bit hard. The voxcpm `CausalTransposeConv1d` class in
`audio_vae_v2.py` defines:

```python
class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x):
        return super().forward(x)[..., : -(self.__padding * 2 - self.__output_padding)]
```

The trap: `padding` and `output_padding` are **named keyword args**
in the subclass `__init__` signature, so Python pulls them out of
the call's kwargs BEFORE the `**kwargs` capture. They're stored
into `self.__padding` (name-mangled) and **never passed to
`super().__init__`**. So PyTorch's `nn.ConvTranspose1d`
internally uses `padding=0`, `output_padding=0` regardless of what
the caller passed.

That changes the output length of `super().forward(x)` from "PyTorch
ConvTranspose1d output with padding P" to "no-padding output,
`L_std = (T_in - 1)*S + K`". The wrapper then slices `[:-(2P - OP)]`
from the END, yielding `T_in * S` samples (verified for
`S ∈ {8, 6, 5, 2}` with `K = 2S, P = ceil(S/2), OP = S%2`).

**What this means for a C++ port.** Don't assume the subclass's
`__init__` signature lines up with the parent's — read what `*args,
**kwargs` actually passes through, then trust that. The legacy
voxcpm2 C++ `causal_transposed_conv1d` used `trim = K - 1` (a
head-shift), which gives the same OUTPUT LENGTH but a completely
wrong slice of the no-padding output. Cumulated across 6 upsample
blocks at increasing sample rates, the audio was shifted by **~46 ms
vs Python** — and `decoded_audio cos vs Python = 0.008` in the
diff harness for months.

The correct C++ formula: emit `T_in * S` samples, taking positions
`[0, T_in * S)` of the standard padding=0 transposed conv output.
For each output position `ot`:

```
y[ot] = sum_k w[k] * x[(ot - k) / S]   for k with (ot - k) % S == 0 and (ot - k)/S in [0, T_in)
```

= a tail-trim of `K - S` samples from the no-padding output. The
ggml graph version is even simpler — just take the first
`T_in * S` rows of `ggml_conv_transpose_1d(s, p=0, d=1)`.

Fixed in both legacy CPU `vae_decode` and the new `vae_decode_graph`.
After the fix: `vae_only cos vs Python = 0.989` (essentially correct),
`decoded_audio cos = 0.683` (remaining drift is now provably upstream
— CFM/TSLM precision cascade).

### Isolate the VAE before chasing audio drift

The diff harness's `decoded_audio` cosine compares end-to-end audio
through ALL of TSLM → RALM → LocDiT → CFM (10 Euler steps × N AR
steps) → VAE. A failing cosine there could come from any of those.

To isolate the VAE: hook `model.audio_vae.decode` in the Python
dumper, save the input latent as `generated_latent`, then add a C++
`vae_only` / `vae_only_graph` stage in `voxcpm2_extract_stage` that
takes the latent via `ref_samples` and runs **only** vae_decode.
Comparing the C++ output of that against Python's `decoded_audio`
directly attributes drift to the VAE without confounding it with
upstream precision drift. (Implementation: ~50 LOC across the
backend + dump + diff main.)

This is the same pattern used elsewhere (e.g. chatterbox `s3gen`
with reference T3 tokens) — feed Python's intermediate as the
boundary input, run only the segment under test. Cheap to add when
the upstream is non-deterministic or non-trivial; saves hours of
"is it the AR or the codec?" guessing.

---

## ggml broadcast hides size-mismatch bugs (May 2026)

When the SR-cond fix went in (`vae_decode_graph` first call), the new
sr_scale/sr_bias tensors were sized from `it->second->ne[1]` — which
returned **4** instead of **2048** because GGUF's `scale_embed` is
stored as ne=[2048, 4] in ggml (C innermost, bucket outer — the
*opposite* of what I'd assumed). So my init allocated 4-element
tensors and only wrote 4 scales (the first 4 channels at bucket 3),
leaving the "broadcast" to fend for itself.

**`ggml_mul(cur ne=[T, 2048], sr_scale ne=[1, 4])` does NOT assert**.
The op compiled and ran. The output was even close to correct — `cos
vs Python = 0.967` (vs 0.989 for the legacy CPU path). The first few
channels got the right scale; the rest got whatever ggml's binary-op
broadcast read from past the tensor end.

**Why this is dangerous.** A small cos drop reads like "Metal
floating-point drift" — so I spent multiple iterations trying to
swap CPU/Metal backends instead of looking at the broadcast input
shapes. The truth was much simpler: the input tensor was the wrong
size.

**How to detect.** When debugging a graph-vs-legacy drift:

1. Dump intermediates per-stage and find the first stage where they
   diverge (here: `block_0_sr` was identical until I added the SR
   cond step; the snake checkpoint came in already-wrong).
2. **Always print `ne[]` of every input tensor** to the failing op.
   I had `C=4` printed in the init trace from the very first run; I
   just didn't look at it carefully because "obviously ne[1] is the
   channel dim".
3. If ggml broadcast looks suspicious, replace it with an explicit
   `ggml_repeat` to a matching shape — that *does* assert on size
   mismatch.

**How to apply.** For multi-axis GGUF tensors with ambiguous shape
order, **take `max(ne[0], ne[1])` for "the non-bucket dim"** instead
of trusting either dim's position. The legacy CPU loop's pointer
arithmetic doesn't care about ne order (it just walks raw bytes),
which is why it kept working through the same ambiguity.

---

## Diff-harness "drift" is mostly the GGUF quant, not a code bug (May 2026)

The voxcpm2 VAE-graph session (commit `3be663cc`) left
`decoded_audio cos vs Python = 0.683` open and tracked as "upstream
CFM / TSLM drift". Several intermediate stages also looked weak:
`cfm_step0_result` 0.937, `tslm_layer_27_out` 0.968.

I spent half a day chasing this assuming F32-vs-bf16 op precision was
the culprit (Python runs voxcpm2 in `dtype=bfloat16`, our C++ in F32).
Hypothesis: rounding every tensor op output to bf16 (via the
`bf16_round_vec` helper in `core/torch_rng.h`) would align us with
Python's bf16-throughout dataflow. Added rounds to `tslm_layer_step`
(Q/K/V/O matmuls, RoPE, attention, FFN, both residual adds), to
`bidir_attn_full`, and to `locdit_forward` (time MLP, in/cond projs,
all 12 layers' attn + FFN).

**The bf16-round changes had near-zero effect** and slightly
**regressed** `cfm_step0_result` (0.937 → 0.925). They were also
costly (synth wall +~15 min from extra sequential op overhead).
Reverted entirely.

**Actual cause: the diff harness loads `voxcpm2-q4_k.gguf`** (4-bit
quantised weights) **but Python's reference uses the bfloat16
weights**. The "drift" was the Q4_K quantisation noise, multiplied
through the network. Re-running the diff with `voxcpm2-f16.gguf`:

| Stage              | Q4_K cos | F16 cos |
| ------------------ | -------: | ------: |
| tslm_prefill_out   |    0.986 | **0.998** |
| dit_single_fwd     |    0.994 | **0.99999** |
| cfm_step0_result   |    0.937 | **0.99992** |
| tslm_layer_27_out  |    0.968 | **0.999** |
| **decoded_audio**  | **0.683**| **0.929** |
| lm_to_dit_hidden   |    0.998 | **0.99996** |

Every intermediate stage hits cos ≥ 0.998 on F16. The remaining 0.07
gap in `decoded_audio` on F16 is most likely AR-stop-step jitter
(C++ and Python stop predictors fire at slightly different patches
when fed the same `mu`) plus residual F16-vs-bf16 mantissa drift in
the VAE. Not the deep transformer body.

**How to apply.**

1. **Before chasing "upstream precision drift" in a diff harness,
   re-run with the F16 (or BF16) GGUF.** If cos jumps to ≥ 0.998,
   the model is bit-correct — the previous drift was just the
   shipped quant level. Don't add bf16 rounds to F32 ops; they
   don't fix quant noise.

2. **The Q4_K → bf16 cos gap IS the quant cost** — it's the same
   number you'd see for any sufficiently deep transformer that's
   bf16 in Python and Q4_K in C++. The acceptance bar should be
   "audio sounds good + ASR roundtrips" (which Q4_K passes for
   voxcpm2), not "diff-harness cos ≥ 0.999 against bf16
   reference" (which Q4_K never will).

3. **For per-stage diff diagnostics, recommend dumping the
   reference against the matching weight precision:** run the
   Python dumper with `voxcpm2-f16` if you want to compare F16
   C++ vs F16-equivalent Python, or accept the Q4_K cosine
   penalty as quant noise.

Cross-ref: [[project_voxcpm2_vae_kwargs_bug]] (real bug fix earlier
in the same session, contrasts with this non-bug).

---

## SANM-encoder family (FunASR / SenseVoice / CosyVoice)

### The same `SenseVoiceEncoderSmall` ships in at least three FunAudioLLM products

Fun-ASR-Nano-2512, Fun-ASR-MLT-Nano-2512, and SenseVoiceSmall all use
the identical 70-block `SenseVoiceEncoderSmall` topology (1 entry
block @ 560→512 with `encoders0`, 49 main blocks via `encoders`,
20 tp blocks via `tp_encoders`, `after_norm` between, `tp_norm` at
end; 512-dim, 4 heads, kernel 11). The encoder weights ARE different
across the three releases (each is fine-tuned for its product) but
the C++ runtime is shared: factor everything that touches a SANM
block into `src/core/sanm.h` (`core_sanm::BlockWeights`,
`BlockParams`, `build_block`) and reuse it from each per-model `*.cpp`.

If we ever port CosyVoice2 / CosyVoice3 the same encoder is in there
too (it's the text-to-codec semantic-encoder bottleneck). One
helper, three+ consumers.

### config.yaml advertises CTC; Fun-ASR's `model.pt` ships none of it

Upstream `config.yaml` and `funasr/models/fun_asr_nano/model.py`
declare a `ctc_decoder` + a `CTC` head; the published checkpoint for
Fun-ASR-Nano-2512 and Fun-ASR-MLT-Nano-2512 ships **only**
`audio_encoder.* + audio_adaptor.* + llm.*` (1261 tensors total, zero
`ctc_decoder.*` / `ctc.ctc_lo.*` keys). If you trust the config you'll
spend a day chasing a CTC path that isn't there. Verify by enumerating
the state-dict prefixes before writing converter code.

The CTC weights live in **SenseVoiceSmall** instead — same encoder
shape, but the published `model.pt` carries `encoder.* + embed.*
+ ctc.*` and skips the LLM. That's the model to use when you want
fast non-AR multi-task ASR; Fun-ASR-Nano is the one to use when you
want LLM-quality text output (and can pay the AR cost).

### Adaptor `MultiHeadedAttention` default heads is 8, not 16 (Fun-ASR)

`funasr.models.llm_asr.adaptor.Transformer` builds the per-block
attention as `MultiHeadedAttention(kwargs.get("attention_heads", 8),
llm_dim, ...)` and `config.yaml`'s `audio_adaptor_conf` does NOT pass
`attention_heads`. So the adaptor's two transformer blocks run at
**8 heads (head_dim = 128)**, not the 16 you'd guess from
`llm_dim / 64`. The CrispASR converter wrote 16 into the GGUF
`funasr.ada_n_heads` KV by mistake; the runtime ignores that KV
and hard-codes 8. Mis-counting drops the adaptor's cosine similarity
to ~0.6 (vs cos=1.0 on the 70-layer encoder which is bit-near-exact)
and produces a one-character-off Chinese transcript (`开放` instead of
`开饭`). The diff harness catches this immediately, but only if you
run it.

### EncoderLayerSANM drops the attn residual when `in_size != size`

```python
if self.in_size == self.size:
    x = residual + dropout(self_attn(x, mask))     # standard
else:
    x = dropout(self_attn(x, mask))                # no residual
```

`encoders0[0]` (the entry block, `in_size=560, size=512`) is the
only one of the 70 blocks that hits this branch. The C++ encoder
block builder needs an `apply_attn_residual` flag (or two code
paths). Forgetting to special-case it produces a 560-dim residual
being added to a 512-dim output, which won't even compile in ggml —
but if you silently pad / project the residual to 512, you get a
model that "works" but has subtly wrong activations from layer 0
onward.

### SANM FSMN memory branch has its own pre-conv-V residual

```python
def forward_fsmn(self, inputs, mask, ...):
    x = inputs.transpose(1, 2)
    x = self.pad_fn(x)
    x = self.fsmn_block(x)               # depthwise Conv1d, no bias
    x = x.transpose(1, 2)
    x += inputs                          # ← residual to PRE-conv V
    return x
```

The full SANM block output is `att_out + fsmn_memory`. Two
non-obvious things: (1) FSMN convolves over **V only** (not Q or K,
and not the post-MHA result), (2) the FSMN branch carries its own
residual to V **before** joining the MHA path. Drawing the dataflow
with one arrow fewer or one arrow more both produce a network that
compiles fine but yields ~cos 0.5 vs the reference.

### `use_low_frame_rate=true` shrinks the prompt placeholder count (Fun-ASR)

config.yaml `audio_adaptor_conf.use_low_frame_rate: true` triggers
a 3× `(x-1)//2+1` reduction in `fake_token_len_i` inside
`FunASRNano.data_load_speech`, even though the adaptor itself has
`downsample_rate: 1` (no actual downsample in the forward graph).
The prompt builder reserves the reduced number of placeholder slots,
and only `adaptor_out[:fake_token_len_i]` gets spliced — the rest
of the adaptor frames are discarded. Looks like a vestige of an
earlier adaptor design; mirror it verbatim and the model still
produces correct transcripts.

### funasr 1.3.1 packaging bug — `from ctc import CTC` is absolute

`funasr/models/fun_asr_nano/model.py` line 20 imports CTC with an
unqualified name; on a clean install this fails with
`ModuleNotFoundError: No module named 'ctc'`. Workaround in
`tools/reference_backends/funasr.py`
(`_ensure_fun_asr_nano_importable`): prepend the `fun_asr_nano/`
directory to `sys.path` before importing. Don't try to upstream-fix
this on user machines.

### Flash-attn crossover sits at ~T_lfr=100-150 (~6 s audio)

`FUNASR_NO_FA=1` (and the same shape `SENSEVOICE_NO_FA=1`) reverts the
SANM + adaptor MHA from `ggml_flash_attn_ext` back to the unfused
`mul_mat + soft_max_ext + mul_mat` path. On Apple M1 Metal:

| Clip | T_lfr | FA ON | FA OFF | Total Δ |
| --- | ---: | ---: | ---: | --- |
| zh.mp3 (5.6 s) | 94 | 196 ms | 166 ms | FA OFF wins 6 % |
| samples/jfk.wav (11 s) | 183 | 258 ms | 370 ms | FA ON wins 7 % |

Below T≈100 the FA kernel-launch overhead eats the win; above T≈150
the fused-softmax+matmul kernel pulls clearly ahead. Default ON is
right for typical ASR workloads (most utterances ≥6 s); realtime
users splitting on tight VAD windows may want the opt-out.

### SenseVoice query-embed pattern — multi-task output via prepended tokens, not separate heads

SenseVoiceSmall's "4 outputs in one CTC pass" trick is **not** four
classifier heads — it's a 16-row embedding table (`embed.weight`,
`(16, 560)`) prepended to the LFR fbank features. The first 4
positions of the encoder output go through the same CTC head as the
rest of the sequence, but the CTC vocab reserves slot ranges for
language IDs (24884-24992), emotions (25001-25009), audio events,
and ITN flags (25016-25017). So the model "writes" the multi-task
predictions to the first 4 positions via the standard CTC argmax,
and the transcript follows from position 4 onward.

This means the C++ runtime needs zero per-task classifier code —
just one CTC head + the query-embed prepend. The runtime decides
which of the 4 query rows to prepend (language hint, ITN flag) at
inference time; the model uses them as the implicit "channel
selector" for the encoder output positions.

If the upstream model adds a new task (e.g. speaker ID) the
infrastructure on our side stays exactly the same — they extend
the query-embed table + the CTC vocab; we just read whichever new
special-token IDs get emitted from position 0..3.

### SenseVoice output-token order ≠ input query order — classify by content, not position

The four query embeds are prepended to the LFR features in input
order `[language, event_query, emotion_query, textnorm]` (see
`sensevoice_gather_query_rows` and the upstream Python ref
`tools/reference_backends/sensevoice.py`). The intuitive assumption
is that the encoder output's four prefix tokens come back in the
same order — they don't. On `samples/jfk.wav` the encoder emits
`<|en|><|ANGRY|><|Speech|><|withitn|>`: position 1 is the **emotion**
(`ANGRY`), position 2 is the **audio event** (`Speech`). The
"event_query" and "emo_query" names in upstream refer to the
embedding-table row IDs (1 and 2), not to which slot they decode
into; the trained model has those slots swapped relative to the
input naming.

This bit when wiring `sensevoice_transcribe_structured()`. Positional
parsing populated `audio_event` with `"ANGRY"` and `emotion` with
`"Speech"`. The fix is content-based classification: maintain three
small static sets (languages, emotions, itn flags) and classify each
`<|…|>` against them; whatever doesn't match those three is the
audio event (which is open-ended — Speech / Music / BGM / Laughter
/ Cough / Sneeze / Sigh / ... so a positive enum is brittle).

Bonus robustness: on degenerate audio the model can emit fewer than
four prefix markers. Positional parsing would mislabel everything
downstream; content-based classification just leaves the missing
field empty.

### SentencePiece detokenise without linking libsentencepiece

For CTC-decode-only consumers, we don't need full SentencePiece —
just a piece-ID → piece-string table. Extract the table once at
convert time via `sp.id_to_piece(i)` for i in range(vocab_size),
write it as `tokenizer.ggml.tokens`, and detokenise at runtime by
looking up each ID and replacing the SentencePiece leading-space
marker `▁` (U+2581, "\xE2\x96\x81") with an ASCII space. ~20 LOC
of C++, no link-time dep. Same pattern canary-ctc / firered-asr
already use.

### FunASR audio_adaptor: use_low_frame_rate config mismatch (PLAN #99)

**Symptom:** Fun-ASR-MLT-Nano-2512 hallucinated the ending of English
transcripts (~last 5 tokens wrong), while Fun-ASR-Nano-2512 worked
correctly through the identical C++ code path.

**Root cause:** The C++ runtime hardcoded `use_low_frame_rate = true`.
This flag controls how many adaptor output frames are spliced into
the Qwen3-0.6B LLM prompt. The Nano config.yaml sets it to `true`
(23 of 183 frames used on JFK); the MLT-Nano config.yaml omits it
(upstream default is `false` — all 183 frames). With the wrong value,
the C++ truncated 87% of the audio context.

**Why the bisect was misleading:** The per-stage diff showed genuine
attention drift (cos 0.998 at `ada_blk0_attn`), which is real F32
cross-implementation non-determinism amplified by peaked attention
patterns. But this drift is benign when the LLM receives the full
audio context — the model tolerates it and still decodes correctly.
The hallucination was caused by frame truncation, not attention
precision. The lesson: when one model variant works and another
doesn't through the same code, compare the config files first.

**Fix:** Converter reads `audio_adaptor_conf.use_low_frame_rate` from
`config.yaml` and writes it as a bool GGUF KV. Runtime reads the KV
at load time, falling back to `true` for pre-fix GGUFs.

### Qwen3-TTS FUSED_QKV bench — cold-cache trap + Q8_0 is neutral on Metal

**Trap:** A naive A/B test of `QWEN3_TTS_FUSED_QKV=1` appeared to show
a 1.83× speedup (144.7 → 79.0 ms/frame). The "improvement" was a
**cold-cache artifact**: the baseline run loaded the Q8_0 GGUF from
`/Volumes/backups/ai/` (external disk) into unified memory and macOS's
file-page cache. The immediately following FUSED_QKV run benefited
from those warm pages and appeared faster. Running both cold, or
properly interleaving the two conditions in sequence, gives:
~129 ms/frame for baseline, ~129 ms/frame for FUSED_QKV — neutral.

**Rule:** Always interleave conditions (A B A B A B) and discard the
first cold run of a new binary invocation when benching on macOS.
The first invocation loads the GGUF from disk; subsequent invocations
reuse page-cached pages. Alternating order eliminates this confound.

**Why FUSED_QKV is neutral for Q8_0 on Metal:** The optimization
byte-concatenates Q + K + V weight rows into a single fused tensor so
one `ggml_mul_mat` replaces three. For F16 weights this reduces Metal
kernel-launch overhead (three small dispatches → one larger one). For
Q8_0, Metal's Q8_0 dequant dispatch already launches a GPU threadgroup
that spans the full head_dim×head_dim tile; fusing to a 4096-output
tensor doesn't improve parallelism and may slightly hurt cache
utilization. The net effect on M1 Metal Q8_0 (28L, 1024 d, 8 KV
heads): no measurable difference.

**F16 case still open:** FUSED_QKV for F16 talkers is untested locally
(no F16 talker GGUF on this Mac). The expected benefit is real for
F16 (fewer kernel launches, larger tiles map better to the GPU
scheduler) — but requires a dedicated bench on a machine with the F16
GGUF and enough RAM to hold it warm.

**Bench setup used (2026-05-23, M1, build-ninja-compile):**
```
QWEN3_TTS_BENCH=1 [QWEN3_TTS_FUSED_QKV=1] \
  crispasr --backend qwen3-tts-customvoice \
  -m .../qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf \
  --codec-model .../qwen3-tts-tokenizer-12hz.gguf \
  --tts "In the beginning..." --tts-output bench_tmp.wav
```
Key output line: `qwen3_tts: ar_loop  N ms (94 frames, X.X ms/frame)`.

---

## WDDM idle-clock-state hysteresis on consumer/laptop NVIDIA SKUs (May 2026)

Issue #81 investigation surfaced a Windows/WDDM behavior that
dominates **any** chunked-streaming inference perf measurement on
consumer/laptop NVIDIA GPUs (RTX A1000 Laptop confirmed; almost
certainly applies to RTX 30/40-series mobile and lower-end GeForce
desktop too). Knowing this saves days of code-bisecting noise.

### The behavior

A cold dGPU starts at low P-state (P5/P8, ~210-510 MHz core /
405-810 MHz mem). When compute work arrives, **WDDM does NOT
immediately boost** to P0/1140 MHz / 5501 MHz mem — it requires
~5-15 s of *sustained* GPU activity to engage P0, and it
aggressively defends idle clocks against bursty workloads with
sub-100 ms kernel groups (which is exactly the shape of chunked
ASR/TTS inference).

Same DLL, same hardware, same workload, two different states:

| State | mean / chunk | RTx | Notes |
|---|---:|---:|---|
| Cold (P5/P8) | 1400 ms | 2.7× | what you get if you bench from idle |
| Warm (P0)    |  185 ms | 21×  | what the model is actually capable of |

**8× wallclock difference** from nothing but driver-side power
state. The "real" performance is the warm number.

### Why this matters for ASR specifically

Parakeet / Canary / FastConformer-style ASR runs the encoder
graph in many short kernel groups (one per 4 s chunk = ~150 ms of
GPU work + ~50 ms host gap). WDDM sees each gap as "GPU is idle"
and starts demoting clocks. On a server SKU (Tesla / Quadro with
TCC mode, or any Linux box) this would never happen — TCC bypasses
WDDM, and Linux uses a more conservative DVFS. **The problem is
specific to WDDM on consumer SKUs.**

### What works to mitigate

1. **Pre-warm the dGPU** with ~10 s of sustained CUDA work before
   measuring. We ship `bench-issue81/gpu_keepalive.py` for this
   (a tight 256×256 fp32 ORT-CUDA add loop, ~50 MB VRAM,
   ~1 % util, but enough to keep WDDM in "GPU is busy" mode).
   For benches, simpler: run the workload-under-test ~200 times
   as a discard warmup, then measure.
2. **NVIDIA Control Panel → 3D Settings → Manage 3D Settings**
   - Global → "Power management mode" → "Prefer maximum
     performance" (no admin, persists across reboots)
   - Same setting under Program Settings tab for `python.exe`
     / `crispasr.exe` (per-app override tends to engage the
     dGPU more aggressively on Optimus laptops than the global
     setting alone)

### What does NOT work (counter-intuitive)

1. **`nvidiaProfileInspector.exe -setProfileSetting
   "_GLOBAL_DRIVER_PROFILE,PreferredPState,1"`** *USED* to help
   on older drivers (e.g., 581.95) but is **actively harmful on
   newer drivers** (we confirmed on Studio Driver 596.36): it
   caps the dGPU at P1 instead of allowing P0, so the GPU
   underclocks even when work demands max clocks. **Default
   driver behavior is better than this override on 596.x+.**
   To remove: `nvidiaProfileInspector.exe -deleteProfileSetting
   "_GLOBAL_DRIVER_PROFILE,PreferredPState"`. The setting was
   itself a workaround that newer drivers obsoleted; recheck
   on every driver bump.

2. **Driver upgrades don't fix it.** We went 581.95 → Studio
   596.36 mid-investigation; same WDDM behavior, same magnitude
   of cold-vs-warm gap. The heuristic is OS-side
   (DXGI/dxgkrnl), not driver-side. Game Ready and Studio
   variants behave identically.

3. **`nvidia-smi -lgc <freq>`** would force-pin the clock but
   needs admin per session AND is blocked on consumer SKUs with
   "user does not have permission" even when elevated.

4. **`gpu_keepalive.py` alone is not enough** if NPI is also
   set (the keepalive engages WDDM but NPI caps the resulting
   clock). Use keepalive WITHOUT NPI on 596.x+.

### Practical bench protocol (Windows + laptop NVIDIA)

```powershell
# 1. Set NV Control Panel "Prefer maximum performance" once (GUI, no admin).
# 2. Ensure no NPI PreferredPState override is active:
#    nvidiaProfileInspector.exe -deleteProfileSetting "_GLOBAL_DRIVER_PROFILE,PreferredPState"
# 3. Set CPU max state to 100% on AC (was 98% by default on some Win11 11):
#    powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
# 4. Warm WDDM (10+ s of sustained CUDA work):
python bench-issue81/probe_postsiglu_leak.py <dll> 200
# 5. Bench immediately:
python tools/benchmark_asr_engines.py --engine crispasr --crispasr-lib <dll> ...
```

Single-shot cold benches on this class of hardware are **noise**.
Report best-of-N back-to-back with warm-up; document the warm-up
explicitly.

### How this was learned the hard way

Initial measurements on a fresh-rebooted machine showed `dll-postsiglu`
(current main) at 14.96 s mean vs `dll-post-cg` (May 11 baseline)
at 5.94 s — a *2.5× regression* that looked like a code bug. We
wrote a leak-probe (`bench-issue81/probe_postsiglu_leak.py`) that
called the same DLL 60-150 times in a tight loop, expecting to
see the per-call time degrade. Both DLLs were rock-stable in the
probe (157-256 ms per call, GPU memory constant).

The actual difference between the probe and the bench was that
the probe ran **after** another GPU-intensive bench had just
finished, so WDDM was already engaged. The bench ran from a
cold state. After running the probe FIRST, immediately followed
by the bench, both DLLs returned to ~2.7-2.9 s mean / 175-184 ms
p50 — matching the original May 11 3.063 s reference closely.

**The "2.5× regression" was a WDDM-state coincidence, not code.**
The actual `d758fe69` (fused norm_affine + siglu) impact on A1000
is +5.7 % wallclock improvement when measured cleanly.

### Sources / cross-refs

- PERFORMANCE.md "A1000 Ampere CUDA A/B" + "Phase 1 update
  (2026-05-23) — fused siglu/norm_affine A1000 verdict"
- `bench-issue81/probe_postsiglu_leak.py` (the probe itself; also
  a useful warmup driver)
- `bench-issue81/gpu_keepalive.py` (the May 11 keepalive — its
  docstring was already saying *"NPI's 'Prefer maximum
  performance' only biases up; it doesn't actually pin P0"*, and
  we now know that even when it does bias up, on modern drivers
  it biases the wrong direction)
- Microsoft DXGI WDDM Display Power Management docs (not linked
  here — search "WDDM idle detection" if you need primary sources)

---

## Chatterbox #83 Round 9 — S3Gen UNet GPU drift on Metal (2026-05-24)

This round is the production-fix landing for the cross-backend GPU
drift documented in Rounds 7 (Metal bisect) and 8 (CUDA P100 confirm).
The actual root cause hunt that previous rounds attempted (which
specific kernel loses precision) finally produced a clear answer:
**there's no single kernel bug to fix.** The drift is path-dependent
through `ggml-alloc`'s in-place buffer reuse decisions interacting
with Metal command-buffer scheduling, and 11 independent fix attempts
either don't help or trade one failure mode for another.

### Production fix: UNet weight residency split

`src/chatterbox_s3gen.cpp` is the only changed file. On GPU init, the
910 `s3.fd.*` (UNet1D ConditionalDecoder) weight tensors are loaded
on the CPU backend buffer via `core_gguf::load_weights_split`; the
other ~1375 tensors (`s3.fe.*` Conformer encoder, `s3.flow.*` front-end,
`s3.tok.*` Whisper-style tokenizer, `s3.se.*` CAMPPlus, `s3.v.*` HiFT
vocoder) stay on the GPU buffer. The ggml scheduler auto-routes ops
to the backend where weights live, so the entire UNet sub-graph runs
on the CPU backend in a single backend split — no per-op pinning, no
GPU↔CPU sync inside the UNet (which is what produced the Round 8 NaN
at T_mel ≥ 200). M1 Metal `s3gen_mel cos_min`: 0.940 → **0.999980**
in the diff harness; intelligible audio in smoke at all T.

Wall-time on M1 (3-iter median, "Ask not what your country can do for
you.", T_mel=200):

| Mode | s/run |
|---|---|
| cpu-only | 67.6 |
| default weight residency (UNet CPU, encoder/vocoder GPU) | 76.8 |
| gpu residency (broken, produces nonsense) | 48.0 |

The default residency mode is wall-time comparable to pure CPU on M1
because the encoder/vocoder are a small fraction of total time. The
gpu-residency mode would be ~30% faster *if* the drift were fixed —
that's the motivation for upstream PRs 09–11, not a practical win for
the M1 dev box. On dGPUs (CUDA, larger Metal Pro/Max) the encoder
fraction is larger and the residency split should buy more.

### Q8_0 × F32 bit-match Metal kernel (committed; upstream PR 09)

`ggml/src/ggml-metal/ggml-metal.metal` gains the Q8_0 counterpart of
the existing Q4_K × Q8_K `GGML_PREC_F32` bit-match path: pre-quantise
the F32 input to Q8_0 via `kernel_quantize_q8_0_f32` (mirrors
`quantize_row_q8_0` ARM NEON), then run a scalar integer-dot
`kernel_mul_mv_q8_0_q8_0` (mirrors `ggml_vec_dot_q8_0_q8_0_generic`).
Two precision-critical details:

- `id = 1.0f / d` uses the unrounded F32 `d`, not the F16 round-trip.
  We initially used `d_back = (float)(half)d` and got cos_min 0.940
  → 0.947 (basically no change). Switching to the unrounded F32 `d`
  matters but didn't move the needle on the UNet drift either,
  because…
- Quantize rounding uses `rint()` (round-to-nearest-even, IEEE
  default), which matches ARM NEON's `vcvtnq_s32_f32`. We tried
  `roundf` semantics (`floor(x + copysign(0.5, x))`) first — also no
  significant change.

The kernel is bit-identical to CPU on all 350 UNet Q8_0 mul_mats. It
just doesn't help the UNet drift because the drift isn't in mul_mat
output. Verified by combining the kernel with weight residency: the
two are independent.

### The exhaustive bisect — 11 attempts, none surgically fix it

The session pivoted from "find the buggy kernel" to "characterise
where the drift comes from" after the bit-match mul_mat kernel
failed to help.

1. **Bit-perfect mul_mat** (the Q8_0 kernel above) → no improvement.
   Drift isn't in mul_mat output.
2. **PIN_CPU_OP=mul_mat at T=102 (diff harness)** → cos=1.000.
   Inserting GPU→CPU→GPU sync at every mul_mat call restores parity.
3. **Same PIN at T=200 (smoke)** → NaN. Confirmed the Round 8 finding.
4. **PIN bisect across op types** at T=102: pinning ANY frequent op
   (norm, mul, add, flash_attn_ext, gelu, reshape, cont, concat,
   permute, mul_mat) restores cos=1.000. Pinning a sparse op (conv_1d,
   soft_max, mish, silu, scale) does not. The pattern is op
   *frequency*, not op identity — sync-barrier density matters.
5. **Per-segment dump bisect** — wrote a debug knob that dumps UNet
   intermediates from each sub-block. Discovered along the way that
   ggml allocator reuses buffers for same-shape outputs (mb_0_out and
   mb_2_out had identical md5 because the allocator placed them at
   the same address after compute finished). `ggml_set_output()` on
   each dumped tensor disables that reuse — required for the dump to
   yield distinct data.
6. **`set_output` on all 62 sub-block intermediates** in the diff
   harness call context → **cos_min = 1.000, max_abs = 0** (bit-perfect).
   `set_output` on 14 (block-level only) → 0.907 (worse than baseline).
   `set_output` on 1 (just db_resnet) → 0.879 (worse). Non-monotonic
   in the number of preservation points.
7. **Same 62-output set_output in the smoke call context** → NaN at
   every T tested (T_mel ∈ {38, 58, 66, 68, 244}).
8. **GGML_NO_INPLACE=1** (global allocator knob to skip the in-place
   reuse path) → cos = -0.97 (sign-flipped garbage). Some downstream
   code relies on in-place semantics in ways we couldn't trace.
9. **GGML_METAL_CONCURRENCY_DISABLE=1** (existing env knob that
   forces serial Metal command encoding) → no effect on drift.
10. **`kernel_norm` audit** — `kernel_norm_fuse_impl` uses
    `float sumf` accumulators end-to-end, F32 `simd_shuffle_xor`
    reduction, F32 shared memory. Not the source.
11. **`kernel_flash_attn_ext` audit** — line 6430 downcasts Q from
    F32 to half: `sq4[j*DK4 + i] = (q4_t) q4[i]` where `q4_t = half4`
    even in the `FA_TYPES_F32` family. Patched `FA_TYPES_F32`'s Q
    triple to `float, float4, simdgroup_float8x8` and rebuilt → cos
    went 0.940 → 0.860 (worse). The F16 Q downcast IS happening but
    "fixing" it changes the numerical ordering in a way that doesn't
    bit-match CPU either.

### The remaining unresolved question

Why does `set_output` on all 62 intermediates achieve
**bit-perfect** parity in the diff harness call context and **NaN**
in the smoke call context, when the chatterbox model, UNet graph
topology, CFM solver loop, and seed are all the same? Invariant
against: random seed, T_mel value, S3-tokenizer presence (.wav vs
precomputed .gguf voice), Metal concurrency on/off, no-inplace,
F32-tile Q in flash_attn. The diff-vs-smoke divergence is structural
in `ggml_gallocr` state across multi-graph sched invocations
(`chatterbox_synthesize_mel` calls fewer ggml graphs into
`c->s3gen_ctx->sched` than `chatterbox_synthesize` does before the
UNet runs), and isolating it would need targeted instrumentation of
the allocator's per-tensor address decisions. Standalone handover
prompt prepared in `docs/prompts/`.

### Three upstream PR drafts (`tools/upstream-prs/`)

- **09 metal-q8_0-bit-match** — the kernel above. Concrete patch
  ready to send to `ggml-org/ggml` (Metal changes can go to either
  repo per the README).
- **10 metal-sched-buffer-reuse-drift** — bug report with full
  bisect evidence. The "set_output on 1 makes it WORSE, on all 62
  makes it BIT-PERFECT" finding is the smoking gun that points
  reviewers at `ggml_gallocr_allocate_node`'s reuse-decision-order
  dependency. No patch — out of scope for an application-side fix.
- **11 metal-sched-nan-large-t** — bug report for the Round 8 NaN.
  Same call-context-sensitivity flavour as 10 but at a different
  symptom level (NaN, not drift).

All three drafts strip CrispASR-internal markers (no `(#83)` refs,
no internal language) per the standing repo rule
([[feedback_strip_local_markers]]).

### Lessons for future ggml-drift hunts on Metal

1. **Pin-any-frequent-op-fixes-it is a sync-barrier-density signal,
   not an op-identity signal.** Don't try to identify "the" buggy op
   by op-type bisect alone — there isn't one. Each "approximate
   enough" GPU op contributes; pinning enough of them to CPU resets
   precision via the sync barrier, regardless of which kernel.

2. **`ggml_set_output` changes allocator reuse decisions in
   path-dependent ways.** A single `set_output` can make drift
   *worse* (0.879 < 0.940 baseline). The reuse decision in
   `ggml_gallocr_allocate_node` depends on traversal order
   (`p_hn->n_children == 1 && p_hn->n_views == 0` — both vary as
   markers are added). For an app-side debug knob to work reliably,
   mark ALL intermediates or NONE, not a subset.

3. **Bit-perfect at small T, NaN at large T** is the canonical Metal
   shape for "mixed-backend op routing exposes a sync race." Round 7
   PIN_CPU_OP=mul_mat hit this. Round 9 set_output on 62 hits the
   same shape from a different angle. Whatever the underlying bug
   is, it scales with the working-set size of the allocator's reuse
   pattern.

4. **The diff harness is not a faithful proxy for the production
   pipeline on Metal.** Same model, same graph, same seed, same T,
   different result (bit-perfect vs NaN). The diff harness call
   context goes through `chatterbox_synthesize_mel_from_tokens`
   which runs fewer ggml graphs into `c->s3gen_ctx->sched` before
   the UNet; the smoke context goes through `chatterbox_synthesize`
   which involves more (S3 tokenizer in `--voice .wav` path, plus
   the vocoder on CPU after). Always test on smoke before claiming a
   Metal fix.

5. **Weight residency split is the safe production answer.** When
   you can't fix the GPU compute, push the entire problematic
   sub-graph to CPU via where the weights live. Avoids per-op
   pinning races. Cost: forgo the GPU speedup for that sub-graph.
   On M1 with a small encoder/vocoder shell, that's nearly free.

### Round 9 follow-up (2026-05-24, evening) — correcting items #6, #7, #11

The investigation continued the same day. **Three of the bisect
items above were wrong**, and the headline "set_output on 62 →
bit-perfect in diff harness / NaN in smoke" was a measurement
artifact. The actual bug shape is much tighter.

**The measurement bug.** `compare_with_row_width` in
`examples/cli/crispasr_diff_main.cpp` aggregated cosine and max_abs
without checking for non-finite data. When the diff harness output
is all-NaN, IEEE-754 silently makes every comparison
short-circuit:

- `if (ad > r.max_abs)` — `NaN > 0.0f` is `false`. `max_abs`
  stays at its initial `0.0`.
- `if (denom > 1e-12)` — `sqrt(NaN_sum_sq) > 1e-12` is `false`.
  The cosine accumulator loop skips every row. `cos_min` stays at
  initial `1.0f`, `cos_mean` stays at `1.0f`.

Result: `[PASS] cos_min=1.000000 cos_mean=1.000000 max_abs=0.00e+00 rms=nan`.
The `rms=nan` was visible the whole time but read as
"cosmetic"; the bogus PASS hid an entirely NaN s3gen_mel. Fixed in
commit `4c2e54c0` — `Report.n_nonfinite` is tracked, surfaced as
`non_finite=N/N`, and forces FAIL.

**With the comparison fixed, the actual picture:**

| Config (GPU residency on Metal, T_mel=102) | Result |
|---|---|
| baseline (no marks) | PASS cos_min=0.940 (real degradation) |
| + PRESERVE (~14 block-out marks) | PASS cos_min=0.907 (slightly worse) |
| + DUMP_UNET (62 marks) | **FAIL non_finite=8160/8160** (all NaN, was bogus PASS 1.000) |
| + minimal 2-mark trigger (see below) | **FAIL non_finite=8160/8160** |
| production (CPU residency) | PASS cos_min=0.999980 |

So item #6's "set_output on all 62 → bit-perfect 1.000" was a
phantom. Item #7's "same 62 → NaN in smoke" was right but
incomplete; **the diff harness was also producing NaN**, just
reported as PASS. Item #11's "F32 Q in flash_attn made cos=0.860"
is also suspect — the comparison was running on whatever data
the patched flash_attn produced; with NaN content this metric
isn't meaningful. The bit-match mul_mat (item #1) and the pin
bisect (items #2–#4) measurements are still valid because their
output stayed finite.

**The actual minimum trigger.** With commit `1a37f4c8`'s granular
bisect knobs, the smallest set of marks that flips the Metal UNet
from "degraded but finite" to "all NaN" is **two specific
`set_output` calls**:

1. `dump_db_resnet` — output of the first DOWN block's
   `causal_resnet_block` (shape `(T_mel, 256)`).
2. Any one of `dump_mb_{0,2,4,6,8,10,11}_out` — output of an
   *even-indexed* MID block (or `mb_11`, the last one).

Both alone are finite. Together → NaN. The parity is reproducible:
`mb_{1,3,5,7,9}_out` paired with `dump_db_resnet` produces
identical finite output (`rms=15.806` deterministic). The same
2-mark trigger fires in both the diff harness and smoke — the bug
is **path-independent**. The handover's "diff-vs-smoke
divergence" was entirely the comparison artifact.

**What that means for the upstream story.** The path-dependence
narrative from upstream-prs/10 was wrong. The real shape is:

- 2 specific `set_output` marks on a long all-F32 Metal graph
  cause the Metal kernel chain to write NaN.
- The parity pattern (even mb_*_out indices trigger; odd don't)
  strongly suggests buffer aliasing in `ggml_gallocr`: when an
  even-indexed mb_out and the db_resnet both reserve their own
  buffers (non-reused), the allocator places them in colliding
  pool regions. Odd indices land elsewhere.
- The visible NaN is then "an early kernel reads from a buffer
  that another kernel is writing to" or "a later kernel reads from
  the wrong-aligned region of an overlapping buffer."

This is a much more tractable upstream bug than the original
phrasing. Repro is now self-contained: two env vars + the public
chatterbox CLI. `tools/upstream-prs/10` rewritten in commit
[TBD] with the tight repro.

### Updated lessons (replace #2 and #4 above)

2'. **`ggml_set_output` is a no-monotonicity sledgehammer for
debug.** A single mark CAN make finite output go all-NaN on Metal.
Don't add `set_output` to "see what's happening at point X" — it
changes the allocator's reuse map and may invent NaN that wasn't
in the un-instrumented compute path. The data you dump is data
the *modified* graph produced, not the *real* graph.

4'. **NaN data must fail the diff harness loudly.** If a tensor
comparison can silently report `[PASS] cos=1.000 max_abs=0`
because both data and reference (or just data) contain NaN, all
your downstream bisect is unreliable. Always count + assert on
non-finite before consuming cosine/max_abs. Same lesson would
apply to any other Metric pipeline that aggregates over potentially
non-finite data.

### Round 9 follow-up #2 (2026-05-24, evening) — gallocr ruled out

The "buffer aliasing in `ggml_gallocr`" hypothesis from the prior
follow-up is **wrong**. The allocator is correct. The bug is
elsewhere.

**The instrumentation.** Added a per-node allocator trace
(`CRISPASR_GGML_ALLOC_TRACE=1`) that emits one line per allocate,
free, in-place-reuse, view-reuse, and FREE_SKIP_OUTPUT event in a
diffable single-line format. Captured pass=37 (the first UNet
allocator pass, n_nodes=2715) for both 1-mark and 2-mark configs.

**The diff.** Of 3044 events in pass=37:
- 2108 lines diverge (~70%).
- Exactly +1 FREE_SKIP_OUTPUT and -1 FREE in the 2-mark case
  (`dump_mb_0_out` now pinned instead of freed).
- 1312 ALLOC events have shifted offsets.
- Max chunk-0 offset reached: 8,310,688 (1-mark) vs 8,701,856
  (2-mark) — exactly +391,168 bytes, the size of `dump_mb_0_out`.
- Both runs use only chunk 0; no spill to a new chunk.

**The overlap check.** Wrote a script that scans the trace and
flags any pair of live tensors whose byte ranges overlap inside
overlapping lifetimes (filtering out the legitimate parent→child
in-place reuse). Result: **0 overlaps in both configurations.** The
allocator's plan is correct; no two distinct live tensors share
bytes. If the bug were aliasing, the script would have flagged it.

**The parity mechanism, finally.** In the 1-mark control trace,
even-indexed `mb_*_out` tensors are allocated at `chunk=0 offset=0`
and odd-indexed at `chunk=0 offset=1271296`. The allocator
naturally alternates them across the two large reusable regions of
the same shape (391,168 bytes). Pinning a *low-offset* output
(via `set_output` → `FREE_SKIP_OUTPUT`) blocks reuse of offset=0
and forces ~1300 downstream allocations to higher offsets. Pinning
a *mid-offset* output (the odd indices) is local — only a small
hole, no cascading shift. Hence the even/odd trigger pattern. The
parity is geometric, not aliasing-driven.

**Where the bug actually is.** It must be at a layer above gallocr:
- a Metal kernel whose correctness depends on the specific
  address pattern produced when downstream allocations are shifted;
- `ggml_metal_op_concurrency_check`'s barrier insertion against the
  shifted address layout;
- the Metal output-staging path that handles `set_output`-flagged
  tensors;
- or sched-level interaction between OUTPUT tensors and split
  boundaries.

`GGML_METAL_CONCURRENCY_DISABLE=1` was already negative, so it's
not the *first-order* barrier check, but it may be the
output-staging side. Item #6 (`kernel_flash_attn_ext` Q downcast)
is still a known F32→half precision loss worth a separate PR; it's
not the 2-mark NaN trigger.

### Updated lesson (replace #2' above)

2''. **`set_output` reshapes the allocator's plan, not its
correctness.** When a debug `set_output` causes downstream NaN, the
gallocr plan is still valid (no live-range overlap); what changed
is the geometric layout of where every subsequent tensor lives.
The bug surfaced by the mark is in whatever code path is sensitive
to *which* addresses the kernels run on, not in the allocator
itself. Before chasing gallocr internals, run the per-node trace
(`CRISPASR_GGML_ALLOC_TRACE=1`) and confirm with an overlap scan;
if it's clean, pivot to the backend's kernel/barrier/output paths.

### Toolbox added

| Knob | Effect |
|---|---|
| `CRISPASR_GGML_ALLOC_TRACE=1` | One trace line per allocator event |
| `CRISPASR_GGML_ALLOC_TRACE_MAX_PASSES=N` | Cap to the first N passes (UNet is at pass=37 in the chatterbox CLI smoke path) |
| `CRISPASR_S3GEN_DUMP_UNET_NO_AUTO_MARK=1` | Lets `DUMP_UNET` dump only tensors kept live by other `MARK_*` knobs — without the implicit "mark every dump point" that triggers the NaN cascade. Required for clean per-stage A/B between CPU and GPU. |
| `CRISPASR_S3GEN_UNET_PROBE_BLOCK1=<N>` | Inline-probes the N-th `causal_block1d` call within one UNet graph. Names + marks `dump_probe_after_{im2col,mul_mat,conv1d,transpose_in,norm,ln_mul,ln_bias,transpose_out,mish}` so the bug can be bisected within a single resnet block. N=0 is `s3.fd.db.0.0.b1`. |

Trace format:
```
ggml-alloc: [TRACE] PASS pass=37 n_nodes=2715 n_leafs=909
ggml-alloc: [TRACE] ALLOC node=<n> op=<op> buf=<b> chunk=<c> offset=<o> size=<s> inplace_from=- view_src=-
ggml-alloc: [TRACE] ALLOC_REUSE node=<n> op=<op> ... inplace_from=<parent> view_src=-
ggml-alloc: [TRACE] FREE node=<n> buf=<b> chunk=<c> offset=<o> size=<s>
ggml-alloc: [TRACE] FREE_SKIP_OUTPUT node=<n> buf=<b> chunk=<c> offset=<o> size=<s>
ggml-alloc: [TRACE] PASS_END pass=37
```

### Round 9 follow-up #3 (2026-05-24, late) — bug localized to the first UNet block; pin workarounds were also a measurement artifact

The new tooling pinned the bug down further. Three things to record:

**1. The bug starts at the very first UNet block.** With
`CRISPASR_S3GEN_DUMP_UNET_NO_AUTO_MARK=1` and only the marks needed
to keep `db_resnet` live, the GPU's first `causal_resnet_block`
output is already wrong: 1280 NaN values across 5 contiguous time
frames (t=260..264) for every channel (256), in a tensor of shape
(T=382, C=256). The downstream `set_output`-marked intermediates
that I previously assumed were the "first" NaN are 100% NaN purely
by propagation. The PROBE_BLOCK1 path further pins this to inside
`causal_block1d` (conv1d→norm→mul→add→transpose→mish), and
because the block2 `causal_conv1d` has `K=3`, its receptive field
expands the 3-frame NaN slice from block1 conv1d output into a
5-frame slice at the resnet block boundary.

**2. The bug is not just precision drift; it is structural.** GPU
vs CPU `db_resnet`: cosine similarity is **-0.09** (essentially
uncorrelated), magnitude is 10× off (GPU `max_abs=14.6` vs CPU
`max_abs=1.9`). This is not the kind of error a small FP16
accumulator drift produces. Setting `GGML_PREC_F32` on conv1d's
internal `mul_mat` makes no difference (rms 13.938 → 13.942,
within noise). The wrongness is in the structural arithmetic of
some Metal kernel chain, not in accumulator precision.

**3. The handover's "pinning any frequent op restores cos = 1.0"
claim was also a measurement artifact.** Re-tested
`CRISPASR_S3GEN_UNET_PIN_CPU_OP=norm`, `=mul_mat`, `=add`, `=cont`
with the fixed `crispasr-diff`. All four now FAIL with
`non_finite=8160/8160`. The original PASS reports came from the
pre-`4c2e54c0` harness that silently scored all-NaN as
`cos=1.000`. So both the "pin to CPU restores baseline" and the
"set_output on 62 → bit-perfect" claims were the same one
measurement bug. With it fixed, **no per-op pin currently fixes
the GPU baseline**. The only working configuration remains the
production weight-residency split (`s3.fd.*` on CPU).

**4. The bug is layout-dependent and moves with `set_output`.**
Adding the PROBE_BLOCK1 named outputs cleared the conv1d NaN (its
intermediates were finite) but pushed the first observable NaN to
`dump_db_tb_0` (the basic_transformer_block output). The final
audio was still NaN. This is the classic signal that the bug is a
GPU kernel correctness issue sensitive to which addresses the
allocator picks — adding marks shifts addresses and the kernel
breakage moves.

**Updated lesson (replaces 2'' above):**

2'''. **GPU-residency UNet on Apple Metal is fundamentally broken
in a way no in-tree workaround addresses.** The production fix
(weight-residency split) is not just a perf-comparable
convenience; it is the only configuration that produces correct
output. Documented per-op pins, single-mark and PRESERVE marks,
F32 precision hints, and `GGML_METAL_CONCURRENCY_DISABLE` all
either don't help or actively break it. A real fix needs Metal
shader-level investigation against the address pattern the
allocator produces for this graph — out of scope for a normal
chatterbox session. Until then: `CRISPASR_S3GEN_UNET_GPU_RESIDENCY`
remains an investigation-only knob, not a user-facing option.

### Round 9 follow-up #4 (2026-05-24, late) — TWO bugs in tandem; one fixed, one worked-around

R9 follow-up #3's "address-dependent Metal kernel bug" conclusion
was wrong. The actual root cause is much simpler and tractable.

**Method.** Captured CPU vs GPU bytes at every probe point in the
first `causal_block1d` (`s3.fd.db.0.0.b1`) using
`CRISPASR_S3GEN_UNET_PROBE_BLOCK1=0` + `DUMP_UNET=<tag>` +
`DUMP_UNET_NO_AUTO_MARK=1`. The probe path's `after_im2col` already
showed cos=-0.085 between the two runs — the VERY first GPU kernel
of the UNet was producing structurally wrong output.

A by-hand inspection (`tools/inspect_im2col_dump.py`) confirmed the
kernel's causal-padding shape was correct (first 2 of every 3 floats
zero, third populated). The bug was the VALUES being populated:
CPU read `x[0,0]=1.93` (the first Gaussian noise sample), GPU read
`x[0,0]=-4.05` (a mel-scale value). The im2col kernel arithmetic
was fine; it was reading the wrong source.

A short diagnostic logged the host-side `unet_input[0..7]` values
before upload and a `ggml_backend_tensor_get` readback right after.
Both bit-identical between CPU and GPU runs. So the host array was
the same and the upload landed correctly.
`ggml_backend_sched_get_tensor_backend(sched, unet_input)` returned
`CPU` even under `GPU_RESIDENCY=1` — confirming sched put the
input on CPU and the existing gallocr trace's `MTL0#unet_input#0`
copy was the data path GPU kernels actually read from.

**Root cause.** With UNet weights GPU-resident, the three runtime
inputs (`unet_input`, `time_emb`, `mask`) auto-assigned to the CPU
backend. The scheduler creates an auto-copy
(`{backend_name}#{input}#{copy_idx}`) for the first Metal split that
consumes them; that copy lives in
`ggml_backend_sched_compute_splits` at `ggml-backend.cpp:1555-1567`.
In this 29-split graph, the copy does NOT correctly deliver the
user's uploaded data to the Metal kernel — the kernel reads stale
content at the destination offset. Under certain `set_output`
layouts (the 2-mark trigger from follow-ups #1–#3) the stale
values turn into NaN.

**Fix has two parts** (both required; one without the other still produces wrong output):

**Part A — ggml-side patch** for `ggml_backend_sched_compute_splits`
(ggml-backend.cpp). On every call, sched_split_graph mutates
`gf->nodes[i]->src[j]` to point at fresh `MTL0#input#0` tensors it
just created in `sched->ctx`. Those tensors live until the NEXT
call's `ggml_free(sched->ctx); sched->ctx = ggml_init(...)`, which
happens at the top of the next sched_split_graph. So between calls,
the user's `gf->nodes[i]->src[j]` is a DANGLING POINTER. The next
call's sched_split_graph reads from that dangling pointer to decide
whether an input copy is needed — and the garbage flags rarely match
`GGML_TENSOR_FLAG_INPUT`, so it skips creating new copies. The
consuming Metal kernel then reads stale content at the previous
input_cpy's offset.

Fix: keep a mutation log in sched, and restore each `node->src[j]`
to its original at end of `ggml_backend_sched_compute_splits`. The
user's gf is left untouched between sched calls; the next call sees
the real `GGML_TENSOR_FLAG_INPUT` tensors and creates fresh copies.
Marked `// CrispASR patch (#83 r9 follow-up #4)` with MUST RE-APPLY.

**Part B — application-side pin of `unet_input` only**. Even with
Part A, the sched copy of `unet_input` (CPU buffer → MTL0#unet_input#0)
produces correct bytes at the kernel (verified by inline
`ggml_backend_tensor_get` right before dispatch) yet downstream
compute diverges. Cause not fully traced; sidestepped by placing
`unet_input` on the consuming Metal backend so no copy is needed.

```cpp
if (c->unet_on_gpu) {
    ggml_tensor* ui = ggml_graph_get_tensor(gf, "unet_input");
    if (ui) ggml_backend_sched_set_tensor_backend(c->sched, ui, c->backend);
}
```

**Bisect of which inputs to pin (all under Part A in place):**

| Pinned to GPU | smoke rms | result |
| - | - | - |
| (none) | 14.6 | broken |
| `unet_input` only | **5.14** | works ✓ |
| `time_emb` only | 14.4 | broken |
| `mask` only | 16.0 | broken |
| `unet_input` + `time_emb` | 209 | catastrophic |
| `unet_input` + `mask` | 5.14 | works ✓ |
| all three | 5.14 | works ✓ |

Pinning `time_emb` forces `mish` onto Metal (its input now lives on
GPU), which changes the resnet block's cross-backend topology and
interacts badly with the rest of the graph. **Pin minimally**.

**Verification matrix (M1 Metal):**

| Configuration | Before | After |
| - | - | - |
| Baseline GPU residency, `--tts "Hello."` smoke | `rms=13.938` | `rms=5.143` (ref 5.115) |
| 2-mark NaN trigger | `rms=NaN` | `rms=5.143` |
| Long text (~9 s) | `rms=13.045` | `rms=4.741` |
| Diff harness `s3gen_mel` | `cos_min=0.940` | `cos_min=0.999976` |
| Production CPU residency (default) | `rms=5.139` | `rms=5.139` (no change) |

`cos_min=0.999976` matches the production weight-residency split's
0.999980 (parity within FP16/F32 round-off). The GPU-residency
path is now correct AND faster (~30% on M1 because the encoder /
vocoder no longer need a backend split).

**Replaces lessons 2'/2''/2'''/3 about Metal kernel debug:**

2''''. **For mixed-residency GPU graphs, pin runtime inputs to the
GPU backend explicitly.** ggml's scheduler prefers CPU for tensors
with `GGML_TENSOR_FLAG_INPUT` to minimise upload overhead. The
auto-generated CPU→GPU copy works for short graphs but fails
silently in long F32 graphs with many backend splits (this case:
29 splits). Explicit pinning is a one-line fix and removes the
failure mode. Diagnostic: call
`ggml_backend_sched_get_tensor_backend(sched, input_tensor)` after
`alloc_graph` — if it's `CPU` when you expect `GPU`, pin it.

3'. **First bisect step for "GPU output differs from CPU" should be
"compare bytes at the FIRST kernel's output."** If they already
differ, the bug is in the input pipeline (upload, sched
placement, sched copy), not in the kernel chain. Three rounds of
shader-level investigation were wasted on what turned out to be
an input-routing issue. The fast diagnostic is:

```
CRISPASR_S3GEN_UNET_PROBE_BLOCK1=0 \
CRISPASR_S3GEN_DUMP_UNET=cpu+gpu-probe \
CRISPASR_S3GEN_DUMP_UNET_NO_AUTO_MARK=1
```

then diff `dump_probe_after_im2col.bin` between CPU and GPU runs.
If cos<<1 from the very first probe, look at sched, not at
shaders.

**Upstream story.** The dangling-pointer bug (Part A) is a real
ggml-side issue that affects any user code which calls
`ggml_backend_sched_alloc_graph + graph_compute` twice on the same
gf — the second call silently loses input copies. Filed at
`tools/upstream-prs/10`.

**Bug B is still open** — the workaround (pinning `unet_input` to
the consuming backend) is shipping in production, but the underlying
divergence between "input on CPU + sched-copies to Metal" and "input
directly on Metal" when downstream kernels see the same input bytes
is not understood. Handover at
`handover-prompts/issue83-r9-followup-5-unet-input-routing.md`
collects every diagnostic we ran, every hypothesis we ruled out, and
the structural shape of the bug (the kernel reads correct bytes but
produces wrong output that compounds through CFM). A real fix needs
either an upstream-quality root-cause patch or a confident
"workaround is the right answer because X" explanation.

**Lesson 7. "Pinning fixes it" is not a root cause.** When a
backend-pin diagnostic resolves a correctness issue, you've found
A fix, not necessarily THE fix. The instinct to ship and move on is
strong; resist it. Sidestepping a bug you can't characterise leaves
the same trap loaded for the next graph topology that triggers it.
Carve out time to chase Bug B-class symptoms to root cause even
after a workaround makes the harness green.

`tools/compare_probe_dumps.py` + `tools/inspect_im2col_dump.py`
are the diagnostic scripts.

### Round 9 follow-up #5 (2026-05-24, very late) — Bug B narrowed to rc.weight residual conv; root cause still unidentified

Continued the Bug B chase from #4. Did not find root cause but
eliminated several hypotheses and localized the divergence point.

**Verified (from #4's open questions):**

- Host `ggml_backend_tensor_get(MTL0#unet_input#0)` immediately before
  the FIRST UNet `im2col` dispatch returns the correct Gaussian-noise
  bytes (`1.9269 1.4873 …`) at `data=0x159114000 offs=0`. Same buffer
  and offset across all 4 dispatches observed (cond + uncond × 2
  steps). The host side is fine.

**New finding — where Bug B actually manifests:**

Used `CRISPASR_S3GEN_UNET_PROBE_BLOCK1=<N>` to compare per-block
intermediates between Path X (workaround) and Path Y (no-pin):

| Probe target | Stage | cos(Path X vs Y) |
| - | - | - |
| Block 0 (`db.0.0.b1`) | after_mul_mat → after_mish | 1.00000 |
| Block 1 (`db.0.0.b2`) | after_mul_mat → after_mish | 1.00000 |
| `dump_db_resnet` (resnet block output) | post-residual-add | 0.32261 |
| `dump_rc_out_db00` (rc conv output, **new probe**) | F32 residual conv | **0.02211** |
| Block 2 (`mb.0.0.b1`) | after_im2col | 0.36914 |
| Block 13 (`mb.5.0.b2`) | after_im2col | 0.70562 |
| Block 27 (`ub.0.0.b2`) | after_im2col | 0.92610 |

So Bug B's first divergence is at the **residual conv** (`rc.weight`)
inside `causal_resnet_block` at `s3.fd.db.0.0`. b1 and b2's conv1d
outputs are bit-identical between the two paths — both read from the
same Metal compute-buffer offset 0 (MTL0#unet_input#0 in Path Y,
unet_input in Path X). The rc conv reads from the **same** offset
yet produces output with `rms=0.0645` (Bug B) vs `rms=0.3631`
(workaround) — a ~5.6× magnitude collapse and `cos=0.022`. Bug B's
values look like rc.weight × ~zero input; workaround's values look
like rc.weight × Gaussian-noise input (σ≈1).

**Eliminated hypotheses (all left rms ≈ 16 unchanged):**

1. **CPU store-buffer / cache coherency.**
   `CRISPASR_FORCE_DMB=1` inserts `__sync_synchronize()` after the
   shared-buffer host memcpy. No effect — the GPU was already seeing
   the writes correctly. (The host-side readback proved this from the
   start; the experiment just rules out a write-buffer race
   conclusively.)
2. **GPU-side cache invalidation.** `CRISPASR_FORCE_BLIT_COPY=1`
   replaces the host memcpy with a Metal blit-encoder copy in its own
   committed command buffer. No effect — the GPU is performing the
   copy itself, fully synchronous to subsequent compute. Bug B
   persists.
3. **Intra-encoder concurrency.** `GGML_METAL_CONCURRENCY_DISABLE=1`
   and `GGML_METAL_GRAPH_OPTIMIZE=0` both leave Bug B unchanged.
4. **Cross-command-buffer race.** Metal splits each
   `graph_compute_async` across `n_cb+1` command buffers; each cb has
   its own `mem_ranges`, so cross-cb conflicts are not tracked at the
   ggml layer. Setting `CRISPASR_METAL_N_CB=2` produces rms=16.217
   (essentially unchanged from default n_cb=1 rms=16.129). Setting
   `n_cb=0` is broken (division-by-zero in `n_nodes_per_cb`). So
   command-buffer fan-out is not the cause.
5. **Missing or stale `mem_ranges` barrier.**
   `CRISPASR_METAL_FORCE_BARRIER=1` emits a Metal memory barrier
   before EVERY metal-op encode (effectively serializing the whole
   graph). Bug B still produces rms=16.183. So no concurrency hazard
   inside the encoder is responsible.

**Bug shape — unanswered:**

Same Metal pipeline, same kernel args, same input bytes (verified
host-side), same Metal buffer offset. b1 and b2's conv1ds read this
input correctly. rc's conv1d does not. The only differences I can
think of between b1's im2col and rc's im2col on the same input:

- b1 uses KH=1, KW=3, p0=2 (causal-padded). Threads with `iiw < 0`
  branch to write `0.0f`.
- rc uses KH=1, KW=1, p0=0. No padding; all threads enter the
  data-read branch.

Both use the same `kernel_im2col_f32` template with identical args
plumbing; there is no KW=1 fast path. Both dispatches are
`(IC=320, OH=1, OW=T_mel)` threadgroups × `(N=1, KH, KW)` threads.

**Loose lead — alloc-plan aliasing within the buf=0 chunk:**

In the alloc trace (with probe + dump_db_resnet preserved), the
first UNet pass shows `MTL0#unet_input#0` at offset 0, then
`node_47 op=IM2COL` (rc's im2col) is allocated at offset 6024704
while `MTL0#unet_input#0` is **freed** by gallocr the moment node_47
is registered. Then `node_51 op=MUL_MAT` (rc's mul_mat) is allocated
**at offset 0**, reusing MTL0#unet_input#0's just-freed slot. The
same plan exists in Path X (with unet_input itself at offset 0
instead of MTL0#unet_input#0), and Path X works — so the pure
aliasing isn't the cause, but the interaction of "sched-injected
input_cpy at offset 0 vs user-pinned input at offset 0" remains
suspicious.

**Operational note — dump artifacts:**

`ggml_set_output` on a probe target keeps that one slot live but
adds new tensors to the graph (the implicit "(transposed) (cont)"
follow-ups), which **changes the alloc plan downstream**. A run
configured with `PROBE_BLOCK1=1 + MARK_DB_RESNET=1` broke the
workaround (rms=17.355). So you cannot freely add probes to compare
"Path X with marks" vs "Path Y with marks" — the markings perturb
the graph in their own right. Future investigators: probe one path
exclusively per run, or accept that mark-induced offset shuffling is
itself part of the experiment.

**Diagnostic knobs added (env-gated, no runtime impact when unset):**

- `CRISPASR_NO_INPUT_PIN` — disables the unet_input pin workaround,
  for reproducing Bug B.
- `CRISPASR_IM2COL_DBG` — logs first 8 host bytes of `op->src[1]` for
  the first UNet `kernel_im2col_f32` dispatch.
- `CRISPASR_FORCE_DMB`, `CRISPASR_FORCE_BLIT_COPY`,
  `CRISPASR_METAL_FORCE_BARRIER`, `CRISPASR_METAL_N_CB` — the
  hypothesis-elimination knobs above.
- `CRISPASR_S3GEN_UNET_PROBE_RC_OUT` — dumps rc's residual-conv
  output at `s3.fd.db.0.0` as `dump_rc_out_db00`.

**Also eliminated: Metal op fusion.** Set
`GGML_METAL_FUSION_DISABLE=1` to disable the cross-op fusion
optimizer; Bug B's smoke is still rms=16.129. So the rc-specific
divergence isn't an artifact of the fusion code path either.

**Structural difference between b1 (works) and rc (broken):**

Both fire `kernel_im2col_f32` on the same `MTL0#unet_input#0` at
the same Metal buffer offset 0, inside the SAME sched split
(SPLIT #0 of the UNet1D graph — verified via `GGML_SCHED_DEBUG=1`,
which shows just one Metal split with unet_input as its input; all
subsequent Metal splits consume only the time_proj outputs that
loop back from CPU). The only mechanical differences are:

| | b1's conv1d | rc's conv1d |
| - | - | - |
| `KW` (kernel width) | 3 | 1 |
| `p0` (left pad) | 2 | 0 |
| im2col output `ne[0]=CHW` | 960 | 320 |
| Threads per threadgroup (`ntptg0, KH, KW`) | (1, 1, 3) | (1, 1, 1) |
| Causal-pad branch enters | yes (for iow=0..1) | never |

So rc dispatches `kernel_im2col_f32` with a 1×1×1 threadgroup — one
active lane, 31 masked-off SIMD lanes on the M1's 32-wide simdgroup.
That's an unusual but documented configuration. Whether this
single-lane threadgroup tickles a driver/compiler bug on M1 specifically
is a leading hypothesis but unverified. The next experiment would
either:

**Tested — eliminated: 1×1×1-threadgroup im2col edge case.**
Added `CRISPASR_S3GEN_RC_AS_MUL_MAT=1` that bypasses `ggml_conv_1d`
for rc when KW=1 and emits a direct
`mul_mat(cont(transpose(residual)), reshape_2d(rc_w))` instead. Test
results on M1:

| Configuration | rms |
| - | - |
| Path X (workaround), `RC_AS_MUL_MAT=1` | 5.143 (matches workaround baseline — correctness preserved) |
| Path Y (Bug B), `RC_AS_MUL_MAT=1` | 16.837 (Bug B still present) |

So the 1×1×1-threadgroup im2col edge case is **NOT** the cause of
Bug B. Bypassing rc's im2col leaves Bug B intact. Whatever's wrong
with Path Y's rc compute is also wrong with a mul_mat-only rc compute
on the same `MTL0#unet_input#0` buffer. The divergence is not in
the rc kernel itself.

This is a significant negative result — it pushes the bug AWAY from
rc and suggests the cos=0.022 measurement of `dump_rc_out_db00` may
be a downstream symptom rather than the causal location. Working
hypotheses to try next:

- The `dump_rc_out_db00` divergence might be a **dump artifact**
  itself: `MARK_DB_RESNET=1 + PROBE_BLOCK1=1` already broke the
  workaround at rms=17.355, so probe-induced offset shifts are
  measurably destabilising. The rc-output dump landing at a
  different offset in Path X vs Path Y could mean we're reading
  different bytes for reasons unrelated to rc's correctness.
- **Capture per-thread reads via a custom Metal kernel.** Replace
  `kernel_im2col_f32` with a debug variant that writes
  `(in*ofs0 + iic*ofs1 + iiw, x[offset_src])` pairs to a side
  buffer. Compare what GPU thread (iic, iow, ikw) saw vs what
  host expected.
- **Probe the SECOND CFM step** specifically. The first step's
  compute uses freshly memcpy'd data; subsequent steps reuse the
  same Metal compute buffer offsets with new memcpy contents. If
  a buffer's state from step N-1 leaks into step N's compute,
  the divergence might only appear from step 1 onward — and the
  cumulative compound through 10 steps drives the rms gap.
- **Pin only `time_emb` or only `mask`** (R9 #4's bisect tested
  these but didn't try probing per-step compute outputs). The
  combinations may encode which sched-injected input_cpy is
  problematic.

**Bug B remains open.** Workaround is still shipping.

**MAJOR ISOLATION (2026-05-24, late) — Bug B is in the UNCOND pass only.**

Each CFM step calls `run_denoiser` twice: first with the conditioned
input `[noise, mu, spk, cond]` (cond pass), then with
`[noise, 0, 0, 0]` (uncond pass) for the CFG combination. With a
dup-named `dump_denoiser_out` tensor + `CRISPASR_S3GEN_UNET_PROBE_DENOISER_OUT=1`
and a static-counter cond/uncond-tagged DUMP_UNET, the step-0
denoiser_out comparison between Path X (workaround) and Path Y
(Bug B) is:

| Pass | bug rms | wrk rms | cos(bug, wrk) |
| - | - | - | - |
| cond   | 3.9634 | 3.9634 | **1.00000** |
| uncond | 0.9474 | 5.0228 | **0.21431** |

The cond pass is BIT-IDENTICAL between the two paths. Bug B only
manifests on the **second** `graph_compute` call within a CFM step.
The cond and uncond runs are structurally identical (same gf, same
sched topology, same alloc plan); only the host-side input data
differs.

This rules out anything intrinsic to sched-copying `unet_input` — the
first sched-copy + compute works perfectly. The bug is in what
happens *between* the cond compute and the uncond compute, or
specifically in the uncond compute reading the newly-memcpy'd
unet_input data.

Working hypothesis: on Apple Silicon, the GPU's L1/L2 caches retain
shared-mode buffer state across consecutive command-buffer
submissions. The cond compute leaves cached data; the uncond
CPU memcpy overwrites system memory, but no Metal API invalidates
the GPU's caches before the next dispatch. The uncond kernel then
reads stale data the memcpy never reached. None of the tested
barriers (`FORCE_DMB`, `FORCE_BLIT_COPY`, `FORCE_BARRIER`,
`METAL_N_CB`) address this — they all run AFTER the memcpy but
before the new compute, where they SHOULD trigger cache
invalidation, but apparently don't on M1 for shared-storage
buffers modified by CPU.

Observation: Bug B's uncond denoiser_out has rms=0.95 (very small,
near-bias-only) while workaround's is rms=5.02. The first 8 values
have CONSISTENT SIGNS and gradients but ~5× smaller magnitude. This
is what you'd expect if the model received near-zero or attenuated
input on the GPU side (consistent with the kernel reading something
other than the noise channel).

Critical next experiment: instead of trying to fix the cache via
barriers, **force MTL0#unet_input#0 onto a private-storage buffer**.
Private mode requires explicit blit transfers (which Metal manages
cache state around) — if Bug B disappears, the bug is confirmed as
"M1 GPU caches shared-storage buffer reads across cmd-buf
boundaries, no Metal API invalidates them when the CPU writes
between cmd-bufs". A more surgical workaround would then be to
keep `unet_input` on a private buffer specifically in this path.

Knobs added this session:

- `CRISPASR_S3GEN_UNET_PROBE_DENOISER_OUT=1` — emits a dup-named
  `dump_denoiser_out` tensor so the existing DUMP_UNET filter
  catches it.
- `CRISPASR_S3GEN_DUMP_UNET=<tag>` — dump files now include
  `<tag>-cond` / `<tag>-uncond` prefixes via a static call counter
  inside `run_denoiser`.
- `CRISPASR_S3GEN_RC_AS_MUL_MAT=1` — bypasses `ggml_conv_1d` for
  the residual conv (the 1×1×1 im2col edge case). Confirmed
  irrelevant to Bug B.

**Conclusive host-vs-GPU divergence proof.** Extended IM2COL_DBG
to read channels 0/80/160/240 (noise/mu/spk/cond) instead of just
channel 0. In Bug B mode with `CRISPASR_IM2COL_DBG=1`:

```
dbg 0 (step 0 cond):    noise=1.9269 mu=0.24 spk=0.17 cond=-6.75
dbg 1 (step 0 uncond):  noise=1.9269 mu=0.00 spk=0.00 cond=0.00
dbg 2 (step 1 cond):    noise=1.8502 mu=0.24 spk=0.17 cond=-6.75
dbg 3 (step 1 uncond):  noise=1.8502 mu=0.00 spk=0.00 cond=0.00
```

All four host_get readbacks return the EXACTLY CORRECT bytes —
mu/spk/cond are zero in uncond passes, populated in cond. Same data
ptr `0x121120000`, same Metal buffer offset 0, same shared-mode
buffer across all four dispatches.

Yet the compute output is still wrong in Bug B (rms=16.245). So
the GPU's view of `MTL0#unet_input#0` in the second `graph_compute`
(uncond) differs from what host_get returns at the SAME byte offset
of the SAME shared-storage Metal buffer.

Already tested and ruled out for fixing this divergence:
`FORCE_DMB`, `FORCE_BLIT_COPY`, `FORCE_BARRIER`, varying
`METAL_N_CB`, `GGML_METAL_CONCURRENCY_DISABLE=1`,
`GGML_METAL_FUSION_DISABLE=1`, `GGML_METAL_GRAPH_OPTIMIZE=0`,
`GGML_METAL_SHARED_BUFFERS_DISABLE=1` (private-storage buffers),
`CRISPASR_S3GEN_RC_AS_MUL_MAT=1`.

This is now an extremely well-characterized but genuinely
intractable Metal/M1 bug. The cond pass works correctly; the
uncond pass doesn't, on identical-shape graphs with identical
sched alloc plans and identical host-visible buffer bytes.

Likely next steps (none cheap):

- Capture an Xcode GPU frame trace of the cond and uncond passes
  side by side; compare buffer state inspection at dispatch time.
- File a minimal reproducer with Apple (chatterbox sched + sched
  tensor_copy + repeated CPU memcpy on a shared-storage Metal
  buffer + back-to-back `commandBufferWithUnretainedReferences`).
- Switch the sched layer's INPUT-flagged `ggml_backend_tensor_copy`
  path to issue a Metal compute kernel (not a memcpy) for the
  CPU→Metal copy step on shared-storage buffers. This would route
  through a known-working Metal command pipeline rather than a
  raw host write.

**Bug B remains open.** Workaround is still shipping.

---

## FA per-head additive mask CUDA kernel — what the upstream signature already gave us (issue #81 #06, May 2026)

### Read the kernel signature before estimating patch size

The first design pass on the per-head FA mask patch (in the now-
retired `issue81-phase1-uar-wip` branch) estimated ~300-500 LOC
across 4-6 .cuh files — touching the launcher, the kernel-body
mask load, the tile_mask broadcast, and possibly the
quantization fast path. The actual minimal patch is **4 LOC of
kernel-body code** (+ ~10 LOC of dispatch gate + CMake/CDef
wiring) for a total of ~45 LOC.

The difference is that the MMA-F16 kernel signature
(`flash_attn_ext_f16` in `fattn-mma-f16.cuh`) **already took all
six mask dims/strides as parameters** — the launcher plumbed
`nb31`, `nb32`, `nb33`, `ne31`, `ne32`, `ne33` through. The
kernel body just never offset by `nb32`:

```cuda
// before:
const half * mask_h = ncols2 == 1 && !mask ? nullptr :
    (const half *) (mask + nb33*(sequence % ne33));
// after (gated by GGML_CUDA_CRISPASR_FA_PERHEAD_MASK):
const half * mask_h = ncols2 == 1 && !mask ? nullptr :
    (const half *) (mask + nb33*(sequence % ne33)
                         + nb32*(zt_Q     % ne32));
```

For `ne32 == 1` (broadcast mask — upstream's only supported case)
`zt_Q % 1 == 0` and the new term is zero, so default-OFF builds
are bit-identical to upstream.

**Lesson**: when sizing a kernel patch, grep for the strides
you'll need in the kernel signature first. If they're already
plumbed but unused (compiler probably already warns), the patch
is a one-liner inside the kernel body; if they aren't, you have
to widen the launcher too. The original 300-500 LOC estimate
predated reading the signature carefully.

### One CPU FA op per layer forces ≈2 scheduler splits, not 1

Sched-debug A/B on the parakeet short clip (3 chunks, 24 layers,
q8_0, GGML_SCHED_DEBUG=2):

| | total splits | CPU splits | CUDA0 splits | `FLASH_ATTN_EXT` total |
|---|---:|---:|---:|---:|
| FA on CPU (pre-patch) | 147 | 72 | 75 | 72 |
| FA on CUDA0 (post-patch) | 3 | 0 | 3 | 72 (all CUDA0) |

Pre-patch: 24 FA ops/chunk × 3 chunks = 72 CPU FLASH_ATTN nodes.
The *splits* they force are ~49 per chunk (147 / 3), not 24 —
each CPU FA breaks the graph into a CPU split for the FA itself
plus a CUDA0 split for the next layer's GPU work, and the inputs
to the FA (KQV proj outputs straddling the boundary) often spawn
their own splits. Post-patch: a single CUDA0 split per chunk
swallows the whole encoder pass.

**Lesson**: when planning a "move op X from CPU to CUDA" patch,
the wallclock win comes from eliminating *the splits the CPU op
forced around itself*, not just from running op X faster. Count
splits, not ops, when estimating impact. The split-count
difference (147 → 3) is also the cleanest correctness signal —
if it doesn't drop near 1-per-chunk after the patch, the
dispatch gate didn't relax the way you thought it did.

### Per-tile mask offset is only correct for ncols2 == 1

The clean 4-LOC mask offset above is mathematically correct when
`ncols2 == 1` — one Q head per MMA tile. For `ncols2 > 1` (the
GQA-folded path: `gqa_ratio > 1` and the launcher chose to fold
multiple Q heads into one tile), the same expression reads
head `zt_Q`'s mask and silently broadcasts it across the
`ncols2` heads in the tile — **wrong outputs, no crash**.

The fix-of-the-fix is a hard gate at dispatch (in `fattn.cu`):
```cpp
const bool mask_is_per_head = (mask && mask->ne[2] != 1);
if (mask_is_per_head && gqa_ratio > 1) {
    return BEST_FATTN_KERNEL_NONE;  // CPU fallback, no regression
}
```

No current CrispASR model has both per-head masks *and*
`gqa_ratio > 1` (parakeet / canary / FastConformer-CTC are all
`gqa_ratio == 1`), so this gate is a no-op for the workloads we
care about. But if a GQA-conformer ever lands, the gate prevents
silent corruption — caller falls back to CPU FA, == upstream
pre-patch behaviour.

**Lesson**: when you patch a per-tile kernel offset under a
broadcast assumption, audit the broadcast cases (`ncols2` here)
and add a dispatch-level gate for the cases your offset doesn't
cover. Silently-wrong-result kernels are the worst class of bug;
crashing is a feature, not a bug, in patches gated default-OFF.

### WDDM warmup doesn't survive a Python process boundary

The published `dll-postsiglu` warmup protocol —
`probe_postsiglu_leak.py <dll> 200` then
`benchmark_asr_engines.py …` — keeps the A1000 warm only if the
two scripts run back-to-back **with a previously hot GPU**. In a
fresh shell on a cold A1000, the second script's Python startup
+ model mmap + JIT prewarm pass (~3-5 s) is enough idle time for
WDDM to drop the GPU from P0 back to P8. Our FA per-head A/B
captured both runs at P8 / 315 MHz the whole time, suppressing
the expected 10-15 % wallclock win to 2 %.

The clean fix is a single-process warmup driver: put the
keepalive loop and the bench in **one** process so there's no
idle gap between them. The probe-then-bench pattern works only
when the GPU is *already* warm (e.g. when an earlier bench in
the same shell session just finished pushing work). Documented
above at "WDDM idle-clock-state hysteresis on consumer/laptop
NVIDIA SKUs (May 2026)".

**Lesson**: a warmup procedure that crosses a process boundary
on Windows + laptop NVIDIA is racing the WDDM idle timer. Treat
the cross-process probe-then-bench pattern as "works iff GPU was
already warm", and use a single-process keepalive when starting
cold.
