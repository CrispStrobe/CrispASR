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

### Capture tensors MUST call `ggml_set_output()`, not just `ggml_set_name()`

Found while wiring stage-by-stage diff captures into the mimo-asr
prefill graph. We had:

```cpp
ggml_tensor* audio_features = ggml_cont(ctx0, x);
ggml_set_name(audio_features, "prefill_audio_features");
ggml_build_forward_expand(gf, audio_features);  // not enough!
// ... later in the same graph ...
ggml_tensor* inputs_embeds = ggml_add(ctx0, text_embeds, x); // shares x with cont above
```

Both `audio_features` (cont of x) and `inputs_embeds` (consumes x)
read from the same MUL output buffer. With only `set_name`, the
backend scheduler treated audio_features as an ordinary intermediate
and reused its buffer when allocating later ops in the same graph.
By the time we read it back via `ggml_graph_get_tensor`, the values
were post-clobber. Symptom: per-stage cosines look fine in isolation
(audio_features cos≈0.998), but a downstream consumer's output
collapses to ~0 against the reference (inputs_embeds cos≈0.003)
even though both inputs to its `ggml_add` extracted at cos≥0.99
individually.

Fix: every tensor that the host plans to extract via
`ggml_graph_get_tensor` (or via `ggml_backend_tensor_get` against a
named graph node) must be marked `ggml_set_output(...)` in addition
to `ggml_set_name(...)`. `set_output` tells the scheduler the buffer
must persist past compute; `build_forward_expand` only marks the node
as a graph target, not as a long-lived output.

The same pattern applies to any backend that mixes "named capture
points" with "downstream consumers of the same intermediate" in one
forward graph.

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

### Pre-emphasis is not Cohere-specific — it's the NeMo cluster default

`AudioToMelSpectrogramPreprocessor` (and the underlying
`FilterbankFeatures`) defaults `preemph=0.97` and applies the filter
`y[i] = x[i] - 0.97 * x[i-1]` at inference time before STFT. **All four
NeMo-cluster models inherit this default** unless their training config
explicitly overrode it: parakeet, canary, canary_ctc, cohere.

Originally we only had pre-emphasis on cohere (because cohere's
inline mel code happened to include it) and missed it on the other
three. Symptom on parakeet-tdt-0.6b-ja: short Japanese clips dropped
content NeMo caught (issue #37 — reazon_meal_11s lost
`お腹すいた … うん`). Stage-by-stage diff against
`tools/dump_reference.py --backend parakeet` showed
`mel_spectrogram cos_mean = 0.990` even before any encoder layer ran.

Fix: add `preemph` field to `core_mel::Params`, apply BEFORE the
center-pad (so `y[0] = x[0]` is preserved against the true first
sample, matching NeMo's `torch.cat((x[:,0:1], x[:,1:] - α*x[:,:-1]))`).
Set `p.preemph = 0.97f` in the parakeet / canary / canary_ctc wrappers.
Cohere keeps its existing inline pre-emphasis (left it as-is to
preserve a tested code path).

Verified post-fix: `mel_spectrogram cos_mean = 0.999451` on the meal
sample and the major deletion is gone. **If you add a new NeMo-cluster
model, set `p.preemph = 0.97f` unless you have read its checkpoint's
config and confirmed it overrides the default.**

### NeMo PerFeatureZ uses Bessel-corrected std with eps OUTSIDE the sqrt

`normalize_batch(x, "per_feature")` is:

```python
var = sum_sq / (T - 1)              # Bessel-corrected
std = sqrt(var); std = 0 if NaN     # NaN guard for T == 1
x = (x - mean) / (std + 1e-5)       # eps OUTSIDE the sqrt
```

Two pitfalls to avoid:
1. Population variance `/N` instead of sample variance `/(N-1)`
2. `sqrt(var + eps)` (eps inside the radical) instead of
   `sqrt(var) + eps`

Both shipped wrong originally and shifted low-variance mel bands enough
to compound through 24 conformer layers. Effect was small on clean
speech and noticeable on conversational JA (issue #37 again). The
Bessel correction matters more on short clips where T is small;
the eps placement matters on quiet silence-heavy bands at any T.

### NeMo STFT uses zero-pad, not reflect-pad

PyTorch's `torch.stft(center=True)` defaults to `pad_mode="reflect"`,
but NeMo's `FilterbankFeatures.stft` explicitly passes
`pad_mode="constant"` (= zero-pad). Our `core_mel` already does zero
center-pad, so this matches without changes. Worth noting because the
"obvious" guess from the PyTorch defaults would be wrong, and reflect
vs zero padding shifts the first/last few frames.

### `use_bias=True` checkpoints fail silently with `try_get` + nullptr mm_bias

`mm_bias` skips the bias add when the bias pointer is nullptr — this
makes the FastConformer block builder backwards-compatible across
parakeet variants that switch `use_bias` between True and False (v3
trained with `use_bias=False`; tdt-0.6b-ja and canary-1b both have it
True). The trap: if the loader uses `require()` for the *weight* and
nothing at all for the *bias*, the field stays nullptr regardless of
whether the tensor exists in the GGUF. A `use_bias=True` checkpoint
loads cleanly, encoder output looks plausible, the model produces
fluent-but-wrong transcripts. Specifically issue #37 / parakeet-ja:
the loader was missing 10 biases per layer × 24 layers = 240 silently
unloaded tensors. After fixing, `encoder_output cos_mean` jumped from
**0.792 → 0.996** vs the NeMo reference on
reazon_meal_11s (`crispasr-diff parakeet …`).

The bias set that needs `try_get` (for parakeet/canary FastConformer):

| Bias slot | GGUF name | Used by |
|---|---|---|
| `attn_q_b`/`k_b`/`v_b`/`out_b` | `attn.{q,k,v,out}.bias` | self-attention |
| `ff1_l1_b`/`l2_b`, `ff2_l1_b`/`l2_b` | `{ff1,ff2}.linear{1,2}.bias` | macaron FFNs |
| `conv_pw1_b`, `conv_pw2_b` | `conv.{pw1,pw2}.bias` | conv module |

Norms (`norm_*.bias`), `pos_bias_u/v`, `conv.dw.bias`, and `conv.bn.bias`
were already loaded via `require()` because they exist in every
variant. Add `try_` for the rest, never `require()`, so v3-style
checkpoints still load.

**Rule of thumb:** when adding a new model whose conditional builder
takes `(weight, bias)` pairs, audit the loader for every `weight`
that has a sibling `bias` in the GGUF. If the model's reference code
has `use_bias` as a config flag, ALL biases should load via `try_get`
— never `require()`.

### Per-layer diff is the only reliable encoder-bug detector

When an encoder produces fluent-but-wrong output and the final
`encoder_output` cos < 0.999, scattered numerical guessing wastes days.
The recipe that cracked issue #37 in <30 minutes:

**1. Capture per-layer reference activations.** In the backend's
`tools/reference_backends/<name>.py`, register PyTorch forward hooks
on every encoder submodule of interest. Use the shared helper:

```python
from . import _hooks
captured = {}
handles = _hooks.capture_modules(captured, [
    ("pre_encode_output", model.encoder.pre_encode),
    *[(f"encoder_layer_{i}", L) for i, L in enumerate(model.encoder.layers)],
])
# ... run the model ...
_hooks.drop_hooks(handles)
out.update(_hooks.finalize(captured, T_max=int(enc_len.item())))
```

Add the stage names to `DEFAULT_STAGES` so they're captured by
default. NeMo conformer modules emit `(B, T, D)` directly — `_hooks`
strips the batch dim and slices to `T_max`.

**2. Dump per-layer C++ activations.** Add a dump entry point to the
runtime (parallel to the production encoder builder) that tags each
layer's output:

```cpp
for (uint32_t il = 0; il < n_layers; il++) {
    cur = build_block(ctx0, cur, ...);
    char nm[64]; snprintf(nm, sizeof(nm), "dump_layer_%u", il);
    ggml_tensor* tag = ggml_cont(ctx0, cur);
    ggml_set_name(tag, nm); ggml_set_output(tag);
    ggml_build_forward_expand(gf, tag);
    cur = tag;  // chain — see below
}
```

Run the graph once, read each tagged tensor with
`ggml_backend_tensor_get`. See `parakeet_run_encoder_dump()` for a
worked example.

**Chain through the tag** (`cur = tag`). Without it, the production
buffer can be reused before the read-back happens — you'll see the
final layer's value flip between runs while upstream layers stay
stable. With it, every dumped tensor stays live through compute.

**3. Wire stages into `crispasr-diff`.** Call
`ref.compare("encoder_layer_K", our_buf, n_elem)` for each layer.
The first K where `cos_mean` drops below ~0.999 is where the bug
lives. Reading the runtime's loader / `build_block` code against
that layer's GGUF tensors usually finds the bug in minutes.

**4. Distinguish mel propagation from encoder-internal bugs.** Add
an `encoder_output_ref_mel` stage that feeds the REFERENCE mel into
the C++ encoder (skipping our `compute_mel`). If its cos matches the
production `encoder_output` cos, mel error isn't the issue — the
bug is encoder-internal. See `parakeet_encoder_with_ref_mel_r()` in
`crispasr_diff_main.cpp`.

For issue #37 the drop appeared at `encoder_layer_0` itself,
immediately after a bit-exact `pre_encode_output`. Reading the
loader against the GGUF tensor list found the missing biases in
<30 seconds. The same recipe applies to LLM hidden states, audio
projector outputs, and RVQ codec stages — any sequential pipeline
that can drift from a Python reference.

**Where the reusable infrastructure lives:**

| Piece | Path | Reusable? |
|---|---|---|
| Forward-hook helper | `tools/reference_backends/_hooks.py` | yes — every backend |
| Per-layer dump pattern | `parakeet_run_encoder_dump()` in `src/parakeet.cpp` | template — copy per backend |
| Reference-mel-input stage | `parakeet_encoder_with_ref_mel_r()` | template — copy per backend |
| Recipe documentation | `examples/cli/crispasr_diff.h` header comment | always-up-to-date |

### Cohere's cblas_sgemm note (kept for the historical record)

Cohere uses `cblas_sgemm` for the power→mel matmul. When we migrated
to the manual accumulator in `core_mel`, the summation order changes
slightly and one SRT timestamp shifted by 80 ms (one encoder frame).
The transcript text is bit-identical. If bit-exact BLAS output becomes
a hard requirement, a BLAS-backed matmul path can be added to
`core_mel` behind a feature flag.

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

### `CRISPASR_FFMPEG=ON` only helps bare Opus

Upstream crispasr's `examples/ffmpeg-transcode.cpp` has known bugs
on mp4-family containers: `.m4a` crashes with `munmap_chunk(): invalid
pointer` on the first audio chunk read, and `.webm` (Opus-in-WebM)
hangs indefinitely after the libavformat headers are parsed. Both use
the same `av_read_frame` + `avcodec_send_packet` loop.

Bare-codec `.opus` files work cleanly in the FFmpeg build. So the
practical advice is: enable `CRISPASR_FFMPEG=ON` only if you need
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
   identical against upstream `crispasr` is the gate.

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
9. **Chat-template natural stop ≠ tokenizer's `<eos>` — over-decoded
   to `max_new_tokens` until caught.** Found independently in
   gemma4-e2b (8.7× speedup, 67.77s → 7.75s on JFK Q4_K) and
   vibevoice-ASR (was running to max_gen=512 every call instead of
   stopping at ~50 tokens). In each case the model is trained to
   emit a chat-specific marker (`<end_of_turn>` for Gemma, `<|im_end|>`
   for Qwen2.5-based models) at the natural assistant-turn end, but
   greedy decode was configured with `cfg.eos_id = ctx->eos_id`,
   the tokenizer's `<eos>` — a different token the model never
   actually emits in conversational use. Fixes: gemma4
   `cfg.eos_id = ctx->end_of_turn_id >= 0 ? ctx->end_of_turn_id : ctx->eos_id`;
   vibevoice ASR: stop on `IM_END (151645) || EOS_TOKEN (151643)`.
   **Why:** every chat-templated LLM has TWO valid stop tokens —
   `<eos>` (= end-of-document) and `<end_of_turn>`-equivalent
   (= end-of-assistant-response). The chat-template tokeniser binds
   the model's training to the latter. Forgetting that means decode
   never naturally terminates and runs to the cap.
   **How to apply:** when porting a chat-templated LLM, immediately
   audit (a) where the runtime declares its stop set and (b) what
   the model's `tokenizer_config.json` / chat template uses as the
   assistant-turn closer. If the test transcript ever produces
   exactly `max_new_tokens` tokens — even if the visible text is
   right — the model is over-running. Bonus: across the repo, this
   pattern was specifically a bug only when chat_template_stop ≠
   tokeniser_eos. Mistral/Voxtral (`</s>`) and Granite
   (`<|end_of_text|>`) coincidentally use the same token for both,
   which masked the issue. Long-term: extend
   `core_greedy_decode::Config` to accept a list of stop tokens.

---

## Weight placement: CPU weights + GPU encoder via scheduler

### The problem: single-token decoder matmuls on GPU

For autoregressive decoders (firered-asr, kyutai-stt, any AED model),
each decode step runs ~128 small matmuls (1×1280 × 1280×1280) through
16 transformer layers. Three approaches were tried on Kaggle T4:

| Approach | Weights | Matmul method | ms/step | Why |
|---|---|---|---|---|
| `ggml_vecmat` + CUDA | GPU | Per-call ggml graph on CUDA | **2,600** | 128 CUDA context cycles × ~20ms overhead each |
| F32 dequant + `cpu_matmul_bt` | GPU→CPU | Dequant Q4_K to F32, plain C matmul | **590** | No SIMD, no OMP on Kaggle |
| **`ggml_vecmat` + CPU** | **CPU** | Per-call ggml graph on CPU | **60** | Native Q4_K SIMD (fused dequant+multiply) |

The native ggml Q4_K kernel on CPU uses AVX2/NEON SIMD for fused
dequant+dot-product, which is 9.3× faster than dequanting to F32 first
and doing plain float matmul. And on CPU there's near-zero per-graph
overhead (no CUDA launch, no D2H/H2D copies).

### The fix: load weights to CPU, let scheduler copy to GPU for encoder

```cpp
// Load to CPU — decoder uses native Q4_K SIMD directly
core_gguf::load_weights(path_model, ctx->backend_cpu, "firered_asr", wl);

// Encoder uses ggml_backend_sched which auto-copies CPU weights to GPU
// for mul_mat ops. Slightly slower than pre-loaded GPU weights but the
// decoder speedup (60ms vs 2600ms per step) dominates.
```

The `ggml_backend_sched` handles the cross-device copy transparently:
when building an encoder graph with `ggml_mul_mat(ctx, weight, input)`
where `weight` is on CPU and the scheduler assigns the op to GPU, it
inserts an automatic H2D copy. This adds ~1s total for the encoder
(16 layers of weight copies), but the decoder saves 28×2540ms = 71s.

### Applicability to other backends

This pattern applies to ANY backend with an autoregressive decoder
where single-token matmuls dominate:

- **kyutai-stt** — 24-layer Mimi decoder, same AED structure
- **omniasr-llm** — LLM decoder with per-token generation
- **voxtral / voxtral4b / qwen3 / granite** — LLM backends with
  autoregressive decode loops

For LLM backends that already use `ggml_backend_sched` for the full
decode graph (voxtral, qwen3, granite), the scheduler handles weight
placement automatically. But if any backend creates per-call mini-graphs
like `ggml_vecmat` did, switching to CPU weights + CPU backend for those
calls gives the same 40× speedup.

**Rule of thumb:** If your decode loop creates >10 tiny ggml graphs per
step on a GPU backend, you're paying more in CUDA overhead than you gain
from GPU compute. Either batch into fewer larger graphs, or use CPU
weights with native quantized SIMD kernels.

### GPU logit projection bug

When the decoder runs on CPU but the logit projection uses a GPU graph
(`project_decoder_logits`), the `ggml_backend_sched` state can become
corrupted if other operations (like decoder weight dequantization) reset
the scheduler between graph builds. Symptom: logits are wrong, EOS is
never generated, decoder runs to max_len. Fix: use CPU for the logit
projection too when decoder weights are on CPU. The Q4_K native kernel
handles the 1280×8667 projection in ~0.4ms on CPU.

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

### BPE merges break GGUF loading (skip add_token_merges)

The gguf Python library's `add_token_merges()` writes string arrays as
GGUF type 9. Our C-side `gguf_init_from_file` rejects type 9 with
`key 'tokenizer.ggml.merges' has invalid GGUF type`. This blocks
model loading entirely (not just tokenizer — the whole GGUF fails).

**Fix:** Don't store merges. The BPE decoder in `core/bpe.h` works
from the token vocab alone (greedy byte-pair matching). Merges would
add 10-30 MB to the header for no benefit in our use case.

Affected converters: gemma4-e2b, mimo-asr (any model with >100K BPE vocab).

### Metal conv_transpose_1d input range tightening (MUST RE-APPLY after every ggml bump)

Upstream ggml's Metal `kernel_conv_transpose_1d` iterates **all** input
positions (`for i in 0..IL`) and uses an `if` to filter the ones that
contribute to each output position. Mathematically only `ceil(K/s0)`
inputs ever contribute to a given output (the K-wide kernel "lands" on
at most that many input strides). For typical Snake-decoder-style
transposed convs (`K = 2*s0`) this is just 2 — but the kernel still
runs the full IL-long loop and pays the branch + memory traffic.

**Symptom:** Qwen3-TTS codec decoder block 1 (IL=320, K=10, s0=5,
output [1605, 384]) hangs the M1 GPU command buffer with
`kIOGPUCommandBufferCallbackErrorImpactingInteractivity` after ~5 sec
because each output-element thread does 320 inner-loop iterations
(of which only 2 contribute) instead of 2.

**Fix (file: `ggml/src/ggml-metal/ggml-metal.metal`, kernel
`kernel_conv_transpose_1d`):** compute the contributing range
`i ∈ [ceil((j - K + 1)/s0), floor(j/s0)] ∩ [0, IL-1]` once before the
input-channel loop and iterate only those positions. ~160× less work
on the codec block-1 shape; brings the kernel well under the macOS
GPU watchdog window. Kernel output is bit-identical to the original
formulation since only the never-contributing iterations are skipped.

After the patch, qwen3-tts codec runs end-to-end on Metal with
cos_min ≥ 0.999983 against the Python reference (8/8 stages PASS).
Marked with `// CrispASR patch` in the kernel. **MUST RE-APPLY after
every ggml bump.**

### Metal `GGML_OP_PAD` only supports right-side pads — and only matters on direct compute

ggml's Metal backend rejects `GGML_OP_PAD` whenever any of the
left-side pad params (`lp0`, `lp1`, `lp2`, `lp3` — op-params indices
0/2/4/6) is non-zero. See
`ggml/src/ggml-metal/ggml-metal-device.m:131-138`:

```c
case GGML_OP_PAD:
    if (ggml_get_op_params_i32(op, 8) != 0) return false; // circular
    return (ggml_get_op_params_i32(op, 0) == 0) && (ggml_get_op_params_i32(op, 2) == 0) &&
           (ggml_get_op_params_i32(op, 4) == 0) && (ggml_get_op_params_i32(op, 6) == 0);
```

`ggml_pad` (right-only) always satisfies this. `ggml_pad_ext` does
not — any left-pad on any dim trips the rejection.

**The catch:** when the rejection fires depends on how the graph is
run. `ggml_backend_sched_graph_compute` queries `supports_op` and
falls unsupported ops back to a CPU sub-backend transparently. So
in scheduler-based code paths a left-pad just gets routed to CPU
and inference works. **`ggml_backend_graph_compute(backend, gf)`
called directly has no fallback — Metal sees the unsupported op and
crashes with `unsupported op 'PAD'`.**

**Repro that bit us (May 2026):** wav2vec2's standalone
`ggml_grouped_conv1d_same` (`src/wav2vec2-ggml.cpp`, called for the
positional conv branch) used `ggml_pad_ext(ctx, x, pad_l, pad_r, …)`
with `pad_l = (K-1)/2` ≈ 63 inside a graph dispatched via
`ggml_backend_graph_compute(m.backend, gf)` where `m.backend` is
`ggml_backend_init_best()` — Metal on Apple Silicon. Crash:
`ggml_metal_op_encode_impl: error: unsupported op 'PAD'` on the
first inference.

**Fix:** zero-pad on the host into a wider input buffer, set the
input tensor at the pre-padded shape, and skip the in-graph PAD op
entirely. Per-call cost is one `vector<float>` zero-fill + memcpy
of `T_pad * C * 4` bytes (≈2.8 MB for wav2vec2-large) — under 1 ms
on M1, well below the noise floor on a 690 ms inference.

**Audit rule going forward:** if you call
`ggml_backend_graph_compute(backend, gf)` directly (without the
scheduler) and the graph contains a left-side `ggml_pad_ext`, you
have a Metal landmine. As of the May 2026 audit, the only live
direct-compute callers are moonshine (`src/moonshine.cpp:572,693,892`),
moonshine_streaming, orpheus_snac, whisper (`src/crispasr.cpp`),
and wav2vec2-ggml. None of them currently have left-pads except the
already-fixed wav2vec2 site. Every other left-pad in the tree
(kyutai_stt, firered_asr, qwen3_tts, marblenet_vad, gemma4_e2b,
vibevoice) runs through a scheduler and is safe by construction.

The dead `build_pos_conv_graph` in `wav2vec2-ggml.cpp:615` retains
the unfixed pattern but is gated off (`include_pos_conv=false`); if
that path is ever revived (e.g. for data2vec multi-layer pos_conv),
port the same CPU-pad fix.

### Qwen3-TTS runtime voice prompt bug: RVQ encode layout mismatch

For `qwen3-tts`, "runtime WAV clone sounds wrong, baked voice-pack
works" was not a talker or Metal problem. The decisive diff pattern was:

- `cenc_se_init` .. `cenc_ds_out` PASS
- `cenc_codes` FAIL
- `runtime_spk_emb` PASS
- `runtime_ref_codes` FAIL
- `runtime_talker_logits` FAIL as a downstream consequence

That localizes the bug to the CPU-side RVQ encode step that turns
encoder embeddings into the 16 reference codebooks. ECAPA, the SEANet
encoder, the transformer, and the CLI wrapper can all be correct while
this still fails.

**Actual problem:** `run_cenc()` handed `cenc_rvq_encode()` embeddings in
time-major row-major layout, but the helper functions still indexed them
as if they were channel-major. The bad assumption existed in all three
helpers:

- `cpu_k1_proj()`
- `rvq_nearest_neighbor()`
- `rvq_subtract()`

So nearest-neighbor search and residual subtraction were operating on
the wrong vectors even though `cenc_ds_out` already matched the Python
reference.

**Fix (file: `src/qwen3_tts.cpp`):**

1. Treat encoder RVQ inputs as `[T, C]` row-major throughout.
2. Write projected tensors in the same `[T, C]` row-major layout.
3. Run nearest-neighbor and residual subtraction against that layout.
4. Keep semantic and acoustic RVQ as independent branches from the same
   encoder embedding, matching `MimiSplitResidualVectorQuantizer.encode()`.

If `qwen3-tts` repeats the reference text or otherwise sounds wrong, the
most useful first diff is:

- `cenc_se_init` .. `cenc_ds_out`
- then `cenc_codes`
- then `runtime_ref_codes`

If the network stages pass but `cenc_codes` fails, the bug is almost
certainly in the RVQ encode layout / residual logic, not the prompt
builder.

### Qwen3-TTS quantization guidance

`qwen3-tts` quantization quality does not track audible output as closely as
many ASR models do. Current CrispASR testing found:

- `f16` talker + `f16` codec is the reference baseline
- `q8_0` talker + `f16` codec is the best tested quantized deployment
- lower-bit talker quants (`q6_k`, `q5_k`, `q4_k`) can still sound usable, but
  drift noticeably in strict tensor diffs
- quantizing the tokenizer / codec hurts earlier than talker-only quantization,
  especially around `runtime_ref_codes`

Practical rule: if you must save memory, quantize the talker first and keep the
tokenizer / codec at `f16`.

### Qwen3-TTS speed optimization tracking (2026-04-30, partial)

The qwen3-tts AR loop runs `1× talker + 15× code_predictor` per 80 ms audio
frame. Profiling showed talker compute dominates (~50-60% of frame time;
28L Qwen3 vs 5L code_predictor). The talker also rebuilds its full graph
and re-allocs the scheduler every frame because `n_past` grows by 1 — the
O15-style graph reuse trick used for code_predictor (cache one T=1 graph,
blit per-step `lm_head` weights) cannot be applied as-is because the
talker's `Lk = n_past + 1` changes on every step.

**Optional optimization paths landed in the code (default-on except where
noted), each with an env switch so a clean A/B is reproducible once a
quiet machine is available:**

1. **Quantized talker GGUFs** — converters work end-to-end:
   - `qwen3-tts-0.6b-talker-q8_0.gguf` (986 MB, 1.86× smaller than F16)
   - `qwen3-tts-0.6b-talker-q4_k.gguf` (533 MB, 3.44× smaller than F16)
   Run any of them via `-m <path>`; auto-detected. Quality guidance from
   the section above still applies (Q8_0 is the recommended default; Q4_K
   sounds usable but drifts in strict diffs).

2. **Fused Q+K+V matmul on F16/F32 talker** — same pattern as qwen3_asr.
   At load time we concatenate `attn_{q,k,v}.weight` per layer into a
   single `attn_qkv_w` on a CPU buffer (Apple Silicon's unified memory
   makes this Metal-readable without explicit copies). The shared
   helper `core_attn::kv_self_attn` already accepts `qkv_w`; we pass it
   from `build_graph_talker_kv`. Default OFF; enable with
   `QWEN3_TTS_FUSED_QKV=1` (the contended-machine bench below was
   inconclusive, so the safer default is the unchanged 3-matmul path
   until a quiet-machine A/B confirms a win). Quantized talkers
   (Q4_K/Q8_0) skip the fuse automatically — their block layout would
   need a converter-side fuse.

   **Bug found while wiring this up:** `core_attn::kv_self_attn`'s
   fused-QKV path used `ggml_view_2d` to split the `[qkv_out, T]` matmul
   output into Q/K/V views. Each view inherits `nb[1] = qkv->nb[1] =
   qkv_out * type_size`, so its rows are non-contiguous (gaps for the
   other Q/K/V). The downstream `ggml_reshape_3d` asserts
   `ggml_is_contiguous(a)` and aborts. qwen3_asr never hit this because
   its LLM only runs at T=1 (single row → trivially contiguous). qwen3-tts
   has a T=147 ICL prefill pass, which immediately exposed the bug.
   Fixed by inserting `ggml_cont` on Q/K/V when T>1; T=1 stays
   zero-copy. Both backends share the helper, so qwen3_asr is unaffected.

3. **`QWEN3_TTS_MAX_FRAMES=N` runtime cap** — short-text TTS regularly
   runs to `max_codec_steps=1500` because the talker rarely emits
   `codec_eos` for a few-word prompt; benchmarks need a hard frame cap
   or they take 4× longer than the audio they produce. The CLI doesn't
   surface `max_codec_steps`; the env var is the bench-only override.

4. **`.local/bench-qwen3/run_all.sh`** — sequential A/B/C/D harness for
   `f16_nofuse`, `f16_fused`, `q8_0`, `q4_k`. Reads JFK 24 kHz as the
   voice prompt, feeds a fixed test sentence, caps at 100 frames.
   Records per-variant `ar_loop` / `ar_breakdown` lines for comparison.

**What we did not learn yet (machine was contended by a Granite GGUF
conversion + other parallel work):** absolute or relative numbers for
fused QKV on M1 Metal. A first noisy run had `f16_nofuse` ≈ 237 ms/frame
and `f16_fused` ≈ 273 ms/frame, suggesting fusion may regress at T=1
on Metal (where 3 small matmuls can launch in parallel and the single
fused matmul has the same memory traffic but worse parallelism). This
is consistent with Apple's M1 having 8 GPU cores that benefit from
finer kernel granularity at low T. **Re-run on a quiet machine before
making the fused path the default.** Until then, keep the env switch.

**What we did learn (confirmed via byte-identical WAV output):** the
F16 fused-QKV path produces **bit-identical** output to the unfused
3-matmul path on M1 Metal at seed=42, JFK voice prompt, 100-frame
synthesis (`md5(f16_fused.wav) == md5(f16_nofuse.wav)`). This means
the fusion is mathematically correct — any future speedup observed is
pure performance, never a quality regression. (Bit-identity surprises
because reordering associative ops typically perturbs FP rounding.
On M1 Metal at this scale, the matmul kernels evidently produce the
same accumulation order regardless of whether QKV is one fused
4096-wide matmul or three independent matmuls.)

**Quantization quality A/B (2026-04-30, JFK voice prompt, fixed seed
42, 100 frames, prompt "Hello world, this is a quick speed benchmark
for the qwen three TTS pipeline"):**

| variant | Qwen3-ASR-0.6B transcript                            |
|---------|------------------------------------------------------|
| F16     | "Hello world. This is a quick three bench vlog."     |
| Q8_0    | "Hello world. This is a friend of yours."            |
| Q4_K    | *language=None — ASR could not classify as English*  |

The earlier LEARNINGS note that lower-bit talker quants "can still
sound usable" was optimistic for short utterances. Q4_K appears
unusable at this prompt length / frame budget — it may recover with
longer warmup, but for short TTS workloads Q8_0 is the floor and Q4_K
is "ship only when disk space is the binding constraint."

**Field regression on non-Metal backends (2026-04-30, fixed in 7298dd5):**
The same O15 graph-reuse change that landed for performance broke
end-to-end synthesis on backends whose `ggml_backend_alloc_buffer`
returns *uninitialised* memory (CUDA, parts of CPU). The fixed-Lk
code-pred graph reads `cp_kv_max_ctx` slots even when only `n_past+T`
are populated, masking the rest with `-inf`. If the never-written
slots happen to contain NaN-coded bytes, `Q·K → score+(-inf)=NaN`
because IEEE-754 NaN propagates through addition; softmax then
returns NaN and the whole talker output turns into noise. Metal
zero-inits buffers by default, so my Mac never saw it. Fix:
`ggml_backend_buffer_clear(cp_kv_buf, 0)` immediately after alloc in
`cp_kv_alloc`. Unconditional; one-time memset per session.

**Diff harness blind spot — closed (2026-04-30, harness in this commit):**
the prefill diff couldn't see the AR loop, so all four O15/FUSED_QKV
combos returned `cos_min=1.0` while end-to-end O15 produced 20 s of
noise. Filled the gap with `cp_step{0..14}_{input_embed,logits}`
stages: Python ref dumps each step via `forward_pre_hook` on
`code_predictor.small_to_mtp_projection` (input) + per-`lm_head[i]`
`forward_hook` (output); C ABI exposes
`qwen3_tts_run_code_pred_step(ctx, embeds, T, n_past, lm_head_idx)`
which mirrors `code_pred_generate_15`'s `skip_plan=(i>=2)` so the
harness drives the same cached-graph reuse path the AR loop does.
This pinned the broken O15 sub-feature exactly: under `QWEN3_TTS_O15=1`
on Q8_0 talker, `cp_step0` (skip_plan=false) and `cp_step1` (graph
builds + caches) both PASS at cos≥0.9999, then `cp_step2` (first
`skip_plan=true` reuse) drops to cos=0.946 and step 3+ collapses to
cos<0 — i.e. the cached-graph reuse, not always-mask / fixed-Lk /
lm_head-slot.

**Root cause and fix (2026-04-30):** `core_attn::kv_self_attn` wrote the
new K/V into the persistent cache via `ggml_cpy` against a
`ggml_view_4d` whose byte offset was `il*nb[3] + n_past*nb[1]` — a
*compile-time literal baked into the graph*. With the O15 cache reuse,
a graph built at step 1 (n_past=2) was reused at step 2 (n_past=3),
so the K/V scatter still wrote into slot 2, clobbering history.
`positions` (already a runtime input populated with `[n_past, n_past+T)`
for RoPE) carries the indices we need; switched the K/V write to
`ggml_set_rows(layer_view, K_new_perm, kv_indices)` when an opt-in
`kv_indices` arg is passed to `kv_self_attn` (default-null path
unchanged for qwen3-asr / voxtral / granite / etc.). `build_graph_code_pred_kv`
passes `positions` as `kv_indices` whenever `O15` is on. Metal kernel
`kernel_set_rows_f16_i32` handles the F32→F16 store natively.

After the fix, all 15 cp_step stages PASS at cos≥0.999924 under both
default and `QWEN3_TTS_O15=1`, and end-to-end synthesis under O15
produces the same 78-frame / 6.24 s output as the default path
(previously: 20 s noise + no EOS) for the "speed benchmark"
prompt against the JFK 24 kHz voice prompt.

**Lessons:**

1. **A literal byte offset baked into a ggml view is graph-state, not
   runtime input.** Anything that wants graph-cache reuse across
   varying `n_past` must use a runtime-indexed scatter
   (`ggml_set_rows`) instead of `ggml_cpy(view_with_offset)`.
2. **The diff harness must exercise the same code paths as the
   production AR loop.** Even prefill-equivalent stages can hide AR-loop
   bugs because the prefill builds a fresh graph; the AR loop reuses
   one. Cover both in any future per-step diff.
3. **`positions` does double duty for free.** It's already a runtime
   I32 tensor with values `[n_past, n_past+T)`; passing it as both the
   RoPE positions input AND the `set_rows` indices avoids any new
   graph-input plumbing in `run_code_pred_kv`.

**Status of the optional perf paths (post-fix):**

- `QWEN3_TTS_O15=1` — diff harness PASS (all 15 cp_step), e2e PASS.
  Now the documented perf path actually works.
- `QWEN3_TTS_FUSED_QKV=1` — diff harness PASS at prefill, byte-
  identical WAV vs default observed at seed=42 / "Hello world…"
  (likely correct at T=1 too, but not yet exercised by a per-step
  diff). Speed: contended-machine bench was inconclusive.
- Both flags together: same correctness profile as each on its own
  (no per-step diff yet that tests their interaction across the 4
  variants).

**Still on the optimization roadmap:**

- ~~**Talker Lk bucketing.**~~ Implemented and benched 2026-05-02 (see
  next section). Net loss on M1 Metal Q8_0 across three system-load
  conditions; kept gated behind `QWEN3_TTS_LK_BUCKET=1`, default OFF.
- **F16 → Q8_0 KV cache.** Halve KV bandwidth for free precision-wise
  (the helper writes via `ggml_cpy(F16 → Q8_0_view)`); needs the read
  path to `ggml_cont` Q8_0 → F16 before flash_attn (currently
  `kv_self_attn` already does a cont in `GQA_MANUAL_CONT` mode, so the
  upgrade is local).
- **Q4_K talker fused QKV.** Converter-side concat respects the Q4_K
  block boundary (rows of size `q_dim=2048`, `kv_dim=1024` divide by
  256 cleanly). Worth doing once the F16 fused-QKV win is established.

### Qwen3-TTS Lk bucketing — net loss on M1 Metal (2026-05-02)

Implemented `QWEN3_TTS_LK_BUCKET=1`: pre-built talker step graphs at
`Lk ∈ {256, 512, 1024, 2048, 4096}` in dedicated per-bucket arenas,
dispatched to the smallest fitting bucket per AR step, K/V scatter
indexed at runtime via `kv_indices=positions` (same `ggml_set_rows`
trick the cp O15 path uses), causal_mask pre-shaped at bucket Lk and
filled at the host with `-∞` past `n_past`. Cached graph plans live
in a dedicated `talker_step_sched` so prefill / embed_text / embed_audio
(which share `c->sched`) don't invalidate the cached step plans.
`kv_alloc` now zero-clears the buffer to keep the masked tail safe on
backends that don't pre-zero (cf. cp_kv fix from commit 7298dd5).

**Correctness (3 runs, very different system loads):** `md5(default.wav)
== md5(lkbucket.wav)` in every pair (`b6fb616...`). The runtime-indexed
scatter + masked-Lk path is byte-identical to the static-offset write
+ no-mask T=1 path. Useful confirmation that `set_rows`+masked-Lk on
Metal preserves softmax semantics exactly.

**Performance (Q8_0 0.6B talker, 100-frame bench, talker_kv vs
talker_bucket compute, first-window — most reproducible measure since
both runs see identical conditions within a back-to-back pair):**

| bench | default ms/call | LK_BUCKET ms/call | delta | load context |
|-------|-----------------|-------------------|-------|--------------|
| 1     | 49              | 58                | +18%  | quiet, load ~3-5 |
| 2     | 79              | 103               | +30%  | HF upload, load 8.85 |
| 3     | 80              | 125               | +56%  | mixed, load 1.98→9.58 |

LK_BUCKET regresses on M1 Metal across all three conditions. Under
sustained contention default's compute *degrades* across windows
(80→90→95→121→112→118→122 ms/call in bench 3) while bucket's stays
flatter — across-pair averages can therefore look like near-parity
(bench 2: 102 vs 101 ms/call avg), but that's default getting slower,
not bucket getting faster. First-window-only is the honest measure.

**Why bucket compute is slower than dynamic at T=1 on M1 Metal**:
1. Bucket reads `Lk_bucket` (256+) K/V positions per layer; dynamic
   reads only `n_past+1` (148→248 over a 100-frame run). 1.05–1.7×
   more F16 KV traffic depending on where in the bucket window we are.
2. Bucket mode requires a `causal_mask` add at every layer; the
   dynamic T=1 path skips the mask entirely (`(T == 1) ? nullptr : mask`).
3. `ggml_set_rows` scatter is slower than the static-offset
   `ggml_cpy(K_new_perm → view_with_baked_offset)` write on Metal —
   indirect index lookup costs more than a constant.
4. The intended win (skip per-step Metal command-buffer rebuild) is
   only ~1 ms/call on quiet M1 Metal — Metal's planner is already fast.
   Under contention alloc costs rise but compute costs rise even
   more, so the win never catches up.

**Decision:** kept gated, default OFF. The implementation is a clean
A/B platform — `QWEN3_TTS_LK_BUCKET=1` enables it for anyone
investigating future variants (e.g. removing the mask cost, or a
`set_rows` Metal kernel improvement). `talker_bucket` bench label is
distinct from `talker_kv` so on/off comparisons stay readable.

### CUDA im2col grid overflow (MUST RE-APPLY after every ggml bump)

Upstream ggml's CUDA im2col kernel uses `OW` (output width) directly as
`grid.y` in the kernel launch: `dim3 block_nums(num_blocks, OW, ...)`.
CUDA grids are limited to 65535 in y and z dimensions. For models that
process raw audio waveforms (kyutai-stt SEANet, vibevoice), the first
convolution has T_out = 176000 — far exceeding 65535.

**Symptom:** `CUDA error: invalid configuration argument` in IM2COL.

**Fix (file: `ggml/src/ggml-cuda/im2col.cu`):**
1. Add `#define MAX_GRIDDIM_Y 65535`
2. Clamp grid.y: `dim3 block_nums(num_blocks, MIN(OW, MAX_GRIDDIM_Y), ...)`
3. Loop inside kernel: `for (int64_t iow = blockIdx.y; iow < OW; iow += MAX_GRIDDIM_Y)`

This patch must be re-applied after every ggml version bump. It's marked
with a `CrispASR patch` comment in the file. First applied in commit
`1552434`, lost in the 0.9.8→0.10.0 bump, re-applied in the current fix.

### F16 mul_mat input saturation on ARM NEON CPU (MUST RE-APPLY after every ggml bump)

ggml-cpu's `ggml_compute_forward_mul_mat` for an F16 weight × F32 input
first converts src1 (F32) to F16 via the type's `from_float` =
`ggml_cpu_fp32_to_fp16` (= `__fp16 tmp = f` on ARM NEON). That cast
**saturates F32 values above 65504 to ±Inf**. The Inf then propagates
through the dot product and the next layer's RMSNorm produces NaN —
even with a corrected F32 accumulator. The actual root cause of issue
#38 is this input saturation, not the (also-buggy) F16 accumulator in
`ggml_vec_dot_f16`.

**Affects any F16 GGUF on the CPU backend** whenever an intermediate
activation exceeds 65504. Most commonly the FFN `silu(gate) * up`
element-wise product, which in a 3072-wide FFN can reach 1e5 even
when the individual mul_mat outputs are 200–400 in magnitude.

**Symptom we hit (issue #38):** `qwen3-tts` F16 talker on
`--gpu-backend cpu` produces only noise — code_predictor block 2's
`ffn_down` sees `silu(gate)*up` with `max_abs ≈ 1.4e5`, the F32→F16
quantize turns ~half the lanes into Inf, the dot product's F32 output
then has ~50% Inf elements, the next block's RMSNorm makes those NaN,
and the AR loop never samples `codec_eos` — runs to the 1500-frame
cap. Q8_0 talker is unaffected because Q8_0's mul_mat path keeps src1
as F32 in its inner loop. Apple Metal matmul kernels also operate in
F32 with the F32 src kept native — that's why the bug only shows up
on CPU and "F16 fails but Q8_0 works on the same input" (geneing's
bisection) was the deciding clue.

**Fix:** add `ggml_vec_dot_f16_f32(F16 weight × F32 input → F32 sum)`
and route the F16 type through it instead of pre-converting src1.
Three files:

1. `ggml/src/ggml-cpu/vec.h` — declare
   `void ggml_vec_dot_f16_f32(int n, float *s, size_t bs, ggml_fp16_t *x,
   size_t bx, float *y, size_t by, int nrc);`
2. `ggml/src/ggml-cpu/vec.cpp` — implement: NEON path uses
   `vcvt_f32_f16(vld1_f16(...))` to load F16 weights as F32 and
   `vfmaq_f32` for the FMA; AVX2+F16C path uses `_mm256_cvtph_ps` +
   `_mm256_fmadd_ps`; scalar fallback `GGML_CPU_FP16_TO_FP32(x[i]) * y[i]`.
3. `ggml/src/ggml-cpu/ggml-cpu.c` — change the F16 type-traits row:
   `vec_dot = ggml_vec_dot_f16_f32` and `vec_dot_type = GGML_TYPE_F32`.
   `ggml_compute_forward_mul_mat` then sees `src1->type == vec_dot_type`
   and skips the F32→F16 convert step entirely.

Defence-in-depth: also patch `ggml/src/ggml-cpu/simd-mappings.h` (both
the SVE arm at ~line 248 and the NEON arm at ~line 363) to wrap
`#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)` in `#if 0 && ...`
so `ggml_vec_dot_f16` (still called directly by `conv_transpose_1d_f16`
and a few other ops, separate from the type-traits dispatch above)
accumulates in F32 instead of `float16x8_t`. Without this defence-
in-depth, the matmul fix alone is enough for qwen3-tts but a future
F16×F16 hot path with similar magnitudes would surface the same
overflow.

All four edits marked with `// CrispASR patch (issue #38)`. **MUST
RE-APPLY after every ggml bump.** Verification: the public F16 GGUF
`cstr/qwen3-tts-0.6b-base-GGUF/qwen3-tts-12hz-0.6b-base.gguf`
(SHA256 `9b719418cf9d31fc8472a60ecdab3553280d4b15a9cbfc1363ce9352d64af6dc`)
on `--gpu-backend cpu` with `samples/jfk_24k.wav` + `Hello world.`
must hit `codec_eos` at ~30 frames (1500 frames pre-fix). The same
GGUF on Metal already worked end-to-end so it's a useful Metal-vs-CPU
regression baseline.

### ggml version bumps: conv_1d_dw shape change (0.9.8 → 0.10.0)

In ggml 0.9.8, `ggml_conv_1d_dw` returned `[OL, 1, channels, 1]` (4D).
In ggml 0.10.0, it returns `[OL, channels, 1]` (3D). Code that does
`ggml_reshape_2d(result, result->ne[0], result->ne[2])` to extract
`[OL, channels]` breaks because `ne[2]` is now `1` instead of `channels`.
Fix: use `ne[1]` instead: `ggml_reshape_2d(ht, ht->ne[0], ht->ne[1])`.

**Rule of thumb:** After any ggml bump, test all backends that use
`ggml_conv_1d_dw` or `ggml_im2col`-based ops — the output tensor
dimensions may shift.

### Qwen3-TTS cp step-0 graph cache (2026-05-02)

The O15 cp-graph cache (commit 7298dd5) handled T=1 only — the
step-0 forward pass (T=2 prefill on `lm_head[0]`) rebuilt every
frame. Added `QWEN3_TTS_CP_STEP0_CACHE=1` (gated, default OFF):
build the T=2 graph once into a per-cache arena and reserve it in a
dedicated `cp_step0_sched`, then reuse forever. Requires O15 (the
mask + `kv_indices=positions` path makes the graph n_past-invariant).

Implementation: `build_graph_code_pred_kv` gained an `arena_ctx`
parameter, mirroring the talker-bucket pattern. `cp_step0_get_or_build`
allocates the arena + graph on first use, caches in
`c->cp_step0_gf` / `c->cp_step0_ctx`. Free path skips `ggml_free` on
caller-supplied arenas (caller manages lifetime).

Validated bit-identical: md5(default WAV) == md5(O15+CP_STEP0_CACHE
WAV) on the qwen-three-bench prompt at seed=42. Bench under heavy
contention showed `cp_step0_cache (14 calls): alloc=0.00 ms` after
first activation (vs default's ~1-3 ms quiet, 5-15 ms contended).
Net win is modest (~1-3 ms/frame quiet) — the per-step compute is
unchanged, only the per-frame build+reset+alloc collapses. Kept
gated until a clean quiet-machine bench confirms the win magnitude.

### Qwen3-TTS Q8_0 KV cache — blocked on Metal cont(Q8_0)

Investigated halving talker KV bandwidth by storing K/V as Q8_0
(LEARNINGS hinted this as a "local upgrade"). Reality on Metal:

- ✓ `ggml_flash_attn_ext` natively supports Q8_0 K and V
  (`ggml-metal-device.m:1167`).
- ✓ `ggml_set_rows` supports F32 → Q8_0 dest (write path works).
- ✓ `ggml_cpy` supports F32 → Q8_0 dest (write path works).
- ✗ `ggml_cont` / `ggml_cpy` / `ggml_dup` with **Q8_0 source** is
  not supported on Metal — only F32, F16, BF16 sources are listed
  (`ggml-metal-device.m:1199-1244`). The `kv_self_attn` read path
  does `ggml_cont(ggml_view_3d(kv_k, hd, Lk, n_kv, ...))` to flatten
  the layer's strided slice from `(hd, max_ctx, n_kv, n_layers)`
  storage. Without Metal-native cont(Q8_0→*), the op falls back to
  CPU — kills the win.

Workarounds: (a) patch ggml-metal to add Q8_0-source `cpy`/`cont`
kernels (mirror llama.cpp's cache-type-k Q8_0 dequant path); persistent
maintenance cost across ggml bumps. (b) restructure KV storage to
per-layer 3D tensors so the layer-extract view becomes the whole
tensor — but the strided-Lk view (positions [0..Lk-1] from
[0..max_ctx-1]) still needs a cont. Neither option is "local" as
the LEARNINGS roadmap implied; deferred until someone wants to
shoulder the kernel work.

### Grouped conv1d via ggml: im2col shape pitfalls

Implementing grouped conv1d as G independent `ggml_im2col` + `ggml_mul_mat`
calls is correct in principle but requires careful tensor shape management:

1. **`ggml_conv_1d` requires F16 kernel** on CPU — can't use Q4_K views directly.
   Must pre-dequantize to F32 or use manual im2col+mul_mat decomposition.

2. **Q4_K tensor views are unsafe** — Q4_K has 256-byte block structure.
   `ggml_view_3d` at arbitrary offsets within a Q4_K tensor misaligns blocks.
   Always dequantize to F32 before taking group-wise views.

3. **im2col output shape**: `OL = (IL + 2*pad - K*dilation) / stride + 1`.
   With "same" padding where `pad = (K-1)/2`, and stride=1, `OL == IL`.
   The im2col tensor is `[K*IC, OL, batch]` — verify ne[0] matches
   the kernel reshape before mul_mat.

4. **Channel-first vs ggml layout**: ggml conv1d expects `[T, C, batch]`
   where ne[0]=T (time-contiguous). This is the SAME memory layout as
   channel-first `[C, T]` in C (where each channel's T values are
   contiguous). No transpose needed.

**Standalone status: DONE (4.9x speedup).** The ggml graph version works after two fixes:
(a) Use `ggml_pad_ext` for asymmetric "same" padding before im2col
(b) Transpose mul_mat output from `[ne[0]=cpg, ne[1]=T]` (time-major)
to channel-first `[cpg, T]` (channel-contiguous).
Result: pos_conv 1588ms → 324ms (4.9x faster) on wav2vec2-large.

**Integrated graph status: BLOCKED.** Folding grouped conv into the full
transformer graph (one `ggml_backend_sched_graph_compute` call) triggers
`view_src bounds` asserts during graph construction with `no_alloc=true`.
The assert fires even when using fresh per-group input tensors (not views
of model weights). Root cause: ggml's `ggml_pad_ext` output tensor +
`ggml_view_2d` combination in `no_alloc=true` contexts. The standalone
approach (separate gallocr per call) works because each call gets its own
allocation context.

### Moonshine Streaming: unit-offset LayerNorm and sliding-window attention

Moonshine Streaming (UsefulSensors, MIT) uses a non-standard LayerNorm
in the encoder where the scale parameter is `gamma + 1.0` instead of
`gamma`. The PyTorch code stores `gamma` as the parameter; applying
standard `gamma * norm(x)` produces wrong output.

**Fix:** Add +1.0 to gamma at conversion time in the GGUF converter.
Then the C++ runtime uses standard `ggml_mul(norm(x), gamma_tensor)`
and gets correct results without special-casing the forward pass.

The encoder also uses per-layer sliding-window attention with configurable
(left, right) windows — e.g., `(16, 4)` for first/last layers, `(16, 0)`
for middle layers. This implements bounded local attention via the `mask`
parameter of `ggml_flash_attn_ext`: fill a `[T, T]` F32 tensor with 0.0
(attend) or `-INFINITY` (block) based on relative position.

The audio frontend has NO mel spectrogram — it processes raw waveform
frames (80 samples = 5ms at 16kHz) with CMVN, an `asinh(exp(k)*x)`
compression (learned scalar k), a Linear+SiLU projection, and two
CausalConv1d layers with stride-2. The asinh function must be built from
ggml primitives: `log(x + sqrt(x*x + 1))`. Causal convolution is
implemented by left-padding the input with `(kernel_size - 1)` zeros
before calling `ggml_conv_1d` with `pad=0`.

### Models with different encoder/decoder hidden sizes

Moonshine Streaming small (enc=620, dec=512) and medium (enc=768, dec=640)
have mismatched hidden sizes. The cross-attention K/V projections take
encoder-dimension inputs. The model includes a learned projection
`decoder.proj.weight` [dec_hidden, enc_hidden] that maps encoder output
to decoder space. The positional embedding for cross-attention
(`decoder.pos_emb.weight`) uses **encoder** hidden size, not decoder.

**Rule of thumb:** When porting an encoder-decoder model, always check
`config.encoder_hidden_size` vs `config.hidden_size`. If they differ,
look for a projection layer in the decoder.

### Gemma4-E2B: USM Conformer + Gemma4 LLM architecture notes

Google Gemma-4-E2B (2.3B, Apache 2.0) combines a USM Conformer audio
encoder (12L, 1024d) with a Gemma4 LLM decoder (35L, 1536d). Key
implementation details that differ from other encoder-LLM backends:

**Per-Layer Embeddings (PLE):** Gemma4 adds a gated per-layer embedding
at each decoder layer. A single large embedding table `embed_tokens_per_layer`
([vocab, n_layers×256]) stores 256-dim vectors per layer per token. At each
layer: gate = sigmoid(W_gate @ ple_slice), proj = W_proj @ ple_slice,
hidden += gate * proj. The table is 262K × 8960 = ~2.3B elements — large
even at F16. Implementation: single `ggml_get_rows` + per-layer `ggml_view_2d`.

**Gemma embedding scale:** Token embeddings are multiplied by sqrt(hidden_size)
before feeding into the LLM, matching Gemma/Gemma2 convention. Missing this
produces subtly wrong outputs.

**Pre/post norms with layer_scalar:** Unlike Llama which uses only pre-norm,
Gemma4 uses both pre-norm and post-norm around attention and FFN, plus an
optional layer_scalar multiplier on the residual contribution.

**Logit softcapping:** Final logits are capped via tanh(logits/30.0) * 30.0.
This is applied after the lm_head matmul and before argmax/sampling.
`ggml_flash_attn_ext` also supports logit_softcap for the conformer encoder
attention (cap=50.0).

**Conformer self-attention:** USM uses per_dim_scale (learned per head dim,
replaces 1/sqrt(d)), chunked attention (chunk=12, left_context=13), and
relative position bias via rel_k_proj. First-pass implementation uses full
attention + per_dim_scale; chunked + rel_pos to be added via diff-testing.

**Conv2D subsampling:** Two Conv2d layers (stride 2) reduce mel [T, 128] →
[T/4, 32, 32] → flatten to [1024, T/4]. Uses ggml_conv_2d with padding.
RMSNorm is applied per-channel (transpose → norm → transpose).

**BPE without merges:** The 262K BPE vocabulary is stored in GGUF but merges
are skipped (GGUF type 9 incompatibility). core_bpe falls back to per-byte
encoding when merge_rank is empty, which works for inference since the model
generates token IDs that are decoded via the vocab lookup.

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
- FireRed decoder: weights on CPU for native Q4_K SIMD (see below)

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
crispasr VAD API stores `start`/`end` via `samples_to_cs()` (line
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

### VAD stitching matches crispasr quality

crispasr stitches VAD segments into one contiguous buffer with 0.1s
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
with "Rust cannot catch foreign exceptions" because crispasr's C++
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

### MarbleNet VAD: NeMo center=True STFT padding

NeMo's `AudioToMelSpectrogramPreprocessor` uses `torch.stft(center=True)`,
which pads the input with `n_fft/2` zeros on each side before the STFT.
Without this padding, the mel frame count is wrong (1098 vs 1101 for
jfk.wav) and all downstream conv outputs are garbage — the model produces
zero probabilities for all frames.

**Fix:** Pad input with `n_fft/2` zeros on each side:
```cpp
int pad = n_fft / 2;
std::vector<float> padded(n_samples + 2 * pad, 0);
memcpy(padded.data() + pad, pcm, n_samples * sizeof(float));
```

Also: NeMo uses `log(mel + 1e-5)` (natural log with dither), NOT
`log10` (whisper) or `ln + z-score` (NeMo conformer). Each NeMo model
has its own mel convention — always check `model_config.yaml`.

**Lesson:** When porting NeMo models, compare mel output against the
PyTorch reference FIRST. A 3-frame mel discrepancy silently corrupts
all downstream activations. The diff-testing protocol saved hours here.

### BatchNorm fusion for conv-only models

MarbleNet is pure convolutions with BatchNorm after each. BN can be
algebraically fused into the preceding conv weights at convert time:
```
W_fused = W * gamma / sqrt(var + eps)
b_fused = -mean * gamma / sqrt(var + eps) + beta
```
This eliminates all BN tensors from the GGUF (84 → 36 tensors, same
accuracy). The pattern applies to any conv+BN model (QuartzNet,
MatchboxNet, SEANet encoder in VoiceCodecs, etc.).

---

### Parakeet-JA: hardcoded mel params + BN fold with existing bias

The parakeet converter had `n_mels=128` hardcoded. The Japanese model
(`parakeet-tdt_ctc-0.6b-ja`) uses 80 mels. This caused a silent
tensor-read overflow: the runtime allocated `128 * 257 * 4 = 131 KB`
for the mel filterbank but the GGUF only stored `80 * 257 * 4 = 82 KB`.

**Fix**: Read `n_mels` (and all mel params) from `model_config.yaml`
instead of hardcoding.

Second bug: the BN fold assumed depthwise conv bias starts at zero
(`b_fused = (0 - mean) * scale + bn_bias`). The Japanese model has
non-zero `depthwise_conv.bias`. Fix: read existing bias first:
```cpp
ggml_backend_tensor_get(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
dw_b[c] = (dw_b[c] - bn_mean[c]) * s[c] + bn_b[c];
```

**TDT decoder issue** (resolved 2026-04-28): The actual cause was
**missing `xscaling`**. NeMo's `RelPositionalEncoding` multiplies the
encoder input by `sqrt(d_model) = 32` between the pre-encode and the
first conformer block when `encoder.xscaling=true`. parakeet-tdt-0.6b-v3
has `xscaling=false` (so the C++ runtime, which also didn't apply the
scale, worked by accident). parakeet-tdt_ctc-0.6b-ja has `xscaling=true`
and was producing near-random encoder activations (cos vs NeMo: 0.149)
— the joint head, fed garbage, saw blank as the argmax for almost
every frame after the first emission, hence "1 token then all blanks".

**Verified bit-exact on a JSUT-basic5000 sample**:
- NeMo:     `'水をマレーシアから買わなくてはならないのです。'`
- crispasr: `'水をマレーシアから買わなくてはならないのです。'`  (F16, identical)

The previous `predictor init matches Python to 4 decimal places`
verification was misleading — at SOS the LSTM input is the blank
token's embedding which is zero (NeMo's `padding_idx` semantics), so
only the LSTM biases were being tested. Encoder weights and the
xscale step weren't exercised by that check.

**Q4_K JA still degrades after ~8 tokens** even with the xscaling fix
— the 80-mel JA encoder is more quantization-sensitive than the v3
128-mel one, and `joint.pred.weight` / `decoder.embed.weight` fall
back to `q4_0` because their dimensions don't divide nicely for q4_k
blocks. Re-quantizing from the new F16 to Q5_K, or marking those two
tensors as F16, would fix this. F16 is bit-perfect.

**Diagnostic protocol that finally cracked it**:
1. Wrote `tools/reference_backends/parakeet.py` so `crispasr-diff
   parakeet …` could compare mel + encoder against NeMo. Initial run
   showed `cos_min=-0.71` on mel — but that was the reference
   backend returning `(n_mels, T)` while the C++ stores `(T, n_mels)`,
   layout mismatch only.
2. Fixed the layout, mel cos became `0.994` (close, residual is
   NeMo's `dither=1e-5 * randn` non-determinism).
3. Encoder cos was `0.149` — too low to explain by mel drift alone.
   Inspected `model_config.yaml`, saw `encoder.xscaling: true`, grep
   confirmed C++ never multiplied by `sqrt(d_model)`.
4. After the xscale fix encoder cos jumped to `0.812` and the JA
   transcript matched NeMo bit-exact at F16.

**Lessons**:
- Never hardcode model hyperparameters in converters. Always read
  from the model config, falling back to defaults only as a last
  resort. (Already learned for `n_mels`; re-applied here for d_model,
  ff_dim, pred_hidden, joint_hidden, AND xscaling.)
- "Predictor init matches" is not the same as "predictor matches" —
  if SOS feeds a zero embedding (padding_idx), only biases are tested.
- A change that's a no-op for one model variant (v3, xscaling=false)
  can be load-bearing for another (JA, xscaling=true). Always
  surface bool/enum architecture flags in metadata.
- `crispasr-diff <backend> <model.gguf> <ref.gguf> <audio.wav>` plus
  a reference backend that captures the same layout the C++ uses is
  the fastest way to localise an encoder discrepancy. Worth the
  ~50 LOC `tools/reference_backends/<name>.py` upfront.

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

### `ggml_backend_sched_alloc_graph` return value isn't optional on Metal

`hybrid_encoder` runs three small graphs per encoder layer (proj, A, B)
and the proj graph silently ignored the return value of
`ggml_backend_sched_alloc_graph`:

```cpp
ggml_backend_sched_reset(sctx->sched);
ggml_backend_sched_alloc_graph(sctx->sched, gf);   // ← return value dropped
ggml_backend_tensor_set(...);
ggml_backend_sched_graph_compute(sctx->sched, gf); // CRASH on Metal
```

When alloc fails, the scheduler's per-tensor backend assignments are in
an indeterminate state. `ggml_backend_sched_graph_compute` then walks
that state and dispatches the projection mul_mat to Metal with whatever
shape happens to live in `op->src[0]->ne[]` — often something that
trips `GGML_ASSERT(ne00 == ne10)` at `ggml-metal-ops.cpp:2029`. The
assert message looks like a shape mismatch in *user code*, but the real
operands are fine — the bug is upstream in the scheduler init.

CPU and CUDA tolerated this by accident. CUDA paths through ggml-cuda
re-derived shape locally; Metal's mul_mat trusts the scheduler.

**Fix:** treat alloc-graph failure as a hard error. Graphs A and B
already used the `if (!alloc) bail` pattern; bringing the proj graph in
line is a 4-line change.

**Lesson:** any time you see `GGML_ASSERT(ne00 == ne10)` fire on a
matmul with operands that *should* match, suspect scheduler-state
corruption upstream — not a real shape bug. Check that every
`alloc_graph` call paired with a `graph_compute` checks its return.

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

## wav2vec2 CNN optimization — 10.8x speedup (April 2026)

### The bottleneck

`WAV2VEC2_BENCH=1` revealed the manual C++ CNN feature extractor was
**88% of total runtime** (95s out of 108s for 11s audio). Seven Conv1d
layers on 176K samples with scalar nested loops.

### The fix (two parts)

**1. Replace CNN with ggml im2col + mul_mat** (47x CNN speedup):
- `ggml_conv_1d` hardcodes F16 im2col which caused precision loss through
  7 layers (all CTC predictions became blank). Fix: call `ggml_im2col`
  with `GGML_TYPE_F32` + `ggml_mul_mat` directly.
- Per-layer ggml graphs (not one big graph) to avoid OOM from im2col
  intermediates on 176K-sample first layer.
- Key insight: `ggml_conv_1d` output has `ne[0]=L_out` (time as fast dim),
  NOT `ne[0]=OC`. Must transpose for bias/norm/gelu, then transpose back.
- Data read from `ggml_backend_tensor_get` is in ggml layout `[C, T]`
  (channel-first), but downstream code expects `[T, C]` row-major. Must
  transpose after the CNN loop.

**2. OpenMP parallelize grouped pos_conv** (3.4x pos_conv speedup):
- `#pragma omp parallel for collapse(2)` over groups × output channels
- Required adding `OpenMP::OpenMP_CXX` to wav2vec2-ggml CMake target

### Results

| Phase | Before | After | Speedup |
|---|---|---|---|
| CNN extract | 95,193 ms | 2,423 ms | **39x** |
| Pos conv | 6,840 ms | 2,050 ms | **3.3x** |
| Encoder graph | 6,277 ms | 5,798 ms | 1.1x |
| **Total** | **108,357 ms** | **10,005 ms** | **10.8x** |

wav2vec2 is now **1.1x realtime** (faster than real-time), up from 0.1x.

### Lessons

- **Always benchmark per-phase first.** The CNN being 88% was invisible
  without per-phase timing. The "obvious" optimization target (transformer
  encoder) was only 6% of runtime.
- **ggml_conv_1d uses F16 im2col** — for models that chain 7+ conv layers,
  use F32 im2col + mul_mat directly instead.
- **ggml tensor layout matters**: `ne[0]` is the fast dimension. For conv1d
  output `ne[0]=L_out`, data is `[C, L]` channel-first when read linearly.
  All downstream code must account for this.

## FireRed decoder ggml native Q4_K — 6.3x speedup (April 2026)

### The bottleneck

`FIRERED_BENCH=1` profiling revealed the decoder was spending **41% of its
time** (23.6s out of 57.9s) on `read_f32_vec` — dequantizing all 16 layers'
Q4_K weight matrices to F32 CPU vectors before the decode loop even started.

Per-step decoding (28 steps × ~650ms) accounted for 31%. The matmuls
themselves were already OpenMP-parallelized but ran on F32 copies of the
weights, missing ggml's native Q4_K kernel optimizations.

### The fix

**`ggml_vecmat` helper**: a micro-graph per matmul call that references the
original Q4_K weight tensors directly via `ggml_mul_mat`. Each call:
1. Creates a tiny ggml context (10 tensors, ~256 KB metadata)
2. Builds a 1-op graph: `mul_mat(weight_Q4K, input_F32)`
3. Assigns weight tensor to the backend, allocates graph, computes, reads output
4. Frees the context

This sounds expensive (graph creation per call) but the Q4_K kernel is so much
faster than the F32 matmul that the overhead is negligible:

| Metric | F32 matmul + OpenMP | ggml Q4_K native | Speedup |
|---|---|---|---|
| Weight init | 23,626 ms | 441 ms | **53.6x** |
| Per-step decode | 650 ms | 70 ms | **9.3x** |
| Total | 57.9 s | 19.4 s | **3.0x** |

From the original manual C++ baseline: **123s → 19.4s = 6.3x total**.

### Why ggml Q4_K is faster than OpenMP F32

1. **No dequantization init**: Q4_K weights stay in 4-bit format. The
   `ggml_mul_mat` kernel fuses dequant + multiply in one SIMD pass.
2. **Memory bandwidth**: Q4_K is 0.56 bytes/weight vs F32's 4 bytes/weight.
   For d=1280, one weight matrix is 0.9 MB (Q4_K) vs 6.6 MB (F32).
   The Q4_K version fits in L2 cache; the F32 version doesn't.
3. **ggml's AVX2 kernels**: hand-tuned Q4_K dot product with `_mm256`
   intrinsics processes 32 weights per instruction.

### Architecture of the fix

- **Greedy path** (beam_size=1): all 8 matmuls per layer use `ggml_vecmat`
  with the original ggml weight tensors. No F32 copies at all.
- **Beam path** (beam_size>1): lazy-loads F32 weights on first beam step.
  This preserves the existing beam search which modifies KV history
  per-hypothesis. The 23.6s init only hits if beam search is actually used.
- **Cross-attn K/V precompute**: uses ggml batch matmul graph (one graph
  for both K and V projections per layer).
- **Norm/bias tensors**: still read to F32 since they're small (~d floats
  each) and needed for CPU LayerNorm.

### Key lesson

**Don't dequantize quantized weights to F32 for CPU matmul — use ggml's
native quantized kernels instead.** Even with the overhead of creating a
tiny ggml graph per matmul call, the native Q4_K path is 9.3x faster
than OpenMP-parallelized F32 matmul. The memory bandwidth savings alone
(7x less data) more than compensate for the graph creation overhead.

## VibeVoice-ASR prompt template verification (April 2026)

**Verified against HF transformers + microsoft/VibeVoice GitHub repo:**

1. **Special tokens**: `<|object_ref_start|>` (151646), `<|box_start|>` (151648),
   `<|object_ref_end|>` (151647). The HF processor defines these as `audio_bos_token`,
   `audio_token`, `audio_eos_token`.

2. **Assistant header required**: The HF processor calls `apply_chat_template` with
   `add_generation_prompt=True`. Qwen2.5's chat template appends `<|im_start|>assistant\n`
   when this flag is set. Without it, the model doesn't know the generation boundary.
   A prior commit incorrectly removed it; restored in 665b5d6.

3. **Qwen2.5-7B attention biases**: `q_proj.bias=True, k_proj.bias=True, v_proj.bias=True,
   o_proj.bias=False`. Our GGUF has all 3 biases stored. The V bias was missing in the
   C++ runtime — fixed in fd4862b.

4. **Tokenizer not in GGUF**: The pre-existing Q4_K on HF (`cstr/vibevoice-asr-GGUF`) has
   `has_tokenizer=0` because the converter couldn't find tokenizer files in the VibeVoice
   snapshot. The converter now falls back to `Qwen/Qwen2.5-7B` for the tokenizer. The GGUF
   needs re-conversion on a machine with ≥16 GB RAM to embed the 152K-token vocab.

5. **Duration format**: The processor inserts audio duration as plain text (e.g., "11.00")
   before tokenization. Qwen2.5 maps digits '0'-'9' to token IDs 15-24, '.' to 13.

---

## VibeVoice-Realtime-0.5B TTS (April 2026)

Full text-to-speech pipeline implemented and verified via ASR round-trip.
17 bugs found and fixed through systematic stage-by-stage diff methodology.

### Final Results

| Input | Parakeet ASR |
|-------|-------------|
| "Hello world" | "Hello world." |
| "Hello, how are you today?" | "Hello, how are you today?" |
| "The quick brown fox jumps over the lazy dog" | "The quick brown fox jumps over the lazy dog." |
| "Good morning everyone" | "Good morning everyone." |

### Architecture

VibeVoice-Realtime-0.5B has a two-LM architecture:

```
Text → Base LM (4L Qwen2, no final norm) → hidden states
  ↓ splice into TTS LM input + type_emb[text=1]
TTS LM (20L Qwen2, with norm) → per-frame condition
  ↓ + DPM-Solver++ (20 steps, cosine schedule, v-prediction)
Prediction head (4× AdaLN+SwiGLU layers) → acoustic latent [64]
  ↓ acoustic connector → TTS LM feedback + type_emb[speech=0]
σ-VAE decoder (7-stage transposed ConvNeXt, 3200x) → 24kHz mono PCM
```

Key components:
- **Voice prompts**: Pre-computed KV caches (`.pt` → `.gguf`, ~2.7 MB) establish
  speaker identity. Required for correct output. Contains `lm` (74 tokens),
  `tts_lm` (251 tokens), `neg_lm` (1 token), `neg_tts_lm` (1 token).
- **CFG (Classifier-Free Guidance)**: Dual KV cache — positive path uses text
  conditioning, negative path uses `<|image_pad|>` tokens through neg base LM.
  Both updated per-frame. `cfg_scale=3.0`.
- **Text/speech interleaving**: TTS_TEXT_WINDOW_SIZE=5 tokens, then
  TTS_SPEECH_WINDOW_SIZE=6 frames, alternating. Matches official pipeline.
- **EOS classifier**: `FC1(896→896) → SiLU → FC2(896→1) → sigmoid > 0.5`.
  Triggers at the correct frame for each text length.
- **Tokenization**: Greedy longest-match on Qwen2 vocab. Text must end with `\n`
  (the official VibeVoiceTextTokenizerFast appends newline).

### Prediction Head Detail

```
AdaLN modulation = Sequential(SiLU, Linear)  ← NOT just Linear!
Input: noisy_latent[64]
  → noisy_proj(64→d_lm)
Condition: c = cond_proj(condition) + t_embedder(timestep)
  where t_embedder = sinusoidal[256] → Linear → SiLU → Linear → [d_lm]
4× HeadLayer:
  adaln_params = linear(silu(c)) → [3*d_lm] → split to shift, scale, gate
  h = rms_norm(x, weight, eps=1e-5) * (1 + scale) + shift
  h = SwiGLU_FFN(h)
  x = x + gate * h
FinalLayer:
  adaln_params = linear(silu(c)) → [2*d_lm] → shift, scale
  h = rms_norm(x, eps=1e-5)  (NO affine weight!)
  h = h * (1 + scale) + shift
  output = linear(h) → [vae_dim=64]
```

### DPM-Solver++ Implementation

Cosine beta schedule with clipping:
```
beta_t = min(1 - cos²((t+1)/T+s)/(1+s) * π/2) / cos²((t/T+s)/(1+s) * π/2), 0.999)
alpha_cumprod = cumulative_product(1 - beta)
```
The `max_beta=0.999` clipping is critical — without it, `α[999]=1e-20` instead of `2.4e-9`.

First-order (steps 0 and 19):
```
x_target = (σ_target/σ_source) * x - α_target * (exp(-h) - 1) * x0_pred
```

Second-order midpoint (steps 1-18):
```
h = λ_target - λ_current,  h_0 = λ_current - λ_previous
r = h_0 / h
D0 = x0_current,  D1 = (1/r) * (x0_current - x0_previous)
x_target = (σ_target/σ_current) * x - α_target*(exp(-h)-1)*D0 - 0.5*α_target*(exp(-h)-1)*D1
```

Where `λ = log(α/σ)`, `α = sqrt(α_cumprod)`, `σ = sqrt(1-α_cumprod)`.

### ggml_conv_transpose_1d

```
Kernel: ne[0]=K, ne[1]=C_out, ne[2]=C_in
Input:  ne[0]=T, ne[1]=C_in
Output: ne[0]=T_out, ne[1]=C_out (4D: [T_out, C_out, 1, 1])
Causal trim: remove K-stride samples from end → T_out = T_in × stride
```

### Voice KV Cache Pre-fill

Voice GGUF tensors store per-layer K/V as `[hd, seq_len, n_kv_heads]` in F16.
Runtime KV cache is `[hd, max_ctx, n_kv_heads, n_layers]`.
Copy per-head with stride matching: each head's `hd × seq_len` slice goes into
the cache at `layer_off + head × (hd × max_ctx × sizeof_f16)`.

### 17 Bugs Found (stage-by-stage diff)

| # | Bug | Impact | How found |
|---|-----|--------|-----------|
| 1 | Cosine schedule (was linear) | Wrong noise levels | Config comparison |
| 2 | v-prediction (was epsilon) | Wrong denoising direction | Config: `prediction_type` |
| 3 | Condition routing: `x += cond` → `c = cond + t_emb` for AdaLN | Wrong conditioning | Reference code reading |
| 4 | Byte-level tokenizer → greedy longest-match | Token mismatch (11 vs 2) | Token ID comparison |
| 5 | Type embed ordering: embed→splice→type → embed→type→splice | cos 0.62→0.9999 | Forward pass tracing |
| 6 | Base LM scope: full prompt → text tokens only | Wrong context | Input dimension comparison |
| 7 | Realtime model: 4 base LM layers (config says 24) | Missing layers crash | Tensor name scan |
| 8 | Dual KV cache for CFG | Non-speech → speech | Reference code reading |
| 9 | Voice KV pre-fill: per-head stride copy | Garbage KV data | KV data dump comparison |
| 10 | Text windowing: 5+6 interleaving | cos 0.988→0.9999 | Official `generate()` tracing |
| 11 | Base LM voice KV cache | Wrong base hidden | KV cache comparison |
| 12 | Type embed in voice mode: re-add after splice | cos 0.27→0.988 | Forward pass tracing |
| 13 | Neg base LM forward: pad → base LM → TTS LM | cos -0.01→0.9999 | Condition comparison |
| 14 | DPM r-ratio sign: `h_0 = λ_cur - λ_prev` | 2nd-order correction wrong | Formula comparison |
| 15 | Beta clipping: max 0.999 | α[999] off by 10¹¹ | Schedule value comparison |
| **16** | **AdaLN SiLU: `linear(silu(c))` not `linear(c)`** | **Solver diverges** | **Internal hook comparison** |
| **17** | **Text newline: append `\n`** | **Word doubling** | **Token ID comparison** |

### Debugging Methodology

Same protocol as wav2vec2, FireRedPunc, and Kyutai STT debugging:

1. **Build Python reference** from model weights (not HF transformers — manual forward)
2. **Dump intermediates** at every stage boundary (token IDs, hidden states, latents)
3. **Compare stage-by-stage** with C++ output using cosine similarity
4. **Find first divergence** — the bug is always at the first cos < 0.999 stage
5. **Read the reference source** (`inspect.getsource`) to find the exact difference
6. **Hook internal modules** (`register_forward_hook`) to capture per-layer values
7. **Verify fix** by re-running comparison after each fix

Key lesson: when each intermediate matches (cos=0.9999) but output is wrong,
the bug is in a later stage or in accumulated precision loss. Keep adding
comparison points until the exact divergent operation is isolated.

For the AdaLN SiLU bug (#16): even with cos=1.0 on test vectors, the DPM solver
diverged in practice. The fix was found by comparing prediction head MAGNITUDE
(rms 0.65 vs 0.26), not direction (cos was misleading). Then hooking the official
model's `adaLN_modulation` module revealed the Sequential(SiLU, Linear) structure
that our manual reimplementation had missed.

### VibeVoice-1.5B Base Model TTS (April 2026)

The 1.5B/7B base models use a fundamentally different TTS architecture
than the Realtime-0.5B streaming model.

**Key discovery: VibeVoice reuses Qwen2 vision tokens for speech:**
```
speech_start_id    = <|vision_start|> = 151652
speech_end_id      = <|vision_end|>   = 151653
speech_diffusion_id = <|vision_pad|>  = 151654
```
NOT `<|object_ref_start|>` (151646) / `<|box_start|>` (151648) as initially assumed.
This mapping is defined in `modular_vibevoice_text_tokenizer.py`.

**Single-LM architecture** (no TTS LM):
```
Prompt: system_prompt + voice_reference + text_input + " Speech output:\n" + <speech_start>
LM generates autoregressively:
  → <speech_diffusion> tokens → each triggers DPM-Solver++ → acoustic latent
  → latent fed back via acoustic connector → next LM step
  → <speech_end> → stop
```

**Voice cloning** uses the acoustic + semantic encoders already in the GGUF:
1. Load reference WAV (24kHz mono)
2. `vibevoice_encode_speech()` → combined features [T_frames, d_lm]
3. Insert between `<speech_start>` and `<speech_end>` in voice section of prompt
4. LM learns speaker identity from these embeddings

**Prompt template** (from `VibeVoiceProcessor`):
```
" Transform the text provided by various speakers into speech output,
  utilizing the distinct voice of each respective speaker.\n"
" Voice input:\n" <speech_start> [voice_embeddings] <speech_end>
" Text input:\n Speaker 1: {text}\n"
" Speech output:\n" <speech_start>
```

**CFG**: cfg_scale=1.5 for base model (lower than Realtime's 3.0 since no
proper negative conditioning path — using zero vector as negative).

**ASR round-trip**: "Hello, how are you today?" → exact match (F16, Q8_0, Q4_K).

| Model | Size (Q4_K) | ASR Result |
|-------|------------|-----------|
| Realtime-0.5B | 607 MB | Perfect (with voice .gguf preset) |
| 1.5B | 1.6 GB | Perfect (with reference WAV) |

### VibeVoice TTS Performance Optimization (April 2026)

**Phase 1: Cached prediction head graph — 25% speedup**

The prediction head (4 AdaLN + SwiGLU layers) is called 280+ times per
TTS synthesis (20 DPM steps × 14 frames). Previously rebuilt the ggml
graph from scratch each time. Now built once with a separate `ggml_context`
and reused via `ggml_backend_sched_reset()`.

| | Before | After | Speedup |
|--|--------|-------|---------|
| Realtime-0.5B Q4_K, "Hello, how are you today?" | 11.85s | 8.82s | **25.6%** |

**Why other graphs can't be cached:**

- **LM decode graph**: KV cache view offsets (`ggml_view_4d` offset parameter)
  change with `n_past` at each step. ggml encodes offsets at graph build time —
  no way to update without rebuilding the graph. Only 14 calls (low impact).
- **Connector graph**: Cacheable but only 14 calls of a 5-op graph (~0.07s total).
- **VAE decoder**: Already built once per synthesis.

**Remaining bottleneck**: 20-layer TTS LM forward pass (~150ms × 14 frames = 2.1s).
Would need fused QKV, persistent KV view indexing, or GPU offload for further gains.

---

## HuggingFace downloads need `hf_xet` since the xet migration (April 2026)

HuggingFace has migrated all model storage to its **xet CAS backend**
(`cas-bridge.xethub.hf.co`). Every `huggingface.co/<repo>/resolve/main/<file>`
URL now 302-redirects to a signed S3 URL on `cas-bridge.xethub.hf.co`.
This affects every newly-uploaded weight and most existing repos.

**The default Python HTTP path doesn't cope.** `huggingface_hub`'s
urllib3-based client times out with `Read timed out` on every shard,
and `huggingface_hub`'s auto-resume just hits the same wall in a loop.
Direct `curl -C - --retry 10` also fails — it gets 70-80% of each shard
then exits with code 18 (partial file) when the xet edge resets the
TLS connection. curl's default `--retry` does not fire on exit 18, and
even with `--retry-all-errors` the recovery loop is fragile.

**`pip install hf_xet` fixes it for both the Python lib and the CLI.**
`hf_xet` is an *optional* PyPI package that ships a Rust-native xet
client. When installed, both the `huggingface_hub` Python library and
the `hf` CLI silently delegate to it — same xet endpoints, dramatically
more robust client. Without it, both tools fall back to the urllib3
path and quietly time out.

**No HF mirror escapes xet.** `hf-mirror.com`, `?download=true`,
ModelScope (404 for VibeVoice repos), and every community re-upload
of the same weights all 302 to `cas-bridge.xethub.hf.co`. xet IS the
storage layer — there is no LFS-direct path on HF anymore.

**The fix workflow:**

```bash
pip install hf_xet              # installs the Rust client
hf download org/Model-7B \
    --local-dir /path/to/dest   # now goes through hf_xet
```

`hf` CLI auto-detects `hf_xet` at runtime; no env var or flag needed.
For the converter scripts that call `snapshot_download(...)` directly,
the same auto-detection kicks in once `hf_xet` is in the env.

**Cost of getting this wrong:** ~30 minutes of failed retries +
partial-file curl loops + searching for non-xet mirrors that don't
exist. Single one-liner away from working downloads.

---

## ggml conv1d: channels-first ops eliminate transpose overhead (April 2026)

### The problem

ggml's `ggml_conv_1d` expects input in `[T, C_in]` layout (time-major),
but audio pipelines naturally work in `[C, T]` layout (channels-first,
matching PyTorch convention). Every conv call in the codebase required:

1. `ggml_cont(ggml_transpose(x))` — copy+transpose [C,T] → [T,C]
2. `ggml_conv_1d(kernel, x_transposed)` — the actual convolution
3. `ggml_cont(ggml_transpose(result))` — copy+transpose back to [C,T]

Each `ggml_cont` on a transposed tensor is a full memcpy. For the
VibeVoice σ-VAE decoder (26 ConvNeXt blocks, 6 upsample stages), this
meant ~224 extra transpose/cont ops on tensors up to 51200 elements.
The depthwise conv path (`ggml_conv_1d_dw`) additionally used F16
im2col, causing precision loss through 26 blocks (cosine sim 0.7-0.8
at output vs 1.0 in F32).

### The solution: `ggml_conv_1d_cf` + `ggml_conv_1d_dw_cf`

New `GGML_OP_CONV_1D_CF` op added to ggml with:
- `ggml_conv_1d_cf(kernel, data, stride, pad, dilation)` — regular conv
  with data `[C_in, T]` (channels-first), output `[C_out, T_out]`
- `ggml_conv_1d_dw_cf(kernel, data, stride, pad, dilation)` — depthwise
  variant, kernel `[K, 1, C]`, same I/O layout

Implementation: direct F32 loop (no im2col), multi-threaded over output
channels. Handles F16/BF16 kernel weights via on-the-fly dequant with
per-channel kernel pre-load (K is typically 7 — fits in a stack buffer).

### Results on VibeVoice TTS VAE decoder

| Metric | Before | After | Change |
|---|---|---|---|
| Op count | 700 | 476 | -32% |
| VAE compute | 5875ms | 4172ms | **-29%** |
| Total TTS | 10.7s | 9.1s | -15% |
| Realtime factor | 0.39x | 0.45x | +15% |

Combined with `--tts-steps 10` (DPM-Solver++ can use fewer steps at
identical quality): 7.3s total, 0.56x realtime — **32% faster** than
baseline.

### Where to use conv1d_cf

Use `ggml_conv_1d_cf` / `ggml_conv_1d_dw_cf` whenever:
- Data is channels-first `[C, T]` (most audio pipelines)
- Kernel is F16/F32 (not block-quantized — K is usually too small)
- You want F32 precision (no im2col F16 accumulation)

Don't use when:
- Data is naturally time-major `[T, C]` (whisper mel, some NeMo paths)
- Kernel is block-quantized (Q4_K/Q8_0) — need dequant support
- GPU acceleration is needed (CPU-only kernel for now)

### Backends that benefit from migration

| Backend | Conv pattern | Expected gain |
|---|---|---|
| wav2vec2-ggml.cpp | 7-layer CNN stem, each with transpose | **High** |
| firered_asr.cpp | 1 depthwise conv with 2 transposes | Medium |
| kyutai_stt.cpp | 1 conv | Low |
| ecapa_lid.cpp | 1 conv | Low |
| vibevoice.cpp | Done (encoder + decoder) | **Done** |

---

## Apple's GPU interactivity watchdog kills heavy compute on Metal (April 2026)

**Symptom:** any sustained Metal compute (TTS synthesis with σ-VAE
decoder, VibeVoice 7B base LM forward at frame ~10) aborts with:

```
error: Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)
ggml_metal_synchronize: error: command buffer 1 failed with status 5
ggml_metal_graph_compute: backend is in error state from a previous command buffer failure - recreate the backend to recover
```

This is **not a ggml bug**. macOS's GPU scheduler kills processes whose
command buffers hold the GPU long enough to starve the windowserver
of frames (~5 sec budget). The threshold is hard-coded in the kernel
extension; you can't disable it from userspace. NVIDIA Windows TDR is
the analogue (default 2 sec, but tunable via registry or compute mode).

**Things that don't fix it:**
- `ggml_backend_synchronize` between graph computes — implicit
  `ggml_backend_tensor_get` already syncs; the watchdog fires
  *during* a single submit, not between them.
- Bumping `ggml_metal_set_n_cb` from 1 to 4 — splits the graph across
  more command buffers, but the heavy ops remain in single buffers.
  Helps a little for medium graphs, doesn't save you on big ones.
- Reducing the cache (graph reuse) — tested by reverting to pre-cache
  vibevoice; same watchdog. Caching is orthogonal.
- Removing custom ggml ops (e.g. `conv_1d_cf`) — tested; same watchdog.

**What actually fixes it:** route the offending graph (or its hot
sub-graph) to the CPU backend. Per-tensor backend hints work great:

```cpp
for (int i = 0; i < ggml_graph_n_nodes(dec_gf); i++) {
    ggml_backend_sched_set_tensor_backend(
        ctx->sched, ggml_graph_node(dec_gf, i), ctx->backend_cpu);
}
ggml_backend_sched_alloc_graph(ctx->sched, dec_gf);
```

The scheduler handles weight copies from Metal-resident buffers
automatically. Encoders, LM, diffusion all stay on Metal — only the
problematic graph runs CPU.

For VibeVoice TTS this means routing only the σ-VAE decoder to CPU.
Cost on M1: ~10–15% of TTS wall time; alternative is the entire
process aborting. End-to-end TTS time still beats real-time
(0.57× RT for Realtime-0.5B).

**Don't preemptively force-CPU on every GPU backend.** The watchdog
is a desktop-display-attached-GPU thing. Linux servers without X
have no watchdog at all; datacenter NVIDIA GPUs have no TDR. Make
it conditional + env-overridable:

```cpp
const char* env = getenv("VIBEVOICE_VAE_BACKEND");  // cpu|gpu|auto
bool force_cpu = env && !strcmp(env, "cpu") ? true
               : env && !strcmp(env, "gpu") ? false
               : backend_is_metal(ctx->backend);   // auto: Metal → CPU
```

---

## ggml-metal scheduler doesn't preserve view-tensor backend mapping across sched_reset (April 2026)

**Symptom:** `GGML_ASSERT(src_backend_id != -1)` in
`ggml_backend_sched_split_graph` on Metal, on the *second* call to
`ggml_backend_sched_alloc_graph` after `ggml_backend_sched_reset`, when
the graph contains views of intermediate tensors (e.g.
`ggml_view_2d(ctx, adaln_out, ...)` over the output of a prior matmul).

**Pattern that triggers it:** caching a graph (saving the
`ggml_cgraph*` + `ggml_context*`) for reuse across multiple
`sched_reset()` cycles. Specifically: when the cached graph lives in
its **own dedicated meta_buf** separate from `ctx->compute_meta` that
other graph builders use.

VibeVoice's `get_pred_head_graph` did this for ~25% TTS speedup. Asserts
on Metal at the second invocation. CPU/CUDA tolerate it.

**Fix that works:** build the graph into the **shared `compute_meta`
buffer** (the same one every other builder uses) on Metal, dedicated
buffer only on CPU/CUDA. Other graph builders that survive sched_reset
on Metal (`run_dec`, `run_connector_stage`, `build_decoder_graph`) all
use compute_meta — that's the existence proof that this pattern is
Metal-safe. The CPU/CUDA dedicated-buffer + cache path keeps the
speedup where it works.

```cpp
// On Metal: build fresh into shared compute_meta (same as every other
// builder that survives sched_reset). On CPU/CUDA: dedicated buffer
// keeps the cached graph reusable across diffusion sub-steps.
std::vector<uint8_t>* meta = backend_is_metal(ctx->backend)
                                 ? &ctx->compute_meta
                                 : &ctx->pred_graph_meta;
```

**Backend name detection:** Metal devices register as `"MTL0"`, `"MTL1"`,
etc — **not** `"Metal"`. Match the `MTL` prefix:

```cpp
static bool backend_is_metal(ggml_backend_t b) {
    if (!b) return false;
    const char* name = ggml_backend_name(b);
    return name && std::strncmp(name, "MTL", 3) == 0;
}
```

CPU is `"CPU"`, CUDA is `"CUDA0"`, Vulkan is `"Vulkan0"` — all
unambiguous.

---

## gguf-py only ships Python implementations for non-K quants (April 2026)

The `gguf` PyPI package's `quants.quantize(arr, qtype)` works for:
- F32, F16
- Q4_0, Q4_1
- Q5_0, Q5_1
- Q8_0

Everything else — **Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, IQ4_NL, IQ4_XS, IQ2_*,
IQ3_S, MXFP4, BF16** — has Python class skeletons but `quantize_blocks`
raises `NotImplementedError`. The actual implementations live in
llama.cpp's C code; gguf-py defers to them at runtime via a separate
binding that isn't shipped on PyPI.

**Implication for streaming converters:** if you want a memory-safe
single-pass converter that quantizes per-tensor (avoiding the F16
intermediate that needs the whole model in RAM), you can only target
the implemented quants. Q4_0 is the closest substitute for Q4_K (same
4.5 bits/weight, only the per-block scale granularity differs); for
ASR/TTS it's perceptually fine.

For real K-quants from Python you need to either (a) ship the F16
intermediate and run llama.cpp's `quantize` binary on it (the canonical
path; needs a 32 GB host for 7B+), or (b) write a C extension binding
to ggml's `ggml_quantize_chunk` (~50 LOC + build setup).

---

## GGUFWriter.add_tensor_info conditional shape conversion gotcha (April 2026)

`gguf.GGUFWriter.add_tensor_info(name, tensor_shape, tensor_dtype,
tensor_nbytes, raw_dtype=None)` applies
`quant_shape_from_byte_shape(tensor_shape, raw_dtype)` to the passed
`tensor_shape` — but **only when `tensor_dtype == np.uint8`** (i.e.
quantized).

For F16 / F32 (where `tensor_dtype` is `np.float16` / `np.float32`),
the shape is stored as-is.

**Implication:** when streaming-writing with `add_tensor_info` directly
(not via the high-level `add_tensor` which infers the right shape from
the numpy array), you must pass:

| Target type | What to pass as `tensor_shape` |
|---|---|
| Q4_0/Q4_1/Q5_*/Q8_0 (uint8 packed) | byte_shape from `quants.quant_shape_to_byte_shape()` |
| F16 / F32 | logical shape directly |

Got this wrong once → produced a GGUF with all F16 dimensions doubled,
which loaded fine but had wrong tensor offsets ("expected 2.18 GB got
1.09 GB on tensor 1"). The C++ loader's `gguf_init_from_file_ptr`
catches it and refuses to open the file.

---

## MSVC `_USE_MATH_DEFINES` must come BEFORE the very first include (April 2026)

POSIX `M_PI` etc are extensions — MSVC's `<math.h>` only exposes them
when `_USE_MATH_DEFINES` is defined **before the first time `<math.h>`
is transitively included anywhere in the translation unit**. Once any
header (a project `.h`, a ggml header, anything that pulls `<math.h>`
or `<cmath>` even indirectly) commits the macro state, defining the
flag later is a no-op.

Wrong:
```cpp
#include "vibevoice.h"     // transitively pulls <math.h>
#include "ggml.h"
#include <cmath>           // already too late
#define _USE_MATH_DEFINES  // ineffective
```

Right:
```cpp
#define _USE_MATH_DEFINES  // FIRST line of code in the file
#include "vibevoice.h"
#include "ggml.h"
#include <cmath>
```

Linux/macOS have `M_PI` in their libc implementations of `<math.h>`
unconditionally, so the bug is invisible there — Windows CI is the only
place it surfaces. We had two patches in a row that put the define in
"the right-looking spot" but not actually first; the second patch was a
clean no-op until I moved the define above all `#include` lines.

---

## clang-format-18 patch versions disagree (April 2026)

CI pins `clang-format-18` (Ubuntu apt → 18.1.3). Homebrew's default
`clang-format` keg is on the latest LLVM (v22.x), and `pip install
'clang-format'` defaults to 18.1.8. Even within the v18 series, 18.1.3
and 18.1.8 disagree on subtle formatting (extra braces, for-loop init
spacing, long-line breaks at slightly different columns).

**Pin the local toolchain to match CI exactly:**

```bash
pip install 'clang-format==18.1.3'    # match Ubuntu's apt build
```

Then put `~/Library/Python/3.11/bin/` (or your platform's pip user-bin)
ahead of `/opt/homebrew/bin` on `PATH`, or alias `clang-format` to that
pinned binary. Without this, every `clang-format -i` you run locally
will silently introduce drift that the lint job rejects.

Documented in README's "Adding a new backend" section so future
contributors don't trip on the same v22-vs-v18 mismatch we hit several
times in one session.

---

## Per-platform fast iterative builds: Ninja + ccache + libomp (April 2026)

Local build was Unix Makefiles + no ccache + tests=ON. Cold rebuild
was ~90 sec, single-file edit ~30 sec.

After `scripts/dev-setup.sh` (auto-detects platform):

| Platform | Installs | Cold build | Touch one file |
|---|---|---|---|
| macOS | brew: `libomp ninja ccache cmake` | 39s → **26s** with libomp | <0.5s |
| Linux | apt: `+ mold libomp-dev ninja ccache` | similar gains + mold linker | <0.5s |
| Windows | choco: `ninja ccache` (manual) | varies | varies |

Three changes carry the speedup:
1. **Ninja** (vs Unix Makefiles): better dependency tracking, lower
   per-target overhead. Single biggest factor for incremental builds.
2. **ccache**: caches compiler output across rebuilds. Massive win on
   touch-one-file cycles. ggml's CMakeLists auto-detects via
   `find_program(GGML_CCACHE_FOUND ccache)`.
3. **libomp**: not a build speedup but a runtime gain (Apple clang
   needs explicit `-DOpenMP_ROOT=/opt/homebrew/opt/libomp`); enables
   multi-threaded matmul on CPU. Worth the install slot anyway.

`mold` linker (Linux only — macOS port doesn't ship a Mach-O linker)
saves additional seconds on the linking step, which dominates
incremental rebuilds when ccache eliminates the compile cost.

Per-platform-aware scripts at `scripts/dev-{setup,build}.{sh,ps1}` —
detect via `uname -s` (Darwin/Linux/MINGW*/MSYS*/CYGWIN*) and apply
only the flags that platform's toolchain accepts. None of these flags
go into CMakeLists.txt itself, so CI/Docker/release builds are
unaffected.


## Per-layer architecture variations are easy to miss (April 2026)

Two cases bit us this session:

### Architectural booleans flipping per model variant

`encoder.xscaling` is `false` for `parakeet-tdt-0.6b-v3` and `true`
for `parakeet-tdt_ctc-0.6b-ja`. The C++ runtime that worked for v3
silently produced near-random encoder activations on the JA model
because we never multiplied by `sqrt(d_model)`. Lesson: **whenever a
config flag is a bool, surface it to GGUF metadata** so old models
that lack the key default to the *correct* behaviour for that family
and the runtime can branch on it. Don't rely on "every model in this
family does X" — sister models flip flags.

### Per-LAYER hparam variations

`google/gemma-4-E2B-it` alternates `sliding_attention` /
`full_attention` layers (5 of 35 are full). The two layer types use
DIFFERENT `head_dim`: 256 for sliding, 512 for full
(`global_head_dim`). A single-`head_dim` graph trips
`ggml_can_mul_mat` at the first full layer. Same model also has
`num_kv_shared_layers=20` — and the direction matters: it's the
**LAST** N layers that reuse K/V from earlier same-`layer_type`
layers, not the first N (`first_kv_shared_layer_idx = num_layers
- N`, see `transformers/models/gemma4/modeling_gemma4.py:1148`).
Plus `use_double_wide_mlp=true`, which means the SAME MLP runs with
2× `intermediate_size` ON the kv-shared layers — not two separate
MLP halves (a misread that wasted us a session of converter work).

Lesson: **runtime hparams aren't always per-model; some are
per-layer.** When a `layer_types` array, `num_kv_shared_layers`-style
counter, or a `global_*` paired with a regular `*` shows up in the
config, plan a per-layer mask in the runtime, not a single value.
The converter should persist the mask (we now do for Gemma4 via
`gemma4e2b.llm.layer_full_mask`) so the runtime can branch without
re-parsing the YAML.

**Sub-lesson: read the actual `forward()` end-to-end before assuming
"this flag is/isn't load-bearing."** Twice with Gemma4 we caught
ourselves pattern-matching ("kv-share = first N layers", "MLP =
two halves") off names + config fields and were both times wrong.
The HF source is the ground truth.

## QAT clipping scalars are NOT just training-time noise (April 2026)

Gemma4's audio tower uses `Gemma4ClippableLinear` everywhere
(q/k/v/o projections, both feed-forwards, lconv1d.linear_start/end).
At init the four buffers `input_min/max, output_min/max` are
`±inf`; QAT trains them down to finite values like `±5..±40`. AND
THE FORWARD HOOK ACTUALLY USES THEM:

```python
def forward(self, hidden_states):
    if self.use_clipped_linears:
        hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
    hidden_states = self.linear(hidden_states)
    if self.use_clipped_linears:
        hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
    return hidden_states
```

We initially read "QAT scalars" and reasoned "those get folded at
inference, harmless to skip." Wrong. Skipping them produced an
encoder that drifted gradually through the conformer and then
collapsed at layer 11 (cos vs HF = 0.51 instead of >0.95). The
network was trained EXPECTING clipped activations on every
ClippableLinear's input and output; without the clamps, deep layers
saw distributions outside their effective input range.

**Confirmation strategy that saved us another full debugging session:**
patch HF locally (`Gemma4ClippableLinear.forward = no_clip_forward`)
and re-run. If your ggml runtime + the patched HF agree at cos=0.51
*and* the bit-exact HF reference is the only one at >0.95, the
delta IS the clipping op. We did that, the cos numbers matched to
within 0.001, decision was unambiguous: stop skipping the scalars.

Operational lesson: **store every config flag and every learned
inference-time scalar in the GGUF, even when the name reads like a
training-time artefact.** The converter should default to "include"
and only skip with a citation. Our SKIP_PATTERNS used to read
`".input_max", ".input_min", ".output_max", ".output_min"` because
"clipping scalars" sounded inert. That was 480 missing scalars in
gemma4-e2b alone.

## ggml_clamp is in-place — copy shared inputs first (April 2026)

`ggml_clamp(ctx, a, min, max)` returns `ggml_view_tensor(ctx, a)` and
the executor writes the clamped result INTO `a`'s underlying memory
when the graph runs (`ggml/src/ggml-cpu/ops.cpp` clamp_f32 has
`dst->data == src->data` after the view).

Practical consequence: if `h` is the post-norm hidden state shared
between Q, K, V projections and each has its own clip bounds:

```cpp
ggml_tensor* Q = clipped_mul_mat(L.q_w, h, L.clip_q);  // clamps h in place to clip_q.in
ggml_tensor* K = clipped_mul_mat(L.k_w, h, L.clip_k);  // K reads h that's ALREADY been clamped to Q's range
```

…you'd get `clamp(clamp(h, q_in), k_in)` for K's input, which is the
intersection of the two ranges instead of `clamp(h, k_in)`. The same
applies to any sibling that reads the same intermediate tensor.

Fix: pre-`ggml_cont` once per consumer so each clamp acts on its own
private buffer:

```cpp
ggml_tensor* h_q = ggml_cont(ctx, h);  // private copy
ggml_tensor* h_k = ggml_cont(ctx, h);
ggml_tensor* h_v = ggml_cont(ctx, h);
Q = clipped_mul_mat(L.q_w, h_q, L.clip_q, /*private_input=*/true);
K = clipped_mul_mat(L.k_w, h_k, L.clip_k, /*private_input=*/true);
V = clipped_mul_mat(L.v_w, h_v, L.clip_v, /*private_input=*/true);
```

This applies to any in-place ggml op (`ggml_silu_inplace`,
`ggml_clamp`, etc.) when the input has multiple downstream consumers.
The "in-place" flag in the docstring isn't documentation — it's a
hazard label.

## Stage-by-stage diff with intermediate dumps localises subtle bugs (April 2026)

The Gemma4 audio encoder went from cos=0.10 → 0.49 → 0.97 over a
session of fixes. Each step we knew which sub-module to attack
because we kept a running per-stage cosine table and could see
exactly where the curve broke.

The infrastructure that worked:

1. Python reference dumper (`tools/reference_backends/gemma4.py`)
   registers forward hooks on every named module (subsample, every
   conformer layer, output_proj, audio→LLM adapter) and writes each
   captured tensor as a named entry in a single GGUF reference
   archive.
2. Runtime exposes the same names via `ggml_set_output()` on the
   intermediate tensors and writes them to `CRISPASR_DUMP_DIR=…`
   when the env var is set. No code change to existing
   forward — just `ggml_set_output()` + a side-channel write block
   right before `ggml_free(ectx)`.
3. A short Python diff script reads both sides and reports per-row
   cosine + max-abs per stage. ~30 LOC.

With all three in place, finding bugs becomes: "look at the table,
find where cos drops, dive into THAT module." We localised:

- mel = bit-exact → bug below mel.
- subsample = 0.73 → axis-order bug in conv2d output flatten
  (`(M, T, C) → (C, M, T)` for HF's C-fast-then-M-slow per-frame
  feature ordering). Subsample → 0.999.
- layer_5 = 0.97 vs layer_11 = 0.51 → drift compounding through
  conformer body. Tested without rel_pos_bias to localise → still
  0.51 → it's something in the body, not pos_bias alone. Patched
  HF's ClippableLinear to no-op → HF's no-clip cos=0.51 matched ours
  exactly → the clipping IS the bug.

Total cycles to find each: 1-2 iterations once the table existed.
Trying to debug without the table: weeks (we know — we tried).

The same harness paid for itself again on the LLM side
(`llm_hidden_layer_*` taps) when those become the next surface to
debug.

## HF audio feature extractors are not interchangeable (April 2026)

`Gemma4AudioFeatureExtractor` looks like a Whisper FE on the
outside (mel-128, 16 kHz, log) but every detail differs:

| Param | Whisper-like | Gemma4 | Why it matters |
|---|---|---|---|
| frame_length / fft_length | 400 / 400 | 320 / 512 | window is 320 samples, fft is 512 (zero-padded), produces 257 freq bins |
| windowing pad | symmetric `n_fft/2` | semicausal: `frame_length//2 = 160` left only | first STFT frame centres at t=0 |
| unfold size | `frame_length` | `frame_length + 1`, drop last | one-sample offset across all frames |
| spectrum | power (re² + im²) | magnitude (`|stft|`) | smaller dynamic range pre-log |
| mel filter norm | Slaney (`2/(hi-lo)`) | none | filter peaks have variable magnitude |
| log | `log10(max(mel, 1e-10))` | `ln(mel + 0.001)` | Whisper floors at -10, Gemma4 at ln(0.001)=-6.91 |
| post-log normalisation | clip-and-rescale to `[-1, 1]` | none | Gemma4 features remain in raw log scale |

Mixing these up is silent: the encoder runs and produces "speech-y"
output that the LLM still tries to decode (it heard "la la la la"
when fed mel features computed with the wrong floor). So the FE
must be **bit-exact replicable** in our runtime, not just
"approximately whisper-style."

We ended up writing a dedicated `g4e_compute_mel_hf_faithful` that
bypasses `core_mel` because the combination of semicausal padding +
unfold-by-frame_length+1 doesn't fit `core_mel`'s parameter set. The
test that catches this is the `mel_spectrogram` stage cos: bit-exact
or it isn't.

## VibeVoice-Realtime-0.5B TTS quality regression — issue #39 (April 2026)

User report: with-voice TTS sounded "very noisy" and "crackling"
compared to the upstream microsoft/VibeVoice-Realtime-0.5B output, even
though our ASR round-trip was perfect on every test sample.

The bug count was four, and only the first showed up in a single-stage
frame-0 cos diff. The latter three required either a per-frame diff
across the autoregressive loop or a sample-wise audio cos against a
noise-pinned official run.

### #1 — CFG negative path was advanced by every text window

**Location:** `src/vibevoice.cpp` `process_text_window` (with-voice path).

The C++ runtime ran the negative TTS-LM forward with neg-base-hidden +
type_emb[1] for every text window, advancing `neg_n_past` by the window
size and writing 5 spurious tokens of K/V into `kv_neg` per window.

The official inference loop in `microsoft/VibeVoice/vibevoice/modular/
modeling_vibevoice_streaming_inference.py:600-900` only ever calls
`forward_tts_lm` for the negative path inside the speech-frame loop
(line 841). Text windows update positive only.

The initial `neg_condition` in the official is
`all_prefilled_outputs["neg_tts_lm"].last_hidden_state[-1]` — the
prefilled state of running TTS LM forward on a single `<|image_pad|>`
token at pos 0 with no past KV. Our voice .gguf stores `voice.neg_lm`
and `voice.neg_tts_lm` (the past_key_values) but NOT
`last_hidden_state`. The fix recomputes the prefill at runtime via the
exact recipe (cheap — one IMAGE_PAD pass per synthesize call) and uses
`run_lm_step` to also write the K/V into `kv_neg` so subsequent speech
frames attend to the same content the official does.

### #2 — DPM-Solver++ timesteps were spaced wrong

**Location:** `src/vibevoice.cpp` `make_ddim_schedule`.

The diffusers `DPMSolverMultistepScheduler` with the default
`timestep_spacing="linspace"` does:

```python
timesteps = (
    np.linspace(0, num_train - 1, num_inference_steps + 1)
      .round()[::-1][:-1]
)
```

For N=20 over T=1000 this is `[999, 949, 899, ..., 50]` — the LAST
timestep is **50, not 0**. The final t=0 cleanup is implicit because
`set_timesteps` appends a `sigma=0` to the sigma array
(`final_sigmas_type="zero"` is the default), so the last solver step
naturally lands on `x = x0`.

Our schedule was `linspace(0, T-1, N).round()[::-1]` with the last
forced to 0:

```cpp
s.timesteps[i] = round((T-1) * (1 - i / (N-1)));
s.timesteps[N-1] = 0;
```

For N=20 this gives `[999, 946, 894, 841, ..., 53, 0]`. That's off by
up to 50 from the official schedule. Per-frame ALPHAS_CUMPROD differs
at every step → small per-step error → catastrophic accumulation:
frame 0 still cos 0.999 (small drift), frame 2 v_cfg cos -0.289
(opposite direction!), frames 3+ unsalvageable.

The fix is one line:

```cpp
s.timesteps[i] = round((T-1) * (N-i) / N);  // i = 0..N-1
```

Lesson: **never substitute "what looks like the same linspace" for the
diffusers reference behavior**. The +1 / drop-last / final-sigma=0
combo encodes a specific final-step semantic that's invisible until
you diff against an actual diffusers run.

### #3 — Voice GGUF stores KV but not the prefilled last_hidden_state

The official voice `.pt` files are full
`{lm, tts_lm, neg_lm, neg_tts_lm}` dicts of `BaseModelOutputWithPast`
objects, each containing `past_key_values` AND `last_hidden_state`. The
inference loop reads `tts_lm_negative_outputs.last_hidden_state[-1]` as
the initial negative_condition for frame 0 of the first speech window.

Our `convert-vibevoice-voice-to-gguf.py` only writes the
`past_key_values` tensors — `last_hidden_state` is dropped. So at
runtime the C++ has to either (a) extend the voice file format, or
(b) recompute the prefill from weights. Recompute is cheap (one
1-token pass through 4 base + 20 TTS layers) and stays compatible
with existing voice .gguf files. Bug #1's fix uses (b).

### #4 — σ-VAE Block1D FFN uses GELU, not SiLU

**Location:** `src/vibevoice.cpp` `build_block1d`.

After fixing #1 and #2 the per-frame latent cos vs the official model
was ≥ 0.9989 across 56 frames of the long test sentence. But audio cos
at zero shift was only **0.8225**, and our amplitude was ~65% of
reference. That mismatched the per-frame match: if the latents are
right and the decoder is right, the audio is right.

The decoder wasn't right. From `microsoft/VibeVoice/vibevoice/modular/
modular_vibevoice_tokenizer.py:591`:

```python
class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim, bias=False):
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=bias)
        self.gelu = ACT2FN["gelu"]            # ← GELU
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=bias)

    def forward(self, x):
        x = self.linear1(x); x = self.gelu(x); x = self.linear2(x)
        return x
```

`build_block1d` had `ggml_silu` in the FFN. SiLU and GELU look similar
on a single forward pass (cos ~0.999 element-wise on Gaussian inputs)
but the σ-VAE decoder runs ~30 ConvNeXt blocks on the latent and the
accumulated activation drift produces audible **crackling** — not
broadband noise, not muffling, but something that sounds like clipped
samples or sample-rate mismatch even though the waveform is in range
and at 24 kHz. RMS comes out about 65% of reference because the gating
non-linearity damps the residual stream slightly more than GELU.

After the fix: audio cos vs official = **0.9991** at the 322 ms shift
caused by C++'s leading-silence trim (essentially bit-exact modulo
F16). Spectral profile bands now in family with reference across
0-12 kHz.

`build_block1d` is shared between the encoder and the decoder, so the
ASR-side acoustic and semantic tokenizers also get GELU. The text LM
that consumes the encoder hidden states is robust to small encoder
drift, so we don't expect ASR behavior change — but worth a regression
spot-check on a known sample.

### Per-frame methodology vs single-stage cos

The lesson worth carrying: **for autoregressive models, a single-stage
cos diff at frame 0 is not enough.** Bugs #1 and #2 produced cos > 0.99
at frame 0 stages (because the chain of errors hadn't accumulated yet)
but cos ≈ 0 by frame 2. Bug #4 produced cos ≥ 0.999 at every per-frame
*latent* stage but cos = 0.82 in the actual audio (the decoder had a
constant per-block multiplicative error).

Three orthogonal harnesses — only all three together pin a bug to a
specific module:

1. **Per-frame conditions/latents diff** (`tools/diff_vibevoice_tts.py`
   + `VIBEVOICE_TTS_DUMP_PERFRAME=1` in C++ + hooks on the upstream
   model in `tools/run_official_vibevoice.py`). Catches LM/CFG/AR
   bugs (#1, #2).
2. **Sample-wise audio cos at zero shift, then at the
   cross-correlation peak.** Catches decoder-side bugs (#4) that
   leave the latents intact.
3. **Spectral band-power table.** Catches systematic activation /
   weight scaling errors that a noise-pinned audio cos may miss when
   the diffusion noise itself drives most of the variance.

Pinning RNG is essential for sample-wise audio cos — both runs must
consume the same Gaussian z per frame. The C++ has
`VIBEVOICE_TTS_NOISE` to read a flat float32 table; the upstream-model
hook captures `speech[0]` (the trajectory's actual seed row) inside
sample_speech_tokens and writes it as `noise.bin` next to the per-frame
dumps.

### Reusable pieces extracted

- `run_qwen2_prefill_no_kv` — Qwen2 transformer forward with no past KV
  and a configurable `prefix` / `n_layers` / `has_final_norm`. Useful
  for the negative-path prefill recipe and for any one-shot Qwen2
  forward where you don't want to persist KV.
- `tools/run_official_vibevoice.py` — pattern for "install upstream
  library + monkey-patch internal methods to capture per-frame
  intermediates". Reusable for any upstream HF model that exposes its
  forward methods (here we patched `sample_speech_tokens` directly
  rather than `forward_tts_lm` because the diffusion entry-point gave
  the cleanest cut at conditions, noise, v_cfg, latent in one place).
- `tools/diff_vibevoice_tts.py` — stage table; could be merged with
  `crispasr-diff` once the no-voice path is stable enough to be the
  reference.
- `VIBEVOICE_TTS_DUMP_PERFRAME=1` — convention for "dump
  `perframe_<stage>_f<NNN>.bin` per frame". Worth standardizing across
  AR-TTS backends.

## Mimi/SEANet encoder: pad_mode varies per-layer (April 2026)

Porting the Qwen3-TTS-Tokenizer-12Hz codec encoder (which inherits
`MimiModel` from `transformers`) cost a session because we copied the
"padding looks the same everywhere" assumption from kyutai_stt's encoder
without checking. It isn't.

### What broke

The codec encoder's runtime-extracted RVQ codes weren't matching the
Python-baked codes for the same audio. End-to-end synthesis with those
codes hung the talker (it never emitted EOS). The diff harness reported:

```
cenc_se_init      cos_min=1.000000  PASS
cenc_se_s0        cos_min=0.999339  PASS  (after stride-4 conv)
cenc_se_s1        cos_min=0.973387        (after stride-5 conv)
cenc_se_s2..s3    cos_min ≈ 0.997
cenc_seanet_out   cos_min=0.981
cenc_xfmr_out     cos_min=0.987
cenc_ds_out       cos_min=0.553   ← huge drop, max_abs=11.4
```

`cenc_ds_out` is the stride-2 downsample applied AFTER the encoder
transformer to halve the frame rate from `encodec_frame_rate` to
`frame_rate` (25 Hz → 12.5 Hz here). That single layer was contributing
nearly all of the encoded-codes divergence.

### Why

`MimiConv1d` takes `pad_mode` from `config.pad_mode` by default, which
the Qwen3-TTS-Tokenizer config sets to `"constant"` (zero pad). All
SEANet convs (init, residual, stride, final) and the encoder
transformer's downsample inside the resblocks use that default.

But `MimiModel.__init__` overrides this for the FRAME-RATE downsample:

```python
self.downsample = MimiConv1d(
    config, hidden_size, hidden_size,
    kernel_size=2 * int(encodec_frame_rate / frame_rate),
    stride=2,
    bias=False,
    pad_mode="replicate",   # ← per-layer override
    layer_idx=...,
)
```

`pad_mode="replicate"` means PyTorch's `F.pad(x, (left, right),
mode='replicate')` repeats the boundary values rather than zeroing
them. For a `[1, 2, 3, 4, 5]` tail with `(left=2, right=1)`:

- `'constant'`: `[0, 0, 1, 2, 3, 4, 5, 0]`
- `'replicate'`: `[1, 1, 1, 2, 3, 4, 5, 5]`

For a low-stride conv that operates near input boundaries (like the
final 2× downsample), the difference in the boundary frames is the
difference between "edge-frame fed real signal" and "edge-frame fed
zeros" — and it propagates undamped through the rest of the codec
because there's no further smoothing layer.

### What fixed it

Adding a `cenc_replicate_pad` helper (built from `ggml_view_2d`,
`ggml_repeat`, and `ggml_concat`) and wiring an opt-in
`cenc_conv1d_ext(..., bool pad_replicate)` on top of the existing
zero-pad path.

```cpp
// Replicate-pad along T: pad_left copies of x[0], pad_right copies of x[T-1].
static ggml_tensor* cenc_replicate_pad(ggml_context* ctx, ggml_tensor* x,
                                        int pad_left, int pad_right) {
    if (pad_left == 0 && pad_right == 0) return x;
    const int T = (int)x->ne[0];
    const int C = (int)x->ne[1];
    ggml_tensor* result = x;
    if (pad_left > 0) {
        ggml_tensor* first = ggml_view_2d(ctx, x, 1, C, x->nb[1], 0);
        ggml_tensor* target = ggml_new_tensor_2d(ctx, x->type, pad_left, C);
        result = ggml_concat(ctx, ggml_repeat(ctx, first, target), result, 0);
    }
    if (pad_right > 0) {
        const int Tn = (int)result->ne[0];
        ggml_tensor* last = ggml_view_2d(ctx, result, 1, C, result->nb[1],
                                          (size_t)(Tn - 1) * result->nb[0]);
        ggml_tensor* target = ggml_new_tensor_2d(ctx, x->type, pad_right, C);
        result = ggml_concat(ctx, result, ggml_repeat(ctx, last, target), 0);
    }
    return result;
}
```

Only the downsample layer takes the replicate path; everything else
keeps `ggml_pad_ext`. After the fix, `cenc_ds_out` jumped from
**cos_min=0.553 → 0.998** (max_abs from 11.4 → 0.99, ~10× reduction).

### Generalisable lesson

When porting any HF model that builds its layer stack from a generic
`Conv1d` wrapper (Mimi, EnCodec, SEANet, ConvNeXt, etc.), grep the
constructor for **per-layer kwargs overrides**. The same wrapper class
gets used 30+ times with `pad_mode='constant'` and once with
`pad_mode='replicate'` (or `bias=False`, or `dilation=2`, or
`groups=8`), and that one outlier is exactly where the bug hides.
Don't assume "same class = same config" — diff against `__init__`
keyword arguments at every call site.

### Diff-harness stages were essential

We could not have found this in any reasonable time without
intra-SEANet checkpoints. The end-to-end `cenc_seanet_out` cos_min was
0.981 and the final `cenc_ds_out` was 0.553 — those numbers alone tell
you "something's off late in the pipeline" but not which layer. Adding
five extra dumps (`cenc_se_init`, `cenc_se_s{0..3}`) localised the
drift to `cenc_ds_out` exclusively, and `cenc_se_s0` PASS-ing
confirmed the SEANet base wasn't the culprit. The cost was 30 lines of
Python hooks plus 30 lines of C++ `ggml_set_output` calls; the payoff
was finding the specific layer in one diff run.

### Other Mimi/Qwen3-TTS-Tokenizer encoder issues this harness caught

1. **Causal mask missing** in the encoder transformer. The Mimi
   encoder uses causal sliding-window attention (`sliding_window=250`,
   but typical T_enc < 250 so effectively just causal). Initial port
   used `ggml_flash_attn_ext(..., mask=nullptr, ...)` = full attention.
   Fix moved `cenc_xfmr_out` from cos_min=−0.37 to 0.987.
2. **Diff-harness layout transposition.** ggml `ne=[T,C]` puts T as
   the innermost (memory-fastest) dim, but Python `(T, C)` numpy puts
   C innermost. The element-by-element comparison was reading
   transposed bytes for every stage. Fix: insert
   `ggml_cont(ggml_transpose(...))` to put the dump tensor in
   `ne=[C,T]` so its flat memory layout matches the Python (T, C)
   numpy layout's flat C-order.
3. **`MimiConv1d`'s `extra_padding` ceil-alignment.** When `T_in %
   stride != 0`, an extra zero is padded on the right so output
   length is `ceil(T_in/stride)` not `floor`. The Python formula:
   ```python
   n_frames = ceil((length - K + pad_total)/stride + 1) - 1
   ideal    = n_frames*stride + K - pad_total
   extra    = max(0, ideal - length)
   ```
   For our T_enc=75 → 38 downsample, `extra=1`. Forgetting this gave
   37 frames instead of 38 and silently truncated downstream
   comparisons.
4. **Memory Layout Bug in CPU RVQ.** The `enc_emb` tensor was correctly
   transposed to `[T, 512]` before output, but the manually implemented
   CPU loop for `cpu_k1_proj`, `rvq_nearest_neighbor`, and `rvq_subtract`
   was still using channels-first indexing (`ci * T + t`). This scrambled
   the embeddings before they even hit the quantizer.
   Fix: Rewrite the helpers to use standard row-major `[T, C]` indexing
   (`t * IC + ci`). This fix was the "silver bullet" that moved
   `cenc_codes` from cos_min=0.77 (garbage) to 0.998+ and restored
   perfectly clear voice cloning in the end-to-end CLI.

The complete recipe — for any future audio codec port — is:
**dump after every stride-changing layer, not just at module
boundaries.** Module boundaries hide the conv-by-conv accumulation;
intra-module dumps localise it.

---

## Granite Speech 4.1 (April 2026)

### Q4K OOM in `granite_speech_init_from_file`: tiny `tctx` for `ggml_new_graph`

`granite_speech_init_from_file` precomputes the Shaw RPE lookup at
load time. For F32 RPE weights it just reads bytes directly; for any
quantized type it took an else-branch that built a one-op cast graph:

```cpp
ggml_init_params tip = {2 * ggml_tensor_overhead(), nullptr, true};  // 736 B
ggml_context* tctx = ggml_init(tip);
ggml_tensor* f32 = ggml_cast(tctx, rpe_w, GGML_TYPE_F32);
ggml_cgraph* tgf = ggml_new_graph(tctx);   // needs ~83 KB!
ggml_backend_sched_reset(ctx->sched);      // and sched is still NULL here!
```

Two compounding bugs:

1. The `tctx` was sized for **2 tensor overheads (736 bytes)** —
   enough to host two `ggml_tensor` objects but not the `ggml_cgraph`
   structure that `ggml_new_graph` allocates inside the same context.
   `ggml_new_graph` (with default `GGML_DEFAULT_GRAPH_SIZE=2048`)
   needs ~83 KB for its node/leaf arrays. Result:
   `ggml_new_object: not enough space in the context's memory pool
   (needed 82976, available 736)`.
2. `ctx->sched` is created **after** the RPE precomputation, so even
   if `tctx` had been sized correctly, the `ggml_backend_sched_reset`
   call would have crashed on a NULL scheduler.

The branch was never tested because for the F16 GGUF the converter
keeps `attn_rel_pos_w` in F32 (so the if-branch fires); only Q4K
hits the else.

**Fix.** Skip the graph entirely. Read the raw bytes via
`ggml_backend_tensor_get` (works for any backend, including Metal —
the runtime copies device → host for you) and dequantize on the CPU
using `ggml_get_type_traits(type)->to_float`. No `ggml_context`, no
graph, no scheduler — and it works for any current or future quant
type without code changes:

```cpp
std::vector<uint8_t> raw(ggml_nbytes(rpe_w));
ggml_backend_tensor_get(rpe_w, raw.data(), 0, raw.size());
const struct ggml_type_traits* tt = ggml_get_type_traits(rpe_w->type);
tt->to_float(raw.data(), emb_table.data(), (int64_t)emb_table.size());
```

**Generalisable lesson.** When you need a one-shot dequantize at
load time (small tensor, runs once), prefer the type-traits route
over a tiny graph. Tiny graphs require correctly sizing both the
tensor budget *and* the graph overhead, plus a live scheduler — and
all three preconditions tend to fail in init paths. Reserve the
graph route for runtime dequantizes that genuinely need GPU.

The same pattern appeared a second time in the CPU encoder loop's
per-layer RPE read (`ggml_backend_tensor_get(rpe_w, emb.data(), 0,
emb.size() * sizeof(float))`) — that call requested
`emb.size() * sizeof(float)` bytes from a tensor whose `ggml_nbytes`
was much smaller in the quantized case, so we read random uninitialised
bytes off the end of the device buffer and they tokenised to noise.
The fix is the same dispatch on `rpe_w->type`.

### Granite encoder graph path is ~2.3× faster end-to-end (default since PLAN #16)

The CPU per-layer loop in `granite_speech_run_encoder` dispatches
~96 small ggml graphs (one per matmul / per layer) to the scheduler.
Each dispatch carries scheduler overhead — `ggml_backend_sched_reset`,
`ggml_backend_sched_alloc_graph`, kernel pipeline lookup. With Metal
that's a full command-buffer round-trip per matmul.

The graph path now builds the **entire 16-layer encoder as a single
ggml graph** with per-block manual attention that includes Shaw RPE
bias (bit-near-identical to the CPU loop, well above
`cos_min ≥ 0.999` on JFK). Numbers on M1 / Q4_K (encoder F32) for
an 11 s JFK clip:

| Variant            | CPU loop      | Graph (default) | Speedup |
|--------------------|--------------:|----------------:|--------:|
| `granite-4.1`      | 4.78 s (2.3×) | **2.31 s (4.8×)** | ~2.1×  |
| `granite-4.1-plus` | 9.41 s (1.2×) | **3.74 s (2.9×)** | ~2.5×  |
| `granite-4.1-nar`  | 19.27 s¹       | **6.41 s (1.7×)** | ~3.0×  |

¹ NAR CPU-loop run was disk-contended; both transcripts byte-for-byte identical.

**Per-block subgraph shape.** Mirrors the projector's windowed
dispatch: `ctx_size=200` blocks, `ceil(T/200)` blocks per layer, all
emitted into one big graph. Per block the graph computes
`scores = Q · K^T + Q · RPE_block` → `softmax · V`. The RPE bias
table is precomputed once per layer at load time
(`core_conformer_ibm::build_shaw_rpe_lookup`) and uploaded into a
single stacked input tensor of shape
`(ctx_size*hd, ctx_size, n_layers)`, sliced per-layer via
`ggml_view_3d` on the layer axis.

**Variant-specific bits in the graph.** The PLUS variant captures
post-norm hidden states at every `cat_hidden_layers` index and
concatenates them with the final encoder output along the feature dim
inside the graph (`ggml_concat` axis 0) — fanout off the residual
stream, no extra compute. The NAR variant adds a self-conditioning
residual at the configurable layer (default 8), taps `softmax(mid_logits)`
so the runner can pull per-frame `blank_prob` for the BPE auxiliary
head's `posterior_weighted_pool`, and exposes the final CTC logits as
a named graph output. The pool itself stays on CPU (windowed reduce
that doesn't fit a single ggml op); `bpe_out_w` matmul rides the same
scheduler. The PLUS / NAR-specific concat tensors live in
`granite_build_encoder` (granite-speech) and `granite_nle_build_encoder`
(NAR) — both share the per-block attention + Macaron + conv module
shape via `core/conformer_ibm.h`.

**Regression that motivated this** *(May 2026, fixed in this commit).*
The earlier `GRANITE_ENCODER_GRAPH=1` baseline silently produced
"ask what you can do for your country" on JFK — only the back half
of the quote — because `ctx->rpe_lookup` was built from **layer 0's**
`attn_rel_pos.weight` and reused for all 16 layers. The "RPE is tied
across layers" assumption was wrong for granite-speech-4.1-2b: each
block stores a distinct table (layer 0 mean ≈ 0.00004, layer 1 mean
≈ -0.003, layer 2 mean ≈ -0.002). Layer 0 still matched the CPU
loop bit-for-bit, but layer 1 diverged immediately in the attention
output, and the drift compounded so badly by layer 16 that the LLM
only latched onto the back half of the audio. The CPU loop was
unaffected because it was already building per-layer RPE locally
inside the encoder forward.

**Lessons.**
1. A "single big graph" is almost always faster than many small
   graphs on GPU backends because the per-dispatch cost dwarfs the
   per-op cost for our typical layer sizes.
2. When two paths share a precomputed table (here: RPE lookup),
   make sure they share the *same* table builder — duplicate
   per-layer logic is a divergence that hides until model weights
   stop being uniform across layers.
3. "Approximate path matches reference on JFK" is only as durable
   as the assumptions baked into the reference. Re-validate
   approximations against ground truth whenever upstream weights
   change shape or stop being tied.

### Quantizer encoder skip rule for Granite Speech

The default quantizer skips `proj.` tensors for Granite Speech
(precision-sensitive), but it was happily quantizing the 16-layer
Conformer encoder (everything ending in `_w` / `weight`, 2D, no
"norm" in the name). Result: JFK encoder cos_min dropped from
**0.999908 (F16)** to **0.929 (Q4_K)**, projector from 0.999995
to 0.922. The transcript was still readable on JFK by luck — the
LLM is robust to small acoustic noise — but the README claim
"encoder + projector kept F32" was false.

**Fix.** Detect `general.architecture == granite_speech` in the
quantizer and skip any tensor whose name starts with `enc.`. This
restores Q4K parity to F16 levels:

| File         | encoder cos_min | projector cos_min | size    |
|--------------|----------------:|------------------:|--------:|
| F16          |        0.999908 |          0.999995 | 5.2 GB  |
| Q4K (enc F32)|        0.999908 |          0.999995 | 2.94 GB |
| Q4K (all)    |        0.929    |          0.922    | 1.7 GB  |

**Lesson.** When the README says "kept at F32," verify with a
diff harness against the BF16 reference. The encoder is a 16-layer
Conformer where Q4_K rounding error compounds across layers; even
if individual matmuls land at cos 0.999, by layer 16 you can be at
0.93. The size saving (2.94 GB → 1.7 GB) is real but the parity
loss is not free — ship both, document both, let the user choose.

### Don't hardcode block-attention `context_size` / `max_pos_emb`

The runtime had `const int C = 200; const int max_pos = 512;` in
five places (init, two encoder forward functions, two RPE precompute
blocks). They happened to match the granite-4.0-1b config, and they
also match 4.1-2b — but only because the upstream config files
re-used the same numbers. They are **per-config values** that belong
in the GGUF header (`granite_speech.enc.context_size` and
`granite_speech.enc.max_pos_emb`), with the old values as defaults
so legacy GGUFs without these keys still load.

**Lesson.** Any constant that comes from the model's `config.json`
gets a GGUF key — even if every release so far has used the same
value. The cost is one converter line and one hparam-load line per
key. The cost of *not* doing it is rediscovering the bug six months
later when a new variant ships with a different value.

### Granite Speech 4.1 variant family: base / plus / nar

The 4.1 release ships **three** variants with significantly different
architectures despite the shared "4.1-2b" name:

| Variant | Decoder         | Encoder change          | Outputs                               | Speed | Status |
|---------|-----------------|-------------------------|---------------------------------------|-------|--------|
| base    | Granite-1B (AR) | —                       | text                                  |   1×  | shipped |
| plus    | Granite-1B (AR) | `cat_hidden_layers:[3]` | text + speaker labels + word timing   |   1×  | not yet |
| nar     | NLENARDecoder   | self-conditioning + CTC | text                                  | 5–20× | encoder DONE |

- **plus** is the smallest delta: encoder forward concatenates the
  layer-3 output with the final layer output, so the projector input
  is 2048-dim instead of 1024-dim (`encoder_hidden_size: 2048`).
  The Q-Former cross-attention KV weights are correspondingly
  `(1024, 2048)` instead of `(1024, 1024)`. Converter: emit a new
  GGUF key `granite_speech.proj.cat_layers` (e.g. `[3]`). Runtime:
  capture the layer-3 hidden state during encoder forward and
  concatenate before passing into the projector.
- **nar** is essentially a different model. `model_type: nle`,
  `architectures: ["NLENARDecoder"]`, character-level CTC tokenizer
  (348 tokens incl. ASCII + Kana), `bpe_output_dim: 100353` auxiliary
  BPE head, `self_conditioning_layer: 8` (encoder layer 8 also
  consumes the running CTC logits), `encoder_layer_indices: [4,8,12,-1]`
  (projector reads from 4 different encoder layers), and the LLM is
  Granite 4.0-1B used as a **non-autoregressive** refiner over
  parallel-decoded tokens. New converter + new runtime, ~1000 LOC.

**Lesson.** "Variant" in a model card can mean anything from a one-line
config diff to a wholly different decoder. Read the
`config.json["model_type"]` field before estimating effort:
`granite_speech` and `granite_speech_plus` share 95% of their code;
`nle` shares ~30%.

### Per-stage bench timer (`GRANITE_BENCH=1`)

The runtime now ships a tiny RAII timer wrapped in `granite_bench_stage`
that prints elapsed wall-clock for `compute_mel`, `run_encoder`, and
`run_projector` when `GRANITE_BENCH=1` is set. Cost when disabled is
one cached `getenv` per stage. Useful for A/B-ing encoder paths
(the 4× speedup above was measured with this), and for spotting
which stage dominates after future changes.

**Lesson.** Per-stage bench instrumentation at the public-API level
is cheap to add and pays for itself the first time you need to compare
two implementation paths. Add it to every new model runtime up front,
not retroactively when you're trying to debug a regression.

### Encoder norm + QKV fusion: 3.7× total speedup, math unchanged

Per-layer attention used to be three operations:

```cpp
cpu_layernorm(normed, hidden);        // CPU pass over (d * T) floats
run_matmul(Q, normed, W_q);           // Metal dispatch #1
run_matmul(KV, normed, W_kv);         // Metal dispatch #2
```

Three problems compound:

1. **Two Metal command-buffer round-trips per layer** — each
   `ggml_backend_sched_graph_compute` is ~1 ms on M1, so 2 × 16 = 32 ms
   of pure dispatch overhead per encoder forward.
2. **A CPU layernorm pass per layer** that allocates fresh `nw[d]`,
   `nb[d]` buffers and runs single-threaded over T frames each call.
3. **A CPU ↔ GPU round-trip on the `normed` tensor** — written by the
   CPU, immediately fed back to the GPU as a graph input.

Folded into a single `run_norm_matmul_pair` graph that does
`ggml_norm` + `mul_mat W_q` + `mul_mat W_kv` on the same input. Both
matmuls are parallel branches. Numbers on M1 / Q4_K (F32 encoder),
JFK 11 s clip:

| Metric              | Before     | After      | Speedup |
|---------------------|-----------:|-----------:|--------:|
| `run_encoder`       | 12,624 ms  |  2,972 ms  | 4.2×    |
| `run_projector`     |  1,038 ms  |    234 ms  | 4.4×    |
| Total transcribe    |    19.6 s  |     5.3 s  | 3.7×    |
| Realtime            |    0.6×    | **2.1×**   |         |

The math is the same so encoder cosine vs PyTorch BF16 reference
stays at ~0.999855 (was 0.999908 — drift of 5e-5 from running the
norm in F32 ggml on Metal vs the previous F32 CPU code path).
Projector dropped 4× too as a side effect of the warmer Metal
kernel pipeline cache after the encoder fusion.

**Lesson.** When a runtime mixes CPU and GPU work in tight loops, the
per-layer CPU ↔ GPU boundary is usually a 5–10× cost compared to the
actual compute. Folding the mixed-mode boundary into a single graph
dispatch is the highest-leverage optimization available — bigger than
micro-optimizing the matmul kernels. The trick is finding ops that
share an input (here: norm + Q + KV all consume the same `hidden`);
ggml schedules them as parallel branches so you also save matmul
wall time on top of the dispatch savings.

### Quantizer F16-encoder downcast: bias and conv_bn caveats

`CRISPASR_GRANITE_ENC_F16=1` for the granite quantizer downcasts
`enc.*` weights from F32 → F16, halving the encoder footprint and
landing at ~2.07 GB instead of 2.94 GB (Q4K with F32 encoder) with
encoder cos_min still 0.999855. Two bugs found during integration:

- **`conv_bn.weight` / `conv_bn.bias` must stay F32.** The runtime
  does in-place BN folding at load time, writing the precomputed
  scale/shift back into the conv_bn tensors via
  `ggml_backend_tensor_set(b.conv_bn_w, scale.data(), 0,
   inner * sizeof(float))`. With F16 storage the request size doesn't
  match `ggml_nbytes(t)` and the call aborts. Fix: skip `conv_bn` in
  the downcast pattern alongside the existing `norm` /
  `running_mean` / `running_var` / `rel_pos` exclusions.
- **1D bias tensors must stay F32.** Metal's
  `ggml_add(matmul_out_f32, bias)` asserts `src[1]->type ==
   GGML_TYPE_F32` (in `ggml-metal-ops.cpp:3074`). Restricting the
  downcast to 2D weight matrices (`ggml_n_dims(t) == 2`) covers both
  this and any future 1D parameter that finds its way into the
  encoder.

**Lesson.** "Quantize / downcast everything that's a weight" is the
right starting point but every special tensor (BN stats, biases,
RPE tables, learned positional embeds) needs an explicit
exclusion. Match the converter's `is_f32_tensor` policy when you
write a quantizer's downcast path — they encode the same domain
knowledge.

### `cat_hidden_layers` indexing: `output_hidden_states` includes the input embedding (CONFIRMED)

Implementing granite-speech-4.1-2b-plus's encoder concat exposed two
bugs in sequence. The first was a *silent decode failure* — encoder
loads cleanly, projector accepts the wider 2048-dim input, LLM runs
26 high-confidence tokens to completion … and the transcript is
empty. Two compounding causes:

1. **The PLUS HF snapshot ships only the unified `tokenizer.json`**
   format. No separate `vocab.json` / `merges.txt`. The converter's
   tokenizer-write path conditionally fired only on the legacy
   files, silently writing zero tokens for PLUS. Every decoded LLM
   token id mapped to "" → empty transcript. Fix: parse
   `model.vocab` + `model.merges` out of `tokenizer.json` when the
   legacy files are missing. Handles both
   `[[left, right], ...]` (newer) and `["left right"]` (older)
   merges layouts.

2. **Off-by-one on cat_layer indexing.** The upstream
   `cat_hidden_layers: [3]` in the config indexes into
   HuggingFace's `output_hidden_states` tuple, where index 0 is the
   *input embedding* (after `input_linear`, before any encoder
   block) and index N is the output of encoder block N-1. So `[3]`
   means "after layer 2" in our 0-indexed encoder loop, not "after
   layer 3". Capturing one layer too late slightly degraded the
   projector's K/V quality, which on JFK was enough to push the
   LLM into a degenerate decode that happened to land on tokens
   the detokeniser had already mapped to "" — so the
   off-by-one bug was *masked* by the tokenizer bug. Fixing only one
   of the two would not have produced a working transcript.

**Lesson 1.** Whenever a model config's "layer index" refers into
HF's `output_hidden_states` API, expect the +1 offset for the
embedding slot. Capture after `il == N - 1` for HF's `[N]`. Add a
special case for `[0]` if you want to be able to feed the projector
the input embedding directly.

**Lesson 2.** When the tokenizer is silently dropping tokens, every
downstream signal (cosines, magnitudes, logit confidences) looks
fine — the decode loop generates real-looking ids and the per-token
probability is ~0.99. The bug only surfaces in the printed transcript
because every id maps to the empty string. Add a debug print of
`granite_speech_decode_tokens(...)` output right after the call when
diagnosing "no transcript" issues — that catches the mismatch in one
shot.

**Lesson 3.** Tokenizer files are split across three layouts in the
HF ecosystem, and any new model release can choose any of them:
- `vocab.json` + `merges.txt` (legacy GPT-2 style)
- `tokenizer.json` with `model.vocab` (dict) + `model.merges`
  (`[[left, right], ...]` or `["left right"]` strings)
- `tokenizer.model` (sentencepiece proto, not used by granite)
A converter that silently skips a layout it doesn't recognise will
ship a useless GGUF. Always make missing tokenizer data a *warning*
in the converter output, not silence.


---

# Kokoro / StyleTTS2 (iSTFTNet) lessons — session 2026-05-01

## Lesson 1 — read the source, not the plan

For Kokoro M3-M7b we caught **9 factual errors** in the plan + briefing
context by spending ~10 minutes downloading the reference Python
package (`pip download kokoro --no-deps`) and reading the actual
forward code at `/tmp/kokoro_pkg/unpacked/kokoro/{modules,istftnet,model}.py`.
Errors that would have silently produced wrong outputs:

- TextEncoder activation: plan said GELU, source has LeakyReLU(0.2).
- Pad-wrap convention: input ids must be `[0, *raw, 0]` before BERT
  (`KModel.forward()` line 131). The plan did not mention this; the
  initial M3 run produced the wrong-length BERT output.
- Voice-pack split: plan/briefing said `[pred 0:128 | dec 128:256]`;
  source has it reversed (`model.py:104,118`).
- `dur_enc_out` shape: plan said `(L, 512)`, actually `(L, 640)`
  because the last block in `DurationEncoder` is AdaLN+cat-with-style.
- Duration formula: plan said `softplus`, source uses
  `sigmoid(x).sum(-1).round().clamp(min=1)` over 50 buckets.
- `dec_encode_out` / `dec_decode_3_out` shapes per plan were both wrong.
- Generator activations: two different `LeakyReLU` slopes (0.1 inside
  the upsample loop, 0.01 default after).
- conv_post output split: plan called them "mag" and "phase", but the
  reference applies `exp` on the mag side and `sin` on the phase side
  (`spec = exp(x[:, :n_fft//2+1, :])`, `phase = sin(x[:, n_fft//2+1:, :])`).
- The "pooler" weight in the GGUF is loaded but unused — `CustomAlbert`
  returns `last_hidden_state` directly, discarding the pooler output.

**Rule:** any plan claim about a model's forward pass should be
treated as a hypothesis until verified against the published reference
code. The cost of the verification is small relative to the cost of a
silent numerical divergence at M11/M12.

## Lesson 2 — depthwise ConvTranspose1d without `groups` support

ggml's `ggml_conv_transpose_1d(kernel, input, s, p, d)` has no
`groups` argument. Kokoro's pool layer (in `AdainResBlk1d` upsample
blocks) is depthwise: kernel ne=`(K=3, 1, C)` with `groups=in_C` and
`out_C/groups=1`. We open-coded the math for the specific
`(s=2, k=3, p=1, op=1)` parameters:

```
y[c, 2t]   = w[c, 1] * x[c, t]
y[c, 2t+1] = w[c, 2] * x[c, t] + w[c, 0] * x[c, t+1]    (x[c, T]=0 boundary)
```

**Derivation** (worth memorising — the M11 diff caught a swapped-end
bug here that survived audio QA because the envelope/energy looked
right): PyTorch ConvTranspose1d emits `y[i] = sum input[j]·weight[k]`
over (j, k) satisfying `j·stride + k − padding = i`. For our
`(s=2, p=1, k=3)` config and `i = 2t+1`, the valid (j, k) pairs are
`(t, 2)` and `(t+1, 0)`. Hence `w[2]·x[t] + w[0]·x[t+1]` — NOT
`w[0]·x[t] + w[2]·x[t+1]`. Initial implementation had the kernel ends
swapped; commit `448c1af` fixed it after the M11 dumper made the bug
visible (every `dec_decode_3_out` channel diverged at the upsample
boundaries even though the C++ produced plausible audio).

Implementation in `kokoro_pool_2x_depthwise` (src/kokoro.cpp): permute
the kernel from `(K, 1, C)` to `(C, K)`, cast to F32 (the F16 view +
F32 mul fails on Metal — see Lesson 3), slice into 3 column views,
compute even/odd separately, then interleave via
`(C, 1, T) ⊕ (C, 1, T) → (C, 2, T) → reshape (C, 2T)`. The interleave
trick relies on ggml's contiguous memory layout: in a `(C, 2, T)`
tensor, element `(c, k, l)` is at byte offset `c·sz + k·C·sz + l·2C·sz`,
which lines up exactly with `(C, 2T)` element `(c, 2l + k)`. ~10
ops, very fast on any backend.

## Lesson 3 — F32 × F16 element-wise ops are not portable

`ggml_mul_mat` always casts F16 weights to F32 internally, so mixed
mul_mat works. But `ggml_mul` (Hadamard) requires same-type operands
on Metal. A pattern like `ggml_mul(F32_tensor, F16_view_of_kernel)`
will fail at the kernel-dispatch level.

Fix: cast the F16 weight to F32 before the element-wise op. Either
cast the whole tensor once (cheap, and ggml caches the result) via
`ggml_cast(ctx, w, GGML_TYPE_F32)`, or precompute slice views of an
already-cast F32 copy.

## Lesson 4 — InstanceNorm1d via transpose+ggml_norm

`ggml_norm(x, eps)` normalises along `ne[0]` per other dim — i.e.
LayerNorm-style across the first dim of every column. Kokoro uses two
*different* normalisations:

- `AdaLayerNorm` (DurationEncoder): LN over channels. With layout
  `(C, T)` ne[0]=C, `ggml_norm` gives the right semantics directly.
- `AdaIN1d` (predictor F0/N + decoder body): InstanceNorm1d, i.e.
  per-channel mean/var across time. With layout `(C, T)`, ggml_norm
  would be wrong. Transpose to `(T, C)`, ggml_norm normalises
  over `ne[0]=T` per channel = per-channel-along-T = instance norm 1D,
  then transpose back. Two extra `ggml_cont(ggml_transpose(...))`
  ops per call, acceptable overhead.

The cost of getting this wrong: cosines look "fine-ish" (because both
ops normalise to roughly unit-variance outputs) but the actual
distributions are different and downstream activations diverge by ~10%.

## Lesson 5 — LSTM output assembly via cpy-into-pre-allocated columns

`core_lstm::lstm_unidir` (src/core/lstm.h) builds the per-timestep
forward graph with a pre-allocated output tensor `(H, T)`, and writes
each step's `h_t` into a column via:

```c
ggml_tensor* slot = ggml_view_2d(ctx, output, H, 1, output->nb[1], (size_t)t * output->nb[1]);
ggml_build_forward_expand(gf, ggml_cpy(ctx, h, slot));
```

The scheduler sequences the cpys before any downstream read of
`output` thanks to ggml's view-tracking via `view_src`
(same mechanism used by `core_attn::kv_self_attn` for the persistent
KV cache). Cost: `O(T × H)` instead of `O(T² × H)` for the naive
concat-chain approach. For Kokoro at L_padded=14 this is irrelevant;
at T_frames=1500 (predictor `pred.shared` LSTM) it would have been
prohibitive.

The lazy zero-init for `h` and `c` (skip the `W_hh @ h_{t-1}` matmul
at t=0) avoids needing an externally-zeroed input buffer — when `h`
or `c` is nullptr, the helper substitutes `b_hh` (correct because
`W_hh @ 0 = 0`) and `i ⊙ g` (correct because `f ⊙ 0 = 0`). Saves an
input tensor and an extra `ggml_set_input` call per call site.

## Lesson 6 — Kokoro phonemizer: libespeak-ng vs popen divergence

Replacing the `popen("espeak-ng -q --ipa=3 -v LANG …")` shell-out with
in-process `espeak_TextToPhonemes()` is the obvious latency win
(~30–50 ms per call saved on shell-quoting + fork) but the two paths
do **not** produce byte-identical output even though both ask for IPA:

- **U+200D ZWJ tie characters.** The popen path emits ZWJ between
  IPA symbols that the espeak CLI considers a "tied" articulation
  (e.g. affricates). The library path with `espeakPHONEMES_IPA` does
  not. The Kokoro-82M tokenizer's 178-symbol vocab does **not**
  include U+200D, so the popen output is silently passing through a
  character the model never trained on. Greedy tokenization drops it,
  but anything stricter (e.g. byte-level tokenizers in future
  variants) would diverge.
- **Sentence separator.** popen joins sentences with `\n`; the
  library returns one chunk per call to `espeak_TextToPhonemes`,
  advancing `textptr` until NULL — we join these with `' '`. So the
  library path actually preserves an inter-sentence space that popen
  drops.

For Kokoro at 178 IPA symbols + greedy tokenization the two paths
produce equivalent token sequences after normalization. They are
**not** equivalent at the phoneme-string level. If you ever wire a
diff harness against the phonemizer (PLAN #56 open item 2), reference
and runtime must use the **same** path or normalize ZWJ + whitespace
before comparing.

**Process-global state.** `espeak_Initialize` and
`espeak_SetVoiceByName` are not thread-safe — they mutate process
globals. The runtime takes a `std::mutex` around any espeak call. Init
is one-shot; voice changes are sticky (avoid the
`SetVoiceByName('en-us')` → `SetVoiceByName('en-us')` no-op cost).
Init failure is sticky too: if `espeak_Initialize` returns < 0
(typically a bad data path), set a flag and stop retrying — the call
costs ~5 ms and would otherwise repeat per `kokoro_synthesize`.

**Sandbox-friendly data path.** `espeak_Initialize`'s third arg is the
data path. macOS .app sandboxes and Linux container images often don't
have espeak-ng-data at the default location; export
`CRISPASR_ESPEAK_DATA_PATH=/path/to/espeak-ng-data` to override it
without recompiling.

**Phonemizer language ≠ model voice language.** Kokoro-82M's IPA vocab
(178 symbols) covers a superset of the languages the bundled voice
packs were trained on. The voice prefix tells you what's actually
supported at the model level: `a/b` (en US/UK), `e` (es), `f` (fr),
`h` (hi), `i` (it), `j` (ja), `p` (pt), `z` (zh). **No** German (`d_*`),
Russian (`r_*`), Korean, or Arabic voices ship with the official
release. End-to-end synth on de/ru/etc. *will* run — espeak-ng emits
correct IPA and the model accepts it through an English voice — but
quality degrades the further the phoneme distribution drifts. Concrete
M1 measurement on 2026-05-01 with `af_heart`:
- Healthy synth (peak ~12000, RMS ~1500): en, fr, ru on 3-4 s
  utterances. Short de works too ("Hallo Welt." peak=11919; "Guten
  Morgen." peak=11418).
- Silence collapse (peak 541, RMS 44): a longer German phrase with
  unusual diphthongs (`dˈɔøtʃən fˈoːneːmˌiːtsɜs`) — the model's
  duration predictor evidently produces near-zero envelopes for
  out-of-distribution phoneme sequences. The phonemes printed
  correctly; only the audio collapsed.

For full multilingual quality, either ship language-matched voice
packs (the project's voice GGUF format is documented in
`models/convert-kokoro-voice-to-gguf.py`) or auto-fall back to a
phonologically-close trained language (e.g. `ff_siwis` for German —
French's nasal vowels are a closer match than English's). espeak-ng
also has its own language-level limits worth knowing:

- **Mandarin** comes back with espeak's tone-number IPA
  (`ni2χˈɑu2 …` — digits 1-5 mark tones). The kokoro vocab has no
  tone digits, so the tokenizer drops them and tone is lost.
- **Japanese kanji** triggers espeak's English fallback —
  日本語 → `(en)tʃˈaɪniːz(ja)…` ("Chinese letter") with explicit
  voice-switch markers that aren't IPA. Kana works fine; kanji
  needs an external Japanese frontend (`pyopenjtalk` / `mecab` +
  `kakasi`) to convert to kana before espeak.

# granite-family DRY refactor — session 2026-05-01

## Lesson 1 — read the structs, not the duplication map

PLAN #55 step 5 listed the windowed Q-Former as shared by both
granite_speech and granite_nle. The actual block weight structs proved
the row was wrong:

| TU | Block struct | Per-layer ops |
|---|---|---|
| `granite_speech.cpp` | `granite_proj_block` (`sa_q/k/v/out_w`, `ca_q/k/v/out_w`, `ffn_up/down_w`) | self-attn + cross-attn + FFN |
| `granite_nle.cpp`    | `granite_nle_proj_block` (`attn_q/k/v/o_w`, `mlp_fc1/fc2_w`)            | cross-attn-only + MLP |

A duplication map written from one TU's vocabulary ("we have a Q-Former
projector here, and granite_speech has one too") will quietly conflate
two different architectures. The fast check is to look at the per-layer
weight struct field names — if `sa_*` / `ca_*` / `ffn_*` does not match
`attn_*` / `mlp_*` 1:1, the architectures are not the same Q-Former.
Don't trust the lift target until both structs have been read.

## Lesson 2 — `is_causal` is a clean axis for KV-cached vs whole-sequence

The Granite-1B body is identical between the autoregressive
(granite_speech, KV-cached, causal mask) and the editing pass
(granite_nle, single non-causal forward over (audio | text_with_slots)).
A single `core_granite_llm::build_decoder(..., bool is_causal)` dispatches
to `core_attn::kv_self_attn` (causal+KV) or an inline non-causal flash
helper. The KV-cache tensor handles + n_past + causal_mask are passed
through as nullptr/0 on the non-causal side.

This works because the backbone math (RMSNorm + GQA flash-attn + RoPE +
SwiGLU + µP residual ×0.22 + final RMSNorm) is identical. The differences
that DON'T fit the flag are kept at the call site:

- LM head: NAR uses tied `token_embd_w` and skips the `1/logits_scaling`
  divide; granite_speech uses a separate `output_w` and applies the divide
  for sampling-correctness.
- Slicing: NAR slices the text portion `[n_audio, N)`; granite_speech
  slices the last token in prefill mode.
- Inputs: NAR builds `inputs_embeds` inside the graph (cast + concat of
  audio + `embed_tokens(text_ids)`); granite_speech receives `inputs_embeds`
  pre-built by the caller.

Trying to push these into the same builder with more flags would make
the helper unreadable. Keep the lift to "the backbone every variant
agrees on" and let plumbing diverge.

## Lesson 3 — sibling-not-merge for Conformer dialects

`core/fastconformer.h` (NeMo: conv subsampling + MHA RPE, used by
parakeet/canary) and `core/conformer_ibm.h` (IBM: Shaw RPE + fused conv
layout + BN folding-at-load, used by granite) are **siblings, NOT a merge
target.** Three structural divergences would force the merged helper into
unreadable conditional branches:

- **Position encoding**: MHA RPE (additive bias on QK) vs Shaw RPE
  (per-position lookup table indexed by relative position, applied
  inside the score sum).
- **Conv module**: full pre-norm + GLU + dw conv + post-norm vs the
  IBM fused layout with BN folded into conv at load time.
- **BN folding**: granite folds BatchNorm into the depthwise conv at
  weight-load (saves a runtime op); fastconformer does not.

The duplication-map row called this out explicitly ("do NOT merge —
Shaw RPE / fused conv layout / BN folding differ") and the refactor
honoured it. Future Conformer variants should pick one sibling or
create a third — never collapse the two.

## Lesson 4 — JFK smoke is faster than the diff harness for incremental refactors

The PLAN #55 acceptance criterion was `crispasr-diff` cosine numbers,
but for a structural-rename refactor (same ggml ops, same order, same
tensor handles) the cheaper proxy is just `crispasr -nt -f samples/jfk.wav`
on the three variants. If the transcript matches byte-for-byte, the
math survived. Steps 3, 4, 5 all gated on JFK smoke — total validation
budget per step was ~10 seconds × 3 variants = ~30s. The full diff
harness would have meant dumping a fresh reference GGUF (minutes) for
no extra signal on a pure rename. Reserve the diff harness for steps
where math actually changes.

## Lesson 5 — parallel workers in the working tree

The mid-step-4 build broke not on the granite refactor but on
`src/mimo_asr.cpp`, which a parallel agent was actively editing
(PLAN #51 in-flight Qwen2 LLM build-out). The honest move is:

- Do not `git stash`, `git checkout`, or `git restore` files outside
  your scope — even briefly. Parallel workers depend on those file
  states being preserved.
- Stage commits with explicit pathspecs (`git add src/...specific/file.cpp`),
  never `git add -A` or `git add .`.
- If the working tree becomes uncompilable due to another worker's WIP,
  wait it out rather than fix-and-revert. The parallel worker is the
  one who knows what they meant.

This bit during step 4 — the build failed with `member access into
incomplete type 'ggml_cgraph'` in mimo_asr.cpp; reverting the granite
refactor wouldn't have fixed it. Waited ~5 minutes, the parallel agent
landed their fix, and the rebuild worked clean.

# Kokoro German backbone via dida-80b — session 2026-05-01

PLAN #56 Option 2b: ship a German-trained Kokoro variant whose
predictor + decoder + StyleEncoder were all trained on German, so
prosody on long German phrases sounds native (the existing Option 2a
df_eva fallback uses the official English-trained predictor; speaker
timbre is German, prosody isn't).

## Lesson 1 — modern PyTorch parametrize WeightNorm vs the legacy form

`models/convert-kokoro-to-gguf.py`'s `fuse_weight_norm()` was originally
written against `torch.nn.utils.weight_norm` (PyTorch <2.1 default),
which stores the reparameterisation as `X.weight_g` and `X.weight_v`.
PyTorch ≥2.1 deprecated this in favour of `torch.nn.utils.parametrize.
register_parametrization(X, 'weight', WeightNorm(...))`, which writes
the same two tensors as `X.parametrizations.weight.original0` (g) and
`X.parametrizations.weight.original1` (v). The fusion math is
identical: `w = v · (g / ||v||)` over all-dims-except-0.

The official `hexgrad/Kokoro-82M` checkpoint and StyleTTS2-LJSpeech
both use the legacy form. Re-trains by community contributors
(dida-80b/kokoro-german-hui-multispeaker-base) use the modern form.
Same architecture, same weight values, different on-disk naming —
hard to spot until the converter silently maps zero `weight_g/_v`
pairs and the resulting GGUF has incomplete weights.

Make weight-norm fusion accept both naming conventions (handle either
suffix → same `(g, v)` pair → same fused `weight`). One small chunk
in `fuse_weight_norm()` keeps both Kokoro variants and any future
PyTorch ≥2.1 re-trains usable by the same converter.

## Lesson 2 — distinguish "checkpoint stub config" from "missing config"

dida-80b's HF model dir ships only a placeholder config.json:

```json
{"model_type":"kokoro","architectures":["KModel"],"language":["de"],
 "custom_pipeline":"text-to-speech"}
```

No vocab, no architecture sizes — the converter would have to fall
back to defaults. But dida-80b was trained with semidark's
`training/kokoro_symbols.py` which is *byte-identical* to Kokoro-82M's
sparse 178-symbol IPA vocab (gaps filled with PUA placeholders so
embedding indices match). Falling back to a *dense* 178-symbol vocab
(which `STYLETTS2_DEFAULT_VOCAB` is) would shift every token id by
the placeholder offsets and the model's embedding rows would line up
with the wrong phonemes — silent quality regression.

Lesson: when a downstream re-train ships a stub config, the cleanest
fix is to let the user point the converter at the upstream's
config.json (`--config /path/to/hexgrad/Kokoro-82M/config.json`). The
vocab IDs are guaranteed to match because the re-train was done
against the same symbol table (`kokoro_symbols.py` makes that
explicit). Defaults that "look right" may quietly poison embedding
indexing.

## Lesson 3 — gated HF datasets sometimes have a sibling that isn't

The user-pointed dataset `dida-80b/hui-german-51speakers` returns 401
even with a valid HF token (private/gated/not-listed-publicly). The
original CC0 source `iisys-hof/HUI-Audio-Corpus-German` is downloadable
only via their LibriVox-pulling pipeline (multi-step). Both blocked
the "extract a fresh voicepack" plan.

Then the user pointed to `kikiri-tts/kikiri-german-{martin,victoria}` —
*same maintainer's* TTS org, which ships pre-extracted `voices/*.pt`
voicepacks alongside the Stage-2 single-speaker fine-tunes
(`kikiri_german_*_ep10.pth`). Apache-2.0, in-distribution to the
dida-80b lineage (kikiri's StyleEncoder shares ancestry with dida-80b's
backbone). Saved a half-day of HUI corpus reconstruction.

Lesson: when a HF resource is gated, look for the *publishing org's
sibling repos* before falling back to a multi-step recreate path. TTS
maintainers often ship "deployable" assets in a separate org from
"research" datasets — kikiri-tts vs dida-80b in this case.

## Lesson 4 — ASR roundtrip beats RMS as a TTS quality gate

Peak/RMS gates catch silence collapse but pass through "loud
gibberish" or wrong-language audio. For PLAN #56 Option 2b we ran
parakeet-v3 (`-l de`) over every (backbone × voice) combination on
the long German phrase. Results separated four voices that all
*passed* the peak/RMS gate (≥ 8000 / ≥ 1000) into a clear quality
ranking by edit-distance to the reference text:

| voice (with dida-80b backbone) | ASR roundtrip                                      | failure mode                  |
|---|---|---|
| dm_martin                      | "...Phonemizers."                                  | none — perfect                |
| df_victoria                    | "...Tester des Deutschen Phonemizers."             | "Test" → "Tester" boundary    |
| dm_bernd                       | "...Phonemetzers."                                 | "izers" → "etzers" 1 phoneme  |
| df_eva                         | "...Phonemetzes."                                  | "izers" → "etzes" 1 phoneme   |

Without ASR roundtrip, the four would have looked equivalent. With it,
dm_martin is byte-perfect and df_victoria recovers "Phonemizers"
correctly — both kikiri voicepacks (in-distribution to the dida-80b
StyleEncoder lineage). The Tundragoon voicepacks lose the trailing
"-izers" suffix even on the German backbone, suggesting the prosody
predictor and the style embedding need to come from the same lineage
to nail unfamiliar suffixes.

Bake this into the methodology: every TTS output that matters gets
ASR-roundtripped. Energy gates are a necessary precondition, not a
sufficient quality signal.

## Lesson 5 — single source of truth for per-language routing

The first cut put `kokoro_resolve_model` and
`kokoro_resolve_fallback_voice` directly in
`examples/cli/crispasr_backend_kokoro.cpp`. That works for the CLI but
duplicates policy if a wrapper (Python, Rust, ...) wants the same
auto-routing for `crispasr.Session(model_path, backend='kokoro')` —
the wrapper would have to re-implement the cascade.

Move resolvers into `src/kokoro.cpp` behind a C ABI
(`crispasr_kokoro_resolve_model_for_lang`,
`crispasr_kokoro_resolve_fallback_voice`), declared in `src/kokoro.h`.
Re-export them with a `_abi` suffix from `src/crispasr_c_api.cpp` so
they're visible in `libcrispasr.dylib` (static-archive symbols are
LTO-pruned otherwise — every public ABI function must be touched by
something the dylib link references). Have the CLI delegate to the C
ABI rather than keep its own copy. Now wrappers call one function and
get identical behaviour — Python's `crispasr.kokoro_resolve_for_lang()`
verified against the CLI by running both on the same model + language
pair and comparing.

# Kokoro quant ceiling — session 2026-05-01

PLAN #56 follow-up: when publishing kokoro GGUFs to HF
(`cstr/kokoro-82m-GGUF`, `cstr/kokoro-de-hui-base-GGUF`), we
quantised both backbones to Q4_K and Q8_0 and ran both
`crispasr-diff kokoro` AND `parakeet-v3` ASR roundtrip on each.

## Lesson 1 — Q4_K is below the quality bar for Kokoro

| Quant | EN ASR roundtrip | DE ASR roundtrip |
|---|---|---|
| F16 | "Hello this is a test of the English Phone Miza." | "Guten Tag, dies ist ein Test des deutschen Phonemizers." (perfect) |
| Q8_0 | identical to F16 | identical to F16 |
| Q4_K | "phone miser" (1 word diff, intelligible) | "Guten A, dies ist ein S des Worten von links." (broken) |

`crispasr-diff` cosine numbers tell the same story: F16 hits 16/16
PASS at cos≥0.999 (with the audio_out RNG-divergent stage at 0.99);
Q8_0 has 7 stages drift below 0.999 but stays ≥0.85 on every stage
and produces ASR-identical output; Q4_K's audio_out cos drops to
~0.03 on the German backbone. Q4_K English just-about works because
parakeet's English G2P is forgiving enough to recover a comprehensible
word from drifted prosody, but German has no such tolerance.

Why: `crispasr-quantize` only quantises the matmul tensors; LSTM
biases, norms, and the 178-symbol embedding table stay F32. Kokoro's
matmul population is small enough (459 tensors total, ~110 quantised)
that a Q4_K hit disproportionately damages the prosody predictor +
decoder. Kokoro is not vibevoice.

Decision: only F16 + Q8_0 are published on HF. Q8_0 is the
recommended default — the disk savings vs F16 are only ~13 % but
the ASR roundtrip stays byte-perfect.

Lesson: peak/RMS gates are cheap and necessary, cosine diff is
necessary, but ASR roundtrip is the *only* signal that distinguishes
"loud noise" from "intelligible speech". For TTS quants, the diff
table without an ASR check would have made Q4_K look "merely degraded"
when in practice it produces unusable output for the target language.

## Lesson 2 — community re-trains use modern parametrize WeightNorm

The Python `kokoro` package's `KModel.__init__` only handles legacy
`weight_g`/`weight_v` keys for its weight-norm-registered modules
(decoder iSTFTNet, text encoder convs, predictor LSTM). Modern
PyTorch ≥2.1 community re-trains (e.g. `dida-80b/kokoro-german-hui-
multispeaker-base`) write keys as
`parametrizations.weight.original{0,1}` instead. Loading that
checkpoint via the upstream `KModel(model=...)` constructor falls
into the bare-prefix-strip fallback (lines 70-75 of
`kokoro/model.py`), which yields silent partial-init: predictor +
decoder + text_encoder weights are essentially unloaded.

Manifestation: `crispasr-diff kokoro` against the F16 German GGUF
shows `text_enc_out` cos = -0.17 (anti-correlated) and everything
downstream cascades. The C++ GGUF is fine — the *Python reference*
is wrong because the upstream loader silently dropped weights.

Workaround: pre-process the dida-80b checkpoint before passing it
to KModel — split `state['net']` into top-level component dicts
(KModel's expected layout) and rename
`parametrizations.weight.original0` → `weight_g`,
`parametrizations.weight.original1` → `weight_v`. After this, the
German backbone's reference passes 14/16 stages at cos≥0.999.
Our converter `models/convert-kokoro-to-gguf.py` already handles
both naming conventions natively, so the C++ GGUF was always fine —
only the Python diff harness needed the workaround.

Lesson: when a downstream community re-train ships in a newer
PyTorch's parametrize form, the upstream package's "convenience"
loader can silently underspecify the forward pass without erroring.
Always cross-check loaded weights against the checkpoint's actual
key set when porting reference dumpers across re-trains, especially
when the cosine diff against your C++ port goes anti-correlated on
the *F16* baseline (= the quant isn't the problem).

## Lesson 3 — multi-companion auto-download via ExtraCompanion

Original `Entry` struct in `src/crispasr_model_registry.cpp` had
exactly one `companion_file/url` slot — fine for moonshine
(tokenizer.bin), vibevoice-tts (one voice), qwen3-tts (codec). Kokoro
needs 3 extras alongside the model: English default voice (inline
companion), German backbone (extra), German default voice (extra).
Adding more `companion_file_2/3/...` slots to `Entry` would force
every existing row to add trailing `nullptr,nullptr` pairs and bloat
the table.

Cleaner: keep `Entry` exactly as-is, add a sibling table
`k_extras: [{backend, ExtraCompanion[]}]` indexed by backend name.
Nullable. The resolver runs the inline companion first, then walks
extras for the backend if any. Existing rows are unchanged; new
multi-companion backends just add one entry to `k_extras`.

This pattern generalises beyond kokoro: any TTS backend that wants
to bundle a default voice + a related model (e.g. a speaker-encoder
companion + a watermark detector + …) plugs into `k_extras` without
touching the Entry struct.

Lesson: when extending a packed config table, prefer a sibling table
keyed by the existing identifier (`backend` name) over widening the
struct. Keeps existing rows touch-free, makes the extension visually
obvious to readers, and the indirection cost is one branch per
auto-download — negligible compared to the HTTP round-trips.

# VibeVoice silent unconditioned-voice fallback — session 2026-05-01

While testing `-m auto -l de` for vibevoice-tts, the synth produced
~1.2 sec of phonetically-English-sounding gibberish ASR-transcribed
as "We never's a lot of...". The phrase asked for ~3 sec of German.

Diagnosed in three steps:

1. Cross-test (emma + DE text) → 3.05 sec, ASR "Dies ist ein Test
   der deutschen Stimme." (close enough — synth works with English
   speaker timbre rendering German text).
2. Cross-test (de-Spk0_man + EN text) → still 1.2 sec gibberish.
3. `wc -c` on the local `.gguf` → file didn't exist; I had been
   passing a non-existent path because the German voicepacks weren't
   in `/Volumes/.../crispasr-models/`. The model + the CLI silently
   accepted the bad path and ran with no voice-prompt loaded.

Root cause: `examples/cli/crispasr_backend_vibevoice.cpp` had

    if (!voice_loaded_ && !params.tts_voice.empty()) {
        if (vibevoice_load_voice(ctx_, params.tts_voice.c_str()) == 0)
            voice_loaded_ = true;
        // <-- silent on failure, falls through to synthesize()
    }

`vibevoice_synthesize` then runs without a voice prompt. The model's
EOS classifier triggers within ~1.2 sec because there's no speaker
context to condition on. The audio sounds like English-shaped noise
because the LM defaults to English-ish phonemes when unconditioned.

Fix (commit `49b99f8`):
- Error out cleanly when `vibevoice_load_voice()` returns non-zero,
  instead of falling through.
- When `--voice` is empty, walk a sibling cascade
  (`vibevoice-voice-<lang>-Spk1_woman.gguf` →
  `vibevoice-voice-<lang>-Spk0_man.gguf` →
  `vibevoice-voice-emma.gguf`) and announce the auto-pick on stderr.
- If still nothing resolves, error out with a clear message.

After the fix the same `-m auto -l de` command pulls
`vibevoice-voice-de-Spk1_woman.gguf` via the registry's
`k_vibevoice_tts_extras`, the CLI auto-picks it, and parakeet-v3
roundtrips byte-perfect German. ~3 sec of audio, exactly as expected.

Lesson: every TTS backend has a "model went into a degenerate
fallback state" failure mode that produces output, not an error.
Loud-noise gates (peak/RMS) wouldn't catch this because the audio
hits typical TTS amplitude — the only signal is the *duration*
(uniformly ~1.2 sec) and the ASR roundtrip ("nonsense English"
when the input was German). Bake "voice failed to load → refuse to
synthesise" into every TTS adapter; never paper over a missing
voice prompt by feeding the model defaults.

# Qwen3-TTS-CustomVoice "speaker_embed = token_embd[spk_id]" — session 2026-05-01

PLAN #57 Phase 1: adding `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` looked
like a registry-only drop-in (same architecture as the already-shipped
Base) but the smoke test failed with `no voice — call
qwen3_tts_load_voice_pack or qwen3_tts_set_voice_prompt`.

## Lesson 1 — "same architecture" doesn't mean "same contract"

The CustomVoice GGUF has 402 tensors vs Base's 478 — 76 tensors fewer
because there's *no speaker_encoder section at all*. The runtime
expected ECAPA output and bailed.

The fix isn't "make the runtime tolerate missing speaker_encoder" —
that's papering over the contract. The actual contract from upstream
`modeling_qwen3_tts.py:2091-2101` is simpler: CustomVoice ships with
a fixed `spk_id` table (9 speakers), and the "speaker embedding" for
synthesis is one row from `talker.token_embd[spk_id]` — the codec
embedding table the GGUF already contains. **No new weights, no
ECAPA forward, no codec encoder.** The runtime just has to know
which row to pull.

Lesson: when a fine-tune drops weights you expected, read the
upstream `generate()` before designing your fix. The path that pulls
fewer weights at training time is usually the one that *also* pulls
fewer weights at inference — but via a different mechanism (here, a
table lookup in place of a forward pass).

## Lesson 2 — dispatch on GGUF metadata, not filename

We added `qwen3tts.tts_model_type` (string: "base" | "custom_voice" |
"voice_design") to the GGUF metadata so the runtime branches on what
it actually loaded, not on the filename or the CLI flags. The same
backend factory (`qwen3-tts`) handles both variants; the same `--voice`
flag has different semantics depending on the loaded GGUF (speaker
NAME vs WAV/voicepack path).

Existing Base flows are untouched — the CustomVoice path skips the
speaker_encoder load, the ref_text guard, and the ICL fusion in the
prefill builder, all keyed off `hp.tts_model_type == "custom_voice"`.

## Lesson 3 — fixed-speaker dialect override is a `language_id` swap

Two of the 9 speakers carry Chinese-dialect tags (`dylan`→Beijing,
`eric`→Sichuan). Upstream re-routes `language_id` to the dialect's
`codec_language_id` token (Beijing=2074, Sichuan=2062) only when the
synthesis language is Chinese-or-auto. The runtime hook is the
existing `qwen3_tts_set_language` — set_speaker_by_name calls it
when a dialect is bound and the user hasn't explicitly chosen a
non-Chinese language. No prefill changes needed for the dialect
case; just `language_id <= 0` → dialect token.

## Lesson 4 — the prefill builder difference is small but precise

Side-by-side, the CustomVoice prefill (modeling_qwen3_tts.py:2174-2227,
non_streaming_mode) is almost identical to the ICL prefill, *minus*
the ref_code fusion and *plus* a separate text+codec_pad block. Shape:

```
[role(3)]                                        # <|im_start|>assistant\n
[bridge(L_codec-1)]: tts_pad×(L_codec-2)+tts_bos | codec_in_emb[:-1]
[text_block(N+1)]:   text_proj(text_content)+tts_eos | codec_pad×(N+1)
[final(1)]:          tts_pad | codec_bos_emb
```

trailing_text_hidden = tts_pad_emb (1 row).

A separate `build_customvoice_prefill_embeds` is cleaner than trying
to make the existing ICL builder skip stages — the trailing semantics
differ (single tts_pad row, never the ICL's variable-length
text-overhang case). Keep them as parallel builders.

## Lesson 5 — ASR roundtrip is the unit test (again)

Per `feedback_tts_validation.md`, every TTS output gets ASR-roundtripped.
Four roundtrips covered the implementation: 3 English speakers
(vivian / aiden / serena) all produced verbatim parakeet-v3
transcriptions; the dialect speaker (dylan, Mandarin) produced clean
audio with the runtime correctly logging the `language_id=2074`
override. Catching a regression in this path without ASR roundtrip
would mean noticing wrong-prosody Chinese audio by ear — a much
weaker gate.

# MiMo-V2.5-ASR (PLAN #51 / issue #45)

End-to-end JFK transcription matches the upstream `MimoAudio.asr_sft`
reference verbatim. Five lessons that bit hard during the port —
documented so the next 7B-class speech-LLM porter doesn't have to
re-discover them.

## Lesson 1 — torch+OpenMP deadlock during bf16→f16 conversion

**MUST APPLY** to every `python models/convert-*-to-gguf.py`
invocation in this project, not just MiMo.

Symptom: `torch.Tensor.to(torch.float16)` on bf16 weights goes through
`at::native::DEFAULT::copy_kernel` which in turn enters
`__kmpc_fork_call` → `__kmp_join_barrier` → `__kmp_suspend_64` → an
indefinite `_pthread_cond_wait`. The master thread is parked waiting
for an OpenMP worker that never wakes up. Process appears alive (RSS
stable, mmap'd weights resident, STAT=S, 0.0% CPU) but the GGUFWriter
temp file stops growing after ~50 tensors. We lost ~30 minutes
diagnosing this on the MiMo-ASR convert run before sample(1) showed
the parked stack.

Workaround (zero downside — the cast is memory-bound, not
compute-bound, so single-threaded is just as fast):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  python models/convert-mimo-asr-to-gguf.py …
```

`PYTHONUNBUFFERED=1` is unrelated to the deadlock but pairs naturally
because (a) `python | tee log.file` blocks `print()` flushes for many
minutes otherwise, hiding genuine progress, and (b) makes diagnosing
the deadlock easy — the last buffered line tells you exactly where
torch parked.

## Lesson 2 — Capture tensors MUST call `ggml_set_output()`

This is a re-statement of the existing "ggml / inference engine"
lesson, but the stakes were so high during MiMo prefill that it bears
repeating: any tensor read out of a graph via
`ggml_graph_get_tensor()` (i.e. for diff-harness extraction) must
call `ggml_set_output()` in addition to `ggml_set_name()`. Without
it, the scheduler treats the named cont/add nodes as ordinary
intermediates and may reuse their buffers when allocating later ops
that consume the same upstream tensor.

We hit this on MiMo as `prefill_inputs_embeds` reading back as
post-clobber data — `cos≈0.003` against the reference even though
the fusion math was provably correct. The fix is one line per
capture (`ggml_set_output(t)`) but the symptom looks like a tensor
math bug, not a scheduler bug, and burned ~3 hours.

## Lesson 3 — `core_gguf::load_weights` materialises tensors into RAM

Specifically for the diff harness: the F16 GGUF for a 7.5B-class
model is ~15 GB. `core_gguf::load_weights` calls `_platform_memmove`
to copy mmap'd source data into a freshly allocated CPU backend
buffer. On a 16 GB Mac, this thrashes swap for 25+ minutes and never
completes (we saw the process parked in `_platform_memmove` with
12.7 GB physical footprint and only 1.5 sec CPU time accumulated
after 25 minutes wall clock). Q4_K weights (4.5 GB) fit comfortably,
so the bug is invisible on production paths but blocks F16 +
fp32-ref strict cos≥0.999 validation that the diff harness assumes.

Decision: validated step 9 against Q4_K instead — the cosines
exactly reproduce step 7 numbers, confirming no regression. Tracked
as PLAN #51a to add mmap-style backend buffer allocation (already a
pattern in upstream llama.cpp, plumbs through to
MTLBuffer-newBufferWithBytesNoCopy on Metal).

Generalisable: when porting any 7B+ class model to a memory-tight
host, prefer Q4_K + bf16-ref validation paths; expect to touch the
loader before F16 + fp32-ref will run.

## Lesson 4 — `process_speechdata.InputSegment` is the contract, copy it byte-for-byte

The MiMo asr_sft prompt has a deceptively simple shape: `[9, T]`
where row 0 is text and rows 1–8 are 8-channel RVQ codes. The
non-obvious bits:

- For TEXT segments (no audio), the upstream calls
  `insert_between(tokens, gs-1, value=-100)` which produces
  `[tok, -100, -100, -100, tok, -100, ...]` with text tokens at
  stride `gs` and `-100` fillers everywhere else. The `-100` is
  meaningless (the LM only reads `ids[:, 0, ::gs]`) but the row
  length must be `len(tokens) * gs` to align with the audio rows.
  Audio rows for text segments are uniformly `speech_zeroemb_idx[c]`.
- For AUDIO segments, row 0 is `[<|sosp|>, ...empty*n_groups...,
  <|eosp|>]` insert-between'd; audio rows have the codes flanked by
  `gs`-wide pads of `speech_zeroemb_idx[c]`.
- The 6 segments concatenate along time, so total `T_total =
  Σ T_seg`. `T_groups = T_total / gs` is what the LM body sees.
- During step decode, the upstream `slm_sample` replicates the new
  text token across all `gs` positions of the new group via
  `next_text_tokens.expand(B, gs, 1)`. The replication doesn't
  matter (LM reads stride-gs only) but matching it makes the C++
  code line up trivially with the Python ref.

If you skip any of these, `prefill_inputs_embeds` looks fine in
isolation but cosine drifts on token-level alignment with the
reference. Always dump `prefill_input_ids` as a stage and compare
it against the Python reference's `wrap.get_input_ids()` output
BEFORE looking at downstream cosines. Most "bug" reports in the
diff harness for [ids, codes] backends are off-by-one in the prompt
construction, not in the model.

## Lesson 5 — `tokenize_simple` is not enough for chat-template prompts

`core_bpe::tokenize_simple` is fine for prompt fragments that contain
only plain text. For chat templates with `<|im_start|>...<|im_end|>`
and other special tokens, you need the **qwen3-asr-style splitter**:
greedy match `<|...|>` against `token_to_id` first, emit those ids
directly, THEN `bytes_to_unicode + bpe_one` for the residue.
`tokenize_simple` byte-encodes the `<`, `|`, `>` characters and
tries to BPE-merge them — produces tokens unrelated to the special-
token id, and the LM never sees the chat structure it was trained on.

In MiMo's case this would have collapsed silently — the per-byte
fallback produces a syntactically valid token sequence that the LM
just doesn't know how to interpret, and you'd see the LM produce
`<|empty|>` tokens forever or random text rather than a clean
crash. The qwen3-asr-style splitter lives in
`src/qwen3_asr.cpp:qwen3_asr_tokenize` and we duplicated it inline
in `mimo_asr_tokenize_text`. If a third backend needs this pattern,
factor it out to `core_bpe::tokenize_with_specials(token_to_id,
merge_rank, text)`.

# Qwen3-TTS-Base 1.7B — `spk_enc_dim` parameterization (session 2026-05-01)

PLAN #57 Phase 1 closed by adding `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
The conversion + registry entry took minutes; the synthesis-broken-
in-a-quiet-way fix took an hour because the failure mode looked
identical to a logits-decoding regression.

## Lesson 1 — hardcoded "obviously fixed" dimensions are a foot-gun

The runtime's ECAPA graph builder and speaker-embedding helpers all
sized buffers at 1024 floats:

```cpp
// build_graph_spk_enc
h = ggml_reshape_1d(ctx0, h, 1024);

// run_spk_enc
std::vector<float> emb(1024);
ggml_backend_tensor_get(..., emb.data(), 0, 1024 * sizeof(float));
```

At the time it was written, every supported variant
(`Qwen3-TTS-12Hz-0.6B-Base`) had `talker.hidden_size = 1024` and
`speaker_encoder_config.enc_dim = 1024`, and they had to be equal:
upstream concatenates `speaker_embed.view(1, 1, -1)` with the
talker's codec input embeddings (which live in `talker.hidden_size`),
so a different `enc_dim` would have asserted in PyTorch.

`Qwen3-TTS-12Hz-1.7B-Base` re-pegs both at 2048. The graph still ran
— `spk.fc_w` is 2048-row Conv1d, `mul_mat` produced a 2048-d vector,
the reshape to 1024 lost the high half — and `run_spk_enc` then
copied 1024 floats into a 1024-d std::vector. The 0813869 fix is
five lines: replace `1024` with `c->hp.spk_enc_dim` (already plumbed
from `kv_u32(g, "qwen3tts.speaker.enc_dim", ...)` at line 4332).

The general rule: any literal whose value happens to equal a config
field is a config field with a typo. If the GGUF metadata exports it,
the runtime reads it. The trapdoor is "obvious" constants that work
fine until a bigger / smaller variant arrives.

## Lesson 2 — half-zero embeddings produce degenerate audio, not a crash

Half-truncating the speaker embedding is the worst failure mode for
debugging: the talker accepts the malformed row, the AR loop runs to
the codec_eos, the codec decodes valid audio frames, and you get
audibly degenerate (but well-formed) speech. None of the existing
asserts fire. ASR roundtrip is the only gate that catches it
cheaply.

The pattern matches `feedback_tts_validation.md`: peak/RMS gates
flag clipping or silence; ASR catches "audio is well-formed but
content is wrong." For 1.7B-Base the F16/Q8_0 ASR roundtrips were
word-exact within a few minutes of the fix; without the gate I'd
have shipped a half-broken model.

## Lesson 3 — `(publish pending)` is a fine intermediate state

The registry entry landed in `e3dc578` with the URL string suffixed
`(publish pending)` while the GGUFs were still on local disk. Same
pattern Orpheus uses (`"~3.4 GB (publish pending)"`). Two reasons it
beats the alternative of waiting:

1. The registry entry is the only change that needs a code review.
   Decoupling it from "upload finishes successfully" lets the PR
   land when the binding decision (alias name, default quant)
   is made, not when the upload pipe drains.
2. If the upload is a parallel-worker handoff, the parallel worker
   can't add the registry entry without round-tripping; landing it
   ahead of upload removes one merge.

The cleanup is one commit (`c4b458b` here) that strips the suffix
once the URL serves a 200 (or 302 to Xet CDN). It's also a useful
trip-wire: a "publish pending" string that lingers in main is a
signal that someone forgot to do the upload, which is recoverable;
a dead URL in main is a worse failure mode.

# Orpheus 3B-FT — PLAN #57 Phase 2 slice (c) (session 2026-05-01)

The first Orpheus end-to-end ship surfaced three lessons that
generalise beyond Orpheus and are worth pinning before the
Kartoffel_Orpheus / lex-au checkpoint swaps land.

## Lesson 1 — `add_special_tokens=True` is a silent prompt-format dependency

`canopyai/Orpheus-TTS:engine_class.py:_format_prompt` builds the
talker prompt as

```python
ids = [audio_start] + tokenizer(f"{name}: {text}").input_ids + end_tokens
```

The `tokenizer(...)` call defaults to `add_special_tokens=True`,
which inserts the Llama-3 `<|begin_of_text|>=128000` BOS at the
start of `input_ids`. The printed prompt template doesn't show
that BOS — you have to dump `input_ids` to see it. **Without that
BOS the C++ port produced well-structured but semantically garbage
codec output**: parakeet ASR returned `"Ineonice perfect of the
Pan 8."` for `"Hello, my name is Tara."`. With it, roundtrip lands
exactly.

The trap: well-formed super-frames are a strong signal that the
talker is "working", which masks the actual issue. Generalisation:
when porting a prompt builder from a Python reference, **always
dump the live `input_ids` from the ref and grep for the model's
BOS before trusting the printed template** — `add_special_tokens`
is the most common source of silent BOS injection but tokenizer
configs (`tokenizer_config.json:add_bos_token`,
`PreTrainedTokenizer.__call__` overrides) can also do it.

## Lesson 2 — greedy sampling can be a debugger trap

`temperature=0` on the Orpheus talker enters a 7-slot loop after a
few super-frames and the AR halts after ~24 tokens. This looks
exactly like "model is broken / sampling path is wrong". The
actual root cause is that the upstream `engine.generate()` defaults
to `temperature=0.6` and relies on the noise to escape modal traps
in the codec super-frame distribution — codec tokens are highly
peaked, so greedy sampling routes back through a small attractor.

When validating a new TTS port, mirror the upstream `engine`
defaults exactly **before** declaring anything broken. Greedy
decoding is the right default for ASR; for TTS where the codebook
is a learned distribution, the upstream default is virtually
always >0 and divergence from that is itself a bug.

## Lesson 3 — stop policy by token-class, not just terminal IDs

Orpheus's prompt template includes `eot_id=128009`,
`audio_eot=128260`, and `audio_eom=128261`. All three look like
they could be sentinel "stop here" tokens, but stopping on them
breaks AR for two reasons:

1. `eot_id=128009` is the Llama-3 `<|eot_id|>` and **already
   appears in the prompt** (between the `name: text` block and the
   audio sentinels). Stopping on it halts decoding after 0 codec
   tokens.
2. `audio_eom=128261` and `audio_pre_end=128009` map to `text_N<10`
   reserved markers in the custom_token block. The reference
   `tokens_decoder` filters them silently rather than terminating.

The correct stop policy is **`audio_end=128257` OR >4 consecutive
non-codec tokens**. The >4 heuristic catches the model jumping out
of the custom_token block when it has nothing left to say. Same
shape will likely apply to Kartoffel_Orpheus / lex-au and to the
other Phase 2 talkers (gwen-tts-0.6B, tada-3b-ml) where the
"terminal" token IDs in the prompt template overlap with text-side
specials.

## Lesson 4 — disable `GGUFWriter(use_temp_file=True)` on full disks

`models/convert-orpheus-to-gguf.py` initially used the writer's
default `use_temp_file=True`, which buffers tensor data to a
`tempfile.SpooledTemporaryFile` (honors `TMPDIR`) and copies to
the output afterward. With `/Volumes/backups` at 100% during this
session, the temp-file write hit silent corruption / multi-MB/s
throughput collapse (cf. the gigabyte-scale slowdown documented in
`.claude/CLAUDE.md` "TMPDIR matters for converter temp files"). 
Setting `use_temp_file=False` writes tensors directly and side-steps the
issue at the cost of holding the full tensor list in RAM during
write — fine for 6.6 GB f16, will need re-evaluation for VoxCPM2.

# MiMo-V2.5-ASR perf wave — PLAN #51b/b' lessons (May 2026)

Three portable lessons from the first decode-side perf pass on the
mimo-asr runtime (HISTORY section 60). 1.46× per-step decode
speedup; the patterns generalise to any speech-LLM with a heavy
prefill branch that's dead at decode.

## Lesson 1 — when a decode branch mathematically zeroes out, elide it

mimo-asr's prefill graph has two parallel branches: a 6-layer audio
input_local_transformer + group_proj + fusion (audio path), and
a text embed lookup + zero-mask (text path). The two are summed at
`prefill_inputs_embeds`. At decode time, every step's text row
holds the new generated token (so `text_zero_mask[g] = 1`,
`speech_active_mask[g] = 0`) and the audio rows hold
`speech_zeroemb_idx[c]` whose combined-mask gates all contributions
to zero. The audio path's contribution to `inputs_embeds` is
provably zero. Reusing the prefill graph means computing 6 audio
layers + the group_proj matmul + the fusion add per step, all of
which are dead ops.

The fix: build a separate `mimo_asr_build_step_graph` that skips
the audio branch outright (`embed_w[next] → 36L LM trunk`). Decode
loop calls it for n_past>0; prefill keeps using the full graph for
n_past=0 (where the audio path is alive).

This pattern generalises. For any multi-modal speech-LLM where
decode-step inputs zero out the prefill-only branch via
build-pattern-fixed masks (`speech_active_mask=0` is the giveaway
— if you can prove the mask is identically zero across all decode
steps, the entire branch is dead), splitting into prefill-graph +
step-graph is a free perf win. Watch for this on:

- granite-speech NAR/AR (Q-Former projector dead at decode)
- voxtral / voxtral4b (audio encoder dead at decode)
- qwen3-asr (whisper encoder dead at decode)

The diff is small — roughly half a graph builder, copy/paste from
the prefill builder minus the audio nodes. The validation is
free: greedy decode is deterministic, so the existing
JFK-transcript-equality test catches any divergence.

## Lesson 2 — the qwen3-tts O15 cached-graph pattern transfers cleanly

The cached T=1 step graph from `src/qwen3_tts.cpp:976-1050`
(opt-in via `QWEN3_TTS_O15=1` for the code-predictor; the talker
runs uncached) ports directly to mimo-asr without modification.
The recipe in three lines:

1. `fixed_kv_len = kv_max_ctx` so graph topology is invariant
   across n_past (every step has the same Lk-shaped K/V read view
   and mask).
2. `kv_indices = positions` (the same I32 tensor that drives RoPE)
   so the K/V cache write becomes a runtime-keyed
   `ggml_set_rows(k_layer, K_new_perm, kv_indices)` instead of a
   static byte offset baked into the graph at build time.
3. Cache the `gf` pointer in the context. On every subsequent step,
   skip `ggml_backend_sched_reset` + `_alloc_graph` (the qwen3-tts
   `skip_plan` path); only update the input tensor values
   (`text_input_ids`, `lm_positions`, `lm_causal_mask`).

Two non-negotiables, both already documented elsewhere in this
file but worth re-stating because they bit hard for qwen3-tts and
will bite again:

- **The KV buffer must be zero-cleared after `ggml_backend_alloc_buffer`.**
  Otherwise the never-written tail slots hold whatever bytes the
  allocator handed back, and the partial-CPU / Metal fallback
  paths read past the masked region anyway. mimo-asr already had
  the `ggml_backend_buffer_clear` in `mimo_asr_kv_init` — keep
  it. (See qwen3-tts commit `7298dd5` for the reference
  regression.)
- **Cache invalidation is a real concern.** The cached graph
  lives in `ctx->compute_meta`. Any other graph-build call into
  the same buffer (a fresh prefill, an `extract_stage` invocation
  from the diff harness, even a different context's graph if you
  share `compute_meta`) clobbers the cached pointer. Set
  `ctx->step_t1_gf = nullptr` at every transcribe entry and after
  every other graph-build call into the same arena. The fact that
  this is `nullptr`-safe (the runner just rebuilds) means the
  failure mode is "no perf win" rather than "crash" — quietly
  bad, hence document the invariant.

The pattern transfers to **any** KV-cached autoregressive
decoder, not just speech-LLMs. Talker / code-predictor / LLM
trunk / NAR — they all have the same shape.

## Lesson 3 — `ggml_set_output` is allocator-pressure, not a feature flag

Lesson 2 of the section above ("Capture tensors MUST call
`ggml_set_output()`") is the safety side. The other side is: in
production paths where you don't actually read the captured
tensor back, **do not** call `ggml_set_output` — it pins the
buffer past its last consumer and forces the allocator to keep it
resident.

mimo-asr's prefill graph captures 5 named stages
(`prefill_audio_features`, `prefill_text_embeds`,
`prefill_inputs_embeds`, `prefill_last_hidden`,
`prefill_text_logits_step0`), only the last of which is consumed
by `mimo_asr_run_lm`. The other four are diff-harness-only.
Gating them behind a `bool diag_captures` parameter (production
transcribe passes `false`, diff harness `extract_stage` passes
`true`, `MIMO_ASR_DIAG=1` env force-enables for transcribe-time
debug) drops 4 `ggml_set_output` calls + 2 `ggml_cont` clones on
the production path. ~5 % wall-clock + a measurably cleaner
allocator picture (smaller peak metadata footprint on the
scheduler).

The pattern: any backend whose stage-extract API and production
forward share a graph builder should parameterise the captures.
Same shape applies to vibevoice (4 prefill stages),
qwen3-asr (3 prefill stages), parakeet (encoder-side stages) —
all currently capture all stages unconditionally.

## Lesson 4 — mmap loader makes large F16 *loadable* but not *fast* on RAM-tight hosts

PLAN #51a (CPU + Metal mmap loaders, commit `9710f80`) and #60a
(`posix_madvise(WILLNEED)` async readahead, commit `f1f4bce`)
together remove the OOM wall: a 16 GB F16 GGUF can now be
*opened* on a 16 GB Mac without `_platform_memmove` thrashing
swap. RSS during load stays around 1-2 GB instead of climbing
to 12+ GB.

**But "loadable" is not the same as "usable in real time."** The
fundamental constraint is unchanged: a 16 GB F16 working set
needs **≥20 GB RAM** to comfortably fit + leave headroom for
activations + KV cache + audio tokenizer + the OS. On a 16 GB
Mac, prefill compute walks layer-by-layer through weights that
keep getting page-evicted between accesses; every re-touched
page costs a synchronous read from disk (~5-10 ms each on a
99%-full external — and our build dir lives on one).
PLAN #51c F16 attempt hit this: 51 min wall clock, 0.1 % CPU,
prefill never finished.

**Generalises to every 7B+-class speech-LLM** in our family
(MiMo-ASR, qwen3-asr-7B variants if we ship them, voxtral4b at
the boundary):

- F16 loads fine on 32+ GB boxes — ship the F16 GGUF for those
  users, document them as the recommended target.
- On 16 GB boxes default to Q4_K — fits comfortably (4-5 GB
  resident), warm decode hits 0.79 s/step (51b/b' result).
- Don't try to bench F16 cold-start on a 16 GB box; the result
  will be dominated by page-fault wait, not by compute. Use
  Q4_K as the bench dtype and rely on the F16 cosine
  validation on a larger box to confirm numerical fidelity.

Diagnostic signature of the thrash mode (so you recognize it
fast next time): `ps -p <pid> -o %cpu,%mem,rss,etime,state` shows
`%CPU=0.0-0.5`, `STAT=SN` or `UN`, `RSS` *decreasing* over time
as the OS pages out the process under memory pressure, while
`vm_stat` reports very low `Pages free`. Don't wait it out —
either move to a larger-RAM box or switch dtype.

# Cross-cutting lessons — session 2026-05-02

## Lesson — `clang-format -i` is destructive when local version ≠ CI version

CI runs `clang-format-18` on Ubuntu (per `.github/workflows/lint.yml`).
Local Homebrew clang-format on macOS is `v22+` and disagrees with v18 on
many wrapping rules. Running `clang-format -i src/...` locally to "be
safe" pulls in v22's preferences across every previously-conformant
line, producing a diff like:

```
 bindings/javascript/emscripten.cpp        | 146 +++++++++++++++---------------
 bindings/ruby/ext/ruby_crispasr_session.c | 130 ++++++++++++++------------
```

— hundreds of whitespace-only changes for a 3-line addition. CI v18
will then *also* refuse the result because v22's reformat doesn't match
v18's style either. Net result: pollutes blame, creates merge conflict
risk with parallel agents, and **doesn't fix what CI is checking**.

The right workflow:

1. Add your code in the existing file's hand-aligned style. Match
   neighbouring lines visually rather than running `-i`.
2. Verify only with `clang-format-18 --dry-run --Werror <files>` if you
   have v18 (Linux), or accept that **only the lines you added** can
   trigger violations and the rest are pre-existing.
3. If v18 ever flags lines you added (CI annotation), reproduce locally
   with `clang-format <newer-version> --dry-run --Werror` — the violations
   that BOTH versions agree on are the real ones to fix. Apply the fix
   manually to those specific lines, never `-i` the whole file.

This bit during the PLAN #56 #5 wrapper sweep — auto-formatting
`emscripten.cpp` and `ruby_crispasr_session.c` produced 276 lines of
whitespace churn for a 17-line logical addition. Reverted with
`git checkout HEAD -- <files>` and re-applied only the new lines by
hand; net diff went from 276 → 17 lines.

Earlier same session (`d393a43` PLAN #53), the inverse trap also
fired: 4 lines I added to `core/activation.h` and `core/conv.h` happened
to be v18-non-conformant *and* not flagged by v22. CI failed; fix
shipped as `21464e3`. So the rule is symmetric: trust neither
"local says clean" nor "auto-format made it clean" — only CI v18 is
authoritative, and the safest local shortcut is just to align by eye.

## Lesson — NeMo ref dumpers must transpose to TimeMels for parakeet / canary

`tools/reference_backends/{parakeet,canary}.py` both use NeMo's
`model.preprocessor(input_signal=sig, length=sig_len)`, which returns
mel features as `(B=1, n_mels, T_mel)` PyTorch tensors. The C++ runtime
side, in turn, uses `core_mel::Layout::TimeMels` — flat ordering with
`n_mels` as the fast (innermost) axis and `T_mel` as the slow axis.

Numpy `(B, n_mels, T_mel)` → after `[0]` strip → `(n_mels, T_mel)`
C-contiguous = n_mels-major flat. That is the OPPOSITE of what the C++
side returns from `<backend>_compute_mel`. Without a transpose, the
diff harness compares `mel_ref[i_mel, t]` against `mel_cpp[t, i_mel]`
element-wise, producing cosine values typically in `[-0.95, 0.95]`
depending on signal symmetry — looks "broken" but with no obvious
arithmetic cause.

The `parakeet.py` template gets it right with:

```python
m = feats[0].transpose(0, 1).contiguous()  # (T_mel, n_mels) — TimeMels
out["mel_spectrogram"] = m.detach().cpu().float().numpy()
```

The `canary.py` initial draft (commit `63f708e`) omitted the transpose
on the assumption that `canary_compute_mel`'s return shape comment
`(n_mels, T_mel)` meant n_mels-major flat. It does NOT — `core_mel`'s
internal layout enum overrides the doc comment. Diagnostic signature:

| stage | cos_min | cos_mean | meaning |
|---|---:|---:|---|
| `mel_spectrogram` | -0.83 | 0.002 | layout swap (or fully unrelated tensors) |
| `mel_spectrogram` | 0.94 | 0.9995 | layout correct, small boundary diff (frame-level outlier) |
| `mel_spectrogram` | 0.999 | 0.9999 | layout correct, F32 ↔ F32 numerical match |

**Rule of thumb:** any new NeMo-backed reference backend should mirror
parakeet.py's transpose unless the C++ side explicitly uses
`Layout::MelsTime`. When in doubt, dump both shapes from a tiny script
and compare against a diff harness PASS on a known-good neighbour
(parakeet's mel diff is the canonical reference).

Encoder cos divergences downstream of a layout-clean mel are then a
separate question — for Q4_K models a cos_min in `[0.3, 0.5]` range
on the encoder output is expected (low-magnitude frames dominated by
quant noise), and isn't a real bug as long as the runtime transcribes
correctly. F16 GGUFs are required for strict cos≥0.999 validation;
on storage-tight machines that's not always feasible (PLAN #5 canary
diff was left at the Q4_K + functional-transcribe gate for that reason).

### Quantised KV cache: write needs `ggml_set_rows`, read needs `Q*→F32` cast (PLAN #60e)

Two CPU-backend mismatches surfaced when first wiring
`CRISPASR_KV_QUANT=q8_0` into `core_attn::kv_self_attn`. Recording
both because the rules generalise to any quantised-cache work:

**1. `ggml_cpy(F32_src, slice-of-Q*-cache)` is not legal — use
`ggml_set_rows` instead.** The CPU backend's `dup_to_q<float>`
(`ggml/src/ggml-cpu/ops.cpp:292-322`) requires
`ggml_is_contiguous(dst)` AND `nb00 == sizeof(src_t)`; otherwise it
hits `GGML_ABORT("not implemented")`. A per-token slice into a
`max_ctx`-strided 4D KV cache is **never** contiguous (the
sub-row stride leaves gaps for the rest of `max_ctx`). Metal's CPY
also rejects non-contig quant dst. The only path that works on
both backends for `F32→Q*` is `ggml_set_rows` — it accepts a
strided dst directly because the row-id tensor encodes destination
offsets explicitly. The `positions` tensor (already populated with
`[n_past..n_past+T)` for RoPE) is exactly the row-id list set_rows
wants, so re-use it as the synthetic kv_indices when the caller
didn't supply one.

**2. CPU `compute_forward_dup` only implements `Q*→F32`, not
`Q*→F16`** (`ops.cpp:560-572`: the switch falls through to "fatal
error" when `src->type` is quantised and `dst->type != F32`).
Metal supports both, but the scheduler can route the cast op to
CPU and that backend then aborts. So when dequantising a quant KV
view on the read path, **always cast to F32**, never F16. F32 K/V
costs marginal extra bandwidth on read (the cache *storage* is
still half-bytes for Q8_0), and `flash_attn_ext` on Metal accepts
F32 K/V natively (`ggml-metal-device.m:1165`).

**Diagnostic signature:** `ggml_compute_forward_dup_to_q<float>` /
`ggml_compute_forward_dup` aborts in the backtrace, with frames
running through `ggml_backend_sched_graph_compute_async`. The
op-source-and-target types tell you which mismatch you hit.

### GGUF in-place patcher beats fresh re-conversion when only layout changes (PLAN #60d)

When a port adds a tensor-layout change (e.g. fusing per-LM-layer
Q/K/V into one `attn.qkv.{weight,bias}` per layer), the natural
instinct is to update the converter and re-run BF16→F16. On a
contested external disk this can be infeasible — the PLAN #60d
attempt sustained ~0.8 MB/min and was killed after 22 min with
~100 MB temp written (same disk-thrash signature as PLAN #51c F16).

The shortcut: write a small python patcher that loads the existing
GGUF via `gguf.GGUFReader`, byte-concat's the per-row Q/K/V data
along axis=0, and re-emits via `gguf.GGUFWriter`. **Bit-identical
to a fresh-from-converter result** because:

- F16/F32 weights: numpy `concatenate(axis=0)` is element concat;
  trivially equivalent to writing rows in the new order.
- Q4_K / Q8_0 weights: each row is independently quantised (no
  super-block ever crosses row boundaries, given `ne[0]` is a
  multiple of the block size — 256 for Q4_K, 32 for Q8_0). So
  byte-concat along axis=0 (the row dim in `GGUFReader.data`'s
  numpy memmap) is bit-identical to re-quantising a fresh fused
  F16 with the same source values.
- F32 biases: trivial element concat.

The patcher runs in ~5 min on the same contested disk because it
only reads the already-quantised GGUF (4.2 GB) instead of the
14 GB BF16 safetensors. See `tools/patch_mimo_asr_fuse_qkv.py` for
the pattern. Quirk: pass `data` (the numpy buffer) without
`raw_shape` to `GGUFWriter.add_tensor` — for `np.uint8` quant
data it correctly interprets the array shape as the byte-shape and
calls `quant_shape_from_byte_shape` to derive the logical shape;
explicit `raw_shape` confuses that branch.

### HF `upload-large-folder` rename-before-replace beats replace-then-re-upload

When you want to keep the old GGUF as an A/B reference under a new
name AND replace the canonical-name file with a new variant, the
**only efficient order** is rename-first, upload-second:

1. `HfApi.move_files(repo_id, ['mimo-asr-q4_k.gguf' →
   'mimo-asr-q4_k-unfused.gguf'], ...)` — server-side metadata
   move, instant, zero bytes transferred.
2. `hf upload-large-folder` of the new file as `mimo-asr-q4_k.gguf`
   — uploads only the new bytes once.

Replace-first is wrong: the original bytes are no longer accessible
by path on HF (only the latest revision per path is served), and
re-uploading the local copy of the original bytes under the new
name re-incurs the full 4.5 GB hash + pre-upload cost. In this
session, three kill+restart cycles all stalled in the pre-upload
phase (no xet activity logged after the initial config dump);
local A/B copy ended up being the practical fallback. Lesson learnt
post-fact — apply for any future variant-rotation publish.

### Don't pre-compute KV-buffer byte sizes from a hardcoded dtype — use `ggml_nbytes()` after creation (PLAN #60e wiring)

Found in `voxtral4b_kv_init` and `granite_speech_kv_init` while
threading `CRISPASR_KV_QUANT` through every backend's `*_kv_init`.
Both had this pattern:

```cpp
size_t k_size = (size_t)ggml_type_size(GGML_TYPE_F16) * hd * max_ctx * n_kv * nl;
ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);
ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, k_size + v_size);
```

The pre-compute is bug-prone for two reasons:

1. **Hardcodes the dtype.** Switching to a quant type (e.g. Q8_0)
   under the env flag silently over-allocates: `ggml_type_size(Q8_0)
   = 34` bytes (one block holding 32 quants), so the formula treats
   each *element* as 34 bytes → ~30× over-alloc for Q8_0 K/V. The
   buffer still works (over-alloc is safe) but wastes RAM.
2. **Skips ggml's row-padding logic.** Quant types pad rows to a
   block-multiple of `ggml_blck_size`; `ggml_nbytes()` accounts for
   this, the manual product doesn't.

Right pattern (already used by mimo_asr / orpheus / qwen3_tts):

```cpp
ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_dtype, hd, max_ctx, n_kv, nl);
ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_dtype, hd, max_ctx, n_kv, nl);
const size_t kbytes = ggml_nbytes(ctx->kv_k);
const size_t vbytes = ggml_nbytes(ctx->kv_v);
ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, kbytes + vbytes);
```

Always size buffers from `ggml_nbytes()` after the tensor exists —
it's the only path that's correct across F16, F32, and every quant
type without per-dtype branches.

### Cached ggml graph + `sched_reset()` corrupts view→buffer-id mapping (multi-backend schedulers)

Symptom: `GGML_ASSERT(src_backend_id != -1) failed` inside
`ggml_backend_sched_split_graph` (somewhere around
`ggml-backend.cpp:1328`) on a *second* invocation of a graph that was
built once and re-run via `ggml_backend_sched_reset()`.

Reproduced on Metal (commit `e586f5e`, original VibeVoice TTS fix)
and Vulkan (issue #47, May 2026). Both are multi-backend ggml
schedulers — they assign every tensor a backend-id by walking the
op tree, and that id mapping for *view* tensors does not survive
`sched_reset()`. On the second invocation the scheduler can't
resolve the parent buffer for the cached view, the assert fires, and
synthesis crashes.

CUDA almost certainly has the same shape bug — same scheduler
architecture, same `sched_reset` semantics — even though no CUDA
user has reported the assert yet. After the Vulkan disprove we
decided to **assume CUDA is broken** rather than wait for a crash
report; the cost (giving up the speculative ~30% cache-reuse
speedup) is preferable to shipping the known-pattern crash to a
user.

**Workaround pattern** (see `backend_needs_fresh_pred_graph()` in
`src/vibevoice.cpp`):

1. Prefix-match the backend name and bail to rebuild-each-call for
   any backend on the bypass list (Metal, Vulkan, CUDA today).
2. Provide an env escape hatch (`CRISPASR_VIBEVOICE_REUSE_PRED_GRAPH=1`
   for vibevoice) that forces the cache-reuse fast path regardless
   of backend. This is the path you flip on once a kernel
   maintainer confirms the upstream scheduler has been fixed for a
   given GPU backend, or to benchmark the speedup the cache would
   save if the bug weren't there.

SYCL / HIP / ROCm aren't in the list yet — add the prefix when
either a report comes in or someone audits the upstream
`sched.cpp` `src_backend_id` reset path and confirms the shape.

The proper long-term fix is upstream ggml — make
`ggml_backend_sched_split_graph` recompute view→backend mapping
from `view_src->buffer` rather than relying on cached ids. Once
that lands, the bypass list collapses back to empty.

# Cross-cutting lessons — session 2026-05-02 (feature-matrix closure)

## Lesson — "streaming" in a backend's name doesn't mean streaming in its API

`moonshine_streaming` (`src/moonshine_streaming.cpp`) and
`kyutai_stt` (`src/kyutai_stt.cpp`) both have streaming-friendly
*architectures* — sliding-window attention in moonshine, 12.5 Hz
Mimi-frame-aligned text emission in kyutai. The PLAN inferred
"~80 LOC each to wire into the unified stream API" from those
architectural facts. **Wrong.**

The actual public APIs are single-shot:

- `moonshine_streaming_transcribe(ctx, pcm, n_samples) -> char*`
- `kyutai_stt_transcribe(ctx, pcm, n_samples) -> char*`

Both internally batch-encode the whole audio (kyutai resamples
16 kHz → 24 kHz then runs the entire Mimi codec at once, then
runs the whole LM loop). To reach low-latency streaming requires
real refactoring: incremental encode (Mimi has sliding-window
context — chunked encoding ≠ batch encoding bit-for-bit), per-call
state carry-over (KV caches, partial decoder state, the
`audio_delay_seconds` lookahead), and numerical regression
gating (chunked vs batch on JFK should match within ~99.9%
cosine; below that the chunk boundaries are emitting different
tokens).

**Realistic estimate: ~300 LOC + 1-2 days of debugging encoder
boundary effects per backend**, not 80. Generalises: when a
backend description says "streaming arch" but the C-API takes
`(pcm, n_samples)` and returns a flat `char*`, assume the
streaming bit hasn't been plumbed yet and budget accordingly.

## Lesson — `MA_NO_DEVICE_IO` defines you forgot you set

`crispasr_audio.cpp` shipped with three miniaudio optimisations
since the start:

```cpp
#define MA_NO_DEVICE_IO  // we don't need playback/capture devices
#define MA_NO_THREADING  // avoid pthreads coupling in the dylib
#define MA_NO_GENERATION // skip synth helpers
```

These were correct when libcrispasr only decoded WAV/MP3/FLAC
files. When PLAN #62d added `crispasr_mic_*` (which uses
`ma_device_init/start/stop/uninit`), the link failed with
"undefined symbol _ma_device_*" — the symbols had been compiled
out by `MA_NO_DEVICE_IO`. Fix: drop `MA_NO_DEVICE_IO` (and
`MA_NO_THREADING`, since device IO needs threading anyway).

Diagnostic signature: `ld: Undefined symbols for architecture
arm64: _ma_device_init / _ma_device_start / _ma_device_stop /
_ma_device_uninit`, all referenced from a *new* TU that just
`#include "miniaudio.h"`. Search for `MA_NO_*` defines in the
TU that has `MINIAUDIO_IMPLEMENTATION` — that's where the
symbols got dropped.

**iOS / tvOS / watchOS caveat**: miniaudio's CoreAudio backend
pulls AVFoundation Objective-C headers (NSString etc.) which a
`.cpp` TU can't parse — would need `.mm`. So on those platforms
keep `MA_NO_DEVICE_IO` defined and stub the `crispasr_mic_*` C
ABI to return failure (so libcrispasr links). Host iOS apps do
mic capture via AVAudioEngine and feed PCM into
`crispasr_stream_feed` — they have the AVAudioSession +
privacy-prompt context the static lib wouldn't. See
`crispasr_audio.cpp:31-36` + `crispasr_mic.cpp:16-21`.

## Lesson — audio-thread → managed-language callback patterns

miniaudio's `ma_device` data callback fires from the audio
thread. Each language binding needed a different trampoline to
keep the native callback alive AND cross safely back to managed
code:

| Wrapper | Pattern |
|---|---|
| Python | `ctypes.CFUNCTYPE(...)` instance held on the `Mic` object so the ctypes thunk doesn't get GC'd. Trampoline copies the buffer via `np.ctypeslib.as_array(...).copy()` because miniaudio reuses the buffer after return. |
| Rust | Boxed `FnMut(&[f32])` closure inside a `TrampolineState` struct, passed via `*mut c_void` userdata. The Box keeps the closure alive; a `Mutex` makes audio→main-thread access safe. |
| Dart | `NativeCallable<...>.listener` — Dart's native interop pattern that automatically marshals the audio-thread call to the main isolate. Must call `trampoline.close()` in the destructor or the isolate leaks. |

Common pitfall (all three): if the user supplies a closure /
function and you don't keep a reference to the trampoline AT
LEAST as long as the device runs, the audio thread fires into
freed memory and you crash. The `Mic` handle in each binding
owns the trampoline by composition.

The audio thread also can't safely raise into managed code — a
Python exception in the callback unwinds through the ctypes
boundary into miniaudio's loop and corrupts the device. All
three trampolines catch + suppress callback exceptions, writing
to stderr instead.

### miniaudio's `MINIAUDIO_IMPLEMENTATION` in a `.cpp` TU breaks on iOS / tvOS / watchOS

Hit on the v0.5.4 release CI iOS build (commit `76bf2d5` →
`1c0e996` fix). When `MINIAUDIO_IMPLEMENTATION` is defined inside
a `.cpp` translation unit on Apple platforms, miniaudio's
CoreAudio backend pulls in `AVFoundation/AVBase.h`, which
includes `Foundation/NSObjCRuntime.h`. That header uses
Objective-C types (`NSString`, `id`, etc.) which the C++
front-end cannot parse — the compiler emits a wall of
`expected unqualified-id` and `unknown type name 'NSString'`
errors.

macOS happens to "work" anyway because the targets we exercise
in regular CI use the macOS build matrix and the same `.cpp`
hits AVFoundation just fine when `TARGET_OS_OSX` is in scope —
but iOS / tvOS / watchOS hit the issue.

Three options:

1. **Rename to `.mm`** and let it be parsed as Objective-C++.
   Cleanest semantically but requires a CMake conditional + adds
   an Objective-C++ source file to a project that's otherwise
   pure C++.
2. **Define `MA_NO_DEVICE_IO`** for iOS/tvOS/watchOS — drops the
   ma_device_* symbols on those platforms but keeps the audio
   decoder intact. We took this path because the static-lib
   build artifact for those platforms is consumed by host apps
   that handle mic capture themselves (via AVAudioEngine + the
   AVAudioSession privacy-prompt context the static lib doesn't
   own anyway). See `src/crispasr_audio.cpp` + `src/crispasr_mic.cpp`
   for the gate (`#if TARGET_OS_IOS || TARGET_OS_TV ||
   TARGET_OS_WATCH`).
3. **Exclude the file from the iOS build entirely** via CMake.
   Loses the audio decoder too — usually too aggressive.

If you ever need to enable mic capture on iOS in the static lib
itself, option 1 is the right move (rename + CMake conditional).
For "wrapper apps own mic capture" use cases, option 2 stays.

### First publish to a language registry can't be CI-only — bootstrap matters

Project-specific shape (PLAN §66) but the generic pattern
applies to every new language-binding push: **all three of
crates.io, PyPI, and pub.dev reject a CI-only first publish**
because they have no prior owner record to verify the OIDC /
token exchange against. The CI workflow you write for `v*` tag
pushes will fail every release until someone manually does the
first publish from a workstation:

- crates.io: `cargo login` + `cargo publish` (one-time, then
  `CARGO_REGISTRY_TOKEN` covers all future releases).
- PyPI: configure a "pending publisher" at
  https://pypi.org/manage/account/publishing/ pointing to the
  workflow file + environment, OR upload v0 with a manual
  `twine upload` first.
- pub.dev: `dart pub publish` is interactive (browser login) for
  the first version of any package; only after that can the
  Admin → Automated Publishing toggle take over.

**When to set up the workflow:** keep the per-tag publish flow
in the repo from day one (so you don't forget) but **comment
out the `tags: ['v*']` auto-trigger and keep
`workflow_dispatch` only** until bootstrap is done. Otherwise
every tag push produces a red CI run that obscures real
failures. Document the bootstrap procedure inline in the
workflow comment block AND in PLAN.md so the human picking it
up later doesn't have to spelunk for the per-registry magic
words.

Resilience pattern that pays off when you DO re-enable the
trigger: every job gets `continue-on-error: true` (so a
misconfiguration of one registry doesn't fail the others) and
a fast "secret/config presence check" at the top that prints a
clear `::warning::` + `exit 78` (neutral) when prerequisites
are missing — instead of letting `cargo` / `twine` emit
cryptic auth errors deep in the log.

### `HF_HOME` overrides token-file lookup — a "401 from a gated repo I just accepted" footgun

Setting `HF_HOME=/some/dir` to redirect the cache (e.g. to keep
weights off a small `~`) silently moves the **token-file lookup**
too. `huggingface_hub.snapshot_download` /
`hf_hub_download` then look for `$HF_HOME/token` instead of the
default `~/.cache/huggingface/token` — and if it's not there, they
fall back to **anonymous** access. On a gated repo that surfaces as
`GatedRepoError(401)` whose message says "must be authenticated"
even though `hf auth whoami` and `api.repo_info()` (which use a
different code path) both work fine and confirm the gate is
already accepted.

Symptom we hit on Kartoffel-Orpheus DE downloads: gate had been
accepted and `cstr` was the logged-in user; the click-through
accept page wasn't actually the issue. Diagnostic that nailed it:
a direct `requests.head` with an explicit `Authorization: Bearer
<token>` from `huggingface_hub.get_token()` returned 200 — same
URL the lib hit, different auth-loading path.

**Three fixes, by escalation:**

1. **Drop the `HF_HOME` override.** Use `cache_dir=` on the
   download call instead — `cache_dir` is per-call scope and
   doesn't affect token loading.
2. **Symlink the default token into the override.** `mkdir -p
   $HF_HOME && ln -sf ~/.cache/huggingface/token $HF_HOME/token`
   — keeps the override semantics for cache while preserving
   default token resolution.
3. **Pass the token explicitly.** `snapshot_download(...,
   token=get_token())` bypasses the env-var dance entirely. Most
   robust; least convenient.

Don't waste cycles re-clicking gate accept pages until you've
verified `requests.head` with the explicit token works. If it
does, the gate is fine and the bug is in the lib's env-var
handling. `HUGGINGFACE_HUB_CACHE` (cache-dir-only) does NOT have
the same effect — only `HF_HOME` reroutes the token. Easy to
assume both env vars behave the same; they don't.

### Gating per-token softmax preserves fast paths on large-vocab decoders (PLAN #61b/#65)

When adding `*_transcribe_with_probs` to a decoder backend, the
natural shape is to refactor the `argmax` lambda into a `pick(L,
float* out_p)` that ALSO emits the picked token's softmax
probability. The temptation is to call it unconditionally so the
new public entry shares one code path with the legacy text-only
entry. Don't.

For small-vocab models this is fine — voxtral's ~32k tokens cost
~32k `expf` per step, microseconds on M1. For large-vocab decoders
the picture changes: mimo-asr's vocab is **151,680 Qwen2 tokens**.
A single per-token softmax adds ~150k `expf` calls — about
0.5–1 ms on M1 and proportionally more on slower backends. Across
30 emitted tokens that's 15–30 ms of pure overhead, which is
measurable on the runtime path the CLI actually uses.

The fix is a one-liner gate at the call site, NOT inside `pick`:

```cpp
const bool capture_probs = (out_token_ids && out_token_probs);
int next = pick(logits, capture_probs ? &prob : nullptr);
```

`pick` then short-circuits the softmax loop when `out_p == nullptr`:

```cpp
if (out_p) {
    float s = 0.f;
    for (int i = 0; i < V; i++) s += expf(L[i] - bv);
    *out_p = (s > 0.f) ? (1.f / s) : 0.f;
}
```

The legacy path (`mimo_asr_transcribe`, called by the CLI) passes
`nullptr` out-vectors → `capture_probs` false → no softmax →
bit-identical fast behaviour. The new path
(`mimo_asr_transcribe_with_probs`, called by the session API) pays
the softmax cost only when the caller actually wants probs. Same
pattern applied to vibevoice (~152k vocab), moonshine, and
omniasr-llm. For small-vocab cases (firered ~9k, canary-ctc ~16k)
the gate is cosmetic but doesn't hurt.

### Best-of-N seed propagation: xorshift mix-in for sampler-owned RNG, `srand` for libc-rand (PLAN #61d)

Adding best-of-N to a backend with multinomial sampling requires
the N runs to draw INDEPENDENT samples. If the per-call seed is
deterministic in the audio (e.g. `seed = audio_ptr ^ n_samples`),
running the same audio N times gives N IDENTICAL samples — no
diversity, best-of-N collapses to single-shot.

Two patterns observed across the LLM-style decoder quartet:

**Pattern A — sampler owns its RNG state** (moonshine, omniasr-llm).
The sampler runs a context-local xorshift, so threading a sticky
`seed_override` field on the context is straightforward:

```cpp
uint64_t rng_state = 0x9E3779B97F4A7C15ull
    ^ (uint64_t)(uintptr_t)audio
    ^ (uint64_t)n_samples
    ^ (ctx->seed_override * 0xBF58476D1CE4E5B9ull);
if (rng_state == 0) rng_state = 0x9E3779B97F4A7C15ull;  // safety
```

The adapter injects `run_index * 0x9E37…` salt for runs 1..N-1
and 0 for run 0 (which falls back to audio-derived seed →
bit-identical to single-shot).

**Pattern B — sampler uses libc `rand()`** (glm-asr, kyutai-stt).
Their multinomial samplers are `static` helpers with no context
arg. Threading per-context xorshift through the helper signature
is invasive. Pragmatic alternative: `*_set_seed(ctx, unsigned)`
calls `srand(seed)` when non-zero, no-op otherwise. Caveat:
`srand` is **process-global** — concurrent best-of-N from
multiple threads racing through the same process would interleave
seed manipulations. Document and serialize at the adapter level
(natural pattern anyway, since best-of-N is a sequential N-run
loop in one transcribe call).

Convention used everywhere: **run 0 always uses seed 0 → sticky
override defaults → legacy single-shot bit-identical behaviour**.
Existing callers who don't pass `--best-of N` get the exact same
trajectory as before; only the explicit best-of-N path draws
different samples.

### Stub `_transcribe` masks broken session API: a runtime can be "working" via the CLI but null via the session API (PLAN #65)

When integrating a backend through the CrispASR session API, the
default move is to call its `*_transcribe(ctx, samples, n_samples)`
entry point. For most backends that works. For **qwen3 and
granite**, that entry point is a 3-line stub that returns null
(`granite_speech_transcribe`) or empty (`qwen3_asr_transcribe`) —
yet the CLI works fine because `crispasr_backend_qwen3.cpp` /
`_granite.cpp` drive the building blocks (`*_compute_mel`,
`*_run_encoder`, `*_tokenize`, `*_embed_tokens`, `*_kv_init`,
`*_run_llm_kv`, `*_token_text`) directly.

Symptom: `crispasr_session_open(...) → crispasr_session_transcribe(...)`
returns a `crispasr_session_result*` with one segment whose
`.text` is empty (or nullptr return). The session call doesn't
error; upstream wrappers see "transcription succeeded with no
text" and report nothing went wrong.

Detection: search the runtime `.cpp` for the public `_transcribe`
entry point and inspect the body. If it's:

```cpp
extern "C" char* qwen3_asr_transcribe(qwen3_asr_context*, const float*, int) {
    return strdup("");  // or: return nullptr;
}
```

…the runtime is a building-block kit, not a complete pipeline.
The session API has to replicate the CLI adapter's flow. Pattern:
read the CLI adapter (`examples/cli/crispasr_backend_*.cpp`),
port the prompt construction + decode loop into the session path
in `c_api.cpp`. Use `core_greedy_decode::run_with_probs` for the
autoregressive loop — that infra was designed for this.

Worth a `// TODO: complete` or `assert(0 && "stub")` at the top
of each stub so the next session can find them. Ideally the
stubs would be deleted in favour of a unified runtime-level
implementation (since the CLI logic is what they SHOULD do), but
that's a larger refactor.

### Replay-from-prefix beam search is `O(B × T²)` — only practical for backends with batched forward + short generations (PLAN #61h)

Adding beam search to an LLM-style decode loop without a per-backend
`*_kv_save` / `*_kv_restore` C-API leaves one option: at every
beam-step, for every beam, embed the *entire* generated suffix and
run it through `forward_fn(emb, n_tokens, n_past=prompt_len)` so
the KV slots `[prompt_len, prompt_len + suffix_len)` get overwritten
with that beam's K/V. The last-position logits are the next-step
expansion candidates. Top-K + global prune. Done.

This works perfectly on paper. The cost is `O(B × T²)` — for `T=75`
generated tokens and `B=2` you do ~5,600 token-forwards vs ~75 for
greedy, a ~75× slowdown. The PLAN #61h scoping note ("acceptable
for `beam_size = 2-4` since the prompt is the dominant cost") was
optimistic: the *audio encoder* dominates wall time, not the prompt
prefill, so the decode-loop blow-up dominates the *post-encoder*
phase and isn't amortised by anything.

What actually matters in practice is whether the backend's forward
is **batched**:

- **glm-asr's `glm_asr_run_llm_kv(emb, n_tokens, n_past)`** — one
  graph, one `ggml_backend_sched_alloc_graph`, one Metal command
  buffer per call. So `O(B × T)` *batched* calls per beam decode,
  each call doing `O(T)` Metal work. Slow but tractable on short
  audio.
- **omniasr-llm / moonshine** — per-step decode is one-token-only
  with implicit `kv_self.n`. Replay would do `O(B × T²)` *single-
  token* graph rebuilds (each one a fresh `ggml_init` + scheduler
  reset + Metal sync). On Metal the per-call overhead is ~1-2 ms,
  so 5,600 calls = ~5-10 s of pure overhead before any compute.

The lesson when scoping a beam-search effort:

1. Check the per-step decode signature first. If it's
   `(token_id, n_past) → logits` (one token at a time, implicit KV
   advance), replay-from-prefix is **not** an acceptable shortcut.
   Either ship `*_kv_save` / `*_kv_restore` for the backend, or
   defer beam search.
2. If the forward is **already batched** (`(emb, n_tokens, n_past) →
   logits`), replay-from-prefix is a fine starting point — the
   helper goes in `src/core/`, the per-backend wiring is ~30 LOC.
   Document the perf characteristic so the next person doesn't
   surprise themselves running `--beam-size 5` on a long clip.
3. The right *long-term* fix for either case is per-beam KV
   branching (a real save/restore, not a state replay). That's a
   per-backend ABI add, not a "shared helper" change. Don't conflate
   the two scopes.

Reference: `src/core/beam_decode.h` (the helper), `src/glm_asr.cpp`
beam dispatch (the batched-forward case that does work), and the
deferred sub-steps in PLAN #61h.

### Branched-KV beam search — `shared_ptr<Holder>` is the right snap-lifetime contract (PLAN #61h follow-up)

The replay-from-prefix lesson above ended on "the right long-term fix
is per-beam KV branching". Implementing that for omniasr-llm,
kyutai-stt, and moonshine surfaced one design subtlety worth its own
note: **what owns the per-beam snapshot lifetime when siblings share a
parent's snap?**

The natural shape of branched beam-search is:

```
restore(parent.snap)
logits = step_fn(parent.last_token, n_past)
new_snap = save_fn()                  // captures state right after step
for each top-K successor (≤ B):
    child = parent + (token, prob)
    child.snap = new_snap             // siblings share!
```

Multiple children of the same parent share `new_snap` — they each call
`restore(new_snap)` on their next round, then make their own fresh
snap from the post-step state. The snap itself is never mutated; it's
a frozen byte buffer. Restore = `ggml_backend_tensor_set(snap → ctx)`.

Manual ref-counting is a footgun. The first attempt tracked refcounts
manually with a `parent_ref_count` array + an "aliased?" check at each
prune step:

```cpp
for each parent: if no successor inherited its snap, snap_free_fn(parent.snap);
```

Three problems: (1) a parent selected by both a `from_finished` carry-
forward AND a normal expansion shares the snap with one survivor and
needs a fresh one for the other → tricky to express; (2) if you
accidentally `snap_free_fn` a snap a successor still references,
later restore reads freed memory and the result is silent corruption,
not a crash; (3) the prune loop ends up with an O(B²) "is this snap
aliased?" check per step.

The fix that's correct without thinking: wrap the snap in a refcounted
holder and let normal scope rules do it.

```cpp
struct Holder {
    Snap snap;
    SnapFreeFn* free_cb;
    ~Holder() { (*free_cb)(snap); }
    Holder(const Holder&) = delete;
};
auto wrap = [&snap_free_fn](Snap s) { return std::shared_ptr<Holder>(new Holder(s, &snap_free_fn)); };

struct BeamS {
    ...
    std::shared_ptr<Holder> snap;
};
```

Then "siblings share" becomes `child.snap = step_snaps[parent_idx]`
(copy-by-value of the shared_ptr — refcount bumps). When `beams =
move(next_beams)`, every old `beams[i].snap` and every unselected
`step_snaps[i]` goes out of scope and the destructor fires
`snap_free_fn` exactly when the last reference dies. No manual
ref-tracking, no aliasing checks, impossible to double-free.

This pattern is now the contract in `core_beam_decode::run_with_probs_branched`.
The four callbacks the caller provides — `save_fn`, `restore_fn`,
`snap_free_fn`, `step_fn` — do not need to know anything about
sibling sharing or refcounting; they just operate on a single snap at
a time. The helper handles the lifetime.

For backend snapshots themselves, `ggml_backend_tensor_get` on the
WHOLE per-layer (moonshine) or contiguous (omniasr-llm, kyutai-stt)
KV tensor is the right cost-vs-correctness trade. Snap sizes:
moonshine-tiny ≈ 3 MB, omniasr-llm-300m ≈ 30 MB, kyutai-stt-1b ≈
50 MB per snap. Times B beams times T steps adds up — but the
single-token decode work dominates anyway, and partial-tensor
snapshots break the moment the KV tensor's stride includes the
unwritten portion of `max_ctx`.

References: `src/core/beam_decode.h` `run_with_probs_branched`
template, `src/moonshine.cpp` / `src/omniasr.cpp` / `src/kyutai_stt.cpp`
beam paths.

## Lesson — chunked-batch is the right streaming default when the encoder is non-causal

PLAN #62c wired kyutai-stt into the unified streaming Session API.
The original ~300 LOC estimate assumed "refactor `_transcribe` into
incremental encode + decode + state carry-over" — true streaming
with sub-second latency. Pre-impl exploration surfaced a deeper
trap than the earlier "streaming in name doesn't mean streaming
in API" lesson: the Mimi encoder transformer
(`src/kyutai_stt.cpp:660`) calls
`ggml_flash_attn_ext(..., nullptr, ...)` — **fully non-causal**,
every encoder frame attends to every other including future ones.

True incremental encoding under that constraint requires either:
- **Re-encode the growing audio every chunk** — O(n²) cost over a
  session, defeats sub-second latency for anything past ~30s;
- **Replace the encoder transformer with sliding-window attention**
  (mimics moshi's `StreamingTransformer` with bounded lookback).
  ~500-700 LOC, deviates from training, fails the "≥99.9% cosine
  vs batch" gate by construction at chunk boundaries.

The chunked-batch alternative — buffer audio in `feed`, every
`step_ms` re-run the existing single-shot `_transcribe` over the
last `length_ms` of audio, replace the cumulative output — sidesteps
both. Each window is bit-exact-batch by construction; latency is
≥ `step_ms`; compute scales with window size, not session length.
This is exactly what whisper's `crispasr_stream_*` already does. It
took ~200 LOC including the dispatch wiring (struct field + four
`if (kyutai_state) return ...` branches).

**Generalisable rule.** When a "streaming" feature request lands,
look at the encoder's attention pattern *before* costing the work:

| Encoder | Streaming options | Default choice |
|---|---|---|
| Causal (left-padding only) | True incremental possible; per-conv left-context cache + per-call KV carry. | Either, depending on latency budget. |
| Non-causal full-sequence | Chunked-batch (bit-exact, simple) **or** sliding-window attention (cheap, lossy). | Chunked-batch unless a consumer has a hard sub-second SLA. |
| Bidirectional with cross-attention | Same as non-causal. | Chunked-batch. |

Chunked-batch buys ~80% of the streaming-API value (live captions,
mic dictation, push-to-talk transcription) at ~30% of the cost. The
remaining 20% (sub-second latency for very long sessions) only
matters when an actual consumer says so — and even then, the
chunked-batch wrapper is the natural adapter layer for the eventual
incremental impl, so the work isn't wasted.

Reference: `src/kyutai_stt.cpp` `kyutai_stt_stream_*`
(the implementation), `src/crispasr_c_api.cpp:387` `crispasr_stream`
struct's `kyutai_stream_state` field + the four
`if (s->kyutai_stream_state)` branches in feed/get_text/flush/close,
and PLAN #62c for the options-evaluated record.

### Beam search needs an EOS *set*, not a single EOS id (PLAN #61h)

The greedy decode helper in `src/core/greedy_decode.h` takes a single
`int eos_id` in its `Config` because every backend that ports through
it terminates on exactly one stop token. The first iteration of
`core_beam_decode::Config` mirrored this — same `int eos_id` — and
glm-asr's adapter passed `hp.eos_token_ids[0]` through.

This was wrong, in a way that didn't fail loudly: glm-asr declares
`n_eos = 3` with `eos_token_ids = {59246, 59253, 59255}` (an LLM-end
token plus two tokeniser sentinels), and the model emits any of the
three depending on prompt. The greedy path's existing `is_eos` lambda
already iterates the whole array. The beam path was checking only
the first id. So:

- ~1/3 of greedy decodes terminate on `eos_token_ids[0]` → beam works
  by accident.
- ~2/3 emit `eos_token_ids[1]` or `[2]` → beam never recognises the
  stop, runs to `max_new_tokens=512`, replay-from-prefix at 512 token
  decode steps with batched calls of length 1..512 = a multi-minute
  stall.

Symptom: beam=2 CPU smoke ran 6+ minutes at 300%+ CPU and never
emitted output. The greedy regression (`-bs 1`) ALWAYS works because
the bypass path uses the existing `is_eos` lambda. So a "greedy still
works, beam stalls" pattern is the smoke signal.

Fix: `Config` gained `std::vector<int> eos_ids` plus an `is_eos(id)`
lambda that returns true on any match (or falls back to the legacy
`eos_id` when the vector is empty, preserving callers that ported
straight from `core_greedy_decode`). Adapter passes
`hp.eos_token_ids[0..n_eos)` through. After the fix, beam=2 CPU on
JFK finishes in 129 s; beam=2/4 GPU in under 1 min; all transcripts
match greedy.

**General lesson.** Greedy-style helpers can model "stop on EOS" as a
single id because they peek at the next token before pushing. Beam
search has to push then check, and prunes globally — so a "missed"
stop token doesn't revert; the beam just keeps going. Whenever a
backend's runtime declares `n_eos > 1` (granite, glm-asr, kyutai-stt
have multi-stop sets; check `hp.eos_token_ids` arrays), the beam-
helper Config needs the *whole set*, not just `[0]`. Worth surfacing
this in any future per-backend beam wiring as a check on the adapter.

## Lesson — different audio-LLM checkpoints in the same family use different prompt templates; don't reuse one orchestrator across all of them

Hit during PLAN #7 phase 1 (voxtral4b streaming). The unified
`run_voxtral_family<Ctx>()` orchestrator in `crispasr_c_api.cpp:1483-1663`
prefills `<s>[INST][BEGIN_AUDIO]` + audio embeds + `[/INST]lang:en[TRANSCRIBE]`
and runs a plain greedy loop. This works for voxtral-3B / qwen3-asr /
granite-speech. **It does not work for Voxtral-Mini-4B-Realtime-2602.**
That model uses a completely different streaming-prompt convention:

- Audio is padded with 32 left-pad tokens of silence + right-aligned +
  10 right-pad tokens of silence (one token = `hop_length × conv_stride
  × stack_4 = 1280` samples).
- Prompt is BOS + 38 × STREAMING_PAD (id=32) = 39 tokens.
- Audio embeddings are *element-wise added* to the first 39 prompt embeddings.
- During decode, each next-token's embedding has the *next* audio
  embedding added to it before the LLM forward — the "audio-injection
  pre_hook" pattern.
- Decode stops when audio is exhausted (not on EOS — the model rarely
  emits EOS; it emits id<1000 streaming-control tokens that get filtered
  from the visible transcript).
- Without the audio-exhausted stop, the model emits trailing garbage
  ("...for your country.**ch.**") because the streaming prompt doesn't
  license post-audio decoding.

The CLI adapter at `examples/cli/crispasr_backend_voxtral4b.cpp:67-220`
implements this correctly because it was hand-written for the realtime
model. The Python `session.transcribe()` path historically went through
`run_voxtral_family` and produced a runtime crash on arbitrary audio
sizes (the [INST]...[TRANSCRIBE] template + no audio padding broke the
projector's stride-8 alignment). PLAN #7 phase 1.5 also fixed this:
the voxtral4b branch in `crispasr_session_transcribe_lang` now routes
through the streaming API (open + feed + flush + get_text), which uses
the streaming-prompt convention and the right audio padding. Same
implementation, two entry points.

**Generalisable rule.** When a new audio-LLM checkpoint shares an
encoder family with an existing backend but has a different "Realtime"
or "Streaming" suffix in its name, **read its tokenizer config and its
model card's prompt example before assuming the family orchestrator
applies**. The C-ABI dispatch can route to a per-checkpoint backend
adapter; the unified `crispasr_session_transcribe` path needs an
explicit branch for any model that breaks the family template — and
once you've built the streaming path with the right convention, the
session.transcribe path can DRY through it.

## Lesson — when an encoder is causal + SWA *and* the LLM has a KV cache, "streaming" is synchronous, not threaded

PLAN #7's original framing (May 2026 PLAN.md §7) called for "encoder
thread + decoder thread + audio ring buffer with cross-thread sync,
~200-300 LOC, high complexity." That estimate predates the PLAN #62b
adapter layer that landed in §66.

With `crispasr_session_stream_open` already giving us synchronous
`feed`/`get_text` semantics at the C ABI, and with audio-thread →
managed-language trampolines templated per the lesson at the top of
this section, the C++ side does NOT need its own threads to deliver
PTT/dictation streaming. `feed()` accumulates PCM (or runs an
incremental encoder if one is wired); `flush()` runs the prefill +
greedy decode loop synchronously; `get_text()` returns the result.
The wrapper layer handles its own audio-thread → main-thread marshaling
exactly as the PLAN #62a/d kyutai/moonshine path does.

The threading framing only earns its keep when *concurrent
encode-during-decode* is needed — i.e., when the goal is live captions
during continuous speech (PLAN #7 phase 2), not PTT. Even there, a
state-machine driven by `get_text()` polls is often enough to deliver
the user-visible behaviour without a real thread.

**Generalisable rule.** When a streaming feature request lands, before
budgeting threads, audit:
1. Is the encoder causal? (Look at the attention mask construction —
   if `mask[k > q]` is the only condition, yes; if `mask` is nullptr
   or bidirectional, no.)
2. Is the encoder bounded-context? (Look for SWA / sliding-window
   attention.)
3. Does the decoder/LLM have a persistent KV cache exposed via
   `n_past`? (Look for a `_kv_init` / `_run_llm_kv(inputs_embeds,
   n_tokens, n_past)` pair.)

If 1+2+3 are all yes, the architecture supports synchronous incremental
streaming. Threads only help if the bench shows you can't keep up with
audio rate at synchronous feed cadence. Don't pre-spend the LOC budget
on threads the architecture didn't need.

PLAN #7 phase 1 + 1.5 shipped the API + the incremental encoder, both
bit-exact-batch on JFK, ~600 LOC + ~150 LOC bench harness, no C++
threads. The remaining latency win (≤240 ms first-token) is
encoder-kernel optimisation work (Q4_K Metal kernels, larger chunks,
decoder thread overlap with next-chunk encode), not a threading-the-
whole-thing problem.

## Lesson — `core_mel::Layout::MelsTime` emits T-fast, n_mels-slow; matching ggml's mel-input convention; don't transpose on the way in

Hit during PLAN #7 phase 1.5. `core_mel::compute(Layout::MelsTime)`
returns `(n_mels, T)` row-major — per-band-row contiguous, i.e. for
band m, all T timesteps are stored sequentially before band m+1's row
starts. So `mel[m * T + t]` indexes element (band m, time t).

The encoder graph (in `voxtral4b.cpp:534`) creates the mel input as
`ggml_new_tensor_2d(F32, T_mel, n_mels)`. ggml's `ne[0]` is the FAST
axis, so this tensor has T fast, n_mels slow — exactly matching
core_mel's output. `ggml_backend_tensor_set(mel_tensor, mel_data, ...)`
just memcpys the bytes; the convention matches **only if** you treat
the data as already-(T, n_mels) layout-compatible.

Phase 1.5 streaming code originally had a "transpose on copy" loop
that wrote `mel_pending` as per-frame interleaved
(`mel_pending[t * n_mels + m]` = m fast, t slow), assuming the encoder
expected per-frame-row layout. The encoder saw mel and time axes
**swapped**: each "time step" the conv stem processed actually
contained values from many time positions of one mel band. The output
was plausible (real numbers, similar magnitudes to batch) but
content-meaningless. The LLM read the resulting audio embeddings as
noise and emitted only streaming-pad tokens (id=32, filtered to empty
transcript).

**Generalisable rule.** When wiring a streaming-mode mel feed into a
graph whose batch sibling consumes `core_mel::compute` output directly,
just memcpy the chunks; do NOT transpose. Keep all internal mel
buffers (`mel_pending`, conv left-context) in the same `(n_mels, T)`
per-band-contiguous layout. Cross-chunk concatenation requires
interleaving per-band rows (each new chunk's row m gets appended
after the previous chunks' row m, not at the end of the linear buffer)
— a ~10-line loop, not a transpose.

Diagnostic signature: for an all-zero input (e.g. the 32 left-pad
tokens of silence in voxtral4b's prompt prefix), the streaming and
batch encoder MUST produce identical outputs (cos = 1.0). Cosine
~0.99 with non-trivial element-wise differences on a zero-input chunk
is the fingerprint of an axis-orientation bug or a precision/dtype
mismatch in the K/V cache. Distinguish: a transpose preserves
sum-of-squares but scrambles per-element values; a dtype mismatch
(F16 vs F32) introduces a uniform low-magnitude noise floor without
scrambling. Both can compound across deep encoders to where the LLM
treats the embeds as silence.

## Lesson — `language="auto"` is a hand grenade for backends that embed language codes literally

Discovered May 2026 via `tools/test-all-backends.py` smoke run: canary
returned an EMPTY transcript on JFK. The diagnostic chain:

1. `whisper_params.h:71` defaults `language="auto"` (since 63d0e5d
   on 2026-04-20 — multilingual UX improvement).
2. `crispasr_run.cpp` triggers LID via whisper-tiny when
   `language=="auto"`.
3. LID calls `crispasr_cache::ensure_cached_file("ggml-tiny.bin",
   <hf-url>)` which **only checked `~/.cache/crispasr/`**.
4. That cache was empty on the test box; the download failed (network
   flaky, file already on disk elsewhere — doesn't matter why).
5. `crispasr_run.cpp:215` then left `params.language="auto"` —
   useless as a fallback because it's the very value that triggered
   LID in the first place.
6. canary's prompt builder (`canary_build_prompt`) embeds the language
   token *literally* — `<|en|>`, `<|de|>`, etc. So `language="auto"`
   becomes `<|auto|>`, which is not in the canary vocab.
7. Prompt build returns `{}`, transcribe silently emits an empty
   string. Caller has no idea why.

**Generalisable rule.** For any backend whose prompt or conditioning
embeds the language code **as a literal token** (canary, qwen3-tts,
likely future vibe variants), `"auto"` is a footgun. The contract has
to be: by the time we hit the backend, `params.language` is a real
ISO code or the backend rejects it loudly. Three places need to
enforce this together — one alone isn't enough:

| Layer | What it does | Why each layer matters |
|---|---|---|
| **Cache resolver** (`ensure_cached_file`) | Probe `$CRISPASR_MODELS_DIR`, `/Volumes/backups/ai/crispasr-models`, `~/.cache/crispasr-models`, `~/.cache/huggingface/hub` before downloading. | Prevents the LID-download-failure from triggering at all when the model is already on disk. Most users have it. |
| **LID-failed fallback** (`crispasr_run.cpp:215`) | When LID returns false AND `language=="auto"`, set to `"en"` with stderr note. | If the cache resolver still misses (genuinely no model anywhere), don't pass the literal `"auto"` downstream. |
| **Backend defensive** (`crispasr_backend_canary.cpp:65`) | If `src=="auto"` or empty still reaches us, normalize to `"en"` with stderr note. | Belt-and-suspenders against future code paths that skip LID entirely (e.g. `--lid-backend=off`). |

**Why we didn't catch it earlier.** Two reasons compound:
- The default flipped from `"en"` to `"auto"` on Apr 20. On most dev
  boxes `~/.cache/crispasr/ggml-tiny.bin` was already populated from
  prior whisper runs, so LID succeeded silently and `params.language`
  got reset to a real code before the backend saw it. The bug only
  surfaces on cleanish caches.
- The test framework prior to `tools/test-all-backends.py` was
  benchmark-shaped (perf tables + gist upload) and didn't gate on
  pass/fail per backend per capability — an EMPTY transcript was
  visible but not actionable.

**Diagnostic next time.** Empty transcript with no obvious crash →
add a stderr line in the backend that names which token / lookup
failed. The original error message was just `"canary: prompt contains
an unknown token"` — useless. Now it prints the offending string,
vocab size, and the resolved src/tgt — which immediately revealed
`<|auto|>`. Costs ~10 LOC, saves an hour of forensics.

Reference: `src/crispasr_cache.cpp::well_known_search_dirs`,
`examples/cli/crispasr_run.cpp:215` (LID-failed branch),
`examples/cli/crispasr_backend_canary.cpp:65` (defensive layer),
`src/canary.cpp:944` (improved error message), commit `f2d5ecb`.

## Lesson — Q4_K is too lossy as the default for CTC-decoded ASR

Discovered May 2026 via `tools/test-all-backends.py` smoke run:
omniasr-ctc-1b-v2-q4_k.gguf produced "fello americas as for your
coutry" (single-char drops on multiple words, 22.7% WER on JFK).
Three-way comparison clinched it:

| Reference | Output | WER |
|---|---|---|
| Python `transformers` (FP32) | "and so my fellow americans ask not …" | 0% (ground truth) |
| Our **Q8_0 GGUF** | identical to Python ref, byte-perfect | 0% |
| Our **Q4_K GGUF** | "and so my fello americas as … or your coutry" | 22.7% |

**Not a runtime / conversion bug.** Q8_0 proves the encoder, CTC
decoder, audio frontend, and ggml ops are all correct.

**Per-tensor sensitivity (all 289 Q4_K-quantized tensors against
Q8_0 ground truth, dequantized to F32):**

  median cosine: 0.99743
  min cosine:    0.99667  (enc.0.attn.out.weight)
  median relRMS: ~0.075   (~7.5 % relative drift per matmul)

The drift is **uniform** across the 48 transformer layers — there's
no single tensor to "promote" to higher precision. Layer 0 has the
worst loss (errors compound through subsequent layers) but every
encoder layer drifts in the same range. Mixed quant won't help
unless you bump the entire encoder up — which is essentially Q8_0.

**Why CTC suffers when seq2seq doesn't.** A CTC head does an argmax
per encoder frame; small logit drift can flip frames at boundaries
between a real character and the blank token, manifesting as
*missing characters* (not substitutions). Seq2seq with internal LM
smoothing (BPE + autoregressive decoder) absorbs the same drift —
the decoder's language model "rescues" borderline frames.

**Generalisable rule:**

| Backend type | Default quantization safe at |
|---|---|
| CTC-decoded (no internal LM) | **Q8_0** — Q4_K is too lossy |
| Seq2seq encoder-decoder (BPE + AR decoder) | Q4_K usually fine |
| LLM-style audio (qwen3, granite, voxtral) | Q4_K usually fine |

**Action taken.** Switched omniasr-ctc shipping default to Q8_0 in
`tools/test-all-backends.py` registry. Q4_K stays available on HF
(551 MB vs 1.0 GB) but is no longer the smoke-test default.

**Diagnostic recipe for next CTC quant question.** Run the model
through Python `transformers` to establish FP32 ground truth, then
A/B Q8_0 vs Q4_K via `tools/test-all-backends.py --backends X
--models <dir>`. If Q8_0 == ref and Q4_K diverges, it's quant
sensitivity, not a code regression — don't bisect git history.

Reference: per-tensor analysis script `/tmp/q4k_sensitivity.py`
(local, one-off — uses `gguf.dequantize` to compare both quants
vs each other). Commit `6df9129`.

## Lesson — runtime QKV/MLP fusion on row-wise quantized weights is just byte-concat

PLAN #7 phase 2. The qwen3_asr fused-QKV path (`qwen3_asr.cpp:1430`)
gates its runtime fuse on `attn_q_w->type == GGML_TYPE_F32 ||
GGML_TYPE_F16` — assuming concatenation along the output axis only
makes sense for floating-point types. **That gate is overly
conservative for row-wise quantized formats.**

Q4_K, Q4_0, Q5_K, Q8_0 etc. all encode each *row* (along the output
dimension) as a self-contained block group: scales + minimums + 4-bit
or 8-bit packed values. Concatenating two Q4_K tensors of shape
`(d_in, n_out_a)` and `(d_in, n_out_b)` along the output axis is the
**same operation** as for F32 — append the bytes. No requantization,
no rebalancing, no rebake.

`voxtral4b.cpp` LLM fused-QKV builds `(d_model=3072, q+2*kv=6144)`
from three separate `(3072, 4096|1024|1024)` Q4_K tensors purely via
`ggml_backend_tensor_get` + `set` byte-copies. Adding the type-gate
extension was a 1-line change relative to the qwen3_asr precedent and
delivered ~7-8 % decode speedup on M1 (56 → 50.4 ms/step on
voxtral4b LLM Q4_K).

**Generalisable rule.** When you see a fused-QKV (or fused gate+up)
runtime path gated only on F16/F32, that's an artifact of the
original author's caution, not a real correctness boundary for
Q-formats. Verify by checking that `ggml_new_tensor_2d` accepts the
quantized type at the fused output shape, then drop the gate.

**Cross-backend portability — DONE May 2026.** Same gate-drop landed
in `qwen3_asr.cpp:1433` (default-on) and `qwen3_tts.cpp:4986` (kept
opt-in). voxtral 3B got the full runtime fuse (~80 LOC mirror of
voxtral4b), but **opt-in only** (`CRISPASR_VOXTRAL_FUSED_QKV=1`)
because A/B on JFK Q4_K showed no measurable speedup — the
voxtral4b 7-8 % win came from its long decode loop (141 tokens
incl. streaming-pad warmup); voxtral 3B's ~30-token normal-prompt
decode amortises the saved kernel-launches into run-to-run noise.
**Lesson learned: fuse value scales with `decode-loop-length ×
saved-kernel-launches × per-launch-overhead`** — short decodes
don't amortise well. Same backend, different prompt convention
→ different ROI.

**One subtle gotcha** when porting the qwen3_asr pattern: the
existing implementation allocates the fused buffer on the CPU
backend (`ggml_backend_cpu_buffer_type()`) — fine for F16/F32 since
the matmul kernel handles cross-backend transfer cheaply, but
catastrophic for Q-formats because every matmul would pay a
Metal→CPU transfer cost. Switch to
`ggml_backend_get_default_buffer_type(ctx->backend)` when extending
the gate. (voxtral4b's implementation already does this.)

## Lesson — not every matmul fusion is a win on Metal Q4_K — measure ROI in saved kernel launches, not saved input reads

Same phase 2 pass, FFN gate+up fuse: combine the per-layer
`(3072, 9216)` gate and up matmuls into a single `(3072, 18432)`
matmul, then split via `ggml_view_2d`. Saves 1 matmul/layer × 26
layers = 26 fewer matmul ops/decode-step. Predicted ~4 % speedup
based on the QKV fuse precedent (which saved 2 matmuls/layer × 26 =
52 ops for ~7-8 %).

**Measured: zero speedup** (per-decode-step 50.4 ms unchanged within
run-to-run noise; prefill marginally slower). Rolled back the fuse
runtime path; left the `core_ffn::swiglu_fused_gate_up` helper in
place for any future caller.

Reason: Metal's Q4_K matmul kernel for the gate/up shape
`(3072, 9216)` is already memory-bandwidth-bound — the kernel reads
~21 MB of Q4_K weights per matmul, computes a 1-vector × 9216-output
matmul, and writes 9216 floats. Reducing two reads of the same
3072-float input vector into one was below the noise floor. The
QKV-fuse win came from saving *two* full Metal kernel launches
(higher fixed-cost amortisation), not from saving the small input
re-read.

**Generalisable rule.** Matmul-fusion ROI is roughly proportional to
`(saved_kernel_launches × per-kernel-overhead)`, not to
`saved_input_reads × input_bytes`. On Metal at 4B-param scale:
per-kernel-overhead ≈ 10-30 µs, the saved input re-read is ≈ free
when input fits in L2. So:

- **Fusing 3-into-1** saves 2 kernel-launches and 2 small input
  re-reads → measurable win.
- **Fusing 2-into-1** saves 1 kernel-launch and 1 small input
  re-read → below noise.

Don't fuse "because you can"; measure ROI first.

The `core_ffn::swiglu_fused_gate_up` helper stays in `core/ffn.h` for
any future caller — a different backend (CPU, CUDA), a much larger
model where matmul time dwarfs launch overhead, or a smaller-quant
weight where the input re-read becomes proportionally significant
might still benefit.

## Lesson — Metal's single GPU queue makes "decoder thread" mostly an API affordance, not a perf win, on Apple-Silicon

PLAN #7 phase 4 added a decoder worker thread to voxtral4b streaming.
The naïve hope: encoder runs on main thread, decoder runs on worker
thread, total wall-clock = max(encoder, decoder) instead of sum.
**Measured on M1: zero wall-clock improvement.** PTT/LIVE-single-
thread/LIVE+decoder-thread all land within ±100 ms on JFK 11 s.

Why: Metal exposes one GPU command queue per device. ggml's Metal
backend submits compute kernels into that queue via command buffers.
Two threads submitting via separate `ggml_backend_sched_compute`
calls still funnel into the same queue, which executes them in
submission order. There's no kernel-level parallelism on M1's
8-core GPU — concurrent compute submissions serialize.

**The architecture is still worth shipping** because:

1. **Mic-driven workloads.** When the audio source is paced (a real
   microphone delivering ~80 ms chunks at audio rate), the decoder
   thread drains decode during the natural gaps between feeds. The
   main thread can return from feed() without waiting for the entire
   decode loop, keeping the audio queue from backing up.

2. **Faster GPUs.** M3 Ultra (76-core GPU), NVIDIA, AMD — these have
   real kernel-level parallelism. Two compute submissions to
   different command buffers actually run concurrently across SIMD-
   groups. The same code that's a no-op on M1 becomes a
   30-40 % wall-clock win on M3 Ultra (estimated).

3. **API ergonomics.** Even on M1, callers get a non-blocking feed:
   `stream.feed(pcm); return` instead of `stream.feed(pcm); wait
   for decode; return`. For UI-driven applications (display
   progressive captions in a TextField), this matters even when
   the underlying compute is serialized.

**Generalisable rule.** Threading a compute pipeline only delivers
parallel-execution wins when the underlying device exposes that
parallelism. On a single-queue GPU, threads buy you API smoothness
(non-blocking returns, separate concerns) but not throughput. Don't
expect throughput wins until you've measured the device's
multi-stream capacity.

For dual-sched true parallelism on devices that support it: PLAN #7
phase 5 (deferred). Two `ggml_backend_sched` instances on the same
backend, each with its own graph allocation buffer. Each submits to
the Metal queue independently; on a multi-stream GPU they overlap.
~150 LOC + per-call routing in voxtral4b_run_llm_kv. Defer until a
consumer measures actual M3 Ultra (or NVIDIA) gain that justifies
the LOC.

## Lesson — audio-injection prompt models give live captions for free; encoder-decoder models don't

PLAN #7 phase 3 shipped voxtral4b live captions in ~150 LOC with no
stable-prefix-commit heuristic. The two-line summary: voxtral4b
*injects* each audio embedding into the next-token's input embedding
during decode, so each emitted token is a deterministic function of
audio context [0..N] only. Token N never depends on audio[N+1..],
so tokens are *immediately commit-safe* the moment they're decoded.

This is the architectural property that distinguishes the two
classes of streaming ASR:

| Class | Examples | Token stability |
|---|---|---|
| **Audio-injection prompt LLM** | voxtral4b, future MoE-of-streaming-LLMs | Each token depends only on past audio. **Tokens commit immediately, no retraction.** |
| **Encoder-decoder ASR** | whisper, parakeet, canary, qwen3-asr (cross-attention path) | Encoder's bidirectional context shifts as more audio arrives. **Mid-stream tokens may be revised.** |

The class on the left is fundamentally simpler for live captions —
no stability heuristic, no retraction logic, no "stable prefix"
commit window. The class on the right requires either chunked-batch
(re-decode-the-whole-window-on-each-step, what PLAN #62c shipped) or
a real stable-prefix algorithm (longest common prefix across N
consecutive decode trials, with retraction when it breaks).

**Generalisable rule.** Before designing live-caption infra for a
new audio-LLM, check the prompt convention: does it inject audio
*per-decode-step* (via a pre_hook adding audio embeds to the next
token's embedding before LLM forward), or does it splice audio into
a static prompt and decode against that frozen context? Per-step
injection → tokens commit immediately. Static splice + cross-
attention → stable-prefix needed.

**Implementation gotcha** (caught during phase 3): when extracting
the decode loop into a helper that's called multiple times during
feed (live mode), the natural "argmax at top of loop, forward at
bottom" pattern emits each token TWICE — once at the bottom of
drain-call N, once at the top of drain-call N+1 (which re-argmaxes
the same logits). Fix: track a `decode_logits_committed` flag on
the stream. The 3-step state machine (argmax+emit → stop-check →
inject+forward) with the flag set after step 1 and cleared after
step 3 ensures one emission per logits.

Reference: `src/voxtral4b.cpp` `vox_stream_drain_decode`, PLAN #7
phase 3, HISTORY §71 final paragraph.

## Lesson — per-stage timing instrumentation reveals architectural latency floors that kernel optimisation can't break

PLAN #7 phase 2 set out to hit a ≤240 ms first-text-token target on
voxtral4b streaming. After 240 ms chunks (2.6× feed speedup),
default-unification (~1 s flush save), and runtime fused QKV (~8 %
decode speedup) — first-text-token went 2.7 s → 1.6 s. Still ≥ 4×
the target.

`CRISPASR_VOXTRAL4B_STREAM_TIMING=1` instrumentation revealed the
breakdown (M1 Q4_K JFK 11 s):

| Stage | ms |
|---|---|
| Encoder drain at flush (right-pad encoder + projector) | ~990 |
| LLM prefill (39-token streaming-prompt) | 252 |
| 8 streaming-pad warmup steps × 50.4 ms each | ~403 |
| First text-emitting decode step | ~50 |

Even with a *fully warm encoder* (drain → 0), the prompt convention
forces ≥ prefill (252 ms) + 8 × decode (403 ms) = **655 ms before the
first text token can emit**. The 8-step streaming-pad warmup is part
of the model's training distribution — the model emits id=32
STREAMING_PAD for the first 8 decode steps before producing text, no
matter how much audio context is available.

**Generalisable rule.** "X ms latency" claims in audio-LLM model
cards can be misleading on a different inference stack. Read the
prompt convention and per-token decode cost FIRST, multiply, and
compare to the target *before* costing perf work. If the
architectural floor exceeds the target, kernel optimisation can only
get you so close — the rest requires either a different prompt
template (model retrain) or a fundamentally different decoder
(different quant, different architecture, different backend).

For voxtral4b-realtime, the model card claims ~97 ms end-to-end
latency — likely measured on a specific Mistral inference stack with
some combination of (much faster Q4_K matmul, smaller warmup, or
parallel decode). A community ggml implementation on M1 Q4_K Metal
at 50.4 ms/step is fundamentally bounded above the marketing number.
Saying so explicitly in `PLAN.md` / `HISTORY.md` saves the next
perf-pass engineer a multi-day chase.

The instrumentation cost was ~30 LOC and the diagnostic value was
~3 days of avoided guessing. Reference: `PLAN.md` §7 phase 2,
`HISTORY.md` §71, `CRISPASR_VOXTRAL4B_STREAM_TIMING=1` env var.

## Lesson — for CTC-decoded ASR, mixed Q4_K head-skip recovers nearly all Q8_0 quality

Follow-up to the previous lesson ("Q4_K is too lossy as the default
for CTC-decoded ASR"). Initial conclusion was "ship Q8_0" because
uniform Q4_K dropped to 22.7% WER on JFK. Per-tensor weight cosine
analysis (`/tmp/q4k_sensitivity.py`) showed all 289 Q4_K tensors
drift uniformly (~0.997 cosine vs Q8_0). No single tensor is the
culprit.

Per-layer **activation** dump (via `OMNIASR_DUMP_DIR=/tmp/q4k-dump
crispasr ...` on Q4_K vs Q8_0) revealed the real signal:

| Stage | cos(Q4K, Q8) | drift |
|---|---|---|
| pcm_norm → pos_conv_out | 1.000 | 0% — frontend untouched (already F16/F32) |
| enc_layer_0 → 32 | 0.998 → 0.991 | accumulates linearly through residual stream |
| **enc_layer_36 → 47** | **0.988 → 0.982** | sharp acceleration |
| logits | 0.9995 | one-frame argmax flips → CTC drops chars |

**The wrong intervention.** Keeping the *last* 12 layers at F16
(`CRISPASR_OMNIASR_KEEP_F16_TAIL=12`) made the output **worse** than
uniform Q4_K — "fello amercas ... cuty" vs uniform's "fello americas
... coutry". F16 math in the tail preserves the accumulated upstream
noise more faithfully than Q4_K math does (Q4_K's quantization
rounding sometimes happens to land back near the right bin).

**The right intervention.** Keep the *first* 4 layers at F16
(`CRISPASR_OMNIASR_KEEP_F16_HEAD=4`). Result:

| Recipe | Size | WER | Output |
|---|---|---|---|
| Q4_K uniform | 551 MB | 22.7% | "fello americas as ... coutry" |
| **Q4_K + head=4** | **658 MB** | **5%** | "fellow americas ... country" — 1-word diff |
| Q4_K + head=8/12/16 | 766–982 MB | 5% | identical to head=4 |
| Q8_0 | 1007 MB | 0% | byte-perfect vs FP32 reference |

The residual "americans → americas" diff is the same as omniasr-LLM
shows on the same audio — likely an aadel4-fine-tune artifact
(model-internal), not quant-related. Promoting more layers to F16
doesn't fix it.

**Generalisable rule for residual-stream architectures.** Quantization
errors compound through the residual stream of a transformer
encoder. Where they manifest (late layers) is *not* where to fix them
(early layers). The intervention has to be at the *entry point*:

| Action | Effect on noise |
|---|---|
| Keep early layers at F16 | Stops noise from entering the residual stream — late layers run Q4_K math on clean signal and the rounding stays bounded |
| Keep late layers at F16 | F16 math preserves accumulated noise faithfully — strictly worse than uniform Q4_K |
| Uniform Q4_K | Quantization rounding occasionally cancels accumulated drift — better than tail-skip alone |

For ASR specifically: head=4 is the smallest cutoff that prevents
the compounding cascade in our 48-layer omniasr-ctc encoder. Exact
N likely depends on encoder depth and residual-stream variance —
expect a similar small constant for other CTC encoders in the
1B+ class.

**Action taken.** `examples/crispasr-quantize/main.cpp` now defaults
`omniasr_keep_head = 4` for omniasr-ctc-detected GGUFs. Override
via `CRISPASR_OMNIASR_KEEP_F16_HEAD=N` (N=0 → uniform Q4_K, useful
for size-vs-quality trade-off study). `CRISPASR_OMNIASR_QUANT_ALL=1`
opts out entirely (matching the granite-mini precedent). The test
registry switches back to `omniasr-ctc-1b-v2-q4_k.gguf` (now meaning
the head=4 mixed variant) at 658 MB.

Reference: `examples/crispasr-quantize/main.cpp` (`omniasr_keep_head`
default), `tools/test-all-backends.py` REGISTRY entry,
`OMNIASR_DUMP_DIR` env (already in `src/omniasr.cpp`).

## Lesson — design test infrastructure for both bandwidth-limited and disk-limited environments

The first version of `tools/test-all-backends.py` had one cache
strategy — download to `--models` dir, leave the file there. Worked
fine on a dev box with a 2 TB SSD. Failed instantly on Kaggle (~20 GB
disk, but bandwidth from HF is 100+ MB/s) where the registry totals
~15 GB across 17 backends and the cumulative downloads quickly fill
the volume.

The pattern that scales to both:

1. **Two named modes**, not a continuous knob:
   - `keep` — the local-dev default. Bandwidth is the bottleneck;
     we want to never re-download.
   - `ephemeral` — the cloud-runner default. Disk is the
     bottleneck; we want to never carry more than one big model
     at a time.

2. **Track downloaded-this-run vs pre-existing.** If a file was
   already on disk when the run started, we never delete it — that
   blob may be the user's curated cache. The implementation returns
   a `FetchedModel(path, downloaded_files: list[Path])` from the
   fetch helper; cleanup only iterates `downloaded_files`.

3. **Pin "reference" models.** The matrix has a small handful of
   models that get cross-used: `whisper-tiny` (every backend triggers
   LID through it when `language="auto"`), `parakeet` (TTS-roundtrip
   capability runners use it as ASR ground truth). Mark them
   `is_reference=True` in the registry — cleanup is a no-op for
   them. Without this, ephemeral mode would download whisper-tiny
   17 times (or fail when the second TTS backend can't find
   parakeet to do its roundtrip).

4. **Cleanup in `try/finally`.** A tier runner that raises must NOT
   leak the multi-GB download. Especially on Kaggle where a single
   leaked blob can fill the volume and brick subsequent backends.

5. **Pre-download disk-space check.** Each backend declares
   `approx_size_mb`; before downloading, check
   `shutil.disk_usage(dir).free` against `approx_size_mb + 2 GB
   margin`. Half-filling the disk and then erroring is worse than
   reporting `SKIP — need X, only Y free`.

**Generalisable rule.** Test infrastructure should ship with two
operational modes corresponding to the two common bottleneck
profiles. A flag, not a heuristic — the user knows whether they're
on a 2 TB SSD or a 20 GB Kaggle volume; let them say so. The
mistake is shipping with one mode and "fixing" the other when it
breaks; that creates churn the day you actually need to run on the
opposite profile.

Reference: `tools/test-all-backends.py` `FetchedModel` /
`cleanup_downloaded` / `Backend.is_reference`, commit `0c90209`.

## Lesson — five Windows release-pipeline bugs masked each other for 5 days

v0.5.4's release page never published. Every retry failed at a
different layer; fixing one bug exposed the next. Five fixes in
total to get v0.5.5 + v0.5.4 (backfilled) shipping clean. All five
were Windows MSVC or Docker specific — Linux + macOS sailed through
the whole time, which is why the failure went undetected so long.

### The cascade

| # | Bug | Manifested as | Fix |
|---|-----|---------------|------|
| 1 | `release.yml` built `crispasr` (library) target, not `crispasr-cli` (binary). Renamed silently in `examples/cli/CMakeLists.txt:12 set(TARGET crispasr-cli)` with `OUTPUT_NAME=crispasr` keeping the binary name unchanged. | Linux/macOS/Windows Package step: `cp build/bin/crispasr` → "No such file or directory". | `4d4eef9`: change all 6 `--target crispasr` → `--target crispasr-cli`. |
| 2 | `cmake/git-vars.cmake` used `git log --date=local` → `Sat May 2 22:41:36 2026 +0200`. Spaces + English month names. | Windows MSVC link.exe re-parsed the `-DCRISPASR_GIT_DATE` value when expanding command files; "May" became `May.obj` → `LNK1181: cannot open input file 'May.obj'`. | `5771566`: switch to `--date=iso-strict` + defensive `string(REPLACE " " "_")`. |
| 3 | `CMAKE_CUDA_ARCHITECTURES="75-real;80-real;..."` propagated as a CMake LIST into `target_compile_definitions` → cl.exe got it as separate `-D` args (`-D80-real -D86-real -D120-virtual"`) with one orphan quote. | Mangled cl.exe line bled into link.exe's argv; the FIRST loose token from the SUBJECT define ("GIT_DATE = ISO-8601") became `GIT_DATE.obj` → `LNK1181: cannot open input file 'GIT_DATE.obj'`. | `cf24ef0`: `string(REPLACE ";" "," ...)` before embedding in the `-D` define + extend GIT_COMMIT_SUBJECT sanitiser to neutralise `=`. |
| 4 | `crispasr_diff_main.cpp:944` used `M_PI`. MSVC's `<cmath>` doesn't define it without `_USE_MATH_DEFINES` set BEFORE the first include. | windows-blas matrix in `build.yml`: `error C2065: 'M_PI': undeclared identifier`. (Different workflow from release.yml; symptomatic side-finding.) | `e6e8fcd`: `#define _USE_MATH_DEFINES` + `#include <cmath>` at top of crispasr_diff_main.cpp. |
| 5 | `libggml-cuda.so` links against `libcuda.so.1` (CUDA driver). Docker base images ship the stub `libcuda.so` (no `.1` suffix); `.1` gets mounted in by host's nvidia container runtime at exec time, not build time. | Both cuda Dockerfiles: `cuMem* / cuDevice*` undefined references at the final crispasr exe link. | `b288283`: `-DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"` for runtime-resolved transitively-undefined symbols. |

### Common pattern

Four of the five (1, 2, 3, 5) are **toolchain-string-injection** bugs:
something user-controlled (a target name, a date, a CUDA arch list,
a commit subject) gets embedded into a build command line, then a
later layer (link.exe, ld, the second-pass arg parser) re-parses
that line on different rules than the first. Anywhere you put a
*value* into a *flag* there's a quoting boundary, and on Windows the
boundary multiplies — cmd.exe, vcvars64.bat, ninja, cl.exe, link.exe
each parse arguments slightly differently.

### Generalisable rules

1. **Sanitise everything that lands in `target_compile_definitions`.**
   The `git-vars.cmake` SUBJECT block already neutralised `;`, `\`, `"`,
   `(`, `)`, `:` — but missed `=` and (via the DATE) spaces +
   localised month names. The defensive list to ALWAYS strip from
   any toolchain-injected string:

   | char | breaks | replace with |
   |------|--------|--------------|
   | `;` | CMake list separator | `,` |
   | space | shell word boundary | `_` |
   | `=` | linker positional split | `-` |
   | `(` `)` | MSVC `/D` and POSIX sh subshell | `[` `]` |
   | `:` | target separator (some shells) | `-` |
   | `\` | escape character | `/` |
   | `"` | quote nesting | `'` |
   | localised words | non-ASCII shell parsers | `_`-prefix or skip |

2. **CMake list variables NEED explicit `;` → `,` before embedding.**
   `target_compile_definitions(... "${CMAKE_CUDA_ARCHITECTURES}")` is
   wrong — CMake lists propagate into the build command line as
   separate arguments. Always `string(REPLACE ";" "," X "${X}")`
   before the `-D` boundary if the value happens to be a list type.

3. **Test what `release.yml` builds before tagging.** The `--target`
   bug had been silently latent since the binary rename. A `make
   release-dry-run` Makefile target that runs the same `cmake --build
   ... --target X` invocation as release.yml would have caught it
   immediately. (Open follow-up: add such a target.)

4. **Docker linker flags for runtime-mounted libs.**
   `-Wl,--allow-shlib-undefined` is the correct posture for any
   driver/runtime lib (CUDA, ROCm, OpenCL) that the host injects at
   container exec time. The stubs dir resolves at build time but
   doesn't carry `SONAME`-suffixed names; the flag tells ld to trust
   that `libfoo.so.1` will appear later.

### Why it took five days

v0.5.4 shipped on April 27. Six retries between then and the
backfill on May 3. Each retry surfaced a different bug because:

- The matrix is large (Linux × macOS × Windows × CPU/CUDA/Vulkan/MSYS2
  + Docker × main/cuda/cuda-12/vulkan/musa/intel = ~16 build paths).
- Failure messages were misleading. "May.obj" sounds like a missing
  file; "GIT_DATE.obj" sounds like a CMake variable issue. Neither
  hint at the actual quoting boundary that's being violated.
- Some bugs only fire on commits whose date or message contains
  specific shell metacharacters. The "May.obj" bug was dormant in
  April (when month abbreviation was "Apr" — link.exe happened to
  not error on `Apr.obj` for reasons I haven't traced). The
  `GIT_DATE.obj` bug only fired because my own commit message
  happened to contain "GIT_DATE = ISO-8601 strict".

The lesson on the human side: when a release pipeline breaks, the
first instinct is to suspect a recent code change. In this case the
breakage was 5 latent bugs, none of which was triggered by a single
change. Always look at *what's different about the inputs to the
pipeline* (commit messages, dates, tag names) — those are the values
that actually cross the toolchain-string-injection boundary.

References: `cmake/git-vars.cmake`, `examples/cli/CMakeLists.txt`,
`examples/cli/crispasr_diff_main.cpp`, `.devops/main-cuda*.Dockerfile`,
`.github/workflows/release.yml`. Commits: `4d4eef9`, `5771566`,
`cf24ef0`, `e6e8fcd`, `b288283`. v0.5.5 first clean release; v0.5.4
backfilled at the v0.5.5 commit.

## Lesson — clang-format MAJOR-version drift silently breaks CI; pin it via a wrapper that refuses other versions

clang-format keeps adding heuristics for line-wrapping, brace-placement,
parameter-pack alignment, etc. Major versions ship with different
defaults. Running v22 on a project formatted with v18 produces ~80
violations across files that "look formatted" — the format passes
locally, fails the lint job's `clang-format-18 --dry-run --Werror`.

**Concrete cost paid in this repo:** a session's worth of work landed
through CI red because the developer's `PATH` had `/opt/homebrew/bin/
clang-format` (v22 from Homebrew's default keg / Xcode's bundle)
ahead of any v18. Required a follow-up commit to re-format every
file touched in the session through `clang-format-18` to clear lint.

**The fix is a wrapper script that refuses non-v18:** `tools/format.sh`
locates clang-format-18 at three known paths
(`/opt/homebrew/opt/llvm@18/bin/clang-format`, `clang-format-18` on
PATH, or any `clang-format` whose `--version` reports v18) and exits
non-zero with a remediation message if none is present. The script's
file scope mirrors `lint.yml` exactly, so local `./tools/format.sh
--check` is bit-equivalent to the CI step.

**Generalisable rule.** For any CI step that pins a tool version
(clang-format, ruff, mypy, prettier, rustfmt …), there's a 1:1 risk
that a developer's bare-PATH binary is a different version. The
defensive pattern: a project-owned wrapper script that asserts the
expected version and exits with a remediation message when wrong.
The wrapper costs ~80 LOC; saves a CI-red-fix-amend-push cycle the
first time it bites.

**Why not just remove the bad version from the system?** Because
`clang-format` (v22 in this case) is also the binary that Xcode /
Homebrew installs as a hard dependency for other dev tools, IDE
integrations, and unrelated projects. Deleting it ripples beyond
the one repo. The wrapper-and-document approach is the lowest-risk
fix that doesn't externalise costs to the rest of the dev environment.

References: `tools/format.sh`, `.github/workflows/lint.yml:62`
(the `apt install clang-format-18` pin), `CLAUDE.md` "clang-format
MUST be v18" section, README.md "Adding a new backend" heads-up.
Commits `7ab5dab` (the 80-violation reformat) + `5711931` (the
follow-up cppcheck fix once v18-format stopped masking it).
