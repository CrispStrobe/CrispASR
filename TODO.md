# CrispASR — TODO

Live tracker of pending work across the unified `crispasr` binary and the
shared `src/core/` infrastructure. Items marked **[next]** are the current
session's immediate targets; **[later]** are queued; **[upstream]** are
blocked on external fixes (tracked in detail in `UPSTREAM.md`).

Historical milestones and the per-model port plans are in `HISTORY.md`.
Technical deep-dives (optimisation notes, RoPE lessons, benchmark tables)
are in `LEARNINGS.md`.

---

## `src/core/` shared helpers — current state

| Helper | File | Consumers | Status |
|---|---|---|---|
| **mel spectrogram** | `mel.{h,cpp}` | 8/8 non-whisper | ✅ done |
| **FastConformer encoder** | `fastconformer.h` | parakeet, canary, canary_ctc, stt_en_fc_ctc | ✅ done |
| **LLM self-attention** | `attention.h` | voxtral (encoder + LLM), voxtral4b/qwen3/granite (KV-cached) | ✅ done |
| **FFN (SwiGLU/SiLU)** | `ffn.h` | qwen3, voxtral, voxtral4b, granite | ✅ done |
| **GGUF loader** | `gguf_loader.{h,cpp}` | all 8 non-whisper | ✅ done |
| **BPE encoder** | `bpe.h` | qwen3, granite | ✅ done |
| **Greedy decode** | `greedy_decode.h` | voxtral, voxtral4b, qwen3, granite | ✅ done |

Remaining extraction opportunities (each saves ~30-60 LOC but has only 1-2 consumers):

- **[done]** ~~KV-cached attention variant for qwen3/voxtral4b/granite LLM~~ —
  all 4 LLM backends migrated to `core_attn::kv_self_attn()`.
- **[done]** ~~Q/K norm variant for qwen3~~ — `kv_self_attn()` accepts
  optional `q_norm_w`/`k_norm_w` params.
- **[done]** ~~Whisper-style audio encoder (voxtral 3B)~~ — migrated to
  `core_attn::encoder_self_attn()` with biased Q/V/O, no K bias, no RoPE.
- **[done]** ~~voxtral4b encoder attention migration~~ — migrated to
  `core_attn::encoder_self_attn()` with `permute_cont = false` (matching
  original no-cont graph structure). Bit-identical on jfk.wav verified.
- **[later]** Sliding-window attention (voxtral4b encoder — single consumer)
- **[later]** µP scale tricks for granite (attention_multiplier, residual_multiplier)

- **[done]** ~~**`src/core/attention.h` — voxtral audio encoder.**~~
  `encoder_self_attn()` added: optional biases on Q/K/V/O, optional RoPE,
  optional mask. Voxtral 3B migrated. Voxtral4b is a candidate but uses
  no-cont permute (needs bit-identity verification with model files).

- **[later]** **`src/core/attention.h` — sliding-window attention.**
  voxtral4b audio encoder uses 750-token SWA. Needs a `sliding_window`
  knob and the mask has to be pre-built by the caller (or constructed
  by the helper from `sliding_window`).

- **[later]** **`src/core/attention.h` — µP scale tricks.**
  Granite uses `attention_multiplier` (0.0078125 = 1/128) as the attention
  scale instead of `1/sqrt(d)` and `residual_multiplier` (0.22) on the
  residual add. Parameterise via `Config::attn_scale` and
  `Config::residual_scale` in the helper.

- **[done]** ~~`src/core/greedy_decode.h` — unified LLM decode loop.~~
  **Done.** Header-only `core_greedy_decode::run()` with an optional
  `PreHook` callback (used by voxtral4b's streaming audio-frame-addition
  path). All 4 LLM backends migrated bit-identically: voxtral (via the
  crispasr_llm_pipeline.h template), voxtral4b (with the pre-hook for
  streaming), qwen3 (with dynamic EOS lookup via tokenize), granite
  (with EOS-filter post-step). Net: ~100 lines of duplicated decode
  loops replaced with a single templated helper.

- **[done]** ~~`src/core/mel::Params::stacked_frames`.~~ **Done
  differently.** Granite is now on `core_mel::compute` without a new
  Params knob: its normalization `v/4+1` turned out to be identical
  to `GlobalClipMax` `(v+4)/4`, and the "drop-last-if-odd + stack
  pairs into 160-mel rows" step stays in the granite wrapper as
  ~15 lines of post-processing. `core_mel` coverage is now 8/8
  non-whisper models.

- **[done]** ~~**`cli.cpp` output writer refactor (task #4).**~~
  Done in a9365d8. All whisper output writers (txt/vtt/srt/csv/lrc/
  score/json/wts) now take `const std::vector<crispasr_segment> &`
  produced once by `cli_whisper_collect_segments(ctx)`. Byte-identical
  regression verified on all 7 output formats.

- **[done]** ~~**`backend-whisper.cpp` wrapper (task #15).**~~
  Done in e120103. `crispasr_backend_whisper.cpp` implements the
  CrispasrBackend interface on top of whisper.h. `--backend whisper`
  now routes through the unified dispatch like every other backend,
  and the `--list-backends` matrix reads the wrapper's capability
  bitmask live (no more hardcoded `kWhisperCaps` constant). The
  default (empty-backend) whisper path stays byte-identical.
  Subsequent commits (a71f617 grammar + auto-dl, fb47aa0 VAD,
  2f43f7c n_processors) filled in the rest of whisper_full_params
  plumbing — the wrapper now advertises: ts-native, word-ts,
  tok-conf, lang-detect, translate, temperature, beam-search,
  grammar, flash, VAD-internal, parallel-processors, auto-dl.
  The only gaps vs the historical cli.cpp path are stereo diarize
  (needs pcmf32s through the dispatch) and -owts karaoke output.

---

## CLI + examples cleanup

- **[done]** ~~Delete the per-model `examples/*-main/` directories
  once `crispasr --backend X` has shipped.~~ **Done.** 7 user-facing
  CLIs (cohere-main, parakeet-main, canary-main, qwen3-asr-main,
  voxtral-main, voxtral4b-main, granite-main) removed. The
  `cohere-quantize.cpp` tool was rescued into
  `examples/crispasr-quantize/` as a standalone GGUF quantizer.
  Kept: `cohere-align` / `nfa-align` (standalone forced-alignment
  tools, still useful when the transcript is pre-existing) and the
  `{qwen3,voxtral}-test-*` differential fixtures.

- **[later]** `tests/CMakeLists.txt` uses `whisper-cli` as the test
  target. Keep that target name (we already preserve it) but move the
  tests over to `$<TARGET_FILE:crispasr>` once the rename has propagated.

---

## Feature parity gaps (non-whisper backends vs whisper)

The whisper backend in CrispASR is the most feature-complete. The
capability matrix in the README shows which features are missing on
each backend. High-value gaps to close:

- **[done]** ~~**Temperature sampling for LLM backends**~~ — landed
  in e4861c3 (pipeline) and 7a7e6cd (run_with_probs). voxtral,
  voxtral4b, qwen3 and granite all honour `-tp N` via the shared
  `core_greedy_decode` helper's `sample_temp` path. Default
  temperature=0 stays on the bit-identical pure-argmax path.
- **[done]** ~~**Best-of-N sampling for LLM backends**~~ — all four LLM
  backends (voxtral, voxtral4b, qwen3, granite) support `--best-of N`
  with `--temperature > 0`. Each run uses a different RNG seed, best
  selected by mean token softmax probability.

- **[later]** **VAD integration in LLM backends.** qwen3 and voxtral
  currently don't chunk long audio; the dispatch layer does VAD slicing
  but the LLM models themselves pad to a fixed 30s window. Variable-
  length mel would let them handle >30s natively.

- **[done]** ~~**Streaming transcription.**~~ **Done.** Generic
  `--stream` (stdin PCM), `--mic` (microphone capture via
  arecord/sox/ffmpeg subprocess), `--live` (continuous mode), and
  `--monitor` (unicode progress symbols) work with all 11 backends.
  Inspired by antirez/voxtral.c. Also added `--alt` for per-token
  confidence display.

- **[later]** **Native voxtral4b streaming protocol.** The model is
  designed for realtime streaming with configurable 240ms-2.4s delay.
  Currently we run it in chunk-and-transcribe mode. Exposing a
  streaming mode through the CLI is a bigger design question.

- **[done]** ~~**Audio understanding mode for voxtral 3B.**~~ `--ask`
  flag switches from transcription to Q&A template. Tested: language
  detection ("The audio is in English.") and summarization work.

---

## Per-model follow-ups

### parakeet
- **[later]** Port the TDT decoder (LSTM predictor + joint head) to
  ggml graphs so it can run on GPU. Currently pure CPU float* loops.
  Risk: per-token LSTM stepping is sequential, so GPU speedup may be
  small. Encoder is already the dominant cost.

### canary
- **[later]** Speech translation quality validation at scale.
  Currently regression-tested on German only.

### cohere
- **[done]** ~~F32→F16 self-attention KV cache upgrade.~~ Already F16
  (decoder + cross-attention KV caches both use GGML_TYPE_F16).

### qwen3 / voxtral
- **[later]** Stop recreating `ggml_backend_sched` on every compute
  call (encoder, prefill, each decode step). Create once at init with
  worst-case node budget; use `ggml_backend_sched_reset()` between
  calls. ~80 LOC per runtime.

### voxtral4b
- **[later]** Reduce right padding from 17 → 10 tokens to match the
  reference `voxtral.c` implementation.
- **[later]** SRT/VTT subtitle output (currently only plain transcript;
  CTC alignment already works via `-am`).

### granite
- **[later]** HF release of quantised GGUFs (`cstr/granite-speech-4.0-1b-GGUF`
  is still pending). Need `cohere-quantize granite-speech-1b.gguf …`
  then upload.
- **[later]** Encoder parallelisation: granite_speech is now linked
  against OpenMP but has no `#pragma omp` annotations yet. Adding
  `#pragma omp parallel for` on the per-layer encoder hot loops
  would deliver a measurable speedup on CPU but shifts the float
  reduction order, so the change needs its own regression gate
  (allow small float drift; transcript must stay correct). Encoder
  Conformer is the dominant cost today (~22.5s on jfk.wav).
- **[done]** ~~Consider porting the per-layer CPU encoder to a single
  ggml graph.~~ Done: `granite_build_encoder()` wired with runner
  function, enable via `GRANITE_ENCODER_GRAPH=1`. Identical output on
  jfk.wav. Shaw RPE omitted (approximate) — follow up if accuracy
  issues surface on other test cases.
- **[done]** ~~Remove dead ggml graph encoder `granite_build_encoder`.~~
  Resurrected — now used by the graph encoder path (`GRANITE_ENCODER_GRAPH=1`).

### canary_ctc (aligner)
- **[done]** ~~Fix single-backend scheduler — currently no CPU fallback
  if the primary backend rejects an op.~~ Already uses the 2-backend
  pattern (GPU primary + CPU fallback) at both scheduler init points
  (`canary_ctc_compute_logits_from_mel_debug` and `canary_ctc_compute_logits`).

---

## Markdown cleanup (this session)

Consolidating ~15 historical notes into three live docs:

- `TODO.md` — this file (replaces all `*-todo.md`)
- `LEARNINGS.md` — technical insights, benchmarks, comparisons
- `HISTORY.md` — condensed chronology of the ports

Remove after consolidation: `canary-todo.md`, `parakeet-todo.md`,
`granite-todo.md`, `voxtral-todo.md`, `voxtral-4b-todo.md`,
`qwen3-asr-todo.md`, `TODO_COHERE_OPTIMIZATION.md`,
`benchmark_cohere.md`, `qwen3-asr-benchmark.md`, `ggml_plans.md`,
`voxtral-comparison.md`, `test_german.md`, `PERFORMANCE.md`.

Keep: `README.md`, `TODO.md`, `LEARNINGS.md`, `HISTORY.md`, `UPSTREAM.md`,
`README_sycl.md`, `ci/README.md`, `models/README.md`, `samples/README.md`,
`hf_readmes/*.md`.

---

## Ground-truth diff infrastructure (new in this session)

The `tools/dump_reference.py` + `crispasr-diff` pair is the new
contributor-facing path for adding backends with confidence. Status:

- **[done]** Unified Python dumper (`tools/dump_reference.py`) with
  plug-in backend modules under `tools/reference_backends/`. Writes
  a single GGUF tensor archive per dump (not scattered `.npy` files).
- **[done]** Shared C++ diff harness (`examples/cli/crispasr_diff.{h,cpp}`)
  loading the archive via `core_gguf::load_weights` and exposing
  `compare(name, data, n)` with cosine sim / max-abs / RMS / top-1
  argmax metrics.
- **[done]** `crispasr-diff` CLI binary built alongside `crispasr`.
  Currently wires up mel-stage comparison for voxtral / voxtral4b /
  qwen3 / granite (the stages their public C API exposes). Parakeet /
  canary / cohere only have all-in-one `transcribe()` entry points so
  they're reported as unsupported.
- **[done]** Worked-example Python backend modules:
  `tools/reference_backends/qwen3.py` and `voxtral.py` are fully
  ported from the legacy `models/*-dump-*.py` scripts.
- **[done]** ~~Port `models/voxtral4b-dump-ref.py` into
  `tools/reference_backends/voxtral4b.py`~~ — done; captures mel,
  encoder_output (post-projector), t_cond, llm_argmax, generated_text.
- **[done]** ~~Port `models/granite-speech-kaggle-groundtruth.py` into
  `tools/reference_backends/granite.py`~~ — done; captures mel,
  per-layer encoder checkpoints, projector_out (Q-Former), llm_argmax,
  text. Strips the Kaggle-specific HF_TOKEN / gist-upload plumbing.
- **[done]** ~~Expose `audio_encoder`-only standalone entry points
  in `parakeet` / `canary` / `cohere` C headers so `crispasr-diff`
  can do stage-by-stage comparison for them too~~ — done in 7ba3c50.
  Each backend now exposes `<name>_compute_mel` and
  `<name>_run_encoder` alongside the existing `<name>_transcribe_ex`
  batch entry point. `crispasr_diff_main.cpp` gained the matching
  dispatch branches. What's still missing is
  `tools/reference_backends/{parakeet,canary,cohere}.py` — those
  backends aren't in REGISTERED_BACKENDS yet because loading NeMo
  checkpoints from PyTorch is non-trivial (needs either
  `nemo_toolkit` or direct .nemo unpacking following the pattern
  in `models/convert-parakeet-to-gguf.py`). Follow-up below.
- **[later]** Write `tools/reference_backends/parakeet.py`,
  `canary.py`, `cohere.py` so the crispasr-diff harness has
  references to compare against. Each follows the pattern already
  in tools/reference_backends/{qwen3,voxtral,voxtral4b,granite}.py.
  Parakeet and canary load .nemo tarballs via torch + tarfile
  (mirroring models/convert-parakeet-to-gguf.py's unpack_nemo()).
  Cohere loads a HF transformers checkpoint.
- **[done]** ~~Migrate `examples/{qwen3,voxtral}-test-*/main.cpp`
  drivers to load their reference data from a crispasr-diff GGUF
  archive via `crispasr_diff::Ref` instead of the inline NPY parser.~~
  Done in this batch. All six drivers (`voxtral-test-encoder`,
  `voxtral-test-llm`, `voxtral-test-e2e`, `qwen3-asr-test-conv`,
  `qwen3-asr-test-llm`, `qwen3-asr-test-trace`) now link
  `crispasr-diff-lib` (the reusable static version of
  `examples/cli/crispasr_diff.{h,cpp}`) and consume a single
  `reference.gguf` produced by `tools/dump_reference.py`. The
  inline `load_npy_f32` parsers are gone. `qwen3-asr-test-bpe` has
  no reference data and stays as-is.
- **[done]** ~~Extend `tools/reference_backends/qwen3.py` to emit
  `trace_input_ids / trace_audio_pad_pos / trace_first_logits /
  trace_generated_ids`~~ — needed by `qwen3-asr-test-trace` for the
  chat-template prompt + splice + forward path, plus `llm_input_ids`
  + full-T `llm_logits` for the `qwen3-asr-test-llm` differential
  test. Trigger via `--stages` or leave as part of the backend's
  `DEFAULT_STAGES` (they're in the default now).
- **[later]** Mirror the same for `tools/reference_backends/voxtral.py`
  so `voxtral-test-llm` stops reporting `[SKIP]`. Needs the Voxtral
  apply_chat_template → processor → embed → splice → forward path.
- **[rejected]** Vosk as a third `--lid-backend` provider. Vosk's
  C++ API exists and would fit the "no Python" constraint, but it's
  an ASR toolkit, not a language detector — standalone LID means
  running the full Kaldi decoder with each candidate language model
  and comparing likelihoods, which is slow and memory-heavy. It also
  drags in Kaldi + OpenFST + BLAS as build dependencies, ~50-100 MB
  of binary and a non-ggml runtime path that would be the first of
  its kind in this repo. If someone still wants it, the dispatcher
  in crispasr_lid.cpp::crispasr_detect_language() has an easy
  extension point — just another `if (be == "vosk")` branch. For
  now we stick with whisper-tiny (shipping) and the future native
  Silero GGUF port.
- **[done]** ~~**Qwen3-ForcedAligner-0.6B as a generic timestamp post-step.**~~
  Fully implemented. `qwen3_asr_align_words()` does the full pipeline:
  mel → encoder → prompt with `<|timestamp|>` markers → single forward
  pass → argmax * 80ms per position. Dispatched in `crispasr_aligner.cpp`
  via filename detection. GGUF on HF: `cstr/qwen3-forced-aligner-0.6b-GGUF`.
  Verified working with voxtral on jfk.wav.
- **[done]** ~~Native GGUF port of Silero's language detector.~~
  **Done.** `src/silero_lid.{h,cpp}` implements a pure-C++ forward pass
  (no ggml graph — manual F32 loops, similar to pyannote_seg). GGUF
  converter at `models/convert-silero-lid-to-gguf.py`. 507 tensors,
  16.1 MB F32 / ~9 MB Q8_0. CLI wiring in `crispasr_lid.cpp`: when
  `--lid-model *.gguf` is passed, the native path runs; falls back to
  sherpa subprocess for `.onnx` models. Verified: English, German,
  Latvian correctly detected across multiple test wavs.
- **[done]** ~~Delete the legacy `models/*-dump-*.py` scripts~~ — done.
  Removed `qwen3-asr-{llm,reference,trace}-dump.py`,
  `voxtral-{encoder,llm}-dump.py`, `voxtral4b-dump-ref.py`, and
  `granite-speech-kaggle-groundtruth.py`. Everything they did is now
  covered by `tools/dump_reference.py` + `tools/reference_backends/`.
  The `voxtral4b-debug-{cpp,light}.py` deep-diagnostic scripts and
  `voxtral-verify-gguf.py` (a converter sanity check) were kept —
  they're useful tools with no GGUF-archive replacement yet.
- **[later]** Delete the empty `examples/{parakeet,canary,cohere,
  qwen3-asr,voxtral,voxtral4b,granite}-main/` directories. They no
  longer have any source files (CMakeLists.txt already dropped them
  from `add_subdirectory`), only stale build artifacts. Untracked,
  so this is a filesystem cleanup rather than a git operation.

---

## Upstream dependencies

Full tracking is in `UPSTREAM.md`. Short summary:

- **[upstream]** whisper.cpp `examples/ffmpeg-transcode.cpp` mp4-family
  container crash. Workaround: pre-convert with ffmpeg one-liner.
- **[upstream]** ggml x86 AVX-VNNI / AVX512-VNNI dispatch for Q8_0 dot
  products. Closes the 5-second gap to ONNX INT8 on x86 servers.
- **[upstream]** NeMo Forced Aligner auxiliary CTC model standalone
  release. Not blocking — our converter extracts it from the `.nemo` tarball.

---

## HF releases

| Repo | Status |
| --- | --- |
| `cstr/parakeet-tdt-0.6b-v3-GGUF` | ✅ shipped |
| `cstr/parakeet_de_med-GGUF` | ✅ shipped |
| `cstr/canary-1b-v2-GGUF` | ✅ shipped |
| `cstr/canary-ctc-aligner-GGUF` | ✅ shipped |
| `cstr/cohere-transcribe-03-2026-GGUF` | ✅ shipped |
| `cstr/qwen3-asr-0.6b-GGUF` | ✅ shipped |
| `cstr/qwen3-asr-1.7b-GGUF` | ✅ shipped |
| `cstr/qwen3-forced-aligner-0.6b-GGUF` | ✅ shipped |
| `cstr/voxtral-mini-3b-2507-GGUF` | ✅ shipped |
| `cstr/voxtral-mini-4b-realtime-GGUF` | ✅ shipped (Q4_K + Q8_0) |
| `cstr/granite-speech-4.0-1b-GGUF` | ✅ shipped (f16, q4_k, q5_0, q8_0) |
| `cstr/granite-speech-3.3-2b-GGUF` | ✅ shipped (f16, q4_k, q5_0, q8_0) |
| `cstr/granite-speech-3.3-8b-GGUF` | ✅ shipped (q4_k, q5_0, q8_0) |
| `cstr/granite-speech-3.2-8b-GGUF` | ✅ shipped (q4_k, q5_0, q8_0) |
| `cstr/stt-en-fastconformer-ctc-large-GGUF` | ✅ shipped (f16, q4_k, q5_0, q8_0) |
| `cstr/stt-en-fastconformer-ctc-xlarge-GGUF` | ✅ shipped (f16, q4_k, q5_0, q8_0) |
| `cstr/stt-en-fastconformer-ctc-xxlarge-GGUF` | ✅ shipped (f16, q4_k, q5_0, q8_0) |
| `cstr/silero-lid-lang95-GGUF` | ✅ shipped (f32 only — 16 MB; quants break accuracy on small conv tensors) |
| `cstr/pyannote-v3-segmentation-GGUF` | ✅ shipped (f32, 5.7 MB) |
| `cstr/wav2vec2-large-xlsr-53-english-GGUF` | ✅ shipped (f16, q4_k, q5_0, q8_0) |
| `cstr/wav2vec2-large-xlsr-53-german-GGUF` | ✅ shipped (q4_k) |

---

## Current session WIP (April 2026)

### Silero LID native port (#56) — DONE ✅
- **Files:** src/silero_lid.{h,cpp}, models/convert-silero-lid-to-gguf.py
- **State:** Fully working. Detects English on jfk.wav, German on German samples, Latvian on Latvian samples. Matches ONNX reference model output.
- **5 bugs fixed:** (1) front-end zero-pad 160/side not reflection-pad 320 left, (2) stride-2 output T=(T-1)/2+1 not T/2, (3) QKV split order K,Q,V not Q,K,V, (4) missing ReLU after stride-1 projections stages 4-7, (5) missing tanh in attention pooling.
- **Architecture:** Learned STFT Conv1d(1→322,k=320,s=160) → magnitude → log(2^20×mag+1) → adaptive norm (17-tap reflected smooth) → 8×(12 dw-sep conv + post-norm transformer + stride-2/1 proj+ReLU) → attention pool (tanh+softmax) → 95-lang + 58-group classifiers.
- **GGUF:** F32 16.1 MB, Q8_0 ~9 MB. Available at `cstr/silero-lid-lang95-GGUF`.

### wav2vec2 ggml rewrite (#63) — DONE ✅
- **Files:** src/wav2vec2-ggml.{h,cpp}
- **State:** Layer-by-layer ggml graphs (~80 MB/layer, reused). CNN+posconv stay manual. Correct output on jfk.wav.
- **Architecture:** Per-layer graph with `ggml_graph_compute_with_ctx` (proven to correctly reference external F16 weights). LM head via `ggml_linear_f32`.
- **Root causes found:**
  1. `ggml_gallocr`/`ggml_backend_sched` corrupt external F16 weight tensors (reallocate over them). Workaround: use `compute_with_ctx`.
  2. Data layout confusion: ggml `[H,T]` stores `data[h+t*H]` = C's `data[t*H+h]` — SAME layout as `[T,H]` row-major. The original code had a spurious transpose that corrupted all data.
  3. `flash_attn_ext` crashes with `mask=nullptr`. Replaced with `mul_mat`-based attention.
  4. Logits `[V,T]` in ggml = `[T,V]` row-major — no transpose needed.
- **Model:** `jonatasgrosman/wav2vec2-large-xlsr-53-english` (33 vocab, 1024 hidden, 24 layers).

### Pyannote v3 native (#57) — DONE ✅
- **Full runtime:** SincNet + 4× biLSTM + 3× Linear + LogSoftmax (440 lines C++)
- **Wired into CLI:** `--diarize-method pyannote --sherpa-segment-model *.gguf` uses native path, falls back to subprocess for .onnx
- **Tested:** 650 frames on jfk.wav, "(speaker 1)" assigned correctly

### Granite speedup (#64) — CLOSED (hardware-blocked)
- 33s for 11s audio at q4_k, 4 threads. Bottleneck: autoregressive LLM decode (26s of 33s).
- `--gpu-backend` flag exists, granite uses `ggml_backend_init_best()`. No code changes needed — just add a GPU.

### iOS + Android CI (#65) — DONE ✅
- Cross-compilation gates for arm64 iOS (Xcode) + arm64-v8a Android (NDK r26d).

### v0.1.0 release — SHIPPED ✅
- Linux 660KB, macOS 484KB, Windows 1437KB — all 3 platforms built via GitHub Actions.
