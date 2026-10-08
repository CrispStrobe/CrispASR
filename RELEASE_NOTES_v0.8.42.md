# v0.8.42 — Live translation, persistent streaming and Hikari

Draft for the next release, covering changes on main since v0.8.41. The version
has not been bumped and the tag has not been created. Final integration and
release-tip validation are still pending. PR #515 and #516 are integrated;
PR #492 remains unmerged.

## Persistent streaming and German Moonshine

- The C session API now exposes Nemotron's cache-aware encoder/RNN-T stream,
  Qwen3-ASR prefix streaming and the existing Voxtral realtime implementation.
  Nemotron honors the stream language prompt and removes language control tags
  consistently from CLI, C ABI text, words and stream output. Canonical backend
  steps make its input independent of microphone packet boundaries.
- `crispasr_session_stream_kind()` reports the loaded implementation: unavailable,
  rolling windows, persistent model caches or text-prefix streaming. All seven
  language wrappers, WASM/JavaScript and the WebSocket ready event expose it.
  Qwen's prefix streaming re-encodes accumulated audio; it retains text prefixes
  rather than encoder/KV caches across calls.
- Stream updates contain cumulative utterance text, including Voxtral's
  underlying consuming deltas. Dart propagates stream errors and reads long
  UTF-8 output without truncation. Feed only new 16 kHz mono PCM, then flush
  and close; the model session must outlive its stream.
- Optional CPU ONNX Runtime support adds German Streaming Tiny/Small in int8
  and F32, the Phreak87 Tiny two-graph ONNX export, and the dattazigzag native
  Q4 Tiny GGUF. These registry entries are MIT licensed. Downloads use pinned
  revisions and isolated bundles, including matching graphs, tokenizer,
  configuration and license files. Companion failures propagate; cached
  incomplete bundles can be repaired. Generic `encoder*.onnx` paths are detected
  from their companion configuration through both CLI and C ABI.
- ONNX backend cleanup shares a nonvirtual helper between explicit shutdown
  and destruction, resolving the destructor diagnostic from deep static analysis.
- Five-graph Moonshine produces incremental drafts with persistent frontend
  state and bounded encoder updates. Final flush replays the actual batch
  frontend/encoder/decoder from retained original PCM, correcting Small int8
  punctuation drift caused by tiny differences in windowed frontend values.
  PCM history costs 64 kB per second. Explicit generation caps work through
  CLI, session batch and persistent streams; clearing the override restores
  each model's duration-derived budget. The legacy two-graph model is buffered.

See [German deployment and SDK setup](docs/german-moonshine.md) and
[streaming semantics](docs/streaming.md#stateful-streams-from-c-and-dart).

## Live transcription and translation

- `--live-translate` translates streaming recognition sentence by sentence.
  `--translate-model` also enables translation on `--stream` / `--mic` runs.
  Sentence commits use agreement between recognition updates and pause handling;
  audio already committed is excluded from subsequent recognition windows.
- Choose Opus-MT/Marian, m2m100, MADLAD, or a translation LLM. Registry names
  `hy-mt2` and `index-translate` resolve the publishers' Apache-2.0 GGUF models.
  Translation LLM output appears as it is generated.
- Draft translation uses the words on which recent recognition updates agree.
  JSON `translation_partial.stable` identifies the stable whole-word prefix;
  the terminal dims the provisional remainder. Candidate translations are reused
  at sentence commit, and LLM translators retain their shared instruction prefix.
- `--translate-revise MODEL` optionally re-translates finished paragraphs with
  more context. `--translate-revise-asr MODEL` also re-recognizes the completed
  utterance before revision. JSON revision events include sentence IDs, timing,
  revised source where applicable, and `final_until_sentence`.
- `--translate-view inplace|scroll` controls terminal presentation. With a slow
  revision pass, the default in-place view replaces the affected sentences and
  prints the final transcript to the normal screen on exit.
  `--translate-revise-backlog N` bounds the revision queue (default 3).
- Added real-time pacing, silence-delimited session handling, first-Ctrl+C exit
  and `CRISPASR_STREAM_TIMING` diagnostics. Windows live transcription and
  translation have an end-to-end CI workflow.

Revision is optional: additional context can improve wording and segmentation,
but an LLM can also add meaning to incomplete speech. Revisions can arrive
seconds later, and contention with the fast path on a shared GPU remains a
measurement task. See [streaming usage and measurements](docs/streaming.md).

## Opus-MT / MarianMT translation

- Added the `marian` backend, GGUF converter, tokenizer, registry models and
  language-pair selection through `--translate-backend marian`.
- Published 24 directions across German, English, French, Italian, Spanish,
  Arabic, Hebrew and Turkish. Supported pairs without a direct model can route
  through English when both component directions exist.
- Most directions download Q8_0; German→Arabic downloads F16 because its tested
  Q8_0 output differed more substantially. Q4_K was evaluated and is not the
  published default. The released checkpoints use CC-BY-4.0.
- Greedy and beam decoding share the m2m100 runtime. Beam search now snapshots
  each beam's decoder cache instead of repeatedly decoding its entire prefix.
  Fixed a decoder-KV allocation leak and propagated `-t` to both runtimes.
- On Apple Silicon, m2m100 defaults to Metal and Opus-MT to CPU, following paired
  measurements in the live pipeline. `CRISPASR_M2M100_GPU=0|1` overrides device
  selection. These defaults do not imply the same result on every device.

F16 matches the driving reference for both greedy and beam-4 decoding on the
recorded sentence sets: 14 German→English, eight English→German and eight for
each of the 22 additional directions. Q8_0 sometimes chooses different valid
wording; it is not claimed to reproduce every F16 token. The German→English
live test measured 23–118 ms per Opus-MT sentence, median about 45 ms, on a
loaded M1. See the [language-pair table](docs/streaming.md#opus-mt-language-pairs)
for exact defaults and quantization results.

## Hikari simultaneous speech translation

- Added native `hikari` support for `sbintuitions/hikari-medium` (MIT): English
  speech directly to German, Japanese or Russian, including streaming output.
- Registry loading fetches the GGUF model and its required Silero VAD companion.
  The runtime and reference path include the speech probabilities used by the
  model's emit/wait policy. CLI, C ABI, architecture detection, live tests and
  regression manifest are wired in.
- Default F16 is approximately 1.5 GB; Q8_0 is approximately 873 MB. F16 matches
  the reference across 161 streaming steps on JFK. Quantized output can differ.
- Initializes unavailable token confidence to the existing 1.0 convention and
  resets constructor state through a nonvirtual helper shared with `reset()`.
- Added CUDA benchmark tooling with pinned model/source receipts. Failed
  downloads/inference, absent timing, incorrect short-clip content, and stale
  transcript files fail validation rather than producing a success report.

Recorded CUDA runs took 234–258 ms per audio-second for F16 and 193–207 ms for
Q8_0 on the tested setup. The tested M1 Metal path is slower than real time;
this release does not claim universal Hikari real-time performance. Detailed
usage and device results are in [the streaming guide](docs/streaming.md#one-model-instead-of-two-hikari-english-speech--de--ja--ru).

## Character timestamps for Arabic and other CTC vocabularies

- Wav2vec2-family forced alignment exposes measured character start/end times
  alongside words. Align-only word and segment JSON include nested `characters`;
  the C ABI and Python, Dart, Rust, Go, Java, C# and Ruby expose the same spans
  in their existing time units.
- Repeated letters retain separate CTC occurrences. Supported Arabic labels stay
  in their original script. Latin-only aligners retain their romanization fallback;
  unsupported or romanized characters receive no invented timestamps.
- Impossible complete CTC paths fail explicitly. Character times use the model's
  frame resolution and are not interpolated subdivisions of word durations.
- Java now converts incoming strings and decodes returned strings as UTF-8
  independently of the host JNA encoding, including Unicode model paths.
  CLI diagnostic previews preserve UTF-8 character boundaries.

Real Arabic Q4 acceptance passes 15 words / 85 character spans against an
independent full-sequence Viterbi calculation, CLI word/segment JSON, Python/
C ABI offsets and invalid accessors, and Java/JNA equality with US-ASCII forced.
The job passes 41 new / 91 existing alignment assertions; local ASAN/UBSAN is
clean. This validates post-logit alignment/output, not human-annotated phonetic
boundaries. See [exact model/audio/source pins and proof](docs/ctc-characters-2026-10-08.json).

## ASR and TTS correctness

- **MioTTS:** CLI and session output rates now come from codec metadata. Public
  v2 reports 44,100 Hz; missing legacy metadata keeps the 24,000 Hz fallback.
  Preset voices work at startup and per request; clearing an override restores
  the startup preset. Temperature and seed controls reach native inference,
  and model downloads include the required tokenizer.
- **Qwen3-ASR hotwords (#488):** C ABI requests now use the same system-turn
  context as CLI and streaming, preserving forced-language assistant prefills
  and explicit questions. Off/on/clear requests are covered by speech checks.
- **MiMo-ASR (#489):** language-selected default instructions match upstream,
  explicit `--ask` remains independent, and automatic language detection avoids
  unnecessary external LID. CLI/library capability tables reflect this behavior.
- **Orukeet (#491):** short names, cached filenames and explicit paths select
  the Parakeet runtime. Fresh download and anonymous C ABI recognition are
  validated. CC-BY-SA still requires acceptance; refusal performs no download.
  CLI notices now distinguish share-alike/custom terms from noncommercial terms.
- **Moonshine German:** the existing native decoder handles stretches between
  pauses separately to improve long-utterance segmentation.
- **Nemotron:** corrected GPU streaming-cache graph resubmission and stale
  allocator addresses on graph re-allocation. Cached transcripts match the
  working path. Language-tag cleanup covers native text, words and sessions.
- **Qwen3.5 / Qwen3.5 MoE:** loaders skip appended MTP blocks. This enables the
  published Index-Translate-2B GGUF. The MoE loader change is compiled; a 35B
  MoE checkpoint was not executed as part of this validation.

## Performance and ggml

- Synced the shared ggml fork to upstream v0.26.0 at `c36dab89`, retaining the
  fork's carried operations and restoring Metal `kernel_mul_mm_hp`. This fixes
  an abort for F32-precision matrix multiplication on affected Apple GPUs.
- Incremental Silero VAD processes each frame once and retains state; the
  measured live-loop VAD work fell from 30–86 ms to 3–6 ms per step.
- Nemotron sessions keep encoder/cache state in backend memory and build one
  graph per chunk. The prompt kernel is a ggml graph on CPU with weights
  dequantized once. A loaded-CPU test measured 67 ms per 320 ms chunk; no
  corresponding Metal speed improvement is claimed.
- Index-Echo keeps `CRISPASR_LLAMA_PIPELINE_DISABLE=1` as an opt-in scheduler
  experiment, with `INDEX_ECHO_BENCH=1` timing/reuse counters. A two-T4 AB/BA
  experiment showed 2.5–3.6% improvement across 48 accepted calls. The default
  is unchanged; graph reuse counters do not prove CUDA graph capture.
- Index-Echo mixed-Q4 preparation audits tensor types and preserves all 177
  source F32 tensors, including recurrent convolution matrices. This tooling
  does not constitute an accepted Q4 model. Candidate transfer and actual
  stage/magnitude, decoded-output and roundtrip acceptance remain pending.

The ggml sync passed fork/platform CI and recorded CUDA speech checks for seven
backends, plus all 67 Index-Echo-2B stages. Canary's per-layer regression floor
is explicitly 0.99 after the sync (recorded x86 layer 18 cosine 0.9977).
Existing wav2vec2 GPU transcript drift remains under investigation.

## Bindings, packaging and developer tools

- WASM exposes `sessionPianoNotes` and `sessionPianoSampleRate` for note
  transcription, with export checks across all five builds and a Node/Embind
  no-session smoke check. Model-in-browser acceptance is a separate task.
- Glint synchronization uses the built-in GitHub token, explicitly dispatches
  CI after a real sync push, and supports a tested dry run.
- CI/regression APT setup replaces the failing Azure Ubuntu mirror with the
  official HTTPS mirror, bounds network waits and preserves error exits.
- Repaired the Apple Metal cache smoke test for the ggml device API and
  Objective-C++17, with explicit Metal linkage. A focused macOS workflow checks
  shared/static builds and asserts both tests are discovered. Actual macOS
  compilation and both lifecycle/disable-env tests pass in both configurations
  without skipping (`37762956547`). Unavailable GPU hardware produces a skip;
  lifecycle coverage does not prove serialized compute pipelines or speed.
- Kaggle regression honors SRT transcripts, downloads declared companions, and
  uses corrected model revisions. Registry URL checking verifies manifest pins.
- The manual Windows CUDA smoke workflow can package matched CUDA-12.6
  forced-MMQ OFF/ON experiments for #483. Both Windows arms pass compile-setting,
  runtime-version and driverless-startup checks; actual artifact manifests pair
  locally and in the final hosted pair job; the complete workflow passes.
  These are experiment artifacts, not changed release defaults or evidence of
  GTX1660/MX150 speed. [Packages and scope](docs/cuda126-mmq-experiment-2026-10-08.md).

## Validation and remaining gates

MioTTS ARM/x86 checks cover codec rate, request behavior and speech readback;
both tested CLI/session outputs have 0% WER at 44.1 kHz. Combined Qwen3/MiMo
checks preserve expected speech and CLI/C ABI equality. Fresh ARM Index-Echo-9B
F16 validation passes 76 checks per clip, full-file output and three Piper
roundtrips, with minimum cosine 0.999997 and maximum relative L2 0.1883%.
These results do not accept experimental Echo Q4 weights.

Integrated source checkpoints have passed platform CI, lint, bindings, WASM
and selected numerical regressions. Orukeet's pinned Q4 model passes real CLI
and C ABI speech acceptance, with the published v0.8.41 failure reproduced.
See [MioTTS/Echo evidence](docs/miotts-echo-integration-2026-10-03.md),
[prompt evidence](docs/asr-prompt-validation-2026-10-03.md) and
[issue/PR triage](docs/issue-pr-triage-2026-10-07.md) for source pins and runs.

Before tagging, refresh this draft against the final main commit and complete
its required checks. Hikari's full pinned deep-lint rerun passes at `f55c7bc8c`.
PR #515 passes final x86/ARM acceptance: all eight cases on each architecture,
15 exact stages per five-graph deployment, scale-negative control, and CLI/C ABI
cap/reset/stream checks. These compare identical deployed ONNX exports against
independent Python ORT execution. PR #516 passes full real Arabic character
alignment and Java/JNA acceptance. Integrated native CI, regression, lint,
Moonshine speech acceptance, Go/Rust/C#/Dart, WASM and Windows live translation
checks pass at their recorded sources; full pinned deep lint is still running.
PR #492's MiMo/CANN changes remain unmerged. Reporter-specific
Windows/Vulkan, Intel Mac and newer NVIDIA hardware retests remain open.
