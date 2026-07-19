# Changelog

## 0.8.15

- Pitch (F0) estimation: `CrispasrSession.pitch()` returns a monophonic
  pitch track from mono 16 kHz float32 PCM, backed by the CREPE backend.
  Adds the `PitchFrame` record typedef (`timeMs`, `f0Hz`, `voicedProb`)
  and the `pitchSampleRate` capability probe.
- The native GGUF architecture auto-detect now recognises `crepe` (and
  `htdemucs`), so plain `CrispasrSession.open()` works for those models;
  previously only the CLI auto-detected them and every binding had to name
  the backend. Passing `backend: 'crepe'` explicitly remains valid and is
  required against an older native library.

## 0.8.11

- Session introspection accessors: CTC vocab, Whisper `no_speech_prob`, and
  `detected_language`, mirrored across the language bindings.
- Docs/quality: add an example, enable `lints/core`, and apply the resulting
  fixes (flow-control braces + FFI typedef casing).

## 0.8.10

- Initial pub.dev release for the current CrispASR 0.8 line.
- Exposes Dart FFI bindings for the unified CrispASR session API, Whisper-shaped legacy API, audio decode helpers, language detection, diarization, chat/text helpers, grammar support and alt-token capture.
- Keeps the package pure Dart FFI: native `libcrispasr` must be installed or bundled separately.

## 0.5.13

- **Whisper alt-token capture (`--alt N` parity)** — per-token
  top-N alternative-candidate capture for whisper greedy
  decode. Closes the last open whisper-cli equivalent gap. New
  `Word.alts` list (with a new `AltToken` value class, default
  `const []` so existing call-sites stay source-compatible)
  on both the low-level `Segment` shape (from `transcribePcm`)
  and the unified session `SessionSegment` shape (from
  `CrispasrSession.transcribe`). New
  `TranscribeOptions.altN` (default 0 = off) plumbs through
  the low-level path; sticky
  `CrispasrSession.setAltN(int)` (matches the
  `setFallbackThresholds` / `setWhisperDecodeExtras` pattern)
  threads through the session path. Pre-0.5.13 dylibs raise
  `UnsupportedError` for the session setter and report
  `Word.alts == []` from both readers, so apps stay loadable
  and just hide the affordance.
- Capture happens inside `whisper_sample_token` so the alts
  are softmax probabilities at the same decode step as the
  chosen token. Beam search is excluded — siblings are
  beam-conditional rather than greedy alternatives. Chosen
  token is not included in the alt list; entries are sorted
  descending by probability.
- Whisper session-transcribe path now also populates
  `SessionSegment.words` (it previously returned only
  segment-level text) — closes a long-standing gap with the
  parakeet / canary backends as a side-benefit of needing
  per-token data for the word-alt mapping. Whisper sub-word
  BPE means a multi-token word's alts cover the first content
  token only (e.g. "kubectl" surfaces alts for "kub"); full
  word-level enumeration is deferred.
- C-ABI symbols: `crispasr_params_set_alt_n`,
  `crispasr_session_set_alt_n`, `crispasr_token_n_alts` /
  `_alt_id` / `_alt_p` / `_alt_text`,
  `crispasr_session_result_word_n_alts` / `_alt_text` /
  `_alt_p`. New whisper getters
  `whisper_full_get_token_n_alts` / `_alt_id` / `_alt_p`
  (plus `_from_state` variants). All nine symbols pinned in
  `bindings_smoke_test.dart`.
- Live test `test/alt_tokens_live_test.dart` (tagged `live`)
  exercises the full stack against `ggml-tiny.en.bin` +
  `samples/jfk.wav`. Asserts ≥1 returned word has alts,
  every alt's p ∈ [0, 1] and the list is descending,
  chosen token excluded from its own alts, and `setAltN(0)`
  on a re-decode actually clears them. Skips silently when
  `CRISPASR_LIB` / `CRISPASR_MODEL` aren't set. Representative
  dev-box result: 22/22 words on JFK get runner-ups (e.g.
  "Americans → America(4.85%), americ(3.84%), American(3.35%)").

## 0.5.12

- **Audio enhancement (RNNoise pre-step)** — new top-level
  `enhanceAudioRnnoise(Float32List pcm)` runs xiph/rnnoise v0.1
  on a 16 kHz mono float32 buffer (upsample to 48 kHz →
  RNNoise frame loop → downsample back) and returns a fresh
  same-length `Float32List`. Backed by the new C-ABI
  `crispasr_enhance_audio_rnnoise`; pre-0.5.12 dylibs raise
  `UnsupportedError` so callers graceful-degrade. State is
  per-call so worker isolates can run enhancement concurrently
  without coordination.

## 0.5.11

- **Whisper text-suppression + prompt-carry extras** — three
  more whisper-only `wparams` knobs the CLI surfaces
  (`--suppress-nst`, `--suppress-regex`,
  `--carry-initial-prompt`) now have Dart bindings. New
  sticky setter
  `CrispasrSession.setWhisperDecodeExtras(suppressNonSpeechTokens:,
  suppressRegex:, carryInitialPrompt:)`. Empty regex clears
  any prior pattern (passes `nullptr` to wparams =
  whisper's "no suppression" sentinel). Pre-0.5.11 dylibs
  raise `UnsupportedError` so callers can graceful-degrade.
- Underlying C-ABI: `crispasr_session_set_whisper_decode_extras`;
  three new sticky session fields default to whisper's
  upstream values so an unmodified session matches stock
  whisper.cpp.

## 0.5.10

- **Whisper decoder-fallback thresholds** — `wparams`
  knobs that decide when the decoder falls back to a higher
  temperature pass (hard audio, low logprob) or treats a
  segment as silence are now exposed via the session API:
  `CrispasrSession.setFallbackThresholds(entropyThold:,
  logprobThold:, noSpeechThold:, temperatureInc:)`. Mirrors
  the CLI's `--entropy-thold` / `--logprob-thold` /
  `--no-speech-thold` / `--temperature-inc` / `--no-fallback`
  flags. `temperatureInc` is clamped to `[0, 1]`; setting
  `0.0` disables the temperature-fallback loop entirely
  (= the CLI's `--no-fallback`). Defaults match
  `whisper_full_default_params` so an unmodified session
  behaves bit-identical to stock whisper.cpp.
- Underlying C-ABI: `crispasr_session_set_fallback_thresholds`.
  Pre-0.5.10 dylibs raise `UnsupportedError`. Non-whisper
  backends silently ignore — none have an analog for these
  fields today.

## 0.4.9

- Initial pub.dev release.
- Dart FFI bindings for the CrispASR C ABI (`src/crispasr_c_api.cpp`).
- Supports all 17 backends: Whisper, Qwen3-ASR, FastConformer, Canary, Parakeet, Cohere, Granite-Speech, Voxtral (Mistral 1.0/4B), wav2vec2, GLM-ASR, Kyutai-STT, Moonshine, FireRed, OmniASR, VibeVoice-ASR, plus FireRedPunc post-processor.
- Unified `Session` API across all backends; legacy `CrispASR` Whisper-shaped API preserved.
- Word-level alignment, speaker diarization, and language ID helpers.
- Auto-download of registered models via the model registry.
