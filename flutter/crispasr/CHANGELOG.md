# Changelog

## 0.8.37

* **Phrase scoring:** `CrispasrSession.scoreTexts(pcm, texts, language:,
  prompt:)` returns log P(text | audio) and a token count for each candidate
  (whisper sessions), for picking the likeliest of a known set of phrases.
  All candidates are decoded in one batch.
* **Strict grammars:** `setGrammarStrict(true)` forbids end-of-text until the
  grammar can be complete, so a constrained decode no longer stops
  mid-phrase. Off by default.
* **New in the bundled native library:** Raon-Speech-9B speech-to-text,
  parakeet-ultra / parakeet-redux, and Nemotron-3-Diarization
  (`sortformer`) with streaming presets.
* **TTS:** letters followed by digits ("d4", "B52") keep their number in
  every language; German reads a lone letter by its name.

## 0.8.36

* **New ASR backends in the bundled native library:** Dolphin CN-Dialect
  (`dolphin`; Mandarin plus Chinese dialects, a two-level
  `sourceLanguage` such as `zh-SICHUAN`), X-ASR zh-en (`xasr`; streaming
  Zipformer2 transducer), Hojo-ASR-Multi-V1 (`hojo-asr`), and registry
  entries for Orukeet and Confucius4-R2T2.
* **Piano transcription** gains Onsets & Frames and hFT-Transformer
  (`onsets-and-frames`, `hft-transformer`), both auto-downloading and
  reachable through the existing piano-notes accessors.
* **Front-end fixes** in the native library change transcripts for Kaldi-fbank
  models (FireRedASR, SenseVoice, Paraformer and relatives), whose mel
  triangles are now built in mel space as the reference does.
* **Session TTS:** three defects fixed that only the session ABI (and so this
  package) could hit; see the v0.8.36 release notes for details.

## 0.8.35

* **`pianoNotesWithPrograms()`** reports the General MIDI program of each
  transcribed note — which instrument played it. `0`-`127` is a GM program,
  `128` is percussion, and `-1` means the model does not identify an
  instrument. Only MT3 fills it in; `piano-transcription` and `basic-pitch`
  report `-1` throughout, and so does any native library predating the
  export, so callers read the sentinel rather than probing for the symbol.
  MT3's multi-instrument transcription was previously flattened to a single
  part before any caller could see it.
* **Basic Pitch convolutions gain SIMD and threading** in the native library:
  3.59x on x86-64 and 2.67x on Apple Silicon at four threads, byte-identical
  output, on by default with `CRISPASR_BASIC_PITCH_FASTCONV=0` as the way
  back.
* Fixes two build-guard defects around the piano accessors: note storage
  accidentally required CREPE to be compiled in, and
  `crispasr_session_piano_notes` was guarded more narrowly than
  `crispasr_session_piano_n_notes`, so a build with MT3 or basic-pitch but
  without piano-transcription reported a note count and returned `nullptr`.

## 0.8.34

* **Breeze-TTS-2 (`bt2-tts`)** adds plain synthesis, voice cloning, and voice
  design. Its GGUFs retain the upstream non-commercial licence.
* **Linux CUDA 12 and CUDA 13 CLI packages return** after v0.8.33's release
  jobs failed to link their dynamically loaded ggml backends (#442).
* **Parakeet long audio is bounded at the encoder allocation itself** and uses
  cgroup-aware available memory, preventing the intermittent 123.6 GB request
  and NULL dereference reported on a 47.5-minute input (#441).
* **Sidon splits oversized recordings by default** instead of asking users to
  do it manually, while inputs that fit keep the same single-pass path (#431).
* M2M100/WMT21 translation now uses each checkpoint's beam-5, 200-token decode
  defaults in both CLI and library sessions, avoiding repetition collapse
  (#439). Zonos rejects unsupported voice cloning and switches per-request
  languages reliably (#435).

## 0.8.33

* **Two new TTS backends**: Supertonic-3 (`--backend supertonic`, 44.1 kHz
  non-autoregressive flow matching, 10 voices, OpenRAIL-M) and FireRedTTS3
  (`--backend fireredtts3`, zero-shot voice cloning, Apache-2.0) — closing the
  last model in #377.
* **Sidon accepts long audio** (#431). The input cap now derives from a memory
  budget instead of a hardcoded frame count, and `CRISPASR_SIDON_SPLIT=1`
  restores very long files as N exact passes cut at energy minima.
* **Zonos no longer emits noise for non-Latin scripts** (#435). A missing
  phonemizer now refuses loudly instead of synthesising a near-empty prompt,
  and the per-request language is honoured.
* **New API**: `set_voice_samples()` (#432), `backend_caps()` /
  `list_backends_with_caps()` and `detect_backends()` (#433) — the last names
  every backend that can open a given file, not just one.
* **Built-in G2P replaces espeak-ng for en/de/fr** in zonos, removing that
  GPL-3.0 dependency for those languages.
* Fixes: supertonic q4_k was unloadable; a shared CAM++ pooling defect affecting
  five backends; `voxcpm2-vae` reachable by `-m auto`.

## 0.8.32

* **Four new backends**: Raon-OpenTTS speech synthesis, Quds Persian
  recognition, and MT3 / Basic Pitch on the music surface, with Standard MIDI
  output (#250, #387). mel-band-roformer, already present on CPU, gains a
  fused GPU graph (#422).
* **Source separation runs on the GPU by default** where one is present —
  HTDemucs and the new fused mel-band-roformer graph (#413, #414, #422).
* **Nemotron `-l auto` no longer conditions on English regardless of the
  detected language**, and reads its 121-entry language table from the model,
  reaching 51 languages that were previously unnameable (#425).
* Subtitle output returns the original script instead of transliterated text,
  and Canary reports wrong-language conditioning instead of failing silently
  (#419).
* Live/streaming WebM decodes past the first 100 ms timeslice (#417), and a
  malformed sample rate can no longer amplify a small file into a
  multi-gigabyte allocation.
* Kokoro honours `--tts-speed` (#423); FunASR, piano transcription, the
  Bananamind vocoder and Sidon all get bit-identical speedups (#305, #416).

## 0.8.31

* **Pocket-TTS** now supports German, Spanish, Italian, Portuguese, and French
  through managed GGUF variants, language-aware `-m auto` routing, and the C
  session ABI (#411).
* **Chatterbox Nano** gains a Finnish checkpoint (#382), while direct T3
  KV-cache views remove redundant per-layer copies (#410).
* Streaming partials avoid repeated audio slicing and can opt into bounded
  tail decoding; CosyVoice3 enables packed Conv1d by default (#404, #406).
* GPU portability improves across Windows CUDA 13, old-CPU CUDA packages,
  Vulkan Chatterbox, Qwen3-TTS HIP fallbacks, and HTDemucs separation
  (#337, #398, #400, #402, #405).
* Silero language identification, server-side transcription progress, safer
  TTS padding, and Go speaker-turn access round out the release (#395, #408,
  #409).

## 0.8.30

* **Audio input**: files whose sample rate differs from the backend's are no
  longer resampled by linear interpolation. A 10 kHz tone decoded to 16 kHz,
  where it must vanish, previously survived at -10.3 dB and folded down into
  the speech band; every 44.1/48 kHz recording carried that alias. Decoding now
  happens at the file's own rate and resamples with the Kaiser-windowed
  polyphase sinc (-89 dB on the same test). `CRISPASR_HQ_RESAMPLE=0` restores
  the old path.
* **GPU binaries**: the v0.8.29 GPU artifacts were built with `-march=native`
  against a runner that has AVX-512, so the official Windows CUDA build died
  with SIGILL on any CPU without it (#374). Nine of eleven GPU jobs were
  affected. Replace any v0.8.29 GPU build.
* A build/host CPU ISA mismatch now fails fast with a message naming the right
  download, instead of an illegal-instruction fault the Windows console
  swallows (#380).
* **Windows**: cached GGUFs larger than 2 GiB were reported missing and
  re-downloaded on every run, because MSVC's `stat` uses a 32-bit size field
  and fails outright above 2 GiB (#393).
* Punctuation restoration returned an empty string — not degraded output,
  nothing — for every SentencePiece `fireredpunc` model (`fullstop-punc`,
  `punctuate-all`).
* VibeVoice-ASR answered in the wrong language: the input was never normalised
  to -25 dBFS before the VAE encoders, and the 1.5B model was sent the 7B's
  prompt format (#369). The model's own `[Silence]` marker no longer leaks into
  transcripts and SRT files.
* Canary long-form seam artifacts are gone — the real dynamic chunking from
  canary-1b-v2 replaces the parakeet-shaped machinery (#375), with opt-in seam
  dedup at chunk boundaries (#365).
* New backend: **Confucius4-TTS** (#377), with native voice cloning via
  `--voice`.
* `/v1/realtime` Nemotron sessions are genuinely incremental — the stream now
  owns frontend, encoder and predictor state across appends instead of
  recomputing the whole buffer every 500 ms (#383).
* New endpoint: `POST /v1/audio/separation` (#381).
* Bare voice names resolve against `--voice-dir` in five TTS adapters, over
  HTTP as well as the CLI (#384).
* Long-form progress is reported on the chunk-encoded, JA-sliced and unified
  dispatch routes, not only the common one (#385).
* Diarization forwards FoxNose speaker turns across the C ABI (#395), and the
  minimum speaker count is clamped to the distinct pyannote tracks (#368).
* f5-tts counts UTF-8 characters, not bytes, in its duration estimate (#372);
  kyutai `stt-1b-en_fr` no longer claims to be English-only (#366); omnivoice
  matches upstream's target-token arithmetic exactly (#363).
* `--align-only` accepts JSON input, for a JSON-to-JSON pipeline (#317).

## 0.8.29

* **Dart binding**: `inputSampleRate` and `outputSampleRate` are now bound
  (#321). They had been added to the five bindings that lacked
  `speechToSpeech`, which skipped the three that already had it — so Dart kept
  s2s while never gaining the getter that says what rate to feed it. Both
  return 0 on a dylib that predates the symbol, matching `separateSampleRate`.
* **Chat**: cancellation and prompt-token counting are bound, plus four defects
  fixed in the Dart chat surface (#361, #362). A session now waits for its
  in-flight calls before freeing, so tearing one down mid-generation is safe.
* `--diarize-method pyannote` never worked out of the box: the managed download
  for its segmentation model was tagged with the licence string `"other"`, which
  the registry treats as restricted, so it refused to fetch and produced no
  speaker turns. It is MIT and ungated.
* Diarization now returns speaker labels through `verbose_json`, which
  previously dropped them entirely, and embeds segments across workers
  (1.6–2.0x on the embed stage).
* Parakeet long-form dropped whole spans of 30–300 s audio (#350) and could
  emit segments out of time order (#356); both fixed, and long-form throughput
  is 2.1x (#353).
* Every GGUF load leaked its weight mapping — `MAP_PRIVATE` with write
  permission, so merely reading the weights privatized the pages.
* CosyVoice3 cloning works through the session API when the reference clip has
  been prepared once through the CLI (#334).
* Windows CUDA packages now ship in split form, without the three NVIDIA
  runtime DLLs (#342) — 296 MB instead of 873 MB for the dev-lib package.
* There is a Dart binding CI job, so this package is compiled on every push.

## 0.8.28

* **HIP/ROCm on Linux**: first release since 0.8.25 with a
  `crispasr-linux-x86_64-hip` tarball. The packaging step rewrote `RUNPATH`
  before asking `ldd` what the binaries needed, so ROCm's OpenMP runtime
  (`libomp.so`, reachable only through that `RUNPATH`) was silently dropped and
  the archive never built (#339).
* The same defect in two more artifact kinds, neither reported (#341):
  `libcrispasr-linux-x86_64-hip` shipped needing an unbundled `libomp.so`, and
  the Python binding tarballs needed `libgomp.so.1` and `libblas.so.3` — so
  `import crispasr` failed in the loader on any host without OpenBLAS and gcc's
  OpenMP. Both are now bundled and gated.
* GPU archives no longer copy the build machine's CUDA/ROCm install into the
  tarball, and carry those toolkit directories in their own `RUNPATH` instead —
  so they resolve without the `/etc/ld.so.conf.d` post-install step.
* CLI tarballs now ship `LICENSE` and `THIRD_PARTY_NOTICES.txt`, which now also
  declare the bundled OpenMP runtimes (`libgomp`, `libomp`).

## 0.8.27

* **Linux users on 0.8.26 should upgrade**: that release published only one of
  its seven Linux binary tarballs (plain x86_64, arm64, CUDA, CUDA 13, Vulkan
  and HIP all failed to build). Two shell bugs in the release workflow, fixed
  (#339).
* Fixed a null-pointer crash in the audio loader on a malformed Ogg file: a
  Vorbis comment header declares its entry count before the array is
  allocated, so an attacker-sized count left a non-zero length with a null
  pointer that the teardown path then indexed. Reachable from
  `crispasr_audio_load` on untrusted input.
* qwen3-tts: `--temperature` now reaches the talker (it had only ever reached
  the code predictor), plus greedy/replay/logit-dump levers for cross-backend
  diagnosis (#337).

## 0.8.26

* CosyVoice3 voice cloning was conditioned on the wrong speaker embedding
  (cosine 0.737 against upstream's CAMPPlus ONNX): the export folds
  `out_nonlinear`'s BatchNorm into the preceding convolution and the fused bias
  was dropped. Now 0.999997. Baked bank voices were never affected (#334).
* CosyVoice3 `--ref-text` is now optional — the reference is auto-transcribed
  and cached — and the decode has upstream's minimum-length floor, so it can no
  longer end at step 0 with no audio (#334).
* New CosyVoice3 RL talker: `--backend cosyvoice3-tts-rl` (#334).
* Voxtral TTS could index its embedding table out of bounds on some inputs: a
  Tekken vocabulary blob may serialize more pieces than the checkpoint
  activates. Bounded, in both `voxtral-tts` and `voxtral4b` (#338).
* qwen3-tts could emit 300 s of audio for one sentence — the frame budget was
  the KV cache ceiling rather than anything derived from the input text (#337).
* madlad400 F16 and Q8_0 artifacts published; F16 verified at cosine 1.000000
  on all 14 stages against the PyTorch reference (#333).
* Kokoro punctuation was being discarded, in German, French and Spanish too,
  and the contextual G2P rules had shipped switched off (#316).
* `-tl` / `--target-lang` was silently discarded by cosyvoice3 and omnivoice
  (#329).
* Rust and Dart diarize ABI mirrors were 24 bytes short; FoxNose exposed, plus
  `session_output_sample_rate` and channel getters (#332).

## 0.8.25

* New ASR backend: GigaAM-v3 (Russian, CTC + RNN-T, punctuation-native `e2e` heads).
* New diarization method `foxnose` — WeSpeaker embeddings + spectral clustering,
  no external binary required.
* Diarization correctness: the pyannote powerset decode table had two entries
  swapped (48.21% -> 33.37% DER), and `--diarize-max-speakers` was picking the
  speaker count instead of bounding it (15.74% -> 7.81% DER).
* Diarization speed: pyannote segmentation now infers in parallel chunks
  (2888 s file, 8 cores: ~50 s -> 18.1 s); speaker embedding parallelised.
* Fixed sherpa diarization hanging indefinitely (#328) and the MSVC build
  failure in glint (#327).
* ggml synced to v0.17.0.

## 0.8.24

- **`Session.setTtsPhonemes(String)`** — synthesize a phoneme string verbatim,
  skipping the grapheme-to-phoneme stage. Honoured by `kokoro` and `piper`;
  other backends throw rather than silently synthesizing the text instead.
- **`SessionSegment.speaker`** — the native per-segment speaker label, in the
  `"(Speaker N) "` form, or `""` when the backend does not diarize natively.
  The ordinals are chunk-local: `Speaker 1` from one `transcribe` call is not
  necessarily the same voice as `Speaker 1` from the next.
- Native improvements that ship with this release (no Dart API change):
  - **English pronunciation for Kokoro** substantially fixed — numbers were
    silently dropped from synthesized text, and the phonemes were in the wrong
    alphabet for the model. Agreement with Kokoro's own G2P goes from 58% to
    99.1%.
  - **MOSS-TTS 4B** no longer generates past its stop token.
  - **CosyVoice3 cross-lingual synthesis** — a clone reading a target language
    no longer carries the reference voice's accent.
  - **VibeVoice** speaker turns and per-utterance timings are now parsed out of
    the model's answer instead of being returned as one raw JSON blob.
  - Punctuation restoration no longer double-capitalizes or doubles a full stop.

## 0.8.23

- No Dart API change. Ships the native v0.8.23 improvements: the restored
  Windows CPU binary, F5-TTS Chinese synthesis, VibeVoice voice-pack
  correctness, broader NVIDIA GPU coverage, faster VAD, the opt-in
  `--strict-pipeline` mode, and several TTS-on-Vulkan fixes.

## 0.8.22

- **TTS provenance opt-out now requires a marking attestation.** Synthesis is
  provenance-marked by default; raw (unmarked) synthesis and an explicit
  marking-responsibility affirmation are exposed for callers that take
  responsibility for labeling AI-generated audio. Marked-output flows are
  unchanged.
- Native improvements that ship with this release (no Dart API change):
  - **Tiron** — a new multi-speaker meeting-ASR model (Whisper large-v3 + inline
    speaker markers, `--diarize` cross-window speaker linking) is reachable via
    `CrispasrSession.open()` on the whisper path.
  - **Source separation ~26× faster** on Linux/Windows — `CrispasrSession.separate()`
    with mel-band-roformer no longer runs the naive scalar path (BLAS + FFT iSTFT
    + threading), output bit-identical.
  - **NaN-robust ASR decode** — canary-qwen and 7 other backends no longer emit
    garbage when a quantized weight goes non-finite.
- No breaking Dart API changes beyond the provenance gate above.

## 0.8.21

- **`--max-new-tokens` is honored across all ASR backends.** Ten backends
  (moss-diarize, canary, canary-qwen, glm-asr, funasr, mimo-asr, moss-transcribe,
  mini-omni2, higgs-stt, higgs) previously hardcoded their decode cap, so a long
  single-pass transcription truncated. Each now forwards the value while keeping
  its own default, so nothing regresses.
- **MOSS TTS synthesis params wired** — moss-tts / moss-tts-local now honor
  max-new-tokens, duration (max-speech-tokens), top-p, top-k, and
  repetition-penalty instead of staying at hardcoded defaults.
- **Chunk-local speaker scope** — diarized segments now carry a `chunk_id`;
  `(Speaker N)` labels are chunk-local, so a consumer uses `chunk_id` to tell
  continuity from an ID swap.
- No Dart API changes.

## 0.8.20

- **iOS is now supported.** The release ships an Apple `crispasr.xcframework`
  (device + simulator slices for iOS, plus macOS/tvOS/visionOS) for the first
  time, so a Flutter iOS app can link the native library. Earlier releases
  shipped no iOS artifact at all.
- **Prebuilt native library bundles load as delivered.** The macOS and Linux
  `libcrispasr` bundles had a broken embedded run-path and could not be
  `dlopen`ed after extraction; they now resolve their bundled `ggml` libraries
  relative to their own location. Android and the desktop bundles are laid out
  as a flat `lib/` directory.
- **canary-qwen long audio fixed.** Clips longer than ~30 s no longer skip
  chunking, return near-empty transcripts, or grow memory without bound — the
  backend had advertised an internal chunker it did not implement (#290).
- No Dart API changes; existing code upgrades untouched.

## 0.8.19

- Beat and downbeat tracking: `CrispasrSession.beats()` returns a beat grid
  from mono 22050 Hz float32 PCM, backed by the Beat This! backend. Adds the
  `Beat` record typedef (`timeS`, `isDownbeat`), a `beatsSampleRate`
  capability probe and a `beatsTempoBpm` median-interval tempo estimate.
  Previously beat tracking was reachable only from the CLI and the C ABI.
- Every downbeat is also reported as a beat: the postprocessor snaps each
  downbeat onto its nearest detected beat, so downbeats are a strict subset —
  filter on `isDownbeat` for the bar grid, and never merge two lists.
- Beat This! is **MIT for code and weights** and uses **no DBN** (postprocessing
  is peak-picking only), so unlike most beat trackers this arm carries none of
  madmom's patent-encumbered, non-commercially-licensed machinery.
- The native GGUF architecture auto-detect recognises `beat-this`, so plain
  `CrispasrSession.open()` works for beat models; `backend: 'beat-this'`
  remains valid and is required against an older native library.

## 0.8.17

- Source separation: `CrispasrSession.separate()` splits interleaved stereo
  PCM into named stems (`drums`/`bass`/`other`/`vocals` for htdemucs), with
  the `Stem` record typedef and a `separateSampleRate` capability probe.
  Previously separation was reachable only from the CLI, the C ABI and
  Python — never from Dart.
- Note the input contract: `separate()` takes **interleaved** stereo
  (`L,R,L,R,…`) at 44100 Hz, and the native side counts samples *per
  channel*. To feed a stem to `pitch()`, downmix to mono and resample to
  16 kHz first — the dartdoc has the recipe.
- Piano transcription: `CrispasrSession.pianoNotes()` returns structured
  note events — `PianoNote = ({int midi, double onMs, double offMs, int
  velocity})` — plus a `pianoSampleRate` probe. Previously the only route
  out of the piano backend was `transcribe()`, whose segment text reads
  like `"C4 v=80"`; parsing that back was lossy.
- Native fixes that ship with this release: htdemucs now segments long
  audio (7.8 s windows, 25% overlap, weighted overlap-add), so separating a
  full song no longer needs tens of GB — peak memory is flat instead of
  growing with length. And the htdemucs loader no longer silently zero-fills
  quantized tensors it cannot read, which had made the q4_k model produce
  garbage; unreadable weights are now fatal.

## 0.8.16

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
