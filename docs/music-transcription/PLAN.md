# Music transcription in CrispASR

Porting the CometBeat / mus-textbook "transcription → SOTA" model roster
(`docs/TRANSCRIPTION_SOTA_HANDOFF.md` in that repo) from ONNX to CrispASR
ggml/GGUF backends.

## NOW — active work

- **Done**: feasibility triage of all 7 handoff workers (below); fixed a live
  `CAP_SEPARATE`/`CAP_STREAMING` bit collision (both were `1u << 22`).
- **In flight**: nothing — awaiting go/no-go on the phase-0 infra.
- **Next**: `core/stft.h` extraction, then CREPE (`src/crepe.cpp`) as the first
  music-analysis backend.
- **Branch**: `feat/music-transcription`, worktree
  `.claude/worktrees/music-transcription`.

---

## Verdict: yes for the neural models, no for two of the seven

The handoff lists 7 workers. They are not the same kind of thing — four are
neural models (a CrispASR port makes sense), two are pure score-level algorithms
(they belong in Dart), and one is already shipped here.

| Worker | CrispASR? | Status / why |
|---|---|---|
| **W-SEP** | ✅ **already done** | HTDemucs (`src/htdemucs.cpp`, §248 full parity, `cstr/htdemucs-GGUF`) + Mel-Band RoFormer (`src/mel_band_roformer.cpp`, waveform bit-exact 2.4e-7). Both shipped with `--separate`, auto-download, C ABI, Python `Session.separate()`. **Don't export Open-Unmix to ONNX — call CrispASR.** |
| **W-CREPE** | ✅ port — start here | 6-layer 1D CNN on raw 16 kHz audio → 360-bin activation. No STFT, no attention, MIT. The single easiest port in the repo's history. |
| **W-PIANO** (slice 1) | ✅ port | Kong/ByteDance high-res piano CNN + biGRU on log-mel. `core/mel.h` covers the front-end; needs a **GRU** in `core/` (only LSTM exists today). |
| **Basic Pitch** | ✅ port | Already ONNX in the app; Apache-2.0, ~4 MB CNN over a harmonic-CQT stack. Needs a **CQT** front-end (absent). |
| **W-HARMONY** | ⚠️ port, licence-gated | Small CRNN/CQT chord model. Architecture is easy; the work is finding a checkpoint whose **licence** is actually permissive. Timebox the checkpoint hunt before the port. |
| **W-DRUMS** | ⚠️ mostly DSP | Onset + band-energy classification is DSP, and DSP belongs where the app is. Only worth a backend if a permissive drum-transcription CNN is chosen. |
| **W-MT3** (slice 2) | ⚠️ frontier, timebox | T5 encoder-decoder over spectrogram frames → MIDI-like tokens, Apache-2.0. The *architecture* is well-trodden in ggml (easier than ONNX, honestly). The risk is the **checkpoint format** — T5X/JAX gin, not HF safetensors — so the converter is the whole job. Feasibility memo before committing. |
| **W-METRE** | ❌ **not CrispASR** | Downbeat DP + metrical quantisation. No model, no tensors. Pure algorithm over a `RhythmGrid`. Keep in Dart. |
| **W-NOTATION** | ❌ **not CrispASR** | Voice separation, staff split, enharmonic spelling — operates on `crisp_notation` score types, not audio. Keep in Dart. |

So: **5 of 7 are worth porting, 1 is already done, 2 should stay in Dart.**

### Why port at all, given ONNX works

1. **W-SEP is the handoff's "biggest lever" and it already exists here**, at
   higher quality than the Open-Unmix fallback the handoff proposes, with
   per-stage cosine parity already validated. That alone justifies the seam.
2. **One runtime for the whole chain.** Separation → F0 → notes currently means
   ONNX Runtime *plus* whatever runs the stems. CrispASR already owns the audio
   IO, resampling, chunking, and model auto-download.
3. **Quantization.** `crispasr-quantize` gives q8_0/q4_k for free; these models
   ship as f32 ONNX. CREPE-full at q8_0 is a phone-sized model.
4. **Metal / CUDA / WASM** come from ggml, not from a per-model ONNX EP story.

The counter-argument is honest and worth stating: for **Basic Pitch and CREPE
specifically**, ONNX already works in the app today, and porting buys speed and
packaging, not capability. The capability wins are W-SEP (done), piano, and MT3.

---

## Architecture: a new task surface, not a `transcribe()` overload

`docs/source-separation-surface.md` already settled this argument for stems: a
task that returns something other than `crispasr_segment`s must **not** be
layered onto `transcribe()`; it gets its own early dispatcher before the ASR
backend is constructed. Music transcription (audio in → note events out) is the
same shape, so it copies that design:

- `src/core/note_events.h` — the result surface, mirroring
  `src/core/separation_io.h` (header-only, unit-testable without linking a
  backend). Carries the Dart-side seam types: `{midi, onMs, offMs, confidence}`
  note events, `{timeMs, f0Hz, voicedProb}` pitch frames.
- `examples/cli/crispasr_music_cli.{h,cpp}` — early route, mirroring
  `crispasr_separate_cli.{h,cpp}`, hooked once from `cli.cpp`.
- `CAP_MUSIC_TRANSCRIBE = 1u << 24` (bit 23 now belongs to `CAP_STREAMING`
  after the collision fix; 22 stays `CAP_SEPARATE`).
- A MIDI writer in `core/` so the CLI can emit `.mid` directly. MusicXML
  engraving stays in Dart — that's `crisp_notation`'s job, not a C runtime's.

**Contract compatibility.** The handoff freezes `contracts.dart`
(`PitchFrame` / `NoteEvent` / `RhythmGrid`). `core/note_events.h` is designed to
be a 1:1 memory-layout match so the Dart FFI binding is a reinterpret, not a
marshal. That is the whole point of the seam — an engine swaps behind it.

---

## Phase 0 — infrastructure (blocks everything else)

The survey turned up three real gaps. None is hard; all are prerequisites.

1. **`core/stft.h` — forward STFT.** `core/istft.h` exists but covers only the
   inverse. HTDemucs rolls its own (`src/htdemucs.cpp:548` `compute_stft`) and
   mel-band-roformer has a second copy. A music backend would be the **third**
   copy. Extract now, before adding to the pile.
   ⚠️ This refactors two *shipped* backends → per the A/B rule, it needs
   byte-identical stem output on both before it lands, gated if not.
2. **`core/cqt.h` — constant-Q / harmonic-CQT.** Absent entirely. Basic Pitch
   and every chord model want log-frequency bins. Built on (1).
3. **`core/gru.h`.** `core/lstm.h` has uni/bidirectional LSTM; the piano model
   needs biGRU. Mirror the LSTM file's structure.

Ordering: (1) → (3) can proceed in parallel with CREPE, which needs neither.

## Phase 1 — CREPE (recommended first backend)

Why first: it needs **zero** new infrastructure. Raw 16 kHz waveform in
(1024-sample frames), 6 conv+batchnorm+maxpool blocks, one 360-unit dense layer
out. No STFT, no attention, no autoregression, no tokenizer. It exercises the
entire new music surface end-to-end — CLI flag, capability bit, note-event
result type, converter, registry, C ABI, bindings — against the simplest
possible model, which is exactly how you want to debug a new surface.

- `models/convert-crepe-to-gguf.py` — from the MIT Keras/`torchcrepe` weights.
- `src/crepe.{h,cpp}` + the 12-point checklist in `docs/contributing.md`.
- Parity: `tools/reference_backends/crepe.py` → `crispasr-diff crepe`, per-stage
  cos ≥ 0.999 vs `torchcrepe`.
- **Acceptance is the decoded output, not cosine** (HARD RULE #3): synth a
  C-major scale → `crepe` → note segmentation → note-F ≥ 0.9 with **zero octave
  errors**, which is the specific failure the handoff wants fixed.

## Phase 2+ — piano, Basic Pitch, harmony, drums, MT3

Sequenced after phase 1 proves the surface. Each follows the same regime:
blueprint read line-by-line → converter → per-stage diff → decoded-output gate →
registry + 12-point checklist. MT3 gets a feasibility memo (checkpoint
conversion viability) **before** any C++ is written.

---

## Open questions

- **Where does the app call this from?** CrispASR has Dart/Flutter bindings, so
  the seam can be FFI. But the handoff's engines are `!kIsWeb`-guarded with a
  pure-Dart web fallback — CrispASR's WASM build could actually *remove* that
  caveat. Worth confirming with the app author before designing the binding.
- **Model hosting.** Existing convention is `cstr/<name>-GGUF` on HF with a
  `license:` YAML tag that must be verified post-upload. CREPE (MIT) and Basic
  Pitch (Apache-2.0) are clean; the chord checkpoint is the one to vet.
