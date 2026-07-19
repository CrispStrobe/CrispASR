# Music transcription in CrispASR

Porting the CometBeat / mus-textbook "transcription → SOTA" model roster
(`docs/TRANSCRIPTION_SOTA_HANDOFF.md` in that repo) from ONNX to CrispASR
ggml/GGUF backends.

## NOW — active work

- **Done**: feasibility triage of all 7 handoff workers (below); fixed a live
  `CAP_SEPARATE`/`CAP_STREAMING` bit collision (both were `1u << 22`; CLI builds
  clean after the move to bit 23). **CREPE converter landed and validated**:
  `models/convert-crepe-to-gguf.py` + `tools/crepe_numpy_parity.py`, cos=1.0 vs
  torchcrepe on both capacities (tiny 1.0 MB, full 44.5 MB at f16).
- **Done**: `src/crepe.{h,cpp}` runtime — **cos = 1.0 vs the numpy spec** on a
  real tone sweep, decoding 220.6 / 440.4 / 881.4 Hz at 0.95–0.97 confidence.
  `tests/test_crepe_parity.cpp` is the acceptance gate.
- **Done**: CREPE wired through the 12-point checklist — `CAP_PITCH = 1u << 24`,
  `examples/cli/crispasr_backend_crepe.cpp` (redirect shim, mirroring the
  htdemucs one), the `--pitch` early dispatcher
  (`examples/cli/crispasr_pitch_cli.{h,cpp}`, mirroring `--separate`), factory /
  roster / arch auto-detect (`crepe`) / filename heuristic in
  `crispasr_backend.cpp`, the session C ABI (`crispasr_session_pitch*` in
  `src/crispasr_c_api.cpp` + `include/crispasr_session.h`), registry entries for
  `cstr/crepe-GGUF` (**tiny is the default**), CMake linkage, README + docs/cli.md.
- **In flight**: nothing.
- **Next**: upload `cstr/crepe-GGUF` (the registry URLs point at it but the repo
  is not published yet); quantize (q8_0/q4_k) and re-measure; then the Dart FFI +
  WASM surfaces. `core/stft.h` extraction is independent (CREPE needs no STFT).

### Performance — measured, M1, quiet box (load 4.0), 10 s audio, median of 3

| model | Metal | CPU |
|---|---|---|
| full (44.5 MB f16) | 20.0 s — **RTF 2.0** | ~400 s — RTF 40 |
| tiny (1.0 MB f16) | 2.8 s — **RTF 0.28** | ~24 s — RTF 2.4 |

CREPE is genuinely expensive: at the reference 10 ms hop it is **1409 MMAC per
frame → 282 GFLOP per second of audio** for `full`, and 36.7 MMAC/frame →
7.3 GFLOP/s for `tiny` (38× cheaper). So **tiny is the shipping default** — it
is also what the handoff asks for ("smallest that hits accuracy"). `full` stays
available and is the right choice offline. Neither is close to real-time on CPU;
the GPU path is not optional here.

Three graph decisions got it from the first working version (RTF 31) to here:

1. **Batching (the big one).** One frame per dispatch wastes the GPU on a model
   this small per-frame. `kBatch = 64` makes each layer one large GEMM.
2. **Channel-fastest layout throughout.** `ggml_conv_1d` ends by permuting back
   to (OL, OC, N), materializing the whole activation every layer. We keep the
   mul_mat's native (OC, OL, N), do bias/relu/BN there — where a plain (OC)
   vector broadcasts along ne[0], ggml's fast path, instead of a stride-0
   (1, OC, 1) broadcast — and pool with `ggml_pool_2d(k0=1, k1=2)`. The one
   transpose im2col forces is deferred until *after* the pool, so it moves half
   the bytes, and the last layer skips it entirely because (OC, OL, N) already
   *is* the channel-fastest flatten the classifier wants.
3. **F32-baked conv kernels** (`ggml_conv_1d` casts an F16 kernel to F32 inside
   the graph — in a persistent graph that re-casts 44 MB per 10 ms frame).
   Gated `CRISPASR_CREPE_NO_BAKE_F32=1`. Honest note: this one measured
   **neutral** here, unlike qwen3-tts CODEC_FASTCONV. Kept gated-on because it
   is provably redundant work, but it was not the win.

Gates: `CRISPASR_CREPE_NO_GPU=1`, `CRISPASR_CREPE_NO_BAKE_F32=1`,
`CRISPASR_CREPE_DEBUG=1`.

### `ggml_conv_1d` returns a tensor whose declared shape contradicts its data for N > 1

**Status: fixed in the fork (`ggml/src/ggml.c`), upstream PR drafted at
`tools/upstream-prs/24-conv-1d-batch-reshape.md` + a standalone repro. NOT yet
merged to main — one audit item is open, see below.**

The im2col is the FIRST `ggml_mul_mat` argument, so the result's ne is
`[N*OL, OC]` (OC slowest). The final `ggml_reshape_3d` declares `[OL, OC, N]`
(N slowest). Those expressions coincide **exactly when N == 1** and differ
otherwise — which is why every shipping caller is correct and this was invisible.

Repro (`tools/upstream-prs/24-conv-1d-batch-reshape.repro.cpp`, standalone,
vs a hand-rolled direct convolution), before the fix:

```
N=1  cos=1.00000000  OK        N=2  cos=0.41129104  MISMATCH        N=3  cos=0.05935857  MISMATCH
```

After: all three `cos=1.0`. Fix reshapes to the true `[OL, N, OC]` then permutes;
the `N == 1` branch is the *unmodified original statement*, so batch-1 callers
are bit-identical **by construction**, not merely by test.

Corroborating facts:

- **Upstream `llama.cpp` has byte-identical code.** Not a fork regression, and
  not fixed upstream.
- **Upstream `test-backend-ops.cpp` has ZERO `conv_1d` cases.** It covers
  `IM2COL` and `MUL_MAT` as ops, but `ggml_conv_1d` is a composite graph
  builder, so the reshape between them is untested. That is the mechanism by
  which this survived.

#### ✅ AUDIT COMPLETE — landed in the fork (`CrispStrobe/ggml@662b05fb`)

The open question was whether any existing caller passes N > 1. Answered, and
**my original safety argument was wrong**:

- **CrispEmbed: zero `ggml_conv_1d` callers.** Unaffected entirely. (Its only 1-D
  conv use is two `ggml_conv_1d_dw` calls, a different function, both N == 1.)
- **CrispASR: 141 call sites** — 11 more than my `grep` found, because
  `ggml_conv_1d_ph` forwards to `ggml_conv_1d` without matching the literal
  string. **136 pass N == 1.** **2 pass N > 1.** 0 unknown.

The two batched callers are `aa_snake_beta_native` in `src/indextts_voc.cpp`
(:508 and :551), which deliberately maps **channels onto the batch axis** so one
depthwise FIR runs across all C channels at once. So "the N == 1 branch is
unmodified, therefore every caller is bit-identical" was **false** — those two
take the new branch.

They are safe for a *different* reason: their filter is `[K,1,1]`, i.e.
**OC == 1**, and with OC == 1 both branches produce the identical flat layout
`n*OL+ol` *and* the identical declared `ne`. Confirmed from the source (the
shape is documented at `indextts_voc.cpp:459-460` and enforced by a downstream
`ggml_reshape_2d` nelements assert) and verified empirically on that exact shape
class at N = 1..4. Neither site compensates for the old transpose, so nothing
depended on the broken layout.

**The branches diverge only when N > 1 AND OC > 1** — which no caller in either
repo does. CREPE would have been the first, which is why it surfaced here.

Gates run: standalone repro (both shape classes, all N) cos = 1.0; CrispASR unit
suite **1032/1032**; CREPE parity unchanged at cos = 1.0.

**Related, not fixed:** `ggml_conv_1d_dw` ends with
`ggml_reshape_3d(..., result->ne[0], result->ne[2], 1)` — it hardcodes `1` into
`ne[2]`, so it is the same bug class and is only correct for N == 1. No current
caller in either repo passes N > 1 to it. Worth fixing in the same upstream PR.

### Two measurement traps hit while benchmarking (both in the dev doc already)

- A run piped to `head -2` reported **0.79 s for 30 s of audio** (RTF 0.026,
  which would have been ~10 TFLOP/s — above M1's FP32 peak). SIGPIPE had killed
  it after two lines. The "too good, and the arithmetic disagrees" smell is what
  caught it; the frame-count-scales check (101 / 1001 / 3001) is what confirmed
  the real runs.
- Load average hit **253** mid-session, making every timing meaningless. Numbers
  above were all re-taken at load 4.0.
- **Branch**: `feat/music-transcription`, worktree
  `.claude/worktrees/music-transcription`.

### CREPE blueprint — the geometry the C++ must hit

Traced from `torchcrepe/model.py` + `core.py` + `convert.py` (the *source*, see
the warning below). Input is a 1024-sample 16 kHz frame, per-frame normalized
(`-= mean`, `/= max(std, 1e-10)`); hop is 10 ms; `pad=True` zero-pads
`WINDOW_SIZE//2` each edge.

Per layer: `F.pad -> conv -> F.relu -> batch_norm -> max_pool2d(2)`.

| layer | K | stride | pad (l, r) | out ch (full / tiny) | T out |
|---|---|---|---|---|---|
| conv1 | 512 | 4 | 254, 254 | 1024 / 128 | 1024 → 256 → 128 |
| conv2 | 64 | 1 | 31, **32** | 128 / 16 | 128 → 64 |
| conv3 | 64 | 1 | 31, **32** | 128 / 16 | 64 → 32 |
| conv4 | 64 | 1 | 31, **32** | 128 / 16 | 32 → 16 |
| conv5 | 64 | 1 | 31, **32** | 256 / 32 | 16 → 8 |
| conv6 | 64 | 1 | 31, **32** | 512 / 64 | 8 → 4 |

Then permute to (T, C) — **C is the fast axis** — flatten to `in_features`
(4 × 512 = 2048 full, 4 × 64 = 256 tiny), `classifier` Linear → 360, sigmoid.
Decode: `cents = 20 * bin + 1997.3794084376191`, `Hz = 10 * 2**(cents/1200)`.

Three traps, all now pinned by `tools/crepe_numpy_parity.py`:

1. **ReLU is BEFORE BatchNorm.** So the conv+BN fold is *invalid*. BN ships as a
   standalone per-channel affine (`_BN.scale`, `_BN.offset`, computed in f64).
2. **conv2..6 padding is asymmetric (31, 32)** and Metal rejects an asymmetric
   `GGML_OP_PAD` — use symmetric `p=32` and drop output column 0.
3. **`torchcrepe.convert.bins_to_cents` applies dithering** (triangular noise),
   so the reference is *non-deterministic*. Disable it when dumping parity
   fixtures, and do not implement it in C++. Also note torchcrepe's default
   decoder is **Viterbi**, not the handoff's weighted-average-around-argmax —
   implement `local_average` (original CREPE) and treat Viterbi as optional.

> ⚠️ **Lesson (HARD RULE #1, the expensive way).** The first converter folded BN
> into the conv, because a fetched *summary* of `model.py` listed the ops as
> "Batch Norm ... ReLU activation" in that order. The real source has the relu
> first. The failure looked like plausible numerics, not a structural bug: layer
> 1 at cos=0.83 with ~2× the reference magnitude — because least-squares fitting
> an affine through a *rectified* signal recovers about half the true scale. What
> caught it in one run was printing `|mine|` and `|ref|` per stage and noticing
> `|mine|` was **identical across four different input frames**. A fetched
> summary of source is not reading the source.

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
