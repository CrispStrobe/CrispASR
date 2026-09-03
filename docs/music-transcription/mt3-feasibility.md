# MT3 feasibility check (PLAN.md §250 / §251 open item)

Research-only, 2026-08-30. All facts below verified against source or the live
GCS bucket, not from memory. **Two claims currently in the repo are wrong** — see
"Corrections" at the end.

## 1. Architecture (verified from `mt3/gin/model.gin` + `mt3/network.py`)

Encoder-decoder T5 1.1-flavoured, `models.ContinuousInputsEncoderDecoderModel`:

- `emb_dim=512`, `num_encoder_layers=8`, `num_decoder_layers=8`, `num_heads=6`,
  `head_dim=64` (→384 qkv), `mlp_dim=1024`, `mlp_activations=('gelu','linear')`
  = **gated-GELU**, ~60 M params.
  ([model.gin](https://raw.githubusercontent.com/magenta/mt3/main/mt3/gin/model.gin),
  [paper](https://arxiv.org/pdf/2111.03017))
- Positional: **`layers.FixedEmbed` sinusoidal ABSOLUTE, added to the embeddings
  — NOT T5 relative-attention buckets**
  ([network.py:180,225](https://raw.githubusercontent.com/magenta/mt3/main/mt3/network.py)).
  Confirmed by the checkpoint: 147 zarr params, zero `relative_attention_bias`.
- Encoder input: continuous 512-d log-mel frame → `continuous_inputs_projection`
  Dense(512→512); no encoder token embedding (`PassThroughVocabulary(size=0)`).
- Logits: untied `logits_dense` (512×1536), separate from `token_embedder`.

Front end (`mt3/spectrograms.py`): 16 kHz mono, `FFT_SIZE=2048`,
`hop_width=128` → **125 fps**, `num_mel_bins=512`, `MEL_LO_HZ=20.0`, via DDSP
`spectral_ops.compute_logmel` (magnitude mel, `safe_log`, tf.signal mel matrix).

Segmentation: `mt3.gin` `TASK_FEATURE_LENGTHS = {'inputs': 256, 'targets': 1024}`
→ 256 frames = **2.048 s non-overlapping segments**, decoder cap 1024 tokens.
(`ismir2021.gin` piano model: inputs 512 = 4.096 s.)

## 2. Vocabulary (verified from `mt3/vocabularies.py` + `event_codec.py`)

`build_codec()` event ranges, in order: `shift` 0..1000 (100 steps/s, 10 s max),
`pitch` 0..127, `velocity` 0..N, `tie` 0..0, `program` 0..127, `drum` 0..127.
Plus 3 specials (0=PAD, 1=EOS, 2=UNK) and 100 `extra_ids`.

- **mt3 (multi-instrument)**: `NUM_VELOCITY_BINS=1`, `PROGRAM_GRANULARITY='full'`,
  `USE_TIES=True` → 1388 classes + 3 + 100 = 1491, padded to **1536** (confirmed:
  `token_embedder.embedding` shape `[1536,512]`).
- **ismir2021 (piano)**: `NUM_VELOCITY_BINS=127`, `'flat'`, `USE_TIES=False` →
  padded to **1664**.

Decoding is **greedy autoregressive** (paper §3). Notes crossing a segment
boundary are carried by the "tie section": each segment begins by re-declaring
active pitches until an `end_tie_section` token.

## 3. Checkpoints — availability, format, size (verified live)

Public, anonymous-readable GCS bucket `gs://mt3/checkpoints/{mt3,ismir2021}`:

```
checkpoints/mt3:       296 objects, 171.6 MB   (147 params)
checkpoints/ismir2021: 296 objects, 172.0 MB   (147 params)
```

Format, confirmed by fetching the objects:

- Each large param is a **zarr v2 directory**: `.zarray` JSON + a single
  gzip-level-1 chunk `0.0`, `dtype "<f4"`, `order C`. Decoded
  `continuous_inputs_projection.kernel/0.0` end-to-end with `gzip` +
  `np.frombuffer` → `(512,512) float32`. **Works.**
- Small params (42 LayerNorm `scale` vectors (memo originally said 51 — corrected by the phase-1 full decode: 147 zarr + 42 inline = 189 params), `(512,)`) are stored **inline in the
  895 KB msgpack `checkpoint`** under `/optimizer/target/...` as flax
  ExtType(code=1) = `[shape, dtype, bytes]`. Decoded with `msgpack` + a 12-line
  `ext_hook`. **Works, no flax needed.**

**Converter needs `msgpack` + `numpy` + `gzip` only. No JAX, t5x, seqio,
TensorStore or TensorFlow.** ~150 lines. License Apache-2.0, code and weights.

## 4. What CrispASR already has vs. what is new

Reusable: `src/t5_translate.cpp` (full ggml T5 enc-dec — RMSNorm, gated-GELU FFN
at lines 734/940, cross-attn, KV cache, greedy; ~80 % of the graph);
`src/core/mel.h` (STFT+mel+log, configurable base/guard/layout);
`models/convert-madlad-to-gguf.py` (T5→GGUF naming convention);
`src/piano_transcription.cpp` (the `--task transcribe-music` note-event surface
from §250). `src/core/beam_decode.h` is not needed — MT3 is greedy.

New work:
1. **Positional encoding swap.** `t5_translate.cpp` implements *only* T5
   relative-attention buckets (`t5_relative_position_bucket`, line 220). MT3 needs
   sinusoidal absolute added to the embeddings and **no** attention bias at all.
   Small but invasive — a `t5.pos_encoding` GGUF key + two branches.
2. **Continuous encoder input**: replace `get_rows(token_embedder)` with
   `mul_mat(continuous_inputs_projection, mel_frames)`. Easy.
3. **tf.signal mel filterbank, 512 bins from n_fft=2048**, `lo_hz=20`, DDSP
   `safe_log` (natural log, `+1e-5`), magnitude not power. `build_htk_fb` is close
   but not identical — needs a tf.signal-compatible variant.
4. **Event codec** (`event_codec.py`): id ↔ `(type, value)` over the 6 ranges.
   ~120 lines of C++, deterministic, unit-testable against Python.
5. **Tie-section state machine + cross-segment note assembly**
   (`note_sequences.py`) — the genuinely fiddly part, plus invalid-token recovery.
6. **Multi-track MIDI writer** (128 programs + drum channel 10). §251 already
   wants a MIDI writer in `core/`; piano_transcription emits its own single track.
7. Converter, `crispasr-diff mt3` ref dumper, registry entry, HF upload.

Sizes: 60 M params → **~120 MB F16**, ~35 MB Q4_K. Fits VPS RAM; Kaggle only
needed for the CUDA pass.

## 5. Successors — is one a better target?

| Model | Params | Weights | License | Notes |
|---|---|---|---|---|
| MT3 | 60 M | zarr+msgpack on public GCS | Apache-2.0 | the reference; Slakh Multi F1 ≈ 62 |
| [MR-MT3](https://arxiv.org/abs/2403.10024) (gudgud96) | ~60 M + memory block | 176 MB, HF `gudgud1014/MR-MT3` | MIT | +memory retention, MIDI-class F1 61.6→66.4, leakage 1.65→1.05. **Adds a 1024-token memory block concatenated to encoder outputs — a new cross-segment recurrence, i.e. strictly more C++ than MT3.** |
| [YourMT3+](https://arxiv.org/html/2407.04822v1) (mimbres) | 45.8 M | 536 MB (git-lfs) | Apache-2.0 | Slakh Multi F1 **74.84** vs 62.0. But: Perceiver-TF encoder + **MoE (2-of-8 experts)** + RoPE + **multi-channel decoder**. Essentially none of `t5_translate.cpp` survives the encoder swap. |
| [mt3-infer](https://github.com/openmirlab/mt3-infer) (openmirlab, MIT) | wrapper | — | MIT | **Highly useful**: unified PyTorch inference for MR-MT3 / MT3-PyTorch / YourMT3, auto-downloads checkpoints. This is the ready-made oracle for the diff harness. Note it *excludes* Magenta MT3 itself due to JAX/t5x dependency conflicts — so a JAX-free ref dumper for MT3 is still on us (but see §3: the checkpoint reads without JAX). |

Verdict: **MT3 first, YourMT3+ never (as a port), MR-MT3 maybe later.** YourMT3+
is the better *model* and the worse *port* — its accuracy win comes precisely
from the parts that are new ggml work. MT3 buys the whole event-codec / tie /
MIDI-assembly stack, which MR-MT3 then reuses for +5 F1 at the cost of one extra
attention block.

## 6. Does anything in the repo already cover this?

No. Checked `PLAN.md` §250/§251 and `docs/music-transcription/PLAN.md`:
`src/piano_transcription.cpp` is piano-only (88 keys, one instrument);
`src/btc_chords.cpp` / `src/crepe.cpp` are chords / monophonic f0, not note
events; Basic Pitch (§250, unclaimed) is instrument-*agnostic* polyphony, still
one track. **MT3 is the only multi-instrument note-event candidate on the
roster** — nothing else produces per-program tracks.

## 7. Effort estimate

Converter 0.5 d · numpy ref dumper + `crispasr-diff mt3` 1.5 d · `t5_translate.cpp`
abs-pos + continuous-input branches 1 d · mel front end to cos ≥ 0.999 1 d · event
codec + tie state machine + note assembly 2 d · multi-track MIDI writer +
CLI/registry/12-point checklist 1 d · decoded-output gate vs `mt3-infer` 1 d.
**Total ~8 working days.**

## 8. Recommendation: **GO — but scheduled after Basic Pitch.**

The blocking risk that PLAN.md cites does not exist. Everything else is a known
shape for this repo.

**Single biggest risk: the tie-section / cross-segment note assembly.** Cosine
parity on encoder and decoder logits will pass early and mean nothing — MT3's
output quality lives entirely in how a 2.048 s segment's re-declared active
pitches are stitched into continuous notes, and a subtly wrong state machine
produces plausible-looking MIDI with wrong durations and leaked instruments. Per
the HARD RULE, the gate must be **note-level F1 against `mt3-infer` on the same
audio**, not cosine. Budget the debugging there, not in the converter.

Secondary risk: greedy autoregressive decode at 1024 tokens × (audio_len / 2.048 s)
segments is the one shape where a generous offline budget does not help — a 4-min
song is ~117 segments. Expect single-digit× real-time on CPU. That is a
performance caveat, not a blocker.

## Corrections to existing repo docs

1. **`PLAN.md:3900` says MT3 is "Large (~1 GB+)". It is 172 MB f32 / ~120 MB F16.**
   Off by 6-8×. (Source: live `storage.googleapis.com/storage/v1/b/mt3/o` listing.)
2. **`docs/music-transcription/PLAN.md:456,568` says "T5X/JAX gin checkpoint …
   the converter is the whole job".** Empirically false — the checkpoint is zarr
   (gzip + `<f4`) plus one msgpack file; I decoded both with stdlib + numpy in
   this session. The converter is the *easiest* part; the tie-section decoder is
   the hard part.
3. `docs/music-transcription/PLAN.md:736` ("Not worth porting… autoregressive")
   remains directionally right on *speed* but was written assuming (1) and (2).
