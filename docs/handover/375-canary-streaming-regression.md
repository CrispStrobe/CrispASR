# Handover — #375 canary streaming regression

**State: a real, reproduced regression in the bisect window is found and fixed
(glint AAC-LC decode). Whether it is the reporter's regression awaits their
confirmation of the input format.** PR #376 is CLOSED — the pre-existing seam
artifact it aimed at is fixed at the root instead: canary long-form now ports
canary-1b-v2's own `.transcribe()` dynamic chunking (30..40 s raw-waveform
chunks, 1 s overlap, per-chunk normalization, NeMo's LCS-alignment merge —
`core/canary_chunk_merge.h`, pinned by `tests/test-canary-chunk-merge.cpp`
against vectors from the nemo 2.7.3 Python functions). The previous 8 s / 2 s
LCS-prefix streaming was parakeet machinery grafted onto canary; it survives
only behind `CRISPASR_CANARY_LEGACY_STREAM=1` as the bisection arm. On
jfk_x12 the legacy gate reproduces `ask not Ask not` / broken clauses ×2
while the default now emits 12 clean repetitions; fleurs_600s has zero
repeated n-grams in 925 words.

## The report

`cdoepmann`, 2026-08-19. Upgraded CrispASR, canary recognition degraded:
"sentences are interrupted and words/phrases are repeated a few times before the
recognition continues". Present on **all quantizations**, and on **CUDA and CPU
alike**. Bisected to `282e5d0b` (good) .. `c08f7a52` (bad).

## Root cause found (2026-08-19, measured)

The signature — all quants, CPU and CUDA alike — is an **input-path** signature.
The window rerouted compressed-audio decode through the in-tree glint decoder
(`f3d82d30`, `bf249d09`, …), and glint's AAC-LC decoder had two defects that
only fire on streams glint's own encoder never produces:

1. **`window_shape` parsed and discarded** — synthesis always used sine
   windows. Real encoders (ffmpeg, Apple, fdk) emit KBD on ~80%+ of frames;
   KBD analysis + sine synthesis breaks MDCT perfect reconstruction.
2. **TNS decode broken**: skipped entirely on EIGHT_SHORT windows (transients),
   no `tns_max_bands` clamp, direction bit ignored, only one filter per window,
   hardcoded 4-bit dequant.

Net effect: **every real-world .aac (ADTS) file decodes at ~17 dB SNR** —
bitrate-independent (96k and 192k both 17.7 dB) — where the old path
(AudioToolbox on macOS, ffmpeg subprocess on Linux) gives 37–66 dB. Since
`f3d82d30`, glint runs BEFORE AudioToolbox/ffmpeg, so even ffmpeg-enabled
builds regressed. Formats measured unaffected: wav/mp3/m4a/flac byte-identical
across the endpoints; opus/webm differ ≤2e-5 (decoder rounding).

End-to-end (canary-1b-v2-q4_k, CPU, `fleurs_60s` re-encoded to AAC): the two
endpoints' transcripts **differ on .aac input** (first divergence on any
input); the degraded audio produces garbled openings and a "transcript is not
in time order after slice merge" warning. The repeated-phrase symptom is the
canary AED **looping on hard audio** — observed firing even on the GOOD build
on one clean-audio chunk ("Ich habe das Gefühl, dass ich das Gefühl habe,
dass …" ×25) — degraded 17 dB input makes those loops far more likely, which
matches the reporter's description exactly.

## The fix

glint upstream `77738f3` (`CrispStrobe/glint`): KBD window tables
(ISO/IEC 14496-3 4.6.11.3.2, α=4 long / α=6 short, left half keyed to the
previous frame's shape), full spec `tns_decode_frame` (short-window TNS via
window-major de-interleave before TNS, region stacking, direction, per-res
dequant). Measured: 17.7 → 67.3 dB (16 kHz), 17.0 → 70.1 dB (44.1 kHz).
glint's full ctest suite green, including fuzz.

**Regression gate** (red-verified): `tools/test_aac_decoder.py` Tier 3 —
foreign encoder with PNS disabled → glint decode, hard 40 dB SNR floor. Old
decoder: 16.4/17.3 dB FAIL; every pre-existing gate passed at 17 dB (the
roundtrip gates own both sides of the contract; the foreign-stream gate only
checked spectrum correlation).

Synced into CrispASR via the sync-glint workflow (run 32288223794). CrispASR's
`test-audio-formats` passes (106 assertions); canary end-to-end on .aac now
matches the good endpoint except chunk-1 loop flips that the good build also
exhibits.

## What remains open

1. **Reporter confirmation** — asked on #375 whether their inputs are .aac
   (ADTS). If their audio is WAV/MP3, this fix is real but not their bug, and
   the hunt resumes (everything else in the window is measured byte-identical
   on two files; a content-dependent delta would need their audio).
   Workaround offered meanwhile: `CRISPASR_AAC_DECODER=ffmpeg` (any non-glint
   value) bypasses glint AAC.
2. **Pre-existing, separate**: the glint decode paths resample 44.1/48 kHz →
   16 kHz with miniaudio's LINEAR resampler: ~28 dB vs AudioToolbox/ffmpeg's
   ~38 dB on the same decode. Same class as the opus/webm paths at BOTH
   endpoints (29.3 dB). Not the regression; worth its own issue.
3. ~~Canary seam-merge artifacts~~ — FIXED by the blueprint port (see the
   State block). PR #376 CLOSED, its branch deleted;
   `CRISPASR_CANARY_SEAM_DEDUP` only means anything under the legacy gate.

## Traps already burned (do not repeat)

* A file-path-scoped search cannot see cross-cutting changes: `73bb9b2f` was
  selected, not bisected, by "the only canary commit in the window".
* Anything that looks like the report on `main` must be diffed against
  `282e5d0b` on the SAME file first — the seam artifacts reproduce on both.
* WAV-only probes cannot rule out the audio path: the regression was
  format-specific (.aac only). Test the formats whose ROUTING changed.
* SNR without shift alignment understates resampled paths (group delay);
  align first, then judge.
* glint's own roundtrip gates (86–135 dB) said nothing about foreign streams:
  the encoder never emits KBD windows or short-window TNS. A test that owns
  both sides of a contract is blind.
* The 8 s / 2 s "NeMo FrameBatchMultiTaskAED analogon" comment in canary.cpp
  was FALSE on every parameter — FrameBatchMultiTaskAED joins NON-overlapping
  chunks with `" ".join`, and canary-1b-v2's shipped path is dynamic 30..40 s
  chunks with a 1 s overlap and `lcs_alignment_merge_buffer`. Read the
  blueprint, not the comment that cites it.
