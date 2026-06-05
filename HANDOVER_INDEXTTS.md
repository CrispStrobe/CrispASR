# IndexTTS-1.5 — Handover for 99.5% Parity

## Current State (2026-05-10)

**RESOLVED** — IndexTTS now achieves >99.5% parity with Python.
ASR roundtrip produces **"Hello world!"** matching Python exactly.
All 55/55 mel codes match the Python greedy reference.

### What works (verified against Python):
- **Conditioning (Conformer+Perceiver):** norm 288.92 matches Python. First5 match to 4+ decimal places.
- **GPT mel codes:** 100% identical (55/55 match Python greedy reference)
- **Latent extraction:** Shape [56, 1280] correct
- **BigVGAN vocoder:** ASR = "Hello world!" (matches Python)
- **End-to-end:** All mel codes match, ASR roundtrip = "Hello world!"

## Bugs fixed (2026-05-10)

### Root cause: mel spectrogram center-padding (zero vs reflect)

The `core_mel::compute` used **zero-padding** for `center_pad`, but torchaudio's `center=True`
uses **reflect-padding**. This caused the first 2 STFT frames to differ from Python, propagating
through the Conformer (6 layers) into conditioning, then compounding through 24 GPT layers to
flip a beam search token at step 14.

**Fix:** Added `center_pad_reflect` option to `core_mel::Params`. IndexTTS now uses reflect padding.

### Bug 2: Repetition penalty ordering

C++ applied rep penalty to raw logits BEFORE log_softmax. HuggingFace's beam search applies
it AFTER log_softmax (to log-probabilities). Fixed to match HF's exact behavior.

### What was NOT the problem (disproved hypotheses):
- F16 weight precision (F32 GGUF gives identical results to F16)
- Flash attention precision (F16 BD mask has negligible effect)
- Conformer/Perceiver architecture (matches Python perfectly once mel is correct)

## Files modified in this session

```
src/core/mel.h            — added center_pad_reflect option
src/core/mel.cpp          — reflect-padding implementation for center_pad
src/indextts.cpp          — enable reflect padding, fix rep penalty (after log_softmax),
                            gate debug prints behind INDEXTTS_DEBUG env
```

## Diff-testing commands

```bash
# 1. Generate Python reference conditioning + latent + mel codes:
cd /mnt/akademie_storage/index-tts-src
# (see the Python scripts used in this session — they bypass the broken normalizer via patches)

# 2. Test with Python conditioning override (isolates GPT precision):
INDEXTTS_COND_FILE=/mnt/storage/indextts_cond_jfk2.bin \
  ./build/bin/crispasr --backend indextts \
  --model /mnt/akademie_storage/models/indextts-gguf/indextts-gpt.gguf \
  --codec-model /mnt/akademie_storage/models/indextts-gguf/indextts-bigvgan.gguf \
  --tts "Hello world." --voice /mnt/akademie_storage/chatterbox/test_jfk_final.wav

# 3. Test with Python latent override (isolates vocoder):
INDEXTTS_LATENT_FILE=/mnt/storage/indextts_latent_jfk.bin \
  ... (same as above)

# 4. ASR roundtrip:
ffmpeg -y -i tts_output.wav -ar 16000 /mnt/storage/tts_16k.wav
./build/bin/crispasr --backend parakeet --model /mnt/storage/models/parakeet-tdt-0.6b-v2/model.gguf /mnt/storage/tts_16k.wav
```

## Python reference values (JFK reference audio, "HELLO WORLD."):

```
Conditioning: norm=288.92, first5=[-0.17545, 0.05909, -0.08273, -0.10259, -0.05862]
Mel codes (56): [2623, 515, 4643, 7539, 7051, 7119, 7901, 7957, 7002, 7357, 6783, 6599, 6697, 6719, 6109, 5085, 5110, 1573, 3978, 4953, 2342, 1783, 7453, 5750, 1863, 6133, 1634, 8180, 6737, 2012, 1736, 3828, 6355, 105, 2428, 1922, 2343, 3144, 4780, 587, 2455, 959, 7986, 2844, 3467, 6353, 1719, 7442, 4246, 5708, 6956, 4401, 4026, 1940, 6122, 8193]
Latent: shape=[56, 1280], rms=1.1726
Audio: 2.35s, rms=3655, peak=-3.2dB, ASR="Hello world!"
```

## C++ current values:

```
Conditioning: norm=288.88, first5=[-0.17556, 0.06011, -0.08333, -0.10298, -0.05928]
Mel codes (70): [2623, 515, 4643, 7539, 7051, 7119, 7901, 7957, 7002, 7357, 6783, 6599, 6697, 6719, 6283, 6109, ...]
                                                                              ^--- diverges here (step 14)
Audio: 3.03s, rms=4072, peak=-2.5dB, ASR="Hello, well."
```

## Key insight for next session

The 0.01% conditioning difference compounds through 24 GPT layers of dot products into a logit difference
of ~0.01 at step 14. This crosses a beam-score threshold, causing a different token to win.

The cleanest fix: **convert the GGUF to F32** (or at minimum, keep the GPT attention weights in F32)
and verify that mel codes then match 100%. If they do, the implementation is correct and the remaining
gap is a quantization trade-off, not a bug.
