---
license: apache-2.0
language:
- en
base_model:
- hexgrad/Kokoro-82M
- yl4579/StyleTTS2-LJSpeech
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- kokoro
- styletts2
- gguf
- crispasr
library_name: ggml
---

# Kokoro-82M — GGUF (ggml-quantised)

GGUF / ggml conversion of [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Kokoro-82M is an open-weight 82 M-parameter TTS model built on the StyleTTS2 architecture. It is multilingual at the model level (178-symbol IPA vocab covering en, es, fr, hi, it, ja, pt, zh) and ships per-language voice packs in [`hexgrad/Kokoro-82M/voices`](https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices). Apache-2.0 weights.

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `kokoro-82m-f16.gguf`  | F16  | 156 MB | Reference quality — bit-true to the PyTorch model on the deterministic stages |
| `kokoro-82m-q8_0.gguf` | Q8_0 | 135 MB | **Recommended** — ASR roundtrip identical to F16 |

Q4_K is **not** published: per the `crispasr-diff kokoro` harness it falls below quality bar (cosine similarity collapses to ~0.1 on `audio_out` and the German backbone produces unintelligible output). Q8_0 saves ~13 % on disk over F16 with no observable quality loss.

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target crispasr-lib

# 2. Pull the model + a voice pack
huggingface-cli download cstr/kokoro-82m-GGUF kokoro-82m-q8_0.gguf --local-dir .
huggingface-cli download cstr/kokoro-voices-GGUF kokoro-voice-af_heart.gguf --local-dir .

# 3. Synthesise
./build/bin/crispasr --backend kokoro \
    -m kokoro-82m-q8_0.gguf \
    --voice kokoro-voice-af_heart.gguf \
    --tts "Hello, this is a test of the English phonemizer." \
    --tts-output hello.wav
```

24 kHz mono WAV. `--language <code>` switches the in-process libespeak-ng phonemizer (en, es, fr, it, pt, hi, ja, zh — and de/ru when paired with the [German backbone](https://huggingface.co/cstr/kokoro-de-hui-base-GGUF)).

## Voice packs

Voices ship in the sister repo [`cstr/kokoro-voices-GGUF`](https://huggingface.co/cstr/kokoro-voices-GGUF) so the model and per-speaker style files can be versioned independently. CrispASR auto-routes by language when both files sit in the same directory.

## Quality verification

ASR roundtrip via `parakeet-tdt-0.6b-v3 -l en`, voice `af_heart`:

| Quant | Synthesised text | Parakeet output |
|---|---|---|
| F16  | "Hello, this is a test of the English phonemizer." | "Hello this is a test of the English Phone Miza." |
| Q8_0 | (same text) | "Hello this is a test of the English Phone Miza." |

Both quants produce ASR-identical output. The trailing "Phone Miza" is parakeet mis-transcribing the rare word "phonemizer", not a TTS artefact.

`crispasr-diff kokoro` against the PyTorch reference:

| Stage (selection) | F16 cos_min | Q8_0 cos_min |
|---|---:|---:|
| token_ids → durations | 1.000 | 1.000 |
| `pred_lstm_out` | 0.999 | 0.997 |
| `dec_decode_3_out` | 0.9999 | 0.9990 |
| `audio_out` (RNG-divergent) | 0.99 | 0.85 |

F16 hits 16/16 PASS at cos≥0.999. Q8_0 hits 9/16 PASS at the same threshold; the failed stages stay above cos≥0.85 and don't perturb ASR roundtrip.

## Architecture

| Component | Details |
|---|---|
| Text encoder | 178-IPA-symbol embedding → 1× CNN block → bidirectional LSTM |
| pl-BERT | 12-layer transformer, 768 d, 12 heads (text prosody features) |
| Predictor | Duration LSTM + F0/N curves |
| Decoder (iSTFTNet) | 3-stage upsample (10×6 = 60×) + ResBlocks + iSTFT (n_fft=20, hop=5) |
| Style dim | 256 (split as 128+128 between predictor and decoder) |
| Audio | 24 kHz mono output; predictor consumes phoneme IDs |

## Conversion

```bash
python models/convert-kokoro-to-gguf.py \
    --input hexgrad/Kokoro-82M \
    --output kokoro-82m-f16.gguf \
    --outtype f16

build/bin/crispasr-quantize kokoro-82m-f16.gguf kokoro-82m-q8_0.gguf q8_0
```

The converter handles both legacy `weight_g/weight_v` and the modern `parametrizations.weight.original{0,1}` WeightNorm forms, so it also runs on community re-trains like the dida-80b German backbone.

## Attribution

- **Original model:** [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0). hexgrad et al.
- **Architecture:** [StyleTTS2-LJSpeech](https://github.com/yl4579/StyleTTS2) by Yinghao Aaron Li.
- **GGUF conversion + ggml runtime:** [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR) — see `src/kokoro.cpp` and `models/convert-kokoro-to-gguf.py`.

## License

Apache-2.0, inherited from the base model.
