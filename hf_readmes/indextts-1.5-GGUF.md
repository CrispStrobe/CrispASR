---
license: apache-2.0
language:
- en
- zh
- ja
- ko
base_model:
- IndexTeam/IndexTTS-1.5
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- indextts
- voice-cloning
- gpt2
- bigvgan
- gguf
- crispasr
library_name: ggml
---

# IndexTTS-1.5 — GGUF (ggml-quantised)

GGUF / ggml conversion of [`IndexTeam/IndexTTS-1.5`](https://github.com/index-tts/IndexTTS) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

IndexTTS-1.5 is a zero-shot voice cloning TTS system: reference audio + text → cloned speech at 24 kHz. Architecture: Conformer conditioning encoder (6L, d=512) → Perceiver resampler (2L, 32 latents) → GPT-2 AR decoder (24L, d=1280, 20 heads) → BigVGAN vocoder (6-stage upsample, anti-aliased SnakeBeta activations). ~500M parameters total. Distributed under **Apache-2.0 license**.

Two GGUF files are needed: the **GPT model** (conditioning + text → mel codes → latent) and the **BigVGAN vocoder** (latent → 24 kHz audio).

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `indextts-gpt-q8_0.gguf`  | Q8_0 | 613 MB | GPT-2 + Conformer + Perceiver — **recommended** |
| `indextts-gpt-q4_k.gguf`  | Q4_K | 347 MB | GPT-2 + Conformer + Perceiver — smallest |
| `indextts-gpt.gguf`       | F16  | 2.2 GB | GPT-2 + Conformer + Perceiver — reference quality, bit-exact Python parity |
| `indextts-bigvgan.gguf`   | F16  | 256 MB | BigVGAN vocoder (shared across all GPT quants) |

All quant levels produce correct speech (ASR roundtrip = "Hello world!"). F16 gives 100% mel-code parity with Python; Q8_0 is the best quality/size trade-off.

## Quick start

```bash
# Easiest: auto-download (~870 MB on first run)
./build/bin/crispasr --backend indextts -m auto \
    --voice reference_speaker.wav \
    --tts "Hello world, this is IndexTTS speaking."

# Output: tts_output.wav (24 kHz mono)
```

Or with explicit paths:

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target crispasr-cli

# 2. Pull model files (pick your preferred quant)
huggingface-cli download cstr/indextts-1.5-GGUF indextts-gpt-q8_0.gguf --local-dir .
huggingface-cli download cstr/indextts-1.5-GGUF indextts-bigvgan.gguf --local-dir .

# 3. Synthesise with voice cloning
./build/bin/crispasr --backend indextts \
    -m indextts-gpt-q8_0.gguf \
    --codec-model indextts-bigvgan.gguf \
    --voice reference_speaker.wav \
    --tts "Hello world, this is IndexTTS speaking."
```

## Features

- **Zero-shot voice cloning** — any 3-10 second reference WAV at 24 kHz (or auto-resampled)
- **Multilingual** — trained on English and Chinese; cross-language cloning works
- **Beam search** — num_beams=3 with repetition_penalty=10.0 (matches Python defaults)
- **No external dependencies** — SentencePiece tokenizer embedded in GGUF, no espeak/phonemizer needed

## Accuracy

With the F16 model, C++ mel codes match Python **100%** (55/55 tokens identical to the Python greedy reference). Conditioning norm matches Python within 0.001%. All quantization levels (Q8_0, Q4_K) produce correct speech verified via ASR roundtrip.

## Architecture details

```
Input text → uppercase → SentencePiece unigram tokenizer (12000 vocab)
Reference audio → 24kHz resample → mel spectrogram (100 bands, hop=256)
  → Conformer encoder (6 blocks, d=512, 8 heads, Conv2d subsampling)
  → Perceiver resampler (2 layers, 32 latents, d=1280, GEGLU FFN)
  → 32 conditioning vectors

GPT-2 AR decoder:
  [32 cond latents | text_embs + text_pos | start_mel + mel_pos]
  → 24 transformer blocks (d=1280, 20 heads, GELU FFN)
  → gpt.ln_f → final_norm → mel_head → beam search (B=3)
  → mel codes (stop token = 8193)

Latent extraction (2nd pass):
  Full sequence → GPT-2 → gpt.ln_f → final_norm → [n_mel+1, 1280]

BigVGAN vocoder:
  latent [T, 1280] → conv_pre → 6× (ConvTranspose1d + AMPBlock1 with
  anti-aliased SnakeBeta) → conv_post → tanh → 24kHz PCM
  + ECAPA-TDNN speaker embedding for voice conditioning
```

## Conversion

```bash
python models/convert-indextts-to-gguf.py \
    --model-dir /path/to/IndexTTS-1.5 \
    --output indextts-gpt.gguf \
    --vocoder-output indextts-bigvgan.gguf
```

## License

Apache-2.0 (same as upstream IndexTTS).

## Citation

```bibtex
@misc{indextts2024,
  title={IndexTTS: An Industrial-Level Zero-Shot Text-to-Speech System with Controllable Timbre},
  author={IndexTeam},
  year={2024},
  url={https://github.com/index-tts/IndexTTS}
}
```
