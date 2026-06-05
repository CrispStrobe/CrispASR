---
license: mit
language:
- en
- fr
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- gguf
- kyutai
- moshi
- mimi
- codec
library_name: ggml
base_model: kyutai/stt-1b-en_fr
---

# Kyutai STT 1B (en/fr) -- GGUF

GGUF conversions and quantisations of [`kyutai/stt-1b-en_fr`](https://huggingface.co/kyutai/stt-1b-en_fr) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `kyutai-stt-1b.gguf` | F16 | 2.0 GB | Full precision |
| `kyutai-stt-1b-q8_0.gguf` | Q8_0 | 1.1 GB | High quality |
| `kyutai-stt-1b-q4_k.gguf` | Q4_K | 636 MB | Best size/quality tradeoff |

All variants produce correct transcription on test audio (JFK speech).

## Model details

- **Architecture:** Mimi neural audio codec encoder (SEANet CNN + 8-layer transformer + RVQ with 32 codebooks at 12.5 Hz) + 16-layer causal transformer LM (2048d, RoPE, SwiGLU, RMSNorm)
- **Parameters:** ~1B
- **Languages:** English, French
- **Audio input:** 24 kHz mono (auto-resampled from 16 kHz)
- **License:** MIT
- **Reference:** [moshi.cpp](https://github.com/Codes4Fun/moshi.cpp) (MIT)

This is a novel **codec-based** ASR architecture: audio is first encoded into discrete tokens via the Mimi neural audio codec, then a causal language model autoregressively predicts text tokens from the audio codes. Unlike encoder-decoder models (Whisper, Parakeet), the entire pipeline is autoregressive.

## Usage with CrispASR

```bash
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -S . -B build && cmake --build build -j8

# Auto-detect backend from GGUF
./build/bin/crispasr -m kyutai-stt-1b-q4_k.gguf -f audio.wav

# Explicit backend
./build/bin/crispasr --backend kyutai-stt -m kyutai-stt-1b-q4_k.gguf -f audio.wav -osrt
```

## Conversion

```bash
python models/convert-kyutai-stt-to-gguf.py --input kyutai/stt-1b-en_fr --output kyutai-stt-1b.gguf
crispasr-quantize kyutai-stt-1b.gguf kyutai-stt-1b-q4_k.gguf q4_k
```
