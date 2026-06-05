---
license: mit
language:
- en
- zh
- es
- pt
- de
- ja
- ko
- fr
- ru
- id
- sv
- it
- he
- nl
- pl
- "no"
- tr
- th
- ar
- hu
- ca
- cs
- da
- fa
- af
- hi
- fi
- et
- el
- ro
- vi
- bg
- is
- sl
- sk
- lt
- sw
- uk
- ms
- hr
- cy
- ka
- ml
- te
- kn
- ta
- sr
- az
- mk
- hy
- bs
- bn
- mr
- ur
- my
- gl
- km
- eu
- ne
tags:
- speech
- asr
- speech-recognition
- gguf
- ggml
- vibevoice
- microsoft
- qwen2
base_model: microsoft/VibeVoice-7B
pipeline_tag: automatic-speech-recognition
---

# VibeVoice-7B — GGUF

GGUF conversions of [`microsoft/VibeVoice-7B`](https://huggingface.co/microsoft/VibeVoice-7B) for use with the `crispasr` CLI from **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

VibeVoice-7B is the **largest model** in Microsoft's VibeVoice family — a 9.3B-parameter speech-LLM (Qwen2.5-7B decoder + dual σ-VAE encoders) with state-of-the-art ASR quality across 50+ languages.

- **60-minute long-form audio** in a single forward pass
- **Built-in speaker diarization** — Speaker IDs per segment
- **Word-level timestamps**
- **Hotword / context injection** via `--context`
- **50+ languages** with automatic language detection
- **MIT licence**

> **Update (April 2026):** Now includes both ASR (encoder + LM) and TTS (σ-VAE decoder + prediction head). TTS requires ≥Q4_K for good quality — Q3_K is too aggressive for the decoder. For faster/smaller TTS, use [VibeVoice-Realtime-0.5B](https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF) or [VibeVoice-1.5B](https://huggingface.co/cstr/vibevoice-1.5b-GGUF).

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `vibevoice-7b-q3_k.gguf` | 4.7 GB | Q3_K — ASR only (TTS quality too low) |
| `vibevoice-7b-q4_0.gguf` | 5.6 GB | Q4_0 — fast decode |
| `vibevoice-7b-q4_k.gguf` | 5.8 GB | **Q4_K — recommended default (ASR + TTS)** |
| `vibevoice-7b-q5_k.gguf` | 6.8 GB | Q5_K — higher quality |
| `vibevoice-7b-q6_k.gguf` | 7.9 GB | Q6_K — near-lossless |
| `vibevoice-7b-q8_0.gguf` | 9.8 GB | Q8_0 — reference quality |
| `vibevoice-7b-f16.gguf` | 17.4 GB | F16 — full precision |

## Quick Start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON   # macOS
cmake --build build -j$(nproc)

# 2. Download the quantised GGUF
huggingface-cli download cstr/VibeVoice-7B-GGUF \
    vibevoice-7b-q4_k.gguf --local-dir .

# 3. Transcribe
./build/bin/crispasr --model vibevoice-7b-q4_k.gguf \
    --file audio.wav --backend vibevoice
```

Audio is automatically resampled from 16 kHz to 24 kHz by the backend.

## Architecture

| Component | Details |
| --- | --- |
| LM decoder | Qwen2.5-7B (28 layers, d=3584, 28/4 heads, GQA) |
| Acoustic encoder | 7-stage ConvNeXt σ-VAE, 3200× downsample |
| Semantic encoder | 7-stage ConvNeXt σ-VAE, 3200× downsample |
| Connectors | FC1 → RMSNorm → FC2 (acoustic + semantic) |
| Prediction head | 4-layer DiT with AdaLN modulation |
| Total parameters | ~9.3B |
| Input | 24 kHz mono PCM |
| Tokenizer | Qwen2.5 BPE (152064 tokens, embedded in GGUF) |

## Hardware Requirements

| Quantization | RAM (approx) | Notes |
| --- | ---: | --- |
| Q3_K | ~6 GB | Minimum for inference |
| Q4_K | ~7 GB | Recommended |
| Q8_0 | ~11 GB | High quality |
| F16 | ~18 GB | Full precision |

GPU acceleration (Metal/CUDA) is strongly recommended for the 7B model. CPU-only inference is very slow (~10× slower than realtime).

## Conversion

Converted from `microsoft/VibeVoice-7B` safetensors using the streaming memory-mapped converter:

```bash
python3 models/convert-vibevoice-stream-gguf.py \
    --input microsoft/VibeVoice-7B \
    --output vibevoice-7b-f16.gguf

# Quantize
./build/bin/crispasr-quantize vibevoice-7b-f16.gguf vibevoice-7b-q4_k.gguf q4_k
```

The streaming converter (`convert-vibevoice-stream-gguf.py`) uses memory-mapped tensor access to avoid loading the full 19 GB model into RAM.
