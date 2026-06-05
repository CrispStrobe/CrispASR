---
license: apache-2.0
language:
- en
base_model:
- sesame/csm-1b
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- csm
- mimi
- conversational
- gguf
- crispasr
library_name: ggml
---

# CSM-1B — GGUF (ggml-quantised)

GGUF / ggml conversion of [`sesame/csm-1b`](https://huggingface.co/sesame/csm-1b) (Conversational Speech Model) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

CSM-1B is a TTS model that generates speech from text using a two-stage transformer architecture:
- **Backbone** (Llama-3.2 1B, 16 layers): generates first-codebook Mimi tokens autoregressively
- **Depth decoder** (Llama-3.2 100M, 4 layers): fills remaining 31 codebooks per frame
- **Mimi codec** (Kyutai, 8-layer transformer + SEANet): converts 32-codebook RVQ tokens to 24 kHz PCM

Released under **Apache 2.0** license.

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `csm-1b-f16.gguf`  | F16  | 3.4 GB | Full precision — reference quality |
| `csm-1b-q8_0.gguf` | Q8_0 | 1.9 GB | Recommended — identical ASR roundtrip |
| `csm-1b-q4_k.gguf` | Q4_K | 1.1 GB | Smallest — minor quality loss |

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target crispasr-cli

# 2. Download model
huggingface-cli download cstr/csm-1b-GGUF csm-1b-q8_0.gguf --local-dir .

# 3. Synthesize
./build/bin/crispasr --backend csm -m csm-1b-q8_0.gguf \
    --tts "Hello, how are you today?" \
    --tts-output hello.wav --seed 42
```

Or with auto-download:
```bash
./build/bin/crispasr -m csm --auto-download \
    --tts "The quick brown fox jumps over the lazy dog." \
    --tts-output fox.wav
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--seed N` | 0 | RNG seed (0 = non-deterministic) |
| `-tp N` | 0.9 | Sampling temperature |
| `--tts-output PATH` | — | Output WAV path (24 kHz mono) |

## Architecture details

- **Text tokenizer**: Llama-3.2 BPE (128,256 tokens)
- **Audio codec**: Mimi (32 codebooks, 2048 entries each, 12.5 Hz frame rate)
- **Backbone**: 16-layer Llama with GQA (32 heads, 8 KV heads), SwiGLU, RMSNorm, RoPE theta=500,000
- **Depth decoder**: 4-layer Llama (8 heads, 2 KV heads), position-specific codebook heads
- **Sample rate**: 24,000 Hz

## Conversion

```bash
python models/convert-csm-to-gguf.py \
    --input sesame/csm-1b \
    --output csm-1b-f16.gguf

# Quantize
./build/bin/crispasr-quantize csm-1b-f16.gguf csm-1b-q8_0.gguf q8_0
./build/bin/crispasr-quantize csm-1b-f16.gguf csm-1b-q4_k.gguf q4_k
```

## Acknowledgements

- [SesameAILabs/csm](https://github.com/SesameAILabs/csm) — original model and inference code
- [Kyutai/mimi](https://huggingface.co/kyutai/mimi) — audio codec
