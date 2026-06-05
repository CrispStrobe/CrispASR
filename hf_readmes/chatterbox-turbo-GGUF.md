---
license: mit
language:
- en
base_model:
- ResembleAI/chatterbox-turbo
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- chatterbox
- chatterbox-turbo
- flow-matching
- meanflow
- hifi-gan
- gguf
- crispasr
library_name: ggml
---

# Chatterbox-Turbo TTS — GGUF (ggml)

GGUF / ggml conversion of [`ResembleAI/chatterbox-turbo`](https://huggingface.co/ResembleAI/chatterbox-turbo) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Chatterbox-Turbo is a distilled 350M-parameter TTS pipeline: GPT-2 tokenizer + AR text-to-speech model + meanflow S3Gen (2-step CFM, vs 10 for base Chatterbox) + HiFTGenerator vocoder. Distributed under **MIT license**.

Two GGUF files are needed: the **T3 model** (text to speech tokens) and the **S3Gen model** (speech tokens to audio).

## Files

| File | Size | Notes |
|---|---:|---|
| `chatterbox-turbo-t3-f16.gguf`     | 964 MB | T3 GPT-2 AR model (24L, 1024D) |
| `chatterbox-turbo-t3-q8_0.gguf`    | 628 MB | Quantized T3, recommended deployment default |
| `chatterbox-turbo-t3-q4_k.gguf`    | 457 MB | Smaller T3 quant for memory-constrained use |
| `chatterbox-turbo-s3gen-f16.gguf`  | 628 MB | S3Gen encoder + meanflow CFM + HiFT vocoder |
| `chatterbox-turbo-s3gen-q8_0.gguf` | 350 MB | Quantized S3Gen, recommended deployment default |
| `chatterbox-turbo-s3gen-q4_k.gguf` | 244 MB | Smaller S3Gen quant for memory-constrained use |

Encoder attention/FFN weights are stored at F32 precision for quality. Vocoder weights (conv_pre, resblocks, conv_post, source fusion, F0 predictor) are F32.

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build -j --target chatterbox

# 2. Pull both model files (Q8_0 recommended)
huggingface-cli download cstr/chatterbox-turbo-GGUF chatterbox-turbo-t3-q8_0.gguf --local-dir .
huggingface-cli download cstr/chatterbox-turbo-GGUF chatterbox-turbo-s3gen-q8_0.gguf --local-dir .

# 3. Synthesise (C API — CLI adapter in progress)
# See test programs in SESSION_HANDOVER.md for usage examples
```

## Architecture

```
Text -> GPT-2 BPE tokenizer (50257 tokens)
     -> T3 GPT-2 AR (24 layers, 1024D, 16 heads, learned pos emb, SwiGLU)
     -> 25 Hz speech tokens (6561 codebook)
     -> UpsampleConformerEncoder (6 pre + 4 post upsample, 512D, 8 heads, rel-pos attn)
        -> Upsample1D: nearest-neighbor 2x + Conv1d(512,512,k=5) + Linear + LayerNorm + xscale
     -> 80-channel mel spectrogram (50 Hz)
     -> Meanflow CFM denoiser (2 Euler steps, linear schedule, no CFG)
        UNet1D: 1 down + 12 mid + 1 up blocks, 256 ch, 4 transformer blocks each
     -> HiFTGenerator vocoder (F0 predictor + SineGen + 3x ConvTranspose1d + iSTFT)
     -> 24 kHz mono WAV
```

### Key differences from base Chatterbox

| Feature | Base Chatterbox | Chatterbox-Turbo |
|---|---|---|
| T3 architecture | Llama (30L, 520M) | GPT-2 Medium (24L, 350M) |
| T3 tokenizer | Character (704 tokens) | BPE (50257 tokens) |
| CFM steps | 10 (cosine schedule) | 2 (linear, meanflow distilled) |
| CFG | Yes (rate=0.7) | No (distilled) |
| Total params | ~520M | ~350M |

## Quality verification

ASR roundtrip using same speech tokens as Python reference:

| Metric | Value |
|---|---|
| ASR output (moonshine-base) | **"Hello world"** (correct) |
| Language detection confidence | **0.939** |
| encoder_out RMS | **0.4602** (exact match to Python) |
| matrix_bd (rel-pos scores) h0[0,0] | **24.70** (matches Python to 2dp) |

## Conversion

```bash
# From HuggingFace model (requires chatterbox-tts pip package):
python models/convert-chatterbox-to-gguf.py \
  --input ResembleAI/chatterbox-turbo \
  --output-dir /path/to/output \
  --variant turbo
```

## Related models

- [`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF) — base Chatterbox (Llama T3, 10-step CFM)
- [`cstr/lahgtna-chatterbox-v1-GGUF`](https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF) — Arabic T3 variant
