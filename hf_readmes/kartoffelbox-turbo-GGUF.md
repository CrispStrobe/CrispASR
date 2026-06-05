---
license: mit
language:
- de
- en
base_model:
- SebastianBodza/Kartoffelbox_Turbo
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- chatterbox
- chatterbox-turbo
- kartoffelbox
- german
- flow-matching
- meanflow
- hifi-gan
- gguf
- crispasr
library_name: ggml
---

# Kartoffelbox-Turbo TTS — GGUF (ggml)

GGUF / ggml conversion of [`SebastianBodza/Kartoffelbox_Turbo`](https://huggingface.co/SebastianBodza/Kartoffelbox_Turbo) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Kartoffelbox-Turbo is a **German fine-tune** of [chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) by SebastianBodza. It shares the same GPT-2 T3 architecture and S3Gen vocoder as chatterbox-turbo — only the T3 AR weights differ. Distributed under **MIT license**.

Three GGUF files are needed: the **T3 model** (Kartoffelbox fine-tune), the **S3Gen model** (shared with chatterbox-turbo), and optionally precomputed **conds** (voice conditioning).

## Files

| File | Size | Notes |
|---|---:|---|
| `kartoffelbox-turbo-t3-f16.gguf` | 958 MB | T3 GPT-2 AR model (24L, 1024D), German fine-tune |
| `kartoffelbox-turbo-t3-q8_0.gguf` | 623 MB | Q8_0 quantized (vocoder/embeddings preserved at F32) |
| `kartoffelbox-turbo-t3-q4_k.gguf` | 452 MB | Q4_K quantized |
| `chatterbox-turbo-s3gen-f16.gguf` | 628 MB | S3Gen encoder + meanflow CFM + HiFT vocoder (shared with turbo) |

The T3 file includes precomputed voice conditioning (`conds.*` tensors). The S3Gen file is identical to the one in [`cstr/chatterbox-turbo-GGUF`](https://huggingface.co/cstr/chatterbox-turbo-GGUF).

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build -j

# 2. Pull model files
huggingface-cli download cstr/kartoffelbox-turbo-GGUF kartoffelbox-turbo-t3-f16.gguf --local-dir .
huggingface-cli download cstr/chatterbox-turbo-GGUF chatterbox-turbo-s3gen-f16.gguf --local-dir .

# 3. Synthesise via CLI
./build/bin/crispasr \
  -m kartoffelbox-turbo-t3-f16.gguf \
  --codec-model chatterbox-turbo-s3gen-f16.gguf \
  --tts "Hallo Welt, wie geht es Ihnen heute?" \
  --tts-output output.wav
```

## Architecture

Same as chatterbox-turbo — see [`cstr/chatterbox-turbo-GGUF`](https://huggingface.co/cstr/chatterbox-turbo-GGUF) for full architecture details.

```
Text -> GPT-2 BPE tokenizer (50257 tokens)
     -> T3 GPT-2 AR (24 layers, 1024D, 16 heads, learned pos emb)
     -> 25 Hz speech tokens (6561 codebook)
     -> UpsampleConformerEncoder (6 pre + 4 post upsample, 512D)
     -> Meanflow CFM denoiser (2 Euler steps)
     -> HiFTGenerator vocoder (F0 + SineGen + iSTFT)
     -> 24 kHz mono WAV
```

### Key differences from chatterbox-turbo

| Feature | Chatterbox-Turbo | Kartoffelbox-Turbo |
|---|---|---|
| Language | English | **German** (+ English) |
| T3 weights | ResembleAI base | SebastianBodza fine-tune |
| speech_cond_prompt_len | 150 | **375** |
| S3Gen / vocoder | Same | Same (shared) |

## Quantized variants

Quantized GGUFs are available for reduced memory usage:

| Variant | T3 size | S3Gen size | Total |
|---|---:|---:|---:|
| F16 | 958 MB | 628 MB | 1,586 MB |
| Q8_0 | ~630 MB | ~350 MB | ~980 MB |
| Q4_K | ~460 MB | ~245 MB | ~705 MB |

Vocoder/F0 weights and embedding tables are preserved at original precision; only transformer attention/FFN weights are quantized.

## Conversion

```bash
# From HuggingFace model:
python models/convert-chatterbox-to-gguf.py \
  --input SebastianBodza/Kartoffelbox_Turbo \
  --output-dir /path/to/output \
  --variant kartoffelbox
```

## Related models

- [`cstr/chatterbox-turbo-GGUF`](https://huggingface.co/cstr/chatterbox-turbo-GGUF) — base Chatterbox-Turbo (English)
- [`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF) — base Chatterbox (Llama T3, 10-step CFM)
- [`SebastianBodza/Kartoffelbox_Turbo`](https://huggingface.co/SebastianBodza/Kartoffelbox_Turbo) — original PyTorch model
