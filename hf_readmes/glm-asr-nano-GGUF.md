---
license: mit
language:
- zh
- en
- yue
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- gguf
- glm
- zhipu
- multilingual
library_name: ggml
base_model: zai-org/GLM-ASR-Nano-2512
---

# GLM-ASR-Nano-2512 — GGUF

GGUF conversions and quantisations of [`zai-org/GLM-ASR-Nano-2512`](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `glm-asr-nano.gguf` | F16 | 4.3 GB | Full precision |
| `glm-asr-nano-q8_0.gguf` | Q8_0 | 2.3 GB | High quality |
| `glm-asr-nano-q4_k.gguf` | Q4_K | 1.3 GB | Best size/quality tradeoff |

All variants produce correct transcription on test audio.

## Model details

- **Architecture:** Whisper encoder (1280d, 32L, partial RoPE) + 4-frame projector + Llama LLM (2048d, 28L, GQA 16/4)
- **Parameters:** 1.5B
- **Languages:** 17 (Mandarin, English, Cantonese, + 14 more)
- **License:** MIT
- **Outperforms OpenAI Whisper V3** on benchmarks (lowest avg error rate 4.10)

## Usage with CrispASR

```bash
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -S . -B build && cmake --build build -j8

# Auto-detect backend from GGUF
./build/bin/crispasr -m glm-asr-nano-q4_k.gguf -f audio.wav

# Explicit backend
./build/bin/crispasr --backend glm-asr -m glm-asr-nano-q4_k.gguf -f audio.wav -osrt
```

## Conversion

```bash
python models/convert-glm-asr-to-gguf.py --input zai-org/GLM-ASR-Nano-2512 --output glm-asr-nano.gguf
crispasr-quantize glm-asr-nano.gguf glm-asr-nano-q4_k.gguf q4_k
```
