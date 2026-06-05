---
license: mit
language:
- en
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- gguf
- moonshine
- lightweight
library_name: ggml
base_model: UsefulSensors/moonshine-tiny
---

# Moonshine Tiny -- GGUF

GGUF conversions and quantisations of [`UsefulSensors/moonshine-tiny`](https://huggingface.co/UsefulSensors/moonshine-tiny) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `moonshine-tiny.gguf` | F32 | 104 MB | Full precision |
| `moonshine-tiny-q8_0.gguf` | Q8_0 | 33 MB | High quality |
| `moonshine-tiny-q4_k.gguf` | Q4_K | 21 MB | Best size/quality tradeoff |

All variants produce correct transcription on test audio.

## Model details

- **Architecture:** Conv1d stem + 6L transformer encoder + 6L transformer decoder (288d, 8 heads, partial RoPE, SiLU/GELU)
- **Parameters:** 27M
- **Languages:** English only
- **WER:** 4.55% (LibriSpeech clean), 11.68% (Other)
- **Performance:** 11.2x realtime on CPU (F32)
- **License:** MIT
- **Source:** [moonshine.cpp](https://github.com/csexton-ua/moonshine.cpp) (MIT)

## Usage with CrispASR

```bash
./build/bin/crispasr -m moonshine-tiny-q4_k.gguf -f audio.wav
./build/bin/crispasr --backend moonshine -m moonshine-tiny-q4_k.gguf -f audio.wav -osrt
```
