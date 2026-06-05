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
- streaming
- lightweight
library_name: ggml
base_model: UsefulSensors/moonshine-streaming-tiny
---

# Moonshine Streaming Tiny -- GGUF

GGUF conversions and quantisations of [`UsefulSensors/moonshine-streaming-tiny`](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `moonshine-streaming-tiny.gguf` | F32 | 168 MB | Full precision |
| `moonshine-streaming-tiny-q4_k.gguf` | Q4_K | 31 MB | Quantized |

## Model details

- **Architecture:** Streaming encoder-decoder ASR. Raw-waveform audio frontend (no mel) + sliding-window transformer encoder (6L, 320d) + autoregressive transformer decoder (6L, 320d, SiLU-gated MLP, partial RoPE)
- **Parameters:** 34M
- **Languages:** English
- **License:** MIT
- **Source:** [`UsefulSensors/moonshine-streaming-tiny`](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny)
- **Designed for:** Low-latency streaming ASR on edge devices

## Usage with CrispASR

```bash
./build/bin/crispasr --backend moonshine-streaming -m moonshine-streaming-tiny-q4_k.gguf -f audio.wav
```

## Notes

- Tokenizer (`tokenizer.bin`) must be in the same directory as the model file
- Streaming architecture: sliding-window attention with 80ms lookahead
- Audio frontend processes raw waveform (no mel spectrogram needed)
