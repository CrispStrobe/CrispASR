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
base_model: UsefulSensors/moonshine-streaming-medium
---

# Moonshine Streaming Medium -- GGUF

GGUF conversions and quantisations of [`UsefulSensors/moonshine-streaming-medium`](https://huggingface.co/UsefulSensors/moonshine-streaming-medium) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `moonshine-streaming-medium.gguf` | F32 | 1014 MB | Full precision |
| `moonshine-streaming-medium-q4_k.gguf` | Q4_K | 183 MB | Quantized |

## Model details

- **Architecture:** Streaming encoder-decoder ASR. Raw-waveform audio frontend (no mel) + sliding-window transformer encoder (14L, 768d) + autoregressive transformer decoder (14L, 640d, SiLU-gated MLP, partial RoPE)
- **Parameters:** 245M
- **Languages:** English
- **License:** MIT
- **Source:** [`UsefulSensors/moonshine-streaming-medium`](https://huggingface.co/UsefulSensors/moonshine-streaming-medium)
- **Designed for:** Low-latency streaming ASR on edge devices

## Usage with CrispASR

```bash
./build/bin/crispasr --backend moonshine-streaming -m moonshine-streaming-medium-q4_k.gguf -f audio.wav
```

## Notes

- Tokenizer (`tokenizer.bin`) must be in the same directory as the model file
- Streaming architecture: sliding-window attention with 80ms lookahead
- Audio frontend processes raw waveform (no mel spectrogram needed)
