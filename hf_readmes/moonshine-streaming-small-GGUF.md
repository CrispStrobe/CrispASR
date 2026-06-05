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
base_model: UsefulSensors/moonshine-streaming-small
---

# Moonshine Streaming Small -- GGUF

GGUF conversions and quantisations of [`UsefulSensors/moonshine-streaming-small`](https://huggingface.co/UsefulSensors/moonshine-streaming-small) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `moonshine-streaming-small.gguf` | F32 | 535 MB | Full precision |
| `moonshine-streaming-small-q4_k.gguf` | Q4_K | 243 MB | Quantized |

## Model details

- **Architecture:** Streaming encoder-decoder ASR. Raw-waveform audio frontend (no mel) + sliding-window transformer encoder (10L, 620d) + autoregressive transformer decoder (10L, 512d, SiLU-gated MLP, partial RoPE)
- **Parameters:** 123M
- **Languages:** English
- **License:** MIT
- **Source:** [`UsefulSensors/moonshine-streaming-small`](https://huggingface.co/UsefulSensors/moonshine-streaming-small)
- **Designed for:** Low-latency streaming ASR on edge devices

## Usage with CrispASR

```bash
./build/bin/crispasr --backend moonshine-streaming -m moonshine-streaming-small-q4_k.gguf -f audio.wav
```

## Notes

- Tokenizer (`tokenizer.bin`) must be in the same directory as the model file
- Streaming architecture: sliding-window attention with 80ms lookahead
- Audio frontend processes raw waveform (no mel spectrogram needed)
