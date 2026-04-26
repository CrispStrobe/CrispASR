---
license: mit
language:
- ar
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- gguf
- moonshine
- lightweight
library_name: ggml
base_model: UsefulSensors/moonshine-tiny-ar
---

# Moonshine Tiny (Arabic) -- GGUF

GGUF conversions and quantisations of [`UsefulSensors/moonshine-tiny-ar`](https://huggingface.co/UsefulSensors/moonshine-tiny-ar) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `moonshine-tiny-ar.gguf` | F32 | 104 MB | Full precision |
| `moonshine-tiny-ar-q4_k.gguf` | Q4_K | 21 MB | Best size/quality tradeoff |

## Model details

- **Architecture:** Conv1d stem + 6L transformer encoder + 6L transformer decoder (288d, 8 heads, partial RoPE, SiLU/GELU)
- **Parameters:** 27M
- **Languages:** Arabic (fine-tuned from English moonshine-tiny)
- **License:** MIT
- **Source:** [`UsefulSensors/moonshine-tiny-ar`](https://huggingface.co/UsefulSensors/moonshine-tiny-ar)

## Usage with CrispASR

```bash
# Auto-download (English tiny only)
./build/bin/crispasr --backend moonshine -m auto -f audio.wav

# Explicit model path
./build/bin/crispasr --backend moonshine -m moonshine-tiny-ar-q4_k.gguf -f audio.wav
```

## Notes

- Moonshine models run on CPU only (GPU not needed for these small models)
- Tokenizer (`tokenizer.bin`) must be in the same directory as the model file
- Tiny models use head_dim=36 which works on CPU flash_attn
