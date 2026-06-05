---
license: other
license_name: moonshine-ai-community
language:
- ja
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- gguf
- moonshine
- lightweight
library_name: ggml
base_model: UsefulSensors/moonshine-base-ja
---

# Moonshine Base (Japanese) -- GGUF

GGUF conversions and quantisations of [`UsefulSensors/moonshine-base-ja`](https://huggingface.co/UsefulSensors/moonshine-base-ja) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `moonshine-base-ja.gguf` | F32 | 235 MB | Full precision |
| `moonshine-base-ja-q4_k.gguf` | Q4_K | 40 MB | Best size/quality tradeoff |

## Model details

- **Architecture:** Conv1d stem + 8L transformer encoder + 8L transformer decoder (416d, 8 heads, partial RoPE, SiLU/GELU)
- **Parameters:** 61M
- **Languages:** Japanese (fine-tuned from English moonshine-base)
- **License:** [Moonshine AI Community License](https://huggingface.co/UsefulSensors/moonshine-tiny-ja/blob/main/LICENSE.txt) (free for <$1M revenue, attribution required)
- **Source:** [`UsefulSensors/moonshine-base-ja`](https://huggingface.co/UsefulSensors/moonshine-base-ja)

## Usage with CrispASR

```bash
# Auto-download (English tiny only)
./build/bin/crispasr --backend moonshine -m auto -f audio.wav

# Explicit model path
./build/bin/crispasr --backend moonshine -m moonshine-base-ja-q4_k.gguf -f audio.wav
```

## Notes

- Moonshine models run on CPU only (GPU not needed for these small models)
- Tokenizer (`tokenizer.bin`) must be in the same directory as the model file
- Base models use head_dim=52 which works on CPU flash_attn
