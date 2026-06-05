---
license: other
license_name: moonshine-ai-community
language:
- zh
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- gguf
- moonshine
- lightweight
library_name: ggml
base_model: UsefulSensors/moonshine-base-zh
---

# Moonshine Base (Chinese) -- GGUF

GGUF conversions and quantisations of [`UsefulSensors/moonshine-base-zh`](https://huggingface.co/UsefulSensors/moonshine-base-zh) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `moonshine-base-zh.gguf` | F32 | 235 MB | Full precision |
| `moonshine-base-zh-q4_k.gguf` | Q4_K | 40 MB | Best size/quality tradeoff |

## Model details

- **Architecture:** Conv1d stem + 8L transformer encoder + 8L transformer decoder (416d, 8 heads, partial RoPE, SiLU/GELU)
- **Parameters:** 61M
- **Languages:** Chinese (fine-tuned from English moonshine-base)
- **License:** [Moonshine AI Community License](https://huggingface.co/UsefulSensors/moonshine-tiny-ja/blob/main/LICENSE.txt) (free for <$1M revenue, attribution required)
- **Source:** [`UsefulSensors/moonshine-base-zh`](https://huggingface.co/UsefulSensors/moonshine-base-zh)

## Usage with CrispASR

```bash
# Auto-download (English tiny only)
./build/bin/crispasr --backend moonshine -m auto -f audio.wav

# Explicit model path
./build/bin/crispasr --backend moonshine -m moonshine-base-zh-q4_k.gguf -f audio.wav
```

## Notes

- Moonshine models run on CPU only (GPU not needed for these small models)
- Tokenizer (`tokenizer.bin`) must be in the same directory as the model file
- Base models use head_dim=52 which works on CPU flash_attn
