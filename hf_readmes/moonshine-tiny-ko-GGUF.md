---
license: other
license_name: moonshine-ai-community
language:
- ko
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- gguf
- moonshine
- lightweight
library_name: ggml
base_model: UsefulSensors/moonshine-tiny-ko
---

# Moonshine Tiny (Korean) -- GGUF

GGUF conversions and quantisations of [`UsefulSensors/moonshine-tiny-ko`](https://huggingface.co/UsefulSensors/moonshine-tiny-ko) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `moonshine-tiny-ko.gguf` | F32 | 104 MB | Full precision |
| `moonshine-tiny-ko-q4_k.gguf` | Q4_K | 21 MB | Best size/quality tradeoff |

## Model details

- **Architecture:** Conv1d stem + 6L transformer encoder + 6L transformer decoder (288d, 8 heads, partial RoPE, SiLU/GELU)
- **Parameters:** 27M
- **Languages:** Korean (fine-tuned from English moonshine-tiny)
- **License:** [Moonshine AI Community License](https://huggingface.co/UsefulSensors/moonshine-tiny-ja/blob/main/LICENSE.txt) (free for <$1M revenue, attribution required)
- **Source:** [`UsefulSensors/moonshine-tiny-ko`](https://huggingface.co/UsefulSensors/moonshine-tiny-ko)

## Usage with CrispASR

```bash
# Auto-download (English tiny only)
./build/bin/crispasr --backend moonshine -m auto -f audio.wav

# Explicit model path
./build/bin/crispasr --backend moonshine -m moonshine-tiny-ko-q4_k.gguf -f audio.wav
```

## Notes

- Moonshine models run on CPU only (GPU not needed for these small models)
- Tokenizer (`tokenizer.bin`) must be in the same directory as the model file
- Tiny models use head_dim=36 which works on CPU flash_attn
