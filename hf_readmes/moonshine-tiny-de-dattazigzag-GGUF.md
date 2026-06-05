---
license: mit
language:
- de
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- gguf
- moonshine
- lightweight
- german
library_name: ggml
base_model: dattazigzag/moonshine-tiny-de
---

# Moonshine Tiny (German, dattazigzag) -- GGUF

GGUF conversions and quantisations of [`dattazigzag/moonshine-tiny-de`](https://huggingface.co/dattazigzag/moonshine-tiny-de) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `moonshine-tiny-de-dattazigzag-f16.gguf` | F16 | 52 MB | Half precision |
| `moonshine-tiny-de-dattazigzag-q4_k.gguf` | Q4_K | 17 MB | Best size/quality tradeoff |

## Model details

- **Architecture:** Conv1d stem + 6L transformer encoder + 6L transformer decoder (288d, 8 heads, partial RoPE, SiLU/GELU)
- **Parameters:** 27M
- **Languages:** German (fine-tuned from English moonshine-tiny)
- **WER:** 36.7% on MLS German test set
- **Training data:** MLS German (~1,967 hours), 10k steps
- **Output:** Lowercase only, no punctuation
- **License:** MIT (inherited from upstream)
- **Source:** [`dattazigzag/moonshine-tiny-de`](https://huggingface.co/dattazigzag/moonshine-tiny-de)

## Usage with CrispASR

```bash
# Explicit model path
./build/bin/crispasr --backend moonshine -m moonshine-tiny-de-dattazigzag-q4_k.gguf -f audio.wav
```

## Notes

- Moonshine models run on CPU only (GPU not needed for these small models)
- Tokenizer (`tokenizer.bin`) must be in the same directory as the model file
- For better German quality, consider `moonshine-base-de-fidoriel` (6.9% WER) or `moonshine-tiny-de-fidoriel` (11.4% WER)
