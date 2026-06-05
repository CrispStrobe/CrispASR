---
license: cc-by-nc-sa-4.0
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
base_model: fidoriel/moonshine-base-de
---

# Moonshine Base (German, fidoriel) -- GGUF

GGUF conversions and quantisations of [`fidoriel/moonshine-base-de`](https://huggingface.co/fidoriel/moonshine-base-de) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `moonshine-base-de-fidoriel-f16.gguf` | F16 | 118 MB | Half precision |
| `moonshine-base-de-fidoriel-q4_k.gguf` | Q4_K | 39 MB | Best size/quality tradeoff |

## Model details

- **Architecture:** Conv1d stem + 8L transformer encoder + 8L transformer decoder (416d, 8 heads, partial RoPE, SiLU/GELU)
- **Parameters:** 61.5M
- **Languages:** German (fine-tuned from English moonshine-base)
- **WER:** 6.9% on Common Voice 22 German test set
- **CER:** 2.4%
- **Training data:** Common Voice 22 German
- **Output:** Proper casing and punctuation
- **License:** [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (inherited from upstream)
- **Source:** [`fidoriel/moonshine-base-de`](https://huggingface.co/fidoriel/moonshine-base-de)

## Usage with CrispASR

```bash
# Explicit model path
./build/bin/crispasr --backend moonshine -m moonshine-base-de-fidoriel-q4_k.gguf -f audio.wav

# Or via backend name (auto-download)
./build/bin/crispasr --backend moonshine-de -m auto -f audio.wav
```

## Notes

- Moonshine models run on CPU only (GPU not needed for these small models)
- Tokenizer (`tokenizer.bin`) must be in the same directory as the model file
- Best quality among the moonshine German variants (6.9% WER vs 11.4% for tiny)
