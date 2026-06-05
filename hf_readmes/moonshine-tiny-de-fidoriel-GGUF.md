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
base_model: fidoriel/moonshine-tiny-de
---

# Moonshine Tiny (German, fidoriel) -- GGUF

GGUF conversions and quantisations of [`fidoriel/moonshine-tiny-de`](https://huggingface.co/fidoriel/moonshine-tiny-de) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `moonshine-tiny-de-fidoriel-f16.gguf` | F16 | 52 MB | Half precision |
| `moonshine-tiny-de-fidoriel-q4_k.gguf` | Q4_K | 17 MB | Best size/quality tradeoff |

## Model details

- **Architecture:** Conv1d stem + 6L transformer encoder + 6L transformer decoder (288d, 8 heads, partial RoPE, SiLU/GELU)
- **Parameters:** 27M
- **Languages:** German (fine-tuned from English moonshine-tiny)
- **WER:** 11.4% on Common Voice 22 German test set
- **CER:** 4.2%
- **Output:** Proper casing and punctuation
- **License:** [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (inherited from upstream)
- **Source:** [`fidoriel/moonshine-tiny-de`](https://huggingface.co/fidoriel/moonshine-tiny-de)

## Usage with CrispASR

```bash
# Explicit model path
./build/bin/crispasr --backend moonshine -m moonshine-tiny-de-fidoriel-q4_k.gguf -f audio.wav

# Or via backend name (auto-download)
./build/bin/crispasr --backend moonshine-tiny-de -m auto -f audio.wav
```

## Notes

- Moonshine models run on CPU only (GPU not needed for these small models)
- Tokenizer (`tokenizer.bin`) must be in the same directory as the model file
- Smaller/faster alternative to moonshine-base-de (17 MB vs 39 MB)
