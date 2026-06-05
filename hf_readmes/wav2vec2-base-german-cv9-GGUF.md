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
- wav2vec2
- german
library_name: ggml
base_model: oliverguhr/wav2vec2-base-german-cv9
---

# Wav2Vec2 Base German (CV9) -- GGUF

GGUF conversions and quantisations of [`oliverguhr/wav2vec2-base-german-cv9`](https://huggingface.co/oliverguhr/wav2vec2-base-german-cv9) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `wav2vec2-base-german-cv9-q4_k.gguf` | Q4_K | ~60 MB | Best size/quality tradeoff |

## Model details

- **Architecture:** Wav2Vec2ForCTC — CNN feature extractor + 12L transformer encoder (768d, 12 heads, post-norm) + CTC head
- **Parameters:** 94M
- **Language:** German
- **Vocab:** 35 characters (CTC greedy decode)
- **License:** MIT
- **Source:** [`oliverguhr/wav2vec2-base-german-cv9`](https://huggingface.co/oliverguhr/wav2vec2-base-german-cv9)
- Small and fast. MIT licensed. Post-norm architecture (wav2vec2-base style).

## Usage with CrispASR

```bash
./build/bin/crispasr --backend wav2vec2 -m wav2vec2-base-german-cv9-q4_k.gguf -f german_audio.wav -l de

# Auto-download (default German model):
./build/bin/crispasr --backend wav2vec2 -m auto --auto-download -l de -f audio.wav
```
