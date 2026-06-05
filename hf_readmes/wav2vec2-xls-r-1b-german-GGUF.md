---
license: apache-2.0
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
base_model: jonatasgrosman/wav2vec2-xls-r-1b-german
---

# Wav2Vec2 XLS-R 1B German -- GGUF

GGUF conversions and quantisations of [`jonatasgrosman/wav2vec2-xls-r-1b-german`](https://huggingface.co/jonatasgrosman/wav2vec2-xls-r-1b-german) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `wav2vec2-xls-r-1b-german-q4_k.gguf` | Q4_K | ~600 MB | Best size/quality tradeoff |

## Model details

- **Architecture:** Wav2Vec2ForCTC — CNN feature extractor + 48L transformer encoder (1280d, 16 heads, stable (pre-norm)) + CTC head
- **Parameters:** 1B
- **Language:** German
- **Vocab:** 37 characters (CTC greedy decode)
- **License:** APACHE-2.0
- **Source:** [`jonatasgrosman/wav2vec2-xls-r-1b-german`](https://huggingface.co/jonatasgrosman/wav2vec2-xls-r-1b-german)
- Largest German wav2vec2 model. Best quality, needs more compute.

## Usage with CrispASR

```bash
./build/bin/crispasr --backend wav2vec2 -m wav2vec2-xls-r-1b-german-q4_k.gguf -f german_audio.wav -l de

# Auto-download (default German model):
./build/bin/crispasr --backend wav2vec2 -m auto --auto-download -l de -f audio.wav
```
