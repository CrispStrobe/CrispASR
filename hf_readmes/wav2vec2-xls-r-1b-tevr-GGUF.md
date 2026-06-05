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
base_model: fxtentacle/wav2vec2-xls-r-1b-tevr
---

# Wav2Vec2 XLS-R 1B TEVR -- GGUF

GGUF conversions and quantisations of [`fxtentacle/wav2vec2-xls-r-1b-tevr`](https://huggingface.co/fxtentacle/wav2vec2-xls-r-1b-tevr) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `wav2vec2-xls-r-1b-tevr-q4_k.gguf` | Q4_K | ~600 MB | Best size/quality tradeoff |

## Model details

- **Architecture:** Wav2Vec2ForCTC — CNN feature extractor + 48L transformer encoder (1280d, 16 heads, stable (pre-norm)) + CTC head
- **Parameters:** 1B
- **Language:** German
- **Vocab:** 256 characters (CTC greedy decode)
- **License:** APACHE-2.0
- **Source:** [`fxtentacle/wav2vec2-xls-r-1b-tevr`](https://huggingface.co/fxtentacle/wav2vec2-xls-r-1b-tevr)
- Byte-level CTC (vocab=256). Handles any character without OOV. German ASR.

## Usage with CrispASR

```bash
./build/bin/crispasr --backend wav2vec2 -m wav2vec2-xls-r-1b-tevr-q4_k.gguf -f german_audio.wav -l de

# Auto-download (default German model):
./build/bin/crispasr --backend wav2vec2 -m auto --auto-download -l de -f audio.wav
```
