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
base_model: oliverguhr/wav2vec2-large-xlsr-53-german-cv13
---

# Wav2Vec2 Large XLSR-53 German (CV13) -- GGUF

GGUF conversions and quantisations of [`oliverguhr/wav2vec2-large-xlsr-53-german-cv13`](https://huggingface.co/oliverguhr/wav2vec2-large-xlsr-53-german-cv13) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `wav2vec2-large-xlsr-53-german-cv13-q4_k.gguf` | Q4_K | ~222 MB | Best size/quality tradeoff |

## Model details

- **Architecture:** Wav2Vec2ForCTC — CNN feature extractor + 24L transformer encoder (1024d, 16 heads, stable (pre-norm)) + CTC head
- **Parameters:** 315M
- **Language:** German
- **Vocab:** 35 characters (CTC greedy decode)
- **License:** APACHE-2.0
- **Source:** [`oliverguhr/wav2vec2-large-xlsr-53-german-cv13`](https://huggingface.co/oliverguhr/wav2vec2-large-xlsr-53-german-cv13)
- Fine-tuned on CommonVoice 13.0 German — newer training data than the original.

## Usage with CrispASR

```bash
./build/bin/crispasr --backend wav2vec2 -m wav2vec2-large-xlsr-53-german-cv13-q4_k.gguf -f german_audio.wav -l de

# Auto-download (default German model):
./build/bin/crispasr --backend wav2vec2 -m auto --auto-download -l de -f audio.wav
```
