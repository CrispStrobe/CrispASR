---
license: apache-2.0
tags:
  - automatic-speech-recognition
  - wav2vec2
  - gguf
  - crispasr
  - english
base_model: jonatasgrosman/wav2vec2-large-xlsr-53-english
pipeline_tag: automatic-speech-recognition
library_name: ggml
language:
  - en
---

# Wav2Vec2 Large XLSR-53 English — GGUF

GGUF conversion of [jonatasgrosman/wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) for English speech recognition.

## Model details

| Property | Value |
|---|---|
| Architecture | Wav2Vec2 (CNN feature extractor + 24 Transformer layers) |
| Hidden size | 1024 |
| Attention heads | 16 |
| CTC vocabulary | 33 tokens |
| Format | GGUF (F16 weights) |
| Size | 627 MB |

The model was pre-trained on 53 languages (XLSR-53) and fine-tuned on English Common Voice data with a CTC head. It accepts 16 kHz mono audio and outputs character-level transcriptions.

## Usage with CrispASR

```bash
crispasr \
  --backend wav2vec2 \
  -m wav2vec2-xlsr-en.gguf \
  audio.wav
```

## Provenance

Weights were converted from the original Hugging Face PyTorch checkpoint into GGUF format with F16 precision for all transformer and feature-extractor parameters.

## License

Apache-2.0 — same as the original model.
