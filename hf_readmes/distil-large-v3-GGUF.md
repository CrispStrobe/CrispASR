---
license: mit
language:
- en
tags:
- gguf
- ggml
- audio
- speech-recognition
- whisper
- distil-whisper
- automatic-speech-recognition
base_model: distil-whisper/distil-large-v3
pipeline_tag: automatic-speech-recognition
---

# Distil Whisper Large v3 (ggml)

ggml conversion of [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR) and the upstream [whisper.cpp](https://github.com/ggml-org/whisper.cpp) lineage it builds on.

## Model Details

- **Architecture**: Whisper encoder (32 layers, 1280-dim) + distilled decoder (2 layers only)
- **Parameters**: 756M (49% smaller than whisper-large-v3)
- **Speed**: 6.3x faster than whisper-large-v3, within 1% WER
- **Language**: English
- **License**: MIT

## Usage

```bash
# Uses the standard whisper backend (auto-detected)
crispasr -m distil-large-v3-q5_0.bin -f audio.wav
```

## Files

| File | Size | JFK Result |
|------|------|-----------|
| distil-large-v3.bin | 1.5 GB | perfect |
| distil-large-v3-q5_0.bin | 513 MB | perfect |

## Why Distil Whisper?

- **6.3x faster** than whisper-large-v3 (2 decoder layers vs 32)
- **Within 1% WER** on standard benchmarks
- **Same encoder** as whisper-large-v3 (32 layers, 1280-dim)
- **Drop-in replacement** — same ggml format, same CLI flags
