---
license: apache-2.0
language:
- en
- multilingual
tags:
- speech
- asr
- gguf
- ggml
- omniasr
pipeline_tag: automatic-speech-recognition
base_model: aadel4/omniASR-CTC-300M-v2
---

# OmniASR CTC-300M-v2 — GGUF

GGUF conversion of [`aadel4/omniASR-CTC-300M-v2`](https://huggingface.co/aadel4/omniASR-CTC-300M-v2) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

OmniASR is Meta's **multilingual ASR** model family supporting **1600+ languages**. Apache-2.0 license.

Works on audio ≤5 seconds (model positional encoding limit). Use `--vad` for longer audio.

## Files

| File | Size |
| --- | ---: |
| `omniasr-ctc-300m-v2-q4_k.gguf` | 194 MB |
| `omniasr-ctc-300m-v2-q8_0.gguf` | 343 MB |
| `omniasr-ctc-300m-v2.gguf` | 623 MB |

## Quick Start

```bash
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

./build/bin/crispasr --backend omniasr -m auto --auto-download -f audio.wav
```

## Conversion

Converted using CrispASR's converter scripts with fixed positional conv weight normalization (per-kernel-position norm, not per-output-channel).
