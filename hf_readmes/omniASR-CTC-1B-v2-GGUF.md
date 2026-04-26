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
base_model: aadel4/omniASR-CTC-1B-v2
---

# OmniASR CTC-1B-v2 — GGUF

GGUF conversion of [`aadel4/omniASR-CTC-1B-v2`](https://huggingface.co/aadel4/omniASR-CTC-1B-v2) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

OmniASR is Meta's **multilingual ASR** model family supporting **1600+ languages**. Apache-2.0 license.

Perfect output on all audio lengths. **Recommended CTC model.**

## Files

| File | Size |
| --- | ---: |
| `omniasr-ctc-1b-v2-q4_k.gguf` | 551 MB |
| `omniasr-ctc-1b-v2-q8_0.gguf` | 1007 MB |
| `omniasr-ctc-1b-v2.gguf` | 1.8 GB |

## Quick Start

```bash
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

./build/bin/crispasr --backend omniasr -m auto --auto-download -f audio.wav
```

## Conversion

Converted using CrispASR's converter scripts with fixed positional conv weight normalization (per-kernel-position norm, not per-output-channel).
