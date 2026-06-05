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
base_model: facebook/omniASR-LLM-300M
---

# OmniASR LLM Unlimited-300M v2 — GGUF

GGUF conversion of Meta's `omniASR_LLM_Unlimited_300M_v2` from [`facebookresearch/omnilingual-asr`](https://github.com/facebookresearch/omnilingual-asr) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

OmniASR is Meta's **multilingual ASR** model family supporting **1600+ languages** (1,627,603,584 parameters). Apache-2.0 license.

## Unlimited vs Standard

This is the **Unlimited** variant — it supports **arbitrarily long audio** input, unlike the standard `omniASR_LLM_300M_v2` which is limited to ~40 seconds. Identical parameter count (1.6B), comparable accuracy.

The "300M" in the name refers to the encoder size; the full model with LLM decoder is 1.6B params.

| Variant | Max Audio | Model |
| --- | --- | --- |
| Standard | ~40s | [omniasr-llm-300m-v2-GGUF](https://huggingface.co/cstr/omniasr-llm-300m-v2-GGUF) |
| **Unlimited** | No limit | This repo |

## Files

| File | Size |
| --- | ---: |
| `omniasr-llm-unlimited-300m-v2-f16.gguf` | 3.26 GB |
| `omniasr-llm-unlimited-300m-v2-q4_k.gguf` | 1.08 GB |

## Quick Start

```bash
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

./build/bin/crispasr --backend omniasr-llm \
    -m omniasr-llm-unlimited-300m-v2-q4_k.gguf \
    -f audio.wav
```

## Source

Converted from [`omniASR-LLM-Unlimited-300M-v2.pt`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-Unlimited-300M-v2.pt) published at [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) (not from HuggingFace). Uses CrispASR's converter with fixed positional conv weight normalization (per-kernel-position norm).

## License

Apache-2.0 (same as upstream).
