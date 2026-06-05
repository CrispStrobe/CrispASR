---
license: mit
pipeline_tag: audio-to-audio
tags:
- audio
- speech
- gguf
- tokenizer
- rvq
library_name: ggml
base_model: XiaomiMiMo/MiMo-Audio-Tokenizer
---

# MiMo Audio Tokenizer (encoder only) -- GGUF

GGUF conversion of the **encoder** from [`XiaomiMiMo/MiMo-Audio-Tokenizer`](https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `mimo-tokenizer-q4_k.gguf` | Q4_K | 377 MB | Encoder + RVQ codebooks |

## Model details

- **Architecture:** 32-layer transformer encoder (1280d, 20 heads) + Conv1d stem + 20 RVQ codebooks
- **Parameters:** ~600M (encoder only, decoder/vocoder excluded)
- **Audio:** 24kHz input, outputs RVQ tokens at 25 Hz (8 channels used by ASR)
- **License:** MIT
- **Source:** [`XiaomiMiMo/MiMo-Audio-Tokenizer`](https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer)

## Notes

- Only the encoder is included (waveform → RVQ tokens). Decoder/vocoder (TTS reconstruction) excluded.
- Used as the first stage of MiMo-V2.5-ASR pipeline (tokenizer → LLM)
