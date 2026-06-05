---
license: apache-2.0
language:
- multilingual
pipeline_tag: voice-activity-detection
tags:
- audio
- vad
- voice-activity-detection
- gguf
- firered
- dfsmn
library_name: ggml
base_model: FireRedTeam/FireRedVAD
---

# FireRedVAD -- GGUF

GGUF conversions of [`FireRedTeam/FireRedVAD`](https://huggingface.co/FireRedTeam/FireRedVAD) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Variant | Size | Params | Notes |
|---|---|---|---|---|
| `firered-vad.gguf` | VAD | 2.4 MB | 588K | Non-streaming, lookback+lookahead |
| `firered-stream-vad.gguf` | Stream-VAD | 2.3 MB | 568K | Streaming (no lookahead) |
| `firered-aed-vad.gguf` | AED | 2.4 MB | 589K | Multi-label: speech/singing/music |

All variants are F32 (no quantization needed — models are already tiny).

## Model details

- **Architecture:** DFSMN (Deep Feedforward Sequential Memory Network) — 8 blocks with depthwise lookback/lookahead convolutions (k=20)
- **Parameters:** ~588K (2.4 MB)
- **Languages:** 100+ (language-agnostic voice activity detection)
- **F1 Score:** 97.57% on FLEURS-VAD-102 (outperforms Silero-VAD, TEN-VAD, FunASR-VAD, WebRTC-VAD)
- **License:** Apache 2.0

## Conversion

```bash
python models/convert-firered-vad-to-gguf.py --input FireRedTeam/FireRedVAD --variant VAD --output firered-vad.gguf
```
