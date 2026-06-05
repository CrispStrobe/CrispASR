---
license: apache-2.0
language:
- zh
- en
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- gguf
- firered
- conformer
- ctc
- mandarin
- chinese
library_name: ggml
base_model: FireRedTeam/FireRedASR2-AED
---

# FireRedASR2-AED -- GGUF

GGUF conversions and quantisations of [`FireRedTeam/FireRedASR2-AED`](https://huggingface.co/FireRedTeam/FireRedASR2-AED) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `firered-asr2-aed.gguf` | F16 | 2.3 GB | Full precision |
| `firered-asr2-aed-q8_0.gguf` | Q8_0 | 1.4 GB | High quality |
| `firered-asr2-aed-q4_k.gguf` | Q4_K | 919 MB | Best size/quality tradeoff |

All variants produce identical transcription on test audio.

## Model details

- **Architecture:** Conformer encoder (16L, d=1280, 20 heads, relative positional encoding with pos_bias_u/v, macaron FFN, depthwise separable conv k=33) + CTC head
- **Parameters:** 1.1B
- **Languages:** Mandarin Chinese, English, 20+ Chinese dialects
- **License:** Apache 2.0
- **CER:** 3.05% (Mandarin average, per paper)
- **Encoder:** Hybrid ggml/CPU — ggml for matmuls, CPU for relative position attention scoring

## Usage with CrispASR

```bash
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -S . -B build && cmake --build build -j8

# Auto-detect backend from GGUF
./build/bin/crispasr -m firered-asr2-aed-q4_k.gguf -f audio.wav

# Explicit backend
./build/bin/crispasr --backend firered-asr -m firered-asr2-aed-q4_k.gguf -f audio.wav -osrt
```

Note: Output is in UPPERCASE (the model was trained with uppercase English text). CTC decoding is used; beam search decoder not yet implemented.

## Conversion

```bash
python models/convert-firered-asr-to-gguf.py --input FireRedTeam/FireRedASR2-AED --output firered-asr2-aed.gguf
crispasr-quantize firered-asr2-aed.gguf firered-asr2-aed-q4_k.gguf q4_k
```
