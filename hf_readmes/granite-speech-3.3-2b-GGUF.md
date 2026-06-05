---
license: apache-2.0
base_model: ibm-granite/granite-speech-3.3-2b
language:
  - en
  - fr
  - de
  - es
  - pt
tags:
  - automatic-speech-recognition
  - gguf
  - crispasr
pipeline_tag: automatic-speech-recognition
---

# granite-speech-3.3-2b-GGUF

GGUF quantisations of [ibm-granite/granite-speech-3.3-2b](https://huggingface.co/ibm-granite/granite-speech-3.3-2b) for [CrispASR](https://github.com/CrispStrobe/CrispASR).

| Quant | Description |
|---|---|
| F16 | Full precision |
| Q8_0 | 8-bit |
| Q5_0 | 5-bit |
| Q4_K | 4-bit K-quant (recommended) |

## Usage

```bash
crispasr -m granite-speech-3.3-2b-q4_k.gguf -f audio.wav
```
