---
license: cc-by-4.0
base_model: nvidia/stt_en_fastconformer_ctc_xlarge
language:
  - en
tags:
  - automatic-speech-recognition
  - gguf
  - crispasr
pipeline_tag: automatic-speech-recognition
---

# stt-en-fastconformer-ctc-xlarge-GGUF

GGUF quantisations of [nvidia/stt_en_fastconformer_ctc_xlarge](https://huggingface.co/nvidia/stt_en_fastconformer_ctc_xlarge) for [CrispASR](https://github.com/CrispStrobe/CrispASR).

| Quant | Description |
|---|---|
| F16 | Full precision |
| Q8_0 | 8-bit |
| Q5_0 | 5-bit |
| Q4_K | 4-bit K-quant (recommended) |

## Usage

```bash
crispasr -m stt-en-fastconformer-ctc-xlarge-q4_k.gguf -f audio.wav
```
