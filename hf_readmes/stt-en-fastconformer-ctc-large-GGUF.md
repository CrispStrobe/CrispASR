---
license: cc-by-4.0
base_model: nvidia/stt_en_fastconformer_ctc_large
language:
  - en
tags:
  - automatic-speech-recognition
  - gguf
  - crispasr
  - fastconformer
  - ctc
  - nemo
pipeline_tag: automatic-speech-recognition
---

# stt-en-fastconformer-ctc-large-GGUF

GGUF quantisations of [nvidia/stt_en_fastconformer_ctc_large](https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large) for [CrispASR](https://github.com/CrispStrobe/CrispASR).

| Quant | Size | Description |
|---|---|---|
| F16 | 222 MB | Full precision |
| Q8_0 | 132 MB | 8-bit |
| Q5_0 | 95 MB | 5-bit |
| Q4_K | 83 MB | 4-bit K-quant (recommended) |

## Architecture

18-layer NeMo FastConformer encoder + Conv1d CTC head. d_model=512, 8 heads, 1024 SentencePiece vocab, English only, 80 log-mel features, ~115M params.

## Usage

```bash
crispasr --backend fastconformer-ctc -m stt-en-fastconformer-ctc-large-q4_k.gguf -f audio.wav
```

## NeMo Family

Same `--backend fastconformer-ctc` supports large (18L/512d), xlarge (24L/1024d), and xxlarge (42L/1024d).
