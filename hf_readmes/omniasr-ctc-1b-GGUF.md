---
license: apache-2.0
language:
- multilingual
tags:
- gguf
- ggml
- audio
- speech-recognition
- transcription
- omniasr
- wav2vec2
- ctc
- automatic-speech-recognition
- en
- de
- fr
- es
- ja
- ko
- zh
- ar
- hi
- pt
- ru
base_model: facebook/omniASR-CTC-1B
pipeline_tag: automatic-speech-recognition
---

# OmniASR-CTC-1B (GGUF)

GGUF conversion of Facebook's [omniASR-CTC-1B](https://github.com/facebookresearch/omnilingual-asr) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

## Model Details

- **Architecture**: wav2vec2 encoder (48L, d=1280) + CTC head
- **Parameters**: ~1B
- **Encoder**: 7-layer CNN frontend (320x downsampling) + 48L transformer with grouped positional convolution
- **Output**: CTC greedy decoding (character-level SentencePiece, 9812 tokens)
- **Languages**: 1100+ (multilingual, trained on 3.6M hours)
- **License**: Apache 2.0
- **Input**: Raw 16 kHz mono PCM (no mel features)

## Usage with CrispASR

```bash
# Auto-detected from GGUF metadata (omniasr-ctc arch)
crispasr --backend omniasr -m omniasr-ctc-1b-q4_k.gguf -f audio.wav

# With language specification
crispasr --backend omniasr -m omniasr-ctc-1b-q4_k.gguf -l de -f audio.wav
```

## Available Files

| File | Quant | Size | Description |
|------|-------|------|-------------|
| `omniasr-ctc-1b.gguf` | F16 | 1.9 GB | Full precision |
| `omniasr-ctc-1b-q4_k.gguf` | Q4_K | 551 MB | Recommended — good balance of quality and size |

## Conversion

Converted using:
```bash
python models/convert-omniasr-ctc-to-gguf.py \
  --input facebook/omniASR-CTC-1B \
  --output omniasr-ctc-1b.gguf
```

Quantized with:
```bash
crispasr-quantize omniasr-ctc-1b.gguf omniasr-ctc-1b-q4_k.gguf Q4_K
```

## Comparison with OmniASR-CTC-300M

| Model | Params | Size (Q4_K) | Accuracy |
|-------|--------|-------------|----------|
| OmniASR-CTC-300M | 300M | 157 MB | Good |
| **OmniASR-CTC-1B** | 1B | 551 MB | Better (deeper encoder) |

Both use the same CTC architecture — the 1B variant has 48 encoder layers (vs 24) and wider hidden dimension (1280 vs 1024).

## Original Model

- **Paper**: [OmniASR: Transcribing Every Language Everywhere All at Once](https://arxiv.org/abs/2502.10219)
- **Code**: [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)
- **Training Data**: 3.6M hours across 1100+ languages
