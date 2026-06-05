---
license: mit
language:
- en
- de
- fr
- es
- bg
- it
- pl
- nl
- cs
- pt
- sk
- sl
tags:
- gguf
- ggml
- punctuation
- text-processing
- xlm-roberta
- nlp
base_model: kredor/punctuate-all
pipeline_tag: token-classification
---

# Punctuate-All (GGUF)

GGUF conversion of [kredor/punctuate-all](https://huggingface.co/kredor/punctuate-all) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

Adds punctuation to unpunctuated ASR output. **12 languages** with ASCII punctuation output. Smaller and faster alternative to fullstop-punc-multilang (base vs large).

## Model Details

- **Architecture**: XLM-RoBERTa-base — 12L, d=768, 12 heads, d_ffn=3072, GELU
- **Parameters**: ~278M
- **Classifier**: Linear(768, 6) — 6 punctuation classes
- **Labels**: none, `.` (period), `,` (comma), `?` (question), `-` (dash), `:` (colon)
- **Vocabulary**: SentencePiece (250,002 tokens)
- **Max sequence**: 512 tokens (auto-chunked)
- **Languages**: en, de, fr, es, bg, it, pl, nl, cs, pt, sk, sl
- **License**: MIT

## Usage with CrispASR

```bash
crispasr --backend wav2vec2 -m wav2vec2.gguf --punc-model punctuate-all-q4_k.gguf -f audio.wav
```

## Available Files

| File | Quant | Size | Description |
|------|-------|------|-------------|
| `punctuate-all-f16.gguf` | F16 | 901 MB | Half precision |
| `punctuate-all-q4_k.gguf` | Q4_K | 154 MB | Recommended |

## Original Model

- **Source**: [kredor/punctuate-all](https://huggingface.co/kredor/punctuate-all)
- **Training**: WMT/Europarl dataset
