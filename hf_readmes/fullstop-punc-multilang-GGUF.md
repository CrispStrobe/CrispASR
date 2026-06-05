---
license: mit
language:
- en
- de
- fr
- it
tags:
- gguf
- ggml
- punctuation
- text-processing
- xlm-roberta
- nlp
- truecasing
base_model: oliverguhr/fullstop-punctuation-multilang-large
pipeline_tag: token-classification
---

# Fullstop Punctuation Multilingual (GGUF)

GGUF conversion of [oliverguhr/fullstop-punctuation-multilang-large](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

Adds punctuation to unpunctuated ASR output. **Multilingual** (English, German, French, Italian) with **ASCII punctuation** output.

## Model Details

- **Architecture**: XLM-RoBERTa-large — 24L, d=1024, 16 heads, d_ffn=4096, GELU
- **Parameters**: ~560M
- **Classifier**: Linear(1024, 6) — 6 punctuation classes
- **Labels**: none, `.` (period), `,` (comma), `?` (question), `-` (dash), `:` (colon)
- **Vocabulary**: SentencePiece (250,002 tokens)
- **Max sequence**: 512 tokens (auto-chunked)
- **Languages**: English, German, French, Italian
- **License**: MIT

## Usage with CrispASR

```bash
crispasr --backend wav2vec2 -m wav2vec2.gguf --punc-model fullstop-punc-q4_k.gguf -f audio.wav
```

## Available Files

| File | Quant | Size | Description |
|------|-------|------|-------------|
| `fullstop-punc-f32emb.gguf` | F16+F32emb | 1.6 GB | Full precision (F32 embeddings) |
| `fullstop-punc-q8_0.gguf` | Q8_0 | 572 MB | High quality |
| `fullstop-punc-q4_k.gguf` | Q4_K | 254 MB | Recommended |

## Example

Input: `and so my fellow americans ask not what your country can do for you ask what you can do for your country`

Output: `And so my fellow americans ask not what your country can do for you, ask what you can do for your country.`

## Conversion

```bash
python models/convert-fullstop-punc-to-gguf.py \
  --input oliverguhr/fullstop-punctuation-multilang-large \
  --output fullstop-punc.gguf
```

## Original Model

- **Source**: [oliverguhr/fullstop-punctuation-multilang-large](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large)
- **Paper**: [fullstop-deep-punctuation-prediction](https://github.com/oliverguhr/fullstop-deep-punctuation-prediction)
- **Training**: Europarl Dataset (political speeches)
