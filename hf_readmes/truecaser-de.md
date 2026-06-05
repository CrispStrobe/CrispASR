---
license: apache-2.0
language:
- de
tags:
- truecasing
- text-processing
- german
- nlp
- lstm
- crf
pipeline_tag: token-classification
---

# Truecaser Models

Truecasing models for restoring proper capitalization in lowercase ASR output. Used by [CrispASR](https://github.com/CrispStrobe/CrispASR) via `--truecase-model`.

## Available Models

| File | Type | Language | Size | F1 | License | Flag |
|------|------|----------|------|-----|---------|------|
| `truecaser-lstm-de.bin` | BiLSTM char-level | German | 3.2 MB | 97.9% | Apache-2.0 | `lstm` or `lstm-de` |
| `truecaser-lstm-en.bin` | BiLSTM char-level | English | 3.2 MB | 93.0% | Apache-2.0 | `lstm-en` |
| `truecaser-lstm-es.bin` | BiLSTM char-level | Spanish | 3.2 MB | — | Apache-2.0 | `lstm-es` |
| `truecaser-lstm-ru.bin` | BiLSTM char-level | Russian | 4.1 MB | — | Apache-2.0 | `lstm-ru` |
| `truecaser-crf-de.bin` | CRF + context | German | 8.5 MB | ~95% | MIT | `crf` |
| `truecaser-de.bin` | Statistical freq | German | 1.7 MB | ~93% | MIT | `auto` |

## BiLSTM Truecaser (recommended)

Converted from [mayhewsw/pytorch-truecaser](https://github.com/mayhewsw/pytorch-truecaser) (Apache-2.0).

- **Architecture**: Embedding(202, 50) → BiLSTM(50→150, 2 layers) → Linear(300, 2)
- **Labels**: L (lowercase), U (uppercase) — per character
- **Training**: WMT monolingual text (de: 2.6M tokens, 97.86% F1; en: Wikipedia, 93.01% F1; es: WMT; ru: LORELEI)
- **Original paper**: Mayhew et al., "NER and POS When Nothing is Capitalized" (2019)
- **Source**: [mayhewsw/pytorch-truecaser v1.0](https://github.com/mayhewsw/pytorch-truecaser/releases/tag/v1.0) — `wmt-truecaser-model-de.tar.gz`

### Example

```
Input:  die schnelle braune katze springt über den faulen hund
Output: Die schnelle braune Katze springt über den faulen Hund
```

Correctly handles:
- Adjective vs noun: "braune" (lowercase) vs "Katze" (capitalize)
- Formal pronouns: "Ihnen" (capitalize)
- Compound words and proper nouns

## CRF Truecaser

Trained on 245K sentences of WMT News Crawl German using [python-crfsuite](https://github.com/scrapinghub/python-crfsuite).

- **Features**: word identity, 3-char suffix, noun suffixes, previous/next word, article context
- **Decode**: Viterbi over linear-chain CRF (3 labels: lc, u1, uc)
- **Training data**: WMT News Crawl 2023 German (8.5 MB model, MIT license)

## Statistical Truecaser

Simple word-frequency lookup trained on WMT News Crawl 2023 German.

- **Entries**: 71,142 unique words
- **Size**: 1.7 MB
- **Approach**: for each word, pick the casing variant (lowercase/capitalize/uppercase) seen most often
- **Training data**: WMT News Crawl 2023 German (278K sentences), MIT license

## Usage with CrispASR

```bash
# BiLSTM (recommended)
crispasr --backend wav2vec2-de -m model.gguf --truecase-model lstm -f audio.wav

# CRF
crispasr --backend wav2vec2-de -m model.gguf --truecase-model crf -f audio.wav

# Statistical
crispasr --backend wav2vec2-de -m model.gguf --truecase-model auto -f audio.wav

# Combined with punctuation restoration
crispasr --backend moonshine -m model.gguf --punc-model punctuate-all --truecase-model lstm -f audio.wav
```

## Conversion

```bash
# BiLSTM: download from mayhewsw, convert to binary
wget https://github.com/mayhewsw/pytorch-truecaser/releases/download/v1.0/wmt-truecaser-model-de.tar.gz
tar xzf wmt-truecaser-model-de.tar.gz
python models/convert-lstm-truecaser-to-bin.py --input wmt-truecaser-de/ --output truecaser-lstm-de.bin

# CRF: train from Wikipedia
python models/train-truecaser-crf.py --output truecaser-crf-de.bin
```
