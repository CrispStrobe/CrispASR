---
license: mit
language:
- en
- ha
- is
- ja
- cs
- ru
- zh
- de
base_model:
- facebook/wmt21-dense-24-wide-x-en
pipeline_tag: translation
tags:
- translation
- m2m_100
- wmt21
- text2text-generation
- multilingual
- encoder-decoder
- gguf
- crispasr
library_name: ggml
arxiv: "2108.03265"
---

# WMT21 Dense 24-Wide (X→EN) — GGUF (ggml)

GGUF / ggml conversion of [`facebook/wmt21-dense-24-wide-x-en`](https://huggingface.co/facebook/wmt21-dense-24-wide-x-en) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

This is the **many-to-English** WMT21 competition model — a 4.7B-parameter dense encoder-decoder transformer trained for high-quality translation from 7 source languages into English. It won the WMT21 shared task for multiple language pairs. Architecture: 24 encoder + 24 decoder layers (d=2048, 32 heads, FFN=16384, ReLU, pre-norm, sinusoidal positions). Distributed under **MIT license**.

## Files

| File | Size | Notes |
|---|---:|---|
| `wmt21-dense-24-wide-x-en-f16.gguf` | 8.8 GB | F16 weights (reference quality) |
| `wmt21-dense-24-wide-x-en-q8_0.gguf` | 4.7 GB | Q8_0 quantized (identical quality to F16 on test set) |
| `wmt21-dense-24-wide-x-en-q4_k.gguf` | 2.5 GB | Q4_K quantized (minor word choice differences) |

## Supported languages (7 → English)

| Code | Language |
|---|---|
| de | German |
| cs | Czech |
| ru | Russian |
| ja | Japanese |
| zh | Chinese |
| is | Icelandic |
| ha | Hausa |

Target is always English (`en`).

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build -j

# 2. Pull model
huggingface-cli download cstr/wmt21-dense-24-wide-x-en-GGUF wmt21-dense-24-wide-x-en-q8_0.gguf --local-dir .

# 3. Translate
./build/bin/crispasr --backend m2m100 -m wmt21-dense-24-wide-x-en-q8_0.gguf \
    --text "Die Maschine lernt schnell und verändert die Welt." \
    -sl de -tl en

# Japanese → English
./build/bin/crispasr --backend m2m100 -m wmt21-dense-24-wide-x-en-q8_0.gguf \
    --text "機械学習は世界を変えています。" \
    -sl ja -tl en
```

## Quality verification

| Input | Language | F16 | Q8_0 | Q4_K |
|---|---|---|---|---|
| Hallo Welt, wie geht es dir heute? | de | Hello world, how are you today? | Hello world, how are you today? | Hello world, how are you today? |
| Die Maschine lernt schnell und verändert die Welt. | de | The machine learns quickly and changes the world. | The machine learns quickly and changes the world. | The machine learns quickly and changes the world. |
| 機械学習は世界を変えています。 | ja | Machine learning is changing the world. | Machine learning is changing the world. | Machine learning is changing the world. |
| Машинное обучение меняет мир. | ru | Machine learning is changing the world. | Machine learning is changing the world. | Machine learning is changing the world. |

## Architecture

```
Text → SentencePiece BPE tokenizer (128K vocab, 8 lang codes)
     → Source lang token (__de__) + text tokens + </s>
     → 24-layer transformer encoder (d=2048, 32 heads, FFN=16384, ReLU, pre-norm)
     → Sinusoidal positional embeddings (pre-computed)
     → 24-layer transformer decoder (self-attn + cross-attn + FFN)
     → Shared embedding LM head (tied weights)
     → English forced as first decoder token
     → Greedy decode → translated English text
```

## Conversion

```bash
python models/convert-m2m100-to-gguf.py \
    --input facebook/wmt21-dense-24-wide-x-en \
    --output wmt21-dense-24-wide-x-en-f16.gguf
```

## Related models

- [`cstr/wmt21-dense-24-wide-en-x-GGUF`](https://huggingface.co/cstr/wmt21-dense-24-wide-en-x-GGUF) — English-to-many (reverse direction)
- [`cstr/m2m100-418m-GGUF`](https://huggingface.co/cstr/m2m100-418m-GGUF) — smaller 100-language any-to-any model
- [`facebook/wmt21-dense-24-wide-x-en`](https://huggingface.co/facebook/wmt21-dense-24-wide-x-en) — original PyTorch model
