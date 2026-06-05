---
license: apache-2.0
language:
- multilingual
base_model:
- google/madlad400-3b-mt
pipeline_tag: translation
tags:
- translation
- t5
- text2text-generation
- multilingual
- encoder-decoder
- gguf
- crispasr
library_name: ggml
arxiv: "2309.04662"
---

# MADLAD-400 3B MT — GGUF (ggml)

GGUF / ggml conversion of [`google/madlad400-3b-mt`](https://huggingface.co/google/madlad400-3b-mt) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

MADLAD-400 is a 3B-parameter T5-based multilingual machine translation model covering **450+ languages**, trained on 1 trillion tokens. It was developed by Google and achieves strong results across a wide range of languages, with special emphasis on low-resource and underrepresented languages. Distributed under **Apache 2.0 license**.

## Files

| File | Size | Notes |
|---|---:|---|
| `madlad400-3b-mt-f16.gguf` | ~5.7 GB | F16 weights (reference quality) |
| `madlad400-3b-mt-q8_0.gguf` | ~3.1 GB | Q8_0 quantized |
| `madlad400-3b-mt-q4_k.gguf` | ~1.8 GB | Q4_K quantized |

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build -j

# 2. Pull model
huggingface-cli download cstr/madlad400-3b-mt-GGUF madlad400-3b-mt-q8_0.gguf --local-dir .

# 3. Translate (uses <2xx> language tags)
./build/bin/crispasr --backend madlad -m madlad400-3b-mt-q8_0.gguf \
    --text "Hello world, how are you today?" \
    -sl en -tl de

# English → Japanese
./build/bin/crispasr --backend madlad -m madlad400-3b-mt-q8_0.gguf \
    --text "Machine learning is changing the world." \
    -sl en -tl ja

# French → Portuguese
./build/bin/crispasr --backend madlad -m madlad400-3b-mt-q8_0.gguf \
    --text "Bonjour le monde!" \
    -sl fr -tl pt
```

## Supported languages (450+)

MADLAD-400 supports over 450 languages using `<2xx>` target language tags (ISO 639 codes). This includes all major world languages plus many low-resource languages. See the [original model card](https://huggingface.co/google/madlad400-3b-mt) for the full language list.

## Architecture

```
Text → SentencePiece tokenizer (256K vocab, shared encoder-decoder)
     → <2xx> target language tag prepended to source text
     → T5 encoder (24 layers, d=1024)
     → T5 decoder (24 layers, d=1024) with cross-attention
     → Greedy decode → translated text
```

## Conversion

```bash
python models/convert-madlad-to-gguf.py \
    --input google/madlad400-3b-mt \
    --output madlad400-3b-mt-f16.gguf
```

## Related models

- [`cstr/m2m100-418m-GGUF`](https://huggingface.co/cstr/m2m100-418m-GGUF) — M2M-100, 100 languages, any-to-any
- [`cstr/wmt21-dense-24-wide-en-x-GGUF`](https://huggingface.co/cstr/wmt21-dense-24-wide-en-x-GGUF) — WMT21 English-to-many (7 langs, highest quality)
- [`cstr/wmt21-dense-24-wide-x-en-GGUF`](https://huggingface.co/cstr/wmt21-dense-24-wide-x-en-GGUF) — WMT21 many-to-English (7 langs)
- [`google/madlad400-3b-mt`](https://huggingface.co/google/madlad400-3b-mt) — original PyTorch model
