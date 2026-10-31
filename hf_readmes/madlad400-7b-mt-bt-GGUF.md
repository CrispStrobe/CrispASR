---
license: apache-2.0
language:
- multilingual
base_model:
- google/madlad400-7b-mt-bt
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

# MADLAD-400 7B MT BT — GGUF (ggml)

GGUF / ggml conversion of [`google/madlad400-7b-mt-bt`](https://huggingface.co/google/madlad400-7b-mt-bt) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

MADLAD-400 7B MT BT is Google's 7.2B-parameter T5-based multilingual translation model fine-tuned with **Back-Translation (BT)** on over **450 languages**. The back-translation training significantly improves fluency, domain robustness, and translation quality across underrepresented and high-resource language pairs alike. Distributed under the **Apache 2.0 license**.

## Files Available

| File | Size | Notes |
|---|---:|---|
| `madlad400-7b-mt-bt-f16.gguf` | ~14.5 GB | Full precision baseline (reference quality) |
| `madlad400-7b-mt-bt-q8_0.gguf` | ~7.6 GB | High fidelity 8-bit quantization |
| `madlad400-7b-mt-bt-q4_k.gguf` | ~4.1 GB | Recommended for general use |
| `madlad400-7b-mt-bt-ref.gguf` | ~1.5 MB | Per-stage verification archive |

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build -j

# 2. Pull model
huggingface-cli download cstr/madlad400-7b-mt-bt-GGUF madlad400-7b-mt-bt-q4_k.gguf --local-dir .

# 3. Translate (uses <2xx> language tags)
./build/bin/crispasr --backend madlad -m madlad400-7b-mt-bt-q4_k.gguf \
    --text "Hello world, how are you today?" \
    -sl en -tl de

# English → Japanese
./build/bin/crispasr --backend madlad -m madlad400-7b-mt-bt-q4_k.gguf \
    --text "Machine learning is changing the world." \
    -sl en -tl ja

# French → Portuguese
./build/bin/crispasr --backend madlad -m madlad400-7b-mt-bt-q4_k.gguf \
    --text "Bonjour le monde!" \
    -sl fr -tl pt
```

## Supported languages (450+)

MADLAD-400 supports over 450 languages using `<2xx>` target language tags (ISO 639 codes). This includes all major world languages plus many low-resource languages. See the [original model card](https://huggingface.co/google/madlad400-7b-mt-bt) for the full language list.

## Architecture

```
Text → SentencePiece tokenizer (256K vocab, shared encoder-decoder)
     → <2xx> target language tag prepended to source text
     → T5 encoder (32 layers, d=1024)
     → T5 decoder (32 layers, d=1024) with cross-attention
     → Greedy decode → translated text
```

## Parity Verification

Converted and verified with [CrispASR](https://github.com/CrispStrobe/CrispASR) per-stage parity test against PyTorch reference outputs:

```bash
crispasr-diff madlad madlad400-7b-mt-bt-q4_k.gguf --ref madlad400-7b-mt-bt-ref.gguf
```

## Related models

- [`cstr/madlad400-3b-mt-GGUF`](https://huggingface.co/cstr/madlad400-3b-mt-GGUF) — MADLAD-400 3B MT
- [`cstr/madlad400-7b-mt-GGUF`](https://huggingface.co/cstr/madlad400-7b-mt-GGUF) — MADLAD-400 7B MT (standard)
- [`cstr/madlad400-10b-mt-GGUF`](https://huggingface.co/cstr/madlad400-10b-mt-GGUF) — MADLAD-400 10B MT
- [`cstr/m2m100-418m-GGUF`](https://huggingface.co/cstr/m2m100-418m-GGUF) — M2M-100, 100 languages, any-to-any
- [`google/madlad400-7b-mt-bt`](https://huggingface.co/google/madlad400-7b-mt-bt) — original PyTorch model

## Citation

```bibtex
@misc{kudugunta2023madlad400,
      title={MADLAD-400: A Multilingual And Document-Level Large Audited Dataset}, 
      author={Sneha Kudugunta and Isaac Caswell and Biao Zhang and Xavier Garcia and Christopher A. Choquette-Choo and Katherine Lee and Derrick Xin and Aditya Kusupati and Romi Stella and Ankur Bapna and Orhan Firat},
      year={2023},
      eprint={2309.04662},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
