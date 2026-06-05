---
license: mit
language:
- ar
- en
base_model:
- oddadmix/lahgtna-chatterbox-v1
- ResembleAI/chatterbox
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- chatterbox
- arabic
- flow-matching
- gguf
- crispasr
library_name: ggml
---

# lahgtna-chatterbox-v1 — GGUF (ggml-quantised)

GGUF / ggml conversion of [`oddadmix/lahgtna-chatterbox-v1`](https://huggingface.co/oddadmix/lahgtna-chatterbox-v1) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Arabic T3 (text-to-speech-tokens) variant of Chatterbox TTS. The T3 language model is finetuned for Arabic text while sharing the same S3Gen vocoder as the base English model. Distributed under **MIT license**.

This repo contains only the **T3 model** (Arabic text → speech tokens). You also need the shared **S3Gen model** from [`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF) to produce audio.

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `chatterbox-t3-f16.gguf` | F16 | 1.1 GB | Arabic T3 — reference quality |

For S3Gen (vocoder), download from the base Chatterbox repo:
```bash
huggingface-cli download cstr/chatterbox-GGUF chatterbox-s3gen-q8_0.gguf --local-dir .
```

## Architecture

Same as base Chatterbox but with Arabic-trained T3:
- T3: 30-layer Llama AR (1024D, 16 heads, RoPE, SwiGLU, CFG)
- Multilingual tokenizer (704 text tokens + Arabic graphemes)
- S3Gen + HiFTGenerator vocoder shared with base English model

## Related models

- [`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF) — Base English Chatterbox (required for S3Gen)
- [`oddadmix/lahgtna-chatterbox-v1`](https://huggingface.co/oddadmix/lahgtna-chatterbox-v1) — Original weights

## Conversion

```bash
python models/convert-chatterbox-to-gguf.py \
    --input oddadmix/lahgtna-chatterbox-v1 \
    --output-dir . --t3-only
```

## License

MIT — same as upstream [oddadmix/lahgtna-chatterbox-v1](https://huggingface.co/oddadmix/lahgtna-chatterbox-v1).
