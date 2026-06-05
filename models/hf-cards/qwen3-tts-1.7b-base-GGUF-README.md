---
license: apache-2.0
language:
  - zh
  - en
  - ja
  - ko
  - de
  - fr
  - ru
  - pt
  - es
  - it
pipeline_tag: text-to-speech
tags:
  - audio
  - tts
  - voice-clone
  - ggml
  - gguf
  - crispasr
  - qwen3-tts
library_name: ggml
base_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
---

# Qwen3-TTS 12Hz 1.7B Base — GGUF (CrispASR)

GGUF / ggml conversions of [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) for use with the `qwen3-tts` backend in **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Qwen3-TTS 12Hz 1.7B Base is the larger sibling of [`cstr/qwen3-tts-0.6b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF), with the same multilingual voice-cloning behaviour:

- 10 supported languages: `zh en ja ko de fr ru pt es it`
- discrete multi-codebook LM architecture with a separate 12 Hz tokenizer / codec
- runtime voice cloning from `(ref_audio, ref_text)` or pre-baked voice-pack GGUFs
- Apache-2.0 licence

The 1.7B variant has a wider talker (`hidden=2048`, vs 1024 for the 0.6B), a matching ECAPA `enc_dim=2048`, and an additional `small_to_mtp_projection` bridge from the talker into the 1024-d code predictor. The CrispASR runtime handles all three differences automatically — same CLI, same GGUF schema.

This repo contains the **talker / code-predictor / speaker-encoder** model. It must be used together with the separate tokenizer / codec GGUF from [`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF).

## Files

File | Size | Notes
--- | --- | ---
`qwen3-tts-12hz-1.7b-base.gguf` | 3.6 GB | F16
`qwen3-tts-12hz-1.7b-base-q8_0.gguf` | 1.9 GB | Q8_0, recommended quantised talker

## Quick Start

Build CrispASR:

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target crispasr-lib
```

Download the talker + tokenizer:

```bash
huggingface-cli download cstr/qwen3-tts-1.7b-base-GGUF \
    qwen3-tts-12hz-1.7b-base-q8_0.gguf --local-dir .

huggingface-cli download cstr/qwen3-tts-tokenizer-12hz-GGUF \
    qwen3-tts-tokenizer-12hz.gguf --local-dir .
```

Voice clone from a reference WAV:

```bash
./build/bin/crispasr \
    --backend qwen3-tts-1.7b-base \
    -m qwen3-tts-12hz-1.7b-base-q8_0.gguf \
    --codec-model qwen3-tts-tokenizer-12hz.gguf \
    --voice clone.wav \
    --ref-text "Exact transcript of clone.wav" \
    --tts "Hello there" \
    --tts-output hello.wav
```

Or let CrispASR pull both files for you on first run:

```bash
./build/bin/crispasr \
    --backend qwen3-tts-1.7b-base -m auto \
    --voice clone.wav \
    --ref-text "Exact transcript of clone.wav" \
    --tts "Hello there" \
    --tts-output hello.wav
```

When `--voice` points to a `.wav`, `--ref-text` is required. When `--voice` points to a `.gguf`, it is treated as a baked voice pack.

## Quantisation Notes

- `qwen3-tts-12hz-1.7b-base.gguf`
  - reference baseline (3.6 GB)
- `qwen3-tts-12hz-1.7b-base-q8_0.gguf`
  - recommended quantised deployment (1.9 GB)
  - ASR-roundtrips word-exact on English prompts in current CrispASR testing

Lower-bit talker quants (`q6_k`, `q5_k`, `q4_k`) can still load but are not numerically faithful to the F16 reference and should be treated as experimental.

The companion tokenizer / codec should stay at F16.

## How this was made

1. The upstream HF safetensors checkpoint was converted to GGUF F16 by [`models/convert-qwen3-tts-to-gguf.py`](https://github.com/CrispStrobe/CrispASR/blob/main/models/convert-qwen3-tts-to-gguf.py).
2. Quantised variants are produced with CrispASR's GGUF quantiser.
3. Inference is implemented in [`src/qwen3_tts.cpp`](https://github.com/CrispStrobe/CrispASR/blob/main/src/qwen3_tts.cpp), using ggml graphs for the talker / code-predictor path and the companion tokenizer GGUF for codec encode/decode.

## Reference implementation

Architecture and behaviour were checked against the official Qwen release:

- upstream model card: [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- upstream repository: [`QwenLM/Qwen3-TTS`](https://github.com/QwenLM/Qwen3-TTS)

The CrispASR runtime is a clean C++ / ggml re-implementation for this repo's backend stack.

## Related

- Smaller sibling: [`cstr/qwen3-tts-0.6b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF)
- Companion tokenizer / codec GGUF: [`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF)
- Upstream Base model: [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- C++ runtime: [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR)

## License

Apache-2.0, inherited from the base model.
