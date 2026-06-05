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
base_model: Qwen/Qwen3-TTS-12Hz-0.6B-Base
---

# Qwen3-TTS 12Hz 0.6B Base — GGUF (CrispASR)

GGUF / ggml conversions of [`Qwen/Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) for use with the `qwen3-tts` backend in **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Qwen3-TTS 12Hz 0.6B Base is Qwen's multilingual **voice-cloning** TTS model:

- 10 supported languages: `zh en ja ko de fr ru pt es it`
- discrete multi-codebook LM architecture with a separate 12 Hz tokenizer / codec
- runtime voice cloning from `(ref_audio, ref_text)` or pre-baked voice-pack GGUFs
- Apache-2.0 licence

This repo contains the **talker / code-predictor / speaker-encoder** model. It must be used together with the separate tokenizer / codec GGUF from [`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF).

## Files

File | Size | Notes
--- | --- | ---
`qwen3-tts-12hz-0.6b-base.gguf` | 1.7 GB | F16
`qwen3-tts-12hz-0.6b-base-q8_0.gguf` | 940 MB | Q8_0, recommended quantised talker

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
huggingface-cli download cstr/qwen3-tts-0.6b-base-GGUF \
    qwen3-tts-12hz-0.6b-base-q8_0.gguf --local-dir .

huggingface-cli download cstr/qwen3-tts-tokenizer-12hz-GGUF \
    qwen3-tts-tokenizer-12hz.gguf --local-dir .
```

Voice clone from a reference WAV:

```bash
./build/bin/crispasr \
    --backend qwen3-tts \
    -m qwen3-tts-12hz-0.6b-base-q8_0.gguf \
    --codec-model qwen3-tts-tokenizer-12hz.gguf \
    --voice clone.wav \
    --ref-text "Exact transcript of clone.wav" \
    --tts "Hello there" \
    --tts-output hello.wav
```

Use a baked voice-pack GGUF:

```bash
./build/bin/crispasr \
    --backend qwen3-tts \
    -m qwen3-tts-12hz-0.6b-base-q8_0.gguf \
    --codec-model qwen3-tts-tokenizer-12hz.gguf \
    --voice my-voice-pack.gguf \
    --tts "Hello there" \
    --tts-output hello.wav
```

When `--voice` points to a `.wav`, `--ref-text` is required. When `--voice` points to a `.gguf`, it is treated as a baked voice pack.

## Quantisation Notes

Current CrispASR validation status:

- `qwen3-tts-12hz-0.6b-base.gguf`
  - reference baseline
- `qwen3-tts-12hz-0.6b-base-q8_0.gguf`
  - recommended quantised deployment
  - end-to-end synthesis is audibly good in current CrispASR testing
  - intermediate activations still drift measurably from F16 in strict tensor diffs

Lower-bit talker quants such as `q6_k`, `q5_k`, and `q4_k` can still load and
produce usable speech, but they are not numerically faithful to the F16
reference. They should be treated as experimental builds, not as quality-
equivalent replacements.

If fidelity matters more than memory:

- use `qwen3-tts-12hz-0.6b-base.gguf`
- keep the companion tokenizer / codec at F16

If you want the best currently-tested size / quality trade-off:

- use `qwen3-tts-12hz-0.6b-base-q8_0.gguf`
- still keep `qwen3-tts-tokenizer-12hz.gguf` at F16

## How this was made

1. The upstream HF safetensors checkpoint was converted to GGUF F16 by [`models/convert-qwen3-tts-to-gguf.py`](https://github.com/CrispStrobe/CrispASR/blob/main/models/convert-qwen3-tts-to-gguf.py).
2. Quantised variants are produced with CrispASR's GGUF quantiser.
3. Inference is implemented in [`src/qwen3_tts.cpp`](https://github.com/CrispStrobe/CrispASR/blob/main/src/qwen3_tts.cpp), using ggml graphs for the talker / code-predictor path and the companion tokenizer GGUF for codec encode/decode.

## Reference implementation

Architecture and behaviour were checked against the official Qwen release:

- upstream model card: [`Qwen/Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- upstream repository: [`QwenLM/Qwen3-TTS`](https://github.com/QwenLM/Qwen3-TTS)
- technical report: [`Qwen3-TTS Technical Report`](https://huggingface.co/papers/2601.15621)

The CrispASR runtime is a clean C++ / ggml re-implementation for this repo's backend stack.

## Related

- Companion tokenizer / codec GGUF: [`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF)
- Upstream Base model: [`Qwen/Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- C++ runtime: [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR)

## License

Apache-2.0, inherited from the base model.
