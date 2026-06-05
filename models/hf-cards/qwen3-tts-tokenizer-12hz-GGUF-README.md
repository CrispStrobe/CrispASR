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
pipeline_tag: audio-to-audio
tags:
  - audio
  - tts
  - speech
  - codec
  - ggml
  - gguf
  - crispasr
  - qwen3_tts_tokenizer_12hz
library_name: ggml
base_model: Qwen/Qwen3-TTS-Tokenizer-12Hz
---

# Qwen3-TTS Tokenizer 12Hz — GGUF (CrispASR)

GGUF / ggml conversions of [`Qwen/Qwen3-TTS-Tokenizer-12Hz`](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz) for use with the `qwen3-tts` backend in **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Qwen3-TTS-Tokenizer-12Hz is the separate **speech tokenizer / codec** used by the Qwen3-TTS family:

- 10 supported languages: `zh en ja ko de fr ru pt es it`
- 12.5 Hz, 16-codebook speech representation
- used for reference-audio encoding, voice-pack baking, and final waveform decode
- Apache-2.0 licence

This repo contains the **tokenizer / codec** only. Use it together with the talker GGUF from [`cstr/qwen3-tts-0.6b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF).

## Files

File | Size | Notes
--- | --- | ---
`qwen3-tts-tokenizer-12hz.gguf` | 342 MB | F16
`qwen3-tts-tokenizer-12hz-q8_0.gguf` | 277 MB | Q8_0 codec quant

## Quick Start

```bash
./build/bin/crispasr \
    --backend qwen3-tts \
    -m qwen3-tts-12hz-0.6b-base.gguf \
    --codec-model qwen3-tts-tokenizer-12hz.gguf \
    --voice clone.wav \
    --ref-text "Exact transcript of clone.wav" \
    --tts "Hello there" \
    --tts-output hello.wav
```

The tokenizer GGUF is used for:

- encoding reference audio into `ref_code`
- baking / loading voice-pack GGUFs
- decoding generated codes back into 24 kHz mono WAV

## Quantisation Notes

Current CrispASR validation status:

- `qwen3-tts-tokenizer-12hz.gguf`
  - reference baseline
- `qwen3-tts-tokenizer-12hz-q8_0.gguf`
  - usable, but numerically less faithful than the F16 codec in strict diff tests

For best fidelity, keep the tokenizer / codec at F16 even when quantising the talker. In current CrispASR testing, codec quantisation drifts earlier in the codec-encoder path than talker-only quantisation.

This repo may also publish lower-bit talker variants in the companion talker
repo. If you use them, the safest pairing is still:

- quantised talker
- `qwen3-tts-tokenizer-12hz.gguf` kept at F16

In other words: if you must choose where to keep precision, keep it in the
tokenizer / codec first.

## How this was made

1. The upstream tokenizer / codec checkpoint was converted to GGUF F16 by [`models/convert-qwen3-tts-tokenizer-to-gguf.py`](https://github.com/CrispStrobe/CrispASR/blob/main/models/convert-qwen3-tts-tokenizer-to-gguf.py).
2. Quantised variants are produced with CrispASR's GGUF quantiser.
3. Encode / decode inference is implemented in [`src/qwen3_tts.cpp`](https://github.com/CrispStrobe/CrispASR/blob/main/src/qwen3_tts.cpp), sharing the same runtime as the `qwen3-tts` talker backend.

## Reference implementation

Architecture and behaviour were checked against the official Qwen release:

- upstream tokenizer model card: [`Qwen/Qwen3-TTS-Tokenizer-12Hz`](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz)
- upstream repository: [`QwenLM/Qwen3-TTS`](https://github.com/QwenLM/Qwen3-TTS)
- technical report: [`Qwen3-TTS Technical Report`](https://huggingface.co/papers/2601.15621)

## Related

- Companion talker GGUF: [`cstr/qwen3-tts-0.6b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF)
- Upstream tokenizer: [`Qwen/Qwen3-TTS-Tokenizer-12Hz`](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz)
- C++ runtime: [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR)

## License

Apache-2.0, inherited from the base model.
