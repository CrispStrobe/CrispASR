---
license: mit
tags:
  - tts
  - text-to-speech
  - voice-cloning
  - flow-matching
  - gguf
  - cpp
language:
  - en
  - zh
library_name: crispasr
pipeline_tag: text-to-speech
base_model: SWivid/F5-TTS
---

# F5-TTS v1 Base — GGUF

Native C++ GGUF conversion of [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) (MIT license) for the [CrispASR](https://github.com/CrispStrobe/CrispASR) runtime.

## Model file

| File | Size | Description |
|------|------|-------------|
| `f5-tts-v1-base-f16.gguf` | 953 MB | F16 weights for DiT/Vocos, F32 for critical paths (AdaLN, RoPE, time embed) |

**Note on quantization:** F5-TTS uses a 32-step iterative ODE solver
where each step runs the full 22-layer DiT twice (for CFG). This means
every weight matrix is used 1408 times per synthesis. Q8_0's ~0.5%
per-operation error compounds multiplicatively across these passes,
producing unintelligible output — even when the conditioning pathway
(AdaLN, timestep MLP) is kept at F32. F16's ~0.001% error survives
the 1408× accumulation. This is inherent to flow-matching architecture,
not a ggml limitation. The converter supports `--quant q8_0` for
experimentation, but F16 is the only recommended format.

## Architecture

- **DiT backbone**: 22-layer Diffusion Transformer with AdaLN-Zero (330M params)
- **Text encoder**: Character-level ConvNeXtV2 (4 blocks, 512-d)
- **Vocoder**: Vocos (8× ConvNeXt + ISTFTHead, 13M params)
- **ODE solver**: 32-step Euler with CFG (strength=2.0, sway=-1.0)
- **Output**: 24 kHz mono PCM
- **Voice cloning**: Zero-shot from 3-15s reference audio + transcript

Single GGUF contains both DiT and Vocos — no separate codec model needed.

## Usage

```bash
# Install / build CrispASR
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -B build && cmake --build build -j$(nproc) --target crispasr-cli

# Synthesize with voice cloning
./build/bin/crispasr --backend f5-tts -m auto \
    --voice reference.wav \
    --ref-text "Transcript of the reference audio" \
    --tts "Hello, how are you today?" \
    --tts-output output.wav --seed 42
```

The `--ref-text` flag is required — F5-TTS conditions on both audio and its transcript for voice cloning.

## Conversion

Converted from `SWivid/F5-TTS` safetensors using:
```bash
python models/convert-f5-tts-to-gguf.py \
    --model-dir /path/to/f5-tts \
    --output f5-tts-v1-base-f16.gguf
```

Quantization beyond F16 is not recommended for flow-matching models.

## License

MIT (same as upstream SWivid/F5-TTS).
