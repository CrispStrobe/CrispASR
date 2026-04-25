---
license: mit
language:
- en
tags:
- tts
- text-to-speech
- vibevoice
- gguf
- crispasr
base_model: microsoft/VibeVoice-Realtime-0.5B
pipeline_tag: text-to-speech
---

# VibeVoice-Realtime-0.5B GGUF

GGUF conversion of [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

## Model variants

| File | Quant | Size | Notes |
|------|-------|------|-------|
| `vibevoice-realtime-0.5b-tts-f16.gguf` | F16 | 2.0 GB | Full precision, reference quality |
| `vibevoice-realtime-0.5b-q8_0.gguf` | Q8_0 | 1.1 GB | Near-lossless quantization |
| `vibevoice-realtime-0.5b-q4_k.gguf` | Q4_K | 607 MB | Smallest, still perfect ASR round-trip |

## Voice prompts

A voice prompt is **required** for TTS. It contains pre-computed KV caches that establish speaker identity.

| File | Speaker | Size |
|------|---------|------|
| `vibevoice-voice-emma.gguf` | Emma (female, English) | 2.7 MB |

Convert additional voices from the [VibeVoice demo voices](https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model):

```bash
python models/convert-vibevoice-voice-to-gguf.py \
    --input en-Carter_man.pt --output vibevoice-voice-carter.gguf
```

## Usage

```bash
# Basic TTS
crispasr --tts "Hello, how are you today?" \
    -m vibevoice-realtime-0.5b-q4_k.gguf \
    --voice vibevoice-voice-emma.gguf \
    --tts-output hello.wav

# Output: 24 kHz mono WAV
```

## Architecture

VibeVoice-Realtime-0.5B is a streaming text-to-speech model with:

- **Base LM**: 4-layer Qwen2 (text encoding with voice context)
- **TTS LM**: 20-layer Qwen2 (speech conditioning, autoregressive)
- **Prediction head**: 4 AdaLN + SwiGLU layers (flow matching denoiser)
- **DPM-Solver++**: 20-step 2nd-order midpoint solver (cosine schedule, v-prediction)
- **Classifier-Free Guidance**: dual KV cache, cfg_scale=3.0
- **sigma-VAE decoder**: 7-stage transposed ConvNeXt (3200x upsample to 24kHz)
- **EOS classifier**: automatic length detection

## Quality verification

All quantizations produce exact ASR round-trip matches:

| Input text | Parakeet ASR output |
|-----------|-------------------|
| "Hello world" | "Hello world." |
| "Hello, how are you today?" | "Hello, how are you today?" |
| "The quick brown fox jumps over the lazy dog" | "The quick brown fox jumps over the lazy dog." |
| "Good morning everyone" | "Good morning, everyone." |

## Conversion

```bash
# From HuggingFace model
python models/convert-vibevoice-to-gguf.py \
    --input microsoft/VibeVoice-Realtime-0.5B \
    --output vibevoice-realtime-0.5b-tts-f16.gguf \
    --include-decoder

# Quantize
crispasr-quantize vibevoice-realtime-0.5b-tts-f16.gguf \
    vibevoice-realtime-0.5b-q4_k.gguf q4_k
```

## License

MIT (same as the original model).
