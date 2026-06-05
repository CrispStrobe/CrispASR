---
license: mit
language:
- en
- zh
tags:
- tts
- text-to-speech
- vibevoice
- gguf
- crispasr
- voice-cloning
base_model: microsoft/VibeVoice-1.5B
pipeline_tag: text-to-speech
---

# VibeVoice-1.5B GGUF

GGUF conversion of [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

This is the **base model** (not the streaming variant). It supports voice cloning from audio samples and multi-speaker synthesis.

## Model variants

| File | Quant | Size | Notes |
|------|-------|------|-------|
| `vibevoice-1.5b-tts-f16.gguf` | F16 | 5.1 GB | Full precision |
| `vibevoice-1.5b-tts-q8_0.gguf` | Q8_0 | 2.8 GB | Near-lossless |
| `vibevoice-1.5b-tts-q4_k.gguf` | Q4_K | 1.6 GB | Smallest, perfect ASR round-trip |

## Usage

Requires a **voice reference audio** (WAV file, 24 kHz mono) for voice cloning:

```bash
# Voice cloning TTS
VIBEVOICE_VOICE_AUDIO=reference_voice.wav \
crispasr --tts "Hello, how are you today?" \
    -m vibevoice-1.5b-tts-q4_k.gguf \
    --tts-output output.wav
```

## Architecture

Single-LM architecture (differs from the streaming Realtime-0.5B):

- **LM**: Qwen2.5-1.5B (d=1536, 28 layers, 12 heads, 2 KV heads)
- **Prediction head**: 4 AdaLN + SwiGLU layers (d=1536)
- **Acoustic encoder**: 7-stage ConvNeXt (3200x downsample from 24kHz)
- **Semantic encoder**: same architecture, 128-dim latent
- **Decoder**: 7-stage transposed ConvNeXt (3200x upsample)
- **DPM-Solver++**: 20-step, cosine schedule, v-prediction

The model generates speech tokens autoregressively — the LM produces
`<|vision_pad|>` (speech_diffusion) tokens that trigger diffusion sampling,
with `<|vision_start|>` / `<|vision_end|>` as control tokens.

## Quality

| Input | Parakeet ASR |
|-------|-------------|
| "Hello, how are you today?" | "Hello, how are you today?" |

## Differences from Realtime-0.5B

| Feature | Realtime-0.5B | 1.5B Base |
|---------|--------------|-----------|
| Architecture | 4L base + 20L TTS LM | Single 28L LM |
| Voice input | Pre-computed .pt prompts | Audio WAV files |
| Voice cloning | No (fixed presets) | Yes (from reference audio) |
| Multi-speaker | No | Yes (up to 4 speakers) |
| Streaming | Yes | No |

## License

MIT (same as original model).
