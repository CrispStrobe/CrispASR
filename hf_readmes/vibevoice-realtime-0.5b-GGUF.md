---
license: mit
language:
- en
- de
- fr
- it
- jp
- kr
- nl
- pl
- pt
- sp
tags:
- tts
- text-to-speech
- vibevoice
- gguf
- crispasr
- voice-pack
- multilingual
base_model: microsoft/VibeVoice-Realtime-0.5B
pipeline_tag: text-to-speech
library_name: ggml
---

# VibeVoice-Realtime-0.5B — GGUF

GGUF conversion of [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

## Model variants

| File | Quant | Size | Notes |
|---|---|---:|---|
| `vibevoice-realtime-0.5b-tts-f16.gguf` | F16 | 2.0 GB | Full precision, reference quality |
| `vibevoice-realtime-0.5b-q8_0.gguf`    | Q8_0 | 1.1 GB | Near-lossless |
| `vibevoice-realtime-0.5b-q4_k.gguf`    | Q4_K | 607 MB | **Recommended** — perfect ASR round-trip |

## Voice prompts

A voice prompt is **required** for TTS. Each `.gguf` voice pack ships pre-computed KV caches that establish a fixed speaker identity (the realtime variant is *not* WAV-cloning — for runtime cloning, use [`cstr/vibevoice-1.5b-GGUF`](https://huggingface.co/cstr/vibevoice-1.5b-GGUF) instead).

This repo bundles all 25 demo voices from [`microsoft/VibeVoice@main demo/voices/streaming_model/`](https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model) — same MIT license as the model.

| File | Speaker | Language |
|---|---|---|
| `vibevoice-voice-emma.gguf`              | Emma  (F)   | English (alias of en-Emma_woman) |
| `vibevoice-voice-en-Emma_woman.gguf`     | Emma  (F)   | English |
| `vibevoice-voice-en-Carter_man.gguf`     | Carter (M)  | English |
| `vibevoice-voice-en-Davis_man.gguf`      | Davis  (M)  | English |
| `vibevoice-voice-en-Frank_man.gguf`      | Frank  (M)  | English |
| `vibevoice-voice-en-Grace_woman.gguf`    | Grace  (F)  | English |
| `vibevoice-voice-en-Mike_man.gguf`       | Mike   (M)  | English |
| `vibevoice-voice-de-Spk0_man.gguf`       | Spk0   (M)  | German |
| `vibevoice-voice-de-Spk1_woman.gguf`     | Spk1   (F)  | German |
| `vibevoice-voice-fr-Spk0_man.gguf`       | Spk0   (M)  | French |
| `vibevoice-voice-fr-Spk1_woman.gguf`     | Spk1   (F)  | French |
| `vibevoice-voice-in-Samuel_man.gguf`     | Samuel (M)  | Indian English |
| `vibevoice-voice-it-Spk0_woman.gguf`     | Spk0   (F)  | Italian |
| `vibevoice-voice-it-Spk1_man.gguf`       | Spk1   (M)  | Italian |
| `vibevoice-voice-jp-Spk0_man.gguf`       | Spk0   (M)  | Japanese |
| `vibevoice-voice-jp-Spk1_woman.gguf`     | Spk1   (F)  | Japanese |
| `vibevoice-voice-kr-Spk0_woman.gguf`     | Spk0   (F)  | Korean |
| `vibevoice-voice-kr-Spk1_man.gguf`       | Spk1   (M)  | Korean |
| `vibevoice-voice-nl-Spk0_man.gguf`       | Spk0   (M)  | Dutch |
| `vibevoice-voice-nl-Spk1_woman.gguf`     | Spk1   (F)  | Dutch |
| `vibevoice-voice-pl-Spk0_man.gguf`       | Spk0   (M)  | Polish |
| `vibevoice-voice-pl-Spk1_woman.gguf`     | Spk1   (F)  | Polish |
| `vibevoice-voice-pt-Spk0_woman.gguf`     | Spk0   (F)  | Portuguese |
| `vibevoice-voice-pt-Spk1_man.gguf`       | Spk1   (M)  | Portuguese |
| `vibevoice-voice-sp-Spk0_woman.gguf`     | Spk0   (F)  | Spanish |
| `vibevoice-voice-sp-Spk1_man.gguf`       | Spk1   (M)  | Spanish |

Each voice pack is ~2-6 MB. The `vibevoice-voice-emma.gguf` filename is kept as the legacy default (referenced by `crispasr -m auto --backend vibevoice-tts` in CrispASR's auto-download manifest); `vibevoice-voice-en-Emma_woman.gguf` is the canonical upstream-named copy.

## Usage

```bash
# English with Emma
crispasr --backend vibevoice-tts \
    -m vibevoice-realtime-0.5b-q4_k.gguf \
    --voice vibevoice-voice-emma.gguf \
    --tts "Hello, how are you today?" \
    --tts-output hello.wav

# Japanese with jp-Spk1_woman
crispasr --backend vibevoice-tts \
    -m vibevoice-realtime-0.5b-q4_k.gguf \
    --voice vibevoice-voice-jp-Spk1_woman.gguf \
    --tts "こんにちは、これは日本語の音声テストです。" \
    --tts-output jp.wav
```

Output: 24 kHz mono WAV. Use `crispasr -m auto --backend vibevoice-tts` to auto-download the model + the default Emma voice.

## Architecture

VibeVoice-Realtime-0.5B is a streaming text-to-speech model:

- **Base LM**: 4-layer Qwen2 (text encoding with voice context)
- **TTS LM**: 20-layer Qwen2 (speech conditioning, autoregressive)
- **Prediction head**: 4 AdaLN + SwiGLU layers (flow matching denoiser)
- **DPM-Solver++**: 20-step 2nd-order midpoint solver (cosine schedule, v-prediction)
- **Classifier-Free Guidance**: dual KV cache, cfg_scale=3.0
- **sigma-VAE decoder**: 7-stage transposed ConvNeXt (3200× upsample to 24 kHz)
- **EOS classifier**: automatic length detection

## Quality verification

All quantisations produce exact ASR round-trip matches on English:

| Input text | Parakeet ASR output |
|---|---|
| "Hello world" | "Hello world." |
| "Hello, how are you today?" | "Hello, how are you today?" |
| "The quick brown fox jumps over the lazy dog" | "The quick brown fox jumps over the lazy dog." |
| "Good morning everyone" | "Good morning, everyone." |

## Conversion

Model:
```bash
python models/convert-vibevoice-to-gguf.py \
    --input microsoft/VibeVoice-Realtime-0.5B \
    --output vibevoice-realtime-0.5b-tts-f16.gguf \
    --include-decoder

build/bin/crispasr-quantize vibevoice-realtime-0.5b-tts-f16.gguf \
    vibevoice-realtime-0.5b-q4_k.gguf q4_k
```

Voice packs (one per upstream `.pt`):
```bash
python models/convert-vibevoice-voice-to-gguf.py \
    --input demo/voices/streaming_model/en-Carter_man.pt \
    --output vibevoice-voice-en-Carter_man.gguf
```

## Attribution

- **Original model:** [`microsoft/VibeVoice-Realtime-0.5B`](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) (MIT) — Microsoft Research.
- **Voice packs:** demo voices from [`microsoft/VibeVoice@main`](https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model) (MIT).
- **GGUF + ggml runtime:** [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR) — see `src/vibevoice.cpp`, `models/convert-vibevoice-to-gguf.py`, and `models/convert-vibevoice-voice-to-gguf.py`.

## License

MIT (same as the upstream model and voice prompts).
