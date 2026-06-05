---
license: apache-2.0
language:
- en
base_model:
- nari-labs/Dia-1.6B
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- dia
- dac
- dialogue
- gguf
- crispasr
library_name: ggml
---

# Dia-1.6B — GGUF (ggml)

GGUF / ggml conversion of [`nari-labs/Dia-1.6B`](https://huggingface.co/nari-labs/Dia-1.6B) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Dia is a dialogue text-to-speech model that generates expressive 44.1 kHz speech from text, with `[S1]` / `[S2]` speaker tags:

- **Text encoder** (12-layer, 1024-d, byte-level vocab 256): encodes the prompt bytes.
- **Audio decoder** (18-layer, 2048-d, GQA 16 query / 4 KV heads, classifier-free guidance): autoregressively emits **9 interleaved DAC codebooks** under a delay pattern `[0,8,9,10,11,12,13,14,15]`.
- **DAC codec** (44.1 kHz): decodes the 9 codebooks to PCM. Shipped as a separate **required** companion file.

Released under **Apache 2.0**.

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `dia-1.6b-f16.gguf` | F16 | 3.0 GB | Main model — reference quality |
| `dac-44khz.gguf`    | —   | 104 MB | DAC codec — **required** companion (download both) |

> Lower-bit quants (Q8_0 / Q4_K) are not published yet: Dia uses `scale=1.0`
> attention (no `1/√d`), which is precision-sensitive, so quants need an
> ASR-roundtrip check before release.

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target crispasr-cli

# 2. Download model + DAC codec
hf download cstr/dia-1.6b-GGUF dia-1.6b-f16.gguf dac-44khz.gguf --local-dir .

# 3. Synthesize (keep the codec beside the model, or pass --codec-model)
./build/bin/crispasr --backend dia -m dia-1.6b-f16.gguf \
    --codec-model dac-44khz.gguf \
    --tts "[S1] Hello there, how are you doing today? I really hope you are having a wonderful and pleasant time." \
    --tts-output hello.wav --seed 42
```

Or with auto-download (pulls the model + DAC companion):
```bash
./build/bin/crispasr -m dia --auto-download \
    --tts "[S1] The quick brown fox jumps over the lazy dog, and then it runs back again." \
    --tts-output fox.wav
```

> **Prompt length matters.** Dia is inconsistent on very short inputs (it may
> emit non-speech) — use prompts of **at least ~100 characters**. Start the
> text with a `[S1]` (or `[S2]`) speaker tag.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--seed N` | 0 | RNG seed (0 = non-deterministic; output varies per seed) |
| `-tp N` | 1.2 | Sampling temperature |
| `--codec-model PATH` | auto | DAC codec GGUF (auto-discovered beside the model) |
| `--tts-output PATH` | — | Output WAV path (44.1 kHz mono) |

## Architecture details

- **Text tokenizer**: byte-level (vocab 256); `[S1]`/`[S2]` map to bytes `0x01`/`0x02`.
- **Encoder**: 12 layers, 1024-d, 16 heads, head_dim 128, RoPE (NeoX half-split), `scale=1.0`.
- **Decoder**: 18 layers, 2048-d; self-attn GQA 16q/4kv; cross-attn MHA (16/16) over the encoder; `scale=1.0`; CFG `cond + cfg_scale·(cond − uncond)`.
- **Codebooks**: 9 DAC channels, delay pattern `[0,8,9,…,15]`, audio vocab 1024.
- **Codec**: Descript Audio Codec (DAC) at 44.1 kHz.

## Conversion

```bash
python models/convert-dia-to-gguf.py \
    --input nari-labs/Dia-1.6B \
    --output dia-1.6b-f16.gguf
```

## Acknowledgements

- [nari-labs/dia](https://github.com/nari-labs/dia) — original model and inference code
- [descript/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) — DAC codec
