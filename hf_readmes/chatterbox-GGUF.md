---
license: mit
language:
- en
base_model:
- ResembleAI/chatterbox
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- chatterbox
- flow-matching
- hifi-gan
- gguf
- crispasr
library_name: ggml
---

# Chatterbox TTS — GGUF (ggml-quantised)

GGUF / ggml conversion of [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Chatterbox is a full TTS pipeline: character tokenizer → T3 (30-layer Llama AR, 520M) → speech tokens → S3Gen (Conformer encoder + UNet1D CFM denoiser, 10 Euler steps) → HiFTGenerator vocoder (conv chains + Snake activations + iSTFT) → 24 kHz WAV. Distributed under **MIT license**.

Two GGUF files are needed: the **T3 model** (text → speech tokens) and the **S3Gen model** (speech tokens → audio).

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `chatterbox-t3-f16.gguf`     | F16  | 1.1 GB | T3 AR model — reference quality |
| `chatterbox-t3-q8_0.gguf`    | Q8_0 | 542 MB | T3 AR model — recommended |
| `chatterbox-t3-q4_k.gguf`    | Q4_K | 287 MB | T3 AR model — smallest |
| `chatterbox-s3gen-f16.gguf`  | F16  | 548 MB | S3Gen + vocoder — reference quality |
| `chatterbox-s3gen-q8_0.gguf` | Q8_0 | 342 MB | S3Gen + vocoder — recommended |
| `chatterbox-s3gen-q4_k.gguf` | Q4_K | 237 MB | S3Gen + vocoder — smallest |

Note: vocoder weights (conv_pre, resblocks, conv_post, source fusion) are kept at F32 in all quant levels for audio quality. Quantization applies to the Conformer encoder, UNet decoder, and T3 Llama layers.

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build -j --target chatterbox

# 2. Pull both model files
huggingface-cli download cstr/chatterbox-GGUF chatterbox-t3-q8_0.gguf --local-dir .
huggingface-cli download cstr/chatterbox-GGUF chatterbox-s3gen-q8_0.gguf --local-dir .

# 3. Synthesise with the built-in default voice
./build/bin/crispasr --backend chatterbox \
    -m chatterbox-t3-q8_0.gguf \
    --codec-model chatterbox-s3gen-q8_0.gguf \
    --tts "Hello there, this is chatterbox speaking." \
    --tts-output out.wav

# 4. (Optional) clone a different speaker — bake a small voice GGUF
# from a reference WAV, then pass it via --voice. Requires the upstream
# python pkg: pip install chatterbox-tts
python models/bake-chatterbox-voice-from-wav.py \
    --input /path/to/reference.wav \
    --output my_voice.gguf

./build/bin/crispasr --backend chatterbox \
    -m chatterbox-t3-q8_0.gguf \
    --codec-model chatterbox-s3gen-q8_0.gguf \
    --voice my_voice.gguf \
    --tts "Cloned voice synthesising arbitrary text." \
    --tts-output cloned.wav
```

See [`docs/tts.md`](https://github.com/CrispStrobe/CrispASR/blob/main/docs/tts.md#voice-cloning)
for the full Chatterbox voice-clone reference, including the
per-call cache used by `--server` mode.

## Architecture

```
Text → Character tokenizer (704 tokens)
     → T3 Llama AR (30 layers, 1024D, 16 heads, RoPE, SwiGLU, CFG)
     → 25 Hz speech tokens (6561 codebook)
     → Conformer encoder (6 pre + 4 post upsample, 512D, 8 heads)
     → 80-channel mel spectrogram
     → UNet1D CFM denoiser (1 down + 12 mid + 1 up, 256 ch, 10 Euler steps)
     → HiFTGenerator vocoder (3× ConvTranspose1d + 9 ResBlocks + Snake + iSTFT)
     → 24 kHz mono WAV
```

## Quality verification

ASR roundtrip on Python reference mel (no source fusion, deterministic):

| Metric | Value |
|---|---|
| ASR output (moonshine-base) | **"Hello world"** (correct) |
| Per-stage cosine vs Python ref | **1.000** (conv_pre through rb_2) |
| Waveform cosine vs torch.istft | **0.93** |
| STFT range | [-0.82, 2.0] (ref [-1.1, 1.7]) |

All quantization levels (F16/Q8_0/Q4_K) produce ASR-identical output on the reference mel.

## Conversion

```bash
python models/convert-chatterbox-to-gguf.py \
    --input ResembleAI/chatterbox \
    --output-dir .
```

Requires `pip install gguf safetensors torch huggingface_hub`.

## Related models

- [`cstr/lahgtna-chatterbox-v1-GGUF`](https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF) — Arabic T3 variant (MIT, shares S3Gen)
- [`cstr/orpheus-3b-base-GGUF`](https://huggingface.co/cstr/orpheus-3b-base-GGUF) — Llama-3.2 + SNAC TTS
- [`cstr/qwen3-tts-0.6b-customvoice-GGUF`](https://huggingface.co/cstr/qwen3-tts-0.6b-customvoice-GGUF) — Qwen3-TTS with fixed speakers

## License

MIT — same as the upstream [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox).
