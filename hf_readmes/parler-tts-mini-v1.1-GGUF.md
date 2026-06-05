---
license: apache-2.0
language:
- en
base_model:
- parler-tts/parler-tts-mini-v1.1
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- parler-tts
- musicgen
- dac
- gguf
- crispasr
library_name: ggml
---

# Parler TTS Mini v1.1 — GGUF (ggml-quantised)

GGUF / ggml conversion of [`parler-tts/parler-tts-mini-v1.1`](https://huggingface.co/parler-tts/parler-tts-mini-v1.1) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Parler TTS is a prompt-conditioned text-to-speech model: describe the desired voice in natural language and the model generates matching speech. Architecture: T5 encoder (flan-t5-large, 24 layers) encodes the voice description, a MusicGen-style causal decoder (24 layers, 9 codebooks) generates DAC audio tokens autoregressively, and a DAC 44 kHz codec decoder synthesises the final waveform. Distributed under **Apache 2.0 license**.

Single GGUF contains all three components (T5 encoder + decoder + DAC codec).

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `parler-tts-mini-v1.1-f16.gguf`  | F16  | 1.8 GB | Reference quality |
| `parler-tts-mini-v1.1-q8_0.gguf` | Q8_0 | 979 MB | Recommended |
| `parler-tts-mini-v1.1-q4_k.gguf` | Q4_K | 569 MB | Smallest (DAC codec kept at F16) |

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build -j --target crispasr-cli

# 2. Download
huggingface-cli download cstr/parler-tts-mini-v1.1-GGUF parler-tts-mini-v1.1-q8_0.gguf --local-dir .

# 3. Synthesise
./build/bin/crispasr --backend parler-tts \
    -m parler-tts-mini-v1.1-q8_0.gguf \
    --instruct "A female speaker with a warm, natural voice delivers her words at a moderate pace in a quiet environment." \
    --tts "Hello, this is a test of Parler TTS." \
    --tts-output output.wav

# 4. Auto-download shortcut
./build/bin/crispasr -m parler-tts --auto-download \
    --instruct "A young man speaks clearly in a studio." \
    --tts "The quick brown fox jumps over the lazy dog." \
    --tts-output fox.wav
```

## Architecture

| Component | Params | Details |
|---|---|---|
| T5 Encoder | ~335M | flan-t5-large encoder, d=1024, 16 heads, 24 layers, gated-GELU FFN, relative position bias |
| Decoder | ~300M | MusicGen-style causal transformer, d=1024, 16 heads, 24 layers, 9 codebooks, sinusoidal PE |
| DAC Codec | ~75M | Descript Audio Codec 44 kHz, 9 codebooks x 1024, Snake activations, 512x upsample |

## Voice description

The `--instruct` parameter controls voice characteristics. Examples:

- `"A female speaker with a warm, natural voice delivers her words at a moderate pace in a quiet environment."`
- `"A young man speaks clearly with an enthusiastic tone in a professional studio setting."`
- `"An elderly woman reads softly with a gentle pace, slight background noise."`

## Conversion

```bash
python models/convert-parler-to-gguf.py \
    --input parler-tts/parler-tts-mini-v1.1 \
    --output parler-tts-mini-v1.1-f16.gguf
```

## Quantization notes

DAC audio codec weights are kept at F16 in all quantized variants — audio codecs are
precision-sensitive and quantization noise produces audible artefacts. Only T5 encoder
and MusicGen decoder weights are quantized. The BPE tokenizer is embedded in the GGUF
(`parler.tokenizer.is_bpe=true`) so the C++ runtime auto-selects the correct algorithm.

## Limitations

- Greedy decoding (temperature=0) produces degenerate output; use temperature=1.0 (default)
- C++ RNG (`std::mt19937`) differs from PyTorch RNG — same seed produces different audio
- Generation quality varies with the voice description — more specific descriptions yield better results
- No streaming support yet — audio is generated in one pass
- Maximum ~30 s audio per generation (2580 AR steps at 44.1 kHz / 512 hop)

## License

Apache 2.0 — same as the upstream model.
