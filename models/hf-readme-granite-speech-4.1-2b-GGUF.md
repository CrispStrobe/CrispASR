---
license: apache-2.0
language:
- en
- fr
- de
- es
- pt
- ja
base_model:
- ibm-granite/granite-speech-4.1-2b
tags:
- asr
- speech
- gguf
- crispasr
---

# granite-speech-4.1-2b — GGUF

GGUF conversions of [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

## Files

| File | Quantisation | Size | Notes |
|---|---|---|---|
| `granite-speech-4.1-2b-f16.gguf` | F16 | ~5.2 GB | Encoder + projector in F32, LLM weights in F16 — full parity reference |
| `granite-speech-4.1-2b-q4_k.gguf` | Q4_K | ~2.94 GB | **Recommended.** LLM layers Q4_K; encoder + projector kept F32 (precision-sensitive). Bit-identical-quality to F16 on encoder + projector |
| `granite-speech-4.1-2b-q4_k-f16enc.gguf` | Q4_K + F16 encoder | ~2.07 GB | LLM Q4_K, encoder + projector F16 (norms / biases / BN stats stay F32). Sweet spot: ~1 GB smaller than the recommended Q4_K with virtually no parity loss |
| `granite-speech-4.1-2b-q4_k-mini.gguf` | Q4_K (aggressive) | ~1.7 GB | Encoder, projector and LLM all Q4_K. Smaller / faster to download but lower cosine parity (~0.93). Still produces correct transcriptions on JFK and similar clips, but expect quality regressions on harder material |

## Cosine parity (vs PyTorch BF16 reference, JFK 11 s clip)

| Stage | F16 cos_min | Q4_K cos_min | Q4_K-f16enc cos_min | Q4_K-mini cos_min |
|---|---|---|---|---|
| mel_spectrogram | 0.999997 | 0.999997 | 0.999997 | 0.999997 |
| encoder_out | 0.999908 | 0.999908 | 0.999855 | 0.929 |
| projector_out | 0.999995 | 0.999995 | 0.999993 | 0.922 |

The encoder is a 16-layer Conformer where Q4_K rounding error compounds
across layers; the recommended Q4_K file pins those weights at F32 to
preserve numerical fidelity. The `-f16enc` file relaxes that to F16 and
ships ~1 GB smaller while keeping cosine essentially indistinguishable
from F16 (every Whisper / Llama / parakeet GGUF in the wild already
runs F16 weights). The `-mini` file applies Q4_K to every quantisable
2D weight including the encoder — useful when disk or download size
matters more than transcript quality.

_Tested with `crispasr-diff granite-4.1 <model.gguf> <ref.gguf> samples/jfk.wav`_

## Architecture

Granite Speech 4.1 2B is a speech-LLM with three components:

- **Encoder**: 16-layer Macaron Conformer (hidden 1024, 8 heads, 15-tap depthwise conv, dual CTC heads for characters + BPE). Input: 80-bin log-mel × 2-frame stacked = 160-dim, 10 ms hop.
- **Projector**: 2-layer BLIP-2 Q-Former with 3 learned queries per 15-frame window (5× temporal downsampling). Combined with encoder's 2× → 10 Hz acoustic token rate for the LLM.
- **LLM**: Granite 4.0-1B (40 layers, 2048 hidden, GQA 16/4, SwiGLU, RoPE θ=10000, μP multipliers).

Total ~2.2 B parameters. Named "2B" to reflect the full system size rather than the base LLM alone.

## Usage with CrispASR

```bash
# auto-download and transcribe
crispasr --backend granite-4.1 -m auto samples/audio.wav

# or with explicit path
crispasr --backend granite-4.1 \
  -m granite-speech-4.1-2b-q4_k.gguf \
  samples/audio.wav
```

Supported tasks via prompt (`-p` flag):

| Task | Prompt |
|---|---|
| ASR (raw) | `can you transcribe the speech into a written format?` |
| ASR (with punctuation) | `transcribe the speech with proper punctuation and capitalization.` |
| AST to English | `translate the speech to English.` |
| AST with punctuation | `translate the speech to English with proper punctuation and capitalization.` |

Supported languages: English, French, German, Spanish, Portuguese, Japanese.

## Conversion

```bash
# Convert HF safetensors → GGUF F16
python models/convert-granite-speech-to-gguf.py \
  --input /path/to/granite-speech-4.1-2b \
  --output granite-speech-4.1-2b-f16.gguf

# Quantise F16 → Q4_K
crispasr-quantize granite-speech-4.1-2b-f16.gguf \
                  granite-speech-4.1-2b-q4_k.gguf q4_k
```

The converter handles all three Granite Speech 4.x releases (4.0-1b, 4.1-2b) from the same script; parameters are read from `config.json` at conversion time.

## Licence

Apache 2.0 — same as the original [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b).
