---
license: apache-2.0
language:
- en
- zh
- ja
- ko
- de
- fr
- es
- pt
- it
- nl
- ru
- ar
- hi
- vi
- th
- id
- ms
- tl
- tr
- pl
- cs
- sv
- da
- no
- fi
- el
- he
- uk
- ro
tags:
- tts
- text-to-speech
- speech-synthesis
- gguf
- crispasr
base_model: openbmb/VoxCPM2
pipeline_tag: text-to-speech
---

# VoxCPM2 GGUF

GGUF conversion of [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

## Model Details

- **Architecture:** TSLM (28L MiniCPM-4) + RALM (8L) + LocEnc (12L) + LocDiT (12L) + AudioVAE
- **Parameters:** ~2B
- **Output:** 48 kHz mono PCM
- **Languages:** 30+ languages including English, Chinese, Japanese, Korean, German, French, Spanish, and more
- **License:** Apache 2.0
- **Voice Cloning:** Supported via reference audio

## Features

- **Tokenizer-free:** Directly generates audio patches via diffusion (no discrete audio tokens)
- **High quality:** CFM-based generation with classifier-free guidance
- **Multilingual:** 30+ language support via CJK-split BPE tokenizer (73k vocab)
- **Voice cloning:** Zero-shot voice cloning from reference audio

## Usage with CrispASR

```bash
# Zero-shot TTS
crispasr -m voxcpm2-f16.gguf \
    --tts "Hello, this is VoxCPM2 speaking." \
    --tts-output output.wav

# Quantized (smaller, faster)
crispasr -m voxcpm2-q4_k.gguf \
    --tts "Hello world" --tts-output output.wav
```

## Files

| File | Size | Description |
|------|------|-------------|
| `voxcpm2-f16.gguf` | 4.63 GB | F16 weights (full precision) |
| `voxcpm2-q4_k.gguf` | ~1.5 GB | Q4_K quantized (faster, slightly lower quality) |
| `voxcpm2-ref.gguf` | 371 KB | Reference activation dump for numerical validation |

## Numerical Validation

The `voxcpm2-ref.gguf` file contains intermediate activation tensors captured from the PyTorch reference implementation. Used with `crispasr-diff` to validate the C++ inference path:

```bash
crispasr-diff voxcpm2-tts voxcpm2-f16.gguf voxcpm2-ref.gguf samples/jfk.wav
```

Current status: **12 pass, 0 fail** across all transformer stages (TSLM, RALM, LocEnc, LocDiT, projections).

Captured stages: `text_input_ids`, `locenc_in`, `locenc_out`, `enc_to_lm`, `tslm_layer_0_out`, `tslm_layer_27_out`, `tslm_prefill_out`, `ralm_prefill_out`, `lm_to_dit_hidden`, `res_to_dit_hidden`, `cfm_step0_z`, `cfm_step0_result`, `stop_logits_step0`.

## Architecture

```
Text → BPE tokenize → TSLM (28L causal MiniCPM-4, GQA 16h/2kv, LongRoPE)
                       ↓
                    FSQ bottleneck (tanh→round→linear)
                       ↓
              RALM (8L causal, no RoPE, GQA 16h/2kv)
                       ↓
         Projections: lm_to_dit + res_to_dit → mu [2048]
                       ↓
    LocDiT (12L bidirectional, CFM Euler solver, 10 steps, cfg=2.0)
                       ↓
              Predicted latent patch [4 frames × 64 dims]
                       ↓
         LocEnc (12L bidirectional) → next TSLM input
                       ↓ (AR loop until stop)
              AudioVAE decoder → 48 kHz PCM
```

## Conversion

Converted using `models/convert-voxcpm2-to-gguf.py` from the CrispASR repository:

```bash
python models/convert-voxcpm2-to-gguf.py \
    --input openbmb/VoxCPM2 \
    --output voxcpm2-f16.gguf
```

## Acknowledgments

Original model by [OpenBMB](https://github.com/OpenBMB/VoxCPM2). Apache 2.0 license.
