---
license: apache-2.0
language:
- de
- fr
- it
- pt
- es
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- gguf
- ggml
- crispasr
- qwen3
base_model: HojoAI/Hojo-ASR-Multi-V1
library_name: ggml
---

# Hojo-ASR-Multi-V1 — GGUF

GGUF conversions of [`HojoAI/Hojo-ASR-Multi-V1`](https://huggingface.co/HojoAI/Hojo-ASR-Multi-V1) for
**[CrispASR](https://github.com/CrispStrobe/CrispASR)** (`--backend hojo-asr`).

Hojo-ASR-Multi-V1 is an encoder–adapter–LLM ASR model for German, French, Italian, Portuguese and Spanish:

| component | details |
| --- | --- |
| audio encoder | Qwen3-Omni audio tower, 32 layers, d=1280, conv stem fusing 8 mel frames into one 1280-d frame |
| adapter | 2 WeNet conformer blocks, d=2560, 1:1 in time (no frame stacking) |
| decoder | Qwen3-4B-Instruct-2507, lm_head tied to the embeddings |
| front end | whisper-large-v3 feature extractor (128 mel bins) |

All weights come from the checkpoint's single `merged_full_model.safetensors`. The 30B Qwen3-Omni checkpoint
named in its config is only read for audio hyper-parameters and is never downloaded.

## Files

| File | Size |
| --- | ---: |
| `hojo-asr-multi-v1-f16.gguf`  | 9.6 GB |
| `hojo-asr-multi-v1-q4_k.gguf` | 4.4 GB (default for `-m hojo-asr`) |

## Verification against the upstream package

A reference dump comes from the upstream Python package, via forward hooks, on German and French FLEURS-style
clips. CrispASR's `crispasr-diff` compares every stage:

| stage (F16, worst frame cos) | German | French |
| --- | --- | --- |
| mel spectrogram | 1.000000 | 1.000000 |
| encoder output | 0.999998 | 0.999999 |
| adapter output | 0.999994 | 0.999995 |
| prefill embeddings | 0.999997 | 0.999997 |
| first-step logits | 1.000000 | 1.000000 |

End to end, the CrispASR CLI's greedy transcript is **identical** to the upstream package's greedy transcript on
both clips at F16, and at Q4_K on French. At Q4_K on German, one word differs ("weit" for "breed").

Decoding is **greedy by default**. The model's `config.yaml` names beam 4, available with `-bs 4`, but CrispASR's
beam search replays each beam's prefix (O(B·T²)) and its ranking does not yet reproduce `transformers`'
`generate()`. Upstream's own greedy output is what the CLI matches.

## Usage

```bash
crispasr -m hojo-asr -f audio.wav            # auto-download Q4_K
crispasr --backend hojo-asr -m hojo-asr-multi-v1-f16.gguf -f audio.wav -l de
```

## Licence

Apache-2.0, as the upstream model. Conversion and runtime: [CrispASR](https://github.com/CrispStrobe/CrispASR).
