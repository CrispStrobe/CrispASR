---
license: apache-2.0
language:
- en
- multilingual
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- gguf
- gemma
- conformer
library_name: ggml
base_model: google/gemma-4-E2B-it
---

# Gemma-4-E2B-it — GGUF

GGUF conversion of [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `gemma4-e2b-it.gguf` | F16 | ~9.5 GB | Full precision |
| `gemma4-e2b-it-q8_0.gguf` | Q8_0 | ~5.0 GB | Near-lossless quant |
| `gemma4-e2b-it-q4_k.gguf` | Q4_K | ~2.8 GB | Standard quant |
| `gemma4-e2b-it-q2_k.gguf` | Q2_K | ~2.2 GB | Smallest, quality drop |

## Model details

- **Architecture:** USM Conformer audio encoder (12L, 1024d, chunked-local attention with relative position bias, LightConv1d, ClippableLinear with QAT scalars) + Gemma4 LLM decoder (35L, 1536d, GQA 8Q/1KV, per-layer embeddings, hybrid sliding/full attention, GeGLU)
- **Parameters:** 2.3B effective (5.1B with embeddings)
- **Audio:** Gemma4AudioFeatureExtractor — 128-bin mel, 16 kHz, frame_length=320, hop=160, fft_length=512, semicausal padding, log(mel + mel_floor=0.001), no normalisation
- **Languages:** 140+ (ASR + speech translation)
- **License:** Apache 2.0
- **Source:** [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)

## What's included vs an upstream Gemma-4 GGUF

This GGUF is built specifically for ASR with CrispASR and includes the audio path that
standard text/vision Gemma-4 GGUFs (unsloth, ggml-org) omit:

- 12-layer audio conformer encoder (~872 tensors total).
- Gemma4MultimodalEmbedder audio→LLM adapter (`embed_audio.embedding_projection`,
  pre-projection RMSNorm).
- All ClippableLinear QAT clipping scalars (`.input_min/max`, `.output_min/max`) — these
  are NOT QAT-only artefacts. HF applies them at inference via
  `Gemma4ClippableLinear.forward`. Skipping them collapses the encoder past layer 5.
- `num_kv_shared_layers`, `layer_full_mask`, `partial_rotary_factor`,
  `global_head_dim`, `use_double_wide_mlp`, `attention_k_eq_v` — all the per-layer
  flags the LLM forward needs to honour.
- Mel filterbank + Hann window resources (HTK no-norm filters,
  `frame_length=320` window; the runtime regenerates these too).

Vision tower tensors are excluded.

## Usage with CrispASR

```bash
# Auto-download (recommended)
./build/bin/crispasr --backend gemma4-e2b -m auto --auto-download -f audio.wav

# Or explicit path
./build/bin/crispasr --backend gemma4-e2b -m gemma4-e2b-it-q4_k.gguf -f audio.wav
```

## Differential testing

CrispASR ships a stage-by-stage differential test against the HF PyTorch
reference. Per-stage cosine similarity vs HF `Gemma4AudioModel`:

```
mel_spectrogram          1.0000   bit-exact (HF FE faithfully reproduced)
audio_subsample_output   0.9994   conv2d + LayerNorm + ReLU
audio_layer_0..11        0.97 — 0.99 (with QAT clip scalars)
audio_tower_output       0.99+
```

Run it yourself:

```bash
# 1. Dump HF reference
HF_HOME=/path/to/hf-cache python tools/dump_reference.py \
    --backend gemma4 --model-dir google/gemma-4-E2B-it \
    --audio samples/jfk.wav --output /tmp/gemma4-ref.gguf

# 2. Compare
build/bin/crispasr-diff gemma4 \
    gemma4-e2b-it-q4_k.gguf /tmp/gemma4-ref.gguf samples/jfk.wav
```

## Conversion provenance

This GGUF was produced by `models/convert-gemma4-e2b-to-gguf.py` (CrispASR repo)
running on Kaggle T4 nodes (16 GB RAM). Conversion config:

- `--outtype f16` then `crispasr-quantize` for Q-variants.
- ClippableLinear QAT scalars persisted as 1-element F32 tensors named
  `audio.layers.{i}.{linear}.input_min/max, output_min/max`.
- Vision tower (`model.vision_tower.*`, `model.embed_vision.*`) skipped.
