---
language:
- en
- fr
- es
- de
- ru
- zh
- ja
- it
- pt
- nl
- ar
- hi
- ko
license: apache-2.0
base_model: mistralai/Voxtral-Mini-4B-Realtime-2602
pipeline_tag: automatic-speech-recognition
tags:
- gguf
- speech-to-text
- realtime
- streaming
- voxtral
---

# Voxtral-Mini-4B-Realtime — GGUF

GGUF quantizations of [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602), a **4.4B-parameter realtime streaming speech-to-text model** with a causal audio encoder and configurable transcription delay.

Converted and tested with [CrispASR](https://github.com/CrispStrobe/CrispASR), a multi-model ASR framework built on ggml.

## Files

| File | Quant | Size | Description |
|------|-------|------|-------------|
| `voxtral-mini-4b-realtime-f16.gguf` | F16 | 8.3 GB | Full precision (reference) — what the quants below are cut from |
| `voxtral-mini-4b-realtime-q8_0.gguf` | Q8_0 | 4.4 GB | 8-bit quantized |
| `voxtral-mini-4b-realtime-q4_k.gguf` | Q4_K | 2.4 GB | 4-bit K-quant (recommended) |

⚠ This table used to name the F16 `voxtral-mini-4b-realtime.gguf`, and no such
file was ever uploaded. The published name is `…-f16.gguf`, matching the quant
suffix CrispASR's `-m auto:f16` resolver expects.

## Performance (CPU, 4 threads, AVX2, jfk.wav 11s)

| Quant | Encoder | Prefill | Decode (ms/tok) | Total | RTFx |
|-------|---------|---------|-----------------|-------|------|
| F16 | 39s | 30s | 430 | 133s | 0.08× |
| Q8_0 | 30s | 9s | 257 | 79s | 0.14× |
| **Q4_K** | **19s** | **3s** | **177** | **49s** | **0.22×** |

Q4_K recommended — 3.5× smaller than F16, 2.7× faster, identical transcription quality.

## Usage

```bash
# Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target voxtral4b-main

# Download Q4_K (recommended)
huggingface-cli download cstr/voxtral-mini-4b-realtime-GGUF \
    voxtral-mini-4b-realtime-q4_k.gguf --local-dir .

# Transcribe
./build/bin/voxtral4b-main -m voxtral-mini-4b-realtime-q4_k.gguf -f audio.wav
```

### With word-level timestamps

```bash
# Also download the CTC aligner
huggingface-cli download cstr/canary-ctc-aligner-GGUF \
    canary-ctc-aligner-q4_k.gguf --local-dir .

./build/bin/voxtral4b-main -m voxtral-mini-4b-realtime-q4_k.gguf \
    -f audio.wav -am canary-ctc-aligner-q4_k.gguf -timestamps
```

### CLI options

```
-m  FNAME   GGUF model file (required)
-f  FNAME   Input audio, 16 kHz mono WAV (required)
-t  N       Threads (default: 4)
-l  LANG    Language hint (default: en)
-n  N       Max new tokens (default: 512)
-am FNAME   CTC aligner GGUF for word timestamps
-timestamps Enable word-level timestamps (requires -am)
-np         Suppress stderr info
```

## Architecture

- **Audio encoder**: 32-layer causal transformer (RoPE, SwiGLU, RMSNorm, sliding window 750)
- **LLM decoder**: 26-layer Mistral (GQA 32/8, SwiGLU, adaptive RMSNorm, sliding window 8192)
- **Projector**: 4-frame stack → Linear(5120→3072) → GELU → Linear(3072→3072)
- **Tokenizer**: Mistral Tekken (150K vocab, 1000 special tokens)
- **Audio injection**: adapter output ADDED to token embeddings (streaming format)

### Key features

- **Natively streaming** architecture with causal encoder
- **13 languages**: en, fr, es, de, ru, zh, ja, it, pt, nl, ar, hi, ko
- **Configurable delay**: 480ms default (6 tokens × 80ms)
- **Apache 2.0** license

## Conversion

```bash
python models/convert-voxtral4b-to-gguf.py \
    --input /path/to/Voxtral-Mini-4B-Realtime-2602 \
    --output voxtral-mini-4b-realtime-f16.gguf

# Then quantize
./build/bin/crispasr-quantize voxtral-mini-4b-realtime-f16.gguf \
    voxtral-mini-4b-realtime-q4_k.gguf q4_k
```

## Credits

- Model: [Mistral AI](https://mistral.ai/) — Apache 2.0
- GGUF conversion: [CrispASR](https://github.com/CrispStrobe/CrispASR)
- Port cross-referenced against [voxtral.c](https://github.com/antirez/voxtral.c), [voxmlx](https://github.com/awni/voxmlx), [voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs)

## Provenance and EU AI Act Art. 53 note

- **Upstream model:** [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) — published by `mistralai`.
- **Upstream licence:** `apache-2.0`. This repository redistributes under the same terms; it grants no rights the upstream licence does not.
- **What was done here:** format conversion and/or quantisation only (GGUF). No training, no fine-tuning, no merging, no distillation, no change to architecture, vocabulary or capability. Only the numeric representation of the upstream weights differs.
- **Training data:** documented — where it is documented at all — by the upstream provider; see the upstream model card. No training data was used, added or selected by this repository. No training-content summary was found on the upstream model card at the time of writing; that documentation gap is upstream's and is not filled here.
- **Provider status:** under Regulation (EU) 2024/1689 the upstream authors remain the provider of this model. Converting the serialisation format does not make this repository the provider of a new general-purpose AI model, and no such claim is made. Questions about training content, copyright policy or model capability belong upstream.
