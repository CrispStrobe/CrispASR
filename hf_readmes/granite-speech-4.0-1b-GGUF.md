---
language:
- en
- fr
- de
- es
- pt
- ja
license: apache-2.0
base_model: ibm-granite/granite-4.0-1b-speech
pipeline_tag: automatic-speech-recognition
tags:
- gguf
- speech-to-text
- granite
- conformer
- qformer
---

# Granite Speech 4.0-1B — GGUF

GGUF quantizations of [ibm-granite/granite-4.0-1b-speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech), a **1B-parameter speech-to-text model** combining a Conformer encoder, BLIP-2 Q-Former projector, and Granite LLM with μP (maximal update parameterization).

Converted and tested with [CrispASR](https://github.com/CrispStrobe/CrispASR), a multi-model ASR framework built on ggml.

## Files

| File | Quant | Size | Description |
|------|-------|------|-------------|
| `granite-speech-1b.gguf` | F16 (enc/proj F32) | 5.2 GB | Full precision (reference) |
| `granite-speech-1b-q8_0.gguf` | Q8_0 | ~3.4 GB | 8-bit quantized LLM |
| `granite-speech-1b-q4_k.gguf` | Q4_K | 2.8 GB | 4-bit K-quant LLM (recommended) |

Encoder and projector weights are kept at F32 for accuracy (they are precision-sensitive due to Shaw relative position embeddings and cross-attention). Only the 40-layer Granite LLM is quantized.

## Performance (CPU, 4 threads, AVX2, jfk.wav 11s)

| Quant | Size | Encoder | Prefill | Decode (ms/tok) | Total |
|-------|------|---------|---------|-----------------|-------|
| F16 | 5.2 GB | 14.1s | 16.6s | 200 | 37s |
| **Q4_K** | **2.8 GB** | **13.5s** | **5.3s** | **115** | **22.5s** |

Q4_K recommended — 1.9× smaller, 1.6× faster, identical transcription quality. The encoder dominates at 13.5s (60% of total) because it runs per-layer on CPU; future optimization will consolidate it into a single ggml graph.

## Usage

```bash
# Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target granite-main

# Download Q4_K (recommended)
huggingface-cli download cstr/granite-speech-4.0-1b-GGUF \
    granite-speech-1b-q4_k.gguf --local-dir .

# Transcribe
./build/bin/granite-main -m granite-speech-1b-q4_k.gguf -f audio.wav
# Output: and so my fellow americans ask not what your country can do for you...
```

### Auto-download

```bash
./build/bin/granite-main -m auto -f audio.wav
```

Downloads Q4_K automatically from this repo on first run.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | 16-layer Conformer (1024 dim, 8 heads, 128 head_dim, SiLU FFN 4096, depthwise conv k=15, batch norm, Shaw relative position embeddings, context_size=200) |
| **Projector** | 2-layer BLIP-2 Q-Former (3 learned query tokens per 15-frame window → 111 audio tokens for ~11s audio) |
| **LLM** | Granite 4.0-1B (40 layers, 2048 dim, GQA 16/4, SwiGLU FFN 4096, RoPE θ=10k, μP: emb×12, attn×1/128, residual×0.22, logits÷8) |
| **Mel** | 80 bins, HTK scale, win_length=400, n_fft=512, hop=160, per-utterance max normalization, frame stacking 2×80→160 |
| **Languages** | English, French, German, Spanish, Portuguese, Japanese |
| **Parameters** | ~1B total (encoder 440M + projector ~55M + LLM ~500M) |
| **License** | Apache-2.0 |

## Conversion

```bash
pip install gguf safetensors torch

python models/convert-granite-speech-to-gguf.py \
    --input /path/to/granite-4.0-1b-speech \
    --output granite-speech-1b.gguf

# Quantize (only LLM weights are quantized; encoder/projector stay F32)
./build/bin/cohere-quantize granite-speech-1b.gguf granite-speech-1b-q4_k.gguf q4_k
```

## Output

Tested on JFK inaugural address (jfk.wav, 11 seconds):

> and so my fellow americans ask not what your country can do for you ask what you can do for your country

Matches the HuggingFace Transformers reference output exactly.

## Credits

- **Model**: IBM Granite team ([ibm-granite/granite-4.0-1b-speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech))
- **Runtime**: [CrispASR](https://github.com/CrispStrobe/CrispASR) (MIT), built on [ggml](https://github.com/ggerganov/ggml)
