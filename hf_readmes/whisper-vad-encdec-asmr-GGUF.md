---
license: mit
language:
- en
- ja
- zh
pipeline_tag: voice-activity-detection
tags:
- audio
- vad
- voice-activity-detection
- gguf
- whisper
- transformer
- encoder-decoder
library_name: ggml
base_model: TransWithAI/Whisper-Vad-EncDec-ASMR-onnx
---

# Whisper-VAD-EncDec-ASMR -- GGUF

GGUF conversion of [`TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`](https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Type | Size | Notes |
|---|---|---:|---|
| `whisper-vad-asmr.gguf` | F32 | 114 MB | Full precision |
| `whisper-vad-asmr-q4_k.gguf` | Q4_K | 22 MB | 4-bit quantized, recommended |

## Model details

- **Architecture:** Whisper-base encoder (6L, 512d, 8 heads) + 2-layer TransformerDecoder (self-attention + cross-attention) + frame classifier (Linear 512->1 + sigmoid)
- **Parameters:** 29.8M (encoder: 25.4M, decoder: 4.4M)
- **Input:** 80-bin log-mel spectrogram (30s chunks, 16kHz, Whisper-style)
- **Output:** 1500 per-frame speech probabilities (20ms per frame)
- **Training data:** ~500 hours of Japanese ASMR audio
- **License:** MIT (see [original repo](https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx))

## Benchmark (CrispASR, CPU, 4 threads)

Tested on 10 diverse audio files (English, German, 1s-89s, clean/noisy):

| VAD Model | Size | Latency | Segmentation quality |
|---|---:|---:|---|
| Silero VAD v5 | 0.9 MB | 10-725 ms | Over-segments (10-55 segments per file) |
| **FireRedVAD** | **2.4 MB** | **~50 ms** | **Clean (1-2 slices)** |
| Whisper-VAD-ASMR (Q4_K) | 22 MB | ~1000 ms | Clean (1-2 slices) |

FireRedVAD remains recommended for production use (smallest, fastest, best F1). This model is useful as an alternative VAD option or for research purposes.

## Usage with CrispASR

```bash
# Use as VAD with any ASR backend
crispasr --backend whisper -m auto --auto-download \
  --vad -vm whisper-vad-asmr-q4_k.gguf \
  -f audio.wav

# Works with all CrispASR backends
crispasr --backend parakeet -m auto --auto-download \
  --vad -vm whisper-vad-asmr-q4_k.gguf \
  -f audio.wav
```

The model is auto-detected by filename pattern (`*whisper*vad*.gguf`) and dispatched through CrispASR's external VAD pipeline.

## Conversion

From the original ONNX model:

```bash
python models/convert-whisper-vad-onnx-to-gguf.py \
  --input TransWithAI/Whisper-Vad-EncDec-ASMR-onnx \
  --output whisper-vad-asmr.gguf

# Quantize
crispasr-quantize whisper-vad-asmr.gguf whisper-vad-asmr-q4_k.gguf q4_k
```

The converter uses ONNX graph topology tracing to correctly map anonymous tensor initializers to named weights (whisper-base encoder was fine-tuned, not frozen).

## Technical notes

- The whisper encoder weights are **fine-tuned** (not identical to `openai/whisper-base`)
- The decoder uses learned position queries (1500 x 512) as input, cross-attending to encoder output
- Frame classifier applies sigmoid to produce per-frame speech probabilities
- VAD segmentation uses hysteresis thresholding. The frame classifier's calibration sits lower than Silero / FireRed — on continuous speech the mean probability is around 0.25–0.30 even when activity is steady — so CrispASR auto-lowers the positive threshold to **0.30** (negative threshold 0.15 via the standard 0.15-step hysteresis) when `--vad-threshold` isn't set explicitly. Pass `-vt 0.5` to keep the legacy stricter behaviour, or `-vt 0.2` for noisier/softer material.
- The runtime casts conv weights to F16 (ggml im2col requirement) and quantized biases to F32 at graph build time
