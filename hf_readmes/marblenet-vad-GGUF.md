---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license
language:
- en
- de
- fr
- es
- ru
- zh
pipeline_tag: voice-activity-detection
tags:
- audio
- vad
- voice-activity-detection
- gguf
- marblenet
- nemo
- nvidia
library_name: ggml
base_model: nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0
---

# MarbleNet VAD -- GGUF

GGUF conversion of [`nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0`](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## Available variants

| File | Size | Notes |
|---|---:|---|
| `marblenet-vad.gguf` | 439 KB | F32, BatchNorm fused into conv weights |

No quantization needed — model is already 91.5K params (439 KB).

## Model details

- **Architecture:** MarbleNet — 1D time-channel separable CNN (6 Jasper blocks: depthwise conv + pointwise conv + BN + ReLU)
- **Parameters:** 91.5K (smallest VAD model in CrispASR)
- **Languages:** Chinese, English, French, German, Russian, Spanish
- **Input:** 80-bin mel spectrogram (16kHz, 512 FFT, 25ms window, 10ms stride)
- **Output:** per-frame speech probability (20ms per frame)
- **Training data:** 2,600h real + 1,000h synthetic + 330h noise
- **License:** [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license) (commercial use OK)

## Benchmark (ROC-AUC from NVIDIA)

| Dataset | ROC-AUC |
|---|---:|
| VoxConverse-test | 96.65 |
| VoxConverse-dev | 97.59 |
| AMI-test | 96.25 |
| Earnings21 | 97.11 |

## Usage with CrispASR

```bash
# Auto-download (439 KB)
crispasr --backend whisper -m auto --auto-download --vad -vm marblenet -f audio.wav

# Or with explicit path
crispasr --backend parakeet -m auto --auto-download --vad -vm marblenet-vad.gguf -f audio.wav
```

## CrispASR VAD comparison

| VAD Model | Size | Latency | Languages |
|---|---:|---:|---|
| Silero VAD v5 | 0.9 MB | ~60 ms | Multilingual |
| **MarbleNet** | **0.4 MB** | **~30 ms** | **6 languages** |
| FireRedVAD | 2.4 MB | ~50 ms | 100+ (recommended) |
| Whisper-VAD-ASMR | 22 MB | ~1000 ms | Experimental |

## Conversion

```bash
python models/convert-marblenet-vad-to-gguf.py \
  --input nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0 \
  --output marblenet-vad.gguf
```

BatchNorm layers are fused into convolution weights at convert time (36 tensors from 84 original).
