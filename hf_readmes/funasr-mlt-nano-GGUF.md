---
license: other
license_name: funasr-model-license-v1.1
license_link: https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/blob/main/LICENSE
language:
- zh
- yue
- en
- ja
- ko
- vi
- th
- id
- ms
- tl
- ar
- hi
- bg
- ru
- de
- fr
- es
- it
- pt
- nl
- pl
- cs
- ro
- el
- fi
- sv
- tr
- fa
- da
- hu
- mk
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- ggml
- gguf
- funasr
- speech-llm
- sanm
- qwen3
- multilingual
library_name: ggml
base_model: FunAudioLLM/Fun-ASR-MLT-Nano-2512
---

# Fun-ASR-MLT-Nano-2512 — GGUF (ggml-quantised)

GGUF / ggml conversion of [`FunAudioLLM/Fun-ASR-MLT-Nano-2512`](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) for use with the `funasr` backend in **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**. Multilingual variant — same architecture as `funasr-nano-GGUF`, broader language coverage (~31 languages including Korean, Vietnamese, Indonesian, Thai, Malay, Filipino, Arabic, Hindi, Bulgarian, German, French, Spanish, Italian, Portuguese, Dutch, Polish, Czech, Romanian, Greek, Finnish, Swedish, Turkish, Persian, Danish, Hungarian, Macedonian, Russian).

The architecture is identical to Fun-ASR-Nano-2512:

- **70-block SenseVoiceSmall SANM encoder** (1 entry block @ 560→512 + 49 main blocks + 20 "tp" blocks, all 512-dim, 4 heads, FSMN k=11 depthwise convolution branch)
- **2-block Transformer audio adaptor** (512 → 2048 → 1024 prelude + 2× MHA blocks at 1024, FFN inner = 256)
- **Qwen3-0.6B LLM decoder** (28 layers, GQA 16/8, head_dim 128, RoPE θ=1e6, RMSNorm eps=1e-6) — the same body as Qwen3-ASR's decoder
- 1261 tensors total; only the LLM weights are tuned differently between Nano and MLT-Nano.

## Architecture note — no CTC path

Upstream `config.yaml` and `funasr/models/fun_asr_nano/model.py` declare a CTC decoder + head, but the published `model.pt` ships **only** `audio_encoder.* + audio_adaptor.* + llm.*` (zero `ctc_decoder.*` / `ctc.ctc_lo.*` keys). The LLM-decoder path is therefore the only viable inference path for these weights, and is what this GGUF and the CrispASR runtime implement.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `funasr-mlt-nano-2512-f16.gguf` | 1.98 GB | F16, full precision reference |
| `funasr-mlt-nano-2512-q8_0.gguf` | 1.27 GB | Q8_0, near-lossless |
| `funasr-mlt-nano-2512-q4_k.gguf` | 897 MB  | **Q4_K — recommended default** |

Mixed-case + punctuation output (mlt-nano's multilingual vocab keeps
both); no `--punc-model` post-processor needed.

## Quick Start

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build-ninja-compile -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-ninja-compile --target crispasr-lib

# Auto-download (recommended Q4_K)
./build-ninja-compile/bin/crispasr -m fun-asr-mlt-nano --auto-download -f samples/jfk.wav

# Or pin a specific file
hf download cstr/funasr-mlt-nano-GGUF funasr-mlt-nano-2512-q4_k.gguf --local-dir .
./build-ninja-compile/bin/crispasr -m funasr-mlt-nano-2512-q4_k.gguf -f samples/jfk.wav
```

## Licence + attribution

Upstream **FunAudioLLM/Fun-ASR-MLT-Nano-2512**:

- **Code** (the `funasr` Python package): Apache-2.0.
- **Model weights**: [**FunASR Model License v1.1**](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/blob/main/LICENSE) (Alibaba) — commercial use OK with attribution. Confirmed on the upstream-tracking discussion in [CrispStrobe/CrispASR#99](https://github.com/CrispStrobe/CrispASR/issues/99).

These GGUF files are a quantised / repackaged distribution of the upstream weights and inherit the FunASR Model License v1.1. Please attribute Alibaba / FunAudioLLM in downstream products.

> If you use this model, please also cite the upstream FunASR work.
> See the [upstream model card](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) for the canonical citation.
