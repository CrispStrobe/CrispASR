---
license: mit
language:
- de
- en
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- ggml
- whisper
- german
- quantized
library_name: ggml
base_model: primeline/whisper-large-v3-turbo-german
---

# whisper-large-v3-turbo-german — GGUF

GGML conversions and quantisations of [`primeline/whisper-large-v3-turbo-german`](https://huggingface.co/primeline/whisper-large-v3-turbo-german) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)** or any crispasr-compatible tool.

## Available variants

| File | Quant | Size | Notes |
|---|---|---|---|
| `ggml-model.bin` | F16 | 1.6 GB | Original conversion, full precision |
| `ggml-model-q5_0.bin` | Q5_0 | 548 MB | Good quality/size tradeoff |
| `ggml-model-q4_k.bin` | Q4_K | 453 MB | Smallest, fastest on CPU |

All variants produce correct German transcription on test audio. Q4_K is recommended for CPU deployment.

## Model details

- **Architecture:** Whisper large-v3 encoder (32 layers) + turbo decoder (4 layers)
- **Parameters:** 809M
- **Languages:** German (primary), English
- **Base model:** [`primeline/whisper-large-v3-turbo-german`](https://huggingface.co/primeline/whisper-large-v3-turbo-german)
- **License:** MIT

The "turbo" variant uses only 4 decoder layers (vs 32 in large-v3), making it ~3x faster at inference with minimal quality loss for German.

## Usage with CrispASR

```bash
# Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -S . -B build && cmake --build build -j8

# Transcribe German audio
./build/bin/crispasr -m ggml-model-q4_k.bin -f german_audio.wav -l de

# With subtitles
./build/bin/crispasr -m ggml-model-q4_k.bin -f german_audio.wav -l de -osrt --split-on-punct
```

## Conversion

Converted from the original HuggingFace model using crispasr's `convert-h5-to-ggml.py`, then quantised with `crispasr-legacy-quantize`:

```bash
python models/convert-h5-to-ggml.py primeline/whisper-large-v3-turbo-german . models
crispasr-legacy-quantize ggml-model.bin ggml-model-q5_0.bin q5_0
crispasr-legacy-quantize ggml-model.bin ggml-model-q4_k.bin q4_k
```
