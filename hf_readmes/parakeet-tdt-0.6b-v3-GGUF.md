---
license: cc-by-4.0
language:
- bg
- cs
- da
- de
- el
- en
- es
- et
- fi
- fr
- hr
- hu
- it
- lt
- lv
- mt
- nl
- pl
- pt
- ro
- ru
- sk
- sl
- sv
- uk
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- ggml
- gguf
- parakeet
- tdt
- fastconformer
- multilingual
library_name: ggml
base_model: nvidia/parakeet-tdt-0.6b-v3
---

# Parakeet TDT 0.6B v3 — GGUF (ggml-quantised)

GGUF / ggml conversions of [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for use with the `parakeet-main` CLI from **[CrispStrobe/CrispASR@parakeet](https://github.com/CrispStrobe/CrispASR/tree/parakeet)**.

Parakeet TDT 0.6B v3 is NVIDIA's 600 M-parameter multilingual ASR model:

- **25 European languages** with automatic language detection (no prompt prefix needed)
- **Built-in word-level timestamps** from the TDT (Token-and-Duration Transducer) decoder — no separate CTC alignment model required
- **6.34 % avg WER** on the HuggingFace Open ASR Leaderboard
- **CC-BY-4.0** licence (friendlier than most ASR models)

This repo provides four quantisations, all converted from the same `.nemo` checkpoint via the streaming `convert-parakeet-to-gguf.py` script and quantised with `crispasr-quantize`.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `parakeet-tdt-0.6b-v3.gguf`        | 1.26 GB | F16, full precision |
| `parakeet-tdt-0.6b-v3-q8_0.gguf`   | 711 MB  | Q8_0, near-lossless |
| `parakeet-tdt-0.6b-v3-q5_0.gguf`   | 516 MB  | Q5_0 |
| `parakeet-tdt-0.6b-v3-q4_k.gguf`   | 467 MB  | **Q4_K — recommended default** |

All quantisations produce identical text on `samples/jfk.wav`:
> And so my fellow Americans. Ask not what your country can do for you. Ask what you can do for your country.

## Quick Start

```bash
# 1. Build the runtime
git clone -b parakeet https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target parakeet-main

# 2. Download a quantisation
huggingface-cli download cstr/parakeet-tdt-0.6b-v3-GGUF \
    parakeet-tdt-0.6b-v3-q4_k.gguf --local-dir .

# 3. Transcribe
./build/bin/parakeet-main \
    -m parakeet-tdt-0.6b-v3-q4_k.gguf \
    -f your-audio.wav -t 8
```

## Word-level timestamps for free

Pass `-v` to dump per-token timestamps from the TDT duration head. Each token spans one or more encoder frames; one frame = **80 ms**. No separate alignment model is required.

```
$ ./build/bin/parakeet-main -m parakeet-tdt-0.6b-v3-q4_k.gguf -f samples/jfk.wav -t 8 -v
  [    0.32s →     0.64s]  ' And'
  [    0.64s →     0.88s]  ' so'
  [    1.04s →     1.28s]  ' my'
  [    1.28s →     1.76s]  ' fellow'        ← f + ell + ow grouped
  [    1.76s →     2.56s]  ' Americans'
  [    2.96s →     3.28s]  '.'
  [    3.28s →     3.84s]  ' Ask'
  [    4.08s →     4.40s]  ' not'
  [    5.28s →     5.92s]  ' what your'
  ...
```

This is roughly **10× tighter** than the cross-attention DTW path used for Cohere Transcribe word timestamps (~360 ms MAE), and comparable to running a separate wav2vec2 + CTC forced alignment model — but at zero extra cost.

## Model architecture

| Component | Details |
| --- | --- |
| Encoder       | 24-layer FastConformer, d=1024, 8 heads, head_dim=128, FFN=4096, conv kernel=9 |
| Subsampling   | Conv2d dw_striding stack, 8× temporal (50 → 12.5 fps) |
| Predictor     | 2-layer LSTM, hidden 640, embed 8193 × 640 |
| Joint head    | enc(1024 → 640) + pred(640 → 640) → ReLU → linear(640 → 8198) |
| Vocab         | 8192 SentencePiece tokens (multilingual) |
| Audio         | 16 kHz mono, 128 mel bins, n_fft=512, hop=160, win=400 |
| Parameters    | ~600 M |

The mel filterbank and Hann window are baked directly into the GGUF (`preprocessor.fb` and `preprocessor.window` from the original `.nemo` checkpoint), so there is no recomputation at runtime. BatchNorm in the convolution module is folded into the depthwise conv weights at load time.

## How this was made

1. The `.nemo` checkpoint was unpacked, NeMo state-dict keys were remapped to ggml-friendly names, and weights were written to GGUF F16 (matmul tensors) + F32 (norms / biases / mel filterbank). A synthetic zero `conv.dw.bias` is added per encoder layer so the runtime BN-fold pass has somewhere to write the absorbed bias shift.
2. Quantised variants are produced by `crispasr-quantize` (the same llama.cpp-style quantiser used for the other GGUF releases).
3. Inference is implemented in `src/parakeet.{h,cpp}`: the FastConformer encoder runs as a single ggml graph (BN folded out), the LSTM predictor and joint head run as manual F32 CPU loops, and the TDT greedy decode loop alternates "advance encoder frame" / "emit token + advance predictor" using the duration head's argmax.

## Supported languages

`bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk`

The model auto-detects the language at inference time. No prompt prefix or `-l` flag is needed.

## Attribution

- **Original model:** [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC-BY-4.0). NVIDIA NeMo team.
- **GGUF conversion + ggml runtime:** [`CrispStrobe/CrispASR@parakeet`](https://github.com/CrispStrobe/CrispASR/tree/parakeet) — community contribution. Encoder graph borrows the dw_striding subsampling + Conformer block patterns from the same fork's `cohere.cpp`.
- **Reference inference:** [`istupakov/onnx-asr`](https://github.com/istupakov/onnx-asr) was the cross-check for the joint head + TDT greedy loop.

## Related

- C++ runtime: **[CrispStrobe/CrispASR@parakeet](https://github.com/CrispStrobe/CrispASR/tree/parakeet)**
- Sister repo (Cohere Transcribe): [`cstr/cohere-transcribe-03-2026-GGUF`](https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF)

## License

CC-BY-4.0, inherited from the base model. Use of these GGUF files must comply with the CC-BY-4.0 license including attribution.
