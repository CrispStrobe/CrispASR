---
license: cc-by-4.0
language:
- en
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- ggml
- gguf
- parakeet
- tdt
- ctc
- hybrid
- fastconformer
- small
- english
library_name: ggml
base_model: nvidia/parakeet-tdt_ctc-110m
---

# Parakeet TDT+CTC 110M — GGUF (ggml-quantised)

GGUF / ggml conversions of [`nvidia/parakeet-tdt_ctc-110m`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) for use with the `crispasr` CLI from **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

The **smallest** Parakeet variant: 110 M parameters, 17-layer FastConformer encoder with both TDT and CTC heads. Designed for low-RAM hosts and high-throughput batch transcription.

- **English-only**, mixed-case + punctuation output
- **Hybrid TDT+CTC** head — runtime defaults to CTC decode (the TDT path needs a 2-LSTM predictor which this model doesn't have; single-LSTM predictor + CTC head is a deliberate trade for size)
- **~45× realtime on M1 Metal** with Q4_K — fastest in the family
- **CC-BY-4.0** licence

This repo provides three quantisations, all converted from the same `.nemo` checkpoint via the `convert-parakeet-to-gguf.py` script and quantised with `crispasr-quantize`.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `parakeet-tdt_ctc-110m.gguf`        | 230 MB | F16, full precision |
| `parakeet-tdt_ctc-110m-q8_0.gguf`   | 139 MB | Q8_0, near-lossless |
| `parakeet-tdt_ctc-110m-q4_k.gguf`   |  91 MB | **Q4_K — recommended default** |

Smoke test on `samples/jfk.wav` (11 s clip, M1 Metal):

| Quant | Time | Realtime | Output |
| --- | ---: | ---: | --- |
| F16  | 0.24 s | 45.7× | "And so, my fellow Americans, askk not what your country can do for you. Ask what you can do for your country." |
| Q8_0 | 0.25 s | 44.6× | (identical) |
| Q4_K | 0.25 s | 43.8× | (identical) |

> Note: the model emits "askk" instead of "ask" on this clip — a small-model artifact, not a runtime bug. Larger Parakeet variants (v2 / tdt-1.1b / tdt_ctc-1.1b) get it right.

## Quick Start

```bash
# 1. Build the runtime
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target crispasr-lib

# 2a. Auto-download via the registry key
./build/bin/crispasr -m parakeet-tdt_ctc-110m --auto-download -f your-audio.wav

# 2b. Or explicit download + load
hf download cstr/parakeet-tdt_ctc-110m-GGUF \
    parakeet-tdt_ctc-110m-q4_k.gguf --local-dir .
./build/bin/crispasr -m parakeet-tdt_ctc-110m-q4_k.gguf -f your-audio.wav
```

The runtime detects `pred_layers=1 + has_ctc=True` at load time and automatically flips to CTC decode — no flag needed. See `parakeet_init_from_file` in `src/parakeet.cpp`.

## When to pick this over the other Parakeet variants

| Scenario | Pick |
| --- | --- |
| English, tightest RAM (mobile / edge / embedded) | **110m** (this repo) |
| English, best WER, ~600 M params | [`cstr/parakeet-tdt-0.6b-v2-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v2-GGUF) |
| Multilingual (25 EU languages) | [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF) |
| English, long-tail vocab | [`cstr/parakeet-tdt-1.1b-GGUF`](https://huggingface.co/cstr/parakeet-tdt-1.1b-GGUF) |

## Model architecture

| Component | Details |
| --- | --- |
| Encoder       | 17-layer FastConformer, d=512, 8 heads, head_dim=64, FFN=2048, conv kernel=9 |
| Subsampling   | Conv2d dw_striding stack, 8× temporal (100 → 12.5 fps) |
| Predictor     | **1-layer LSTM**, hidden 640 (smaller than the standard 2-LSTM) |
| Joint head    | enc(512 → 640) + pred(640 → 640) → ReLU → linear(640 → 1029) — TDT, 5 durations |
| CTC head      | linear(512 → 1025) — used by default at runtime |
| Vocab         | 1024 SentencePiece tokens (English) + blank |
| Audio         | 16 kHz mono, **80 mel bins**, n_fft=512, hop=160, win=400 |
| Parameters    | ~110 M |

## Why the runtime auto-flips to CTC

Hybrid TDT+CTC models normally let the user choose between the two decoders. But this checkpoint ships with `pred_layers=1` — a single LSTM rather than the usual two — which is too shallow for the TDT prediction network to produce useful joint scores. Upstream's recommended decode is CTC for this checkpoint. The C++ runtime detects this at load (`pred_layers < 2 && has_ctc`) and sets `decode_ctc=true` automatically. You can opt back into TDT (not recommended) by removing that auto-flip in `parakeet_init_from_file`.

## Attribution

- **Original model:** [`nvidia/parakeet-tdt_ctc-110m`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) (CC-BY-4.0). NVIDIA NeMo team.
- **GGUF conversion + ggml runtime:** [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR).

## License

CC-BY-4.0, inherited from the base model.
