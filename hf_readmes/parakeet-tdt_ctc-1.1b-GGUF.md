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
- english
library_name: ggml
base_model: nvidia/parakeet-tdt_ctc-1.1b
---

# Parakeet TDT+CTC 1.1B — GGUF (ggml-quantised)

GGUF / ggml conversions of [`nvidia/parakeet-tdt_ctc-1.1b`](https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b) for use with the `crispasr` CLI from **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

The **largest hybrid** Parakeet — 1.1 B parameters, 42-layer FastConformer encoder with **both** TDT and CTC heads. The hybrid head gives you two decode strategies on the same encoder: native TDT word timestamps (default), or CTC if you need shallow-fusion biasing.

- **English**, mixed-case + punctuation output (vocab includes uppercase + punctuation tokens, unlike the pure `parakeet-tdt-1.1b`)
- **Hybrid TDT+CTC** — default decode is TDT; pass `--parakeet-decoder ctc` for the CTC head
- **CC-BY-4.0** licence

This repo provides three quantisations, all converted from the same `.nemo` checkpoint via the `convert-parakeet-to-gguf.py` script and quantised with `crispasr-quantize`.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `parakeet-tdt_ctc-1.1b.gguf`        | 2.15 GB | F16, full precision |
| `parakeet-tdt_ctc-1.1b-q8_0.gguf`   | 1.27 GB | Q8_0, near-lossless |
| `parakeet-tdt_ctc-1.1b-q4_k.gguf`   | 810 MB  | **Q4_K — recommended default** |

Smoke test on `samples/jfk.wav` (11 s clip, M1 Metal):

| Quant | Time | Realtime | Output |
| --- | ---: | ---: | --- |
| F16  | 0.74 s | 14.8× | "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country." |
| Q8_0 | 2.12 s | 5.2×  | (identical) |
| Q4_K | 2.67 s | 4.1×  | (identical) |

> Note: this checkpoint's Q4_K/Q8_0 run slower than the pure `parakeet-tdt-1.1b` quants on M1 (CTC + TDT both wired in, plus a per-tensor q4_0 fallback on the joint head). F16 is the fastest precision here.

## Quick Start

```bash
# 1. Build the runtime
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target crispasr-lib

# 2a. Auto-download via the registry key
./build/bin/crispasr -m parakeet-tdt_ctc-1.1b --auto-download -f your-audio.wav

# 2b. Or explicit download + load
hf download cstr/parakeet-tdt_ctc-1.1b-GGUF \
    parakeet-tdt_ctc-1.1b-q4_k.gguf --local-dir .
./build/bin/crispasr -m parakeet-tdt_ctc-1.1b-q4_k.gguf -f your-audio.wav

# 2c. Switch to the CTC head (e.g. when adding hotword biasing)
./build/bin/crispasr -m parakeet-tdt_ctc-1.1b --parakeet-decoder ctc -f your-audio.wav
```

## When to pick this over the other Parakeet variants

| Scenario | Pick |
| --- | --- |
| English 1.1B with proper casing + punctuation in output | **tdt_ctc-1.1b** (this repo) |
| English 1.1B, lowercase output, faster Q4_K/Q8_0 | [`cstr/parakeet-tdt-1.1b-GGUF`](https://huggingface.co/cstr/parakeet-tdt-1.1b-GGUF) |
| English, best WER per FLOP | [`cstr/parakeet-tdt-0.6b-v2-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v2-GGUF) |
| Multilingual (25 EU languages) | [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF) |
| Tight RAM | [`cstr/parakeet-tdt_ctc-110m-GGUF`](https://huggingface.co/cstr/parakeet-tdt_ctc-110m-GGUF) |

## Model architecture

| Component | Details |
| --- | --- |
| Encoder       | **42-layer** FastConformer, d=1024, 8 heads, head_dim=128, FFN=4096, conv kernel=9 |
| Subsampling   | Conv2d dw_striding stack, 8× temporal (100 → 12.5 fps) |
| Predictor     | 2-layer LSTM, hidden 640 |
| Joint head    | enc(1024 → 640) + pred(640 → 640) → ReLU → linear(640 → 1029) — TDT, 5 durations |
| CTC head      | linear(1024 → 1025) |
| Vocab         | 1024 SentencePiece tokens (English, mixed case + punctuation) + blank |
| Audio         | 16 kHz mono, **80 mel bins**, n_fft=512, hop=160, win=400 |
| Parameters    | ~1.1 B |

Same 42-layer encoder as `parakeet-tdt-1.1b`, but with an added CTC head and a mixed-case + punctuated vocab.

## Attribution

- **Original model:** [`nvidia/parakeet-tdt_ctc-1.1b`](https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b) (CC-BY-4.0). NVIDIA NeMo team.
- **GGUF conversion + ggml runtime:** [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR).

## License

CC-BY-4.0, inherited from the base model.
