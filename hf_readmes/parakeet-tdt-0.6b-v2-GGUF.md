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
- fastconformer
- english
library_name: ggml
base_model: nvidia/parakeet-tdt-0.6b-v2
---

# Parakeet TDT 0.6B v2 — GGUF (ggml-quantised)

GGUF / ggml conversions of [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) for use with the `crispasr` CLI from **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Parakeet TDT 0.6B v2 is NVIDIA's English-only 600 M-parameter ASR model — the original Open ASR Leaderboard topper before v3 spread capacity across 25 European languages. On plain English, v2 is often stronger than v3 since it didn't have to share encoder capacity with 24 other languages.

- **English-only**, mixed-case + punctuation output
- **Built-in word-level timestamps** from the TDT (Token-and-Duration Transducer) decoder — no separate CTC alignment model required
- **CC-BY-4.0** licence (friendlier than most ASR models)

This repo provides three quantisations, all converted from the same `.nemo` checkpoint via the `convert-parakeet-to-gguf.py` script and quantised with `crispasr-quantize`.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `parakeet-tdt-0.6b-v2.gguf`        | 1.24 GB | F16, full precision |
| `parakeet-tdt-0.6b-v2-q8_0.gguf`   | 735 MB  | Q8_0, near-lossless |
| `parakeet-tdt-0.6b-v2-q4_k.gguf`   | 468 MB  | **Q4_K — recommended default** |

All three precisions produce the same text on `samples/jfk.wav`:
> And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

## Quick Start

```bash
# 1. Build the runtime
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target crispasr-lib

# 2a. Auto-download via the registry key
./build/bin/crispasr -m parakeet-v2 --auto-download -f your-audio.wav

# 2b. Or explicit download + load
hf download cstr/parakeet-tdt-0.6b-v2-GGUF \
    parakeet-tdt-0.6b-v2-q4_k.gguf --local-dir .
./build/bin/crispasr -m parakeet-tdt-0.6b-v2-q4_k.gguf -f your-audio.wav
```

## When to pick v2 over v3

| Scenario | Pick |
| --- | --- |
| English only, want best WER | **v2** (this repo) |
| Multilingual, 25 EU languages | v3 — [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF) |
| Tight RAM, English | smaller hybrid — [`cstr/parakeet-tdt_ctc-110m-GGUF`](https://huggingface.co/cstr/parakeet-tdt_ctc-110m-GGUF) |
| Long-tail English vocab, willing to pay 2x compute | larger — [`cstr/parakeet-tdt-1.1b-GGUF`](https://huggingface.co/cstr/parakeet-tdt-1.1b-GGUF) |

## Model architecture

| Component | Details |
| --- | --- |
| Encoder       | 24-layer FastConformer, d=1024, 8 heads, head_dim=128, FFN=4096, conv kernel=9 |
| Subsampling   | Conv2d dw_striding stack, 8× temporal (50 → 12.5 fps) |
| Predictor     | 2-layer LSTM, hidden 640 |
| Joint head    | enc(1024 → 640) + pred(640 → 640) → ReLU → linear(640 → 1029) |
| Vocab         | 1024 SentencePiece tokens (English, mixed case + punctuation) |
| Audio         | 16 kHz mono, 128 mel bins, n_fft=512, hop=160, win=400 |
| Parameters    | ~600 M |

Same FastConformer encoder + TDT decoder as v3 — just trained on English-only data with an English-only BPE.

## How this was made

1. The `.nemo` checkpoint was unpacked, NeMo state-dict keys were remapped to ggml-friendly names, and weights were written to GGUF F16 (matmul tensors) + F32 (norms / biases / mel filterbank).
2. Quantised variants are produced by `crispasr-quantize` (the same llama.cpp-style quantiser used for the other GGUF releases).
3. Inference uses `src/parakeet.{h,cpp}`: FastConformer encoder runs as a single ggml graph (BN folded out), LSTM predictor + joint head run as CPU F32 loops, TDT greedy decode alternates "advance encoder frame" / "emit token + advance predictor" using the duration head's argmax.

## Attribution

- **Original model:** [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (CC-BY-4.0). NVIDIA NeMo team.
- **GGUF conversion + ggml runtime:** [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR).

## License

CC-BY-4.0, inherited from the base model. Use of these GGUF files must comply with the CC-BY-4.0 license including attribution.
