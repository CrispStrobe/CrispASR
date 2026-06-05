---
license: cc-by-4.0
language:
- de
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
- german
- medical
library_name: ggml
base_model: johannhartmann/parakeet_de_med
---

# Parakeet-DE-Med — GGUF (ggml-quantised)

GGUF / ggml conversions of [`johannhartmann/parakeet_de_med`](https://huggingface.co/johannhartmann/parakeet_de_med) for use with the `parakeet-main` CLI from **[CrispStrobe/CrispASR@parakeet](https://github.com/CrispStrobe/CrispASR/tree/parakeet)**.

`parakeet_de_med` is Johann Hartmann's PEFT decoder+joint fine-tune of [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) specialised for **German medical documentation** (Arztbriefe). On the German medical test set it scores **3.28% WER** vs the base model's 11.73% — a 72% relative reduction.

The fine-tune freezes the encoder and trains only the TDT decoder + joint head (18.1M out of 627M parameters, 2.89%). This means:
- The architecture is identical to `parakeet-tdt-0.6b-v3` (24-layer FastConformer encoder, 2-layer LSTM predictor, 8198-class TDT joint head)
- The same GGUF converter, runtime, and CLI work as-is
- The frozen encoder still uses the base model's auto-language detection — for clean German speech this works well, for accented or noisy audio you may want to fall back to a different runtime (see [comparison table](#which-runtime-should-i-use))

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `parakeet_de_med.gguf`        | 1.26 GB | F16, full precision |
| `parakeet_de_med-q8_0.gguf`   | 711 MB  | Q8_0, near-lossless |
| `parakeet_de_med-q5_0.gguf`   | 516 MB  | Q5_0 |
| `parakeet_de_med-q4_k.gguf`   | 467 MB  | **Q4_K — recommended default** |

All quantisations produce the same text on the German verification clip:
> Leider zu spät. Leider zu spät.

## Quick start

```bash
# 1. Build the runtime
git clone -b parakeet https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target parakeet-main

# 2. Download a quantisation
huggingface-cli download cstr/parakeet_de_med-GGUF \
    parakeet_de_med-q4_k.gguf --local-dir .

# 3. Transcribe German audio
./build/bin/parakeet-main \
    -m parakeet_de_med-q4_k.gguf \
    -f german_audio.wav -t 8
```

The runtime is the same `parakeet-main` binary used for the base parakeet-tdt-0.6b-v3. All the usual flags work: `-vad-model` for Silero VAD slicing, `-ck N` for fixed chunking, `-ml N` for max chars per line, `-osrt`/`-ovtt`/`-ot` for subtitle output, `-v` for per-token timestamps via the TDT duration head.

## Word-level timestamps

Like the base parakeet model, this fine-tune emits TDT durations as part of decoding, so word-level timestamps come for free at one encoder frame = **80 ms** granularity. No separate forced alignment model needed:

```
$ ./build/bin/parakeet-main -m parakeet_de_med-q4_k.gguf -f german.wav -t 8 -v
[ 0.32s →  0.64s]  Der
[ 0.64s →  1.04s]  Patient
[ 1.04s →  1.32s]  klagt
[ 1.32s →  1.92s]  über
...
```

## Which runtime should I use?

For German speech specifically:

| Use case | Right tool |
| --- | --- |
| **German medical documentation** | **`parakeet_de_med-q4_k.gguf`** ← this repo |
| General German ASR with explicit language control | `canary-1b-v2-q4_k.gguf` (`-sl de -tl de`) |
| German → English translation | `canary-1b-v2-q4_k.gguf` (`-sl de -tl en`) |
| General multilingual ASR (auto-detect) | `parakeet-tdt-0.6b-v3-q4_k.gguf` |
| Lowest English WER | `cohere-transcribe-q4_k.gguf` |

## Architecture (inherited from base)

| Component | Details |
| --- | --- |
| Encoder       | 24-layer FastConformer (frozen), d=1024, 8 heads, head_dim=128, FFN=4096, conv kernel=9 |
| Subsampling   | Conv2d dw_striding stack, 8× temporal (100 → 12.5 fps) |
| Predictor     | 2-layer LSTM, hidden 640, embed 8193 × 640 (**fine-tuned**) |
| Joint head    | enc(1024 → 640) + pred(640 → 640) → ReLU → linear(640 → 8198) (**fine-tuned**) |
| Vocab         | 8192 SentencePiece tokens (multilingual, but generation biased toward German medical) |
| Audio         | 16 kHz mono, 128 mel bins, n_fft=512, hop=160, win=400 |
| Parameters    | 627M total, 18.1M trained (2.89%) |

## Attribution

- **Base model:** [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC-BY-4.0). NVIDIA NeMo team.
- **Fine-tune:** [`johannhartmann/parakeet_de_med`](https://huggingface.co/johannhartmann/parakeet_de_med) (CC-BY-4.0). Johann Hartmann. Trained on 976 German medical documentation samples for 5 epochs with PEFT decoder+joint strategy.
- **GGUF conversion + ggml runtime:** [`CrispStrobe/CrispASR@parakeet`](https://github.com/CrispStrobe/CrispASR/tree/parakeet).

## Related

- C++ runtime: **[CrispStrobe/CrispASR@parakeet](https://github.com/CrispStrobe/CrispASR/tree/parakeet)**
- Base multilingual model (auto-detect): [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF)
- Encoder–decoder companion (canary, with explicit language control + speech translation): [`cstr/canary-1b-v2-GGUF`](https://huggingface.co/cstr/canary-1b-v2-GGUF)
- Cohere Transcribe (lowest English WER): [`cstr/cohere-transcribe-03-2026-GGUF`](https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF)

## License

CC-BY-4.0, inherited from both the base model and the fine-tune. Use of these GGUF files must comply with the CC-BY-4.0 license including attribution to NVIDIA NeMo team and Johann Hartmann.
