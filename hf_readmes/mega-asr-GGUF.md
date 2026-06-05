---
license: apache-2.0
language:
- ar
- cs
- da
- de
- el
- en
- es
- fa
- fi
- fil
- fr
- hi
- hu
- id
- it
- ja
- ko
- mk
- ms
- nl
- pl
- pt
- ro
- ru
- sv
- th
- tr
- vi
- yue
- zh
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- robust-asr
- ggml
- gguf
- qwen3
- mega-asr
- crispasr
library_name: ggml
base_model:
- Qwen/Qwen3-ASR-1.7B
- zhifeixie/Mega-ASR
---

# Mega-ASR 1.7B — GGUF

GGUF / ggml conversions of [`zhifeixie/Mega-ASR`](https://huggingface.co/zhifeixie/Mega-ASR) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Mega-ASR is a robust ASR adaptation of Qwen3-ASR-1.7B. The upstream release consists of:

- `Qwen3-ASR-1.7B` base model
- `mega-asr-merged/adapter_model.safetensors`, a LoRA adapter trained for degraded / in-the-wild audio
- `audio_quality_router/best_acc_model.safetensors`, a lightweight router that decides whether to enable the LoRA

This GGUF release uses the **always-on robust path**: the Mega-ASR LoRA is merged into the Qwen3-ASR-1.7B weights offline, then converted to a normal Qwen3-ASR GGUF. CrispASR runs it through the standard `qwen3` backend; no Python, PEFT, safetensors, or runtime LoRA switching is needed.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `mega-asr-1.7b-f16.gguf` | 4.4 GB | F16 merged-LoRA model |
| `mega-asr-1.7b-q4_k.gguf` | 1.3 GB | Q4_K, recommended default |
| `mega-asr-router.gguf` | 2.8 MB | Converted upstream audio-quality router weights; auxiliary artifact, not used by current CrispASR inference |

## Quick Start

Build CrispASR:

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target crispasr-lib
```

Download the recommended quant:

```bash
huggingface-cli download cstr/mega-asr-GGUF \
    mega-asr-1.7b-q4_k.gguf --local-dir .
```

Transcribe:

```bash
./build/bin/crispasr \
    --backend mega-asr \
    -m mega-asr-1.7b-q4_k.gguf \
    your-audio.wav
```

Older CrispASR builds can use the same file with the canonical backend name:

```bash
./build/bin/crispasr \
    --backend qwen3 \
    -m mega-asr-1.7b-q4_k.gguf \
    your-audio.wav
```

## Auto-Download

Current CrispASR `main` registers `mega-asr` as an alias for the Qwen3-ASR runtime:

```bash
./build/bin/crispasr --backend mega-asr -m auto your-audio.wav
```

This downloads `mega-asr-1.7b-q4_k.gguf` and runs it with the existing Qwen3-ASR backend.

## Validation

Smoke test on `samples/jfk.wav`, CPU-only VPS build:

```bash
./build/bin/crispasr \
    --backend mega-asr \
    -m mega-asr-1.7b-q4_k.gguf \
    --no-timestamps \
    samples/jfk.wav
```

Output:

```text
And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
```

The runtime reports the expected Mega-ASR / Qwen3-ASR-1.7B geometry:

```text
audio 24 layers, llm 28 layers, vocab 151936
```

## Router Status

The upstream Mega-ASR router chooses between the clean base model and the robust LoRA path. This GGUF release currently ships the practical always-on robust model.

The small `mega-asr-router.gguf` file is included so the router weights are available in GGUF form, but CrispASR does not yet use it to switch between base and LoRA models at runtime. Dynamic routing would require either loading both Qwen3-ASR-1.7B variants or adding runtime LoRA switching.

The router's frontend is a `LogMelSpectrogram` using 16 kHz audio, 80 Slaney mel bins, `n_fft=400`, `hop_length=160`, `win_length=400`, `log10(clamp(mel, 1e-10))`, then `(x + 4) / 4`. CrispASR already has the reusable pieces for this in `src/core/mel.h` (`build_slaney_fb` + `core_mel::compute()`), so if dynamic routing is added later it should reuse that shared mel path rather than adding another spectrogram implementation.

For degraded audio, the always-on merged model is the intended useful path. For consistently clean audio, use the regular Qwen3-ASR models.

## How this was made

1. Downloaded upstream [`zhifeixie/Mega-ASR`](https://huggingface.co/zhifeixie/Mega-ASR).
2. Merged `mega-asr-merged/adapter_model.safetensors` into `Qwen3-ASR-1.7B` offline.
3. Converted the merged Hugging Face checkpoint with CrispASR's existing Qwen3-ASR converter.
4. Quantised the F16 GGUF with CrispASR's `crispasr-quantize`.

## Upstream

- Model weights: [`zhifeixie/Mega-ASR`](https://huggingface.co/zhifeixie/Mega-ASR)
- Code: [`xzf-thu/Mega-ASR`](https://github.com/xzf-thu/Mega-ASR)
- Paper: [`Mega-ASR: Towards In-the-Wild^2 Speech Recognition via Scaling Up Real-world Acoustic Simulation`](https://huggingface.co/papers/2605.19833)
- Base model: [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

## License

Apache-2.0, inherited from the upstream Mega-ASR project and Qwen3-ASR base model.
