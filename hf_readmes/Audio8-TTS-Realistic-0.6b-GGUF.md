---
license: cc-by-nc-4.0
language:
- en
- zh
- de
- fr
- it
- es
- ja
- ko
- nl
- pl
- yue
base_model:
- Edge0/Audio8-TTS-Realistic-0.6b
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- arktts
- audio8
- gguf
- crispasr
library_name: ggml
---

# Audio8-TTS-Realistic-0.6b — GGUF (ggml-quantised)

GGUF / ggml quantised weights of [`Edge0/Audio8-TTS-Realistic-0.6b`](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b) for use with **[CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Audio8-TTS is a Dual-Autoregressive text-to-speech and zero-shot voice cloning model operating at 44.1 kHz. It couples a slow semantic transformer with a fast acoustic transformer and a causal neural vocoder (Snake activation + causal ConvNeXt blocks).

## Quantization Details

Standard GGUF quantization corrupts 1D convolutional layers with small kernel taps ($K \le 16$) and discrete acoustic centroids ($k=1024$). CrispASR applies precision-preserving carve-outs:
- **`Q4_K` (1.28 GB)**: Attention and FFN linear weights quantised to `Q4_K` / `Q6_K`; vocoder convs and discrete acoustic embeddings preserved at `FP16`.
- **`Q8_0` (1.49 GB)**: Attention and FFN weights quantised to `Q8_0`; achieves **1.000000** cosine parity across all stages against PyTorch.
- **`F16` (1.88 GB)**: Unquantised IEEE FP16 baseline.

## Quickstart with CrispASR

### CLI Inference (auto-downloads from registry)
```bash
crispasr --backend arktts -m auto --tts "The quick brown fox jumps over the lazy dog." --tts-output speech.wav
```

### Direct GGUF Model Path
```bash
crispasr --backend arktts -m arktts-q4_k.gguf --tts "CrispASR delivers high quality text to speech with low latency." --tts-output speech.wav
```

## Performance & Memory

- **Autoregressive Generation**: ~150–220 ms/frame (~21.5 frames/s = Real-Time Factor ~0.21x on 8 CPU threads).
- **Long Audio Scaling**: Chunked causal vocoder decoding bounds peak memory usage strictly under 2.5 GB even for minutes of continuous audio.
