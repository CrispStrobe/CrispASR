---
license: apache-2.0
language:
- ar
- de
- el
- en
- es
- fr
- it
- ja
- ko
- nl
- pl
- pt
- vi
- zh
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- gguf
- conformer
- crispasr
base_model: CohereLabs/cohere-transcribe-03-2026
---

# cohere-transcribe-03-2026 — GGUF

GGUF weights for **[CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)** — Cohere's open-source 2B-parameter ASR model, #1 on the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) (avg WER 5.42, as of March 2026).

This conversion enables high-performance CPU inference via **[CrispASR](https://github.com/CrispStrobe/CrispASR/tree/ggml)** — a crispasr-style C++ runtime for the Cohere Conformer-encoder / Transformer-decoder architecture.

> **License**: Apache 2.0 (inherited from source model). See [original model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) for full terms.

---

## Files

| File | Size | Type | RTFx (8 threads) |
|------|------|------|------------------|
| `cohere-transcribe.gguf` | 3.85 GB | F16 | 0.80x |
| `cohere-transcribe-q8_0.gguf` | 2.05 GB | Q8_0 | 1.03x |
| `cohere-transcribe-q6_k.gguf` | 1.62 GB | Q6_K | 1.05x |
| `cohere-transcribe-q5_1.gguf` | 1.45 GB | Q5_1 | 1.06x |
| `cohere-transcribe-q5_0.gguf` | 1.38 GB | Q5_0 | 1.07x |
| `cohere-transcribe-q4_k.gguf` | 1.21 GB | Q4_K | 1.08x |

**RTFx** measured on `jfk.wav` (11s) using 8 CPU threads. Higher is faster. 1.0x means real-time.

---

## Quick Start

### 1. Build CrispASR

```bash
git clone -b ggml https://github.com/CrispStrobe/CrispASR
cd CrispASR && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) cohere-main
```

### 2. Download a GGUF

```bash
huggingface-cli download cstr/cohere-transcribe-03-2026-GGUF \
    cohere-transcribe-q4_k.gguf \
    --local-dir .
```

### 3. Transcribe

```bash
./bin/cohere-main \
    -m cohere-transcribe-q4_k.gguf \
    -f audio.wav \
    -l en \
    -t 8
```

---

## Implementation Notes (Critical for Correctness)

### Mel normalization
Per-feature normalization uses **biased standard deviation** `std = sqrt(mean(diff²) + ε)`, matching the ONNX reference. Using the Bessel-corrected (unbiased) formula produces a `sqrt(T) ≈ 20×` larger denominator for T ≈ 417 frames and completely corrupts the encoder output.

### Conformer Attention Scaling
The self-attention mechanism in the Conformer encoder **must** be scaled by `1/sqrt(head_dim)` before the softmax. Omitting this results in saturated attention scores and repetitive "garbage" output (e.g., "what what what...").

### Encoder preprocessing
1. **Pre-emphasis**: `y[n] = x[n] - 0.97·x[n-1]`
2. **Center-pad**: `n_fft/2 = 256` samples on each side
3. **STFT**: Hann window (length 400, zero-padded to 512), hop 160, rfft → power spectrum
4. **Mel Filterbank**: 128 bins → log → per-feature norm (biased std)

### Conv subsampling
5 convolutions with 3 stride-2 steps reducing T_mel → T_enc ≈ T_mel/8:
`conv0(ReLU) → conv2(DW) → conv3(PW,ReLU) → conv5(DW) → conv6(PW,ReLU) → linear(d=1280)`

### Cross-Attention Pre-computation
For high performance, cross-attention Key and Value tensors are pre-computed once per utterance from the encoder output. In this implementation, these projections are performed as part of the encoder's GGML compute graph to leverage backend acceleration.

### Decoder activation
Transformer decoder FFN uses **ReLU** (not SiLU/Swish).

---

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | 48-layer Conformer, d=1280, heads=8, head_dim=160, ffn=5120, conv_kernel=9 |
| **Decoder** | 8-layer causal Transformer, d=1024, heads=8, head_dim=128, ffn=4096, max_ctx=1024 |
| **Vocab** | 16,384 SentencePiece tokens |
| **Audio** | 16 kHz mono, 128 mel bins, n_fft=512, hop=160, win=400 |
| **Parameters** | ~2B |

---

## Related

- **Source model**: [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- **C++ runtime**: [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR) — also hosts ports of `parakeet-tdt-0.6b-v3`, `canary-1b-v2`, and a universal multilingual forced aligner (`nfa-align`)
- **Open ASR Leaderboard**: [hf-audio/open_asr_leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

### Sister GGUF releases in the same family

- [`cstr/cohere-transcribe-onnx-int4`](https://huggingface.co/cstr/cohere-transcribe-onnx-int4) — ONNX INT4 export of the same Cohere model
- [`cstr/cohere-transcribe-onnx-int8`](https://huggingface.co/cstr/cohere-transcribe-onnx-int8) — ONNX INT8 export of the same Cohere model
- [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF) — NVIDIA's 600M multilingual ASR with built-in word timestamps (faster, smaller, 25 EU languages)
- [`cstr/parakeet_de_med-GGUF`](https://huggingface.co/cstr/parakeet_de_med-GGUF) — German medical PEFT fine-tune of the parakeet base
- [`cstr/canary-1b-v2-GGUF`](https://huggingface.co/cstr/canary-1b-v2-GGUF) — NVIDIA's 978M multilingual ASR + speech translation with explicit `-sl/-tl` flags
- [`cstr/canary-ctc-aligner-GGUF`](https://huggingface.co/cstr/canary-ctc-aligner-GGUF) — universal multilingual subword forced aligner (25 EU languages, ~78 ms MAE)

### Use case → which runtime?

| Need | Right tool |
| --- | --- |
| **Lowest English WER** (Open ASR Leaderboard #1) | **`cohere-main`** ← this repo |
| Multilingual ASR + free word timestamps | `parakeet-main` ([cstr/parakeet-tdt-0.6b-v3-GGUF](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF)) |
| Multilingual ASR + speech translation + explicit language control | `canary-main` ([cstr/canary-1b-v2-GGUF](https://huggingface.co/cstr/canary-1b-v2-GGUF)) |
| Multilingual subword forced alignment of any transcript | `nfa-align` ([cstr/canary-ctc-aligner-GGUF](https://huggingface.co/cstr/canary-ctc-aligner-GGUF)) |
| English-only character-level forced alignment (~30 ms MAE) | `cohere-align` (uses wav2vec2-large-xlsr-53-english) |
