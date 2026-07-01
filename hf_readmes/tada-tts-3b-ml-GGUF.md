---
license: llama3.2
language:
- en
- es
- ja
- zh
- de
- fr
- it
- pt
- ko
- ar
base_model:
- HumeAI/tada-3b-ml
- meta-llama/Llama-3.2-3B
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- tada
- llama
- flow-matching
- voice-cloning
- gguf
- crispasr
library_name: ggml
---

# TADA-3B-ML — GGUF (ggml-quantised)

GGUF / ggml conversion of [`HumeAI/tada-3b-ml`](https://huggingface.co/HumeAI/tada-3b-ml) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

TADA-3B-ML is a multilingual text-to-speech model built on Meta Llama 3.2 3B with a flow-matching (FM) speech decoder and custom Hume codec. Key property: **1:1 token alignment** — every text token maps to exactly one speech vector, eliminating transcript hallucination. Supports 10 languages (en, es, ja, zh, de, fr, it, pt, ko, ar). 24 kHz mono output.

**License:** Apache-2.0 / Llama 3.2 Community License ("Built with Llama").

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `tada-tts-3b-ml-f16.gguf` | F16 | ~8.2 GB | Reference quality (LLM + FM head) |
| `tada-tts-3b-ml-q4_k.gguf` | Q4_K | ~6.2 GB | **Recommended** — good quality |
| `tada-tts-3b-ml-q8_0.gguf` | Q8_0 | ~5.6 GB | Near-lossless |
| `tada-codec-f16.gguf` | F16 | ~1.0 GB | Codec decoder (required companion) |
| `tada-ref.gguf` | F32 | ~466 KB | Default voice reference (JFK prompt, ~5 s) |
| `tada-encoder-f16.gguf` | F16 | ~187 MB | Reference encoder for `--make-ref` voice cloning |
| `tada-aligner-<lang>.gguf` | Q8_0 | ~906 MB | CTC aligner for `--make-ref` / `--align` (ar ch de en es fr it ja pl pt) |

The Q4_K file uses a TADA-aware quantization policy (tail=14): large transformer block matrices are quantized, while the last 14 token-embedding rows and all `tada.*` flow-matching tensors are kept at F16. This preserves the timing and acoustic conditioning paths where quantization noise matters most.

## Architecture

```
Text Input
  |
BPE Tokenize (Llama-3.2 128K vocab)
  |
Llama-3.2-3B AR Forward (28L, 3072d, 24 heads / 8 KV)
  + acoustic embedding (512d) + gray-code time embedding
  |
Flow-Matching Speech Head (6L AdaLN + SwiGLU, 10 Euler steps)
  |-- noise → speech vector (528d)
  |
TADA Codec Decoder (DAC upsampler)
  |-- speech vectors → 24 kHz PCM
  |
Output: float32 mono @ 24 kHz
```

## Quick start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target crispasr

# 2. Pull model + codec + default voice reference
huggingface-cli download cstr/tada-tts-3b-ml-GGUF \
  tada-tts-3b-ml-q4_k.gguf tada-codec-f16.gguf tada-ref.gguf \
  --local-dir .

# 3. Synthesize (with default voice reference)
./build/bin/crispasr --backend tada \
  -m tada-tts-3b-ml-q4_k.gguf \
  --codec-model tada-codec-f16.gguf \
  --voice tada-ref.gguf \
  --tts "Hello, this is a test of the TADA speech synthesis system." \
  --tts-output hello.wav \
  --seed 42
```

For near-lossless quality use `tada-tts-3b-ml-q8_0.gguf`; for the F16 reference quality use `tada-tts-3b-ml-f16.gguf`.

Recent CrispASR builds can also resolve this repo through the model registry:

```bash
./build/bin/crispasr --backend tada -m auto --auto-download \
  --tts "Hola, esto es una prueba del sistema TADA." \
  --tts-output hello.wav
```

### Voice cloning from your own audio

Build a reference voice from any `.wav` with the built-in `--make-ref` pipeline
(no Python needed). It needs the `tada-encoder-*.gguf` + `tada-aligner-*.gguf`
from this repo — add `--auto-download` and they are fetched automatically:

```bash
# 1. Build a reference GGUF from a voice sample + its EXACT transcript
./build/bin/crispasr --backend tada-3b-ml -m tada-tts-3b-ml-f16.gguf --auto-download \
  --make-ref --voice your-voice.wav \
  --ref-text "Exact words spoken in your-voice.wav." \
  --make-ref-output my-voice.gguf

# 2. Synthesize in that voice
./build/bin/crispasr --backend tada-3b-ml \
  -m tada-tts-3b-ml-q4_k.gguf --codec-model tada-codec-f16.gguf \
  --voice my-voice.gguf \
  --tts "Text to speak in the cloned voice." --tts-output output.wav
```

The `--ref-text` must match the audio (it drives the text↔audio alignment). For
non-English audio pass `--language <code>` to select `tada-aligner-<code>.gguf`.
Or pass any voice reference GGUF directly via `--voice /path/to/voice.gguf`.

The bundled `tada-ref.gguf` encodes a short JFK clip as the default voice.


### Forced-alignment word timestamps (`--align`)

The TADA aligner (a wav2vec2 CTC model over the Llama-3.2 BPE vocab) also does
forced alignment: given audio + its transcript it emits frame-accurate word
timings. Same assets as `--make-ref` (auto-downloaded):

```bash
crispasr --backend tada-3b-ml -m tada-tts-3b-ml-f16.gguf --auto-download \
  --align --voice speech.wav --ref-text "exact transcript" \
  --align-format srt   # srt (default) | json | plain
```

Multilingual: pass `--language <code>` to use `tada-aligner-<code>.gguf`
(ar ch de en es fr it ja pl pt). Note it is a *forced* aligner — it needs the transcript, it is not
a standalone recogniser.


## Source model

- **Upstream:** [`HumeAI/tada-3b-ml`](https://huggingface.co/HumeAI/tada-3b-ml) (BF16 safetensors)
- **Base model:** [`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B)
- **Codec:** [`HumeAI/tada-codec`](https://huggingface.co/HumeAI/tada-codec)
- **Paper:** [arXiv:2602.23068](https://arxiv.org/abs/2602.23068)
- **Converted with:** `models/convert-tada-to-gguf.py`, `models/convert-tada-codec-to-gguf.py`, and `crispasr-quantize`
- **Runtime:** [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR)

## Validation

The Q4_K model was validated with CrispASR by synthesizing "Please call Stella." and ASR-roundtripping the output with Whisper tiny.en:

```
Please call Stella.
```

## Runtime fixes (CrispASR ≥ commit 0a95b326)

This GGUF was produced alongside the following CrispASR runtime fixes; use a build from that commit or later:

- **Voice prompt AR continuation** — C++ now correctly inserts the n_prompt PAD token slots that Python uses for the voice replay phase. Without this fix, long voice references caused the AR generation loop to run 0 iterations, producing near-silence.
- **BF16 noise parity** — FM denoising noise is now rounded to BF16 before scaling, matching PyTorch's `torch.randn().to(bfloat16)` behaviour and eliminating subtle residual drift.
- **Vulkan contiguity** — `ggml_cont()` wrappers added around strided 2D views in the B2 FM graph; required for Vulkan element-wise kernels.
- **Mixed-precision Q4_K** — `crispasr-quantize` tail=14 keeps the last 14 token-embedding rows and all TADA tensors at F16, stabilising the timing path for the 3B model.

## Checksums

```text
0a738e55d88af1ed5c4106bf12bf6cf8e4f134c50a2963b75d9d94c26c10f0cf  tada-tts-3b-ml-f16.gguf
5606670f6787fbb186fc3371ac418d1e6b7fbfe879f4fc8b6c82294eb9a15efa  tada-tts-3b-ml-q4_k.gguf
23da2b8230c2e7b1f8753b80200d059c75e0752ef37626917078952e5363cf2d   tada-tts-3b-ml-q8_0.gguf
ef5652e7a346c8a55dd6692676da2827320fd141042e87175880e032e1953082   tada-codec-f16.gguf
7efcc96795dd2b27577a4a81eb52d0c3add5ffa67f325fba5a938f3f98067ace  tada-ref.gguf
```
