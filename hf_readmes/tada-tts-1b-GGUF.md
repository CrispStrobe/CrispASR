---
license: llama3.2
language:
- en
base_model:
- HumeAI/tada-1b
- meta-llama/Llama-3.2-1B
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

# TADA-1B — GGUF (ggml-quantised)

GGUF / ggml conversion of [`HumeAI/tada-1b`](https://huggingface.co/HumeAI/tada-1b) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

TADA-1B is a text-to-speech model built on Meta Llama 3.2 1B with a flow-matching (FM) speech decoder and custom Hume codec. TADA uses **1:1 token alignment**: every text token maps to one speech vector before the codec decoder renders 24 kHz mono PCM. This repo packages the talker model, required codec decoder, and a ready-to-use reference voice prompt for CrispASR's `tada-1b` backend.

**License:** Llama 3.2 Community License. See the upstream [`HumeAI/tada-1b`](https://huggingface.co/HumeAI/tada-1b) model card for the original model terms.

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `tada-tts-1b-f16.gguf` | F16 | ~3.3 GB | Reference-quality talker |
| `tada-tts-1b-q4_k.gguf` | Q4_K | ~2.6 GB | **Recommended** — fits 6 GB RAM |
| `tada-codec-f16.gguf` | F16 | ~1.0 GB | Codec decoder, required companion |
| `tada-ref.gguf` | F32 | ~17 KB | Default voice reference (8-token JFK prompt) |
| `tada-encoder-f16.gguf` | F16 | ~187 MB | Reference encoder for `--make-ref` voice cloning |
| `tada-aligner-<lang>.gguf` | Q8_0 | ~906 MB | CTC aligner for `--make-ref` / `--align` (en) |

The Q4_K file uses a TADA-aware quantization policy (tail=8): large transformer block projection matrices are quantized, while the last 8 token-embedding rows and all `tada.*` flow-matching tensors are kept at F16. This preserves the timing and acoustic conditioning paths where quantization noise matters most.

## Architecture

```
Text Input
  |
BPE Tokenize (Llama-3.2 128K vocab)
  |
Llama-3.2-1B AR Forward (16L, 2048d, 32 heads / 8 KV)
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
huggingface-cli download cstr/tada-tts-1b-GGUF \
  tada-tts-1b-q4_k.gguf tada-codec-f16.gguf tada-ref.gguf \
  --local-dir .

# 3. Synthesize (with default voice reference)
./build/bin/crispasr --backend tada-1b \
  -m tada-tts-1b-q4_k.gguf \
  --codec-model tada-codec-f16.gguf \
  --voice tada-ref.gguf \
  --tts "Please call Stella." \
  --tts-output tada.wav \
  --seed 42
```

For F16 quality, replace `tada-tts-1b-q4_k.gguf` with `tada-tts-1b-f16.gguf`.

Recent CrispASR builds can also resolve this repo through the model registry:

```bash
./build/bin/crispasr --backend tada-1b -m auto --auto-download \
  --tts "Hello from TADA one billion." \
  --tts-output hello.wav
```

### Voice cloning from your own audio

Build a reference voice from any `.wav` with the built-in `--make-ref` pipeline
(no Python needed). It needs the `tada-encoder-*.gguf` + `tada-aligner-*.gguf`
from this repo — add `--auto-download` and they are fetched automatically:

```bash
# 1. Build a reference GGUF from a voice sample + its EXACT transcript
./build/bin/crispasr --backend tada-1b -m tada-tts-1b-f16.gguf --auto-download \
  --make-ref --voice your-voice.wav \
  --ref-text "Exact words spoken in your-voice.wav." \
  --make-ref-output my-voice.gguf

# 2. Synthesize in that voice
./build/bin/crispasr --backend tada-1b \
  -m tada-tts-1b-q4_k.gguf --codec-model tada-codec-f16.gguf \
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
crispasr --backend tada-1b -m tada-tts-1b-f16.gguf --auto-download \
  --align --voice speech.wav --ref-text "exact transcript" \
  --align-format srt   # srt (default) | json | plain
```

Multilingual: pass `--language <code>` to use `tada-aligner-<code>.gguf`
(en). Note it is a *forced* aligner — it needs the transcript, it is not
a standalone recogniser.


## Source model

- **Upstream:** [`HumeAI/tada-1b`](https://huggingface.co/HumeAI/tada-1b)
- **Base model:** [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B)
- **Codec:** [`HumeAI/tada-codec`](https://huggingface.co/HumeAI/tada-codec)
- **Paper:** [arXiv:2602.23068](https://arxiv.org/abs/2602.23068)
- **Converted with:** `models/convert-tada-to-gguf.py`, `models/convert-tada-codec-to-gguf.py`, and `crispasr-quantize`
- **Runtime:** [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR)

## Validation

The Q4_K model was validated with CrispASR by synthesizing "Please call Stella." and ASR-roundtripping the output with Whisper tiny.en:

```
Please call Stella!
```

## Runtime fixes (CrispASR ≥ commit 0a95b326)

This GGUF was produced alongside the following CrispASR runtime fixes; use a build from that commit or later:

- **Voice prompt AR continuation** — C++ now correctly inserts the n_prompt PAD token slots that Python uses for the voice replay phase. Without this fix, long voice references caused the AR generation loop to run 0 iterations, producing near-silence.
- **BF16 noise parity** — FM denoising noise is now rounded to BF16 before scaling, matching PyTorch's `torch.randn().to(bfloat16)` behaviour and eliminating subtle residual drift.
- **Vulkan contiguity** — `ggml_cont()` wrappers added around strided 2D views in the B2 FM graph; required for Vulkan element-wise kernels.
- **Mixed-precision Q4_K** — `crispasr-quantize` tail=8 keeps the last 8 token-embedding rows and all TADA tensors at F16, stabilising timing for the 1B model.

## Checksums

```text
7be26395d37412dff5fd2bbeb47b3f584c3172a4cd0ac3793208c82b107b28cf  tada-tts-1b-f16.gguf
0be99404ff8f959c30ab2e31cf2041cb1cd0df9c2079752384ec10e6ac16b862  tada-tts-1b-q4_k.gguf
ef5652e7a346c8a55dd6692676da2827320fd141042e87175880e032e1953082   tada-codec-f16.gguf
0f20e4076a8ac18ddd939299d751b9e9d57e46eeb87b77060d4f4096a4329835  tada-ref.gguf
```
