# cohere-whisper.cpp

A [whisper.cpp](https://github.com/ggerganov/whisper.cpp)-style C++ inference engine for
**Cohere Transcribe** (CohereForAI/cohere-transcribe-v0.1) — a high-quality multilingual ASR model
with a 48-layer Conformer encoder and 8-layer causal Transformer decoder.

## Model Architecture

| Component | Spec |
|-----------|------|
| Encoder | 48-layer Conformer, d=1280, heads=8, head_dim=160, ffn=5120, conv_kernel=9 |
| Decoder | 8-layer causal Transformer, d=1024, heads=8, head_dim=128, ffn=4096, max_ctx=1024 |
| Vocab | 16,384 SentencePiece tokens |
| Audio | 16 kHz mono, 128 mel bins, n_fft=512, hop=160, win=400 |

## GGUF Weight File

Download from HuggingFace: [CrispStrobe/cohere-transcribe-gguf](https://huggingface.co/CrispStrobe/cohere-transcribe-gguf)

```
cohere-transcribe.gguf   (~2.5 GB)
```

Mixed precision: encoder/decoder weight matrices in **F16**, all biases/norms/BN params in **F32**.

## Build

Requires a C++17 compiler, CMake ≥ 3.14, OpenMP (for multi-threaded GEMM).

```bash
git clone https://github.com/CrispStrobe/cohere-whisper.cpp
cd cohere-whisper.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) cohere
# Link cohere-main (pthread workaround):
g++ -O3 -march=native -fopenmp \
    examples/cohere-main/CMakeFiles/cohere-main.dir/cohere-main.cpp.o \
    -o bin/cohere-main \
    src/libcohere.a ggml/src/libggml.a ggml/src/libggml-base.a ggml/src/libggml-cpu.a \
    -lpthread -lm -lstdc++
```

## Usage

```bash
./build/bin/cohere-main \
    -m cohere-transcribe.gguf \
    -f audio.wav \
    -l en \
    -t 8
```

**Options:**
- `-m MODEL.gguf` — path to GGUF weight file (required)
- `-f AUDIO.wav` — 16 kHz mono WAV (required; resampled internally if needed)
- `-l LANG` — language code, default `en`
- `-t THREADS` — CPU threads, default 4

**Output:** transcript printed to stdout, progress/timing to stderr.

## Export GGUF from Source Model

Requires the original `model.safetensors` from HuggingFace (cohere-transcribe-v0.1):

```bash
cd test_cohere   # directory containing export_gguf.py
pip install gguf sentencepiece safetensors

python export_gguf.py \
    --model-dir ./local_cohere_model \
    --output cohere-transcribe.gguf
```

Options:
- `--f32` — store ALL weight matrices as F32 (doubles file size, no quality benefit for encoder)
- `--f32-encoder` — store only encoder weight matrices as F32 (not needed; F16 is accurate)

## How It Works

1. **Audio preprocessing**: pre-emphasis (α=0.97) → center-pad (n_fft/2) → STFT →
   mel filterbank (128 bins) → log → per-feature normalization using **biased std** `sqrt(mean(diff²))`
   (matching the ONNX reference).

2. **Encoder**: 48-layer Conformer with Transformer-XL relative-position self-attention,
   Conv module (GLU → depthwise BN → SiLU → pointwise), Macaron FF (scale 0.5), output projection
   to dec_d=1024.

3. **Decoder**: 8-layer causal Transformer with pre-norm self-attention, cross-attention
   to encoder output, FFN with **ReLU** activation.

4. **Decoding**: greedy argmax. Prompt tokens
   `<|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>`
   prepended before generation.

## Known Limitations / Status

- **Speed**: CPU-only, naive O(n³) attention and DFT. ~5 min for 4s audio on 4 cores.
  See `optimize.md` for the full optimization roadmap.
- **Precision**: encoder in F16, cross-K diff vs ONNX: max=0.68, mean=0.076 (acceptable).
- **No batching**, no streaming, no beam search yet.
- CMake pthread link requires manual `g++` step (see Build above).

## Verified Output

For `sample2_16k.wav` ("The quick brown fox jumps over the lazy dog."):
```
Top token after prompt: 749 (▁The, logit=26.15)  ✓ matches ONNX
Transcript: The quick brown fox jumps over the lazy dog.
```
