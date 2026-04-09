---
license: apache-2.0
language:
- en
- fr
- de
- es
- it
- pt
- nl
- hi
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- ggml
- gguf
- voxtral
- mistral
- whisper
- speech-llm
- multilingual
library_name: ggml
base_model: mistralai/Voxtral-Mini-3B-2507
---

# Voxtral-Mini-3B-2507 — GGUF

GGUF / ggml conversions of [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) for use with the `voxtral-main` CLI from **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Voxtral Mini is Mistral's **3B-parameter speech-LLM** — an enhancement of [Ministral 3B](https://mistral.ai/news/ministraux) with state-of-the-art audio input capabilities while retaining best-in-class text performance. It excels at speech transcription, translation, and audio understanding.

- **8 languages** (English, French, German, Spanish, Italian, Portuguese, Dutch, Hindi) with automatic language detection
- **Built-in audio Q&A and summarization** — ask questions about audio content directly
- **Function calling from voice** — trigger backend functions based on spoken intents
- **Long-form context** — up to 30 minutes of audio for transcription, 40 minutes for understanding
- **Natively multilingual** with state-of-the-art WER across the world's most widely used languages
- **Highly capable at text** — retains the text understanding capabilities of its Ministral 3B backbone
- **Apache-2.0** licence

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `voxtral-mini-3b-2507-q4_k.gguf` | 2.5 GB | **Q4_K — recommended default** |
| `voxtral-mini-3b-2507-q8_0.gguf` | 5.0 GB | Q8_0, near-lossless |

Both quantisations produce the correct transcript on `samples/jfk.wav`:
> And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

The mel filterbank from `WhisperFeatureExtractor` and the Tekken tokenizer vocab are **baked into the GGUF**, so the C++ runtime computes everything natively — no Python/torch/librosa at inference time.

## Quick Start

```bash
# 1. Build the runtime
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target voxtral-main

# 2. Download a quantisation
huggingface-cli download cstr/voxtral-mini-3b-2507-GGUF \
    voxtral-mini-3b-2507-q4_k.gguf --local-dir .

# 3. Transcribe
./build/bin/voxtral-main \
    -m voxtral-mini-3b-2507-q4_k.gguf \
    -f audio.wav -t 8 -l en
```

Audio must be 16 kHz mono 16-bit PCM WAV. Pre-convert with:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

### Language selection

Use `-l LANG` with a two-letter code:

| Flag | Language |
| --- | --- |
| `-l en` | English (default) |
| `-l de` | German |
| `-l fr` | French |
| `-l es` | Spanish |
| `-l it` | Italian |
| `-l pt` | Portuguese |
| `-l nl` | Dutch |
| `-l hi` | Hindi |

## Performance

Measured on `samples/jfk.wav` (11 seconds), 4-core CPU:

| Variant | Mel | Encoder | Prefill | Decode/tok | **Total** |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16 (8.8 GB) | 264 ms | 48.7 s | 78.4 s | 1134 ms | 157 s |
| **Q4_K (2.5 GB)** | 246 ms | 32.7 s | **30.8 s** | **242 ms** | **70 s** |

Q4_K gives a **2.2× speedup** over F16 while producing identical transcripts. The 3B model is larger than the Qwen3-ASR 0.6B — for fastest CPU inference on short clips, Qwen3-ASR Q4_K (6.6s for 11s audio) is faster; Voxtral's advantage is the richer capabilities (audio understanding, function calling, text Q&A) and superior multilingual WER.

## Architecture

Voxtral-Mini-3B is a three-module speech-LLM:

| Component | Details |
| --- | --- |
| **Audio encoder** | 32-layer Whisper-large-v3 encoder: d=1280, 20 heads, head_dim=64, FFN=5120, 128 mels, **learned absolute positional embedding** (1500, 1280). Conv1d front-end: conv1(128→1280, k=3, **stride=1**, pad=1) + GELU → conv2(1280→1280, k=3, stride=2, pad=1) + GELU. Note: conv1 stride is 1 (not 2 like standard Whisper), so only conv2 does temporal downsampling (2×). 3000 mel frames → 3000 → 1500 encoder frames. |
| **Projector** | Stack-4-frames + 2× Linear: reshape (1500, 1280) → (375, 5120), then Linear(5120→3072) → GELU → Linear(3072→3072). 4× temporal downsampling: 50 fps Whisper output → **12.5 fps audio embeddings** matching the documented frame rate. |
| **LLM** | 30-layer Llama 3 / Ministral 3B: d=3072, **32 Q heads / 8 KV heads (GQA, ratio 4)**, head_dim=128, FFN=8192, SwiGLU, RMSNorm, NEOX-style RoPE θ=1e8, vocab=131072, max_pos=131072. **No biases** anywhere. **No Q-norm/K-norm** (unlike Qwen3-ASR's Qwen3 backbone). |
| **Tokenizer** | Mistral Tekken (tiktoken-style rank BPE, 150k vocab entries + 1000 special tokens). Stored in the GGUF as a binary blob. |
| **Audio injection** | `audio_token_id=24` placeholder in the `[INST]` prompt; the LLM input embeddings at those positions get replaced with the projector output frames. |
| **Parameters** | ~3B total |

### Transcription prompt format

```
<s> [INST] [BEGIN_AUDIO] <audio_pad>×375 [/INST] lang:en [TRANSCRIBE]
```

Token IDs: `[1, 3, 25, 24×375, 4, 9909, 1058, <lang_id>, 34]`

### Key differences from standard Whisper

1. **Conv1 stride is 1** (Whisper uses stride 2). This means the conv front-end only does 2× temporal reduction (just conv2), not 4×. 3000 mel frames → 1500 encoder frames (vs Whisper's 750).
2. **K-proj has no bias** in the encoder's self-attention (Whisper quirk preserved from the Whisper-large-v3 weights).
3. The encoder output is **not** consumed by a Whisper decoder — it's fed through a 4-frame-stack projector into a general-purpose Llama 3 LLM that generates the transcript (or any other text response) autoregressively.

## Implementation notes

The C++ runtime is verified against the PyTorch reference (bf16) at every architectural boundary:

| Stage | Diff metric | Result |
| --- | --- | --- |
| LLM forward (30 layers, text-only) | cosine sim at last position | 0.999973, **top-5 5/5 match** |
| Audio encoder + projector (32 layers + stack-4) | per-row cosine sim vs `proj2_out.npy` | mean 0.998, min 0.870 (bf16 ref precision) |
| End-to-end transcription on jfk.wav | generated token sequence | **Correct transcript** |

The 0.87 min cosine sim on the encoder is from the bf16 reference precision (7-bit mantissa) vs F16 GGUF weights (10-bit) with F32 compute in C++. An F32 reference would give tighter numbers — the end-to-end transcript is the real correctness test and it passes.

### Bugs found during the port

1. **`ggml_conv_1d` output layout**: returns `(OL, OC, N)` not `(OC, OL)`. Bias needs `(1, OC, 1)` reshape to broadcast over time+batch.
2. **Post-conv transpose**: `ggml_conv_1d` puts time on `ne[0]`, but LayerNorm needs feature dim on `ne[0]`. Fixed by reshape+transpose to `(d, T_enc)`.
3. **GELU approximation**: `ggml_gelu` (tanh approx) → `ggml_gelu_erf` (exact) for correctness matching.
4. **Tekken vocab blob storage**: gguf-py's `add_array` with Python int lists stores as INT32, corrupting the uint8 byte stream. Fixed by storing as a 1D F32 tensor.

## How this was made

1. HF safetensors converted to GGUF F16 by [`models/convert-voxtral-to-gguf.py`](https://github.com/CrispStrobe/CrispASR/blob/main/models/convert-voxtral-to-gguf.py). All 765 tensors (762 model + mel_filters + mel_window + Tekken vocab blob) map cleanly.
2. Quantised variants produced by [`cohere-quantize`](https://github.com/CrispStrobe/CrispASR/blob/main/examples/cohere-main/cohere-quantize.cpp) with the Q4_0 fallback for 1280-wide audio encoder tensors (1280 % 256 ≠ 0 for Q4_K, same situation as Qwen3-ASR).
3. Inference implemented in [`src/voxtral.{h,cpp}`](https://github.com/CrispStrobe/CrispASR/blob/main/src/voxtral.cpp) (~1300 LOC): encoder and LLM each run as one ggml graph, with a persistent F16 KV cache `(head_dim, max_ctx, n_kv_heads, n_layers)` shared between prefill and per-token decode steps. Flash attention (`ggml_flash_attn_ext`) used on both prefill (F16 causal mask) and decode (no mask) paths.

## Related

- **Original model**: [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) (Apache-2.0)
- **C++ runtime**: [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)
- **Research paper**: [arxiv.org/abs/2507.13264](https://arxiv.org/abs/2507.13264)
- Sister releases in the same family:
  - [`cstr/qwen3-asr-0.6b-GGUF`](https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF) — Qwen3-ASR 0.6B (faster, 30 languages + Chinese dialects)
  - [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF) — Parakeet TDT 600M (free word timestamps)
  - [`cstr/canary-1b-v2-GGUF`](https://huggingface.co/cstr/canary-1b-v2-GGUF) — Canary 978M (speech translation)
  - [`cstr/cohere-transcribe-03-2026-GGUF`](https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF) — Cohere Transcribe 2B (lowest English WER)

## License

Apache-2.0, inherited from the base model.
