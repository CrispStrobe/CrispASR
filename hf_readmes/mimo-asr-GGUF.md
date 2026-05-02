---
license: mit
language:
- zh
- en
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- gguf
- mimo
- qwen2
library_name: ggml
base_model: XiaomiMiMo/MiMo-V2.5-ASR
---

# MiMo-V2.5-ASR — GGUF

GGUF conversion of [`XiaomiMiMo/MiMo-V2.5-ASR`](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR) for **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**. Pure C++ inference — no Python, no Transformers, runs on Apple Silicon (Metal), CPU, and CUDA.

The runtime is functional end-to-end: greedy decode through the 36-layer Qwen2 LM, full prefill + step-decode KV-cached graphs, prompt construction matching the upstream `MimoAudio.asr_sft` reference exactly. JFK transcription test passes verbatim.

## Available variants

| File | Quant | Size | Layout | Recommended |
|---|---|---|---|---|
| `mimo-asr-f16.gguf` | F16 | 14.9 GB | separate Q/K/V | Full precision; needs ~16 GB RAM during inference |
| `mimo-asr-q4_k.gguf` | Q4_K | 4.2 GB | **fused QKV** | **Default** — fits in 8 GB RAM, no quality loss visible on JFK |

The default `mimo-asr-q4_k.gguf` (re-uploaded May 2026, PLAN #60d) ships with per-LM-layer Q/K/V projections fused into a single `model.layers.{i}.attn.qkv.{weight,bias}` tensor pair, yielding ~1.7× faster per-step decode on M1 vs the prior unfused layout (3058 ms/step → 1806 ms/step on a contended-disk run; ~1.1-1.2× pure-compute on a quiet box). The CrispASR runtime auto-detects either layout: the F16 file above keeps working unchanged via the separate-Q/K/V fallback path. Re-upload of a fused F16 is queued behind disk-headroom availability.

Pair with **[`cstr/mimo-tokenizer-GGUF`](https://huggingface.co/cstr/mimo-tokenizer-GGUF)** — the audio tokenizer is a separate model that converts 16 kHz PCM → 8-channel RVQ codes that this LM consumes.

## Architecture

- **Audio path** — 6-layer input_local_transformer (1024d, 64 heads, GS=4 group size, SiLU, sinusoidal RoPE on Q/K) + 8-channel RVQ codebook embeddings + linear group-downcast to 4096d
- **LM** — 36-layer Qwen2 (hidden=4096, 32 attn heads, 8 KV heads, intermediate=12288, RMSNorm, SwiGLU, RoPE θ=640K, max_pos=8192)
- **LM head** — untied, vocab=151680
- **Total params** — ~7.5B
- **Languages** — Mandarin (with Wu / Cantonese / Hokkien / Sichuanese dialect support), English, code-switching
- **License** — MIT (matches upstream)

## Usage with CrispASR

```bash
# Build (one-time)
git clone https://github.com/CrispStrobe/CrispASR.git
cd CrispASR
cmake -B build-ninja-compile -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-ninja-compile --target crispasr

# Download both halves
hf download cstr/mimo-asr-GGUF mimo-asr-q4_k.gguf --local-dir models/
hf download cstr/mimo-tokenizer-GGUF mimo-tokenizer-q4_k.gguf --local-dir models/

# Transcribe
build-ninja-compile/bin/crispasr \
  --backend mimo-asr \
  -m models/mimo-asr-q4_k.gguf \
  --codec-model models/mimo-tokenizer-q4_k.gguf \
  -f samples/jfk.wav
```

If `--codec-model` is omitted, the runtime auto-discovers `mimo-tokenizer-q4_k.gguf` (or `mimo-tokenizer.gguf`, `mimo-audio-tokenizer.gguf`) next to the LM file.

### Expected output (JFK sample)

```
And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
```

This matches the upstream Python `MimoAudio.asr_sft` reference verbatim.

### Performance

On Apple M1, Metal backend, Q4_K, warm-cache:

| Phase | Time |
|---|---|
| LM load (mmap, lazy) | ~1 s |
| Audio tokenize (11 s sample) | ~0.5 s |
| Prefill (T_groups=71) | ~3 s |
| Step decode (~25 tokens) | ~20 s with the fused-QKV file (≈0.8 s/token; was ~30 s pre-fusion) |
| **End-to-end** | **~25-30 s for 11 s audio (~0.4× realtime)** |

Per-step decode is the bottleneck; PLAN #60d (May 2026) fused the per-LM-layer Q/K/V projections into one matmul, replacing 3 mul_mat + 3 ggml_add per layer × 36 layers with 1 + 1, for a measured ~1.7× speedup at the same disk-pressure level. KV cache reuse via cached step graphs (PLAN #51b') is also live. Future perf wins: F16 with fused QKV (queued behind disk headroom), `CRISPASR_KV_QUANT=q8_0` for hour-long inputs (PLAN #60e env-flag is already plumbed; default stays F16 until per-backend rollout completes).

## Validation

Stage-by-stage cosine similarity against the bf16 PyTorch reference on JFK (Q4_K weights, bf16 ref):

| Stage | cos_mean | cos_min |
|---|---|---|
| `prefill_audio_features` | 0.998 | 0.992 |
| `prefill_text_embeds` | 0.996 | 0.901 |
| `prefill_inputs_embeds` | 0.998 | 0.901 |
| `prefill_last_hidden` | 0.963 | 0.963 |
| `prefill_text_logits_step0` | 0.981 | 0.981 |

Argmax of step-0 logits is token 1597 (`' And'`), matching the Python reference. The strict cos≥0.999 gate is tracked under F16+fp32 ref but requires >28 GB RAM; in practice the Q4_K + bf16-ref ceiling reflects quantisation noise, not bugs.

## Conversion (reproducibility)

```bash
# Set OMP_NUM_THREADS=1 to avoid a torch+OpenMP deadlock during bf16→f16
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  python models/convert-mimo-asr-to-gguf.py \
    --input XiaomiMiMo/MiMo-V2.5-ASR \
    --output mimo-asr-f16.gguf \
    --outtype f16

build-ninja-compile/bin/crispasr-quantize \
  mimo-asr-f16.gguf mimo-asr-q4_k.gguf q4_k
```

Vocab is padded to 151680 entries (151643 BPE + 30 special + 7 unused slots) and `tokenizer.ggml.merges` is populated (151291 entries). Earlier filenames (`mimo-asr.gguf`, `mimo-asr-q2_k.gguf`) shipped before commit `2191a70` with truncated vocab + missing merges and were removed from the repo on 2026-05-01.

## Citation

```bibtex
@misc{mimo2025v25asr,
  title = {MiMo-V2.5-ASR},
  author = {Xiaomi MiMo Team},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR}
}
```

## License

MIT — same as upstream.
