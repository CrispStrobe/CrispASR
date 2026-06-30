---
license: apache-2.0
language:
  - en
  - zh
library_name: crispasr
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - tts
  - gguf
  - dots-tts
  - flow-matching
  - crispasr
base_model: dots/dots.tts-soar
---

# dots.tts-soar — GGUF (CrispASR)

GGUF conversions of [**dots.tts-soar**](https://huggingface.co/dots/dots.tts-soar),
a continuous-latent autoregressive text-to-speech model, for use with
[**CrispASR**](https://github.com/CrispStrobe/CrispASR) — a portable
C/C++ speech engine built on `ggml`. No Python or PyTorch needed at
inference time; runs on CPU (Metal/CUDA/Vulkan optional).

> dots.tts generates 48 kHz speech patch-by-patch in a continuous latent
> space (no discrete audio codec). A Qwen2.5-1.5B language model drives a
> flow-matching DiT head that predicts acoustic latents, which a BigVGAN
> vocoder renders to waveform.

## Files

| File | Size | Contents | Use |
|------|------|----------|-----|
| `dots-tts-soar-f16.gguf`         | 4.6 GB | Full model, F16 | Reference quality |
| `dots-tts-soar-q8_0.gguf`        | 3.1 GB | **Mixed-quant** (Q8_0 LLM + PatchEncoder, **F16 DiT**) | Recommended default |
| `dots-tts-soar-q4_k.gguf`        | 2.0 GB | **Mixed-quant** (Q4_K LLM + PatchEncoder, **F16 DiT**) | Smallest footprint |
| `dots-tts-soar-vocoder-f16.gguf` | 0.36 GB | BigVGAN 48 kHz vocoder | Required companion |
| `dots-tts-soar-spk-f16.gguf`     | 0.01 GB | CAM++ speaker encoder | Optional (voice cloning) |

Each core GGUF needs the **vocoder** companion. Pick **one** core + the vocoder.

## ⚠️ Why the quants are "mixed"

The flow-matching **DiT head is kept at F16 in every quant.** It runs in a
classifier-free-guidance Euler ODE loop (≈16 steps × 18 layers × 2 CFG =
hundreds of forwards per utterance); per-step quantization noise compounds
and **derails generation** — validated: a fully-Q8 DiT pushes the
flow-match cosine to ~0.994 and produces no-EOS runaway / garbled audio.

The LLM (cos 0.999 quantized) and PatchEncoder (cos 0.9999 quantized) are
robust, so the quants shrink **those** layers while leaving the DiT,
projections, embeddings and denormalization statistics at source precision.
Result: `q8_0` and `q4_k` are ASR-roundtrip-verbatim against F16, at a
fraction of the size. **Do not re-quantize the DiT yourself.**

## Usage (CrispASR)

```bash
# Build CrispASR (see repo README), then:
./build/bin/crispasr --backend dots-tts \
  -m dots-tts-soar-q8_0.gguf \
  --tts "Hello world." \
  --tts-output out.wav
```

The **vocoder companion is auto-discovered** as a sibling file next to the
core model (`dots-tts-soar-vocoder-*.gguf` in the same directory), so no
extra flag is needed. To point at a vocoder elsewhere, pass
`--codec-model dots-tts-soar-vocoder-f16.gguf` (there is **no**
`--tts-vocoder` flag). The optional CAM++ speaker encoder
(`dots-tts-soar-spk-f16.gguf`) is discovered the same way and only used
when you pass `--voice` for cloning.

Or let the model registry fetch the default (F16 core + vocoder)
automatically with `-m auto --backend dots-tts`.

### Tuning knobs (env)

| Variable | Default | Effect |
|----------|---------|--------|
| `CRISPASR_DOTS_MAX_PATCHES`   | 200 | Hard cap on generated audio patches |
| `CRISPASR_DOTS_ODE_STEPS`     | 16  | Flow-matching Euler steps (quality vs speed) |
| `CRISPASR_DOTS_EOS_THRESHOLD` | 0.5 | End-of-speech probability threshold |

## Architecture

- **LLM**: Qwen2.5-1.5B (28 layers, 12 Q / 2 KV heads, hidden 1536,
  head_dim 128, RoPE θ=1e6) — autoregressive driver.
- **PatchEncoder**: 24-layer VAE semantic encoder (RMSNorm, no RoPE,
  no QK-norm) — re-encodes generated latents back into the LLM stream.
- **DiT flow-matching head**: 18 layers (RoPE + AdaLN), CFG Euler ODE —
  predicts continuous acoustic latents. **Kept at F16.**
- **Vocoder**: BigVGAN, 48 kHz output.
- **Speaker**: CAM++ (optional, for voice conditioning).

`patch_size=4`, `latent_dim=128`, `out_ds_rate=2`, `dit_dim=1024`.

## License

Apache-2.0, inherited from the upstream
[dots/dots.tts-soar](https://huggingface.co/dots/dots.tts-soar) model.

## Conversion

Converted and validated stage-by-stage (per-component cosine against the
PyTorch reference) with the CrispASR diff harness. Mixed-quant produced by
`crispasr-quantize`, which automatically preserves the DiT and all
sampling-critical tensors at F16.
