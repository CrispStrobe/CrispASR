---
license: mit
language:
- en
- zh
- es
- pt
- de
- ja
- ko
- fr
- ru
- id
- sv
- it
- he
- nl
- pl
- no
- tr
- th
- ar
- hu
- ca
- cs
- da
- fa
- af
- hi
- fi
- et
- el
- ro
- vi
- bg
- is
- sl
- sk
- lt
- sw
- uk
- lv
- hr
- ne
- sr
- tl
- ms
- ur
- mn
- hy
- jv
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- diarization
- speaker-diarization
- timestamps
- hotwords
- ggml
- gguf
- vibevoice
- qwen2
- speech-llm
- multilingual
- long-form
library_name: ggml
base_model: microsoft/VibeVoice-ASR
---

# VibeVoice-ASR — GGUF

GGUF / ggml conversions of [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR) for use with the `crispasr` CLI from **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

VibeVoice-ASR is Microsoft's **7B-parameter speech-LLM** capable of transcribing **up to 60 minutes of audio in a single pass**. Unlike most ASR models it outputs structured JSON containing **who** (speaker diarization), **when** (timestamps), and **what** (content) simultaneously — with support for **customised hotwords** and **50+ languages**.

- **60-minute long-form audio** in a single forward pass (streaming segmentation for >60 s)
- **Built-in speaker diarization** — Speaker IDs in every output segment
- **Word-level timestamps** — Start time / End time per segment
- **Hotword / context injection** — pass terms or metadata via `--context`
- **50+ languages** with automatic language detection
- **MIT licence**

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `vibevoice-asr-q4_k.gguf` | ~5 GB | **Q4_K_M — recommended default** |
| `vibevoice-asr-f16.gguf` | 16 GB | F16 — reference quality |

## Quick Start

```bash
# 1. Build CrispASR
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON   # macOS
cmake --build build -j$(nproc)

# 2. Download the quantised GGUF
huggingface-cli download cstr/vibevoice-asr-GGUF \
    vibevoice-asr-q4_k.gguf --local-dir .

# 3. Transcribe
./build/bin/crispasr --model vibevoice-asr-q4_k.gguf \
    --file audio.wav --backend vibevoice
```

Audio must be **24 kHz** mono. Pre-convert with:
```bash
ffmpeg -i input.mp3 -ar 24000 -ac 1 -c:a pcm_s16le output.wav
```

### With hotwords / context

```bash
./build/bin/crispasr --model vibevoice-asr-q4_k.gguf \
    --file meeting.wav --backend vibevoice \
    --context "ACME Corp, John Smith, Q3 earnings"
```

## Output Format

The model outputs **structured JSON** per segment:

```json
[
  {
    "start_time": "0:00:01.234",
    "end_time":   "0:00:04.567",
    "speaker_id": "Speaker_1",
    "text":       "And so, my fellow Americans …"
  },
  …
]
```

This differs from most ASR models — the LLM generates the JSON itself; no separate diarization step is needed.

## Architecture

VibeVoice-ASR is a four-module speech-LLM:

| Component | Details |
| --- | --- |
| **Acoustic σ-VAE encoder** | ConvNeXt-style 7-stage causal encoder, `depths=[3,3,3,3,3,3,8]`, `ratios=[8,5,5,4,2,2]`, `n_filters=32`. Each stage: `RMSNorm → depthwise-Conv1d → γ-scale → residual` + `RMSNorm → FFN → γ-scale → residual`. Total temporal downsampling **3200×** (24 kHz → 7.5 Hz). Output: `(T', 64)` VAE mean with `fix_std=0.5`. |
| **Semantic encoder** | Same ConvNeXt architecture, same strides. Output: `(T', 128)` mean (no sampling). |
| **SpeechConnectors** | Two independent `Linear(vae_dim → 3584) + RMSNorm + Linear(3584 → 3584)` projectors. Combined features = `acoustic_connector(at_mean) + semantic_connector(st_mean)`. |
| **LLM decoder** | Qwen2-7B: `d=3584`, 28 layers, 28 Q heads / 4 KV heads (GQA), `head_dim=128`, FFN=18944 (SiLU), RMSNorm `ε=1e-6`, RoPE `θ=1e6`, vocab=152064, max_pos=32768. |
| **Parameters** | ~7B total |

### Prompt format

```
<|im_start|>system
You are a helpful assistant that transcribes audio input into text output in JSON format.
<|im_end|>
<|im_start|>user
<|speech_start|><|speech_pad|>×N<|speech_end|>
This is a X.XX seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content
<|im_end|>
<|im_start|>assistant
```

Where `N = ceil(samples / 3200)`.

### Audio preprocessing

1. Resample to **24 kHz** mono
2. Normalise to **−25 dBFS** RMS: `audio *= 10^(−25/20) / (rms + 1e-6)`, then clip if |audio| > 1

### What's baked into the GGUF

- All acoustic encoder weights (F16 for conv weights except depthwise which stays F32)
- Both SpeechConnectors
- Full Qwen2-7B LLM weights
- Qwen2.5 tokenizer vocabulary (embedded as `tokenizer.ggml.tokens`)
- Architecture hyperparameters as GGUF metadata keys

## Implementation notes

The C++ runtime validates against the PyTorch reference at every pipeline boundary using `tools/dump_reference.py --backend vibevoice`:

| Stage | Key | Notes |
| --- | --- | --- |
| Audio normalisation | `audio_norm` | −25 dBFS RMS, clip guard |
| Acoustic encoder mean | `at_enc_mean` | (T', 64) — deterministic (mean, no noise) |
| Semantic encoder mean | `st_enc_mean` | (T', 128) — always deterministic |
| Acoustic connector | `at_conn_out` | (T', 3584) |
| Semantic connector | `st_conn_out` | (T', 3584) |
| Combined features | `speech_features` | elementwise sum of both connectors |
| Generated tokens | `llm_argmax` | greedy decode |

**Note on acoustic sampling**: the Python model uses `std_dist_type='gaussian'` (adds per-batch noise during training/inference for robustness). The C++ runtime uses the VAE mean directly (deterministic), which is equivalent to `dist_type='none'` and gives reproducible, comparable output for diff-testing.

## How this was made

1. `microsoft/VibeVoice-ASR` safetensors converted to GGUF F16 by [`models/convert-vibevoice-to-gguf.py`](https://github.com/CrispStrobe/CrispASR/blob/main/models/convert-vibevoice-to-gguf.py). Tensors loaded in native dtype (BF16) to avoid OOM on the large embedding table; converted to F16/F32 per-tensor at write time.
2. Quantised variants produced by the CrispASR `quantize` binary (Q4_K_M).
3. Full pipeline implemented in [`src/vibevoice.{h,cpp}`](https://github.com/CrispStrobe/CrispASR/blob/main/src/vibevoice.cpp): two encoder graphs + two connector graphs + Qwen2 autoregressive decoder with F16 KV cache.

## Related

- **Original model**: [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR) (MIT)
- **Reference code**: [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)
- **C++ runtime**: [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)
- **Technical report**: [arxiv.org/pdf/2601.18184](https://arxiv.org/pdf/2601.18184)
- Sister releases:
  - [`cstr/qwen3-asr-0.6b-GGUF`](https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF) — Qwen3-ASR 0.6B (fast, 30 languages)
  - [`cstr/voxtral-mini-3b-2507-GGUF`](https://huggingface.co/cstr/voxtral-mini-3b-2507-GGUF) — Voxtral Mini 3B (audio Q&A)
  - [`cstr/granite-speech-3.3-8b-GGUF`](https://huggingface.co/cstr/granite-speech-3.3-8b-GGUF) — Granite Speech 8B
  - [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF) — Parakeet TDT 600M

## License

MIT, inherited from the base model.
