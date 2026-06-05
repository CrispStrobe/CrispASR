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
- ggml
- gguf
- qwen3
- speech-llm
- multilingual
library_name: ggml
base_model: Qwen/Qwen3-ASR-0.6B
---

# Qwen3-ASR 0.6B — GGUF (ggml-quantised)

GGUF / ggml conversions of [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) for use with the `qwen3-asr-main` CLI from **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

Qwen3-ASR 0.6B is Alibaba's **speech-LLM** ASR model:

- **30 languages + 22 Chinese dialects** with automatic language detection
- **6.42 % avg WER** on the HuggingFace [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
- **Apache-2.0** licence
- **Speech-LLM architecture**: Whisper-style audio encoder (2D-conv subsampler + 18-layer Transformer + projector head, 896 → 1024) feeds frames into a stock **Qwen3 0.6B LLM** (28 layers, GQA 16/8, head_dim=128, Q-norm/K-norm, SwiGLU, RoPE θ=1e6) via embedding splice at `<|audio_pad|>` placeholder positions in a ChatML prompt. The LLM autoregressively generates the transcript.

This is the **first speech-LLM** in the CrispASR family — every other model in the set uses a dedicated CTC / transducer / encoder-decoder. The Qwen3-ASR runtime ships with a persistent KV cache so per-token decode is O(1) in cache size, not O(N) full re-forwards.

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `qwen3-asr-0.6b.gguf`        | 1.88 GB | F16 |
| `qwen3-asr-0.6b-q8_0.gguf`   | 961 MB  | Q8_0, near-lossless |
| `qwen3-asr-0.6b-q4_k.gguf`   | 676 MB  | **Q4_K — recommended default**, faster than realtime on a 4-core CPU |

All quantisations produce the correct transcript on `samples/jfk.wav`:
> And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.

The mel filterbank from `WhisperFeatureExtractor` is **baked into the GGUF** as `audio.mel_filters` (along with `audio.mel_window`), so the C++ runtime computes the log-mel spectrogram natively without needing torch / librosa / scipy at inference time.

## Quick Start

```bash
# 1. Build the runtime
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target qwen3-asr-main

# 2. Download a quantisation
huggingface-cli download cstr/qwen3-asr-0.6b-GGUF \
    qwen3-asr-0.6b-q4_k.gguf --local-dir .

# 3. Transcribe
./build/bin/qwen3-asr-main \
    -m qwen3-asr-0.6b-q4_k.gguf \
    -f your-audio.wav -t 8
```

Audio must be 16 kHz mono 16-bit PCM WAV. Pre-convert with:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

## Performance

Measured on `samples/jfk.wav` (11 seconds), Apple-class 4-core CPU:

| Variant | Mel | Encoder | Prefill | Decode/tok | **Total** |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16  | 250 ms | 2660 ms | 3032 ms | 151 ms | 10.3 s |
| Q8_0 | 236 ms | 2459 ms | 2840 ms | 137 ms |  9.5 s |
| **Q4_K** | 246 ms | 2851 ms | 2721 ms | **118 ms** | **9.3 s** |

Q4_K runs **slightly faster than realtime** with no quality loss on this clip.

## Architecture

| Component | Details |
| --- | --- |
| Audio encoder | 18-layer Whisper-style pre-LN Transformer, d=896, heads=14, head_dim=64, FFN=3584 |
| Conv subsampler | 3 × Conv2D stride-2 (1→480→480→480), then linear (480·16=7680 → 896). Output frame rate ~13 frames / second of audio (~77 ms / frame) |
| Projector | ln_post → proj1 (896→896) → GELU → proj2 (896→1024) |
| LLM | Qwen3 0.6B: 28 layers, hidden=1024, **16 Q heads / 8 KV heads (GQA)**, head_dim=128, FFN=3072, SwiGLU, RMSNorm, **per-head Q-norm / K-norm**, NEOX-style RoPE θ=1e6 |
| Vocab | 151 936 tokens (Qwen2 BPE, GPT-2 byte encoding) |
| Audio | 16 kHz mono, 128 mel bins, n_fft=400, hop=160, win=400 (matches `WhisperFeatureExtractor`) |
| Audio injection | `<|audio_pad|>` placeholder positions in ChatML prompt get their token embedding replaced with the encoder output frames |
| Parameters | ~900 M |

## Implementation notes (correctness)

The C++ runtime is verified to F16 numerical precision against the PyTorch reference at every architectural boundary on `samples/jfk.wav`:

| Stage | Diff metric | Result |
| --- | --- | --- |
| Conv front-end (per-chunk Conv2D + flatten + linear) | max abs vs `conv_out.npy` | 1.43e-4 |
| Full audio encoder (18 layers + projector) | per-row cosine sim vs `proj2_out.npy` | mean 1.000000, min 0.999999 |
| Qwen3 LLM forward (28 layers, no audio) | per-position cosine sim vs `llm_logits.npy` | mean 0.999999, top-1 9/9 |
| End-to-end (audio → spliced embeds → LLM → greedy decode) | reproduced reference token sequence | 26 / 26 |
| Mel filterbank (C++ STFT vs `WhisperFeatureExtractor`) | max abs vs `mel_input.npy` | 2.2e-2 |

### Bugs that would have been hours of debugging

A few non-obvious gotchas the port had to handle:

1. **`ggml_permute` semantics** are inverted from the obvious reading: `permute(t, p0, p1, p2, p3)` means "source axis i goes to NEW position `p_i`", not "new axis i comes from source axis `p_i`".
2. **PyTorch hooks fire pre-GELU** when registered on an `nn.Conv2d` module — the `F.gelu` is applied externally in the forward function.
3. **`cu_seqlens` is GPU-only**: `eager_attention_forward` (used on CPU) **ignores** `cu_seqlens` and does standard full self-attention. The "windowed attention" path only kicks in for FlashAttention2 on GPU. **Don't apply the windowed mask on CPU** — the reference produces full-attention output.
4. **`WhisperFeatureExtractor.mel_filters` shape is `(n_freqs=201, n_mels=128)`**, not `(n_mels, n_freqs)` as the parameter ordering might suggest.
5. **Qwen3 attention output width** is `hd × n_q_heads = 2048`, not `d_model = 1024`. The o_proj is `(2048 → 1024)`, so the attention output is reshaped to `(2048, T)` before o_proj.
6. **mrope sidestep**: Qwen3-ASR uses interleaved multi-modal RoPE with `mrope_section=[24,20,20]`. For text-only or 1D-position input (which includes our spliced audio frames), the three mrope sections all receive identical position_ids and **collapse to standard 1D RoPE**. The simpler RoPE matches the reference perfectly for our use case.

See [`qwen3-asr-todo.md`](https://github.com/CrispStrobe/CrispASR/blob/main/qwen3-asr-todo.md) in the runtime repo for the complete work log.

## How this was made

1. The HF safetensors model was converted to GGUF F16 by [`models/convert-qwen3-asr-to-gguf.py`](https://github.com/CrispStrobe/CrispASR/blob/main/models/convert-qwen3-asr-to-gguf.py). All 612 tensors map cleanly. The mel filterbank (from `WhisperFeatureExtractor.mel_filters`) and Hann window are baked into the GGUF as `audio.mel_filters` / `audio.mel_window`.
2. Quantised variants are produced by `crispasr-quantize` (the same llama.cpp-style quantiser used for the other GGUF releases in this family).
3. Inference is implemented in [`src/qwen3_asr.{h,cpp}`](https://github.com/CrispStrobe/CrispASR/blob/main/src/qwen3_asr.cpp): the encoder and the LLM each run as one ggml graph, with a persistent F32 KV cache `(head_dim, max_ctx, n_kv_heads, n_layers)` shared between prefill and per-token decode steps.

## Reference implementation

[`predict-woo/qwen3-asr.cpp`](https://github.com/predict-woo/qwen3-asr.cpp) (MIT) was read for architecture discovery and tensor name mapping. **No source code was vendored** — the CrispASR runtime is a re-implementation in this repo's existing FastConformer / cohere-style ggml infrastructure, sharing structures with the four other ASR runtimes in the family.

## Supported languages

`ar cs da de el en es fa fi fil fr hi hu id it ja ko mk ms nl pl pt ro ru sv th tr vi yue zh` plus 22 Chinese dialects (auto-detected at inference time).

## Attribution

- **Original model**: [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) (Apache-2.0). Alibaba Cloud Qwen team.
- **GGUF conversion + ggml runtime**: [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR) — community contribution.
- **Reference implementation**: [predict-woo/qwen3-asr.cpp](https://github.com/predict-woo/qwen3-asr.cpp) (MIT) — used for architecture discovery only, no code vendored.

## Related

- C++ runtime: **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**
- Sister releases in the same family:
  - [`cstr/cohere-transcribe-03-2026-GGUF`](https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF) — Cohere Transcribe 2B (Open ASR Leaderboard #1)
  - [`cstr/parakeet-tdt-0.6b-v3-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF) — Parakeet TDT 600M (free word timestamps)
  - [`cstr/canary-1b-v2-GGUF`](https://huggingface.co/cstr/canary-1b-v2-GGUF) — Canary 978M (speech translation)
  - [`cstr/canary-ctc-aligner-GGUF`](https://huggingface.co/cstr/canary-ctc-aligner-GGUF) — universal multilingual forced aligner

## License

Apache-2.0, inherited from the base model.
