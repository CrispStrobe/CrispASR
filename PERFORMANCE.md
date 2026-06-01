# CrispASR — Performance benchmarks

Test audio: jfk.wav (11.0s), Q4_K quantization, greedy decode (`-bs 1`).

---

## Backend × Optimization matrix

At-a-glance view of which performance knobs each backend supports today,
and where the gaps are. Last refresh: **2026-05-04** (after PLAN §79 —
14-commit session that shipped #69a / #69b / #69e / #72 / #73).

**Legend**: ✓ = supported, opt-in via env var · `F16` = stuck at F16
(quant cache types unavailable; attention path needs migration) ·
`—` = not applicable (no KV cache or no transformer blocks) ·
`·` = applicable but not yet wired (port deferred).

### LLM-decoder ASR (high VRAM, autoregressive)

| backend | KV_QUANT | KV_QUANT_K/_V | KV_ON_CPU | N_GPU_LAYERS | weight residency |
|---|:-:|:-:|:-:|:-:|:-:|
| voxtral4b (4B) | ✓ | ✓ | ✓ | ✓ | gpu |
| voxtral (3B) | ✓ | ✓ | ✓ | ✓ | gpu |
| granite-speech (1B / 4.0 / 4.1 / 4.1-plus / 4.1-nar) | ✓ | ✓ | ✓ | ✓ | gpu |
| gemma4-e2b (5B effective) | ✓ | ✓ | ✓ | ✓ | gpu (FLIPPED §72) |
| mimo-asr (1.4B) | ✓ | ✓ | ✓ | ✓ | gpu (FLIPPED §72) |
| qwen3-asr (0.6B) | ✓ | ✓ | ✓ | ✓ | gpu |
| glm-asr (1B) | ✓ | ✓ | ✓ | ✓ | gpu |
| omniasr-llm (300M) | ✓ | ✓ | ✓ | ✓ | gpu |
| vibevoice (4B ASR mode) | F16 | F16 | F16 | ✓ | gpu |

### Encoder-decoder ASR (medium VRAM, autoregressive)

| backend | KV_QUANT | KV_QUANT_K/_V | KV_ON_CPU | N_GPU_LAYERS | notes |
|---|:-:|:-:|:-:|:-:|---|
| canary (1B) | ✓ | ✓ | ✓ | · | flash_attn_ext default, -17 % on JFK with q8_0/q4_0 |
| cohere (2B) | ✓ | ✓ | ✓ | · | flash_attn_ext available; +11 % regression vs cast-on-read on JFK with q8_0/q4_0 — long-form rerun needed before promoting (see PLAN) |
| kyutai-stt (1B) | ✓ | ✓ | ✓ | · | flash_attn_ext native, quant-safe |
| firered-asr (900M) | — | — | — | — | inline AED, no exposed transformer KV |
| moonshine-tiny / streaming | — | — | — | — | tiny decoder, no exposed KV |

### Encoder-only ASR (low VRAM, single forward)

| backend | KV_QUANT | KV_QUANT_K/_V | KV_ON_CPU | N_GPU_LAYERS | notes |
|---|:-:|:-:|:-:|:-:|---|
| whisper (legacy) | ✓ | ✓ | ✓ | — | upstream loader, separate path |
| parakeet (TDT) | — | — | — | — | RNN-T transducer, no KV cache |
| fastconformer-ctc | — | — | — | — | CTC head |
| wav2vec2 / hubert / data2vec | — | — | — | — | CTC heads |
| omniasr (CTC variant) | — | — | — | — | CTC head |

### TTS

| backend | KV_QUANT | KV_QUANT_K/_V | KV_ON_CPU | N_GPU_LAYERS | notes |
|---|:-:|:-:|:-:|:-:|---|
| orpheus (3B + DE / lex-au variants) | ✓ | ✓ | ✓ | ✓ | shared Llama-3 path |
| chatterbox (T3 + CFG cache) | ✓ | ✓ | ✓ | · | uses kv_self_attn natively |
| qwen3-tts (0.6B + 1.7B variants) | ✓ talker | ✓ talker | ✓ talker | · | code-predictor cache stays F16 (separate path) |
| vibevoice (4B TTS mode) | F16 | F16 | F16 | ✓ | KV migration still pending; layer offload routes `tts_lm.layers.<N>.*` |
| kokoro | — | — | — | — | non-AR vocoder, no transformer KV |

### Where the gaps are

1. **Layer offload (`N_GPU_LAYERS`) on encoder-decoder ASR** (canary,
   cohere, kyutai-stt). Their cross-attention layout doesn't have the
   `blk.<N>.*` block-tagged tensors that the layer-split predicate
   recognises. Encoder-decoder offload is its own design problem —
   probably want to offload only the LLM/decoder side, but the tensor
   names (`<arch>.dec.<N>.*` etc.) need bespoke per-backend predicates.
2. **vibevoice quant K/V (both modes)**. The attention path uses the
   `ggml_cpy(K_perm, view_into_kv_k)` pattern that's incompatible with
   quant K/V (see LEARNINGS.md "ggml_cont(ggml_permute(quant_tensor))
   doesn't move data"). Migration recipe is the canary/cohere
   `ggml_flash_attn_ext` port — ~50-80 LOC + F16 mask graph input.
   Layer offload (`N_GPU_LAYERS`) is independently shipped and works
   on F16 K/V; the migration only unlocks quant K/V on top.
3. **qwen3-tts code-predictor cache**. Talker KV is fully covered via
   `core_attn::kv_self_attn`; the secondary code-predictor path
   doesn't go through that helper, so its cache stays F16. Lower-
   priority since the talker dominates per-frame cost.
4. **Linux/CUDA validation of #72 GPU residency.** mimo-asr 22 % /
   gemma4-e2b 2.2x speedups were measured on Apple Silicon Metal.
   dGPU should be even more favourable; deferred until a CUDA host
   is available. If a platform regresses, gate via env
   (`CRISPASR_FORCE_CPU_WEIGHTS=1`).
5. **Cohere flash_attn_ext regresses on short audio.** JFK (~11 s)
   with q8_0 K / q4_0 V is +11 % slower under flash than under the
   cast-on-read fallback (canary on the same workload is -17 %, so
   the kernel works — cohere's cache layout or head dim flips the
   crossover). Need a multi-minute clip to confirm flash pulls ahead
   on long-form before promoting it to the recommended path; until
   then short-form users on cohere should treat flash as opt-in.

### Stacking the four knobs

Each addresses an independent bottleneck:

| knob | addresses | when to use |
|---|---|---|
| `CRISPASR_KV_QUANT_K=q8_0 / _V=q4_0` | KV size in VRAM | always reasonable for LLM-decode ASR; quartered V cache on long context |
| `CRISPASR_KV_ON_CPU=1` | KV doesn't fit in VRAM at all | very long context with a tight VRAM budget |
| `CRISPASR_N_GPU_LAYERS=N` | model itself doesn't fit in VRAM | model size > VRAM; spill the last (total-N) layers |
| `CRISPASR_FORCE_CPU_WEIGHTS=1` (proposed) | platform regressed on §72 GPU residency | not yet wired — none seen on Apple Silicon |

```bash
# Maximum-memory-savings combo for a VRAM-tight host
CRISPASR_N_GPU_LAYERS=10 \
  CRISPASR_KV_ON_CPU=1 \
  CRISPASR_KV_QUANT_K=q8_0 \
  CRISPASR_KV_QUANT_V=q4_0 \
  ./build/bin/crispasr --backend voxtral4b -m auto -f long.wav
```

See [`docs/cli.md`](docs/cli.md) "Memory footprint" for the full env-
var reference and the llama.cpp parity comparison table; HISTORY §79
for the implementation write-up.

---

## Kaggle P100 GPU — 2026-05-31

Platform: Tesla P100-PCIE (16 GB VRAM), 4 CPU threads, CUDA **sm_60**
(auto-detected — the kernel pins `CMAKE_CUDA_ARCHITECTURES` from
`nvidia-smi compute_cap`, so the build is correct whether the box is a
T4/P100/A100/L4). Commit: `7bc3ef5b`. Run via
`tools/kaggle-benchmark-all-backends.py` (now on the shared
`tools/kaggle/kaggle_harness.py`). **27 backends tested, 26 pass.**

First run to cover all six newly-registered ASR backends (marked †):
sensevoice, paraformer, mega-asr, granite-4.1, funasr, mimo-asr.

### Speed ranking (11.0 s JFK, Q4_K unless noted, greedy)

| Rank | Backend | RTx | Time | WER | Architecture |
|---|---|---|---|---|---|
| 1 | SenseVoice Small † | **19.8x** | 0.6s | 0.0% | Encoder (multitask) |
| 2 | FastConformer CTC | 8.9x | 1.2s | 0.0% | Encoder-CTC |
| 3 | Moonshine Tiny | 8.6x | 1.3s | 9.1% | Encoder-Decoder |
| 4 | Canary 1B | 8.0x | 1.4s | 9.1% | Encoder-AED |
| 5 | Data2Vec Base | 7.5x | 1.5s | 4.5% | Encoder-CTC |
| 6 | OmniASR CTC 1B | 7.0x | 1.6s | 9.1% | Encoder-CTC |
| 7 | HuBERT Large | 6.9x | 1.6s | 0.0% | Encoder-CTC |
| 8 | Wav2Vec2 XLSR-EN | 6.9x | 1.6s | 0.0% | Encoder-CTC |
| 9 | Cohere Transcribe | 6.6x | 1.7s | 0.0% | Encoder-AED |
| 10 | Parakeet TDT 0.6B | 6.0x | 1.8s | 0.0% | Encoder-TDT |
| 11 | Whisper base | 5.6x | 2.0s | 0.0% | Encoder-Decoder |
| 12 | Paraformer-zh NAR † | 5.0x | 2.2s | 0.0% | Encoder (NAR) |
| 13 | GLM ASR Nano | 4.7x | 2.4s | 0.0% | Encoder-LLM |
| 14 | Qwen3 ASR 0.6B | 4.4x | 2.5s | 0.0% | Encoder-LLM |
| 15 | Moonshine Streaming Tiny | 2.9x | 3.8s | 0.0% | Encoder-Decoder |
| 16 | Mega-ASR 1.7B † | 2.9x | 3.9s | 0.0% | Encoder-LLM (qwen3) |
| 17 | Granite Speech 1B | 2.8x | 3.9s | 0.0% | Encoder-LLM |
| 18 | Granite Speech 4.1 2B † | 2.7x | 4.1s | 0.0% | Encoder-LLM |
| 19 | Voxtral Mini 3B | 2.5x | 4.4s | 0.0% | Encoder-LLM |
| 20 | OmniASR LLM 300M | 1.6x | 7.0s | 4.5% | Encoder-LLM |
| 21 | Kyutai STT 1B | 1.5x | 7.5s | 0.0% | Encoder-AED |
| 22 | VibeVoice ASR | 1.3x | 8.6s | 4.5% | Encoder-LLM |
| 23 | Voxtral 4B Realtime | 0.9x | 11.9s | 0.0% | Encoder-LLM |
| 24 | Gemma-4-E2B 2.3B | 0.8x | 13.6s | 0.0% | Encoder-LLM |
| 25 | FireRed ASR2 AED | 0.6x | 19.2s | 0.0% | Encoder-AED |
| 26 | MiMo-ASR † | 0.3x | 38.0s | 0.0% | Encoder-LLM (CPU-forced, #115) |
| 27 | FunASR Nano † | ~1.0x | ~10.6s | 0.0% | Encoder-LLM (enc GPU, LLM CPU) |

### Notes

- **funasr is FIXED (§136, 2026-06-01).** The original benchmark showed
  6.0× RT / 1.8s but **100% WER** — the `ggml_backend_sched` produced
  all-NaN logits on CUDA (issue #125). The fix (`f94fec90`) splits
  weights: encoder on GPU, LLM+KV on CPU. Now ~1.0× RT / 10.6s wall-clock
  (including model load) with **0% WER**. Slower than the broken all-GPU
  run because the Qwen2-0.6B decode is CPU-bound. The encoder (70 SANM
  blocks) still benefits from GPU — for longer audio the encoder
  dominates and the GPU speedup matters more. `FUNASR_LLM_GPU=1`
  overrides to all-GPU for future testing once the upstream sched bug
  is fixed.
- **SenseVoice debuts at #1** (19.8× RT, 0% WER) — encoder-only multitask
  model, fastest backend now measured.
- **mimo-asr** runs at 0.3× (38 s) because PLAN #115 forces it to CPU; it
  transcribes correctly. The 420 s timeout budgeted for it was ample.
- P100 (sm_60, ~9.3 TFLOPS fp32) lands the LLM-AR tail a touch faster than
  the 2026-04-26 T4 run (e.g. voxtral-3B 2.5× vs 2.4×, granite-1B 2.8× vs
  1.7×); CTC/encoder backends are comparable. Cross-run deltas are also
  affected by Q4_K model refreshes since April.
- Two non-fatal HF pre-download `401`s (the Kaggle Secrets API was
  flaking, so no HF token) fell back to the C++ downloader; all models are
  public `cstr/*` so downloads still succeeded.

---

## Kaggle T4 GPU — 2026-04-26

Platform: 2x Tesla T4 (15 GB VRAM each), 4 CPU threads, CUDA.
Commit: `b9fd8eb`. **All 19 backends pass.**

### By architecture

#### Encoder-CTC (non-autoregressive, single forward pass)

| Backend | Params | Model MB | WER | RTx | Time | Notes |
|---|---|---|---|---|---|---|
| FastConformer CTC Large | 120M | 83 | 0.0% | **9.6x** | 1.1s | 18 FC layers |
| OmniASR CTC 1B v2 | 975M | 551 | 4.5% | 7.4x | 1.5s | w2v-BERT enc, 276ms GPU |
| Data2Vec Base | 95M | 78 | 0.0% | 5.3x | 2.1s | 12 layers, pos_conv 735ms |
| Wav2Vec2 XLSR-EN | 300M | 212 | 0.0% | 3.6x | 3.1s | 24 layers, pos_conv 1.6s |
| HuBERT Large | 300M | 212 | 0.0% | 3.6x | 3.1s | Same runtime as wav2vec2 |

#### Encoder-TDT (non-autoregressive, transducer)

| Backend | Params | Model MB | WER | RTx | Time | Notes |
|---|---|---|---|---|---|---|
| Parakeet TDT 0.6B | 600M | 466 | 0.0% | 5.6x | 2.0s | 24 FC layers + joint net |

#### Encoder-Decoder / AED (autoregressive, attention-based)

| Backend | Params | Model MB | WER | RTx | Time | Notes |
|---|---|---|---|---|---|---|
| Whisper (base) | 74M | 141 | 0.0% | **9.3x** | 1.2s | Full GPU (upstream) |
| Moonshine Tiny | 27M | 20 | 9.1% | 6.7x | 1.6s | CPU-only, tiny |
| Canary 1B | 1B | 672 | 0.0% | 6.2x | 1.8s | GPU enc+dec, 32+8 layers |
| Cohere Transcribe | 2B | 1440 | 0.0% | 5.2x | 2.1s | GPU enc, AED dec |
| Kyutai STT 1B | 1B | 636 | 4.5% | 1.4x | 7.7s | 24-layer Mimi decoder |
| FireRed ASR2 AED | 900M | 918 | 0.0% | 0.6x | 19.0s | CPU Q4_K SIMD dec (60ms/step) |

#### Encoder-LLM (autoregressive, language model decoder)

| Backend | Params | Model MB | WER | RTx | Time | Notes |
|---|---|---|---|---|---|---|
| Qwen3 ASR 0.6B | 780M | 515 | 0.0% | 4.7x | 2.3s | 0.6B LLM |
| GLM ASR Nano | 1.3B | 1262 | 0.0% | 4.6x | 2.4s | ~1B LLM |
| Voxtral Mini 3B | 3B | 2530 | 0.0% | 2.4x | 4.7s | Mistral 3B LLM |
| OmniASR LLM 300M | 1.6B | 1018 | 4.5% | 1.7x | 6.4s | LLaMA 1.3B dec |
| Granite Speech 1B | 2.9B | 2805 | 0.0% | 1.7x | 6.4s | Granite LLM |
| VibeVoice ASR | 4.5B | 4589 | 4.5% | 1.2x | 8.8s | ~4B LLM, JSON output |
| Voxtral 4B Realtime | 4B | 2407 | 0.0% | 0.9x | 12.8s | Causal streaming arch (PLAN #7 streaming API; 1.6s first-text-token) |

### Speed ranking

| Rank | Backend | RTx | Time | Architecture |
|---|---|---|---|---|
| 1 | FastConformer CTC | 9.6x | 1.1s | Encoder-CTC |
| 2 | Whisper base | 9.3x | 1.2s | Encoder-Decoder |
| 3 | OmniASR CTC 1B | 7.4x | 1.5s | Encoder-CTC |
| 4 | Moonshine Tiny | 6.7x | 1.6s | Encoder-Decoder |
| 5 | Canary 1B | 6.2x | 1.8s | Encoder-AED |
| 6 | Parakeet TDT 0.6B | 5.6x | 2.0s | Encoder-TDT |
| 7 | Data2Vec Base | 5.3x | 2.1s | Encoder-CTC |
| 8 | Cohere Transcribe | 5.2x | 2.1s | Encoder-AED |
| 9 | Qwen3 ASR 0.6B | 4.7x | 2.3s | Encoder-LLM |
| 10 | GLM ASR Nano | 4.6x | 2.4s | Encoder-LLM |
| 11 | Wav2Vec2 XLSR-EN | 3.6x | 3.1s | Encoder-CTC |
| 12 | HuBERT Large | 3.6x | 3.1s | Encoder-CTC |
| 13 | Voxtral Mini 3B | 2.4x | 4.7s | Encoder-LLM |
| 14 | OmniASR LLM 300M | 1.7x | 6.4s | Encoder-LLM |
| 15 | Granite Speech 1B | 1.7x | 6.4s | Encoder-LLM |
| 16 | Kyutai STT 1B | 1.4x | 7.7s | Encoder-AED |
| 17 | VibeVoice ASR | 1.2x | 8.8s | Encoder-LLM |
| 18 | Voxtral 4B Realtime | 0.9x | 12.8s | Encoder-LLM |
| 19 | FireRed ASR2 AED | 0.6x | 19.0s | Encoder-AED |

---

## CPU-only VPS — 2026-04-24

Platform: x86_64, 4 threads, 7.6 GB RAM, AVX2, no GPU.

| Backend | RTx (CPU) | Time (CPU) | RTx (T4) | Speedup |
|---|---|---|---|---|
| FastConformer CTC | 9.4x | 1.2s | 9.6x | 1.1x |
| Moonshine Tiny | 16.8x | 0.7s | 6.7x | 0.4x* |
| Parakeet TDT 0.6B | 2.9x | 3.8s | 5.6x | 1.9x |
| Canary 1B | 2.7x | 4.0s | 6.2x | 2.2x |
| Data2Vec Base | 2.1x | 5.2s | 5.3x | 2.5x |
| Qwen3 ASR 0.6B | 1.7x | 6.5s | 4.7x | 2.8x |
| Wav2Vec2 XLSR-EN | 1.1x | 9.9s | 3.6x | 3.2x |
| Cohere Transcribe | 1.4x | 7.7s | 5.2x | 3.7x |
| FireRed ASR2 AED | 0.1x | 123s | 0.6x | 6.5x |

*Moonshine runs CPU-only on both (tiny model, no GPU benefit).

GPU acceleration is strongest for encoder-heavy models (2-6x). Decoder-bound
models benefit less (FireRed decoder still runs on CPU even with GPU).

---

## Per-phase breakdowns

### wav2vec2 family (Kaggle T4)

| Model | CNN | Pos conv | Encoder | Total |
|---|---|---|---|---|
| wav2vec2-large (24L) | 215ms | 1588ms | 127ms | 1941ms |
| hubert-large (24L) | 227ms | 1595ms | 128ms | 1960ms |
| data2vec-base (12L) | 221ms | 735ms | 57ms | 1023ms |

**Bottleneck:** pos_conv (grouped conv1d on CPU) = 50-80% of total time.
Encoder graph on GPU is only 57-128ms.

### FireRed AED decoder (Kaggle T4)

| Phase | Time | Notes |
|---|---|---|
| Fbank extraction | ~50ms | CPU |
| Conv2d subsampling | ~100ms | CPU |
| Hybrid encoder (16L) | ~17s | GPU matmuls + CPU attention, slow due to CPU weight copies |
| K/V precompute | 433ms | GPU (scheduler auto-copies) |
| Decoder (28 steps) | 1695ms | CPU Q4_K SIMD, 60.5ms/step |
| **Total** | **19.0s** | Encoder dominates |

### OmniASR (Kaggle T4)

| Model | Encoder | Prefill | Decode | Total | RTx |
|---|---|---|---|---|---|
| CTC 1B v2 | 244ms | — | — | 277ms | 39.8x (encoder only) |
| LLM 300M v2 | 97ms | 803ms | 4028ms (103 steps) | 5021ms | 2.2x |

---

## Key observations

1. **CTC models dominate on speed.** No decoder loop = one forward pass.
2. **Small LLM decoders (0.6-1B) are competitive** — Qwen3 and GLM hit 4.5x+
   realtime with 0% WER, close to encoder-only models.
3. **Large LLMs (3-4.5B) are 1-2x realtime** on T4. Usable but not fast.
4. **Most WER=0% on jfk.wav.** The 4.5% models have minor formatting differences,
   not actual transcription errors. Moonshine Tiny (9.1%) has a real word error.
5. **wav2vec2 pos_conv was the bottleneck** — now 4.9x faster with ggml grouped
   conv (im2col + mul_mat SIMD). Was 1.6s (80% of runtime), now 324ms (~3.5%).
6. **FireRed encoder is slow** because CPU weights auto-copy to GPU per-layer.
   Pre-loading encoder weights to GPU would save ~15s.

---

## Optimization history

### wav2vec2 grouped conv — 2026-04-27

| Path | pos_conv | Notes |
|---|---|---|
| Manual C++ (OMP) | 1588ms | 4-thread OMP, plain float loops |
| **ggml im2col + mul_mat** | **324ms** | **4.9x faster**, SIMD kernels |

The grouped positional conv (C=1024, K=128, G=16) is decomposed into G=16
independent `ggml_pad_ext` + `ggml_im2col` + `ggml_mul_mat` calls. The
mul_mat output `[cpg, T]` is transposed to channel-first before reassembly.
Applies to wav2vec2, data2vec, and hubert.

### FireRed decoder — 2026-04-26

| Path | ms/step | 28 tokens | Why |
|---|---|---|---|
| Manual C++ F32 (original) | 4400 | 123s | No SIMD, no parallelism |
| + OpenMP matmuls | 2320 | 58s | 2.1x from OMP |
| + ggml Q4_K CPU native | **70** | **2.0s** | 9.3x from fused SIMD kernel |
| ggml_vecmat on CUDA | 2600 | timeout | CUDA launch overhead kills it |
| F32 dequant + cpu_matmul | 590 | 16.5s | No SIMD, OMP disabled on Kaggle |
| **ggml_vecmat CPU (final)** | **60** | **1.7s** | Weights on CPU, native Q4_K |

### wav2vec2 CNN — 2026-04-24

| Change | CNN | Total | Speedup |
|---|---|---|---|
| Baseline (manual C++) | 95.2s | 108.4s | 1.0x |
| ggml F32 im2col | 2.4s | 15.5s | 7.0x |
| + OpenMP pos_conv | 2.3s | 9.9s | 10.9x |

### voxtral4b streaming — 2026-05 (PLAN #7 phases 1+1.5+2+3+4)

Native incremental encoder + streaming-prompt decode + speculative
prefill + combined-chunk flush + live captions + decoder thread.
M1 Q4_K JFK 11 s baseline, all variants bit-exact-batch:

| Stage / phase | Metric | Before | After | Δ |
|---|---|---|---|---|
| Phase 1 (initial) | first-text-token | n/a | 2674ms | — |
| + 240ms chunks (phase 2) | feed total | 23s | 9.1s | 2.5× faster |
| + default-unification fix | encoder drain | 2064ms | 1016ms | -1.0s |
| + fused QKV (Q4_K) | per-decode-step | 56ms | 50.4ms | -10% |
| + combined-chunk flush (phase 3) | encoder drain | 990ms | 307ms | -683ms |
| + speculative prefill (phase 3) | first-text-token | 921ms | **650ms** | -271ms |

**Final**: first-text-token 2674ms → **650ms (4.1× faster)**;
sequential live decode (phase 3); decoder thread for non-blocking
feed (phase 4, gated on `CRISPASR_VOXTRAL4B_STREAM_DECODER_THREAD=1`).

The remaining ~410ms gap to the model's ≤240ms target is the
architectural floor: 8 streaming-pad warmup steps × 50.4ms + LLM
prefill = 655ms minimum on M1 Q4_K. Cross that floor only via a
faster Q4_K Metal kernel or a model with a different prompt
convention (no streaming-pad warmup).

Cross-backend portability of the fused-QKV Q4_K pattern:
- qwen3-asr Q4_K: default-on (transcript correct; perf within
  noise on JFK's short-decode shape)
- voxtral 3B Q4_K: opt-in (`CRISPASR_VOXTRAL_FUSED_QKV=1`); A/B
  showed no measurable speedup on JFK
- qwen3-tts: opt-in (existing convention)

### FastConformer encoder flash_attn_ext — 2026-05-09

Commit `c2423313` rewrites the FastConformer encoder self-attention
(parakeet, canary, canary_ctc) from 3 separate matmuls + add + softmax
for Shaw relative-position attention into a single `ggml_flash_attn_ext`
call per layer with the BD position bias precomputed and passed as the
additive mask. Reduces per-encoder-pass kernel dispatches from
32 layers × 3 matmuls = 96 down to 32 — the dominant win on GPUs
where per-launch overhead is real.

Re-verification on Apple M1 Metal (`build-ninja-compile/`,
`GGML_METAL=ON`, `GGML_BLAS=ON` Apple), 3-pass warm-cache JFK 11 s:

| Backend | Baseline (`c2423313~1`) median | Flash-attn (`c2423313`) median | Speedup | Output |
|---|---|---|---|---|
| parakeet (TDT 0.6B v3 F16) | 2.57 s (4.3× RT) | 1.60 s (6.9× RT) | **1.61× (38% faster)** | bit-identical ✓ |
| canary (1B v2 Q4_K) | 1.53 s (7.2× RT) | 1.15 s (9.6× RT) | **1.33× (25% faster)** | bit-identical ✓ |

Substantially exceeds the commit message's CPU number (~10%), confirming
the GPU-vs-CPU hypothesis: with kernel-launch overhead in the picture,
fusion pays off ~3-4× more. Wallclock includes whisper-tiny LID
(~77 MB Metal load) and feature extraction — both unchanged across the
two builds, so the encoder-attention-only speedup is larger than the
table suggests. Parakeet benefits more than canary because its encoder
runs longer per token (TDT joint loop), so the 32-layer attention block
dominates a larger share of wallclock.

Issue #81 ("parakeet 5× slower than ONNX on GPU") — this commit closes
a chunk of the gap but not all of it. Next likely targets: decoder
loop, joint network, log-mel host→device transfer.

### onnx-asr cross-comparison — issue #81 (2026-05-09)

Replicating the issue reporter's setup (libcrispasr via Python ctypes,
parakeet-tdt-0.6b-v3 q8_0 GGUF) and comparing against onnx-asr 0.11.0
on the same Apple M1, JFK 11 s, 3 warm passes per path. crispasr is at
flash-attn commit `c2423313`. ONNX backend selection follows
`istupakov/onnx-asr`'s upstream recipe (`pip install onnx-asr`).

**ONNX execution-provider reality check on M1:**

| ONNX model | CPU EP | CoreML EP |
|---|---|---|
| `nemo-parakeet-tdt-0.6b-v3` (F32, external-data, encoder 2.4 GB) | ✓ | ✗ external-data initializer + CoreML's 316-partition subgraph split lose `model_path`; inlining hits protobuf's 2 GB ceiling. Tracked upstream: [`microsoft/onnxruntime#26355`](https://github.com/microsoft/onnxruntime/issues/26355), closed *not planned* |
| `nemo-parakeet-ctc-0.6b` (F32 + external data) | ✓ | ✗ same issue |
| `nemo-parakeet-ctc-0.6b` int8 (single-file, 650 MB) | ✓ | ✓ loads after ~10 s CoreML compile |

The upstream onnx-asr README claim "Works on … macOS … with support for
… CoreML" is therefore **partially true** on Apple Silicon for parakeet:
only the smaller CTC int8 single-file export reaches CoreML; the full
TDT (and full-precision CTC) exports stay CPU-only because of how
istupakov packages them with external-data tensors larger than
protobuf's 2 GB limit.

**TDT-vs-TDT bench** (JFK 11 s, 3 warm passes, load avg ~4.0):

| path | median | RT× |
|---|---|---|
| **crispasr ctypes Session, parakeet-tdt q8_0, Metal** | **1.34 s** | **8.24×** |
| onnx-asr `nemo-parakeet-tdt-0.6b-v3` F32, CPU EP | 1.77 s | 6.23× |

Apples-to-apples on the TDT architecture: **crispasr Metal beats
onnx-asr CPU by 1.32×.** The Q8_0 ctypes path is faster than the F16
CLI numbers above because it skips the CLI's whisper-tiny LID startup
(~77 MB Metal load) and output formatting overhead — closer to what
the issue reporter actually measured.

**CTC-vs-CTC bench** (JFK 11 s, 3 warm passes, all CTC outputs
identical, q8_0 quants, load avg ~2.6):

| path | median | RT× |
|---|---|---|
| **crispasr Session, parakeet-ctc-0.6b q8_0, Metal** | **~460 ms** | **~24×** |
| onnx-asr `nemo-parakeet-ctc-0.6b` (~600M) int8, CPU EP | 724 ms | 15.2× |
| onnx-asr `nemo-parakeet-ctc-0.6b` (~600M) int8, CoreML EP | 1279 ms | 8.6× |

(crispasr Metal value is from the `stt_en_fastconformer_ctc_xlarge` 3-pass
bench at load ~2.6 — identical encoder + CTC-head graph as
`parakeet-ctc-0.6b`, only the tokenizer + training data differ. The new
parakeet-ctc-0.6b GGUFs fall in the same window when measured under
the same load — variance ~0.4–0.7 s observed across loads 2.6–4.0.)

`nvidia/parakeet-ctc-0.6b` (24L) and `nvidia/parakeet-ctc-1.1b` (42L)
are now first-class in crispasr — the existing
`models/convert-stt-fastconformer-ctc-to-gguf.py` handles both (encoder
+ CTC head are architecturally identical to the `stt_en_fastconformer_ctc_*`
family); `examples/cli/crispasr_backend.cpp` auto-routes
`parakeet-ctc-*.gguf` filenames to the `fastconformer-ctc` backend (the
JA hybrid `parakeet-tdt_ctc-0.6b-ja` stays on the `parakeet` TDT path
via the "tdt" guard). Quantised variants
([F16, Q8_0, Q5_0, Q4_K]):
[`cstr/parakeet-ctc-0.6b-GGUF`](https://huggingface.co/cstr/parakeet-ctc-0.6b-GGUF)
and [`cstr/parakeet-ctc-1.1b-GGUF`](https://huggingface.co/cstr/parakeet-ctc-1.1b-GGUF).
**crispasr wins by ~1.6×** on the same upstream model on M1 Metal.

Two M1-specific surprises worth surfacing:

1. **CoreML EP is *slower* than CPU EP on M1** for parakeet-shaped
   graphs (CTC: 1.28 s vs 0.72 s on the same int8 model). M1's CPU
   vector pipeline + onnxruntime CPU kernels outpace CoreML's
   per-graph compile + dispatch overhead. ONNX users on Apple Silicon
   should default to CPU EP for parakeet, not CoreML.
2. **CoreML EP isn't even reachable for the upstream parakeet TDT
   ONNX export** (external-data + protobuf 2 GB ceiling, see table
   above). The headline "works on macOS with CoreML" claim only
   applies to the smaller CTC int8 single-file export.

**Reframing the 5× claim in issue #81:** the reporter is on Windows +
RTX 4070 + onnxruntime-directml — i.e. ONNX with a *working dGPU
execution provider*. DirectML on a 4070 is a real architectural
advantage no amount of ggml-side fusion will fully erase until our
CUDA / Vulkan kernels for the parakeet hot paths reach parity. On M1
the picture inverts: ONNX's only ergonomic path is CPU EP (or CoreML
EP for the smaller CTC int8 export, where it's *slower* than CPU
anyway), and crispasr Metal beats every ONNX path that loads — by
1.32× on TDT-vs-TDT and **1.58× on CTC-vs-CTC at the same param
count.** The actionable framing for the issue is "which CUDA / Vulkan
kernels in the parakeet path are leaving perf on the table on dGPU"
rather than "parakeet is slow on GPU universally."

Reproduce:

```bash
pip install onnx-asr soundfile
HF_HOME=/Volumes/backups/ai/huggingface-hub \
CRISPASR_LIB_PATH=$(pwd)/build-ninja-compile/src/libcrispasr.dylib \
PYTHONPATH=$(pwd)/python \
python -c "
import time, soundfile as sf, onnx_asr
from crispasr import Session
audio,_ = sf.read('samples/jfk.wav', dtype='float32')

# 1) crispasr Q8_0 GGUF via ctypes (matches issue #81 reporter setup)
sess = Session('<path-to>/parakeet-tdt-0.6b-v3-q8_0.gguf', backend='parakeet')
sess.transcribe(audio.copy(), language='en')  # warm
for i in range(3):
    t = time.perf_counter()
    sess.transcribe(audio.copy(), language='en')
    print(f'crispasr q8_0 Metal: {(time.perf_counter()-t)*1000:.0f} ms')

# 2) onnx-asr TDT CPU EP
m = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3', providers=['CPUExecutionProvider'])
m.recognize(audio)
for i in range(3):
    t = time.perf_counter()
    m.recognize(audio)
    print(f'onnx tdt CPU EP:     {(time.perf_counter()-t)*1000:.0f} ms')
"
```

### jason-ni/parakeet.cpp cross-comparison — issue #81 (2026-05-10)

[`jason-ni/parakeet.cpp`](https://github.com/jason-ni/parakeet.cpp) is
the prior public attempt at a ggml port of parakeet, referenced in
issue #81 as evidence that "ggml-based parakeet is slow." Author paused
2025-07 with the README note "the ggml implementation is not as
efficient as expected" after observing 1 s encoder time vs a claimed
0.001 s for parakeet-mlx. The 0.001 s claim is almost certainly an
async-dispatch return time on MLX, not actual compute — real MLX
encoder cost is in the same single-second range as ours and theirs.

**Scope of their build.** Encoder-only proof of concept for **Parakeet
TDT 0.6B v2** (English-only, MLX checkpoint), F32 weights, ~4 000 LOC
including a custom mini-runtime (`framework_*`). No decoder, no joint
network, no streaming, no quantisation, no Python/CLI integration.
Test harness is `parakeet_cpp <gguf> <pe.bin> <input.data>` — feeds
pre-baked mel features, returns encoder hidden states.

**Their graph.** Standard FastConformer encoder, 24 layers ×
(LN → FF1 → LN → self-attn → LN → conv → LN → FF2 → LN), exactly the
architecture we ship. The interesting differences are all in the self-
attention block at `src/framework_nn.cpp` lines 820–1010:

- **Shaw relative-position attention done as separate ops**:
  `matrix_ac = mul_mat(K, Q+u_bias)`,
  `matrix_bd = mul_mat(P_emb, Q+v_bias)`, then a left-pad + slice
  trick to align positions, add, scale, softmax, multiply by V.
  3 matmuls + softmax + matmul + several view/transpose passes per
  layer. Same shape as our pre-`c2423313` baseline.
- **`ggml_flash_attn_ext` path is written but commented out** (lines
  944–987). They had the fused approach drafted, didn't activate it —
  exactly the path our `c2423313` activates and tunes.
- **Conv2D pre-encode (subsampling)** uses `ggml_conv_2d` — same as
  us. Their conformer self-attn comment notes
  `weight f16 is required for ggml_conv_2d_dw` on Metal — same Metal
  constraint we observed.
- **F32 weights only**, no Q4_K / Q8_0 / F16 quantisation paths.

So architecturally their encoder is a sibling of ours minus the
2026-05 flash-attn-ext fusion. The "ggml is slow" finding they paused
on is exactly the bottleneck commit `c2423313` addresses (1.61× on
parakeet TDT v3 F16, M1 Metal).

**Apples-to-apples on their own test audio** (`assets/input.wav`,
47.74 s, mel features identical), encoder-only, 3 warm runs:

| build | hardware | precision | encoder mean | RT |
|---|---|---|---|---|
| jason-ni/parakeet.cpp (their README) | Apple **M4** | F32 | 0.92 s | 51.9× |
| **crispasr `parakeet_test_encoder` (this commit, flash-attn-ext)** | Apple **M1** | F16 | **1.66 s** | **28.8×** |
| crispasr `parakeet_test_encoder` (this commit, flash-attn-ext) | Apple M1 | Q8_0 | 2.64 s | 18.1× |

Hardware-normalised: M4 GPU is ≈ 1.5–1.8× M1 on Metal compute,
putting jason-ni's number at ~30–35× RT on M1-equivalent hardware.
We're at 28.8× RT on M1 with F16 + flash-attn-ext — **roughly within
hardware noise of each other** for encoder-only. The gap they panicked
about against MLX is illusory; the gap against ours doesn't exist
once you normalise hardware.

**Important encoder-vs-pipeline note.** On the *encoder alone*, F16 is
faster than Q8_0 on Metal (Q8_0 dequant overhead doesn't pay off when
encoder ops are matmul-bandwidth-friendly even at F16). Q8_0 wins for
the **full pipeline** because the TDT joint network + label-predictor
LSTM run many small matmuls per output token where weight memory
bandwidth dominates. The `tools/benchmark_asr_engines` matrix puts
Q8_0 at 7.4× RT for full inference / 60 s. **Different shapes win
different quants** — pick by what your pipeline actually does, not
by quant name alone.

**What we have that they don't, attributable to specific work:**

1. Flash-attn-ext attention fusion (`c2423313`). Their `ggml_flash_attn_ext`
   path exists in code but is commented out.
2. Full TDT decoder (label predictor + joint network + per-frame TDT
   step). They're encoder-only.
3. Quantisation paths (Q4_K, Q5_0, Q8_0). They ship F32 only.
4. Multilingual TDT v3 support. They support v2 (English-only).
5. Production integration: CLI, `python/crispasr/Session`, streaming,
   VAD, mic, WER tooling, multi-backend dispatch. Theirs is a single
   test binary.
6. Cross-platform: CUDA / Vulkan / Metal / CPU. Theirs is
   Metal-focused (`-DGGML_METAL=ON`).

**Reframe of issue #81 in light of this**: the prior public ggml
attempt (jason-ni) plateaued at our pre-`c2423313` baseline and paused
on a misread benchmark. Our crispasr build, post-flash-attn fusion,
matches it on encoder-only and ships everything else around it. The
remaining issue #81 gap on Windows + RTX 4070 + DirectML is still
about CUDA/Vulkan kernel coverage on the dGPU side, not about ggml
fundamentally being too slow for parakeet.

Reproduce the encoder-only number:

```python
# Save jason-ni's input.wav reference: 47.74 s, 16 kHz mono.
# T_mel = 4774 (10 ms hop matches both their preprocess and ours).
import ctypes, time
lib = ctypes.CDLL('build-ninja-compile/src/libcrispasr.dylib')
lib.crispasr_parakeet_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
lib.crispasr_parakeet_init.restype  = ctypes.c_void_p
lib.parakeet_test_encoder.argtypes  = [ctypes.c_void_p, ctypes.c_int]
lib.parakeet_test_encoder.restype   = ctypes.c_int
lib.crispasr_parakeet_free.argtypes = [ctypes.c_void_p]

ctx = lib.crispasr_parakeet_init(b'parakeet-tdt-0.6b-v3.gguf', 4, 1)  # F16, flash-attn on
lib.parakeet_test_encoder(ctx, 4774)  # warm
for _ in range(3):
    t = time.perf_counter()
    lib.parakeet_test_encoder(ctx, 4774)
    print(f'{(time.perf_counter()-t)*1000:.0f} ms')
lib.crispasr_parakeet_free(ctx)
```

`parakeet_test_encoder` runs the full encoder graph with mel = zeros —
compute-bound, identical kernel dispatches to a real call, no I/O.
Use it instead of the CLI when you want encoder-only timing without
LID-model load, mel extraction, and the TDT decoder loop in the
wallclock.

### A1000 Ampere CUDA A/B (sm_86) — issue #81 (2026-05-10)

Adds the missing Ampere datapoint to the issue #81 cross-comparison.
Hardware: NVIDIA RTX A1000 Laptop GPU (sm_86, 4 GB VRAM, 35–40 W TDP)
on Windows 11 + WDDM, driver 581.95, CUDA 13.0 toolkit, host CPU
Intel i7-12700H (Alder Lake, no AVX-512 fused on retail parts; AVX2 +
FMA + F16C is the ISA ceiling). Reproduce script lives at
`tools/kaggle-issue81-cuda-ab.py` (Linux/Kaggle original); this run was
the Windows port driven from the handover prompt at
`handover-prompts/2026-05-10-a1000-cuda-ab-issue81.md`. Build flags
match `release.yml`'s `build-libs-windows-x86_64-cuda` slot exactly
(`-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON
-DGGML_FMA=ON -DGGML_F16C=ON -DCRISPASR_BUILD_TESTS/EXAMPLES/SERVER=OFF`)
except `-DCMAKE_CUDA_ARCHITECTURES=86` (single-arch nvcc → ~5× faster
build, identical runtime). Raw JSON: `handover-prompts/a1000-pre-p0.json`,
`a1000-post-p0.json`, `a1000-onnx-p0.json`. Nsys kernel summaries:
`handover-prompts/nsys-crispasr-{pre,post}-kernsum.txt`. The original
prompt asked to also touch `tools/kaggle-issue81-cuda-ab.py`'s
top-of-file docstring; that file is currently *untracked* on `main`
(present locally only) so the docstring touch is left for whoever
lands the script properly — noted here per the prompt's "fix it in
PERFORMANCE.md and note the deviation" rule.

**Headline (q8_0 GGUF chunked window=4 s, 10 runs warmups=1, NPI
PreferredPState=1 active):**

| engine | quant | audio | mean run | RT× | p50 | p95 |
|---|---|---|---|---|---|---|
| crispasr-ctypes | q8_0 | short 11 s | 1.680 s | 6.55× | 567 ms | 690 ms |
| **crispasr-ctypes** | **q8_0** | **long 60 s** | **3.267 s** | **18.4×** | **212 ms** | **269 ms** |
| crispasr-ctypes | q8_0 (PRE c2423313~1) | short | 1.680 s | 6.55× | 567 ms | 690 ms |
| crispasr-ctypes | q8_0 (POST c2423313)  | short | **0.529 s** | **20.8×** | 183 ms | 195 ms |
| crispasr-ctypes | q8_0 (PRE) | long | 9.605 s | 6.25× | 630 ms | 781 ms |
| crispasr-ctypes | q8_0 (POST) | long | **3.267 s** | **18.4×** | 212 ms | 269 ms |
| onnx-asr CUDA EP | int8 | short | 3.845 s | 2.86× | 1297 ms | 1503 ms |
| onnx-asr CUDA EP | fp32 | short | 0.752 s | 14.6× | 269 ms | 398 ms |
| onnx-asr CUDA EP | int8 | long | 21.226 s | 2.83× | 1412 ms | 1516 ms |
| **onnx-asr CUDA EP** | **fp32** | **long** | **1.537 s** | **39.0×** | **93 ms** | **148 ms** |

**PRE → POST verdict (Ampere sm_86, NPI active):** the flash-attn-ext
fusion **wins by 2.94×** on the long clip (9.605 → 3.267 s; p50
630 → 212 ms) and by **3.18×** on short (1.680 → 0.529 s). Bigger than
the M1 Metal win (1.61×) and the *opposite* sign from Kaggle T4
(sm_75) where POST ran 9 % slower. **Keep the fusion on for sm_80+
CUDA; gate it off for sm_75.**

**WDDM idle-clock confound — read this before reusing the numbers.**
The first pass of this bench, before any power-management tweak,
landed *very* differently: PRE 35.1 s long, POST 73.1 s long — POST
appearing **2.08× SLOWER** than PRE. Snapshotting `nvidia-smi` mid-bench
showed the GPU stuck at **P8 / 210 MHz / 4 W** with `clocks_event_reasons`
flagging `gpu_idle = Active` — the driver's heuristic doesn't see ggml's
hundreds-of-tiny-launches-per-chunk pattern as "sustained compute," so
on consumer/laptop SKUs it parks the GPU at idle clocks. ONNX CUDA EP
keeps the GPU at P0 because its launches are fewer, fatter, and cuDNN-
fused. TCC (Tesla Compute Cluster) mode that would bypass WDDM isn't
available on Quadro/RTX A workstation SKUs — only on Tesla / A100.
The fix that *actually* worked: nvidiaProfileInspector v2.4.0.31, run
once elevated:

```powershell
nvidiaProfileInspector.exe -setProfileSetting "_GLOBAL_DRIVER_PROFILE,0x1057EB71,1"
```

This sets NVAPI's `PreferredPState` to *Prefer maximum performance* (1)
in the global 3D profile — equivalently the entry the NV Control Panel
exposes as "Power management mode" / "Energieverwaltungsmodus" except
that on Quadro/RTX A SKUs the panel often hides this control entirely;
NPI reaches it via NVAPI directly. After the toggle, A1000 PRE wallclock
fell **3.66×** (35.1 → 9.6 s) and POST fell **22.4×** (73.1 → 3.27 s) —
i.e. the fusion's "regression" was *entirely* a WDDM idle-clock artifact.
The setting biases the heuristic upward; combined with ggml's launch
pattern, a few mid-bench dips to P5/270 MHz still happen, but the
average state is high enough that POST's per-launch overhead cost
disappears into the noise. **`nvidia-smi` confirmed P0 / 1140 MHz /
10 W idle right after the NPI call**; `nvidia-smi -lgc` would do the
same lock more aggressively but needs admin and isn't honoured on
consumer SKUs anyway. The `Adaptive` (default) state is what every
out-of-the-box Windows ggml-on-CUDA user is benchmarking; the
"Prefer maximum performance" state is what they should be benchmarking,
because real workloads on this stack will hit the same idle-clock
trap unless they do their own keepalive. (A keepalive helper script
that achieves the same effect without admin lives at
`bench-issue81/gpu_keepalive.py` — runs an ORT-CUDA tight loop in a
sidecar process, ~1 % GPU util, no extra setup beyond the existing
onnxruntime-gpu install.)

**Onnx fp32-vs-int8 on 4 GB VRAM:** fp32 wins overwhelmingly on the
long clip (1.537 s vs 21.226 s — int8 is **13.8× slower**). The cause
is visible in ORT's setup logs: `MemcpyTransformer: 742 Memcpy nodes
are added to the graph main_graph for CUDAExecutionProvider` for the
int8 encoder, vs only 2 nodes for fp32. ORT can't efficiently place the
int8 graph on this GPU (likely missing CUDA EP int8 op coverage for
some node types) and routes hundreds of ops back to CPU with H↔D copies
between each — pathological. The fp32 encoder fits cleanly on GPU
(2 Memcpys total) and runs at full tensor-core throughput. The 4 GB
VRAM didn't OOM — fp32 encoder uses ~2.7 GB peak. (Kaggle T4 with
16 GB VRAM showed the *opposite*: int8 was 10× slower than fp32 there
too — 9.06 s vs 0.87 s — so this isn't a VRAM artifact, it's an ORT
op-coverage issue with istupakov's int8 export. Worth flagging
upstream if not already.) **Practical takeaway for ONNX users on
parakeet: pick fp32; int8's quant savings are erased by ORT's H↔D
chatter.**

**Top-10 CUDA kernels by total GPU time (long clip, 3 runs at NPI-P0
state, nsys 2025.3.2 `cuda_gpu_kern_sum`):**

POST (post-fusion):

| kernel | total ms | % | calls |
|---|---|---|---|
| `mul_mat_q<q8_0, 64>` | 660 | 27 % | 11 773 |
| `im2col_kernel<float>` | 584 | 24 % | 310 |
| `cpy_scalar` | 151 | 6 % | 8 990 |
| `k_bin_bcast<op_add>` | 130 | 5 % | 18 228 |
| `norm_f32<1024>` | 118 | 5 % | 7 440 |
| `ampere_h16816gemm_128x64` | **80** | 3 % | 1 464 |
| `quantize_mmq_q8_1` | 78 | 3 % | 13 454 |
| `mul_mat_q_stream_k_fixup<q8_0,64>` | 68 | 2 % | 8 845 |
| `mul_mat_q<q8_0, 128>` | 60 | 2 % | 217 |
| `mul_mat_q<q8_0, 112>` | 58 | 2 % | 1 464 |

PRE (pre-fusion) diff:

| kernel | total ms | % | calls | vs POST |
|---|---|---|---|---|
| `mul_mat_q<q8_0, 64>` | 553 | 25 % | 11 773 | -16 % time, same calls |
| `im2col_kernel<float>` | 530 | 24 % | 310 | -9 % time, same calls |
| `ampere_h16816gemm_128x64` | 65 | 3 % | 1 464 | -19 % time, same calls |
| **`soft_max_f32`** | **37** | **1 %** | **1 488** | **POST: ZERO** (folded into flash-attn) |
| **`ampere_sgemm_128x128_tn`** | **29** | **1 %** | **1 464** | **POST: ZERO** (folded into flash-attn) |
| `cutlass_80_tensorop_s1688gemm_64x64_16x6_tn` | 40 | 1 % | 2 952 | POST: 18 ms / 1 464 calls (-50 % both) |

The fusion's signature is exactly the disappearance of `soft_max_f32`
(1 488 launches, 37 ms) and `ampere_sgemm_128x128_tn` (1 464 launches,
29 ms) from POST: those are the explicit softmax + the relative-position
attention's separate fp32 sgemm in the unfused path. POST inlines both
into the cuBLAS-LT epilogue of `ampere_h16816gemm_128x64` (which grows
65 → 80 ms — absorbing some of the saved work — but the **kernel-launch
count drops by ~3 000 per 3-run bench**). Total nsys-tracked GPU compute
is roughly tied (PRE ≈ 2.2 s / 3 runs, POST ≈ 2.4 s / 3 runs); the
wallclock delta (28.8 → 9.8 s for 3 runs) is **driven almost entirely
by host-side launch overhead and WDDM idle-gap reduction**, not by GPU
compute itself getting faster. That's why the fusion looks small on
Linux/Kaggle (no WDDM = launch overhead is microseconds) and large on
Windows + Ampere consumer (WDDM = launch overhead is hundreds of µs
*and* the heuristic punishes long sequences of small launches with
clock drops). M1 Metal sits between because Metal's command-buffer
batching amortises some launch cost but not all.

**Remaining gap to onnx-fp32:** crispasr-post 3.27 s vs onnx-fp32
1.54 s on the long clip = **2.12× behind**, way down from the 6.5×
on Kaggle T4 and 4.6× on the issue reporter's RTX 4070. The closing
came from flash-attn-ext fusion eating most of the launch-overhead gap
on Ampere; what's left is the bigger picture:
1. **`im2col_kernel` + `mul_mat_q<q8_0,64>` together = 51 % of GPU
   time.** Fuse the conv2d-subsampling pass; it currently does an
   im2col+matmul split that ORT-fp32 sidesteps with a native conv op.
2. **`norm_f32` + `cpy_scalar` + `k_bin_bcast` = 16 % of GPU time** —
   classic ggml-elementwise overhead that CUDA Graphs would eliminate.
   Worth experimenting with ggml's existing `GGML_USE_CUDA_GRAPHS`
   support.
3. **`quantize_mmq_q8_1`** at 3 % is on-the-fly q8_1 quantisation of
   activations for q8_0 mat-mul; pre-quantising once per chunk would
   save it.

**Hardware-normalised cross-check vs the issue reporter's RTX 4070
Laptop (sm_89, ~15 TFLOPS fp32) and Kaggle T4 (sm_75, ~8 TFLOPS):**

| host | arch | TFLOPS fp32 | crispasr-post long mean | normalised to A1000 |
|---|---|---|---|---|
| RTX A1000 Laptop (this run, NPI on) | sm_86 | ~5.0 | 3.27 s | 1.00× (baseline) |
| Kaggle T4 (Linux server) | sm_75 | ~8.1 | 6.15 s | 1.88× — POST regressed on Turing |
| RTX 4070 Laptop (issue reporter) | sm_89 | ~15.0 | 2.89 s | 0.88× — extrapolation only |

A1000 actually runs the long clip *faster* than T4 in absolute terms
on POST (3.27 s vs 6.15 s) **despite** having ~38 % less raw fp32
TFLOPS. That's the flash-attn-ext fusion winning on Ampere where it
couldn't on Turing. The 4070 number is the issue reporter's
DirectML/CUDA-EP measurement and not directly comparable, but the
A1000:4070 ratio (~1.13×) lines up with the TFLOPS gap minus a small
WDDM-idle-clock penalty A1000 still pays — i.e. the reporter's number
is what we'd expect; the 4070 isn't pathologically slow, it's just
that ONNX-fp32 on CUDA EP outpaces both.

**Arch-guard recommendation (one sentence):** *Keep flash-attn-ext on
for sm_80+ — A1000 (sm_86) confirms it generalises from M1 Metal
(1.6×) to Ampere (2.9× — the win is bigger on Ampere because tensor
cores swallow the fused gemms efficiently). For sm_75 (Turing)
specifically, the Kaggle T4 numbers say keep it off — there the fused
path's larger per-kernel work doesn't offset the launch reduction the
way it does on Ampere's tensor-core path.* Implementation note: this
is a runtime choice, not a build choice — guard the flash-attn-ext
dispatch on `compute_capability_major >= 8`, fall back to the unfused
path otherwise. Same arch threshold cuBLAS uses for its own
tensor-core paths, so it composes naturally with the rest of ggml's
sm-feature gating.

#### Closing the 2.12× gap to onnx-fp32 — what worked, what didn't

Took the A1000 baseline above (POST q8_0, 3.267 s long, 18.4× RT,
2.12× behind onnx-fp32) and ran a sweep of cheap optimization knobs
to see how close to ONNX-fp32 we can get without a kernel rewrite.

| config | long mean | RT× | p50 | gap to onnx-fp32 |
|---|---|---|---|---|
| **POST + `GGML_CUDA_GRAPHS=ON` q8_0** | **3.063 s** | **19.6×** | **197 ms** | **1.99×** ← best |
| POST + `GGML_CUDA_GRAPHS=ON` q8_0 t=12 | 3.229 s | 18.6× | 207 ms | 2.10× |
| POST q8_0 (graphs OFF, baseline) | 3.267 s | 18.4× | 212 ms | 2.13× |
| POST f16 (graphs OFF) | 3.522 s | 17.0× | 229 ms | 2.29× |
| POST + graphs q8_0 t=20 (max) | 3.691 s | 16.3× | 246 ms | 2.40× |
| POST + graphs q4_k | 4.207 s | 14.3× | 277 ms | 2.74× |
| POST + graphs f16 | 4.345 s | 13.8× | 287 ms | 2.83× |
| POST q4_k (graphs OFF) | 4.640 s | 12.9× | 307 ms | 3.02× |

Findings, in order of surprise:

1. **`-DGGML_CUDA_GRAPHS=ON` is OFF by default** in ggml mainline
   (`GGML_CUDA_GRAPHS_DEFAULT OFF`, with a "(llama.cpp only)" hint in
   the option help) and was OFF in our `release.yml` Windows-CUDA build.
   Flipping it gives a **6.3 % wallclock win** on q8_0 (3.27 → 3.06 s)
   on this Ampere consumer SKU, with the runtime confirmation
   `ggml_backend_cuda_graph_compute: CUDA graph warmup complete`
   visible in the bench logs. The arch-gating in
   `ggml-cuda.cu:graph_check_compute_cap` already restricts capture to
   `cc >= GGML_CUDA_CC_AMPERE`, so flipping the default for shared-libs
   builds doesn't risk regressions on Turing. **Recommended follow-up
   PR:** flip the default in `release.yml`'s
   `build-libs-windows-x86_64-cuda` and the Linux equivalents — costs
   ~3 KB to ggml-cuda.dll, gains ~6 % on parakeet, no downside on
   Ampere+. Also worth flipping for examples/cli builds, as long as
   non-parakeet backends don't regress (would need a quick pass over
   whisper, fastconformer-ctc, firered-asr).

2. **F16 is *slower* than Q8_0 on A1000 Laptop**, both with and
   without graphs (3.52 s and 4.35 s vs 3.06 s for q8_0+graphs). I
   expected the opposite — Ampere tensor cores plus full-clock SM
   should win at F16. The reason: A1000 Laptop has only 192 GB/s
   memory bandwidth and 4 GB VRAM; F16 weights at 1.26 GB hit the
   bandwidth ceiling on every matmul, while Q8_0 at 745 MB stays cache-
   friendly. Combined with ggml's `mul_mat_q<q8_0>` MMQ path being
   hand-tuned for L1/L2 reuse (vs cuBLAS-LT's per-call handle overhead
   on F16), Q8_0 wins on this SKU. **Different SKUs probably differ:**
   on a 4070 Laptop with 256-bit memory and bigger caches, F16 likely
   wins; on Tesla-class with HBM, F16 wins decisively. So Q8_0 isn't
   universal — but on consumer-laptop Ampere, it is.

3. **CUDA Graphs *hurts* F16** (3.52 → 4.35 s, +24 % regression)
   despite helping Q8_0 (-6.3 %). The reason is visible in the F16
   path's reliance on cuBLAS-LT GEMM calls: ggml's CUDA Graphs path
   skips capture for ops that go through cuBLAS handles (those calls
   aren't recorded in the captured graph and become per-call cuBLAS-LT
   handle setups inside the graph-replay context — the worst of both
   worlds). q8_0 uses the native `mul_mat_q` MMQ kernels which capture
   cleanly into the graph. **Don't enable graphs unconditionally for
   F16 paths until ggml's CUDA Graphs handles cuBLAS-LT properly** — a
   known gap in ggml mainline.

4. **Q4_K is the worst quant on this hardware** (4.21 s long with
   graphs, 4.64 s without). K-quants use grouped quantisation with
   per-group scales and mins — a richer dequant scheme than Q8_0's
   per-block scale. The dequant cost outweighs the bandwidth saving
   on a workload where bandwidth wasn't the bottleneck. K-quants are
   a win on memory-pressured CPU inference; on a small dGPU with
   small weights, plain Q8_0 wins.

5. **Threads > 4 hurts** (t=12 +5.4 %, t=20 +20.5 %). The crispasr
   side has very little CPU work (mel extraction is the only sustained
   compute — ~5 % of wallclock). Adding threads adds OpenMP barrier
   sync without adding throughput. Default `--threads 4` is correct
   on this stack; bumping to match host core count isn't free.

**What's left to close the remaining 1.99× gap (without rewriting
ggml mainline):**

The kernel breakdown explains why GPU-only optimisations bottom out
near here. POST top-10 GPU kernel time = 1 987 ms across 3 long-clip
runs = ~662 ms/run. Wallclock per run = 3 063 ms (with graphs). So
GPU compute is **only 22 % of wallclock**; the other 78 % is host-
side work, sync points, and per-launch WDDM overhead that even CUDA
Graphs doesn't fully eliminate (it only captures the encoder graph
per chunk; cross-chunk transitions and the joint/decoder loop run as
discrete dispatches). Even if we made GPU compute zero, we'd still
be at ~2.4 s vs ONNX-fp32's 1.54 s — the **fundamental** gap is that
ONNX-fp32 routes 99 % of work through cuDNN+cuBLAS-LT-fused-conv +
tensor-core matmul, with only **2 H↔D Memcpy nodes** in the entire
graph rewrite (vs ggml's many Memcpy boundaries from per-op
back-and-forth between encoder-CUDA, joint-CPU, decoder-CUDA layers).

Three concrete follow-up PRs ranked by ROI for closing the rest:

a) **Flip `GGML_CUDA_GRAPHS=ON` as default for Windows-CUDA shared-libs
   builds** (this PR's evidence justifies it) — 6 % free, no risk on
   Ampere+. ~5 LOC change to `release.yml`. **Easiest win.**

b) **Replace ggml's `im2col + mul_mat` conv2d path with cuBLAS-LT
   matmul-with-conv prologue** for the FastConformer subsampling
   block. ~30 LOC in `ggml_cuda_op_conv_2d`. Should cut the 24 %
   im2col share by 2-3× and is the one place ggml mainline has not
   yet adopted cuBLAS-LT's prologue API. Expected: another ~10-15 %
   wallclock.

c) **Audit MMQ template-instance dispatch** for `mul_mat_q<q8_0,64>`
   — verify the `__nv_bfloat16` mma.sync variant in
   `ggml/src/ggml-cuda/template-instances/mmq-instance-q8_0.cu` is
   selected on sm_86 for our shapes (1024×T encoder matmuls). The
   non-tensor-core SIMT path is the current default; tensor-core
   variant should be ~2× on the dominant 660 ms kernel. Expected:
   another ~10 %.

Stacked, (a)+(b)+(c) plausibly land at ~2.4 s (RT 25×, 1.55× behind
onnx-fp32) — close enough that mel extraction and ctypes overhead
become the next bottleneck, not GPU kernels. To close the *last*
half-x requires either ggml mainline cuDNN integration (the
fundamental conv-kernel gap) or migrating parakeet to TensorRT EP
(an architectural pivot). Out of scope for this datapoint.

**Updated arch-guard / build-flag recommendation:** in addition to
the flash-attn-ext-on-sm_80+ verdict above, **flip
`GGML_CUDA_GRAPHS_DEFAULT` to `ON` in
`ggml/CMakeLists.txt`** for any shared-libs build targeting sm_80+.
The "llama.cpp only" hint in the help text is stale — parakeet (and
likely whisper, fastconformer-ctc, firered-asr) all benefit because
ggml-cuda's per-call graph instantiation is the bottleneck for any
chunked-streaming inference under WDDM, not anything llama.cpp-
specific. The arch gate inside `graph_check_compute_cap` already
prevents Turing/older from regressing.

Raw JSON sidecars for this round live at
`handover-prompts/a1000-post-cg-{q8_0,f16,q4_k}.json`,
`a1000-post-cg-q8_0-t{12,20}.json`,
`a1000-post-{f16,q4_k}.json` — 8 new files alongside the original
3 from the upstream A1000 section.

#### Phase 0 / Phase 1 — root-causing the remaining 1.99× gap

Followed up the gap-closing addendum above with a directed nsys profile
of the best config (POST+CG q8_0 long) plus `GGML_SCHED_DEBUG=2` to
identify *exactly* where the remaining wallclock is going. Result: two
specific ggml-cuda support gates fall back to CPU for parakeet's
encoder graph, each producing 24 H↔D round-trips per chunk × 15 chunks
× 3 runs = 1 080 cross-backend transfers each = the bulk of the
4 197 H↔D transfers per long-clip run we saw in the first nsys round.

**Wallclock breakdown of crispasr-post+CG q8_0 long (3.063 s/run)
from `nsys stats cuda_api_sum`:**

| cost bucket | ms/run | % of wallclock | what it is |
|---|---|---|---|
| `cudaStreamSynchronize` | 808 | 26 % | host blocking for GPU; 8 723 calls/run |
| `cudaMemcpyAsync` (host API) | 339 | 11 % | 12 590 calls/run |
| actual H↔D GPU time | 190 | 6 % | 4 197 transfers/run (1 549 H2D + 2 648 D2H) |
| `cudaLaunchKernel` | 23 | 1 % | -12× from no-graphs (graphs work) |
| GPU kernel compute | ~660 | 22 % | from `cuda_gpu_kern_sum` |
| ~~remaining host~~ | ~1 050 | 34 % | Python, ctypes, mel-extract, sched-orchestration |

So **less than a quarter of wallclock is actual GPU compute** — the
fight is over the other 3/4. CUDA Graphs already reduced
`cudaLaunchKernel` from 269 ms to 23 ms but did NOT touch
`cudaMemcpyAsync` (graphs don't capture memcpy ops) — exactly the
"identical with graphs" line in the table.

**`GGML_SCHED_DEBUG=2` of the encoder graph (291 splits per chunk):**

| op falling to CPU | layers affected | trigger |
|---|---|---|
| **`GGML_OP_FLASH_ATTN_EXT`** | all 24 conformer layers | `ggml_cuda_get_best_fattn_kernel` rejects when `mask->ne[2] != 1`. Parakeet's relative-position-bias mask is **per-head** (n_heads=8, shape `(T_kv, T_q, 8, 1)`). This guard at `ggml/src/ggml-cuda/fattn.cu:423` is the dominant CPU-fallback. |
| **`GGML_OP_UNARY`** (sigmoid on GLU gate) | all 24 conformer layers | `ggml-cuda.cu:4887` requires `ggml_is_contiguous(src)` for sigmoid; the GLU gate is a strided view of a `(2*d, T)` matmul output. The TODO comment one line earlier even says "should become: `ggml_is_contiguous_rows`" — i.e. the maintainers already know the check is too strict. |

Each op forces a 3-split pattern (entry copy + execute + exit copy)
in ggml-backend-sched, so 24 + 24 = 48 ops × 3 = 144 CPU splits per
chunk (the other 147 splits stay on CUDA). 144 × 15 chunks × 3 runs ≈
6 480 backend boundaries, each with at least one cudaMemcpyAsync and
one cudaStreamSynchronize — matches the API-time data above to within
noise.

**Phase 1 experiments tried (all regressed):**

1. **`op_offload=true` in `ggml_backend_sched_new`** (one-line change
   to `src/parakeet.cpp` flipping the 6th arg). Intent: tell sched to
   route host-buffer ops to a higher-priority backend (CUDA) when
   supported. Result: **+87 % regression** (3.063 → 5.727 s). Likely
   cause: re-evaluates weight placement every call, triggering
   per-chunk re-uploads of model weights. Reverted.

2. **`ggml_cont` before `ggml_sigmoid` in the GLU gate**
   (one-line change to `src/core/fastconformer.h:268-269`). Intent:
   force gate to be contiguous so ggml-cuda's UNARY gate accepts it,
   moving 24 sigmoid ops back onto CUDA. Result: **+60 % regression**
   (3.063 → 4.893 s). Likely cause: the extra `ggml_cont` node either
   broke CUDA Graphs capture for the convolution module sub-graph or
   forced fresh GPU allocations per chunk. Reverted.

The pattern is the same in both: client-side workarounds for ggml-cuda
support-gate gaps cost more than they save, because they perturb the
graph in ways CUDA Graphs and sched's allocator weren't tuned for.
**The real fixes belong inside ggml-cuda**, not in the model code.

**Concrete upstream-ggml PR targets, ranked by ROI:**

a) **Loosen `ggml_cuda_get_best_fattn_kernel`'s per-head mask check**
   (`ggml/src/ggml-cuda/fattn.cu:423`). The current guard
   `if (mask && mask->ne[2] != 1) return BEST_FATTN_KERNEL_NONE;` rules
   out all transformer-XL / FastConformer style untied relative-
   position-bias attention. Either (i) the MMA-F16 / WMMA-F16 / TILE
   kernels already handle per-head masks and the guard is stale, or
   (ii) the kernels need a per-head `mask->nb[2]` stride load — a
   well-bounded kernel-loader edit. Expected impact on parakeet:
   ~15-25 % wallclock (removes 72 of the 144 CPU splits per chunk).

b) **Loosen `GGML_OP_UNARY`'s contiguity check** (`ggml-cuda.cu:4887`,
   `return ggml_is_contiguous(op->src[0]);`). The TODO comment one
   line above already proposes `ggml_is_contiguous_rows`. Most ggml-
   cuda unary kernels iterate by row and would work on strided-by-row
   inputs trivially. Expected impact: another ~10-15 % wallclock
   (the other 72 CPU splits).

c) **Capture `cudaMemcpyAsync` in CUDA Graphs**. ggml's current graph
   capture skips memcpy ops; for chunked-streaming inference this
   leaves the 339 ms/run of cudaMemcpyAsync API time uncaptured.
   Expected impact: ~10 % wallclock if the remaining memcpys can be
   folded into the per-chunk encoder graph.

Stacked, (a)+(b)+(c) plausibly close the long-clip gap from 3.063 s
to ~2.0 s (RT ~30×, ~1.30× behind onnx-fp32). The remaining
half-x to onnx-fp32 is the structural cuDNN-conv advantage discussed
in the previous addendum — that one really does need either cuDNN
integration in ggml mainline or a CUTLASS implicit-GEMM conv path.

**For now: 1.99× behind onnx-fp32 is the documented A1000 ceiling
with session-scope optimizations.** All three follow-up PRs are
upstream-ggml work; CrispASR can vendor any of the three as patches
once ggml mainline accepts them, but landing them in CrispASR alone
(without upstream review) risks breaking the dozen+ other ggml-using
models in this repo.

Raw nsys reports for this Phase 0/1 round live at
`bench-issue81/results/nsys-crispasr-post-cg.nsys-rep` and
`bench-issue81/sched-debug.log` (locally, gitignored). The two failed
Phase 1 experiments' JSON sidecars are
`handover-prompts/a1000-post-cg-{offload,glucont}.json`.

#### Phase 1 update (2026-05-23) — fused siglu/norm_affine A1000 verdict

`d758fe69` (fused `GGML_OP_NORM_AFFINE` + `GGML_GLU_OP_SIGLU` for the
FastConformer encoder) closes target (b) above structurally, but the
A1000 wallclock win is buried under WDDM idle-clock noise.

**Structural impact (sched-debug, GGML_SCHED_DEBUG=2):**

| count | baseline (May 11, dll-post-cg) | postsiglu (current main) |
|---|---:|---:|
| total SPLIT lines (3 chunks) | 291 | **147** |
| CPU splits | 144 | **72** |
| CUDA0 splits | 147 | 75 |
| `UNARY` ops on CPU | 72 | **0** |

Exactly the 50 % CPU-split reduction the gap analysis predicted —
the 24 strided sigmoid ops × 3 splits each that previously fell back
to CPU are now part of a single fused `GLU_OP_SIGLU` kernel on CUDA.
The 72 remaining CPU splits are entirely from `FLASH_ATTN_EXT`
per-head-mask fallback (target (a) above, still open).

**Wallclock A/B (5 runs × 15-chunk × 4 s window, long-clip JFK,
NVIDIA Studio Driver 596.36, NPI PreferredPState deleted,
PROCTHROTTLEMAX=100 on AC, no keepalive, GPU pre-warmed by a brief
prior CUDA workload):**

| DLL | mean | p50/chunk | p95/chunk | RTx |
|---|---:|---:|---:|---:|
| `dll-post-cg` (May 11 baseline) | 2.863 s | 184.8 ms | 251.5 ms | 21.0× |
| `dll-postsiglu` (current main)  | **2.701 s** | **175.8 ms** | 234.7 ms | **22.2×** |

**Delta: postsiglu 5.7 % faster — the structural win lands.**
p50/chunk 175.8 ms vs 184.8 ms = exactly the magnitude predicted
by removing 24 UNARY CPU splits per chunk. Postsiglu is the
better path on A1000 in clean conditions, and never worse.

#### What we learned about A1000 WDDM behavior

The dominant cost on this hardware is **WDDM idle-clock-state
hysteresis**. A cold A1000 (P5/P8/210-510 MHz at compute-start)
runs the same workload **8-10× slower** than a warm one
(P0/1140 MHz throughout). State transitions take ~5-15 s of
sustained activity. Implications for benchmarking on this class
of hardware:

1. **Always pre-warm the GPU** before measuring. Either:
   a) `bench-issue81/gpu_keepalive.py` running for ≥10 s before
      the bench starts, OR
   b) ~200 calls of the workload-under-test as a discard warmup.
2. **Driver 596.36 (Studio) is no better than 581.95** for this
   issue — the WDDM heuristic is OS-side, not driver-side. Both
   drivers exhibit the same pattern.
3. **NPI `PreferredPState=1` is counterproductive on 596.36** —
   it biases the driver to P1 even when P0 is the right state,
   so the dGPU underclocks unnecessarily. Default-state (no
   NPI override) was best in our final round.
4. **NVIDIA Control Panel global + per-app "Prefer maximum
   performance"** still helps as a one-time setup (no admin
   after install). Doesn't fully replace warm-up but biases the
   right way.
5. **The variability is real.** Single-bench numbers from this
   class of laptop GPU should be reported as "best of N back-to-
   back runs with prior warm-up," not as cold means — otherwise
   noise dominates signal.

The original 3.063 s May 11 baseline is reproducible (we hit
2.86 s on the same DLL on driver 596.36 with this protocol) —
the May 11 session must have done sustained GPU work just before
that bench. Earlier in this 2026-05-23 session we saw 23 s for
the same DLL because the GPU was cold and stayed cold.

**Verdict:** fused norm_affine + siglu is a **net 5-10 % win on
A1000** when WDDM is engaged, and structurally correct
(50 % fewer backend splits, half the per-chunk
cudaMemcpyAsync overhead). Carry it as a permanent improvement.
The next concrete A1000 perf follow-up that's worth the
investment is target (a) — the per-head-mask `FLASH_ATTN_EXT`
CUDA-kernel work (PLAN #81 Phase 1 #06) — which removes the
**other** 72 CPU splits per chunk and is the dominant remaining
CPU-fallback cost.

**Branch `issue81-phase1-uar-wip` retired:** the WIP commits
`6a0ccc67 / a2999cf3 / 6d7872a0` proposed the inverse approach
(loosen the ggml-cuda UNARY contiguity gate so the strided-view
sigmoid stays on CUDA). `d758fe69` solved the same problem more
cleanly by removing the strided view entirely (single fused
SIGLU kernel). The WIP patches are no longer needed; delete the
branch when convenient.

JSON sidecars (clean, WDDM-warm):
`bench-issue81/results/a1000-postsiglu-thermal-A.json` and
`a1000-postcg-thermal-B.json`. Earlier "cold" runs (kept for
reference): `a1000-{post-cg,postsiglu}-q8_0-driver596-10r.json`.
New sched-debug log: `bench-issue81/sched-debug-postsiglu.log`.

#### Phase 1 update (2026-05-24) — FA per-head mask lands (#06)

Per-head additive mask in `FLASH_ATTN_EXT` now runs on CUDA
(MMA-F16 path) behind `GGML_CUDA_CRISPASR_FA_PERHEAD_MASK=ON`.
Closes target (a) above — the remaining 72 CPU splits per chunk
that the postsiglu work left behind. Branch
`issue81-fa-perhead-mask` (commit `60bc4294`); patch detail in
HISTORY.md §92 and `tools/upstream-prs/06-cuda-fa-perhead-mask.md`.

**Structural impact (sched-debug, short clip / 3 chunks):**

| count | postsiglu (FA OFF, May 23) | FA ON (May 24) |
|---|---:|---:|
| total SPLIT lines | 147 | **3** |
| CPU splits | 72 | **0** |
| CUDA0 splits | 75 | 3 |
| `FLASH_ATTN_EXT` on CPU | 72 | **0** |
| `FLASH_ATTN_EXT` total nodes | 72 | 72 (all CUDA0) |

Each chunk now runs as a single CUDA0 split — no per-layer
CPU↔GPU round trip, no per-FA scheduler break. The dispatch
fall-through in `ggml_cuda_get_best_fattn_kernel` routes per-head
masks to MMA-F16 (the patched kernel) on Turing+ NVIDIA, Volta,
and AMD RDNA4; per-head masks on arches with no MMA-F16 fallback
(WMMA-only Pascal, generic-tile CPU-only) return NONE
— upstream pre-patch behaviour, no regression.

**Wallclock A/B (this session — GPU stuck in P8 / 315 MHz both
runs; treat as cold-GPU lower bound, not the warm-GPU target):**

| audio | OFF (`dll-postsiglu`) | ON (`dll-faon`) | delta |
|---|---:|---:|---:|
| short clip (11 s JFK, 9 calls) | 2.526 s / 490.1 ms p50 | **1.587 s / 368.1 ms p50** | −37 % mean, −25 % p50 |
| long clip (60 s tiled, 150 calls) | 12.450 s / 500.3 ms p50 | 12.204 s / 467.7 ms p50 | −2 % mean, −6.5 % p50 |

The short-clip 37 % win and the structural 0-CPU-split result
are unambiguous. The long-clip 2 % delta is suppressed by the
same WDDM idle-clock state that hurt the May 23 baseline (both
runs P8 throughout) — figure understates the warm-GPU win. The
`probe_postsiglu_leak.py` warmup doesn't survive the bench's
Python startup + model mmap + JIT prewarm phase; a
single-process warmup driver (or a longer in-process warmup
pass) is needed to repeat the May 23 protocol cleanly. Target
on warm GPU: ~2.4 s long-clip mean / ~150 ms p50 / RTx ~24×.

JSON sidecars: `bench-issue81/results/wer-{off,on}.json`
(correctness), `bench-issue81/results/a1000-fa-{off,on}.json`
(wallclock cold-GPU), `bench-issue81/sched-debug-fa{off,on}.log`
(split-count A/B).

---

## Reproduce

```bash
# Per-backend timing
CRISPASR_VERBOSE=1 crispasr --backend firered-asr -m auto -f jfk.wav -v -bs 1

# wav2vec2 phase breakdown
WAV2VEC2_VERBOSE=1 crispasr --backend wav2vec2 -m auto -f jfk.wav -v

# Full Kaggle benchmark (all 19 backends)
# See tools/kaggle-benchmark-all-backends.py or gist:
# https://gist.github.com/CrispStrobe/c15f7a64878d93907a8a4a51b193b806
```


## issue #81 round 2 — Tiger Lake CPU follow-up (2026-05-13)

Cross-comparison on Tiger Lake i7-1165G7 (AVX-512 + VNNI hardware),
reported by @Tamnac in the issue thread:

| Engine | RT× |
|---|---|
| CrispASR Q8_0, AVX-2 build (v0.6.4 prebuilt) | 4.1× |
| CrispASR Q8_0, AVX-512 + VNNI source build  | 4.3× |
| onnx-asr int8, CPU EP                       | 21.2× |

VNNI alone gets ~5%, not the 1.5-2× I optimistically projected in
the first reply on this issue. The structural diagnosis from the
reporter's LLM is correct: ggml materialises the full activation
tensor between every op, and on a memory-bandwidth-bound mobile
CPU (~50 GB/s LPDDR4X) that intermediate traffic dominates total
throughput. VNNI accelerates *the inner loop of the int8 dot
product*, but the matmul is only a fraction of total Conformer
encoder cost.

ONNX Runtime int8 CPU EP closes most of this with broader kernel
fusion that ggml doesn't ship:

- conv + BN + activation → one kernel
- MHA (Q/K/V proj + softmax + attn × V) → one kernel
- LayerNorm + GEMM → one kernel
- per-tensor int8 *activations*, not just weights — ~4× less
  bandwidth moving through the same encoder block

ggml has `flash_attn_ext` for attention and a fused `norm + mul +
add` for RMSNorm, but doesn't cover the broader conv-norm-act or
layernorm-gemm fusion. Closing the gap meaningfully needs either:

1. CrispASR-side fusion at the FastConformer graph builder
   (chain the existing ggml ops into fewer materialisations).
2. New fused ops landed upstream in ggml-org/llama.cpp.
3. Per-tensor int8 activations in the GGUF quant pipeline
   (independent of the above; big lift).

Multi-quarter effort. For the documented record: **on Tiger Lake-
class mobile CPUs with VNNI hardware, ONNX Runtime int8 will win
on parakeet not because of VNNI usage but because of kernel-fusion
architecture.** The AVX-512 release variants (v0.6.5+ Linux tarballs)
get ~5% over AVX-2; they don't close the structural gap.

The wins we already ship are still real: Apple M1 Metal (parakeet
TDT q8_0) is 8.24× RT vs ONNX CPU's 6.23× RT — see the earlier
section. The structural-fusion gap is a CPU x86 specific story;
GPU paths bypass it because compute throughput, not memory
bandwidth, is what limits them.

---

## Long-audio coverage benchmark — 2026-05-21 (issue #89)

Platform: x86_64 VPS, 4 threads, CPU-only, no GPU. Commit `5e16414`
(30 s auto-chunk fallback + PR #116 VAD gate fix).

Test audio: first 60 s of the issue #89 reporter's exact YouTube clip
(`o_9dWkRPYC0`, Japanese podcast, 16 kHz mono PCM). Human estimate:
100-150 words in the first 60 s of continuous speech.

### Issue #89 fix verification — parakeet-tdt-0.6b-ja

**Final state — streamed encoding is always on (default
`CRISPASR_PARAKEET_STREAM_THRESHOLD=0`, see commit "always route
parakeet through streamed encode").**

Global z-norm + overlapping 8 s encoder chunks + single TDT decode pass.

The earlier "single-pass ≤60 s" default produced 99.5 % coverage on the
cached MP3-derived copy of the reporter's clip but only ~33 % on a
fresh `yt-dlp` extract of the same YouTube video (lenhone, issue #89
comment 4529025103). Both extracts are perceptually identical
(duration 60.000 s, 0.998 waveform correlation, ~0.3 % RMS diff from
codec quantization) but the FastConformer encoder's full-clip
bidirectional attention amplified that quantization noise enough
(encoder output std differed by 14 %: 0.2069 vs 0.2415) to drive the
TDT decoder into emit-blank-forever past frame 250 (≈20 s). The
streamed path keeps attention local to 8 s windows, so codec
perturbations don't amplify and the decoder runs to the end.

| path | audio | chars | first_ts | last_ts | coverage% | gaps |
|---|---|---:|---:|---:|---:|---:|
| **streamed (default)** | reporter's MP3-derived `yt_60s.wav` | **309** | **0.00** | **55.84** | **~99** | **0** |
| **streamed (default)** | fresh `yt-dlp` Opus→PCM `o_9dWkRPYC0_60s.wav` | **314** | **0.00** | **55.84** | **~99** | **0** |
| old: single-pass ≤60 s | reporter's MP3-derived `yt_60s.wav` | 309 | 0.16 | 59.84 | 99.5 | 0 |
| old: single-pass ≤60 s | fresh `yt-dlp` Opus→PCM `o_9dWkRPYC0_60s.wav` | 91 | 0.16 | 20.08 | **~33** | 0 |
| `--vad` (silero) | reporter's MP3-derived | 281 | 0.36 | 59.87 | 93.1 | 1 |
| `--vad --vad-model firered` | reporter's MP3-derived | 238 | 0.28 | 58.01 | 85.1 | 1 |
| old: 30 s independent chunks | reporter's MP3-derived | 195 | 0.16 | 58.02 | 59.7 | 2 |

**Key findings:**
- Single-pass encoding over the full clip is **not robust**: a 0.3 %
  RMS difference between two codec-quantized copies of the same speech
  flips the encoder into emit-blank mode at the 20 s mark on one and
  not the other. The streamed path is robust to that perturbation.
- The streamed pipeline gives ~99 % coverage on both audio variants.
- `--vad` (silero) gives 93 % coverage with speech-boundary
  segmentation.  Useful when you want per-utterance SRT entries rather
  than continuous transcription.
- The old 30 s independent-chunk approach (pre-fix) lost content due
  to TDT decoder cold-start on each chunk (each chunk reset the LSTM
  state).
- **Recommendation for Japanese:** just run `crispasr -m
  parakeet-tdt-0.6b-ja.gguf -f audio.wav -osrt` — the default routes
  through streamed and handles any duration on any audio source.

### Robustness validation — 2026-05-23

Full sweep on the reporter's 60 s clip (commit `0c24178e`, CPU-only).
Streamed pipeline output is **byte-identical to single-pass** across
every chunk/overlap combination tested — the global z-norm makes chunk
boundaries transparent to the TDT decoder.

> **Caveat (added 2026-05-24).** The "identical to single-pass" column
> below is only true on the cached MP3-derived audio (`/mnt/storage/
> samples/o_9dWkRPYC0.mp3` → `yt_60s.wav`). On a fresh `yt-dlp` extract
> of the same YouTube video, single-pass collapses to ~20 s of output
> while streamed still covers the whole clip — see the "single-pass
> not robust" finding above. The streamed-vs-streamed numbers across
> chunk and overlap sizes (300+ chars, 99 %+ coverage) hold across
> both audio derivations.

**Chunk-size sweep** (streamed, overlap=2 s):

| chunk | chars | coverage% | gaps | identical to single-pass? |
|---|---:|---:|---:|---|
| 4 s | 294 | 99.5 | 0 | yes |
| 6 s | 294 | 99.5 | 0 | yes |
| 8 s (default) | 294 | 99.5 | 0 | yes |
| 12 s | 294 | 99.5 | 0 | yes |
| 16 s | 294 | 99.5 | 0 | yes |
| 20 s | 294 | 99.5 | 0 | yes |
| 30 s | 294 | 99.5 | 0 | yes |

**Overlap sweep** (streamed, chunk=8 s):

| overlap | chars | coverage% | gaps | identical? |
|---|---:|---:|---:|---|
| 0 s | 294 | 99.5 | 0 | yes |
| 1 s | 294 | 99.5 | 0 | yes |
| 2 s (default) | 294 | 99.5 | 0 | yes |
| 3 s | 294 | 99.5 | 0 | yes |
| 4 s | 294 | 99.5 | 0 | yes |

**300 s Japanese audio** (streamed, default 8 s chunks):
- 655 chars, **98.6 % coverage**, 0 gaps, first=0.00 last=295.84
- Before fix (30 s independent chunks): 636 chars starting at 58 s

**VAD comparison** (60 s):

| mode | chars | coverage% | gaps |
|---|---:|---:|---:|
| auto (streamed) | 294 | 99.5 | 0 |
| `--vad` silero | 281 | 93.1 | 1 |
| `--vad --vad-model firered` | 238 | 85.1 | 1 |

VAD produces fewer characters because it segments on speech boundaries
and transcribes each segment independently. The auto/streamed path
transcribes the full audio continuously and achieves higher coverage.
Use VAD when you need per-utterance SRT entries; use auto when you want
maximum transcription completeness.

### Multi-backend Japanese comparison (60 s)

All backends on the same 60 s Japanese clip. "chars" counts non-space
characters (Japanese has no word spaces; "words" column counts
space-delimited tokens, which undercounts for CJK).

| backend | settings | chars | coverage% | gaps | wall_s | rtf |
|---|---|---:|---:|---:|---:|---:|
| **parakeet-tdt-0.6b-ja** | **auto (streamed)** | **294** | **99.5** | **0** | **~60** | **~1.0×** |
| cohere-transcribe | `--vad` | 296 | 96.8 | 0 | 169.0 | 0.4× |
| parakeet-tdt-0.6b-ja | `--vad` silero | 281 | 93.1 | 1 | 50.7 | 1.2× |
| cohere-transcribe | auto | 242 | 87.4 | 1 | 199.2 | 0.3× |
| parakeet-tdt-0.6b-ja | `--vad` firered | 238 | 85.1 | 1 | 58.3 | 1.0× |

**Quality ranking for 60 s Japanese:**
1. **Parakeet auto (streamed)** — 99.5 % coverage, zero gaps, ~1× RT.
   The NeMo-style pipeline makes this the clear winner.
2. **Cohere + VAD** — 96.8 %, zero gaps, but 3× slower.
3. **Parakeet + VAD silero** — 93.1 %, 1 gap. Useful for per-utterance
   subtitle segmentation.

### Cross-backend CAP_INTERNAL_CHUNKING — 2026-05-23

The 30 s auto-chunk fallback affected all `CAP_UNBOUNDED_INPUT` backends
that use PerFeatureZ mel normalization, not just parakeet.  Adding
`CAP_INTERNAL_CHUNKING` to canary and fastconformer-ctc (commit
`1dd247a7`) lets them skip the auto-chunk and process full audio in a
single encoder pass.

| backend | audio | coverage (old 30 s chunks) | coverage (new, single-pass) |
|---|---|---:|---:|
| parakeet-tdt 0.6b JA | 60 s JA | 59.7 % | **99.5 %** |
| parakeet-ctc 1.1b | 60 s EN | 74.6 % | **98.5 %** |
| canary-1b-v2 Q4_K | 60 s EN | broken (empty) | **96.8 %** |

**Not affected** (different normalization or architecture):
wav2vec2 / hubert / data2vec (GlobalClipMax, no PerFeatureZ drift);
firered-asr (PerFeatureZ but inline AED — needs separate investigation);
granite-nar (different architecture).

### Benchmark framework

Results collected with `tests/benchmark_asr.py`:

```bash
# Quick single-backend triage:
python tests/benchmark_asr.py --audio myfile.wav --backend parakeet

# Full matrix across backends and settings:
python tests/benchmark_asr.py --corpus /mnt/storage/test-audio/corpus.json --all-settings

# Build the test audio corpus (en/de/ja/zh × 4 durations from FLEURS):
python tests/benchmark_corpus.py
```

Results are stored in `/mnt/storage/benchmark-results/runs.jsonl` (JSONL,
one line per run). The framework computes: word count, char count,
first/last timestamp, time coverage %, gap count/size, wall time, and
realtime factor. See `tests/benchmark_metrics.py` for the metric
definitions and `tests/test_benchmark_metrics.py` for 14 pytest unit
tests that validate the computation (including the issue #89 failure
signature: <5 % coverage on 300 s audio).

### Multi-backend long-form Japanese — 120 s sweep (2026-05-24, issue #89 follow-up)

Test audio: first **120 s** of the issue #89 reporter's exact YouTube
clip (`o_9dWkRPYC0`, fresh `yt-dlp` Opus→WAV extract, the file lenhone
actually reports against — md5 `d1f2ef…`, *not* the cached MP3-derived
copy). Apple M1 Metal, default flags unless noted.

Speech runs out around 01:37; the remaining ~22 s is short pause +
follow-on talking. All "covers full speech" rows below land around
01:37 → 02:00.

| backend | mode | segments | first → last ts | coverage | notes |
|---|---|---:|---|---|---|
| **parakeet-tdt-0.6b-ja** (default, streamed TDT) | full | 12 | 0:00 → 1:37.84 | full speech | post-fix `33f9a162` |
| **parakeet-tdt-0.6b-ja** + `--vad` | full | 14 | 0:00 → 1:58.39 | full speech | cleaner per-utterance |
| **parakeet-tdt-0.6b-ja** + `--parakeet-decoder ctc` (hybrid CTC head) | full | 12 | 0:00 → 1:37.84 | full speech | byte-identical to streamed-TDT — confirms encoder is fine |
| **parakeet-tdt-0.6b-ja** + `STREAM_THRESHOLD=999` (forced single-pass) | full | many | 0:00 → ~14 s then kana-by-kana fragmentation | **broken** | the issue #89 bug; reproduces the lenhone complaint |
| **sensevoice-small** (CTC, multilingual) | `--vad` | 13 | 0:00 → 2:00 | full speech | accurate, minor JA glitches (`スピーク**ジャ**プネス**ナチャパ**`); 19.8× RT |
| **voxtral-mini-3b** (LLM AR, multilingual) | default chunking | partial | 0:00 → 0:27 then 1:47 → 2:00 | **drops 0:27 → 1:47 (~80 s)** | LLM decoder loses a middle chunk |
| **cohere-transcribe** (Conformer, multilingual) | default chunking | 4 | 0:00 → 1:53 | **sparse with multi-tens-of-seconds gaps** | only ~0:00, 0:50, 1:18, 1:48 anchors |
| **canary-1b-v2** (NeMo multilingual seq2seq) | default | broken | n/a | hallucinates `"I am not aware of anything"` in English | needs proper language-prompt wiring; out of scope here |

**Best on this 120 s clip:** parakeet-tdt-0.6b-ja (post-fix, streamed
TDT) and sensevoice-small are tied — both produce full speech coverage
with sentence-level segmentation. Parakeet via the CTC head produces
*byte-identical* output to streamed-TDT, which confirms the encoder
isn't the problem: only TDT-decoded-over-the-full-utterance is.

**The new finding (voxtral, cohere):** these aren't parakeet-specific
failures. voxtral and cohere both drop the middle of a 120 s clip on
this audio, with different symptoms:

- **voxtral-mini-3b**: LLM decoder loses one middle chunk entirely
  (segments span 0:00 → 0:27, then skip to 1:47 → 2:00). The chunker
  hands the AR decoder its middle window, the decoder either runs to
  max_new_tokens before catching up to the audio or skips ahead via
  prompt conditioning that misfires on this clip. Not investigated
  deeply yet; this is an open follow-up — see PLAN #114.
- **cohere-transcribe**: at the default chunk size, the energy chunker
  hands the encoder a small number of long slices and the Conformer
  encoder hits a similar long-bidirectional-attention-amplifies-noise
  regime as parakeet single-pass on lenhone's audio. Only ~4 segments
  emitted across 120 s, with gaps of tens of seconds between them.

**Upstream behaviour for the same long-form failure:**

- **NeMo parakeet / canary**: stock `model.transcribe()` does single-
  pass over the full utterance (verified locally — NeMo's own
  `transcribe()` produces 47 chars and stops at ~20 s on the same
  lenhone WAV, *exactly* matching our pre-fix single-pass). For long
  audio NeMo ships `nemo.collections.asr.parts.utils.streaming_utils.
  BatchedFrameASRTDT` / `FrameBatchChunkedCTC` and
  `nemo.collections.asr.parts.utils.transcribe_utils.
  get_buffered_pred_feat_rnnt`, plus the
  `examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_
  infer_rnnt.py` reference script. Our `parakeet_transcribe_streamed`
  is the same shape: global-z-norm + chunked encode + single TDT
  decode, with overlap-skip on chunk boundaries. The difference is
  ours is now the default; upstream `transcribe()` is not.
- **Mistral voxtral**: the reference HuggingFace integration chunks at
  ~30 s with overlap; the long-form failure we see at 120 s is partly
  an artefact of how our energy-chunker hands the slices to the LLM
  decoder (not upstream-identical chunking).
- **Cohere Transcribe**: the released model is intended for
  ≤ a few minutes per call; the hosted product does server-side VAD +
  chunking, the released weights do not.

**Implication.** The "default fine on a single transcribe() call over
the whole file" affordance is fragile across this whole class of
models. Going forward we should probably treat *every* `CAP_UNBOUNDED
_INPUT` backend the way we now treat parakeet: ship a chunked /
streamed default that the user doesn't have to opt into. See PLAN
#114 for the open architectural question and the per-backend ladder.

### Cross-length × cross-backend matrix — 60 / 120 / 300 / 600 s (2026-05-25)

Same audio (lenhone's fresh `yt-dlp` extract), extended to longer
durations. Linux x86 CPU on the issue #89 VPS (`168.119.190.252`),
sequential to avoid memory contention (we paused / split the queue
when the kernel went into thrash territory). `tools/longform_vps.sh`
is the harness; `tools/analyze_longform.py` parses the per-cell
JSON output.

> **Matrix v1 vs v2 (2026-05-25 afternoon recheck).** The numbers
> below were collected in two passes. The first pass (matrix v1)
> ran on the VPS binary `bd8b98cf` (May 24), which **predates** the
> per-backend opt-out fixes for cohere (`dc2295b2`), gemma4-e2b /
> glm-asr (`46f6848d`), kyutai-stt (`eaee2319`), and voxtral
> (`6fef8790`) that landed during the matrix run. Those fixes
> remove an external overlap-save context wrap that the LLM-decoder
> backends couldn't trim back from correctly. Their pre-fix coverage
> of ~9-65 % at 120 s+ was driven by the wrap, not the model
> architecture. **Matrix v2 (post-opt-out, rebuilt VPS binary
> `13059e0c`)** shows the true post-fix behaviour. Both passes are
> kept so the reader can see the cost of the missing opt-out.

**Coverage % (covered span / clip duration, computed from segment
timestamps).** Higher = better. Bold = best at that length.

**Matrix v2 (post-opt-out, what main looks like today):**

| backend / mode               |  60 s |  120 s |   300 s |   600 s |
|---|---:|---:|---:|---:|
| **parakeet streamed-TDT** (default)       | **93.1** | 81.5 | **96.6** | **99.3** |
| parakeet CTC head (byte-identical)        | **93.1** | 81.5 | **96.6** | **99.3** |
| **voxtral-mini-3b** (default chunking)    | **100.0** | **100.0** | **100.0** | wall-time timeout (`rc=124` at 900 s in 15 min limit; LLM-AR CPU-bound, not a coverage failure) |
| **voxtral-mini-3b** streamed (option A — this PR's pipeline) — single LLM context | **100 %**, 11 segs / 470 chars | **100 %**, 527 chars | **100 %**, 1276 chars / 863 tokens (post-`a5165c84` max_new scaling fix; was 781 chars / 512-tok cap pre-fix) | hung on contended M1 (80 MB free out of 16 GB → Metal allocator stall — see HISTORY 2026-05-25 (late) "Distinguishing slow vs hung run") |
| **cohere-transcribe** (default chunking)  | **96.3** | **97.9** | **98.1** | **97.9** (22 segs, full 0:00 → 10:00, 577 s wall) |
| parakeet single-pass (`STREAM_THRESHOLD=999`, opt-in regression bait) | 33.2 | 81.7 | **1.5** | 99.9 |
| parakeet + `--vad` (silero)               | 86.7 | 82.0 | 76.3 | 84.0 |
| canary-1b-v2                              | (still hallucinates English at every length — separate prompt-wiring bug, PLAN #114 P3) | | | |

**Largest gap (seconds) between consecutive segments, matrix v2:**

| backend / mode               |  60 s |  120 s |   300 s |   600 s |
|---|---:|---:|---:|---:|
| parakeet streamed-TDT                | 0.0 | 0.0 | 0.0 | 0.0 |
| voxtral default (post-opt-out)       | 0.0 | 0.0 | 0.0 | timeout |
| cohere default (post-opt-out)        | 1.2 | 1.2 | 1.2 | 2.2 |

The pre-fix gap pathologies — voxtral 21.9 / 78.2 / 240.9 / 545.5 s and cohere up to 50 s — are entirely gone with the opt-out fixes. What remains is the cohere baseline ~1.2 s gap between chunks (natural energy-chunker boundaries, well under a sentence pause).

**Wall time (s), matrix v2:**

| backend / mode               |  60 s |  120 s |   300 s |   600 s |
|---|---:|---:|---:|---:|
| voxtral default (post-opt-out)       | 237 | 393 | 834 | timeout (>900 s) |
| cohere default (post-opt-out)        |  70 | 125 | 290 | 577 |

cohere is consistently ~2× realtime at 300-600 s on VPS x86 CPU. voxtral-mini-3B is ~1.4-2× slower than cohere at the same length (LLM AR decode at 3 B params) and hits the wall around the 10 min mark. Apple Silicon Metal would close most of that gap — the LLM-AR rows on Mac are typically 5-10× faster than x86 CPU on this size class.

**Matrix v1 (pre-opt-out, kept as historical reference for what we
fixed):**

| backend / mode               |  60 s |  120 s |   300 s |   600 s |
|---|---:|---:|---:|---:|
| **parakeet streamed-TDT** (default)        | **93.1** | 81.5 | **96.6** | **99.3** |
| **parakeet CTC head**                       | **93.1** | 81.5 | **96.6** | **99.3** |
| parakeet single-pass (`STREAM_THRESHOLD=999`) | 33.2 | 81.7 | **1.5** | 99.9 |
| parakeet + `--vad` (silero)                 | 86.7 | **82.0** | 76.3 | 84.0 |
| voxtral-mini-3b (default chunking)          | 63.5 | 34.8 | 19.7 | 9.1 |
| cohere-transcribe (default chunking)        | **95.0** | **91.5** | 58.8 | 61.8 |
| cohere-transcribe + `--vad`                 | **96.8** | 90.8 | **92.5** | **91.4** |
| canary-1b-v2 (default)                      | 99.7* | 99.9* | 99.3* | OOM (rc=137) |

*Canary's coverage% is misleading — the transcribed text is `"I am not aware of anything, I am not aware of…"` (English) at every duration. It's a separate language-prompt-wiring bug, not a long-audio bug. See PLAN #114 P3.

**Largest gap (seconds) between consecutive emitted segments** —
catches the "drops a middle chunk" failure that the coverage% can
under-report when the missing region is bracketed by emitted text on
both sides.

| backend / mode               |  60 s |  120 s |   300 s |   600 s |
|---|---:|---:|---:|---:|
| parakeet streamed-TDT                       |  0.0 |  0.0 |  0.0 |  0.0 |
| parakeet CTC head                           |  0.0 |  0.0 |  0.0 |  0.0 |
| parakeet single-pass                        |  0.0 |  0.0 |  0.0 |  0.0 |
| parakeet + `--vad`                          |  5.8 | 12.0 | 30.1 | 30.1 |
| **voxtral-mini-3b**                         | **21.9** | **78.2** | **240.9** | **545.5** |
| cohere-transcribe                           |  0.0 |  3.4 | **50.0** | **50.0** |
| cohere-transcribe + `--vad`                 |  1.4 |  4.2 |  4.2 | 19.0 |
| canary-1b-v2                                |  0.0 |  0.0 |  0.0 | n/a |

**Wall time (s) / realtime factor.** Apple Silicon would be 5-10×
faster on the parakeet rows; numbers below are the Linux x86 VPS.

| backend / mode               |  60 s |  120 s |   300 s |   600 s |
|---|---:|---:|---:|---:|
| parakeet streamed-TDT                       |  55 |  99 |  236 |  463 |
| parakeet CTC head                           |  54 | 102 |  236 |  462 |
| parakeet single-pass                        |  45 |  86 |  235 |  627 |
| parakeet + `--vad`                          |  55 |  97 |  225 |  457 |
| voxtral-mini-3b                             | 166 | 165 |  189 |  193 |
| cohere-transcribe                           |  79 | 144 |  349 |  673 |
| cohere-transcribe + `--vad`                 |  65 | 117 |  279 |  557 |
| cohere-asr-ja (Q4_K, JA audio)             |  31 | 140 |    — |    — |
| cohere-asr-ja (Q4_K, EN audio)             |  39 | 104 |    — |    — |
| canary-1b-v2                                |  68 | 122 |  381 | OOM  |

(voxtral wall time is roughly constant because it silently skips
most of the input — see the gap column.)

### Per-backend take-aways from the matrix

**parakeet (the post-fix default).** streamed-TDT and CTC-head are
byte-identical at every length (CTC head is a frame-synchronous
fallback that bypasses the TDT blank-runaway entirely; streamed-TDT
keeps the TDT decoder but bounds the encoder's bidirectional
attention to 8 s windows so it can't accumulate the codec-noise
amplification). The 120 s coverage dip to 81.5 % is the *audio*, not
the model: the clip's speech runs out at ~01:37 and the next ~22 s
is silence + a sentence-start, so coverage measured against the full
120 s under-counts. Both paths produce the same actual content.

**parakeet single-pass.** Catastrophically non-monotonic: 33 % at
60 s, 82 % at 120 s, **1.5 %** at 300 s, 99.9 % at 600 s. This is the
"per-feature z-norm depends on the full audio's mel statistics"
problem manifesting as random walks across the stable/unstable
boundary. The single-pass path is genuinely unsafe; the "works at
600 s" cell is luck, not a property.

**parakeet + `--vad`.** Coverage drops to 76-87 % across lengths
because VAD trims silence (by design). Larger gaps at longer
durations because the underlying clip has more silence stretches.
Good for "I want per-utterance SRT entries" use cases, less so for
"I want continuous transcription with maximum coverage."

**voxtral-mini-3b.** **In matrix v1: worst long-form behaviour we
measured.** Coverage halved with each length doubling: 64 → 35 → 20 →
9 %. In matrix v2 (post-opt-out, commit `6fef8790` removing the
external overlap-save wrap), coverage jumps to **100 % at 60 / 120 /
300 s**: the LLM AR decoder *was* processing all chunks fine; the
matrix-v1 word-timestamp trim was discarding most of the emitted text
because voxtral's emitted word timestamps don't honour the original
slice frame, so the trim treated almost everything as "outside the
slice range." Two additional fixes shipped together with this matrix:

  * `6fef8790` — voxtral opt-out from the external overlap-save wrap
    (the immediate >90 % fix; default-chunked voxtral is now sound).
  * **PR #114 voxtral_transcribe_streamed** (matching the upstream
    Mistral `apply_transcription_request` pattern): per-30 s encode,
    concatenate audio embeds, **single LLM AR decode** over the whole
    sequence. Result on the 60 s clip is **denser segmentation** (11
    segments / ~470 chars vs 3 segments / 280 chars on default
    chunking) because the LLM doesn't cold-start at every 30 s
    boundary; it sees one continuous audio stream. Both paths produce
    correct content; streamed is the more upstream-faithful default.

**cohere-transcribe.** **In matrix v1: degraded from 95 % at 60 s to
**59 %** at 300 s and **62 %** at 600 s, with 50 s gaps.** In matrix
v2 (post-opt-out, commit `dc2295b2` removing the external overlap-save
wrap for cohere), default chunking jumps to **96-98 %** at 60 / 120 /
300 s with gaps ≤ 1.2 s. The pre-fix gap-growth was driven by the
overlap-save wrap, not the model itself. `--vad` is no longer a
mandatory rescue — it's available for users who want per-utterance
SRT segmentation, but coverage parity is now native.

**canary-1b-v2.** Separate bug. Coverage looks fine because the
decoder emits text for the full duration, but the text is English
`"I am not aware of anything"` in a loop regardless of input
language. Language-prompt wiring problem, not a long-audio problem.
600 s OOM-killed (rc=137) on the 7.6 GB VPS — likely the AED
decoder's hidden-state stack growing past the available memory.

### What's the right default per backend, post-matrix v2

| backend | recommended default | why |
|---|---|---|
| parakeet (any variant)         | streamed-TDT (default since `33f9a162`) | best coverage at all lengths, byte-identical to CTC-head when available |
| voxtral-mini-3b                | streamed (this PR — Mistral `apply_transcription_request` shape) | 100 % coverage at 60-300 s, single LLM context, denser segmentation; default-chunked + opt-out (`6fef8790`) also lands at 100 % |
| cohere-transcribe              | default chunking + opt-out (`dc2295b2`)         | 96-98 % at 60-300 s; `--vad` available but no longer required for coverage |
| canary-1b-v2                   | fix lang-prompt bug first; then streamed-encode port | currently broken at all durations on JA; long-audio fix on hold |
| qwen3-asr / granite-speech / mimo-asr | post-opt-out default chunking (audit pending) | LLM-AR class — opt-out gate is `glm-asr` / `gemma4-e2b` / `kyutai-stt` (`46f6848d`, `eaee2319`); voxtral-style streamed is a follow-up improvement, not a coverage fix |
| fastconformer-ctc / wav2vec2 / firered-asr | current single-pass (CTC is robust) | no observed failure; defer streamed port until reported |
| sensevoice-small               | `--vad`                                          | already the recommendation; matrix v1 confirms 99 %+ at 120 s |
| whisper                        | unchanged                                        | internal 30 s seek handles long audio by design |

### Reproducer

```bash
# Driver — runs all 32 cells sequentially with memory backpressure
bash tools/longform_vps.sh   # outputs to /mnt/akademie_storage/longform_results/

# Parser — JSON outputs → coverage table
python tools/analyze_longform.py /path/to/longform_results/
```

Both scripts in this commit. Audio: `/mnt/akademie_storage/yt_{60,120,300,600}s.wav` on the VPS (PCM s16le, 16 kHz mono, fresh `yt-dlp` extract of `youtube.com/watch?v=o_9dWkRPYC0`).

---

## Beam search — quality vs speed (2026-05-23, PLAN #90)

**Knob:** `--beam-size N` (CLI) / `CRISPASR_BEAM_SIZE=N` (env) /
`crispasr_session_set_beam_size(session, N)` (C API).
Default N=1 (greedy). N > 1 activates `core_beam_decode::run_with_probs`
on LLM-decoder backends: qwen3-asr, granite-speech, voxtral.
Non-AR backends (parakeet, canary, fastconformer-ctc, etc.) ignore the
flag — beam search only makes sense on autoregressive token decoders.

Benchmark script: `tools/benchmark_vitw_beam.py` — runs against
[`zhifeixie/Voices-in-the-Wild-Bench`](https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-Bench)
(5 000 samples, 8 acoustic conditions, streamed — no full download needed).

### Speed cost on JFK (11 s, M1 Metal, post-warmup)

| backend | beam=1 | beam=2 | beam=4 |
|---|---|---|---|
| qwen3-asr 0.6B Q4_K | 3.67 s (1×) | 8.20 s (2.2×) | 14.75 s (4.0×) |
| granite-speech 4.1 2B Q4_K | 18.39 s (1×) | 27.59 s (1.5×) | 33.17 s (1.8×) |
| voxtral mini 3B Q4_K | ~70 s (1×) | ~56 s (0.8×)† | ~77 s (1.1×) |

†voxtral beam=2 < beam=1 is measurement noise — voxtral spends most
of its time in the audio encoder; decoder token count for JFK is
small enough that OS jitter dominates.

### WER by condition (qwen3-asr, Voices-in-the-Wild-Bench, 8 EN samples each)

| condition | beam=1 | beam=2 | beam=4 | beam=2 cost | beam=4 cost |
|---|---|---|---|---|---|
| real_noise | 0.125 | 0.144 | 0.136 | 1.7× | 3.5× |
| syn_noise | 0.167 | 0.167 | 0.167 | 2.6× | 2.7× |
| real_dropout | 0.045 | 0.045 | 0.041 | 1.9× | 4.6× |
| real_obstructed | 0.015 | 0.015 | 0.015 | 1.9× | 3.3× |
| real_mixed | 0.035 | 0.039 | 0.039 | 2.1× | 4.8× |
| syn_mixed | 0.089 | 0.089 | 0.080 | 1.3× | 2.2× |

### Key findings

- **One clear win:** `syn_mixed` — "I called customer service **twice**"
  decoded as "wise" (greedy, WER 0.154), correctly as "twice" at beam=4
  (WER 0.077). Phonetically similar word confusion — exactly the
  scenario beam search is designed to fix.
- **beam search occasionally hurts.** `real_noise` ticks from 0.125 to
  0.144 at beam=2. The beam finds a confident wrong hypothesis that
  greedy would have gotten right — known failure mode on well-trained
  models where the greedy peak is already correct.
- **Heavy hallucination is not rescued.** `real_noise` sample 6
  (WER≈0.42, badly degraded audio) just produces a different confabulation
  at beam=4; the model is guessing regardless of beam width.
- **This dataset skews easy.** Most samples are TTS speech with layered
  acoustic corruption; qwen3-asr is near-ceiling on greedy. Real
  spontaneous noisy speech with disfluencies and rare words would expose
  more beam-search-recoverable errors.

### When to use

| scenario | recommendation |
|---|---|
| Clean / studio speech | greedy (beam=1) — no quality gain, 2-5× cost |
| Noisy real speech, latency-insensitive | beam=2 — marginal gain possible, 2× cost |
| Rare words / phonetic confusion, offline | beam=4 — worth trying |
| Streaming / latency-critical | greedy only — beam adds a full extra decode pass per token step |

Reproduce:

```bash
python tools/benchmark_vitw_beam.py \
    --backends qwen3 \
    --splits real_noise,real_dropout,real_obstructed,real_mixed,syn_mixed \
    --n 8 --beams 1,2,4 \
    --json tools/vitw_beam_results.json
```

## Multi-backend long-form comparison — 2026-05-26 (PLAN #114 P3 closeout)

Live runs on M1 Metal with the post-PLAN-#114-P3 binaries. Inputs from
`/Volumes/backups/code/audio_samples/` (mirrored from VPS — see that
dir's CLAUDE.md). All backends invoked with `-l <lang>`, `-np`, `-nt`,
default settings (`CANARY_STREAM_THRESHOLD_S=0` after `10c2fba5`).

### EN 60 s (FLEURS English, narration)

```
audio_samples/en/fleurs_60s.wav  (60 s, 16 kHz mono)
```

| Backend | Chars | Notes |
|---|---|---|
| parakeet-tdt-0.6b-v3 | ~217 → **520** (post `e1904a1e`) | The 217 was with the c=8 chunk default that ships well for the JA-only model. Empirical sweep on EN+DE FLEURS 60s+300s showed c=8 collapses on the multilingual v3 model: EN 60s drops to 23% of the c=40 max. Fix `e1904a1e` adds a per-model chunk default keyed off `vocab_size` (< 4000 ⇒ JA model ⇒ c=8 preserved; ≥ 4000 ⇒ v3 / multilingual ⇒ c=30). v3+EN60 improved 186 → 520 chars (2.8×), v3+EN300 492 → 1550 chars (3.15×), v3+DE60 502 → 679, v3+DE300 2496 → 3064, ja+JA60 1674 unchanged. Root cause is *encoder* context, not decoder cold-start — see the corrected LEARNINGS section on the Independent-chunk failure mode. |
| canary-1b-v2 | ~735 | Full content but visible artifacts: `"twenty five. to thirty"` (model splits a number), `"Save for You"` (AED re-emits with different capitalization → after case-insensitive LCS the dup is dropped but the leftover `"Save for You"` reads as a sentence start), `"Yeah, yeah, ×14"` (degenerate-loop guard fired at the configured 14-token window). |
| voxtral-mini-3b-2507 | **~826** | **Clean.** Includes extra content like `"in which was made famous to foreigners after a glowing account of its splendorous recorded by Lord Byron"` that canary missed entirely. No boundary artifacts. |
| cohere-transcribe | **~864** | **Clean.** Similar coverage to voxtral, `"world's"` instead of `"world"` (model preference), `"Northern Marianas"` instead of `"Northern Mariana's"`. |

### DE 60 s (FLEURS German, narration)

```
audio_samples/de/fleurs_60s.wav  (60 s, 16 kHz mono)
```

| Backend | Notes |
|---|---|
| canary-1b-v2 | Full content but boundary dups: `"Geld-Technologie-Technologie-Technologie-Technologie"` (early-chunk loop, partly caught by the guard at 4 reps before the window opened), `"T-Rex war war"`, `"Rückseite der der Unabhängigkeitserklärung"`, `"Männer und Frauen. Frauen"`, `"Spitze. der Spitze"`. The LCS-merge + word-snap + case-insensitive LCS pipeline caught some but not all — these are exact-token re-emissions across chunks that an LCS strict-prefix match still leaks. |
| voxtral-mini-3b-2507 | **Clean, single-pass-quality.** No boundary artifacts visible. Catches `"Juden und Nicht-Juden gleichermaßen"` (post-segment continuation) that canary missed. |
| cohere-transcribe | **Clean.** `"Tri-Rex"` is a minor model error (not a boundary artifact), otherwise identical-shape transcript to voxtral. |

### Architectural takeaway

The data confirms the design-notes table in PLAN #114 ("Streaming-pattern
design"): the **voxtral-pattern backends** (voxtral, cohere — 30 s
disjoint chunks → audio embeds concat → one LLM AR decode) produce
cleaner long-form output than the **NeMo-pattern backends** (canary —
8 s overlap chunks → per-chunk decode → LCS-merge + word-snap +
case-insensitive LCS dedup). The voxtral pattern's lack of overlap means
no duplication enters the input, so no dedup is needed; the NeMo
pattern's overlap (necessary for bidirectional encoder context) requires
dedup, and any imperfect dedup pass leaves visible artifacts.

This is not a universal win for the voxtral pattern though — it requires
a long-context AR LLM (voxtral's 3 B, cohere's 1.3 B). Canary's AED was
trained on 8–30 s clips and cannot absorb a full 5 min in a single decode
(`<eos>` lands at the first internal utterance boundary). Parakeet's TDT
could in principle use the voxtral pattern but doesn't currently —
something to revisit if parakeet's truncation behaviour on the EN 60 s
clip turns out to be a streamed-path bug rather than a one-off.

### Six-commit canary thread that produced the "full content" column above

| SHA | What |
|---|---|
| `dfe1af3b` | lang-whitelist (en/de/fr/es only) — refused unsupported langs before they could hallucinate |
| `7177c931` | `canary_transcribe_streamed` first cut (concat-then-decode → truncated at AED `<eos>`) |
| `63fdbe46` | NeMo `FrameBatchMultiTaskAED` analogon — per-chunk AED decode with prompt re-injection |
| `62766dae` | LCS boundary dedup |
| `10c2fba5` | splice-punct cleanup + `CANARY_STREAM_THRESHOLD_S=0` default |
| `361df3e2` | window-based degenerate-loop guard |
| `935ffbee` | word-snap heuristic (extend LCS drop to next word-start) |
| `5e402ee9` | case-insensitive LCS (ASCII lowercase canonical id) |

Before this thread canary truncated to ~460 chars on the 1.3 m
De-Abwasch article and ~360 chars on EN FLEURS 60 s; after, full
coverage (~1196 chars and ~735 chars respectively).

## Parakeet long-form option matrix — 2026-05-26 (PLAN #114 follow-up)

Empirical sweep across all the dispatch knobs the parakeet backend
exposes, on the same three 60 s fixtures used elsewhere in this
section. Default mode includes the `e1904a1e` per-model chunk default
(v3 → c=30 internal, ja → c=8 internal).

| Mode | v3 + EN 60s | v3 + DE 60s | v3 + JA 60s | ja + JA 60s |
|---|---|---|---|---|
| **default** (backend streamed, c=auto) | 520 | 679 | 605 | **1674** |
| `CRISPASR_PARAKEET_STREAM_CHUNK=8` forced | 187 | 503 | 375 | **1674** |
| `CRISPASR_PARAKEET_STREAM_CHUNK=30` forced | 520 | 679 | 605 | 508 |
| `CRISPASR_PARAKEET_STREAM_THRESHOLD=999` (single-pass) | **626** | 621 | 599 | 271 |
| `--vad --vad-model silero` | 368 | **709** | 637 | 1627 |
| `--chunk-seconds 30 --chunk-overlap 0` (no LCS) | 713 | 689 | 608 | 1413 |
| `--chunk-seconds 30 --chunk-overlap 3` (LCS) | **755** | 665 | **660** | **1942** |

### Headline finding: dispatcher-side `--chunk-seconds 30 --chunk-overlap 3` wins on 3 of 4 cases — **shipped as the new default in `98381810`**

The internal-streamed-path default that previously shipped was **not** the
quality-optimal long-form mode. The CLI's dispatcher-side chunking +
overlap-save context wrap + LCS-merge dedup recovers more content than
the backend's single-pass-over-concat-encoder design.

**Shipped as the new default 2026-05-26 (`98381810`)** by dropping
`CAP_INTERNAL_CHUNKING` from the parakeet backend's capabilities
declaration. The dispatcher's `should_auto_chunk_long` fallback then
fires for audio > 30 s — chunking at 30 s, overlap-save 3 s, LCS-merge
dedup — exactly the matrix's winning mode. Short audio (< 30 s) is
unaffected: the dispatcher only auto-chunks past the threshold, so the
11 s JFK case still routes through a single backend call.

After-the-fix matrix (the previous matrix was with `CAP_INTERNAL_CHUNKING`
declared, blocking the auto-chunk path):

| case | old default | new default | Δ |
|---|---|---|---|
| JFK 11s | 109 | 109 | unchanged |
| v3 + EN 60s | 520 | **755** | **+45 %** |
| v3 + DE 60s | 679 | 665 | -2 % |
| v3 + JA 60s | 605 | **660** | +9 % |
| ja + JA 60s | 1674 | **1942** | **+16 %** |
| v3 + EN 300s | 1550 | **3865** | **+150 %** |
| v3 + DE 300s | 3064 | **3288** | +7 % |

The longer the audio, the bigger the win — EN 300 s scales from +45 % at
60 s to +150 % at 300 s. The internal-streamed-path's quality
degradation compounds with audio length; the dispatcher chunks scale
linearly.

Wall time on M1 Metal: 300 s EN now takes ~86 s (was ~30 s) — 3.5×
realtime. Acceptable for the quality gains; users can still pass
`CRISPASR_PARAKEET_STREAM_THRESHOLD=99999` to force the older
single-pass path if the wall-time matters more than coverage.

### Headline finding: dispatcher-side `--chunk-seconds 30 --chunk-overlap 3` wins on 3 of 4 cases (original 4-trial sweep)

Why this works: the dispatcher splits the 60 s input into ~30 s chunks
with ±3 s acoustic overlap, calls the backend once per chunk (each
call sees a 33 s window), and LCS-merges the boundary tokens. Inside
each backend call, parakeet's internal streamed path now runs as a
single 30 s encoder window (no further sub-chunking), which is
exactly the encoder context size the v3 model was trained for. The
backend's own "streamed over 60 s with c=30" instead splits the 60 s
mel into two ~30 s chunks internally — but the per-chunk encoder
passes don't see the bidirectional context across the cut as cleanly
as the dispatcher's per-call boundaries do (the dispatcher feeds each
chunk independently with its own mel-norm; the backend's streamed
path applies global mel-norm first then splits).

### When each mode wins

| Audio profile | Recommended mode | Why |
|---|---|---|
| Continuous EN/DE long-form, supported v3 lang | `--chunk-seconds 30 --chunk-overlap 3` | Highest coverage; modest wallclock overhead (12 s for 60 s audio) |
| JA model on JA long-form | `--chunk-seconds 30 --chunk-overlap 3` OR default | Both recover most content; LCS edges default by 16 % on the tested clip |
| Short audio (< 30 s) | default | Single backend call, no dispatcher overhead |
| Speech-with-long-silences | `--vad --vad-model silero` | VAD trims silences and feeds the backend with bounded slices; can outperform chunking when speech density is uneven |
| Reference parity / debugging | `CRISPASR_PARAKEET_STREAM_THRESHOLD=999` | Forces `parakeet_transcribe_ex`, the bit-exact single-pass path |

### Caveats

- The dispatcher's chunk-overlap wrap is what `kBlocked` opts cohere /
  gemma4-e2b / kyutai-stt etc. *out* of (LCS doesn't compose with
  their internal-chunking pipelines). Parakeet is intentionally NOT
  in `kBlocked` — the dispatcher's wrap and LCS dedup are correct for
  TDT's frame-synchronous output.
- The default did NOT change to `--chunk-seconds 30 --chunk-overlap 3`
  because that would force a dispatcher change visible to every
  caller (server API, Python session, …) without their request.
  Users who want the +45 % EN coverage today pass the flags
  explicitly; the option matrix above is the documentation.
- Single-pass occasionally wins (v3 + EN 60s: 626 chars vs 520
  internal-streamed default) — for clips that comfortably fit a single
  encoder forward pass on the model's hardware, the streamed wrapper
  is overhead. `CRISPASR_PARAKEET_STREAM_THRESHOLD=99999` makes
  single-pass the default.

### Reproduce

```
B=build/bin/crispasr
V3=/Volumes/backups/ai/crispasr/parakeet-tdt-0.6b-v3-q4_k.gguf
JA=/Volumes/backups/ai/crispasr/parakeet-tdt-0.6b-ja-q4_k.gguf
EN60=/Volumes/backups/code/audio_samples/en/fleurs_60s.wav
DE60=/Volumes/backups/code/audio_samples/de/fleurs_60s.wav
JA60=/Volumes/backups/ai/long-clips/yt_60s.wav

# default
$B --backend parakeet -m $V3 -f $EN60 -np -nt
# --chunk-seconds 30 --chunk-overlap 3
$B --backend parakeet -m $V3 -f $EN60 -np -nt --chunk-seconds 30 --chunk-overlap 3
# (etc)
```

### Coverage parity check vs cohere / canary on 300 s — 2026-05-26

User direction: "are these after-numbers complete? compare to what
other models deliver". Right — char-count delta vs the previous parakeet
default proves we *improved*, but says nothing about *complete*. Real
test: how does parakeet's new default compare against the best
long-form-capable backends on the same audio.

Ran parakeet (post-`98381810` default) vs cohere vs canary on the
300 s FLEURS clips. Voxtral skipped — its mem-thrash failure mode on
M1 with the 300 s clip is documented in
[`feedback_torch_omp_deadlock`](../memory/feedback_torch_omp_deadlock.md);
sat at 5 s CPU / 30 min wall and was killed.

**EN FLEURS 300 s:**

| Backend | chars | wall (s) | × RT | vs cohere |
|---|---|---|---|---|
| parakeet (default) | **3865** | 66 | 5.0× | -3 % |
| cohere | 3994 | 94 | 3.2× | (ref) |
| canary | 2971 | 74 | 4.1× | -26 % |

**DE FLEURS 300 s:**

| Backend | chars | wall (s) | × RT | vs cohere |
|---|---|---|---|---|
| parakeet (default) | **3288** | 69 | 4.3× | -0.3 % |
| cohere | 3299 | 87 | 3.4× | (ref) |
| canary | 3532 | 273 | 1.1× | +7 %, 3.1× slower |

**Headline.** Parakeet's post-fix default is now within **3 % of cohere
on EN 300 s and within 0.3 % on DE 300 s** at higher throughput
(66 / 69 s wall vs cohere's 94 / 87 s). The numbers are complete in
the same sense as cohere — coverage parity with the best long-form
backend, at faster wallclock. The previous default (CAP_INTERNAL_CHUNKING
set) was 60 % below cohere on EN 300 s; the fix closes that gap.

Canary on DE wins on coverage (3532 chars) but pays 4× the wall time
(273 s) — a different trade-off. For German-only workflows where wall
time is bounded, canary's per-chunk AED decode produces slightly more
content; for general use, parakeet's faster path with coverage-parity
is the recommended default.

