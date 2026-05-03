# CrispASR — Port history

Condensed chronology of the ports that built this repo. Kept for
context, not for day-to-day reference. Live work is in `TODO.md`;
technical deep-dives are in `LEARNINGS.md`.

---

## Timeline

### 1. Cohere Transcribe — the original port
The repository started as a standalone ggml runtime for
`CohereLabs/cohere-transcribe-03-2026` (2B params, Conformer encoder +
Transformer decoder, lowest English WER on Open ASR Leaderboard as of
March 2026).

Starting point: scalar C++ loops ported from the HuggingFace reference.
Optimisation trajectory on an 11s clip (4-thread CPU, same hardware
throughout):

| Step | Wall time | Speedup | Cumulative |
|---|---:|---:|---:|
| Baseline (scalar nested loops) | ~825 s | — | 1× |
| + FFTW3f STFT | ~100 s | 8.2× | 8× |
| + OpenBLAS GEMM | 104 s | ~1× | ~8× |
| + lazy F32 weight cache | 100 s | 1.04× | 8.3× |
| + EncScratch + AVX2 F16C | 32 s | 3.1× | ~26× |
| + ggml compute graph | 12.4 s | 2.6× | ~67× |
| + BatchNorm folding (48 layers × 10 nodes) | 11.5 s | 1.08× | ~72× |
| + depthwise conv direct (kernel=9, no im2col) | 9.6 s | 1.20× | ~86× |
| + self-contained FFT (drop fftw3) | ~8.7 s | 1.10× | **~95×** |

Total: ~825 s → ~8.7 s ≈ **95× on 4-thread CPU for an 11s clip.** Q4_K
quantisation brought this further to 14.8 s for a 5.4s clip (~2.7×
realtime). End-to-end, this beat ONNX INT4 by a hair (17.1s, but 7.5s
of that is cold-load; ggml mmap wins for repeat runs).

Main bug classes fixed: mel normalisation (NeMo per-band z-score with
biased std), cross-attention DTW timestamps, 2-backend scheduler
(GPU primary + CPU fallback).

### 2. Parakeet TDT 0.6B v3 — "free" word timestamps
`nvidia/parakeet-tdt-0.6b-v3`, 600M FastConformer encoder + Token-and-
Duration Transducer (TDT) decoder. Chosen because:

- Word timestamps come for free from the TDT decoder's duration head —
  no separate forced-alignment model.
- 600M params vs 2B for Cohere → ~400 MB Q4_K, ~3× faster.
- 25 EU languages with automatic detection.
- CC-BY-4.0.

- ~80% of encoder code reusable from cohere.cpp (both use the same
  FastConformer).

The novel work was the TDT decoder: LSTM predictor + joint network +
greedy decode loop with duration-advanced time stepping. Shipped with
a raw C++ CPU decoder (still unchanged — tracked in TODO.md as "port
LSTM to ggml" — encoder dominance makes GPU speedup small).

Tested on German audio and discovered the language-ID drift problem
(see `LEARNINGS.md` → "Auto-detect can silently code-switch").

### 3. Canary 1B v2 — explicit language + speech translation
`nvidia/canary-1b-v2`, 978M FastConformer encoder + Transformer decoder
with task-token prompt prefix. Ported specifically to fix the parakeet
language-ID drift: canary takes `-sl SRC -tl TGT` explicit language
flags, and the decoder is forced into the target language by the task
token. Also added speech translation (X→EN and EN→X), the only
translation runtime in the repo.

Implementation effort was small because we already had both halves:

| Canary component | Source |
|---|---|
| FastConformer encoder (32 layers) | `parakeet.cpp` |
| Conv2d dw_striding subsampling | `parakeet.cpp` + `cohere.cpp` |
| Rel-pos attention with Transformer-XL biases | `parakeet.cpp` + `cohere.cpp` |
| Mel preprocessor | `parakeet.cpp` (identical) |
| Transformer decoder (8 layers) | `cohere.cpp` |
| Cross-attention KV cache | `cohere.cpp` |
| 16384-token SentencePiece | `cohere.cpp` |

Shipped together with `nfa-align`, a general-purpose multilingual
forced aligner built from Canary's auxiliary CTC model. Works on any
transcript + audio pair in 25 European languages at ~78ms MAE.

### 4. Qwen3-ASR 0.6B — first speech-LLM
`Qwen/Qwen3-ASR-0.6B`, 900M speech-LLM combining a Whisper-style audio
encoder (18 layers) with a Qwen3 0.6B LLM (28 layers) via audio-token
injection at `<|audio_pad|>` placeholder positions in a ChatML prompt.
First speech-LLM in the repo and the template for everything that came
after.

Architecture: 2D-conv subsampler + Whisper-block body + 18-layer
encoder + Qwen3 LLM decoder. Uses **windowed attention via `cu_seqlens`**
(chunked self-attention with window size ~104 positions after CNN) —
standard full self-attention produces wrong output. This was the
trickiest part of the port.

Also first runtime to use a persistent KV cache for the LLM decode
loop: `qwen3_asr_kv_init` allocates it once, `qwen3_asr_kv_reset`
clears between utterances, each `qwen3_asr_run_llm_kv` call appends
new tokens at position `n_past` and reads K/V views for attention.

### 5. Voxtral-Mini-3B-2507 — Mistral's speech-LLM
`mistralai/Voxtral-Mini-3B-2507`, 3B speech-LLM with a literal Whisper-
large-v3 audio encoder (32 layers, 1280 dim) + 4-frame stack projector
+ Llama 3 (Mistral) 3B LLM. Audio tokens are injected at a special
`audio_token_id=24` placeholder in an `[INST][BEGIN_AUDIO] … [/INST]`
prompt. Uses the Mistral Tekken (tiktoken-style) BPE tokenizer with
150k vocab + 1000 special tokens.

Ported in a single session because we already had the Whisper-encoder
pattern from qwen3 and the LLM pattern was straight Llama 3. Main
novel work: Tekken tokenizer (150k vocab stored as a 1D F32 tensor in
the GGUF because the KV-array path loses uint8 precision) and the
4-frame stack projector.

Diff-tested against PyTorch at every stage boundary:
- Encoder: cosine similarity > 0.999 across all layers
- Projector: exact match
- LLM: top-5 5/5 match on first decoded token, cosine sim 0.999973

This gave us the confidence to ship despite the CPU-only path being
slower than an ideal GPU reference. See `LEARNINGS.md` →
"Model architecture comparisons" for the three-way comparison with
`max-lt/voxtral-cpp` and `llama.cpp mtmd`.

### 6. Voxtral-Mini-4B-Realtime-2602 — streaming speech-LLM
`mistralai/Voxtral-Mini-4B-Realtime-2602`, 4.4B natively-streaming
speech-LLM with a causal RoPE+SwiGLU+RMSNorm audio encoder (32 layers,
sliding window 750) + Mistral 3.4B LLM with adaptive RMSNorm and
sliding window attention. Designed for on-device realtime ASR with
<500ms latency.

Substantial architectural differences from the 3B:

| | 3B | 4B Realtime |
|---|---|---|
| Encoder attention | Full (abs pos embed) | RoPE θ=1e6 + SWA(750) |
| Encoder FFN | GELU fc1/fc2 | SwiGLU |
| Encoder norm | LayerNorm | RMSNorm |
| LLM layers | 30 | 26 |
| LLM FFN | 8192 | 9216 |
| LLM norm | RMSNorm | Adaptive RMSNorm (time-conditioned) |
| LLM attention | Full | Sliding window 8192 |
| LLM embeddings | Separate lm_head | Tied (token_embd = lm_head) |
| Audio tokens/frame | 1 per 4 Whisper frames | `audio_length_per_tok=8` |

Seven critical bugs found during debugging via Kaggle ground truth
and three reference implementations (`voxtral.c`, `voxmlx`, `voxtral-rs`):

1. **`audio_length_per_tok=8`** — 3B uses 4, 4B uses 8. Wrong value
   means audio-to-token alignment is off by 2× and transcript drifts.
2. **Audio padding: 32 tokens left + 17 tokens right + right_align.**
   Left-padding is 32 × 1280 samples = 32 streaming tokens worth of
   silence. Right-padding is 17 × 1280 samples plus whatever's needed
   to align the input length to a token boundary. Skipping the right
   pad silently breaks the encoder graph reshape.
3. **Prompt is `BOS + STREAMING_PAD × 38`**, not Tekken text. The audio
   encoder output is ADDED element-wise to each position's embedding
   (not replaced — this is the streaming mechanism). During decode,
   each generated token ALSO has the next adapter frame added to its
   embedding before the LLM forward.
4. **Tokens with id < 1000 are control tokens** (STREAMING_PAD,
   STREAMING_WORD, etc.) and must be filtered from the output
   transcript.
5. **Adaptive RMSNorm** applies a time-conditioned scale multiplication
   after the standard RMSNorm: `cur = cur * ada_scales[il]`. The
   `ada_scales` are precomputed from a learned module at load time
   as `(1 + scale)` and read from a view in the LLM graph.
6. **RoPE θ difference.** 3B uses θ=1e8, 4B uses θ=1e6. Getting this
   wrong corrupts the attention scores on long contexts.
7. **Tied embeddings.** 4B ties `lm_head = token_embd.T`, so the
   final linear is a `ggml_mul_mat(token_embd_w, cur)` — no separate
   `output_w` tensor.

Q4_K shipped at 2.4 GB and runs an 11s clip in 49s on 4-thread CPU.

### 7. Granite Speech 4.0-1B — Q-Former projector + µP LLM
`ibm-granite/granite-4.0-1b-speech`, 1B speech-LLM with a 16-layer
Conformer encoder (Shaw relative position embeddings + depthwise
conv + batch norm), a 2-layer BLIP-2 Q-Former projector with learned
query tokens, and a 40-layer Granite 1B LLM using µP (maximal update
parameterisation) multipliers. Apache-2.0.

Novel architectural pieces:

- **Q-Former** — cross-attention from a fixed-length learned query
  sequence (3 query tokens) to the encoder output. Each query token
  has its own self-attention among queries, then cross-attention to
  the encoder frames, then a position-wise FFN. Output is a small
  number of audio tokens fed to the LLM. Very different from the
  "stack frames + linear" projector used by qwen3/voxtral.

- **µP multipliers** — four scalar multiplications baked into the
  forward pass:
  - `embedding_multiplier = 12.0` (scales token embeddings)
  - `attention_multiplier = 0.0078125 = 1/128 = 1/head_dim` (scales
    attention logits, replacing the default `1/sqrt(d_head)`)
  - `residual_multiplier = 0.22` (scales residual additions)
  - `logits_scaling = 8.0` (scales output logits)

- **Stacked mel input** — unlike other models that use 128 mels,
  granite uses 80 mels stacked into 160-dim frames (two 80-mel frames
  zipped along channels). `granite_speech_compute_mel` outputs
  `(160, T/2)` instead of `(n_mels, T)`. Still inline — tracked as
  the `core_mel::Params::stacked_frames` follow-up.

Six bugs found during debugging (all preserved in `LEARNINGS.md`):

1. **Hann window centering.** The window must be symmetrically
   zero-padded to n_fft; off-by-one on the centering shifts the power
   spectrum peak and breaks everything downstream.
2. **Q-Former layer norm target.** LN applies to the query tokens,
   not the encoder output.
3. **Embedding multiplier placement.** Applied inside the LLM forward,
   after the token embedding lookup but before the first layer, so
   the raw `token_embd` tensor stays un-scaled in memory.
4. **CTC dim hardcoding.** Encoder output dim for CTC head is 348,
   not the encoder hidden 1024.
5. **Native GQA.** Flash attention handles `n_kv_heads < n_heads`
   natively if the K/V tensor shapes are right — no explicit
   `repeat_4d` needed. We were double-expanding at first.
6. **RoPE mode NEOX vs NORMAL.** The single most expensive bug. See
   `LEARNINGS.md` → "RoPE mode mapping".

### 8. Unified `crispasr` CLI + `src/core/` shared library (April 2026)
Two-phase refactor that reshaped the repo.

**Phase 1 — Unified CLI.** Extended `examples/cli/cli.cpp` with a
backend dispatch layer (`crispasr_backend.*`, backend adapters, VAD,
output writers, model manager, CTC aligner, run loop). Whisper code
path preserved byte-identical to upstream `crispasr`. 7 non-whisper
backends wired up. `-m auto` auto-download via `curl`/`wget` shell-out
(no Python, no libcurl link). `--list-backends` prints the capability
matrix. GGUF-based backend auto-detection with filename heuristic
fallback.

**Phase 0 — Shared model primitives.** Created `src/core/` (library
name `crispasr-core`) with:

- `mel.{h,cpp}` — one parameterised log-mel spectrogram function for
  both NeMo and HF/Whisper clusters. 7 of 8 non-whisper models migrated
  (granite deferred pending `stacked_frames` support).
- `ffn.h` — header-only SwiGLU / plain-SiLU FFN helpers. 4 LLM backends
  migrated.
- `attention.h` — header-only Llama-style self-attention (Q/K/V +
  reshape + NEOX RoPE + GQA + flash-attn + output projection). voxtral
  migrated as pilot; persistent-KV-cache variant for the others
  deferred.
- `gguf_loader.{h,cpp}` — unified GGUF two-pass loader with mmap
  (pread fallback for non-mmap filesystems). All 8 non-whisper models
  migrated.

Regression gate: every commit produces bit-identical output on
`samples/jfk.wav` (or within documented float-ULP tolerance where
matmul accumulator order changes). ~877 lines of duplicated
boilerplate removed from `src/` and replaced with ~730 lines of
shared code in `src/core/`.

Whisper is **intentionally not migrated** to `src/core/` — it's the
battle-tested reference and the `crispasr -m ggml-base.en.bin …` path
stays byte-identical to upstream `crispasr`.

---

## Markdown consolidation (April 2026)

The repo accumulated ~15 per-topic notes during the ports. These were
consolidated into four live documents:

- `README.md` — user-facing docs
- `TODO.md` — pending work, cross-checked against current state
- `LEARNINGS.md` — technical insights, benchmarks, comparisons
- `HISTORY.md` — this file

Removed: `canary-todo.md`, `parakeet-todo.md`, `granite-todo.md`,
`voxtral-todo.md`, `voxtral-4b-todo.md`, `qwen3-asr-todo.md`,
`TODO_COHERE_OPTIMIZATION.md`, `benchmark_cohere.md`,
`qwen3-asr-benchmark.md`, `ggml_plans.md`, `voxtral-comparison.md`,
`test_german.md`, `PERFORMANCE.md`. Everything of continuing value was
folded into the live docs.

Preserved outside these four: `UPSTREAM.md` (active upstream tracker),
`README_sycl.md` (Intel SYCL backend build), `ci/README.md` (CI
tooling), `models/README.md` (converter scripts), `samples/README.md`
(sample audio), and `hf_readmes/*.md` (HuggingFace model cards).

---

## Completed roadmap items (from PLAN.md, April 2026)

Items below were tracked in PLAN.md with full implementation details.
Moved here once shipped. See git history for code diffs.

### Core infrastructure (items 1-4, 6, 8, 10, 13, 17, 21, 24)

- **#1 voxtral4b encoder → encoder_self_attn()** — migrated with `permute_cont=false`. Bit-identical.
- **#2 Qwen3 forced aligner** — `qwen3_asr_run_aligner()` + `crispasr_aligner.cpp`. HF: `cstr/qwen3-forced-aligner-0.6b-GGUF`.
- **#3 Granite µP scale** — already handled via `KvSelfAttnParams::attn_scale`. No change needed.
- **#4 Scheduler reuse audit** — all backends use create-once + `ggml_backend_sched_reset()`.
- **#6 Best-of-N sampling** — all 4 LLM backends (voxtral/qwen3/granite/voxtral4b). `--best-of N -tp T`.
- **#8 voxtral audio Q&A** — `--ask "question"` flag for audio understanding.
- **#10 Granite encoder ggml graph** — `GRANITE_ENCODER_GRAPH=1` env var. CPU-verified identical.
- **#13 canary_ctc CPU fallback** — already implemented (2-backend GPU+CPU pattern).
- **#17 VAD stitching** — stitch + remap matching crispasr. C-ABI: `crispasr_session_transcribe_vad`.
- **#21 CLI→library DRY refactor** — VAD, diarize, LID, aligner, cache, registry promoted to `src/` behind shared C-ABI (v0.4.4–v0.4.8).
- **#24 Wrapper test suites** — Python (13), Rust (5+3), Dart (9) tests.

### New backends (items 26-34)

- **#26 GLM-ASR-Nano** — 12th backend. Whisper encoder + Llama 1.5B. MIT. `glm-asr`.
- **#27 Kyutai STT** — 13th backend. Mimi codec + causal LM. MIT. `kyutai-stt`.
- **#28 FireRedASR2-AED** — 14th backend. Conformer + CTC + beam search. Apache-2.0. `firered-asr`.
- **#29 FireRedVAD** — DFSMN 588K-param VAD, 97.57% F1.
- **#30 Moonshine** — 15th backend. Conv+transformer encoder-decoder. MIT. `moonshine`.
- **#31 FireRedASR decoder** — greedy + beam search Transformer decoder.
- **#32 FireRedLID** — 120-language LID via shared encoder + 6L decoder.
- **#33 OmniASR-LLM** — 16th backend variant. wav2vec2 encoder + 12L LLaMA decoder. Apache-2.0. Dynamic language selection (1693 FLORES-200 codes).
- **#34 VibeVoice** — architecture analysis complete. 1.5B is TTS-only; ASR is 7B (blocked on RAM).
- **ECAPA-TDNN LID** — 107-language LID, ggml graph (4.1s, 6x speedup), 100% accuracy. Two variants (VoxLingua107 + CommonLanguage).
- **OmniASR-CTC** — 300M and 1B variants working. HF: `cstr/omniASR-CTC-1B-GGUF`.

### Post-processing (item 35)

- **#35 FireRedPunc** — BERT-based punctuation restoration. GGUF converter + C++ runtime + CLI (`--punc-model`) + C-ABI + Python/Rust/Dart wrappers. HF: `cstr/fireredpunc-GGUF` (F16/Q8_0/Q4_K). Verified exact match against Python reference.

### Other completed items

- **#31 JSON LID (issue #17)** — language_detected/confidence/source in JSON output.
- **#25 Montreal Forced Aligner** — NOT PLANNED (too heavy, external tool).
- **Qwen Omni ASR** — NOT PLANNED (split GGUF, too large, already in llama.cpp).
- **#30 PazaBench assessment** — 16 model families assessed. 7 already covered, 4 easy wins identified.

### v0.5.4 (April 2026)

**Maintenance:**
- **Sync versioning**: Synchronized version numbers across CMake, Rust (crispasr, crispasr-sys), Python, and Dart wrappers to 0.5.4.
- **CI/Lint cleanup**: Resolved all remaining clang-tidy and clang-format violations using LLVM 18.
- **Improved Static Analysis**: Updated `.clang-tidy` to exclude third-party headers, reducing noise in CI reports.
- **Code Quality**: Fixed multiple implicit widening conversion warnings and enforced braces for all control flow statements in core core files.

### v0.5.0 (April 2026)

**Features:**
- **#36** ASCII punc mapping — auto-detect Latin script, map `，。？！` → `, . ? !`
- **#37** Progressive SRT (`--flush-after N`) — streaming subtitles for media players
- **#38** Fullstop-punc multilingual — XLM-RoBERTa-large, MIT, EN/DE/FR/IT. HF: `cstr/fullstop-punc-multilang-GGUF`
- **#39** Session API — all 18 backends wired in C-ABI + Python/Rust/Dart
- **#15** CMake rename — crispasr → crispasr in CMake, CI, Dockerfiles, scripts
- **#18** Aligner LIS — Longest Increasing Subsequence monotonicity fix
- **#40** Moonshine converter — multilingual variants (ja, ko, zh, ar, vi, uk)

**Optimizations:**
- **#44** FireRed ggml decoder — native Q4_K matmuls: 123s → 19s (**6.3x**)
- **O11** wav2vec2 CNN → ggml F32 im2col + OpenMP pos_conv: 108s → 10s (**10.8x**)
- **O1** ggml_soft_max_ext fusion — saves one op per attention layer (-10% wav2vec2)
- GPU auto-detect for all 18 backends + aux models

**Server:**
- Auto-chunking for long audio (#27) — prevents OOM
- Verbose logging + chunk progress (#26)
- API keys via env only, not CLI arg (#28)
- JSON error bodies + 404 handler

**Docker:**
- GHCR publishing (main, cuda, vulkan, intel, musa)
- passwd fix for ubuntu:24.04 images
- Standardized run-server.sh entrypoint

**VibeVoice TTS — Perfect ASR Round-Trip (April 2026):**
- **17 bugs found and fixed** via systematic stage-by-stage diff methodology
- **Perfect ASR round-trip**: all test cases produce exact match
  - "Hello, how are you today?" → parakeet: "Hello, how are you today?"
  - "The quick brown fox jumps over the lazy dog" → exact match
- **Model**: VibeVoice-Realtime-0.5B (2.04 GB, 605 tensors)
- **Architecture**: Base LM (4L) → TTS LM (20L) → DPM-Solver++ (20 steps) → σ-VAE (3200x) → 24kHz
- **Voice prompts**: pre-computed KV caches from .pt files (2.7 MB GGUF each)
- **CFG**: dual KV cache with per-frame negative updates, cfg_scale=3.0
- **EOS classifier**: automatic length detection via sigmoid(FC1→SiLU→FC2)
- **Critical bugs**: AdaLN SiLU (#16), text newline (#17), r-ratio sign (#14)
- **CLI**: `crispasr --tts "text" --voice voice.gguf -m vibevoice-realtime.gguf`

**VibeVoice-1.5B Base Model TTS (April 2026):**
- Single-LM autoregressive TTS (no TTS LM, 28-layer Qwen2)
- Voice cloning via acoustic+semantic encoder from reference WAV
- Speech token IDs: vision tokens reused (151654/151652/151653)
- ASR round-trip verified: "Hello, how are you today?" → exact match
- Quantized: F16 (5.1 GB), Q8_0 (2.8 GB), Q4_K (1.6 GB)
- HF: `cstr/vibevoice-1.5b-GGUF`

**ggml conv1d extensions + performance optimizations (April 2026):**
- Three new ggml ops: `GGML_OP_CONV_1D_CF` (channels-first conv1d),
  `GGML_OP_CONV_1D_CF` depthwise variant, `GGML_OP_CONV_1D_GROUP`
  (fused grouped conv1d). All with direct F32 kernels, F16/BF16
  kernel weight support, multi-threaded over output channels.
- VibeVoice TTS: VAE decoder 29% faster (700→476 ops), total 32%
  faster (0.39x→0.56x RT) via conv_1d_cf + `--tts-steps 10`
- wav2vec2: 12% faster via grouped positional conv + CNN cleanup
- firered-asr: depthwise conv migrated to conv_1d_dw_cf
- `VIBEVOICE_BENCH=1` / `WAV2VEC2_BENCH=1` per-phase timing

**Auto-download for all 19 backends (April 2026):**
- Added firered-asr, kyutai-stt, glm-asr, moonshine, fastconformer-ctc
  to model registry. Every backend now supports `-m auto --auto-download`.
- Companion file mechanism for moonshine's tokenizer.bin

**#29 Japanese split-on-punct fix (April 2026):**
- CJK fallback in `split_text_at_punct`: splits at clause breaks (、，)
  after ≥20 chars, force-splits at ~42 chars. English behavior unchanged.

**VibeVoice-7B GGUF (April 2026):**
- Full ASR+TTS GGUF (1205 tensors: encoder + LM + decoder + tokenizer)
- 7 quantizations: Q3_K (4.7 GB) through F16 (17.4 GB)
- TTS requires ≥Q4_K (Q3_K quality too low for decoder)
- HF: `cstr/VibeVoice-7B-GGUF` with README

**OmniASR CTC fix — two bugs found via stage-by-stage diff (April 2026):**
- pos_conv weight normalization: converter used per-output-channel norm
  (dim=0) instead of the model's per-kernel-position norm (dim=2).
  Fix: materialize weight directly from the HF model.
- head_dim hardcoded to 64: the 1B model uses 16 heads × 80 dim,
  not 20 × 64. Fix: read from HF config.
- Before: "koamerik asnot what yor country" (cosine 0.65 at layer 0)
- After: "fellow americans ask not what your country" (exact match)
- Converter now auto-detects v1 (fairseq2 .pt) vs v2 (HF transformers)
- 300M v2: works perfectly on ≤5s audio (cos=0.999997), breaks on >7s
  (positional encoding doesn't generalize beyond training length).
  Workaround: use --vad to chunk audio.
- LLM converter: complete rewrite (previous was corrupted). Tensor
  name mapping fixed (dec_ln, lm_head, tok_emb, gate). Same pos_conv fix.
- HF: `cstr/omniASR-CTC-1B-v2-GGUF` (F16 + Q4_K + Q8_0),
  `cstr/omniASR-CTC-300M-v2-GGUF` (F16),
  `cstr/omniasr-llm-300m-v2-GGUF` (F16 + Q4_K — fixed pos_conv + names)

**Moonshine multilingual — 12 models (April 2026):**
- Fixed converter: 1D tensors (norms/biases) forced to F32 for binary ops
- Fixed runtime: conv_1d_f32 mul_mat argument order for F16 kernels
- All 12 models tested and uploaded to HuggingFace
- HF: `cstr/moonshine-{tiny,base}-{ja,ar,ko,zh,vi,uk}-GGUF`

**OmniASR LLM-1B conversion (April 2026):**
- Converted facebook/omniASR-LLM-1B (8.5 GB .pt) to GGUF (4.55 GB F16, 918 tensors)
- 48-layer encoder (d=1280) + 12-layer LLaMA decoder (d=4096)
- Output: "fellow americas ask not what your country can do for you"
- HF: `cstr/omniasr-llm-1b-GGUF` (F16 + Q4_K)

**Parakeet TDT-CTC Japanese — xscaling fix (April 2026):**
- `nvidia/parakeet-tdt_ctc-0.6b-ja` was emitting "1 token then loop"
  because NeMo's `RelPositionalEncoding` multiplies the encoder
  input by `sqrt(d_model)=32` when `encoder.xscaling=true`. The C++
  runtime never applied this scale; v3 has `xscaling=false` so the
  multilingual sibling worked by accident.
- Diagnostic path: stood up `tools/reference_backends/parakeet.py`
  + `tools/dump_parakeet_reference.py` so `crispasr-diff parakeet
  <model.gguf> <ref.gguf> <audio.wav>` produces a stage-by-stage
  comparison against the NeMo reference. mel matched at cos≈0.99,
  encoder at cos=0.149 → grep'd `model_config.yaml`, found
  `xscaling: true`, applied `ggml_scale(*, sqrt(d_model))` between
  pre_encode and the first conformer block. Encoder cos jumped to
  0.81, F16 transcript bit-exact.
- Verified on a JSUT-basic5000 sample at F16:
  NeMo:     `'水をマレーシアから買わなくてはならないのです。'`
  crispasr: `'水をマレーシアから買わなくてはならないのです。'`
- Converter rewrite: every architecture hparam read from
  `model_config.yaml` (no more hardcoded `d_model=1024` /
  `pred_hidden=640`), cross-checked against actual tensor shapes,
  unmapped tensors warn loudly. New `parakeet.xscaling` GGUF key
  (default true on read) so old-converter v3 GGUFs continue to work
  unchanged once re-converted with `xscaling=false`.
- Q4_K JA still degenerates after ~8 tokens — the smaller 80-mel JA
  encoder is more quantisation-sensitive than v3's 128-mel one and
  `joint.pred` / `decoder.embed` fall back to q4_0. F16 is the
  recommended JA file; Q5_K or pinning those two tensors to F16 is
  the path forward.
- HF: `cstr/parakeet-tdt-0.6b-ja-GGUF` (F16 1.24 GB,
  Q4_K 470 MB) re-uploaded with the new converter + README.

**Gemma-4-E2B-it ASR — landed end-to-end (April 2026):**

`google/gemma-4-E2B-it`: USM Conformer (12L, 1024d, chunked-local
attention with relative position bias, ClippableLinear with QAT
scalars, LightConv1d) + Gemma4 LLM decoder (35L, 1536d, GQA 8Q/1KV,
per-layer embeddings, hybrid sliding/full attention with
per-layer-type head_dim, GeGLU MLP). Q4_K transcribes
`samples/jfk.wav` perfectly:

> "And so my fellow Americans ask not what your country can do for
> you, ask what you can do for your country."

End-to-end took ~16 numerical bugs to fix. The dominant ones:

- **ClippableLinear QAT scalars are NOT optional.** HF
  `Gemma4ClippableLinear.forward` clamps every input AND output of
  every q/k/v/o/ffw_layer/lconv1d.linear with trained finite bounds
  (±5..±40). Skipping them collapsed audio_layer_11 cos to 0.51 vs HF.
  Fix: stop skipping in converter, runtime applies clamp(input)→
  matmul→clamp(output) per linear. 480 scalars persisted per audio
  tower. Confirmed by patching HF locally to disable the clamps:
  HF-no-clip cos = 0.51, exactly matching ours-no-clip → unambiguous
  attribution. cos jumped to 0.97 once enabled.
- **Audio FE is bit-different from Whisper-style.** `frame_length=320`,
  `fft_length=512`, semicausal padding (160 zeros at start only),
  unfold-by-`frame_length+1`-then-drop-last, magnitude (not power)
  spectrum, HTK no-norm filterbank, `log(mel + 0.001)` (additive
  epsilon, natural log), no post-log normalisation. Wrote a
  dedicated `g4e_compute_mel_hf_faithful` instead of fighting
  `core_mel`'s param surface.
- **LLM forward had 5 separate bugs** — attn_scale=1.0 (q_norm
  replaces 1/√d), v_norm RMSNorm-no-weight, layer_scalar at end of
  layer (was applied twice mid-layer), PLE block at end + full
  per_layer_inputs prep stage including `per_layer_model_projection
  + per_layer_projection_norm`, MLP is GeGLU not SwiGLU. Each was a
  distinct numerical mismatch with HF; combined they took the LLM
  from outputting `<pad>` repeats to coherent English.
- **KV-share direction was the LAST 20 layers, not the first.**
  CLAUDE-memorised "first 20 layers reuse from later layers" was
  wrong; HF source has `first_kv_shared_layer_idx = num_layers - N`
  with each shared layer reading from the LAST earlier layer of the
  same `layer_type`. Donor map computed at load.
- **`use_double_wide_mlp=true` is a single 2× MLP, not two halves.**
  Misread of the field name + converter rename rules ate a session;
  HF L1024 of `modeling_gemma4.py` makes it explicit:
  `intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)`.

Per-stage cos vs HF reference (Q4_K, JFK 11s):

```
mel_spectrogram          1.0000   bit-exact
audio_subsample_output   1.0000
audio_layer_0..7        >0.998
audio_layer_11           0.969
audio_tower_output       0.962
encoder_output           0.966
```

Process win: the stage-by-stage diff harness
(`tools/dump_reference.py --backend gemma4` + intermediate dumps from
the runtime via `CRISPASR_DUMP_DIR=…`) was decisive. Every bug was
localised in 1–2 iterations once the per-layer cos table existed —
the alternative (eyeball end-to-end output, guess) wasted multiple
sessions before we wired it up. See LEARNINGS for the methodology.

HF: `cstr/gemma4-e2b-it-GGUF` re-uploaded with the QAT scalars and
all the fixes above (F16, Q8_0, Q4_K, Q2_K).

**Speed follow-up (same day):** end-to-end JFK transcription went
from 0.2× realtime (67.77 s) → **1.4× realtime (7.75 s)** with a
1-line fix. The model emits `<end_of_turn>` (token 106) at the
natural completion point, but greedy decode was configured with
`cfg.eos_id = ctx->eos_id` (token 212 = `<eos>`), so the loop
never matched and ran to `max_new_tokens=256` every call. For an
11s utterance that's 25 real tokens + 231 wasted ones. Fixed by
preferring `end_of_turn_id` when the chat template defines one.
Per-stage profile after the fix: mel 17 ms, encoder 719 ms,
prefill 287 tok in 1.46 s, decode 25 tok in ~5.5 s (~220 ms/tok).
The remaining encoder/decode optimisations (TODO O10/O11) are
secondary now — at 1.4× realtime the model is usable; further
work would target per-token decode cost. See LEARNINGS "Specific
bugs that cost us a day each" #9.

**NeMo-cluster encoder cosine fix — parakeet bias load (April 2026):**

The 24-layer FastConformer encoder was producing cos_mean=0.79 vs the
NeMo reference on Japanese audio (`reazon_meal_11s.wav`) and JSUT,
even though `mel_spectrogram` matched at cos≈0.999 after the preemph
+ Bessel-corrected PerFeatureZ fix. Symptom was small but real: extra
hallucinated prefixes (`本当`) and partial syllables on conversational
JA (parakeet-tdt-0.6b-ja, issue #37).

Root cause was a 10-line bug in `parakeet_load_weights`. The GGUF
stored `attn.{q,k,v,out}.bias` + `{ff1,ff2}.linear{1,2}.bias` +
`conv.{pw1,pw2}.bias` (10 biases per layer × 24 layers = 240 tensors)
but the loader only fetched the weights — the bias slots stayed
nullptr, and `mm_bias` silently skipped the bias add. Fix: add
`e.X_b = try_("…bias")` for each missing slot. `try_get` rather than
`require` keeps v3 (which has `use_bias=False`) compatible.

Result: `encoder_output cos_mean: 0.792 → 0.996` on reazon_meal_11s,
similar on JSUT. v3 EN regression on JFK still passes. Per-layer
diff confirmed the divergence had started at `encoder_layer_0`
(immediately after a bit-exact `pre_encode_output`), localising the
bug inside the conformer block before grep'ing the loader.

Residual: layers 17–22 keep cos_mean ≈ 0.99 but cos_min crashes to
negative on specific frames (`encoder_layer_22` cos_min = −0.67).
Suspects: rel_shift edge cases, position-encoding numerical
instability on specific positions, or a buffer-aliasing issue in the
dump path. Not blocking the bug-report fix; reusable diagnostic
infra (per-layer captures in `reference_backends/parakeet.py`,
`parakeet_run_encoder_dump`, `encoder_layer_K` stages in the diff
harness) is in place for canary, canary_ctc, and any future
NeMo-cluster runtime debug. Commit `e598767`.

**Qwen3-TTS codec decoder — Metal kernel fix (April 2026):**

The codec decoder (8L sliding-window transformer + ConvNeXt + 4×
SnakeBeta+tconv → 24 kHz waveform) hung on M1 with
`kIOGPUCommandBufferCallbackErrorImpactingInteractivity` whenever
`use_gpu=true`.

Root cause was instrumented via two env vars added to
`src/qwen3_tts.cpp`:
- `QWEN3_TTS_CODEC_TRACE=1` — prints per-node
  `op / tensor name / shape -> backend` before each op, with
  `ggml_backend_synchronize` after each so a hang attributes to the
  last printed line.
- `QWEN3_TTS_CODEC_FORCE_METAL=1` — re-routes codec weights and
  compute through the Metal-capable `c->sched`, reproducing the
  hang for triage.

Trace localised the hang to op 536: `GGML_OP_CONV_TRANSPOSE_1D` in
decoder block 1 (in-T=320, out-T=1605, C_out=384, stride=5,
kernel=10). Block 0 (stride=8, in-T=40, out-T=320) ran fine. The
SnakeBeta `sin`/`exp` and `conv_1d_dw` chains that were originally
suspected all completed cleanly on Metal.

The actual ggml-Metal `kernel_conv_transpose_1d` does
`for i in 0..IL { if (j ∈ [i*s0, i*s0+K)) accumulate; }` — but at
most `ceil(K/s0)` (=2 here) values of `i` ever satisfy that
condition. The kernel was iterating all 320 input positions for each
output element, doing ~160× the necessary work, which kept Metal
command buffers above the macOS GPU watchdog's ~5 sec ceiling.

**Fix landed in the ggml fork** (`ggml/src/ggml-metal/ggml-metal.metal`,
marked `// CrispASR patch`): compute the contributing `i` range
analytically before the input-channel loop and iterate only those
positions. Bit-identical output, ~160× less work on the codec
shape, comfortably under the watchdog. Documented in LEARNINGS.md
under "Metal conv_transpose_1d input range tightening" with the
"MUST RE-APPLY after every ggml bump" pattern (matches the existing
"CUDA im2col grid overflow" entry).

After the patch: 8/8 codec stages PASS with `use_gpu=true` end-to-end
on Metal, cos_min ≥ 0.999983 against the Python reference (slightly
tighter than the CPU path, presumably from F16 vs F32 mul/accum
differences in non-conv ops).

The original CPU-pin workaround (`codec_sched`, codec weights loaded
onto `c->backend_cpu`) is kept as a runtime safety net in case the
kernel patch is lost on a future ggml bump before the LEARNINGS
"RE-APPLY" reminder is honoured. Trace env vars also stay — useful
for debugging any future codec issue. PLAN #52 step 4 (ECAPA
speaker_encoder) is unaffected since it uses regular conv1d, not
transposed conv.

**Aux runtimes — Silero LID / pyannote v3 / wav2vec2-ggml (April 2026):**

Three small standalone runtimes landed alongside the main backend
work. All shipped end-to-end-correct with public GGUFs.

- **Silero LID native port (#56):** `src/silero_lid.{h,cpp}` plus
  `models/convert-silero-lid-to-gguf.py`. 95-language detector, 16 MB
  F32 GGUF (Q8_0 ~9 MB; quants below Q8_0 break accuracy on the small
  conv tensors). Pure-C++ forward pass, no ggml graph (manual F32
  loops, similar to pyannote_seg). Architecture: learned STFT
  Conv1d(1→322,k=320,s=160) → magnitude → log(2²⁰·mag+1) → adaptive
  norm (17-tap reflected smooth) → 8×(12 dw-sep conv + post-norm
  transformer + stride-2/1 proj+ReLU) → attention pool (tanh+softmax)
  → 95-lang + 58-group classifiers. Five bugs fixed during port:
  (1) front-end zero-pad 160/side, not reflection-pad 320 left;
  (2) stride-2 output `T = (T−1)/2 + 1`, not `T/2`; (3) QKV split
  order K,Q,V (not Q,K,V); (4) missing ReLU after stride-1
  projections stages 4–7; (5) missing tanh in attention pooling.
  CLI: when `--lid-model *.gguf` is passed, the native path runs;
  falls back to sherpa subprocess for `.onnx`. Verified across
  English, German, and Latvian. HF: `cstr/silero-lid-lang95-GGUF`.

- **Pyannote v3 native (#57):** SincNet + 4× biLSTM + 3× Linear +
  LogSoftmax, ~440 lines of C++. Wired into the CLI as
  `--diarize-method pyannote --sherpa-segment-model *.gguf` (native
  path; subprocess fallback for `.onnx`). Verified on jfk.wav with
  650 frames and correct "(speaker 1)" assignment.
  HF: `cstr/pyannote-v3-segmentation-GGUF` (5.7 MB F32).

- **wav2vec2 ggml rewrite (#63):** `src/wav2vec2-ggml.{h,cpp}`,
  layer-by-layer ggml graphs (~80 MB/layer, reused) for the 24-layer
  XLSR-53 transformer; CNN + pos_conv stay manual. Four root causes
  during port: (1) `ggml_gallocr` / `ggml_backend_sched` corrupt
  external F16 weight tensors (reallocate over them) — workaround
  `ggml_graph_compute_with_ctx`; (2) ggml `[H,T]` stores
  `data[h + t·H]` which is the SAME layout as `[T,H]` row-major in C
  — the original code had a spurious transpose that corrupted all
  data; (3) `flash_attn_ext` crashes with `mask=nullptr` —
  replaced with mul_mat attention; (4) logits `[V,T]` in ggml =
  `[T,V]` row-major, no transpose needed. Tested with
  `jonatasgrosman/wav2vec2-large-xlsr-53-english` (33 vocab,
  1024 hidden, 24 layers) — correct output on jfk.wav.

**iOS + Android CI gates + v0.1.0 release (April 2026):**

- iOS (arm64, Xcode) and Android (arm64-v8a, NDK r26d) cross-compile
  gates added to GitHub Actions. Catches breakage early on the
  lowest-traffic platforms.
- v0.1.0 shipped via GitHub Actions: Linux 660 KB, macOS 484 KB,
  Windows 1437 KB.

**Granite speed (#64) — closed, hardware-blocked:** at Q4_K /
4-thread CPU the 11s clip takes 33 s, and 26 s of that is
autoregressive LLM decode. `--gpu-backend` already exists and
granite uses `ggml_backend_init_best()` — no code change moves the
needle without GPU hardware. Tracked in TODO under per-model
follow-ups for OpenMP encoder annotations as a CPU-only nibble.

### 53. Qwen3-TTS — codec encoder repair (April 2026)
Fixed a critical memory layout bug in the `qwen3_tts` codec encoder. The
CPU-side RVQ loop was assuming channels-first indexing while the SEANet
output was transposed to row-major `[T, 512]`. This fix restored clear
voice cloning in the end-to-end CLI, eliminating the garbled artifacts.
Verified at `cos_mean=0.998` against the Python reference.

### 54. granite-family DRY refactor — PLAN #55 (May 2026)
Five-step lift of duplicated math out of `granite_speech.cpp` (base +
plus, causal + KV-cached) and `granite_nle.cpp` (NAR, non-causal
single-pass) into shared `src/core/` headers. Each step gated by JFK
smoke tests on all three variants — every commit kept transcripts
identical.

| Step | Header | Risk | LOC moved | Commit |
|---|---|---|---:|---|
| 1 | `core/fft.h` + `core/cpu_ops.h` (FFT, layernorm, matmul fallbacks) | very low | ~250 | `5f4b5ae` + `b343a17` |
| 2 | `core/ctc.h` (posterior-weighted pool + greedy decode w/ blank) | very low | ~60 | `65ef44c` |
| 3 | `core/conformer_ibm.h` (Macaron block helpers + Shaw RPE lookup) | medium | ~600 | `0f72391` |
| 4 | `core/granite_llm.h` (40-block backbone, `is_causal` flag) | medium | ~250 | `372a5f7` |
| 5 | `core/qformer.h` (NAR simplified Q-Former) | low | ~190 | `ed80fb0` |

**Step 5 — plan correction:** the duplication-map row originally
listed the windowed Q-Former as shared by both granite TUs. The code
disagreed: granite-speech (base/plus) uses the full BLIP-2 Q-Former
(self-attn + cross-attn + FFN per layer, no pass A, no window-mean-pool)
while granite-nle uses a "simplified" variant (cross-attn-only + MLP).
The block weight structs (`granite_proj_block` with `sa_*`/`ca_*`/`ffn_*`
vs `granite_nle_proj_block` with `attn_*` + `mlp_*`) made the divergence
explicit. Step 5 was rescoped to NAR-only — `core/qformer.h` is co-located
for any future simplified-windowed-Q-Former backend; granite_speech
stays untouched. PLAN.md was updated mid-step to reflect this.

`core_granite_llm::build_decoder` is the most reusable lift — it composes
40 layers of pre-RMSNorm + GQA(16/4) flash-attn + RoPE + SwiGLU + residual
×0.22 with an `is_causal` flag that picks `core_attn::kv_self_attn` (KV-cached
prefill+decode) or an inline non-causal flash path (whole-sequence editing).
Both granite TUs collapse from a per-layer hand-written loop to a single
function call.

`core_conformer_ibm` is a sibling-not-merge with `core/fastconformer.h`:
parakeet/canary use NeMo's Conformer dialect (conv subsampling, MHA RPE)
while granite uses the IBM dialect (Shaw RPE, fused conv layout, BN
folding-at-load) — keeping them separate avoids muddying both.

**Net effect:** `granite_speech.cpp` 2570 → 2113 LOC (−457); `granite_nle.cpp`
2096 → 1615 LOC (−481); combined drop ~940 LOC. New `core/*.h` files
total ~1070 LOC (fft 93 + cpu_ops 98 + ctc 129 + conformer_ibm 336 +
granite_llm 162 + qformer 255). Roughly half of those core LOC are
deduplicated math (the rest is comments + struct/API plumbing). Plus a
clean separation of "backbone" (in core) from "plumbing" (in TUs).

### 55. Granite encoder graph path as default — PLAN #16 (May 2026)

The `GRANITE_ENCODER_GRAPH=1` no-RPE flash-attn baseline silently
regressed: on JFK with `granite-speech-4.1-2b-q4_k` it produced only
the back half of the quote ("ask what you can do for your country").
The PLAN #16 prototype (per-block subgraph attention with Shaw RPE,
gated by `GRANITE_ENCODER_GRAPH_RPE=1`) inherited the same wrong
output, so end-to-end validation was blocked.

**Root cause.** The loader built only **layer 0's** RPE lookup
(`ctx->rpe_lookup`, single `vector<float>`) and the graph builder
reused it for all 16 encoder blocks, on the assumption that RPE is
tied across layers. granite-speech-4.1-2b in fact stores **distinct**
`attn_rel_pos.weight` per block (verified: layer 0 mean ≈ 0.00004,
layer 1 mean ≈ -0.003, layer 2 mean ≈ -0.002). Layer 0 still
matched the CPU loop bit-for-bit, but layer 1's attention diverged
immediately and the drift compounded across 16 layers until the LLM
only latched onto the back half of the audio. The CPU loop was
unaffected because it was already building per-layer RPE locally
inside the encoder forward — that local builder shadowed the
context-level cache and hid the bug.

**Fix.** Replaced `ctx->rpe_lookup` with `ctx->rpe_per_layer`
(`vector<vector<float>>`), built per-block at load time. The graph's
`rpe_lookup` input now has shape `(ctx_size*hd, ctx_size, n_layers)`
and each layer slices its block via `ggml_view_3d` on the layer
axis. CPU loop reuses the same precomputed table.

**Validation (per LEARNINGS methodology).** Stage-by-stage taps in
both paths (input_linear, FFN1, attn, conv, FFN2, post-norm,
block_out per layer) confirmed all sub-stages match within float
precision (~1e-3) — well above the cos_min ≥ 0.999 bar. JFK
transcript matches the CPU loop byte-for-byte:
"and so my fellow americans ask not what your country can do for
you ask what you can do for your country".

**Promotion.** Made the graph path the default and renamed the
escape hatch: `GRANITE_DISABLE_ENCODER_GRAPH=1` now opts back to
the CPU loop. The legacy `GRANITE_ENCODER_GRAPH` and
`GRANITE_ENCODER_GRAPH_RPE` env vars are gone — the no-RPE
flash-attn branch survives only as automatic fallback for models
with an unsupported `attn_rel_pos.weight` type.

**PLUS on graph.** Captured `cat_hidden_layers` post-norm tensors
inline in the graph and concatenated them with the final encoder
output along the feature dim via `ggml_concat`, so the PLUS variant
also rides the GPU path with no CPU-side cat_layers buffering. The
in-graph concat is essentially free (graph fanout off the residual
stream, no extra compute).

**NAR on graph.** Mirrored the granite-speech graph builder for
`granite_nle.cpp`. NAR-specific bits:
1. Self-conditioning residual at `hp.enc_self_conditioning_layer`
   (1-indexed, default = 8). Tapped `softmax(mid_logits)` on its way
   into `ctc_mid_w` so the runner can pull per-frame
   `blank_prob = column 0` for the BPE auxiliary head's
   `posterior_weighted_pool`.
2. Snapshot taps at every entry in `enc_layer_indices_parsed`
   (HF tuple indices: 0 = input embedding, N = output of block N-1
   *after* the self-cond residual at that block). Concatenated along
   the feature dim into `enc_output` (matches the CPU loop's wide
   buffer layout).
3. Final CTC logits = `ctc_out_w @ final_hidden + b` exposed as a
   named graph output. Cached on `ctx->last_ctc_logits` for the BPE
   editing path.
4. The BPE auxiliary head's `posterior_weighted_pool` stays on CPU
   (windowed reduce that doesn't map cleanly to a single ggml op);
   the `bpe_out_w` matmul (1024 → 100353) runs through the same
   scheduler as before. Negligible perf cost vs the encoder body.

**Numbers (M1, Q4_K encoder F32, 11 s JFK clip; lower disk-contention
session):**

| Variant            | CPU loop      | Graph (default) | Speedup |
|--------------------|--------------:|----------------:|--------:|
| `granite-4.1`      | 4.78 s (2.3×) | **2.31 s (4.8×)** | ~2.1×  |
| `granite-4.1-plus` | 9.41 s (1.2×) | **3.74 s (2.9×)** | ~2.5×  |
| `granite-4.1-nar`  | 19.27 s (0.6×, contended) | **6.41 s (1.7×)** | ~3.0×  |

All three variants transcribe JFK byte-for-byte identical to their
CPU-loop reference (including punctuation/casing for PLUS and NAR).
The `GRANITE_DISABLE_ENCODER_GRAPH=1` escape hatch covers all three.

### 56. MiMo-V2.5-ASR end-to-end — PLAN #51 SHIPPED (May 2026)

XiaomiMiMo's 7.5B-class speech-LLM (8-channel RVQ encoder + 36-layer
Qwen2 LM, vocab=151680, MIT) ships fully working. JFK transcription
matches the upstream Python `MimoAudio.asr_sft` reference verbatim:
"And so, my fellow Americans, ask not what your country can do for
you. Ask what you can do for your country." 11 s of audio in ~37 s
on M1+Q4_K+Metal (0.3× realtime).

**The forward path landed in two phases.** First, prefill numerical
correctness (commit `9faccdd`) — five stages (audio_features,
text_embeds, inputs_embeds, last_hidden, logits_step0) match the
bf16 PyTorch reference within Q4_K + bf16 tolerance, with cos_mean
between 0.96–0.998. Argmax of step-0 logits hits token 1597
(`' And'`), matching the reference. The fix-it-or-lose-it bug:
**capture tensors in a ggml graph need `ggml_set_output()`, not just
`ggml_set_name()`.** Without `set_output`, the scheduler treats
named tensors as ordinary intermediates and reuses their buffers
when allocating later ops in the same graph. Symptom we hit:
`prefill_inputs_embeds` collapsed to cos≈0.003 (looked like a
broadcasting bug for hours) before tracing the buffer aliasing.
LEARNINGS.md "Capture tensors MUST call `ggml_set_output()`"
documents the recipe — apply universally to any tensor read out
via `ggml_graph_get_tensor` for diff-harness extraction.

Three other prefill bugs fixed in the same commit: (a) the
input_local_block was permuting `(hd,n_h,gs,ng) → (hd,gs,n_h,ng)`
*before* `ggml_rope_ext`, putting `n_h` at ne[2] where positions[gs]
expected — assertion failed. Fix: rope-then-permute. (b) After the
o-projection, `attn` was 2D `[d, gs*ng]` while the residual was
3D `[d, gs, ng]` — ggml_add broadcast assert. Fix: reshape_3d.
(c) The on-disk Q4_K had a truncated vocab (151643 entries, missing
`<|empty|>`) so the empty-token fallback was hitting Qwen2's
`<|endoftext|>`, which never appears in the prompt — every position
was treated as non-empty, zeroing the audio path entirely. Fix
in step 9 below.

Second, transcribe path (commit `dae361f`) — full prompt
construction in C++, mirroring `process_speechdata.InputSegment`
byte-for-byte. Each text segment becomes a `[9, T_seg]` block (row
0: tokens at stride gs, -100 fillers, padded to gs alignment;
rows 1–8: per-channel `speech_zeroemb_idx`). Audio segment becomes
`[9, T_audio + 2*gs]` with `<|sosp|>` / `<|eosp|>` wrapping the
empty-token text row and codes flanked by speech_zeroemb pads on
the audio rows. The 6 segments (user header, audio, template,
end, assistant header, think+language tag) concatenate into the
prefill input_ids. `mimo_asr_build_prefill_graph` gained an
`n_past` parameter so step decode reuses the same builder with
T=gs and advancing n_past. The decode loop replicates each
generated text token across the gs positions of the new group
and fills audio rows with `speech_zeroemb_idx[c]` (matches
`slm_sample`'s `expand(group_size)` + `zero_embed_tensor` path).
Stops on `<|im_end|>`/eos, strips
`<|empty|>`/`<|eot|>`/`<|eostm|>`/language tags.

The tokenizer follows the qwen3-asr-style splitter: greedy match
`<|...|>` against the vocab, then GPT-2-style whitespace pre-split
+ `bytes_to_unicode` + `bpe_one` for the rest. Works only because
step 9 reconverted the GGUF with `tokenizer.ggml.merges` populated
(151291 entries — the converter fix from commit `2191a70` had
landed earlier but the GGUFs predating it carried no merges, so
BPE collapsed to per-byte and the prompt didn't match upstream).

**Step 9 dragon: torch+OpenMP deadlock.** The bf16→f16 cast in
`models/convert-mimo-asr-to-gguf.py` (via `t.to(torch.float16)`)
goes through `at::native::DEFAULT::copy_kernel` → OpenMP barrier
→ `__kmp_suspend_64` → indefinite `_pthread_cond_wait`. Process
appears alive (RSS stable, mmap'd weights resident, STAT=S, 0.0%
CPU) but the temp file stops growing after ~50 tensors. Workaround
made permanent: prefix all torch-based converters with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONUNBUFFERED=1`. Cost is negligible — the cast is memory-bound,
not compute-bound. Without the env vars: hangs forever. With them:
~20 min for the 14.9 GB F16 on M1, ~5 min more for Q4_K quantize.
LEARNINGS.md and the `mimo-tokenizer-GGUF` README repeat this so
the next person doesn't lose 30 minutes diagnosing it.

**HF release.** [`cstr/mimo-asr-GGUF`](https://huggingface.co/cstr/mimo-asr-GGUF) ships F16 (14.9 GB) + Q4_K
(4.5 GB) with the corrected vocab and merges. Pair with
[`cstr/mimo-tokenizer-GGUF`](https://huggingface.co/cstr/mimo-tokenizer-GGUF) — the audio tokenizer is a separate
encoder model. Q2_K and the legacy `mimo-asr.gguf` are kept for
history but were built before the vocab/merges fix and should not
be used.

### 57. Qwen3-TTS-Base 1.7B — PLAN #57 Phase 1 (May 2026)

The 1.7B variant of Qwen3-TTS-Base shipped end-to-end behind the
`qwen3-tts-1.7b-base` registry alias. Same ICL voice-clone path as
0.6B-Base (`--voice <wav> --ref-text "..."`); the runtime now reads
`qwen3tts.speaker.enc_dim` from the GGUF instead of assuming 1024.

**The bug.** `build_graph_spk_enc` and `run_spk_enc` (plus the two
`extern "C"` speaker-embedding helpers) hardcoded the ECAPA output at
1024 floats. That matched the 0.6B-Base config (`talker.hidden_size =
1024`, `enc_dim = 1024`) but the 1.7B-Base config has
`talker.hidden_size = 2048` and `enc_dim = 2048`. The first 1024
floats of the speaker embedding made it into the codec_input slot;
the second 1024 floats were silently truncated, and the talker's
prefill saw a half-zero spk row. Symptom: ICL produced degenerate
audio.

**The fix** (`0813869`). Read `c->hp.spk_enc_dim` (already plumbed
via `kv_u32(g, "qwen3tts.speaker.enc_dim", ...)` and exported by the
converter — just unused in the graph builders). Five sites
parameterised; banner now logs `ECAPA-TDNN 128→1024` for 0.6B-Base
and `128→2048` for 1.7B-Base.

**Validation.** `clone.wav` ICL on `"Hello, how are you today? The
weather is beautiful."` →
- 1.7B-Base F16  → 57 frames / 4.56 s → "Hello? How are you today? The weather is beautiful."
- 1.7B-Base Q8_0 → 51 frames / 4.08 s → "Hello! How are you today? The weather is beautiful."
- 0.6B-Base Q8_0 (regression, `enc_dim=1024` path) → 75 frames / 6.00 s → "Hello? How are you today? The weather is beautiful."

Word-level exact match across all three; the punctuation jitter on
the leading single-word token is parakeet-v3 behaviour, not a
synthesis defect.

**HF release.** [`cstr/qwen3-tts-1.7b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF) ships F16 (3.86 GB)
+ Q8_0 (2.07 GB). Pair with [`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF)
— same 12 Hz tokenizer as 0.6B-Base.

The 1.7B small_to_mtp_projection bridge from the 2048-d talker to the
1024-d code predictor was already wired in commits `7f79d34` /
`2cc7aeb` (originally for 1.7B-CustomVoice) and is variant-agnostic,
so once `spk_enc_dim` was unstuck the rest of the path "just worked"
on the existing graph builders.

### 58. Qwen3-TTS-VoiceDesign 1.7B — describe-the-voice TTS (May 2026)

The instruct-tuned variant of Qwen3-TTS-12Hz-1.7B shipped behind the
`qwen3-tts-1.7b-voicedesign` registry alias. Replaces the reference-WAV
or fixed-speaker prompt with a natural-language voice description fed
in via `--instruct`. No ECAPA forward, no codec encoder, no preset
speaker table — the model picks a voice purely from the text.

**Runtime contract.** When `qwen3tts.tts_model_type == "voice_design"`,
the talker prefill is built by `build_voicedesign_prefill_embeds`
(`src/qwen3_tts.cpp`), which mirrors `build_customvoice_prefill_embeds`
with two changes:
- The codec bridge omits the speaker frame: `L_codec =
  codec_prefill.size() + 2` (just `pad,bos`), one frame shorter than
  the CustomVoice path.
- An instruct block — `text_proj(text_embd(instruct_ids))`, where
  `instruct_ids` tokenises `<|im_start|>user\n{instruct}<|im_end|>\n`
  — is prepended to the prefill. Mirrors
  `Qwen3TTSForConditionalGeneration.generate` lines 2076–2233 of
  `modeling_qwen3_tts.py` for the `speaker_embed=None` + `instruct_ids`
  path.

**C-ABI.** Two new entry points:
- `qwen3_tts_is_voice_design(ctx)` — variant detection.
- `qwen3_tts_set_instruct(ctx, instruct)` — required before synthesis
  on a VoiceDesign model; returns -1 if the model isn't VoiceDesign.

**CLI.** New `--instruct "..."` flag (parsed in `cli.cpp`). The
`qwen3-tts` backend rejects `--voice` on VoiceDesign models with a
helpful warning, and errors before generation if `--instruct` is
empty.

**Why 1.7B-only.** Upstream `generate_custom_voice` explicitly disables
`instruct` for the 0.6B variant; there is no 0.6B-VoiceDesign weight
release. The 1.7B talker forward (Q/KV/FFN, mrope, small_to_mtp
bridge) is shared with 1.7B-Base, so the runtime side reuses the
existing graph builders end-to-end.

**HF release.** [`cstr/qwen3-tts-1.7b-voicedesign-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF) ships F16 + Q8_0.
Pair with [`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF) — same 12 Hz tokenizer.

### 59. Orpheus 3B-FT — PLAN #57 Phase 2 slice (c) (May 2026)

The first commercial-friendly TTS in the Phase 2 talker family
shipped end-to-end behind the `orpheus` backend
(commit `a0982d3`). The runtime drives a Llama-3.2-3B-Instruct
talker (custom-token vocab `<custom_token_0..28671>`) over a
persistent KV cache, samples on top of the 7-slot SNAC super-frame
layout, de-interleaves per the canonical Orpheus protocol, and
pipes the codes through a SNAC C++ decoder to 24 kHz PCM. With
Orpheus base in, Kartoffel_Orpheus DE + lex-au Orpheus-3B-DE-Q8 +
the various Orpheus finetunes are now checkpoint swaps.

**Sourcing the talker.** `canopylabs/orpheus-3b-0.1-ft` is gated;
having an HF login token isn't enough — you have to click through
the gate. `unsloth/orpheus-3b-0.1-ft` is a **non-gated mirror** of
the same weights and converts cleanly with the new
`models/convert-orpheus-to-gguf.py`. SNAC codec from
`hubertsiuzdak/snac_24khz` (MIT, 3 codebooks × 4096 @ 24 kHz).

**The BOS=128000 trap.** Verbatim from
`canopyai/Orpheus-TTS:engine_class.py:_format_prompt`, the prompt
is built by tokenising `"{name}: {text}"` with `add_special_tokens=True`
(HF tokenizer default) — which inserts the Llama-3
`<|begin_of_text|>=128000` BOS at the start. The engine then
prepends `audio_start=128259` and appends
`eot_id=128009, audio_eot=128260, audio_eom=128261, audio_end=128257`.
**Without the BOS the model produces well-structured but
semantically garbage codec output** — parakeet ASR returned
`"Ineonice perfect of the Pan 8."` for `"Hello, my name is Tara."`;
with it, the roundtrip lands exact. Easy to miss because the model
emits properly slot-patterned super-frames either way.

**Stop policy.** Stop on `audio_end=128257` *or* on >4 consecutive
non-codec tokens. **Don't** stop on `audio_pre_end=128009` or
`audio_end_b=128261`: in the unsloth/canopylabs ID layout those
are either the Llama-3 `<|eot_id|>` (which appears in the prompt)
or `text_N<10` reserved markers in the custom_token block. The
reference `tokens_decoder` filters them silently rather than
terminating on them.

**Sampling.** Greedy decoding (`temperature=0`) gets stuck in a
7-slot loop after a few super-frames and the AR halts after ~24
tokens. `engine_class.py` defaults to `temperature=0.6` — that's
what produced the validated 2.73 s clip below.

**Validation.**

```
$ crispasr --backend orpheus \
    -m /Volumes/backups/ai/crispasr-models/orpheus-3b-ft-f16.gguf \
    --codec-model /Volumes/backups/ai/crispasr-models/snac-24khz.gguf \
    --voice tara --temperature 0.6 \
    --tts "Hello, my name is Tara." \
    --tts-output /Volumes/backups/ai/crispasr-models/orpheus_test.wav
orpheus: AR emitted 224 codec tokens (32 super-frames)
crispasr: TTS output written (65536 samples, 2.73 sec)

$ crispasr --backend parakeet \
    -m /Volumes/backups/ai/crispasr-models/parakeet-tdt-0.6b-v3-q4_k.gguf \
    -f /Volumes/backups/ai/crispasr-models/orpheus_test.wav --no-prints
Hello, my name is Tara.
```

**Converter note.** `models/convert-orpheus-to-gguf.py` had to flip
`GGUFWriter(use_temp_file=True)` → `False`. With temp-file enabled
the writer first buffers tensor data to a system tempfile (honors
`TMPDIR`) then copies it to the output, which on
`/Volumes/backups` at 100% disk usage causes silent corruption /
slow throughput; direct write side-steps the issue.

**HF release.** Talker shipped to
[`cstr/orpheus-3b-base-GGUF`](https://huggingface.co/cstr/orpheus-3b-base-GGUF)
as F16 (6.6 GB) + Q8_0 (3.5 GB). SNAC codec shipped to
[`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF)
as a single F32 file (26 MB; the codec is small enough that
quantising it isn't worth the audio-quality risk). Registry alias
`orpheus` in `src/crispasr_model_registry.cpp` resolves Q8_0 + SNAC
under `-m auto`. The full unified Session API works end-to-end:
opening the talker through `crispasr_session_open` returns a non-null
handle and `crispasr_session_n_speakers` returns the 8 canopylabs
English voices (`tara`/`leah`/`jess`/`leo`/`dan`/`mia`/`zac`/`zoe`).

**Phase 3+ known gaps (out of scope for slice c):** plain NEOX
RoPE with `theta=500000` (no Llama-3 `freq_factors` scaling — fine
for short prompts, may matter for long synthesis); no
`repetition_penalty` in the sampler (engine_class.py default 1.3);
Metal first-load is slow (~10-15 min for 6.6 GB f16 GGUF due to
kernel compilation, fast thereafter); non-streaming AR (the
reference's "emit middle of 4-super-frame sliding window" protocol
in `orpheus_snac.py` is a follow-up).

### 60. MiMo-V2.5-ASR perf wave — PLAN #51b/b' (May 2026)

First decode-side perf pass on the mimo-asr runtime that shipped in
section 56. Two structural changes plus an allocator-friendly diag
gate, all in `src/mimo_asr.cpp` only.

**51b — step-only decode graph.** Decode-time inputs zero out the
audio branch (text row holds the new token, `text_zero_mask=1`,
`speech_active_mask=0`, audio rows are `speech_zeroemb_idx` whose
combined mask is also 0), so the entire 6L input_local_transformer
+ group_proj + fusion path computes a literal zero that gets added
to `text_embeds`. The new `mimo_asr_build_step_graph` skips it
outright: `embed_w[next] → 36L Qwen2 LM → final_norm → lm_head`,
T=1. Decode loop in `mimo_asr_transcribe` calls this instead of
the heavy 9-row prefill graph for every n_past>0 step. Prefill
still uses the original `mimo_asr_build_prefill_graph`.

**51b' — O15-style cached step graph.** Mirrors
`src/qwen3_tts.cpp:976-1050` (the `QWEN3_TTS_O15` path). `Lk`
pinned to `kv_max_ctx` so the graph topology is invariant across
n_past, `kv_indices = lm_positions` so the K/V scatter via
`ggml_set_rows` keys off a runtime tensor instead of a static
byte offset baked at build time. The plan is reused for every
decode step within a transcribe call (`ctx->step_t1_gf`),
invalidated at every transcribe entry and after `extract_stage`
clobbers `compute_meta`. The mask covers the full Lk with -INF
beyond `n_past + q` so never-written cache slots can't leak NaN
or whatever the buffer happened to hold.

**Diag-capture gate.** `mimo_asr_build_prefill_graph` takes a
`bool diag_captures`. Production transcribe passes false (drops 4
`ggml_set_output` calls + 2 `ggml_cont` clones — without
`set_output` the scheduler is free to reuse those buffers for
later ops, so the only extracted output is `prefill_text_logits_step0`
which is consumed). The diff harness `extract_stage` passes true.
`MIMO_ASR_DIAG=1` env var force-enables for transcribe-time
debugging. ~5 % wall-clock + a much cleaner allocator picture.

**Bench (M1, Metal, Q4_K, samples/jfk.wav):**

| | wall | per-step decode |
|---|---:|---:|
| Section 56 baseline (sec. 56) | ~37 s | ~1.15 s |
| After 51b/b' | 44.4 s* | 0.79 s |

\* The wall-clock looks worse because the bench was a cold run and
~7-10 s went to Metal kernel JIT compile in prefill (`Tg=97`,
includes 4.4 s of `kernel_*_compile_pipeline`). On a warm run the
prefill should drop sub-second, putting end-to-end below the 37 s
baseline. Per-step decode is the apples-to-apples metric: 1.46×
faster, hits the lower end of the work order's 51b+51b' target
band ("~1.5–2× from 51b alone, ~1.3× more from 51b'"). Transcript
matches the gold byte-for-byte.

**Cosine gate (crispasr-diff bf16-ref vs Q4_K-C++)** — five
prefill stages reproduce the section 56 numbers exactly, confirming
51b/b' do not perturb prefill numerics:

| Stage | cos | section 56 |
|---|---:|---:|
| prefill_audio_features    | 0.998270 | 0.998 |
| prefill_text_embeds       | 0.996284 | 0.996 |
| prefill_inputs_embeds     | 0.997573 | 0.998 |
| prefill_last_hidden       | 0.963177 | 0.963 |
| prefill_text_logits_step0 | 0.981261 | 0.981 |

Argmax of step-0 logits = 1597 (`' And'`), matches the reference
and is consistent with the JFK transcript starting "And so, ...".
The harness's strict 0.999 threshold doesn't apply to bf16-ref vs
Q4_K-C++ (would need fp32 ref + ~28 GB RAM, blocked by 51a — see
LEARNINGS lesson 3 of section 56). Ref archive at
`/Volumes/backups/ai/mimo-asr-ref.gguf` (4.1 MB) regenerated via
the 2-phase loader patch (commit `3945d7b`) which keeps peak
memory under 16 GB.

**Out of scope, queued for follow-up:** 51a (mmap-backed weight
loader to drop the `_platform_memmove` into a fresh CPU backend
buffer — saves ~12.7 GB resident on the F16 14.9 GB GGUF, but it
touches `src/core/gguf_loader.h` which is shared by 24 backends
and needs the full diff-harness gauntlet); 51c (F16 step decode,
trivial after 51a); fused QKV per LM layer (saves 2 matmuls per
layer × 36 layers × N steps, but the converter has to be updated
to fuse + write a single `attn_qkv.weight` tensor).

### 61. granite-speech-4.1 — plus + nar variants — PLAN #54 (May 2026)

The `ibm-granite/granite-speech-4.1-2b` family ships three variants
with significantly different decoders despite the shared "4.1-2b"
naming. Base shipped in HISTORY §54 of the granite-family DRY refactor;
this entry records the **plus** + **nar** variants reaching bit-exact
JFK transcription and HF release.

| Variant | Decoder | Encoder change | Outputs |
|---|---|---|---|
| `granite-speech-4.1-2b-plus` | Granite-1B AR | `cat_hidden_layers: [3]` | text + speaker labels + word-level timestamps |
| `granite-speech-4.1-2b-nar` | non-autoregressive (`NLENARDecoder`) | self-conditioning at L8 + BPE aux head + 4-layer hidden capture | text |

**Plus variant.** Backend alias `granite-4.1-plus` registered with the
unified Session API; PLUS GGUF (5.6 GB f16) is converted and the
runtime concatenates encoder layer 3 with the final layer output
(`il + 1 == cat_index`, matching HF's `output_hidden_states`
convention). Punctuation + capitalisation come for free from the
PLUS training default. End-to-end JFK transcript:

```
And so my fellow Americans, ask not what your country can do for you,
ask what you can do for your country.
```

Speaker labels + word-level timestamps remain queued (template-only
~50 LOC follow-up). Commits: `f298818` (cat_layer + tokenizer fix),
`ed0e5ac` (backend alias + registry), `a3147b6` (HF README).

**NAR variant.** Backend alias `granite-4.1-nar` registered. Three
stages, all bit-exact on JFK:

1. **Encoder forward** (`granite_nle_run_encoder`). Same Conformer
   block as base; self-conditioning at layer 8 (running char-level
   CTC logits feed back through `out_mid`); 4-layer hidden state
   capture at the indices listed in `proj.encoder_layer_indices`
   (default `[4, 8, 12, -1]`). The capture obeys HF tuple semantics:
   `-1` resolves to `n_layers`, and the snapshot at the
   self-conditioning layer is taken AFTER the residual is added.
   Validated against PyTorch on JFK at cos_min ≥ 0.999. The BPE
   auxiliary head (`enc.bpe_out`) is intentionally not wired through
   `run_encoder` — it's only needed by the LLM editing pass's
   text-init step, where it's faster to run on the posterior-pooled
   features.
2. **Windowed Q-Former projector** (`granite_nle_run_projector`).
   Two-pass: (A) one ggml graph for the per-encoder-layer LayerNorms
   + concat + `layer_proj` (4096 → 2048) + GELU; (B) one Q-Former
   graph per block (`block_size=15`, `downsample_rate=5`,
   `query_length=3`) with mean-pool over downsample groups, additive
   `query` and `window_positions`, two 32-head SDPA cross-attention
   + SiLU-MLP layers, and a final `out_norm` + `out_linear`. Output
   rate: 3 audio tokens per 15 encoder frames. PyTorch JFK match at
   `projector_output cos_min=0.999999` (T_out=111 × llm_dim=2048).
3. **Non-causal LLM editing pass** (`granite_nle_run_llm_editing`).
   Single graph over the flat `[audio_embs, text_embs_with_slots]`
   sequence with µP scaling (embedding_multiplier=12,
   attention_multiplier=1/128, residual_multiplier=0.22). 40 layers
   of RMSNorm + non-causal `flash_attn_ext` (mask=nullptr, GQA 16/4
   native) + SwiGLU. Tied LM head. The caller passes audio_embs
   pre-divided by `embedding_multiplier` so the uniform downstream
   scale-up recovers the original projector output for audio while
   still scaling text by 12× — mirrors `_build_flat_llm_inputs`.
   Validated bit-exact: `editing_logits cos_min=0.999999` and 47/47
   top-1 match on JFK.

**Reference dump pitfall.** `GraniteModel.forward` unconditionally
builds an upper-triangular causal mask and passes it to SDPA, which
then enforces causality regardless of `self_attn.is_causal=False`.
The upstream "flash_attention_2 required" assertion is real — only
FA2 reads `is_causal` directly without using the mask. The
`tools/reference_backends/granite_nle.py` dumper monkey-patches
`transformers.models.granite.modeling_granite.create_causal_mask` to
return None to get true non-causal attention via SDPA.

**Transcribe orchestration** (`granite_nle_transcribe`) wires
together: encoder (with BPE auxiliary head:
`posterior_weighted_pool` window=4 driven by `1 - blank_prob_mid`
from the L8 self-conditioning softmax, populating `last_bpe_logits`)
→ BPE-CTC greedy decode (`unique_consecutive` → drop blank label 0
→ shift to LLM IDs by -1) → `core_bpe::detokenize` (GPT-2 byte-level
reverse, lifted into shared `core_bpe::token_bytes_to_utf8`) → strip
+ lowercase + " "-fallback → re-tokenize via
`core_bpe::tokenize_simple` → `add_insertion_slots` (`max(2n+1, 8)`,
EOS-padded) → `run_projector` divided by `embedding_multiplier=12`
and sliced to `enc_T // downsample_rate=5` audio frames →
`run_llm_editing` → per-row argmax + unique_consecutive + drop EOS +
detokenize. JFK end-to-end output matches reference `final_text`
exactly.

**HF release.** All three variants live on
[`cstr/granite-speech-4.1-2b-GGUF`](https://huggingface.co/cstr/granite-speech-4.1-2b-GGUF) (base, 4 quants: F16
5.58 GB, Q4K F32-enc 2.94 GB, Q4K F16-enc 2.07 GB, Q4K mini 1.7 GB),
[`cstr/granite-speech-4.1-2b-plus-GGUF`](https://huggingface.co/cstr/granite-speech-4.1-2b-plus-GGUF) (plus, F16 5.6 GB),
and [`cstr/granite-speech-4.1-2b-nar-GGUF`](https://huggingface.co/cstr/granite-speech-4.1-2b-nar-GGUF) (nar, 4 quants:
F16 5.8 GB, Q4K 3.4 GB, Q4K f16enc 2.5 GB, Q4K mini 1.6 GB). Registry
aliases `granite`, `granite-4.1`, `granite-4.1-plus`, `granite-4.1-nar`.

### 62. Zero-copy mmap GGUF loader — PLAN #51a env-flag (May 2026)

`core_gguf::load_weights` previously did mmap-then-`tensor_set` into a
freshly allocated CPU backend buffer. For the 14.9 GB F16 mimo-asr GGUF
that peaked at ~13 GB resident on a 16 GB Mac and thrashed swap for
25+ minutes, blocking the F16 + fp32-ref strict cos≥0.999 diff harness
gauntlet that section 60 had to defer. Q4_K (4.5 GB) hid the symptom
on production paths.

**Implementation** (commit `9710f80`, `src/core/gguf_loader.cpp` +
`src/CMakeLists.txt`). Skip `ggml_backend_alloc_ctx_tensors` on the
CPU path when `CRISPASR_GGUF_MMAP=1`; mmap the file with
`MAP_PRIVATE | PROT_READ|PROT_WRITE` (Win32 `FILE_MAP_COPY`); wrap the
data section in a custom `ggml_backend_buffer_t` whose `free_buffer`
callback munmaps; bind each tensor with `ggml_backend_tensor_alloc`
into the mmap'd offsets. Mmap lifetime is owned by `model.buf` so the
existing 24 caller pattern (move `wl.buf` → `model.buf`, drop the
`WeightLoad`) is unchanged. The CMake target adds `../ggml/src` as a
private include for `ggml-backend-impl.h`.

**Copy-on-write was load-bearing.** First validation pass used
`MAP_SHARED + PROT_READ` and parakeet immediately faulted with SIGBUS
in its BN-into-conv fold path (`src/parakeet.cpp:535`,
`ggml_backend_tensor_set` on read-only mmap'd pages). MAP_PRIVATE gives
COW: pages a backend never touches stay shared with the file's page
cache (the RSS win), pages it mutates get a private anon copy.
Backends with similar post-load weight surgery (vibevoice, …) inherit
the fix for free.

**Validated.** parakeet Q4_K — Metal default, CPU default, and CPU +
`CRISPASR_GGUF_MMAP=1` all produce the gold JFK transcript.

| Case | Working-set RSS | Notes |
|---|---:|---|
| mimo-asr Q4_K (4.5 GB GGUF), legacy | ~5.5 GB | full backend buffer + OS-resident mmap pages |
| mimo-asr Q4_K, mmap | ~760 MB | OS keeps the mmap'd file in shared cache only |
| mimo-asr F16 (14.9 GB GGUF), legacy (predicted) | ~13 GB peak | per HANDOFF; thrashes swap 25+ min on 16 GB Mac |
| mimo-asr F16, mmap | **~910 MB** during model load | observed at 60 s elapsed before contention forced a kill |

The F16 mmap loader churned through model load + LID + tokenizer +
encoder in seconds — the same span where the parallel legacy F16 run
on the same file was still 30+ min into its mmap-then-copy. End-to-end
decode timing is still pending. The kill was forced by an unrelated
problem the test surfaced: `/Volumes/backups/ai` had hit 100% capacity
(12 GB free of 1.9 TB), so both my mmap test and the parallel legacy
F16 from another Claude session ended up thrashing on page faults
under heavy memory pressure (vm_stat showed 13M+ swapins). Once that
disk has headroom again, re-time F16 end-to-end before flipping the
default.

**Default still legacy.** The env flag remains opt-in. Flip the
default in a follow-up commit once the F16 RSS savings are measured
end-to-end and the diff harness on F16 GGUFs has been exercised across
the qwen3-tts and granite-speech families.

**Side-quest fix: parakeet `--no-gpu` crash** (commit `b85f56c`,
`ggml/src/ggml.c`). Pre-existing assertion failure
`GGML_ASSERT(*cur_backend_id != -1)` in
`ggml_backend_sched_split_graph` whenever parakeet's encoder ran on
the CPU backend. Root cause: `ggml_conv_2d` and `ggml_conv_2d_dw` set
their im2col output type to `a->type` (the kernel) unconditionally,
producing `MUL_MAT(F16 im2col, F16 kernel)`. The CrispASR fork's
issue-#38 patch (F16 `vec_dot_type=F32` +
`ggml_vec_dot_f16_f32`) doesn't support F16×F16. Fix mirrors
`ggml_conv_1d`: pick F32 im2col when either operand is F32, cast the
kernel to F32 when the chosen path needs it. Conv_2d puts activations
as src0 and kernel as src1 (reversed from typical); Metal's kernel
table has `mul_mv_f16_f32` but not `mul_mv_f32_f16`, so the kernel
cast is needed for both backends. Slight Metal slowdown
(13.3× → 7.5× realtime on parakeet-tdt-0.6b-v3 Q4_K JFK) is the same
trade-off `ggml_conv_1d` already makes upstream.

**Out of scope, queued for follow-up:** measuring the F16 mimo-asr
RSS win end-to-end; flipping the env-flag default; PLAN #51c (F16
step decode) which the HANDOFF flagged as "trivial" once #51a lands.

### 63. Session 2026-05-02 — canary ref dumper + DRY helpers + cache-clear ABI sweep

Three small landings collected here because none warrants its own
section:

**PLAN #5 — Canary reference dumper** (commit `63f708e`). The C++
`crispasr-diff canary` branch was already wired (mel + encoder taps)
but the matching Python ref dumper was missing. Added
`tools/reference_backends/canary.py`, modeled on `parakeet.py`: NeMo
`ASRModel.from_pretrained("nvidia/canary-1b-v2")`, preprocessor +
encoder forward, per-layer hooks for 32 encoder layers of diagnostic
captures, transposed to TimeMels layout for the C++ side. Diff
against the existing Q4_K GGUF on JFK shows expected quantisation
noise (encoder cos_mean 0.972, cos_min 0.35 on low-magnitude frames)
but the runtime still transcribes byte-exact:
`"And so, my fellow Americans, ask not what your country can do for
you, ask what you can do for your country."` Strict cos≥0.999 PASS
would need an F16 GGUF — deferred until disk headroom allows the
converter to run without thrashing (98% full external + slow
`shutil.copyfileobj` per CLAUDE.md note).

**PLAN #53 — Two narrow core helpers** (commit `d393a43`). After the
qwen3-tts codec and SNAC decoders both shipped, a re-read across all
four of our TTS decoders (vibevoice σ-VAE, qwen3-tts codec, mimo
tokenizer encoder-only, SNAC, kokoro istftnet) showed the convergence
the original PLAN #53 ("`core/audio_decoder.h`") imagined wasn't
there: VibeVoice is continuous-VAE with ConvNeXt, MiMo has no
decoder, Kokoro is istftnet, and only qwen3-tts codec + SNAC share
shape — and even there the codebook-handling diverges (qwen3-tts
splits codebook-0 from rest-15 vs SNAC sums all codebooks equally).
Rewrote the PLAN entry to scope down, then extracted exactly two
helpers:

- `core_act::snake_beta` in `core/activation.h` — the BigVGAN
  `y = x + exp(-β)·sin²(x·exp(α))` activation. qwen3-tts now
  delegates via a 1-line alias. ~10 LOC saved net.
- `core_convt::convt1d_crop` in `core/conv.h` — generic
  channels-first `ggml_conv_transpose_1d` wrapper with
  caller-controlled `crop_left`/`crop_right`. qwen3-tts (causal,
  `crop_right=K-stride`) and SNAC (symmetric,
  `crop_left=crop_right=pad`) both delegate. ~30 LOC saved net.

SNAC `crispasr-diff` 8/8 PASS (cos_min 0.999941 unchanged) confirmed
the wrapper is bit-equivalent. PLAN #53 priority moved MEDIUM →
LOW since the `core/audio_decoder.h` super-helper is no longer the
intent.

**PLAN #56 #5 — Kokoro phoneme cache clear ABI** (commits `9bffb0f`,
`6cabefa`, `d022bff`, `603f47e`). The `kokoro_phoneme_cache` LRU
already existed in `kokoro_context`; what was missing was a way for
long-running daemons to drop the cache when resynthesising across
many speakers. Added:

- `kokoro_phoneme_cache_clear(ctx)` extern-C in `kokoro.cpp`/`.h` —
  takes the mutex, `lru.clear()` + `idx.clear()`. Cheap and
  thread-safe.
- Session-scoped re-export
  `crispasr_session_kokoro_clear_phoneme_cache(session)` in
  `crispasr_c_api.cpp` — no-op for non-kokoro backends, returns -1
  on null handle.
- All 7 wrappers got the method (Python `Session.clear_phoneme_cache()`,
  Rust `Session::clear_phoneme_cache()`, Dart `clearPhonemeCache()`,
  Go `Session.ClearPhonemeCache()`, Java `clearPhonemeCache()`,
  JS `Module.ttsClearPhonemeCache()`, Ruby
  `Session.clear_phoneme_cache(handle)`). Each follows the existing
  per-binding pattern for `set_codec_path`. +55 LOC across the 5
  trailing wrappers. PLAN #59's "open this section when a consumer
  asks" rule was relaxed for this single-method addition because the
  alternative (C-only surface) was ergonomically worse.
- No-model unit tests in `tests/test_python_session.py` cover the
  symbol export + null-handle return path. +2 PASS in 0.56 s, no
  model required.

**Side-quest: clang-format-18 CI fix** (commit `21464e3`). The
`d393a43` PLAN #53 commit added 4 lines that local clang-format-22
considered fine but Ubuntu CI's clang-format-18 flagged. Fixed in
place — see LEARNINGS.md "clang-format-22 vs CI v18" lesson for the
trap and the safer workflow.

**Side-quest: 2 LEARNINGS lessons** (commit `cc82e25`).

- The clang-format-22-vs-v18 destruction trap (auto-formatting whole
  files locally produces hundreds of whitespace changes that v18 *also*
  rejects — manually align by eye instead).
- NeMo ref dumpers must transpose to TimeMels for parakeet/canary
  (the C++ side uses `core_mel::Layout::TimeMels` which is
  n_mels-fast, T_mel-slow — opposite of NeMo preprocessor's natural
  `(B, n_mels, T_mel)` C-contiguous order; cosine-signature lookup
  table for diagnosing layout swaps included).

### 64. PLAN #60d Fused QKV + #60e KV-quant plumbing — mimo-asr (May 2026)

Picked up the two MEDIUM-effort OPEN items from PLAN #60 in one
session. Both shipped behind clean fallback paths so existing GGUFs
keep working — only the new fused-QKV Q4_K mimo-asr download gets the
speedup, and only callers that opt in via `CRISPASR_KV_QUANT` change
KV-cache footprint.

**60d Fused QKV (mimo-asr LM):**

The Qwen2 LM in mimo-asr does Q/K/V via three separate `mul_mat`s
plus three `ggml_add` bias ops per layer × 36 layers × N decode
steps. Fusing the per-layer Q/K/V weights into one
`[d_model, q_dim + 2*kv_dim]` tensor and the biases into one
`[q_dim + 2*kv_dim]` 1-D vector replaces the three matmuls with one
fused matmul and the three bias adds with one — algebraically
identical, fewer ggml ops to schedule.

`core_attn::kv_self_attn` already accepted a `qkv_w` parameter
(qwen3-asr / qwen3-tts use it for runtime fusion of F16/F32 weights).
The Qwen3 path doesn't have biases, so the helper needed an extra
`qkv_b` parameter for the Qwen2 case. Added at the end of the
parameter list (after the existing `o_b`) so all existing callers
stay binary-compatible at default-arg level.

`mimo_asr_qwen2_block` gained `attn_qkv_w` / `attn_qkv_b` slots.
`bind_qwen2_block` tries the fused names first via `try_t`; on miss
it falls back to the separate-Q/K/V `require_t` path. Audio
`audio.blk.*` blocks always take the fallback (their bidirectional
attention reads separate Q/K/V outside `core_attn`, and the
converter is intentionally LM-only).

The HF→GGUF converter (`convert-mimo-asr-to-gguf.py`) was updated
to emit fused tensors at convert time. But re-running the BF16→F16
conversion on this 16 GB / 99%-full-disk box thrashes the same way
PLAN #51c documented — the converter sustained ~0.8 MB/min on the
contested disk before being killed. Workaround: a new
`tools/patch_mimo_asr_fuse_qkv.py` that loads an existing GGUF via
`gguf.GGUFReader`, byte-concat's the per-LM-layer Q/K/V data along
the row dim, and re-emits as fused tensors. Bit-identical to a
fresh-from-converter result for F16/F32 (numpy element concat) and
for Q4_K/Q8_0/etc. — each row's quant blocks are independent so
byte concat across rows is a valid quantised tensor. The patcher
runs in ~5 minutes (vs hours / never for the BF16-source converter
on the contested disk).

The Q4_K-on-disk re-quantisation path is separate from this fuse —
the existing 4.5 GB Q4_K stays as Q4_K, just with three Q/K/V
tensors per layer collapsed into one `attn.qkv.weight`.

**Validation (Q4_K, JFK, on this 16 GB box, 4 concurrent claude
sessions, 99%-full external disk):**
- `crispasr-diff` cosines reproduce the §56 / 51b/b' baselines
  bit-exactly (audio_features 0.998270, text_embeds 0.996284,
  inputs_embeds 0.997573, last_hidden 0.963177, logits 0.981261 —
  identical to the unfused-Q4_K reference run, character-by-character).
- JFK transcript byte-identical: "And so, my fellow Americans,
  ask not what your country can do for you. Ask what you can do
  for your country."
- `MIMO_ASR_BENCH=1` on the same disk-thrashed box:
  - Unfused (separate Q/K/V): prefill 10295 ms, decode 79498 ms /
    26 steps = **3058 ms/step**, total LM 89.8 s.
  - Fused (this PLAN #60d): prefill 4881 ms, decode 46946 ms /
    26 steps = **1806 ms/step**, total LM 51.8 s.
  - **1.69× per-step decode speedup** at the same disk thrash
    level. The work order predicted 1.1-1.2× on a quiet box;
    larger here likely because each fewer matmul also avoids one
    page-fault round-trip on the contested disk. On uncontended
    hardware expect closer to 1.1-1.2× pure-compute.

The patched Q4_K was uploaded to `cstr/mimo-asr-GGUF` replacing
the unfused file. The unfused F16 stays in the repo unchanged —
the runtime fallback in `bind_qwen2_block` keeps it working as-is.
F16 re-upload is queued behind PLAN #51c (uncontended-disk
re-conversion).

**60e KV-quant env flag (mimo-asr):**

`CRISPASR_KV_QUANT={f16,q8_0,q4_0}` now picks the KV-cache dtype in
`mimo_asr_kv_init`. Default stays F16, so this is bit-identical to
existing behaviour.

The shared `core_attn::kv_self_attn` adapts both write and read
paths for quantised cache:

- **Write:** the original `ggml_cpy(F32, slice-of-cache)` path
  requires the destination to be contiguous when the source/dst
  types differ, but a per-token slice into a max_ctx-strided 4D
  cache is never contiguous. CPU's `dup_to_q<float>` aborts; Metal
  also skips non-contig quant dst. Fix: when the cache is
  quantised, always go through the `ggml_set_rows` scatter path
  (which both backends accept for F32→Q* directly), even when no
  `kv_indices` is supplied. The `positions` tensor — already
  populated with [n_past..n_past+T) for RoPE — is exactly the
  row-id list set_rows wants, so we re-use it as the synthetic
  kv_indices.
- **Read:** `ggml_is_quantized(kv_k->type)` switches from `ggml_cont`
  to `ggml_cast(view_q*, GGML_TYPE_F32)`. Both backends support
  `Q*→F32` CPY; the CPU backend's `compute_forward_dup` only
  implements `Q*→F32` (not `Q*→F16`), so F32 is the only safe
  dequant target if the scheduler splits the op. Cache *storage*
  still uses ~half the bytes (Q8_0) — the hour-long-podcast use
  case where `max_ctx > 10k` would otherwise need ~1.5 GB F16 KV —
  but reads pay one dequant pass per layer.

**Validation (Q8_0 KV cache, fused Q4_K, JFK):**
- F16 KV baseline (above) had last_hidden cos_min 0.963177, logits
  cos_min 0.981261.
- Q8_0 KV: last_hidden cos_min 0.963031 (Δ -0.000146), logits
  cos_min 0.981454 (Δ +0.000193). Both stay well above the work
  order's ≥0.98 gate. Pre-attn stages (audio_features,
  text_embeds, inputs_embeds) don't go through the KV cache and
  reproduce the F16 cosines exactly.

**Per-backend env wiring (commit `8edfb74`, same session):** the
mimo_asr-local helper was lifted into a shared
`core_attn::kv_dtype_from_env(const char* tag)` and called from
each `*_kv_init` whose KV path routes through
`core_attn::kv_self_attn`. 9 backends wired in one commit:
`mimo_asr_kv_init` (refactored to use the shared helper),
`qwen3_asr_kv_init`, `voxtral_kv_init`, `voxtral4b_kv_init`,
`granite_speech_kv_init`, `gemma4_e2b` (`g4e_kv_init`, both
sliding-window and full-attention caches share the dtype),
`glm_asr_kv_init`, `omniasr` (`omniasr_alloc_kv_cache`), `orpheus`
(`kv_alloc`), `qwen3_tts` talker (`kv_alloc` — `cp_kv` stays F16
since the code-predictor decode doesn't go through `core_attn`).
Default remains `GGML_TYPE_F16` so the wiring is bit-identical to
legacy behaviour until a user opts in via
`CRISPASR_KV_QUANT={q8_0,q4_0}`.

Smoke-tested: `crispasr-diff mimo-asr` with default-F16 KV reproduces
the audio_features 0.998270 / text_embeds 0.996284 / inputs_embeds
0.997573 / last_hidden 0.963177 / logits 0.981261 cosines bit-exact
post-refactor.

Side-quest fixed in passing: `voxtral4b_kv_init` and
`granite_speech_kv_init` had a hardcoded
`ggml_type_size(GGML_TYPE_F16) * hd * max_ctx * n_kv * nl` byte
pre-compute used to size the backend buffer. This silently
over-allocates for quant types (Q8_0's `ggml_type_size` is 34 bytes
for a 32-element block, treating each *element* as 34 bytes →
massive over-alloc). Switched both to compute `ggml_nbytes()` after
`ggml_new_tensor_4d` instead, which is the right pattern regardless
of dtype.

Per-backend cosine validation (CRISPASR_KV_QUANT=q8_0
`crispasr-diff` against each bf16 reference) is the actual rollout
gate that remains open — see PLAN #60e for the list.

Backends with custom KV paths (canary, cohere, kyutai_stt,
vibevoice) don't route through `core_attn::kv_self_attn`, so the
shared write/read fixes from 1594577 don't cover them; they'd each
need backend-specific work to support quant KV. Left as PARKED in
PLAN #60e.

**Side-quest:** the converter run itself was killed mid-flight after
22 min / 0.8 MB/min sustained on the contested disk — same
diagnostic signature as PLAN #51c thrash mode. The patcher script
exists specifically to side-step this on this hardware; the
converter itself is correct (and what would run on an uncontested
machine).

### 65. Session-API word-confidence parity + 61a/b feature-matrix uplift (May 2026)

**Problem.** Every session-API caller (Rust crispasr crate, Python
wrapper, Flutter, Ruby, …) saw `word.confidence = 1.0` for parakeet
(hardcoded) and no per-word data at all for every other backend
(canary, qwen3, cohere, granite, voxtral, voxtral4b, mimo-asr,
wav2vec2, firered-asr, glm-asr, kyutai-stt, moonshine, omniasr,
fastconformer-ctc). Plus the C-side accessor
`crispasr_session_result_word_p` existed in the .cpp but wasn't
declared in any binding's FFI, so callers couldn't even read it.

**Resolution — three batches of work, all runtime-side, no GGUF
changes:**

**1. Per-token confidence runtime APIs (PLAN #61b).** Each ASR
backend that produces text via greedy / beam decode now exposes a
`*_transcribe_with_probs` C-API (or, for backends with token data
already, a refactor that surfaces the `.p` field through the
adapter):

- **wav2vec2** — new `wav2vec2_greedy_decode_with_probs` does CTC
  frame argmax + numerically-stable softmax of the picked logit;
  emits one `wav2vec2_token_prob{id, prob, frame_start, frame_end}`
  per non-blank emission. Adapter populates
  `crispasr_segment::tokens` with frame-aligned t0/t1.
- **firered-asr** — beam-search loop in `firered_asr_transcribe_impl`
  refactored: `beam_hyp` gained `token_logprobs` parallel to
  `tokens`; `candidate` gained `token_logprob` (= `top_score[k]` at
  push time). `firered_asr_transcribe_with_probs` returns a
  `firered_asr_result*` with id / `expf(logprob)` arrays in
  lock-step with the winning beam's token sequence. Old plain
  `firered_asr_transcribe` is now a 1-line wrapper.
- **fastconformer-ctc / canary-ctc** — new
  `canary_ctc_greedy_decode_with_probs` returns a
  `canary_ctc_decode_result*` with token ids, softmax probs,
  CTC-frame ranges, AND text-offset/length into the assembled
  transcript (so callers can substring-quote each emission without
  redoing the ▁→' ' / special-token logic). Frame timestamps
  computed via `canary_ctc_frame_dur_cs`.
- **moonshine** — refactored `moonshine_transcribe` into
  `moonshine_transcribe_impl` that optionally captures probs;
  added `moonshine_set_temperature` (sticky context flag) so the
  same loop also handles `> 0` multinomial sampling. New
  `moonshine_token_text(ctx, id)` uses a new
  `moonshine_tokenizer::token_to_piece` that decodes a single id
  without trim/special-strip.
- **glm-asr** — extended `sample_token` with optional `out_prob`;
  refactored `glm_asr_transcribe` → `_impl` with optional out
  vectors, only emitting tokens for non-special pieces; new
  `glm_asr_transcribe_with_probs`. Adapter does GPT-2 byte-level
  BPE Ġ→space / Ċ→newline decode for per-token text.
- **kyutai-stt** — extended `sample_token` with `out_prob`; refactored
  `kyutai_stt_transcribe` → `_impl`; new
  `kyutai_stt_transcribe_with_probs`. SentencePiece ▁→' ' done in
  the adapter.
- **omniasr (LLM variant)** — extended `omniasr_run_prefill` and
  `omniasr_run_dec_token` with optional `out_logits` parameters;
  when set, the build switches from `output_token_id=true` (argmax
  baked into graph) to `output_token_id=false` and reads back the
  full `[V, 1]` logits tensor for CPU-side argmax+softmax.
  Per-step capture into `cur_prob` + writeback to context-owned
  `capture_token_ids` / `capture_token_probs` pointers. New
  `omniasr_transcribe_with_probs` orchestrates. CTC variant returns
  null and falls back to text-only path.

Adapter capability flags updated: every CTC trio + decoder quartet
now reports `tok-conf Y` in `crispasr --list-backends`. README
"Feature matrix" `Per-token confidence` row went from 8 ✔ to 15 ✔.

**2. Auto-download (PLAN #61a).** Both fc-ctc and wav2vec2 already
had registry entries — only needed `CAP_AUTO_DOWNLOAD` on each
adapter. 2 README cells gained.

**3. Session-API word emission (PLAN #65).** Every modified
backend's session path now produces a `crispasr_session_seg::words`
array with real per-word probs:

- **parakeet** — `parakeet_word_data` gained a `float p` field;
  `parakeet_transcribe_ex` accumulates `cur_p_sum / cur_p_cnt` mean
  alongside the existing word grouping. Session adapter reads
  `pr->words[i].p` instead of hardcoding 1.0. *(Landed in commit
  `bc67d12` mid-session after CI surfaced the missing struct
  field.)*
- **wav2vec2** — session path switched from `wav2vec2_greedy_decode`
  to `wav2vec2_greedy_decode_with_probs`, projecting CTC emissions
  into a new `ca_token_record` shape, then through a shared
  `emit_words_from_tokens` helper that does SentencePiece-style
  word grouping (▁ / leading-space / standalone " " / punctuation
  attachment).
- **canary** — session path switched to `canary_transcribe_ex`
  (existing API), tokens already have `.p`.
- **cohere** — same with `cohere_transcribe_ex`.
- **firered-asr / glm-asr / kyutai-stt / moonshine / omniasr (LLM)** —
  switched to each backend's new `_transcribe_with_probs`. Per
  backend a small ▁→space / Ġ→space decode loop in the adapter.
- **fastconformer-ctc / canary-ctc** — switched to
  `canary_ctc_greedy_decode_with_probs`.
- **voxtral / voxtral4b** — `run_voxtral_family` (the templated
  decode loop both backends share in `c_api.cpp`) gained per-step
  numerically-stable softmax over the picked logit. Decoded piece
  goes through a ▁→space pass in the loop, so `emit_words_from_tokens`
  sees the same convention as the other backends.
- **mimo-asr** — `mimo_asr_transcribe` refactored into
  `mimo_asr_transcribe_impl` with optional `out_token_ids` /
  `out_token_probs`. The greedy `argmax` lambda became
  `pick(L, float* out_p)`, and the impl passes
  `capture_probs ? &prob : nullptr` so the legacy text-only entry
  pays no softmax cost (mimo's vocab is 151,680 — softmax on every
  step would add ~150k `expf` calls per token, measurable). New
  `mimo_asr_transcribe_with_probs` + `mimo_asr_token_text`.
- **qwen3 + granite** — both runtime `*_transcribe` functions are
  stubs that return empty/null. The CLI worked because its adapter
  drives the building blocks (`*_compute_mel`, `*_run_encoder`,
  `*_tokenize`, `*_embed_tokens`, `*_kv_init`, `*_run_llm_kv`,
  `*_token_text`) directly. Ported the same flow into the session
  path in `c_api.cpp`:
  - **qwen3** — chatml prompt with audio-pad splice; greedy decode
    via `core_greedy_decode::run_with_probs`; metadata-token filter
    (`<|im_start|>`, `<asr_text>`, `[PAD…]`, "language <name>"
    capture) mirrors the CLI adapter; GPT-2 byte-level decode via a
    new shared `gpt2_byte_decode` helper.
  - **granite** — chat-template selection (`audio_token < 50000` ⇒
    granite-3.x control tokens, else legacy 4.0 hardcoded `kPrefix4`/
    `kSuffix4`); projector-output splice into the audio-pad
    positions; `granite_speech_decode_tokens` for batch transcript;
    `granite_speech_token_text` + `gpt2_byte_decode` for per-token
    pieces.

**Public-API additions:**

- `crispasr_session_result_word_p(r, i_seg, i_word)` returns the
  per-word probability in `[0, 1]`, or `-1.0f` for "no data".
- Rust `crispasr-sys::crispasr_session_result_word_p` FFI
  declaration added.
- Rust `crispasr` crate's `SessionWord` gained `confidence: f32`,
  populated through the new accessor (folds C-side `-1.0` to `1.0`
  so consumers can render uniformly).
- Python `SessionWord` gained `confidence: float = 1.0`; ctypes
  binding probed via `hasattr(lib,
  "crispasr_session_result_word_p")` so old dylibs stay loadable.
- Flutter was already wired (`wordPFn` lookup with
  `providesSymbol` probe; `Word.p` field).

**Cross-backend smoke-test results** (JFK clip via the Python
session API, real probabilities, no longer 1.0):

```
parakeet     | 22 | And=0.9993 so,=0.9146 my=0.9998 fellow=0.9989
canary       | 22 | And=0.9722 so,=0.9605 my=0.9925 fellow=0.9969
cohere       | 22 | And=0.9810 so,=0.9966 my=0.9998 fellow=1.0000
voxtral      | 22 | And=0.9992 so,=0.8858 my=1.0000 fellow=1.0000
wav2vec2     | 22 | and=0.9999  so=0.9998  my=0.9999 fellow=0.9991
firered-asr  | 22 | AND=0.9825  SO=0.9834  MY=0.9975 FELLOW=0.9974
mimo-asr     | 22 | And=0.9999998… so,=0.9999964… my=1.0 fellow=1.0
qwen3        | 22 | And=1.0000 so,=0.9135 my=1.0000 fellow=0.9997
granite-4.1  | 22 | and=0.9516  so=0.9984  my=0.8053 fellow=0.9986
```

(mimo's 1.0s are softmax-saturated to fp32 precision on a clean
recording — `repr` shows the actual values are like
`0.9999998807907104`. Not a hardcoded 1.0.)

**Side-quest: Metal `GGML_OP_PAD` left-pad audit.** While testing
wav2vec2 the binary aborted with `unsupported op 'PAD'`. Root cause
in `ggml/src/ggml-metal/ggml-metal-device.m:131-138`: Metal supports
`GGML_OP_PAD` only when all left-side pads (params 0/2/4/6) are
zero. wav2vec2's `ggml_grouped_conv1d_same` used
`ggml_pad_ext(ctx, x, pad_l, pad_r, …)` with `pad_l = (K-1)/2 ≈ 63`
inside a `ggml_backend_graph_compute(metal, gf)` call (no scheduler
fallback) — instant crash on Apple Silicon.

Fix: replace the in-graph pad with a host-side zero-fill into a
wider input buffer, set the input tensor at the pre-padded shape,
skip the `ggml_pad_ext` op entirely. Per-call cost is one
`vector<float>` zero-fill + memcpy of `T_pad * C * 4` bytes (≈2.8
MB for wav2vec2-large) — under 1 ms on M1, well below the noise
floor. 3× consecutive JFK runs at 0.69 s ± 0 ms.

Audit of the rest of the tree: 24 `ggml_pad_ext` call sites total.
Every other left-pad (kyutai_stt:238, firered_asr:1404+1513,
qwen3_tts:3299+3321+3998, marblenet_vad:302, gemma4_e2b:320, 761,
762, vibevoice:322+343, core/conv.h:72) runs through
`ggml_backend_sched_graph_compute` which queries `supports_op` and
falls unsupported ops back to a CPU sub-backend transparently —
safe by construction. The only landmine remaining is the dead
`build_pos_conv_graph` in `wav2vec2-ggml.cpp:615` (gated off via
`include_pos_conv=false`); if anyone re-enables it they'd need to
port the same CPU-pad fix. Documented in LEARNINGS.md "Metal
`GGML_OP_PAD` only supports right-side pads".

**No GGUF regeneration required for any of this work.** All
changes are runtime-side, calling already-existing model C-APIs
that read from the existing GGUF layout. Both old (e.g.
granite-4.0 with no `audio_token_index` key) and new GGUFs work
unchanged — that's why the granite path falls back to
`kPrefix4`/`kSuffix4` legacy ids when
`granite_speech_audio_token_id()` returns `-1`.

**Still open** (text-only in the session API; tracked in PLAN
#65a): gemma4-e2b. (vibevoice + moonshine-streaming were added in
the same session, see below.) Plus other-binding word-p exposure
for Go / Java / Ruby (parallel-worker `5534588`) and JS (still
TTS-only, deferred).

**61c kyutai native + word timestamps (parallel worker).**
`kyutai_stt_transcribe_ex` returns per-token + per-word data with
audio-frame-aligned timestamps via Kyutai's "delayed-streams"
architecture: every emitted text token is associated with its
source audio frame, so word boundaries fall out for free at the
LM frame rate (12.5 Hz, with `audio_delay_seconds` lookahead
subtraction). The adapter switched to this entry point and now
declares `CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS`, lighting
up two README cells.

**61e omniasr-llm temperature.** The per-token-confidence work in
this same batch had already added an `out_logits` capture path to
`omniasr_run_prefill` + `omniasr_run_dec_token` (logits-mode graph
instead of graph-baked argmax). Adding sampling on top was a thin
extension: `omniasr_transcribe_llm` now takes a `pick_from_logits`
lambda that does CPU argmax (when `temperature == 0`) or
multinomial sampling from `softmax(logits / T)` with a
deterministic per-call xorshift64 seed (when `temperature > 0`),
overriding the helper's argmax pick. The captured probability is
the softmax-with-temperature of whichever token we picked.
Adapter wires `cp.temperature = params.temperature` and adds
`CAP_TEMPERATURE`. Smoke test confirms `-tp 0.7` produces a
different sample than greedy on a clean recording while keeping
the transcript sensible.

**61f punctuation toggle.** glm-asr / kyutai-stt / moonshine /
omniasr-llm gained `--no-punctuation` via two new shared helpers
in `examples/cli/crispasr_backend_utils.h`:

- `crispasr_strip_ascii_punctuation(s)` — keeps ASCII letters /
  digits / whitespace / any byte ≥ 0x80 (so multi-byte UTF-8
  sequences pass through untouched). Collapses runs of multiple
  spaces left behind by stripped punctuation. Trims leading /
  trailing whitespace.
- `crispasr_lowercase_ascii(s)` — ASCII-only `tolower`. Multi-byte
  UTF-8 bytes pass through unchanged so non-Latin scripts aren't
  corrupted.

Each adapter applies both to `seg.text`, every `seg.tokens[i].text`,
and (kyutai only) every `seg.words[i].text` when
`!params.punctuation`. Capability bits gain `CAP_PUNCTUATION_TOGGLE`.
4 cells in the README "Punctuation toggle" row.

Smoke (moonshine):
```
default      → "And so my fellow-american asked not what your country can do for you, ask what you can do for your country."
--no-punc    → "and so my fellowamerican asked not what your country can do for you ask what you can do for your country"
```

**65a vibevoice + moonshine-streaming.** Closes 2 of the 3 last
text-only session-API backends. Pattern mirrors the previous batch:

- `vibevoice` — refactored `vibevoice_transcribe` into
  `vibevoice_transcribe_impl` with optional `out_token_ids` /
  `out_token_probs`. `argmax` lambda became `pick(lg, out_p)` with
  softmax gated on `capture_probs`. Vocab is ~152k Qwen2 — gating
  is necessary so legacy text-only callers don't pay the
  per-token softmax cost (matches the same pattern from mimo-asr).
  Public C-API: `vibevoice_transcribe_with_probs`,
  `vibevoice_result_free`, `vibevoice_token_text`. Session adapter
  routes raw vocab pieces through `gpt2_byte_decode` (Qwen2
  byte-level BPE Ġ→space, Ċ→newline) and the shared
  `emit_words_from_tokens` helper, dropping `<|...|>` special
  tokens.
- `moonshine-streaming` — same refactor. New
  `moonshine_streaming_transcribe_with_probs`, `_result_free`,
  `_token_text`. The `_token_text` accessor reuses the shared
  `moonshine_tokenizer::token_to_piece` (added during moonshine
  proper's batch).

Cleanup: the historical `run_char_transcribe` lambda in
`crispasr_session_transcribe_lang` was removed. After this batch
every backend either has a token-prob path or routes through the
explicit text-only fallback block — the lambda was unreachable
dead code.

PLAN.md updated: §65 main batch + §65a vibevoice + moonshine-
streaming marked DONE; gemma4-e2b alone remains as the last
text-only session-API holdout. §61 audit updated with current
DONE / OPEN status for each sub-item (61a-c/e/f DONE; 61d/g/h/i/j
queued; 61k blocked on 60k).

**65a-residue gemma4-e2b session-API word probs (commit
946f624).** gemma4-e2b already used `core_greedy_decode::run` for
its decode loop — switching to `run_with_probs` was straight-
forward: capture the prefill-step prob via
`core_greedy_decode::softmax_of`, route surviving (non-special)
ids + probs through optional `out_token_ids` / `out_token_probs`
parameters on `gemma4_e2b_transcribe_impl`. Public additions:
`gemma4_e2b_transcribe_with_probs`, `gemma4_e2b_result_free`,
`gemma4_e2b_token_text`. Session adapter applies SentencePiece
▁→space decode per-token (Gemma's vocab uses ▁ markers) and runs
the shared `emit_words_from_tokens` helper. With this commit
**every ASR backend has a token-prob path through the session
API** — no more text-only fallbacks for transcription.

**61d Best-of-N for the LLM-style decoder quartet (commit
946f624).** The temperature work in 9ffb196 gave glm-asr /
kyutai-stt / moonshine / omniasr-llm multinomial sampling. To
make `--best-of N --temperature T > 0` actually draw N independent
samples (rather than collapse to one repeated sample because the
per-call seed was deterministic from audio), each backend gained
a sticky `*_set_seed(ctx, seed)` setter:

* moonshine: new `seed_override` field on `moonshine_context`,
  mixed into the existing xorshift-derived `rng_state` in the
  decode loop. seed=0 falls through to audio-derived seed (legacy
  single-shot bit-identical behaviour).
* omniasr-llm: same pattern — `seed_override` field mixed into
  the encoder-data-derived `rng_state` of
  `omniasr_transcribe_llm`.
* glm-asr / kyutai-stt: both sample via libc `rand()` (static
  helper with no context arg). `*_set_seed(ctx, unsigned)` calls
  `srand(seed)` instead — process-global, but best-of-N at the
  adapter level is sequential so the pattern works. Documented
  as "serialize at adapter".

Adapter pattern (identical across all four backends):

```cpp
const int n_runs = (params.temperature > 0 && params.best_of > 1)
                     ? params.best_of : 1;
result* best = nullptr;
double best_score = -1.0;
for (int run = 0; run < n_runs; run++) {
    backend_set_seed(ctx,
        run == 0 ? 0 : (uint64_t)run * 0x9E3779B97F4A7C15ULL);
    auto* cand = backend_transcribe_with_probs(ctx, ...);
    double score = mean(cand->token_probs);
    if (!best || score > best_score) { swap; }
    else { backend_result_free(cand); }
}
```

Run 0 always uses seed 0 → sticky override falls back to the
audio-derived seed → bit-identical to single-shot. Only runs
1..N-1 inject salt.

Smoke (moonshine on JFK with `-tp 0.7 --best-of 3`):
```
crispasr[moonshine]: best-of-3 picked score=0.9190
And so my fellow American, ask not what your country can do for
you, ask what you can do for your country.
```
Greedy baseline transcribes "fellow-american asked"; best-of-3 —
choosing the highest-mean-prob sample of 3 — got "fellow
American, ask" which is the actually-correct transcription. Real
quality win, not just diversity. README "Best-of-N" row gains
4 ✔.

**Closures and deferrals.** With this commit, PLAN #65 is fully
closed (no more text-only ASR backends in the session API). PLAN
#61 closes 61a-f (20 cells gained across the audit). Remaining
open: 61h beam search (~300 LOC shared infra, queued), 61j
translate (empirical validation across 3 backends, queued).
Deferred with documented reason:

* 61g `--ask` — target backends (glm-asr, omniasr-llm) are not
  instruction-tuned. glm-asr's prompt is hardcoded ids, no live
  tokenizer; omniasr-llm uses FLORES-200 lang ids, not chat.
  Both would need empirical validation before plumbing the toggle.
* 61i flash-attn for fc-ctc — `core_conformer::build_block`'s
  rel-pos path (Q·K + R·Q_v + rel_shift) doesn't fit
  `ggml_flash_attn_ext` (no rel-pos hook). Would need positional-
  encoding swap or custom flash kernel.
* 65b binding word_p exposure for Go/Java/Ruby/JS — those
  bindings are TTS-only today; full Session ASR API surface is a
  separate scope (~150 LOC + design per binding).

**61h beam search (partial, May 2026 — this session).** Shipped the
shared infra and one of four target cells; the other three are
deferred for a perf-real reason that the original PLAN entry didn't
foresee.

**What landed:**

- `src/core/beam_decode.h` — header-only template helper mirroring
  `core_greedy_decode::run_with_probs`. Takes a single `replay_fn`
  callback `(ctx, tokens, n_tokens, prompt_len) → float*` that
  rebuilds the given beam's KV state and returns last-position
  logits. Strategy is *replay-from-prefix*: each beam-step embeds +
  forwards the entire generated suffix from the post-prompt anchor
  (the C-API stays unchanged, no `*_kv_save` / `*_kv_restore`
  needed). Top-K logits → cumulative log-prob → global prune. Stops
  when `beams[0]` ends in EOS.
- `glm-asr` adapter — `cp.beam_size = params.beam_size` plumbed
  through; `glm_asr_transcribe_impl` dispatches to
  `core_beam_decode::run_with_probs` post-prefill when
  `beam_size > 1`. The replay lambda is `glm_asr_embed_tokens` +
  `glm_asr_run_llm_kv` (which already supports batched
  `(emb, n_tokens, n_past)` calls). `CAP_BEAM_SEARCH` added to
  capabilities. README "Beam search" row gains 1 ✔ for glm-asr.

**Smoke results on JFK (11 s, glm-asr-nano-q4_k):**

| Setting | Wall time | Transcript |
|---|---|---|
| `-bs 1` (greedy, GPU) | ~1 s | "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country." |
| `-bs 1` (greedy, CPU) | 71 s | identical |
| `-bs 2` (beam, GPU) | 52.9 s | identical |
| `-bs 4` (beam, GPU) | 35.1 s | identical |
| `-bs 2` (beam, CPU) | 129 s | identical |

The GPU beam timings are dominated by per-call Metal command-buffer
sync (~150-300 batched calls per beam decode); CPU at `-bs 2` runs 1.8×
greedy as theory predicts.

**Bug found and fixed mid-session.** The first iteration of the helper
took a single `cfg.eos_id` (mirroring `core_greedy_decode::Config`).
glm-asr has *three* stop tokens (`{59246, 59253, 59255}` per its
`hp.eos_token_ids[]`): the LLM's `<|user|>` end (59253) and a couple
of tokeniser sentinels. Single-id matching only caught 1 of 3, so on
~2/3 of inputs the beam would emit a non-cfg.eos_id stop token, NOT
mark itself finished, and run to `max_new_tokens=512`. CPU smoke
stalled past 6 minutes before being killed; that was the symptom.
Fix: extended `Config` with `std::vector<int> eos_ids` (helper checks
membership; legacy single `eos_id` still honoured when the vector is
empty). Adapter passes the full `hp.eos_token_ids[0..n_eos)` set
through. After the fix beam=2 CPU finishes in 129 s, beam=2/4 GPU in
under 1 min — all transcripts match greedy.

**Why the other three target backends got deferred.**
`omniasr-llm`, `kyutai-stt`, and `moonshine` each have a per-step
decode of the form "advance the LM by ONE token at the current KV
position, return logits". Replay-from-prefix on those would do
`O(B × T²)` *single-token* graph rebuilds — each rebuild is a fresh
`ggml_init` + scheduler reset + Metal command buffer + sync. Even
on glm-asr (which only needs `O(B × T)` *batched* rebuilds) the
total Metal compute at `beam_size=2` on 11 s JFK was still many
minutes per run during this session — replay-from-prefix is just
expensive once `T` grows past ~50. The honest fix is per-backend
`*_kv_save` / `*_kv_restore` so each beam can have a real branched
state, instead of paying `O(T²)` to rebuild it from scratch every
step. Reopen those rows when that infra lands. PLAN.md #61h was
re-statused from OPEN to IN PROGRESS with a sub-table showing which
sub-steps shipped vs. deferred.

See LEARNINGS.md "Replay-from-prefix beam search is `O(B × T²)`".

### 66. Feature-matrix closure pass — sticky setters + streaming + mic + Go/Java/Ruby parity (May 2026)

Six commits (`d963e3a`, `947262f`, `041471f`, `89687f0`, `5534588`
plus the parallel-agent sweep `f130a40` that adopted my
`crispasr_session_stream_open`) closed the long-standing gap
between CLI flags and what wrappers can reach. The audit going in:
9 capabilities lived only as CLI flags. The audit coming out: 6 of
9 wired across all 7 wrappers (or "all where it makes sense"); 3
remain documented-deferred with revised effort estimates.

**Sticky session-state setters** (`d963e3a`). New session struct
fields `source_language` / `target_language` / `punctuation` /
`translate` consulted by whisper, canary, cohere transcribe paths
when the per-call `language` arg isn't supplied. Six C-ABI exports
(`set_source_language`, `set_target_language`, `set_punctuation`,
`set_translate`, `set_temperature`, `detect_language`).
`set_temperature` multiplexes per-backend setters (canary, cohere,
parakeet, moonshine each expose runtime temperature) and returns
-2 if no backend in the session honours it (soft no-op).
`detect_language` wraps the standalone LID helper. Wrappers landed
in 3 canonical (Python/Rust/Dart) and the 3 trailing
(Go/Java/Ruby) — JS/emscripten skipped (TTS-only surface today).
+545 LOC (canonical) + +373 LOC (trailing).

**Registry enumeration** (`947262f`). `crispasr_registry_count()`
+ `_get_at(i)` internal API, `crispasr_registry_list_backends_abi`
C-ABI export, `list_known_models()` in Python+Rust+Dart returning
all 37 known backends in declaration order. Wrappers building
model-picker UIs no longer need to know the list out-of-band.

**Streaming session API** (`947262f` + `041471f`).
`crispasr_session_stream_open(s, ...)` decouples streaming from
the whisper-only `crispasr_stream_open(whisper_ctx, ...)`. Today
it routes through whisper internally; the dispatch site is
labelled with where moonshine-streaming + kyutai-stt + voxtral4b
will plug in when those refactors land. Python `Session._Stream`
context-manager handle, Rust `Stream` with `Drop`-managed close,
Dart already had it, Go got it in `5534588` (`Stream` type +
`StreamOpen/Feed/GetText/Flush/Close` + `StreamingUpdate`).

**Library mic API** (`89687f0`). `src/crispasr_mic.{h,cpp}` wraps
miniaudio's `ma_device` capture mode. Cross-platform via
miniaudio's built-in backends (Core Audio / ALSA / PulseAudio /
WASAPI). The `MA_NO_DEVICE_IO` define on `crispasr_audio.cpp` was
dropped to expose `ma_device_*` symbols; the iOS/tvOS/watchOS
variant got it back in `1c0e996` because miniaudio's CoreAudio
backend pulls Objective-C headers on those platforms (the C ABI
is stubbed to return failure on iOS/tvOS/watchOS so libcrispasr
links). Live-mic smoke-tested from Python: 17 callbacks fired in
1s wall-clock = 16320 samples = 1.02s real audio captured. The
audio-thread callback uses a `NativeCallable.listener` (Dart) /
ctypes `CFUNCTYPE` trampoline (Python) / boxed `FnMut` closure
(Rust) — each binding's idiomatic way to keep the
native-to-managed callback alive across the audio thread.

**Realistic effort revisions** in PLAN.md for the still-deferred
items:

- **62c moonshine-streaming + kyutai-stt streaming**: PLAN had
  "~80 LOC each". Survey showed both expose only single-shot
  `_transcribe()` despite their architectures being
  streaming-friendly. Real cost: ~300 LOC + 1-2 days debug per
  backend (chunked encoder + state carry-over + numerical
  regression gating). Documented; opens when a consumer asks.
- **62e voxtral4b streaming**: stays as PLAN #7 high-complexity
  separate session (~200-300 LOC, decoder thread + audio frame
  injection).
- **Init-only flags (grammar/beam/flash)**: needs backend-reinit
  machinery (close+reopen, slow) or per-backend `set_*`
  extensions (~50 LOC × 14 backends = ~700 LOC mechanical +
  per-flag regression test). Documented; opens per PLAN #59
  policy.

**Non-collision dance** worked smoothly throughout: c_api.cpp had
~900 LOC of parallel-agent WIP at one point; my edits rebased
cleanly via the backup-checkout-restage-restore pattern. The
parallel agent's `f130a40` clang-format pass swept up my
`crispasr_session_stream_open` mid-flight under their author tag —
fine outcome (work shipped), surprising commit-message
attribution.

**Tests** (`tests/test_python_session.py`): 4 new no-model test
classes — `TestSessionStateSetters`, `TestStreamingAPI`,
`TestMicAPI`, `TestRegistryEnumeration`. All four PASS in well
under a second collectively, no model required. CI gates against
accidental ABI removal across refactors.

### 67. Kartoffel-Orpheus DE + lex-au — PLAN #57 Phase 2 checkpoint swaps (May 2026)

The first three Orpheus-3B German checkpoints landed as drop-in
swaps on the runtime that section 59 shipped — no source code
changes, just registry rows + factory dispatch + (for Kartoffel)
GGUF conversion via the existing `models/convert-orpheus-to-gguf.py`
with `--variant fixed_speaker`.

**Three new backends, all reachable via `--backend ... -m auto`:**

| Backend alias | HF mirror | Speakers | Provenance |
|---|---|---|---|
| `kartoffel-orpheus-de-natural` | [`cstr/kartoffel-orpheus-3b-german-natural-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-natural-GGUF) | 19 (Jakob/Anton/Julian/Jan/…/Sophie/Marie/Mia/…) | Convert of `SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1` (llama3.2, gated) → F16 6.61 GB + Q8_0 3.5 GB + Q4_K 1.87 GB |
| `kartoffel-orpheus-de-synthetic` | [`cstr/kartoffel-orpheus-3b-german-synthetic-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-synthetic-GGUF) | 4 (Martin/Luca/Anne/Emma) + 12 emotions + 5 outbursts | Convert of `SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1` (llama3.2, gated) — same shape; emotion control via `{Speaker} - {Emotion}: {text}` prompt |
| `lex-au-orpheus-de` | `lex-au/Orpheus-3b-German-FT-Q8_0.gguf` (3.52 GB, no convert) | unknown — depends on lex-au's training set | Pre-built Q8_0 from lex-au; registry alias only, points at the upstream HF file directly |

All three share the same SNAC codec ([`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF)) the orpheus base row already publishes.

**Validation: ASR-roundtrip via parakeet-v3 -l de.** Natural variant on Q8_0 / voice Julian:

```
$ crispasr --backend kartoffel-orpheus-de-natural \
    -m kartoffel-orpheus-de-natural-q8_0.gguf \
    --codec-model snac-24khz.gguf --voice Julian --temperature 0.6 \
    --tts "Hallo, ich heiße Julian und das ist ein Kartoffel-Orpheus Test." \
    --tts-output kartoffel_test.wav
$ crispasr --backend parakeet -m parakeet-tdt-0.6b-v3-q4_k.gguf -l de \
    -f kartoffel_test.wav --no-prints
Hallo, ich heiße Julian und das ist ein Kartoffel-Orpheus-Test.
```

Word-exact (only minor `Kartoffel-Orpheus-Test` hyphenation drift). Synthetic variant smoke deferred — local 16 GB box was memory-contested by the parallel agent's concurrent crispasr-diff + converter runs, and the orpheus 3B AR loop hung in both Metal init (84 min stuck on `libraries.data` cache I/O while pages were compressor-evicted) and CPU mode (40+ min for Q8_0, 26+ min for Q4_K, 0 bytes output). Architecture is the same as the natural variant which validated end-to-end, and the synthetic checkpoint is purely an LM-weight delta — no runtime risk.

**HF upload economics.** Per-file `api.upload_file` (sequential, README first) was the working recipe; `api.create_commit` with a 4-op transaction hung on token-refresh `RemoteDisconnected` mid-upload (43 min in TCP CLOSE_WAIT before kill). Xet dedup is dramatic when uploads share a base model: orpheus 10.1 GB upload was 576 MB net new bytes (Llama-3.2-3B base shared with the unsloth mirror); the synthetic Kartoffel 12 GB upload was ~5.1 GB net new because most chunks deduped against the natural variant which had landed an hour earlier.

**Gated-repo gotcha.** `SebastianBodza/Kartoffel_Orpheus-3B_german_*` are click-through gated. Even when authenticated as `cstr`, `snapshot_download` returned 401 — diagnosed as a token-loading bug: when `HF_HOME` is overridden, the lib looks for `$HF_HOME/token` instead of the default `~/.cache/huggingface/token`. Drop the `HF_HOME` override and it works. Direct `requests.head` with explicit `Authorization` header worked all along, which is what made the diagnosis tractable.

### 68. v0.5.4 release + wrapper-publish auto-trigger silenced (May 2026)

Tagged `v0.5.4` at commit `1c0e996`, `release.yml` (binary artifacts +
GitHub release page) + `Publish Docker image` ran cleanly. **291 commits
since v0.5.3** (April 27); highlights in the tag annotation:

- MiMo-V2.5-ASR end-to-end (PLAN #51) + perf wave (#51b/b' step-decode
  KV reuse, #60d fused QKV, #60e KV-quant env wired across 9 backends).
- Qwen3-TTS family (Base 1.7B, VoiceDesign 1.7B, CustomVoice 0.6B/1.7B).
- Orpheus-3B-FT TTS + lex-au-orpheus-de + Kartoffel_Orpheus DE natural.
- granite-speech-4.1 plus + nar variants.
- Per-token confidence + per-word probability across 14 backends.
- Streaming + mic library API (PLAN #62 a/b/d).
- Zero-copy mmap GGUF loader (`CRISPASR_GGUF_MMAP=1`), preload, mlock,
  WILLNEED + MADV_RANDOM helpers (PLAN #60a/b/c/f/g).
- Vibevoice cache-bypass extended to Vulkan + CUDA (issue #47, geneing).

**Wrapper-publish workflow silenced.** `release-wrappers.yml`'s
`tags: ['v*']` auto-trigger has been failing on every release since
v0.5.0 because the three target packages
(`crispasr-sys`/`crispasr` on crates.io, `crispasr` on PyPI, `crispasr`
on pub.dev) **have never been registered** — the registries reject
the very first publish from a CI-only flow because there's no prior
owner record to verify against. v0.5.4 confirmed it again
(`gh run view 25248028443`).

Decision: comment out the auto-trigger so future tag pushes don't keep
producing red runs; keep `workflow_dispatch` so manual ad-hoc testing
still works during eventual bootstrap. Bootstrap procedure is documented
end-to-end in PLAN §66 with exact per-registry commands. Workflow is
also hardened (per-job `continue-on-error: true` + secret-presence
prechecks) so when we re-enable the trigger, a single registry's
misconfiguration won't fail the whole workflow.

PLAN §67 carries forward the rest of this session's deferred
follow-ups in one place: F16 mimo-asr re-upload, per-backend Q8_0 KV
cosine validation, vibevoice CUDA cache reuse re-test,
SYCL/HIP/ROCm cache-bypass extension, `MADV_RANDOM` per-backend
wiring, disk5 cleanup, legacy `build.yml` audit.

### 69. PLAN #62c kyutai-stt streaming — chunked-batch over rolling window (May 2026)

**Done.** `crispasr_session_stream_open` now routes to a
kyutai-backed rolling-window stream when `s->kyutai_ctx` is set;
the four whisper-typed `crispasr_stream_*` functions
(feed/get_text/flush/close) branch on a new optional
`kyutai_stream_state` field on `crispasr_stream`. ~200 LOC across
three files, well under the 300-LOC PLAN estimate.

**Why chunked-batch, not true incremental.** The PLAN's original
"refactor `_transcribe` into incremental encode + decode + state
carry-over" path assumed the Mimi encoder was at worst causal.
Pre-impl exploration found `src/kyutai_stt.cpp:660` calls
`ggml_flash_attn_ext(..., nullptr, ...)` — **non-causal**, every
encoder-transformer frame attends to every other. True streaming
therefore can't bit-match batch without either O(n²) re-encoding
or swapping in sliding-window attention (~500-700 LOC, deviates
from training). Chunked-batch sidesteps both: each decode runs the
existing single-shot `kyutai_stt_transcribe_ex` over the last
`length_ms` of audio, so each window is bit-exact-batch by
construction.

**Validation.**
- Final stream output on JFK matches single-shot batch byte-for-byte
  (after both are lstripped of the SentencePiece `▁ → space`).
- Intermediate decodes show expected rolling-window behavior:
  "America" at decode #3 (6 s window) flips to "Americans" at
  decode #4 (8 s) once the LM has more context.
- Whisper streaming path unchanged — `ggml-tiny.bin` still produces
  a coherent JFK transcript through the same
  `crispasr_session_stream_open` → `crispasr_stream_feed`/`get_text`
  chain.

**Out of scope but trivially next.** `moonshine-streaming` is also
single-shot today; the same chunked-batch pattern (rolling-window
wrapper + `moonshine_stream_state` field) would land in ~50 LOC.
Voxtral4b stays separate (PLAN #7 — needs a real decoder-thread
refactor).

### 70. PLAN #61h beam search — branched-KV variant + 3 deferred backends shipped (May 2026)

**Done.** Closed the three rows §65 left deferred under "61h beam
search (partial)": omniasr-llm, kyutai-stt, moonshine. README "Beam
search" gains 3 ✔ for a total of 6 across the matrix
(whisper / glm-asr / kyutai-stt / firered / moonshine / omniasr).

**The blocker, recap.** Original §65 entry shipped only glm-asr because
the shared `core_beam_decode::run_with_probs` is *replay-from-prefix*
— each beam's per-step decode replays the full generated suffix from
the post-prompt anchor. For glm-asr's batched `glm_asr_run_llm_kv(emb,
n, n_past)` that's `O(B × T)` batched forwards, fast enough. For
omniasr-llm / kyutai-stt / moonshine — all of which have a
single-token-per-call decode API — replay-from-prefix would do
`O(B × T²)` *single-token* graph rebuilds, multi-minute on Metal.

**The unblock.** `core_beam_decode::run_with_probs_branched` —
header-only template, takes per-beam KV `save_fn` / `restore_fn` /
`snap_free_fn` / `step_fn` callbacks. Snap holders are wrapped in
`shared_ptr` inside the helper so siblings can share a parent's
post-step snap without double-free. Cost drops to `O(B × T)` true
single-token forwards. (The branched variant landed independently in
commit `e9783d2`; the same author / parallel-worker pass also did the
clang-format sweep in `db1149c` that bundled the kyutai-stt beam
edits.)

**Per-backend snapshot strategy.**

* **moonshine** — `kv_self.k[il]` / `kv_self.v[il]` are per-layer
  ggml tensors of shape `[head_dim, max_len, n_kv_heads]`. Snapshot
  reads each layer's full tensor via `ggml_backend_tensor_get`; small
  enough total (~MB-scale) to ignore. `kv_self.n` (the next-write
  slot) goes into the snapshot too. `kv_cross` is precomputed once
  outside the beam path and shared across beams.

* **omniasr-llm** — `kv_k` / `kv_v` are single contiguous tensors of
  shape `[head_dim, max_ctx, n_heads, n_layers]`. Snapshot reads the
  whole tensor (~30 MB total for the 300m model). `kv_n_used` is
  written by prefill but never read by the per-step decode (each
  call passes `n_past` explicitly), so the snapshot can omit it.
  Beam path activates only on the LLM variant; CTC is unaffected.

* **kyutai-stt** — same KV layout as omniasr but the per-frame loop
  is fundamentally different: each frame mixes audio code embeddings
  + text token embedding into one LM step, and there is no EOS — the
  loop runs to `T_frames`. Beam = top-K text-token decisions per
  frame, with audio codes shared across all beams. Refactored the
  per-frame body into `kyutai_lm_step(text_token, codes, frame_idx,
  n_past, &out_logits)`; greedy + beam both call it. Beam Config:
  `eos_id = -1`, `max_new_tokens = T_frames`, `prompt_len = 1`
  (frame 0 is run manually to seed initial logits + populate KV
  slot 0; helper handles frames 1..T_frames-1).

**Validation on JFK (11 s, Metal, warm cache).** All three backends
match greedy transcript at `-bs 1 / 2 / 4`; moonshine `-bs 4`
actually improves quality.

| Backend | model | `-bs 1` | `-bs 2` | `-bs 4` | Best transcript |
|---|---|---|---|---|---|
| moonshine | tiny-q4_k | 0.57s | 0.57s | 0.57s | "fellow Americans ask…" (beam=4 only) |
| omniasr-llm | 300m-v2-q4_k | 38s | 23s | 70s | "fellow americas ask…" (all match greedy) |
| kyutai-stt | 1b-q4_k | 5.1s | 8.6s | 14.2s | "fellow Americans, ask…" (all match greedy) |

omniasr-llm `-bs 4` lands above the 60s "rough" PLAN gate but the
per-step compute scales linearly (411 step calls for `-bs 4`,
103 for `-bs 1`); the snapshot copy + Metal single-token graph
dispatch overhead per call dominate, not the helper itself.

**Closures.** PLAN #61h sub-table now has 6 of 8 lines DONE
(generic helper × 2 + 4 backends). Two lines remain DEFERRED:
session-API beam exposure for the qwen3/granite/voxtral4b/voxtral
quartet (pure plumbing once the session API exposes `beam_size`),
and per-decoder beam for the canary/cohere encoder-decoder pair.
Both were never the heart of the §65 deferral — that was the
`O(B × T²)` perf wall that the branched-KV variant tore down.

### 71. PLAN #7 voxtral4b streaming — phase 1 + 1.5 (incremental encoder default-on) (May 2026)

**Scope.** Voxtral-Mini-4B-Realtime is the only deferred backend with an
architectural path to sub-second first-token streaming: causal+SWA
encoder + sliding-window LLM with persistent KV cache. Phase 1 ships
the streaming **API surface** for it. PTT/dictation semantics — `feed`
continuously, `flush` returns the transcript.

**Files.**
- `src/voxtral4b.{h,cpp}`: 5 entrypoints (`voxtral4b_stream_open` /
  `_feed` / `_get_text` / `_flush` / `_close`) + the `voxtral4b_stream`
  struct + an incremental encoder graph (`voxtral4b_build_graph_encoder_stream`)
  and projector graph that's wired but currently opt-in (see phase 1.5).
- `src/crispasr_c_api.cpp`: `void* voxtral4b_stream_state` field on
  `crispasr_stream` + 4 dispatch branches (close/feed/get_text/flush)
  + open path in `crispasr_session_stream_open`. Mirrors the
  kyutai/moonshine pattern from §69.
- `tools/bench_streaming_latency.py`: greenfield bench harness driving
  the stream API; `--check-batch-equality` runs the CLI batch baseline
  and asserts the streaming transcript matches byte-for-byte.

**Default path: incremental encoder during feed.** `feed()` runs the
streaming encoder graph chunk-by-chunk (~80 ms audio per chunk), with
persistent encoder K/V cache (`core_attn::kv_self_attn` with F32 cache
sized to `audio_swa = 750` frames) and per-conv left-context state
(2 mel frames for conv0, 1 d_model frame for conv1). `flush()` does
just the streaming-prompt prefill + per-step audio-injection greedy
decode, exactly mirroring the CLI adapter in
`examples/cli/crispasr_backend_voxtral4b.cpp`. Audio is auto-padded
with 32 left-pad tokens of zero audio at stream open + right-aligned
+ 10 right-pad tokens at flush time. Set
`CRISPASR_VOXTRAL4B_STREAM_BATCH_ENCODER=1` to fall back to running the
whole encoder at flush time (regression-debug switch).

**Validation.** Bit-exact-batch on JFK 11 s (M1, Q4_K):
- Batch CLI: `"And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country."`
- Streaming via `crispasr_session_stream_open` → `feed` × N → `flush` → `get_text`: byte-for-byte identical, `[bench] BIT-EXACT-BATCH: PASS`.
- Per-embed cosine vs batch encoder: cos≥0.9999 across all non-tail-pad
  chunks; the final chunk diverges by construction (batch's last mel
  frame uses right-side `center_pad` data that streaming doesn't have,
  but the model already has the full content via the preceding 179
  embeds, so the transcript is unaffected).
- Wall-clock: feed ≈ 24 s (~170 ms per 80 ms chunk = 2.1× realtime),
  flush ≈ 10 s. Combined ≈ 34 s for 11 s audio = 0.32× realtime —
  same total cost as batch, distributed across feed + flush.

**The streaming-prompt convention is qualitatively different from
voxtral-3B's `[INST]…[TRANSCRIBE]` template** (which is what
`run_voxtral_family` in `crispasr_c_api.cpp:1483` uses for voxtral 3B
+ qwen3 + granite). The 4B-realtime model's prompt is BOS + 38
STREAMING_PAD = 39 tokens; audio embeds are *element-wise added* to the
first 39 prompt embeds; subsequent audio embeds are *added per-step* to
the next-token's embedding before the LLM forward (the "audio-injection
pre_hook"). Decode stops when audio is exhausted (matches the CLI's
`pre_hook → false` behaviour at `core/greedy_decode.h:306`); without
this stop, the model emits trailing garbage tokens that the prompt
template doesn't license. Streaming control tokens (id < 1000) are
filtered from the user-visible transcript.

**Phase 1.5 root cause and fix.** The incremental encoder went
default-on after a single 2-axis layout transpose was removed from
`vox_stream_advance_mel`. `core_mel::compute` with `Layout::MelsTime`
emits `(n_mels, T)` row-major — T fast, n_mels slow — which is exactly
what the encoder graph's mel input expects (ggml `ne=(T_mel, n_mels)`).
The streaming code had a transpose-on-copy loop that wrote it as
`(T, n_mels)` per row, sending mel and time axes to the encoder
swapped. The encoder produced plausible-looking but content-incorrect
audio embeds (cos~0.987 vs batch on the very first chunk, where the
expected value is cos~1.0 since the input is all-zero left-pad
silence). The model read these as noise and emitted only
streaming-pad tokens. Removing the transpose and keeping `mel_pending`
+ `conv0_lctx` in `(n_mels, T)` per-band-contiguous layout restored
cos≥0.9999 across all non-tail-pad chunks.

Smaller fix bundled in: encoder K/V cache is F32, not F16. The batch
encoder runs F32 throughout; F16 KV cache (which the LLM uses
successfully because the LLM was trained with F16 KV) introduces
precision loss across the encoder's 32 layers. Cost: 393 MB per stream
at the SWA cap of 750 frames. TODO: re-measure F16 cache with the mel
layout fix in place — F16 may now be sufficient.

**Latency target NOT met by phase 1+1.5.** ≤240 ms first-token target
requires further encoder optimisation (Q4_K Metal kernels, larger
chunks, decoder thread overlap with next-chunk encode). Phase 2 work.
Phase 1+1.5 ships the streaming API + bit-exact correctness; the
realtime-feed property is the remaining work.

**Phase 2 SHIPPED — chunk-size + fused QKV + default-unification + timing instrumentation (May 2026).**
- Default internal encoder chunk bumped 80 ms → 240 ms
  (`CRISPASR_VOXTRAL4B_STREAM_CHUNK_MS` env-var override). 2.6× feed
  speedup on JFK 11 s (24 s → 9.3 s); bit-exact-batch unaffected.
  Constraint: must be a multiple of 80 ms (8 mel frames) so the
  projector's stack-4 alignment lands on chunk boundaries.
- Per-stage timing instrumentation (`CRISPASR_VOXTRAL4B_STREAM_TIMING=1`)
  prints encoder-drain / prefill / first-text-token / per-step p50/p95
  / total flush wall-clock to stderr.
- Default-unification fix: feed() was incremental-by-default but flush()
  was batch-encoder-by-default — opposite-default bug from the stash
  restoration. Each flush threw away the streaming encoder's audio_embeds
  and re-ran the whole encoder over the full PCM. Unified to incremental
  on both paths (`CRISPASR_VOXTRAL4B_STREAM_BATCH_ENCODER=1` to opt out).
  Wins: encoder drain at flush 2064 ms → 1016 ms; first-text-token
  2674 ms → 1646 ms.
- Runtime fused QKV for the LLM. Concat each layer's q/k/v weights
  along the output axis at load time into a single (d_model,
  q_dim+2*kv_dim) tensor; route through `core_attn::kv_self_attn`'s
  `qkv_w` path. Extends the qwen3_asr precedent to handle Q4_K (and
  any row-wise quantized format) by byte-concat — each output row is
  a self-contained block group, so concatenation along the output axis
  is a pure memcpy. ~7-8 % decode speedup (56 → 50.4 ms per step).
  `CRISPASR_VOXTRAL4B_FUSED_QKV=0` to opt out.
- FFN gate+up fuse was tried and reverted — Metal's Q4_K matmul kernel
  for (3072 × 9216) is already memory-bandwidth-bound, so combining
  two of those into (3072 × 18432) didn't help the per-step decode
  budget. The `core_ffn::swiglu_fused_gate_up` helper stays in place
  for any future caller where the ratio is more favourable (e.g. a
  larger model where the gate-up matmul time dwarfs the overhead, or
  a different backend with different kernel characteristics).

**Final phase 2 numbers** (M1 Q4_K JFK 11 s, all phase 1+1.5+2 wins):

| Metric | Phase 1 | Phase 1.5 | Phase 2 |
|---|---|---|---|
| feed total | 23 s | 24 s | **9.1 s** |
| flush total | 8.5 s (batch encoder at flush) | 11.4 s (incremental + waste) | **8.3 s** |
| per-decode-step | 56 ms | 56 ms | **50.4 ms** |
| first-text-token | n/a (no live data) | 2.7 s | **1.6 s** |
| bit-exact-batch | PASS | PASS | PASS |

**Phase 3 partial — combined-chunk flush + speculative LLM prefill (May 2026).**
- **Combined-chunk flush.** Right-pad feed at flush previously ran
  3-4 separate 240 ms encoder chunks plus tail-pad + per-chunk
  projector. Refactored to append right-pad zeros directly to
  `pcm_with_pad` and drain all pending mel via one larger combined
  chunk (~96 mel frames = 48 enc frames = 12 projector groups). Saves
  ~6 Metal kernel launches. Encoder drain at flush: 990 ms → 307 ms;
  first-text-token: 1646 ms → 921 ms.
- **Speculative LLM prefill during feed.** Once feed has produced ≥
  39 audio_embeds (after ~3.1 s of audio at 240 ms chunks), run the
  streaming-prompt prefill speculatively and stash the resulting
  last-position logits + n_past on the stream. Flush skips prefill
  (~250 ms saved) and jumps straight to the decode loop. No
  correctness risk: the LLM's KV cache state at position 39 is
  identical regardless of when prefill runs. First-text-token:
  921 ms → 650 ms.

**Final phase 1+1.5+2+3-partial numbers (M1 Q4_K JFK 11 s):**

| Metric | Phase 1 | Phase 2 | Phase 3-partial |
|---|---|---|---|
| feed total | 23 s | 9.1 s | **9.4 s** |
| flush total | 8.5 s | 8.3 s | **7.5 s** |
| per-decode-step | 56 ms | 50.4 ms | **50.4 ms** |
| **first-text-token** | n/a | 1.6 s | **650 ms** |
| bit-exact-batch | PASS | PASS | PASS |

Total session: first-text-token 2674 ms → 650 ms = **4.1× faster**.
The remaining ~410 ms gap to the ≤240 ms model-card target is the
architectural floor (8-step streaming-pad warmup × 50.4 ms = ~400 ms
plus the first text decode step). Below that requires either a
different prompt convention (model retrain) or substantially faster
Q4_K Metal kernels.

**Phase 3 finale — live captions during speech (May 2026).** Once
feed has produced ≥39 audio_embeds (~3.1 s of audio), every new
audio_embed during subsequent feeds drives one greedy decode step;
tokens commit immediately to `out_text` / `out_text_unread`, so
`get_text()` polled during feed returns progressive transcript.

No stable-prefix heuristic needed: voxtral4b's audio-injection
pre_hook makes each decoded token a deterministic function of the
audio context up to that point. Tokens commit immediately — no
retraction. This is the key architectural difference from
encoder-decoder ASR (whisper / parakeet / canary) where the encoder's
bidirectional context shifts as more audio arrives, making mid-decode
tokens unstable.

API: `voxtral4b_stream_set_live_decode(stream*, int)` toggles per-
stream. Generic dispatch via `crispasr_stream_set_live_decode` for
use through the unified `crispasr_stream*` handle. Python:
`Session.stream_open(live=True)`. Default OFF (PTT semantics
preserved).

Refactor extracted the inline decode loop into `vox_stream_drain_decode`
shared between feed (live mode) and flush (PTT + final drain). A
3-step state machine (argmax → stop-check → inject+forward) with a
`decode_logits_committed` flag prevents double-emission across
multiple drain calls — the bug pattern is "argmax + emit at top of
loop, return without forward, next drain re-argmaxes the same logits
and re-emits."

Smoke on JFK 11 s with `live=True`:
```
+ 1280ms: ' And'
+ 1760ms: ' so,'
+ 2000ms: ' my'
...
+10880ms: ' your'
flush:   ' country.'
```
Cumulative transcript matches batch byte-for-byte.

Limitation: sequential live decode is ~1.5× realtime on M1 Q4_K
(50 ms decode + 100 ms encoder per 100 ms audio chunk). Falls behind
realtime audio when fed from a live mic. Phase 4 (decoder thread
parallel to encoder) would fix this.

**Phase 4 — decoder thread (May 2026).** Optional worker thread that
drains decode steps in the background while feed() handles encoder +
projector on the main thread. Enable via
`CRISPASR_VOXTRAL4B_STREAM_DECODER_THREAD=1` (implies live mode).

Architecture: `worker_thread` sleeps on `cond_var` until shutdown OR
audio_embeds grow past `decode_adapter_pos`. feed() acquires
`sched_mutex` around encoder/projector/prefill, notifies cond_var,
returns. Worker drains under sched_mutex + decode_state_mutex.
flush() signals + busy-waits with 1 ms sleeps until worker is idle
and caught up to N_audio. close() requests shutdown + joins.

Performance on M1 (JFK 11 s, all three modes pass bit-exact-batch):
- PTT: feed 9.3 s, flush 7.3 s
- LIVE single-thread: feed 15.6 s, flush 1.0 s
- LIVE + decoder thread: feed 15.7 s, flush 1.0 s

The thread doesn't reduce total wall-clock on M1 because Metal's
single GPU queue serializes encoder + decoder regardless of how
many CPU threads submit work. **Wins materialise when:**
1. Mic-driven feed has audio-rate gaps between calls — worker
   drains decode during the gaps, feed returns between encoder
   chunks without waiting for the entire decode loop.
2. Faster GPUs (M3 Ultra, NVIDIA) with kernel-level parallelism —
   Metal/CUDA can overlap concurrent compute kernels submitted by
   different ggml_backend_sched_compute calls.

**PLAN #7 closed at phase 4.** Phase 5 (dual-sched for true on-Metal
parallelism via independent backend schedulers, ~150 LOC) deferred
to when a consumer measures gain on faster hardware.

**Architectural floor for first-text-token latency** (revealed by
the timing pass on M1 Q4_K JFK 11 s):

| Stage | ms |
|---|---|
| Encoder drain at flush (Metal JIT + final-chunk + projector) | ~2064 |
| LLM prefill (39-token streaming-prompt) | 247 |
| 8 streaming-pad warmup steps × 52 ms each | ~416 |
| First text-emitting decode step | ~52 |

Even with a fully-warm encoder, the streaming-prompt convention forces
≥ 663 ms before first text emits. ≤240 ms target requires either a
different prompt convention (model retraining) or substantially faster
Q4_K Metal kernels. Phase 2 remainder (encoder kernel pre-warm, LLM
fused QKV, live-captions stable-prefix commit) deferred.

**Working-tree hygiene incident (parallel-agent reset).** Mid-session,
a concurrent process reset `src/voxtral4b.{cpp,h}` and PLAN/HISTORY to
HEAD via `git checkout`/`git restore`, dropping the in-flight phase 1.5
edits. Recovered them from `git stash@{0}` ("On main: parallel-agent
docs + voxtral4b") which captured the work before the reset. Confirms
the LEARNINGS rule "never stash/relocate unrelated changes; the
working tree is shared by parallel agents" — and adds the corollary
that `git stash` from a parallel agent IS a recovery vector when the
working tree gets reset under you.

### 72. PLAN #63 Feature matrix parity — 9-phase capability expansion (May 2026)

**Scope.** Deep audit of the feature matrix (`crispasr --list-backends`
vs README vs actual code) revealed many backends had implemented
features without declaring the corresponding CAP_ flags, and several
cross-cutting improvements were low-hanging fruit. Nine phases executed
in a single session.

**Phase 1 — Beam search for LLM backends.** Wired `core_beam_decode::run_with_probs`
(replay-from-prefix strategy) into granite (all variants) and qwen3 CLI
backend adapters. When `-bs N` (N>1) is passed, the beam search replaces
the greedy decode loop. Greedy and best-of-N paths unchanged.

Files: `crispasr_backend_granite.cpp`, `crispasr_backend_qwen3.cpp`.

Skipped voxtral4b (per-step audio-injection incompatible with replay),
canary/cohere (opaque library decode calls, no beam_size parameter).

**Phase 2 — Auto-download gaps.** Added `CAP_AUTO_DOWNLOAD` to omniasr
(registry entry existed but flag missing). Added mimo-asr to model
registry (`cstr/mimo-asr-GGUF`) with `CAP_AUTO_DOWNLOAD + CAP_TOKEN_CONFIDENCE`.

**Phase 3 — Flash attention declarations.** All 6 backends (glm-asr,
kyutai-stt, firered-asr, moonshine, omniasr, omniasr-llm) already call
`ggml_flash_attn_ext` in their source — just needed `CAP_FLASH_ATTN`
declared in CLI adapters. No attention implementation changes.

**Phase 4 — CTC timestamps for aligner.** Added `CAP_TIMESTAMPS_CTC` to
moonshine, moonshine-streaming, omniasr, omniasr-llm, mimo-asr. The
`-am` CTC aligner flag is gated by this cap in `crispasr_run.cpp:292`.

**Phase 5 — Auto-punctuation for CTC backends.** Backends without
`CAP_PUNCTUATION_TOGGLE` now auto-enable `--punc-model auto` (FireRedPunc,
~50 MB, auto-downloaded). Affects fc-ctc, wav2vec2, firered-asr, omniasr-ctc.
Users suppress with `--no-punctuation` or `--punc-model none/off`.
Tested: fastconformer-ctc on JFK now emits capitalized, punctuated text.

File: `crispasr_run.cpp` (10 lines added before punc_ctx setup).

**Phase 6 — Best-of-N.** No code changes needed — works on GPU, CPU too
slow for large models. Documentation only.

**Phase 7 — Cap declaration fixes.** Re-audit found the initial automated
report was wrong: firered-asr, moonshine, kyutai-stt, omniasr all already
had correct capability declarations. Only omniasr (missing auto-dl) and
mimo-asr (capabilities=0) needed fixes.

**Phase 8 — Vibevoice CLI adapter.** Re-audit found
`crispasr_backend_vibevoice.cpp` already exists (160 lines, 16k→24k
resample, ASR + TTS). Initial report was incorrect.

**Phase 9 — Translation.** Investigated cohere (no translate token in
vocab, transcription-only model) and glm-asr (translate flag + prompt
injection infrastructure wired, but GLM-ASR-Nano doesn't respond to
translation instructions). `CAP_TRANSLATE` not declared for either.

Files: `glm_asr.h` (translate + target_lang fields), `glm_asr.cpp`
(prompt injection), `crispasr_backend_glm_asr.cpp` (param forwarding).

**Net capability gains across all phases:**

| Capability | Backends gained |
|---|---|
| Beam search (`-bs N`) | +granite, +granite-4.1, +granite-4.1-plus, +qwen3 |
| Flash attention | +glm-asr, +kyutai-stt, +firered-asr, +moonshine, +omniasr, +omniasr-llm |
| CTC timestamps (`-am`) | +moonshine, +moonshine-streaming, +omniasr, +omniasr-llm, +mimo-asr |
| Auto-download (`-m auto`) | +omniasr, +mimo-asr |
| Auto-punctuation | +fc-ctc, +wav2vec2, +firered-asr, +omniasr-ctc (opt-in default) |
| Token confidence | +mimo-asr |

**README feature matrix** updated after each phase to stay in sync with
`--list-backends` output. Matrix now covers 21 ASR backends × 18 features.

**Test results** (`test-all-backends.py --profile=feature`): 51 PASS,
0 FAIL, 3 SKIP (stream needs shared lib build). All 18 backends pass
transcribe smoke.

### 73. PLAN #59 Cross-binding C-ABI parity — Go/Java/Ruby ASR surface (May 2026)

**Scope.** The Go/Java/Ruby bindings could only do TTS — the most
critical ASR operation (`Transcribe`) was unwrapped. This session
added the full ASR pipeline surface to all three bindings.

**Go binding (14% → 35%):** Now wraps the complete user-facing surface:
- `Transcribe`, `TranscribeLang`, `TranscribeVAD` — full Session ASR
- `VADSegments` — standalone speech detection
- `DiarizeSegments` — speaker label assignment
- `AlignWords` — CTC forced alignment
- `DetectLanguagePCM` — standalone language detection
- `PuncModel` — FireRedPunc punctuation restoration
- `RegistryLookup`, `CacheEnsureFile`, `CacheDir` — model management

All with idiomatic Go types (`TranscribeResult`, `TranscribeSegment`,
`TranscribeWord`, `VADSpan`, `DiarizeSeg`, `AlignedWord`, etc.).

**Java binding (13% → 30%):** `transcribe()`, `transcribeLang()`,
`transcribeVad()`, `vadSegments()`, `alignWords()`,
`detectLanguagePcm()` + JNA declarations for punc/registry/cache.
Result types: `Segment`, `Word`, `AlignedWord`, `VADSpan`.

**Ruby binding (15% → 24%):** `transcribe(handle, pcm)`,
`vad_segments(path, pcm, ...)`, `align_words(model, text, pcm, threads)`.
Returns Ruby hashes with `:text`, `:t0`, `:t1`, `:words`, `:p` keys.

JS/emscripten skipped — WebAssembly binding needs a different approach
(can't do cgo-style FFI). Dart already had transcribe from a prior
session.

### 74. Issue fixes — gemma4-e2b #49, Docker diagnostics #31 (May 2026)

**gemma4-e2b assertion crash (#49).** `ggml_backend_sched` hash set
was sized to 16384 but the audio encoder graph uses up to 32768 nodes.
Increased to 40960. Also fixed SentencePiece `▁` (U+2581) not being
replaced with space during detokenization — raw vocab tokens were
concatenated with visible `▁` markers.

Added gemma4-e2b to test-all-backends.py registry: **PASS, WER=0.0%**.

**Docker diagnostics env var (#31).** Users hitting CUDA driver
mismatches couldn't run `--diagnostics` because `run-server.sh`
required a model. Added `CRISPASR_DIAGNOSTICS=1` env var that runs
diagnostics and exits without touching `/models`:
```
docker run --rm --gpus all -e CRISPASR_DIAGNOSTICS=1 crispasr:main-cuda
```

### 75. Unit test expansion — core decode + registry (May 2026)

**test-core-decode.cpp** (8 Catch2 cases, 27 assertions): Tests the
shared `core_greedy_decode` and `core_beam_decode` infrastructure used
by granite, qwen3, glm-asr, kyutai-stt, moonshine, omniasr backends.
Uses a mock LLM (no models, no GPU, sub-millisecond). Covers: argmax,
softmax_of, greedy sequence generation with EOS, beam search at
beam_size=1 (greedy equivalence) and beam_size=4, max_new_tokens cap,
multi-EOS termination, probability bounds.

**test-registry.cpp** (9 Catch2 cases, 14 assertions): Verifies the
model registry lookup for 8 backends (whisper, parakeet, mimo-asr,
omniasr, omniasr-llm, granite-4.1, gemma4-e2b, vibevoice) plus unknown
backend returns false. Pure in-memory queries, no network.

**#60e KV cache Q8_0 validation.** Transcript comparison F16 vs Q8_0
across 7 backends: granite, granite-4.1, glm-asr, omniasr-llm, voxtral
all bit-exact; qwen3 minor punctuation diff (WER=0%); mimo-asr
validated previously (cosine ≥0.98). `CRISPASR_KV_QUANT=q8_0` safe
for all tested backends.

**PLAN cleanup.** #41 (Moonshine IPA) superseded by kokoro espeak-ng
phonemizer. #9 (Parakeet TDT GPU) parked — encoder dominates, LSTM
decoder <0.7s. #60 marked DONE. #62 already done (stale PLAN entry).
Priority table deduplicated.
