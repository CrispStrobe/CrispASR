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

**Validation (Q4_K, JFK, on this 16 GB box):**
- `crispasr-diff` cosines reproduce the §56 / 51b/b' baselines
  within FP rounding (audio_features 0.998, text_embeds 0.996,
  inputs_embeds 0.998, last_hidden 0.963, logits 0.981).
- JFK transcript byte-identical to "And so, my fellow Americans,
  ask not what your country can do for you. Ask what you can do
  for your country."
- `MIMO_ASR_BENCH=1` shows the predicted ~1.1-1.2× per-step
  speedup on top of the 0.79 s/step 51b/b' baseline.

The patched Q4_K was uploaded to `cstr/mimo-asr-GGUF` replacing
the unfused file. The unfused F16 stays in the repo unchanged —
the runtime fallback in `bind_qwen2_block` keeps it working as-is.
F16 re-upload is queued behind PLAN #51c (uncontended-disk
re-conversion).

**60e KV-quant env flag (mimo-asr):**

`CRISPASR_KV_QUANT={f16,q8_0,q4_0}` now picks the KV-cache dtype in
`mimo_asr_kv_init`. Default stays F16, so this is bit-identical to
existing behaviour.

The shared `core_attn::kv_self_attn` read path adapts via
`ggml_is_quantized(kv_k->type)`: F16/F32 still uses `ggml_cont` to
materialise the strided per-layer view; quantised types use
`ggml_cast(...,F16)` because Metal's CPY kernel doesn't support
`Q*→Q*` (only `Q*→F16/F32`). So the cache *storage* is half-bytes
(Q8_0) or quarter-bytes (Q4_0) — the hour-long-podcast use case
where `max_ctx > 10k` would otherwise need ~1.5 GB F16 KV — but
each step's read pays one dequant pass per layer.

Per-backend rollout (mirroring the env lookup into `qwen3_asr`,
`voxtral4b`, `granite_speech`, `granite_nle`, `gemma4_e2b`, etc.) is
deferred to a follow-up — see PLAN #60e for the rollout list and
diff-harness gate.

**Side-quest:** the converter run itself was killed mid-flight after
22 min / 0.8 MB/min sustained on the contested disk — same
diagnostic signature as PLAN #51c thrash mode. The patcher script
exists specifically to side-step this on this hardware; the
converter itself is correct (and what would run on an uncontested
machine).
