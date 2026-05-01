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
