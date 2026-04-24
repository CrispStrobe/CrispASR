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
path preserved byte-identical to upstream `whisper-cli`. 7 non-whisper
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
stays byte-identical to upstream `whisper-cli`.

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
- **#17 VAD stitching** — stitch + remap matching whisper.cpp. C-ABI: `crispasr_session_transcribe_vad`.
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
