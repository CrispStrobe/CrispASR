# CrispASR — Pending work

Pending roadmap items. Each is self-contained with files, approach, and
effort estimate. Completed items have been moved to `HISTORY.md`.

**Current state (May 2026, v0.5.5):** 20 ASR + 3 TTS backends (+ Chatterbox T3 in progress), unified CLI,
OpenAI-compatible server + WebSocket streaming, shared `src/core/` library, FireRedPunc
post-processor, C-ABI + Go/Java/Ruby/JS/Python bindings, CI on 6 platforms.
All backends support `-m auto --auto-download`. Three new ggml ops
(`conv_1d_cf`, `conv_1d_dw_cf`, `conv_1d_group`). ggml bumped to 0.10.0.
Feature matrix expanded to 21 backends (README). test-all-backends.py
passes 18/18 transcribe + 51/54 feature tests (3 stream skips, no failures).

> **‼️ Tooling pin: `clang-format` MUST be v18.** CI pins it
> (`.github/workflows/lint.yml`). Homebrew's default `clang-format` and
> Xcode's bundled `clang-format` both ship v22, which silently
> re-wraps lines and breaks CI lint. Use `./tools/format.sh` (refuses
> non-v18) or the explicit path
> `/opt/homebrew/opt/llvm@18/bin/clang-format`. Never `clang-format`
> bare from `PATH`. See `CLAUDE.md` + `LEARNINGS.md` for the full
> lesson.

---

## Priority ordering

| Priority | Item | Effort | Status |
|---|---|---|---|
| **MEDIUM** | [#52 Qwen3-TTS](#52-qwen3-tts) — perf pass | Medium | talker + code_predictor + codec + ECAPA + codec_encoder all done; only step-4 perf pass open (~137 ms/frame → real-time) |
| **HIGH** | [#57 Commercial-friendly TTS expansion](#57-commercial-friendly-tts-backend-expansion) | Phased | Phases 1-2 DONE; Phase 3 vocoder FIXED (2026-05-03) — full pipeline T3→S3Gen→HiFT→WAV produces correct "Hello world."; remaining: C API integration, F0 predictor, voice cloning |
| **MEDIUM** | [#51c MiMo-V2.5-ASR F16 step decode](#51c-f16-step-decode) | Small | F16 step-decode validation blocked behind ≥32 GB box (see PLAN #51c); base runtime + Q4_K shipped → HISTORY §56 |
| **MEDIUM** | [#56 Kokoro multilingual phonemizer](#56-kokoro-multilingual-phonemizer-espeak-ng) | Small | espeak-ng + DE backbone shipped; HF GGUFs published 2026-05-01; auto-download wired; only Mandarin tones / JA kanji + diff-harness phonemizer-step polish remain |
| **MEDIUM** | [#58 MOSS-Audio-4B-Instruct](#58-moss-audio-4b-instruct) | Large | first audio-understanding (not just ASR) backend; introduces DeepStack cross-layer feature injection |
| **MEDIUM** | [#59 Cross-binding C-ABI parity](#59-cross-binding-c-abi-parity) | Medium | Go now has full surface (✅ all 11 capabilities). Java has transcribe+align+LID. Ruby has transcribe. JS needs WebAssembly approach |
| **DONE** | [#60 llama.cpp/llamafile perf trick ports](#60-cross-backend-perf-tricks-llamacpp--llamafile-ports) | 14 items | 60a-g DONE; 60e Q8_0 KV validated on 7 backends (all bit-exact or WER=0%); 60h-n parked/skip |
| **DONE** | #41 Moonshine IPA / phoneme | — | Superseded by kokoro espeak-ng phonemizer (#56) |
| **PARKED** | [#9 Parakeet TDT GPU](#9-parakeet-tdt-decoder-gpu) | Medium | Encoder 85%+ of time; LSTM+joint <0.7s; sequential steps limit GPU benefit |
| **DONE** | [#11 WebSocket server](#11-websocket-streaming-server) | Medium | RFC 6455 WS on port+1, binary PCM in, JSON text out |
| **DONE** | [#7 voxtral4b streaming](#7-native-voxtral4b-streaming) | High | Phases 1-4 shipped → HISTORY §71 |
| **DONE** | [#62 Streaming + mic library API](#62-streaming--mic-library-api) | M-L | All wrappers ship it |
| **DONE** | [#63 Feature matrix parity](#63-feature-matrix-parity) | Phased | All 9 phases → HISTORY §72 |
| **BLOCKED** | [#42 VibeVoice-ASR 7B](#42-vibevoice-asr-7b) | High | Needs ≥16 GB RAM |
| **BLOCKED** | [#43 Fun-ASR-Nano](#43-fun-asr-nano) | Medium | License unclear |

**Recently completed** (full write-ups in HISTORY.md): #11 WebSocket server → §76, #63 Feature matrix parity → §72, #59 binding parity → §73, gemma4 #49 + Docker #31 → §74, tests + KV Q8_0 + cleanup → §75. Earlier: #5→§63, #16→§55, #51→§56, #51b→§60, #53→§63, #54→§61, #55→§54, #56→§63, #60d→§64.

---





## 40. More Moonshine model variants

Convert + upload to HuggingFace:
- ~~`moonshine-base` (61.5M, better WER)~~ **DONE** (cstr/moonshine-base-GGUF)
- `moonshine-streaming-tiny/small/medium` — different architecture, needs new runtime
- ~~`moonshine-tiny-{ja,ar,ko,zh,vi,uk}` (multilingual)~~ **DONE** (12 repos on HF)
- ~~`moonshine-base-{ja,uk,vi,zh,ar,ko}` (multilingual)~~ **DONE** (12 repos on HF)

Converter fix: 1D tensors (norms, biases) forced to F32; conv_1d_f32 mul_mat
argument order fixed for F16 kernels.

---

## ~~41. Moonshine phoneme / IPA output~~ — **SUPERSEDED by kokoro espeak-ng phonemizer (#56)**

---

## ~~5. Reference backends for parakeet/canary/cohere~~ — **DONE → [HISTORY §63](HISTORY.md)**

---

## 7. Native voxtral4b streaming

### Phase 1 — SHIPPED via batch-encoder-at-flush

**Status (May 2026):** Streaming API surface delivered. `crispasr_session_stream_open`
on a voxtral4b session returns a stream handle that accepts `feed()` /
`flush()` / `get_text()`. Bit-exact-batch on JFK validated. Phase 1
ships as the supported PTT/dictation path (`feed` continuously, `flush`
returns the transcript).

**Implementation:** `voxtral4b_stream_*` C ABI in `src/voxtral4b.{h,cpp}` +
adapter wiring (`src/crispasr_c_api.cpp` ~387/450/543/570/609/3343).
Mirrors the kyutai/moonshine streaming-state field pattern. `feed()`
accumulates PCM only; `flush()` runs the batch mel + encoder once over
the full PCM, then performs the streaming-prompt prefill (BOS + 38
STREAMING_PAD = 39 tokens, audio embeds added to first 39 prompt embeds)
and the per-step audio-injection greedy decode loop, exactly mirroring
`examples/cli/crispasr_backend_voxtral4b.cpp`. Audio is auto-padded with
32 left-pad tokens of silence at stream open and right-aligned + 10
right-pad tokens at flush time.

**Bench (M1, Q4_K, JFK 11s):** feed = 1 ms (no per-chunk work),
flush = 8.5 s, transcript byte-for-byte = batch.

**Latency target NOT met by phase 1:** ≤240 ms first-token requires
incremental encoder during `feed`. Phase 1.5 below.

### Phase 1.5 — SHIPPED, incremental encoder

The incremental encoder is now the default path. Audio embeds bit-match
the batch encoder on JFK (cos≥0.9999 across all non-tail-pad chunks;
the final chunk diverges by construction because batch's last mel frame
includes right-side `center_pad` data that streaming doesn't emit).
Transcript matches batch byte-for-byte. Set
`CRISPASR_VOXTRAL4B_STREAM_BATCH_ENCODER=1` to fall back to the whole-
clip batch encoder at flush (kept as a regression-debug switch).

**Root cause that gated 1.5.** A two-axis layout transpose in
`vox_stream_advance_mel`. `core_mel::compute` with `Layout::MelsTime`
emits `(n_mels, T)` row-major (T fast, n_mels slow), which is exactly
what the encoder graph's mel input expects. The streaming code had a
"transpose on copy" loop that wrote it as `(T, n_mels)` — sending
mel-and-time-axes to the encoder swapped, which produced plausible-
looking but content-incorrect audio embeds. Fix: remove the transpose;
keep `mel_pending` and `conv0_lctx` in `(n_mels, T)` per-band-
contiguous layout end-to-end.

A second smaller fix: F32 K/V cache instead of F16. The batch encoder
runs F32 throughout; the F16 KV cache the LLM uses (which works because
the LLM was trained with F16 KV) was applying to the encoder where it
introduced precision loss across 32 layers. F32 cache is 393 MB at the
SWA cap of 750 frames × 32 layers × 32 heads × 64 head_dim — heavy but
tractable. (TODO: measure whether F16 cache loses cos≥0.999 once the
mel layout is correct; might be safe to revert.)

**Wall-clock (M1, Q4_K, JFK 11 s, no other workload).**
- feed = ~24 s (~170 ms per 80 ms chunk = 2.1× realtime). Encoder is
  the dominant cost during feed.
- flush = ~10 s (LLM decode-loop dominates).
- Combined = ~34 s for 11 s audio = 0.32× realtime. Same total cost as
  batch transcribe; streaming distributes it across feed + flush
  instead of bunching it at flush.

**Bit-exact-batch validation:** `tools/bench_streaming_latency.py
--check-batch-equality` PASSES on JFK with the incremental encoder
default-on.

### Phase 2 — SHIPPED (chunk-size + fused QKV + timing instrumentation)

**Shipped May 2026.**
- **240 ms internal encoder chunks default** (`CRISPASR_VOXTRAL4B_STREAM_CHUNK_MS`
  override). 2.6× feed speedup vs the original 80 ms chunks (24 s →
  9.3 s for JFK 11 s on M1 Q4_K). Bit-exact-batch unaffected.
- **Per-stage timing instrumentation** (`CRISPASR_VOXTRAL4B_STREAM_TIMING=1`).
  Stderr prints encoder-drain / prefill / first-text-token / per-step
  p50/p95 / total flush wall-clock. Used as the phase-2 perf-pass dev
  loop.
- **Default-unification fix.** feed() defaulted to incremental encoder,
  flush() defaulted to batch encoder. The mismatch made flush re-run
  the whole batch encoder despite feed having done it already. Unified
  to incremental on both paths (`CRISPASR_VOXTRAL4B_STREAM_BATCH_ENCODER=1`
  to opt out). Saved ~1 s on flush; first-text-token 2.7 s → 1.6 s.
- **Runtime fused QKV (LLM)**. Concat each layer's q/k/v weights along
  the output axis at load time into a single (d_model, q_dim+2*kv_dim)
  tensor; route through `core_attn::kv_self_attn`'s `qkv_w` path.
  Extends the qwen3_asr precedent to handle Q4_K (and any row-wise
  quantized format) by byte-concat. ~7–8 % decode speedup (56 → 50.4 ms
  per step). `CRISPASR_VOXTRAL4B_FUSED_QKV=0` to opt out.
- FFN gate+up fuse was tried and reverted — Metal's Q4_K matmul kernel
  for the FFN dimension (3072 × 9216) is already memory-bandwidth-bound,
  so combining two into (3072 × 18432) didn't help. The
  `core_ffn::swiglu_fused_gate_up` helper stays in place for any
  future caller where the ratio is more favourable.

**Final phase 2 numbers** (M1 Q4_K JFK 11 s, all phase 1+1.5+2 wins):
- feed:  23 s → 9.1 s (2.5× faster)
- flush: 10 s → 8.3 s
- decode: 56 → 50.4 ms/step
- first-text-token: 2.7 s → 1.6 s
- bit-exact-batch: PASS

**Architectural finding (M1 Q4_K JFK 11 s).** The ≤240 ms first-text-
token target is **bounded below by the streaming-prompt convention**:

| Stage | ms (post-phase-2) |
|---|---|
| Encoder drain at flush (right-pad encoder + projector) | ~990 |
| LLM prefill (39-token streaming-prompt) | 252 |
| 8 streaming-pad warmup steps × 50.4 ms each | ~403 |
| First text-emitting decode step | ~50.4 |

Even with a fully-warm encoder (encoder drain → 0), the prompt
convention forces ≥ prefill (252) + 8 × decode (403) = **655 ms
before the first text token can emit**. Beating 240 ms requires
either a different prompt convention (model retraining) or a
substantially faster Q4_K Metal kernel — neither is a quick win.

### Phase 3 partial — combined-chunk flush + speculative prefill (May 2026)

**Shipped May 2026.**
- **Combined-chunk flush.** The right-pad zeros at flush were being
  fed through `voxtral4b_stream_feed` which ran 3-4 separate 240 ms
  encoder chunks (plus tail-pad + per-chunk projector). On M1 Q4_K
  the per-chunk Metal kernel-launch overhead doesn't amortise well
  for the small final chunks. Refactored: append right-pad zeros
  directly to `pcm_with_pad` without triggering the per-feed encoder
  loop; then drain all pending mel (residual + right-pad's ~80
  frames) via one larger combined chunk (~96 mel frames = 48 enc
  frames = 12 projector groups). Saves ~6 kernel launches.
  **Encoder drain at flush: 990 ms → 307 ms; first-text-token: 1646 ms
  → 921 ms.**
- **Speculative LLM prefill during feed.** Once feed has produced ≥
  39 audio_embeds (after ~3.1 s of audio at 240 ms chunks), run the
  streaming-prompt prefill speculatively and stash the resulting
  last-position logits + n_past on the stream. Flush then skips the
  prefill (~250 ms cost) and jumps straight to the decode loop.
  No correctness risk: the LLM's KV cache state at position 39 is
  identical regardless of when prefill runs.
  **First-text-token: 921 ms → 650 ms.**

**Final phase 1+1.5+2+3-partial numbers (M1 Q4_K JFK 11 s):**

| Metric | Start | End | Δ |
|---|---|---|---|
| feed total | 24 s | 9.4 s | 2.5× faster |
| flush total | 10 s+ | 7.5 s | -2.5 s+ |
| per-decode-step | 56 ms | 50.4 ms | -10 % |
| **first-text-token** | **2674 ms** | **650 ms** | **4.1× faster** |
| bit-exact-batch | PASS | PASS | unchanged |

The 650 ms first-text-token is now within 2.7× of the ≤240 ms target.
Remaining gap is the model's 8-step streaming-pad warmup × 50.4 ms =
~400 ms (architectural — model needs to "buffer" enough audio
context before emitting text), plus the encoder drain at flush
(~290 ms, dominated by right-pad encoder work). The architectural
floor of `~400 ms warmup + first-decode` means ≤240 ms first-text
remains out of reach without retraining the model with a different
prompt convention.

### Phase 3 — SHIPPED — live captions during speech (May 2026)

**Live captions with no stable-prefix heuristic.** Once feed has
produced ≥39 audio_embeds (after ~3.1 s of audio), every new
audio_embed during subsequent feeds drives one greedy decode step;
tokens commit immediately to `out_text` / `out_text_unread`, so
`get_text()` polled during feed returns progressive transcript.

**Why no stable-prefix needed:** voxtral4b's audio-injection
pre_hook makes each decoded token a deterministic function of the
audio context up to that point. Tokens commit immediately — no
retraction. This is the key architectural difference from
encoder-decoder ASR (whisper / parakeet / canary) where the
encoder's bidirectional context shifts as more audio arrives,
making mid-decode tokens unstable and requiring a stable-prefix
commit heuristic.

**API:** `voxtral4b_stream_set_live_decode(stream*, int enabled)`
toggles per-stream. Generic dispatch via
`crispasr_stream_set_live_decode` for use through the unified
`crispasr_stream*` handle. Python: `Session.stream_open(live=True)`.
Default off (PTT semantics preserved).

**Smoke result on JFK 11 s with `live=True`:**
```
+ 1280ms: ' And'
+ 1760ms: ' so,'
+ 2000ms: ' my'
+ 2240ms: ' fellow'
+ 3200ms: ' Americans,'
...
flush:   ' country.'
```
Final concatenated transcript matches batch byte-for-byte.

**Limitation:** sequential live decode is ~1.5× realtime on M1 Q4_K
(50 ms decode + 100 ms encoder per 100 ms audio chunk = 150 ms
processing per 100 ms audio). Falls behind realtime audio when fed
from a live mic at audio rate — phase 4 (decoder thread parallel
to encoder) would fix this. Today's path is useful for:
faster-than-realtime offline streaming, post-utterance live
caption rendering, and as the substrate for phase 4.

### Phase 4 — SHIPPED — decoder thread (May 2026)

**Optional worker thread** that drains decode steps in the background
while feed() handles encoder + projector on the main thread. Enable
via `CRISPASR_VOXTRAL4B_STREAM_DECODER_THREAD=1` (implies live mode).

**Architecture:**
- `worker_thread` sleeps on `cond_var` until shutdown OR
  audio_embeds grow past decode_adapter_pos
- feed() acquires `sched_mutex` around encoder/projector/prefill,
  then notifies cond_var so worker picks up new audio_embeds
- worker drains under sched_mutex + decode_state_mutex
- flush() signals + waits until worker is idle + caught up to
  N_audio (busy-wait with 1ms sleeps + cond_var re-notifies)
- close() requests shutdown + joins the worker

**Performance characterisation on M1 (JFK 11 s):**
- PTT: feed 9.3 s, flush 7.3 s — bit-exact-batch
- LIVE single-thread: feed 15.6 s, flush 1.0 s — bit-exact-batch
- LIVE + decoder thread: feed 15.7 s, flush 1.0 s — bit-exact-batch

The thread doesn't reduce total wall-clock on M1 because Metal's
single GPU queue serializes encoder and decoder regardless of how
many CPU threads submit work. **The architectural win is for:**
1. **Mic-driven workloads** with audio-rate gaps between feeds: the
   worker drains decode during user-typed/spoken pauses; feed()
   returns between encoder chunks without waiting for the entire
   decode loop.
2. **Faster GPUs** with kernel-level parallelism (M3 Ultra, NVIDIA):
   Metal/CUDA can overlap concurrent compute kernels submitted by
   different ggml_backend_sched_compute calls, giving real
   encoder+decoder overlap.

### Phase 5 (deferred) — dual-sched for true on-Metal parallelism

- **Two ggml_backend_sched on the same backend.** Encoder uses one,
  LLM uses the other. They submit independently to the Metal
  command queue. On a GPU with kernel-level parallelism, encoder
  and LLM kernels overlap on different SIMDgroups. Estimated 30-40 %
  total wall-clock reduction on M3 Ultra. ~150 LOC of sched
  duplication + per-call routing in voxtral4b_run_llm_kv.
- **Right-pad encoder pipelining (full speculative).** Pre-encode
  right-pad zeros during user idle time with state save/restore.
  Could shave another ~290 ms off first-text-token but interacts
  with phase 5's two-sched architecture.

---

## 9. Parakeet TDT decoder GPU

Port LSTM predictor + joint head from CPU loops to ggml graphs. LSTM
is sequential → per-step kernel launches. Encoder already 85%+ of time.

**Assessment (May 2026):** JFK 11s takes 4.39s total. Encoder dominates
(~3.7s). The LSTM predictor (2×640×640) + joint head (640→8198) run
~22 steps for JFK — the CPU loops take <0.7s. The LSTM is inherently
sequential (each step depends on prev hidden state), so GPU kernel
launch overhead would eat most of the theoretical gain. On CPU, the
tight C loops are already near-optimal for these matrix sizes.

**Verdict:** PARKED. Not worth the complexity. Would only matter for
GPU inference on very long audio (100+ tokens), where the encoder
speedup from GPU is already the dominant improvement.

**Effort:** ~150 LOC. Small gain.

---

## ~~11. WebSocket streaming server~~ — **DONE → [HISTORY §76](HISTORY.md)**

---


## ~~16. Shaw RPE for granite graph~~ — **DONE → [HISTORY §55](HISTORY.md)**

`GRANITE_DISABLE_ENCODER_GRAPH=1` is the unified escape hatch.

---


## 42. VibeVoice-ASR 7B

**BLOCKED:** Needs ≥16 GB RAM for conversion. Converter OOMs on 8 GB due
to Qwen2.5-7B embedding (152064 × 3584 = 2.1 GB F32).

**Fix:** Use `safe_open` per-tensor conversion. Then Q4_K → ~4 GB.

Full architecture analysis in HISTORY.md #34. C++ runtime partially
implemented (`src/vibevoice.cpp`). F16 im2col precision issue in
depthwise conv needs fixing.

---

## 43. Fun-ASR-Nano

**BLOCKED:** License unclear. Issue filed at `FunAudioLLM/Fun-ASR#99`.
No response. HF model card has no license field.

---

## ~~51. MiMo-V2.5-ASR runtime~~ — **DONE → [HISTORY §56](HISTORY.md) + [§64](HISTORY.md)**

Base runtime + Q4_K + fused-QKV layout shipped. Sub-items 51a (mmap
loader → [HISTORY §62](HISTORY.md), env flag `CRISPASR_GGUF_MMAP=1`)
and 51b (step-decode KV cache reuse → [HISTORY §60](HISTORY.md))
also DONE. Only 51c (F16 step decode) is still open — blocked
behind ≥32 GB RAM for end-to-end validation.

### 51c. F16 step decode

Q4_K dequant on every matmul is the largest single cost at decode
time. F16 weights are ~2× larger but skip the dequant loop
entirely.

**Status (May 2026): code path works, validation deferred to a
larger-RAM box.**

PLAN #51a's CPU mmap loader landed (commit `9710f80`) — Metal
mmap loader landed too (same commit) — and #60a added the
`posix_madvise(WILLNEED)` readahead hint (commit `f1f4bce`).
Together these mean **no code change is needed for 51c** — just
point `crispasr` at the F16 GGUF with `CRISPASR_GGUF_MMAP=1`. We
verified the load path works (no OOM, mmap'd weights at 1.9 GB
RSS on a 16 GB box, prefill compute starts).

What we couldn't validate end-to-end on this box:

- **JFK transcript byte-equality on F16**: prefill compute
  thrashes because the 16 GB F16 working set doesn't fit in 16 GB
  RAM. Pages get evicted as compute walks layers, every
  re-access faults from the disk5 external (99% full, often
  contended by other workers). One bench attempt ran for 51 min
  with 0.1% CPU and never finished prefill.
- **Decode speedup measurement**: same root cause — needs warm
  cache, which we can't achieve.

The ceiling is **hardware, not code**: 16 GB F16 weights need
≥20 GB RAM to comfortably fit + leave headroom for activations +
KV cache + audio tokenizer. On a 32+ GB box this should "just
work" and hit the work order's ≥1× realtime target.

Files **not** touched (no code change required):
- `src/mimo_asr.cpp` — the runtime is dtype-agnostic; F16 weights
  flow through the existing `core_attn::kv_self_attn` matmul kernels
  on Metal without modification.
- `src/core/gguf_loader.cpp` — already wired (60a + #51a).

Validation deferral notes:
- Run `CRISPASR_GGUF_MMAP=1 ./build-ninja-compile/bin/crispasr --backend mimo-asr -m /path/to/mimo-asr-f16.gguf --codec-model /path/to/mimo-tokenizer-q4_k.gguf -f samples/jfk.wav` on a 32+ GB box to validate transcript + bench.
- If F16 prefill hits ≥1× realtime as predicted, ship the F16
  GGUF as the recommended quant and demote Q4_K to a memory-tight
  fallback. Until then both are shipped on `cstr/mimo-asr-GGUF`
  with Q4_K as the default.

Effort: **0 LOC** (validation only). The originally-scoped
"Effort: Small" assumed code work that turned out to be unneeded
once the mmap loader landed.

---

## 52. Qwen3-TTS

User-requested follow-on to the VibeVoice TTS work. Apache-2.0
collection: [Qwen/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
[HF collection](https://huggingface.co/collections/Qwen/qwen3-tts).

- **Six repos in the collection** (all BF16 safetensors, Apache 2.0):
  - `Qwen/Qwen3-TTS-Tokenizer-12Hz` — RVQ codec, 16 codebooks × 2048,
    12.5 FPS at 24 kHz. Non-DiT lightweight architecture (8L
    encoder + 8L decoder).
  - `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base` — base talker LM with
    voice clone (3s reference audio).
  - `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice` — fine-tuned,
    fixed speakers.
  - `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` — instruction-tuned
    (voice description → speech).
- **Architecture:** "Discrete Multi-Codebook LM" — Qwen3 backbone
  with a 16-codebook output head. No DiT; direct AR generation of
  RVQ codes. ~97ms end-to-end latency, 10 languages incl.
  en/de/zh/ja/ko/it.
- **Status (May 2026):** **base + CustomVoice + VoiceDesign 0.6B/1.7B all live** — talker forward, ICL prefill, code-predictor sampling, codec decoder, ECAPA speaker_encoder forward, codec encoder forward all DONE. ASR roundtrip word-exact across all variants. Open: only the **performance pass** below.
- **Shipped milestones** (commit references in HISTORY §57/§58 + per-model status table under #57):
  1. ✓ Talker forward (28L Qwen3 + Q/K-norm + flash-attn + F16 KV cache) — `talker_logits` cos=1.000000 (`2b85b78`).
  2. ✓ ICL prefill builder — `talker_logits_via_icl` cos=1.000000 (`b939d4f`).
  3. ✓ Code predictor with sampling — fixed silent-output trap (`9608202`, `69c135c`).
  4. ✓ TTS→ASR roundtrip on parakeet-v3.
  5. ✓ Codec decoder (Tokenizer-12Hz) — diff harness 8/8 PASS at cos≥0.999983 (`d1f47b1`, `48c6c1a`). Required a Metal `kernel_conv_transpose_1d` patch in our ggml fork (input-range tightening — see LEARNINGS, MUST RE-APPLY on every ggml bump).
  6. ✓ ECAPA speaker_encoder runtime forward — cos=0.999999 (`c0a9cb3`, `8a4c49e`, `38040b4`). C ABI: `qwen3_tts_compute_speaker_embedding(audio, n, sr)` + `qwen3_tts_set_voice_prompt[_with_text]`.
  7. ✓ Codec encoder runtime forward — diff 3 stages cos≥0.999 (`ef11c01`, `10302b4`). Closes the bake-script loop.
- **Performance pass (in progress, partial wins shipped).** Quiet-bench Q8_0 0.6B with all defaults: ~96 ms/frame (talker ~49 + cp ~45). Real-time at 12.5 fps = 80 ms/frame, so ~16 ms/frame still over budget; talker compute is the dominant remaining cost. Shipped: **`QWEN3_TTS_O15=1` is default-on** (commit `5e21e4a`) — cp graph reuse saves ~14 ms/frame on cp_pred under contention, ~2-3 ms/frame quiet, bit-identical WAV. Gated, byte-identical, kept default-OFF: `QWEN3_TTS_FUSED_QKV=1` (talker fused QKV, F16/F32 only, no clean quiet bench yet); `QWEN3_TTS_LK_BUCKET=1` (talker Lk bucketing, **net loss on M1 Metal Q8_0** — see LEARNINGS); `QWEN3_TTS_CP_STEP0_CACHE=1` (cp T=2 step-0 graph cache, ~1-3 ms/frame quiet savings, bit-identical). Investigated: Q8_0 KV cache — blocked on Metal `cont(Q8_0)` source (only F32/F16/BF16 sources supported); needs Metal kernel patch or KV layout restructure to land. Still open: F16 FUSED_QKV clean quiet-machine bench (the existing impl + bench harness needs a contention-free run to land a default-flip decision); Q4_K talker fused QKV; the larger lift of fusing 15 cp steps into one graph (needs on-device top-k sampling, ~3 ms/frame upper bound after O15 since most overhead is already gone).
- Debug knobs: `QWEN3_TTS_{BENCH,DEBUG,DUMP_DIR}` env vars; diff harness via `tools/reference_backends/qwen3_tts.py` + `crispasr-diff qwen3-tts`.
- **Reuse:** the talker is essentially Qwen3-0.6B/1.7B with a
  multi-codebook output head — `core_attn::kv_self_attn` +
  `core_ffn::swiglu` again. The codec needs new code for RVQ
  decoding; that work is shared with MiMo (#51) and overlaps in
  shape with the VibeVoice σ-VAE decoder, so a `core_audio_decoder`
  helper is worth landing alongside the runtime (see #53).

**Effort:** Large. ~1500 LOC across runtime + codec + reference
backend. The two TTS targets (Qwen3-TTS and any future expansion)
share enough that landing one substantially de-risks the other.

---

## ~~54. granite-speech-4.1 plus / nar variants~~ — **DONE → [HISTORY §61](HISTORY.md)**

All three variants (`granite-4.1`, `granite-4.1-plus`, `granite-4.1-nar`) shipped bit-exact on JFK; HF GGUFs published. Open follow-up: speaker labels + word-level timestamps for the `plus` variant via chat_template (~50 LOC, template-only).

---

## ~~53. Two narrow extractions for shared TTS-codec patterns~~ — **DONE → [HISTORY §63](HISTORY.md)**

`core_act::snake_beta` + `core_convt::convt1d_crop` shipped (qwen3-tts codec + SNAC both delegate).

---


## ~~55. granite-family DRY refactor~~ — **DONE → [HISTORY §54](HISTORY.md)**

---


## 56. Kokoro multilingual phonemizer (espeak-ng)

Kokoro/StyleTTS2 is multilingual at the model level — the 178-symbol IPA
vocab covers en, de, fr, ru, cmn, ja and more — but until this work the
runtime always shelled out to `popen("espeak-ng -q --ipa=3 -v LANG …")`,
which (a) cost ~30–50 ms per call on the shell-quoting + fork path,
(b) needed `espeak-ng` on `$PATH`, and (c) emitted U+200D ZWJ tie
characters and newline-separated sentence chunks that the GGUF
tokenizer then has to silently absorb.

This item replaces the popen path with in-process libespeak-ng calls
behind a CMake AUTO probe, while keeping popen as a runtime fallback
so existing builds don't regress.

### Done (this session)

- `src/CMakeLists.txt`: `CRISPASR_WITH_ESPEAK_NG` cache string
  (`AUTO`/`ON`/`OFF`, default `AUTO`). AUTO probes `pkg-config
  espeak-ng` first, then a Homebrew/Linux fallback
  (`/opt/homebrew`, `/usr/local`, `/usr`). When found, defines
  `CRISPASR_HAVE_ESPEAK_NG=1` and links `libespeak-ng` via PUBLIC so
  it propagates into `crispasr` / `libcrispasr.dylib`. `ON` makes a
  missing lib a hard error; `OFF` skips the probe entirely.
- `src/kokoro.cpp`:
  1. `kokoro_phoneme_cache` — bounded LRU (1024 entries,
     mutex-protected) keyed on `lang \0 text`, lives in
     `kokoro_context`.
  2. `phonemize_espeak_lib()` — gated on `CRISPASR_HAVE_ESPEAK_NG`.
     Lazy `espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, …,
     espeakINITIALIZE_PHONEME_IPA | espeakINITIALIZE_DONT_EXIT)`
     behind a process-global mutex; sticky-init-failure flag so we
     don't keep retrying. `CRISPASR_ESPEAK_DATA_PATH` env var
     overrides the data dir for sandboxed apps. Voice changes are
     sticky. Loops `espeak_TextToPhonemes` until `textptr==NULL`,
     joining chunks with spaces.
  3. `phonemize_popen()` — the old shell-out, kept as a runtime
     fallback. `kokoro_synthesize` now calls `phonemize_cached()`
     which tries cache → lib → popen.
- `examples/cli/crispasr_backend_kokoro.cpp`: maps `-l/--language`
  to `cp.espeak_lang`. `auto` keeps the default (en-us) since
  espeak has no auto-detect mode.
- Smoke-tested standalone against libespeak-ng: en-us, de, fr,
  cmn, ru, ja all produce IPA. Compared lib vs popen: see
  LEARNINGS.md "Kokoro phonemizer: libespeak-ng vs popen
  divergence" for the ZWJ + sentence-join behaviour.
- Build verified: `otool -L libcrispasr.dylib` shows
  `libespeak-ng.1.dylib`; `nm libkokoro.a` has the three espeak
  symbols.
- **End-to-end synth check** (against
  `/Volumes/backups/ai/crispasr-models/kokoro-82m-f16.gguf` +
  `kokoro-voice-af_heart.gguf`):
  | lang | phonemes | duration | peak | RMS | verdict |
  |---|---|---:|---:|---:|---|
  | en  | clean | 3.45 s | 11443 | 1545 | ✅ healthy |
  | de  | clean | 4.08 s |   541 |   44 | ❌ near-silence on long phrases (no German voice — see open #1) |
  | fr  | clean | 3.40 s | 12374 | 1434 | ✅ healthy |
  | ru  | clean | 3.38 s | 11375 | 1506 | ✅ healthy |
  | cmn | espeak tone numbers (`ni2χˈɑu2…`) | 3.20 s | 11731 | 1627 | ⚠️ audio plays but tones unmodelled — open #2 |
  | ja  | kanji fallback (`(en)tʃˈaɪniːz(ja)…`) | 8.38 s | 15460 | 1581 | ⚠️ partial — kana works, kanji becomes English — open #3 |

  Short German phrases ("Hallo Welt.", "Guten Morgen.") synthesize
  fine with `af_heart`; the silence collapse only triggers on longer
  out-of-distribution phoneme sequences. See LEARNINGS.md "Kokoro
  phonemizer: libespeak-ng vs popen divergence" for full results.

### Open

1. **German voice pack — DE is a primary target language.** Kokoro-82M
   ships voices only for `a/b` (en US/UK), `e` (es), `f` (fr), `h` (hi),
   `i` (it), `j` (ja), `p` (pt), `z` (zh). No `d_*` (de), no `r_*` (ru),
   no Korean/Arabic. Three options ordered by effort:

   **Option 1 — Closer-language voice fallback (SHIPPED 2026-05-01).**
   Measured against the long German phrase ("Guten Tag, dies ist ein
   Test des deutschen Phonemizers."):

   | voice | peak | RMS | duration | verdict |
   |---|---:|---:|---:|---|
   | `af_heart` (English) |   541 |   44 | 4.08 s | silence collapse |
   | `ff_siwis` (French)  | 20577 | 2318 | 4.22 s | healthy, French-accented |
   | `ef_dora` (Spanish)  | 15036 | 1613 | 3.35 s | healthy, Spanish-accented |

   Wired into `examples/cli/crispasr_backend_kokoro.cpp` as an
   auto-fallback. Selection table:

   | `-l` value | preferred voice | rationale |
   |---|---|---|
   | `de`, `de-*`, `de_*` | `df_victoria` (Option 2b — kikiri-tts, Apache-2.0) → `df_eva` (Option 2a — Tundragoon, Apache-2.0) → `ff_siwis` | in-distribution to dida-80b backbone first; Tundragoon as second tier; French as last resort |
   | everything else without a native pack (ru, ko, ar, …) | `ff_siwis` (French) | non-silence baseline |

   Resolution: `--voice` (explicit) → cascade above → empty (helpful
   error). Explicit `--voice` always wins. Voice GGUFs live at
   `/Volumes/backups/ai/crispasr-models/kokoro-voice-{af_heart,
   ef_dora, ff_siwis, df_eva, dm_bernd, df_victoria, dm_martin}.gguf`.

   **Option 2a — Recovered Tundragoon's German voice packs (DONE,
   SHIPPED 2026-05-01).**
   The only public German Kokoro voice pack on HF was
   `Tundragoon/Kokoro-German` (Apache-2.0) — the user account was
   deleted in early 2026 and the HF repo is 404. **Voices recovered**
   from `r1di/kokoro-fastapi-german`'s Git LFS (`api/src/voices/v1_0/
   {df_eva,dm_bernd}.pt`, sparse + LFS pull). They are
   `[512, 1, 256]` F32 (vs the 510 of official Kokoro voices —
   Tundragoon's fine-tune used a slightly larger max_phonemes; the
   GGUF voice loader reads max_phonemes from the file so this is fine).

   End-to-end synth with the **official** Kokoro-82M model on the
   long German phrase ("Guten Tag, dies ist ein Test des deutschen
   Phonemizers."):

   | voice | peak | RMS | duration | note |
   |---|---:|---:|---:|---|
   | `df_eva` (German F)  | 14716 | 1648 | 3.50 s | healthy, German speaker |
   | `dm_bernd` (German M)| 19185 | 2374 | 3.88 s | healthy, German speaker |

   Both produce non-silent, German-timbred audio with the official
   Kokoro-82M weights — **the matching Tundragoon model fine-tune
   (`kokoro-german-v1_1-de.pth`) is not required.** That model is
   *unrecovered* (only available from the deleted HF repo per
   `r1di/docker/scripts/download_model.py`), but voices alone are
   sufficient for this fallback path. Caveat: predictor + decoder
   weights are still the official English-trained Kokoro-82M's, so
   prosody is not fully native German. Better than ff_siwis (German
   speaker timbre instead of French), worse than Option 2b.

   GGUF artefacts at
   `/Volumes/backups/ai/crispasr-models/kokoro-voice-{df_eva,dm_bernd}.gguf`.
   Wired as the German auto-fallback (Option 1 table above).

   **Option 2b — Native German backbone via dida-80b (SHIPPED 2026-05-01).**

   Sources (all Apache-2.0 weights + Apache-2.0 recipe + CC0 dataset):
   - Recipe: <https://github.com/semidark/kokoro-deutsch> — clone
     locally (recurse-submodules: `StyleTTS2/` + `kokoro/`).
     `scripts/extract_voicepack.py` is the tool for fresh per-speaker
     voicepacks; we did not need to run it (kikiri-tts ships
     pre-extracted voicepacks — see below).
   - Backbone: <https://huggingface.co/dida-80b/kokoro-german-hui-multispeaker-base>
     — `first_stage.pth` + `config.json`. Stage-1 multispeaker base
     fine-tune of Kokoro-82M on HUI-Audio-Corpus-German (51 speakers,
     51 h, 10 epochs A40, mel loss 0.583 → 0.326).
   - Pre-extracted voicepacks (kikiri-tts org, dida-80b maintainer):
     <https://huggingface.co/kikiri-tts/kikiri-german-victoria> +
     <https://huggingface.co/kikiri-tts/kikiri-german-martin>. Each
     ships `voices/{victoria,martin}.pt` extracted via the kikiri
     synthetic StyleEncoder which shares lineage with the dida-80b
     base — saves us from running `extract_voicepack.py` ourselves
     (the underlying HUI corpus is gated and would require a multi-step
     LibriVox-pulling pipeline to reproduce).

   What this adds over Option 2a:
   - **Predictor + decoder are German-trained.** Solves the root
     cause behind the af_heart silence collapse on long German
     phrases — voices alone (Option 2a) only cover the speaker
     timbre, not the prosody/duration distribution.
   - StyleEncoder is German-trained → kikiri voicepacks are in-
     distribution. Pairs cleanly with the dida-80b backbone.

   Steps taken:
   1. ✓ `models/convert-kokoro-to-gguf.py` extended for the modern
      `torch.nn.utils.parametrize` WeightNorm form
      (`parametrizations.weight.original0/original1`) used by dida-80b,
      tolerated the missing `module.` DataParallel prefix on bert keys,
      and added `--config` so the official Kokoro-82M `config.json`
      can be reused (dida-80b ships only a HF-hub stub config without
      vocab; the 178-symbol IPA vocab IDs are byte-identical per
      semidark's `training/kokoro_symbols.py`).
   2. ✓ Converted to
      `/Volumes/backups/ai/crispasr-models/kokoro-de-hui-base-f16.gguf`
      (163.7 MB at F16; 459 tensors mapped, 0 skipped — same byte size
      as `kokoro-82m-f16.gguf`, confirming identical architecture).
   3. ✓ Pulled kikiri voicepacks `voices/{victoria,martin}.pt`
      (510×1×256 F32) via `huggingface_hub.hf_hub_download` and
      converted them with the existing
      `models/convert-kokoro-voice-to-gguf.py` to
      `kokoro-voice-{df_victoria,dm_martin}.gguf` (~510 KB each,
      `[510,1,256]` F32 — direct passthrough, no converter changes).
   4. ✓ C ABI: new `crispasr_kokoro_resolve_model_for_lang()` and
      `crispasr_kokoro_resolve_fallback_voice()` in `src/kokoro.h` /
      `src/kokoro.cpp`, re-exported with the `_abi` suffix from
      `src/crispasr_c_api.cpp` so the dylib (and every wrapper that
      links against it) gets them.
   5. ✓ CLI: `examples/cli/crispasr_backend_kokoro.cpp` now delegates
      to the C ABI. When `-l de*` AND the user-passed model basename
      starts with `kokoro-82m`, the backend silently swaps to a
      sibling `kokoro-de-hui-base-f16.gguf` if present, then loads
      the German fallback voice from the new cascade
      `df_victoria → df_eva → ff_siwis`.
   6. ✓ Python wrapper: `crispasr.kokoro_resolve_for_lang(model, lang)`
      returns `KokoroResolved(model_path, voice_path, voice_name,
      backbone_swapped)`; surfaced from `crispasr/__init__.py`.

   End-to-end measurements on the long German phrase
   ("Guten Tag, dies ist ein Test des deutschen Phonemizers."), each
   ASR-roundtripped through `parakeet-v3 -l de` so we measure
   intelligibility and not just envelope:

   | model + voice | peak | RMS | sec | ASR roundtrip |
   |---|---:|---:|---:|---|
   | official + df_eva (Option 2a) | 14726 | 1648 | 3.50 | "...Phonemizer." (lost trailing 's') |
   | dida-80b + df_eva             | 23477 | 1830 | 3.50 | "...Phonemetzes." (1 word boundary error) |
   | dida-80b + df_victoria        | 12052 | 1177 | 4.22 | "...Tester des Deutschen Phonemizers." (1 word boundary error) |
   | dida-80b + dm_bernd           | 18948 | 2693 | 3.88 | "...Phonemetzers." (1 word boundary error) |
   | **dida-80b + dm_martin**      | 18100 | 1546 | 3.98 | **"...Phonemizers." (perfect)** |

   All four German voices clear the gate (peak ≥ 8000, RMS ≥ 1000)
   on the dida-80b backbone, and three of four are word-perfect except
   for one minor token-boundary error each. dm_martin is byte-perfect
   round-trip; df_victoria handles "Phonemizers" correctly which df_eva
   misses. This is the "fully native German signal path" the option
   promised: predictor + decoder + StyleEncoder distribution all
   German.

   For deployable single-speaker production quality, run Stage-2
   fine-tuning on one HUI speaker (~half-day on an A40) — out of
   scope of this PLAN item; track separately if needed.

   **Option 3 — Extract a style embedding via the English-trained
   StyleEncoder (only if 2a + 2b are blocked).**
   Same recipe as Option 2a's recovery effort but starting from a
   fresh German recording (Common Voice DE, public-domain
   audiobook). `[max_phon=510, 1, 256]` style tensor through
   StyleTTS2's StyleEncoder, save as `.pt`, convert. Strictly worse
   than Option 2b because the predictor/decoder aren't German-aware;
   keep as last-resort.

   **Status:**
   1. ✓ Option 1 shipped (auto-fallback table per-language).
   2. ✓ Option 2a shipped (df_eva + dm_bernd recovered from r1di's
      Git LFS, Apache-2.0; works with both backbones).
   3. ✓ Option 2b SHIPPED (dida-80b backbone + kikiri-tts voicepacks,
      all Apache-2.0; truly native German prosody on long phrases).
      Auto-routing kicks in when both `kokoro-82m-f16.gguf` and
      `kokoro-de-hui-base-f16.gguf` sit in the same directory.
   4. Option 3 not needed.

   **Follow-ups:**
   - ✅ HF GGUF mirrors published (2026-05-01):
     [`cstr/kokoro-82m-GGUF`](https://huggingface.co/cstr/kokoro-82m-GGUF),
     [`cstr/kokoro-de-hui-base-GGUF`](https://huggingface.co/cstr/kokoro-de-hui-base-GGUF),
     [`cstr/kokoro-voices-GGUF`](https://huggingface.co/cstr/kokoro-voices-GGUF)
     — F16 + Q8_0 backbones (Q4_K dropped — see LEARNINGS), 7 voicepacks.
   - ✅ Auto-download via `src/crispasr_model_registry.cpp` (PLAN #56).
     New `ExtraCompanion` mechanism in the registry — backends with >1
     auxiliary file (kokoro: English voice + German backbone + German
     voice) can list extras alongside the inline `companion_file`.
     `crispasr --backend kokoro -m auto -l de` now pulls all 4 files
     and auto-routes to the German backbone.
   - ✅ Wrapper TTS surface across Rust/Go/Java/JS/Ruby
     (commit `4f476c3`, 2026-05-01). Each binding gets
     `Session.{open,setVoice,setCodecPath,synthesize,close}` plus
     `kokoroResolveForLang(model, lang)` returning the same
     `KokoroResolved` shape as the Python wrapper.
   - Stage-2 fine-tune on one HUI speaker (~half-day A40) for
     deployable single-voice production quality. Out of scope here.
2. **Mandarin tone numbers.** espeak-ng outputs digit-suffixed
   tone markers (`ni2χˈɑu2`) that aren't in the kokoro-82m IPA vocab
   (178 symbols) and likely get dropped at tokenization, losing tone
   info. Investigate whether `--ipa=2` (without tone numbers) plus a
   separate tone embedding would work, or whether to switch to a
   different Mandarin G2P (e.g. `pypinyin`).
3. **Japanese kanji.** espeak-ng falls back to English pronunciation
   for kanji (e.g. 日本語 → "Chinese letter"), inserting `(en)…(ja)`
   voice-switch markers that aren't IPA. For full Japanese support,
   pre-process input with a Japanese frontend (`pyopenjtalk` /
   `mecab` + `kakasi`) to convert kanji → kana before espeak.
4. ~~**Diff harness reference backend.**~~ **DONE — phonemizer-step
   diff (May 2026).** The model-side reference dumper at
   `tools/reference_backends/kokoro.py` already covered the 16 model
   stages; the phonemizer step is now covered by a separate sibling
   tool `tools/check_kokoro_phonemizer_parity.py` that exercises the
   newly-exposed `kokoro_phonemize_text_{lib,popen}` C ABI on a fixed
   `(lang, text)` suite (en / de / fr / ru / cmn / ja / it / es / pt)
   and reports drift between the two paths. Default mode normalises
   away the documented benign U+200D ZWJ tie chars (LEARNINGS §6);
   `--strict` does byte-exact comparison. Initial run surfaces 1 real
   substantive divergence in cmn (`ni2χˈɑu2` vs `niɜχˈɑ‍u2`) — that's
   #56 #2's symptom, captured automatically now. No-model unit tests
   in `tests/test_python_session.py` cover the symbol export +
   null-args return path.
5. ~~**Optional polish.**~~ **DONE + CROSS-BINDING.**
   `kokoro_phoneme_cache_clear()` + session-scoped
   `crispasr_session_kokoro_clear_phoneme_cache()` ABI exports for
   long-running daemons that resynthesize across many speakers. Wrappers
   landed across all 7 bindings (Python `Session.clear_phoneme_cache()`,
   Rust `Session::clear_phoneme_cache()`, Dart `clearPhonemeCache()`,
   Go `Session.ClearPhonemeCache()`, Java `clearPhonemeCache()`, JS
   `Module.ttsClearPhonemeCache()`, Ruby `Session.clear_phoneme_cache()`).
   No-model unit tests cover the symbol export + null-handle return path.

### Effort

Small individually. Open items 2 + 3 are each an afternoon if we
go the pre-processing route. Open item 1 is "policy" — a one-line
fallback in the backend or a docs change. Open item 4 is ~150 LOC.
Open item 5 is ~20 LOC if asked.

---

## 57. Commercial-friendly TTS backend expansion

May 2026 sweep through high-traffic HF TTS models. Filter is **permissive
license + reusable architecture + reasonable effort**. Sequenced so each
phase unlocks a family of finetunes — finishing Phase 3 (Chatterbox stack)
also unlocks Phase 5's CFM solver, etc.

License triage that drives the ordering:

| ✅ Permissive (commercial OK) | ⚠️ Llama-3.2 community (commercial OK with attribution) | ❌ Non-commercial — defer |
|---|---|---|
| Qwen3-TTS-{Base,CustomVoice} (Apache 2.0) | Orpheus-3B family + Kartoffel_Orpheus (llama3.2) | SebastianBodza/Kartoffelbox-v0.1 (CC-BY-NC-ND) |
| ResembleAI/chatterbox base (MIT) | HumeAI/tada-3b-ml (llama3.2) | marduk-ra/F5-TTS-German (CC-BY-NC) |
| SebastianBodza/Kartoffelbox_Turbo (CC-BY-4.0, gated) | | mlx-community/fish-audio-s2-pro (Fish-Audio Research) |
| oddadmix/lahgtna-chatterbox-v0/v1 (MIT) | | amphion/Vevo1.5 (CC-BY-NC-ND) |
| openbmb/VoxCPM2 (Apache 2.0) | | mlx-community/Voxtral-4B-TTS-2603 (CC-BY-NC; upstream Mistral Apache OK) |
| FINAL-Bench/Darwin-TTS-1.7B-Cross (Apache 2.0) | | |
| AMAImedia Qwen3-1.7B-TTS-Cross-Darwin AWQ (Apache 2.0) | | |
| g-group-ai-lab/gwen-tts-0.6B (MIT) | | |
| kugelaudio/kugelaudio-0-open (MIT) | | |

License gaps to resolve before depending on a model: CosyVoice 3
(`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` — model card silent;
v1/v2 were Apache 2.0 but v3 not yet confirmed).

### Phase 1 — DONE

All four Phase 1 variants shipped to HF and registered as backend
aliases:

| Variant | Backend alias | HF repo | HISTORY |
|---|---|---|---|
| Qwen3-TTS-CustomVoice 0.6B | `qwen3-tts-customvoice` | [`cstr/qwen3-tts-0.6b-customvoice-GGUF`](https://huggingface.co/cstr/qwen3-tts-0.6b-customvoice-GGUF) | per-model status table below |
| Qwen3-TTS-CustomVoice 1.7B | `qwen3-tts-1.7b-customvoice` | [`cstr/qwen3-tts-1.7b-customvoice-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-customvoice-GGUF) | — |
| Qwen3-TTS-Base 1.7B | `qwen3-tts-1.7b-base` | [`cstr/qwen3-tts-1.7b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF) | [§57](HISTORY.md) |
| Qwen3-TTS-VoiceDesign 1.7B | `qwen3-tts-1.7b-voicedesign` | [`cstr/qwen3-tts-1.7b-voicedesign-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF) | [§58](HISTORY.md) |

The CustomVoice contract surfaced from a config.json diff: a fixed
`spk_id` token (e.g. `vivian:3065`, `dylan:2878`) is prepended to the
talker prefill instead of an ECAPA forward; the speaker embedding is
just `talker.get_input_embeddings()(spk_id)`. Dialect override on the
`spk_is_dialect` table swaps `language_id` (e.g. dylan → beijing 2074).
Pending: extend `tools/reference_backends/qwen3_tts.py` so
`crispasr-diff qwen3-tts` covers the CustomVoice prefill path
(today's diff coverage is ICL/Base only).

Skipped: **havok2/Kartoffelbox-v0.1_0.65h2** (checkpoint variant of
CC-BY-NC-ND blocked Kartoffelbox-v0.1).

The Kartoffel_Orpheus DE + lex-au-orpheus-de checkpoints rolled into
Phase 2 (per-model status table).

### Phase 2 — talker pattern (qwen3_tts.cpp reuse)

Models with a Llama/Qwen-style AR talker + a small audio-token codec.
The talker forward fits directly into the `core_attn::kv_self_attn` +
`core_ffn::swiglu` pattern that #52 already uses.

- **Orpheus-3B backbone** (`canopylabs/orpheus-3b-0.1-ft` —
  use `unsloth/orpheus-3b-0.1-ft` non-gated mirror in practice;
  llama3.2 license) — Llama-3.2-3B + SNAC codec. New backend
  `orpheus`. **DONE (May 2026, commit `a0982d3`)** — talker AR
  forward + SNAC C++ decode shipped end-to-end; ASR-roundtrip on
  `"Hello, my name is Tara."` returns the input verbatim through
  parakeet-v3. With Orpheus base in, Kartoffel_Orpheus + lex-au +
  the various Orpheus finetunes are checkpoint swaps. Phase 3+
  follow-ups (out of scope for slice (c)): greedy decoding loops
  (ship-default must pass `--temperature 0.6`); Llama-3 RoPE
  freq scaling unimplemented; no `repetition_penalty`; Metal
  first-load is slow (~10-15 min for 6.6 GB f16 due to kernel
  compilation, fast thereafter); non-streaming AR (sliding-window
  protocol from `orpheus_snac.py` is a follow-up).
- **g-group-ai-lab/gwen-tts-0.6B** (MIT) — likely a Qwen3-TTS-style
  talker variant; need a weight inspection before sizing. If the
  shape matches, it's a #52 registry add.
- **HumeAI/tada-3b-ml** (llama3.2) — 3B Llama backbone + custom
  codec. Talker reuse high; codec is a new component. Defer until
  Orpheus lands so the SNAC vs Hume-codec contrast informs whether
  a `core_audio_codec` helper makes sense (overlaps with #53).

### Phase 3 — Chatterbox stack (CFM solver)

This is the family-unlock phase. Building a flow-matching (CFM) ODE
solver in ggml is the gating piece; once it's in, three commercial-OK
models become checkpoint-only adds.

- **ResembleAI/chatterbox** (MIT) — full pipeline: BPE tokenizer →
  T3 (0.5B Llama AR) → S3Gen (CosyVoice-style CFM, ~12 ODE steps)
  → HiFT-GAN-style vocoder → 24 kHz PCM. Plus voice encoder for
  cloning. New backend `chatterbox`.
- **SebastianBodza/Kartoffelbox_Turbo** (CC-BY-4.0, gated) — German
  TTS. **NOT a Chatterbox checkpoint swap** — inspection (2026-05-04)
  revealed GPT-2 architecture (`tfmr.h.N`, fused `c_attn` QKV,
  LayerNorm+bias, learned `wpe` positional embeddings, standard MLP
  `c_fc`/`c_proj`). This is a Tortoise-TTS variant, not Chatterbox
  Llama T3. Needs its own runtime or a GPT-2 adapter in chatterbox.cpp.
  Re-scoped from XS to M effort. **Caveat from model card: training
  loss diverged late; paralinguistic tags likely non-functional.**
- **oddadmix/lahgtna-chatterbox-v1** (MIT) — Arabic T3 variant.
  **DONE** — same Llama architecture as base, T3 converted to GGUF,
  shares S3Gen. Published [`cstr/lahgtna-chatterbox-v1-GGUF`](https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF).

#### Phase 3 implementation status (May 2026)

Full C++ pipeline running end-to-end with real weights:

| Component | Files | Tensors | Status |
|---|---|---|---|
| GGUF converter | `models/convert-chatterbox-to-gguf.py` | — | ✅ T3 1.1GB + S3Gen 574MB |
| T3 Llama AR (30L) | `src/chatterbox.{h,cpp}` | 292 | ✅ KV-cached, perceiver, character tokenizer |
| Perceiver resampler | (in chatterbox.cpp) | 12 | ✅ Cross+self attention, 32 conditioning tokens |
| Conformer encoder (6+4) | `src/chatterbox_s3gen.cpp` | ~200 | ✅ ggml graph, simplified attention (no rel-pos) |
| UNet1D denoiser (14 blocks) | (in chatterbox_s3gen.cpp) | 910 | ✅ Causal conv + BasicTransformer + CFG |
| HiFTGenerator vocoder | (in chatterbox_s3gen.cpp) | 328 | ✅ FIXED — all stages cos=1.0 vs Python; ASR "Hello world." |
| Reference backend | `tools/reference_backends/chatterbox.py` | — | ✅ Dumps 7 stages to GGUF |

ASR roundtrip validation:
- Python vocoder on Python mel → parakeet: **"Hello world."** ✅
- C++ vocoder on Python mel → parakeet: **"Hello world."** ✅ (fixed 2026-05-03)
- All ggml graph stages match Python to cos=1.000 (no source fusion)
- Deterministic waveform cosine similarity: 0.93 vs torch.istft
- GGUF weights verified matching Python to 5-6 significant figures

Bugs fixed (2026-05-03):
1. **iSTFT transposed data access** — was `data[frame*C+f]`, correct `data[f*T+frame]` (ggml ne[0]=T fast)
2. **Missing ReflectionPad1d((1,0))** at last upsample stage
3. **Ad-hoc source STFT** → proper SineGen + windowed DFT (Box-Muller + Hann + center)
4. **Nyquist term** in Hermitian iDFT missing imaginary component

GGUFs shipped (2026-05-04):
- [`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF) — T3 F16/Q8_0/Q4_K (1.1G/542M/287M) + S3Gen F16/Q8_0/Q4_K (548M/342M/237M). All quants ASR-verified "Hello world."
- [`cstr/lahgtna-chatterbox-v1-GGUF`](https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF) — Arabic T3 F16 (1.1 GB), shares S3Gen with base

Remaining for production quality:
1. **C API integration** — register in crispasr_c_api.cpp, CLI adapter (`--backend chatterbox`)
2. **F0 predictor** — currently source fusion assumes F0≈0 (unvoiced); voiced speech needs F0 net
3. **Conformer relative position attention** — pos_bias_u/v + linear_pos (encoder quality)
4. **Voice cloning** — VoiceEncoder LSTM + S3Tokenizer + CAMPPlus
5. **Kartoffelbox_Turbo** — needs GPT-2 T3 runtime (see Phase 3 prose)

The CFM solver landed here is **also** the gating piece for Phase 4
CosyVoice 3 (license permitting) and partially for Fish-Speech S2
(blocked on license anyway). Ship it once, three families light up.

### Phase 4 — codec-head additions to existing audio LMs

Already-supported encoder/decoders in the tree get a TTS direction by
adding a codec head + sampling path. Cheaper than a full new backend.

- ~~**Voxtral-TTS**~~ — **BLOCKED, May 2026 license re-survey.**
  Upstream `mistralai/Voxtral-4B-TTS-2603` is **CC-BY-NC 4.0**, not
  Apache 2.0 as previously assumed. The model card states the license
  is inherited from the voice-reference training datasets (EARS,
  CML-TTS, IndicVoices-R, Arabic Natural Audio) which are themselves
  NC, so the constraint is constitutional and can't be cleansed by
  re-quantization. `TrevorJS/voxtral-tts-q4-gguf` tags itself
  Apache-2.0 but that's incorrect. Same blocker class as F5-TTS-German
  / Vevo1.5 below. Moved to deferred.
- **FINAL-Bench/Darwin-TTS-1.7B-Cross** (Apache 2.0) + AWQ
  variant `AMAImedia/Qwen3-1.7B-TTS-Cross-Darwin-NOESIS-AWQ-INT4` —
  Qwen3-1.7B talker + "Darwin" codec. The 1.7B talker is a #52
  shape bump; the AWQ INT4 path is not currently supported and
  should not block (use bf16/fp16). Codec is new — assess after
  Orpheus's SNAC integration.

### Phase 5 — new architectures (medium-large, standalone value)

- **openbmb/VoxCPM2** (Apache 2.0, 1.26k likes) — CPM-backbone TTS
  with diffusion/flow head. Entirely new arch family in the tree.
  High user demand → worth the spend after Chatterbox lands so we
  can reuse whatever flow-matching utilities the CFM solver
  produces. Estimate: comparable to VibeVoice work (~1.5k LOC).
- **kugelaudio/kugelaudio-0-open** (MIT) — multi-component pipeline,
  needs deeper config read before sizing. Defer.

### Deferred / explicitly skipped

| Model | Reason |
|---|---|
| SebastianBodza/Kartoffelbox-v0.1 + havok2 derivative | CC-BY-NC-ND-4.0 — can't ship and can't even fine-tune. Recommend Kartoffelbox_Turbo (CC-BY-4.0) as the German Chatterbox path. |
| marduk-ra/F5-TTS-German | CC-BY-NC. F5-TTS arch is a DiT — would need new ggml ops, not worth the spend on an NC model. |
| mlx-community/fish-audio-s2-pro-* | Fish-Audio Research license — commercial requires separate Fish Audio license. |
| amphion/Vevo1.5 | CC-BY-NC-ND. Also voice conversion, different I/O contract. |
| mistralai/Voxtral-4B-TTS-2603 + all derivatives (mlx-community 4-bit, TrevorJS Apache-2.0-tagged GGUF) | Upstream weights are CC-BY-NC 4.0 inherited from voice-ref training data (EARS / CML-TTS / IndicVoices-R / Arabic Natural Audio). Constitutional, not cleanable. The "use upstream Apache 2.0 weights" plan turned out to be based on a wrong assumption (May 2026 re-survey). |
| KevinAHM/pocket-tts-onnx, Pendrokar/xvapitch_nvidia | ONNX-only, niche, no clear demand. |
| NeuralAudioAI/NA_base, tokenaii/horus | Insufficient public info — re-evaluate if asked. |
| FunAudioLLM/Fun-CosyVoice3-* + ayousanz/cosy-voice3-onnx | License unverified on v3. Earlier CosyVoice generations were Apache 2.0; needs confirmation before committing to CFM solver work for it. |

### Per-model status

| Phase | Model | License | Status | Effort |
|---|---|---|---|---|
| 1 | Qwen3-TTS-CustomVoice 0.6B | Apache 2.0 | **DONE + SHIPPED — runtime spk_id path; 4 ASR roundtrips passed (vivian / aiden / serena / dylan-dialect); registry alias `qwen3-tts-customvoice`; published as [`cstr/qwen3-tts-0.6b-customvoice-GGUF`](https://huggingface.co/cstr/qwen3-tts-0.6b-customvoice-GGUF) (Q8_0 968 MB).** | S |
| 1 | Qwen3-TTS-CustomVoice 1.7B | Apache 2.0 | **DONE + SHIPPED — `small_to_mtp_projection` applied per-step (steps 1..14), ASR roundtrips word-exact on Q8_0/ryan + F16/vivian. Registry alias `qwen3-tts-1.7b-customvoice`; factory dispatch wired. Published as [`cstr/qwen3-tts-1.7b-customvoice-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-customvoice-GGUF) (F16 3.84 GB + Q8_0 2.04 GB).** | S |
| 1 | Qwen3-TTS-Base 1.7B | Apache 2.0 | **DONE — runtime parameterised `spk_enc_dim` (was hardcoded 1024) so the 1.7B's 2048-d ECAPA output stops getting truncated; registry alias `qwen3-tts-1.7b-base` + HF model card landed. ASR-roundtrip word-exact on F16/Q8_0 (clone.wav English ICL). Published as [`cstr/qwen3-tts-1.7b-base-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-base-GGUF) (F16 3.86 GB + Q8_0 2.07 GB).** | S |
| 1 | Qwen3-TTS-VoiceDesign 1.7B | Apache 2.0 | **DONE (commit `bd3eb71`) — natural-language voice description via `--instruct`. New `build_voicedesign_prefill_embeds` mirrors CustomVoice but omits the speaker frame from the codec bridge and prepends an instruct block tokenised as `<\|im_start\|>user\n{instruct}<\|im_end\|>\n`. New C-ABI: `qwen3_tts_set_instruct` + `qwen3_tts_is_voice_design`. ASR-roundtrip word-exact on F16/Q8_0 (parakeet-v3 verbatim modulo terminal punctuation). Published as [`cstr/qwen3-tts-1.7b-voicedesign-GGUF`](https://huggingface.co/cstr/qwen3-tts-1.7b-voicedesign-GGUF) (F16 3.84 GB + Q8_0 2.04 GB). 1.7B-only — no 0.6B-VoiceDesign weight release upstream.** | S |
| 2 | Orpheus-3B base | llama3.2 | **DONE (commits `a0982d3` + `a4f7c49` + `1f62647` + `5025150`) — talker AR forward + SNAC C++ decoder shipped; ASR-roundtrip word-exact on `"Hello, my name is Tara."` (parakeet-v3 verbatim). Published as [`cstr/orpheus-3b-base-GGUF`](https://huggingface.co/cstr/orpheus-3b-base-GGUF) (F16 6.6 GB + Q8_0 3.5 GB) + [`cstr/snac-24khz-GGUF`](https://huggingface.co/cstr/snac-24khz-GGUF) (F32 26 MB). Unified Session API + all 6 wrappers wired (`crispasr_session_set_speaker_name`, `n_speakers`, `get_speaker_name`); orpheus default temperature now 0.6f (was 0.0f / greedy / loops). Phase 3+ gaps tracked in slice prose above.** | M |
| 2 | Kartoffel_Orpheus DE natural | llama3.2 | **DONE + SHIPPED — converted + quantized (F16 6.61 GB / Q8_0 3.5 GB / Q4_K 1.87 GB), ASR-roundtrip word-exact on Q8_0/Julian via parakeet-v3 -l de. Published as [`cstr/kartoffel-orpheus-3b-german-natural-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-natural-GGUF). Registry alias `kartoffel-orpheus-de-natural` + factory dispatch live (commit `d5b55a7`). 19 fixed German speakers (Jakob, Anton, Julian, Jan, Alexander, Emil, Ben, Elias, Felix, Jonas, Noah, Maximilian, Sophie, Marie, Mia, Maria, Sophia, Lina, Lea).** | XS |
| 2 | Kartoffel_Orpheus DE synthetic | llama3.2 | **DONE + SHIPPED — converted + quantized (F16 6.61 GB / Q8_0 3.5 GB / Q4_K 1.87 GB). Published as [`cstr/kartoffel-orpheus-3b-german-synthetic-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-synthetic-GGUF) (commit `927877e`). Registry alias `kartoffel-orpheus-de-synthetic` + factory dispatch live. 4 speakers (Martin / Luca / Anne / Emma) + 12 emotions (Neutral, Happy, Sad, Excited, Surprised, Humorous, Angry, Calm, Disgust, Fear, Proud, Romantic) + 5 outbursts (haha, ughh, wow, wuhuuu, ohhh) via `{Speaker} - {Emotion}: {text}` prompt syntax. End-to-end synth verification deferred (local 16 GB box memory-contested by parallel agent's converters; orpheus 3B AR loop hung in both Metal init and CPU mode); architecture + Kartoffel checkpoint-swap path validated via natural variant's word-exact roundtrip. Xet dedup made the synth upload only ~5.1 GB net new bytes despite 12 GB nominal size.** | XS |
| 2 | lex-au Orpheus-3B-DE-Q8 | llama3.2 (HF tags Apache-2.0; underlying Llama-3.2-FT) | **DONE — registry alias `lex-au-orpheus-de` added pointing at the existing `lex-au/Orpheus-3b-German-FT-Q8_0.gguf` (3.52 GB). Factory dispatch wired. SNAC companion shared with the base orpheus row.** | XS |
| 2 | gwen-tts-0.6B | MIT | queued — needs weight inspection first | S–M |
| 2 | tada-3b-ml | llama3.2 | queued | M |
| 3 | Chatterbox base | MIT | **DONE + SHIPPED** — vocoder fixed, F16/Q8_0/Q4_K quantized, ASR "Hello world." on all quants. Published as [`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF) (T3: 1.1G/542M/287M + S3Gen: 548M/342M/237M). Remaining: C API wiring, F0 predictor, voice cloning. | L |
| 3 | Kartoffelbox_Turbo DE | CC-BY-4.0 (gated) | **BLOCKED** — NOT a checkpoint swap. Uses GPT-2 architecture (fused QKV, LayerNorm+bias, learned pos embeddings), not Chatterbox Llama T3. Needs own runtime. Re-scoped to M effort. | M |
| 3 | lahgtna-chatterbox-v1 AR | MIT | **DONE + SHIPPED** — T3 converted to GGUF (shares S3Gen with base). Published as [`cstr/lahgtna-chatterbox-v1-GGUF`](https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF) (T3 F16 1.1 GB). | XS |
| 4 | Voxtral-TTS (Mistral upstream) | CC-BY-NC 4.0 | **BLOCKED — license inherits from voice-ref training data; moved to Deferred. See Phase 4 prose.** | — |
| 4 | Darwin-TTS-1.7B-Cross | Apache 2.0 | queued | M |
| 5 | VoxCPM2 | Apache 2.0 | queued — large new arch | L |
| 5 | kugelaudio-0-open | MIT | needs scoping | TBD |

### Effort

Phase 1 is hours. Phase 2 is one new backend (Orpheus) + N
checkpoint adds. Phase 3 is the CFM solver + Chatterbox runtime —
the largest single piece, but unlocks Phase 5's VoxCPM2 partially.
Phase 4 is bolt-ons. Phase 5 is standalone large.

Sequencing rationale: do Phase 1 immediately (free coverage), then
Phase 2 because Orpheus reuses #52's talker code most directly,
then Phase 3 because CFM is the biggest force-multiplier, then
Phase 4 (codec heads) as opportunistic, then Phase 5 (VoxCPM2) once
flow-matching utilities exist.

---

## 58. MOSS-Audio-4B-Instruct

[`OpenMOSS-Team/MOSS-Audio-4B-Instruct`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Instruct)
— Apache-2.0, ~4 B params, released 2026-04. First **audio-
understanding** model in the queue (not just ASR): speech, music,
environmental sounds, scene QA, time-aware ASR, multi-step
reasoning. Mandarin + English. The Instruct variant is the entry
point; the family also has 8B and Thinking (CoT) variants sharing
the same architecture.

### Architecture summary (from `config.json`)

- **Audio encoder** — 32-layer Whisper-style transformer trained
  from scratch (not a stock Whisper checkpoint). 1280 d / 20 heads,
  GELU FFN 5120 d, 128 mel bins, max 1500 source positions, sliding-
  window attention with window=100. Output rate 12.5 Hz after
  downsample (rate=8). The novel bit: **cross-layer feature taps**
  at layers 8, 16, 24 (in addition to the final 32) — these are
  carried through the adapter into the LM via DeepStack injection
  (see below).
- **DeepStack adapter** — adapter MLP (8192 d hidden) projects each
  of the 4 encoder taps into LM-embedding space (2560 d) with
  independent weights. The 4 projections are added as residuals
  into LM block inputs at indices 0, 1, 2, 3 (so the encoder's
  multi-resolution features inject continuously through the LM's
  early layers). This preserves low-level prosody / transients
  alongside high-level semantics in a way single-tap projectors
  (qwen3-asr / voxtral / granite-speech) can't.
- **Time-aware tokens** — explicit time-marker tokens are inserted
  between audio frame embeddings at fixed intervals. The LM learns
  "what happened when" natively; supports word-level + sentence-
  level timestamp ASR + time-based QA without a separate aligner.
- **LM** — 36-layer Qwen3 (hidden=2560, 32 Q / 8 KV head_dim=128,
  SwiGLU, RMSNorm, RoPE θ=1 M, max_pos=40 960, vocab=151 936,
  untied lm_head). No sliding window; full attention.

### Effort breakdown

| Component | LOC | Reuse |
|---|---:|---|
| Audio mel front-end (128-bin) | ~50 | `core_mel` |
| 32-layer Whisper-style encoder | ~150 | ~70 % from `qwen3_asr.cpp` encoder |
| Encoder sliding-window attention | ~50 | reuse pattern from `voxtral4b` |
| **DeepStack 4-tap output capture** | ~80 | **new** — needs encoder builder hooks at L8/16/24/32 |
| **DeepStack 4-projection adapter** | ~60 | **new** — 4× MLP, run once after encoder |
| **DeepStack injection into LM blocks 0–3** | ~120 | **new** — adds a fixed-shape residual at `cur` before block-N's first norm |
| Time-marker tokenization | ~100 | **new** — chat template builder + per-frame interval logic |
| Qwen3 LM body | ~50 | full reuse (`core_attn::kv_self_attn` + `core_ffn::swiglu`) |
| Greedy / sampler decode | ~80 | `core_bpe::tokenize_with_specials` + step builder pattern from `mimo_asr.cpp` |
| Converter (HF → GGUF) | ~250 | `models/convert-mimo-asr-to-gguf.py` template |
| Diff harness reference + 6 stages | ~200 | `tools/reference_backends/mimo_asr.py` template |
| Backend wrapper for main CLI | ~120 | `crispasr_backend_mimo_asr.cpp` template |
| **Total** | ~**1200–1500 LOC** | comparable to PLAN #51 |

Headline new helper: a **DeepStack injection block** (probably
`core_deepstack::inject(ctx, cur, projector_w, projector_b,
encoder_tap)`) that's reusable for any future model adopting this
pattern. The 4 projection heads are independent matmul + bias adds
applied to the captured encoder taps; injection is a residual add
at the input of LM blocks 0..3.

### What we'd need to dump from the Python ref

Stage taps for the diff harness:
- `mel_in` `[T_mel, 128]`
- `enc_l8` / `enc_l16` / `enc_l24` / `enc_l32` `[T_enc, 1280]`
  (the four DeepStack taps)
- `adapter_proj_{0,1,2,3}` `[T_enc, 2560]` (post-projection)
- `lm_inputs_embeds` `[T_total, 2560]` (pre-block-0)
- `lm_block_3_in` `[T_total, 2560]` (after the last DeepStack
  injection — this is where a multi-tap bug would show up)
- `lm_last_hidden` + `lm_logits_step0` (standard tail)

Six-to-eight stages, similar to mimo-asr's prefill captures.

### Risks / open questions

1. **DeepStack injection point semantics** — does the projection
   replace the LM block's input or get added as a residual? Need
   to read `processing_moss_audio.py` + the model's `forward()` to
   confirm. If it's a *replace* (not residual), the injection
   builder is simpler but the math is more sensitive.
2. **Time-marker token vocab** — are these dedicated special tokens
   in the Qwen3 BPE, or are they synthesized in the embedding
   space? The vocab=151 936 has slots beyond Qwen3's 151 643 BPE +
   30 special — likely the extra ~263 are time markers.
3. **Sliding-window encoder attention with mask=100** — already a
   pattern (`voxtral4b`), but interacts non-trivially with the
   12.5 Hz downsample. Confirm causal vs bidirectional via Python
   ref hook.
4. **Family extensibility** — 8B variant has the same architecture
   per the README, just bigger LM hidden + layer count. If we
   parameterize by config, all four (4B/8B × Instruct/Thinking)
   share one runtime. Worth doing up front.

### Why "audio understanding, not just ASR" matters here

The 24 ASR-style backends in CrispASR all map audio → text
transcription. None handle "describe the music in this clip", "is
the speaker happy", "summarise this 10-minute meeting", or
"transcribe with word-level timestamps". MOSS-Audio is the first
candidate that covers that ground with an open license (Apache-2.0)
and a reasonable size (4 B → ~2.5 GB Q4_K). Adding it expands
CrispASR's surface meaningfully — analogous to how qwen3-tts
expanded scope to TTS.

### Sequencing

Don't start until:
- mimo-asr perf follow-ups (51a/b/c) are at least scoped — they'll
  inform DeepStack's KV-reuse strategy.
- Orpheus / Qwen3-TTS-1.7B (PLAN #57 phases 1–2) finish — those are
  active sessions and the parallel-worker contention is high.

Probable kickoff: mid-to-late May 2026 if the queue clears.

---

## Ecosystem expansion (lower priority)

### New backends from PazaBench assessment (see HISTORY.md #30)

| Model | License | Approach | Priority |
|---|---|---|---|
| Wav2Vec2 Conformer | Apache-2.0 | Conformer attention variant | Medium |
| Qwen2-Audio 7B | Apache-2.0 | Whisper encoder + Qwen2 LLM | Medium |
| OmniASR larger (1B/3B/7B) | Apache-2.0 | Same converter, bigger models | Medium |
| NeMo Canary-Qwen-2.5b | Apache-2.0 | FastConformer + Qwen2.5 decoder | Medium |
| Paza / Phi-4 | MIT | 14B multimodal, defer to llama.cpp | Low |
| **XiaomiMiMo/MiMo-V2.5-ASR** | TBD (check) | LLM-style multimodal speech (similar to Qwen3-ASR pattern) | Medium — user-requested in #35 |
| **google/gemma-4-E2B** | Gemma terms | Conformer + Gemma 4 decoder; matches "Gemma 4 Audio" entry below | Medium — user-requested in #35 |

### From llama.cpp (MIT)

| Model | Architecture | Notes |
|---|---|---|
| Ultravox | Whisper encoder + Llama 3.2 1B/8B | Speech understanding |
| Gemma 4 Audio | Conformer, chunked attention | Streaming, multimodal |
| LFM2-Audio | Conformer variant | Position embeddings |

### Post-processing

| Model | License | Type | Priority |
|---|---|---|---|
| FireRedPunc | Apache-2.0 | BERT punct (zh+en) | **DONE** |
| fullstop-multilingual | MIT | XLM-R punct (en/de/fr/it) | Medium |
| bert-restore-punctuation | MIT | BERT punct+truecase (en) | Medium |
| xashru/punctuation | Apache-2.0 | XLM-R+BiLSTM-CRF (40+ langs) | Low |

### Optimizations (cross-cutting, from survey + CrispEmbed comparison)

| # | Optimization | Applies to | Expected gain | Status |
|---|---|---|---|---|
| O1 | `ggml_soft_max_ext` fusion | wav2vec2, canary, fastconformer | -10% wav2vec2 | **DONE** |
| O11 | wav2vec2 CNN → ggml | wav2vec2 family | **10.8x** | **DONE** |
| O9/#44 | FireRed ggml Q4_K decoder | firered-asr | **6.3x** | **DONE** |
| O10 | Sliding window attention | voxtral4b | Already implemented | **DONE** |
| O2 | Fused QKV pre-merge | LLM decoders | ~10-15% attn (GPU) | API ready in core/attention.h; CPU gain <1%, defer to GPU |
| O3 | Temperature sampling | glm-asr, kyutai-stt | Feature parity | **DONE** |
| O5 | Pipelined mel+encode | LLM backends, CPU | ~15-20% | TODO |
| O4 | Beam search for LLMs | Audio-LLM backends | Quality | TODO |
| O6 | Batched encoder (GPU) | All + GPU | 3-5x | TODO |
| O7 | Speculative decoding | LLM backends | 2-4x decode | TODO |
| O12 | `ggml_conv_1d_cf` channels-first conv | vibevoice VAE | **-29% VAE, -15% total** | **DONE** |
| O13 | `ggml_conv_1d_group` + CNN cleanup | wav2vec2 family | **-12% total** (pos -12%, CNN -22%) | **DONE** |
| O14 | `--tts-steps` configurable DPM steps | vibevoice TTS | **-31% diffusion** | **DONE** |
| O15 | Remove redundant neg base LM | vibevoice TTS | Eliminated 60 LOC of wasted compute | **DONE** |

**From COMPARISON.md (llama.cpp patterns):**
- `ggml_soft_max_ext` with baked scale (O1) — already in llama.cpp, saves one `ggml_scale` op per attention layer
- Chunked window attention (O10) — llama.cpp uses for Gemma4A Conformer
- Conv2d subsampling via ggml ops — llama.cpp does this for Qwen3-ASR encoder

**From CrispEmbed (shared core patterns):**
- Fused QKV (O2) — CrispEmbed pre-merges Q/K/V weights at init, one matmul instead of 3
- SentencePiece Viterbi DP tokenizer — CrispEmbed has proper optimal tokenization
- Lazy graph allocation (`no_alloc=true` + scheduler) — reduces memory churn

**From LEARNINGS.md (FireRed decoder triage):**
- Small per-step ggml graphs are SLOWER than CPU loops (scheduling overhead)
- BUT: native Q4_K matmuls via ggml are 9.3x faster than F32 OpenMP (lesson: never dequant)

### Audio format support

- `.m4a`, `.mp4`, `.webm` crash with upstream ffmpeg integration — needs fix or robust fallback
- `.aiff`, `.wma`, raw PCM not supported without pre-conversion
- Consider bundling a lightweight M4A/AAC decoder or improving the ffmpeg path
- Only move LARGE, REUSED matmuls onto ggml/GPU
- Persistent subgraphs per decode step > one-off graphs

### Other

- **OmniASR-LLM beam search** — beam=2+ with N hypothesis KV caches
- ~~**TTS module** — VibeVoice-Realtime-0.5B text-to-speech~~ **DONE** — perfect ASR round-trip on all test cases. 17 bugs found via stage-by-stage diff. Uses DPM-Solver++, dual KV CFG, voice prompts, EOS classifier, text/speech interleaving.
- ~~**ggml_conv_1d_dw F16 im2col fix**~~ **DONE** — solved via `ggml_conv_1d_dw_cf` (direct F32, no im2col)

---

## Publish language wrappers to package registries

Today the Rust, Dart, and Python wrappers all live in this repo and (for
Python) require a `pip install -e .` from a clone. Move all three onto
their language-native registries so users can install with one command.

**Status (2026-04-25):** All three wrappers now have publishable
metadata + dry-runs pass. The CI workflow `release-wrappers.yml` is
wired up but cannot run until the **one-time registry setup** below
is complete.

| Wrapper | Pre-flight | Blocker |
|---|---|---|
| Python `crispasr` 0.5.4 | sdist + wheel build clean | PyPI trusted-publisher must be configured |
| Dart `crispasr` 0.5.4 | `dart pub publish --dry-run` passes (warnings only) | pub.dev automated publishing must be configured |
| Rust `crispasr-sys` 0.5.4 | `cargo publish --dry-run` clean (5.9 KiB) | needs `CARGO_REGISTRY_TOKEN` repo secret |
| Rust `crispasr` 0.5.4 | publish-order dependent on `crispasr-sys` | same |

### One-time registry setup (must happen before first tag)

1. **PyPI** — go to https://pypi.org/manage/account/publishing/ and add
   a "pending publisher": owner `CrispStrobe`, repo `CrispASR`,
   workflow `release-wrappers.yml`, environment `pypi`. Then push any
   `v*` tag.
2. **crates.io** — generate a token at https://crates.io/me, add it
   as the `CARGO_REGISTRY_TOKEN` secret on the GitHub repo.
3. **pub.dev** — go to https://pub.dev/packages/crispasr/admin (after
   first manual publish or claim) → enable automated publishing → set
   tag pattern `v{{version}}`. Alternatively for the first publish,
   run `dart pub publish` locally with the package owner's credentials.

### Pattern (matches crispasr approach)

All three wrappers are thin FFI/ctypes shims over the C ABI in
`src/crispasr_c_api.cpp`. They do **not** bundle the native library — the
user must have `libcrispasr.{so,dylib,dll}` installed (Homebrew, apt, or
built from source). This keeps the wheels/crates/pub packages tiny and
avoids a per-platform build matrix on every release.

| Wrapper | Registry | Effort | Notes |
|---|---|---|---|
| Python | PyPI | Low | Add `python/pyproject.toml`; pure-Python wheel; `_helpers.c` builds at install if a C toolchain is present, else falls back to ctypes-only path |
| Rust   | crates.io | Low | `crispasr-sys` then `crispasr` (two `cargo publish` calls); already has `Cargo.toml` |
| Dart   | pub.dev | Low | `flutter pub publish --dry-run` then `flutter pub publish`; already has `pubspec.yaml` |

### Library discovery (Python)

Update `_find_lib()` in `python/crispasr/_binding.py` to probe, in order:
1. `$CRISPASR_LIB_PATH` env var (explicit override)
2. `sys.prefix/lib/` (system or virtualenv install)
3. Standard Homebrew/Linux paths (`/opt/homebrew/lib`, `/usr/local/lib`, `/usr/lib`)
4. Existing repo-relative fallbacks (for `pip install -e .` from a clone)

If none found, raise `RuntimeError` with a helpful message linking to
install docs (the same pattern Tesseract / faster-whisper use).

### Release automation

Add a tag-triggered workflow `.github/workflows/release-wrappers.yml`
that, on `v*` tags, runs in parallel:
- `python -m build && twine upload` (PyPI, OIDC trusted-publishing — no API token)
- `cargo publish -p crispasr-sys && cargo publish -p crispasr` (crates.io, `CARGO_REGISTRY_TOKEN` secret)
- `dart pub publish --force` (pub.dev, OIDC publishing)

Trigger only on tag push, not on every commit. Version bumps stay
manual — bump `pyproject.toml` / `Cargo.toml` / `pubspec.yaml` together
in the same commit that creates the tag.

### Future: bundled wheels for Python

After the pure-Python release is out, add a follow-up release pipeline
using `cibuildwheel` to produce manylinux2014 + macOS arm64/x64 +
Windows wheels with `libcrispasr.*` bundled inside via `auditwheel` /
`delocate` / `delvewheel`. Same for Rust if we ever want
`crispasr-sys` to vendor the native build like `tch-rs` /
`onnxruntime-sys` do. Defer until pure-Python wheel is out and stable.


---

## 59. Cross-binding C-ABI parity

The Session API surface for TTS (incl. qwen3-tts Base / CustomVoice /
VoiceDesign variant routing) is fully wrapped across all 7 bindings as
of commit `65e0a61` + the Dart follow-up. **The non-Session ABI (~80
exports) is still C-ABI-only or partially-wrapped on most bindings.**
This entry tracks closing those gaps.

### Coverage matrix (May 2026)

C-ABI exposes 127+ unique `crispasr_*` exports in
`src/crispasr_c_api.cpp`. Coverage by binding:

| Binding | Symbols wrapped | Approx % | ASR Transcribe | TTS Session | Variant detect | Align | Diarize | LID | VAD | Streaming | Punc | Registry | Cache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rust (`crispasr-sys`) | 56 | ~44% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Python (`_binding.py`) | 53 | ~42% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dart (`flutter/crispasr`) | ~30 | ~24% | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Go (`bindings/go`) | ~45 | ~35% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Java (JNA) | ~38 | ~30% | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅¹ | ✅¹ | ✅¹ |
| Ruby (C ext) | ~30 | ~24% | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅¹ | ❌ | ❌ |
| JS (emscripten) | 18 | ~14% | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

¹ JNA declarations added, idiomatic Java wrapper methods pending.

Rust + Python are the canonical / "full-coverage" wrappers. The other
five track the high-traffic surface (transcribe + TTS) and were swept
together in `4f476c3` (set_speaker_name) and `65e0a61` (set_instruct +
variant detect).

### Capabilities reachable only from C-ABI / Rust / Python

For each, ~3-12 exports + an idiomatic result type per binding:

- **Forced alignment** — `crispasr_align_words`, `align_words_abi`,
  `align_result_*`. Word-level timestamps from a transcript + audio.
- **Diarization** — `crispasr_diarize_segments[_abi]`. Speaker segment
  spans.
- **Language ID** — `crispasr_detect_language[_pcm]`,
  `crispasr_lid_free_cache`. Pre-transcribe LID for routing.
- **VAD** — `crispasr_vad_segments`, `crispasr_compute_vad_slices`,
  `crispasr_stitch_vad_slices`, `crispasr_vad_remap_timestamp`,
  `crispasr_vad_free`. Standalone VAD + slice stitching.
- **Streaming** — `crispasr_stream_open/feed/get_text/flush/close`,
  `crispasr_stream_run_decode`. Online ASR with a step buffer.
- **Punctuation** — `crispasr_punc_init/process/free/free_text`.
  FireRedPunc post-processor.
- **Model registry** — `crispasr_registry_lookup[_abi]`,
  `registry_lookup_by_filename[_abi]`,
  `crispasr_detect_backend_from_gguf`. Backend / file resolution.
- **Cache** — `crispasr_cache_dir_abi`,
  `crispasr_cache_ensure_file_abi`. Auto-download dir + lookup.

### Effort

Per binding ~150-300 LOC (extern decls + idiomatic methods + result
types + smoke test). Five trailing bindings × 9 capability surfaces ×
~30 LOC each ≈ 1.5 kLOC total. Each capability is independent — can
be staged.

Suggested ordering once a consumer asks:
1. Streaming (Go/Java first — common deployment shapes for ASR servers).
2. VAD + alignment (mobile use cases via Dart).
3. Diarization + LID + punctuation (transcription pipelines).
4. Registry + cache (CLI-style consumers).

### When to do this

Not now. The qwen3-tts sweep was justified because PLAN #57 Phase 2
unblocks needed it. Open this section when a concrete consumer shows
up asking for, say, "Java VAD" or "Go streaming". Reference commits
for the pattern: `4f476c3` (TTS surface sweep) and `65e0a61`
(variant detection sweep). Same shape applies to every other capability.

---

## 60. Cross-backend perf tricks (llama.cpp / llamafile ports)

Catalogue of optimizations worth porting from upstream llama.cpp +
Justine Tunney's llamafile, broken into discrete actionable items
(60a, 60b, …). Prioritized for our specific shape: Apple Silicon M1,
16 GB RAM, 7B-class speech-LLMs (MiMo-ASR, qwen3-asr, voxtral4b),
often-contested external disk.

| Item | Status | Tier | Effort | Notes |
|---|---|---|---|---|
| [60a — madvise WILLNEED](#60a-posix_madvisewillneed-on-mmapd-weights) | **DONE** | T1 | done | Async kernel readahead, both mmap branches |
| [60b — wrap_iface forward-compat](#60b-wrap_iface-forward-compat-set_tensor_2d--get_tensor_2d--reset) | **DONE** | T1 | done | 3 delegations added to mmap_wrap_iface |
| [60c — pre-touch / `--preload` flag](#60c-pre-touch--preload-flag) | **DONE** | T1 | done | `CRISPASR_GGUF_PRELOAD=1` page-walks before return |
| [60d — Fused QKV per LM layer](#60d-fused-qkv-per-lm-layer) | **DONE** (mimo-asr) | T2 | done | runtime + converter + GGUF patch script; Q4_K re-uploaded; F16 re-upload deferred |
| [60e — KV cache quantization](#60e-kv-cache-quantization-f16--q8_0--q4_0) | **OPEN** (env-flag landed) | T2 | M | `CRISPASR_KV_QUANT=q8_0/q4_0` plumbing done for mimo-asr; per-backend rollout pending |
| [60f — `--mlock` flag](#60f---mlock-flag) | **DONE** | T3 | done | `CRISPASR_MLOCK=1` pins pages after WILLNEED |
| [60g — `MADV_RANDOM` post-prefill](#60g-madv_random-post-prefill) | **DONE** | T3 | done | `core_gguf::mmap_advise_random()` exposed |
| [60h — Linux huge pages](#60h-linux-huge-pages-map_hugetlb) | PARKED | T3 | S | Linux-only; we don't have Linux production targets |
| [60i — Read-only mmap mode](#60i-read-only-mmap-mode) | PARKED | T3 | XS | Per-backend; safety net not yet needed |
| [60j — Speculative decoding](#60j-speculative-decoding) | PARKED | T3 | L | No obvious draft model in our family |
| [60k — GBNF grammar-constrained decode](#60k-gbnf-grammar-constrained-decoding) | PARKED | T3 | M | No structured-output consumer yet |
| [60l — tinyBLAS x86 kernels](#60l-tinyblas-llamafile-specific) | SKIP | T3 | — | Not relevant on Apple Silicon |
| [60m — APE multi-arch binary](#60m-ape-multi-arch-binary-llamafile-specific) | SKIP | T3 | — | Distribution, not perf |
| [60n — CUDA graphs](#60n-cuda-graphs--cuda-specific) | SKIP | T3 | — | We don't ship a CUDA backend |
| [60o — MTLBinaryArchive pipeline cache](#60o-mtlbinaryarchive-metal-pipeline-cache) | **OPEN** | T1 | M | Save 30–60 s per cold start on every Apple Silicon consumer |

**Tiers:** T1 = directly attacks cold-start / loader. T2 = decode-time
speedup, work-order calls out. T3 = situational / parked / skip.

**Suggested order for remaining OPEN items:**
60d (Fused QKV, mimo-asr LM) → 60e (KV cache quantization,
shared `core_attn` surgery). Both want a fresh session — see the
hand-off prompt at the end of the May 2026 perf-wave session.

---

### 60a. `posix_madvise(WILLNEED)` on mmap'd weights — **DONE → [HISTORY §63](HISTORY.md)**

`CRISPASR_GGUF_MMAP=1` triggers `POSIX_MADV_WILLNEED` on both CPU + Metal mmap branches. Open follow-up: Windows `PrefetchVirtualMemory` (Win8+).

---

### 60b. wrap_iface forward-compat: `set_tensor_2d` / `get_tensor_2d` / `reset` — **DONE → [HISTORY §63](HISTORY.md)**

---

### 60c. Pre-touch / `--preload` flag — **DONE → [HISTORY §63](HISTORY.md)**

`CRISPASR_GGUF_PRELOAD=1` page-walks all mmap'd weights synchronously. Open follow-up: `--preload` CLI flag in `whisper_params.h` if a user asks; Linux `MADV_POPULATE_READ` for a single-syscall kernel-walk.

---

### 60d. Fused QKV per LM layer — **DONE (mimo-asr Q4_K, voxtral4b Q4_K, qwen3-asr Q4_K, voxtral 3B Q4_K opt-in) → [HISTORY §64, §71](HISTORY.md)**

mimo-asr Q4_K runtime fuse (May 2026, HISTORY §64). voxtral4b Q4_K
runtime fuse (May 2026, HISTORY §71, ~7-8 % decode speedup on M1).

**May 2026 portability pass.** The Q4_K / row-wise-quantized type-gate
extension (drop the F16/F32 check, switch CPU→default-backend buffer
allocation) ported to:

- `qwen3_asr.cpp:1433` — gate dropped, default-on. Q4_K users pick up
  the fuse path. Transcript bit-correct on JFK; perf delta on
  short-decode tasks like JFK is sub-noise (decode loop is ~30 tokens),
  but consistent with voxtral4b's pattern: longer decodes amortise
  more.
- `qwen3_tts.cpp:4986` — gate dropped. Stays opt-in via
  `QWEN3_TTS_FUSED_QKV=1` (matching prior F16/F32 default-off behaviour
  pending a clean quiet-machine bench).
- `voxtral.cpp` (3B) — full runtime fuse added (~80 LOC, mirror of
  voxtral4b). **Opt-in via `CRISPASR_VOXTRAL_FUSED_QKV=1`** because
  A/B on JFK Q4_K showed no measurable speedup (16.37 s vs 16.43 s,
  within run-to-run noise). The voxtral4b 7-8 % win came from its
  long decode loop (141 tokens incl. streaming-pad warmup); voxtral
  3B's ~30-token normal-prompt decode is too short for the saved
  kernel-launch overhead to surface above noise. ~500 MB extra
  memory when enabled — opt-in keeps memory-tight deployments unpenalised.

**Lesson learned:** the fuse value scales with the per-matmul
kernel-launch overhead × decode-loop length. For Metal Q4_K on M1
with ~30 µs/launch, voxtral4b's 30-layer × 141-step decode amortises
the fuse meaningfully; voxtral 3B's 30-layer × 30-step decode does
not. Captured in LEARNINGS § "not every matmul fusion is a win on
Metal Q4_K — measure ROI in saved kernel launches, not saved input
reads."

---

### 60e. KV cache quantization (F16 → Q8_0 / Q4_0) — **OPEN (env-flag landed, May 2026)**

**Status:** OPEN — env-flag plumbing landed for mimo-asr; per-backend
rollout pending validation on long-form inputs.

llama.cpp ships `--kv-quant-type k Q8_0` etc. Halves (Q8_0) or
quarters (Q4_0) KV memory with near-zero quality loss for ASR.
Currently our `kv_max_ctx` for mimo-asr is capped by the
`mimo_asr_kv_init(prompt_groups + max_new + 16)` budget — with F16
KV at 36 layers × 8 KV heads × 128 head_dim × 369 ctx ≈ **~57 MB**.
Q8_0 would drop that to ~28 MB.

Not load-bearing for JFK (11 s). **Essential for hour-long podcast
ASR** where `max_ctx` balloons past 10k groups (~1.5 GB F16 KV).

**What landed:**
- `src/core/attention.h` — `core_attn::kv_self_attn` now reads
  `kv_k->type` directly: F16 (and F32) takes the existing
  `ggml_cont` path, while a quantized cache (Q8_0, Q4_0, …) uses
  `ggml_cast(...,F16)` to dequantize-on-read. Metal supports
  `Q*→F16` CPY for all standard quant types per
  `ggml-metal-device.m:1198–1250`. The cache *storage* keeps the
  ~half-bytes (Q8_0) / quarter-bytes (Q4_0) saving; reads pay one
  dequant pass per layer per step.
- `src/mimo_asr.cpp` — `mimo_asr_kv_init` reads `CRISPASR_KV_QUANT`
  (`f16` default, `q8_0`, `q4_0`) and allocates `ctx->kv_k` /
  `ctx->kv_v` with the requested dtype. The `ggml_cpy(F32→Q*)`
  write path (and `ggml_set_rows` scatter for the cached step
  graph) is supported on Metal for Q8_0 / Q4_0.

**Per-backend env wiring (DONE):** `core_attn::kv_dtype_from_env()` is the
shared lookup. The 9 backends that route their KV cache through
`core_attn::kv_self_attn` all call it from their `*_kv_init` and
allocate `kv_k` / `kv_v` with the chosen dtype:

- `mimo_asr_kv_init` (validated — see HISTORY §64)
- `qwen3_asr_kv_init`
- `voxtral_kv_init`
- `voxtral4b_kv_init`
- `granite_speech_kv_init`
- `gemma4_e2b` (`g4e_kv_init`, both sliding-window and full-attention caches)
- `glm_asr_kv_init`
- `omniasr_init_kv_cache`
- `orpheus` (`kv_alloc`)
- `qwen3_tts` talker (`kv_alloc`) — `cp_kv` (code-predictor cache)
  intentionally stays F16 since its decode path doesn't go through
  `core_attn::kv_self_attn`

Default stays F16 across all of them. `CRISPASR_KV_QUANT=q8_0`
or `=q4_0` opts in.

**Per-backend transcript validation (May 2026):**

| Backend | F16 vs Q8_0 KV | Result |
|---|---|---|
| mimo-asr | bit-exact (HISTORY §64, cosine ≥0.98) | ✅ |
| granite | bit-exact | ✅ |
| granite-4.1 | bit-exact | ✅ |
| glm-asr | bit-exact | ✅ |
| omniasr-llm | bit-exact | ✅ |
| voxtral | bit-exact | ✅ |
| qwen3 | minor punct diff (`;`→`,`), WER=0% | ✅ acceptable |

All 7 tested backends produce correct transcripts with Q8_0 KV.
Remaining untested: voxtral4b, gemma4-e2b, orpheus, qwen3-tts
(these use KV but haven't been validated yet — low priority since
the wiring is shared).

**Backends with custom KV paths (skipped — would need separate
quant-write fixes):** canary (Conformer encoder + RNN-T), cohere
(encoder-decoder), kyutai_stt (depthwise/streaming decoder),
vibevoice. These don't route through `core_attn::kv_self_attn`,
so they can't piggy-back on the shared write/read fixes; they'd
each need backend-specific work to support quant KV.

---

### 60f. `--mlock` flag — **DONE → [HISTORY §63](HISTORY.md)**

`CRISPASR_MLOCK=1` runs `mlock()` after WILLNEED + preload in both mmap branches. Open follow-up: `--mlock` CLI flag in `whisper_params.h` if a user asks.

---

### 60g. `MADV_RANDOM` post-prefill — **DONE → [HISTORY §63](HISTORY.md)**

`core_gguf::mmap_advise_random(ggml_backend_buffer_t)` exposed. Open follow-up: per-backend wiring (1-line call between prefill and decode loop) — defer until a 32+ GB-box benchmark shows measurable benefit; on Q4_K the perf delta is marginal.

---

### 60h. Linux huge pages (`MAP_HUGETLB`)

**Status:** PARKED. **Tier 3.**

mmap with `MAP_HUGETLB` reduces TLB misses ~10% on big models.
Doesn't exist on macOS. Not currently a Linux production target —
revisit if/when we ship a Linux-first packaging target.

---

### 60i. Read-only mmap mode

**Status:** PARKED. **Tier 3.**

Currently `MappedFile` opens with `MAP_PRIVATE + PROT_READ|PROT_WRITE`
(copy-on-write writable). Some backends (parakeet's BN-into-conv
fold) need the writable path. For backends that don't mutate
weights, `PROT_READ` would catch accidental mutation as a SIGSEGV
instead of silent CoW. Could gate per-backend in
`core_gguf::load_weights` later. Safety net, not perf.

---

### 60j. Speculative decoding

**Status:** PARKED. **Tier 3.** **Reason:** no obvious draft model.

llama.cpp's main throughput trick for autoregressive generation:
small "draft" model proposes 4-8 tokens, large "target" model
verifies in one prefill. Typical 2-3× decode speedup with no
quality loss when draft acceptance rate is high.

For our family, no obvious draft model exists for mimo-asr (7.5B,
36L Qwen2). Could investigate using qwen3-asr-0.6B as a drafter,
but it's a different tokenizer / vocab / audio preprocessing
stack. Not worth the complexity unless someone specifically asks.

---

### 60k. GBNF grammar-constrained decoding

**Status:** PARKED. **Tier 3.** **Reason:** no consumer.

Useful for structured ASR outputs (JSON, fixed-format timestamps,
PII redaction templates). Not for raw transcription. Park until a
consumer asks.

---

### 60l. tinyBLAS (llamafile-specific)

**Status:** SKIP. **Tier 3.** **Reason:** wrong target architecture.

Justine Tunney's bespoke x86 quantized matmul kernels. Faster
than llama.cpp's reference quants on certain CPUs. Apple Silicon
Metal kernels are already fast, and our x86 paths are CI-only
(no production users on x86 yet). Skip.

---

### 60m. APE multi-arch binary (llamafile-specific)

**Status:** SKIP. **Tier 3.** **Reason:** distribution trick, not perf.

Cosmopolitan libc — one binary runs on Linux/macOS/Windows. Skip
until we ship a packaged binary to end users (currently we ship
source + per-platform builds in CI).

---

### 60n. CUDA graphs / CUDA-specific

**Status:** SKIP. **Tier 3.** **Reason:** wrong backend.

Doesn't apply to Metal. If we ever ship a CUDA backend (not
currently a target — we use ggml-cuda for build-time only),
revisit.

### 60o. MTLBinaryArchive Metal pipeline cache

**Status:** OPEN. **Tier 1.** **Effort:** M (~half day source patch
in upstream `ggml/src/ggml-metal/`). **Source:** raised by CrisperWeaver
PLAN §5.18 — the highest-leverage perf item the Flutter app's CI
sweep is currently waiting on.

**Problem.** ggml-metal compiles MSL pipelines lazily for each unique
tensor shape on first use, then caches them in-memory only. Every
fresh process pays 30–60 s of MTLLibrary + MTLComputePipelineState
JIT before the first `ggml_metal_encode` lands. Affects:

* Every `flutter test` / `crispasr` CLI invocation on the dev box
  (~30–60 s startup tax per run).
* Every CI sweep — measured at ~25 min for the single-process
  multi-backend pass; projected ~5 min if pipelines were warm.
* Every end-user app launch on macOS / iOS / iPadOS where pipelines
  are recompiled across the whole loaded model on first transcribe.

**Fix.** Use Apple's first-party `MTLBinaryArchive` API to write
freshly-compiled pipeline state objects to a per-device disk cache
on shutdown and reload them on startup. Same pattern Apple's own
MPS / MLX use. Sketch:

* Patch `ggml/src/ggml-metal/ggml-metal-device.m`:
  - On `ggml_metal_device_init`, attempt
    `[device newBinaryArchiveWithDescriptor:]` from
    `${GGML_METAL_PIPELINE_CACHE}` (default
    `~/Library/Caches/ggml-metal/<device-name>.archive`).
  - When `ggml_metal_compile_pipeline` produces a new
    `id<MTLComputePipelineState>`, also call
    `[archive addComputePipelineFunctionsWithDescriptor:]` so the
    next process can rehydrate it.
  - On exit (or via an explicit `ggml_metal_pipeline_cache_save`),
    `[archive serializeToURL:]` flushes to disk.
* Joins the existing `// CrispASR patch` set in ggml-metal — same
  rebase discipline as the conv_transpose_1d perf patch.
* Cache invalidation: include device name + ggml-metal source
  hash in the archive path so a kernel change auto-busts the cache.

**Risk.** Low — Apple's API is stable since iOS 14 / macOS 11. Worst
case the archive fails to load and we fall back to the existing JIT
path, exactly today's behaviour.

**Why Tier 1.** The 30–60 s saved compounds across every consumer
of CrispASR (CLI, CrisperWeaver, the wrapper bindings, CI). Same
order-of-magnitude impact as 60a (madvise WILLNEED) had on cold-mmap
weight loads.

---

## ~~65. Session-API word-confidence parity~~ — **DONE → [HISTORY §65](HISTORY.md)**

Main batch + 65a (vibevoice / moonshine-streaming) + 65a-residue
(gemma4-e2b token-prob API) all landed. Go/Java/Ruby brought to
parity in `5534588` + `d963e3a`. **Only residual:** JS / emscripten
word-accessor surface — leaving until a JS consumer asks (the
current JS binding is TTS-focused).

---

## 61. Feature matrix uplift

The README "Feature matrix" was missing checkmarks for many cells
where the underlying model already supported the feature. Tracker
for closing the remaining gaps.

### 61a-f — **DONE → [HISTORY §65](HISTORY.md)**

| Sub-item | Outcome |
|---|---|
| 61a Auto-download for fc-ctc + wav2vec2 | 2 ✔ |
| 61b Per-token confidence × 7 backends | 7 ✔ (full row, 15/15) |
| 61c Kyutai native + word timestamps | 2 ✔ |
| 61d Best-of-N × 4 LLM-style decoders | 4 ✔ |
| 61e Temperature for omniasr-llm | 1 ✔ |
| 61f Punctuation toggle × 4 LLM-style decoders | 4 ✔ |
| **Subtotal** | **20 cells gained** |

### 61g. Audio Q&A (`--ask`) — DEFERRED

glm-asr is an ASR fine-tune (hardcoded prompt ids, no live
tokenizer for arbitrary instructions); omniasr-llm uses FLORES-200
language conditioning, not chat. Both would need empirical
validation showing the model honours an instruction prompt before
plumbing the toggle. Out of scope until a backend lands that's
actually instruction-tuned.

### 61h. Beam search for LLM family + enc-dec — IN PROGRESS

**Tier:** 3. **Effort:** ~300 LOC for shared decoder + 30 LOC per
backend. **Cells:** 8 (LLM quartet + qwen3/granite/voxtral4b +
canary/cohere/moonshine via per-model loop).

| Sub-step | Outcome |
|---|---|
| Generic `core_beam_decode` helper (header-only) — replay-from-prefix variant | DONE → [HISTORY §65](HISTORY.md) |
| Branched-KV variant (`run_with_probs_branched`) — per-beam `save`/`restore`/`step` callbacks | DONE — `O(B × T)` single-token forwards |
| glm-asr beam path (`-bs N`) | DONE — 1 ✔ (replay-from-prefix; batched LM helper) |
| moonshine LLM-side beam | DONE — 1 ✔ (branched-KV; per-layer `kv_self.{k,v}` snapshot) |
| omniasr-llm beam | DONE — 1 ✔ (branched-KV; whole-tensor `kv_k` / `kv_v` snapshot) |
| kyutai-stt per-frame text-token beam | DONE — 1 ✔ (branched-KV; one pick per Mimi frame, audio codes shared across beams) |
| qwen3/granite/voxtral4b/voxtral session-API beam | DEFERRED — pure plumbing once the session API exposes `beam_size` |
| canary/cohere/moonshine encoder-decoder beam (per-decoder loop) | DEFERRED — separate scope from the LLM beam path |

**What landed (May 2026 follow-up).** The original entry deferred
omniasr-llm / kyutai-stt / moonshine because `replay_fn` does
`O(B × T²)` single-token forwards for backends with one-token-at-a-
time decode. Resolved by adding `core_beam_decode::run_with_probs_branched`
— takes per-beam KV `save_fn`/`restore_fn`/`snap_free_fn`/`step_fn`
callbacks. Cost drops to `O(B × T)`. Snap holders are refcounted via
`shared_ptr` inside the helper so siblings can share a parent's snap
without double-free.

For each backend, KV snapshots are full-tensor `ggml_backend_tensor_get`
on either per-layer (`moonshine`) or contiguous (`omniasr-llm`,
`kyutai-stt`) K/V tensors. Cross-attention KV (moonshine, kyutai's
audio codes) stays shared across beams.

Smoke results on JFK (11 s, Metal, warm cache):

| Backend | model | `-bs 1` | `-bs 2` | `-bs 4` |
|---|---|---|---|---|
| moonshine | tiny-q4_k | 0.57s | 0.57s | 0.57s (improved transcript) |
| omniasr-llm | 300m-v2-q4_k | ~9.5s pure / 38s wall | ~22s | ~42s pure / 70s wall |
| kyutai-stt | 1b-q4_k | 5.1s | 8.6s | 14.2s |

Wall scales ~linearly with beam (each beam adds ~1× greedy compute).
Transcripts match greedy at all beam sizes; moonshine's `-bs 4`
actually improved quality on JFK ("fellow Americans" vs greedy's
"fellow-american"). omniasr-llm at `-bs 4` lands above the 60s
"rough" gate but well within order-of-magnitude.

**Still deferred and why.** The session-API quartet (qwen3, granite,
voxtral4b, voxtral) and the classic encoder-decoder backends (canary,
cohere) need either session-API plumbing (just `set_beam_size`
exposure) or a per-decoder beam path that reuses the cross-attention
KV across all beams. Both are pure plumbing — reopen when the wave
of enc-dec backends has a clear quality win to point at.

### 61i. Flash attention for fc-ctc — DEFERRED

`core_conformer::build_block`'s rel-pos path (`Q·K + R·Q_v +
rel_shift`) doesn't fit `ggml_flash_attn_ext` — the kernel has no
rel-pos hook. Would need either a positional-encoding swap or a
custom flash kernel. Reopen after PLAN #58 / Conformer rewrite.

### 61j. Translate + source/target lang for voxtral4b / glm-asr / omniasr-llm — OPEN

**Tier:** 3. **Effort:** ~100 LOC + empirical validation.
**Cells:** 3-6.

Try the translate template each model honours; ASR-roundtrip a
known X→Y pair; if sensible, add `CAP_TRANSLATE | CAP_SRC_TGT_LANGUAGE`.

### 61k. Grammar (GBNF) — BLOCKED on PLAN #60k

**Tier:** 4. **Cells:** 8 (qwen3, voxtral, voxtral4b, granite,
glm-asr, moonshine, omniasr-llm, kyutai-stt).

When 60k lands, every backend that token-by-token decodes through a
sampler can constrain output. Pure plumbing per backend.

### Validation gate

Each step must pass: golden JFK transcript unchanged, the new ✔
shows up in `crispasr --list-backends`, README matrix line updated,
`warn_unsupported` no longer fires for the toggled flag.

---

## 62. Streaming + mic library API

May 2026 — the C-ABI exposes `crispasr_stream_*` (open / feed /
get_text / flush / close) for low-latency rolling-window decoding,
but it's tied to a `whisper_context*` and only the Dart wrapper
surfaces it. Several backends are architecturally streaming
(moonshine-streaming, kyutai-stt 12.5 Hz frame-aligned, voxtral4b
240ms latency) but called as batch through the unified Session API.
Mic capture is CLI-only (`--mic` shells out to `rec`/`arecord`/
`ffmpeg`) — no library API exists.

This item closes the gap end-to-end so dictation / push-to-talk /
real-time captioning use cases can ship from any wrapper without
subprocess hacks.

### Status

| Piece | Status |
|---|---|
| `crispasr_session_stream_*` C-ABI | ✅ takes `crispasr_session*` |
| Python `Session.stream_*()` | ✅ DONE (stream_open/feed/get_text/flush/close + context manager) |
| Rust `Session::stream_*()` | ✅ DONE (Stream struct with feed/get_text/flush, Drop auto-close) |
| Dart `Session.stream*()` | ✅ DONE |
| Go `Session.StreamOpen()` | ✅ DONE |
| Library mic API (`crispasr_mic_*`) | ✅ DONE via miniaudio `ma_device` |
| Mic in Python/Rust/Dart | ✅ DONE — `Mic.open(callback)` in all three |
| moonshine-streaming wired to stream API | ✅ chunked-batch over rolling window |
| kyutai-stt wired to stream API | ✅ chunked-batch over rolling window |
| voxtral4b native streaming | ✅ DONE (PLAN #7 phases 1-4) |

### Sub-items

#### 62a. Python + Rust streaming wrappers — DONE

Already shipped. Python `Session.stream_open()` with context manager.
Rust `Session::stream_open()` with `Stream` struct (Drop auto-close).
Go `Session.StreamOpen()` added in #59 session. All use `hasattr` /
`providesSymbol` guards for older dylib compat.

#### 62b. Generalise `crispasr_stream_open` to a session handle — DONE

`crispasr_session_stream_open(crispasr_session*, ...)` ships and is
what all wrappers call. Legacy `crispasr_stream_open` alias kept.

#### 62d. Library-level mic API via miniaudio `ma_device` — DONE

`crispasr_mic_open/start/stop/close` shipped in `src/crispasr_mic.{h,cpp}`.
Wrapped in Python (`Mic` context manager), Rust (`Mic::open(callback)`),
Dart (`Mic.open`), Go (`Mic{open,start,stop,close}`), Java (JNA),
Ruby (C ext). Cross-platform via miniaudio backends.

#### 62c. kyutai-stt streaming — SHIPPED via chunked-batch

**Original scope** assumed true incremental encoding (refactor SEANet
conv chain for per-conv left-context state + per-call KV carry-over).
Pre-impl exploration surfaced a second trap: the Mimi encoder
transformer (`src/kyutai_stt.cpp:660`) calls `ggml_flash_attn_ext(...,
nullptr, ...)` — **fully non-causal**, every frame attends to every
other. True incremental encoding therefore can't bit-match batch
without either re-encoding the growing audio (O(n²) per session) or
replacing the encoder transformer with sliding-window attention
(~500–700 LOC, deviates from training).

**Chosen path: chunked-batch over a rolling window** (~200 LOC, no
encoder refactor). Mirrors whisper's `crispasr_stream_*`: each
`step_ms` re-runs the existing single-shot `kyutai_stt_transcribe_ex`
over the last `length_ms` of audio. Bit-exact match to batch on each
window. Latency ≥ `step_ms`; for audio longer than `length_ms` the
window only holds the tail (same trade-off whisper streaming already
accepts). Validated end-to-end on JFK: final stream output matches
single-shot batch byte-for-byte after stripping the leading
SentencePiece `▁ → space`.

Wired into `crispasr_session_stream_open` via a new optional
`kyutai_stream_state` field on `crispasr_stream`; the four
whisper-typed `crispasr_stream_*` functions branch on it.

**Moonshine-streaming** also shipped via the same chunked-batch
pattern (~120 LOC in `src/moonshine_streaming.{h,cpp}` +
`moonshine_streaming_state` field on `crispasr_stream` + four
dispatch branches). E2E validated word-perfect against batch on
JFK using `moonshine-streaming-tiny-f32.gguf`.

**Found + fixed in passing — `moonshine-streaming` F16 batch crash.**
The F16 GGUF (`moonshine-streaming-tiny-f16.gguf`, 84 MB) crashed
inside `audio_frontend_cpu` with "tensor read out of bounds":
`ggml_backend_tensor_get(t, dst, 0, n_elems * sizeof(float))`
overflowed by 2× when reading 2D weights stored as F16
(`ggml_nbytes(t) == n_elems * sizeof(ggml_fp16_t)`). The CPU
frontend hard-coded F32 byte counts; the converter quantizes 2D+
weights to F16 in F16 mode, so the read size mismatched. Fix:
new `tensor_get_as_f32` helper in `src/moonshine_streaming.cpp`
that branches on `t->type` and dequantizes if F16, applied at the
3 weight-read sites (linear_w / conv1_w / conv2_w). Biases +
scalar `log_k` are always F32 per the converter's 1D path.
F16 batch + streaming + F32 regression all word-perfect on JFK
post-fix.

**Voxtral4b native streaming** — see #62e / PLAN #7.

Sub-second latency for kyutai/moonshine via true incremental
encoding remains the deferred path. The new `kyutai_stt_stream`
struct in `src/kyutai_stt.cpp` is the adapter layer; the internals
get swapped without ABI breaks if a consumer eventually hits the
latency wall.

#### 62e (deferred). Voxtral4b native streaming — see PLAN #7

Already tracked separately. ~200-300 LOC, decoder thread + audio
frame injection. High complexity, separate session.

### Sequencing

a + b + d shipped as 947262f (a/b spec) + 041471f (Python+Rust
streaming) + 89687f0 (mic). Go wrapper sticky setters + streaming
shipped in this PLAN-uplift commit. c and e remain deferred per
the revised effort estimates above — open them when a consumer
explicitly needs sub-second latency on kyutai/moonshine/voxtral4b.

### Init-only flag refactor (related deferral)

CLI flags `--temperature`, `--beam-size`, `--flash-attn`, `--grammar`
are baked into backend contexts at `_init_from_file()` time on every
backend. Surfacing them as session-level setters means tearing down
+ reopening the context (~15-30s per swap depending on model size)
or refactoring per-backend init to accept post-init parameter
updates. The temperature setter (`crispasr_session_set_temperature`)
already works via per-backend runtime setters that 4 backends
expose; the others (beam/flash/grammar) would need either:

- **Backend-reinit machinery** (close + reopen + load weights again)
   — easy to write, slow to use, fine for "set once at session
   creation" use cases.
- **Per-backend `set_*` extensions** — each backend exposes a
   runtime setter (parakeet/canary/cohere already have
   `set_temperature`; extend to `set_beam` etc.). Per-backend work,
   no unified machinery.

Realistic effort: ~50 LOC per backend × 14 backends = ~700 LOC
mechanical, but each one needs a regression test that the new flag
actually changes output. **Defer until a consumer asks for a
specific flag on a specific backend** (per PLAN #59 policy).

---

## 66. Wrapper publishing bootstrap — required before language registries can ship

**Status:** OPEN, auto-trigger silenced. The `tags: ['v*']` push
trigger on `release-wrappers.yml` is now COMMENTED OUT so future tag
pushes don't keep producing red runs while we're not ready to
bootstrap. Workflow stays in the repo on `workflow_dispatch` only —
manual dispatch still works for ad-hoc testing during bootstrap.
Failed on every release since v0.5.0; confirmed again on v0.5.4
(`gh run view 25248028443`).

The CI workflow pushes to three registries automatically on every
`v*` tag, but **none of the packages currently exist on those
registries**:

- crates.io: `crispasr-sys` and `crispasr` do not exist (404).
- PyPI: `crispasr` does not exist (404).
- pub.dev: `crispasr` does not exist.

All three registries require **manual bootstrap** — the first
version of any package can't be published by an OIDC / token CI
flow because the registry has no prior owner record to verify
against. After the first manual publish, automated publishing
takes over via the existing workflow.

### Bootstrap steps (one-time, requires repo admin credentials)

1. **crates.io** (Rust, simplest):
   ```bash
   cargo login   # paste API token from https://crates.io/me
   cargo publish --manifest-path crispasr-sys/Cargo.toml --allow-dirty
   sleep 30   # wait for crates.io index
   cargo publish --manifest-path crispasr/Cargo.toml --allow-dirty
   ```
   Then add `CARGO_REGISTRY_TOKEN` repo secret (Settings → Secrets
   → Actions). Subsequent tag pushes auto-publish.

2. **PyPI** (uses trusted publishing / OIDC):
   - Visit https://pypi.org/manage/account/publishing/ and create a
     pending publisher with:
     - Owner: `CrispStrobe` (or org owning the repo)
     - Repository: `CrispASR`
     - Workflow: `release-wrappers.yml`
     - Environment: `pypi`
   - Push a `v*` tag and the OIDC handshake creates the package.
     (No manual `twine upload` needed — the pending-publisher
     mechanism IS the bootstrap path.)

3. **pub.dev** (Dart, hardest — `dart pub publish` requires a
   logged-in interactive shell for the first version):
   ```bash
   cd flutter/crispasr
   dart pub get
   dart pub publish   # interactive: confirm, log in via browser,
                      # accept the package contents
   ```
   Then visit https://pub.dev/packages/crispasr/admin and enable
   "Automated publishing" with:
   - Repository: `CrispStrobe/CrispASR`
   - Tag pattern: `v{{version}}`

### Resilience improvements landed alongside this entry

`release-wrappers.yml` is updated so when we DO re-enable the
auto-trigger, a single registry's misconfiguration doesn't fail the
whole workflow:

- Auto-trigger on `tags: ['v*']` is currently **commented out**.
  Re-enable by un-commenting the two lines (`push:` /
  `tags: ['v*']`) after bootstrap completes.
- Each job runs a fast secret/config presence check at the top and
  echoes a clear "skipping: registry X not configured" instead of
  letting `cargo` / `twine` emit cryptic auth errors deep in the
  log.
- Each job uses `continue-on-error: true` so the others still try.
- Workflow comment block updated to reference this PLAN section.

After bootstrap + re-enabling the trigger, the next tag push should
publish all three wrappers cleanly.

---

## 67. Deferred follow-ups carry-over (mid-May 2026 session)

Captured here so they don't get lost between sessions.

### 60d F16 mimo-asr re-upload (HF)

The Q4_K fused-QKV file is on HF
(`cstr/mimo-asr-GGUF/mimo-asr-q4_k.gguf`, 4.2 GB). The F16 variant
on HF is still the legacy unfused layout — the runtime fallback
keeps it working but it doesn't get the 1.7× per-step decode that
fused QKV unlocks. Re-conversion needs a fresh BF16→F16 run,
which on this 16 GB / 99%-full-disk box sustained ~0.8 MB/min and
was killed at 22 min (PLAN #51c disk-thrash signature). Run on a
32+ GB box with non-99%-full external. Then
`tools/patch_mimo_asr_fuse_qkv.py` patches it to the fused layout
(~5 min vs hours for a fresh quantize).

### 60e per-backend Q8_0 KV cosine validation

Env wiring (`CRISPASR_KV_QUANT={f16,q8_0,q4_0}`) landed across 9
backends (mimo_asr, qwen3_asr, voxtral, voxtral4b, granite_speech,
gemma4_e2b, glm_asr, omniasr, orpheus, qwen3_tts) — defaults stay
F16 so it's bit-identical until opted in. **Only mimo-asr has been
diff-harness validated at q8_0** (last_hidden 0.963031 vs F16
0.963177; logits 0.981454 vs 0.981261, both ≥0.98 gate). The
remaining 8 backends need their own
`CRISPASR_KV_QUANT=q8_0 crispasr-diff <backend>` pass before any
default-flip per backend.

Effort: ~1 diff-harness run per backend, ~5 min each on warm
cache. Zero code work — wiring is in place.

### Vibevoice CUDA cache reuse re-test

`backend_needs_fresh_pred_graph()` defensively bypasses the
pred-head graph cache on Metal + Vulkan + CUDA (CUDA added on the
"shape suggests it's broken too" presumption). When a CUDA box is
available, run `CRISPASR_VIBEVOICE_REUSE_PRED_GRAPH=1` and confirm
TTS runs without `GGML_ASSERT(src_backend_id != -1)`. If the cache
works there, drop the `CUDA` prefix from the bypass list and
recover the ~30% per-synthesis caching speedup.

If the assert fires, the env hatch stays disabled by default and
the proper upstream-ggml fix (recompute view→backend mapping
from `view_src->buffer` in `ggml_backend_sched_split_graph`)
becomes the next step.

### SYCL / HIP / ROCm cache-bypass extension

Same shape as CUDA — these multi-backend GPU schedulers probably
need the bypass too but no user has reported. Extend
`backend_needs_fresh_pred_graph()` prefix list when a report comes
in or when a kernel maintainer audits the upstream
`ggml_backend_sched_split_graph` reset path on those backends.

### Per-backend `MADV_RANDOM` post-prefill wiring (PLAN #60g)

`core_gguf::mmap_advise_random()` is exposed but no backend calls
it yet. Add a single call between prefill and the decode loop in
`mimo_asr_transcribe`, `qwen3_asr_transcribe`, `voxtral_transcribe`,
etc. when a 32+ GB-box benchmark demonstrates measurable benefit
(on Q4_K the readahead delta is marginal; F16 is where it would
matter, and we can't reliably measure F16 on 16 GB).

### Disk5 cleanup

`/Volumes/backups` sits at 99% full, 30 GB free. The
`/Volumes/backups/ai/crispasr-models/mimo/mimo-asr-q4_k.gguf`
unfused (4.2 GB) is now superseded by `mimo-asr-q4_k.fused.gguf`
and the HF copy of the fused. Safe to delete the local unfused
once future A/B testing isn't needed.

### CI: legacy `build.yml`

`.github/workflows/build.yml` is the legacy whisper.cpp CI matrix
(triggers on `branches: [master]` which doesn't exist + `tags: v*`).
Has been failing on every tag push since v0.4.x. Doesn't block
releases (the new `ci.yml` / `release.yml` are the actual gates).
Either delete or repair when convenient — pending audit on whether
any build-matrix combination there isn't covered by the new
`ci.yml` matrix.

---

## 63. Feature matrix parity

**Motivation:** Feature matrix audit (May 2026) revealed many backends
have capabilities the infrastructure already supports but that aren't
wired up. The `core_greedy_decode` / `core_beam_decode` / flash-attn
infrastructure is shared — hooking new backends into existing machinery
is often a small patch per backend.

**Audit baseline:** `crispasr --list-backends` output + `test-all-backends.py
--profile=feature` run (51 PASS, 0 FAIL after registry cleanup).

### Phase 1 — Beam search for LLM backends — PARTIALLY DONE

**DONE:** granite (all variants) + qwen3 — wired `core_beam_decode::run_with_probs`
with replay-from-prefix in the CLI backend adapters. Tested with -bs 4 on JFK.

**SKIPPED (architectural blockers):**
- voxtral4b: per-step audio-injection decode loop incompatible with replay-from-prefix
- canary/cohere: decode is inside opaque library calls, no beam_size parameter exposed

### Phase 2 — Auto-download gaps (SMALL)

| Backend | HF repo exists | Missing from registry |
|---|---|---|
| omniasr (CTC) | cstr/omniASR-CTC-1B-v2-GGUF | no auto-dl flag |
| omniasr-llm | cstr/omniasr-llm-300m-v2-GGUF | no auto-dl flag |
| gemma4-e2b | (needs upload) | no model, no auto-dl |
| mimo-asr | cstr/mimo-asr-GGUF | no auto-dl flag |

**Approach:** Add `model_url` + `model_filename` to each backend's
registry entry in `src/crispasr_model_registry.cpp`. Then set
`CAP_AUTO_DOWNLOAD` in `capabilities()`.

**Effort:** Trivial, ~1 hour for all four.

### Phase 3 — Flash attention audit — DONE

Backends claiming flash attention in `--list-backends`:
whisper, parakeet, canary, cohere, granite, voxtral, voxtral4b, qwen3,
vibevoice, qwen3-tts, orpheus.

Backends that **should** have flash attention (use standard ggml
self-attention with Q/K/V matmuls) but don't declare it:
glm-asr, kyutai-stt, firered-asr, moonshine, omniasr, omniasr-llm.

**Approach:** Each backend allocates ggml graphs for self-attention.
Flash attention is a ggml-level optimization (`GGML_OP_FLASH_ATTN_EXT`)
that replaces the Q×K^T → softmax → ×V chain. Requirements:
1. Q/K/V must be contiguous tensors with standard layout
2. `ggml_flash_attn_ext` must be called instead of the manual chain
3. Causal mask must be passed correctly

For each backend: check if the attention implementation uses manual
matmul chains or already calls `ggml_flash_attn_ext`. If manual,
refactor to use `ggml_flash_attn_ext` behind a `flash_attn` flag.

All 6 backends (glm-asr, kyutai-stt, firered-asr, moonshine, omniasr,
omniasr-llm) already call `ggml_flash_attn_ext` in their source — they
just didn't declare `CAP_FLASH_ATTN`. Added the flag to each CLI
adapter. No code changes to the attention implementations needed.

### Phase 4 — Word timestamps via CTC aligner for all backends — DONE

The `-am` (alignment model) flag works with any backend that produces
text output. Currently documented for: granite, voxtral, voxtral4b,
qwen3, fc-ctc, wav2vec2, glm-asr, firered, moonshine, omniasr.

**Missing documentation/testing for:** moonshine-streaming, omniasr-llm,
vibevoice, mimo-asr, gemma4-e2b.

**Approach:** These backends already produce text. The aligner runs
independently on (text, audio) pairs. Just needs:
1. Test with `-am canary-ctc-aligner.gguf` on each backend's output
2. Add `-am` to the feature matrix if it works
3. Update test-all-backends.py capabilities

Added `CAP_TIMESTAMPS_CTC` to moonshine, moonshine-streaming, omniasr,
omniasr-llm, mimo-asr. The `-am` aligner flag now accepts these backends
(gated by `CAP_TIMESTAMPS_CTC` in `crispasr_run.cpp:292`).

### Phase 5 — Punctuation for CTC backends via auto-punc — DONE

CTC backends (fc-ctc, wav2vec2, omniasr-ctc, firered-asr) produce
lowercase text without punctuation. The `--punc-model auto` flag
auto-downloads FireRedPunc (~50 MB) and restores punctuation.

**Idea:** Make `--punc-model auto` the default when a CTC backend is
detected and `--no-punctuation` is not set. This would give all CTC
backends punctuated output by default, matching user expectations.

**Risk:** Adds ~50 MB download + ~200 ms latency per segment. Should
be opt-in initially, maybe promoted to default in a future release.

Implemented: `crispasr_run.cpp` auto-sets `punc_model = "auto"` when
the backend lacks `CAP_PUNCTUATION_TOGGLE` and `--no-punctuation` is
not set. FireRedPunc (~50 MB) auto-downloads on first use. Users can
suppress with `--punc-model none` or `--no-punctuation`. Tested on
fastconformer-ctc: raw CTC output gets capitalization + commas.

### Phase 6 — Best-of-N for LLM backends (LOW priority)

Best-of-N (`-bo N`) runs N independent decodes at temperature > 0 and
picks the highest-scoring result. Currently times out on CPU for large
models (omniasr-llm 300s, glm-asr 300s, kyutai-stt 90s).

**CLI side:** not a code issue — the feature works, it's just too slow
on CPU with Q4_K models > 500 MB. On GPU or with smaller models it
completes. No code changes needed; just document the GPU requirement.

**SDK side (gap discovered 2026-05-03 by CrisperWeaver §5.8) — DONE:**

- `crispasr_session_set_best_of(session, n) → int rc` shipped in
  `src/crispasr_c_api.cpp` (commit 62be9b1, 2026-05-03). Whisper
  wires directly to `greedy.best_of`; all other session backends use
  an external N-pass wrapper in `transcribe_lang` that picks the
  candidate with the highest average per-word confidence — backends
  emitting word-level probs (canary, qwen3, voxtral, granite, cohere)
  benefit most.
- `crispasr_params_set_best_of(params, n)` added to the legacy
  `whisper_full_params` path so the Dart wrapper (which uses
  `whisper_full` directly, not the session API) can set it too.
- `int bestOf` field on `TranscribeOptions` in
  `flutter/crispasr/lib/src/crispasr.dart`, default 0 (disabled);
  values > 1 call through to the new params setter.

CrisperWeaver can now expose the slider against `TranscribeOptions.bestOf`
for whisper sessions and `crispasr_session_set_best_of` for everything else.

### Phase 7 — Capability declaration fixes — DONE

Re-audit found the initial agent report was wrong: firered-asr,
moonshine, kyutai-stt, omniasr all already had correct capability
declarations in their CLI backend adapters.

Actual fixes applied:
- omniasr: added CAP_AUTO_DOWNLOAD (had registry entry but not the cap flag)
- mimo-asr: capabilities was 0, added CAP_AUTO_DOWNLOAD + CAP_TOKEN_CONFIDENCE
- mimo-asr: added model registry entry (cstr/mimo-asr-GGUF)

### Phase 8 — vibevoice CLI backend adapter — ALREADY EXISTS

Re-audit found `crispasr_backend_vibevoice.cpp` already exists with
full ASR + TTS support (160 lines, 16k→24k resample, voice loading).
Initial agent report was incorrect.

### Phase 9 — Translation for backends that support it — INVESTIGATED

| Backend | Result |
|---|---|
| cohere | No translate token in vocab — model is transcription-only, not feasible |
| glm-asr | Code wired (translate flag + prompt injection), but GLM-ASR-Nano doesn't respond to translation instructions — model wasn't trained for it. Infrastructure ready for translation-capable GLM variants. |

No `CAP_TRANSLATE` declared for either — models can't actually translate.

### Summary — expected matrix after all phases

| Feature gained | Backends affected |
|---|---|
| Beam search | +granite, +qwen3 (DONE); canary/cohere/voxtral4b blocked |
| Cap declarations | +firered, +moonshine, +kyutai-stt, +omniasr, +omniasr-llm |
| Auto-download | +omniasr, +omniasr-llm, +mimo-asr |
| Flash attention | +glm-asr, +kyutai-stt, +firered, +moonshine, +omniasr |
| Word timestamps (-am) | +moonshine-streaming, +omniasr-llm, +vibevoice, +mimo-asr |
| Auto-punctuation | +fc-ctc, +wav2vec2, +omniasr-ctc, +firered (opt-in) |
| vibevoice CLI adapter | +vibevoice (CLI path) |


## 69. Layer + KV CPU-offload knobs (llama.cpp parity)

**Effort:** Medium-large per backend (~150-200 LOC for the
weight-residency-aware variant — see scope note below). Originally
estimated at 50-80 LOC; that turned out to be the *compute-only*
pattern, which doesn't actually solve the VRAM-pressure problem
this PLAN entry exists for.

**Two valid designs, only one is useful for #60's case.**

**(a) Compute-only offload** (~50 LOC, kokoro / vibevoice pattern).
Walk the built graph, call
`ggml_backend_sched_set_tensor_backend(sched, node, backend_cpu)`
on the output tensors of layers `il >= n_gpu_layers`. The sched
handles GPU↔CPU transfers at the op boundary. Weights stay on
GPU buffer; only the compute moves. Useful for "this op is broken
on this GPU" (Metal conv_transpose_1d hang, Intel Vulkan workgroup
limit), useless for "model doesn't fit."

**(b) Weight-residency offload** (~150-200 LOC, llama.cpp's
`--n-gpu-layers` pattern). Allocate two backend buffers at load
time: first N layers go on `c->backend` (GPU), rest on
`c->backend_cpu`. Each `voxtral4b_block` stores tensor pointers
that already live on the right backend. Graph builder produces
ops naturally placed by sched (each op's compute follows its
input weights). Properly fits the VRAM use case — N can be tuned
down until the GPU residence fits in available VRAM.

**For #60's voxtral4b VRAM problem, only (b) is useful.** The user
worked around it with `CRISPASR_KV_QUANT=q4_0 + CRISPASR_GGUF_MMAP=1`
(both already shipped); that combo is the recommended interim
answer until (b) lands.

**Background.** External feature request via #60: Voxtral 4B
specifically needs the ability to offload N transformer blocks to
CPU and / or pin the KV cache to CPU on GPU-weight builds. llama.cpp
exposes `--n-gpu-layers N` for this; we currently have no
equivalent. Today's escape hatches against VRAM pressure are:

- `CRISPASR_KV_QUANT=q8_0|q4_0` — halve / quarter the KV cache
  (already shipped, PLAN #60e, 9+ backends incl. voxtral4b)
- `CRISPASR_GGUF_MMAP=1` — don't double-allocate weights on load
  (already shipped, PLAN #51a / HISTORY §62)

That covers steady-state footprint, but not the case where the
model itself doesn't fit in VRAM. For that we'd need per-block
placement.

### 69a. N-layer CPU offload — `--n-gpu-layers` / env equivalent

The pattern to mirror is `src/kokoro.cpp:2174`'s per-op CPU pinning,
generalised to "first N transformer blocks":

```cpp
const int n_gpu_layers = env_int_default("CRISPASR_N_GPU_LAYERS", -1);  // -1 = all
if (n_gpu_layers >= 0 && n_gpu_layers < (int)hp.n_layers) {
    for (int il = n_gpu_layers; il < (int)hp.n_layers; il++) {
        // Pin block `il`'s Q/K/V/O/FFN tensors to backend_cpu via
        // ggml_backend_sched_set_tensor_backend().
    }
}
```

Where: per-backend graph builder (`build_graph_llm_kv` for voxtral4b,
analogous for voxtral / qwen3_asr / glm_asr / granite_speech /
gemma4_e2b / orpheus / mimo_asr / omniasr-llm). Tag the layer index
on each block's tensors when constructing the graph; the loop above
walks them and CPU-pins at sched-alloc time.

CLI plumbing: `whisper_params.n_gpu_layers` (already exists from
upstream whisper at `examples/common.h:25`!) → forward into each
backend's `init()` path, threaded through to the graph builder.

Per-backend implementation list (rough order: highest-VRAM first):

1. **voxtral4b** (3.4 B LLM) — direct request from #60. Largest VRAM
   footprint of the LLM-decode backends.
2. **voxtral** (3 B Mistral)
3. **qwen3_asr** (0.6 B Qwen3)
4. **granite_speech** + granite-4.1 / 4.1-plus / 4.1-nar
5. **gemma4_e2b** (E2B = ~5 B effective; widening capabilities just
   landed in cf20c08, layer offload would be the natural follow-up)
6. **glm_asr**
7. **orpheus** (Llama 3.2 3 B talker)
8. **mimo_asr**
9. **omniasr-llm**
10. **vibevoice-tts** — Qwen2 7B; may need the most help on VRAM

Skip: ASR-only encoders (parakeet, canary, cohere, fc-ctc, wav2vec2,
firered-asr, moonshine) — encoder graphs aren't layered the same way
and the VRAM footprint isn't a problem.

### 69b. KV-only CPU offload (`CRISPASR_KV_ON_CPU=1`)

Allocate `ctx->kv_buf` on `ctx->backend_cpu` instead of `ctx->backend`
even when GPU weights are active. Implementation: change one line
per backend's `kv_init`:

```cpp
ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, k_size + v_size);
// →
ggml_backend_t kv_backend = env_bool("CRISPASR_KV_ON_CPU")
                              ? ctx->backend_cpu : ctx->backend;
ctx->kv_buf = ggml_backend_alloc_buffer(kv_backend, k_size + v_size);
```

The expensive part isn't the alloc — it's that every attention step
copies the KV slice GPU↔CPU↔GPU. Typically slower than just using
`KV_QUANT=q4_0` to fit KV in VRAM. Land it for users who legitimately
need it (very long context, very small VRAM headroom) but document
that `KV_QUANT` should be tried first.

### 69e. Asymmetric K-vs-V cache quantization (llama.cpp parity)

Today our `KV_QUANT=<type>` flag (#60e, shipped) applies the same
precision to both K and V. llama.cpp exposes the two halves
independently (`--cache-type-k` / `--cache-type-v`) because the
sensitivity profiles are very different:

- **V** quantizes down well — it gets used as `softmax(QK^T) · V`.
  The softmax already concentrates probability mass, so per-element
  errors get averaged across attended positions. `q4_0` V is
  typically indistinguishable from F16 in PPL.
- **K** is the fragile half — `QK^T / sqrt(d)` produces attention
  scores *before* the softmax, and softmax exponentiates. Errors
  here distort *which* positions get attended to. K usually wants
  Q8_0 or higher for the same PPL floor.

The common llama.cpp recipe is `-ctk q8_0 -ctv q4_0` — about 40 %
more KV memory savings than symmetric Q8_0, with PPL barely moved
on Llama-class models. We have headroom to push V lower than the
Q8_0 we shipped in #60e.

Most useful on the LLM-decode backends where KV is the dominant
memory pressure (voxtral4b, granite-speech-4.x, qwen3, mimo-asr).

#### Implementation

Split the env knob:

```
CRISPASR_KV_QUANT       (legacy; sets both)  → keep for back-compat
CRISPASR_KV_QUANT_K     (new; overrides KV_QUANT for K only)
CRISPASR_KV_QUANT_V     (new; overrides KV_QUANT for V only)
```

In each backend's `kv_init`, pick `k_type` and `v_type` independently
from these env vars (default both to `KV_QUANT`, default that to
F16). The K and V buffers already get separate `ggml_tensor`s in
all current backends, so the change is:

```cpp
ggml_type k_type = parse_kv_quant("CRISPASR_KV_QUANT_K", default_kv);
ggml_type v_type = parse_kv_quant("CRISPASR_KV_QUANT_V", default_kv);
ctx->kv_k = ggml_new_tensor_3d(meta, k_type, ...);
ctx->kv_v = ggml_new_tensor_3d(meta, v_type, ...);
```

(parse_kv_quant() falls back to `CRISPASR_KV_QUANT` when its
type-specific var is unset.)

#### Validation

Use the same #60e methodology — bit-exact at F16 (no-op), WER=0
floor at Q8_0/Q8_0, and confirm PPL/WER stays at the WER=0 floor
for `-ctk q8_0 -ctv q4_0` on each LLM-decode backend. The Q8_0/Q4_0
asymmetric pair is the one that ships; Q4_0/Q4_0 (symmetric) is
expected to degrade and serves as the sanity check that V is the
forgiving half. Compare on JFK + a few longer LibriSpeech clips
because the longer the context the more the V error matters.

#### Effort

Small. The plumbing change is ~10 LOC per backend, and we already
have parse helpers from #60e. Validation across 7 LLM-decode
backends is the time sink. Estimate: 1 backend in a session, full
roll-out across LLM-decode backends in a week.

#### Out of scope

- Per-layer K/V quant (some llama.cpp users go finer than `-ctk` /
  `-ctv` — e.g. lower precision in middle layers). Adds a third
  dimension of ablation surface that isn't justified until the
  whole-cache asymmetric pair is shipped and validated.

### Approach (do these in order)

1. Land `CRISPASR_N_GPU_LAYERS` for voxtral4b (the requesting user's
   target). Validate against #60's reproducer.
2. Land `CRISPASR_KV_ON_CPU` for voxtral4b — same backend, ~5 LOC.
3. Land `CRISPASR_KV_QUANT_K` / `_V` (#69e) for voxtral4b — drop in
   beside the existing KV_QUANT path; validate the `q8_0`/`q4_0`
   pair against the WER=0 floor from #60e.
4. If validation passes and there's pull, replicate to the rest of
   the LLM-decode backends one at a time. Don't preemptively roll
   it out everywhere — the per-backend test surface (each one's
   layer-tagging needs verifying) doesn't scale linearly.

### Files touched (per backend, approximate)

```
include/crispasr.h                       — public API tweak (n_gpu_layers field)
src/<backend>.{h,cpp}                    — graph builder layer tag
                                            + kv_init backend pick
examples/cli/whisper_params.h            — already has n_gpu_layers
examples/cli/cli.cpp                     — surface --n-gpu-layers flag
                                            (or rely on env var only)
examples/cli/crispasr_backend_<name>.cpp — forward params.n_gpu_layers
                                            into init()
docs/cli.md                              — document the new flag(s)
```

### Out of scope

- A `--cpu-mask` style fine-grained tensor offload (per-tensor CPU
  pinning across the whole graph): too much UI surface for the
  user-side win.
- Per-tensor mmap-from-disk inference (`memory-mapped weights` with
  cold-loaded tensors): that's a separate, larger feature — track as
  its own item if it ever becomes a priority.

---

## 70. Streaming TTS via chunked VAE decode (latency win, vibevoice / qwen3-tts)

**Effort:** Medium-large.

**Background.** Issue #52 surfaced a chunked-VAE patch from
[`geneing`](https://github.com/CrispStrobe/CrispASR/issues/52#issuecomment-4366745018)
that re-runs the σ-VAE decoder on small chunks of the latent stream
instead of one big graph. Their measurement showed a speed regression
because re-running the ggml graph N times pays per-call setup
(`sched_reset` + `sched_alloc_graph` + the kernel-launch ramp-up) on
every chunk. So that patch isn't useful for the Intel-Arc Vulkan
workgroup-limit bug it was filed against — that's already fixed by
the CPU fallback in `31795a7` / `VIBEVOICE_VAE_BACKEND=cpu`.

**But chunking is the right shape for a latency feature, not a
throughput one.** If we ever want streaming TTS — the listener
starts hearing audio before AR completes — we'll need chunked VAE
*plus* the rest of the pipeline. A `--stream` mode for `--tts` would
look like: emit a 24 kHz PCM chunk every K AR steps, written to
stdout / streamed over HTTP, while the AR loop continues. Time-to-
first-byte drops from "full TTS wall-clock" to "K AR steps + one
chunked-VAE pass."

This is **not the same project as the Intel-Vulkan workgroup fix.**
We'd want a chunked VAE that's well-engineered for latency rather
than borrowed from a workgroup-limit workaround.

### Three pieces required

1. **Persistent VAE compute-graph reused across chunks.** The
   per-call `sched_reset` + `sched_alloc_graph` overhead is what
   killed geneing's prototype's speed. Pattern to mirror is
   qwen3-tts's `O15` graph reuse (see `src/qwen3_tts.cpp:1037`):
   build the graph once at `Lk = max_chunk_latents`, pin the
   tensor topology, reuse the cached gallocr plan across all
   chunk decode calls. Net cost is one `set_rows`-style
   "where to write this chunk's output" op per call, not a full
   rebuild.

2. **Causal padding on the σ-VAE conv stack.** The σ-VAE
   transposed-conv stack has receptive field that crosses chunk
   boundaries — naive chunking will produce phase artefacts at
   the boundaries. Causal padding (left-pad each chunk with the
   previous chunk's tail context, drop the first L padding samples
   from the output) makes the chunk decode equivalent to the full
   decode at chunk boundaries. Reference: kokoro and voxtral4b
   already use causal-conv1d padding for streaming-encoder paths;
   the σ-VAE side has a different topology but the math is the
   same.

3. **Chunked transfer in the HTTP TTS endpoint.** Once #58's
   `POST /v1/audio/speech` lands (vkrmch's PR), wire chunked-
   transfer-encoded audio output for clients with `Accept:
   audio/wav; chunked` (or a `stream=true` request field). cpp-
   httplib has chunked-transfer support out of the box. Without
   this piece the latency win can't reach the network — server
   would compute chunks fast but still wait until the last chunk
   to flush.

### Backends in scope

- **vibevoice TTS** (σ-VAE decoder) — primary target, the patch
  origin. Largest latency win because vibevoice is positioned as
  the realtime TTS backend.
- **qwen3-tts codec decode** — different architecture (12 Hz codec
  vocoder, not a σ-VAE) but the same chunked-decode-with-graph-
  reuse pattern applies. Already has graph reuse via `O15`; would
  extend that to chunked output.
- **kokoro iSTFTNet generator** — different shape again
  (deterministic vocoder, not a diffusion VAE). Chunking is
  cleaner here because the generator is straight-line; harder
  because the iSTFT inverse window has the same boundary
  artefact problem.

Skip out-of-scope: orpheus uses the SNAC codec which already
emits 24 kHz PCM in a single forward pass — chunking has no
latency win there.

### Approach

Pre-work: revisit geneing's
[chunked_vibevoice.patch](https://github.com/user-attachments/files/27326191/chunked_vibevoice.patch)
as a starting point — it nailed the chunking decomposition;
where it gave up was on the per-call overhead. Land the graph-
reuse fix first (mostly mechanical), benchmark to confirm the
regression is gone, then layer in the causal-padding and HTTP
chunked transfer.

### Files touched

- `src/vibevoice.cpp` (and `vibevoice_tts.cpp`) — chunked decode
  path with graph reuse + causal padding
- `examples/cli/crispasr_backend_vibevoice.cpp` — `--stream`
  output path: write each chunk's PCM to `stdout` as they
  complete, instead of buffering and writing one WAV at end
- `examples/cli/cli.cpp` — surface `--tts-stream` flag
- `examples/server/server.cpp` — chunked-transfer wiring for
  `/v1/audio/speech` (depends on #58 landing first)
- `docs/tts.md` — document the new flag + the streaming env
  var(s)
- `LEARNINGS.md` — document the per-call ggml graph overhead
  trap and the graph-reuse cure (geneing's patch is the
  cautionary tale)

### Out of scope for v1

- Multi-chunk look-ahead (lower latency at cost of slightly worse
  boundary behaviour) — a single look-ahead chunk is already a
  meaningful tuning knob; tuning past that adds complexity that
  isn't justified until we measure how good the v1 latency is.
- Non-vibevoice / non-qwen3-tts backends — kokoro / orpheus
  chunking is its own work item if anyone needs it.
- Any changes to AR decoding itself — the AR loop stays
  unchanged; only the post-AR codec / VAE side is chunked.

## 71. Test-runner under-invocation + cap-honesty audit (2026-05-04)

`tools/test-all-backends.py --profile feature` was run against the
fresh main with the widened cap tuples (commit 3c2864e). The suite
was stopped at 15/24 backends due to slow external-disk model loads
(USB-2-class 30 MB/s, granite-1b's 2.8 GB → ~90 s of I/O alone before
each cap), but enough data landed to triage every advertised cap.

### Findings shipped this session

#### Real cap drift (binary-side)

- **omniasr / omniasr-llm `CAP_PUNCTUATION_TOGGLE`** — both declared
  the cap but their CTC vocab is lowercase + unpunctuated by design.
  Verified against JFK: output is `"and so my fellow americas ask
  not ..."`. **Fix shipped**: removed the cap from
  `crispasr_backend_omniasr.cpp:22` and dropped `"punctuation"` from
  the test tuples. Audit re-run is clean.

#### Test-runner under-invocation (runner-side, not real drift)

- **`test_lid` ran with empty extra args** — non-whisper backends only
  trigger the framework LID pre-step on `-dl` / `-l auto`. **Fix
  shipped**: pass `-dl` always, and widen the regex to also accept the
  framework's `crispasr: LID -> language = '<lang>'` log line shape.
- **`test_word_timestamps` only tests native** — backends that declare
  only `CAP_TIMESTAMPS_CTC` need `-am <aligner.gguf>` to emit per-word
  entries via the post-step CTC aligner. **Fix shipped (partial)**:
  the runner now queries `--list-backends-json` and SKIPs (rather than
  FAILs) when the backend has only `timestamps-ctc`. Wiring an actual
  aligner default is deferred — see "Open work" below.

### Open work

- **CAP_LANGUAGE_DETECT honesty across non-whisper backends** — DONE.
  Read parakeet, qwen3, glm-asr, gemma4-e2b transcribe paths; only
  qwen3 had any LID-related code (a "language <name>" prefix scrape
  off the model output, fired only when the system prompt asks for
  translation), and it didn't fire on plain `-dl`. Dropped the cap
  from all four backends so `-dl` correctly routes through the
  framework's whisper-tiny pre-step LID. qwen3's stderr line was
  also tightened (ISO code + p=1.000 format) for the cases where it
  does fire — it's no longer cap-claimed but remains informational.
  test_lid runner now invokes without `--no-prints` because the
  framework + qwen3 LID lines are gated on it (whisper's path uses
  CRISPASR_LOG_INFO and ignores the gate, which is why the original
  whisper-only test ran with --no-prints).
- **Wire a default CTC aligner into `test_word_timestamps`** — the
  current SKIP is honest but it under-tests 15 backends that all
  declare `CAP_TIMESTAMPS_CTC`. Pick a small, English-friendly
  aligner (granite-aligner candidate or wav2vec2-en) and download it
  on first need.
- **Big-model perf in the test loop** — the runner spawns crispasr
  fresh for each cap, so big models pay the full model-load tax on
  every cap. For local laptop dev with `/Volumes/backups` (USB-class
  30 MB/s), this turns granite/voxtral/gemma4-e2b into 5+ min/cap.
  Two ergonomic options: (a) stage models to internal SSD before a
  full feature-profile run, or (b) add a `--cap-once` mode that runs
  all caps in a single process via the session API. (a) is fastest,
  (b) is right for CI.
- **Untracked variant backends (11)** — `data2vec`, `hubert`,
  `granite-4.1-{nar,plus}`, `kartoffel-orpheus-de-{natural,synthetic}`,
  `lex-au-orpheus-de`, `qwen3-tts-1.7b-{base,customvoice,voicedesign}`,
  `qwen3-tts-customvoice`. Each is a minor variant of an architecture
  already covered by a tracked backend; the audit reports them but
  doesn't gate. Add as separate Backend tuples only if a variant
  starts diverging in caps from its parent.

### Why this came up

The capability widening commit (3c2864e) made the test script's
*intent* match the binary's *claims*. Running the resulting suite
exposed the gap between "claim" and "behaviour": most failures were
runners under-invoking, but two backends actually shipped with
mis-declared caps. The audit script alone could not detect those
because both directions of drift looked clean.
