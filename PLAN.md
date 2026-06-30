# CrispASR — Pending work

Pending roadmap items. Each is self-contained with files, approach, and
effort estimate. Completed items have been moved to `HISTORY.md`.

> **Numbering convention:** `§N` refers to PLAN items (sections in this
> file). `#N` refers to GitHub issues on CrispStrobe/CrispASR. They are
> independent sequences and numbers may collide. When in doubt, PLAN
> items are always written as `§N` and GitHub issues as `#N`.

**Latest release: v0.6.12** (commit `345ecfdc`). Full notes in [`RELEASE_NOTES_v0.6.12.md`](RELEASE_NOTES_v0.6.12.md).

## Gemma-4 12B (gemma4_unified) ASR support (OPEN)

The remaining open item for full 12B support (a new converter map + backend audio path for the 640-dim unified
encoder) is a larger port, scoped but not started.

## crispasr-diff harness extensions — catch decode-policy / quality bugs (OPEN)

Motivated by #197: the stage-cosine diff validates the forward pass against a
Python reference *pinned to a non-default mode* (`text_do_sample=False`) for
determinism, so it structurally cannot see (a) a production default diverging
from upstream, (b) decode-policy bugs (greedy looping / cut-off words), or (c)
perceptual pathologies. The cheap config-parity guard is **DONE** (a
defaults-audit unit test, `tests/test-tada-params.cpp`, asserting the C++
defaults equal upstream `InferenceOptions`). Three larger extensions remain:

1. **Per-step talker logits in the diff.** Dump the talker logits at each
   generation step in both the Python reference and the C++ runtime and compare
   them. Validates the text-decoder *input* so the sampler is a faithful port
   over verified logits — today only the FM/codec stages are diffed.
2. **Generation-health regression gate (non-diff).** A small en/fr/de suite
   asserting objective signals on the raw output: EOS reached within a bound, no
   token-loop (n-gram repeat rate under threshold), no excessive trailing
   low-energy frames (the #192 "trailing background noise" symptom), audio
   duration within a sane band of the text length. Catches perceptual
   pathologies cosine can't, without trusting noisy ASR roundtrips.
3. **Replay-token dual-mode reference.** Dump the Python *sampled* token ids and
   replay them in C++ (instead of re-sampling) so the sampling-enabled
   downstream stages can be diffed deterministically despite torch-vs-`mt19937`
   RNG mismatch — gives the sampling path a ground-truth diff, not just greedy.

Generalise the defaults-audit pattern across backends: a per-backend table of
(param → upstream default) checked against the params struct, so "a knob is
declared but dead / defaults diverge from upstream" fails CI everywhere.

> **Audit 2026-06-12** — code-verified all items against HISTORY.md and codebase.
> **Newly closed (stale in table until this audit):** #42 VibeVoice-ASR 7B
> (shipped with GGUFs + layer offload), #43 Fun-ASR-Nano (shipped 2026-05-20),
> #59 Cross-binding C-ABI parity (all 7 bindings 149/149 symbols, 2026-06-04),
> #155 CONV_TRANSPOSE_1D GPU (Vulkan+CUDA done, 2026-06-10),
> §WASM Browser build (all backends + HF Space, 2026-06-10).
> **Updated:** #52 O15 broken on CUDA; #56 diff-harness done, only JA kanji remains;
> #81 Nemotron un-deferred → HIGH for dictation.
> **Still open:** #52 perf pass, #51c F16 (RAM-blocked), #56 JA kanji
> (needs MeCab/KaKaSi), #58 MOSS (in progress), #75 server
> round 2, #66 wrapper publishing, O5/O6/O7 (O6 GPU-only, O7 needs draft
> models), #101 OmniVoice, #102 RapidTP.
> **Prior audit closed:** #96, #73 FA, #61j, #94, #93, #103, #100 A+B, O4, #115.

**Current state (May 2026, v0.6.11):** 20 ASR + 3 TTS + 1 speaker-verification backends (+ Chatterbox T3 in progress), unified CLI,
OpenAI-compatible server + WebSocket streaming, shared `src/core/` library, FireRedPunc
post-processor, C-ABI + Go/Java/Ruby/JS/Python/Dart bindings, CI on 6 platforms.
All backends support `-m auto --auto-download`. Three new ggml ops
(`conv_1d_cf`, `conv_1d_dw_cf`, `conv_1d_group`). ggml bumped to 0.10.0.
Feature matrix expanded to 21 backends (README). Speaker identification via
TitaNet embeddings + profile DB, integrated into diarize pipeline.
test-all-backends.py passes 18/18 transcribe + 51/54 feature tests (3 stream skips, no failures).

> **‼️ Tooling pin: `clang-format` MUST be v18.** CI pins it
> (`.github/workflows/lint.yml`). Homebrew's default `clang-format` and
> Xcode's bundled `clang-format` both ship v22, which silently
> re-wraps lines and breaks CI lint. Use `./tools/format.sh` (refuses
> non-v18) or the explicit path
> `/opt/homebrew/opt/llvm@18/bin/clang-format`. Never `clang-format`
> bare from `PATH`. See `CLAUDE.md` + `LEARNINGS.md` for the full
> lesson.

---

## #201 follow-up — generate a TADA voice ref from audio+transcript at query time (OPEN)

The switch-voice half of #201 shipped (commit `a5cd7510`): the server reloads a
prebaked `tada-ref-*.gguf` per request, no restart. The remaining half from the
reporter is **on-the-fly ref generation** — accept raw audio (+ transcript) at
inference time and bake the reference server-side, instead of requiring an
offline `--make-ref` pass first.

Today a `.wav` passed to `voice` is rejected (CLI adapter + session ABI both
`return -2`/warn) because the make-ref pipeline isn't loaded in the server.

**Approach:** the make-ref pipeline already exists in C++ (`src/tada_encoder.{h,cpp}`
+ wav2vec2 aligner runtime; CLI `--make-ref` drives it via wav2vec2 aligner → BPE
tokenization → DP alignment → WavEncoder → LocalAttentionEncoder → ref GGUF in
memory). To expose it at query time:
1. Load the encoder + aligner GGUFs in the TADA backend `init()` when configured
   (new flags, e.g. `--make-ref-encoder` / `--make-ref-aligner`, already parsed
   for the CLI path — reuse them). Keep them optional: only pay the ~1.3 GB
   (178 MB encoder + 1.1 GB aligner) when ref-baking is enabled.
2. In `synthesize()` / the server handler, when `voice` is a `.wav`, run the
   in-memory make-ref to produce prompt_values/positions and feed them straight
   into the context (skip the GGUF round-trip), or bake a temp ref GGUF and
   `tada_load_prompt` it. Requires a transcript — take it from a `ref_text`
   request field (the consent gate + `consent_attestation` already exist for
   `.wav` voices on `/v1/audio/speech`).
3. Cache the baked ref keyed by (audio hash, transcript) so repeat requests with
   the same clip don't re-run the aligner.

**Files:** `examples/cli/crispasr_backend_tada.cpp` (init load + synthesize bake),
`examples/cli/crispasr_server.cpp` (`ref_text` field + `.wav` voice path),
`src/tada_tts.{h,cpp}` (a `tada_make_ref_from_pcm` entry that returns prompt
state without a GGUF), `src/tada_encoder.*` (reuse). Aligner is language-specific
(`tada-aligner-<lang>.gguf`) — must match the audio language.

**Effort:** Medium-large. The pipeline is ported and CLI-proven; the work is
wiring it into the server lifecycle + per-request path + caching, plus the
~1.3 GB memory cost gate. Lower priority than the switch-voice half (shipped),
which already covers the common "I baked refs, let me pick one live" workflow.
Tracked on #201 (left open for this).

---

## §192 follow-up — native-Vulkan TADA garbled output FIXED (codec on CPU)

**FIXED 2026-06-29** (branch `fix/tada-vulkan-repeat-f16`). The garbled/empty
native-Vulkan output was **the codec, not the FM head** — the prior "FM
time-dimension divergence" localization was a red herring. Now: native Vulkan
"I went to school and back in four hours" (seed 1) → intelligible, **identical to
Metal**; +2 sentences +seed2 all ASR-round-trip cleanly; deterministic; CPU
output byte-identical to before (fix is Vulkan-gated). Still gated
(`CRISPASR_TADA_VULKAN_NATIVE=1`), default remains CPU-fallback — ask reporter
(BergmannAtmet, RADV) to confirm before flipping the default.

**Actual root cause (the FM premise was wrong; the codec is the culprit).** A/B'd
Metal vs Vulkan native on the same input: both produce **bit-identical** durations
(`time_before = [38,2,9,6,8,8,5,10,9,5,211]`) and 522 frames — the talker + FM +
duration decode AGREE across GPUs. Metal renders them intelligibly; Vulkan
rendered empty audio. So the divergence is purely in **audio rendering = the
codec** (`tada_codec.cpp`). Pinned down with per-stage `TADA_CODEC_DUMP` (Vulkan
vs CPU, identical 522-frame features): the **attention encoder MATCHES** across
backends (`dump_attn`, `dump_layer0` identical), but the **DAC decoder front-end
explodes** — `dump_dac_in` (the `in_conv` Conv1d → im2col+mul_mat) jumps to
rms ~37 / range ±800 on Vulkan vs rms ~0.85 / ±8 on CPU (~43×), propagating to the
output; the final `Tanh` masks the range but the audio is distorted. It is
**size-dependent**: short inputs ("Hello world") render *correctly* on Vulkan
(per-stage dump bit-comparable to CPU), and only break past some sequence length.

**NOT what earlier notes claimed (each verified, not assumed).** It is **not** a
`ggml_backend_sched` cross-backend-copy bug — `GGML_SCHED_DEBUG=2` shows the codec
runs as a *single Vulkan split*, no CPU offload, no cross-backend copies. It is
**not** missing Vulkan kernels — ggml-vulkan has `conv_transpose_1d`/`col2im_1d`/
`im2col`/`mul_mat`/`flash_attn_ext`, and the codec uses **no ISTFT** (DAC *conv*
decoder). It is **not** precision — F32 weights change nothing (MoltenVK
downconverts src0→f16); and the real in_conv weight is tiny (absmax 0.068), so no
overflow. It is **not even a conv/im2col kernel bug** — a standalone repro of
`conv_1d` / `im2col` / the full `wn_conv1d` (transpose+cont→conv→reshape+transpose)
at the codec's exact dims and T=522 is **bit-correct** on MoltenVK, and capping
`GGML_VK_FORCE_MAX_ALLOCATION_SIZE` doesn't help. So it's a **graph-scale
gallocr/aliasing-class corruption** in the large real codec graph — the in_conv
output is wrong despite a correct input (`dump_attn` matches) — that only bites at
length and isn't reproducible in a minimal harness. (Also disproven for the FM red
herring: F32 FM weights and whole-FM-on-CPU both left output bit-identical;
`time_before` already matched across backends.)

**Fix.** When the codec's GPU backend is Vulkan, run the whole codec on the CPU
backend (`src/tada_codec.cpp`, `tada_codec_init_from_file_impl`). The codec is a
one-shot decode (not the AR loop) and its input features are bit-identical
Metal-vs-Vulkan (Metal renders them correctly), so CPU rendering is faithful. The
talker/FM keep their native-Vulkan path. Opt back into the broken native codec
with `CRISPASR_TADA_CODEC_VULKAN_NATIVE=1` (debug only). **Open follow-up (to keep
the codec on GPU):** the op-level repros rule out the conv kernels, so the
remaining lever is the graph-scale corruption — best chased on RADV (real
hardware, where the talker/FM native path is also pending validation) with a
ggml-vulkan gallocr/graph dump, not MoltenVK. A **chunked codec decode** (decode in
time-windows under the breaking length, where short inputs are proven correct) is
the pragmatic GPU-native workaround and sidesteps the root cause entirely.

**Chunked-decode design — DEFERRED, low priority (not queued; design-only).**
Decision 2026-06-29: do **not** implement this now. The codec-on-CPU fix already
solves #192 (validated, shipped, default). The codec is a one-shot decode, *not*
the AR bottleneck, so a GPU-native codec buys little; the corruption is
MoltenVK-only locally while the reporter is on RADV (it may not even reproduce
there). Net: high implementation+validation complexity (window seams, absolute
RoPE bookkeeping, empirical sample-trim — the codec emits `n_frames*480 − ~6`
samples, non-exact) for a marginal, driver-specific payoff. Pick this up only if
a RADV user actually needs the codec on GPU. Design preserved below for whoever
does.

Add `tada_codec_decode` a chunked wrapper, gated (e.g. `CRISPASR_TADA_CODEC_CHUNK=<N>`,
default off; auto-on for the Vulkan-native codec). The codec upsamples 480×
(strides 4·4·5·6) and is feed-forward; the only cross-frame coupling is (a) the
block attention (each frame attends to its block + the previous block) and (b) the
conv receptive field across the 4 decoder blocks. So decode is chunkable with a
context margin:
  - Split the `n_frames` features into windows of `N` (start ~192, below the
    observed break; tune by bisecting the threshold between 93 ok and 522 broken).
  - For window `[a,b)`, run the existing `build_decode_graph` on the *extended*
    slice `[a-CTX, b+CTX)` (clamped), `CTX` ≈ 64 frames to cover both the conv
    receptive field and the prev-block attention reach.
  - **RoPE positions must be absolute**: set `codec_pos[i] = (a-CTX)+i`, not 0-based,
    so attention matches the full decode for the kept region.
  - Build the block mask from the *sliced* `token_masks[a-CTX : b+CTX]` (block ids
    are relative-safe — the mask only encodes same/prev block).
  - Keep only the PCM for `[a,b)`: trim `CTX*480` samples from each side (except at
    the true sequence start/end); concatenate windows.
  - Validate: ASR-roundtrip vs the CPU codec, and check for boundary clicks (peak
    discontinuity at window seams); widen `CTX` / add a short crossfade if needed.
Risk: a conv vocoder can click at seams; the CTX-trim should prevent it but needs
the ASR + waveform check. Keep CPU-codec as the default fallback until validated.

**Caveat — MoltenVK can't fully validate GPU numerics.** MoltenVK's `mul_mm` /
`mul_mat_vec` downconvert src0 to f16 regardless of stored dtype (storing F32 FM
weights changed nothing; `GGML_VK_DISABLE_F16=1` barely moved cond cosine,
0.99919→0.99966). So the talker `cond` is ~0.9997 vs CPU on MoltenVK — fine here
(durations/candidates still agree with Metal), but a reminder that exact CPU
parity isn't achievable on MoltenVK. On RADV (f32-native matmul) the talker is
more precise; the codec-on-CPU fix is the operative change regardless.

**Preconditions (hard-won — see LEARNINGS §192).** (1) Build with
`-DGGML_METAL=OFF` and verify the `backend=Vulkan0` log line — Metal silently
wins otherwise. (2) **Free the disk first** — benchmarking with `/Volumes/backups`
near-full gives SIGBUS + nondeterministic frame counts that masquerade as a
signal. (3) Re-run each config twice for determinism before forming a hypothesis.

---

## Priority ordering

| Priority | Item | Effort | Status |
|---|---|---|---|
| **HIGH** | [§176 Runtime optimization pass](#176-runtime-optimization-pass--2026-06-20-audit) | Phased | 20 sub-items (§176a–§176t). **16 DONE or MOSTLY DONE**: §176a (flash-attn via core_attn), §176b (bucket cache 8 backends), §176d (BLAS 9 backends), §176e (context cache all support runtimes), §176f (mel BLAS+OMP), §176g (embd cache 3 backends), §176h (F5-TTS fused graph), §176i (cross-KV F16, 5 backends), §176j (iterative FFT), §176m (nemotron memmove), §176o (embed fast path), §176p (MOSS flash), §176q (greedy alloc), §176r (beam top-K), §176s (encoder cache 16/17), §176t (weight pre-cache). **4 OPEN**: §176c (device-resident KV), §176k (FireRed KV), §176l (Kyutai RVQ), §176n (VoxCPM2 Metal). |
| **MEDIUM** | [#52 Qwen3-TTS](#52-qwen3-tts) — perf pass | Medium | talker + code_predictor + codec + ECAPA + codec_encoder all done; step-4 perf pass open (~137 ms/frame → real-time). **O15 broken on CUDA and default-OFF** (`61c42bfb`) — main perf lever disabled. **2026-06-13 Kaggle P100:** dedicated-sched fix (`baef21aa`) didn't help — O15=ON still rc=-6 SIGABRT at 6.0s. Crash is on the *first* code_pred call (not cached reuse), so root cause is `ggml_set_rows`-based KV scatter or the fixed-Lk causal mask on CUDA, not sched sharing. Baseline O15=OFF: 27.4 ms/frame, WAV OK. |
| **HIGH** | [#57 Commercial-friendly TTS expansion](#57-commercial-friendly-tts-backend-expansion) | Phased | Phases 1–3 + Turbo + native voice cloning shipped (→ HISTORY §82). **#83 S3Gen production fix LANDED** — UNet weight-residency split + `parallel=true` sched cache-coherency fix; M1 Metal diff cos_min 0.940→0.999976, intelligible at all T. **Remaining:** Kartoffelbox_Turbo DE. → see HISTORY + upstream-prs/09–11. |
| **MEDIUM** | [#51c MiMo-V2.5-ASR F16 step decode](#51c-f16-step-decode) | Small | F16 step-decode validation blocked behind ≥32 GB box (see PLAN #51c); base runtime + Q4_K shipped → HISTORY §56 |
| **DONE** | [#56 Kokoro multilingual phonemizer](#56-kokoro-multilingual-phonemizer-espeak-ng) | Small | espeak-ng + DE backbone shipped; Mandarin tone strip done; JA kanji g2p DONE (`1e7755e5`) — MeCab via dlopen (BSD-3-Clause, MIT-clean). → see HISTORY |
| **MOSTLY DONE** | [#58 MOSS-Audio-4B-Instruct](#58-moss-audio-4b-instruct) | Large | Runtime + GGUFs shipped, diff cos≥0.999, Kaggle P100 CUDA PASS. **Remaining:** flash-attn encoder, sweep transcript-extraction fix. → see HISTORY |
| **DONE** | [#59 Cross-binding C-ABI parity](#59-cross-binding-c-abi-parity) | Medium | **DONE 2026-06-04.** All 7 bindings at 100% C-ABI parity (149/149 symbols, `0b64a6d7` + `4835a241`). → see HISTORY |
| **DONE** | [#104 Stateful TDT frame-streaming](#104-stateful-frame-streaming-tdt-decode-for-parakeet-long-form-issue-89) | M-L | **DONE 2026-05-23.** Global z-norm + chunked encode + single decode → 99.5 % (was 59.7 %). **§216 2026-06-21: the streamed path actually COLLAPSES on v3/multilingual (5 min→75 w); non-JA now default to NeMo-exact single-pass (100 %) + silence-split longform; `streamed` kept as JA path + `CRISPASR_PARAKEET_*` gates.** → see HISTORY |
| **PARKED** | [#9 Parakeet TDT GPU](#9-parakeet-tdt-decoder-gpu) | Medium | Encoder 85%+ of time; LSTM+joint <0.7s; sequential steps limit GPU benefit |
| **DONE** | [#42 VibeVoice-ASR 7B](#42-vibevoice-asr-7b) | High | **DONE.** GGUFs at `cstr/VibeVoice-7B-GGUF` (Q3_K 4.7 GB – F16 17.4 GB). → see HISTORY |
| **DONE** | [#43 Fun-ASR-Nano](#43-fun-asr-nano) | Medium | **DONE 2026-05-20.** Full LLM-decoder runtime shipped; GGUFs at `cstr/funasr-{nano,mlt-nano}-GGUF`; byte-identical diffs; ~9× RT on M1 Metal. → see HISTORY |
| **DONE** | [#80 nano-cohere-transcribe-inspired tweaks](#80-nano-cohere-transcribe-inspired-perf--chunking-tweaks) | Small | 80a parked; **80b DONE**; **80c DONE**; **80d DONE** 2026-05-23 (audit: no fixes needed — all backends use energy chunker); 80e low-priority warmup deferred → see HISTORY |
| **DONE** | [#81 Nemotron-Speech-Streaming-EN-0.6B](#81-nemotron-speech-streaming-en-06b--first-cache-aware-streaming-native-asr) | M-L | **DONE 2026-06-15.** All 4 streaming presets working. → see HISTORY |
| **DONE** | [#157 Add streaming capabilities for other runtimes](#157-add-streaming-capabilities-for-other-runtimes) | — | Granite + Voxtral4b done 2026-06-19; greedy_decode.h got dual-hook run_with_probs_cb. → see HISTORY |
| **MOSTLY DONE** | [§221 TADA encoder `--make-ref`](#221-tada-encoder---make-ref) | Medium | C++ encoder runtime + GGUF converters + diff harness shipped. GGUFs at `cstr/tada-encoder-GGUF`. **WIP:** cos_mean=0.94 parity (F16 precision), C++ BPE tokenizer for end-to-end `--make-ref`. → see HISTORY §221 |
| **DONE** | [#172 Wyoming protocol server — Home Assistant Assist](#172-wyoming-protocol-server--home-assistant-assist) | Small | **DONE 2026-06-19.** `--wyoming-port N` starts a Wyoming JSONL/TCP server alongside HTTP. → see HISTORY |
| **DONE** | [#173 `--tts-play` / `--tts-play-device`](#173-tts-play--tts-play-device--local-speaker-output) | Small | **DONE 2026-06-19.** `crispasr_speaker.{h,cpp}` wraps miniaudio playback; pre-resamples to hardware-native rate (avoids 4× upsampler artefacts). → see HISTORY |
| **DONE** | [#158 transcribe_streaming for opaque-C-library backends](#158-transcribe_streaming-for-opaque-c-library-backends) | Medium | **DONE 2026-06-19.** All 7 backends done: `_transcribe_cb` C entry points added to moss_audio, gemma4_e2b, moonshine_streaming, kyutai_stt, mimo_asr, nemotron; GLM-ASR uses exported step APIs directly. → see HISTORY |
| **DONE** | [#86 Per-backend flash-attention wiring](#86-per-backend-flash-attention-wiring-crisperweaver-driven) | — | All backends now route through core helpers (`core_attn`, `core_sanm`, `core_conformer`) that unconditionally use `ggml_flash_attn_ext`. → see HISTORY |
| **LOW** | [#87 `gpu_backend` runtime selector](#87-gpu_backend-runtime-selector-multi-backend-ggml-build) | ~1 week | Needs ggml-side multi-backend dispatch to land first. CrisperWeaver UI placeholder ready when the C-side is. |
| **LOW** | [#95 IndexTTS Chinese TN binary alternative](#95-indextts-15-chinese-tn--binary-alternative-to-the-python-wetext-hook) | survey only | Python `INDEXTTS_TEXT_NORMALIZER` hook shipped 2026-05-19. Hand-roll (#95a) is the right next step *when* a user reports a digit/date prompt that breaks; OpenFST vendoring (#95b) only after #95a grows past ~5 cases. |
| **IN PROGRESS** | [#97 More Parakeet variants](#97-more-parakeet-variants) | Small per-variant | TDT/TDT+CTC DONE; **parakeet-rnnt 0.6b+1.1b DONE 2026-05-24**. **parakeet-unified-en-0.6b surveyed 2026-06-13:** Unified-FastConformer-RNNT (24L, 600M), jointly trained offline+streaming with shared params. NOT converter-only — 8× subsampling (vs 4×) + Dynamic Chunked Convolutions are new. ~80% overlap with #81 nemotron streaming work. Offline mode may work through existing converter (standard bidir FC + RNNT). realtime-EOU blocked on #81 cache-aware streaming. |
| **DONE** | [#98 Hotwords / contextual biasing](#98-hotwords--contextual-biasing) | Phased | **Phase A+B DONE.** CTC-WS Aho-Corasick trie wired into parakeet CTC + TDT; LLM prompt injection for qwen3-asr + voxtral. → see HISTORY |
| **DONE** | [#110 Global diarization timeline](#110-global-diarization-timeline) | Medium | Sherpa/ecapa now runs once on the full audio (not per-slice). → see HISTORY |
| **LOW** | [#106 TEN-VAD](#106-ten-vad--low-latency-cross-platform-vad) | Small | Technically feasible VAD backend: C-compatible, 16 kHz / 10-16 ms frames, prebuilt libs + ONNX path. License is the gate: Apache 2.0 plus extra no-compete / own-app-only conditions from Agora. |
| **MOSTLY DONE** | [#114 Long-form transcribe — chunking-default ladder for voxtral / cohere / canary](#114-long-form-transcribe--make-chunkingstreamed-the-default-for-all-asr-backends-issue-89-follow-up) | Medium | Chunking/streamed-default shipped for all ASR backends (parakeet/voxtral/cohere/canary/gemma4/mimo); per-chunk AED re-injection + LCS dedup + word-snap. **Remaining:** EN-FLEURS retokenization artifacts (out of scope). → see HISTORY |
| **DONE** | [#105 WhisperX word alignment models](#105-whisperx-word-alignment-models-wav2vec2-ctc-zoo) | Phased | **DONE 2026-05-23.** All 10 WhisperX common languages (fr/es/it/ja/zh/nl/uk/pt/ar/cs) converted, uploaded to `cstr/*-GGUF`, registry aliases wired. → see HISTORY |
| **DONE** | [#115 mimo-asr baseline broken](#115-mimo-asr-baseline-broken-silent-empty-on-short-segfault-on-long) | Small-Medium | **GPU is the default** (Option B: split-load + prefill-graph decode). → see HISTORY |
| **MOSTLY DONE** | [#125 Issue #125 — multi-backend bug sweep from montvid](#125-issue-125--multi-backend-bug-sweep-from-montvid-12-findings) | Medium | 12 montvid findings; P1–P6b all DONE (`f72d3db1`/`72b74486`/`5f0aefc0`/`8bfaff23`/`b936b488`/`ba0e388e`/`043b3ae5`). **Remaining:** P0 Blackwell retest; mimo-asr `-np` empty-transcript retest. → see HISTORY |
| **DONE (no-flip)** | §208 Chatterbox S3Gen CFM single-GPU raw-gallocr cached path | Medium | **DONE 2026-06-21 — correct, but a perf DUD, default OFF.** `CRISPASR_S3GEN_UNET_GALLOCR=1` runs the batch-2 CFG UNet on one GPU backend via raw `ggml_gallocr`, caching graph+alloc across all Euler steps (legal because `ggml_gallocr_alloc_graph` doesn't mutate the graph, unlike the sched). Parity vs legacy: step-wise `x_rms` match (Δ≤1e-4), spectral corr 0.999105, identical ASR; Bug B does not recur (no cross-backend copy). **No speedup:** instrumented host build+`sched_alloc` = ~4–7 ms/step of ~1887 ms/step (0.3%) — the CFM is compute-bound (Metal GEMM), not overhead-bound as the §207 handover assumed. Shipped env-gated, default OFF (don't flip without a real speedup); kept as the clean single-backend reference + bisection gate. → HISTORY §208, LEARNINGS §208. |
| **DONE** | §205 Chatterbox full-sweep CUDA crash + q8/Metal CFM NaN | Medium | **DONE 2026-06-21.** Kaggle P100 full-backend-sweep crashed chatterbox with `free(): corrupted unsorted chunks` → non-power-of-two FFT heap overflow in the VE/S3Tok/CAMPPlus/CosyVoice3 mel (`fft_radix2_wrapper` on N=400/1920); fixed with a mixed-radix DFT (pow2 callers bit-identical). Then the q8 s3gen CFM NaN'd on M1 Metal — properly diagnosed (NOT "compound F16 accumulation"): Metal's q8 mat-vec kernel requantises the CFM activations to q8, single-pass NaN at Euler step 1 on the batch=2 CFG graph. Fix: dequantise the quantized CFM (`s3.fd.*`) weights to F16 at load, GPU-resident (correct `mul_mm_f16_f32_hp` path, full GPU speed; `CRISPASR_S3GEN_UNET_CPU=1` for the slower CPU route). ASan-clean + ASR roundtrip OK; quality identical to CPU route (`conv_pre rms` 3.380 vs 3.379). **Open:** Kaggle CUDA re-run to confirm q8 chatterbox now passes the sweep (the Metal CFM handling is M1-only; the FFT fix alone should suffice on CUDA). → see HISTORY §205, LEARNINGS §205. |
| **DONE** | [§130 Zonos TTS](#130-zonos-tts--transformer--dac-codec-apache-20) | Medium | **DONE 2026-06-09.** End-to-end synthesis + ASR roundtrip verified. → see HISTORY |
| **DONE** | [§131 OuteTTS](#131-outetts--llm--wavtokenizer-codec-cc-by-40) | S-M | **WORKING — speech output confirmed via ASR roundtrip.** WavTokenizer decoder validated cos≥0.999 all stages. → see HISTORY |
| **DONE** | [§139 Beam search — remaining ASR backends](#139-beam-search--remaining-asr-backends-issue-136-follow-up) | Phased | **18/24 done** (was 10). All feasible backends shipped 2026-06-01/02. → see HISTORY |
| **DONE** | [#156 Permissive G2P phonemizer (replace espeak-ng GPL dep)](#156-permissive-g2p-phonemizer) | Phased | **DONE 2026-06-08**: Pre-generated IPA pronunciation dicts (EN 126K, DE 667K, FR 257K, ES 600K) at cstr/g2p-dicts — 99.5% piper-compatible. → see HISTORY |
| **DONE** | [#155 CONV_TRANSPOSE_1D GPU optimization](#155-conv_transpose_1d-gpu-optimization-issue-155) | Small | **DONE 2026-06-10.** Crash fixed (`f8fc8b8e`); CUDA/HIP 9× speedup (1200→130 ms, `5f600f25`); Vulkan `col2im_1d` kernel ported (`cad7fbac`) — codec stays fully on-GPU. → see HISTORY |
| **DONE** | [§WASM Browser build](#wasm-browser-build--all-backends-multithreaded) | Medium | **DONE 2026-06-10.** All backends, multithreaded (`-pthread` + `PTHREAD_POOL_SIZE=8`). → see HISTORY |
| **LOW** | [#127 Coverage gaps from 2026-05-26 sweep close-out](#127-coverage-gaps-from-the-2026-05-26-overlap-save-sweep-close-out) | Small | Three loose ends: **(a) omniasr-llm DONE** — Kaggle P100 sweep v4: short=106 long=490 (4.6x scaling), correct JFK transcript, no timeout (was M1-only issue). **(b)** mimo-asr local test doesn't run in CI (4.2 GB Q4_K doesn't fit runner disk). **(c) cohere-asr-ja DONE** — correct repo is `CKHO/cohere-asr-ja-GGUF` (not `cstr/`); Kaggle P100: rc=0, 108 chars, perfect JFK: "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country." Still needs JA fixture sweep for PERFORMANCE.md table. |

**Recently completed:** many items shipped (parakeet streamed-default, #81 FA per-head mask, #110 diarization, #98 hotwords, paraformer-zh, SenseVoice, Fun-ASR-Nano, …) — full write-ups in HISTORY.md.

**Open follow-ups from §79 — we want all of these:**
- **#73 cohere long-form rerun.** flash_attn_ext is shipped on canary + cohere (commit 193a736). JFK (~11 s) numbers: canary q8_0/q4_0 -17 % under flash (win), but cohere q8_0/q4_0 is +11 % under flash vs cast-on-read on the same workload. F16 is a tie on both. Before promoting flash as cohere's recommended path, validate on a multi-minute clip — if the crossover is workload-dependent the docs need to recommend cast-on-read for short audio and flash for long. Until then PERFORMANCE.md notes flash as available-but-regresses-on-JFK for cohere.
- **#72 Linux/CUDA validation — DONE.** **2026-06-13 Kaggle P100: gemma4-e2b GPU=5.1s vs CPU=10.8s → 2.12× speedup** (JFK 11s Q4_K, sweep v2 `baef21aa`). Confirms Metal wins translate to CUDA with even larger margin. mimo-asr CUDA also PASS (rc=0, 1.1s JFK + long audio OK).
- **encoder-decoder #69a** (canary, cohere, kyutai-stt). Cross-attention layout has no `<prefix><N>.*` block-tagged tensors; needs bespoke per-backend predicates. Own design problem.

**Issue #81 A1000 work — Phase 1 verdict in (2026-05-23):**
- `d758fe69` (fused `GGML_OP_NORM_AFFINE` + `GGML_GLU_OP_SIGLU` for FastConformer encoder) closes target (b) of the gap analysis. Measured **+5.7 % wallclock win** on A1000 Laptop (2.701 s vs 2.863 s baseline, p50/chunk 175.8 ms vs 184.8 ms, RTx 22.2× vs 21.0× — clean WDDM-warm conditions on Studio Driver 596.36). Sched-debug: CPU splits 144→72, UNARY-on-CPU 72→0. **Carry as permanent improvement.** Full write-up in PERFORMANCE.md "Phase 1 update (2026-05-23)" subsection. WIP branch `issue81-phase1-uar-wip` (commits `6a0ccc67 / a2999cf3 / 6d7872a0`) is superseded — delete when convenient.
- **#06 FA per-head mask** is the next concrete A1000 perf step. Removes the other 72 CPU splits per chunk (per-head additive mask in `fattn.cu:423` + the four kernel variants). Scoped at 2-3 days, ~300-500 LOC across `fattn.cu` / `fattn-common.cuh` / `fattn-mma-f16.cuh` (and optionally `-wmma-f16.cu` / `-tile.cu` / `-vec.cuh`). Expected wallclock gain ~10-15 % on top of postsiglu (target ~2.4 s long-clip / RTx ~25× / ~1.5× behind onnx-fp32). Don't start until WDDM-warm bench protocol below is followed for the new baseline.
- **WDDM warm-up protocol** (Windows/laptop NVIDIA only): cold A1000 sits at P5/P8/210-510 MHz during compute and runs 8-10× slower than warm; engage WDDM by running `bench-issue81/probe_postsiglu_leak.py <dll> 200` (or ~10 s of `gpu_keepalive.py`) BEFORE measuring. The 3.063 s May 11 reference is reproducible with this protocol; single-shot cold benches are noise. Documented in PERFORMANCE.md "What we learned about A1000 WDDM behavior" + LEARNINGS.md "WDDM idle-clock-state hysteresis on consumer/laptop NVIDIA SKUs".

**A1000 2026-06-12 follow-ups — DONE:** firered AED beam batching (4.48×, `88ad4b9d`) + parakeet-ja auto-VAD (`f950bd1e`). → see HISTORY.

---

## §214 follow-up — chatterbox T3 batched-CFG (B=2) — deliver the win (OPEN)

§214 shipped the batched-CFG B=2 T3 decode (`CRISPASR_CHATTERBOX_T3_CFG_B2=1`,
default OFF) — greedy-token bit-identical to legacy on CPU (all quants), GPU+F16,
and GPU+quant (via F16 dequant). It works; what's left is *delivering* the
speedup and proving it. Files: `src/chatterbox.cpp` (`build_graph_t3_kv_b2`,
`run_t3_kv_b2`, `ensure_t3_b2_f16_weights`, decode loop ~§214). See HISTORY +
PERFORMANCE §214.

1. **Quiet-machine A/B + default-flip decision (HIGH).** The CPU floor ~34 %
   (75→50 ms/tok) was measured on a contended M1 (load 8–12) — unreliable.
   Re-measure on a quiet host (alternating order, min-of-N, deterministic
   token-parity gate `CRISPASR_CHATTERBOX_TEMP=0`). If the win holds, propose
   flipping the default ON for the CFG path (keep the env + legacy path forever
   for bisection). ~1 h on a quiet box.

2. **Cached / bucketed B=2 step graph — CLOSED, confirmed DUD (measured 2026-06-21).**
   `CHATTERBOX_BENCH_B2=1` split each step into build+alloc vs compute: **CPU
   0.41 ms/step build+alloc = 0.8 %** (compute 53 ms/step); **GPU+F16 1.55 ms =
   0.7 %** (compute 232 ms/step). Per-step rebuild+alloc is <1 % on both backends
   — bucketing it could save at most ~1 %, exactly the §208 lesson. NOT worth it,
   and a cached B2 graph would risk reintroducing the §186 Lk-bucket `buffer is
   nil` Metal crash that per-step rebuild currently sidesteps. Don't pursue. (Side
   datum: GPU per-step compute 232 ms ≫ CPU 53 ms, so B2 on GPU is *not* a speed
   win over the CPU default — its value is enabling T3-on-GPU at all + the
   GPU+quant F16-dequant path, not beating CPU.)

3. **GPU speedup (LOW — first data says GPU loses to CPU).** First measurement
   (2026-06-21, residual load): GPU+F16 B2 compute **232 ms/step** vs CPU **53
   ms/step** — GPU stays ~4× slower per step (the long-standing Metal
   kernel-launch-overhead × 30L × steps story; B2 doesn't change it). So GPU B2's
   value is *enabling* T3-on-GPU (sidesteps the §186 bucket crash) + the GPU+quant
   F16-dequant path, NOT speed. A quiet-box re-measure could refine the 4× but is
   unlikely to flip it; the production T3 default stays CPU.

4. **Generalize B=2 to the other CFG backends — see §215.**


## §215 — batched-CFG (B=2) for the remaining TTS backends (OPEN)

Full-tree audit (§214, see PERFORMANCE.md): only **s3gen CFM** and **chatterbox
T3** batch cond+uncond into one B=2 forward; every other CFG backend runs **two
sequential B=1 passes** and blends post-hoc — correct, but it reads the full
weight set twice and doubles the dispatch count per step. Where the per-step
forward is dispatch/bandwidth-bound *and* the step count is high, B=2 reads each
weight once (gianni measured −42 % on T3). Apply the **same pattern** to each:
batch over `ne[2]=2` for the heavy GEMMs, split any per-batch attention/KV-cache,
tag `GGML_PREC_F32`, gate behind an env (default OFF), validate with a greedy
token/output-parity gate vs the legacy sequential path, and on **GPU + quantized
weights dequant the batched-against weights q*→F16 GPU-resident** (the s3gen
`dequant_cfm_f16` / T3 `ensure_t3_b2_f16_weights` trick — Metal's batched `ne[2]=2`
quant mat-vec misses the PREC_F32 `mul_mv_q*_K` exact-dot kernel and degenerates).

Prioritized by expected payoff (high step count × dispatch-bound first):

1. **§215a dia (HIGH).** `src/dia_tts.cpp run_dia_synth` already builds a B=2
   *encoder*; the **decoder** AR loop still runs cond/uncond as two passes with
   two KV caches (`run_dia_decode_step`). Mirror chatterbox T3: B=2 decode-step
   graph, split per-batch KV write/read, F16-dequant on GPU+quant. Largest payoff
   — long AR token loop with CFG every step.
2. **§215b tada (HIGH).** `src/tada_tts.cpp` runs the talker twice per step
   (`run_talker_kv` for pos + `kv_neg_*` for neg, ~lines 1286–1313). Lockstep
   `n_past` like T3 → reuse both caches, batch the GEMMs, split attention. Same
   shape as chatterbox; should port almost directly.
3. **§215c zonos (MED).** `src/zonos_tts.cpp` keeps two separate KV caches
   (`kv_k`/`kv_k_uncond`) and decodes sequentially (~lines 1740–1842). Batch the
   AR backbone B=2, split the dual-KV attention. Dual-KV CFG + (optional) random
   speaker embed must stay independent per batch.
4. **§215d voxcpm2 (MED).** `src/voxcpm2_tts.cpp` LocDiT runs `locdit_call` twice
   per ODE step (cond `mu` + zero-`mu`, ~lines 2752–2778). Diffusion (fewer steps
   than an AR loop) so lower payoff, but each LocDiT forward is heavy; B=2 over the
   DiT blocks could still help. Keep the cfg-zero-star blend per batch.
5. **§215e f5 (MED).** `src/f5_tts.cpp` runs `dit_forward` twice per CFG step
   (v_cond + v_uncond, ~lines 1563–1566). Same B=2-DiT shape as voxcpm2. Note
   §176h found a standalone B=2 DiT graph for F5 *correct* but a runtime B=2
   corrupted batch-1 (F5-runtime-specific, see [[project_ggml_batched_fused_graph_alloc_bug]])
   — so F5 needs the parity gate run especially carefully; may not be worth it.
6. **§215f cosyvoice3 (LOW — likely WON'T).** `src/cosyvoice3_tts.cpp` explicitly
   declined batching (~lines 3027–3030): the estimator is a 22-block diffusion run
   twice per step, per-call overhead small vs the forward. Only revisit if a
   profile shows the dispatch overhead actually matters; otherwise document as a
   deliberate non-goal.
7. **§215g kugelaudio (BLOCKED).** `src/kugelaudio.cpp` CFG is a TODO (cfg_scale
   read but unused, negative path not implemented). Implement sequential CFG first
   (correctness), then consider B=2.

Shared infra: factor the F16-dequant-of-matmul-weights helper (currently
duplicated in `ensure_t3_b2_f16_weights` + s3gen `dequant_cfm_f16`) into a
`core_*` helper if a third backend needs it — but only on the third consumer, per
the "don't extract single-consumer helpers" rule.

---

## §210 follow-up — shape-stable bucketed decode for the remaining LLM/AR backends (CUDA-graph capture) (OPEN, CONDITIONAL)

PR #207 made **granite-speech**'s single-token LLM decode a *cached, shape-stable*
graph (fixed-`Lk` KV bucket written via `ggml_set_rows` at a runtime index, in-graph
`ggml_argmax`, fused F16 embed), which unlocks **two** wins:
- **CUDA-graph capture** — the ~1.4k-op step replays as one `cudaGraphLaunch`,
  ~9–13× on the decode (RTX 5090). Engages **automatically** in ggml-cuda for any
  capturable, shape+pointer-stable graph routed through the sched — no per-backend
  CUDA code.
- **Metal gallocr allocate-once** (`granite_dec_use_gallocr`) — reserve one
  `ggml_gallocr` against the cached graph, reuse every step (skip per-step
  `sched_reset`+`sched_alloc_graph`).

The question: which other LLM/AR-decoder backends would benefit? Survey done
2026-06-30 (see [[LEARNINGS]] §210; the gating property is shape-stability —
`fixed-KV bucket` vs growing `Lk = n_past + T`).

### ⚠️ READ FIRST — the honest cost/benefit (don't mass-port)

1. **The headline (CUDA-graph) win is gated to Ampere+ (sm_80+)** — see
   `ggml/src/ggml-cuda/ggml-cuda.cu:4329` (`< GGML_CUDA_CC_AMPERE` →
   `disable_due_to_gpu_arch`). The project's usual CUDA test GPUs — **T4 (sm_75),
   P100 (sm_60) — are gated OUT.** Only RTX 5090 / A100-class actually benefit.
2. **On Metal there is NO throughput win.** §210 measured the M1 granite decode at
   host-encode **1.8 %** / GPU **98 %** — it's GPU-bound (weight-bandwidth-bound
   Q4_K GEMVs), so the shape-stable rewrite buys only memory-pressure robustness
   (alloc balloon 68–236 ms → ~0) on Metal, not speed. The real Metal/Apple-Silicon
   lever is GPU-side (kernel fusion, lower-bandwidth quant), not graph replay.
3. **Each port is a MANUAL graph rewrite + byte-identical diff-harness validation**
   — not mechanical, and **NOT delegable to agents** (runtime graph code, per
   [[feedback_no_agents_for_runtime_graphs]]). Budget one focused session per
   backend.

**Therefore: port a backend ONLY when it's actually deployed on Ampere+ CUDA
servers, and measure first** (`CRISPASR_METAL_PROFILE` for the host/GPU split;
confirm the GGML_LOG_DEBUG "disabling CUDA graphs due to …" line does NOT fire on
an Ampere+ GPU). For smaller LLM decoders the host-encode fraction may be larger
than granite's 1.8 %, which would change the calculus — so don't assume, measure.

### Templates (copy these — already shape-stable)

- **`src/granite_speech.cpp`**: `granite_build_argmax_decode` (~L2474, fixed
  `bucket_len`, in-graph argmax, fused F16 embed when `token_embd` non-quant),
  `granite_dec_use_gallocr` (~L2362, gates gallocr vs sched: gallocr on Metal/CPU,
  **sched on CUDA so capture still fires**).
- **`src/mimo_asr.cpp`**: did it independently — cached `step_t1_gf` +
  `step_t1_fixed_kv_len` (L211), `set_rows` scatter-write at runtime `kv_indices`
  (L958, L1026), skip-plan reuse "update inputs only, no reset/alloc"
  (L1507–1522). Good second reference for the set_rows path.

### Already done (no work): `granite_speech`, `mimo_asr`, `dots_tts` (Metal gallocr, `ec74c5a0`).

### Candidates — growing-shape (`Lk = n_past + T`) + naive per-step rebuild + `sched_reset`/`alloc`

ASR-LLM (prioritized by likely server deployment):
1. **voxtral** — `src/voxtral.cpp`, `Lk` at L1019; per-step graph rebuild +
   `sched_reset`+`sched_alloc_graph` at L1159/L1192/L1244.
2. **qwen3_asr** — `src/qwen3_asr.cpp`, `Lk` at L1210.
3. **voxtral4b** — `src/voxtral4b.cpp`, `Lk` at L1411 (decode via
   `core_greedy_decode`).
4. **gemma4_e2b** — `src/gemma4_e2b.cpp`, `Lk` at L995 (decode via
   `core_greedy_decode::run_with_probs_cb`, ~L1631).
5. **glm_asr** — `src/glm_asr.cpp`, `Lk` at L1322.
6. **higgs_stt** — `src/higgs_stt.cpp`, `Lk` at L1061 (Qwen3-1.7B decoder).
7. **ark_asr** — `src/ark_asr.cpp`, `ark_build_decoder_graph` (L643) /
   `ark_run_decoder` (L729), `Lk` at L674 (Qwen2.5-3B decoder).
8. **lfm2_audio** — `src/lfm2_audio.cpp`, `Lk` at L933; already on a gallocr (L875,
   the §206 weight-less-first-op fix) but growing-shape, so not yet capturable.

TTS AR decoders (LOWER priority — heavy per-step compute = even more GPU-bound, so
the Metal payoff is ~nil and even the CUDA payoff is diluted by larger per-step
GPU time): `csm_tts`, `indextts`, `bark_tts`, `moss_audio`, `mini_omni2` are naive
growing-shape. `qwen3_tts`, `chatterbox` (T3), `tada_tts`, `parler_tts`,
`vibevoice` already have per-backend perf work (`set_rows`/gallocr present) — audit
individually before touching.

### Per-backend recipe (mirror granite)

1. Allocate KV at `kv_max_ctx` once; pick a fixed `bucket_len` (≤ cache cap).
2. Rewrite the step graph to a fixed `[0, bucket_len)` KV view; write the new
   token's K/V via `ggml_set_rows` at runtime index `n_past` (not a growing
   `ggml_view` sized to `n_past`). Mask is `(bucket_len, 1)`, set host-side each
   step. Topology must be byte-identical across steps.
3. Move argmax in-graph (`ggml_argmax`); keep `logits` as a graph output for
   callers needing the vocab.
4. Make the embed **capturable**: if `token_embd` is k-quant, the in-graph
   `GET_ROWS` host-syncs and disables capture — either fuse a F16 embed (granite)
   or pass a pre-computed F32 embed input (granite's `fused_embed` branch).
5. Cache the cgraph; gate gallocr-vs-sched like `granite_dec_use_gallocr` (gallocr
   on Metal/CPU, sched on CUDA/HIP so capture engages; force sched if a CPU layer
   split exists). Add an env opt-out.
6. Bound `n_past < bucket_len` (granite's OOB guard, `c5035969`).

### Validation gate (mandatory, per [[feedback_methodology]])

- **Byte-identical transcript** vs the legacy path on jfk + fleurs_60s (the granite
  gate). This is the correctness bar — no merge without it.
- **Measure** before/after: `CRISPASR_METAL_PROFILE` (host vs GPU µs split) and a
  per-step compute-µs accumulator like `CRISPASR_GRANITE_DEC_PROFILE`. On a real
  Ampere+ GPU, confirm capture engages (no "disabling CUDA graphs" debug line) and
  A/B the decode RTFx.
- M1 wall time is noise (§210) — gate on the instrumented per-step quantity, not
  end-to-end wall.

**Effort:** ~1 focused session per backend (graph rewrite + diff-harness parity +
profile). Do the highest-deployment ASR backend first; stop if its measured CUDA
A/B on Ampere+ doesn't justify the next.

---

## §219 — more permissive audio input formats for crispasr_audio_load (OPEN)

Done so far (ffmpeg-free): WAV/MP3/FLAC (miniaudio), Ogg Vorbis (stb_vorbis),
Opus (libopus/opusfile), AIFF/W64/RF64 (miniaudio dr_wav, free), and AAC/M4A/
ALAC/CAF on **Apple** via AudioToolbox `ExtAudioFile`. Remaining, by value:

1. **AAC/M4A on Windows + Android (native, HIGH).** Mirror the Apple AudioToolbox
   fallback with Media Foundation (`IMFSourceReader`, Windows) and NDK
   `MediaExtractor`+`MediaCodec` (Android) — free OS decoders, no ffmpeg/GPL.
   Same `crispasr_audio.cpp` fallback hook, platform-`#if`'d. Linux still has no
   permissive AAC decoder (fdk-aac is non-OSI/patent-disclaimed) → optional
   ffmpeg dynamic fallback only.
2. **AMR-NB/WB (MED).** opencore-amr (Apache-2.0) — telephony/voicemail speech.
   Direct decode (simple `#!AMR` framing) → ma_resampler to 16 k. pkg-config +
   FetchContent-static (autotools upstream → compile the amrnb/amrwb sources).
3. **WebM/Matroska Opus|Vorbis (MED).** libwebm (BSD-3) demux → feed the already-
   linked libopus / stb_vorbis. Web/browser audio.
4. **Speex / WavPack (LOW).** BSD libs; Speex is obsolete (Opus superseded it),
   WavPack niche. Only if a corpus needs them.
5. **AU / Sun .snd (LOW).** Tiny PCM/µ-law parser; cheap if requested.

Pattern is fixed: permissive decoder (or OS-native) behind the miniaudio custom
backend / fallback hook, CMake-gated (pkg-config primary + FetchContent static
fallback for no-system-lib platforms), greedy decode → 16 k. ffmpeg stays an
optional dynamic fallback only for what none of the above covers (AAC-on-Linux,
WMA, AC-3/DTS, .ape).

---

## §166 follow-up — WASM `asr*` session surface needs a build-verify (OPEN)

Round 4 (2026-06-13, see HISTORY) added a backend-agnostic ASR session surface to
the WASM/JS binding (`bindings/javascript/emscripten.cpp`:
`asrOpen`/`asrTranscribe`/`asrSet*`) — the WASM ASR path was whisper-only
(`init`/`full_default`) before. Verified by inspection (all C-ABI decls present;
mirrors the existing `tts*` Embind patterns) but **not built locally**: emsdk
won't resolve a fetchable arm64-mac SDK on this dev box (every version errors at
manifest resolution; no Homebrew emscripten either). **Open:** build via
`build-wasm.sh` / the WASM CI and smoke-test `asrTranscribe` in node/browser once
an emcc toolchain is available. The §166 native-wrapper + server + Node-addon
parity is DONE (HISTORY).

---

## §175 Surgical DRY — share pure helpers across the CLI/library boundary (MOSTLY DONE)

**Context.** The `CrispasrBackend` adapter layer (`examples/cli/`) and the
session C ABI (`src/crispasr_c_api.cpp`, the dylib every binding + the HTTP
server loads) are two **separate** implementations of each backend's
transcribe/synthesize. This is structural: `CrispasrBackend` and
`whisper_params` live in `examples/cli/` and `libcrispasr` (`src/`) links
nothing from there, so the session ABI *cannot* call the adapters — it
reimplements inline. contributing.md §6 already institutionalises this as a
"9 edit points" checklist. See memory `project-session-abi-reimplements-cli`.

**Scope decision (2026-06-19).** Do **not** unify the dispatch layer (hoisting
`CrispasrBackend`+`whisper_params` into `src/` and routing both consumers
through it touches all ~25 backends, the two impls have diverged — sticky
fields, voice-key caching, server error strings, best-of-N, hotwords,
punctuation toggles — and regresses CLI + 7 bindings + server; cannot promise
"no breaking changes" without a multi-week test matrix). Instead extract the
small, **pure, copy-pasted** helpers that cross the boundary cleanly as
header-only `inline` functions in `src/core/` that both sides include (no link
dependency across the boundary).

**Gate every item on before/after regression** — bit-identical output required:
unit test the pure helper + run the Kaggle `lang-spec-sweep`, `crispasr-diff`
stage harness, and `tests/regression/manifest.json` before and after.

Candidates (audited 2026-06-19):

1. **ISO-639-1 → English language-name map — DONE.** `src/core/lang_names.h`
   header-only `inline core_lang::iso_to_english()`. All 4 former copies now
   delegate: `crispasr_backend_utils.h` (`crispasr_iso_to_english_lang`),
   `crispasr_c_api.cpp` (`ca_iso_to_english_lang`), `gemma4_e2b.cpp`
   (`g4e_lang_name`). Unit test: `tests/test-core-lang-names.cpp`.

2. **GPT-2 byte-level BPE decoder — DONE (variant a).** 6 copies of the
   full 256-byte `byte_decoder()` + `decode_token()` / `gpt2_byte_decode()`
   now delegate to `core_bpe::token_bytes_to_utf8()` (already in
   `src/core/bpe.h`): `crispasr_c_api.cpp`, `lfm2_audio.cpp`,
   `vibevoice.cpp`, `crispasr_backend_{qwen3,moss_audio,mimo_asr}.cpp`.
   Unit test: `tests/test-core-bpe.cpp`. Variant (b) — the simpler
   `Ġ→space, Ċ→newline` inline replacements in glm_asr / c_api — is
   2 call sites, not worth a helper.

3. **Language-instruction prompt templates.** "Transcribe the speech in
   <lang>." now duplicated CLI↔C-ABI per backend (§174). Semantically coupled
   to each model's chat template, so lower priority / higher coupling — leave
   unless it grows; revisit if a 3rd consumer appears.

Not candidates: `crispasr_strip_ascii_punctuation` / `crispasr_lowercase_ascii`
/ `crispasr_backend_should_use_gpu` are CLI-only (already DRY via
`crispasr_backend_utils.h`; the C ABI does not duplicate them).

---

## §177 VibeVoice #171 — remaining layers after the chunking fix (OPEN)

The server/CLI chunking divergence is fixed (§176, `crispasr_tts_chunking.cpp`
prefix guard). Two layers from GH #171 remain, both needing the reporter's
RDNA4 box (AMD RX 9070 XT, gfx1201) — we get clean audio on Metal +
Vulkan/MoltenVK and cannot reproduce locally.

1. **RDNA4 coopmat2 flash-attn garbage.** Confirmed by the reporter:
   `GGML_VK_DISABLE_COOPMAT2=1` fixes every broken sample. This is the
   upstream ggml coopmat2 shader on RDNA4 (already filed upstream — see
   `docs/prompts`/upstream-PR notes #19). **Action:** once upstream lands, or
   as a stopgap, auto-disable coopmat2 flash-attn for the vibevoice TTS graph
   on gfx12xx (the `VIBEVOICE_TTS_FLASH_ATTN=0` knob already bisects LM
   attention; `VIBEVOICE_VAE_BACKEND=cpu` isolates the decoder). Set a safe
   RDNA4 default rather than relying on the user-set env var.

2. **Cross-request statefulness** ("1st request correct; after N different
   sentences the same input gives garbage" — server only, CLI is one
   synthesize per process). The §176 chunking fix only *de-amplifies* this
   (removes the 2-calls-per-request multiplier); it is not the root cause.
   Suspect a scratch / cached voice-prompt KV buffer reused across
   `vibevoice_synthesize()` calls that a numerically-bad request (RDNA4
   coopmat2) poisons for subsequent ones — the logs show pos resets to 309
   and "pre-filled KV from voice prompt" every request, so the leak is likely
   in a *shared buffer behind* that refill, not the position counter.
   **Action:** audit `vibevoice_synthesize` for state that survives between
   calls (voice-KV cache tensors, compute scratch); add a session-level
   "synthesize is idempotent across calls" test (synthesize A, then B×5, then
   A again → byte-identical to the first A) on a reproducible backend. Ask the
   reporter to re-test multi-request after §176 + `GGML_VK_DISABLE_COOPMAT2=1`
   to see whether a residual leak remains once the numerics are clean.

---

## WASM Browser build — all backends, multithreaded

### Goal

Compile CrispASR to WebAssembly so all ASR/TTS/LID/VAD/alignment
backends run in the browser. Multithreaded via `-pthread` +
`PTHREAD_POOL_SIZE=8` (requires COOP/COEP headers on the hosting page).

### Architecture

The `bindings/javascript/emscripten.cpp` exposes ~60 functions via Embind
(`--bind`): whisper ASR (`init`/`full_default`), the backend-agnostic ASR session
surface (`asrOpen`/`asrTranscribe`/`asrSet*`, added 2026-06-13 — see the §166
follow-up above), TTS synthesis (`ttsSynthesize` + the `tts*` setters), kokoro
language routing, and registry helpers.

All ~70 backend static libs link into `crispasr-lib` via the `whisper`
alias target. The JS binding links against `whisper`, pulling everything in.

### What was done

1. **`build-wasm.sh`** — Self-contained build script. Drives `emcmake cmake`
   with CPU-only flags (no CUDA/Metal/Vulkan/BLAS/OpenMP), SIMD128 optional,
   `CRISPASR_WASM=ON`. Builds the `libwhisper` target.

2. **`CMakeLists.txt`** — Added `CRISPASR_WASM` option (default ON under
   Emscripten). When enabled, `add_subdirectory(bindings/javascript)` is
   wired into the top-level build. Threading: `-pthread` in C/CXX flags
   (already present), `USE_PTHREADS=1` + `PTHREAD_POOL_SIZE=8` in the
   JS binding's link flags (already present).

3. **`crispasr_cache.cpp`** — Added `#ifdef __EMSCRIPTEN__` guards:
   - `dir()` returns `/models` (Emscripten MEMFS path)
   - `fetch()` returns false with a diagnostic (models are pre-loaded
     by JS via `FS.writeFile`)

4. **`bindings/javascript/CMakeLists.txt`** — Existing, links `whisper`
   with `--bind`, `MODULARIZE=1`, `EXPORT_NAME=whisper_factory`,
   `FORCE_FILESYSTEM=1`, `ALLOW_MEMORY_GROWTH=1`, `USE_PTHREADS=1`,
   `PTHREAD_POOL_SIZE=8`. No changes needed.

5. **`bindings/javascript/emscripten.cpp`** — Existing, ~700 lines,
   exposes the full session C-ABI. No changes needed.

### Deployment requirements

The hosting page MUST set HTTP headers:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

These enable `SharedArrayBuffer` which the pthread worker pool requires.
HuggingFace Spaces (Docker SDK) sets these automatically.

### Model loading flow

1. JS fetches GGUF model via `fetch()` (or from IndexedDB cache)
2. JS writes to Emscripten MEMFS: `Module.FS.writeFile('/models/model.gguf', data)`
3. JS calls `whisper_factory()` to instantiate the module
4. JS calls `Module.init('/models/model.gguf')` or `sessionTranscribe(pcm, lang)`

### Output files

- `build-wasm/bin/libwhisper.js` — Emscripten JS loader
- `build-wasm/bin/libwhisper.wasm` — WebAssembly binary
- `build-wasm/bin/libwhisper.worker.js` — Worker for pthreads

### Effort

Small — infrastructure was 90% present from upstream whisper.cpp heritage.
Only needed: `build-wasm.sh`, wiring `add_subdirectory(bindings/javascript)`
into top-level CMakeLists, and `__EMSCRIPTEN__` guards in `crispasr_cache.cpp`.

### Relation to CrispEmbed WASM

CrispEmbed's WASM build (June 2026) took the opposite threading approach:
single-threaded, no `-pthread`, no SharedArrayBuffer requirement.
This was appropriate for the math OCR decoder which runs single-threaded.
CrispASR needs multithreading for real-time ASR inference, so it uses
the full pthread path with COOP/COEP headers.

---

## 40. More Moonshine model variants

Convert + upload to HuggingFace:
- ~~`moonshine-base` (61.5M, better WER)~~ **DONE** (cstr/moonshine-base-GGUF)
- `moonshine-streaming-tiny/small/medium` — different architecture, needs new runtime
- ~~`moonshine-tiny-{ja,ar,ko,zh,vi,uk}` (multilingual)~~ **DONE** (12 repos on HF)
- ~~`moonshine-base-{ja,uk,vi,zh,ar,ko}` (multilingual)~~ **DONE** (12 repos on HF)
- ~~`moonshine-{base,tiny}-de` German fine-tunes~~ **DONE** — fidoriel (6.9%/11.4% WER, NC-SA) + dattazigzag (MIT)

Converter fix: 1D tensors (norms, biases) forced to F32; conv_1d_f32 mul_mat
argument order fixed for F16 kernels.

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

## 97. More Parakeet variants

The runtime in `src/parakeet.cpp` already dispatches TDT vs CTC via the
`has_ctc` GGUF flag (line 1669), and `models/convert-parakeet-to-gguf.py`
reads every hparam from `model_config.yaml` + cross-checks against actual
tensor shapes (line 314+). So most NVIDIA Parakeet checkpoints that share
the FastConformer-encoder + TDT-or-CTC-decoder shape should be
converter-runs with no new C++.

### Shipped — TDT / TDT+CTC (2026-05-20)

All four converted, smoke-tested on JFK at M1 Metal, and uploaded to HF
with READMEs. Registry entries + `-m <name>` lookup + C ABI surface all
wired in `crispasr_model_registry.cpp` + `crispasr_model_mgr_cli.cpp`
(fix landed on the same branch — see commit `d8325847`). One C++ fix
along the way: `parakeet_init_from_file` now auto-flips to CTC decode
when `pred_layers < 2 && has_ctc` (commit `0a902517`); without this
the 110m's single-LSTM predictor would have failed silently.

- [x] `nvidia/parakeet-tdt-0.6b-v2` → [`cstr/parakeet-tdt-0.6b-v2-GGUF`](https://huggingface.co/cstr/parakeet-tdt-0.6b-v2-GGUF) (468 MB Q4_K, 11.4× rt) — `-m parakeet-v2`
- [x] `nvidia/parakeet-tdt-1.1b` → [`cstr/parakeet-tdt-1.1b-GGUF`](https://huggingface.co/cstr/parakeet-tdt-1.1b-GGUF) (808 MB Q4_K, 16× rt, lowercase) — `-m parakeet-tdt-1.1b`
- [x] `nvidia/parakeet-tdt_ctc-110m` → [`cstr/parakeet-tdt_ctc-110m-GGUF`](https://huggingface.co/cstr/parakeet-tdt_ctc-110m-GGUF) (91 MB Q4_K, 45× rt, auto-CTC) — `-m parakeet-tdt_ctc-110m`
- [x] `nvidia/parakeet-tdt_ctc-1.1b` → [`cstr/parakeet-tdt_ctc-1.1b-GGUF`](https://huggingface.co/cstr/parakeet-tdt_ctc-1.1b-GGUF) (810 MB Q4_K, mixed-case + punct) — `-m parakeet-tdt_ctc-1.1b`

Each GGUF available in three precisions (F16, Q8_0, Q4_K). All work via:
- the unified CLI: `crispasr -m <name> --auto-download -f audio.wav`
- C ABI: `crispasr_registry_lookup_abi(<name>, ...)` returns
  filename + URL; existing init functions consume the path directly.

Filename-heuristic dispatch in `crispasr_backend.cpp:370-372`
unchanged — `parakeet-tdt_ctc-*.gguf` matches "parakeet" with the
`!contains_ci("tdt")` guard preventing accidental fc-ctc routing.

### Done — parakeet-rnnt 0.6b + 1.1b (2026-05-24)

- **`nvidia/parakeet-rnnt-0.6b`** + **`nvidia/parakeet-rnnt-1.1b`** — standard RNN-Transducer (no duration head).
  - `parakeet_rnnt_decode` in `src/parakeet.cpp` — blank→advance t by 1,
    real token→stay on same frame, `max_per_step=10` anti-loop cap; hotword
    biasing wired.
  - Converter RNNT detection: `joint.joint_net.2.weight` key detection → sets
    `n_tdt_durations=0`; runtime dispatches via `use_rnnt = !use_ctc && n_tdt_durations==0`.
  - In-memory nemo loading avoids disk extraction (BytesIO + torch.load).
  - 0.6b: 24-layer encoder, Q4_K 447 MB; `cstr/parakeet-rnnt-0.6b-GGUF`.
  - 1.1b: 42-layer encoder, Q4_K 770 MB; `cstr/parakeet-rnnt-1.1b-GGUF`.
  - Both smoke-tested on JFK (correct transcript). Registry entries added. Committed + pushed.

**Still open:**
- **`nvidia/parakeet_realtime_eou_120m-v1`** — streaming + end-of-utterance head. Needs cache-aware FastConformer streaming (cf. PLAN #81 Nemotron), plus an EOU head. Not a converter-only job.
- **`nvidia/parakeet-unified-en-0.6b`** — **surveyed 2026-06-15 via Kaggle kernel.**
  Target class: `EncDecRNNTBPEModel` (same as other parakeets). Encoder:
  `ConformerEncoder` with new `att_chunk_context_size` param (dynamic chunked
  convolutions for unified offline+streaming). Kaggle NeMo 2.7.3 lacks this
  param — needs `nemo_toolkit>=2.8` or later. Alternatively: extract weights
  directly from the .nemo zip (torch checkpoint format) + read config from the
  Hydra config embedded in `save_restore_connector`. The existing
  `convert-parakeet-to-gguf.py` should work once the model loads — same
  `EncDecRNNTBPEModel` class, same weight naming. The C++ runtime needs
  `att_chunk_context_size` support in `core/fastconformer.h` (dynamic conv
  kernel selection per chunk, similar to nemotron's streaming conv cache).
  **2026-06-16 v4 kernel — direct zip extraction WORKS.** Custom
  `NeMo2Unpickler` reads `data.pkl` + storage files from the NeMo 2.x zip
  without instantiating the NeMo model (bypasses `att_chunk_context_size`).
  989 tensors extracted with actual data. Hparams: d_model=1024, n_layers=24,
  vocab=1025, pred=640, joint=640 — **same architecture as standard parakeet**.
  **v5 kernel — GGUF conversion succeeded (1181 MB F16).** Synthetic NeMo 1.x
  tar (config + tokenizer from parakeet-rnnt + weights) fed to existing
  converter. CrispASR test SIGABRT: runtime assumes 4x subsampling but
  parakeet-unified uses 8x (3 strided convs vs 2). **Fix needed:**
  `parakeet_build_pre_encode` in `src/parakeet.cpp` (or
  `core/fastconformer.h`) must handle `subsampling_factor=8` — 3 Conv2d
  layers with strides [1,2] instead of 2. Tensor shapes are already in
  the GGUF; the graph builder just needs the extra conv layer.

### Won't do

- `parakeet-ctc-0.6b-Vietnamese` — already runtime-supported (CTC); ship
  if a Vietnamese user asks. Tracked as a known gap, not active work.

See also: [#98 Hotwords](#98-hotwords--contextual-biasing) — orthogonal
feature that lights up biasing on every Parakeet variant once the CTC-WS
trie lands.

---

## 98. Hotwords / contextual biasing

User-supplied vocabulary that the ASR should prefer when in doubt
(names, jargon, product terms, place names). Distinct from "improve
overall WER" — hotwords only help on the biased subset, but on that
subset the lift is large (e.g. NeMo CTC-WS reports F1 jumps from ~30 %
to ~80 % for OOV name spotting; FunASR SeACo-Paraformer ~58 % F1 lift
on AISHELL-NER).

### Upstream-support survey (May 2026)

| Backend | Upstream support | Mechanism |
|---|---|---|
| parakeet-tdt, -ctc, -rnnt, -tdt_ctc, -unified | YES | NeMo CTC-WS (Aho-Corasick phrase-boost trie + shallow fusion); MBS Transducer hotwords (Feb 2026) |
| fastconformer-ctc | YES | Same NeMo CTC-WS pipeline |
| funasr / fun-asr-nano / mlt-nano (paraformer) | YES | Native `hotword=` kwarg in upstream `AutoModel.generate`; SeACo-Paraformer-style hotword encoder pre-decoder |
| qwen3-asr | YES | DashScope `vocabulary_id` + free-text `-c <context>` |
| voxtral / voxtral4b | YES | `context_bias` API field, up to 100 entries |
| granite-speech-4.1-2b-plus | YES | Keyword list baked into LLM prompt (the `-plus` variant is the keyword-prompted one) |
| mimo-asr | YES (research) | PromptASR cross-attends a text-prompt encoder into the speech encoder; HF card doesn't expose a flag yet |
| whisper | partial | `initial_prompt` only — decoder text-prompt conditioning, no hard bias |
| firered-asr-llm | partial | `(prompt, speech, transcript)` triplet on the LLM variant supports free-text prompt |
| cohere-transcribe | NO | API takes model / language / file / temperature only |
| moonshine | NO | Whisper-style encoder-decoder; no `initial_prompt` either |
| kyutai-stt | NO | "Contextual accuracy" from delayed-streams modelling, no keyword list API |
| omniasr (CTC head, Meta) | NO upstream | But our CTC-WS trie would just work — model-agnostic on the logit stream |
| glm-asr | NO | Plain transformers / vLLM / SGLang inference, no documented hotword field |

### Phased implementation

**Phase A — generic CTC-WS phrase-boost trie** (covers parakeet-ctc /
parakeet-tdt / fastconformer-ctc / omniasr in one shot)

- New shared helper `src/core/asr_context_bias.{h,cpp}` —
  Aho-Corasick trie over piece-id sequences with a configurable boost
  score per matched phrase; emits a per-frame log-prob bias vector that
  the CTC / TDT decoder shallow-fuses into its argmax / beam scoring.
- Wire-in points: `parakeet_ctc_decode` and `parakeet_tdt_decode` in
  `src/parakeet.cpp:999+` and `parakeet.cpp:1670` dispatch. Phrase
  tokenisation goes through the same SentencePiece model the backend
  already loads, so users supply human-readable strings.
- CLI: `--hotwords "Acme Corp,Sandra Berenz,GPU-PB"` and/or
  `--hotwords-file <path>` (one phrase per line, optional `^N` boost
  suffix); env var `CRISPASR_HOTWORDS=...` for the OpenAI-server path.
- Estimated ~250–400 LOC including beam-search rescoring path; pure CPU,
  no ggml graph needed.
- Reference impl to mirror: NeMo CTC-WS notebook + the `BoostingTree`
  C++ in TurboBias (`arxiv.org/html/2508.07014v1`).

**Phase B — `--hotwords` → LLM prompt-prefix helper** (covers funasr,
granite-plus, voxtral, qwen3-asr)

- Tiny shared helper in `src/core/` that renders a hotword list into
  the backend's expected prompt format (each backend has a different
  template — granite uses `Keywords: …`; funasr uses a hotword token
  block; qwen3 uses free-text context). One template registry, one
  call site per backend.
- Wire-in points: `funasr_transcribe_ex` (the LLM-decoder prompt
  builder we just shipped), `granite_nle_transcribe`, voxtral,
  qwen3-asr. All four already have a "system prompt" path the helper
  plugs into.
- Estimated ~150 LOC + per-backend template strings.

**Phase C — parakeet TDT joint-net boost (Transducer-native)**

- Mirror NeMo's MBS hotwords for Transducer: add a per-step bias on the
  joint-net output when the partial hypothesis matches a prefix in the
  trie. More accurate than shallow-fusion at the cost of being
  TDT-specific (CTC path already covered by Phase A).
- Defer until Phase A is shipped + benchmarked; only worth the
  complexity if Phase A on TDT undershoots NeMo's reference numbers.

### Out of scope

- **Whisper `initial_prompt`** — already supported upstream via the
  whisper.cpp loader. If a user really wants Whisper biasing, the
  current path is to set `--initial-prompt`; no new CrispASR work.
- **MiMo PromptASR exposure** — the architecture supports it but
  upstream doesn't expose a flag and the HF card has no hotword
  example. Park until upstream ships an API.
- **Cohere / Moonshine / Kyutai-STT / GLM-ASR** — no upstream support,
  no architectural hook; would require training a side-channel which
  is outside this engine's scope.

### Validation

- Add a `tests/test_hotwords.py` that runs a synthetic clip with a
  rare name (e.g. "Berenz") through each Phase-A backend with and
  without `--hotwords Berenz`, asserts the unbiased transcript
  mispells it and the biased one nails it.
- For Phase B, point at the upstream reference Python and assert the
  prompt-prefix matches byte-for-byte.

### Effort estimate

- Phase A: 2–3 days (helper + 4 wire-ins + tests + docs).
- Phase B: 1 day (helper + 4 wire-ins + tests).
- Phase C: 1–2 days, only if Phase A undershoots on TDT.

Total: ~1 week of work covering 9 of 14 backends.

### Sources

- NeMo CTC-WS tutorial: `tutorials/asr/ASR_Context_Biasing.ipynb`
- NVIDIA word boosting docs: `docs.nvidia.com/nemo-framework/.../word_boosting.html`
- Fast Context-Biasing CTC-WS paper: `arxiv.org/html/2406.07096v1`
- TurboBias / GPU-PB: `arxiv.org/html/2508.07014v1`
- FunASR Paraformer hotword: `modelscope/FunASR examples/industrial_data_pretraining/paraformer/README.md`
- Voxtral context_bias: `docs.mistral.ai/studio-api/audio/speech_to_text`
- Qwen3-ASR-Toolkit: `QwenLM/Qwen3-ASR-Toolkit`
- Xiaomi PromptASR: `arxiv.org/pdf/2309.07414`
- icefall shallow-fusion: `k2-fsa/icefall docs/source/decoding-with-langugage-models/shallow-fusion.rst`

---

## 42. VibeVoice-ASR 7B

**DONE.** Full ASR+TTS GGUF (1205 tensors) at `cstr/VibeVoice-7B-GGUF`,
7 quantizations Q3_K (4.7 GB) through F16 (17.4 GB). Layer offload
validated on M1 Metal (ASR 28L, TTS 20L). See HISTORY for details.

---

## funasr — perf follow-ups (LOW priority, not blocking)

The 2026-05-20 funasr port ships with `ggml_flash_attn_ext` on the
encoder + adaptor (FA on by default, opt out with `FUNASR_NO_FA=1`)
and `core_attn::kv_self_attn` on the LLM body. Three opportunistic
optimisations that didn't make the first cut and are worth ~one bench
session each when somebody wants to push the numbers:

1. **Per-step LLM decode graph cache.** On JFK (T_lfr=183, 29 tokens
   decoded) the decode loop runs at 37.6 ms/token; the unfused
   memory-bound floor for F16 Qwen3-0.6B on M1 is ~6 ms/token, so
   ~30 ms is unaccounted-for graph build + sched alloc overhead.
   Pattern: build the step graph once at `funasr_kv_init` time with
   `kv_indices` runtime input (so K/V writes go to a runtime slot
   via `ggml_set_rows` instead of the default static-offset
   `ggml_cpy`) and `fixed_kv_len = kv_max_ctx` (so topology stays
   constant). Each decode step then only writes the positions /
   kv_indices / causal_mask / inputs_embeds inputs and re-runs the
   cached graph. Expected savings: 5-10 ms/tok ≈ 15-25 % of total
   decode time. Same pattern qwen3_asr could adopt.

2. **Encoder graph cache by T_lfr bucket.** At T_lfr=183 the encoder
   takes 258 ms; back-to-back calls on similar-length clips pay the
   graph build cost each time. Bucket to {128, 256, 512, 1024, 2048}
   like voxcpm2's TSLM (HISTORY 2026-05-19) — pad the inputs to the
   bucket and emit a static mask that drops the trailing rows. The
   first call to each bucket pays the build cost; everything after
   reuses it. Expected savings: 10-20 ms per call once warm.

3. **Fused LLM QKV — DONE (shipped at initial port time).** Pattern from
   `qwen3_asr.cpp`: concat Q/K/V weights at load, one matmul per layer.
   Already implemented in `funasr_init_from_file` (unconditional, no
   env-var opt-out — fused at load if Q/K/V shapes + types match).

4. **Single-token embed fast path — DONE (2026-06-20, HISTORY §180).**
   `CRISPASR_FUNASR_EMBED_FAST` (default ON). AR decode hot path called
   `funasr_embed_tokens({next_id})` once per step, paying full graph-build
   + sched-alloc overhead for a single GET_ROWS. New path dequants one
   row directly from `token_embd.weight`, skipping the graph. Transcript
   byte-identical. VPS A/B: embed step ~1.6× faster (15–26 vs 27–41
   ms/tok). On M1 Metal (~37 ms/tok total decode), this is the
   lowest-hanging fruit from item #1 above.

4. **Two-pass: CTC fast pass → Fun-ASR-Nano LLM rescore.**
   RapidAI/RapidSpeech.cpp claims a "CTC fast pass + LLM rescoring" path
   for FunASR-Nano. The state of CTC support for Fun-ASR-Nano took a
   couple of investigative passes to map cleanly; **the situation as
   of 2026-05-21**:

   - **Official upstream `FunAudioLLM/Fun-ASR-Nano-2512/model.pt`** —
     1880 MB, **1261 tensors, 0 with `ctc` in the name**. Prefix
     breakdown: `audio_encoder` (914), `llm` (311), `audio_adaptor`
     (36). Same shape for `Fun-ASR-MLT-Nano-2512/model.pt`
     (independent binary, also 0 CTC). Verified locally 2026-05-21
     against the cached HF snapshots; consistent with `840e36dd`
     "no-CTC finding".
   - **Official FunASR framework `funasr/models/fun_asr_nano/`** —
     does ship CTC code. `model.py`'s `FunASRNano.__init__` sets
     `self.ctc_decoder = None` by default and only builds a CTC
     head when `ctc_decoder` is present in `kwargs` (i.e. set in
     the training config). `ctc.py` defines the standard `CTC`
     module (single Linear `ctc_lo` + `CTCLoss`). Recipes live in
     `examples/industrial_data_pretraining/fun_asr_nano/`.
   - **`csukuangfj/funasr-nano-with-ctc`** (sherpa-onnx / k2-fsa
     maintainer, Apache-2.0) — the only public *trained* CTC head.
     Recipe in `config.yaml`: `ctc_decoder: Transformer` with
     `n_layer: 5`, `ffn_dim: 2048`, `encoder_dim: 512`, `llm_dim:
     512`; encoder/adaptor/LLM frozen, only CTC trained;
     `effective_save_name_excludes: - llm.` so the saved `model.pt`
     (599 MB) is encoder + adaptor + CTC head, no LLM.
   - **`manyeyes/Fun-ASR-Nano-2512-CTC-onnx`** (and `-int8-onnx`),
     **`Oulasong/Funasr_Nano_MLT_ONNX`**, **`jiyilin123/FunASR-CTC-Nano-INT8-ONNX`**
     — almost certainly downstream ONNX/int8 conversions of
     csukuangfj's trained CTC head. manyeyes' description text
     ("encoder与Fun-ASR-Nano-2512-CTC中的encoder一致") confirms
     they reuse the upstream frozen encoder, which is csukuangfj's
     setup.

   So upstream released a pure LLM-style ASR by choice — the
   framework supports CTC as opt-in, but the released checkpoint
   omits the head. CTC is one trained-from-scratch head away, and
   csukuangfj has already done that training under Apache-2.0.

   ### Two viable two-pass patterns

   | Source for Pass 1 | Encoder forwards | Vocab mapping | Trust |
   |---|---|---|---|
   | **csukuangfj's CTC head + upstream encoder/adaptor/LLM** | **one** (shared) | none (CTC head was trained against Qwen3 tokenizer / SANM frame rate; same tokens.txt as upstream) | single-author training, no published WER, but framework-blessed recipe |
   | **SenseVoice-Small (already in CrispASR)** | two (different encoder weights) | needed (SenseVoice's CTC vocab vs Fun-ASR-Nano's Qwen3 tokenizer differ) | gold — official Alibaba release |

   The first pattern is the cleaner architecture (single encoder
   forward, no cross-vocab) but inherits csukuangfj's training
   trust. The second is more conservative but pays an extra
   encoder pass and a vocab translation step.

   ### Phase A — measurement only (no code)

   Before writing any C++:

   - Tensor-list csukuangfj's `model.pt` to confirm his encoder
     weights are byte-identical to upstream's (they should be —
     he loaded them with `freeze: true`). Same shape, same dtype,
     same values modulo precision conversion.
   - Run csukuangfj's CTC head + upstream's encoder+adaptor on a
     small Chinese + English benchmark set we have ground-truth
     transcripts for. Measure CER/WER vs upstream's pure-LLM
     path; the framework's CTC training was auxiliary
     (`detach_ctc_decoder: true`, `ctc_weight: 1.0` as one of
     several losses), so CTC quality won't match the LLM head —
     we just need it within ~3-5% relative for rescore to be a
     net win.
   - If quality is within bounds, write up the result in
     LEARNINGS.md and proceed to Phase B.

   ### Phase B — implementation

   - Grow `models/convert-funasr-to-gguf.py` to (optionally) pick
     up `ctc_decoder.*` tensors from a separate "with-ctc"
     checkpoint, written as `funasr-nano-with-ctc-q4_k.gguf` or
     similar. Auto-download from `cstr/funasr-nano-with-ctc-GGUF`
     (we'd mirror csukuangfj's weights under our HF account, with
     attribution).
   - Opt-in flag `CRISPASR_FUNASR_TWOPASS=1` that requires the
     companion `with-ctc` GGUF to be loadable; fall back to the
     single-pass path when it's missing. New `--asr-rescore` CLI
     flag for explicit selection.
   - Wire one encoder forward, fork into CTC head + LLM head.
     Pass 1: greedy CTC → per-frame token probs. Pass 2: skip
     LLM if avg per-frame confidence >0.95; otherwise use CTC
     top-K hypothesis as decode-prefix candidates for the LLM.
     Expected savings: 2-4× on high-confidence clips, neutral on
     hard audio (still pays both heads but only one encoder).

   ### Fallback (only if Phase A's quality measurement fails)

   Use SenseVoice-Small as the fast pass. Pays two encoder
   forwards and a vocab translation, but inherits gold-source
   trust. Same opt-in flag, different model lookup.

None of these affect correctness — they're pure throughput pickings.

---

## §187 Cross-runtime embed fast path sweep (MOSTLY DONE)

Single-token embed fast path for all LLM-based AR decode runtimes.
Same pattern as funasr §180 item #4: per-step `embed_tokens({next_id})`
called in the decode loop; fast path dequants one row directly from the
token embedding weight, skipping graph-build + sched overhead.

| Backend | Env var | Status |
|---------|---------|--------|
| funasr | `CRISPASR_FUNASR_EMBED_FAST` | **DONE** §180 |
| glm_asr | `CRISPASR_GLM_ASR_EMBED_FAST` | **DONE** §187 |
| moss_audio | `CRISPASR_MOSS_AUDIO_EMBED_FAST` | **DONE** §187 |
| qwen3_asr | `CRISPASR_QWEN3_ASR_EMBED_FAST` | **DONE** §187 |
| gemma4_e2b | `CRISPASR_GEMMA4_E2B_EMBED_FAST` | **DONE** §187 (+ sqrt(d) scale) |
| outetts | — | Already had n==1 fast path at port time |
| orpheus | — | Already had n==1 fast path at port time |
| lfm2_audio | — | Already does direct row reads (no graph) |
| mini_omni2 | — | N/A — never calls embed with n=1 |
| granite_speech | — | OPEN (model too large for VPS bench) |

### Other perf candidates from bench data

1. **Deepen shallow bench stages.** vibevoice, wav2vec2-ggml, m2m100,
   t5_translate only have `_total` stages — add finer-grained stages.
2. **§175 DRY: `src/core/lang_names.h`** — 4 copies → 1 header.
3. **Encoder graph cache by bucket** — funasr, sensevoice, qwen3_asr.

---

## §201 Kaggle CUDA backend failures — full sweep 2026-06-20 (OPEN)

The first full-coverage Kaggle GPU sweep (`tools/kaggle-benchmark-all-backends.py`,
streamed to `cstr/crispasr-kaggle-progress/full-backend-sweep/`, see PERFORMANCE.md
2026-06-20) ran 59 backends. The first pass showed 10 "failures", but a follow-up
audit + a fixed-subset re-test (run tag `voicefix-retest`) found **2 were benchmark
mistakes, not bugs**, leaving **7 genuine failures** (+1 pending). Of those 7,
**fastpitch + speecht5 are now fixed** (HISTORY §204), leaving **5 open**:

**RESOLVED — were benchmark args/category, fixed in the script (commit on main):**
- ✅ **vibevoice-1.5b** — is a *TTS* model (`vibevoice-1.5b-tts-q4_k.gguf`), was
  wrongly run as ASR → empty. Moved to TTS; now PASSES (367 KB).
- ✅ **vibevoice-tts** — voice-conditioned; the bare `--tts` gave it no voice.
  With `--voice <ref.wav>` it now PASSES (255 KB).

**Pending one more confirm:**
- [ ] **f5-tts** — once given a reference voice it *runs* (was a fast-fail before),
  but **TIMEOUT at 120 s** in the re-test. Bump the smoke timeout (≥240 s) and
  re-run to settle pass-vs-stuck; passes on M1 Metal locally.

**FIXED — moved to HISTORY §204 (FastPitch + SpeechT5 GPU/CUDA):**
- ✅ **fastpitch** (TTS) + **speecht5** (TTS) — both fixed by `69dc1789`. Two
  Metal-masked HiFi-GAN bugs (sched-allocated decoder/vocoder graphs computed on
  `backend_cpu`; channel-major `convt1d_decomp` on a time-major input). Both
  validated on M1 Metal. CUDA re-test pending. Full write-up in HISTORY §204.

**RESOLVED — lfm2-audio GPU backbone fixed (HISTORY §206):**
- ✅ **lfm2-audio** (ASR) — fully fixed. The crash was embed lookups on CUDA
  (device-ptr deref, like §204). The garbage was the ROOT CAUSE found via the
  diff harness + GGML_SCHED_DEBUG (`63d4c013`): the backbone's weight-less leading
  RMSNorm made `ggml_backend_sched` put the input + first op on CPU and feed the
  GPU a miscomputed cross-backend copy. Fix: compute the backbone directly on
  ctx->backend via gallocr (single-backend, no copy) in `backbone_step` + `run_lfm`.
  GPU transcribes JFK verbatim; GPU diff matches the PyTorch ref like CPU.
  **GPU is default again**; `CRISPASR_LFM2_AUDIO_CPU=1` forces CPU. PyTorch ref +
  GPU diff (`CRISPASR_DIFF_USE_GPU=1`) shipped. Follow-up: GPU-decode graph caching
  (AR decode is currently dispatch-bound, GPU ~slower than CPU on JFK).

**RESOLVED — kugelaudio GPU empty-output (HISTORY §209):**
- ✅ **kugelaudio** (TTS) — fixed by `bdb3f42f`. Same §206 root cause: the
  Qwen2.5-7B LM's weight-less leading RMSNorm made `ggml_backend_sched` put the
  input + first op on CPU and feed the GPU a miscomputed copy → garbage LM → no
  speech-diffusion token → empty output. Fix: backbone graphs compute directly on
  ctx->backend via gallocr; only the VAE decoder stays on the sched (its `ggml_pad`
  is Metal-unsupported → CPU fallback; CUDA supports PAD so it runs on GPU there).
  Validated on M1 Metal (q4_k): GPU generates 83200 samples, ASR-roundtrips to
  "Hello there.".

**GENUINE bugs — fail on CUDA even with correct args (TODO):**
- [x] **orpheus** (TTS) — **FIXED (§215, HISTORY).** The 0-byte/SIGSEGV was the
  §176b Lk-bucket decode's dedicated step-sched: a fresh `ggml_backend_sched`'s
  first `alloc_graph` left the cross-backend input copies unbacked on GPU, so the
  first decode step bound a garbage device buffer (crash in `ggml_metal_op_norm`
  on node #0, before any set_rows). §213 was right that neither the talker AR
  decode nor SNAC was the culprit — it was the bucket's separate sched. Fix:
  run the cached bucket graph on the already-warm prefill sched `c->sched`
  (drop `ar_step_sched`) + read the KV from the `ggml_set_rows` result for the
  Metal write→read edge. GPU + CPU bucket both ASR-roundtrip verbatim on M1.
  Stays opt-in (`CRISPASR_ORPHEUS_BUCKET=1`): correct now, but ~30 % slower than
  the non-bucket path on M1 unified memory (over-read); may still win on CUDA.
  **CUDA cross-check still pending** (Kaggle `chr1str/crispasr-orpheus-talker-cuda`
  end-to-end `orpheus_synthesize`).
- [ ] **chatterbox** (TTS) — 0-byte (~14 s) with `--voice <wav> --i-have-rights`;
  the #83 S3Gen GPU fix was Metal-validated — re-check the CUDA S3Gen path.
- [ ] **cosyvoice3** (TTS) — **dies in 0.1 s** even with a reference voice; passes
  on M1 Metal. Flow-matching + HiFT — earliest/cheapest to bisect. **Note
  (2026-06-21):** §205's mixed-radix FFT fix explicitly covers CosyVoice3's
  `n_fft=400` mel (same heap overflow as chatterbox). Likely resolved — needs
  Kaggle CUDA re-test to confirm.

**Method:** per-backend JSONs in the dataset have timing context; reproduce on a
CUDA worker (Kaggle T4/P100 or the A1000) with `CRISPASR_VERBOSE=1` +
`CRISPASR_<BACKEND>_DEBUG=1`. Remaining open: orpheus, cosyvoice3 — chatterbox
fixed in §205, lfm2-audio in §206, kugelaudio in §209 (lfm2 + kugelaudio shared
the §206 sched weight-less-first-op root cause). Several
pass on M1 Metal → CUDA-path-specific; cross-check Metal first. Small models also
fit the 8 GB CPU-only VPS, where the diff harness can drive the fix if the bug
reproduces on CPU. The fastpitch+speecht5 fix (§204) came exactly this way — the
HiFi-GAN layout bug reproduced on CPU locally; lfm2-audio (§206) likewise (the
backbone diverges identically on CPU-vs-Metal diffing).

---

## CosyVoice3-0.5B-2512 TTS — DONE

_Done — see HISTORY.md + git log._

## 51c. MiMo-V2.5-ASR F16 step decode — open

Base runtime + Q4_K + fused-QKV layout shipped → HISTORY §56 + §64.
Sub-items 51a (mmap loader → HISTORY §62) and 51b (step-decode KV
cache reuse → HISTORY §60) also DONE. Only this F16 step decode is
still open — blocked behind ≥32 GB RAM for end-to-end validation.

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
- **Performance pass (in progress, partial wins shipped).** Quiet-bench Q8_0 0.6B with all defaults: ~79 ms/frame (talker ~30 + cp ~49) on a quiet M1 — already under the 80 ms/frame real-time budget at 12.5 fps. Under normal system load ~129 ms/frame; talker and cp scale proportionally with Metal contention. Shipped: **`QWEN3_TTS_O15=1` is default-on** (commit `5e21e4a`) — cp graph reuse saves ~14 ms/frame on cp_pred under contention, ~2-3 ms/frame quiet, bit-identical WAV. Gated, byte-identical, kept default-OFF: `QWEN3_TTS_FUSED_QKV=1` (**Q8_0 bench done 2026-05-23: neutral — interleaved A/B on this M1 shows 129 vs 129 ms/frame; keep default-OFF for Q8_0**; **F16 case benched 2026-05-24: inconclusive — interleaved A/B (6 runs) on loaded machine (model DL + build concurrent) shows σ≈47 ms/frame exceeding any signal; mean baseline 212 vs mean fused 191 ms/frame, warm-up baseline 133 ms/frame consistent with Q8_0 quiet result; keep default-OFF for F16 same as Q8_0; clean quiet-machine bench still open**); `QWEN3_TTS_LK_BUCKET=1` (talker Lk bucketing, **net loss on M1 Metal Q8_0** — see LEARNINGS); `QWEN3_TTS_CP_STEP0_CACHE=1` (cp T=2 step-0 graph cache, claimed 1-3 ms/frame quiet savings — **loaded-machine bench neutral within noise; quiet-machine confirmation still pending**, bit-identical). Investigated: Q8_0 KV cache — blocked on Metal `cont(Q8_0)` source (only F32/F16/BF16 sources supported); needs Metal kernel patch or KV layout restructure to land. Still open: F16 FUSED_QKV clean quiet-machine bench (F16 GGUF now at `/Volumes/backups/ai/crispasr/qwen3-tts-12hz-0.6b-base.gguf`; rerun on a quiet machine with no background I/O); Q4_K talker fused QKV bench (needs Q4_K talker GGUF); the larger lift of fusing 15 cp steps into one graph (needs on-device top-k sampling, ~3 ms/frame upper bound after O15 since most overhead is already gone).
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

## 96. voxcpm2-tts perf — switch to per-step ggml graph (Metal-ready) — DONE (graph-default flipped 2026-06-04)

### Where we are (2026-05-19)

Voice cloning is **structurally correct** end-to-end. The CLI sample-rate
bug (24 kHz header on 48 kHz PCM, which sounded like half-speed
distortion) is fixed (`0321fa5e`). Q4_K cloning of "Hello world" with the
JFK reference now ASR-roundtrips to "Hello world." Diff harness verifies
every prefill stage matches Python upstream at cos_mean ≥ 0.98 under
`VOXCPM2_USE_REF=1`.

### Why this is the next priority

Synthesis is slow: 19 s wall-clock for 0.8 s of audio on M1 CPU (Q4_K,
"Hi", no ref). `VOXCPM2_BENCH=1` (added in `a3dcdd21`) shows the
per-AR-step breakdown:

  - **cfm (LocDiT × 20 calls)**     63.7%   (1186 ms / step)
  - **tslm_step (28 layers)**       25.3%   (471 ms / step)
  - locenc (12 layers × 5 tokens)   10.1%
  - everything else                 <1% each

Inside `cfm_euler_solve`, `locdit_forward` accounts for ~100% of CFM
time. Per call it does ~30 `matmul_mv_ggml` invocations, each of which
builds + computes + frees its own tiny ggml graph (`ggml_init` /
`ggml_new_graph` / `ggml_backend_graph_compute` / `ggml_free`). That's
~600 graph builds per AR step just for CFM.

### Why we can't just flip the backend

Tried `g_cpu_backend = ggml_backend_init_best()` (Metal) in `1635e4fa`
— SIGSEGV on first kernel dispatch. The input tensors in
`matmul_mv_ggml` live in a CPU-side mem buffer (`ggml_init` with nullptr
mem_buffer) that the Metal backend can't read directly.

### Right fix: per-step graph

Build a single `ggml_cgraph` per `locdit_forward` call. Inputs flow in
via `ggml_backend_tensor_set(named_input, host_data)`, outputs read out
via `ggml_backend_tensor_get`. With that pattern, weights can also live
on the Metal buffer (load via `load_weights(path, ctx->backend, ...)`)
and the whole pipeline runs on GPU.

Same shape for `tslm_layer_step` and `ralm_layer_step` (the #2 hotspot,
25% of AR-step time).

Reference patterns: `qwen3_tts.cpp:1019 build_graph_talker_kv`,
`chatterbox.cpp:1168 build_graph_t3_kv`, `vibevoice.cpp` — they all do
this: backend pool (`c->backend`, `c->backend_cpu`), pre-allocated
`compute_meta` arena, one `build_graph_*` per forward, single
`ggml_backend_graph_compute`.

### Estimated scope

- `voxcpm2_context`: add `backend`, `backend_cpu`, `compute_meta`,
  `galloc` (~30 LOC).
- `build_locdit_graph` (~300 LOC): in_proj / cond_proj / time_mlp /
  delta_time_mlp / concat / 12 × (RMSNorm / GQA attention with RoPE /
  SwiGLU FFN / residuals) / final norm / out_proj.
- `locdit_forward_graph` wrapper (~100 LOC) that compiles t_sin /
  dt_sin inputs in C++, sets all named inputs, computes, reads out the
  velocity.
- `build_tslm_step_graph` (~250 LOC) similar pattern with KV cache as a
  backend tensor that the graph updates in-place.
- Gate behind `VOXCPM2_USE_GRAPH=1` initially so both paths coexist
  during validation. Verify via diff harness (`cfm_step0_result` cosine
  must stay ≥ 0.93 vs Python, matching current zero-shot). Once
  verified, swap defaults + remove the per-matmul path.

### Expected speedup

- CPU-only: 2-5× on CFM (cross-op scheduling, no malloc/free churn).
  Per-AR-step drops from ~1.9 s to ~0.7-1.0 s.
- Metal: 10-50× on CFM (the actual compute moves to GPU). Per-AR-step
  under 200 ms.

### Validation gate

The existing diff harness (`build-ninja-compile/bin/crispasr-diff
voxcpm2-tts ...`) already prints `cfm_step0_result cos_mean`. The graph
rewrite must keep that number at ≥ 0.93 (current value with
Python-supplied mu+noise). If it drops, the graph has a numerical bug —
bisect by stage. Re-run `VOXCPM2_BENCH=1` to confirm the per-AR-step
times collapse to the targets above.

### Progress (2026-05-19)

LocDiT graph done (`VOXCPM2_USE_GRAPH=1`). Backend pool + per-call
`build_locdit_graph` + `locdit_forward_graph` wrapper added; CFM
`locdit_call` lambda picks the graph or legacy path per env. Diff
harness on `voxcpm2-q4_k.gguf` zero-shot ref: 14 pass / 0 fail / 3 skip
— `cfm_step0_result cos_mean=0.9826` (above the 0.93 gate);
`dit_input_seq cos_mean=0.9984`; all TSLM/RALM stages unchanged
(0.99+). Voice-cloning smoke test ("Hello world" + jfk.wav) still
ASR-roundtrips to "Hello world."

Bench (M1, OMP=8, "Hello world" zero-shot, 6 AR steps):

| Path  | AR loop (ms) | CFM/step (ms) | Wall (ms) |
| ----- | -----------: | ------------: | --------: |
| legacy | 15 306      | 2 398         | 24 211    |
| graph  |  6 815      | 1 035         | 18 519    |

→ 2.3× CFM speedup on CPU. Below the `~400 ms` CPU target from the
plan — remaining overhead is per-call `ggml_init` / `gallocr_alloc`,
not the per-matmul work; caching the graph across CFM Euler iterations
(qwen3_tts bucket pattern) is the next CPU win. Metal still requires
moving weights off `backend_cpu`, blocked on the matching TSLM-step
graph (otherwise the legacy CPU paths SIGSEGV reading Metal memory).

### Progress (2026-05-19 follow-up)

TSLM step graph done. `build_tslm_step_graph` + `tslm_step_graph` use
`core_attn::kv_self_attn` (NEOX RoPE w/ LongRoPE freq_factors, GQA
expansion, flash-attn) with a backend-resident KV tensor.
`init_tslm_kv_backend` + `sync_tslm_kv_cpu_to_backend` transpose the
legacy `vox_kv_cache` (pos-major) into the qwen3 layout (kvh-major)
once per synthesis before the AR loop's first graph step. AR loop
routes through the graph under the same `VOXCPM2_USE_GRAPH=1` gate
as LocDiT.

Diff harness on `voxcpm2-q4_k.gguf` zero-shot ref: still 14 pass /
0 fail / 3 skip; no regression. Smoke "Hello world" zero-shot
ASR-roundtrips correctly.

Bench (M1 CPU, OMP=8, "Hi" zero-shot, 4 AR steps, contended):

| Path                  | AR loop (ms) | cfm/step | tslm/step |
| --------------------- | -----------: | -------: | --------: |
| legacy                | 24 706       | 5 110    |       158 |
| graph (LocDiT + TSLM) | 18 731       | 1 700    |     1 781 |

Net: AR -24%, total -12%. LocDiT graph wins by ~14 s; TSLM graph
loses ~6.5 s. **TSLM step is slower in absolute terms on CPU** —
the 28-layer per-call graph build + `gallocr_alloc_graph` overhead
exceeds the matmul-overhead savings for T=1. Both graphs together
are still net positive overall, and the per-step graph is the
prerequisite for moving weights to Metal (where the GPU compute
savings will dominate the build overhead).

### Progress (2026-05-19 second follow-up)

LocDiT graph cached across CFM Euler iterations (built once into a
dedicated arena, gallocr-reserved once on first use); same pattern
for TSLM step with a single qwen3-style bucket at `Lk=128`
(`fixed_kv_len=128` + `kv_indices=positions` so the K/V scatter is
runtime-indexed). Single bucket because all current synth paths fit
under 128 (zero-shot ≪ 20 positions; "Hello world" + jfk.wav clone
~80 positions). Longer prefills fall through to the dynamic per-call
build automatically.

Bench (M1 CPU, OMP=8, "Hello world" zero-shot, 6 AR steps):

| Path                              | AR loop | cfm/step | tslm/step |
| --------------------------------- | ------: | -------: | --------: |
| legacy                            | 15.3 s  | 2398 ms  |    55 ms  |
| graph, uncached                   |  6.8 s  | 1035 ms  |    38 ms  |
| graph, cached (LocDiT)            |  8.0 s  |  837 ms  |    52 ms  |
| graph, cached (LocDiT + TSLM)     |  6.0 s  |  625 ms  |   180 ms  |

Steady-state CFM per step in the cached path: **~410 ms** (target
~400 ms from the plan, met). Steady-state TSLM step: ~180 ms (par
with legacy, vs ~1781 ms uncached). Voice cloning end-to-end (jfk
ref + "Hello world") AR loop drops to **4.1 s**.

### Progress (2026-05-20)

Weights now load on `c->backend = ggml_backend_init_best()` when
`params.use_gpu` is true. Apple Silicon Metal uses unified-memory
"shared" buffers — `tensor->data` stays CPU-readable, so the
remaining legacy `matmul_mv_ggml` paths (TSLM/RALM prefill, LocEnc,
VAE encode/decode, FSQ, stop) keep working against Metal-resident
weight pointers. Dropped `GGML_PREC_F32` on the LocDiT bidirectional
flash-attn since Metal's `supports_op` refuses any FA op tagged
PREC_F32 (chatterbox-style); the per-stage cosine bar tolerates the
resulting F16 simdgroup drift.

Diff harness `voxcpm2-q4_k.gguf` (CPU path, use_gpu=false): still
14 pass / 0 fail / 3 skip. Smoke "Hello world" zero-shot AND voice
clone (jfk.wav ref) ASR-roundtrip correctly on Metal.

Bench (M1, OMP=8, "Hello world" zero-shot, 6 AR steps):

| Path                  | TSLM prefill | AR loop  | Total   |
| --------------------- | -----------: | -------: | ------: |
| legacy                |     ~5 000 ms| 15.3 s   | 48.7 s  |
| graph cached (CPU)    |    ~4 000 ms |  6.0 s   | 26.2 s  |
| graph cached (Metal)  |       80 ms  |  5.0 s   | 14.1 s  |

Per-substep Metal (CPU shown for comparison):

| Substep    | CPU cached | Metal     |
| ---------- | ---------: | --------: |
| cfm/step   |    625 ms  |   702 ms  |
| tslm/step  |    180 ms  |    82 ms  |
| locenc     |    160 ms  |    34 ms  |

`TSLM prefill 5 s → 80 ms` (≈60×) is the dominant Metal win — the
3-positions × 28-layers prefill is matmul-dense and lights up the
GPU's bandwidth. CFM is roughly the same on Metal as on CPU because
the cached graph is already near optimal; the per-call shape (T=11,
12 layers, n_q=16 GQA on n_kv=2) doesn't have enough independent work
to shine on the GPU.

### Progress (2026-05-20 follow-up)

Multi-bucket TSLM Lk: 5-bucket array (128/256/512/1024/2048) replacing
the single-bucket cache. `tslm_pick_bucket(needed_lk)` picks the
smallest that fits; each bucket is built lazily on first hit via the
existing `build_tslm_step_graph(fixed_kv_len, kv_indices=positions)`
pattern. Long-prefill inputs (multi-sentence cloning, voice
instructions) now stay on the cached path instead of falling through
to the dynamic build at >127 positions.

Validated: long-text clone exercising the 256-bucket
(122 prefill + 81 AR = 203 positions) ASR-roundtrips correctly; both
"built tslm step bucket Lk=128" and "Lk=256" fire in the log. Diff
harness still 14 pass / 0 fail / 3 skip on `voxcpm2-q4_k.gguf`.

### VAE decode profile (2026-05-20)

`VOXCPM2_BENCH=1` now also dumps per-upsample-block timings inside
`vae_decode`. M1 OMP=8, "Hello world" zero-shot, 6 AR steps (i.e. 7
patches × 4 = 28 latent frames):

| Block | up | Cc   | Tc     | Upsample  | Residual |
| ----- | -: | ---: | -----: | --------: | -------: |
| 0     | 8  | 1024 |    224 | 2 957 ms  |   498 ms |
| 1     | 6  |  512 |  1 344 | 1 531 ms  |   695 ms |
| 2     | 5  |  256 |  6 720 | 1 146 ms  |   594 ms |
| 3     | 2  |  128 | 13 440 |   418 ms  |   318 ms |
| 4     | 2  |   64 | 26 880 |   185 ms  |   177 ms |
| 5     | 2  |   32 | 53 760 |    91 ms  |   123 ms |

Totals: upsample 6 327 ms (72 %), residual 2 405 ms (28 %). The
dominant cost is `causal_transposed_conv1d` on the deepest channel
counts (block 0 at Cc=1024 × in_ch=2048). Inner `ic` loop is
strided (ic-stride T_in across x, ic-stride out_ch×ksize across
weight) so the compiler can't auto-vectorise. The arithmetic is
already OMP-parallelised across (oc, ot).

### Progress (SIMD-friendly conv layouts, 2026-05-20)

Rewrote `causal_transposed_conv1d` and `causal_conv1d` (non-depthwise,
ksize>1 path) to lay the reconstructed weight as `[k, oc, ic_inner]`
and transpose x to `[t, ic_inner]` per call. The inner ic dot product
is now contiguous on both axes — auto-vectorisable via NEON on M1.

Bench (M1, OMP=8, "Hello world" zero-shot, 6 AR steps):

|                       |  Old (b94)  | New (SIMD layout) |
| --------------------- | ----------: | ----------------: |
| Block 0 upsample (ms) |      2 957  |              615  |
| VAE decode total (ms) |      8 772  |            3 875  |
| Synth wall total (ms) |     14 766  |            6 800  |

~4.8× on the deepest block-0 upsample; ~2.3× on total VAE; ~2.2× on
total synth wall. Block 5 residual gets noisier (transpose cost
shows up at large T_in with small in_per_grp); future work could
gate the transpose on `in_per_grp >= 128`.

### Progress (VAE decode ggml graph + transposed-conv fix, 2026-05-20)

Full `vae_decode_graph` shipped — single cgraph over the whole upsample
stack (input convs, 6 upsample blocks × {SR cond + snake + transposed
conv + 3 residual units}, final snake/conv/tanh). New helpers
`snake1d_ggml`, `causal_conv1d_ggml`, `causal_transposed_conv1d_ggml`,
plus `vae_wn_init_ggml` building a dedicated arena + backend buffer
for all WN-scaled weights and SR-cond per-bucket slices.

**Two pre-existing bugs fixed (both paths).** (1) `causal_transposed_conv1d`
used `trim = K - 1` (head-shift) where Python's `CausalTransposeConv1d`
expects a tail-trim of `K - S` (= take first `T_in * S` of the no-padding
output — Python's wrapper captures `padding`/`output_padding` as named
kwargs that are NEVER forwarded to `nn.ConvTranspose1d.__init__`, so
`super().forward(x)` returns the no-padding result that the wrapper
then slices `[:-(2P - OP)]` from the END). Legacy head-shift cumulated
to ~46 ms of audio offset over 6 upsample blocks → `decoded_audio`
cos=0.008. After fix: cos=0.683 (remaining drift is upstream).
(2) `vae_wn_init_ggml`'s SR-cond tensors were sized from
`it->second->ne[1]` (=4 for the bucket dim) instead of channel count
(=2048). ggml's binary-op broadcast silently mishandled the 4-vs-2048
mismatch instead of asserting — cos=0.967 instead of cos=0.989 for
`vae_only_graph`. Fix: take `max(ne[0], ne[1])` for the non-bucket
dim. Per-block graph output now bit-identical to legacy CPU on every
channel.

Also aligned CPU `snake1d` to Python's `1/(α + 1e-9)` formula.

Added `vae_only` / `vae_only_graph` diff-harness stages that take
Python's `generated_latent` as input and run the C++ VAE in
isolation — backed by a Python-side hook on `model.audio_vae.decode`
in the reference dumper.

**Validation.**

| Stage              | Before | After |
| ------------------ | -----: | ----: |
| decoded_audio      | cos=0.008 FAIL | cos=0.683 (upstream-limited) |
| vae_only (CPU)     | — | cos=**0.989** |
| vae_only_graph     | — | cos=**0.989** (Metal + CPU graph both match) |
| Upstream stages    | 13 PASS | 13 PASS (unchanged) |

ASR roundtrip: EN/DE/ZH all transcribe back exactly through
parakeet-tdt-v3 / qwen3-asr.

### Progress (Q4_K-vs-F16 drift investigation, 2026-05-20)

The follow-up "upstream drift bringing `decoded_audio` cos to ~0.95"
turned out to be a non-bug. The Q4_K diff harness's apparent
upstream drift (`cfm_step0_result` 0.937, `tslm_layer_27_out` 0.968,
`decoded_audio` 0.683) is dominated by Q4_K weight quantisation, not
by F32-vs-bf16 op precision. Re-running the diff against
`voxcpm2-f16.gguf` instead of `voxcpm2-q4_k.gguf`:

| Stage              | Q4_K cos | F16 cos |
| ------------------ | -------: | ------: |
| tslm_prefill_out   |    0.986 | **0.998** |
| dit_single_fwd     |    0.994 | **0.99999** |
| cfm_step0_result   |    0.937 | **0.99992** |
| tslm_layer_27_out  |    0.968 | **0.999** |
| **decoded_audio**  | **0.683**| **0.929** |

Every intermediate stage hits cos ≥ 0.998 on F16. No code change
needed — the C++ implementation is bit-correct. The diff harness's
default Q4_K archive just multiplies Q4_K quant noise through the
network. ASR roundtrip on Q4_K still works perfectly (EN/DE/ZH).

Tried adding `bf16_round_vec` calls after each tensor op in
`tslm_layer_step`, `bidir_attn_full`, and `locdit_forward` (full
investigation in LEARNINGS.md §"Diff-harness 'drift' is mostly the
GGUF quant, not a code bug"). Near-zero effect, slight regression on
`cfm_step0_result`, +15 min synth runtime — reverted.

### Still TODO

- The F16 `decoded_audio` cos=0.929 still has a 0.07 gap vs Python
  — likely AR stop-step jitter (C++ and Python stop predictors fire
  at slightly different patches) plus residual VAE F16-vs-bf16
  drift. Low priority; both Q4_K and F16 sound natural and
  ASR-roundtrip cleanly in EN/DE/ZH.
- ~~Once the above is investigated (or accepted as inherent), flip
  default to `VOXCPM2_USE_GRAPH=1`.~~ **DONE 2026-06-04.** Accepted
  as inherent (Q4_K quant noise, not a code bug). Default flipped:
  `vox_env_bool_default_on("VOXCPM2_USE_GRAPH")` — graph path is now
  default, opt-out via `VOXCPM2_USE_GRAPH=0`. Validated on two
  independent platforms:
  - **VPS (Hetzner x86_64 CPU):** "Hello world" Q4_K — legacy 1062.8s
    vs graph 670.8s (1.58x). Identical ASR roundtrip ("Hello world.").
    WAV correlation 0.833 (expected: F16 simdgroup drift through
    6-step AR + 20-step CFM Euler).
  - **Kaggle (x86_64 CPU):** two prompts — "Hello world" 1.46x
    (24.2→16.5s), long sentence 1.61x (88.7→55.2s). Perfect
    ASR roundtrip on both, identical WAV sizes. Kernel:
    `chr1str/crispasr-voxcpm2-graph-ab`.

---


## 54-follow-up. granite-speech-4.1 plus speaker labels + word timestamps — open

Variants 4.1 / 4.1-plus / 4.1-nar shipped bit-exact on JFK → HISTORY
§61. Remaining: speaker labels + word-level timestamps for the `plus`
variant via chat_template (~50 LOC, template-only).

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
  | cmn | espeak tone numbers (`ni2χˈɑu2…`) | 3.20 s | 11731 | 1627 | ✅ **FIXED 2026-05-23**: `strip_cmn_tone_numbers` removes digits after phonemization |
  | ja  | kanji fallback (`(en)tʃˈaɪniːz(ja)…`) | 8.38 s | 15460 | 1581 | ⚠️ partial — kana works, kanji becomes English — needs MeCab/KaKaSi (open #3) |

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
   pre-process input with a Japanese frontend to convert kanji → kana
   before espeak. **MIT-clean approach (2026-06-16 survey):**
   - MeCab (BSD-3-Clause) + unidic-lite (MIT) for morphological
     analysis → kanji reading extraction
   - NO kakasi/pykakasi (GPL-3.0 — viral license, incompatible)
   - Feed kana output to existing espeak-ng `ja` voice for IPA
   - Implementation: either libmecab C API via dlopen (like espeak)
     or a pre-built kanji→kana lookup dictionary shipped as a flat
     file (avoids runtime MeCab dependency). The dict approach is
     simpler but less accurate on rare/compound words.
   - fugashi (MIT) is a Python MeCab wrapper; cutlet (MIT) does
     kanji→romaji. Either can generate the offline dict.
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

_Done — see HISTORY.md + git log._

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
| fullstop-multilingual | MIT | XLM-R punct (en/de/fr/it) | **DONE** — runtime in fireredpunc.cpp |
| punctuate-all (kredor) | MIT | XLM-R-base punct (12 langs) | **DONE** — `--punc-model punctuate-all` |
| 1-800-BAD-CODE PCS | Apache-2.0 | XLM-R punc+truecase+SBD (47 langs) | **DONE** — `--punc-model pcs` |
| CT-Transformer (FunASR) | Apache-2.0 | SANM 3-layer (vocab 272727), Chinese+English; production-default in FunASR/RapidPunc | Medium — `modelscope/punc_ct-transformer_zh-cn-common-vadrealtime-vocab272727-pytorch`. SANM block primitives already in CrispASR (`src/core/sanm.h`, used by funasr + sensevoice). Adds a Chinese punc option distinct from BERT-style FireRedPunc; VAD-realtime variant emits punc per-segment for streaming. New backend alias `ct-punc`. |
| truecaser-lstm (BiLSTM) | Apache-2.0 | mayhewsw char-level BiLSTM (3.2 MB, 97.9% F1) | **DONE** — `--truecase-model lstm` (recommended) |
| truecaser-crf | MIT | CRF + context features (24 MB) | **DONE** — `--truecase-model crf` |
| truecaser-de (statistical) | MIT | Wikipedia word-freq (375K entries, 9 MB) | **DONE** — `--truecase-model auto` |
| bert-restore-punctuation | MIT | BERT punct+truecase (en) | Low |
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
| O4 | Beam search for LLMs | Audio-LLM backends | Quality | **DONE** — 18/24 backends via `core_beam_decode` (§139, §61h); only mimo-asr blocked on #115 |
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

### Coverage matrix (May 2026, post-#107)

C-ABI exposes 136+ unique `crispasr_*` exports in
`src/crispasr_c_api.cpp` (9 new in #107 P6 — pluggable speaker
embedder, agglomerative clustering, pyannote cache). Coverage by
binding:

| Binding | Symbols wrapped | Approx % | ASR Transcribe | TTS Session | Variant detect | Align | Diarize | **Diarize embedder²** | LID | VAD | Streaming | Punc | Registry | Cache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rust (`crispasr-sys`) | 65 | ~48% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Python (`_binding.py`) | 67 | ~49% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dart (`flutter/crispasr`) | ~39 | ~29% | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Go (`bindings/go`) | ~54 | ~40% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Java (JNA) | ~42 | ~31% | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅¹ | ✅¹ | ✅¹ |
| Ruby (C ext) | ~30 | ~22% | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅¹ | ❌ | ❌ |
| JS (emscripten) | 18 | ~13% | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

¹ JNA declarations added, idiomatic Java wrapper methods pending.
² **Diarize embedder** column covers the #107 P6 surface:
  `crispasr_speaker_embedder_*_abi`, `crispasr_speaker_cluster_abi`,
  `crispasr_pyannote_cache_*_abi`. Adapters dispatch by spec string
  (`auto`/`titanet`/`indextts`/`ecapa`/.gguf path). Without this
  surface, callers can still run pyannote-only diarization via
  `crispasr_diarize_segments_abi`; the embedder adds globally
  stable speaker IDs across long files.

Rust + Python are the canonical / "full-coverage" wrappers. The other
five track the high-traffic surface (transcribe + TTS) and were swept
together in `4f476c3` (set_speaker_name) and `65e0a61` (set_instruct +
variant detect).

### Capabilities reachable only from C-ABI / Rust / Python

For each, ~3-12 exports + an idiomatic result type per binding:

- **Forced alignment** — `crispasr_align_words`, `align_words_abi`,
  `align_result_*`. Word-level timestamps from a transcript + audio.
- **Diarization (segment-level)** — `crispasr_diarize_segments[_abi]`.
  Speaker segment spans via energy / xcorr / vad-turns / pyannote
  methods. Missing in **Java, Ruby, JS**.
- **Diarization (embedder + clustering)** — `crispasr_speaker_embedder_*_abi`,
  `crispasr_speaker_cluster_abi`, `crispasr_pyannote_cache_*_abi` (#107
  P6). Pluggable speaker-embedding adapters (TitaNet, IndexTTS-BigVGAN
  ECAPA-TDNN) + agglomerative cosine clustering + pyannote-seg cache
  for globally stable speaker IDs across long files. Missing in
  **Java, Ruby, JS** (which also lack the segment-level surface above).
- **Language ID** — `crispasr_detect_language[_pcm]`,
  `crispasr_lid_free_cache`. Pre-transcribe LID for routing.
- **VAD** — `crispasr_vad_segments`, `crispasr_compute_vad_slices`,
  `crispasr_stitch_vad_slices`, `crispasr_vad_remap_timestamp`,
  `crispasr_vad_free`. Standalone VAD + slice stitching.
- **Streaming** — `crispasr_stream_open/feed/get_text/flush/close`,
  `crispasr_stream_run_decode`. Online ASR with a step buffer. (PR #112
  `--stream-punc` is a CLI-orchestration flag, not a library surface
  — wrappers using `crispasr_stream_*` inherit no change.)
- **Punctuation** — `crispasr_punc_init/process/free/free_text`.
  FireRedPunc post-processor.
- **Model registry** — `crispasr_registry_lookup[_abi]`,
  `registry_lookup_by_filename[_abi]`,
  `crispasr_detect_backend_from_gguf`. Backend / file resolution.
- **Cache** — `crispasr_cache_dir_abi`,
  `crispasr_cache_ensure_file_abi`. Auto-download dir + lookup.

### #107 diarize-pipeline binding follow-up (deferred)

The full diarization surface — both the segment-level
`diarize_segments` family and the new #107 P6 embedder + clustering
+ cache primitives — landed in Python, Rust, Dart/Flutter, and Go.
Three bindings still have nothing wired:

- **Java** (`bindings/java/`) — JNI binding currently exposes only
  `crispasr_session_*` (transcription) and `*speaker_name*` (TTS
  preset-voice). Adding diarize means JNI wrappers for
  `crispasr_diarize_segments_abi` + the 9 new `crispasr_speaker_*_abi`
  / `crispasr_pyannote_cache_*_abi` exports plus an idiomatic Java
  helper class. ~250 LOC.
- **Ruby** (`bindings/ruby/`) — only exposes `Session.transcribe`.
  Treats `Segment#speaker_turn_next?` as the only speaker field
  (whisper tinydiarize). Wider diarize surface needs Ruby FFI
  bindings. ~200 LOC.
- **JavaScript / WASM** (`bindings/javascript/`) — emscripten build,
  no speaker surface at all today. Closing the gap depends on the
  WASM build expanding what it links in (pyannote-seg, titanet,
  indextts_voc all need to be in the WASM target). Bigger effort —
  start with `crispasr_diarize_segments_abi` (no model deps beyond
  the existing wasm whisper) and defer the embedder primitives.

Same "when to do this" rule as the rest of #59 applies: open when a
concrete consumer asks. The Python / Rust / Dart / Go quartet covers
the active CrispASR usage today.

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

### Follow-up: Rust binding directory location (low priority)

The Rust crates live at the repo root as `crispasr/` (high-level) +
`crispasr-sys/` (FFI). The crate **names** are correct and idiomatic
(`crispasr` / `crispasr-sys`, the `-sys` split, published on crates.io)
— **do not rename them**. The smell is purely the top-level *directory*
`crispasr/`, which visually collides with the repo/project name (a
reader at the root can't tell it's specifically the Rust binding vs the
core). It is, however, consistent with the repo's per-ecosystem
top-level pattern (`python/`, `flutter/`).

Optional cleanup: relocate **both** dirs (they're siblings; the
inter-crate dep is a relative `path = "../crispasr-sys"` and there is no
Cargo workspace, so moving them together preserves the link) under
`bindings/rust/` to match the C-family bindings. **Consumer-safe**:
crates.io consumers resolve by name+version, not in-repo path, so a
move does not break them. Before moving, audit: (a) any downstream repo
using a `git` + `path` dependency on the subdir (e.g. CrispEmbed /
CrisperWeaver), (b) internal CI / `scripts/` / `build_go` refs, (c)
docs path references. Do it deliberately in one commit — never a blind
rename. Not worth churn unless the root-dir ambiguity actively bothers.

---

## 60o. MTLBinaryArchive Metal pipeline cache — open

Parent #60 (cross-backend perf tricks) shipped 60a–g → HISTORY §63 /
§64 / §71 / §75 (madvise WILLNEED, wrap_iface, preload, fused QKV,
KV Q8_0 on 9 backends, mlock, MADV_RANDOM). 60h–n parked. Only 60o
below is still open.

### 60o (OPEN). MTLBinaryArchive Metal pipeline cache

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

## 65-residual. JS / emscripten word-accessor surface — open

Parent #65 (session-API word-confidence parity) shipped → HISTORY §65
(main batch + vibevoice / moonshine-streaming + gemma4-e2b token-prob
API + Go/Java/Ruby parity in `5534588` + `d963e3a`). Only residual:
JS/emscripten word accessors — leaving until a JS consumer asks (the
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

### 61h. Beam search for LLM family + enc-dec — DONE

_Done — see HISTORY.md + git log._

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

## 73-follow-up. Long-context cohere FA vs cast-on-read benchmark — DONE (2026-06-04)

Parent #73 (quant-safe KV cache write for canary / cohere / kyutai_stt)
shipped → HISTORY §79. #71 + #72 also there (test-runner under-invocation
+ cap-honesty audit; gemma4_e2b / mimo_asr GPU residency for Q4_K weights
— gemma4 2.2× on M1, mimo-asr -22 %, Linux/CUDA validation deferred).

**Benchmark result (VPS x86 CPU, 2 threads, cohere-transcribe-q4_k.gguf,
FLEURS EN):** flash wins by 26% on 300s audio (820s vs 1115s), loses by
13% on 60s audio (203s vs 179s). Crossover between 1-5 min. Flash
stays as default (`-fa`); short-clip users can opt out with `-nfa`.
Full results in PERFORMANCE.md §5.

## 74. Feature-matrix uplift round 2 — chatterbox family + matrix tooling ✓

After §79b shipped chatterbox + 3 sibling variants and the audit-drift cleanup brought test-all-backends.py to 39/39 backends, four follow-ups surfaced from re-reading the cap matrix. They cluster by user-visible value:

### 74a. Auto-route by `-l <lang>` for chatterbox family — DONE

_Done — see HISTORY.md + git log._

## 75-followups. /v1/audio/speech OpenAI round 2 — open

Parent #75 round 1 shipped → HISTORY §81 (PR #63 merged + corrective
batch + 75a/75b + 75d chunking + 75c-opt-1 server-side speed
resampler). Remaining follow-ups: 75c-opt-2 (native-backend duration
knobs) and 75e (streaming / mp3 / upload).

Remaining gaps documented in follow-up items: 75c-opt-2 (native-backend duration knobs), 75e (streaming response, mp3/opus encoding, voice upload/delete).

---

## 80. nano-cohere-transcribe-inspired perf + chunking tweaks

After studying [Deep-unlearning/nano-cohere-transcribe](https://github.com/Deep-unlearning/nano-cohere-transcribe)
(pure-PyTorch port of `CohereLabs/cohere-transcribe-03-2026`, 1.5–3.6×
faster than the native `transformers` path on CUDA), this entry surveys
which of its tricks port to CrispASR. The headline trick — CUDA-graph
capture of the per-token decoder step — turns out to be largely
redundant on Metal+ggml because PR #73's `sched_reserve` already paid
that bill. The smaller wins are still useful and are the focus here.

### 80a. Decoder graph-once, replay-N — measurement, parked

**Inspiration.** nano-cohere `_graph.py` pre-allocates fixed-shape KV
buffers `[B, H, max_kv, Dh]`, writes new K/V at a runtime `pos_idx` via
`index_copy_`, captures one CUDA graph per `(B, T_enc)`, and replays it
per token. Eliminates per-step kernel-launch overhead (3.62× win at
bs=1 on A100).

**Mapped to ggml.** The structural blocker in CrispASR is
`core/attention.h:148` — the F16 fast path bakes `n_past` into the
view byte-offset (`n_past * kv_k->nb[1]`), so each step rebuilds the
graph at a different topology. The quant path already uses
`ggml_set_rows(indices)` (line 173), the same indirection nano uses;
extending it to F16 is mechanical (verified: Metal/CUDA/Vulkan all
support `ggml_set_rows` with F16 destination). With that change, the
step graph could be built once and replayed N times, only updating
`embd`/`position`/`indices`/`sa_mask` via `ggml_backend_tensor_set`.

**Why this is parked.** Baseline measurement on cohere q4_k + Metal
(MTL0):

| clip            | total wall | enc compute  | dec build | dec alloc | dec compute | dec build+alloc / total |
| --------------- | ---------- | ------------ | --------- | --------- | ----------- | ----------------------- |
| JFK 11 s        | 1356.8 ms  | 1114.3 (82%) |   3.0 ms  |  14.8 ms  |  115.7 ms   | **1.31 %**              |
| JFK ×3 = 30 s   | 3838.6 ms  | 3295.7 (86%) |   5.1 ms  |  28.4 ms  |  293.7 ms   | **0.87 %**              |

The encoder dominates (82–86%). Per-step decoder CPU overhead is
already in the noise on Metal because PR #73's
`ggml_backend_sched_reserve(... max-size graph ...)` (cohere.cpp:2210)
pre-sizes the gallocr so `gallocr_needs_realloc` returns false on
every step. A graph-once-replay-N change would save at most ~1 % of
total wall, with substantial code disruption (constant-shape K/V
read views also need a per-step mask, since `sa_L = offset + n_tokens`
also currently scales the read view).

**Decision.** Park until a CUDA backend ships — at that point the win
flips (CUDA per-kernel-launch overhead is exactly what nano fights),
and the F16 `ggml_set_rows` path may be needed anyway. Cohere already
has the `gf_decode_1` field declared (line 502) for that future
change.

### 80b. Energy-minimum chunk boundaries — DONE

_Done — see HISTORY.md + git log._

## 81. Nemotron-Speech-Streaming-EN-0.6B — first cache-aware streaming-native ASR

[`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
— NVIDIA Open Model License (NVOML), 600 M params, released
late 2025 / early 2026. **Cache-Aware FastConformer + RNN-T**, the
first model in the queue that's *streaming-native* rather than
batch-with-chunked-streaming-on-top. Mentioned by an outside reporter
in issue #85; `parakeet` and `kyutai-stt` are the closest
streaming-capable backends we ship today and neither targets
nemotron's latency/accuracy frontier.

### Why this is interesting

Quality-vs-latency curve from the published Open ASR Leaderboard
numbers (same eval set as our existing parakeet / canary / whisper):

| Chunk size | Right-context lookahead | Avg WER | Notes |
|---|---:|---:|---|
| 1120 ms | 13 frames (~1.04 s) | 6.93 % | best accuracy |
| 560 ms | 6 frames (~0.48 s) | 7.07 % | |
| 160 ms | 1 frame (~0.08 s) | 7.67 % | |
| 80 ms | 0 frames | 8.43 % | unique low-latency point |

Same `.gguf` for all four — `att_context_size=[70, R]` is a runtime
knob, not a retraining artifact. Reference points on our current
leaderboard:

| Model (batch) | Avg WER | RTFx |
|---|---:|---:|
| `parakeet-tdt-0.6b-v3` (we ship this) | 6.34 % | high |
| `nemotron-streaming-en-0.6b` (1.12 s chunk) | 6.93 % | streaming |
| `canary-1b-v2` (we ship this) | 7.15 % | 749 |
| `whisper-large-v3` (we ship this) | 7.44 % | 145 |

Headline read: nemotron is **0.6 pp worse than batch parakeet on
average but better than canary-1b-v2 and whisper-large-v3** — and
crucially gets there with a fixed-size step rather than reading the
whole utterance. On AMI (conversational meetings) it actually wins
all of those (11.73 % vs 11.31/16.01/15.95). The 80 ms / 0-lookahead
/ 8.4 % WER point has no equivalent in our current lineup; it's the
real reason to consider this model.

### License — NVIDIA Open Model License (NVOML)

Source-available, **not** OSI-open-source.

- Commercial + non-commercial use ✅
- Derivatives + fine-tunes ✅ (you own them)
- Redistribution ✅ with attribution: every copy must include a
  `Notice` file containing *"Licensed by NVIDIA Corporation under
  the NVIDIA Open Model License"*
- No explicit field-of-use, **but** subject to NVIDIA's external
  "Trustworthy AI" terms
- Patent litigation = automatic termination
- Bypassing safety guardrails (without a "substantially similar
  Guardrail") = automatic termination
- NVIDIA claims no rights in outputs

For our purposes (publishing `cstr/nemotron-speech-streaming-en-0.6b-GGUF`):
legally fine — same shape as Llama / Cosmos redistributions. Just
need the NVOML attribution `Notice` in the HF README. Less
permissive than `parakeet-tdt-0.6b` (CC-BY-4.0) so downstream users
inherit the NVOML, not Apache.

### Architecture summary

```
WAV (16 kHz mono)
  → log-mel (80 bins, NeMo per-feature norm)
  → 4× conv subsampling pre-encode
  → 24× Cache-Aware FastConformer block
       · macaron FFN
       · multi-head SA w/ rel-pos shift  ← cache-aware (cached K/V from prior chunks)
       · depth-wise conv (kernel 9)      ← cache-aware (8-frame overlap-cache per layer)
       · macaron FFN
  → predictor LSTM (1-layer)
  → joint network + softmax
  → RNN-T blank/non-blank greedy (no TDT durations)
```

Frame stride after the 4× subsampling is 80 ms. Native PnC
(punctuation + capitalization). 530 k hours of training data
(NVIDIA Riva ASR set + Granary, including YouTube-Commons 109 k h,
YODAS2 102 k h, LibriLight 49 k h, Mosel 14 k h plus the standard
LibriSpeech / Fisher / WSJ / VoxPopuli / MLS / Common Voice /
Earnings22 mix).

### What we already have (~60–75 % reuse)

| Piece | Status | Where |
|---|---|---|
| FastConformer encoder body (24L, macaron FFN, SA + rel-pos, DW conv) | ✅ ready | `src/core/fastconformer.h` (shared by parakeet, canary, canary_ctc) |
| 4× conv subsampling pre-encode | ✅ ready | `core_conformer::build_pre_encode` |
| RNN-T predictor LSTM + joint head + greedy decode | ✅ ready (as TDT, where pure RNN-T = `n_tdt_durations=0`) | `src/parakeet.cpp` |
| Log-mel preprocessor (NeMo-style, 80 bins, per-feature normalization) | ✅ ready | `src/core/mel.cpp` |
| KV-cache infrastructure (`kv_cache_write` + offset-indexed reads) | ✅ ready | `src/core/attention.h` |
| Streaming CLI pipeline (`--stream`, `--stream-json` after #84) | ✅ ready | `examples/cli/crispasr_run.cpp` |
| BPE tokenizer (NeMo SentencePiece) | ✅ ready (parakeet pattern) | `src/core/bpe.h` |
| Streaming-aware backend example (per-layer state, chunk-by-chunk graph rebuild) | ✅ partial | `src/moonshine_streaming.cpp` |
| GGUF converter for NeMo `.nemo` checkpoints | ✅ partial | `models/convert-parakeet-to-gguf.py` template |

### What's missing — the actual new work

1. **Cache-aware FastConformer encoder graph.** Existing
   `core_conformer::build_block` consumes the whole `T` and emits
   the whole `T`; for streaming we need a variant that takes
   `(cached_K, cached_V, conv_state)` per layer and emits new K/V +
   new conv state alongside the audio output. Probably lives next
   to `fastconformer.h` as `fastconformer_streaming.h` so we don't
   regress parakeet/canary's bit-identical batch graphs.
2. **Per-layer streaming state on the context.** Modeled on
   `moonshine_streaming`'s pattern — per-layer K/V tensors persisted
   across `transcribe()` calls.
3. **`att_context_size` runtime knob.** Trivial — controls how many
   frames feed the encoder per step and the left-cache trim policy.
   No retraining, just masking + cache management differ.
4. **`.nemo` → GGUF converter.** ~80 % shared with parakeet's
   converter; cache-aware blocks have the same tensor names plus
   new metadata (`encoder.streaming.att_context_left_frames`,
   `encoder.streaming.dw_conv_state_size`, etc.).
5. **`examples/cli/crispasr_backend_nemotron.cpp`.** Thin adapter
   driving the streaming encoder per chunk and running the existing
   RNN-T greedy decode on each chunk's output. Shape is between
   `crispasr_backend_parakeet.cpp` and
   `crispasr_backend_moonshine_streaming.cpp`.
6. **`--live` integration.** Once the backend exists, `--stream` and
   `--stream-json` (#84) Just Work. The new backend would be a much
   better fit for `--live` than the current chunked-batch backends
   because per-chunk cost is constant rather than `O(window²)`.

### Effort breakdown

| Component | LOC | Reuse |
|---|---:|---|
| 80-bin log-mel front-end | ~30 | full reuse from parakeet |
| Cache-aware SA layer (K/V append + trim, rel-pos shift on `[cache | new]`) | ~120 | **new** — conceptually small but graph-topology-sensitive |
| Cache-aware DW conv layer (last 8 frames cached) | ~80 | **new** — pattern is identical to RNN state, just per-layer |
| Per-layer streaming state struct + lifecycle | ~60 | **new** — `nemotron_stream_state` on context |
| `att_context_size=[70,R]` runtime knob | ~30 | **new** — masking + trim policy |
| RNN-T predictor + joint + greedy (no durations) | ~50 | full reuse from parakeet |
| `.nemo` → GGUF converter | ~200 | ~80 % parakeet template + new streaming metadata |
| Diff harness reference (mel, enc per chunk, predictor, joint) | ~150 | parakeet template |
| Backend wrapper for main CLI | ~250 | midway between parakeet + moonshine_streaming |
| HF README + NVOML notice + auto-download registry entry | ~80 | template |
| **Total** | ~**1000–1300 LOC** | smaller than #58 (no LM body, no DeepStack injection) |

Headline new helper: a **streaming-aware FastConformer block builder**
that's reusable for any future cache-aware NeMo model (the same
streaming protocol covers Parakeet-Streaming, Canary-Streaming,
FastConformer-CTC-streaming if NVIDIA ever ships those — they're
rumored in the NeMo 26.x roadmap).

### What we'd need to dump from the Python ref for the diff harness

Stage taps:
- `mel_in` `[T_mel, 80]` — first chunk only
- `enc_pre_encode` `[T_enc, 512]` — after 4× subsampling
- `enc_block_0_post_sa` `[T_enc, 512]` — after first SA block
  (validates rel-pos + cache concat)
- `enc_block_0_post_conv` `[T_enc, 512]` — after first DW conv
  (validates conv-state cache)
- `enc_out` `[T_enc, 512]` — final encoder output for the chunk
- `predictor_state_after_blank` `[1, 640]`
- `joint_logits_step0` `[1, vocab+1]`

Six stages — a touch lighter than parakeet's diff harness because we
share the mel front-end. Multi-chunk validation needs the diff
harness to run twice with carried state (or a third entry point that
takes a list of chunks); the current `crispasr-diff` is single-shot
so this is a small extension to the harness itself.

### Risks / open questions

1. **Cache layout in the GGUF.** NeMo's `.nemo` archive embeds the
   streaming config but not the runtime cache shape. Need to pin
   the cache tensor layout (per-layer `[Dh, 70, n_heads]` for SA K
   and V, `[8, d_model]` for DW conv state) and check it survives
   the converter round-trip.
2. **Rel-pos shift on `[cache | new]`.** ~~The existing
   `core_conformer::rel_shift` operates on a square `(2T-1, T)`
   tensor~~ **SOLVED 2026-06-15.** The asymmetric rel_shift uses the
   same stride-trick formula as the symmetric case:
   `view_3d(BD_raw, T_full, T_new, H, s1-s0, s2, (T_new-1)*s0)`.
   Implemented in `nemotron_build_block_streaming`.
3. **Decoder state across chunks.** RNN-T predictor is autoregressive
   over emitted tokens, not over time; its LSTM state carries
   forward across chunks naturally. Confirm with the Python ref
   that emit-token-then-blank-to-advance still works at chunk
   boundaries (the joint sees encoder output for the current
   chunk; the predictor sees its state from the last emitted
   token, which may be from a previous chunk).
4. **Runtime tradeoff at 80 ms.** A 5-chunk window of left context
   + 1 frame new = 71 frames per encoder forward. On Apple Silicon
   Metal the per-launch overhead at that small a `T` may dominate
   — measure before claiming the 8.4 % WER point is actually usable
   live. CUDA path likely fine (graph capture amortizes the launch
   cost the way nano-cohere relied on, see #80a).
5. **PnC-stripped vs PnC-emitting WER comparison.** Open ASR
   Leaderboard scoring strips PnC; the 6.93 % avg is on stripped
   text. Our `firered_punc` post-processor adds PnC to backends that
   don't emit it natively — for nemotron we'd want to *not* run it
   (the model emits PnC already and re-punctuating can hurt). Wire
   a backend capability flag so `--punctuation` is a no-op when the
   backend is PnC-native.

### When to do this

**Un-deferred 2026-06-12** — dictation use case is the concrete
demand. The streaming pipeline has had weeks of real use since
`--stream-json` (#84) landed. The 80 ms / 0-lookahead / 8.4 % WER
operating point has no equivalent in our lineup and is the right
fit for live dictation where sub-200 ms latency matters.

**Realistic estimate 3–5 days of focused work** for someone who's
already touched parakeet/canary, plus benchmarking against the
upstream Open ASR Leaderboard table to confirm parity on the
published WERs. The model is `nemotron-3.5-asr-streaming-0.6b`
(multilingual, 39 langs incl. de-DE, OpenMDW-1.1 license).

**2026-06-13 progress:** Scaffold done (converter, runtime, CLI, registry).
Pre-encode validated cos=1.0. Root cause of empty output identified: the
model is **streaming-only** (`att_context_style=chunked_limited`). Running
full bidirectional attention produces out-of-distribution activations
where blank wins at every frame. The next step is implementing the
cache-aware chunked encoder: per-layer K/V cache (56 frames left context)
+ conv state (8 frames) + attention masking to limit context window.
Kaggle diff harness (v8) uploaded ref GGUF to `cstr/nemotron-3.5-asr-streaming-GGUF`.

**2026-06-14 progress:** Fixed tensor name mismatches, prompt_id mapping,
mel frame count, pre-encode causal padding. Confirmed via Kaggle NeMo
ground truth that encoder output range [-0.91, 0.51] matches exactly
and NeMo's decoder produces correct text. Per-frame values diverged
due to the conformer conv module bug.

**2026-06-15 — WORKING.** Root cause found and fixed: **GLU gate/value
swap** in the conformer conv module. `ggml_siglu` does
`sigmoid(first) * second` but NeMo's `glu` does `first * sigmoid(second)`.
Fix: `ggml_siglu` → `ggml_siglu_swapped`. One-line change.

Output: "And so my fellow Americans ask not what your country can do for
you. Ask what you can do for your country." — matches NeMo exactly.

Additional fixes along the way:
- Tensor names: conv.bn→conv.ln, prompt_kernel.linear1→prompt_kernel.0
- Prompt_id: NeMo uses en-US=0 (not alphabetical index 7)
- Mel: drop_last_frame=false (NeMo keeps all frames)
- Pre-encode: ggml_pad_ext for true causal padding, F32 weights in converter
- Chunked_limited attention mask (chunk-based, not banded)

**Cleanup done (2026-06-15):**
- Debug prints removed, Q4_K dequant support added to tensor_to_f32
- F16 and Q4_K both produce correct text (Q4_K ~2x faster: 14s vs 24s)
- Fixed GGUFs uploaded to HF, broken ref GGUF deleted
- Streaming block GLU also fixed (ggml_siglu → ggml_siglu_swapped)
- Bidirectional+mask path is now the default (no NEMOTRON_BATCH needed)
- Worktree cleaned up, HF README updated with Q4_K recommendation
- German tested (works on DE audio; EN audio with DE prompt gives
  expected degradation)

**2026-06-15 streaming + sched:**
- Sched migration done (gallocr → ggml_backend_sched) — see §168
- Streaming encoder architecture complete: `nemotron_build_block_streaming`
  with cache_last_channel, asymmetric rel-pos bias (stride-trick rel_shift),
  Q-from-new / KV-from-context, causal conv padding
- `CRISPASR_NEMOTRON_STREAMING=1` + `CRISPASR_NEMOTRON_CONTEXT_PRESET=N`
  env vars for A/B testing (0=R3/chunk4, 3=R13/chunk14)
- Quality problem: streaming output diverges from full-sequence by frame 10;
  preset 3 gives recognizable text, preset 0 gives blank. See §168 for
  root cause analysis.
- 3 live integration tests added (init, JFK, F16/Q4_K parity)

**2026-06-15 — STREAMING FIXED (`7f4feff9`).** Root cause: conv module
zero-padded K-1=8 frames instead of prepending cached pre-DW-conv signal.
NeMo's `CausalConv1D.update_cache` does `torch.cat([cache, x])`.
All 4 presets now produce correct text:
- Preset 0 (R=3, 160ms): "And so, my fellow Americans..."
- Preset 3 (R=13, 1120ms): "And so, my fellow Americans..."
- F16 preset 0: "And so my fellow Americans..."

**Remaining:** WER benchmarking on standard test sets, GPU perf testing,
consider making streaming the default path.

---

## 168. GPU scheduler migration — gallocr-only backends

**Status:** partially done (4/7 migrated, streaming encoder incomplete).

### Completed (2026-06-15, commits `f3a57d7c`, `d393b506`, `8f481aa2`)

Four backends migrated from `ggml_gallocr` to `ggml_backend_sched`:
- **nemotron** — encoder, chunked encoder, pre-encode graph
- **paraformer** — encoder, decoder
- **dia_tts** — encoder, cross-attn, decoder loop, DAC decode
- **outetts_wavtok** — decode graph

All verified: identical output on CPU, ASR roundtrip for TTS, 435 unit tests pass.

### Second batch (2026-06-15, commit `f748b94d`)

Two more backends migrated:
- **audioseal** — generator encode + detect graphs
- **lfm2_audio** — encoder, backbone, adapter graphs; F16 verified

**lfm2 Q4_K finding:** the published `lfm2-audio-1.5b-q4_k.gguf` was quantized
with an older quantizer that Q4_K'd `lfm.embed_tokens` (should be F16). Even
after re-quantizing with correct rules (embed_tokens F16), Q4_K still produces
0 tokens — the hybrid conv+attention backbone is too precision-sensitive. **Q5_K
works** and gives identical output to F16. Need to re-publish GGUFs on HF with
Q5_K as minimum quant.

### Remaining gallocr backends

| Backend | gallocr | init_best | Status |
|---------|:-------:|:---------:|--------|
| `voxcpm2_tts` | 27 | ✅ | persistent gallocr with reserve — complex migration |

### Nemotron streaming encoder — QUALITY PROBLEM

The streaming (chunked) encoder path is architecturally complete:
- `nemotron_build_block_streaming`: FFN1 new-only, Q-new/KV-context,
  causal conv, FFN2+LN new-only, cache_last_channel
- Asymmetric rel-pos bias via stride-trick rel_shift (same formula as
  symmetric, offset = T_new-1 instead of T-1)
- Gated by `CRISPASR_NEMOTRON_STREAMING=1`

**FIXED (2026-06-15, `7f4feff9`).** Root cause was hypothesis #3: the conv
module zero-padded the left by K-1=8 on every chunk instead of prepending
the cached pre-DW-conv signal from the previous chunk. NeMo's
`CausalConv1D.update_cache` does `torch.cat([cache, new_x], dim=-1)` — our
code was using `ggml_conv_2d_dw_direct(..., pad_left=K-1)` which fills zeros.

Fix: added `conv_cache_in` / `conv_cache_out` to the streaming block. Each
layer stores the last K-1 frames of the post-GLU signal and prepends them
on the next chunk. All 4 context presets now produce correct text.

### Env vars added

| Var | Effect |
|-----|--------|
| `CRISPASR_NEMOTRON_STREAMING=1` | Enable streaming chunked encoder |
| `CRISPASR_NEMOTRON_CONTEXT_PRESET=N` | Attention context preset (0-3) |
| `CRISPASR_NEMOTRON_NO_WINDOW_MASK=1` | Disable banded attention mask |
| `CRISPASR_NEMOTRON_DEBUG=1` | Encoder/decoder debug prints |

---

## 86. Per-backend flash-attention wiring (CrisperWeaver-driven)

**Status:** open. Plumbing complete in 0.6.2 (commit `ff5536a6`),
kernel-level wiring per backend is the remaining work.

### Context

CrisperWeaver's *Settings → Performance → ASR flash-attention*
toggle ships via `crispasr_session_open_with_params` (open params
struct v2 added in 0.6.2). The toggle threads through to a thread-
local `g_open_flash_attn_tls` that every backend's init arm can
read to set `cparams.flash_attn` on its context_params struct.

**Today only whisper actually consumes the flag at the kernel
level** — `whisper_context_params` already has a `flash_attn` field
that whisper.cpp branches on internally (it switches to
`ggml_flash_attn_ext` for the QKV → softmax → V product when set).

Other backends accept the toggle but their compute graphs still
build the historical `ggml_soft_max_ext(KQ)` path. For users on
Metal, that's a measurable perf gap on every long-running LLM
backend (orpheus, voxtral, qwen3 ASR, qwen3-tts, granite-speech,
chatterbox-T3, gemma4-e2b).

### Per-backend status (updated 2026-05-23)

**RESOLVED.** All backends now route through shared core modules
(`core_attn::kv_self_attn`, `core_attn::encoder_self_attn`,
`core_sanm::build_block`, `core_conformer::build_block`) that
unconditionally use `ggml_flash_attn_ext`. The table below was
written before the core helpers were consolidated; by the time
individual wiring was attempted (May 2026), every backend already
had flash attention via its core module.

| Backend | Core module | Flash attn | Status |
|---|---|---|---|
| whisper | upstream whisper.cpp | ✅ | DONE (upstream) |
| parakeet | `core_conformer::build_block` | ✅ | DONE |
| canary | `core_attn::kv_self_attn` | ✅ | DONE |
| qwen3 (asr) | `core_attn::kv_self_attn` | ✅ | DONE |
| cohere | `core_attn::kv_self_attn` | ✅ | DONE |
| granite_speech | `core_attn::kv_self_attn` + `core_conformer_ibm` | ✅ | DONE |
| voxtral | `core_attn::kv_self_attn` | ✅ | DONE |
| voxtral4b | `core_attn::kv_self_attn` + `encoder_self_attn` | ✅ | DONE |
| vibevoice | `core_attn::kv_self_attn` | ✅ | DONE |
| qwen3_tts | `core_attn::kv_self_attn` | ✅ | DONE |
| orpheus | `core_attn::kv_self_attn` | ✅ | DONE |
| kokoro | `core_attn::encoder_self_attn` | ✅ | DONE |
| chatterbox | `core_attn::kv_self_attn` | ✅ | DONE |
| funasr | `core_sanm::build_block` + `core_attn::kv_self_attn` | ✅ | DONE |
| sensevoice | `core_sanm::build_block` | ✅ | DONE |
| paraformer | `core_sanm::build_block` + manual cross-attn | ✅ | DONE |
| t5_translate | manual (T5 rel-pos bias) | ❌ | N/A — T5 additive bias incompatible with fused kernel |

No further work needed. The original "recommended order" is moot.

### Effort estimate

~4–6 hours per LLM backend (orpheus/voxtral/qwen3/granite/chatterbox),
~2 hours per Conformer (parakeet/canary/cohere). Total realistic
sweep: 2–3 focused days, can be split across separate PRs since
each backend is independent.

---

## 87. `gpu_backend` runtime selector (multi-backend ggml build)

**Status:** open. Needs ggml-side support to land first.

### Context

CrisperWeaver's *Settings → Performance* exposes an "ASR on GPU"
boolean today. The deeper knob — picking BETWEEN Metal, CUDA,
Vulkan at runtime when more than one is built into libcrispasr —
isn't doable yet because each `*_init_from_file` calls a single
`ggml_backend_*_init()` directly, and the CMake flag picks which
ggml backend gets compiled in.

### What ggml supports today

`ggml_backend_*_init()` returns a per-backend handle. Multiple
backends CAN compile into one binary (`-DGGML_METAL=ON
-DGGML_VULKAN=ON` builds both); each backend's init function lives
behind its own `#ifdef`, and the runtime can call any of them. What
ggml doesn't yet have is a uniform "auto-pick the best available"
selector — that's the missing piece.

### Approach when we tackle it

1. Add `crispasr_select_backend(const char* hint)` helper that
   resolves a hint string (`"auto" / "metal" / "cuda" / "vulkan" /
   "cpu"`) to a `ggml_backend_t` using `#ifdef GGML_METAL` /
   `#ifdef GGML_CUDA` / `#ifdef GGML_VULKAN` chains. `auto`
   prefers Metal on macOS, CUDA on Linux+NV, Vulkan on Linux+AMD
   or Windows, CPU as fallback.
2. Refactor every per-backend `*_init_from_file()` to take a
   `ggml_backend_t* preferred_backend` param (or read a thread-
   local set by a new `crispasr_session_open_params_v3`).
3. Add `gpu_backend_hint` (string field) to the open params struct
   v3.
4. Plumb through CrisperWeaver's `AdvancedTranscribeOptions` →
   `LoadModel` → `openWithParams`.

### Effort estimate

~1 week of focused work — touches every backend's init path. Best
done as a separate phased PR, one backend per commit.

---

## 90. Session-API beam_size — per-backend wiring

May 2026:
  * Shipped `crispasr_session_set_beam_size` (commit 958e6bd7).
  * Whisper consumes it natively (switches sampling strategy to
    BEAM_SEARCH with the supplied width).
  * **Five backends wired** via runtime `<backend>_set_beam_size`
    setters in their per-backend C API:
    - kyutai-stt (also kyutai / moshi-stt aliases) — setter
      pre-existed; just needed to be called from
      `transcribe_single`.
    - moonshine — same.
    - omniasr (LLM variant; CTC ignores) — same.
    - **glm-asr** — added `glm_asr_set_beam_size` (new public
      symbol) + dispatch wire.
    - **firered** — added `firered_asr_set_beam_size` (new
      public symbol) + dispatch wire.
  * CrisperWeaver's batch worker pool drives all six (whisper +
    five) end-to-end via its sticky-setter protocol; beamSearch
    is pool-eligible across that whole set.
  * `nm libcrispasr` confirms all five `<backend>_set_beam_size`
    plus the unified `crispasr_session_set_beam_size` are
    exported.

**DONE (2026-05-23, commit `0c24178e`).** All three remaining backend families wired:

| Backend | Approach |
|---|---|
| qwen3-asr | Beam branch in `crispasr_c_api.cpp` session path: replay via `qwen3_asr_embed_tokens` + `qwen3_asr_run_llm_kv`; `kv_reset` after beam search. |
| granite / granite-4.1 / granite-4.1-plus / granite-4.1-nar | Same pattern: `granite_speech_embed_tokens` + `granite_speech_run_llm_kv`; `kv_reset` after beam. |
| voxtral | `run_voxtral_family` gained a `beam_size` param; decode-piece logic factored into a shared `decode_piece` lambda (U+2581→space detokenisation) used by both beam and greedy paths. |

`voxtral4b` uses a streaming path, not `run_voxtral_family` — not in scope for this item.

**Parakeet TDT/RNNT beam search shipped 2026-06-01 (`b3cdcebd`, issue #136).**
Label-looping beam with per-beam LSTM state snapshots and per-beam
hotword trie tracking. Wired via `parakeet_set_beam_size()` C API
and `--beam-size` / `-bs` CLI flag. Overhead: ~3 % at beam=2, ~12 %
at beam=4 (encoder-dominated pipeline). See §139 for remaining gaps.

`s->beam_size == 1` (default) keeps the existing greedy path bit-identical; no regression.

**Functional regression test added (2026-05-30).**
`tests/test-session-beam.cpp` — Catch2 test with two tiers:
  - `[unit][beam]` — setter API (null guard, width clamping). No model.
  - `[beam][.live]` — end-to-end via session API. Gated on
    `CRISPASR_MODEL_WHISPER` / `CRISPASR_MODEL_GLM_ASR` env vars.
    Verifies: beam_size=1 byte-identical to default (no-regression),
    beam 2–4 produce non-empty well-formed output on jfk.wav.

---

## 91. CrispASR CLI features missing from CrisperWeaver

While auditing the feature matrix for §90 we found a handful of
CLI knobs CrisperWeaver doesn't expose. Tracked here so the
gap is visible:

* `--offset-t MS` / `--duration MS` — process only a time window
  of the audio. Useful for "transcribe minute 5–10" workflows.
  Needs an engine-side `audio[t0:t0+d]` slice + timestamp shift
  (similar mechanics to the existing resume-offset routing).
  Estimate: 1 day end-to-end (CrispASR Dart binding + UI).
* ~~`--alt N` / `--alt-n N`~~ — alternative-candidate tokens —
  **shipped May 2026 (0.5.13 + CrisperWeaver §5.8)**. Whisper
  internals carry a parallel `alts` vector on `whisper_segment`
  (mirrored at every `tokens.{clear,push_back,resize}` site
  through the fallback-temperature loop, `result_len`
  truncation, and `max_len` wrap-segment splitter). New
  `wparams.alt_n` (default 0 = off). Capture happens inside
  `whisper_sample_token` — beam search is excluded because
  siblings are beam-conditional rather than greedy
  alternatives. Six new public getters
  (`whisper_full_get_token_n_alts` / `_alt_id` / `_alt_p` +
  `_from_state` variants); new C-ABI for both the low-level
  (`crispasr_token_n_alts` / `_alt_id` / `_alt_p` /
  `_alt_text`) and the unified session result
  (`crispasr_session_result_word_n_alts` / `_alt_text` /
  `_alt_p`); sticky session setter
  `crispasr_session_set_alt_n`. The whisper session-transcribe
  path now also populates `seg.words` via
  `emit_words_from_tokens` (it previously emitted only
  segment text — closing a long-standing gap with the parakeet
  / canary backends as a side benefit). Dart 0.5.13 + smoke
  test pinned. CrisperWeaver surfaces this as
  `AdvancedOptions.altN` (0..5 slider in the Whisper-only
  section) and a tap-to-pick chip row in the segment edit
  dialog.

  Deferred follow-ups (low priority — v1 covers the common
  case):
  - **Beam-search alt capture.** Siblings ≠ greedy
    alternatives; needs a different capture path and a
    different chip-walk UX. Defer until a user actually asks.
  - **Full word-level alt enumeration.** Sub-word BPE means
    multi-token words ("kubectl" → `["kub","ect","l"]`) only
    surface alts for the first content token. Whole-word
    alternates would need a per-word token-tree expansion —
    not free; v1's first-token alts already catch most real
    mishears.
  - **Widget test for the alt-picker popover** (CrisperWeaver
    side; Riverpod + l10n scaffolding nontrivial).
  - ~~**Live end-to-end test**~~ — **shipped** as
    `flutter/crispasr/test/alt_tokens_live_test.dart`. Opens a
    session against `ggml-tiny.en.bin`, sets `altN=3`,
    transcribes `samples/jfk.wav`, and asserts ≥1 word has
    alts, p ∈ [0, 1] and descending, chosen token excluded,
    `setAltN(0)` actually clears on the next decode. Tagged
    `live` so model-less CI still passes. On the dev box
    whisper-tiny gives 22/22 words runner-ups on JFK with
    real morphological alternatives like "Americans →
    America / americ / American".
* Whisper decoder fallback knobs (`--word-thold`,
  `--entropy-thold`, `--logprob-thold`, `--no-speech-thold`,
  `--no-fallback`, `--temperature-inc`) — already in the Dart
  binding's TranscribeOptions, just not exposed in
  CrisperWeaver's Advanced Options widget. Trivial — half a day
  to add the UI rows + localised strings.
* Subtitle line formatting (`--max-len`, `--split-on-word`,
  `--split-on-punct`) — whisper context-params field today;
  CrisperWeaver formats post-hoc instead. Quality-of-life win
  for SRT export. ~1 day.
* Token suppression (`--suppress-nst`, `--suppress-regex`) —
  niche; whisper-specific.
* `--carry-initial-prompt` — sticky vs reset behaviour for the
  initial prompt across segments. Edge case, ~1 hour.
* `--print-confidence` — per-token confidence in JSON / WTS
  exports. Segments already carry a `confidence` field;
  exporters could surface per-token.

None of these are blocking; they're listed so the next
parity-pass audit doesn't have to re-discover them.

---




## 92. All-backend regression suite (nightly CI)

**Status:** 32 ASR + 21 TTS regression entries, 0 PLACEHOLDERs.
CI workflow at `.github/workflows/regression.yml` runs nightly + on PR
(smoke-only). Matrix: 22 ASR + 7 TTS = 29 backends on GH free tier.
**First nightly run (2026-06-16):** smoke+preflight+14 backends PASS.
6 failures diagnosed and fixed: 4 decode-drift backends got WER
tolerance (`transcript_tolerance`), 1 bad revision SHA corrected,
1 HF rate limit (transient). Re-triggered; expecting full green.
Next: flip `skip_diff` on backends where ref archives exist, promote
to release gating.

**Why:** the ggml-assertion-hardening regression in 0.6.x cycle
demonstrated that we silently inherit upstream behaviour changes —
both in ggml and in HF-hosted weights — without any test catching
them. A nightly regression that pins every artifact to a specific
revision SHA closes that gap.

**Architecture (shipped, see `tests/regression/`):**

```
manifest.json    per-backend pins: GGUF revision + reference path
                 + expected transcript + cosine thresholds
run_one.py       driver: HF download (pinned) → crispasr (assert
                 transcript) → crispasr-diff (assert cos_min)
regression.yml   nightly cron at 04:00 UTC, matrix per backend
```

Reference dumps live in
[`cstr/crispasr-regression-fixtures`](https://huggingface.co/cstr/crispasr-regression-fixtures);
the `fixtures.revision` SHA in `manifest.json` pins the whole
fixture set together so a re-dump can't silently shift CI's
expectations.

**Next steps (each ~1 hour per backend):**

1. **Add parakeet-tdt-0.6b-v3** (English). Need to cache the
   `.nemo` source locally first; `nvidia/parakeet-tdt-0.6b-v3`
   isn't downloaded on the dev box right now.
2. **Add canary** + **cohere** + **kyutai-stt** + **moonshine**.
   All have reference modules in `tools/reference_backends/`.
3. **Add the TTS family** (kokoro, indextts, qwen3-tts, chatterbox,
   vibevoice). These need WAV-output checksums or an
   ASR-roundtrip rather than transcript equality.
4. **Promote to release gating.** Once stable, hook into
   `release.yml`'s pre-publish job so a regression aborts the
   tag.

**Time budget per nightly run:** ~5-15 min per backend (download
+ build cache hit + run). With matrix fan-out, wall time stays
under 30 min for the full suite.

**Storage:** fixtures repo grows by ~1-50 MB per backend. After
20+ backends still well under 1 GB. The GGUF cache lives in
`$RUNNER_TEMP` and is discarded with the runner — no GitHub
Actions cache eviction concerns.

**Cost on free tier:** public-repo Actions minutes are unlimited,
so the only constraint is wall-clock fairness on the shared
runner pool. Nightly cadence is the sweet spot — catches
regressions within 24 h without burning capacity that would
delay PR feedback.




## 93. CMake target rename: `crispasr` → `crispasr-lib`

**Status:** DONE (commit `11148b23`).

**Why:** the CMake target `crispasr` produces the **library**
(`libcrispasr.so`), while the CLI **binary** is produced by target
`crispasr-cli` (which outputs `bin/crispasr`). The target name vs.
output binary name divergence is a long-standing trap that has now
caused two CI regressions this session:

  - GH regression workflow run 25735206584 — built only the library,
    found `bin/crispasr` absent (fixed in commit `08d1872f`).
  - Kaggle `crispasr-regression-suite` run on 2026-05-12 14:34 —
    same root cause (fixed in this commit, applied to
    `tools/kaggle/crispasr-regression.py`).

Both fixes are one-line and obvious **after** you know — but the
trap reliably re-bites anyone writing a new CI workflow because
the natural mental model is "target `crispasr` builds the
`crispasr` binary." Renaming the library target makes the
distinction explicit.

**Plan:**

1. `src/CMakeLists.txt`: rename `add_library(crispasr ...)` →
   `add_library(crispasr-lib ...)`. Update every
   `target_link_libraries(... crispasr ...)` and any
   `target_*(crispasr ...)` call elsewhere in the tree.
2. Preserve the **output binary name** `libcrispasr.{so,dylib,a}`
   via `set_target_properties(crispasr-lib PROPERTIES OUTPUT_NAME
   crispasr)` — no .so/.dylib filename change, no ABI break.
3. Update consumers that use the CMake target name:
   - `bindings/go/whisper.go` cgo LDFLAGS (currently `-lcrispasr`,
     unchanged since OUTPUT_NAME stays `crispasr`).
   - `bindings/ruby/ext/dependencies.rb` graphviz walk (queries by
     target name `crispasr` — needs a 1-line update).
   - `.github/workflows/{ci,release,regression}.yml` `--target`
     args (mostly already use `crispasr-cli` for the CLI binary,
     but any `--target crispasr-lib` referring to the library needs
     the rename).
   - `tools/kaggle/crispasr-regression.py` similarly.
4. Add a CMake alias for one release cycle:
   `add_library(crispasr ALIAS crispasr-lib)` so external repos
   that depend on `target_link_libraries(... crispasr)` keep
   working while they migrate.

**Effort:** ~2 hours including consumer audit + CI re-runs.
Drop-in once green on every workflow.

**Don't do this in a patch release.** Even with the alias the
churn is visible to anyone bisecting a build issue. Schedule for
the next minor (0.7.0).




## 94. Auto-generate Go bindings `#cgo LDFLAGS` from CMake graphviz — DONE

_Done — see HISTORY.md + git log._

## 95. IndexTTS-1.5 Chinese TN — binary alternative to the Python `wetext` hook

**Status: open.** Today CrispASR ships `INDEXTTS_TEXT_NORMALIZER=<shell
cmd>` plus `tools/wetext-normalize.py` (commit `1bfe7c5a`,
2026-05-19). That covers users who already have Python + wetext
installed. Some deployments (single-binary distribution, Windows
without Python, embedded) need a no-Python path. This section
catalogs the realistic options so the next person doesn't have to
re-do the survey.

### The actual functionality gap

Default in-process `preprocess_indextts_text()` handles:
- CJK char split (port of upstream `tokenize_by_CJK_char`)
- Subset of `char_rep_map`: `，。：；！？、…“”‘’（）《》【】「」—～·` → ASCII
  before the CJK splitter
- ASCII upper-case

What it does NOT handle (full `wetext.Normalizer(lang='zh', operator='tn')`):
- Arabic numeral → hanzi (`2025年` → `二零二五年`, `123` → `一百二十三`)
- Pinyin tone-digit restoration (`xuan4`, `受不liao3` patterns)
- Dates (`2025/01/11` → `二零二五年一月十一日`)
- Times (`8:00` → `八点`)
- Currency (`¥12999` → `一万二千九百九十九元`)
- Phone numbers (`13800001234` digit-by-digit)
- Math / measurements / fractions / percent
- English contractions inside Chinese text

The model itself can't pronounce raw digits cleanly — it often fails
to emit `stop_mel_token` on un-pronounceable inputs and burns through
`max_mel_tokens=600` before giving up. So this isn't pure cosmetics:
TN is what makes digit-containing Chinese prompts actually work.

### Options, in order of pragmatic preference

**95a. Hand-roll the high-leverage rules in C++** (recommended first
step). Target the failures that actually break TTS — start with
digit-string → hanzi for the 1-billion range (`零一二三四五六七八九`
plus units `十百千万亿`), the `年/月/日` pattern, the `点/分` time
pattern, and pinyin tone-digit lookup. Estimated 300-600 LOC of
focused C++, no external dependency. Covers ~90 % of real prompts.
Edge cases (currency formats, mixed math, address parsing) silently
stay un-normalized. Lives in `src/indextts.cpp:preprocess_indextts_text`
or a sibling helper.

Files:
- `src/indextts.cpp` — extend the preprocessor with a
  `normalize_chinese_numbers()` pass that runs before the existing
  CJK split.
- `tests/` — golden inputs for the digit/date/time cases (just
  string-in, string-out; no model needed).
- `LEARNINGS.md` — new sub-section once the first user-reported
  edge-case lands so the next contributor knows what's covered.

When to do it: when an issue lands like "我有 3 个苹果 → weird audio"
or `2025年` produces a hang. Don't start it speculatively.

**95b. Vendor `kaldifst` (the C++ WFST runtime) + OpenFST + ship the
compiled `.fst` rule data.** The byte-identical-to-upstream path.

Ingredients:
- `kaldifst` C++ class `TextNormalizer` (k2-fsa/kaldifst on GitHub,
  a few thousand LOC of C++ wrapped around OpenFST). Apache-2.0.
- `OpenFST` (~30-50 K LOC of well-defined C++). Apache-2.0.
  Builds cleanly as a CMake subproject but the build profile cost
  is real — CrispASR today has no WFST dependency.
- Chinese TN rule data from `pengzhendong/wetext`:
  `fsts/zh/tn/tagger.fst` (812 KB) + `fsts/zh/tn/verbalizer.fst`
  (88 KB) + optionally `verbalizer_remove_erhua.fst` (88 KB).
  Plus the orchestration glue from `wetext.utils.normalize` (the
  preprocess → tagger → token-parser → verbalizer → postprocess
  flow) — `token_parser.py` is ~200 lines of recursive-descent
  parsing of the tagger output that would need to be rewritten in
  C++.

Estimated effort: 3-5 days of focused work for someone comfortable
with OpenFST. Result: same output as `wetext.Normalizer` byte for
byte, no Python at runtime.

When to do it: only after #95a has grown past ~5 hand-rolled cases
and the maintenance burden becomes visible — or if a downstream
deployment specifically can't take a Python dependency. Don't
speculate.

Files:
- `third_party/openfst/` — submodule (~50 K LOC, large diff).
- `third_party/kaldifst/` — submodule (~5 K LOC).
- `models/indextts-zh-tn.fsts` or co-distributed via `-m auto` —
  the ~1 MB of compiled FST data.
- `src/indextts.cpp` — new `normalize_chinese_wetext()` function
  invoked when `INDEXTTS_TEXT_NORMALIZER=wetext` (a sentinel
  value) is set, alongside the existing shell-command form.
- CMakeLists.txt — `add_subdirectory(third_party/openfst)` +
  link against kaldifst.

**95c. Static-bundle the Python sidecar via PyInstaller /
cibuildwheel.** Ship Python + wetext + dependencies as a
~50-80 MB single binary co-distributed with `crispasr`. Solves
the "no Python on the box" use case without porting any code.

Major downsides:
- Build system gets meaningfully more complex (a separate Python
  bundling pipeline per platform).
- ~50-80 MB of bloat per release.
- Not idiomatic for a C++ project; CrispASR's distribution story
  today is "one binary + model files".

Almost certainly **not** worth doing. Listed only so future
contributors don't spend time discovering it independently.

**95d. Tiny FST reader in our own C++ — consume OpenFST's
binary `.fst` directly without linking OpenFST.** Middle ground
between #95a (hand-roll rules) and #95b (vendor 30-50 K LOC of
OpenFST + kaldifst). Same upstream rule data as #95b — `.fst`
files from `pengzhendong/wetext` — but skip the libraries.

Why this might be the right shape: the wetext / kaldifst rule
chain is small (one tagger + one verbalizer, no runtime
composition needed if we pre-compose at build time), and
OpenFST's `.fst` binary format is documented in the OpenFST
manual. A single-pass interpreter that reads StdVectorFst
(the variant the wetext compiler emits) and does deterministic
traversal is genuinely small.

Sketch:

- **Parser** (~200 LOC). OpenFST `.fst` header is fixed-layout
  (magic, version, fst-type, arc-type, properties, n_states,
  flags); per-state records carry `final_weight` + arc lists;
  per-arc records carry `ilabel`, `olabel`, `weight`,
  `nextstate`. The wetext rule files use `StdVectorFst`
  (TropicalArc) — uniform fixed-size records. Read once at
  startup into a flat `std::vector<State>` with arc slices.
- **Traverser** (~150 LOC). For each input UTF-8 token,
  consume from the current state along matching `ilabel` arcs,
  emitting `olabel` symbols and tracking weights. Use the
  Tropical semiring (sum of weights along the path; pick the
  min-weight path on ambiguity). Wetext's TN rules are
  deterministic in practice — most input strings have a unique
  matching path — so the traversal is essentially a DFA lookup
  with epsilon transitions, not a full lattice search.
- **Symbol table** (~50 LOC). Wetext ships `tokens.txt` /
  `chars.txt` alongside the `.fst` files mapping symbol IDs to
  UTF-8 strings. Load once, look up by ID during emission.
- **Token-parser glue** (~200 LOC). Port wetext's
  `token_parser.py` (the `tagger output → struct → verbalizer`
  flow) into C++. This is independent of the FST library — same
  ~200 LOC of recursive-descent parsing whether we use OpenFST
  or roll our own.
- **Build-time pre-composition** (optional, ~0 LOC at runtime if
  we ship a pre-composed `.fst`). The tagger ∘ verbalizer
  composition can be done once offline using upstream OpenFST
  on the dev machine and the result checked in alongside the
  rule data. Run-time then only needs the StdVectorFst
  interpreter, not composition.

Total: ~500-800 LOC of C++, zero new dependencies, ~1 MB of
checked-in or auto-downloaded `.fst` data (same as #95b).

Trade-offs vs #95b:

- **Pro:** No third-party submodule. Build profile unchanged.
  Easier to reason about (it's small enough to fit in one
  reviewer's head). Cross-compilation footprint identical to
  the rest of CrispASR.
- **Con:** Not byte-identical to upstream wetext on edge cases
  involving FST features we don't implement (composition
  shortcuts, special weight semirings, on-the-fly relabeling).
  Acceptable iff we pin to the specific wetext rule files and
  treat them as a frozen artifact.

Estimated effort: 2-4 days for someone who reads the OpenFST
binary format spec end-to-end and can verify against
upstream's `fstprint` output on a dozen small inputs. Strictly
less effort than #95b (3-5 days + ongoing OpenFST submodule
maintenance) and strictly more correct than #95a (hand-rolled
rules will never catch up to wetext's coverage).

Files:

- `src/indextts_zh_tn.{h,cpp}` — the parser + traverser +
  symbol-table loader; ~500-800 LOC.
- `src/indextts.cpp` — invoke `normalize_chinese_wetext_native()`
  when `INDEXTTS_TEXT_NORMALIZER=native` is set (third sentinel
  alongside the existing shell-command form).
- `models/indextts-zh-tn-{tagger,verbalizer}.fst` — checked in
  (1 MB) or auto-downloaded via the model registry.
- `tools/dump-openfst-text.py` — dev-time helper that reads a
  `.fst` via upstream `pynini`/`openfst` Python bindings and
  dumps the byte layout for verification.
- `tests/indextts_zh_tn_test.cpp` — golden inputs (the same set
  used in #95a) matched against `wetext` Python output byte for
  byte.

When to do it: after #95a's hand-rolled list passes 2-3 entries
(confirming the use case is alive) but before contemplating
#95b's OpenFST submodule. #95d is the "we want byte-stable
behaviour without the dependency tax" sweet spot.

### What looks like an alternative but isn't

- **ICU `Transliterator`** — does Unicode normalization (NFC, NFKC,
  case folding) and pre-defined transforms like `Han-Latin`. No
  number → hanzi, no date parsing, no rule set comparable to wetext.
- **`libnumber2chinese` / `cn2an` / `pypinyin`** — Python libraries
  with no coherent C/C++ port. Fragments exist, no drop-in.
- **HuggingFace `tokenizers` normalizers** — tokenizer-side
  normalization (lowercase, NFC, strip accents), nowhere near
  wetext-equivalent rule coverage.

### Trigger to start work

This section sits idle until **one of**:

1. A user files an issue with a digit/date/pinyin-tone-digit prompt
   that produces broken audio. Then go to #95a; pick the smallest
   rule set that fixes the reported case.
2. The hand-rolled list reaches 2-3 entries (signal: the use case is
   actually alive). At that point #95d (tiny FST reader, ~500-800
   LOC, zero deps) becomes the right next step rather than letting
   #95a grow into a pile of one-off rules.
3. Only if #95d turns out to be wrong (FST features we don't
   implement keep biting), fall back to #95b (vendor OpenFST + kaldifst).
   Don't go to #95b speculatively.

Don't pre-emptively vendor OpenFST. CrispASR's clean "ggml + minor
deps" profile is a feature.

---

## 100. MeloTTS + OpenVoice2 — multilingual TTS with native CJK + voice cloning — Phase A DONE

_Done — see HISTORY.md + git log._

## 101. OmniVoice — single-stage NAR diffusion TTS with voice cloning

Surveyed via RapidAI/RapidSpeech.cpp ("single-stage non-autoregressive
diffusion TTS, multilingual + voice cloning"). RapidSpeech ships a
`convert_omnivoice_to_gguf.py` that merges an LLM component + audio
tokenizer into one GGUF — same packaging pattern as our Fun-ASR-Nano
and MiMo-ASR ports.

### Open questions before scoping

**Upstream source unclear.** RapidSpeech.cpp's README doesn't link
the upstream OmniVoice repo or HF model card; my websearch hit a wall
(the name collides with several other "Omni" projects). Before
sizing the port we need:

1. Confirm the upstream model identifier (likely on ModelScope or HF —
   search for "OmniVoice" + "diffusion TTS"; check
   `FunAudioLLM`/`ZAI`/`Beijing-Academy-of-AI` namespaces).
2. License — Apache-2.0 / MIT / non-commercial? Skip if non-commercial.
3. Parameter count and codec choice (RVQ? CFM target like voxcpm2?
   raw mel like MeloTTS?).
4. **Differentiation vs voxcpm2.** voxcpm2 is also CFM-diffusion with
   voice cloning (Apache-2.0, 30 langs, native 48 kHz). If OmniVoice
   doesn't bring something distinct — a different codec, better CJK,
   smaller footprint, faster inference — it's redundant.

### Conditional port plan (only if the upstream survey clears)

Assume the model is **non-autoregressive diffusion** (per RapidSpeech's
description). The likely shape based on the conversion script hint
("LLM component + audio tokenizer"):

- A text-conditioning LLM (likely 0.5-1B, similar to qwen3-tts talker)
- An NAR diffusion head over a discrete audio codec (likely 12-25 Hz
  RVQ like qwen3-tts or 48 Hz CFM-mel like voxcpm2)

Reuse map:
- Diffusion solver: voxcpm2's CFM solver (`src/voxcpm2_tts.cpp`) or
  Chatterbox-S3Gen's CFM solver (HISTORY §82) — pick the one that
  matches OmniVoice's solver type (DPM-Solver++ vs Euler vs HuangEuler)
- Codec: if RVQ, reuse mimo-tokenizer (`src/mimo_audio_tokenizer.cpp`);
  if mel-CFM, reuse voxcpm2 VAE
- Talker: another qwen3-tts-style talker port (#52 family); no new
  primitives needed

### Triggers

- Survey clears with a permissive license + measurable advantage over
  voxcpm2 on one of {CJK quality, model size, latency}.
- Otherwise this stays in survey-only mode — voxcpm2 + qwen3-tts +
  the (planned) melotts/openvoice2 already cover the same space.

---

## 102. RapidTP-Aligns — dedicated NN timestamp predictor (survey)

RapidAI ships `RapidTP-Aligns` ("语音的时间戳预测") as a standalone
**timestamp-only model** that runs on raw audio independently of any
ASR. CrispASR currently produces word-level timestamps via either:

- **Native** (whisper, parakeet TDT, canary, cohere, kyutai-stt) —
  emitted by the decoder
- **CTC forced aligner** (`-am canary-ctc-aligner.gguf` or
  `-am qwen3-forced-aligner.gguf`) for LLM-style backends that lack
  native timestamps (granite, voxtral, qwen3, glm-asr, omniasr-llm,
  funasr, etc.) — requires the *text* as input and aligns it back
  to the audio

A dedicated NN predictor would be a **third path**: predict timestamps
**from audio alone**, without needing the ASR's text. Useful when:

- We want timestamps independent of which ASR ran (cross-backend
  consistency on the same audio).
- The ASR's text is wrong but we still want trustworthy segment
  boundaries — useful for diarization / VAD post-processing.
- Streaming: predict end-of-utterance / silence boundaries one
  forward pass instead of running a full CTC alignment.

### Open questions

1. **What architecture** does upstream RapidTP-Aligns ship? The repo
   description is one line in Chinese; the README doesn't surface
   model identifier or arch. Likely a small Conformer-CTC over
   raw audio outputting frame-level boundary labels.
2. **License + upstream weights.** ModelScope-hosted? FunASR derivative?
   Confirm before starting.
3. **Quality vs our CTC aligners.** Our `canary-ctc-aligner-q4_k.gguf`
   (~80 MB) is fast and accurate enough that we haven't seen
   complaints — without a clear improvement margin a dedicated NN
   timestamp predictor is incremental.

### Trigger

- User reports CTC aligner failing on a specific audio class
  (e.g. heavily code-switched, multi-speaker overlap, music behind
  speech) that a dedicated timestamp predictor might handle better.
- OR: we add streaming ASR endpoint that needs sub-100-ms
  end-of-utterance prediction (currently we use VAD silence heuristic).

Until then: stays survey-only. Existing CTC aligners are good enough
for the dominant use cases.

---

## 103. Silero VAD version bump — verify and align with v6 — DONE

_Done — see HISTORY.md + git log._

## 104. Stateful frame-streaming TDT decode for parakeet long-form (issue #89)

**Priority: HIGH** — the auto path (no `--vad`, no `--chunk-seconds`) tops
out at ~82 % coverage on 60 s Japanese audio. Users expect >95 %.

### Problem

The current chunking approach (`kLongAudioFallbackChunkSeconds = 30`)
processes each chunk independently: fresh mel + z-norm + fresh TDT
decoder LSTM state. The decoder cold-starts each chunk and loses 5-20 s
of content from each chunk's interior (not just at boundaries). Sweep of
chunk sizes 15-30 s × overlap 3-8 s on the issue #89 reporter's 60 s
Japanese audio (parakeet-tdt-0.6b-ja, CPU-only):

| chunk | overlap | chars | coverage% | max_gap |
|-------|---------|-------|-----------|---------|
| 20s   | 3s      | 278   | 82.4%     | 3.4s    |  ← best without VAD
| 15s   | 5s      | 278   | 70.9%     | 5.3s    |
| 30s   | 3s      | 195   | 59.7%     | 19.4s   |  ← current default
| 60s   | 0s      | 294   | 99.5%     | 0s      |  ← single-pass (fails on Vulkan/AMD)
| VAD silero | —  | 281   | 93.1%     | 3.7s    |

Counterintuitively, more overlap hurts (extends z-norm window →
distribution shift). The ceiling is the decoder cold-start, not boundary
stitching.

### NeMo's approach

NeMo's `FrameBatchASR` / `BatchedFrameASRTDT` uses a fundamentally
different architecture:

| | NeMo streaming | CrispASR chunking |
|---|---|---|
| Frame step | 1.6-4 s | 15-30 s |
| Buffer | 4 s rolling | chunk + overlap |
| Z-norm | Over 4 s buffer | Over 30-33 s chunk |
| Decoder | **Stateful** across frames | Independent per chunk |
| Overlap ratio | 60-150 % of frame | 10-20 % of chunk |

Key: NeMo keeps the TDT LSTM predictor state between frames. Each 1.6 s
step feeds the decoder with the previous hidden state, so it never
cold-starts. CrispASR reinitializes the LSTM (`lstm_init_state`) for
every chunk.

### Implementation plan

**Phase 1: Stateful TDT decode (the core change)**

The split API already exists (`parakeet.h`):
```c
float* parakeet_encode(ctx, samples, n_samples, &T_enc, &d_model);
parakeet_result* parakeet_decode_frames(ctx, enc_frames, T_enc, d_model, t_offset_cs);
```

Changes needed:

1. **Add `parakeet_decode_frames_stateful`**: like `parakeet_decode_frames`
   but accepts/returns the LSTM hidden state (`parakeet_lstm_state`).
   The TDT decode loop in `parakeet.cpp:1003` already uses `state` — just
   need to make it an in/out parameter instead of initializing to SOS.

2. **Add streaming mel with running z-norm**: instead of computing z-norm
   per chunk, maintain running mean/variance across frames (NeMo's
   `get_norm_consts_per_frame` approach). Add a
   `parakeet_mel_streaming_context` that tracks the statistics.

3. **Wire into `crispasr_run.cpp`**: when the long-audio fallback
   triggers and the backend is parakeet/canary, use the streaming
   decode path instead of independent chunk transcription:
   ```
   for each 4s frame:
     mel = compute_mel(frame, streaming_mel_ctx)  // running z-norm
     enc = parakeet_encode(mel)
     result = parakeet_decode_frames_stateful(enc, &lstm_state)
     merge results with LCS
   ```

**Phase 2: Tuning**

4. Frame size sweep (1.6s, 2s, 4s, 8s) on the benchmark corpus.
5. Running z-norm warmup: first frame gets per-frame z-norm, subsequent
   frames use exponential moving average (NeMo's approach).
6. LCS delay tuning: `lcs_delay = (buffer - frame) / model_stride`.

### Files to modify

- `src/parakeet.h` — add `parakeet_decode_frames_stateful`, `parakeet_mel_streaming_context`
- `src/parakeet.cpp` — expose LSTM state in/out, add streaming mel helper
- `src/core/mel.h` / `mel.cpp` — add running z-norm mode
- `examples/cli/crispasr_run.cpp` — streaming decode path in `process_one_input`
- `examples/cli/crispasr_backend_parakeet.cpp` — streaming transcribe method
- `tests/test-issue-89-long-audio-fallback.cpp` — pin streaming path activation

### Effort

Medium-Large. Phase 1 is ~200-300 LOC of new code (the hot loop is
<50 lines — the LSTM state threading is the main work). Phase 2 is
benchmarking and tuning.

### Success criteria

`python tests/benchmark_asr.py --audio yt_60s.wav --backend parakeet-ja --settings auto`
reports **coverage ≥ 95 %** on the issue #89 reporter's Japanese audio,
without `--vad`.

### Trigger

Immediate — issue #89 is open and the current fix is a partial
mitigation (prevents 0-output catastrophe but doesn't match NeMo quality).

---

## 105. WhisperX word alignment models — wav2vec2 CTC zoo

WhisperX does word alignment as a post-process: ASR text first, then a
language-keyed wav2vec2/CTC aligner refines timestamps at the word
level. The shipped defaults include:

- Torchaudio bundles: `WAV2VEC2_ASR_BASE_960H` (`en`),
  `VOXPOPULI_ASR_BASE_10K_FR` (`fr`),
  `VOXPOPULI_ASR_BASE_10K_DE` (`de`),
  `VOXPOPULI_ASR_BASE_10K_ES` (`es`),
  `VOXPOPULI_ASR_BASE_10K_IT` (`it`)
- Hugging Face checkpoints:
  `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`,
  `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`,
  `jonatasgrosman/wav2vec2-large-xlsr-53-dutch`,
  `Yehor/wav2vec2-xls-r-300m-uk-with-small-lm`,
  `jonatasgrosman/wav2vec2-large-xlsr-53-portuguese`,
  `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`,
  `comodoro/wav2vec2-xls-r-300m-cs-250`

CrispASR now has three CTC aligner families wired into `-am`:

- `canary-ctc-aligner.gguf`
- `qwen3-forced-aligner.gguf`
- `wav2vec2-aligner` / `wav2vec2-aligner-en` / `wav2vec2-aligner-de`
  aliases, plus any raw wav2vec2 / HuBERT / data2vec CTC GGUF path

That means the runtime family support exists, but not all of WhisperX's
language models are converted/uploaded yet. The remaining path splits
into two cases:

- Torchaudio bundle models need a direct runtime path or explicit
  converter support.
- Hugging Face wav2vec2 checkpoints are the easy case: add a generic
  wav2vec2-aligner family, register the common language aliases, and
  reuse the existing CTC alignment plumbing.

### Implementation plan

1. DONE: Add a generic `wav2vec2-aligner` family in the aligner registry
   and C-ABI path so `-am` can dispatch beyond canary/qwen3.
2. DONE: Add initial aliases for `en` and `de` using the already-hosted
   wav2vec2 GGUFs.
3. DONE: All 10 WhisperX common languages converted, quantized, and
   uploaded to `cstr/*-GGUF` HF repos: `fr`, `es`, `it`, `ja`, `zh`,
   `nl`, `uk`, `pt`, `ar`, `cs`. Registry entries and auto-download
   wired. Verified 2026-05-23.
4. DONE: All models are native HF wav2vec2 checkpoints converted via
   `models/convert-wav2vec2-to-gguf.py`. Torchaudio bundles not needed.
5. TODO: Benchmark the new aligners against `canary-ctc-aligner.gguf`.
6. TODO: Document in docs/cli.md which `-am` aliases are available.

### Trigger

- A user wants WhisperX-style word alignment parity in CrispASR.
- A backend needs better language coverage than the current canary/qwen3
  aligners provide.
- We want to close the gap between `whisperX.load_align_model(...)` and
  CrispASR's current `-am` model surface.

---

## 106. TEN-VAD — low-latency cross-platform VAD

TEN-VAD looks technically feasible as an additional VAD backend. The
upstream repo ships cross-platform C bindings, prebuilt libs, and an
ONNX path, and its documented runtime target is 16 kHz audio with
10/16 ms hop sizes. That fits CrispASR's existing VAD surface well
enough to add as a fourth backend, alongside Silero, FireRedVAD,
MarbleNet, and Whisper-VAD-EncDec.

Why it is a good fit:

- C-compatible API and native libs already exist for Linux, macOS,
  Windows, Android, iOS, and Web.
- The model is lightweight relative to Silero and is explicitly aimed at
  low-latency streaming turn detection.
- The upstream README already documents Python, C, Java, Go, and JS
  usage, so the packaging shape is familiar.

Main caveat:

- The upstream license is Apache 2.0 with additional no-compete / own-
  app-only conditions, so we should treat distribution as blocked until
  legal review or a clear internal-only use case is approved.

### Implementation plan

1. Keep the technical integration plan ready, but do not wire or ship
   the backend until the license decision is explicit.
2. Decide whether to use the prebuilt native lib path, the ONNX path,
   or both.
3. Add a `ten-vad` backend alias in the VAD registry and CLI so
   `--vad -vm ten-vad` works, if approved.
4. Add auto-download metadata for the chosen artifact(s) and keep the
   existing Silero default unchanged.
5. Run the usual boundary benchmark against Silero and FireRedVAD on
   the same short-gap / sentence-end test set.
6. Document sampling-rate handling clearly: 16 kHz in, resample other
   inputs before inference.

### Trigger

- A user wants a lower-latency or lower-footprint VAD than Silero.
- We want a fourth backend option with native cross-platform coverage.
- The license review confirms the additional no-compete / own-app-only
  conditions are acceptable for our distribution model, or we confine it
  to an internal-only path.

---

## 114. Long-form transcribe — make chunking/streamed the default for all ASR backends (issue #89 follow-up)

**Status (2026-05-25):** parakeet portion DONE 2026-05-24 via `33f9a162`. Remaining work is per-backend, prioritised below by failure severity from the 60 s + 120 s sweeps on lenhone's fresh `yt-dlp` audio. A 60/120/300/600 s × all-multilingual-backends matrix is running on the VPS to extend the data; numbers update PERFORMANCE.md as they land.

### Per-backend long-audio status

Three columns: **CAP flag** = how it routes through `crispasr_run.cpp`'s auto-chunk gate (`crispasr_long_audio_fallback`), **path** = what the backend actually does for inputs > a few minutes, **status** = empirical result on lenhone's clip.

| backend | CAP flag | path | status (lenhone audio, 60 s / 120 s) | NeMo / upstream equivalent |
|---|---|---|---|---|
| **parakeet** (tdt / tdt_ctc / rnnt) | `CAP_INTERNAL_CHUNKING` | `parakeet_transcribe_streamed`: global z-norm + 8 s overlapping encoder windows + concat + single TDT decode | **✓ DONE 2026-05-24 (`33f9a162`)** — 60 s: 7 segs, full speech; 120 s: 12 segs, full speech to 1:37.84 (clip's speech end) | matches `nemo.collections.asr.parts.utils.streaming_utils.BatchedFrameASRTDT` shape; we made it the default, NeMo's `transcribe()` defaults to single-pass and reproduces the same 20 s collapse |
| **canary-1b-v2** (multi-task AED) | `CAP_INTERNAL_CHUNKING` declared but **no streamed path**; falls back to single-pass encoder | hallucinates English `"I am not aware of anything"` loop on the lenhone 120 s — root cause uncertain (likely missing `<lang>` / `<task>` prompt tokens at the boundary) | ✗ **broken on long JA** | NeMo: `FrameBatchMultiTaskAED` in `streaming_utils.py` |
| **fastconformer-ctc** (en-only) | `CAP_INTERNAL_CHUNKING` declared, single-pass encoder | CTC argmax is frame-synchronous so doesn't have the TDT blank-runaway failure mode; full-pass works on moderate lengths | ~ "works in practice; not formally chunked" | NeMo: `FrameBatchChunkedCTC` |
| **voxtral-mini-3b** (LLM AR) | no `CAP_UNBOUNDED_INPUT` → CLI auto-chunk fires at 30 s | energy chunker hands the LLM 30 s slices, no LCS dedup, AR decoder loses track at chunk boundaries | ✗ **120 s: 0:00→0:27 then jumps to 1:47→2:00** (~80 s dropped in the middle) | Mistral upstream: chunked at ~30 s **with overlap** + manual stitching |
| **cohere-transcribe** (Conformer) | no `CAP_UNBOUNDED_INPUT` → CLI auto-chunk at 30 s | Conformer encoder hits a similar long-attention regime as parakeet single-pass on the chunk-context window | ✗ **120 s: only 4 segments across 120 s, multi-tens-of-seconds gaps** | Cohere hosted does server-side VAD + chunking; released weights aren't designed for long inputs |
| **qwen3-asr** (LLM AR) | no `CAP_UNBOUNDED_INPUT`; CLI auto-chunk + LCS dedup (PLAN #80c) | chunked + LCS overlap merge | works on short clips, slow on long; unknown failure mode on 120 s+ (TBD by VPS matrix) | n/a (Alibaba upstream) |
| **granite-speech** / **granite-4.1** (LLM AR) | no `CAP_UNBOUNDED_INPUT`; CLI auto-chunk | chunked, no LCS dedup yet | TBD by VPS matrix | n/a (IBM upstream) |
| **gemma4_e2b** | no `CAP_UNBOUNDED_INPUT`; `prefers_vad()=true` | CLI auto-VAD for >30 s (silence-bounded segments match ~30 s training window) | ✓ JFK 11 s correct; long audio auto-VAD | n/a (Google upstream) |
| **kyutai-stt** | no `CAP_UNBOUNDED_INPUT` | CLI auto-chunk; streaming-native model | likely OK by design | upstream is cache-aware streaming |
| **mimo-asr** (LLM AR, multilingual) | no `CAP_UNBOUNDED_INPUT` | CLI auto-chunk | TBD (4.5 GB model — heavy) | n/a (Xiaomi upstream) |
| **sensevoice-small** (CTC) | no `CAP_UNBOUNDED_INPUT`; CLI auto-chunk; works well with `--vad` | CTC-style decode, robust to chunking | **✓ 120 s with `--vad`: 13 segs, 0:00 → 2:00, full speech** (minor glitches `スピーク**ジャ**プネス…`) | FunASR upstream |
| **firered-asr** | `CAP_UNBOUNDED_INPUT` | full-audio encoder pass | untested at 120 s+ | n/a (XiaoMi/Xiaohongshu upstream) |
| **wav2vec2** | `CAP_UNBOUNDED_INPUT` | full-audio encoder pass; CTC head | CTC is robust; tested up to 60 s | n/a (Meta upstream) |
| **whisper-large/medium/small/base/tiny** | n/a (whisper has its own internal seek loop) | 30 s windows internal to `whisper.cpp` | ✓ designed for this | upstream is `whisper.cpp` itself |

### Why parakeet was the loudest

Lenhone happened to use parakeet-tdt-0.6b-ja. The other backends are not safe — they just hadn't been reported. Treating long-form as "the caller wraps with `--vad` or `--chunk-seconds N`" is a footgun: most users don't, and the failure modes are silent (no error, just missing text). Stock `crispasr -m auto -f long.wav` should produce a *complete* transcript on any duration on any multilingual backend.

### Roadmap (priority order, empirical-data-driven)

**P0 — Verify scope with the cross-length matrix (in progress).**
Running the 60/120/300/600 s × multilingual-backend matrix on VPS (`/tmp/longform_vps.sh`, PID 3572547) to fill in the gaps in the table above (qwen3, granite, kyutai, gemma4 cells; verify voxtral/cohere/canary failure modes hold past 120 s; quantify sensevoice as the multilingual baseline winner). Results land in `PERFORMANCE.md` "Long-form ASR cross-backend matrix" once done. Without this we can't prioritise the fixes properly.

**P1 — cohere-transcribe: default `--vad` for any input > 30 s.**
Cheapest fix (one capability flag flip + one CLI gate in `crispasr_run.cpp`). The released Cohere weights aren't trained for long inputs in the first place; the hosted product does VAD on the server side. Doing the same on the client side is faithful to the release intent and produces full-coverage output in our 60 s and 120 s tests.

**P2 — voxtral / qwen3-asr / granite-speech / mimo-asr: chunk + LCS dedup.**
We already have `crispasr_lcs::merge_overlapping_hypotheses` from PLAN #80c. Wire it as the default for LLM-AR backends with overlap ≥ 2 s. The 120 s voxtral mid-drop is the smoke test — if LCS+overlap fixes that cleanly, generalise to the other LLM-AR backends. Avoid the LCS-dedup-disabled case on `chunk-overlap 0` (the existing test #114 / #148 gate already covers this).

**P3 — canary-1b-v2: lang-whitelist DONE `dfe1af3b` 2026-05-26; `canary_transcribe_streamed` still open.**

**First half — DONE.** canary-1b-v2's BPE vocab includes every ISO-639 `<|xx|>` token (200+), but the model is trained on en/de/fr/es only. Passing `-l ja` built the prompt successfully and ran the decoder, which then produced hallucinated output — mixed Cyrillic + Greek garbage on JFK with `-l ja` ("И така, мои сънародници, не питайте, τι может да направи ваша страна, ..."). Fix: static `{"en", "de", "fr", "es"}` whitelist in `crispasr_backend_canary.cpp` rejects unsupported langs before invoking `canary_transcribe_ex`, with a clear message pointing at parakeet for ja/zh and qwen3/voxtral for the broader multilingual set. Smoke: `-l en` JFK unchanged; `-l ja` JFK now errors out instead of producing garbage.

**Second half — SHIPPED `7177c931` 2026-05-26 (concat) → `63fdbe46` (per-chunk re-injection).**

First cut (`7177c931`) was parakeet-pattern: full-mel + 8 s/2 s overlapping chunked encode + concat → ONE AED decode. `canary_transcribe_ex`'s post-encode body (cross-KV + prompt + greedy decode + DTW timestamps) extracted into a static `canary_finish_from_encoder` helper that both entry points share. Hit the AED-trained-on-single-utterance limitation: synthetic concatenated short clips emitted `<eos>` at the chunk boundary after producing one full JFK transcript.

**Replaced (`63fdbe46`) with NeMo `FrameBatchMultiTaskAED` analogon.** Each chunk gets its OWN AED decode with the language/task prompt re-injected, then per-chunk transcripts are concatenated. Closes the boundary-`<eos>` bug because each chunk's decoder sees a fresh prompt and doesn't carry "we just finished an utterance" state across the splice.

Verified locally on M1 Metal against real long-audio fixtures fetched from VPS (`/mnt/akademie_storage/yt_{60,120}s.wav` → `/Volumes/backups/ai/long-clips/`):

| input | single-pass | first cut concat (`7177c931`) | per-chunk (`63fdbe46`) | per-chunk + LCS dedup (`62766dae`) |
|---|---|---|---|---|
| JFK 11 s, default path | "...for you, ask what you can do for your country." ✓ | (single-pass, unchanged) | (single-pass, unchanged) | (single-pass, unchanged) |
| JFK 11 s, forced streamed | n/a | "...for you." (truncated) | "...for you, Country can do for you. Ask…" (boundary dup) | "...for you, . Ask what you can do for your country." (dup gone; minor `, . ` artifact) |
| yt_60s.wav (Japanese, -l en) | empty | one short hallucination | romanized JA + "yeah" loops (60 s → 11 s wall, 5.3× RT) | same content, 18 s wall (3.3× RT) |

LCS dedup (`62766dae`) lands the boundary-dup polish. The remaining `, . ` splice artifact on JFK forced-streamed is cosmetic (the LCS match falls between the comma of chunk 1 and the period of chunk 2). A punctuation-cleanup pass is the next polish; until then `CANARY_STREAM_THRESHOLD_S` default stays at 30 (single-pass on short audio). The 60 s OOD-audio "yeah" loop is the AED decoder hitting a no-repetition-penalty failure mode; same shape as the funasr `!`-loop fix in PLAN #125 P1, applies symmetrically here as a follow-up.

**Status.** Default `CANARY_STREAM_THRESHOLD_S=30` retained. Functionally, the long-audio path is correct now — no truncation, no boundary duplication, just a cosmetic punctuation artifact at the splice. Flipping default to `0` (always streamed) is the next promotion step, gated on the punctuation polish.

**P4 — fastconformer-ctc: optional streamed wrapper.**
Lower priority. CTC's frame-synchronous decode doesn't fall into the TDT blank-runaway trap, so this is a portability improvement rather than a correctness fix. Defer until P1-P3 ship and a user reports a real failure on en-only long audio.

### Streaming-pattern design: NeMo vs Voxtral, what's tied vs what's a knob

The two long-audio architectures we ship come from two upstream traditions and have different decoder-class requirements. This section pins what's user-tunable and what's structurally fixed per-backend.

**Pattern A — NeMo `BatchedFrameASRTDT` / `FrameBatchMultiTaskAED`**:
overlap-chunks (8 s + 2 s) → per-chunk encoder pass → LCS-merge dedup at the
boundary. Encoder needs bidirectional context across cuts, so chunks must
overlap; LCS dedup is mandatory consequence. Per-chunk decoder reset is
required for AED-class decoders (canary's `<eos>` semantics — see PLAN #114
P3 "AED-trained-on-single-utterance" footnote). For frame-synchronous
decoders (parakeet TDT/RNN-T, fastconformer CTC) the per-chunk reset is
optional; we currently concat the encoder output and run a single decode
(hybrid mode).

**Pattern B — Mistral voxtral `apply_transcription_request`**:
disjoint 30 s chunks (no overlap) → per-chunk encoder pass → audio embeds
concatenated → one LLM AR decode over the whole thing. No dedup needed
because no duplicated audio in the input. Requires the decoder to be a
long-context AR LLM (voxtral's 3 B, qwen3-asr's 0.6/1.7 B). Bad fit for
AED (canary) or frame-synchronous decoders (parakeet TDT) — both would
need a single decode over the whole encoder output, which doesn't compose
the same way.

**Per-backend fit:**

| Backend | Decoder class | Current pattern | Other pattern feasible? |
|---|---|---|---|
| parakeet (TDT/RNN-T/CTC) | frame-synchronous | NeMo-overlap + single decode (hybrid) | Voxtral pattern technically works (TDT has no `<eos>`) — but no quality upside; just trades the LCS dedup for an awkward "no-overlap encoder seeing a 30 s window of audio with no bidirectional context across cuts" trade |
| canary (AED) | AR with implicit `<eos>` | NeMo-overlap + per-chunk decode + LCS dedup | Voxtral pattern fails — concat-then-decode emits `<eos>` at first chunk-boundary, as we observed in `7177c931` |
| voxtral (long-context LLM) | AR LLM | Voxtral-disjoint + single decode | NeMo pattern works but throws away the LLM's long-context capability — per-chunk LLM resets lose speech context coherence |
| qwen3-asr, granite, glm-asr, mimo-asr, gemma4-e2b, kyutai-stt | AR LLM | Mostly voxtral-shaped (no overlap) | NeMo pattern technically possible — would need per-chunk prompt re-injection at each LLM reset |

**What's a knob today:**

- `PARAKEET_STREAM_THRESHOLD_S` / `CANARY_STREAM_THRESHOLD_S` / `KYUTAI_CHUNK_S` etc. — per-backend duration thresholds for picking single-pass vs streamed.
- `CRISPASR_GEMMA4_AUTO_CHUNK` — opt-in to streamed chunking (default abort > 30 s).
- The opt-out list in `examples/cli/crispasr_chunk_context_gate.h::kBlocked` — backends that refuse the dispatcher's overlap-save context wrap.

**What's NOT a runtime knob (and probably shouldn't be):**

- The pattern choice itself. For canary, pattern B fails by design; for voxtral, pattern A throws away the model's strength. Forcing the wrong pattern via a CLI flag would let users misuse the binary with no quality recovery.
- The encoder's overlap requirement. Conformer-style encoders need overlap; Whisper-style block-causal encoders don't. Backend-specific.

**What COULD become a knob (future):**

- For parakeet (the only genuinely hybrid case), expose `PARAKEET_STREAM_PATTERN=hybrid|voxtral-disjoint` to let users experiment with the disjoint-chunk variant. Low priority — no observed quality win to motivate the implementation effort.
- For voxtral, expose `VOXTRAL_STREAM_CHUNK_S` (currently fixed at 30 s) to let memory-constrained users trade per-decode RAM for more chunks.

**Bottom line for the user-choice question:** the pattern per backend is structurally tied to its decoder class. We expose duration thresholds, chunk sizes, and opt-out lists as runtime knobs, but the high-level pattern (overlap+per-chunk vs disjoint+single-decode) is a per-backend property, not a config dial.

### Decision: don't blanket-VAD everyone

Considered earlier as Option 1. Rejected because:
- VAD trims leading/trailing silence per segment → coverage on continuous speech drops 99 % → 93 % even on the audio where it works perfectly. Wrong default for narration-style content.
- VAD output is per-utterance SRT entries, not paragraph-level. Worse for users who want one continuous transcription.
- Per-backend defaults (P1: VAD for cohere, P2: chunk+LCS for LLM-AR, P3: streamed-encode for canary) match the actual failure mode and don't pay the VAD coverage cost on backends that don't need it.

### Trigger conditions for completion

- 60 s + 120 s + 300 s + 600 s VPS matrix shows full speech coverage on every multilingual backend (current parakeet bar)
- `tests/test-issue-89-long-audio-fallback.cpp` extended with assertions for cohere VAD-default, voxtral LCS-default, canary streamed-encode
- `PERFORMANCE.md` long-form section updated with the post-fix numbers per backend
- `docs/cli.md` long-form recommendation table per backend (currently only parakeet has a per-backend story)

### Files (tentative)

- `examples/cli/crispasr_backend_cohere.cpp` — `CAP_LONGFORM_PREFERS_VAD` flag or similar
- `examples/cli/crispasr_backend_voxtral.cpp` / `_granite.cpp` / `_qwen3.cpp` — wire `core_lcs::merge_overlapping_hypotheses` into the chunked output stitching path
- `src/canary.cpp` — new `canary_transcribe_streamed` (parakeet pattern, with AED prompt re-injection at boundaries); `examples/cli/crispasr_backend_canary.cpp` route through it
- `examples/cli/crispasr_run.cpp` — auto-chunking gate refactored to per-backend ladder; `CAP_LONGFORM_PREFERS_VAD` honored
- `tests/test-issue-89-long-audio-fallback.cpp` — extend
- `tests/benchmark_asr.py` — multi-backend long-form scoring against the same 60/120/300/600 s fixtures (`longform_vps.sh` is the prototype)
- `PERFORMANCE.md` — per-duration cross-backend table

---

## 115. mimo-asr baseline broken — silent empty on short, segfault on long

**Status (2026-06-08): DONE.** GPU is the default since `a429bb45`.
Option B (prefill-graph reuse for decode steps, embed tables CPU-resident
via `load_weights_split`) is the production path — validated on RTX 3090
+ Kaggle P100, 2.4× realtime. k-quant CUDA GET_ROWS fix (`3bf9a599`)
landed as a safety net (supports all-GPU-resident weights at 2.0× RT).
Option B is faster than Option C (step-graph with per-tensor tagging)
would be (10 ms/step vs 31 ms/step), so Option C is **not worth pursuing**.
`CRISPASR_MIMO_FORCE_CPU=1` override available for debugging.

**Kernel run 1 (P100, `bf4b5c3c`) — refutes the prefill hypothesis.**
With `CRISPASR_MIMO_FORCE_GPU=1` the GPU **prefill is correct**: all five
`mimo_dump` stages (audio_features, text_embeds, inputs_embeds, last_hidden,
text_logits_step0) match CPU with **no NaN/Inf** (`first GPU-diverging stage:
none`). But the run **segfaults `rc=-11` at 16.5 s** — *after* the prefill,
in the **decode step**. So option C is NOT in `mimo_asr_build_prefill_graph`;
it's the per-token decode path (build_decode_graph + fresh-cgraph-per-token),
same shape as #125 P0's sched src-mutation-on-re-laid-out-graph — which the
`95d74455` hardening was supposed to fix but evidently doesn't on P100.

**Kernel run 2 (P100, gdb) — root cause.** The backtrace is
`dequantize_row_q4_K` → `ggml_backend_cpu_graph_compute` →
`mimo_asr_transcribe_impl`. So it is NOT the sched src-mutation class at all:
with `force_gpu` the weights are GPU-resident, but the sched was still built
with **both** `[CUDA, CPU]` backends, so its placement heuristic offloaded a
**decode** op to the **CPU** backend — and `dequantize_row_q4_K` (a CPU
function) then read a **GPU-resident Q4_K** weight's CUDA pointer as host
memory → SIGSEGV. The prefill survived because none of its ops got
CPU-routed; exactly one decode op does.

**Run 3 (`3ef9f87e`) — single-GPU-backend sched is wrong.** Building the
sched with `{CUDA}` only *aborts* in `ggml_backend_sched_new` (`signo=6`):
ggml requires a CPU backend as the mandatory last/fallback entry. Reverted.
So the real bug is a specific **decode op CUDA can't run**, which the sched
then offloads to CPU where it dereferences a GPU-resident Q4_K weight. The
decode step reads the embedding table.

**Root cause (definitive, runs 4 + dtype check).** `GGML_SCHED_DEBUG=2` is a
no-op in Release, so it was found by reading the CUDA supports_op + the gguf
tensor dtypes: CUDA's `GET_ROWS` supports_op (ggml-cuda.cu:5004) lists
F16/F32/BF16/I32/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 — **not Q4_K** — and mimo's
`llm.embed.weight` + `audio.emb.*` are **Q4_K**. So `get_rows(embed[Q4_K])`
is CUDA-unsupported → the sched routes it to CPU → `dequantize_row_q4_K` reads
the GPU-resident weight's device pointer → SIGSEGV. Same shape as CSM §135: a
converter quantized a tensor that must stay gather-friendly (token embeddings
should never be Q4_K). (The set_rows theory was wrong — set_rows is fine.)

**Fix (runtime).** Under `force_gpu`, load only the get_rows'd
`embed`/`audio.emb` tables on CPU via `load_weights_split`; every matmul
weight stays GPU-resident for the speedup (the small embed output is copied
GPU-ward by the sched). Validating on kernel run 5 → expect GPU JFK PASS; if
green, flip `--gpu` on by default + mark option C DONE. Cleaner long-term:
the converter keeps `llm.embed`/`audio.emb` at F16/Q8_0 (CUDA-gatherable) so
no runtime split is needed — fold into the next mimo-asr GGUF re-bake.

**Status (2026-05-26):** option A shipped. **(2026-06-08: superseded — see top of section.)**

The smoking-gun commit is `89111260` ("perf #72: load weights to GPU when use_gpu=true"), which flipped `core_gguf::load_weights(..., ctx->backend_cpu, ...)` to `..., ctx->backend, ...`. The same commit message foresaw the regression — *"If a platform regresses, add a CRISPASR_FORCE_CPU_WEIGHTS=1 escape hatch — none seen yet"*.

Worth noting that the sched src-mutation-log hardening (`a5a518c8`/`95d74455`, montvid's Blackwell fix) does NOT address this — verified empirically on M1 Metal post-rebuild: same silent-empty. Two different mimo-asr bugs with the same observable symptom on different platforms.

**Option A shipped:**

1. `5a570b7b` — first pass: just pinned weights to `ctx->backend_cpu`. Insufficient — exposed a second failure mode where Metal compute can't resolve CPU-resident weight buffers (`ggml_metal_buffer_get_id: error: tensor 'llm.embed.weight' buffer is nil`). The §56 working configuration (CPU weights + Metal compute) no longer holds because the ggml scheduler tightened cross-backend tensor resolution since then.
2. `c887881e` — complete fix: force `ctx->backend = ctx->backend_cpu` unconditionally, ignore `params.use_gpu`. Verified on M1 Metal locally: JFK transcribes correctly in 297 s, matches HISTORY §56 reference verbatim. (Slow — pure CPU LLM on M1 — but correct.) Kaggle Linux x86_64 CPU build also verified passing on `b85698670`: `prefill 15.8 s, decode 7.0 s over 26 steps, total_lm 22.8 s`.

Cost of option A: loses the documented 22 % M1 Metal speedup from PLAN #72. Acceptable until C lands because the alternative is shipping a backend that produces no output.

**Option C (open):** proper GPU graph fix. mimo's `mimo_asr_build_prefill_graph` doesn't emit per-tensor backend tagging that current `ggml_backend_sched` needs to route weight reads from a CPU buffer through to a Metal/CUDA compute path. Two sub-options:
  - **C1.** Tag the embed weight (and any other CPU-resident tensors) for the appropriate backend before graph build, so sched can insert the needed copy nodes. Cheaper.
  - **C2.** Build the whole prefill graph on the user-selected backend with weights resident there too, and find what actually breaks the prefill emission (similar shape to chatterbox Bug B from issue #83 — see [[project_chatterbox_gpu_bug_s3gen]] — which took ten candidate hypotheses to land on the real cause). Restores the 22 % speedup. Higher value, requires Kaggle GPU run with the patched binary (currently quota-blocked).

**Progress toward C (2026-06-02):** `CRISPASR_MIMO_FORCE_GPU=1` env-gated diagnostic path exists — split CPU/GPU weight loading with Q4_K embed tables kept CPU-resident (`1cc91461`), fixing the CUDA `get_rows` crash. But the code at `src/mimo_asr.cpp:326-344` still describes this as a workaround; the proper per-tensor backend tagging in `mimo_asr_build_prefill_graph` is acknowledged as missing.

### Original repro (still valid for regression-guarding option C)

### What we see

```
$ ./build/bin/crispasr -m /Volumes/backups/ai/crispasr/mimo-asr-q4_k.gguf \
    --backend mimo-asr -f samples/jfk.wav -of /tmp/out -otxt
... whisper LID detects 'en' p=0.977 ...
mimo_tokenizer: loaded 569 tensors  encoder=32L/1280  rvq=20 stages
mimo_asr_transcribe: audio 176000 samples -> 276 code frames
mimo_asr_transcribe: prompt T_total=388 (T_groups=97)
mimo_asr: kv cache 51 MiB k=f16 v=f16 (on gpu, head_dim=128 max_ctx=369 n_kv=8 n_layers=36)
$ echo $?
0
$ ls /tmp/out*
# (nothing)
```

Two failure modes, same backend:

- **11 s JFK:** exit 0, no `.txt` / `.srt` produced, no error printed. Last log line is the kv-cache allocation; whatever happens inside the decode either returns an empty segment list silently or the segment-emission path is broken.
- **5 min audio:** segfault at ~159 s wallclock (`Segmentation fault: 11`, exit 139), well after the same init sequence.

Logs in `/Volumes/backups/ai/bench-results/overlap-bug-check/mimo-asr.{default,nooverlap}.log`.

### Why this is its own item, not part of #114

PLAN #114 already covers mimo-asr in the LLM-AR chunk-boundary class ("loses track at chunk boundaries"). That's the long-audio content-loss failure mode. This is different — the 11 s JFK case can't possibly trigger chunk-boundary loss (single chunk, single decode), and it still produces zero text. The backend can't transcribe *anything* in its current state.

### Suspected blast radius

mimo-asr is the only backend out of the 16 we A/B-swept that exhibits this. The other LLM-class backends (qwen3, voxtral, gemma4-e2b, granite) all transcribe short audio fine; only their long-audio behaviour was affected. So whatever regressed in mimo-asr is mimo-specific, not a shared-helper change.

### Next steps

1. `git bisect` on `src/mimo_asr.cpp` since HISTORY §56 (last known good: Q4_K shipped + ~50 % WER on a librispeech subset). Bisect harness: `crispasr -m mimo-asr-q4_k.gguf -f samples/jfk.wav -of /tmp/x -otxt && [ -s /tmp/x.txt ]`.
2. If the bisect fingers a refactor on the segment-emission side, check whether `mimo_asr_transcribe` is now returning early without writing into the segments vector.
3. If it's the kv-cache or decoder graph, see whether `max_ctx=369` is being exceeded silently.
4. For the 5-min segfault: separate question, may be the same root cause (loop overruns when more decode iterations execute) or genuinely independent. Triage after the JFK-emit case is fixed.

### Effort

Small if the bug is in segment emission (one missing `.push_back`-shaped fix). Medium if the decoder graph itself is wrong. The repro is trivial and runs in under a minute.

---

## 125. Issue #125 — multi-backend bug sweep from montvid (12 findings)

External user `montvid` ran every available backend on the issue #89 reporter's 50:47 EN FLAC plus the project's own `samples/jfk.wav` smoke fixture, all on **CrispASR v0.6.10 commit `eaee2319`** (the 2026-05-25 morning build, just after the per-backend opt-out fix train), hardware NVIDIA RTX PRO 6000 Blackwell sm_120 + CUDA 12.6. Reference build for the regression bisect: **commit `f23d9485`** (v0.6.9, 2026-05-21 "fix(paraformer): suppress cppcheck invalidPointerCast"). 12 report files attached to the issue; all 12 cached at `/Volumes/backups/code/issue125-attachments/`. The reporter's analysis is high-quality — each finding pins the failing file:line and proposes a concrete fix shape.

This PLAN section reproduces the priority ordering, status, and fix shape for each finding so the next contributor can pick any item off the list independently.

### P0 — Regression I shipped: mimo-asr segfault on Blackwell sm_120 (report 12) — bisect reattributed 2026-05-26

**TL;DR.** v0.6.9 `f23d9485` works on `samples/jfk.wav`. v0.6.10 `eaee2319` segfaults during decode (`rc=139`). Reporter bisected to `6b492b2b` (FA per-head mask) but the bisect grep filtered to `ggml/src/ggml-cuda/` + mimo files and missed `ggml/src/ggml-backend.cpp`, which has a second behaviour-changing commit between the two builds. After review on 2026-05-26 the FA-mask attribution is ruled out and the real suspect is **`0f0f0793` "fix(#83): ggml sched src-mutation log + UNet input pin"**.

**Why FA per-head mask is NOT the cause.**

- Every line of `6b492b2b` is wrapped in `#ifdef GGML_CUDA_CRISPASR_FA_PERHEAD_MASK ... #endif` (both `fattn-mma-f16.cuh` and `fattn.cu`).
- The CMake option `GGML_CUDA_CRISPASR_FA_PERHEAD_MASK` defaults `OFF` (`ggml/CMakeLists.txt:211`).
- No CI / release script anywhere in the repo sets the flag ON — `grep -rn "FA_PERHEAD"` returns only the two CMakeLists.txt entries.
- With the macro undefined the compiled binary is byte-identical to upstream for that path. A self-built `eaee2319` (which is what the reporter has, `/opt/crispasr-main/build/bin/crispasr`) gets OFF by default.

**Why `0f0f0793` is the real suspect.**

The patch adds an *unconditional* src-mutation log + restore in `ggml-backend.cpp` (the generic scheduler, used by every backend including CUDA). Intent: fix chatterbox CFG's cond+uncond gf reuse (chatterbox's S3Gen UNet runs the same gf twice per CFM step, the second call lost the rewired inputs). Bug in the original patch:

- The restore loop ran only on the *success* path of `compute_splits`. If any compute step returned early on a non-`GGML_STATUS_SUCCESS`, the mutation log was retained.
- `split_graph` did not reset `n_src_mutations` at its start, and neither did `sched_reset`. So a next call appended on top of stale entries.
- On the next successful `compute_splits`, the restore loop walked stale `(node, j)` pairs from the *previous* gf and wrote `m->orig_src` to `m->node->src[m->j]`. If the previous gf had been re-laid-out (which mimo-asr does — it builds a fresh cgraph per AR-decoded token), `m->node` is a dangling pointer → write to freed memory → segfault.

mimo-asr's profile fits this exactly: many compute calls per transcribe (one per AR token after the audio adaptor encoder pass). Any one early-return cascades into the next call.

**Hardening shipped: `a5a518c8` "fix(ggml-backend): restore src[j] on every exit path of compute_splits".**

- Extracted the restore loop into `ggml_backend_sched_restore_src_mutations()`.
- Called on every exit path of `compute_splits` (the two early-error returns + the success path).
- Called defensively at the start of `split_graph` and inside `sched_reset` so stale entries from any prior aborted compute are dropped *before* `ggml_free(sched->ctx)`. The restore writes to the user's gf (`orig_src` pointers captured before the rewire), not to `sched->ctx`, so it is safe to call before the free.
- Tightened the realloc-on-grow to assign-after-success.
- Stripped the `// CrispASR patch (#83 r9 follow-up #4)` and `MUST RE-APPLY` markers per `feedback_strip_local_markers.md`.

**Local M1 Metal verification (2026-05-26).**

- `crispasr --backend parakeet ... samples/jfk.wav` — JFK transcribes correctly via the hardened scheduler.
- `crispasr --tts "Hello world test." --tts-output /tmp/tts-smoke.wav -m chatterbox-t3-q8_0-regen.gguf` — CFM solver runs cond+uncond, produces `vocoder mel rms=4.625` (ref rms=5.115; broken-baseline before `0f0f0793` was rms=13.938). The original chatterbox fix is preserved.

**Outstanding: Blackwell sm_120 validation.**

We have no CUDA hardware to confirm the fix end-to-end on the reporter's failing case. **Ask reporter (montvid) to rebuild from `95d74455` or later** and rerun:

```bash
cd /opt/crispasr-main && git pull && cmake --build build -j
/opt/crispasr-main/build/bin/crispasr --backend mimo-asr -m auto --auto-download \
    -f /opt/crispasr-main/samples/jfk.wav -l en -np -nt
```

Expected on the hardened scheduler: transcript matches v0.6.9 reference (`ANd so, my fellow Americans, ask not what your country can do for you.. ASk what you can do for your country..`).

If the segfault persists, next debug step is: `gdb --args` + capture the backtrace + share with us. The other plausible suspects in that case are (a) the `42d1e011` / `db5e22a7` Metal-debug commits leaking into CUDA via shared infrastructure (unlikely — Metal-only files); (b) a separate ggml-cuda bug exposed by Blackwell sm_120 specifically.

**Followups outside this commit.**

1. **Wider validation matrix for any future `ggml-cuda/fattn*` change.** Reporter's request: add mimo-asr, glm-asr, gemma4-e2b, voxtral, granite (multi-head audio-LLM backends) to the GPU validation matrix before landing a kernel change. The 2026-05-24 validation block in HISTORY mentions only a single CTC backend on a single GPU — not enough coverage.
2. **Hold the upstream FA-mask PR draft** at `tools/upstream-prs/06-cuda-fa-perhead-mask.md`. The kernel correctness story still depends on a wider validation matrix; do not submit until that lands.

Cross-refs: `a5a518c8` hardening commit; HISTORY 2026-05-26 follow-up; PLAN #115 (existing JFK-empty-on-Apple-Silicon entry; same root cause, different symptom).

### P1 — funasr / fun-asr-mlt-nano produce `!` loops at every length (reports 01, 07, 08, 09) — DONE `f72d3db1` 2026-05-26

**Shipped:** degenerate-loop guard in the AR decode (bail after the same token id repeats > 20× in a row, with a clear "decode degenerated" message); `frames_spliced` and `fake_token_len` surfaced at `CRISPASR_VERBOSE=1` so the next reporter can confirm/rule out the "encoder collapsed" hypothesis; registry entries added for funasr, fun-asr-mlt-nano, sensevoice, paraformer in `tools/test-all-backends.py` so future regressions get caught loud by the default JFK assertion. Local M1 Metal smoke shows funasr produces the canonical JFK transcript — the `!`-loop is platform/CUDA-specific. **Still open (longer-term):** root-cause the audio adaptor / encoder collapse on Blackwell CUDA; honour `-l` on the funasr prompt template. Both are followups, not blockers, since the loop guard prevents the 60 KB `!` symptom regardless.



**TL;DR.** Both funasr variants emit 60 KB of `!` regardless of audio content or `-l` value. Reproduces on **`samples/jfk.wav` in 3 seconds** (report #07 control test), so this is not a long-audio bug — the model is dead at any length. Report #08 ruled out the "Chinese-prompt vs English-audio mismatch" hypothesis by showing `-l zh` produces byte-identical output to `-l en`. Greedy argmax in `src/funasr.cpp:1375-1400` has no rep-penalty / temperature / degenerate-loop guard; once the joint logits collapse the decoder locks on token id ~5 (`!` in Qwen3 vocab) until `max_new_tokens`.

The remaining suspects are the audio adaptor (`audio_adaptor.*` tensors) and the encoder. Diagnostic the reporter requests: log `frames_spliced` in `funasr_init_from_file` — if it's 0 on an 11 s JFK clip, the encoder/adaptor is the failure.

Report #09 separately notes that **funasr and fun-asr-mlt-nano have no entry in `tools/test-all-backends.py`** — so this regression has been silently shipping since the 2026-05-20 port landed. Sensevoice and paraformer are also absent. The reporter proposes the smallest possible regression guard: assert `"merica"` (case-insensitive) appears in the JFK output text — that catches `!`-loops, language flips, and any obvious decode collapse.

**Fixes:**

1. **Add funasr + fun-asr-mlt-nano + sensevoice + paraformer to `tools/test-all-backends.py`.** Each needs a registry entry with the published GGUF location at `cstr/*-GGUF` (sample skeleton in report #09). The default JFK assertion (`werv < threshold`) at `tools/test-all-backends.py:670` catches the `!`-loop cleanly because `wer("…", "!!!!…") ≈ 1.0`.
2. **Add a degenerate-loop guard in the funasr argmax loop.** Bail after the same `next_id` repeats > 20× in a row. Cheap, model-agnostic stop-loss. Either separately or as part of the broader `core_greedy_decode` work.
3. **Honour `-l` on the funasr path.** — **DONE** `1b491c3d` 2026-06-12. Added `funasr_set_language()` C API + dynamic prompt builder matching upstream `get_prompt(language=...)`. Wired from CLI backend adapter (`params.language`) and session API (`s->source_language`). Kaggle-verified on CUDA: all 4 language configs (default, -l en, -l English, -l zh) produce correct JFK transcripts.
4. **Diagnose the audio adaptor.** Print `frames_spliced` on init at verbose mode; ideally diff the adaptor output against the upstream FunASR Python reference on JFK. If the adaptor is the bug, the prompt fix above is window-dressing.

Cross-refs: HISTORY 2026-05-20 "funasr: FunAudioLLM/Fun-ASR-{Nano,MLT-Nano}-2512 port lands"; 2026-05-21 "funasr: fix MLT-Nano hallucination (PLAN #99)" — only addressed a different failure mode (Chinese-prefix tail drift on the first ~20 correctly-decoded tokens); the present `!`-from-step-0 case is new.

### P2 — firered-asr declares `CAP_UNBOUNDED_INPUT` but pe_maxlen ≈ 50 s (report 04) — DONE `72b74486` 2026-05-26

**Shipped:** `CAP_UNBOUNDED_INPUT` dropped from `crispasr_backend_firered_asr.cpp`; defensive length check in `firered_asr_transcribe_impl` aborts with a clear "input too long (T_sub=X > pe_maxlen=Y; ~Z s of audio after subsampling)" message when callers bypass the VAD dispatcher. Local M1 Metal smoke: JFK still produces the canonical transcript. Long-audio routes through the per-VAD-segment dispatch path. The "JFK silence-only output without `--vad`" subtask (vocab/blank-id mismatch in the auto-downloaded GGUF) was not reproduced in local testing — keeping it on the followup list if a new report surfaces it.



**TL;DR.** firered-asr's encoder has `pe_maxlen = 5000` (relative positional encoding window, ≈ 50 s at 10 ms hop after subsampling). The backend declares `CAP_UNBOUNDED_INPUT`, which tells the dispatcher to bypass per-segment VAD dispatch and pass the whole audio buffer in one call. On a 50 min file `T_sub ≈ 300 000` frames, way past the PE window; the relative-shift attention reads past the PE buffer with no bounds check, producing silent OOB / numerically degenerate output. JFK on the same backend produces a byte-perfect transcript (report #07), so the model itself is fine.

**Fixes:**

1. **Drop `CAP_UNBOUNDED_INPUT` from firered-asr's registry entry.** One-line fix in `examples/cli/crispasr_backend_firered_asr.cpp`. Each VAD segment is well under 50 s; the existing `--vad` path will then dispatch per-segment correctly.
2. **Add an explicit length check** in `firered_asr_transcribe`: return an error with a clear message if `T_sub > pe_maxlen` instead of silently OOB.
3. **Investigate the JFK silence-only output without `--vad`.** Reporter notes that short clips also sometimes return `<Sil>!` only — likely a vocab/blank-id mismatch in the auto-downloaded GGUF, separate from the cap bug.

### P3 — omniasr-llm's `is_streaming` guard prevents chunking on non-streaming GGUFs (report 03) — DONE `5f0aefc0` 2026-05-26

**Shipped:** `src/omniasr.cpp` chunking decision rewritten as `(is_streaming || force_seg) && T_enc > 1`. The segment-marker injection at L1334 stays gated on `is_streaming` (non-streaming variants chunk without the marker — each chunk is decoded as if it's a complete utterance, which is what the model was trained on). Local M1 Metal smoke: JFK still produces the canonical transcript with the known 1-word "americas"→"americans" slip. The wallclock concern (6.8× RT means ~30 min wall for a 50 min file on CPU) is unchanged — chunking dodges the OOM but the realistic remedy for speed is GPU offload for the LLM head.



**TL;DR.** `src/omniasr.cpp:1356-1368` gates the per-segment chunking on `is_streaming` (set from `hp.n_special_tokens == 3` at L1331). For GGUFs without the streaming-mode flag, `n_segments` stays at 1 and the entire 50-minute audio gets fed to a 512-token LLM. The model produces correct text on JFK (report #07), so this is purely a long-audio dispatcher bug.

**Fix:**

Drop the `is_streaming` gate for the chunking decision — always segment past `segment_secs`. Keep the streaming gate only for the segment-marker token injection at L1450 (which actually depends on the special-token vocab). Reporter's patch sketch:

```cpp
const int seg_frames = (int)(hp.segment_secs * 16000.0f) / total_stride;
const bool force_seg = (seg_frames > 0 && T_enc > seg_frames);
if ((is_streaming || force_seg) && T_enc > 1) { ... }
```

Even with this fix, the JFK measurement at 6.8× RT means a 50-minute file would need ~30 min wall on CPU — chunking dodges the OOM, but the realistic remedy for wallclock is GPU offload for the LLM head.

### P4 — gemma4-e2b hallucinates on long audio, works on short (reports 02, 07) — DONE `8bfaff23` 2026-05-26

**Shipped:** defensive 30 s training-window guard in `crispasr_backend_gemma4_e2b.cpp::transcribe()`. Inputs > 30 s abort by default with a clear error message ("input is N s (> 30 s training window). Use --vad to segment, chunk externally, or set CRISPASR_GEMMA4_AUTO_CHUNK=1 to chunk internally"). This stops the symptom — silently-wrong LLM commentary on long audio — and forces the user to route via the segmenter instead. **`CRISPASR_GEMMA4_AUTO_CHUNK=1`** opt-in (`9b5a0a2a`) chunks at 30 s boundaries internally with the same `t_offset_cs` arithmetic as kyutai-stt P6b; off by default because we haven't validated quality at chunk boundaries on long gemma4-e2b output. Local M1 Metal smoke: JFK still produces "ANd so my fellow Americans ask not what your country can do for you, ask what you can do for your country..". **2026-06-13: added `prefers_vad()` override** — auto-enables VAD for >30 s audio (same pattern as parakeet-ja `f950bd1e`). This gives the model silence-bounded segments matching its ~30 s training window, which is the correct approach vs hard-chunking at arbitrary 30 s boundaries. The `CRISPASR_GEMMA4_AUTO_CHUNK` env var path is kept as a fallback but VAD is now the default recommendation. **Kaggle P100 sweep v5 validated 2026-06-13:** 60s audio (5× JFK concatenated) now produces **471 chars** correct transcript via auto-VAD (was 10 chars `<Eos>n0t.!` with hard-chunk). Both default and `CRISPASR_GEMMA4_AUTO_CHUNK=1` paths produce identical 471-char output (VAD fires before the internal chunker). **Still open (longer-term):** Audit the prompt-wiring sanity logs the reporter originally asked for (`audio_soft_token_id`, `proj_dim` vs `d_model`).



**TL;DR.** JFK 11 s transcribes verbatim ("ANd so my fellow Americans ask not what your country can do for you, ask what you can do for your country.."). On the 50 min file the model emits unrelated LLM commentary ("…a holistic view of the self and the concept of the energy body…") starting with `<Eos>!` — meaning it emitted `<end_of_sequence>` immediately after the prompt and then continued into a generic response. So this is a **chunking / long-context bug**, not the audio-soft-token-id mismatch hypothesised in report #02 before the control test in #07.

**Fixes:**

1. **Audit the chunking path** — most likely the dispatcher hands the entire file to the LLM in one prompt without segmenting; the model hits `<eos>` after the first chunk's worth of audio and then continues in an "I see you started a sentence, let me complete the topic" mode.
2. **Sanity log** at init: `audio_soft_token_id`, `proj_dim` vs `d_model`, "audio projection weights found" — even though the report #07 control test ruled these out, the original report #02 asked for them and they're cheap to surface.

### P5 — mimo-asr tokenizer GGUF not in auto-download manifest (report 06) — DONE `b936b488` 2026-05-26

**Shipped:** `src/crispasr_model_registry.cpp` mimo-asr entry now declares `mimo-tokenizer-q4_k.gguf` from `cstr/mimo-tokenizer-GGUF` as the companion file, so `--auto-download` fetches both LM and tokenizer into `~/.cache/crispasr/` where `discover_audio_tokenizer()` finds it without further configuration. Error message in `crispasr_backend_mimo_asr.cpp` now spells out three options for resolving a missing tokenizer (`--auto-download`, `--codec-model`, or manual `hf download`). `docs/architecture.md` mimo-asr section documents the tokenizer-is-a-separate-file requirement and the `--codec-model` override. Confirmed that `discover_audio_tokenizer`'s candidate list already includes the canonical filename, so the auto-download lands where the runtime looks.



**TL;DR.** `--auto-download` fetches the 36-layer Qwen2-based LM but **not** the separate `mimo-tokenizer-q4_k.gguf` (~395 MB audio tokenizer) required to actually transcribe. The user gets exit code 1 with "no audio tokenizer GGUF found. Pass --codec-model PATH or place mimo-tokenizer-q4_k.gguf next to the LM" and a 0-byte output. The `--codec-model` flag is undocumented; `docs/cli.md` doesn't mention it under the mimo-asr section.

Reporter's verified workaround (report #06):

```bash
hf download cstr/mimo-tokenizer-GGUF --local-dir ~/.cache/crispasr/mimo-tokenizer
crispasr --backend mimo-asr -m auto --auto-download \
  --codec-model ~/.cache/crispasr/mimo-tokenizer/mimo-tokenizer-q4_k.gguf \
  -f samples/jfk.wav -l en -np -nt
```

This works on v0.6.9 (per report #12). On v0.6.10 it still hits the P0 segfault above; needs P0 to land first before this is testable.

**Fixes:**

1. **Add `mimo-tokenizer-q4_k.gguf` to the auto-download manifest** for the mimo-asr backend. Source: `cstr/mimo-tokenizer-GGUF`.
2. **Document `--codec-model`** under the mimo-asr section of `docs/cli.md`. Include `discover_audio_tokenizer()`'s search-path convention (the three filenames it tries next to the LM).
3. **Improve the error message** to include a concrete `huggingface-cli download` line — the current message tells the user the flag exists but not how to populate the file.

### P6 — kyutai-stt: three separate issues (reports 05, 10, 11)

**TL;DR.** kyutai-stt has three distinct bugs, all visible on short audio.

#### P6a. Drops the final word on the 11 s JFK clip (report 10) — DONE `ba0e388e` 2026-05-26

**Shipped:** `crispasr_backend_kyutai_stt.cpp::transcribe()` constructs a padded buffer of `n_samples + 8000` (500 ms @ 16 kHz of zeros) and feeds *that* to `kyutai_stt_transcribe_ex`. Tokens emitted during the silence-tail keep their `t_offset_cs` arithmetic and land a few cs past the original input end; word timestamps stay correct since the model is causal. Local M1 Metal smoke: `samples/jfk.wav` now produces "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country." (full final word + sentence-end punctuation, was truncated to "...your c" before).

#### P6b. 0.07× RT on 50 min file (reports 05, 11) — DONE `043b3ae5` 2026-05-26

**Shipped:** `crispasr_backend_kyutai_stt.cpp::transcribe()` extracted the per-call logic into a `transcribe_one()` helper and wraps with a chunking loop that splits inputs > 30 s into 30 s windows. Each chunk gets its own silence-tail flush from P6a so boundaries close cleanly; per-chunk `t_offset_cs` = `caller_offset + chunk_start / 16 kHz * 100 cs`, so token + word timestamps land in the right global window. Local M1 Metal validation on `first90.wav` (the 90 s Japanese clip from issue #89, the original long-audio failure case) produced a coherent three-chunk transcript in 568 s wall = 6.3 s/s — finite and linear, vs the previously-reported 14 s/s degradation that grew worse with input length. The single-chunk path (n_samples ≤ 30 s) is unchanged, so the JFK regression test still passes. GPU offload (`docs/architecture.md:248` TODO) remains the longer-term remedy for absolute wallclock on CPU.

#### P6c. Streaming model on a batch dispatcher fundamentally mismatched — DEFERRED

The architecture entry at `docs/architecture.md:176` lists kyutai-stt as "Mimi codec + causal LM | CPU". The reporter's evidence supports treating this backend as **streaming-only** in the CLI — the batch path is a footgun. With P6b shipped, the wallclock is bounded and linear (no more "hung" appearance), so the defensive `--force-long-audio` cap is now a UX nicety rather than a correctness fix. Deferred until a user reports it; we have higher-value followups for the next session.

### Cross-finding observations from the issue

- **JFK as the universal control test.** Reports #07 and #12 both demonstrate that running the project's own 11 s fixture isolates "model is dead" from "long-audio dispatcher is broken" in <2 minutes — the two failure classes have completely different fix sites. This is the reporter's most actionable methodology observation. Future "broken backend" reports should run JFK first.

- **Backend registry coverage.** Reporter found four backends missing from `tools/test-all-backends.py` (funasr, fun-asr-mlt-nano, sensevoice, paraformer). The script advertises itself as the source of truth in `docs/regression-matrix.md`; a gap there means a backend can ship broken indefinitely. Worth a parallel audit for the 18-backend registry the script does cover, to confirm nothing else has silently grown out of date.

- **Capability-flag honesty.** firered-asr declaring `CAP_UNBOUNDED_INPUT` while having a 50 s PE window is the same class of failure as the previous voxtral / cohere / gemma4-e2b / glm-asr / kyutai-stt cases that drove the opt-out fix train (`dc2295b2` etc.). Worth a defensive sweep of every `CAP_UNBOUNDED_INPUT` declaration to confirm the *encoder* actually is unbounded, not just the dispatcher's input shape.

- **GPU validation matrix.** The 2026-05-24 `#81 #06 FA per-head mask` patch was validated only on parakeet-tdt-0.6b-v3 on A1000 sm_86; the present mimo-asr regression on Blackwell sm_120 is the proximate cost of that narrow validation. The `tools/upstream-prs/06-cuda-fa-perhead-mask.md` PR draft should not be submitted upstream until validated on a wider matrix.

### Priority for this PLAN section

Reporter's classification + ours:

| # | finding | reporter severity | our action priority |
|---|---|---|---|
| 12 | mimo-asr segfault on Blackwell (regression from `6b492b2b`) | regression | **P0** |
| 01/07/08/09 | funasr / fun-asr-mlt-nano `!`-loop + no CI coverage | broken backend | P1 |
| 04 | firered-asr `CAP_UNBOUNDED_INPUT` + 50 s PE window | broken on long audio, model OK | P2 |
| 03 | omniasr-llm `is_streaming` chunking gate | broken on long audio, model OK | P3 |
| 02 | gemma4-e2b long-audio hallucinations | broken on long audio, model OK | P4 |
| 06 | mimo-asr auto-download manifest gap | UX bug, blocked by P0 segfault | P5 |
| 05/10/11 | kyutai-stt batch path slow + final-word truncation | partly design-limit, partly bug | P6 |

PLAN #115 (existing) folds into P0 + P5 here.

### Files (tentative)

- `ggml/src/ggml-cuda/CMakeLists.txt` — default `GGML_CUDA_CRISPASR_FA_PERHEAD_MASK` to OFF for non-parakeet builds, or gate it on a backend capability declaration
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, `fattn.cu` — kernel gate tightening (P0 option 2)
- `tools/test-all-backends.py` — add funasr / fun-asr-mlt-nano / sensevoice / paraformer registry entries (P1)
- `src/funasr.cpp` — degenerate-loop guard + per-variant prompt selection + `params.language` wiring (P1)
- `examples/cli/crispasr_backend_firered_asr.cpp` — drop `CAP_UNBOUNDED_INPUT` (P2)
- `src/firered_asr.cpp` — length check + clear error past `pe_maxlen` (P2)
- `src/omniasr.cpp` — `is_streaming || force_seg` chunking gate (P3)
- `examples/cli/crispasr_backend_gemma4_e2b.cpp` — chunking audit (P4)
- `examples/cli/crispasr_backend_mimo_asr.cpp` + auto-download manifest — fold tokenizer in (P5)
- `docs/cli.md` — document `--codec-model` for mimo-asr (P5)
- `examples/cli/crispasr_backend_kyutai_stt.cpp` — silence tail or `finalize` (P6a), 30 s chunking (P6b)
- `tests/test-*` — regression assertions per fix
- `tools/upstream-prs/06-cuda-fa-perhead-mask.md` — do not submit upstream until P0 wider-matrix validation lands

### Trigger conditions for completion

- mimo-asr v0.6.11 (next release) does not segfault on Blackwell, validated by montvid or by us on the same GPU class.
- funasr + fun-asr-mlt-nano produce a non-degenerate transcript on JFK; CI regression assertion in place.
- firered-asr, omniasr-llm, gemma4-e2b all transcribe a 5-min EN clip without hangs, dropped content, or hallucinations.
- mimo-asr `--auto-download` fetches both LM and tokenizer; `docs/cli.md` documents `--codec-model`.
- kyutai-stt JFK transcript ends on `country.`; 5-min EN clip completes in linear wall-time.

Reporter contact: `montvid` on GitHub issue #125. The 12 reports are reproducible verbatim; their environment is well-documented enough that we can re-run on our VPS to cross-check before claiming any fix.

---

## 127. Coverage gaps from the 2026-05-26 overlap-save sweep close-out

Three small holes the sweep + #115 bisect surfaced. None are urgent; recording so the next contributor doesn't rediscover the same gaps.

### a. omniasr-llm — overlap-save bug status unknown

The original 5 min sweep and the 90 s rerun both came back `BOTH_EMPTY` for `omniasr-llm-300m-v2-q4_k.gguf`: default and `--chunk-overlap 0` both hit the 20 min per-pass wallclock on M1. Probably *slow*, possibly *also has the truncation bug like its sibling backends*. Can't tell without a faster box.

**Fix shape.** Re-run `./tools/check-overlap-save-bug.sh omniasr-llm` with `PER_RUN_TIMEOUT=2400` on a Linux x86 host (the VPS) or a Kaggle CPU kernel. If default produces materially less output than no-overlap, add to the opt-out list in `examples/cli/crispasr_chunk_context_gate.h`. If both produce the same content, mark VERIFIED-OK in the harness comment.

### b. mimo-asr — local test coverage is in place but doesn't run in CI

`tools/test-all-backends.py` has had a `mimo-asr` registry entry since 2026-05-02 (commit `2aeaf4c4`); the `test_transcribe` function explicitly handles `EMPTY` output (line 693-695). Locally the test would have caught PLAN #115's silent-empty regression at runtime — but CI doesn't run `test-all-backends.py` against large-model backends (mimo Q4_K is 4.2 GB, doesn't fit in the standard runner disk budget per pre-release), so the regression shipped in v0.6.10 anyway.

**Fix shape.** Either (a) Kaggle scheduled-CI workflow that runs the full `test-all-backends.py` against the 4 LLM-class backends (mimo, voxtral, gemma4-e2b, granite-4.1-2b) on each main push — patterns in `tools/kaggle/crispasr-regression.py` already handle the model-download + heartbeat parts; (b) cheaper, a documented `make smoke-llm-backends` target that release scripts run before tagging. (a) is more reliable; (b) is one afternoon of work.

### c. cohere-asr-ja-v0.1 — no benchmark numbers in PERFORMANCE.md

Issue #123 added the JA variant to the registry + README, but no row in any of `PERFORMANCE.md`'s cohere tables (the long-form coverage at line 1374, the cross-backend matrix at line 1535, the per-length wall-time table at line 1555). The English `cohere-transcribe` is benchmarked across multiple Japanese / English / multilingual clips; the JA fine-tune isn't.

**Fix shape.** Run the JA variant on the same fixture set the English one used (TedX / JSUT clips per the model card; `samples/jfk.wav` is English so won't exercise the JA tuning). Drop one extra row into each cohere table with the JA numbers. ~30 min of inference + table updates once the fixtures are downloaded.

---

## 128. Piper TTS — lightweight VITS runtime (MIT)

Native C++ runtime for [rhasspy/piper](https://github.com/rhasspy/piper)
VITS models. MIT-licensed. Fills a gap in the TTS lineup: tiny models
(~15-50 MB Q4_K per language) with zero-shot latency, useful on mobile
(CrisperWeaver) and for fast previews.

### Why

Current smallest TTS is Kokoro at ~75 MB. Piper voices are **~15 MB**
Q4_K per language and run in single-digit ms per sentence on CPU.
250+ community voices across 30+ languages. Especially strong for
German (thorsten-medium 6.1% MOS, karlsson, kerstin, pavoque, ramona,
eva_k). The "just works" option when download budget is tight.

### Architecture

VITS (Variational Inference with adversarial learning for end-to-end
Text-to-Speech):
- Text encoder: small transformer (6 layers, 192-d)
- Duration predictor: 2-layer conv (already in `core/conv.h`)
- Flow: 4 affine coupling layers (inverse autoregressive flow)
- HiFi-GAN decoder: 4 upsample blocks + multi-receptive-field fusion

Phoneme frontend: **espeak-ng** — already vendored in the tree for
Kokoro (#56). The `kokoro.cpp` espeak-ng integration
(`espeak_TextToPhonemes`) is directly reusable; Piper's phoneme
alphabet is espeak-ng IPA, same as Kokoro's.

### Reuse from existing code

| Component | Reuse source | New code needed |
|---|---|---|
| espeak-ng phonemizer | `kokoro.cpp` `espeak_TextToPhonemes` | None — same API |
| Text encoder (transformer) | `core/attention.h` `core_attn::kv_self_attn` + `core/ffn.h` | Minimal glue |
| 1D convolutions | `core/conv.h` (`core_conv_1d`, `core_conv_1d_dw`) | None |
| Duration predictor | `core/conv.h` | ~50 LOC adapter |
| Affine coupling flow | **NEW** | ~200 LOC — `core/affine_coupling.h` |
| HiFi-GAN decoder | `chatterbox_s3gen.cpp` HiFT vocoder (4 upsample + MRF) | Adapt from chatterbox; ~300 LOC delta |
| iSTFT / audio output | `core/fft.h` + `chatterbox_s3gen.cpp` istft | None |
| GGUF loader | `core/gguf_loader.h` | None |
| Audio resampler | `core/audio_resample.h` | None |

The affine coupling layer is the only truly new primitive. It's a
simple invertible transform: `y = x * exp(s(x)) + t(x)` where `s`
and `t` are small conv nets. ~200 LOC including forward + inverse.
**This should go into `core/affine_coupling.h`** per DRY — it will
also be needed by MeloTTS (#100 Phase A, same VITS family) and any
future normalizing-flow TTS.

### Concrete steps

1. **Converter** — `models/convert-piper-to-gguf.py`. Read the `.onnx`
   + `.onnx.json` config. Export text encoder, duration predictor, flow
   coupling layers, HiFi-GAN decoder as GGUF tensors. Embed the
   phoneme-to-id map as GGUF KV metadata.
2. **`core/affine_coupling.h`** — forward-pass affine coupling layer.
   Input: (B, C, T) → split channels → compute s,t via conv stack →
   apply transform → concat. ~200 LOC. Reusable by #100 MeloTTS.
3. **`src/piper_tts.cpp` + `src/piper_tts.h`** — backend runtime.
   - `piper_tts_init_from_file(path)` → load GGUF, build encoder +
     flow + decoder graphs
   - `piper_tts_synthesize(ctx, text)` → espeak-ng phonemize → encoder
     → duration → flow → HiFi-GAN → float32 PCM
   - Wire into Session API (`crispasr_c_api.cpp`)
4. **Registry** — add `piper` backend to `crispasr_model_registry.cpp`.
   Host converted GGUFs at `cstr/piper-*-GGUF` on HF.
5. **Test** — ASR roundtrip: piper synth → parakeet transcribe → verify.
   Add to `tools/test-all-backends.py`.

### Effort

**Small-Medium.** espeak-ng and HiFi-GAN are already in the tree.
The affine coupling is small. Main work is the converter + wiring.
~1-2 days for a working EN/DE prototype, +1 day per additional
language voice.

### Trigger

Immediate — Piper's size/speed makes it the best candidate for
CrisperWeaver mobile and HF Space demos. Also the simplest new
TTS architecture to add (no LLM, no codec, no diffusion).

---

## 129. F5-TTS — DiT flow-matching TTS (MIT)

Native C++ runtime for [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS).
MIT-licensed. High-quality zero-shot voice cloning from 5-15s of
reference audio.

### Why

F5-TTS's architecture (Diffusion Transformer + flow matching) is
distinct from everything currently in the tree. It produces noticeably
higher quality than the current AR-based engines (orpheus, qwen3-tts)
for voice cloning tasks, and the MIT license is cleaner than the
llama3.2-derived Orpheus weights. ~330M params, ~660 MB F16.

### Architecture

- **Text encoder**: char-level ConvNeXt blocks (not a transformer) —
  novel in the tree
- **DiT backbone**: 22-layer Diffusion Transformer with AdaLN-Zero
  conditioning. Each layer: LayerNorm → self-attention → cross-attention
  → FFN with adaptive scale/shift from the diffusion timestep embedding
- **Flow matching**: conditional OT path (rectified flow). The ODE
  solver is Euler (simplest) or midpoint
- **Vocoder**: Vocos (iSTFT-based, ConvNeXt stack → STFT magnitudes +
  phases → iSTFT). ~14M params, separate checkpoint

### Reuse from existing code

| Component | Reuse source | New code needed |
|---|---|---|
| Self-attention | `core/attention.h` `core_attn::kv_self_attn` | None |
| Cross-attention | `core/attention.h` | None |
| FFN (SwiGLU) | `core/ffn.h` `core_ffn::swiglu` | None |
| AdaLN-Zero | **NEW** — `core/adaln.h` | ~100 LOC |
| ODE solver (Euler/midpoint) | `chatterbox_s3gen.cpp` `cfm_euler_solve` | Adapt — ~50 LOC delta |
| iSTFT | `core/fft.h` + `chatterbox_s3gen.cpp` | None |
| Mel spectrogram | `core/mel.h` | None |
| ConvNeXt block | **NEW** — `core/convnext.h` | ~150 LOC |
| GGUF loader | `core/gguf_loader.h` | None |
| Reference audio embedding | `core/mel.h` + concat | ~50 LOC |

**New `core/` primitives** (both reusable by other future models):
- `core/adaln.h` — Adaptive Layer Norm with zero-init scale/shift.
  Used by all DiT-family models. ~100 LOC. Would also serve any
  future DiT image/video model if one is ever ported.
- `core/convnext.h` — ConvNeXt V2 block (depthwise conv → LayerNorm →
  pointwise up → GELU → pointwise down + residual). ~150 LOC. Vocos
  vocoder and F5's text encoder both use this. MeloTTS (#100) could
  also benefit if its text encoder is ConvNeXt-flavored.

### Concrete steps

1. **Converter** — `models/convert-f5-tts-to-gguf.py`. Export DiT
   (22 layers), text encoder (ConvNeXt), Vocos vocoder as separate or
   combined GGUF. Embed char-level vocabulary as KV metadata.
2. **`core/adaln.h`** — AdaLN-Zero: `scale, shift = linear(timestep_emb)`;
   `out = (1 + scale) * layernorm(x) + shift`. Zero-init at construction.
3. **`core/convnext.h`** — ConvNeXt V2 block.
4. **`src/f5_tts.cpp` + `src/f5_tts.h`** — backend runtime.
   - Reference audio: load WAV → mel → concat with text embeddings as
     conditioning (masked infilling: text tokens mark where to generate,
     ref mel provides voice identity)
   - DiT forward: 22 layers × N ODE steps (default 32)
   - Vocos: mel → magnitude + phase → iSTFT → 24 kHz PCM
5. **Registry + C API + Session wiring.**
6. **Voice cloning API**: `session.set_voice("ref.wav", ref_text="...")` —
   same API as qwen3-tts base. Mel from ref audio is the conditioning.

### Status — **DONE** (2026-05-30)

Full native C++ runtime operational. End-to-end pipeline: WAV ref →
mel spectrogram → text tokenization → ConvNeXtV2 text encoder →
InputEmbedding + ConvPosEmbed → 32-step Euler ODE with CFG (22-layer
DiT) → Vocos vocoder (8× ConvNeXt + ISTFTHead) → 24 kHz WAV.

All 22 DiT layers match PyTorch reference at cos=1.000. ASR roundtrip
verified (whisper transcribes generated audio correctly). CLI wired
(`--backend f5-tts`), C API wired, model registry entry added.

Key bug found during port: x_transformers `RotaryEmbedding` interleaves
frequencies (`stack + rearrange`, not `cat`), so paired RoPE elements
share the same frequency — a standard rotation. Original analysis in
the handover was incorrect. See `handover-prompts/f5-tts-129-continuation.md`
Bug 10 for details.

Performance: unified ggml graph (ggml_rope_ext + ggml_flash_attn_ext)
gives 5x speedup (DiT forward ~2 min → ~24 sec per ODE step).

Quantization: F16 (953 MB) is the only viable precision. Q8_0/Q4_K
tested with arch-specific conditioning-pathway skip rules — still
produce unintelligible output because QKV/FFN error compounds through
1408 iterative forward passes. `crispasr-quantize` skips F5-TTS.
HF repo `cstr/f5-tts-GGUF` has the F16 GGUF only.

---

## 130. Zonos TTS — transformer + DAC codec (Apache 2.0)

**Status (2026-06-09): DONE.** End-to-end synthesis verified. ASR
roundtrip: "Hello world." → verbatim; fox sentence → verbatim on Q8_0
and selective Q4_K.

**Bugs fixed (diff-harness driven):**
1. RoPE type NEOX → NORMAL (`GGML_ROPE_TYPE_NORMAL`): Zonos uses
   x_transformers `apply_rotary_emb` which does consecutive-pair
   rotation. NEOX (half-split) was wrong; fixed prefill_hidden cos
   0.984 → 0.996.
2. GatedMLP: `fc2(y * silu(gate))` where y=chunk0, gate=chunk1. Old
   code used `swiglu_fused_gate_up` which applied silu to chunk0.
3. Delay pattern offset: `step >= k` (not `step > k`).

**Quantization:**
- Uniform Q4_K inflates EOS logit at AR step 0 by ~0.9 units
  (−1.125 → >0), P(EOS) > 60 %, all seeds fail.
- Fix: selective Q4_K keeps `heads.*` + `embeddings.*` +
  `prefix_conditioner.*` at F16 (82 MB overhead). Backbone (210
  projection tensors) quantized. Result: 931 MB, ~25 % of seeds
  still hit step-0 EOS; resolved by 3-retry guard in
  `zonos_tts_synthesize` (20/20 seeds pass after ≤ 2 retries).
- DAC 44kHz codec cannot be block-quantized: all Conv1d weights have
  kernel-size as ne[0] (≤ 16 < 32 min for Q8_0). F16 only (104 MB).

**GGUFs shipped:** `cstr/zonos-v0.1-transformer-GGUF` (F16 3.0 GB,
Q8_0 1.6 GB, selective-Q4_K 931 MB) + `cstr/dac-44khz-GGUF` (F16
104 MB). Default `-m auto` → Q8_0.

**Integration:** `docs/contributing.md` checklist completed:
backend detection, C ABI sibling discovery + `set_codec_path`,
model registry, docs/tts.md, docs/architecture.md, README.md,
Python/Go/Dart binding comments.

Native C++ runtime for [Zyphra/Zonos](https://github.com/Zyphra/Zonos).
Apache 2.0 licensed. Unique in the lineup for its fine-grained acoustic
conditioning (pitch, speaking rate, emotion via speaker embeddings).

### Why

Zonos's speaker conditioning is richer than any current backend:
controllable pitch, speaking rate, and emotion arrays from reference
audio. This makes it the best candidate for expressive TTS where the
user wants to *tune* the output voice character, not just clone it.
~500M params. 44.1 kHz output (highest SR in the lineup, tied with
MeloTTS).

### Architecture

- **Text encoder**: character-level transformer + language embedding
- **AR backbone**: ~24-layer transformer generating DAC audio codes
  (8 codebooks × 50 Hz). Similar shape to orpheus (Llama AR → codec)
  but with conditioning injection at every layer
- **DAC codec decoder**: Descript Audio Codec — residual VQ → upsampling
  conv stack → 44.1 kHz waveform. Structurally similar to SNAC
  (orpheus) but different codebook structure
- **Speaker conditioning**: reference audio → mel → small encoder →
  embedding vector. Injected via cross-attention or AdaLN at each
  AR layer. The conditioning also accepts explicit float arrays for
  pitch/rate/emotion override

### Reuse from existing code

| Component | Reuse source | New code needed |
|---|---|---|
| AR transformer (Llama-style) | `orpheus.cpp` talker forward / `core/attention.h` + `core/ffn.h` | Minimal — conditioning injection is the delta |
| KV cache | `orpheus.cpp` / `qwen3_tts.cpp` | None |
| RVQ codebook dequant | `core/rvq.h` (`rvq_dequantize`) | None — DAC uses same RVQ pattern |
| Upsampling conv decoder | `orpheus_snac.cpp` SNAC decoder / `indextts_voc.cpp` BigVGAN | Adapt — DAC has different layer count/strides |
| Speaker embedding | `chatterbox_campplus.cpp` CAMPPlus / `titanet.cpp` | Adapt — Zonos uses its own encoder but the TDNN pattern overlaps |
| Mel spectrogram (for ref) | `core/mel.h` | None |
| GGUF loader | `core/gguf_loader.h` | None |
| Greedy decode loop | `core/greedy_decode.h` | None |

**New `core/` primitive:**
- `core/dac_decoder.h` — Descript Audio Codec upsampling decoder.
  Similar to SNAC but with different stride pattern (256× total
  upsampling from 50 Hz codes to 44.1 kHz). ~300 LOC. Reusable by
  any future DAC-based model.

The conditioning-injection mechanism (pitch/rate/emotion floats →
layer-wise adaptive bias) is backend-specific and belongs in
`zonos_tts.cpp`, not `core/`.

### Concrete steps

1. **Converter** — `models/convert-zonos-to-gguf.py`. Export text encoder,
   AR transformer, DAC decoder, speaker encoder as single GGUF. Embed
   character vocab + conditioning config as KV metadata.
2. **`core/dac_decoder.h`** — DAC RVQ → conv upsample → 44.1 kHz.
3. **`src/zonos_tts.cpp` + `src/zonos_tts.h`** — backend runtime.
   - `zonos_tts_init_from_file(path)` → load, build graphs
   - `zonos_tts_set_conditioning(pitch, rate, emotion[])` → session state
   - `zonos_tts_synthesize(ctx, text)` → AR decode → DAC decode → PCM
   - Voice cloning: `set_voice("ref.wav")` → speaker embedding extraction
4. **Registry + C API + Session wiring.**
5. **New session setters** in `crispasr_c_api.cpp`:
   `crispasr_session_set_pitch(float)`,
   `crispasr_session_set_speaking_rate(float)`,
   `crispasr_session_set_emotion(float *array, int len)`.

### Effort

**Medium.** The AR transformer reuses orpheus/qwen3-tts patterns heavily.
DAC decoder is a new codec but structurally similar to SNAC. The unique
part is the conditioning system. ~2 weeks.

### Trigger

Medium priority — start after F5-TTS (#129) so the `core/adaln.h`
primitive exists (Zonos may use AdaLN for conditioning injection,
depending on the exact layer design).

---

## 131. OuteTTS — LLM + WavTokenizer codec (CC BY 4.0)

Native C++ runtime for [OuteAI/OuteTTS](https://github.com/OuteAI/OuteTTS).
CC BY 4.0 (commercial OK with attribution). Zero-shot voice cloning
from a brief reference clip.

### Why

OuteTTS is architecturally closest to what CrispASR already runs well:
a GPT-style LLM generating discrete audio tokens decoded by a learned
codec. The pattern is nearly identical to orpheus (Llama → SNAC) and
indextts (GPT-2 → BigVGAN). This makes it the **lowest-effort new TTS
backend** among the four — mostly converter + registry work, minimal
new runtime code.

### Architecture

- **LLM backbone**: Llama-style 1B transformer (OuteTTS-0.3-1B uses
  a custom 1B arch; OuteTTS-0.2-500M uses a 500M variant)
- **Audio tokenizer**: WavTokenizer — single-codebook VQ at 40 or 75 Hz.
  Encoder: conv stack → quantize. Decoder: conv stack → 24 kHz waveform.
  Structurally simpler than SNAC (1 codebook vs 3) but similar decode
  pattern
- **Voice cloning**: reference audio → WavTokenizer encode → prepend
  tokens to LLM context. Same in-context-learning pattern as
  qwen3-tts-base

### Reuse from existing code

| Component | Reuse source | New code needed |
|---|---|---|
| Llama-style AR forward | `orpheus.cpp` / `qwen3_tts.cpp` | Minimal — same core_attn + core_ffn |
| KV cache + greedy/beam | `core/greedy_decode.h` / `core/beam_decode.h` | None |
| Conv-stack codec decoder | `orpheus_snac.cpp` / `indextts_voc.cpp` | Adapt for WavTokenizer strides — ~200 LOC |
| Reference audio encoding | `core/mel.h` → WavTokenizer encoder | ~300 LOC for the encoder conv stack |
| BPE tokenizer | `core/bpe.h` | None |
| GGUF loader | `core/gguf_loader.h` | None |
| Audio resampler | `core/audio_resample.h` | None |

No new `core/` primitives needed. WavTokenizer's decoder is a
standard conv-upsample stack — put it in `src/outetts_wavtok.cpp`
(backend-specific, not generic enough for `core/` since it's a
single-codebook design unlike the multi-codebook RVQ in `core/rvq.h`).

### Concrete steps

1. **Converter** — `models/convert-outetts-to-gguf.py`. Export LLM
   weights + WavTokenizer encoder/decoder. Embed vocab + audio config
   as KV metadata.
2. **`src/outetts_wavtok.cpp`** — WavTokenizer encode (for ref audio)
   + decode (for generated tokens). ~400 LOC total.
3. **`src/outetts.cpp` + `src/outetts.h`** — backend runtime.
   - Load GGUF, build LLM + WavTokenizer graphs
   - Voice cloning: ref WAV → WavTokenizer encode → token sequence
   - AR decode: text tokens + ref tokens → generate audio tokens
   - WavTokenizer decode → 24 kHz PCM
4. **Registry + C API + Session wiring.**
5. **Test** — ASR roundtrip + voice similarity check.

### Status — DONE

_Done — see HISTORY.md + git log._

## Priority update — new TTS backends

(Merged into the main priority table above. Piper §128 DONE. F5 §129
in progress. Zonos §130 and OuteTTS §131 queued.)

Sequencing: **§128 → §129 → §130 → §131**. Piper first (smallest,
most reuse, biggest size-class gap to fill). F5 second (introduces
`core/adaln.h` + `core/convnext.h` that §130 may need). Zonos third
(introduces `core/dac_decoder.h`). OuteTTS last (least new value
given orpheus/indextts coverage).

---

## §136 — funasr CUDA !-loop fix (issue #125)

**Status:** DONE — fix confirmed on Kaggle P100 (v16, 2026-06-01).

**Root cause:** `ggml_backend_sched` with `[CUDA,CPU]` misroutes funasr's
Qwen2-0.6B LLM decoder on CUDA. Produces Inf at layer 2, all-NaN by
layer 3. Not caused by the Q/K/V split pattern specifically — v15 proved
QKV fusion alone does NOT fix it. The exact sched bug is upstream/unknown.

**Fix (commit `f94fec90`):** weight-split (encoder GPU, LLM+KV CPU) +
QKV fusion + KV zeroing. `FUNASR_LLM_GPU=1` overrides to all-GPU.
See LEARNINGS §136 for the 16-version Kaggle investigation.

**Future:** file upstream ggml issue with the minimal repro (funasr Q8_0
model, dual-backend sched, all weights on GPU → Inf at LLM layer 2).
If fixed upstream, revert the weight split via `FUNASR_LLM_GPU=1`.

---

## §138 SpeechT5 + Dia + Parler + FastPitch TTS stubs → working backends

**Status (2026-06-01):**

### SpeechT5 TTS (microsoft/speecht5_tts)
- **Encoder**: cos > 0.999 all 12 layers ✅
- **Converter**: `models/convert-speecht5-to-gguf.py` — F16 weights, F32 biases
- **Runtime**: `src/speecht5_tts.cpp` — encoder + decoder w/ KV cache + postnet + HiFi-GAN
- **GGUF**: `/mnt/storage/speecht5/speecht5-tts-f16.gguf` (300 MB)
- **Status**: Pipeline runs e2e, produces audio. Decoder content mismatch needs investigation.
- **Next**: validate decoder per-layer against Python reference

### Dia 1.6B TTS (nari-labs/Dia-1.6B)
- **Encoder**: cos = 1.000000 all 12 layers ✅
- **Decoder layer 0**: cos = 0.999 ✅
- **Decoder step 0 argmax**: channel 0 = 568, matches Python ✅
- **Converter**: `models/convert-dia-to-gguf.py` — F32 weights (scale=1.0 attention sensitive)
- **Runtime**: `src/dia_tts.cpp` — encoder + cross-attn + AR decoder (18L GQA CFG) + DAC decode
- **DAC**: `models/convert-dac-to-gguf.py` + `/mnt/storage/dia/dac-44khz.gguf` (104 MB)
- **GGUF**: `/mnt/storage/dia/dia-1.6b-f16.gguf` (3.2 GB F16)
- **Status**: 11 bugs fixed. Audio produced (2.15s) but ASR says music/noise. Full 18-layer decoder precision needs validation. CFG filtering now matches Python blueprint.
- **Next**: validate decoder layers 1-17, test with F32 GGUF, investigate DAC decode fidelity
- **Key insight**: Dia's `scale=1.0` attention (no 1/sqrt(d)) makes softmax extremely sensitive to precision. Every computation must match Python exactly or codes diverge.

### Parler TTS / FastPitch
- Not started yet — Dia and SpeechT5 took priority
- FastPitch has ~1000 LOC stub, converter exists, NeMo model needed
- Parler has ~857 LOC stub with 3 TODOs, T5 encoder + DAC decoder

**Files changed**: `src/dia_tts.cpp`, `src/speecht5_tts.cpp`, `src/core/hifigan.h`, `src/funasr.cpp`, `models/convert-dia-to-gguf.py`, `models/convert-speecht5-to-gguf.py`

## §139 Beam search — remaining ASR backends (issue #136 follow-up)

Parakeet TDT/RNNT beam search shipped in `b3cdcebd` (2026-06-01).
Whisper, glm-asr, kyutai-stt, moonshine, firered-asr, granite,
qwen3, voxtral, omniasr already had beam search via `core_beam_decode.h`
or native runtime support. This section tracks the remaining gaps.

### Current coverage

| Backend | Beam search | Mechanism |
|---|---|---|
| whisper | ✔ | native upstream |
| parakeet | ✔ | TDT/RNNT label-looping beam (`b3cdcebd`) |
| granite / granite-4.1 / granite-4.1-plus | ✔ | `core_beam_decode` replay-from-prefix |
| qwen3-asr | ✔ | `core_beam_decode` replay-from-prefix |
| voxtral | ✔ | `core_beam_decode` replay-from-prefix |
| glm-asr | ✔ | `core_beam_decode` branched KV snapshots |
| kyutai-stt | ✔ | `core_beam_decode` branched KV snapshots |
| moonshine | ✔ | native `moonshine_set_beam_size` |
| firered-asr | ✔ | native beam (default beam=3) |
| omniasr | ✔ | wired; CTC variant ignores |
| gemma4-e2b | ✔ | `core_beam_decode` replay-from-prefix (`f5b28564`) |
| canary | ✔ | `core_beam_decode` branched KV snapshots (§90 runtime + `f5b28564` adapter) |
| cohere | ✔ | `core_beam_decode` branched KV snapshots (§90 runtime + `f5b28564` adapter) |
| m2m100 | ✔ | `core_beam_decode` replay-from-prefix (`84d86a99`) |
| madlad/t5 | ✔ | `core_beam_decode` replay-from-prefix (`84d86a99`) |
| moonshine-streaming | ✔ | `core_beam_decode` branched KV snapshots (`61136713`) |
| funasr | ✔ | `core_beam_decode` replay-from-prefix (`206e6e2a`) |
| voxtral4b | ✔ | `core_beam_decode` replay + audio adapter injection (`f4d9b803`) |

### Done — shipped 2026-06-02

**gemma4-e2b** — DONE (`f5b28564`). Replay-from-prefix beam via
`core_beam_decode`. `gemma4_e2b_set_beam_size()` API. No local model
to benchmark (auto-download is ~3.3 GB); validated compilation.

**canary** — DONE (`f5b28564`). Runtime beam existed from §90;
adapter wiring added (`CAP_BEAM_SEARCH` + `canary_set_beam_size()`
call). Fixed pre-existing bug: beam path skipped `spiece_to_text()`.
Benchmarked: beam=4 +84 % user time, identical text.

**cohere** — DONE (`f5b28564`). Same as canary — runtime existed,
adapter wiring added. Benchmarked: beam=4 +30 % user time, identical
text. OOMs on longer audio at beam=4 with the F16 model (KV snapshot
size).

**m2m100** — DONE (`84d86a99`). Replay-from-prefix beam for text
translation. Benchmarked: beam=4 is 3.4-6.4× user time; identical
output on clean inputs.

**madlad/t5** — DONE (`84d86a99`). Same pattern as m2m100.

**moonshine-streaming** — DONE (`61136713`). Extracted per-step
decode into a lambda, wired `run_with_probs_branched` with CPU-side
KV snapshot/restore (self-attention only). Benchmarked: beam=4
+56 % user time on tiny Q4_K, identical text.

**funasr** — DONE (`206e6e2a`). The "monolithic API" assessment was
wrong — `funasr_embed_tokens` + `funasr_run_llm_step` were already
factored out. Standard `core_beam_decode::run_with_probs` replay.

**voxtral4b** — DONE (`f4d9b803`). Replay lambda injects audio
adapter frames at the correct offsets during suffix replay, matching
the streaming pre_hook's behavior. No local model to benchmark.

### Done — shipped 2026-06-15 (§167 batch)

**nemotron** — DONE (`641c701e`). RNNT beam search ported from
parakeet's `parakeet_rnnt_beam_decode`. Also MAES (`ec6507d2`).
Benchmarked: beam=4 2.16× decode time, identical text. MAES removes
spurious `<en-US>` tags and improves punctuation.

**moss-audio** — DONE (`5c437124`). Full `core_beam_decode::run_with_probs`
replay wired. 4B model needs Kaggle for A/B testing.

**granite-nle** — DONE (`fe94e976`). CTC beam via
`core_ctc::prefix_beam_search` with gamma in the BPE-CTC decode step.
Compile-verified; no local model.

**mimo-asr** — API STUB (`5c437124`). `set_beam_size` setter + field;
actual beam decode blocked on PLAN #115 runtime stability. 9-stream
architecture needs KV snapshot/restore.

**lfm2-audio** — API STUB (`9511dd5b`). Hybrid Mamba+attention needs
both KV and conv state save/restore. Deferred to Kaggle session.

### Not applicable

| Backend | Reason |
|---|---|
| wav2vec2, hubert, data2vec | CTC beam via `core_ctc` (already has gamma) |
| fastconformer-ctc | CTC-only |
| sensevoice | CTC beam via `core_ctc` (already wired: `sensevoice_set_beam_size`) |
| paraformer | Non-autoregressive (CIF-based) |

### Priority (remaining)

1. ~~**gemma4-e2b**~~ — **DONE**
2. ~~**canary**~~ — **DONE**
3. ~~**cohere**~~ — **DONE**
4. ~~**m2m100**~~ — **DONE**
5. ~~**madlad/t5**~~ — **DONE**
6. ~~**moonshine-streaming**~~ — **DONE**
7. ~~**funasr**~~ — **DONE** (was easier than expected)
8. ~~**voxtral4b**~~ — **DONE** (adapter injection in replay lambda)
9. ~~**nemotron**~~ — **DONE** (RNNT beam + MAES, §167a/b)
10. ~~**moss-audio**~~ — **DONE** (replay, §167g)
11. ~~**granite-nle**~~ — **DONE** (CTC beam + gamma, §167e)
12. **mimo-asr** — API stub, blocked on PLAN #115
13. **lfm2-audio** — API stub, needs KV+conv save/restore

**Score: 21 of 24 ASR backends now support beam search** (was 18 at
§139 close). mimo-asr and lfm2-audio have API stubs, blocked on
runtime issues. paraformer is NAR (not applicable).

## §140 GPU / ggml_backend_sched for CPU-only TTS backends — MOSTLY DONE

**Status (2026-06-08):** 6/6 backends now have `ggml_backend_sched` wired.
speecht5/piper/parler-tts/outetts were already migrated (discovered during
audit). pocket-tts migrated this session: `core_gguf::load_weights` (mmap) +
`ggml_backend_sched` + `ggml_backend_init_best`. All 6 have `use_gpu`
passed via `g_open_use_gpu_tls` from the C API.

| Backend | GPU status | Notes |
|---|---|---|
| **fastpitch** | **DONE** | sched + init_best + core_gguf::load_weights. CUDA fix §204 (`69dc1789`): decoder+vocoder graphs must `sched_graph_compute`, not `backend_cpu` compute (see step 6) |
| **speecht5** | **DONE** | sched + init_best (use_gpu default=true). Shared `core_hifigan` CUDA fix §204 (`69dc1789`) |
| **piper** | **DONE** | sched + init_best (use_gpu default=false in params, overridden by C API) |
| **parler-tts** | **DONE** | sched + init_best (use_gpu default=false in params, overridden by C API) |
| **outetts** | **DONE** | sched + init_best (use_gpu default=false in params, overridden by C API) |
| **pocket-tts** | **DONE** | migrated 2026-06-08: core_gguf::load_weights + sched + init_best |

### Pattern (from FastPitch)

1. Add `#include "core/gguf_loader.h"` + `#include "ggml-alloc.h"`
2. Replace raw `gguf_init_from_file` weight loading with `core_gguf::load_weights(path, backend, ...)`
3. Add `ggml_backend_t backend` + `ggml_backend_sched_t sched` to context
4. Init: `backend = params.use_gpu ? ggml_backend_init_best() : backend_cpu`
5. Create sched: `ggml_backend_sched_new(backends, nullptr, n_be, graph_size, false, false)`
6. Each sub-graph: `sched_reset` → `sched_alloc_graph` → set inputs → `sched_graph_compute`
   **Gotcha (§204):** once a graph is `sched_alloc_graph`'d it MUST be run with
   `ggml_backend_sched_graph_compute(sched, gf)`, never
   `ggml_backend_graph_compute(backend_cpu, gf)`. The latter dereferences
   GPU-resident weights on the CPU backend — a no-op on unified-memory Metal but
   an illegal access on CUDA. fastpitch shipped with this mistake on its
   decoder+vocoder graphs (the encoder/pitch graphs were correct).
7. Free: `sched_free` → `buffer_free` → `backend_free` (GPU before CPU)
8. Wire `p.use_gpu = g_open_use_gpu_tls` in crispasr_c_api.cpp open dispatch

---

## 155. CONV_TRANSPOSE_1D GPU optimization (issue #155)

**Status (2026-06-10):** Core decomposition landed in `5f600f25` (PR #160
by @Rafa00127, cleaned up). New `GGML_OP_COL2IM_1D` op with CPU (F32) and
CUDA (F32/F16/BF16) kernels. Qwen3-TTS codec: **1200 ms → 130 ms** (9×).

**Approach:** Decompose `conv_transpose_1d(w, x, stride)` into:
1. Pre-permute `w[K, OC, IC]` → `w_perm[IC, K*OC]` at load time
2. `col = ggml_mul_mat(w_perm, x)` — highly-optimized GEMM
3. `y = ggml_col2im_1d(col, stride, OC, p0)` — lightweight gather
4. Crop + transpose → channels-first output

The old `ggml_conv_transpose_1d` stays as fallback when `w_perm == NULL`.

### Phase 1: Generalize conv.h helpers — IN PROGRESS

Add `convt1d_decomp()` to `src/core/conv.h` — general-purpose version of
`convt1d_causal_decomp()` that supports symmetric cropping (crop_left =
crop_right, used by most decoders) not just causal right-trim.

Add `permute_convt1d_weight()` utility — de-duplicates the inline permutation
lambda from qwen3_tts.cpp for reuse across all backends.

### Phase 2: Wire into shared decoder headers

**2a. `src/core/hifigan.h`** — HiFi-GAN vocoder (SpeechT5, FastPitch,
      Kokoro, MeloTTS, OpenVoice2, Piper-TTS)
      Symmetric crop: `pad = (K − stride) / 2`.
      Add `up_w_perm` field, branch in `conv_transpose_1d()`.

**2b. `src/core/seanet_decoder.h`** — SEANet family (SNAC/Orpheus, future
      CSM/Bark/Mimi).
      Symmetric crop: `crop_left = crop_right = stride / 2`.
      Add `up_w_perm` to `BlockSlots`, branch in `build_decoder_block()`.

**2c. `src/core/dac_decoder.h`** — DAC decoder (Zonos, Parler, Dia).
      Symmetric crop: `pad = stride / 2`.
      Add `up_w_perm` to `DacDecoderBlock`, branch in `convt1d()`.

### Phase 3: Wire into standalone runtimes

**3a. `src/kokoro.cpp`** — HIGH PRIORITY. Has CPU-pinning workaround for
      Metal `conv_transpose_1d` hang. Decomposition eliminates the hack.
**3b. `src/indextts_voc.cpp`** — BigVGAN v2 upsample blocks.
**3c. `src/chatterbox_s3gen.cpp`** — S3Gen vocoder.
**3d. `src/audioseal.cpp`** — decoder + detector.
**3e. Remaining** — csm_tts, vibevoice, voxcpm2_tts, tada_codec,
      pocket_tts, kugelaudio (single ConvTranspose1d each, lower priority).

### Phase 4: Metal kernel for `GGML_OP_COL2IM_1D`

Port the CUDA gather kernel to Metal. Simple kernel (1 thread per output
element). Files: `ggml-metal.metal` (shader), `ggml-metal-impl.h` (kargs),
`ggml-metal-ops.cpp` (dispatch), `ggml-metal-device.cpp` (pipeline lookup),
`ggml-metal-device.m` (supports_op), `ggml-metal-ops.h` / `ggml-metal-device.h`
(declarations).

### Phase 5: Remove Kokoro Metal workaround

Once Phase 4 lands, remove the CPU-pinning hack in kokoro.cpp (the
`conv_transpose_1d` pin loop ~line 2330).

### Implementation order

Phase 1 → 2a–2c → 3a → 4 → 5 → 3b–3e (incremental).

**Applies to:** qwen3-tts codec (done), orpheus SNAC, outetts WavTokenizer,
pocket-tts Mimi, Zonos DAC, Parler DAC, Dia DAC, Kokoro HiFi-GAN,
SpeechT5 HiFi-GAN, FastPitch HiFi-GAN, MeloTTS, OpenVoice2, Piper-TTS,
IndexTTS BigVGAN, Chatterbox S3Gen, AudioSeal, CSM Mimi, VibéVoice,
VoxCPM2, TADA codec, KugelAudio — every TTS backend with strided
upsampling conv.

---

## §TTS-PROV: TTS AI-provenance compliance (watermark, C2PA, consent, disclaimer)

**Status:** Phases 1-5 merged to main. AudioSeal needs diff-harness validation.

**Motivation:** EU AI Act Article 50 requires machine-readable marking of
AI-generated content. Voice cloning adds deepfake-specific duties: deployer
disclosure and consent. This plan covers all five layers.

### Phase 1: Watermark + file metadata ✅ MERGED (c4da639c)

- Spread-spectrum frequency-domain watermark (`crispasr_watermark.h`)
- WAV LIST/INFO chunk (ISFT="CrispASR (AI-generated audio)", ICMT)
- MP3 ID3v2 TXXX frames (AI_GENERATED=true, GENERATOR=CrispASR)
- C API: `crispasr_watermark_embed()`, `crispasr_watermark_detect()`
- Wired into server (`/v1/audio/speech` streaming + non-streaming) and CLI
- Tests: `test_watermark.cpp` (7 cases), `test_server_wav_writer.cpp` (19 cases)

### Phase 2: Consent gate for voice cloning ✅ MERGED (b47c6214)

- CLI: `--i-have-rights` flag required for `--voice <file.wav>`; refuses
  without it with clear explanation of what it attests
- Server: `consent_attestation` field required in `/v1/audio/speech` JSON
  when voice ends in `.wav`; returns 400 with guidance if missing
- Both paths log `[CONSENT]` with ISO 8601 timestamp, voice path, attestation text
- Files: `whisper_params.h` (new fields), `cli.cpp`, `crispasr_run.cpp`, `crispasr_server.cpp`

### Phase 3: Spoken disclaimer for voice-cloned output ✅ MERGED (b47c6214)

- `crispasr_tts_disclaimer.h`: synthesizes "This audio was generated by
  artificial intelligence" using the loaded TTS backend with neutral/default
  voice (not the clone). Cached after first call (thread-safe, `std::call_once`).
- Prepended with 300ms silence gap to every voice-cloned output
- Wired into CLI + server (streaming + non-streaming) paths

### Phase 4: C2PA signed provenance metadata ✅ MERGED (b47c6214)

- `cmake/Findc2pa.cmake`: find module for c2pa-c library
- `crispasr_c2pa.h`: wrapper header with C2PA manifest JSON
  (digitalSourceType=trainedAlgorithmicMedia), signing via c2pa-c
- `scripts/generate-c2pa-cert.sh`: self-signed P-256 cert (10-year validity)
- CLI flags: `--c2pa-cert`, `--c2pa-key`
- Compile-time gated on `CRISPASR_HAVE_C2PA`; no-op + startup warning when absent

### Phase 5: AudioSeal ggml port ✅ COMPLETE — 100% cosine parity

**What:** Meta AudioSeal (MIT) neural watermark — more robust than spread-spectrum
against adversarial removal, lossy compression, time-stretching.

**Parity metrics (16000 samples, F32 weights, verified 2026-06-06):**
- Full output cosine:    **1.000000**
- Watermark-only cosine: **1.000000**
- Max absolute error:    0.000302
- Watermark RMS ratio:   0.999

**Shipped:**
- `src/audioseal.h` / `src/audioseal.cpp`: ggml implementation of SEANet
  encoder-decoder. Generator embeds watermark; detector returns per-frame
  probability + decoded 16-bit message. C API: `audioseal_embed()`,
  `audioseal_detect()`, `audioseal_embed_stage()`, `audioseal_init_from_file()`.
- `models/convert-audioseal-to-gguf.py`: loads via `audioseal` package,
  remaps state_dict keys, writes 113 tensors. F32 default (89 MB), `--f16`
  for smaller (44.6 MB). Verified against live `facebook/audioseal`.
- `tools/reference_backends/audioseal_ref.py`: reference dump script.
- `tests/test_audioseal.cpp`: 10 unit + 3 live tests (53 total across suites).
- `tests/test_audioseal_cosine.cpp`: standalone cosine comparison binary.
- `--watermark-model` CLI flag to load AudioSeal GGUF as upgrade path.
- Debug: `AUDIOSEAL_DEBUG=1` for shape traces, `AUDIOSEAL_DUMP_STAGES=1`
  for per-stage binary dumps to `/tmp/`.

**Key implementation details:**
- LSTM: proper recurrent gate computation with time-step unrolling
  (~12 ops/step × 50 steps × 2 layers). Zero initial state via
  `ggml_scale(x[:,0], 0)`. Outputs concatenated via `ggml_concat`.
- Padding: all encoder downsampling uses `ggml_pad_ext` for external
  padding (PyTorch F.pad semantics). NOTE: `ggml_pad_ext` has reversed
  parameter convention — `lp0` = RIGHT padding, `rp0` = LEFT padding.
- Decoder: ConvTranspose1d with manual output crop matching PyTorch's
  padding removal. Crop offset also uses the reversed convention.
- Message: `nn.Embedding(32, 128)` via `ggml_get_rows` + `ggml_sum_rows`,
  broadcast via `ggml_repeat`.
- Graph size: 8192 nodes (default 2048 insufficient for full model).

**Architecture (verified against live model 2026-06-05):**

```
Generator (14.7M params, 44.6 MB GGUF):
  Encoder:  Conv1d(1,32,7) → (ResBlock+ELU+Down)×4 → LSTM(512,2) → ELU → Conv1d(512,128,7)
  Message:  Embedding(32,128) indexed by 2*bit_pos+bit_value, summed
  Decoder:  Conv1d(128,512,7) → LSTM(512,2) → (ELU+Up+ResBlock)×4 → ELU → Conv1d(32,1,7) → tanh
  Output:   input + decoder_output

Detector (8.6M params):
  Encoder:  same as generator encoder
  Reverse:  ConvTranspose1d(128,32,k=320,s=320) — back to input resolution
  Head:     Conv1d(32,18,k=1) → channels 0-1=detection, 2-17=message bits

ResBlock: ELU → Conv1d(C,C/2,k=3) → ELU → Conv1d(C/2,C,k=1) + identity skip
Ratios: [2,4,5,8] → hop_length=320 (20ms at 16kHz)
Sample rate: 16 kHz
```


## 156. Permissive G2P phonemizer

espeak-ng is GPLv3 — statically linking it makes the binary GPLv3.
This plan replaces the espeak dependency with modular, permissively-
licensed phonemization backends.

### Architecture (shipped)

```
phonemize(lang, text) → IPA string
  │
  ├─► Pre-generated IPA dict (piper-compatible, auto-download from HF)
  │     EN: 126K words (3 MB)   — 99.5% piper match
  │     DE: 667K words (23 MB)  — 100% piper match
  │     FR: 257K words (6.6 MB)
  │     ES: 600K words (18 MB)
  │
  ├─► phonemize_builtin_en(...)     CMUdict + ARPAbet→IPA (76% match)
  │     ├─ CMUdict lookup (126K words, BSD) — auto-download
  │     ├─ Neural G2P (GRU seq2seq, 29→74, ~4 KB) — from GGUF or file
  │     └─ LTS rules (digraph/trigraph)
  │
  ├─► phonemize_builtin_{de,fr,es}  OLaPh MIT dicts + LTS rules
  │     ├─ OLaPh IPA dicts — auto-download from HF
  │     └─ Per-language LTS rules (always available, zero deps)
  │
  ├─► phonemize_espeak_dlopen(...)  GPL loaded at runtime, MIT binary
  │
  └─► phonemize_espeak_popen(...)   subprocess fallback
```

**Dict sources** (hosted at `cstr/g2p-dicts` on HuggingFace):
- Pre-generated IPA dicts: phonetic data generated by running espeak-ng
  on vocabulary from CMUdict/OLaPh/open-dict-data. The IPA output is
  factual linguistic data, not GPL-encumbered (same principle as compiler
  output not inheriting the compiler's license).
- OLaPh dicts (MIT): from iisys-hof/olaph, 13 languages.
- CMUdict (BSD): from cmusphinx/cmudict, English ARPAbet.

Each language module is a header-only file (`core/g2p_XX.h`) with:
- `struct context` (dict + optional neural model)
- `word_to_ipa(ctx, word)` → IPA string
- `text_to_ipa(ctx, text)` → IPA string
- `load_*_file(dict, path)` → dict loader

### Phase 1+2 — DONE (2026-06-07)

- `core/g2p_en.h` — English: ARPAbet→IPA table (39 phonemes),
  CMUdict loader, LTS rules, GRU cell + neural G2P struct + base64
  JSON weight loader (MeloTTS format), base64 decoder
- `core/g2p_de.h` — German: LTS rules (sch/ch/ei/eu/au/sp/st/z/w/ä/ö/ü/ß),
  IPA dict loader (open-dict-data format), auto-download
- `core/g2p_fr.h` — French: LTS rules (ch/gn/nasal vowels ɑ̃/ɔ̃/ɛ̃/œ̃,
  oi→wa, eau→o, -tion→sjɔ̃, silent finals, mute -e/-es/-ent,
  intervocalic s→z, accented vowels, u→y), IPA dict loader
- `core/g2p_es.h` — Spanish: LTS rules (seseo c/z→s, b/d/g allophonic
  lenition β/ð/ɣ, yeísmo ll→ʝ, ch→tʃ, rr→trill, ñ→ɲ, jota g/j→x,
  initial r→trill vs intervocalic r→ɾ tap, silent h, qu→k)
- `espeak_dlopen.h` — cross-platform dlopen loader (3 function pointers)
- `phonemizer.h/cpp` — cascade interface with auto-loading + inventory filter
- Wired into `piper_tts.cpp` (built-in G2P as final fallback)
- Phoneme inventory validated: all output chars verified against piper's
  154-char `phoneme_id_map`. Fixed combining tie U+0361 (not in map).
  Added `filter_to_inventory()` for runtime validation.
- Vowel quality fixes: AH0→ə (was ʌ), IY0→i (was iː), ER→ɜː (was ɜːɹ),
  dropped secondary stress ˌ. Matches espeak-ng output exactly.
- Dict auto-download: CMUdict (BSD) + OLaPh dicts (MIT, 13 langs) from
  HuggingFace `cstr/g2p-dicts`. Selectable via `--g2p-dict` CLI flag
  or `CRISPASR_G2P_DICT_SOURCE` env var. open-dict-data (CC-BY-SA) as alt.
- Full wiring: CLI `--g2p-dict` → `whisper_params` → C API
  `crispasr_session_set_g2p_dict()` → Go `SetG2PDict()` → server startup.
- Tests: 202 unit assertions (85 EN + 44 DE + 22 FR + 21 ES + 30 piper)
  across 38 test cases + 4 live TTS→ASR roundtrips.

### Phase 3 — open

**a. More languages (piper-plus port)**
- piper-plus (ayutaz/piper-plus, MIT, branch: dev) has rule-based G2P
  for 8 languages. Their French (1197 lines) and Spanish (620 lines)
  are more thorough than ours (handle NFD normalization, PUA codepoint
  mapping, syllabification, stress assignment). Consider porting their
  implementations for higher accuracy.
- No German in piper-plus yet — our `g2p_de.h` fills that gap.

| Language | LTS rules | Dictionary (OLaPh MIT) | Status |
|----------|-----------|------------------------|--------|
| English | CMUdict + neural G2P + LTS | 13 MB (en-us) | **DONE** |
| German | Auslautverh. + open-syl + compound | 40 MB (1.12M) | **DONE** |
| French | nasals/oi/eau/silent finals | 6 MB | **DONE** |
| Spanish | seseo/lenition/yeísmo | 16 MB | **DONE** |
| Portuguese | — | OLaPh TBD | need LTS |
| Italian | — | 1.6 MB OLaPh | need LTS |
| Dutch | — | 5.6 MB OLaPh | need LTS |
| Swedish | — | 737 KB OLaPh | need LTS |
| Czech | — | 2.6 MB OLaPh | need LTS |
| Danish | — | 206 KB OLaPh | need LTS |
| Finnish | — | 2.9 MB OLaPh | need LTS |
| Polish | — | 1.5 MB OLaPh | need LTS |
| Japanese | — | piper-plus | need rules |
| Chinese | — | piper-plus | need rules |
| Korean | — | piper-plus | need rules |
| Other | — | espeak-ng dlopen/popen | available |

**OLaPh** (iisys-hof/olaph, MIT): multilingual phonemization framework
with dictionaries for 13 languages. German alone has 1.12M entries.
All dicts use the same `word\t/IPA/` format our loaders already support.
Paper: arXiv 2509.20086v3. Auto-downloadable via `crispasr_cache`.

**b. GGUF-embedded dicts (TTS.cpp pattern)**
- TTS.cpp embeds phonemizer rules in the GGUF itself via
  `phonemizer.rules.keys` / `phonemizer.rules.phonemes` arrays
- Could embed CMUdict / neural G2P weights per-model for zero
  external dependencies at runtime

**c. Gruut CRF (MIT) for German/multilingual OOV**
- rhasspy/gruut: dictionary + CRF G2P model (18 MB, SQLite + CRFsuite)
- Higher quality than LTS rules for unknown words (compounds, loanwords)
- CRFsuite is BSD — native C/C++ dependency
- Gruut supports: de, en, fr, es, it, nl, pt, ru, sv, cs, ar, fa, sw
- Port path: extract SQLite lexicon + CRFsuite model, write C++ feature
  extractor (~100 lines), link against libcrfsuite (BSD)
- Lower priority now that OLaPh MIT dicts (1.12M DE entries) cover most
  words; gruut adds value mainly for truly novel OOV words

**d. Neural G2P weight distribution**
- MeloTTS v3 GGUF has `melotts.g2p_en_json` (base64, ~4 KB weights)
- Could publish standalone `g2p_en.json` for download
- Weight loader already implemented in `g2p_en.h`

### Files

```
src/core/g2p_en.h          — English G2P (ARPAbet→IPA, CMUdict, LTS, neural)
src/core/g2p_de.h          — German G2P (IPA dict, LTS rules)
src/core/g2p_fr.h          — French G2P (LTS rules, nasal vowels, IPA dict)
src/core/g2p_es.h          — Spanish G2P (seseo, lenition, yeísmo, IPA dict)
src/espeak_dlopen.h        — cross-platform dlopen for libespeak-ng
src/phonemizer.h           — cascade interface + filter_to_inventory()
src/phonemizer.cpp         — implementations + auto-loading + auto-download
tests/test-g2p-en.cpp      — 85 assertions (ARPAbet, LTS, CMUdict, neural, inventory)
tests/test-g2p-de.cpp      — 30 assertions (digraphs, vowels, consonants, dict)
tests/test-g2p-fr.cpp      — 22 assertions (digraphs, nasals, vowels, silent finals)
tests/test-g2p-es.cpp      — 21 assertions (seseo, lenition, yeísmo, jota)
tests/test-espeak-phonemize.cpp — 30 assertions (piper synthesis)
tests/test-piper-roundtrip.sh   — 4 live TTS→ASR tests
```

---

## 157. voxcpm2-tts — Vulkan / CUDA GPU acceleration (scheduler-based)

### Context

VoxCPM2 graph-ified its AR-loop hot paths (TSLM step, LocEnc, LocDiT
CFM, VAE decode) in §96 for Metal. On Apple Silicon (unified memory)
both the graph paths (`ggml_backend_graph_compute`) and the legacy CPU
paths (`tensor_data_f32`, `matmul_mv_ggml`, `rms_norm_cpu`) access the
same shared-storage buffer, so everything works.

On discrete GPUs (Vulkan, CUDA) the default buffer is device-local VRAM
that is not host-visible. The legacy CPU paths would SIGSEGV
dereferencing GPU pointers. As of #158 fix (2026-06-07, `c6299251`),
voxcpm2 falls back to CPU when the backend buffer is not host-visible —
correct but loses all GPU acceleration.

### Why a scheduler

The Vulkan backend's `supports_buft` only accepts its own device-local
buffer type; it rejects host-pinned buffers. So there is no single
buffer type that satisfies both CPU-pointer legacy paths and GPU graph
compute. The solution is `ggml_backend_sched`:

1. Load weights to **CPU buffer** (legacy paths can dereference them).
2. Create `ggml_backend_sched` with `[gpu_backend, cpu_backend]`.
3. Graph paths call `ggml_backend_sched_graph_compute()` instead of raw
   `ggml_backend_graph_compute()`. The scheduler copies weight data to
   GPU device-local buffers as needed and allocates intermediates on GPU.
4. Legacy paths (TSLM/RALM prefill, FSQ, stop_score) stay on CPU with
   CPU-accessible weights.

This mirrors the qwen3_tts architecture (`ggml_backend_sched` +
`ggml_backend_sched_graph_compute`).

### Prerequisites / risks

- **Upstream scheduler bugs**: our tools/upstream-prs #10 documents
  dangling `src[j]` pointers across `sched_split_graph` calls (affects
  graph reuse, which the cached LocDiT/LocEnc/TSLM graphs trigger); #16
  documents cross-backend copy insertion failure for small mixed-backend
  graphs. Both need fixes applied or merged upstream before the scheduler
  is safe here.
- **galloc→sched migration**: the existing per-graph `ggml_gallocr`
  pre-reservation (`ggml_gallocr_reserve` + `ggml_gallocr_alloc_graph`)
  must be replaced with `ggml_backend_sched_reserve` +
  `ggml_backend_sched_graph_compute`. Touches every graph path.
- **No hardware to test**: need Vulkan or CUDA device to validate.

### Scope

1. `voxcpm2_init_from_file`: when `!ggml_backend_buft_is_host(buft)`,
   load weights to CPU, keep GPU backend, create `ggml_backend_sched`.
2. Replace `ctx->galloc` with `ctx->sched` (`ggml_backend_sched`).
3. For each graph entry point (`locdit_forward_graph`,
   `locenc_forward_graph`, `tslm_step_graph`, `vae_decode_graph`):
   replace `ggml_gallocr_alloc_graph` + `ggml_backend_graph_compute`
   with `ggml_backend_sched_graph_compute`.
4. Validate: diff harness `voxcpm2-q4_k.gguf` zero-shot + voice-clone.
5. Stretch: graph-ify TSLM/RALM prefill (loop `tslm_step_graph` N_pos
   times) to run prefill on GPU too.

### Status

DONE (2026-06-07, `df6cf31e`) via GPU weight mirrors — weights on CPU
for legacy paths + GPU mirror copies for graph-build functions. No
`ggml_backend_sched` needed (avoids per-call PCIe copy overhead).
Graph paths run entirely on GPU; legacy paths stay on CPU.
Memory overhead: ~2× model size on discrete GPUs.

## §163 LFM2-Audio — ASR + TTS + S2S (ALL PHASES DONE)

**Status**: Complete. 27 commits `470d56f2`–`cd77c75e`. ASR + TTS + S2S
all working, fully wired per `docs/contributing.md`, Kaggle GPU-tested.

### Phase 1 — ASR — DONE

_Done — see HISTORY.md + git log._

## §164 Mini-Omni2 — ASR + TTS + S2S (DONE)

_Done — see HISTORY.md + git log._

## §165 — beam_size default greedy + issue #161 (DONE)

_Done — see HISTORY.md + git log._

## §166 — Vulkan conv_transpose_1d_f16 + voxcpm2 graph bugs (issue #164)

### Part 1 — conv_transpose_1d_f16 shader — DONE `570bb76d`

The f16 variant passed `A_TYPE=float16_t` without enabling the required
`GL_EXT_shader_explicit_arithmetic_types_float16` extension. Strict glslang
(Vulkan SDK 1.4.350+) rejected the shader → LNK2019 on ggml-vulkan.dll.

Fix: enable the extension + cast kernel element to float in `fma()`.

### Part 2 — VOXCPM2_USE_GRAPH NaN + SIGABRT — DONE

_Done — see HISTORY.md + git log._

## §167 — Beam search, MAES, and KV caching expansion

Cross-cutting feature push to bring beam search, MAES (Modified Adaptive
Expansion Search), and KV caching to backends that lack them. Builds on
§139 (beam search round 1, all LLM backends done) and §165 (beam_size
defaults).

Branch: `feat/beam-maes-cache`

### Survey results (2026-06-15)

**Beam search gaps:**

| Backend | Architecture | Gap | Effort |
|---|---|---|---|
| nemotron | RNNT (FastConformer enc + LSTM pred) | Stub only — `set_beam_size` + field exist but `nemotron_rnnt_decode` is greedy | LOW — copy `parakeet_rnnt_beam_decode` pattern |
| sensevoice | CTC (encoder-only) | Greedy CTC; `core_ctc::prefix_beam_search` available | LOW — wire existing infra |
| granite-nle | CTC (encoder-only) | Greedy CTC; same `core_ctc` | LOW |
| mimo-asr | LLM (Qwen2 + RVQ) | No beam wired; `kv_self_attn` present | MEDIUM — `core_beam_decode` replay |
| moss-audio | LLM (InternLM2 + audio) | No beam wired; `kv_self_attn` present | MEDIUM — `core_beam_decode` replay |
| lfm2-audio | Hybrid Mamba+attention | No beam; needs KV + conv state save/restore | HIGH |

**MAES gaps:**

| Backend | Current | Gap |
|---|---|---|
| nemotron | greedy RNNT | No MAES; direct port from `parakeet_rnnt_maes_decode` |
| sensevoice | greedy CTC | `core_ctc::prefix_beam_search` gamma parameter unused |
| granite-nle | greedy CTC | Same |
| wav2vec2 | Has gamma | Already done (reference) |
| parakeet | Has full MAES | Already done (reference) |

**KV caching gaps:**

| Backend | Current | Gap |
|---|---|---|
| firered-asr | Hand-rolled per-step AED, no persistent KV | Migrate to `core_attn::kv_self_attn` |

### Phase 1 — VPS (small models, ≤1.2 GB)

**§167a: Nemotron RNNT beam search**
- Model: `nemotron-3.5-asr-streaming-0.6b-q4_k.gguf` (458 MB)
- Implement `nemotron_rnnt_beam_decode` from `parakeet_rnnt_beam_decode`
- Wire `ctx->decode_beam_size` in transcribe path
- A/B: greedy vs beam_size=4 on jfk.wav — WER, wall time, RSS

**§167b: Nemotron RNNT MAES**
- Depends on §167a
- Implement `nemotron_rnnt_maes_decode` from `parakeet_rnnt_maes_decode`
- Add `nemotron_set_maes()` + C API surface
- A/B: beam=4 vs MAES(beam=4, steps=2, gamma=2.3)

**§167c: Firered AED decoder KV cache**
- Model: `firered-asr2-aed-q4_k.gguf` (919 MB)
- Migrate decoder attention to `core_attn::kv_self_attn`
- Regression gate: identical text output
- A/B: current vs KV-cached — wall time improvement

### Phase 2 — VPS or Kaggle (medium models, need model files)

**§167d: Sensevoice CTC beam + gamma**
- Model: needs GGUF on disk (not currently available)
- Wire `sensevoice_set_beam_size` → `core_ctc::prefix_beam_search(gamma)`
- A/B: greedy vs beam=8 + gamma=2.3

**§167e: Granite-NLE CTC beam + gamma**
- Model: needs GGUF on disk
- Wire beam into `granite_nle` CTC decode
- A/B: greedy vs beam=8 + gamma=2.3

### Phase 3 — Kaggle (large models, GPU needed)

**§167f: MIMO-ASR beam** — `core_beam_decode::run_with_probs` replay
**§167g: Moss-Audio beam** — `core_beam_decode::run_with_probs` replay (4B, GPU)
**§167h: LFM2-Audio beam** — `core_beam_decode::run_with_probs_branched` + conv state

### Benchmark results (2026-06-15)

**§167a Nemotron beam** on jfk.wav (Q4_K, CPU, 4 threads, median of 3):

| | Greedy (beam=1) | Beam (beam=4) |
|---|---|---|
| Decode time | 8.8s | 19.0s (2.16×) |
| Text | `...ask not...for you. <en-US> Ask what...` | identical |
| RSS | 679 MB | 679 MB |

**§167b Nemotron MAES** on jfk.wav:

| | Beam=4 | MAES(4,2,2.3) |
|---|---|---|
| Text | `...for you. <en-US> Ask what...country. <en-US>` | `...for you, ask what...country.` |
| Quality | Spurious `<en-US>` tags, missing commas | Proper punctuation, no tags |
| Overhead | baseline | +15% decode time |

**§167c Firered KV cache**: Already implemented (sa_k/sa_v per beam).
**§167d Sensevoice CTC beam**: Already implemented (core_ctc).
**§167e–h**: Compile-verified; no local models for A/B testing.

## §169 — Qwen3-ASR ChatML language prompt (non-English script output)

**Status:** OPEN

**Problem:** Qwen3-ASR supports 30 languages including Arabic, but our
implementation (`src/qwen3_asr.cpp`) skips the ChatML prompt and builds
the token sequence as bare `<|audio_start|>...<|audio_end|>` without a
system/user message. The original HF model uses:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Transcribe the following audio in Arabic.
<|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>
<|im_start|>assistant
```

Without this, the model auto-detects language but may romanize
non-Latin scripts (observed: Arabic AA0010.wav → `istagel` instead of
native Arabic `استغل`; AA001.wav → `مرحبًا` works, so auto-detect is
partial).

**Impact:** All 30 Qwen3-ASR languages work in auto-detect mode for
clean audio, but explicit language selection (`-l ar`) has no effect
and ambiguous cases get romanized.

**Fix:**
1. Build the full ChatML prompt in `qwen3_asr_transcribe_with_probs`
   when `-l LANG` is set (map ISO 639-1 → English name → prompt text)
2. Fall back to bare audio-only prompt when no language specified
   (current behaviour, preserves auto-detect)
3. Wire `--language` / `-l` through the CLI adapter's transcribe call

**Files:** `src/qwen3_asr.cpp` (~50 LOC), `examples/cli/crispasr_backend.cpp`

**Effort:** MEDIUM — the ChatML token IDs (`<|im_start|>` etc.) need
to be resolved from the GGUF vocab. The word-level timestamp alignment
loop (lines 2074-2091) also needs adjustment since the prompt prefix
shifts positions.

**Test:** Arabic audio from `atishay23/Arabic_Audio` (AA001.wav →
should output `مرحبًا` in native script with `-l ar`).

**2026-06-16: GGUFs uploaded to `cstr/parakeet-unified-en-0.6b-GGUF`**
(F16 1181 MB + Q4_K). v8: n_mels=128 fix → test_rc=0 (no crash) but
transcript garbage. Tokenizer verified correct (same as parakeet-rnnt).
Root cause: synthetic config mismatch vs actual training config. Need
to diff C++ mel output against NeMo Python reference to find divergence.

## 158. transcribe_streaming for opaque-C-library backends

**Context:** `transcribe_streaming` now implemented for Qwen3, Voxtral, Granite,
Voxtral4b (all expose `_run_llm_kv` so the adapter owns the loop). Seven
remaining autoregressive ASR backends hide their decode loop inside an opaque
C library. Each needs:
1. A `*_transcribe_cb(ctx, pcm, n, fn, userdata)` C entry point added to the
   library implementation — fires `fn(token_id, text_piece, userdata)` per token.
2. A `transcribe_streaming` override in the adapter (`.cpp` in `examples/cli/`)
   that calls the new entry point and maps it to `on_text`.
3. Fall back to base-class batch path for beam search (best_of-N already
   handled by single-run greedy in the callback path).

### Backends in scope

| Backend | C library | token-text API | Notes |
|---------|-----------|----------------|-------|
| GLM-ASR | `src/glm_asr.cpp` | `glm_asr_token_text()` | BPE decode lambda already in adapter |
| Kyutai-STT | `src/kyutai_stt.cpp` | `kyutai_stt_token_text()` | |
| Gemma4-E2B | `src/gemma4_e2b.cpp` | `gemma4_e2b_token_text()` | supports translate flag |
| MiMo-ASR | `src/mimo_asr.cpp` | `mimo_asr_token_text()` | |
| MOSS-Audio | `src/moss_audio.cpp` | `moss_audio_token_text()` | |
| Moonshine-Streaming | `src/moonshine_streaming.cpp` | `moonshine_streaming_token_text()` | |
| Nemotron (RNN-T) | `src/nemotron.cpp` | `nemotron_token_to_str()` | RNN-T already emits per-frame; callback fires on non-blank frames |

### C library callback signature (uniform across all)

```c
typedef void (*crispasr_token_cb)(int token_id, const char* text, void* userdata);
int glm_asr_transcribe_cb(struct glm_asr_context* ctx,
                           const float* pcm, int n_samples,
                           crispasr_token_cb cb, void* userdata);
// ... same pattern for others
```

### Adapter pattern (same for all)

```cpp
void transcribe_streaming(const float* samples, int n_samples, int64_t,
                           const whisper_params& params,
                           crispasr_stream_callback on_text) override {
    if (params.beam_size > 1) {
        CrispasrBackend::transcribe_streaming(...);  // batch fallback
        return;
    }
    std::string acc;
    glm_asr_transcribe_cb(ctx_, samples, n_samples,
        [](int /*id*/, const char* text, void* ud) {
            auto& [a, fn] = *static_cast<std::pair<std::string, crispasr_stream_callback>*>(ud);
            a += text;
            fn(a, false);
        }, &std::make_pair(acc, on_text));  // actual impl uses local lambda
    on_text(acc, true);
}
```

**Status:** **DONE 2026-06-19** (`7f2c1f3a`). All 7 implemented.

### Implementation details

The actual callback signatures use per-token `(int tok_id, float prob, void* userdata)` —
not `text` in the callback, since text lookup via `*_token_text()` is done in the adapter.

| Backend | C entry point | BPE decode | Notes |
|---------|--------------|-----------|-------|
| GLM-ASR | step APIs (`glm_asr_embed_tokens`, `glm_asr_run_llm_kv`) | GPT-2 (Ġ→space, Ċ→newline) | Full greedy loop in adapter; EOS={59246,59253,59255} |
| MOSS-Audio | `moss_audio_process_cb` | GPT-2 full byte_decoder | `moss_audio_process` refactored to `moss_audio_process_impl` |
| Gemma4-E2B | `gemma4_e2b_transcribe_cb` | SentencePiece ▁→space | Added `gemma4_e2b_is_control_token` to filter bos/eos/sot/eot |
| Moonshine-Streaming | `moonshine_streaming_transcribe_cb` | SentencePiece ▁→space | — |
| Kyutai-STT | `kyutai_stt_transcribe_cb` | SentencePiece ▁→space | `emit_token` lambda already filters padding; callback fires inside it |
| MiMo-ASR | `mimo_asr_transcribe_cb` | GPT-2 full byte_decoder (inlined in adapter) | `core_bpe::token_bytes_to_utf8` not exported → inline table |
| Nemotron | `nemotron_transcribe_cb` | SentencePiece ▁→space | RNN-T: fires per non-blank emitted frame; `nemotron_transcribe_ex` → `nemotron_transcribe_impl` wrapper |

Forward-declaration fix needed for nemotron: `nemotron_transcribe` called `nemotron_transcribe_impl` before
its definition — added a static forward decl.

## 172. Wyoming protocol server — Home Assistant Assist

**DONE 2026-06-19** (`55bc0039`). `--wyoming-port N` starts a Wyoming
peer-to-peer JSONL/TCP server alongside the HTTP API. One `crispasr-server`
instance replaces both `wyoming-faster-whisper` and `wyoming-piper` for HA.

See HISTORY 2026-06-19 and `docs/server.md#wyoming-protocol-home-assistant-assist`.

## 173. `--tts-play` / `--tts-play-device` — local speaker output

**DONE 2026-06-19** (`c6021b43` + `e78ad149`). Plays TTS output through the
local speaker immediately after synthesis, with the watermark already embedded.

Device is opened at hardware-native rate (sampleRate=0); the PCM is pre-resampled
via linear interpolation before the device starts — avoids miniaudio's 4×
upsampler artefacts at 24→96 kHz on Core Audio devices.

See HISTORY 2026-06-19 and `docs/tts.md#local-speaker-output-tts-play`.

## §176 Runtime optimization pass — 2026-06-20 audit

Full code-read survey of every runtime. Detailed findings in
PERFORMANCE.md "Runtime optimization audit — 2026-06-20". This section
tracks the TODO items.

### Tier 1 — High impact, broadly applicable

#### §176a Wire `ggml_flash_attn_ext` in remaining AR decoders

**Status:** MOSTLY DONE — all backends using `core_attn::kv_self_attn` already use `ggml_flash_attn_ext` unconditionally. MOSS encoder DONE (`294bedff`).
**Effort:** Small per backend (flag passthrough)
**Backends:** Orpheus, OuteTTS, Chatterbox (PLAN #86 stub), Parler
(decoder SA), SpeechT5, Dia, Zonos, TADA, Pocket-TTS, CSM (backbone +
depth decoder), MOSS encoder (32 layers), Voxtral/4B encoder, CosyVoice3
**Approach:** `core_attn::kv_self_attn` already supports flash_attn_ext;
most backends just never pass `flash_attn=true`. The flag exists in most
context_params structs. Wire the flag through and default it ON.
**Impact:** 2-5× attention bandwidth reduction at long sequences.

#### §176b Lk-bucketed graph caching for AR decode steps

**Status:** PARTIAL — Chatterbox T3 DONE (§186), Orpheus DONE (§190), OuteTTS DONE, Zonos DONE, TADA DONE. F5-TTS DiT done differently (§183).
**Effort:** Medium (template from qwen3-tts)
**Backends done:** Chatterbox T3 (§186), Orpheus (§190), OuteTTS, Zonos, TADA, CosyVoice3 (step_t1_gf), VibeVoice TTS LM (§201 2026-06-20), Parler (§176b+c 2026-06-21, opt-in).
**Note:** VoxCPM2 TSLM already has Lk-buckets (`get_or_build_tslm_step_graph`). VibeVoice pred head already cached by n_frames (`get_pred_head_graph`). Both can be removed from remaining.
**Remaining:** SpeechT5 (self-attn only; §202 handled cross-attn), Dia, Pocket-TTS,
LFM2 (T=1 decode graph fixed-topology), KugelAudio (LM T=1 + pred head + VAE decoder).
**Handover prompts:** `docs/prompts/176b-dia-tts.md`,
`docs/prompts/176b-pocket-tts.md`, `docs/prompts/176b-lfm2-kugelaudio.md`,
`docs/prompts/176bc-speecht5-self-attn-kv.md`.
**Note on LFM2/KugelAudio:** device-resident KV already present — only graph-cache
overhead remains. For T=1 decode, graph topology is FIXED (no mask); can cache a
single graph, simpler than Lk-bucketing.
**Parler note (§176b+c 2026-06-21, opt-in `CRISPASR_PARLER_BUCKET=1`, default OFF):** Implemented
device-resident self-attn + cross-attn KV (`ggml_set_rows` write, fixed-Lk read) and
7 cached bucket graphs on a dedicated `dec_step_sched`. Numerically equivalent to the
legacy host-KV path but NOT bit-exact — the fixed-Lk padded reduction shifts FP rounding
(step-1 bit-identical, later steps ~1e-4), so greedy decoding produces a different *valid*
generation. Gotchas found: (1) read from the `set_rows` *result*, not the bare KV tensor,
or Metal races the in-place write; (2) rebuild bucket graphs per utterance or repeated
synthesize calls fault on stale tensor→buffer pointers. With those fixed, the sched
allocation is reused across steps (only re-allocated on a bucket switch) — the earlier
"reuse faults on Metal" was a symptom of (1)/(2), not a real limitation.
**Perf (M1 Metal, F16, 600 steps, back-to-back best-of-3):** ~1.2–1.9× faster than legacy
(legacy absolute swings with machine load; bucket is consistently faster). The two wins
that flipped it from an early M1 *regression* to a clear win: (a) dropping the per-step
`ggml_cont` of the Lk KV window (a D×Lk copy that grew with the bucket), (b) reusing the
sched allocation across steps. Still opt-in (`CRISPASR_PARLER_BUCKET=1`) because the
default path must stay byte-identical and CUDA is unvalidated; consider flip-to-default
after a CUDA run.
**crispasr-diff validation:** vs F32 PyTorch ground truth — F16 legacy 108/108 (100%)
PASS, F16 bucket 108/108 (100%) PASS (bucket bit-identical to legacy); Q8_0 bucket
byte-identical to legacy (both 23% vs F32 = pure quant gap). Reference at
`hf.co/cstr/parler-tts-mini-v1.1-GGUF/parler-mini-v1.1-ref.gguf`. Diff capped to 40 decode
steps for parler via `PARLER_DIFF_MAXGEN` (else it runs the full ~2580-step default).
**Approach:** Qwen3-TTS demonstrates with 5 pre-built graphs at fixed Lk
sizes. MIMO has a simpler single-bucket `step_t1_gf`. FunASR has the
infrastructure but disabled due to full-window attend; needs Lk-bucketing
to be efficient (round up to nearest power of two).
**Impact:** 6-14 ms/step graph-build overhead eliminated across 100-1000
steps. Largest single latency win project-wide.

#### §176c Migrate host-side KV to device-resident 4D tensors

**Status:** OPEN
**Effort:** Medium per backend
**Backends:** SpeechT5 (cross-attn KV DONE §202; self-attn KV still host-side), Dia,
Pocket-TTS, VoxCPM2 (all use `std::vector<float>` KV that grows and re-uploads every step).
Parler DONE (§176b+c 2026-06-21, opt-in — see §176b note).
LFM2 and KugelAudio already have device-resident KV (no §176c work needed there).
**Approach:** Follow IndexTTS/CSM pattern: 4D on-device tensor
`[head_dim, max_ctx, n_heads, n_layers]` with `ggml_view_4d` +
`ggml_cpy` writes. Eliminates O(step × layers × hidden) host↔device
bandwidth per step.
**Impact:** Eliminates the dominant data-movement bottleneck for these
backends at long output sequences.

#### §176d BLAS/ggml for scalar CPU matmul hotpaths

**Status:** PARTIAL — TitaNet ASP DONE, Silero LID DONE, FireRed VAD DONE (§193), Parakeet DONE (§194), Nemotron DONE (`325432f6`), MeloTTS weight cache DONE (§195), OpenVoice2 WaveNet DONE (`5e7d8704`), Granite Speech cpu_linear DONE (`eaa6dff2`), Piper weight cache DONE (§196)
**Effort:** Medium-Large (per-backend refactor)
**Targets (ordered by compute dominance):**
- TitaNet ASP TDNN: DONE — cblas_sgemm under HAVE_ACCELERATE (prior)
- Silero LID: DONE — cblas_sgemm (§ e1a0725e 2026-06-20, 4.5×)
- FireRed VAD: DONE — cblas_sgemm for all DFSMN cpu_linear calls (§193 2026-06-20)
- Parakeet LSTM+joint: DONE — cblas_sgemv for lstm_step_layer + joint_proj_enc + joint_step (§194 2026-06-20)
- MeloTTS: `cpu_multihead_attention_relpos` O(H×T²×D) × 6 layers — DONE: QKV cblas_sgemm + per-head Q@K^T + attn@V cblas_sgemm (§198); rel-pos window stays scalar (O(T×2W×D)≪O(T²×D))
- Piper-TTS: same `cpu_multihead_attention_relpos` — DONE: same cblas_sgemm pattern (§199)
- OpenVoice2 WaveNet: 16 layers × T × K=5 × C=192 → ggml_conv_1d
- Granite Speech `cpu_linear`: naive dequant matmul → ggml or OMP
- Nemotron LSTM/joint: same pattern as Parakeet
- F5-TTS text encoder ConvNeXt pw_up/pw_down: DONE — replaced O(T×D×inter_dim) scalar loops with f5_linear (cblas_sgemm) calls (§200 2026-06-20); Vocos blocks were already using f5_linear
**Impact:** These are the dominant compute paths in their respective
runtimes and currently run as unvectorized nested loops.

### Tier 2 — Medium impact, moderate effort

#### §176e Context caching for support runtimes

**Status:** DONE — all support runtimes now cached or per-session.
**Effort:** Small per backend (template: Silero VAD static cache)
**Backends done:** WhisperEncDec VAD, MarbleNet VAD, Pyannote segmentation, FireRed VAD, CTC aligner, ECAPA-TDNN LID (EcapaLidCache). FireRedPunc is per-session (init once, free at session end). RNNoise is per-call but lightweight (no model weights, just DSP state).
**Approach:** The Silero VAD `g_silero_cache_mtx` + static context
pattern prevents 70× init/free regression. Replicate for each backend:
static or per-pipeline cached context, mutex-guarded.
**Impact:** Eliminates repeated init/free overhead in diarization and
post-processing pipelines that call these per-segment.

#### §176f Parallel STFT in mel.cpp + BLAS mel projection

**Status:** DONE — BLAS mel projection (§189 2026-06-20) + STFT OMP gated path
(2026-06-20, default off)
**Effort:** Small
**Files:** `src/core/mel.cpp`, `src/core/kaldi_fbank.cpp`
**Done (§189):** Mel projection replaced with `cblas_sgemm` for both MelsFreqs
and FreqsMels layouts; `crispasr-core` now links Accelerate/MKL/OpenBLAS.
**Done (STFT):** OMP-parallel STFT frame loop, gated (see below).
**Impact:** Entry point for ~40 backends. BLAS gives AMX/SIMD for free on
the mel projection matrix multiply.

**STFT done (gated, default OFF):** OpenMP-parallel STFT frame loop in
`core_mel::compute`, opt-in via **`CRISPASR_MEL_PARALLEL=1`** (needs an OMP
build; `crispasr-core` now links `OpenMP::OpenMP_CXX` when found). Bit-identical
to serial (each frame writes its own `power` row + thread-private scratch;
verified: cohere transcript identical default vs parallel). Kept OFF by default
because the `fft` callable is per-backend (`cohere_fft_r2c`, `glm_fft`,
`voxtral_fft_wrapper`, …) and parallel correctness needs each to be re-entrant;
flip a backend to default-on once its fft is audited. Speed is arch/load
dependent (allocating ffts contend on malloc across threads) — retest per arch
on a quiet machine; **`CRISPASR_MEL_TIMING=1`** prints per-call STFT ms +
thread count for that A/B. (The mel projection half of §176f is already done by
§189's `cblas_sgemm`; this STFT half stacks on top of it.)

**fft re-entrancy AUDIT (2026-06-20) — all clear.** Read every `FftR2C`
callable passed to `core_mel::compute`: `cohere/nemotron/parakeet/canary/
canary_ctc` are byte-identical pure in-place Cooley-Tukey (stack-only locals,
write only the caller's `out`); `glm_asr` is a captureless lambda; `qwen3_asr`
recurses on the caller's buffer; `voxtral` uses `thread_local` scratch. **None
use shared mutable static state → the parallel STFT is correctness-safe for every
backend.** So the only remaining gate to flipping a backend default-on is a
quiet-machine per-arch perf bench. Added `core_mel::Params::allow_parallel_stft`
(default false) as the per-backend opt-in (env var still overrides globally), and
`tests/test-core-mel-parallel.cpp` (unit, `[unit][mel]`) pins parallel ≡ serial
bit-identity on an OpenMP build. No backend flipped yet (perf unproven on the
loaded dev box).

#### §176g CPU embedding cache for AR TTS backends

**Status:** PARTIAL — Chatterbox DONE (§188 2026-06-20), Zonos DONE (§191 2026-06-20), CosyVoice3 DONE (§192 2026-06-20)
**Effort:** Small per backend (template: qwen3-tts `CpuEmbdCache`)
**Backends done:** Chatterbox (§188), Zonos (§191), CosyVoice3 (§192). **Remaining:** Orpheus (vocab 156938 → ~1.7 GB F32, skip), OuteTTS,
TADA, VibeVoice, Pocket-TTS
**Approach:** Copy raw quantized embedding bytes from GPU buffer to CPU
at init; `get_row_into` dequantizes via `ggml_get_type_traits(type)->
to_float`. Eliminates ~17 Metal command-buffer round-trips per AR frame
(measured in qwen3-tts).

#### §176h F5-TTS: collapse 22 mini-graphs + batch CFG

**Status:** PARTIAL — single fused graph DONE (§183 2026-06-20). **Batch CFG (B=2)
ATTEMPTED + REVERTED (§203 2026-06-20) — blocked by an unisolated F5-runtime
corruption; NOT worth pursuing** (the §183 fusion already captured the win, B=2
adds only ~1.2×).
**Effort:** Medium
**File:** `src/f5_tts.cpp`
**Done:** §183 fused all 22 DiT blocks into a single ggml graph (eliminated 1408
alloc+compute round-trips per synthesis). Weight pre-cache (§184/§185) already avoids
all dequantization in the hot path.
**Batch CFG verdict (§203):** the B=2 batched graph is provably correct (input
verified, batch-0 bit-exact vs serial) but batch-1 is corrupted in F5 only. A
standalone reproducer with the **byte-identical 979-node graph** computes batch-1
perfectly across every variation — so it is NOT a ggml allocator/graph bug. Root
cause is F5-runtime-specific (values?), uncracked after ~45 experiments. Do not
retry without first bisecting F5 toward the working repro. See HISTORY §203,
LEARNINGS, and `repro_batch.cpp`.
**Remaining (separate, still open):** Vocos ConvNeXt ggml port (currently CPU
scalar C++, blocks GPU dispatch).

#### §176i Cross-KV in F16 (not F32)

**Status:** DONE (2026-06-21)
**Effort:** Small
**Backends:** M2M-100, T5, SpeechT5, Dia, Parler
**Approach:** Cross-attention K/V is projected once from the encoder and
read-only thereafter. F16 halves memory with no accuracy impact. Changed
allocation dtype to `GGML_TYPE_F16` and added `ggml_fp32_to_fp16_row`
conversion in the write path. All five backends updated:
- `t5_translate.cpp`: alloc + write path
- `m2m100.cpp`: alloc + write path
- `speecht5_tts.cpp`: alloc only (uses `ggml_cpy` which auto-converts)
- `dia_tts.cpp`: alloc + staging vectors + graph inputs + write path + dump
- `parler_tts.cpp`: alloc + buffer size + staging vectors + graph inputs + write path (both bucket and legacy paths)
Build + 688 unit tests pass. FireRed ASR cross-KV unchanged (separate architecture).

#### §176j Replace recursive FFT (core/fft.h) with iterative

**Status:** DONE — already iterative as of prior refactor
**File:** `src/core/fft.h`
`fft_radix2_inplace` is already the iterative in-place Cooley-Tukey with O(1)
extra memory (bit-reversal via swaps). `fft_radix2` and `fft_radix2_wrapper`
use `thread_local` scratch that grows-once. No further change needed.
**Also:** CosyVoice3 HiFT source DFT is O(n²) — must use FFT.

### Tier 3 — Targeted wins

#### §176k FireRed ASR: add KV cache for decoder self-attention

**Status:** OPEN
**Effort:** Medium
**File:** `src/firered_asr.cpp`
**Approach:** Currently grows `std::vector<float>` per beam per layer
and does O(T²) scalar attention. Add pre-allocated 4D KV cache
(`core_attn` pattern) with flash_attn_ext. Highest-impact single-backend
optimization remaining.

#### §176l Kyutai STT: vectorized RVQ encode

**Status:** OPEN
**Effort:** Medium
**File:** `src/kyutai_stt.cpp`
**Approach:** Brute-force O(T×32×2048×256) codebook search is the
dominant cost. Options: SIMD-vectorized exhaustive search, or product
quantization / FAISS-style IVF for approximate nearest-neighbor.

#### §176m Nemotron: streaming KV cache trim

**Status:** DONE — memmove replaces vector::erase (`6e416c85`)
**File:** `src/nemotron.cpp`
**Done:** `memmove` + `resize` instead of `vector::erase(begin, begin+N)`,
eliminating O(N) element-shifting per chunk eviction.

#### §176n VoxCPM2: fix Metal buffer type mismatch

**Status:** OPEN
**Effort:** Medium (investigation)
**File:** `src/voxcpm2_tts.cpp`
**Approach:** Currently CPU-only due to SIGSEGV from `matmul_mv_ggml`
allocating input tensors in CPU-side mem buffer. Root cause is likely a
buffer type mismatch in `ggml_backend_sched` allocation strategy. Fixing
unlocks GPU for the entire pipeline + flash attn.

#### §176o embed_tokens micro-graph elimination

**Status:** MOSTLY DONE
**Effort:** Small
**Done (HISTORY §180, §187, 2026-06-20):** FunASR (`CRISPASR_FUNASR_EMBED_FAST`),
GLM-ASR (`CRISPASR_GLM_ASR_EMBED_FAST`), MOSS-Audio
(`CRISPASR_MOSS_AUDIO_EMBED_FAST`), Qwen3-ASR
(`CRISPASR_QWEN3_ASR_EMBED_FAST`), Gemma4-E2B
(`CRISPASR_GEMMA4_E2B_EMBED_FAST`, + sqrt(d) scale).
**Already had fast path:** Orpheus, OuteTTS (n==1 direct row read at port time).
**Already optimal:** LFM2-Audio (direct row reads, no graph).
**N/A:** Mini-Omni2 (never calls embed with n=1).
**Remaining:** Granite-Speech (model too large for VPS bench).
**A/B:** funasr embed step ~1.6× faster on VPS.

#### §176t read_tensor_f32 weight pre-cache for VITS-family TTS

**Status:** PARTIAL
**Effort:** Small per backend (mechanical)
**Done:** Piper-TTS (`CRISPASR_PIPER_WEIGHT_CACHE`, 14% speedup, §adbfb274),
MeloTTS (`CRISPASR_MELOTTS_WEIGHT_CACHE`, 16% speedup, §237000c4).
**Remaining:** OpenVoice2 (same VITS architecture, same read_tensor_f32 pattern).
SpeechT5 has only 1 call (not worth it). FastPitch uses ggml graphs (no need).
**Approach:** Pre-populate `unordered_map<tensor*, vector<float>>` at model
load, modify `read_tensor_f32` to check cache first via module-level context
pointer. Cost: ~2× model RAM (acceptable for <200 MB TTS models).
**Impact:** 14-16% end-to-end synthesis speedup by eliminating repeated
`ggml_backend_tensor_get` + F16→F32 dequant per synthesis call.

#### §176p MOSS Audio: wire encoder flash attention

**Status:** DONE — `ggml_flash_attn_ext` wired for 32-layer encoder (`294bedff`)
**File:** `src/moss_audio.cpp`
**Done:** Replaced manual mul_mat+soft_max+mul_mat with `ggml_flash_attn_ext`.
F16 mask for flash_attn compat. Layer offload and fused QKV still open.

#### §176q greedy_decode.h: eliminate per-token alloc

**Status:** DONE — `thread_local static` probs vector (`294bedff`)
**File:** `src/core/greedy_decode.h`
**Done:** `sample_temp` probs vector is now `thread_local static`;
eliminates one `malloc(vocab×8)` per AR decode step across all 30+ backends.

#### §176r beam_decode.h: heap-based top-K

**Status:** DONE — K-element min-heap (`294bedff`)
**File:** `src/core/beam_decode.h`
**Done:** `top_k_log_softmax` uses a K-element min-heap instead of
vocab-sized `std::vector<int>` + `partial_sort`. Eliminates ~128–600 KB
alloc per beam expansion step.

#### §176s Encoder graph caching by shape

**Status:** MOSTLY DONE — 16 backends cached.
**Effort:** Small-Medium
**Backends done:** SenseVoice, Paraformer, FunASR, Nemotron, Canary,
Canary-CTC, Moonshine, Moonshine-Streaming, Parakeet, Qwen3-ASR,
GLM-ASR, Voxtral, Voxtral4B, MOSS Audio, Granite Speech, Kyutai STT.
**Remaining:** OmniASR (encoder+decoder fused in one graph — needs
refactor to split).
**Approach:** Metadata-swap technique: swap `compute_meta` with a
persistent `cached_enc_meta` before calling the graph builder, swap
back after. The cached graph arena persists across calls. Invalidate
when the topology key (T_mel, T_lfr, n_samples, etc.) changes.

#### §176u Chatterbox CPU backend thread count (wired; default tuning open)

**Status:** PARTIAL — bug fixed + parity-proven (`bca36bde`); optimal default
+ speedup magnitude OPEN (needs a quiet-machine A/B — owner: §176 campaign).
**Effort:** Small.
**Finding:** `chatterbox.cpp` called `ggml_backend_cpu_init()` but never
`ggml_backend_cpu_set_n_threads()`, so the compute-bound T3 AR decode (≈88% of
synth wall) ran at `GGML_DEFAULT_N_THREADS=4` on every machine and the CLI `-t`
flag was silently ignored for chatterbox.
**Shipped:** wire `c->n_threads` to the CPU backend, default
`min(8, hardware_concurrency)` (never below `-t`), env override
`CRISPASR_CHATTERBOX_THREADS`.
**Parity:** bit-identical — ggml CPU matmul splits output rows per-thread with
no cross-thread reduction, so output is thread-count-independent. Verified:
4-thread vs 8-thread emitted speech tokens identical (32/32).
**Open / handed to §176:** could NOT quantify the speedup — measured only on a
box at load avg 16–38 (multi-session), where absolute timing was unusable and 8
threads even over-subscribed/regressed. Needs a quiet-machine interleaved A/B
(alternating order, min-of-arm) to (a) confirm the win and (b) pick the optimal
default/cap (E-core penalty on Apple silicon?). Same applies to whether the
global CLI default `min(4, hw)` should rise for other compute-bound backends.


---

### §222 Higgs-Audio-v3 TTS (bosonai/higgs-tts-3-4b) — Qwen3-4B + 8-codebook discrete codec

**Status:** SURVEYED. **Priority:** LOW (non-commercial license).
**Issue:** [#198](https://github.com/CrispStrobe/CrispASR/issues/198).
**Effort:** Medium-Large.

**Architecture** (inspected from `sglang-omni/sglang_omni/models/higgs_tts/`):

- **Backbone:** Qwen3-4B-Base (36L, 2560-d, 32 heads, 8 KV heads, GQA 4:1,
  SwiGLU 9728, RoPE θ=1M, QK-norm). `body.layers.*` in checkpoint.
- **Multi-codebook embedding:** 8 codebooks × 1026 vocab fused into one
  `[8*1026, 2560]` weight; codes offset-indexed then summed across codebooks.
  `tied.embedding.modality_embeddings.0.embedding.weight`.
- **Multi-codebook head:** tied with embedding; `linear(hidden, 8*1026)` →
  reshape to `[L, 8, 1026]` for per-codebook sampling.
- **Delay pattern:** codebook `c` is delayed by `c` steps (BOC/EOC padding).
  AR generates one `[8]` code vector per step; de-stagger recovers `[T, 8]`.
- **Audio codec** (Higgs Audio V2 Tokenizer, embedded in checkpoint under
  `tied.embedding.modality_embeddings.0.model.*`, 529 tensors):
  - `semantic_model`: HuBERT/WavLM-style (conv feature extractor + transformer
    encoder) — only used for encoding (voice cloning), NOT decode path.
  - `acoustic_encoder`/`acoustic_decoder`: DAC-style (Snake1d + ConvTranspose1d
    upsampler). Same architecture as TADA codec / `core/seanet_decoder.h`.
  - `quantizer`: 8 RVQ codebooks (in/out projections per codebook).
  - `fc`/`fc1`/`fc2`: linear projections between semantic and acoustic spaces.
  - Decode path: `codes → quantizer.decode → fc2 → acoustic_decoder → 24kHz PCM`.
- **Emotion/style control:** inline `<|category:value|>` tokens (21 emotions,
  singing/whispering, speed, pitch, sound effects).
- **102 languages**, 85 at Tier 1 (<5% WER).
- **Size:** 9.3 GB BF16 → ~2.5 GB Q4_K.

**Component mapping to existing CrispASR code:**

| Higgs Component | CrispASR Analogue | New code |
|---|---|---|
| Qwen3-4B backbone | `qwen3_asr.cpp` / `qwen3_tts.cpp` | Reuse |
| QK-norm (q_norm, k_norm) | — | ~10 lines (RMSNorm on Q,K) |
| Fused multi-codebook embedding | — | ~30 lines |
| Fused multi-codebook head | — | ~20 lines |
| Delay pattern stagger/de-stagger | — | ~30 lines |
| DAC acoustic decoder | `tada_codec.cpp`, `core/seanet_decoder.h` | Adapt |
| RVQ quantizer decode | `core/snac.h` pattern | Adapt |
| fc2 Linear | — | Trivial |

**License:** Boson Higgs TTS 3 Research and Non-Commercial License. Production
APIs and revenue-generating use require a separate commercial license. Not a
total blocker but deprioritises this vs Apache/MIT backends.

**URLs:**
- HF model: https://huggingface.co/bosonai/higgs-tts-3-4b (9.3 GB single safetensors)
- HF codec (standalone): https://huggingface.co/bosonai/higgs-audio-v2-tokenizer
- Serving framework: https://github.com/sgl-project/sglang-omni
- Upstream HF PR for tokenizer: https://github.com/huggingface/transformers/pull/40294
- GitHub issue: https://github.com/CrispStrobe/CrispASR/issues/198

**Source code** (`git clone https://github.com/sgl-project/sglang-omni`,
key files under `sglang_omni/models/higgs_tts/`):
- `modeling.py` — `HiggsFusedMultiTextEmbedding` + `HiggsFusedMultiTextHead` (58 lines)
- `sampler.py` — delay-pattern state machine, per-row + batched CUDA-graph path
- `audio_codec.py` — vocoder facade, loads codec from TTS checkpoint
  (`tied.embedding.modality_embeddings.0.model.*` prefix)
- `_vendored/higgs_audio_v2_tokenizer_hf.py` — 940-line full codec:
  DAC encoder/decoder (Snake1d), HuBERT semantic model, RVQ quantizer,
  semantic encoder/decoder bridges
- `model.py` — sglang model class, backbone prefix map
  (`tied.embedding.text_embedding.` → `backbone.model.embed_tokens.`,
  `body.layers.` → `backbone.model.layers.`, etc.)
- `utils.py` — `apply_delay_pattern` / `reverse_delay_pattern`, BOC/EOC
- `weight_loader.py` — `DiscreteWeightMapper` for checkpoint key remapping
- `hf_config.py` — `HiggsMultimodalQwen3Config`

**Testing:** 9.3 GB model requires Kaggle GPU kernel (won't fit 8 GB VPS).


## §ARK — ARK-ASR-3B support (⚠️ EXPERIMENTAL / WIP; branch feat/arkasr-3b)

**STATUS 2026-06-29**: ⚠️ **experimental / WIP** — wired through CLI, session C
ABI, model registry, and docs; shipped to main + GGUF published
(cstr/ark-asr-3b-GGUF). Core ASR VALIDATED on **both GPU and CPU**: jfk.wav →
verbatim English; De-Abwasch 79 s → verbatim German. CLI: `crispasr -m
ark-asr-3b-q8_0.gguf --backend ark-asr -f audio.wav`. **GPU is now the default**
(Metal-validated; force CPU with `CRISPASR_ARKASR_CPU=1`).

Done: full wiring per docs/contributing.md (C API, adapter, factory, CMake lib+cli,
registry, session ABI + symbols in libcrispasr.dylib + available_backends, live
test, docs). Bindings audited: ASR dispatch is automatic (verified ark-asr in
python Session.available_backends()); added ark-asr to the python set_ask
docstring enumeration; dart/go/rust/java/ruby/wasm need no change (illustrative
lists + generic set_ask docstrings; no new setter). `-l` injection (§9b). GGUFs
published cstr/ark-asr-3b-GGUF
(f16 7 GB / q8_0 4 GB / q4_k 3.3 GB, all verbatim) + HF model card.

**Diff harness RUN** (vs PyTorch bf16 reference, jfk): log-mel cos 0.999993,
first_logits (q8_0) cos 0.999646, audio_embeds mean cos 0.999445 (one low-mag
frame at 0.953 = q8 noise, harmless — logits pass). Pipeline matches the blueprint.

Both perf follow-ups settled by measurement (gated `CRISPASR_ARKASR_TIMING=1`):
- **(a) step-graph cache = DUD** — per-step build+alloc is 0.3–0.5% of each step
  (~0.45 ms vs ~120 ms compute); decode is fully compute-bound on the 3B forward.
- **(b) GPU "no tokens" no longer reproduces** — GPU is verbatim on M1 Metal,
  ~5.6× faster prefill, ~neutral per-token decode, ~1.7× overall. Default. CUDA unvalidated.

**Language drift FIXED** by matching the reference's single-pass whole-audio
(commit: CAP_UNBOUNDED_INPUT + single-pass ark_asr_transcribe). The drift was a
chunking artifact (independent 30 s windows re-detected language); the RoPE encoder
has no positional cap, so the reference encodes the whole clip in one pass. De-Abwasch
79 s now verbatim German throughout. Long audio > `CRISPASR_ARKASR_MAX_SINGLE_PASS_S`
(default 300 s) falls back to internal chunking (drift can return there → use --vad).
Also strip the model's leading `.` transcript-opening token in output cleanup.

Cross-chunk language conditioning DONE: the >cap chunked fallback now seeds each
window's assistant turn with the previous chunk's transcript tail (32 tokens) so
the model continues in the same language instead of re-detecting/translating per
window. Validated: De-Abwasch with forced 30 s chunking
(CRISPASR_ARKASR_MAX_SINGLE_PASS_S=30) → German throughout (chunk 2 previously
translated to English). Opt out with CRISPASR_ARKASR_NO_CHUNK_CONTEXT=1.

Open (minor): CUDA validation (only Metal validated). Nice-to-have, not a blocker.

Port of [AutoArk-AI/ARK-ASR-3B](https://huggingface.co/AutoArk-AI/ARK-ASR-3B):
a 19-language ASR model = **Whisper-large-v3 encoder with partial RoPE** +
**MLP adapter (merge-4)** + **Qwen2.5-3B decoder** with audio-token injection.
On-policy distilled (THUNLP/OPD). BF16 safetensors, 8.14 GB, trust_remote_code.

### Architecture (from config.json + modeling_*.py, mirrored in `.arkasr-ref/`)

**Audio encoder** (`WhisperSpecialEncoder`, modeling_audio.py):
- Standard Whisper conv stem: `conv1` k=3 s=1 p=1, `conv2` k=3 s=2 p=1, both GELU.
  Input 128 mel bins → d_model 1280. 3000 mel frames → 1500 encoder frames.
- 32 pre-norm encoder layers: `self_attn_layer_norm` → attn → +res →
  `final_layer_norm` → fc1(5120) → gelu → fc2 → +res. 20 heads, head_dim 64.
  q/v/out_proj have bias; **k_proj has NO bias** (Whisper convention).
- **Partial interleaved RoPE** applied to Q and K (`WhisperRoPESdpaAttention`):
  `RotaryEmbedding(dim = head_dim//2 = 32)`, base=10000, rope_ratio=1. Rotates
  **only the first 32 of 64 head dims** (dims 32..63 pass through unchanged),
  adjacent-pair (interleaved) rotation = ggml `GGML_ROPE_TYPE_NORMAL` n_dims=32.
  See [[feedback-x-transformers-partial-rope]]. Non-causal, no attention mask.
- **No final encoder layer_norm** — `whisper.layer_norm = nn.Identity()`.

**Adapter** (`AudioMLPAdapter`):
- `layer_norm` = LayerNorm(1280) over encoder output (this replaces the dropped
  encoder LN).
- merge_factor=4: reshape (B,1500,1280) → (B,375,5120) (concat 4 consecutive
  frames). If T%4≠0 truncate to multiple of 4.
- `adapting` = Linear(5120→4096) → GELU → Linear(4096→2048).
- Output: 375 audio embeddings (2048-dim) for a full 30 s clip.

**Decoder**: stock Qwen2.5-3B — hidden 2048, 36 layers, GQA 16 heads / 2 KV
(head_dim 128), intermediate 11008 SwiGLU(silu), RMSNorm eps 1e-6, rope_theta
1e6, vocab 151936, **tied embeddings** (lm_head = embed_tokens). Reuse the
existing Qwen2 graph (cosyvoice3-LM / qwen3_asr / mimo_asr).

**Audio injection**: `audio_token_id` 151663 placeholders in the embedded
prompt are overwritten by the first N adapter frames, where
`N = ((mel_frames+1)//2)//4` computed from the *real* (unpadded) audio length.
Encoder runs on the 30 s-padded mel; the first N merged frames map to real audio
(right-padding), the rest are dropped.

### Prompt / decode recipe (from processing_arkasr.py)
Token stream (add_special_tokens=False, no newlines):
`<|user|>`(151665) `<|begin_of_audio|>`(151666) `<|audio|>`×N(151663)
`<|end_of_audio|>`(151667) `<|assistant|>`(151668)
Greedy decode (HF default greedy; max 256 new tokens), stop at EOS
`<|im_end|>`(151645). bad_words: block all special tokens except EOS. Tokenizer
is Qwen2 GPT-2 byte-level BPE, vocab 151936 + added specials (added_tokens.json).
Mel: WhisperFeatureExtractor — feature_size 128, n_fft 400, hop 160,
sampling_rate 16000, chunk_length 30 (n_samples 480000, nb_max_frames 3000),
dither 0.0. >30 s audio → chunk into 30 s windows.

### Implementation checklist (per docs/contributing.md)
1. [ ] `models/convert-arkasr-to-gguf.py` — lazy safetensors → GGUF F16, embed
       Qwen2 BPE vocab+merges + special tokens, hparams under `arkasr.*` +
       `arkasr.whisper.*`, `general.architecture="arkasr"`. (agent-scaffolded)
2. [ ] `src/ark_asr.{h,cpp}` — C runtime: GGUF load, mel, whisper+RoPE encoder
       graph, adapter, Qwen2 decoder w/ KV-cache greedy decode, BPE detok. (ME)
3. [ ] `examples/cli/crispasr_backend_ark_asr.cpp` — CLI adapter.
4. [ ] Register: `crispasr_backend.cpp` factory + `detect_backend_from_gguf`
       (filename "ark"/"arkasr" + arch "arkasr"); `crispasr_model_registry.cpp`.
5. [ ] `src/CMakeLists.txt` + `examples/cli/CMakeLists.txt` library + link.
6. [ ] `src/crispasr_c_api.cpp` — 9 edit points (session ABI mirrors CLI,
       see [[project-session-abi-reimplements-cli]]).
7. [ ] `examples/crispasr-quantize/main.cpp` — keep norms/bias F32; F16+Q8_0+Q4_K.
8. [ ] Diff harness: `tools/reference_backends/arkasr.py` + register in
       `tools/dump_reference.py` + `crispasr_diff_main.cpp` branch. (agent-scaffold)
9. [ ] `tests/` live test + `tests/env-live-tests.sh`.
10.[ ] README / docs / bindings docstrings.

### Validation methodology
crispasr-diff stage cosines vs PyTorch reference at: mel, encoder (per-layer +
final), adapter, decoder embed-with-audio-injected, per-layer hidden, logits.
Then ASR-roundtrip an English + a German FLEURS clip (verbatim text gate).

### BLOCKER — disk/compute for validation
Both local volumes are ~full (`/Volumes/backups` 7.5 GB free, `/Users` 9.9 GB).
The 8.14 GB safetensors + ~6 GB F16 GGUF do not fit. Scaffolding + graph code is
done on the M1; **download + conversion + diff validation must run on the VPS**
(`/mnt/storage`, `/mnt/akademie_storage`) or after freeing local space.
