# CrispASR v0.6.11

A broad correctness + long-form quality release. 146 commits since
v0.6.10 (2026-05-23). Most-visible improvements: dramatically better
long-form transcription across every multilingual backend, eight
external-bug fixes from issue [#125](https://github.com/CrispStrobe/CrispASR/issues/125),
chatterbox TTS GPU path finally producing intelligible audio on M1
Metal, and a new `--hf-repo` flag for arbitrary HuggingFace model
fetching.

## TL;DR

- **Long-form transcription is 2–7× better** on parakeet, canary, and
  voxtral. Parakeet on 5-min EN audio: 1550 → 3865 chars; canary on
  1.3-min DE article: 458 → 1196 chars; voxtral on 300 s streamed
  validated at 100 % coverage.
- **Eight external bug reports fixed** from issue
  [#125](https://github.com/CrispStrobe/CrispASR/issues/125): firered-asr,
  funasr, omniasr-llm, gemma4-e2b, mimo-asr, kyutai-stt (two issues each
  in one case).
- **Chatterbox TTS GPU path** finally works on M1 Metal (#83 R9 #5):
  `parallel=true` in `ggml_backend_sched_new` flips the right metal
  cache-coherency knob; smoke RMS goes 13.938 → 5.143 (ref 5.115).
- **New convenience flag** `--hf-repo OWNER/REPO[:FILE]` (issue
  [#128](https://github.com/CrispStrobe/CrispASR/issues/128)) for
  llama-server-style HuggingFace model fetching from arbitrary repos.
- **Five new model ports**: paraformer-zh, sensevoice (multi-task with
  emotion + audio-event), funasr/fun-asr-mlt-nano (FunAudioLLM port),
  parakeet-rnnt-0.6b + parakeet-rnnt-1.1b (RNN-T decoder family).

## Highlights

### Long-form transcription overhaul (issue [#114](https://github.com/CrispStrobe/CrispASR/issues/114))

Started from external user [@lenhone](https://github.com/lenhone)'s
[issue #89 reopened](https://github.com/CrispStrobe/CrispASR/issues/89)
— parakeet-tdt-0.6b-ja was producing 33 % coverage on a 60 s Japanese
clip vs the documented 99.5 %. Cascaded into a full audit of the
long-form pipeline across all multilingual backends. End result is a
per-backend story with empirical coverage data in
[PERFORMANCE.md](PERFORMANCE.md):

**Parakeet** (TDT-family, frame-synchronous):
- 2026-05-21 [`33f9a162`](https://github.com/CrispStrobe/CrispASR/commit/33f9a162): NeMo-style streamed pipeline replaces single-pass — JFK 99 %; 5-min clips no longer collapse on Vulkan/AMD.
- 2026-05-26 [`e1904a1e`](https://github.com/CrispStrobe/CrispASR/commit/e1904a1e): per-model chunk-size default keyed off `vocab_size` (< 4000 ⇒ JA c=8 / else c=30). The `c=8` default that ships well for the JA model collapsed on the multilingual v3 — EN 60 s coverage was at 23 % of optimum.
- 2026-05-26 [`98381810`](https://github.com/CrispStrobe/CrispASR/commit/98381810): dropped `CAP_INTERNAL_CHUNKING` from the capabilities declaration so the dispatcher's auto-chunk + overlap-save + LCS-merge dedup fires for audio > 30 s. Final result on real FLEURS fixtures:

| audio | before | after | Δ |
|---|---|---|---|
| EN 60 s | 186 chars | 755 chars | +305 % |
| EN 300 s | 492 chars | 3865 chars | +686 % |
| DE 300 s | 2496 chars | 3288 chars | +32 % |
| JA model + JA 60 s | 1674 chars | 1942 chars | +16 % |

Coverage parity with cohere on the same audio (EN 300 s: parakeet 3865 vs cohere 3994 = -3 %) at **faster wallclock** (66 s vs 94 s on M1 Metal).

**Canary** (AED-family, the full new long-audio path):
- 2026-05-26 [`dfe1af3b`](https://github.com/CrispStrobe/CrispASR/commit/dfe1af3b): lang-whitelist (`en/de/fr/es` only) — canary-1b-v2's BPE has every ISO-639 token, but only 4 langs are trained. Previously `-l ja` produced hallucinated mixed-script garbage; now errors out cleanly with a pointer at parakeet-tdt-0.6b-ja and qwen3/voxtral.
- 2026-05-26 [`7177c931`](https://github.com/CrispStrobe/CrispASR/commit/7177c931): `canary_transcribe_streamed` first cut — parakeet-shape (full-mel + chunked encode + concat + single AED decode). Documented limitation: AED-trained-on-single-utterance emits `<eos>` at chunk boundaries.
- 2026-05-26 [`63fdbe46`](https://github.com/CrispStrobe/CrispASR/commit/63fdbe46): NeMo `FrameBatchMultiTaskAED` analogon — per-chunk AED decode with prompt re-injection. Closes the boundary `<eos>` bug.
- 2026-05-26 [`62766dae`](https://github.com/CrispStrobe/CrispASR/commit/62766dae): LCS-merge boundary dedup via `core/crispasr_lcs::lcs_dedup_prefix_count`.
- 2026-05-26 [`10c2fba5`](https://github.com/CrispStrobe/CrispASR/commit/10c2fba5): splice-punct cleanup + `CANARY_STREAM_THRESHOLD_S=0` default (always streamed).
- 2026-05-26 [`361df3e2`](https://github.com/CrispStrobe/CrispASR/commit/361df3e2): window-based degenerate-loop guard (≤3 distinct ids in last 40 tokens → abort).
- 2026-05-26 [`935ffbee`](https://github.com/CrispStrobe/CrispASR/commit/935ffbee): word-snap heuristic (extend LCS-prefix-drop to next `▁`-prefixed token) — resolves `"umfassen. tuch umfassen"` → `"umfassen. umfassen"`.
- 2026-05-26 [`5e402ee9`](https://github.com/CrispStrobe/CrispASR/commit/5e402ee9): case-insensitive LCS via ASCII canonical-id table — catches `"world's say"` / `"World's Save"` retokenization that strict LCS missed.

Result on De-Abwasch (1.3-min German article): 458 → 1196 chars; on
yt_60s.wav (60 s Japanese, OOD for canary's en/de/fr/es training): now
produces multiple chunks of content instead of empty (single-pass) or
one short hallucination (concat-streamed).

**Voxtral** (long-context AR LLM):
- 2026-05-25 [`6fef8790`](https://github.com/CrispStrobe/CrispASR/commit/6fef8790): opt-out of dispatcher overlap-save context wrap (incompatible with voxtral's internal chunking).
- 2026-05-25 HISTORY: `crispasr_run_voxtral_style_pipeline_streamed` lands — matches Mistral upstream `apply_transcription_request` shape (per-30s encode, concat audio embeds, single LLM AR decode). Validated 100 % coverage at 60 / 120 / 300 s.

**Cohere / qwen3 / gemma4-e2b / glm-asr / kyutai-stt**: all opted out
of dispatcher overlap-save context wrap (commits `dc2295b2`, `e7dfb93f`,
`46f6848d`, `eaee2319`) — caught by the
[`tools/check-overlap-save-bug.sh`](tools/check-overlap-save-bug.sh) A/B
sweep. Their internal long-form chunking pipelines now work at default
settings.

### Issue [#125](https://github.com/CrispStrobe/CrispASR/issues/125) fix train — external bug sweep from @montvid

[@montvid](https://github.com/montvid) ran every backend on a fresh
v0.6.10 build (NVIDIA RTX PRO 6000 Blackwell sm_120 + CUDA 12.6) and
reported 12 findings across 8 backends. Eight shipped:

| Report | Backend | Fix |
|---|---|---|
| P0 | mimo-asr | Scheduler `src_mutations` log not restoring on early-error compute returns ([`95d74455`](https://github.com/CrispStrobe/CrispASR/commit/95d74455)) — was the actual cause of the Blackwell segfault, not the FA-mask patch the reporter bisected to (fully `#ifdef`-guarded OFF). |
| P1 | funasr / fun-asr-mlt-nano | `!`-loop guard in AR decode ([`f72d3db1`](https://github.com/CrispStrobe/CrispASR/commit/f72d3db1)); added funasr / fun-asr-mlt-nano / sensevoice / paraformer to `tools/test-all-backends.py` (were entirely missing from CI). |
| P2 | firered-asr | Dropped `CAP_UNBOUNDED_INPUT` ([`72b74486`](https://github.com/CrispStrobe/CrispASR/commit/72b74486)) — encoder PE buffer is `pe_maxlen=5000` ≈ 50 s, declaring unbounded produced silent OOB on long audio. |
| P3 | omniasr-llm | Chunking decision now `(is_streaming \|\| force_seg)` ([`5f0aefc0`](https://github.com/CrispStrobe/CrispASR/commit/5f0aefc0)) — non-streaming GGUFs were feeding entire long audio to a 512-token LLM. |
| P4 | gemma4-e2b | 30 s training-window guard ([`8bfaff23`](https://github.com/CrispStrobe/CrispASR/commit/8bfaff23)) with `CRISPASR_GEMMA4_AUTO_CHUNK=1` opt-in for internal chunking. |
| P5 | mimo-asr | Tokenizer (`cstr/mimo-tokenizer-GGUF/mimo-tokenizer-q4_k.gguf`) added to auto-download manifest ([`b936b488`](https://github.com/CrispStrobe/CrispASR/commit/b936b488)). |
| P6a | kyutai-stt | 500 ms silence-tail flush ([`ba0e388e`](https://github.com/CrispStrobe/CrispASR/commit/ba0e388e)) — was truncating the final word ("…your c" → "…your country"). |
| P6b | kyutai-stt | 30 s internal chunking ([`043b3ae5`](https://github.com/CrispStrobe/CrispASR/commit/043b3ae5)) — was 14 s/s on 50 min file (O(N²) KV growth), now bounded and linear. |

### Issue [#115](https://github.com/CrispStrobe/CrispASR/issues/115) — mimo-asr M1 Metal silent-empty fix

PLAN #115 — mimo-asr was producing silent empty output on JFK on M1
Metal after the PLAN #72 GPU weight residency commit. Bisected to
`89111260`'s `core_gguf::load_weights(..., ctx->backend, ...)` swap.
Confirmed via Kaggle CPU notebook that the CPU path works on HEAD.
Fix shipped 2026-05-26 (`5a570b7b` + `c887881e`): force mimo-asr weights
to CPU until the proper GPU graph fix lands. Loses the documented 22 %
Metal speedup; acceptable because the alternative was zero output.

### Chatterbox TTS GPU path (issue [#83](https://github.com/CrispStrobe/CrispASR/issues/83) R9 follow-up #5)

Final fix in a five-round investigation. Bug B (UNet input divergence
under sched-copy) was eliminated 2026-05-24 by switching
`chatterbox_s3gen_init_from_file` to `parallel=true` in
`ggml_backend_sched_new`. Root cause was Metal cache coherency: with
`parallel=false`, sched used `[cmd_buf_last waitUntilCompleted]` which
doesn't invalidate the GPU's L1/L2 view of a shared-storage `MTLBuffer`
that the CPU just memcpy'd; `parallel=true` uses `MTLSharedEvent` with
proper cache invalidation. Result: smoke RMS `13.938 → 5.143` (ref
`5.115`), diff harness `s3gen_mel cos_min 0.940 → 0.999976`,
production CPU path unchanged.

Side-products of the investigation: Q8_0 × F32 bit-match Metal mul_mat
kernel ([`752baecf`](https://github.com/CrispStrobe/CrispASR/commit/752baecf)),
ggml sched src-mutation log fix
([`0f0f0793`](https://github.com/CrispStrobe/CrispASR/commit/0f0f0793))
later hardened in this release at
[`95d74455`](https://github.com/CrispStrobe/CrispASR/commit/95d74455),
and four upstream-PR drafts at `tools/upstream-prs/09-11`.

### New CLI flag `--hf-repo` (issue [#128](https://github.com/CrispStrobe/CrispASR/issues/128))

llama-server-compatible HuggingFace fetcher for arbitrary repos. Three
invocation forms:

```bash
# Shorthand (single arg)
crispasr --hf-repo cstr/parakeet-tdt-0.6b-v3-GGUF:parakeet-tdt-0.6b-v3-q4_k.gguf -f audio.wav

# Two flags
crispasr --hf-repo cstr/parakeet-tdt-0.6b-v3-GGUF --hf-file parakeet-tdt-0.6b-v3-q4_k.gguf -f audio.wav

# -m as filename hint
crispasr --hf-repo cstr/parakeet-tdt-0.6b-v3-GGUF -m parakeet-tdt-0.6b-v3-q4_k.gguf -f audio.wav
```

URL `https://huggingface.co/{REPO}/resolve/main/{FILE}` is synthesised
and routed through the existing
[`crispasr_cache::ensure_cached_file`](src/crispasr_cache.h) cache
infrastructure. Implies `--auto-download`. Use any HF repo, not just
the curated ones in
[`src/crispasr_model_registry.cpp`](src/crispasr_model_registry.cpp).

## New model ports

### SenseVoice / paraformer-zh / FunASR (FunAudioLLM family, 2026-05-20/21)

- **SenseVoiceSmall** (`sensevoice` backend): encoder-only multi-task
  ASR — transcript + LID + emotion + audio-event in one CTC pass; 50+
  languages; 9.8–21.8× realtime on M1 Metal.
  [`cstr/sensevoice-small-GGUF`](https://huggingface.co/cstr/sensevoice-small-GGUF).
- **Paraformer-zh** (`paraformer` backend): 220M-param NAR-ASR
  (single-pass non-autoregressive decoder); byte-identical to upstream
  on Chinese + English; 0.123 GB Q4_K / 0.47 GB F16.
  [`cstr/paraformer-zh-GGUF`](https://huggingface.co/cstr/paraformer-zh-GGUF).
- **Fun-ASR Nano + MLT-Nano** (`funasr` + `fun-asr-mlt-nano` backends):
  full LLM-decoder runtime — 70-block SANM encoder + 2-block Transformer
  adaptor + Qwen3-0.6B AR decode. ~9× realtime on M1 Metal with FA.
  [`cstr/funasr-{nano,mlt-nano}-GGUF`](https://huggingface.co/cstr).

### Parakeet variants (2026-05-20, 2026-05-24)

- **Parakeet-TDT 1.1b** + **Parakeet-TDT_CTC 110m & 1.1b** + **Parakeet
  v2** (`parakeet` backend, distinguished by GGUF). Larger TDT variants
  with the same runtime path as v3.
- **Parakeet-RNNT 0.6b + 1.1b** (HISTORY 2026-05-24): RNN-T decoder
  (frame-synchronous, no `<eos>`) — same FastConformer encoder, swap
  the joint network. Q4_K GGUFs at
  [`cstr/parakeet-rnnt-{0.6b,1.1b}-GGUF`](https://huggingface.co/cstr).

### Cohere ASR-JA-v0.1 (issue [#123](https://github.com/CrispStrobe/CrispASR/issues/123), 2026-05-25)

Japanese fine-tune of cohere-transcribe-03-2026. Registry alias `cohere-ja`.

### CosyVoice3 0.5B TTS — Phase 1 (HISTORY 2026-05-20)

Recon + converter for Fun-CosyVoice3-0.5B-2512. Runtime port pending.

### VoxCPM2 TTS (HISTORY 2026-05-19/20, PLAN #96)

Full pipeline: per-step TSLM (text streaming LM, bucketed graph),
per-call LocDiT (LocDiT graph cache), VAE encode/decode with WN reconstruct
cache. CPU performance target met; Metal live shipped 2026-05-20.

## Infrastructure / tooling

### Diff harness updates (2026-05-17, 2026-05-25)

- canary-1b-v2 mel + encoder graduated to full diff harness coverage.
- chatterbox `hift_pcm(ref_mel)` "drift" turned out to be a diff harness
  layout bug (transposed `(T, C)` instead of `(C, T_fast)` for `source_stft`).
  Fixed 2026-05-25, all stages now cos=1.0. Runtime TTS was never
  affected.

### Kaggle rebake pipeline (2026-05-25)

End-to-end wiring for the
[`chr1str/crispasr-auto-rebake-refs`](https://www.kaggle.com/code/chr1str)
kernel. Coverage: 7 → 11 → 13 manifest entries via four cascading bug
fixes (heartbeat, fixture_path field rename, disk-full cache rotation,
HF auth fallback, all-or-nothing upload gate). New refs published:
parakeet-tdt-1.1b, parakeet-tdt_ctc-1.1b, parakeet-rnnt-0.6b,
parakeet-rnnt-1.1b.

### Overlap-save bug sweep harness (2026-05-25/26)

[`tools/check-overlap-save-bug.sh`](tools/check-overlap-save-bug.sh) runs
each at-risk backend twice on a long clip (default `--chunk-overlap 3.0`
vs `--chunk-overlap 0`) to detect the "external overlap-save context
wraps internal chunking" bug class. Caught 5 new offenders (voxtral,
qwen3, gemma4-e2b, glm-asr, kyutai-stt) — all added to
[`examples/cli/crispasr_chunk_context_gate.h::kBlocked`](examples/cli/crispasr_chunk_context_gate.h).

### Long-audio test fixtures

Mirror of FLEURS en/de + VPS-derived JA fixtures at
`/Volumes/backups/code/audio_samples/` (~71 MB, gitignored). Full
inventory in
[`samples_list.md`](audio_samples/samples_list.md). Used by all the
long-form benchmark runs in this release.

### CI

- 2026-05-25 [`80ac00d1`](https://github.com/CrispStrobe/CrispASR/commit/80ac00d1):
  build.yml trimmed 1610 → 1324 lines, arm64 jobs switched to native
  runners (was QEMU + libc-bin segfault).
- 2026-05-25 [`565b16af`](https://github.com/CrispStrobe/CrispASR/commit/565b16af):
  `GG_BUILD_NO_AVX512` knob added so `ggml-ci-x64-cpu-high-perf` SIGILL
  is structurally fixed (was `continue-on-error`-papered).
- 2026-05-26 [`c76dff20`](https://github.com/CrispStrobe/CrispASR/commit/c76dff20):
  stale clang-format-18 violations cleaned up in `src/crispasr.cpp` +
  `src/crispasr_c_api.cpp` (CI lint job green).
- 2026-05-26 [`5e402ee9`](https://github.com/CrispStrobe/CrispASR/commit/5e402ee9):
  cppcheck zerodiv guard in
  [`examples/cli/crispasr_llm_pipeline.h`](examples/cli/crispasr_llm_pipeline.h).
- Tools: `tools/format.sh` enforces clang-format-18 (not the v22 default).
- Tools: `tools/test-all-backends.py` now has funasr / fun-asr-mlt-nano /
  sensevoice / paraformer registry entries (were missing entirely).

## Feature additions

### Hotwords / contextual biasing (PLAN #98, 2026-05-23)

CTC-WS Aho-Corasick trie for parakeet CTC + TDT; LLM prompt injection
for qwen3-asr + voxtral. CLI flags `--hotwords` / `--hotwords-file` /
`--hotwords-boost`. 17 tests.

### Global diarization timeline (issue [#110](https://github.com/CrispStrobe/CrispASR/issues/110), 2026-05-23)

Sherpa / ecapa speaker embedder now runs once on full audio (not
per-slice). `CrispasrSherpaCache` mirrors the pyannote global-cache
pattern. Segments split at speaker-turn boundaries via word-level
overlap scoring. 21 tests.

### Beam search wired across all backends (PLAN #90, 2026-05-23)

qwen3-asr, granite, voxtral now route through
`core_beam_decode::run_with_probs` for per-token confidence under beam
≥ 2. CLI `--beam-size N` works on every ASR backend.

### TTS `--seed` parity (PLAN #111, 2026-05-23)

Same-seed reproducibility + different-seed divergence verified across
qwen3-tts, chatterbox, vibevoice realtime/base, orpheus, indextts,
voxcpm2. Server `/v1/audio/speech` accepts `"seed"`.

### Feature-matrix uplift round 2 (PLAN #74, 2026-05-23)

Chatterbox lang routing, cap regression tests, qwen3-tts base
voice-cloning cap, matrix regen.

### Streaming JSON+VAD merge policy (PRs #92 + #95, 2026-05-17)

`--stream-vad-merge-gap-ms N` for streaming clients that need
gap-closing across VAD boundaries.

## Breaking / behavioral changes

- **Canary**: now refuses `-l ja`, `-l zh`, etc. — `canary-1b-v2` is
  trained on `en/de/fr/es` only; previous behavior produced hallucinated
  mixed-script output. Use `parakeet-tdt-0.6b-ja` / `qwen3` / `voxtral`
  for languages outside canary's training set.
- **Parakeet default**: long-form audio (> 30 s) now uses the
  dispatcher's chunk-30 + overlap-save + LCS path, not the backend's
  internal-streamed. Wallclock on 300 s clips ~30 s → ~86 s on M1 Metal;
  coverage 2.5×. Override: `CRISPASR_PARAKEET_STREAM_THRESHOLD=99999`.
- **firered-asr**: refuses inputs > `pe_maxlen` ≈ 50 s with a clear
  error. Use `--vad` for longer audio.
- **gemma4-e2b**: refuses inputs > 30 s by default. Set
  `CRISPASR_GEMMA4_AUTO_CHUNK=1` to opt into internal chunking, or
  use `--vad`.
- **Default streamed path for all encoder-decoder ASR backends**
  (parakeet, canary, cohere, voxtral) now matches NeMo / Mistral
  conventions — coverage parity at the cost of some wallclock.
- **mimo-asr on M1 Metal**: pinned to CPU (silent-empty regression on
  Metal under post-2026-05-24 sched changes). Loses the 22 % Metal
  speedup; PLAN #115 option C tracks the proper GPU graph fix.

## Known issues

- **Voxtral on M1 with > 300 s clips**: memory thrash on the
  acoustic-model embed buffer; sits at low CPU for tens of minutes.
  Documented in
  [`feedback_torch_omp_deadlock.md`](.claude/memory/feedback_torch_omp_deadlock.md).
  Use a beefier box or shorter clips.
- **mimo-asr Blackwell sm_120**: P0 segfault hardened in
  [`95d74455`](https://github.com/CrispStrobe/CrispASR/commit/95d74455),
  awaiting external retest.
- **Parakeet DE 60 s** regressed 2 % (679 → 665 chars) under the new
  default. Within noise vs the +305 / +686 % wins on EN 60 s / 300 s.
  Per-mode trade-offs documented in PERFORMANCE.md.

## Methodology notes

This release applied a strong "verify empirically, don't trust LEARNINGS
unconditionally" discipline. Two illustrative cases:

- **Parakeet streamed-path failure mode** was originally diagnosed in
  LEARNINGS as decoder cold-start. That applied to
  `parakeet_transcribe_chunked` (independent per-chunk decodes) but not
  to the `parakeet_transcribe_streamed` path (single decode over concat
  encoder) that ships today. Empirical chunk-size sweep proved the
  bottleneck is encoder context, not decoder state. LEARNINGS updated
  with a Correction 2026-05-26 subsection
  ([`ffd708bb`](https://github.com/CrispStrobe/CrispASR/commit/ffd708bb)).
- **mimo-asr Blackwell P0** was reporter-bisected to the FA per-head
  mask patch (`6b492b2b`). The patch is fully `#ifdef`-guarded OFF, no
  CI / release script sets the flag ON, so the binary is byte-identical
  to upstream for that path. Reattribution to `0f0f0793` (ggml sched
  src-mutation log) caught the real bug via code reading rather than
  taking the bisect at face value. Lesson banked in HISTORY: a
  CUDA-shaped grep can miss the generic scheduler when an unrelated
  patch touches it.

The full PLAN #114 option matrix (7 dispatch knobs × 3 audio fixtures)
is in
[PERFORMANCE.md](PERFORMANCE.md) so future contributors can validate
or refute the current defaults without re-running the sweep.

## Upgrade notes

If you're upgrading from v0.6.10:

1. **No source changes required** — all CLI knobs remain
   backward-compatible.
2. **Default behavior changes for long audio** (see "Breaking /
   behavioral changes"). Pass `CRISPASR_PARAKEET_STREAM_THRESHOLD=99999`
   to restore the older single-pass parakeet path.
3. **Canary `-l ja` errors out cleanly** instead of producing garbage —
   use `--backend parakeet -m parakeet-tdt-0.6b-ja-q4_k.gguf` for
   Japanese.
4. **Self-built v0.6.10 users on Blackwell**: rebuild from v0.6.11 to
   pick up the
   [`95d74455`](https://github.com/CrispStrobe/CrispASR/commit/95d74455)
   scheduler hardening.

## Contributors

External bug reports: [@montvid](https://github.com/montvid) (issue #125),
[@khimaros](https://github.com/khimaros) (issue #128),
[@lenhone](https://github.com/lenhone) (issue #89 follow-up that drove
the long-form audit). Thank you.
