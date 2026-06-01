# CrispASR — Pending work

Pending roadmap items. Each is self-contained with files, approach, and
effort estimate. Completed items have been moved to `HISTORY.md`.

> **Numbering convention:** `§N` refers to PLAN items (sections in this
> file). `#N` refers to GitHub issues on CrispStrobe/CrispASR. They are
> independent sequences and numbers may collide. When in doubt, PLAN
> items are always written as `§N` and GitHub issues as `#N`.

**Latest release: v0.6.11** (2026-05-26, commit `b77a74eb`). 146 commits since v0.6.10 — long-form transcription overhaul (#114), issue [#125](https://github.com/CrispStrobe/CrispASR/issues/125) fix train, chatterbox GPU path final fix, `--hf-repo` flag (#128), five new model ports. Full notes in [`RELEASE_NOTES_v0.6.11.md`](RELEASE_NOTES_v0.6.11.md).

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

## Priority ordering

| Priority | Item | Effort | Status |
|---|---|---|---|
| **MEDIUM** | [#52 Qwen3-TTS](#52-qwen3-tts) — perf pass | Medium | talker + code_predictor + codec + ECAPA + codec_encoder all done; only step-4 perf pass open (~137 ms/frame → real-time) |
| **HIGH** | [#57 Commercial-friendly TTS expansion](#57-commercial-friendly-tts-backend-expansion) | Phased | Phases 1-3 DONE; Turbo WORKING; F0 wired in; native voice cloning shipped → HISTORY §82; **#83 production fix LANDED 2026-05-24 → HISTORY 2026-05-24 + LEARNINGS Round 9** — S3Gen UNet weight residency split (`s3.fd.*` on CPU, encoder/vocoder GPU): M1 Metal cos_min 0.940→**0.999980** in diff harness, intelligible audio at all T; comparable wall-time to pure CPU on M1. Q8_0×F32 bit-match Metal kernel committed (commit `752baecf`, upstream-PR-quality, drafted as PR 09). 3 upstream PR drafts in `tools/upstream-prs/09-11` covering Metal Q8_0 kernel + ggml-alloc drift bug report + scheduler NaN-at-large-T bug report. Linux CPU smoke validated on VPS. **R9 follow-up #4 2026-05-24**: two bugs found. **Bug A** (ggml sched dangling src pointers across `alloc_graph` calls): **FIXED** with a per-call mutation log in `ggml/src/ggml-backend.cpp` that restores `node->src[j]` originals at end of compute. Repro is the chatterbox CFG cond+uncond pair on the same gf; characterized and upstream-PR drafted at `tools/upstream-prs/10`. **Bug B** (`unet_input` divergence under sched-copy): **STILL OPEN — current code only WORKS AROUND it** by pinning `unet_input` to Metal in `cfm_euler_solve::run_denoiser`. With Bug A patched, the sched CPU→GPU copy delivers correct bytes to the kernel (verified by inline `tensor_get` before im2col dispatch) yet downstream compute still diverges (smoke rms ~16). Pinning `time_emb` is actively harmful (rms ~209), so the workaround is narrow. Without a real fix, any future user of `ggml_backend_sched` with a similar topology will hit this. Handover prompt for the follow-up at `handover-prompts/issue83-r9-followup-5-unet-input-routing.md`. **R9 follow-up #5 2026-05-24**: **Bug B FIXED via `parallel=true` in `ggml_backend_sched_new`**. After eliminating ~10 candidate hypotheses (cache barriers, blit copies, concurrency, fusion, optimize, n_cb variants, private-storage buffers, im2col edge case, rc-as-mul_mat) and proving the divergence is between host's and GPU's view of the same shared-storage Metal buffer on the uncond pass, the root cause is sched's between-submission synchronisation. With `parallel=false` (the chatterbox default until this fix) sched uses `[cmd_buf_last waitUntilCompleted]`, which doesn't invalidate the GPU's L1/L2 cached view of a shared-storage `MTLBuffer` that the CPU just memcpy'd between submissions. With `parallel=true` sched uses `ggml_backend_event_record` / `event_wait` → on Metal that's `MTLSharedEvent` `encodeSignalEvent` / `encodeWaitForEvent`, which carry proper GPU cache invalidation. Switched `chatterbox_s3gen_init_from_file` to `parallel=true`. Removed the unet_input pin workaround and the `CRISPASR_NO_INPUT_PIN` env override. Verification: GPU residency smoke `rms 16.x → 5.143`, CPU residency smoke `rms 5.139` unchanged (no regression), diff harness `s3gen_mel cos_min = 0.999976` (matches prior workaround baseline). LEARNINGS R9 #5 closes with new lessons 7' and 8 ("Check sched's `parallel` flag for Metal cache-coherency-shaped bugs"). End-to-end status: smoke rms `13.938 → 5.143`, diff `s3gen_mel cos_min 0.940 → 0.999976`, 2-mark trigger `NaN → 5.291`. Production CPU-residency path unchanged. Also open: Kartoffelbox_Turbo DE |
| **MEDIUM** | [#51c MiMo-V2.5-ASR F16 step decode](#51c-f16-step-decode) | Small | F16 step-decode validation blocked behind ≥32 GB box (see PLAN #51c); base runtime + Q4_K shipped → HISTORY §56 |
| **MEDIUM** | [#56 Kokoro multilingual phonemizer](#56-kokoro-multilingual-phonemizer-espeak-ng) | Small | espeak-ng + DE backbone shipped; HF GGUFs published 2026-05-01; auto-download wired; only Mandarin tones / JA kanji + diff-harness phonemizer-step polish remain |
| **MEDIUM** | [#58 MOSS-Audio-4B-Instruct](#58-moss-audio-4b-instruct) | Large | first audio-understanding (not just ASR) backend; introduces DeepStack cross-layer feature injection |
| **MEDIUM** | [#59 Cross-binding C-ABI parity](#59-cross-binding-c-abi-parity) | Medium | Go now has full surface (✅ all 11 capabilities). Java has transcribe+align+LID. Ruby has transcribe. JS needs WebAssembly approach |
| **DONE** | [#104 Stateful TDT frame-streaming](#104-stateful-frame-streaming-tdt-decode-for-parakeet-long-form-issue-89) | M-L | **DONE 2026-05-23.** Global z-norm + chunked encode + single decode → 99.5 % (was 59.7 %). Extended to canary (96.8 %) and fastconformer-ctc (98.5 %) via `CAP_INTERNAL_CHUNKING`. See HISTORY 2026-05-21. |
| **PARKED** | [#9 Parakeet TDT GPU](#9-parakeet-tdt-decoder-gpu) | Medium | Encoder 85%+ of time; LSTM+joint <0.7s; sequential steps limit GPU benefit |
| **BLOCKED** | [#42 VibeVoice-ASR 7B](#42-vibevoice-asr-7b) | High | Needs ≥16 GB RAM |
| **BLOCKED** | [#43 Fun-ASR-Nano](#43-fun-asr-nano) | Medium | License unclear |
| **DONE** | [#80 nano-cohere-transcribe-inspired tweaks](#80-nano-cohere-transcribe-inspired-perf--chunking-tweaks) | Small | 80a parked; **80b DONE**; **80c DONE**; **80d DONE** 2026-05-23 (audit: no fixes needed — all backends use energy chunker); 80e low-priority warmup deferred |
| **DEFERRED** | [#81 Nemotron-Speech-Streaming-EN-0.6B](#81-nemotron-speech-streaming-en-06b--first-cache-aware-streaming-native-asr) | M-L | NVOML license, ~60–75 % reuse from parakeet/canary; the new bit is cache-aware FastConformer streaming. Wait for `--stream-json` (issue #84) to settle + a second user request (only mention so far is issue #85) before starting. |
| **DONE** | [#86 Per-backend flash-attention wiring](#86-per-backend-flash-attention-wiring-crisperweaver-driven) | — | All backends now route through core helpers (`core_attn`, `core_sanm`, `core_conformer`) that unconditionally use `ggml_flash_attn_ext`. Only t5_translate excluded (T5 rel-pos bias incompatible). |
| **LOW** | [#87 `gpu_backend` runtime selector](#87-gpu_backend-runtime-selector-multi-backend-ggml-build) | ~1 week | Needs ggml-side multi-backend dispatch to land first. CrisperWeaver UI placeholder ready when the C-side is. |
| **LOW** | [#95 IndexTTS Chinese TN binary alternative](#95-indextts-15-chinese-tn--binary-alternative-to-the-python-wetext-hook) | survey only | Python `INDEXTTS_TEXT_NORMALIZER` hook shipped 2026-05-19. Hand-roll (#95a) is the right next step *when* a user reports a digit/date prompt that breaks; OpenFST vendoring (#95b) only after #95a grows past ~5 cases. |
| **IN PROGRESS** | [#97 More Parakeet variants](#97-more-parakeet-variants) | Small per-variant | TDT/TDT+CTC DONE; **parakeet-rnnt 0.6b+1.1b DONE 2026-05-24** — RNNT decoder + Q4_K GGUFs (447 MB / 770 MB) uploaded to `cstr/parakeet-rnnt-{0.6b,1.1b}-GGUF`. realtime-EOU + unified-en still open. |
| **DONE** | [#98 Hotwords / contextual biasing](#98-hotwords--contextual-biasing) | Phased | **Phase A+B DONE.** CTC-WS Aho-Corasick trie wired into parakeet CTC + TDT; LLM prompt injection for qwen3-asr + voxtral. `--hotwords` / `--hotwords-file` / `--hotwords-boost` CLI. 13+4 tests. Phase C deferred. → HISTORY 2026-05-23. |
| **DONE** | [#110 Global diarization timeline](#110-global-diarization-timeline) | Medium | Sherpa/ecapa now runs once on the full audio (not per-slice). `CrispasrSherpaCache` mirrors the pyannote global-cache pattern. Segments split at speaker-turn boundaries via word-level overlap scoring. 13+8 tests. → HISTORY 2026-05-23. |
| **LOW** | [#106 TEN-VAD](#106-ten-vad--low-latency-cross-platform-vad) | Small | Technically feasible VAD backend: C-compatible, 16 kHz / 10-16 ms frames, prebuilt libs + ONNX path. License is the gate: Apache 2.0 plus extra no-compete / own-app-only conditions from Agora. |
| **MOSTLY DONE** | [#114 Long-form transcribe — chunking-default ladder for voxtral / cohere / canary](#114-long-form-transcribe--make-chunkingstreamed-the-default-for-all-asr-backends-issue-89-follow-up) | Medium | **Parakeet DONE 2026-05-24 (`33f9a162`) + per-model chunk-default `e1904a1e` + drop `CAP_INTERNAL_CHUNKING` `98381810`, all 2026-05-26.** Empirical option matrix (PERFORMANCE.md) showed the dispatcher-side `--chunk-seconds 30 --chunk-overlap 3` path beats the backend's internal-streamed default on 3 of 4 cases; shipped that as the new default by dropping `CAP_INTERNAL_CHUNKING` from the capabilities declaration, which lets the dispatcher's `should_auto_chunk_long` fallback fire for audio > 30 s. Net result on the 7-fixture regression: 6 of 7 improved, 1 small DE regression (-2 %). EN 300 s up +150 % (1550 → 3865 chars); JA model on JA 60 s up +16 % (1674 → 1942); JFK 11 s unchanged (under the 30 s threshold). **Voxtral + cohere DONE 2026-05-25** via (1) parallel-track per-backend opt-out fixes (`dc2295b2` cohere, `46f6848d` gemma4-e2b/glm-asr, `eaee2319` kyutai-stt, `6fef8790` voxtral) — default chunking now lands at 96-100 % coverage at 60-300 s; (2) **voxtral_transcribe_streamed** in HISTORY 2026-05-25 (matches upstream Mistral `apply_transcription_request` shape — per-30s encode, concat audio embeds, single LLM AR decode). Matrix v1 in PERFORMANCE.md was on the pre-opt-out binary and overstated the failures; matrix v2 (post-opt-out) is the correct picture and shows >90 % coverage everywhere. **Canary lang-whitelist DONE 2026-05-26 (`dfe1af3b`)** — root cause was the BPE vocab having every ISO-639 `<|xx|>` token while the model is trained on en/de/fr/es only; passing `-l ja` produced mixed-script garbage instead of an error. Now refuses unsupported langs in the backend wrapper with a pointer at parakeet-tdt-0.6b-ja/zh and qwen3/voxtral. **`canary_transcribe_streamed` SHIPPED 2026-05-26 (`7177c931`)** + **per-chunk re-injection `63fdbe46`** — first cut was parakeet-pattern concat-then-decode (truncated at chunk boundaries because AED-trained-on-single-utterance treats splice points as `<eos>`); replaced with NeMo `FrameBatchMultiTaskAED`-shaped per-chunk decode (each chunk gets its own AED pass with the language/task prompt re-injected, results concatenated). Verified on real long-audio fixtures fetched from VPS (`/mnt/akademie_storage/yt_{60,120}s.wav` → `/Volumes/backups/ai/long-clips/`): JFK single-pass unchanged; JFK forced streamed now produces a complete transcript (with a boundary-overlap duplication artifact); 60 s Japanese clip produces multiple chunks of output instead of empty (single-pass) or one short hallucination (concat-streamed). **Boundary-overlap dedup `62766dae` + splice-punct cleanup + always-streamed default `10c2fba5` + degenerate-loop guard `361df3e2`** — P3 fully closed. JFK now produces `"...for you. Ask what you can do for your country."` through the always-streamed default, semantically equivalent to single-pass `"...for you, ask..."` with the splice converted to a sentence boundary by the LCS-dedup + punct-cleanup pair. The window-based loop guard (≤3 distinct ids in last 40 generated tokens → abort) addresses canary's BPE 2-token cycle (`▁yeah` + `,` alternating) that the funasr-style consecutive-id guard missed. 60 s Japanese clip: 12 s wall (4.9× RT), ~14 yeahs before the chunk's decoder aborts vs ~85 before the guard. **Real-language validation 2026-05-26** on `audio_samples/{en,de}/fleurs_60s.wav` + `audio_samples/multi/De-Abwasch-article.wav` (1.3 m DE article): streamed delivers **~2-3× more content** than single-pass in every case (single-pass truncates at the encoder amplification limit; en 60s 362→667 chars, de 60s 419→774 chars, De-Abwasch 458→1233 chars). **Word-snap heuristic `935ffbee`** — after LCS-prefix-drop, if the next surviving token doesn't start with `▁` (sentencepiece word-start), extend the drop until the next word-start token. De-Abwasch 1233 → 1196 chars, fragments resolved: `"Geschirrtuches umfassen. tuch umfassen"` → `"Geschirrtuches umfassen. umfassen"`, `"irrspülmaschine"` → `"Geschirrspülmaschine"`, `"Gefühl. onär ist"` → `"Gefühl. ist"`. Remaining EN FLEURS artifacts like `"World's Save for You"` duplicating `"world's say for you"` are model-retokenization (different token ids from capitalization across chunks) — out of scope for word-snap, would need case-insensitive LCS or beam-over-chunks. **Still open:** qwen3 / granite / omniasr-llm chunk-context audit — Kaggle harness committed `fe44abd2`, awaits empirical run. |
| **DONE** | [#105 WhisperX word alignment models](#105-whisperx-word-alignment-models-wav2vec2-ctc-zoo) | Phased | **DONE 2026-05-23.** All 10 WhisperX common languages (fr/es/it/ja/zh/nl/uk/pt/ar/cs) converted, uploaded to `cstr/*-GGUF`, registry aliases wired. Only benchmarking + docs remain. |
| **OPTION A DONE** | [#115 mimo-asr baseline broken](#115-mimo-asr-baseline-broken-silent-empty-on-short-segfault-on-long) | Small-Medium | Discovered 2026-05-25 by `tools/check-overlap-save-bug.sh`. `crispasr -m mimo-asr-q4_k.gguf --backend mimo-asr -f samples/jfk.wav` (11 s) runs to exit 0 producing **no output**; on 5 min audio it segfaults at ~159 s. Both pre-overlap and post-overlap configs reproduce — it is **NOT** the overlap-save bug, it's a separate mimo-asr regression from PLAN #72 commit `89111260` (CPU→GPU weight residency change). Bisected via Kaggle Linux x86_64 CPU build (kernel `chr1str/crispasr-mimo-asr-cpu-validate`, JFK PASS) + local M1 Metal control. **Option A shipped `c887881e` 2026-05-26**: force `ctx->backend = ctx->backend_cpu` unconditionally — JFK transcribes correctly in 297 s, matches HISTORY §56 reference verbatim. Loses the 22 % M1 Metal speedup. **Option C still open** — proper GPU graph fix via per-tensor backend tagging in `mimo_asr_build_prefill_graph`. Independently, montvid (Blackwell sm_120) hit a related-but-distinct mimo segfault that was fixed by the sched src-mutation hardening `a5a518c8`; verified that hardening does NOT address the M1 Metal silent-empty (different bug, same backend). |
| **MOSTLY DONE** | [#125 Issue #125 — multi-backend bug sweep from montvid](#125-issue-125--multi-backend-bug-sweep-from-montvid-12-findings) | Medium | External user `montvid` ran every backend on v0.6.10 `eaee2319` on a 50 min EN FLAC + the project's own `samples/jfk.wav`, hardware NVIDIA RTX PRO 6000 Blackwell sm_120. 12 well-attested findings. **P0 mimo-asr CUDA segfault** — bisect reattributed from `6b492b2b` (FA mask, ruled out) to `0f0f0793` (sched src-mutation log without all-exits restore); hardening shipped as `a5a518c8`, awaiting Blackwell retest. **P1 funasr `!`-loop guard + funasr/sensevoice/paraformer registry entries** DONE `f72d3db1`. **P2 firered-asr drop `CAP_UNBOUNDED_INPUT` + length check** DONE `72b74486`. **P3 omniasr-llm chunking gate** DONE `5f0aefc0`. **P4 gemma4-e2b 30s training-window guard** DONE `8bfaff23`. **P5 mimo-asr tokenizer auto-download manifest + docs** DONE `b936b488`. **P6a kyutai-stt silence-tail flush** DONE `ba0e388e`. **P6b kyutai-stt 30s internal chunking** DONE `043b3ae5` (90 s Japanese: 568 s wall = 6.3 s/s, finite + linear vs the previous 14 s/s degradation). **P6c streaming-only design-limit guardrail** DEFERRED (P6b made wallclock predictable, so the cap is now a UX nicety). **Still open**: P0 external Blackwell retest; gemma4-e2b long-audio quality validation under `CRISPASR_GEMMA4_AUTO_CHUNK=1` (env-gated chunking added `9b5a0a2a`). JFK as universal control test is the reporter's #1 methodology contribution. |
| **MEDIUM** | [§130 Zonos TTS](#130-zonos-tts--transformer--dac-codec-apache-20) | Medium | queued — AR transformer reuses orpheus pattern, needs DAC decoder + conditioning system |
| **DONE** | [§131 OuteTTS](#131-outetts--llm--wavtokenizer-codec-cc-by-40) | S-M | **WORKING — speech output confirmed via ASR roundtrip.** WavTokenizer decoder validated cos≥0.999 all stages. 8 bugs fixed (GroupNorm vs LayerNorm, SiLU vs GELU, AdaNorm/pos_net order, iSTFT padding="same", magnitude clipping, newline token, text lowercasing, repetition penalty). Speaker prompt support via `--voice speaker.json`. Model registry + GGUF detection + docs wired. |
| **LOW** | [#127 Coverage gaps from 2026-05-26 sweep close-out](#127-coverage-gaps-from-the-2026-05-26-overlap-save-sweep-close-out) | Small | Three loose ends: (a) omniasr-llm overlap-save status unknown — both A/B passes timed out at 20 min wallclock on M1 even at 90 s clip; needs a faster box. (b) mimo-asr local test coverage in place since `2aeaf4c4` but doesn't run in CI because the 4.2 GB Q4_K doesn't fit the runner disk budget — PLAN #115 shipped despite a working `EMPTY`-detecting test because the test wasn't run pre-tag. (c) `cohere-asr-ja-v0.1` registered + README'd (issue #123) but no row in any of `PERFORMANCE.md`'s cohere tables — JA fine-tune needs the same TedX/JSUT fixture sweep the English one had. |

**Recently completed** (full write-ups in HISTORY.md): **Issue #89 reopened — parakeet streamed-encode is now the default → HISTORY 2026-05-24** (lenhone's `yt-dlp` clip reproduced 33 % coverage where the cached MP3 derivation gave 99.5 %; same TDT model collapses on the bad audio in NeMo's stock `transcribe()` too; encoder is bit-for-bit to NeMo via the diff harness; root cause is model-level TDT-single-pass instability that bidirectional attention amplifies past ~20 s; `33f9a162` makes the streamed path the default for any duration). **#81 FA per-head additive mask → HISTORY 2026-05-24** (CUDA MMA-F16 kernel patch +87 LOC behind `GGML_CUDA_CRISPASR_FA_PERHEAD_MASK` default-OFF; byte-identical JFK transcript, 0 CPU FA splits, -37 % short-clip on A1000; `tools/upstream-prs/06-cuda-fa-perhead-mask.md` + `872303bf` write-up). **CI cleanup → HISTORY 2026-05-25** (test #148 catch_discover_tests CLI-parser fix `4fda4be5`; build.yml trimmed 1610 → 1324 lines and arm64 switched to native runners `80ac00d1`; `GG_BUILD_NO_AVX512` knob added to `ci/run.sh` and enabled on `ggml-ci-x64-cpu-high-perf` `565b16af` so the AVX512 SIGILL is structurally fixed instead of `continue-on-error`-papered; `tools/upstream-prs/13-ci-no-avx512-knob.{md,patch}` for upstream submission). **#110 Global diarization timeline → HISTORY 2026-05-23** (sherpa/ecapa runs once on full audio; `CrispasrSherpaCache` mirrors pyannote pattern; segment splitting at speaker turns; 21 tests). **#98 Hotwords A+B → HISTORY 2026-05-23** (CTC-WS Aho-Corasick trie for parakeet CTC/TDT; LLM prompt injection for qwen3-asr/voxtral; `--hotwords` CLI; 17 tests). **Paraformer-zh NAR-ASR → HISTORY 2026-05-21** (220M params, single-pass NAR decode; F16/Q4_K/Q8_0 at `cstr/paraformer-zh-GGUF`; byte-identical on Chinese + English; 4 integration tests). **#86 Flash-attn → DONE** (all backends already wired via core helpers). **#90 Session beam_size all backends → HISTORY 2026-05-23** (qwen3-asr, granite, voxtral wired via `core_beam_decode::run_with_probs`; commit `0c24178e`). **#74 Feature-matrix uplift round 2 → HISTORY 2026-05-23** (74a chatterbox lang routing, 74b cap regression tests, 74c qwen3-tts base voice-cloning cap, 74d matrix regen; commit `b848152a`). **#111 TTS `--seed` parity → HISTORY 2026-05-23** (qwen3-tts, chatterbox, vibevoice realtime/base all show same-seed reproducibility and different-seed divergence on the local backup models; qwen3 env precedence fixed so CLI/request seed wins; IndexTTS stays effectively deterministic on the tested prompt/reference). **#99 funasr MLT-Nano hallucination fix → HISTORY 2026-05-21** (root cause: `use_low_frame_rate` hardcoded true in C++, but MLT-Nano's upstream config omits it (default false) — only 23/183 adaptor frames were spliced into the LLM prompt, truncating 87% of audio context; fix: converter reads the flag from config.yaml into a GGUF KV, runtime reads it at load time; also fixed `ada_n_heads` 16→8 in converter; GGUFs re-uploaded to `cstr/funasr-{nano,mlt-nano}-GGUF`). **SenseVoiceSmall → HISTORY 2026-05-20** (encoder-only multi-task ASR: transcript + LID + emotion + audio-event in one CTC pass; 50+ langs; 9.8-21.8× realtime on M1 Metal; reuses the SANM block helper from the funasr port unchanged; `cstr/sensevoice-small-GGUF` 0.47 GB F16, wired into `-m auto`). **Fun-ASR-Nano + MLT-Nano → HISTORY 2026-05-20** (full LLM-decoder runtime — 70-block SANM encoder + 2-block Transformer adaptor + Qwen3-0.6B AR decode; 77/77 PASS byte-identical on Chinese + English diffs; ~9× realtime on M1 Metal with FA-default-on; both GGUFs at `cstr/funasr-{nano,mlt-nano}-GGUF`). **#57 chatterbox native voice clone → §82** (six-commit sprint shipping all four upstream cond extractors — VoiceEncoder LSTM, S3Tokenizer V2, CAMPPlus, 24 kHz Matcha mel — plus a Kaiser-windowed sinc resampler and atomic 5-cond install in `chatterbox_set_voice_from_wav`'s `.wav` branch; `--voice ref_24k.wav` produces real cloned speech without any python). **#69 + #72 + #73 cap-honesty + KV/layer offload knobs → §79** (14-commit session shipping `CRISPASR_KV_QUANT_K/_V` + `KV_ON_CPU` on 14 backends, `N_GPU_LAYERS` on 10 backends, gemma4/mimo GPU-residency 2.2x / 22 % faster, plus cap-honesty cleanup on parakeet/glm-asr/qwen3/gemma4/omniasr). **vibevoice #69a follow-up → §79b** (mode-aware `tts_lm.layers.` / `lm.layers.` prefix predicate). #78 Chatterbox vocoder → §78. #11 WebSocket server → §76, #63 Feature matrix parity → §72, #59 binding parity → §73, gemma4 #49 + Docker #31 → §74, tests + KV Q8_0 + cleanup → §75. Earlier: #5→§63, #16→§55, #51→§56, #51b→§60, #53→§63, #54→§61, #55→§54, #56→§63, #60d→§64.

**Open follow-ups from §79 — we want all of these:**
- **#73 cohere long-form rerun.** flash_attn_ext is shipped on canary + cohere (commit 193a736). JFK (~11 s) numbers: canary q8_0/q4_0 -17 % under flash (win), but cohere q8_0/q4_0 is +11 % under flash vs cast-on-read on the same workload. F16 is a tie on both. Before promoting flash as cohere's recommended path, validate on a multi-minute clip — if the crossover is workload-dependent the docs need to recommend cast-on-read for short audio and flash for long. Until then PERFORMANCE.md notes flash as available-but-regresses-on-JFK for cohere.
- **#72 Linux/CUDA validation** of the gemma4_e2b / mimo_asr GPU-residency flip. Hardware-blocked from the dev host; expect even larger wins on dGPU than the 22 %–220 % observed on Apple Silicon Metal.
- **encoder-decoder #69a** (canary, cohere, kyutai-stt). Cross-attention layout has no `<prefix><N>.*` block-tagged tensors; needs bespoke per-backend predicates. Own design problem.

**Issue #81 A1000 work — Phase 1 verdict in (2026-05-23):**
- `d758fe69` (fused `GGML_OP_NORM_AFFINE` + `GGML_GLU_OP_SIGLU` for FastConformer encoder) closes target (b) of the gap analysis. Measured **+5.7 % wallclock win** on A1000 Laptop (2.701 s vs 2.863 s baseline, p50/chunk 175.8 ms vs 184.8 ms, RTx 22.2× vs 21.0× — clean WDDM-warm conditions on Studio Driver 596.36). Sched-debug: CPU splits 144→72, UNARY-on-CPU 72→0. **Carry as permanent improvement.** Full write-up in PERFORMANCE.md "Phase 1 update (2026-05-23)" subsection. WIP branch `issue81-phase1-uar-wip` (commits `6a0ccc67 / a2999cf3 / 6d7872a0`) is superseded — delete when convenient.
- **#06 FA per-head mask** is the next concrete A1000 perf step. Removes the other 72 CPU splits per chunk (per-head additive mask in `fattn.cu:423` + the four kernel variants). Scoped at 2-3 days, ~300-500 LOC across `fattn.cu` / `fattn-common.cuh` / `fattn-mma-f16.cuh` (and optionally `-wmma-f16.cu` / `-tile.cu` / `-vec.cuh`). Expected wallclock gain ~10-15 % on top of postsiglu (target ~2.4 s long-clip / RTx ~25× / ~1.5× behind onnx-fp32). Don't start until WDDM-warm bench protocol below is followed for the new baseline.
- **WDDM warm-up protocol** (Windows/laptop NVIDIA only): cold A1000 sits at P5/P8/210-510 MHz during compute and runs 8-10× slower than warm; engage WDDM by running `bench-issue81/probe_postsiglu_leak.py <dll> 200` (or ~10 s of `gpu_keepalive.py`) BEFORE measuring. The 3.063 s May 11 reference is reproducible with this protocol; single-shot cold benches are noise. Documented in PERFORMANCE.md "What we learned about A1000 WDDM behavior" + LEARNINGS.md "WDDM idle-clock-state hysteresis on consumer/laptop NVIDIA SKUs".

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
- **`nvidia/parakeet-unified-en-0.6b`** — recent "unified" variant; needs a model-card / architecture read before scoping.

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

**BLOCKED:** Needs ≥16 GB RAM for conversion. Converter OOMs on 8 GB due
to Qwen2.5-7B embedding (152064 × 3584 = 2.1 GB F32).

**Fix:** Use `safe_open` per-tensor conversion. Then Q4_K → ~4 GB.

Full architecture analysis in HISTORY.md #34. C++ runtime partially
implemented (`src/vibevoice.cpp`). F16 im2col precision issue in
depthwise conv needs fixing.

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

3. **Fused LLM QKV.** Pattern from `qwen3_asr.cpp`: concat the per-block
   Q/K/V weights along the output axis at load time (byte-concat for
   F16/Q4_K — no requantization needed) and submit one matmul per
   layer instead of three. qwen3_asr opts in via
   `CRISPASR_QWEN3_ASR_FUSED_QKV`; funasr would mirror with
   `CRISPASR_FUNASR_FUSED_QKV`. Expected savings: 5-10 % on decode.

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

## CosyVoice3-0.5B-2512 TTS — DONE

Phases 1–5 → HISTORY. **Phase 6** (native arbitrary-WAV cloning:
speech_tokenizer_v3 + CAMPPlus + matcha-mel ggml ports,
`--voice ref.wav --ref-text "..."`) → HISTORY
§"2026-05-29 cosyvoice3 Phase 6". s3tokenizer_v3 is byte-exact vs the
ONNX reference (crispasr-diff `s3tok_tokens` max_abs=0). Open follow-up:
multilingual voice bank (bake ~10 en/de/zh/ja voices into
`cosyvoice3-voices.gguf`); the runtime mel has a sub-1% STFT delta vs
whisper (2/264 token flips, cloning-irrelevant) if ever worth chasing.

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

## 96. voxcpm2-tts perf — switch to per-step ggml graph (Metal-ready) — Metal live

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
- Once the above is investigated (or accepted as inherent), flip
  default to `VOXCPM2_USE_GRAPH=1`.

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

Repaired GGUF refresh (2026-05-08):
- **Base Chatterbox:** regenerated `chatterbox-t3-f16-regen.gguf` because the old base T3 export lacked the real HF BPE tokenizer metadata and made C++ feed the wrong text-token sequence. The repaired F16/Q8_0/Q4_K have been uploaded to HF under canonical `chatterbox-t3-*` names.
- **Chatterbox Turbo:** regenerated `chatterbox-turbo-s3gen-f16-regen.gguf` because the active breakage was in the downstream S3Gen/vocoder companion, while `chatterbox-turbo-t3-f16.gguf` was not the artifact under repair. The repaired S3Gen F16/Q8_0/Q4_K have been uploaded to HF under canonical `chatterbox-turbo-s3gen-*` names.
- **Turbo T3 quant coverage:** `chatterbox-turbo-t3-f16.gguf` did not need regeneration, but Q8_0/Q4_K were produced from the canonical F16 in `/Volumes/backups/ai/crispasr/` and uploaded as `chatterbox-turbo-t3-q8_0.gguf` / `chatterbox-turbo-t3-q4_k.gguf` so explicit quant selection and auto-resolve have matching T3 files.
- **HF publication:** keep `-regen` locally for traceability where a file was repaired, but upload/replace under canonical filenames by stripping `-regen` so registry auto-resolve and auto-download keep using the existing names.
- Current runtime status: T3 tokenizer, conditioning, prefill, CFG, step-0 logits, forced step-1 logits, S3Gen replay, and HiFT replay are matched against Python; remaining nondeterministic mismatch is isolated to CPU `torch.multinomial` sampler parity after logits are already correct.

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
| qwen3-asr session-API beam | DONE — `run_with_probs` replay in `transcribe_single` (commit 0c24178e) |
| granite* session-API beam | DONE — `run_with_probs` replay in `transcribe_single` (commit 0c24178e) |
| voxtral session-API beam | DONE — `run_with_probs` via `run_voxtral_family` (commit 0c24178e) |
| voxtral4b | ❌ N/A — streaming path, no beam hook |
| canary beam | DONE — `run_with_probs_branched` (KV snap of kv_k/kv_v; cross-KV shared) |
| cohere beam | DONE — `run_with_probs_branched` (KV snap of kv_k/kv_v; cross-KV shared) |

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

**All backends now wired.** canary and cohere use
`run_with_probs_branched` with per-backend KV save/restore callbacks
(snapshot monolithic `kv_k`/`kv_v` tensors; cross-attention KV shared
across beams). voxtral4b stays out of scope (streaming API, no beam hook).

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

## 73-follow-up. Long-context cohere FA vs cast-on-read benchmark — open

Parent #73 (quant-safe KV cache write for canary / cohere / kyutai_stt)
shipped → HISTORY §79. #71 + #72 also there (test-runner under-invocation
+ cap-honesty audit; gemma4_e2b / mimo_asr GPU residency for Q4_K weights
— gemma4 2.2× on M1, mimo-asr -22 %, Linux/CUDA validation deferred).
Only residual: long-context perf comparison of cohere flash-attn vs
cast-on-read — JFK is too short to surface the long-context win.

## 74. Feature-matrix uplift round 2 — chatterbox family + matrix tooling ✓

After §79b shipped chatterbox + 3 sibling variants and the audit-drift cleanup brought test-all-backends.py to 39/39 backends, four follow-ups surfaced from re-reading the cap matrix. They cluster by user-visible value:

### 74a. Auto-route by `-l <lang>` for chatterbox family — DONE

`-l de` with `--backend chatterbox` now auto-routes to `kartoffelbox-turbo`; `-l ar` goes to `lahgtna-chatterbox`. Implemented in `examples/cli/crispasr_run.cpp` (the PLAN #74a block at the model-resolution step), mirroring the kokoro `-l de` routing pattern. Only fires when `-m auto` is in effect so an explicit model path is never overridden.

### 74b. CAP_TRANSLATE / CAP_SRC_TGT_LANGUAGE capability regression gate — DONE

`tests/test_backend_caps.py` (new file) runs `crispasr --list-backends-json` and asserts that the known translate / src-tgt-language / voice-cloning backends declare exactly the right caps, and that preset-speaker backends (qwen3-tts-customvoice, voicedesign) do NOT declare voice-cloning. Also includes a live translate smoke-test (whisper + samples/jfk.wav) that runs when the model is present. 6/6 tests pass.

### 74c. CAP_VOICE_CLONING — DONE

`qwen3-tts` (base 0.6B) and `qwen3-tts-1.7b-base` now declare `CAP_VOICE_CLONING`. Implementation: `Qwen3TtsBackend` gained an `is_base_` flag set at construction; `crispasr_backend.cpp` dispatches the two base aliases to `crispasr_make_qwen3_tts_base_backend()` and the customvoice/voicedesign aliases to the original factory. Chatterbox and vibevoice-1.5b already had the cap. Feature matrix regenerated (`tools/gen-feature-matrix.py`).

### 74d. Generated sortable/filterable feature matrix — DONE

`tools/gen-feature-matrix.py` already existed and was regenerated after 74c. `docs/feature-matrix.md` and `docs/feature-matrix.html` are the single-source-of-truth artifacts; 52 backends × 19 caps.

### 74e. Beam search for chatterbox T3 — TIER 3 (deferred)

Chatterbox T3 AR decode currently uses sampling (temp / top_p / min_p / repetition_penalty). Adding beam search would unlock `--beam-size N` for the backend and add a row to the feature matrix. Honest scope: ~200-300 LOC for parallel decode paths + length-normalised score accumulation + KV cache replication-or-sequential-with-separate-caches + early-termination handling. Plus validation that beam decode doesn't amplify the open Conformer rel-pos parity gap (matrix_bd 7.08 vs 10.06).

**Blocker:** the rel-pos parity gap dominates output quality today; beam search would be polish on a runtime that still has unresolved structural issues. Defer until parity closes — at that point beam search is an obvious win, but not before. Tracking here so it doesn't get lost.

---

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

`src/core/audio_chunking.h` — two functions: `find_energy_min_split` and
`split_at_energy_minima`. Wired into `src/cohere.cpp` (replaces fixed
`30 * sample_rate` cut). JFK single-chunk path unaffected.

### 80c. CRISPASR_VERBOSE env override for cohere CLI (DONE this session)

Aligns cohere with the gemma4 / granite / firered / omniasr /
moonshine_streaming convention (env var or `-v` bumps backend
verbosity to 2 to print the perf report). One line in
`examples/cli/crispasr_backend_cohere.cpp`. Side benefit of the
investigation; without it, 80a couldn't be measured from the CLI.

### 80d. Cross-backend audit of fixed-time chunking — DONE 2026-05-23

Audited all 13 AR-decoder backends. **No fixes needed.** All backends
rely on the global slicer (`crispasr_energy_chunk_slices`) which already
uses `split_at_energy_minima`. Cohere's public API path (line 2084)
explicitly calls `split_at_energy_minima` too. The only fixed-time loop
is cohere's internal encoder KV scatter (line 2186), which operates on
encoder frames not raw audio — not a candidate for energy chunking.

### 80e. Eager warmup follow-up — DONE 2026-05-23

Implemented as virtual `warmup()` on `CrispasrBackend`. Transcribes
0.5 s of silence to amortize first-call overhead. Overridden for
parakeet, canary, and cohere. Opt-in via `--warmup` or
`CRISPASR_WARMUP=1`. Server mode always warms up. A/B on CPU shows
no benefit (no GPU kernels to compile); on Metal/CUDA saves 100-200 ms
on first call.

### Out of scope (rejected ideas from nano)

* **Meta-init + `assign=True`** — PyTorch trick. CrispASR's mmap'd
  GGUF is already the moral equivalent.
* **bf16 autoselect** — backend dtype is controlled by the GGUF
  quant choice, not a runtime switch.
* **Decoder-batched chunk packing** with longest-first sort — the
  CrispASR cohere path already encodes-then-decodes per-chunk
  serially (line 1834-1899 recurses); converting to batched chunks
  needs a different graph shape and a real-world workload that hits
  this regime (multi-file CLI calls). Investigate when the demand
  appears.

---

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
2. **Rel-pos shift on `[cache | new]`.** The existing
   `core_conformer::rel_shift` operates on a square `(2T-1, T)`
   tensor; with cached frames the table is `(2(L+R+1)-1, R+1)` —
   different layout, different shift indexing. Either extend
   `rel_shift` with a left-context offset or build a streaming-
   specific variant.
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

**Not yet.** Two reasons to wait:

1. The streaming pipeline only just got `--stream-json` (issue #84)
   and the rolling-buffer fix; we want a few weeks of real wrappers
   building on it before we add a backend that's specifically
   tailored to it. If the structured-output API needs to evolve
   (e.g., adding word-level probability streams), better to find
   that out before nemotron locks in expectations.
2. The user demand is one outside-reporter mention so far (issue
   #85). Worth waiting until either a second user asks or a
   concrete production use case (live captioning, voice agent)
   needs sub-200 ms ASR. The 6.93 % batch number isn't an
   improvement over what we ship — only the streaming-native
   property is.

When demand materializes: **realistic estimate 3–5 days of focused
work** for someone who's already touched parakeet/canary, plus
benchmarking against the upstream Open ASR Leaderboard table to
confirm parity on the published WERs.

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

**Status:** seed shipped (`parakeet-tdt-0.6b-ja` end-to-end);
expand by adding manifest entries + uploading reference dumps.

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

**Status:** open, low-risk mechanical rename. Recommended after
the next release cycle so external consumers can absorb it.

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
     but any `--target crispasr` referring to the library needs
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




## 94. Auto-generate Go bindings `#cgo LDFLAGS` from CMake graphviz

**Status:** open, ~half-day. Recommended before the v0.7.0 cycle.

**Why:** the hand-maintained `#cgo LDFLAGS` in `bindings/go/whisper.go`
has now bitten three releases in a row — v0.6.3 (`-ltitanet`
missing → commit `acda0622`), v0.6.4 implicitly (would have failed
if the Rust translate path tested), and v0.6.4 actually
(`-ltext-lid-dispatch -llid-fasttext -llid-cld3 -lindextts
-lt5_translate` missing → Bindings Tests (Go) #82). The fail mode
is repetitive and the fix is mechanical, which is exactly the
shape that should be automated.

The Ruby binding already solved this — `bindings/ruby/ext/dependencies.rb`
asks CMake for a `--graphviz` dependency dot, walks the graph
reachable from the `crispasr` target, and emits the right
`-l<name>.a` list. Result: every new `target_link_libraries(crispasr
PUBLIC <X>)` in `src/CMakeLists.txt` propagates to Ruby consumers
with zero edits on the binding side.

**Plan:**

1. Move the graphviz-walk logic out of `bindings/ruby/ext/`
   into a shared `tools/cmake_graphviz_targets.py` (~80 LOC).
   Takes a target name + dot path; returns the topologically-
   sorted list of static-lib targets reachable from it.
2. New step in the Go binding's build pipeline: before
   `go build`, run
   `cmake --graphviz=/tmp/crispasr.dot -S . -B build_go ...`
   and `python tools/cmake_graphviz_targets.py crispasr
   /tmp/crispasr.dot > bindings/go/cgo_libs.txt`. The .txt
   becomes a generated artifact; `whisper.go` reads it via
   `//go:embed cgo_libs.txt` and a small init() that splits
   it into the `#cgo LDFLAGS` slice at build time.
3. Drawback: `//go:embed` doesn't compose with `#cgo
   LDFLAGS:` directives — those are evaluated at the cgo
   preprocessing stage, before any Go code runs. So we'd
   actually need to **generate `whisper.go`'s `#cgo` block
   from the .txt at build time**, or use a separate
   `bindings/go/cgo_ldflags.go` with `#cgo LDFLAGS: ${ldflags}`
   and have a `go generate` step that writes it. Either is
   ~30 LOC + a Makefile/CMake target.
4. Mirror the same generated list in `bindings/go/whisper.go`'s
   darwin `#cgo` line.
5. CI: add a check that detects drift — if the hand-edited
   `whisper.go` `#cgo LDFLAGS` doesn't match
   `cmake_graphviz_targets.py crispasr` output, fail the lint
   job with a clear message. Best of both worlds: humans can
   still edit if needed but drift is loud.

**Why not just always run codegen instead of letting humans edit:**
Go's tooling makes generated source files awkward (e.g., `go
fmt` may not run on them, IDE jump-to-symbol may miss them).
Keeping a hand-maintained file with a drift-check is the
least-intrusive path for Go consumers who pull via `go get`.

**Tracking:** every commit message that fixes the next round
of "missing -lX" should reference this section and cross-check
whether the auto-gen is finally in place.

---

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

## 100. MeloTTS + OpenVoice2 — multilingual TTS with native CJK + voice cloning

Surveyed via RapidAI/RapidSpeech.cpp ("OpenVoice2: MeloTTS + voice
cloning") and the upstream `myshell-ai/MeloTTS` + `myshell-ai/OpenVoice`
repos. Both **MIT-licensed**.

**Why it matters for CrispASR:** our current TTS lineup covers
European languages well (Kokoro EN/ES/FR/HI/IT/PT/DE; VibeVoice EN/ZH;
qwen3-tts multilingual; voxcpm2 30 langs) but the *native CJK
pronunciation quality* gap is real — Kokoro's ZH/JA voicepacks
are weaker than purpose-built CJK TTS (informal A/B from
HISTORY: Kokoro ZH was the motivator for the IndexTTS port). MeloTTS
ships **EN / ES / FR / ZH / JA / KO** as first-class targets, with
proper g2p (Mandarin tones, kanji readings, Korean hangul). OpenVoice2
adds zero-shot voice cloning on top — the cloner is a separate
~30M tone-color converter run as a post-step on MeloTTS output.

### Phase A — MeloTTS standalone (no cloning)

**Architecture** (from upstream MeloTTS):
- VITS-style: text encoder + flow + HiFi-GAN-style decoder
- Per-language g2p: pypinyin (zh), pyopenjtalk (ja), g2pkk (ko),
  English/Spanish/French via espeak-ng (we already vendor espeak-ng
  for kokoro #56, so that path is free)
- ~70M params; F16 ≈ 140 MB, Q4_K ≈ 50 MB per language pack
- 44.1 kHz output

**Ports needed:**
- New backend `melotts`; new converter `models/convert-melotts-to-gguf.py`
- Per-language g2p: zh + ja + ko need new code. zh and ja are blocked
  by the same issue #95 (text normalization, but pure g2p — not
  WFST-scale). Lightweight: vendor `cn2an` rules + a pinyin lookup
  table for zh; ja kanji-reading needs pyopenjtalk or a stripped-down
  C port (out of scope for first cut — start with **hiragana-only ja**
  and document the limit).
- VITS forward fits in existing primitives — text encoder is a
  shallow transformer, flow uses a few coupling layers we don't yet
  have but they're trivial (affine coupling). HiFi-GAN-style decoder
  reuses what we built for chatterbox-S3Gen / indextts BigVGAN.

**Effort:** Medium per language. EN/ES/FR ride free off espeak-ng;
ZH ≈ 1 week; JA/KO ≈ 1-2 weeks each because of g2p.

### Phase B — OpenVoice2 tone-color cloning

**Architecture:** Separate `tone_color_converter.pth` (~30M) that
takes a MeloTTS mel + a reference audio's tone-color embedding and
warps the mel toward the reference voice. Extracted speaker
embedding pipeline is similar to chatterbox-VoiceEncoder (already
ported, `src/chatterbox_campplus.h`).

**Ports needed:**
- Reuse VoiceEncoder LSTM from chatterbox (HISTORY §82 sprint)
- New small `tone_color_converter` graph — couple of conv blocks +
  flow. Q4_K ≈ 15 MB.
- Plumb `--voice <ref.wav>` through the melotts backend, run TCC as
  a post-process before final iSTFT / vocoder pass.

**Effort:** Small once Phase A is in — the chatterbox VoiceEncoder
port did the hardest part.

### Why this isn't redundant with our existing stack

| Capability | Kokoro | qwen3-tts | voxcpm2 | indextts | melotts+openvoice2 |
|---|:-:|:-:|:-:|:-:|:-:|
| Native Mandarin g2p (pypinyin-equivalent) | ⚠ tones rough | ✔ | ✔ | ✔ | **✔** |
| Native Japanese g2p (kanji) | ✗ | partial | ✔ | ✗ | **partial** (hiragana only at first) |
| Native Korean g2p | ✗ | ✔ | ✔ | ✗ | **✔** |
| Voice cloning | per-voicepack | ICL ref | ✔ | ✔ | **✔ (TCC)** |
| Model size for one language | ~75 MB | ~500 MB | ~700 MB | ~1.2 GB | **~50 MB Q4_K** |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 | **MIT** |

The size column is the unlock: MeloTTS Q4_K is **~10× smaller** than
qwen3-tts and **~25× smaller** than voxcpm2, which puts CJK-quality
TTS in reach on the HF free tier (#hf-space) and on mobile via
CrisperWeaver (#86 / #87).

### Triggers

- A user requests Japanese / Korean TTS — current options are
  qwen3-tts (heavy, ICL only) or nothing.
- HF Space mobile demo needs sub-100 MB CJK TTS (#hf-space).

Don't start until at least one trigger fires — current TTS coverage
is broad and this is opt-in coverage depth.

---

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

## 103. Silero VAD version bump — verify and align with v6

RapidSpeech.cpp documents shipping with Silero VAD **v6**. CrispASR
ships Silero as the default VAD (`--vad`, auto-downloaded ~885 KB) —
need to confirm which version is in the registry and bump if older.

### Concrete steps

1. **Identify current pinned version.** Find the Silero VAD URL in
   `src/crispasr_model_registry.cpp` (registry key likely `silero-vad`
   or under the moonshine/whisper companion-file lists). Note the
   upstream commit / `silero-vad/silero_vad.onnx` filename revision.
2. **Compare to upstream.** Check
   [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
   releases — v6 was released early 2026 and improves on v5
   short-segment latency + adds multilingual robustness fixes.
3. **Convert + diff.** If v6 weights aren't already in our GGUF
   form, port the upstream PyTorch state dict to GGUF following
   the existing Silero VAD converter (path: `models/convert-silero-vad-to-gguf.py`
   if it exists, otherwise mirror the FireRedVAD converter shape).
4. **A/B on a multi-minute clip.** Compare segment boundaries from
   v5 vs v6 on `samples/jfk.wav` and a longer real-world recording.
   Expect a ~10-30 ms improvement on segment-end alignment and
   slightly fewer false positives on breath noises.
5. **Bump registry URL** in `src/crispasr_model_registry.cpp` to
   point at the new GGUF; preserve the v5 download as a fallback
   alias so existing user caches don't redownload.

### Effort

Small (~half day) **if** v6 PyTorch weights are openly downloadable
(they are — Silero is MIT). Marginal if the existing pinned version
is already v6.

### Trigger

- Any session that touches VAD code can do this opportunistically.
  No standalone trigger needed.

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
| **gemma4_e2b** | no `CAP_UNBOUNDED_INPUT` | CLI auto-chunk | TBD | n/a (Google upstream) |
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

**Status (2026-05-26):** option A shipped, option C still open.

The smoking-gun commit is `89111260` ("perf #72: load weights to GPU when use_gpu=true"), which flipped `core_gguf::load_weights(..., ctx->backend_cpu, ...)` to `..., ctx->backend, ...`. The same commit message foresaw the regression — *"If a platform regresses, add a CRISPASR_FORCE_CPU_WEIGHTS=1 escape hatch — none seen yet"*.

Worth noting that the sched src-mutation-log hardening (`a5a518c8`/`95d74455`, montvid's Blackwell fix) does NOT address this — verified empirically on M1 Metal post-rebuild: same silent-empty. Two different mimo-asr bugs with the same observable symptom on different platforms.

**Option A shipped:**

1. `5a570b7b` — first pass: just pinned weights to `ctx->backend_cpu`. Insufficient — exposed a second failure mode where Metal compute can't resolve CPU-resident weight buffers (`ggml_metal_buffer_get_id: error: tensor 'llm.embed.weight' buffer is nil`). The §56 working configuration (CPU weights + Metal compute) no longer holds because the ggml scheduler tightened cross-backend tensor resolution since then.
2. `c887881e` — complete fix: force `ctx->backend = ctx->backend_cpu` unconditionally, ignore `params.use_gpu`. Verified on M1 Metal locally: JFK transcribes correctly in 297 s, matches HISTORY §56 reference verbatim. (Slow — pure CPU LLM on M1 — but correct.) Kaggle Linux x86_64 CPU build also verified passing on `b85698670`: `prefill 15.8 s, decode 7.0 s over 26 steps, total_lm 22.8 s`.

Cost of option A: loses the documented 22 % M1 Metal speedup from PLAN #72. Acceptable until C lands because the alternative is shipping a backend that produces no output.

**Option C (open):** proper GPU graph fix. mimo's `mimo_asr_build_prefill_graph` doesn't emit per-tensor backend tagging that current `ggml_backend_sched` needs to route weight reads from a CPU buffer through to a Metal/CUDA compute path. Two sub-options:
  - **C1.** Tag the embed weight (and any other CPU-resident tensors) for the appropriate backend before graph build, so sched can insert the needed copy nodes. Cheaper.
  - **C2.** Build the whole prefill graph on the user-selected backend with weights resident there too, and find what actually breaks the prefill emission (similar shape to chatterbox Bug B from issue #83 — see [[project_chatterbox_gpu_bug_s3gen]] — which took ten candidate hypotheses to land on the real cause). Restores the 22 % speedup. Higher value, requires Kaggle GPU run with the patched binary (currently quota-blocked).

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
3. **Honour `-l` on the funasr path.** The CLI's `params.language` doesn't reach `funasr.cpp`'s `PROMPT_PREFIX` (it's a `static const char*`, not a function of lang). Per-variant prompt selection + the `cstr/funasr-mlt-nano-GGUF` having an English/neutral prefix.
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

**Shipped:** defensive 30 s training-window guard in `crispasr_backend_gemma4_e2b.cpp::transcribe()`. Inputs > 30 s abort by default with a clear error message ("input is N s (> 30 s training window). Use --vad to segment, chunk externally, or set CRISPASR_GEMMA4_AUTO_CHUNK=1 to chunk internally"). This stops the symptom — silently-wrong LLM commentary on long audio — and forces the user to route via the segmenter instead. **`CRISPASR_GEMMA4_AUTO_CHUNK=1`** opt-in (`9b5a0a2a`) chunks at 30 s boundaries internally with the same `t_offset_cs` arithmetic as kyutai-stt P6b; off by default because we haven't validated quality at chunk boundaries on long gemma4-e2b output. Local M1 Metal smoke: JFK still produces "ANd so my fellow Americans ask not what your country can do for you, ask what you can do for your country..". **Still open (longer-term):** long-audio quality validation, then promote to default chunking. Audit the prompt-wiring sanity logs the reporter originally asked for (`audio_soft_token_id`, `proj_dim` vs `d_model`).



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

**Completed 2026-05-30.** Full end-to-end pipeline working with speaker-conditioned voice cloning.

**Files:**
- `models/convert-outetts-to-gguf.py` — OLMo safetensors → GGUF (113 tensors)
- `models/convert-wavtokenizer-to-gguf.py` — WavTokenizer decoder → GGUF (162 tensors)
- `src/outetts.cpp` + `src/outetts.h` — LLM runtime + speaker prompt + C ABI
- `src/outetts_wavtok.cpp` + `src/outetts_wavtok.h` — WavTokenizer decoder (validated cos≥0.999)
- `examples/cli/crispasr_backend_outetts.cpp` — CLI adapter
- `tools/reference_backends/outetts_create_speaker.py` — speaker profile creation
- `tools/reference_backends/outetts_wavtok_diff.py` — per-stage diff harness

**Bugs found (8 total):**
1. pos_net uses **GroupNorm(32) + SiLU**, not LayerNorm + GELU
2. pos_net comes **before** AdaNorm, not after (VocosBackbone.forward order)
3. ISTFTHead clips `exp(mag)` to max 1e2 (magnitude safeguard)
4. iSTFT uses `padding="same"` trim `(win-hop)/2=480`, not `center=True`'s `n_fft/2=640`
5. iSTFT needs direct inverse RFFT for non-power-of-2 n_fft=1280
6. Newline token is `Ċ` (U+010A) in GPT-NeoX byte-level BPE
7. Text must be lowercased (model trained on lowercase)
8. All norms use eps=1e-6, not 1e-5

**Voice cloning:** `--voice speaker.json` where JSON is created by
`tools/reference_backends/outetts_create_speaker.py` (WavTokenizer encoder +
whisper word timestamps → per-word codes + durations).

---

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

