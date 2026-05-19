# CrispASR — Pending work

Pending roadmap items. Each is self-contained with files, approach, and
effort estimate. Completed items have been moved to `HISTORY.md`.

**Current state (May 2026, v0.6.2):** 20 ASR + 3 TTS + 1 speaker-verification backends (+ Chatterbox T3 in progress), unified CLI,
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
| **HIGH** | [#57 Commercial-friendly TTS expansion](#57-commercial-friendly-tts-backend-expansion) | Phased | Phases 1-3 DONE; Turbo WORKING; F0 wired in; native voice cloning shipped → HISTORY §82; **#83 GPU-Metal drift re-bisected 2026-05-19 → HISTORY 2026-05-19** — T3 round-4 kernel fixes (Q4_K×Q8_K + `_hp` mul_mm + PREC_F32) clean T3 on GPU; the remaining user-audible bug lives in S3Gen UNet1D compound Metal precision drift across mul_mat/FA/norm/add/gelu/tanh/softplus, not a single op (op-bisect via `CRISPASR_S3GEN_UNET_PIN_CPU_OP=<op>`). Metal default flipped to full CPU (T3 launch overhead × 86 AR steps makes M1 GPU 1.5× slower than CPU); `CRISPASR_CHATTERBOX_T3_GPU=1` opts back in. Remaining: deep ggml-metal precision audit for the S3Gen UNet kernels (or wait for M3/M4 + CUDA bisect), Kartoffelbox_Turbo DE |
| **MEDIUM** | [#51c MiMo-V2.5-ASR F16 step decode](#51c-f16-step-decode) | Small | F16 step-decode validation blocked behind ≥32 GB box (see PLAN #51c); base runtime + Q4_K shipped → HISTORY §56 |
| **MEDIUM** | [#56 Kokoro multilingual phonemizer](#56-kokoro-multilingual-phonemizer-espeak-ng) | Small | espeak-ng + DE backbone shipped; HF GGUFs published 2026-05-01; auto-download wired; only Mandarin tones / JA kanji + diff-harness phonemizer-step polish remain |
| **MEDIUM** | [#58 MOSS-Audio-4B-Instruct](#58-moss-audio-4b-instruct) | Large | first audio-understanding (not just ASR) backend; introduces DeepStack cross-layer feature injection |
| **MEDIUM** | [#59 Cross-binding C-ABI parity](#59-cross-binding-c-abi-parity) | Medium | Go now has full surface (✅ all 11 capabilities). Java has transcribe+align+LID. Ruby has transcribe. JS needs WebAssembly approach |
| **PARKED** | [#9 Parakeet TDT GPU](#9-parakeet-tdt-decoder-gpu) | Medium | Encoder 85%+ of time; LSTM+joint <0.7s; sequential steps limit GPU benefit |
| **BLOCKED** | [#42 VibeVoice-ASR 7B](#42-vibevoice-asr-7b) | High | Needs ≥16 GB RAM |
| **BLOCKED** | [#43 Fun-ASR-Nano](#43-fun-asr-nano) | Medium | License unclear |
| **MEDIUM** | [#80 nano-cohere-transcribe-inspired tweaks](#80-nano-cohere-transcribe-inspired-perf--chunking-tweaks) | Small | 80c done; 80b energy chunker in progress; 80a parked (measurement: <1 % of wall on Metal); 80d/80e TODO |
| **DEFERRED** | [#81 Nemotron-Speech-Streaming-EN-0.6B](#81-nemotron-speech-streaming-en-06b--first-cache-aware-streaming-native-asr) | M-L | NVOML license, ~60–75 % reuse from parakeet/canary; the new bit is cache-aware FastConformer streaming. Wait for `--stream-json` (issue #84) to settle + a second user request (only mention so far is issue #85) before starting. |
| **MEDIUM** | [#86 Per-backend flash-attention wiring](#86-per-backend-flash-attention-wiring-crisperweaver-driven) | 2–3 days | Plumbing shipped 0.6.2; kernel-level wiring per backend is the remaining work. Whisper done; orpheus/chatterbox-T3 are the next-best pickings. |
| **LOW** | [#87 `gpu_backend` runtime selector](#87-gpu_backend-runtime-selector-multi-backend-ggml-build) | ~1 week | Needs ggml-side multi-backend dispatch to land first. CrisperWeaver UI placeholder ready when the C-side is. |
| **LOW** | [#95 IndexTTS Chinese TN binary alternative](#95-indextts-15-chinese-tn--binary-alternative-to-the-python-wetext-hook) | survey only | Python `INDEXTTS_TEXT_NORMALIZER` hook shipped 2026-05-19. Hand-roll (#95a) is the right next step *when* a user reports a digit/date prompt that breaks; OpenFST vendoring (#95b) only after #95a grows past ~5 cases. |

**Recently completed** (full write-ups in HISTORY.md): **#57 chatterbox native voice clone → §82** (six-commit sprint shipping all four upstream cond extractors — VoiceEncoder LSTM, S3Tokenizer V2, CAMPPlus, 24 kHz Matcha mel — plus a Kaiser-windowed sinc resampler and atomic 5-cond install in `chatterbox_set_voice_from_wav`'s `.wav` branch; `--voice ref_24k.wav` produces real cloned speech without any python). **#69 + #72 + #73 cap-honesty + KV/layer offload knobs → §79** (14-commit session shipping `CRISPASR_KV_QUANT_K/_V` + `KV_ON_CPU` on 14 backends, `N_GPU_LAYERS` on 10 backends, gemma4/mimo GPU-residency 2.2x / 22 % faster, plus cap-honesty cleanup on parakeet/glm-asr/qwen3/gemma4/omniasr). **vibevoice #69a follow-up → §79b** (mode-aware `tts_lm.layers.` / `lm.layers.` prefix predicate). #78 Chatterbox vocoder → §78. #11 WebSocket server → §76, #63 Feature matrix parity → §72, #59 binding parity → §73, gemma4 #49 + Docker #31 → §74, tests + KV Q8_0 + cleanup → §75. Earlier: #5→§63, #16→§55, #51→§56, #51b→§60, #53→§63, #54→§61, #55→§54, #56→§63, #60d→§64.

**Open follow-ups from §79 — we want all of these:**
- **#73 cohere long-form rerun.** flash_attn_ext is shipped on canary + cohere (commit 193a736). JFK (~11 s) numbers: canary q8_0/q4_0 -17 % under flash (win), but cohere q8_0/q4_0 is +11 % under flash vs cast-on-read on the same workload. F16 is a tie on both. Before promoting flash as cohere's recommended path, validate on a multi-minute clip — if the crossover is workload-dependent the docs need to recommend cast-on-read for short audio and flash for long. Until then PERFORMANCE.md notes flash as available-but-regresses-on-JFK for cohere.
- **#72 Linux/CUDA validation** of the gemma4_e2b / mimo_asr GPU-residency flip. Hardware-blocked from the dev host; expect even larger wins on dGPU than the 22 %–220 % observed on Apple Silicon Metal.
- **encoder-decoder #69a** (canary, cohere, kyutai-stt). Cross-attention layout has no `<prefix><N>.*` block-tagged tensors; needs bespoke per-backend predicates. Own design problem.

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
| Python (`_binding.py`) | 62 | ~46% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dart (`flutter/crispasr`) | ~39 | ~29% | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Go (`bindings/go`) | ~54 | ~40% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Java (JNA) | ~38 | ~28% | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅¹ | ✅¹ | ✅¹ |
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

## 74. Feature-matrix uplift round 2 — chatterbox family + matrix tooling

After §79b shipped chatterbox + 3 sibling variants and the audit-drift cleanup brought test-all-backends.py to 39/39 backends, four follow-ups surfaced from re-reading the cap matrix. They cluster by user-visible value:

### 74a. Auto-route by `-l <lang>` for chatterbox family — TIER 1 (cheap, high value)

Today `--backend chatterbox` always loads the English base. A user passing `-l de` and `--backend chatterbox` should get auto-routed to `kartoffelbox-turbo`; `-l ar` should go to `lahgtna-chatterbox`. Mirrors the existing kokoro `-l de` → German backbone routing pattern. ~20 LOC in `examples/cli/crispasr_backend.cpp` dispatch (or in chatterbox adapter's `init`). No new cap; pure DX win.

### 74b. CAP_TRANSLATE / CAP_SRC_TGT_LANGUAGE in test-all-backends.py — TIER 1 (cheap, completes audit)

The binary's caps enum already has `CAP_TRANSLATE` (1 << 5) and `CAP_SRC_TGT_LANGUAGE` (1 << 12); declared by canary, granite-4.1, granite-4.1-plus, voxtral, qwen3 per `--list-backends-json`. The test script's `CAPABILITIES_KNOWN` doesn't have either. Adding them lights up granite-4.1-plus's translate path and exposes a regression gate for the others. ~30 LOC: extend `CAPABILITIES_KNOWN`, add `_test_translate` tier handler that runs `--translate -l de samples/jfk.wav` and asserts non-empty German output.

### 74c. CAP_VOICE_CLONING — TIER 2 (new cap bit, cross-cutting)

Currently no way to express "this backend accepts a reference WAV via `--voice <wav>`." Backends that do: chatterbox, qwen3-tts (base + 1.7b-base), vibevoice (1.5B base — distinct from the realtime preset path). New cap bit in `examples/cli/crispasr_backend.h`, declarations in those adapters, test-script tier that runs `--voice samples/jfk.wav --tts "test" --tts-output /tmp/cloned.wav` and asserts non-zero peak. ~80 LOC.

### 74d. Generated sortable/filterable feature matrix (`docs/feature-matrix.html`) — TIER 2 (tooling)

Today's README matrix is a hand-maintained Markdown table — every backend addition requires editing 17 rows. Replace with a generation script that calls `crispasr --list-backends-json`, normalizes, emits both:
  * `docs/feature-matrix.md` — checked-in Markdown table (regenerated by the script; CI gate could check freshness)
  * `docs/feature-matrix.html` — vanilla-JS standalone (clickable column-header sort, top-of-page filter input). No external JS deps. Single self-contained file, viewable offline.

README links to both. ~200 LOC across the generator script + the HTML template. Single source of truth: the binary's JSON. Eliminates the hand-edit drift that #61 / #63 / #71 had to keep chasing.

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

### 80b. Energy-minimum chunk boundaries (in progress)

Cohere's long-form path at `src/cohere.cpp:1834-1901` cuts at exactly
`30 * sample_rate` samples — which slices mid-word. Port the
`_find_split_point_energy` helper from nano-cohere `chunk.py:61-80`:
within the last `boundary_context_seconds` (default 5 s) of each 30 s
window, scan in `min_energy_window_samples` (default 1600 = 100 ms)
non-overlapping slices and pick the lowest-RMS one as the cut point.

**Files:**
* `src/core/audio_chunking.h` — new header. Two functions:
  `audio_chunking::find_energy_min_split(span, search_start,
  search_end, win_samples) -> int` and
  `audio_chunking::split_at_energy_minima(span, sample_rate,
  max_chunk_s, search_window_s, win_samples) -> std::vector<std::pair<
  size_t,size_t>>` returning [begin,end) sample ranges.
* `src/cohere.cpp:1834-1901` — replace the fixed-step loop with a
  call to `split_at_energy_minima`; iterate the returned ranges.

**Acceptance:** JFK transcript byte-identical (single-chunk path
unaffected); 60 s synthetic clip's chunk boundary lands within a
quiet sub-window (verifiable by inspecting the cut-time printed by
`COHERE_VLOG`).

### 80c. CRISPASR_VERBOSE env override for cohere CLI (DONE this session)

Aligns cohere with the gemma4 / granite / firered / omniasr /
moonshine_streaming convention (env var or `-v` bumps backend
verbosity to 2 to print the perf report). One line in
`examples/cli/crispasr_backend_cohere.cpp`. Side benefit of the
investigation; without it, 80a couldn't be measured from the CLI.

### 80d. Cross-backend audit of fixed-time chunking (TODO)

Survey the long-form chunking strategy in every AR-decoder backend
(canary, voxtral, voxtral4b, vibevoice, kyutai-stt, gemma4_e2b,
qwen3-tts-talker). Anything cutting at `N * sample_rate` boundaries
inherits the same mid-word problem and gains from
`audio_chunking::split_at_energy_minima`. Investigate-only here; each
hit is its own one-line change.

### 80e. Eager warmup follow-up (TODO, low priority)

nano-cohere's `from_pretrained` does a 1 s silence transcribe on init
to amortize Metal kernel compile / first-shape gallocr setup. Easy to
add a `cohere_warmup(ctx)` call wired into model load. Worth ~50–
150 ms saved on the first user-visible call but not on steady state.
Leave for a later polish pass; if added, every backend gets its own
warmup hook.

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

### Per-backend status

| Backend | Has `use_gpu` field | Has `flash_attn` field | Compute graph branch | Effort |
|---|---|---|---|---|
| whisper | ✅ (whisper.cpp upstream) | ✅ | ✅ | DONE |
| parakeet | ✅ | ❌ | — | Small (Conformer encoder) |
| canary | ✅ | ❌ | — | Small (encoder SA + decoder XA) |
| qwen3 (asr) | ✅ | ❌ | — | Medium (Whisper-like enc + Qwen3 LLM) |
| cohere | ✅ | ❌ | — | Small (Conformer encoder) |
| granite_speech | ✅ | ❌ | — | Medium (Conformer + Granite LLM) |
| voxtral | ✅ | ❌ | — | Medium (Whisper enc + Mistral 3B) |
| voxtral4b | ✅ | ❌ | — | Medium (causal enc + SWA decoder) |
| vibevoice | ✅ | ❌ | — | Small (σ-VAE encoder + Qwen2.5 talker) |
| qwen3_tts | ✅ | ❌ | — | Medium (talker + code-predictor) |
| orpheus | ✅ | ❌ | — | Small-Medium (Llama 3.2 3B AR loop) |
| kokoro | ✅ | ❌ | — | Small (StyleTTS2-derived; less impact) |
| chatterbox | ✅ | ❌ | — | Medium (T3 AR + S3Gen flow-matching) |

### Approach (per-backend recipe)

For each backend with a transformer attention block:

1. Add `bool flash_attn` to `*_context_params` (default-init to true
   in `*_context_default_params()`).
2. Plumb `g_open_flash_attn_tls → cparams.flash_attn` in
   `crispasr_session_open_explicit`'s arm for that backend (mirror
   the existing `cparams.use_gpu` line).
3. In the compute graph, swap the QKV path:
   - Before: `ggml_soft_max_ext(ggml_mul_mat(K, Q), scale)` then
     `ggml_mul_mat(V_T, softmax_out)`.
   - After: branch on `cparams.flash_attn`:
     - true → `ggml_flash_attn_ext(Q, K, V, mask, scale, ...)`
     - false → keep the historical path.
4. Verify the Metal kernel for `flash_attn_ext` exists for the
   relevant head dim. Most are covered (D ∈ {64, 80, 96, 112, 128})
   but check before committing.
5. Diff-harness round-trip: dump pre/post-attention tensors with
   the flag on vs off, confirm the kernels agree to 1e-3 (Metal
   F16 fast-math drift is the typical failure mode — use
   `GGML_PREC_F32` op_param if needed; same fix as #83).

### Recommended order

1. **orpheus + chatterbox-T3** — Llama-style AR loops, biggest
   wall-clock win on long generations (3 B Llama / T3 dominate
   the synth budget). Reuses the same `ggml_flash_attn_ext` pattern
   per-block.
2. **voxtral / voxtral4b / qwen3 ASR / granite-speech** — LLM-based
   ASR backends where flash-attn helps both the Whisper-style
   encoder and the LLM decoder.
3. **qwen3-tts** — talker is Qwen3-style transformer; same recipe
   as qwen3 ASR.
4. **parakeet / canary / cohere** — Conformer encoders. Lower per-
   call benefit but sums up for batch-transcribe users.
5. **vibevoice / kokoro** — smallest impact; do last or skip.

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

**Still gapped** — three backend families need new C surface on
their high-level transcribe API before the session wrapper can
plumb `s->beam_size` through. Each is more substantial than the
"add a runtime setter" pattern above because the beam-decode
path doesn't live in the backend library; the CLI wraps it
externally:

| Backend | What's needed |
|---|---|
| granite / granite-4.1 / granite-4.1-plus / granite-4.1-nar | The granite library exposes `granite_speech_transcribe(ctx, samples, n_samples)` — no beam, no params struct, just text. The CLI wrapper at `crispasr_backend_granite.cpp` runs its OWN beam decode via `core_beam_decode::run_with_probs(ctx_, logits, replay, cfg)` after pulling logits out of granite. Wiring beam search into the session API needs either: (a) move the `core_beam_decode` call into the granite library + new public `granite_speech_transcribe_with_beam(ctx, samples, n_samples, beam_size)`, or (b) expose granite's logits + replay buffer publicly so the session dispatcher can run `core_beam_decode` itself. Option (a) is cleaner — ~3-4 hours per granite variant. |
| voxtral / voxtral4b | Beam search via internal `voxtral_decode_beam` (not exposed). Same options: (a) public `voxtral_transcribe_with_params(ctx, samples, n_samples, voxtral_decode_params{beam_size, ...})` or (b) expose internal beam decoder. |
| qwen3-asr | Same shape — internal beam, no public surface. |

Each needs ~3-4 hours of careful per-backend work (read beam
path, decide a/b, write the new surface, wire from
`transcribe_single`, add a Catch2 smoke test). Total estimate
~1-1.5 days for the remaining three families. Each is
independent — can ship one at a time.

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
2. The hand-rolled list grows past ~5 cases (track in
   `src/indextts.cpp`). At that point the marginal cost of #95b
   (vendoring OpenFST) becomes lower than continuing to grow the
   hand-rolled rules. Cross the threshold deliberately.

Don't pre-emptively vendor OpenFST. CrispASR's clean "ggml + minor
deps" profile is a feature.
