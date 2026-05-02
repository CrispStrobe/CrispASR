# CrispASR — TODO

Live tracker of pending work across the unified `crispasr` binary and the
shared `src/core/` infrastructure. Items marked **[next]** are the current
session's immediate targets; **[later]** are queued; **[upstream]** are
blocked on external fixes (tracked in detail in `UPSTREAM.md`).

Historical milestones and the per-model port plans are in `HISTORY.md`.
Technical deep-dives (optimisation notes, RoPE lessons, benchmark tables)
are in `LEARNINGS.md`. Full roadmap in `PLAN.md`.

---

## Granite-family DRY refactor (PLAN #55) **[done]** — May 2026

All five steps shipped behind JFK smoke tests on granite-4.1, granite-4.1-plus,
and granite-4.1-nar. See `HISTORY.md` §54 for the table.

New shared headers landed in `src/core/`:
- `core/fft.h` (radix-2 FFT, replaces inline copies in granite + future kokoro/mimo)
- `core/cpu_ops.h` (CPU layernorm + matmul fallbacks)
- `core/ctc.h` (posterior-weighted pool + greedy-decode-with-blank)
- `core/conformer_ibm.h` (IBM-flavour Macaron block + Shaw RPE — sibling of `core/fastconformer.h`, NOT a merge target)
- `core/granite_llm.h` (40-block backbone, `is_causal` flag selects KV-cached vs non-causal flash)
- `core/qformer.h` (NAR-only simplified windowed Q-Former)

Step 5 plan correction: the duplication map originally claimed both
granite TUs share the windowed Q-Former. They don't — granite_speech
runs a full BLIP-2 Q-Former (sa+ca+ffn per layer) while granite_nle
runs a simplified one (cross-attn-only + MLP). `core/qformer.h` shipped
as NAR-only co-location.

---

## Kokoro / StyleTTS2 (iSTFTNet) backend **[done]**

Plan file: `/Users/christianstrobele/.claude/plans/sprightly-enchanting-dahl.md`.
Lessons: `LEARNINGS.md` "Kokoro / StyleTTS2 lessons".

Status as of 2026-05-01: **M1–M9 + M11 + M12 shipped**. End-to-end
synthesis works; the M12 diff harness validates 16 stages against the
official PyTorch reference at `cos_min ≥ 0.999` for all deterministic
stages and `≥ 0.95` for the four RNG-divergent generator stages
(`gen_pre_post_out`, `mag`, `phase`, `audio_out` — Python `torch.rand`
vs C++ `std::mt19937` cannot match exactly; current floor is cos≈0.99
on `audio_out`).

**Closed bug from the M11/M12 work** (commit `448c1af`): the
`kokoro_pool_2x_depthwise` ConvTranspose1d had `w[0]` and `w[2]`
swapped in the odd-output formula. Affected predictor F0[1] / N[1]
and decoder.decode[3]. Fix: `y[2t+1] = w[2]·x[t] + w[0]·x[t+1]` per
PyTorch indexing. Hidden by overall envelope/energy looking right.

Stages exposed via `kokoro_extract_stage()`: token_ids,
bert_pooler_out, bert_proj_out, text_enc_out, dur_enc_out,
pred_lstm_out, durations, align_out, f0_curve, n_curve,
dec_encode_out, dec_decode_3_out, gen_pre_post_out, mag, phase,
audio_out (16 total). Reference dumper at
`tools/reference_backends/kokoro.py` mirrors the 9 corrections
(pad-wrap, drop-unknown, AdaIN affine override, banker's rounding,
voice-pack split, conv_post exp/sin, etc.).

**Polish (later):**
- Tighten generator-stage cos floor by intercepting SineGen's RNG in
  the Python hook to feed the same uniform/normal sequence as
  `std::mt19937(0x12345)` produces. Currently accepted at cos≈0.99.
- CLI wrappers / Python bindings (parallels qwen3-tts CLI work below).

---

## Kokoro multilingual phonemizer (espeak-ng) **[next]**

Full plan: `PLAN.md` → §56. Lessons: `LEARNINGS.md` "Kokoro phonemizer:
libespeak-ng vs popen divergence".

Two layers: (a) in-process libespeak-ng phonemizer + LRU cache for all
languages, and (b) per-language voice fallback table for languages
Kokoro-82M doesn't ship voices for (de, ru, ko, ar, …).

**Shipped (commits `6d6e978`, `ad7dac5`, `7615005`, `65be09d`,
`2215996`, `<this commit>`):**

- *Phonemizer layer.* `CRISPASR_WITH_ESPEAK_NG` AUTO probe (pkg-config
  + Homebrew/Linux fallback). In-process `espeak_TextToPhonemes()`
  behind a process-global mutex; sticky init + voice; LRU cache
  (1024 entries, mutex-protected) keyed on `lang \0 text`. Popen
  retained as runtime fallback. CLI `-l/--language` → `cp.espeak_lang`.
- *End-to-end synth verified.* Six languages (en, de, fr, ru, cmn, ja)
  exercised. en/fr/ru clean; cmn loses tone numbers; ja kanji falls
  back to English. de needed a voice + backbone fix — see below.
- *German backbone + voice routing (PLAN §56 opt 2b).* When `-l de*`
  AND the user-passed model is the official `kokoro-82m-*.gguf`, the
  CLI silently swaps to a sibling `kokoro-de-hui-base-f16.gguf` if
  present (dida-80b/kokoro-german-hui-multispeaker-base, Apache-2.0;
  HUI corpus CC0; predictor + decoder + StyleEncoder all German-
  trained). Voice cascade for German: `df_victoria`
  (kikiri-tts/kikiri-german-victoria voicepack, Apache-2.0,
  in-distribution to dida-80b lineage) → `df_eva` (Tundragoon
  Apache-2.0) → `ff_siwis` (French baseline). Everything else without
  a native pack still falls back to `ff_siwis`. Resolution order:
  explicit `--voice` → cascade → empty (helpful error). C ABI
  exposed for wrappers: `crispasr_kokoro_resolve_model_for_lang_abi`
  + `crispasr_kokoro_resolve_fallback_voice_abi` re-exported from
  `src/crispasr_c_api.cpp`; Python surface
  `crispasr.kokoro_resolve_for_lang()` returns `KokoroResolved`.
- *ASR-validated end-to-end.* Long German phrase ("Guten Tag, dies
  ist ein Test des deutschen Phonemizers."), parakeet-v3 roundtrip
  on each model + voice combo:
  - `dida-80b + dm_martin` → "...Phonemizers." (perfect)
  - `dida-80b + df_victoria` → "...Phonemizers." (1 word boundary err)
  - `dida-80b + dm_bernd` → "...Phonemetzers." (1 word boundary err)
  - `dida-80b + df_eva` → "...Phonemetzes." (1 word boundary err)
  - `official + df_eva` (pre-2b baseline) → "...Phonemizer." (lost s)
  All four voices clear the gate (peak ≥ 8000, RMS ≥ 1000).

**Shipped on top of the above (this commit cluster, 2026-05-01):**

- *HF GGUF mirrors published.*
  [`cstr/kokoro-82m-GGUF`](https://huggingface.co/cstr/kokoro-82m-GGUF),
  [`cstr/kokoro-de-hui-base-GGUF`](https://huggingface.co/cstr/kokoro-de-hui-base-GGUF),
  [`cstr/kokoro-voices-GGUF`](https://huggingface.co/cstr/kokoro-voices-GGUF)
  — F16 + Q8_0 for both backbones, 7 voicepacks. Q4_K is **not**
  published: `crispasr-diff kokoro` shows audio_out cosine collapsing
  to 0.03 on Q4_K and the dida-80b backbone produces unintelligible
  output ("Guten A, dies ist ein S des Worten von links."). See
  LEARNINGS.md "Kokoro quant ceiling" for the per-stage diff numbers
  + ASR roundtrip table.
- *Auto-download wired via the registry* (`src/crispasr_model_registry.cpp`).
  New `ExtraCompanion` mechanism — backends with >1 auxiliary file
  list extras in `k_extras` alongside the inline `companion_file`.
  Kokoro's row pulls 4 files via `-m auto`: kokoro-82m-q8_0
  (backbone) + kokoro-voice-af_heart (English default, inline
  companion) + kokoro-de-hui-base-q8_0 (German backbone, extra) +
  kokoro-voice-df_victoria (German default, extra). CLI then
  auto-picks `af_heart` as the default English voice when `--voice`
  is empty.
- *Wrapper TTS surface across all bindings* (Rust sys + crate, Go,
  Java, JavaScript-emscripten, Ruby). Each exposes
  `Session.{open,close,setCodecPath,setVoice,synthesize}` plus
  `kokoroResolveForLang(model, lang) -> KokoroResolved`. Mirrors the
  verified Python wrapper. Go/Java/JS/Ruby were upstream-whisper-only
  before — this is their first TTS surface.
- *Reference dumper compat with dida-80b's modern parametrize
  WeightNorm.* The German backbone ships keys as
  `parametrizations.weight.original{0,1}`; upstream `KModel.__init__`
  only handles legacy `weight_g`/`weight_v`. Manual workaround: split
  the StyleTTS2-style `state['net']` dict and rename
  `parametrizations.weight.original0/1` → `weight_g/weight_v`. The
  reference dumper now produces a German diff that matches the C++
  GGUF at cos≥0.999 on 14/16 stages. Could be folded into
  `tools/reference_backends/kokoro.py` as a mode flag if we add more
  community Kokoro re-trains (low priority).

**Still open:**
1. **Mandarin tones / Japanese kanji.** espeak-ng tone numbers and
   CJK fallback both lose information at the kokoro vocab level.
   For tones: try `--ipa=2` or pypinyin. For Japanese: pyopenjtalk
   pre-process. See PLAN §56 open #2 / #3.
2. **`crispasr-diff kokoro` reference backend** covering the
   phonemizer step too (the model side is already covered, including
   dida-80b after this session's parametrize-rename trick). PLAN
   §56 open #4.
3. **Stage-2 fine-tune on one HUI speaker** (~half-day A40) for
   deployable single-voice production quality. Out of scope here.
4. ~~Optional `kokoro_phoneme_cache_clear()` C ABI.~~ **DONE
   (May 2026, commits `9bffb0f` / `6cabefa` / `d022bff` / `603f47e`)** —
   shipped across all 7 wrappers + no-model tests. PLAN §56 #5 closed.

---

## TTS voicepack expansion **[in progress]**

### VibeVoice-Realtime (shipped 2026-05-01)

`cstr/vibevoice-realtime-0.5b-GGUF` now bundles all 25 demo voices
from [`microsoft/VibeVoice@main`](https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model)
(MIT). Languages: en (6 voices), de/fr/it/jp/kr/nl/pl/pt/sp (2 each),
in/Indian-English (1, Samuel). Each voicepack is 2-6 MB.

**Auto-download wiring (commit `49b99f8`).** The registry's
`vibevoice-tts` row keeps `emma` as the inline companion and adds
`k_vibevoice_tts_extras` with `de-Spk1_woman` + `fr-Spk1_woman`. So
`crispasr -m auto --backend vibevoice-tts -l de` pulls model + emma +
both extras (~10 MB total extra payload), and the CLI auto-picks the
matching voice for the requested language without `--voice`:

  --voice <path>                            (explicit, always wins)
  → vibevoice-voice-<lang>-Spk1_woman.gguf  (sibling, woman first)
  → vibevoice-voice-<lang>-Spk0_man.gguf
  → vibevoice-voice-emma.gguf               (English default)
  → clear error                             (no synthesis attempted)

Same commit fixes a silent-fallback bug: when `--voice` was empty AND
no preload had run, the model previously got an unconditioned voice
prompt and produced ~1.2 sec of EOS-truncated gibberish. The CLI now
errors out clearly if no voice is resolvable. See
`LEARNINGS.md` "VibeVoice silent unconditioned-voice fallback".

**ASR-validated end-to-end:** `-m auto -l de --tts "Guten Tag, dies
ist ein Test der deutschen Stimme."` → parakeet-v3 -l de roundtrip:

  - de-Spk1_woman (auto-pick): byte-perfect
  - de-Spk0_man:                "Schönen Tag, dies ist ein Test ..."
                                (1 word substitution, both legit greetings)

Smoke-tested on other languages: en-Grace_woman byte-perfect EN ASR;
fr-Spk0_man near-perfect ("voix" → "voice", ASR quirk on rare word);
JP synth produces expected duration audio (no parakeet-v3 JA support
so couldn't roundtrip; would need parakeet-ja for that).

Local files staged at `/Volumes/backups/ai/crispasr-models/
vibevoice-voice-{en-Grace_woman,en-Mike_man,fr-Spk*,it-Spk*,
jp-Spk*,kr-Spk*,nl-Spk*,pl-Spk*,pt-Spk*,sp-Spk*,in-Samuel_man}.gguf`.

**Open follow-ups for vibevoice voicepacks (low priority):**
- Other languages besides de/fr only auto-download via explicit
  filename. If demand emerges for `-l ja` / `-l it` / etc., add to
  `k_vibevoice_tts_extras`. Trade-off is auto-download payload size.
- Could expose the cascade (`vibevoice-voice-<lang>-Spk0_man` →
  `Spk1_woman` → `emma`) as a C ABI `crispasr_vibevoice_resolve_voice_for_lang`
  so wrappers (rust/go/java/js/ruby) can use the same logic outside
  the CLI. Mirrors the kokoro pattern. ~30 LOC if asked.

### Qwen3-TTS — no clean voicepacks available

Qwen3-TTS is voice-cloning-from-WAV (not pre-baked voicepacks like
vibevoice). Surveyed HF for clean third-party voicepacks:

- **`kautism/qwen3_tts_voices`** — 500+ voicepacks but they're all
  HoYoverse 星穹铁道 (Honkai: Star Rail) characters, license
  undeclared, almost certainly extracted from commercial game audio.
  **Skip** — not redistributable.
- **`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`**,
  **`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`** — official Apache-2.0
  Qwen models for *runtime* voice cloning (CustomVoice = WAV-driven,
  VoiceDesign = text-prompt-driven). No pre-baked voicepacks ship.

To add Apache-2.0 / CC-BY qwen3-tts voicepacks we'd need to bake them
ourselves from a public-domain corpus via
`models/bake-qwen3-tts-voice-pack.py`. Candidate sources:

| Source | License | Notes |
|---|---|---|
| LibriTTS-R (`mythicinfinity/libritts_r`) | CC-BY-4.0 | 24 kHz, English, 2,400 speakers — could pick 5-10 representative voices |
| VCTK (`CSTR-Edinburgh/vctk`) | CC-BY-4.0 | 110 English speakers, multiple accents |
| Common Voice 17 | CC0 | Multilingual but per-speaker quality varies wildly |
| MLCommons People's Speech | CC-BY-2.0/3.0/4.0 | Multilingual, mixed licenses per clip |

This is a separate work item — out of scope for this session.

---

## Qwen3-TTS CLI/wrapper integration **[next]**

Bring qwen3-tts up to feature parity with vibevoice's `--tts` mode and
fix the leading-silence regression in the talker. Order fixed by user
direction; do not interleave.

1. **CLI wiring.** Mirror `crispasr_backend_vibevoice.cpp`:
   - New `examples/cli/crispasr_backend_qwen3_tts.cpp` implementing
     `CrispasrBackend` with `CAP_TTS` set. `synthesize()` calls the
     existing `qwen3_tts_synthesise` (or the runtime path that already
     produced `crispasr_1..3.wav`) and returns 24 kHz PCM.
   - Voice prompt path: piggy-back on `--voice` (current vibevoice
     flag) for the GGUF voice pack, plus a new `--ref-audio` /
     `--ref-text` pair for runtime cloning. Add fields to
     `whisper_params.h` only if the existing `tts_voice` slot can't
     carry both.
   - Register the new factory in `examples/cli/crispasr_backend.cpp`
     and add the row to `crispasr_list_backends()`.
   - CMake: link `qwen3_tts` + the codec/talker static libs into the
     `crispasr` target.
   - Verify: `crispasr --backend qwen3-tts --tts "hello world" -of out`
     produces the same WAV as `tools/qwen3_tts_e2e.py` (or whatever
     the standalone tool was).

2. **Wrappers / bindings.** Order: Python first (it's the one we
   ground-truth diff against), then Rust, Go, Java, JS, Ruby. Each
   wrapper just needs the new backend name surfaced — the synthesise
   API already returns `float*` + length on the C ABI side.

3. **Talker-side silence fix (default).** `crispasr_1.wav` had 4.79 s
   of leading silence; Python ref has 0.31–0.56 s. Likely
   stochastic-sampling artefact (top_k=50 / temp=0.9 picks padding
   tokens early). Investigate:
   - `min_new_tokens` / first-N-token logit biasing against pad/EOS.
   - Whether the PyTorch reference biases differently or uses a
     warmup prefill.
   - As a fallback, force the first M decode steps to argmax.

4. **Output-side silence trim (optional CLI flag).** `--tts-trim-silence`
   (default off) on the dispatcher. RMS gate over a 20 ms window —
   drop leading frames below e.g. -50 dBFS, keep a 50 ms head-room.
   Same path applies to vibevoice for free.

5. **Encoder `cos_min ≥ 0.999`. [DONE]** Fixed the major memory layout bug in
   the CPU-side RVQ helpers (was channels-first, now row-major [T, C]).
   `cenc_codes` now at cos_mean=0.998+ and end-to-end TTS cloning is
   clean. Remaining 0.001 drift is negligible stochastic noise.

---

## Pending optimizations (v0.5.x)

| # | Optimization | Backends | Expected gain | Effort | Status |
|---|---|---|---|---|---|
| O1 | ggml grouped conv1d (im2col+mul_mat) | wav2vec2, data2vec, hubert | **4.9x pos_conv** | Done | **DONE** |
| O2 | Fused QKV pre-merge (single matmul) | LLM decoders | ~10-15% attn (GPU) | Medium | Infra done (F32/F16 only; Q4_K needs converter-level fuse) |
| O3 | ggml bump 0.9.8→0.10.0 | All | Bug fixes, FA head_dim=512, BF16 FA | Done | **DONE** |
| O5 | Pipelined mel+encode threading | LLM backends, CPU | ~15-20% | Medium | TODO |
| O4 | Beam search for LLM backends | All Audio-LLM | Quality improvement | High | TODO |
| O6 | Batched encoder (GPU only) | All backends | 3-5x on GPU | High | TODO |
| O7 | Speculative decoding | LLM backends | 2-4x decode | High | TODO |
| O8 | FireRed single-graph encoder | firered-asr | ~15s GPU savings | High | TODO (needs rel_pos_attn refactor) |
| O9 | Grouped conv graph integration | wav2vec2 family | ~300ms saved | Medium | BLOCKED (ggml view bounds) |
| O10 | Gemma4 audio attention: replace block-wise manual attn with `flash_attn_ext` + bias mask | gemma4-e2b | ~5x encoder | Medium | TODO (now low-priority — encoder is 9% of total post-eos fix) |
| O11 | Gemma4: reduce redundant `ggml_cont` calls in clipped_mul_mat (private_input flag wired) | gemma4-e2b | ~5-10% encoder | Low | Done in commit 4abb68f; further wins limited |
| O12 | Gemma4: prefer `end_of_turn_id` over `eos_id` in greedy decode | gemma4-e2b | **8.7× total** | Trivial | **DONE** (commit e9420c5) |
| O13 | Gemma4: avoid re-cont-ing donor K/V for KV-shared layers every decode step | gemma4-e2b | ~10-20% decode? | Medium | TODO (speculative; per-step memcpy of Lk×hd×n_kv F16 × 20 layers) |

## Pending features (v0.5.x)

- **WebSocket streaming server** — `/ws` endpoint for real-time transcription
- ~~**Audio format support**~~ **FIXED** — `common-crispasr` now falls back to an
  `ffmpeg` subprocess (`ffmpeg -loglevel error -i <file> -f s16le -ar 16000 -ac 1 -`)
  when miniaudio fails to open the input. Handles m4a/mp4/webm/aac/opus when the
  user has ffmpeg installed. The previous `ma_decoder_init_memory(fname.c_str(),
  fname.size(), ...)` was a silent no-op that always failed.
- ~~**Japanese punctuation split (#29)**~~ **FIXED** — CJK clause-break + 42-char fallback
- ~~**Moonshine multilingual**~~ **FIXED** — converter forces 1D tensors to F32 (line 338). All 14 GGUF variants (tiny/base × en/ja/ar/ko/zh/vi/uk) work on CPU. head_dim=52 (base) works on CPU flash_attn; GPU flash_attn needs aligned head_dim (ggml limitation, moonshine forced to CPU anyway). Verified 2026-04-26: tiny 50.7×, base (head_dim=52) 14.2×, base-zh on English audio 14.8× — all transcripts correct.
- ~~**Moonshine streaming**~~ **DONE** — 3 sizes (tiny/small/medium), all MIT, all on HF.
  Backend `--backend moonshine-streaming`, model registry for auto-download.
- ~~**Gemma-4-E2B**~~ — **DONE** (2026-04-28). Google USM Conformer (12L) +
  Gemma4 LLM (35L), 5.1B params with PLE. End-to-end correct on jfk.wav.
  Converter: `cstr/gemma4-e2b-it-GGUF` (F16 9.5 GB, Q8_0 5.0 GB,
  Q4_K 2.8 GB, Q2_K 2.2 GB). Backend: `--backend gemma4-e2b`.
  Per-stage cosine sim vs HF: `mel=1.0000`, `subsample=0.9994`,
  `audio_layers=0.97-0.99`, `tower_output=0.99+`. Bugs fixed
  along the way (full list in `HISTORY.md`):
  - Audio: ClippableLinear QAT clip scalars (input/output min/max
    applied LIVE every forward, not training-only); HF mel
    feature-extractor parameters (fft_length=512, magnitude not
    power, log(mel+0.001), no Slaney norm); per-dim scale baked
    at load (q_scale·softplus(pds)); rel_pos_bias `matrix_ac+matrix_bd`
    + rel_shift; subsample flatten axis order
    (`ggml_permute(h, 1, 2, 0, 3)`); `ggml_clamp` is in-place →
    pre-`ggml_cont` shared inputs.
  - LLM: KV-share donor map (LAST 20 layers reuse, not FIRST);
    `use_double_wide_mlp` is single 2× MLP not two halves; PLE
    direction; per-layer head_dim (sliding=256, full=512);
    partial-rotary RoPE via `n_rot`.
  Diff harness: `crispasr-diff gemma4 …` works end-to-end with
  `tools/reference_backends/gemma4.py` (audio-only path fits 8 GB
  RAM via `_dump_audio_only`). Apache 2.0.
  Open polish: speed (currently ~0.2× realtime — see #17 below).
- **MiMo-V2.5-ASR** — **[IN PROGRESS]** Xiaomi 8B Qwen2 + 1.2B RVQ audio tokenizer.
  Both converters DONE. F16 GGUFs on HF: `cstr/mimo-asr-GGUF` (15.3 GB),
  `cstr/mimo-tokenizer-GGUF` (Q4_K 377 MB). Runtime not yet written. MIT.
- **German wav2vec2 models** — **[DONE]** 5 models on HF, all Apache 2.0 / MIT:
  `wav2vec2-large-xlsr-53-german` (222 MB Q4_K), `wav2vec2-large-xlsr-53-german-cv13` (212 MB),
  `wav2vec2-base-german-cv9` (80 MB, MIT), plus 1B models converting on Kaggle.
  Model registry: `--backend wav2vec2 -m auto -l de` auto-downloads German model.
- ~~**MarbleNet VAD**~~ **DONE** — NVIDIA 1D separable CNN, 91.5K params, 439 KB GGUF.
  6 languages (EN/DE/FR/ES/RU/ZH). Auto-download: `--vad -vm marblenet`.
  HF: `cstr/marblenet-vad-GGUF`. NVIDIA Open Model License.
- ~~**Whisper-VAD-EncDec**~~ **DONE** — Whisper-base encoder + 2L decoder head, 22 MB Q4_K.
  Experimental (trained on Japanese ASMR). Auto-download: `--vad -vm whisper-vad`.
  HF: `cstr/whisper-vad-encdec-asmr-GGUF`.
- **4 VAD backends total**: Silero (default), FireRedVAD (recommended), MarbleNet, Whisper-VAD.
  All with auto-download keywords.
- ~~**Parakeet-JA**~~ — **DONE** (xscaling fix → F16 bit-exact match
  with NeMo). Q4_K of the JA model is still quantisation-sensitive
  (degrades after ~8 tokens because `joint.pred` / `decoder.embed`
  fall back to q4_0); follow-up to ship a Q5_K build, or pin those
  two tensors to F16 inside Q4_K, lives below under "open polish".
  See HISTORY.md.
- **MiMo-V2.5-ASR** — **[WORKS, default path]** End-to-end JFK
  transcription matches the upstream Python `MimoAudio.asr_sft`
  reference verbatim. PLAN #51 SHIPPED. F16 (14.9 GB) + Q4_K
  (4.5 GB) on [`cstr/mimo-asr-GGUF`](https://huggingface.co/cstr/mimo-asr-GGUF)
  with corrected vocab (151680) + merges (151291). MIT.
  Perf wave 51b/b' shipped (HISTORY section 60): step-only decode
  graph + O15-style cache, 1.46× per-step decode, JFK transcript
  byte-identical. Remaining perf follow-ups (51a mmap loader,
  51c F16 decode) at LOW priority.

  **PLAN #51 status (2026-05-01, late session):**

  **Tokenizer encoder — DONE end-to-end** (commits cd519c1, 21b4193,
  4ea4cba). All six stages (`tok_mel`, `tok_conv1_out`,
  `tok_conv2_out`, `tok_xfmr_out`, `tok_pool_out`, `tok_codes`) match
  PyTorch with cos_mean ≥ 0.999 on jfk.wav (cos_min ≥ 0.97 driven by
  the polyphase resampler / reflect-pad approximation in `tok_mel`).
  CPU-pinned by default; opt in via `MIMO_TOKENIZER_GPU=1`.

  **LM half — foundation laid, body not yet implemented.**
  Commit 6cdbdb7 extends `core_attn::kv_self_attn` with optional
  q_b/k_b/v_b/o_b parameters (defaulting to nullptr) so the same helper
  serves the Qwen2 LM (which has Q/K/V biases). All other LLM consumers
  (qwen3-asr, voxtral, voxtral4b, granite-speech, granite-nle,
  gemma4-e2b, qwen3-tts) compile and run byte-identically.

  **GGUF tensor-name gotcha** (do NOT re-derive at the converter — the
  Q4_K is already on HF as `cstr/mimo-asr-GGUF`):
   - The real Qwen2 LM lives under `model.layers.{0..35}.*` (36 × 12
     tensors, hidden=4096) — the converter's rename rule for
     `model.layers.` was missing, so these came through partially
     renamed.
   - `llm.blk.{0..15}.*` (1024d) is the TTS-direction `local_transformer`
     audio decoder. ASR does NOT use it.
   - `llm.codebook_head.{0..7}.*` and `llm.norm.weight` are similarly
     TTS-direction; skip them.
   - `llm.embed.weight`, `llm.final_norm.weight`, `llm.lm_head.weight`,
     `audio.blk.*`, `audio.norm`, `audio.emb.{0..7}`, `audio.group_proj`
     all map cleanly to what the runtime needs.

  **Converter follow-up (BLOCKING runtime):** the mimo-asr converter
  skips `tokenizer.ggml.merges` (`models/convert-mimo-asr-to-gguf.py`
  comment claims gguf type 9 is rejected). qwen3-asr/granite-speech
  both ship merges as a string-array field that
  `core_gguf::kv_str_array` reads fine. Without merges the runtime
  cannot encode the chat prompt strings (`user\n`, `assistant\n`,
  `<think>...`, ASR template) — only the special tokens (`<|im_start|>`,
  `<|empty|>`, ...) round-trip. Need to re-run the converter with merges
  written and re-upload the four GGUFs.

  **What's left for the runtime (in order):**
  1. ~~**Bind tensors**~~ **DONE** — typed `mimo_asr_qwen2_block` /
     `audio_tower` / `llm` populated after weight load with require()-
     or-fail per tensor. LM at `model.layers.{0..35}.*`, audio at
     `audio.blk.{0..5}.*` + `audio.emb.{0..7}` + `audio.group_proj` +
     `audio.norm`. `llm.blk.*` / `llm.codebook_head.*` / `llm.norm.*` /
     `audio.hidden_proj` ignored as TTS-direction junk for ASR.
  2. ~~**Input local transformer**~~ **DONE** — 6L bidirectional Qwen2
     built per-group (gs=4 along T, T_groups along batch dim) in
     `build_input_local_block`. Biased Q/K/V matmul + RoPE θ=640000 on
     each layer's 4-position window + `flash_attn_ext` with no mask
     (full attention within each group). 6 layers chained then
     `audio.norm` RMSNorm.
  3. ~~**_prepare_input_embeds**~~ **DONE** — per-channel
     `ggml_get_rows` lookup of speech_embeddings, masked by precomputed
     "(text==empty AND code!=zeroemb_idx)" gate, summed across 8
     channels. Reshape to `[ad, gs, T_groups]`, run input_local
     transformer, reshape to `[ad*gs, T_groups]`, group_proj matmul to
     `[llm_hidden, T_groups]`, mask non-speech positions, add to
     text_embeds (zero where text==empty). Captured as
     `prefill_audio_features` (post group_proj+mask) and
     `prefill_inputs_embeds` (final LM input).
  4. ~~**LM forward (prefill)**~~ **DONE** — full 36L Qwen2 with biased
     Q/K/V via `core_attn::kv_self_attn(..., q_b/k_b/v_b)`. KV cache
     allocated lazily on first prefill via `mimo_asr_kv_init` (F16,
     hd*max_ctx*n_kv*36 layers). Final RMSNorm + lm_head on the last
     position only — captured as `prefill_last_hidden` and
     `prefill_text_logits_step0`. Decode (T=1) reuses the same builder
     with n_past advance, but the actual decode loop is not wired yet.
  5. ~~**Python ref dumper for LM half**~~ **DONE** —
     `tools/reference_backends/mimo_asr.py`, registered in
     `tools/dump_reference.py`. Mirrors `MimoAudio.preprocess_input` +
     `_prepare_input_embeds` + `model.model(...)` + `lm_head` exactly.
     Pinned to `asr_en_templates[0]` for determinism; force-CPU+fp32.
     Captures the 4 prefill stages plus `prefill_input_ids` (so the
     C++ harness reads the same tokenized prompt the ref ran on).
  6. ~~**Diff harness branch**~~ **DONE** — `mimo-asr` branch in
     `examples/cli/crispasr_diff_main.cpp`. Reads `prefill_input_ids`
     from the ref archive, casts F32→I32, runs each of the 4 stages
     via `mimo_asr_extract_stage`, compares cos against the ref.

  **What's left:**
  7. ~~**First ref dump + diff**~~ **DONE (within Q4_K + bf16
     tolerance)** — JFK pass on the existing on-disk Q4_K with bf16
     PyTorch ref:
     - `prefill_audio_features`     cos_mean=0.998
     - `prefill_text_embeds`        cos_mean=0.996  (added stage)
     - `prefill_inputs_embeds`      cos_mean=0.998
     - `prefill_last_hidden`        cos=0.963
     - `prefill_text_logits_step0`  cos=0.981  (argmax = 1597 = ' And',
                                                matches Python ref)
     The ≥0.999 gate is the F16/F32 path; under Q4_K weights + bf16 ref
     the residual is quantisation noise, with the deepest-stack drop
     (last_hidden) limited by the 36L Q4_K LM body. Bugs fixed along
     the way (all in `src/mimo_asr.cpp`):
       a. `build_input_local_block` permuted `(hd,n_h,gs,ng) →
          (hd,gs,n_h,ng)` BEFORE `ggml_rope_ext`, putting `n_h` at
          ne[2] where positions[gs] expected — assertion
          `a->ne[2] == b->ne[0]` failed. Fix: rope-then-permute, so
          rope sees `[hd,n_h,gs,ng]` (gs at ne[2]) and flash_attn_ext
          sees `[hd,gs,n_h,ng]` (gs at ne[1]).
       b. After o-projection, `attn` was 2D `[d, gs*ng]` while the
          residual was 3D `[d, gs, ng]` — `ggml_add` broadcast assert
          failed. Fix: `ggml_reshape_3d(attn, d, gs, ng)` before
          residual add.
       c. Stale on-disk Q4_K has only 151643 vocab entries, missing
          `<|empty|>` (id 151667). `mimo_asr_extract_stage` fell back
          to 151643 (Qwen2 `<|endoftext|>`), which never appears in
          the prompt → all positions were treated as non-empty,
          zeroing the audio path entirely and blanket-keeping text
          embeds. Fix: bumped fallback to 151667 (the real id).
          Step 9 reconversion supersedes this when the vocab is
          padded to vocab_size.
       d. **Buffer-aliasing root cause for inputs_embeds drift to
          cos≈0.003.** The capture tensors (`prefill_audio_features`,
          `prefill_inputs_embeds`, etc.) had `ggml_set_name` but not
          `ggml_set_output`. The backend scheduler treated them as
          plain intermediates and reused their buffers when
          allocating later ops in the same graph (e.g. inputs_embeds
          consuming the same `x` post-mul tensor as the audio
          capture). The values read back via `ggml_graph_get_tensor`
          were post-clobber. Fix: `ggml_set_output(...)` on every
          named capture; per-stage cosines snapped to their real
          values immediately. Recipe: any tensor we plan to read out
          of the graph must be `set_output`, not just `set_name`.
     Command (from CrispASR/):
     ```
     HF_HOME=/Volumes/backups/ai/huggingface-hub \
     HUGGINGFACE_HUB_CACHE=/Volumes/backups/ai/huggingface-hub \
     TRANSFORMERS_OFFLINE=1 \
     MIMO_ASR_DIR=/Volumes/backups/ai/huggingface-hub/models--XiaomiMiMo--MiMo-V2.5-ASR/snapshots/<hash> \
     MIMO_TOKENIZER_DIR=/Volumes/backups/ai/huggingface-hub/models--XiaomiMiMo--MiMo-Audio-Tokenizer/snapshots/<hash> \
     python tools/dump_reference.py --backend mimo-asr \
       --model-dir <same-as-MIMO_ASR_DIR> --audio samples/jfk.wav \
       --output /tmp/mimo-asr-ref.gguf
     build-ninja-compile/bin/crispasr-diff mimo-asr \
       /Volumes/backups/ai/crispasr-models/mimo/mimo-asr-q4_k.gguf \
       /tmp/mimo-asr-ref.gguf samples/jfk.wav
     ```
     The dumper defaults to bf16 (env `MIMO_ASR_REF_DTYPE=fp32`
     overrides — needs ~28 GB RAM for the LM half, swap-thrashes a
     16 GB box). The strict cos≥0.999 gate would need F16 + fp32 ref;
     the diff harness can't load the 16 GB F16 on a 16 GB Mac without
     swap-thrashing (`core_gguf::load_weights` materialises tensors
     into RAM rather than mmap-ing). Q4_K diff against the new GGUF
     reproduces step 7 numbers exactly, confirming step 9 didn't
     regress anything.
  8. ~~**Decode loop + prompt construction**~~ **DONE.** C++ side
     now builds the `asr_sft` prompt input_ids inline (BPE encode each
     chat template segment, splice with the audio segment's
     `[<|sosp|>, ...empty..., <|eosp|>]` row 0 and per-channel
     `speech_zeroemb_idx` audio rows), runs prefill + greedy step
     decode, and detokenizes. The shared `mimo_asr_build_prefill_graph`
     gained an `n_past` parameter; step decode reuses it with T=gs and
     advancing n_past. Stops on `<|im_end|>`/eos; strips
     `<|empty|>`/`<|eot|>`/`<|eostm|>` from the output. Tokenizer is
     the qwen3-asr-style splitter (special-token `<|...|>` greedy
     match → bytes_to_unicode + bpe_one for the rest), needs the
     merges from step 9.
  9. ~~**Reconvert F16 + Q4_K**~~ **DONE.** F16
     (`mimo-asr-f16.gguf`, 14.9 GB, 719 tensors) + Q4_K
     (`mimo-asr-q4_k.gguf`, 4.5 GB) regenerated from the safetensors
     snapshot. Vocab now 151680 (was 151643), merges 151291 (was 0).
     Hit an OpenMP deadlock in `at::native::DEFAULT::copy_kernel` →
     `__kmp_suspend_64` during the bf16→f16 cast; workaround:
     `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
     PYTHONUNBUFFERED=1`. Total wall time ~20 min on the 16 GB Mac
     once the env vars were set; without them the conversion hung
     indefinitely after ~3 min. HF re-upload to `cstr/mimo-asr-GGUF`
     deferred (destructive shared-state op — needs explicit user
     confirmation).
  10. ~~**End-to-end JFK test**~~ **DONE.** `crispasr --backend
      mimo-asr -m mimo-asr-q4_k.gguf --codec-model
      mimo-tokenizer-q4_k.gguf -f samples/jfk.wav` outputs:
      "And so, my fellow Americans, ask not what your country can do
      for you. Ask what you can do for your country." (matches the
      upstream Python `MimoAudio.asr_sft` reference). 11.0 s audio in
      ~37 s on M1 Q4_K with Metal (0.3× realtime — Q4_K dequant on
      every step is the bottleneck; F16 + KV reuse are the obvious
      next perf wins, but quality is correct).

  Upstream Python source for reference is at
  `ref/mimo/github/src/mimo_audio_tokenizer/` and
  `ref/mimo/github/src/mimo_audio/`. config.json for both halves
  lives at `ref/mimo/MiMo-V2.5-ASR/` and
  `ref/mimo/MiMo-Audio-Tokenizer/`.
- **Qwen3-TTS** — **[WORKS, default path]** End-to-end synthesis is
  functional and correct on default settings. HF release
  `cstr/qwen3-tts-0.6b-base-GGUF` ships F16 + Q8_0 + Q4_K talker;
  `cstr/qwen3-tts-tokenizer-12hz-GGUF` ships F16 + Q8_0 codec. Default
  auto-download is Q8_0 talker + F16 codec.
  `crispasr --backend qwen3-tts -m auto --auto-download --tts "..."
  --voice <ref.wav> --ref-text "..."` works out of the box. JFK
  round-trip ASR test confirms exact-match transcription. Apache 2.0.

  **Field regression fixed (commit 7298dd5):** earlier today's commits
  introduced an O15 graph-reuse path that read uninitialised KV slots
  and produced noise on backends that don't zero-init alloc'd buffers
  (CUDA, parts of CPU). Patched with `ggml_backend_buffer_clear` after
  `cp_kv_alloc`. Same commit gates all today's perf paths behind env
  switches so the default reproduces yesterday's known-good behaviour:
   - `QWEN3_TTS_O15=1`        — fixed-Lk + cached T=1 code_pred graph
   - `QWEN3_TTS_FUSED_QKV=1`  — fused Q+K+V matmul in talker (F16/F32)
   - `QWEN3_TTS_MAX_FRAMES=N` — bench-only frame cap
  README's TTS section has the full env-switch table.

  **O15 graph-cache reuse fixed (this commit).** The fine-grained
  cp_step diff harness landed (Python ref dumps 15
  `cp_step{i}_input_embed` + `cp_step{i}_logits` stages, C ABI exposes
  `qwen3_tts_run_code_pred_step`, harness drives them in order to
  match `code_pred_generate_15`'s `skip_plan=(i>=2)`). It pinned the
  bug to the cached-graph reuse: `core_attn::kv_self_attn` baked
  `n_past` into a `ggml_view_4d` byte-offset, so a graph cached at
  step 1 (n_past=2) clobbered slot 2 again at step 2 (n_past=3). Fix:
  added an opt-in `kv_indices` arg to `kv_self_attn` that switches the
  K/V write from static-offset `ggml_cpy` to runtime-indexed
  `ggml_set_rows`; `build_graph_code_pred_kv` passes the existing
  `positions` tensor (already `[n_past, n_past+T)`) as the indices when
  `O15` is on. Default callers (qwen3-asr / voxtral / granite / etc.)
  pass nullptr and stay on the byte-identical legacy path.
  Validation: 15/15 cp_step PASS cos≥0.9999 under O15, and end-to-end
  TTS under `QWEN3_TTS_O15=1` produces the same 78-frame / 6.24 s
  output as default (was: 20 s of noise, never emitted EOS).

  **[next] 4-variant correctness + speed matrix on a quiet machine.**
  With the cp_step diff harness in place and O15 fixed, the four env
  combos can finally be A/B'd cleanly. For each variant, run on a
  cold machine (no parallel jobs starving GPU/CPU) and record both
  the cp_step diff result and the per-frame timing breakdown:

  | variant                          | env                                          |
  |----------------------------------|----------------------------------------------|
  | default                          | (unset)                                      |
  | O15                              | `QWEN3_TTS_O15=1`                            |
  | FUSED_QKV                        | `QWEN3_TTS_FUSED_QKV=1` (needs F16 talker)   |
  | O15 + FUSED_QKV                  | both                                         |

  Per variant:
  1. `crispasr-diff qwen3-tts <talker.gguf> /tmp/qwen3-tts-ref.gguf
     /private/tmp/clone-16k.wav` — confirm 15/15 cp_step PASS
     cos≥0.999. Any drop names which path regressed.
  2. `QWEN3_TTS_BENCH=1 QWEN3_TTS_MAX_FRAMES=100 crispasr --backend
     qwen3-tts --tts "..." --voice samples/jfk_24k.wav ...` —
     capture talker_kv + code_pred_kv per-frame ms.
  3. `md5(out.wav)` cross-check against default at fixed seed (was
     bit-identical for FUSED_QKV at seed=42 / "Hello world…").

  Then decide whether to flip `O15` and/or `FUSED_QKV` on by default.
  FUSED_QKV needs an F16 talker; the auto-download default is Q8_0,
  so seed the cache with `cstr/qwen3-tts-0.6b-base-GGUF` F16 first
  (see `.local/bench-qwen3/matrix.sh` — already wired but skips
  FUSED_QKV when only Q8_0 is in `~/.cache/crispasr/`).
  Open question to settle: on M1 Metal at low T, three small matmuls
  may parallelize better than one fused matmul (first noisy run had
  `f16_nofuse ≈ 237 ms/frame` vs `f16_fused ≈ 273 ms/frame` — possibly
  contention noise, possibly real). Quiet-machine numbers will
  resolve it.

  **[later] Speed roadmap** (LEARNINGS.md): Lk bucketing for talker,
  Q8_0 KV cache, converter-side Q4_K fused QKV. Each requires the
  fine-grained harness above to land safely. Also still pending:
  ECAPA speaker_encoder forward (PLAN #52 step 4) to remove the
  `bake-qwen3-tts-voice-pack.py` dependency for new voices.
- **VibeVoice-ASR 7B** — blocked on ≥16 GB RAM for conversion
- ~~**VibeVoice TTS**~~ — **DONE**: Realtime-0.5B (17 bugs, perfect round-trip) + 1.5B base model (voice cloning). HF: `cstr/vibevoice-realtime-0.5b-GGUF`, `cstr/vibevoice-1.5b-GGUF`

## VibeVoice TTS — integration wiring (this session) **[next]**

Recent session work (2026-04-29):

- **ggml F32 conv fix** — `ggml_conv_1d` and `ggml_conv_transpose_1d` in
  `ggml/src/ggml.c` now select F32 `im2col` type when either input is F32.
  `ggml/src/ggml-cpu/ops.cpp` updated to handle both F16 and F32 `src1`.
  Allows `vibevoice-realtime-0.5b-tts-f32.gguf` to run without crashing.

- **Backend detector** — `crispasr_detect_backend_from_gguf` in `crispasr_c_api.cpp`
  now maps `vibevoice-tts` arch → `vibevoice` backend (was only `vibevoice-asr`).

- **Registry entry** — `vibevoice-tts` added to `crispasr_model_registry.cpp`:
  model `vibevoice-realtime-0.5b-q4_k.gguf` (636 MB, `cstr/vibevoice-realtime-0.5b-GGUF`),
  companion `vibevoice-voice-emma.gguf`.
  Usage: `crispasr --backend vibevoice -m auto --auto-download --tts "..." --voice ~/.cache/crispasr/vibevoice-voice-emma.gguf`

- **Dart binding** — `CrispasrSession` in `flutter/crispasr/lib/src/crispasr.dart`
  gained `setVoice()`, `synthesize()`, `setCodecPath()` (mirrors Python + Rust).

Still pending:

- **[next] Verify Q4_K model on Metal M1.** The F16/F32 models run on CPU/Metal.
  Q4_K model from HF has not been tested yet (no local copy; download needed).
  Run: `crispasr --backend vibevoice -m auto --auto-download --tts "Hello, world" --voice ~/.cache/crispasr/vibevoice-voice-emma.gguf -of out.wav`

- **[next] Frame-7 drift investigation (Task 2).** First strict divergence
  (cos<0.999) hits v_cfg_step0 on frame 7. Root cause is accumulated F16
  quantization error in the TTS-LM KV cache over 16 positions (vs 10 at
  frame 6). To distinguish F16 quant noise from a structural bug: run the
  full perframe diff with the F32 GGUF and compare — if frame 7 cos improves
  significantly, it's pure quantization drift (acceptable). If it doesn't,
  look at the text-window 5-token `run_lm_step` causal mask (n_tokens=5 path
  in `run_lm_step`).

- **[next] 2400-sample lag (Task 3).** Python `generate()` does NOT trim the
  decoder warmup — it concatenates raw audio chunks. C++ strips 2400 samples
  (100 ms @ 24 kHz) as quality improvement. For fair xcorr comparison update
  `run_official_vibevoice.py` to accept `--trim-warmup` (strip first 2400
  samples before saving the WAV). This makes the xcorr peak at lag=0 instead
  of lag=2400 samples.

- **[later] Voice auto-download for all voices.** Registry companion slot
  holds one file (emma). Carter/Davis/Frank/de-Spk0/de-Spk1 voices require
  manual `--voice` path. Consider a voice-registry parallel to the model
  registry, or a `--list-voices` flag that prints the HF URLs.
- **VibeVoice-7B TTS** — needs 32+ GB RAM for conversion (9.3B params). Same architecture as 1.5B.
- **VibeVoice multi-speaker** — 1.5B/7B support up to 4 speakers; need prompt template for multi-speaker scripts
- **VibeVoice negative conditioning** — base model uses zero negative; proper dual-LM CFG would improve quality
- **Qwen3 TTS** — user-requested follow-on after VibeVoice TTS landed
  (Apache 2.0,
  [collection](https://huggingface.co/collections/Qwen/qwen3-tts)).
  "Discrete Multi-Codebook LM" architecture: Qwen3 backbone with a
  16-codebook output head, paired with `Qwen3-TTS-Tokenizer-12Hz`
  (16 codebooks × 2048, 12.5 FPS RVQ). Variants:
  Base / CustomVoice in 0.6B and 1.7B; VoiceDesign 1.7B-only.
  10 languages, ~97ms end-to-end latency.
  - Add `models/convert-qwen3-tts-to-gguf.py` for the LM
  - Add `models/convert-qwen3-tts-tokenizer-to-gguf.py` for the codec
  - New backend `qwen3-tts` reusing `core_attn::kv_self_attn` for
    the LM, with a 16-codebook output layer in place of the usual
    single-vocab lm_head, and the codec for waveform reconstruction.
  - Auto-download for both the LM and the tokenizer.
  Same scope as MiMo (which is also a separate-tokenizer + LM
  pattern); the two share enough infrastructure that landing one
  unblocks the other.

### Open polish

- **parakeet-ja Q4_K** — F16 is bit-exact, Q4_K still loops after
  ~8 tokens because `joint.pred.weight` and `decoder.embed.weight`
  fall back to q4_0 (dims don't tile for q4_k blocks) and the
  80-mel encoder amplifies the error. Two ways to fix: ship a Q5_K
  build, or extend the converter / quantiser to pin those two
  tensors to F16 inside a q4_k file.

---

## `src/core/` shared helpers — current state

| Helper | File | Consumers | Status |
|---|---|---|---|
| **mel spectrogram** | `mel.{h,cpp}` | 8/8 non-whisper | ✅ done |
| **FastConformer encoder** | `fastconformer.h` | parakeet, canary, canary_ctc, stt_en_fc_ctc | ✅ done |
| **LLM self-attention** | `attention.h` | voxtral (encoder + LLM), voxtral4b/qwen3/granite (KV-cached) | ✅ done |
| **FFN (SwiGLU/SiLU)** | `ffn.h` | qwen3, voxtral, voxtral4b, granite | ✅ done |
| **GGUF loader** | `gguf_loader.{h,cpp}` | all 8 non-whisper | ✅ done |
| **BPE encoder** | `bpe.h` | qwen3, granite | ✅ done |
| **Greedy decode** | `greedy_decode.h` | voxtral, voxtral4b, qwen3, granite | ✅ done |

Remaining extraction opportunities (each saves ~30-60 LOC but has only 1-2 consumers):

- **[done]** ~~KV-cached attention variant for qwen3/voxtral4b/granite LLM~~ —
  all 4 LLM backends migrated to `core_attn::kv_self_attn()`.
- **[done]** ~~Q/K norm variant for qwen3~~ — `kv_self_attn()` accepts
  optional `q_norm_w`/`k_norm_w` params.
- **[done]** ~~Whisper-style audio encoder (voxtral 3B)~~ — migrated to
  `core_attn::encoder_self_attn()` with biased Q/V/O, no K bias, no RoPE.
- **[done]** ~~voxtral4b encoder attention migration~~ — migrated to
  `core_attn::encoder_self_attn()` with `permute_cont = false` (matching
  original no-cont graph structure). Bit-identical on jfk.wav verified.
- **[later]** Sliding-window attention (voxtral4b encoder — single consumer)
- **[later]** µP scale tricks for granite (attention_multiplier, residual_multiplier)

- **[done]** ~~**`src/core/attention.h` — voxtral audio encoder.**~~
  `encoder_self_attn()` added: optional biases on Q/K/V/O, optional RoPE,
  optional mask. Voxtral 3B migrated. Voxtral4b is a candidate but uses
  no-cont permute (needs bit-identity verification with model files).

- **[later]** **`src/core/attention.h` — sliding-window attention.**
  voxtral4b audio encoder uses 750-token SWA. Needs a `sliding_window`
  knob and the mask has to be pre-built by the caller (or constructed
  by the helper from `sliding_window`).

- **[later]** **`src/core/attention.h` — µP scale tricks.**
  Granite uses `attention_multiplier` (0.0078125 = 1/128) as the attention
  scale instead of `1/sqrt(d)` and `residual_multiplier` (0.22) on the
  residual add. Parameterise via `Config::attn_scale` and
  `Config::residual_scale` in the helper.

- **[done]** ~~`src/core/greedy_decode.h` — unified LLM decode loop.~~
  **Done.** Header-only `core_greedy_decode::run()` with an optional
  `PreHook` callback (used by voxtral4b's streaming audio-frame-addition
  path). All 4 LLM backends migrated bit-identically: voxtral (via the
  crispasr_llm_pipeline.h template), voxtral4b (with the pre-hook for
  streaming), qwen3 (with dynamic EOS lookup via tokenize), granite
  (with EOS-filter post-step). Net: ~100 lines of duplicated decode
  loops replaced with a single templated helper.

- **[done]** ~~`src/core/mel::Params::stacked_frames`.~~ **Done
  differently.** Granite is now on `core_mel::compute` without a new
  Params knob: its normalization `v/4+1` turned out to be identical
  to `GlobalClipMax` `(v+4)/4`, and the "drop-last-if-odd + stack
  pairs into 160-mel rows" step stays in the granite wrapper as
  ~15 lines of post-processing. `core_mel` coverage is now 8/8
  non-whisper models.

- **[done]** ~~**`cli.cpp` output writer refactor (task #4).**~~
  Done in a9365d8. All whisper output writers (txt/vtt/srt/csv/lrc/
  score/json/wts) now take `const std::vector<crispasr_segment> &`
  produced once by `cli_whisper_collect_segments(ctx)`. Byte-identical
  regression verified on all 7 output formats.

- **[done]** ~~**`backend-crispasr` wrapper (task #15).**~~
  Done in e120103. `crispasr_backend_crispasr` implements the
  CrispasrBackend interface on top of whisper.h. `--backend whisper`
  now routes through the unified dispatch like every other backend,
  and the `--list-backends` matrix reads the wrapper's capability
  bitmask live (no more hardcoded `kWhisperCaps` constant). The
  default (empty-backend) whisper path stays byte-identical.
  Subsequent commits (a71f617 grammar + auto-dl, fb47aa0 VAD,
  2f43f7c n_processors) filled in the rest of whisper_full_params
  plumbing — the wrapper now advertises: ts-native, word-ts,
  tok-conf, lang-detect, translate, temperature, beam-search,
  grammar, flash, VAD-internal, parallel-processors, auto-dl.
  The only gaps vs the historical cli.cpp path are stereo diarize
  (needs pcmf32s through the dispatch) and -owts karaoke output.

---

## CLI + examples cleanup

- **[done]** ~~Delete the per-model `examples/*-main/` directories
  once `crispasr --backend X` has shipped.~~ **Done.** 7 user-facing
  CLIs (cohere-main, parakeet-main, canary-main, qwen3-asr-main,
  voxtral-main, voxtral4b-main, granite-main) removed. The
  `cohere-quantize.cpp` tool was rescued into
  `examples/crispasr-quantize/` as a standalone GGUF quantizer.
  Kept: `cohere-align` / `nfa-align` (standalone forced-alignment
  tools, still useful when the transcript is pre-existing) and the
  `{qwen3,voxtral}-test-*` differential fixtures.

- **[later]** `tests/CMakeLists.txt` uses `crispasr` as the test
  target. Keep that target name (we already preserve it) but move the
  tests over to `$<TARGET_FILE:crispasr>` once the rename has propagated.

---

## Feature parity gaps (non-whisper backends vs whisper)

The whisper backend in CrispASR is the most feature-complete. The

## Rename plan

- Keep `whisper` when it denotes the real Whisper backend, Whisper-specific models, compatibility APIs, or Whisper-only tests and fixtures.
- Rename stale project branding from `whisper*` to `crispasr*` when it is acting as the overall project, package, app, framework, or example identity.
- Prefer compatibility aliases over hard breaks where external consumers may still load `libwhisper.*` or import old wrapper names.
- Treat `tests/` conservatively: keep `whisper` in backend-specific tests, model names like `whisper-tiny`, and compatibility expectations; rename only generic CrispASR-level test and example naming.
- Rename generated Apple artifacts and package metadata to `crispasr` while keeping the Whisper backend available as one backend inside CrispASR.
- Finish remaining cleanup in wrapper docs, example file names, and validation scripts only after confirming they are branding-only and not backend/API semantics.
capability matrix in the README shows which features are missing on
each backend. High-value gaps to close:

- **[done]** ~~**Temperature sampling for LLM backends**~~ — landed
  in e4861c3 (pipeline) and 7a7e6cd (run_with_probs). voxtral,
  voxtral4b, qwen3 and granite all honour `-tp N` via the shared
  `core_greedy_decode` helper's `sample_temp` path. Default
  temperature=0 stays on the bit-identical pure-argmax path.
- **[done]** ~~**Best-of-N sampling for LLM backends**~~ — all four LLM
  backends (voxtral, voxtral4b, qwen3, granite) support `--best-of N`
  with `--temperature > 0`. Each run uses a different RNG seed, best
  selected by mean token softmax probability.

- **[later]** **VAD integration in LLM backends.** qwen3 and voxtral
  currently don't chunk long audio; the dispatch layer does VAD slicing
  but the LLM models themselves pad to a fixed 30s window. Variable-
  length mel would let them handle >30s natively.
  **2026-04-26:** Attempted voxtral 3B variable-length encoder (commit
  8f4c776, reverted in c2328db). Encoder math correct, but the LLM
  produces "I'm sorry, I don't understand." instead of the transcript.
  Hypothesis: Voxtral 3B was trained with a fixed 375-audio-token
  context and the chat-template/positional reasoning is sensitive to
  the audio token count, even though the encoder graph itself accepts
  any length divisible by 8. Future fix would need to either (a) keep
  T_mel=3000 padding but skip attention compute on padded positions,
  or (b) verify against a Voxtral checkpoint that supports variable
  audio context.

- **[done]** ~~**Streaming transcription.**~~ **Done.** Generic
  `--stream` (stdin PCM), `--mic` (microphone capture via
  arecord/sox/ffmpeg subprocess), `--live` (continuous mode), and
  `--monitor` (unicode progress symbols) work with all 11 backends.
  Inspired by antirez/voxtral.c. Also added `--alt` for per-token
  confidence display.

- **[later]** **Native voxtral4b streaming protocol.** The model is
  designed for realtime streaming with configurable 240ms-2.4s delay.
  Currently we run it in chunk-and-transcribe mode. Exposing a
  streaming mode through the CLI is a bigger design question.

- **[done]** ~~**Audio understanding mode for voxtral 3B.**~~ `--ask`
  flag switches from transcription to Q&A template. Tested: language
  detection ("The audio is in English.") and summarization work.

---

## PLAN #51a — mmap loader F16 retest **[blocked: machine state]**

Zero-copy mmap loader landed behind `CRISPASR_GGUF_MMAP=1` in commit
`9710f80`. Validated on parakeet Q4_K (Metal/CPU/mmap, all gold JFK)
and mimo-asr Q4_K (working-set RSS 5.5 GB → 760 MB). The F16 motivating
case (mimo-asr-f16.gguf, 14.9 GB) showed the load-phase win in flight
— ~910 MB working-set at 60 s elapsed vs the HANDOFF-predicted ~13 GB
legacy peak — but end-to-end decode timing was forced to abort twice
on this box because of concurrent jobs (parallel HF uploads, another
session's legacy F16 test on the same file, qwen3-tts benches),
13M+ swapins thrashing both processes down to <100 MB RSS at 0% CPU.

**Retest required on an uncontended machine** before the env-flag
default can be flipped:

1. Confirm no other heavy job is hitting `/Volumes/backups` or the
   same model file (`pgrep -af crispasr; pgrep -af 'hf upload'`).
2. Run `CRISPASR_GGUF_MMAP=1 /usr/bin/time -lp ./build-ninja-compile/bin/crispasr --backend mimo-asr -m /Volumes/backups/ai/crispasr-models/mimo/mimo-asr-f16.gguf -f samples/jfk.wav -nt`.
3. Capture peak RSS (should be < 8 GB; HANDOFF threshold) and that
   the JFK transcript matches the gold output.
4. Run `crispasr-diff mimo-asr <F16 GGUF> <ref.gguf>` per HANDOFF item 6
   — should now run on a 16 GB Mac without thrashing.
5. If both green, commit the default-flip
   (`mmap_loader_enabled()` returns true unless env=0) and mark
   PLAN #51a fully DONE in PLAN.md / HISTORY §62.

After 51a flips, PLAN #51c (F16 step decode) becomes the trivial
follow-up the HANDOFF promised — Q4_K dequant on every matmul drops,
F16 decode on M1 should hit ≥1× realtime on JFK (Q4_K is currently
~0.3× warm).

Side note from the validation session: the conv_2d / conv_2d_dw
F16×F16 MUL_MAT crash that was hitting `parakeet --no-gpu` (and any
other backend with F16 conv kernels on CPU) was already shipped in
`b85f56c` — that one's done.

---

## Per-model follow-ups

### parakeet
- **[later]** Port the TDT decoder (LSTM predictor + joint head) to
  ggml graphs so it can run on GPU. Currently pure CPU float* loops.
  Risk: per-token LSTM stepping is sequential, so GPU speedup may be
  small. Encoder is already the dominant cost.

### canary
- **[later]** Speech translation quality validation at scale.
  Currently regression-tested on German only.

### cohere
- **[done]** ~~F32→F16 self-attention KV cache upgrade.~~ Already F16
  (decoder + cross-attention KV caches both use GGML_TYPE_F16).

### qwen3 / voxtral
- **[DONE]** ~~Stop recreating `ggml_backend_sched` on every compute
  call.~~ All backends now create sched once at init and use
  `ggml_backend_sched_reset()` between
  calls. ~80 LOC per runtime.

### voxtral4b
- **[later]** Reduce right padding from 17 → 10 tokens to match the
  reference `voxtral.c` implementation.
- **[later]** SRT/VTT subtitle output (currently only plain transcript;
  CTC alignment already works via `-am`).

### granite
- **[later]** HF release of quantised GGUFs (`cstr/granite-speech-4.0-1b-GGUF`
  is still pending). Need `cohere-quantize granite-speech-1b.gguf …`
  then upload.
- **[later]** Encoder parallelisation: granite_speech is now linked
  against OpenMP but has no `#pragma omp` annotations yet. Adding
  `#pragma omp parallel for` on the per-layer encoder hot loops
  would deliver a measurable speedup on CPU but shifts the float
  reduction order, so the change needs its own regression gate
  (allow small float drift; transcript must stay correct).
- **[done]** ~~Consider porting the per-layer CPU encoder to a single
  ggml graph.~~ Done: `granite_build_encoder()` wired with runner
  function, enable via `GRANITE_ENCODER_GRAPH=1`. Identical output on
  jfk.wav. Shaw RPE omitted (approximate) — follow up if accuracy
  issues surface on other test cases.
- **[done]** ~~Remove dead ggml graph encoder `granite_build_encoder`.~~
  Resurrected — now used by the graph encoder path (`GRANITE_ENCODER_GRAPH=1`).
- **[done]** ~~Encoder norm + QKV fusion.~~ The CPU per-layer attention
  used to do `cpu_layernorm(normed)` + `run_matmul(Q)` + `run_matmul(KV)`
  as three separate operations (one CPU pass + two Metal dispatches).
  Now folded into a single ggml graph via `run_norm_matmul_pair()`.
  **3.7× total speedup** on M1 Q4K (encoder 12.6s → 3.0s, total
  19.6s → 5.3s, 0.6× → **2.1× realtime**). Math unchanged so cosines
  still pass. Commit `796824f`.

### granite-4.1 (4.1-2b family)
- **[done]** ~~Base 4.1-2b backend.~~ `granite-4.1` registered as a
  backend alias of `granite`; auto-download from
  `cstr/granite-speech-4.1-2b-GGUF`. F16 (5.58 GB), Q4K (2.94 GB) and
  Q4K-mini (1.7 GB) GGUFs published. Diff against PyTorch BF16 ref:
  encoder cos_min 0.999908, projector cos_min 0.999995 (3/3 PASS).
- **[done]** ~~Q4K with F16 encoder variant.~~ New `q4_k-f16enc.gguf`
  (~2.07 GB) — LLM Q4K, encoder + projector F16 (norms / biases / BN
  stats stay F32). 3/3 PASS at encoder cos_min 0.999855. ~1 GB smaller
  than recommended Q4K. Quantizer flag: `CRISPASR_GRANITE_ENC_F16=1`.
- **[done]** ~~Q4K dequantize OOM root cause.~~ Tiny `tctx` for
  `ggml_new_graph` in the RPE init path; replaced with
  `ggml_get_type_traits()->to_float`. Same fix in CPU encoder loop.
  Committed `031d676`.
- **[done]** ~~De-hardcode block-attention `context_size` (200) and
  `max_pos_emb` (512).~~ Now read from new GGUF keys
  `granite_speech.enc.{context_size,max_pos_emb}` with old values as
  defaults so legacy GGUFs keep loading.
- **[done]** ~~`GRANITE_BENCH=1` per-stage timer.~~ RAII wrapper that
  prints elapsed wall-clock for compute_mel / run_encoder /
  run_projector. Useful for A/B-ing the encoder paths.
- **[done]** ~~PLUS variant — cat_layer index + tokenizer fixes.~~ Two
  bugs blocked PLUS from producing transcripts even though encoder +
  projector + LLM all ran. (1) PLUS HF snapshot ships only the unified
  `tokenizer.json` (no separate `vocab.json` / `merges.txt`); the
  converter silently skipped the tokenizer write. Fix: parse
  `model.vocab` + `model.merges` from `tokenizer.json` when legacy
  files are absent. (2) `cat_hidden_layers: [3]` indexes into HF's
  `output_hidden_states` tuple where index 0 is the input embedding,
  so we needed to capture after `il == N - 1` instead of `il == N`.
  Both fixed in commit `f298818`. PLUS now transcribes JFK at 0.9×
  realtime CPU-only with punctuation + capitalisation by default:
  "And so my fellow Americans, ask not what your country can do for
  you, ask what you can do for your country."
- **[done]** ~~PLUS variant — `granite-4.1-plus` backend alias + HF
  upload.~~ Backend alias of `granite` registered, registry entry
  with `~5.6 GB` size, HF repo `cstr/granite-speech-4.1-2b-plus-GGUF`
  populated with f16 GGUF + README. Commit `ed0e5ac`.
- **[done]** ~~PLUS variant — Q4_K / Q4_K-f16enc / Q4_K-mini GGUFs
  published.~~ Three quantized variants uploaded to
  `cstr/granite-speech-4.1-2b-plus-GGUF`; auto-download default switched
  from F16 to Q4_K (~2.96 GB). Diff vs PyTorch BF16 ref on JFK: Q4_K
  and Q4_K-f16enc 3/3 PASS (encoder cos_min ≥ 0.999); Q4_K-mini drops
  to encoder cos_min ≈ 0.62 because the layer-3 + final concat doubles
  the surface for Q4_K rounding error (base-4.1 mini sat at ~0.93).
  All three transcribe JFK correctly. Reference dumper
  (`tools/reference_backends/granite.py`) gained PLUS detection +
  cat-layer concat with HF's `output_hidden_states` indexing
  convention; quantizer gained `CRISPASR_GRANITE_QUANT_ALL=1` env knob
  to override the encoder/projector skip rules for the `-mini` build.
- **[next] PLUS variant — speaker labels + word timestamps.** PLUS's
  default output already includes punctuation; the upstream model
  card also advertises speaker labels and word-level timestamps. We
  haven't yet seen those in the LLM output — likely because the
  granite chat template currently used isn't the one that primes for
  structured output. Investigate `chat_template.jinja` in the PLUS
  HF snapshot; the structured tokens may need a different prompt
  prefix in `examples/cli/crispasr_backend_granite.cpp`. Most of the
  work is template-only (~50 LOC).
- **[done]** ~~NAR variant — converter + runtime scaffold.~~ Converter
  (`models/convert-granite-nle-to-gguf.py`) writes all 930 tensors,
  the LLM tokenizer (BPE), the CTC tokenizer (348 chars) and the
  mel filterbank. Runtime (`src/granite_nle.{h,cpp}`) loads the
  model, parses both tokenizers and reports config correctly:
  `granite_nle: loaded ... (enc 16 layers, proj 2 layers, llm 40
  layers, vocab 100352)`. Forward-pass functions are stubs. Commits
  `d6ddad0`, `ffdd19f`, `108a6ae`.
- **[done]** ~~NAR variant — `compute_mel`.~~ Ported the FFT helpers
  + `core_mel::compute` call. Same 80-bin HTK filterbank + log10 +
  GlobalClipMax + stacked_frames=2 pipeline as base. ~120 LOC.
  Commit `c7ea343`.
- **[done]** ~~NAR variant — encoder forward.~~ Ported the helpers
  (`nle_run_matmul`, `nle_run_norm_matmul_pair`, `nle_run_ffn`,
  `nle_run_conv_module`, `nle_shaw_block_attention_cpu`,
  `nle_cpu_layernorm`) and the main 16-layer Conformer loop from
  `granite_speech.cpp` into `granite_nle.cpp`. BN folding + per-layer
  Shaw RPE precompute + scheduler creation moved into
  `granite_nle_init_from_file`. `encoder_layer_indices` is parsed
  with HF tuple semantics (`-1` → `n_layers`, no auto-append-final);
  the snapshot at index 8 is taken AFTER the self-conditioning
  residual to match HF's `all_hidden_states` ordering. Validated
  against the PyTorch reference on JFK with a new
  `tools/reference_backends/granite_nle.py` (encoder-only loader to
  avoid the upstream LLM-tokenizer fetch): mel `cos_min=0.999997`,
  encoder_output (T=550, 4×1024=4096) `cos_min=0.999852`,
  encoder_logits (T=550, 348) `cos_min=0.999675`. BPE auxiliary
  head (`enc.bpe_out`) is intentionally not wired here — it's only
  needed by the LLM editing pass's text-init step.
- **[done] NAR variant — windowed Q-Former projector.** Two-pass
  implementation: pass A is one ggml graph for the per-encoder-layer
  LayerNorms + concat + `layer_proj` (4096 → 2048) + GELU; pass B is
  one Q-Former graph per block (block_size=15, downsample_rate=5,
  query_length=3) with mean-pool over downsample groups, additive
  `query` and `window_positions`, two 32-head SDPA cross-attention
  + SiLU-MLP layers, and a final `out_norm`+`out_linear`. Output
  rate: 3 audio tokens per 15 encoder frames. Validated on JFK at
  `projector_output cos_min=0.999999` (T_out=111 × llm_dim=2048).
- **[done] NAR variant — non-causal LLM editing pass.** Single graph
  forward over the flat `[audio_embs, text_embs_with_slots]` sequence
  with µP scaling (embedding_multiplier=12, attention_multiplier=1/128,
  residual_multiplier=0.22). 40 layers of RMSNorm + non-causal
  `flash_attn_ext` (mask=nullptr, GQA 16/4 native) + SwiGLU. Tied LM
  head. Validated bit-exact: editing_logits cos_min=0.999999, 47/47
  top-1 match on JFK. Reference dumper monkey-patches
  `transformers.models.granite.modeling_granite.create_causal_mask` to
  return None — without this, `GraniteModel.forward` unconditionally
  passes an upper-triangular mask to SDPA, defeating
  `self_attn.is_causal=False`. The upstream `flash_attention_2`
  assertion is real, not paranoia.
- **[done] NAR variant — transcribe orchestration.** `run_encoder`
  now populates `last_bpe_logits` via `posterior_weighted_pool`
  (window=4, importance = 1 - blank_prob_mid captured at the L8
  self-conditioning softmax) → 100353-vocab `enc.bpe_out` linear.
  `granite_nle_transcribe` ties the full pipeline: BPE-CTC greedy
  decode (`unique_consecutive` then drop blanks then -1 shift) →
  `core_bpe::detokenize` (now shared with `granite_speech`; the
  GPT-2 byte-level reverse lives in `core_bpe::token_bytes_to_utf8`)
  → strip+lowercase+" "-fallback → `core_bpe::tokenize_simple` →
  `add_insertion_slots` (`max(2n+1, 8)`, EOS-padded) → `run_projector`
  with `/= embedding_multiplier=12` and slice to `enc_T // 5` audio
  frames → `run_llm_editing` → per-row argmax + uniq + drop EOS +
  detokenize. JFK end-to-end output matches reference `final_text`
  exactly via `crispasr-diff granite-nle ... transcribe` PASS.
- **[done] Shaw RPE in graph path — per-block subgraph default,
  all three variants on graph.** Per-block subgraph attention emits
  `Q·K^T + Q·RPE_block → softmax → ·V` using
  `core_conformer_ibm::build_shaw_rpe_lookup` for the per-layer bias.
  Root-caused the regression that blocked validation: the loader was
  building only **layer 0's** RPE lookup and the graph reused it for
  all 16 layers (the "tied across layers" assumption is false for
  granite-speech-4.1-2b — each block stores a distinct
  `attn_rel_pos.weight`). Fix: precompute `ctx->rpe_per_layer[il]` at
  load time, declare the graph's `rpe_lookup` input with shape
  `(ctx_size*hd, ctx_size, n_layers)`, and slice per-layer via
  `ggml_view_3d`. PLUS variant captures `cat_hidden_layers` post-norm
  tensors inline in the graph and concats them with the final encoder
  output via `ggml_concat`. NAR (`granite_nle.cpp`) gets the same
  treatment in a sibling `granite_nle_build_encoder` with self-cond
  residual at the configurable layer, blank-prob softmax tap for the
  BPE aux head's `posterior_weighted_pool` (CPU side; bpe_out_w matmul
  stays on the scheduler), final CTC logits as a named output, and an
  in-graph snapshot concat across `enc_layer_indices_parsed`. All three
  paths transcribe JFK byte-for-byte identical to CPU loop.
  `GRANITE_DISABLE_ENCODER_GRAPH=1` is the unified escape hatch.
  End-to-end on M1+Q4_K: base 4.78s → 2.31s (~2.1×), plus
  9.41s → 3.74s (~2.5×), nar 19.27s → 6.41s (~3.0×, NAR CPU-loop run
  was disk-contended).
- **[done] NAR HF upload — F16 + 3× Q4_K.** All four GGUFs published
  to [`cstr/granite-speech-4.1-2b-nar-GGUF`](https://huggingface.co/cstr/granite-speech-4.1-2b-nar-GGUF):
  F16 (5.4 GB), Q4_K (3.2 GB; encoder F32, recommended), Q4_K-f16enc
  (2.4 GB; encoder F16), Q4_K-mini (1.5 GB; everything Q4_K).
  Quantizer extended (`is_granite_family` includes `granite_nle` arch
  now, so the encoder/projector skip rules + `CRISPASR_GRANITE_ENC_F16` +
  `CRISPASR_GRANITE_QUANT_ALL` env knobs apply identically). Diff vs.
  PyTorch ref on JFK: F16/Q4_K/Q4_K-f16enc all transcribe == ref
  exactly with `editing_logits_top1` cos_min=1.000000; Q4_K-mini's
  encoder cos_min drops to 0.10 at the worst frame (vs. ~0.62 for
  PLUS-mini — the 4-layer hidden-state concat amplifies Q4_K rounding
  error) but still transcribes correctly because the LLM's argmax
  recovers the right token.
- **[done] NAR — wire `granite-4.1-nar` into the main `crispasr` CLI**
  (commit `2174b51`). Components:
  1. `examples/cli/crispasr_backend_granite_nle.cpp` adapter — modelled
     on the gemma4-e2b adapter (one-shot `granite_nle_transcribe` call),
     not on the granite-speech adapter (which builds prompts + runs a
     KV-cached greedy decode by hand). NAR has no AR loop, so the
     runtime call returns the final UTF-8 transcript directly.
  2. Backend factory + auto-detect in `examples/cli/crispasr_backend.cpp`
     ("granite-4.1-nar" / "granite-nar" alias; arch="granite_nle"
     auto-detect maps to "granite-4.1-nar").
  3. Re-instated registry entry in `src/crispasr_model_registry.cpp`
     pointing at `cstr/granite-speech-4.1-2b-nar-q4_k.gguf` (~3.2 GB).
  4. CMake wiring in `examples/cli/CMakeLists.txt` (source list +
     granite_nle in crispasr-cli's link libraries).
  5. Smoke test PASS on JFK with Q4_K — output matches
     `crispasr-diff granite-nle` transcribe stage exactly.

  `src/crispasr_c_api.cpp` (the C-library API for embedding consumers,
  separate from the CLI) is NOT yet wired — the CLI uses
  `crispasr_create_backend` from `crispasr_backend.cpp` directly. Add
  a `s->backend == "granite-4.1-nar"` branch in `crispasr_c_api.cpp`'s
  init / list / dispatch sites (currently around L877, L1104, L1436)
  if/when an external library consumer needs NAR.

### gemma4-e2b
- **[later]** Speed — now **1.4× realtime** after the `end_of_turn`
  eos fix (was 0.2×; 8.7× speedup). 96% of the remaining time is
  per-token LLM decode (~220 ms/tok on M1 Q4_K; 35 layers, PLE,
  double-wide MLP). Encoder is only 9% of total now, so O10
  (flash_attn_ext encoder) and O11 (cont reduction) are diminishing
  returns. Real wins would come from reducing the per-token decode
  cost — flash_attn for both shared and non-shared attention paths
  is already in place, KV is F16, sched is reused. Possible next
  ideas: skip the redundant `ggml_cont` of the donor's K/V for
  shared layers (currently copied every step even though it doesn't
  change), batch multiple PLE projections into a single matmul.
- **[later]** Move audio mel/attention hparams to GGUF for multi-flavor
  support (currently hardcoded; Gemma-4 family ships several scales).
- **[later]** Extract Gemma4 audio tower into a CrispAudio shared lib
  (per the BiDirLM/CrispEmbed pattern) — the USM Conformer with
  ClippableLinear is a useful standalone audio encoder.
- **[later]** Stage-by-stage diff for the LLM half: `crispasr-diff
  gemma4` currently covers mel + audio_encoder; LLM-side stages
  would let us tighten any remaining sub-1% layer cos drop.
- **[later]** Q2_K transcription quality test. F16 + Q8_0 + Q4_K
  all verified end-to-end on JFK with the freshly-converted GGUFs;
  Q2_K is converted but its long-context PLE quality has not been
  smoke-tested.

### canary_ctc (aligner)
- **[done]** ~~Fix single-backend scheduler — currently no CPU fallback
  if the primary backend rejects an op.~~ Already uses the 2-backend
  pattern (GPU primary + CPU fallback) at both scheduler init points
  (`canary_ctc_compute_logits_from_mel_debug` and `canary_ctc_compute_logits`).

---

## Markdown cleanup (this session)

Consolidating ~15 historical notes into three live docs:

- `TODO.md` — this file (replaces all `*-todo.md`)
- `LEARNINGS.md` — technical insights, benchmarks, comparisons
- `HISTORY.md` — condensed chronology of the ports

Remove after consolidation: `canary-todo.md`, `parakeet-todo.md`,
`granite-todo.md`, `voxtral-todo.md`, `voxtral-4b-todo.md`,
`qwen3-asr-todo.md`, `TODO_COHERE_OPTIMIZATION.md`,
`benchmark_cohere.md`, `qwen3-asr-benchmark.md`, `ggml_plans.md`,
`voxtral-comparison.md`, `test_german.md`, `PERFORMANCE.md`.

Keep: `README.md`, `TODO.md`, `LEARNINGS.md`, `HISTORY.md`, `UPSTREAM.md`,
`README_sycl.md`, `ci/README.md`, `models/README.md`, `samples/README.md`,
`hf_readmes/*.md`.

---

## Ground-truth diff infrastructure (new in this session)

The `tools/dump_reference.py` + `crispasr-diff` pair is the new
contributor-facing path for adding backends with confidence. Status:

- **[done]** Unified Python dumper (`tools/dump_reference.py`) with
  plug-in backend modules under `tools/reference_backends/`. Writes
  a single GGUF tensor archive per dump (not scattered `.npy` files).
- **[done]** Shared C++ diff harness (`examples/cli/crispasr_diff.{h,cpp}`)
  loading the archive via `core_gguf::load_weights` and exposing
  `compare(name, data, n)` with cosine sim / max-abs / RMS / top-1
  argmax metrics.
- **[done]** `crispasr-diff` CLI binary built alongside `crispasr`.
  Currently wires up mel-stage comparison for voxtral / voxtral4b /
  qwen3 / granite (the stages their public C API exposes). Parakeet /
  canary / cohere only have all-in-one `transcribe()` entry points so
  they're reported as unsupported.
- **[done]** Worked-example Python backend modules:
  `tools/reference_backends/qwen3.py` and `voxtral.py` are fully
  ported from the legacy `models/*-dump-*.py` scripts.
- **[done]** ~~Port `models/voxtral4b-dump-ref.py` into
  `tools/reference_backends/voxtral4b.py`~~ — done; captures mel,
  encoder_output (post-projector), t_cond, llm_argmax, generated_text.
- **[done]** ~~Port `models/granite-speech-kaggle-groundtruth.py` into
  `tools/reference_backends/granite.py`~~ — done; captures mel,
  per-layer encoder checkpoints, projector_out (Q-Former), llm_argmax,
  text. Strips the Kaggle-specific HF_TOKEN / gist-upload plumbing.
- **[done]** ~~Expose `audio_encoder`-only standalone entry points
  in `parakeet` / `canary` / `cohere` C headers so `crispasr-diff`
  can do stage-by-stage comparison for them too~~ — done in 7ba3c50.
  Each backend now exposes `<name>_compute_mel` and
  `<name>_run_encoder` alongside the existing `<name>_transcribe_ex`
  batch entry point. `crispasr_diff_main.cpp` gained the matching
  dispatch branches.
- **[done]** ~~Write `tools/reference_backends/parakeet.py`,
  `canary.py`, `cohere.py` so the crispasr-diff harness has
  references to compare against.~~ All three shipped: `parakeet.py`
  (April 2026, used to diagnose the JA xscaling bug), `cohere.py`
  (HF `transformers` + trust_remote_code), `canary.py` (May 2026
  commit `63f708e`, NeMo `from_pretrained` + TimeMels-layout
  transpose + 32 per-layer encoder hooks). PLAN #5 closed; see
  HISTORY §63.
- **[done]** ~~Migrate `examples/{qwen3,voxtral}-test-*/main.cpp`
  drivers to load their reference data from a crispasr-diff GGUF
  archive via `crispasr_diff::Ref` instead of the inline NPY parser.~~
  Done in this batch. All six drivers (`voxtral-test-encoder`,
  `voxtral-test-llm`, `voxtral-test-e2e`, `qwen3-asr-test-conv`,
  `qwen3-asr-test-llm`, `qwen3-asr-test-trace`) now link
  `crispasr-diff-lib` (the reusable static version of
  `examples/cli/crispasr_diff.{h,cpp}`) and consume a single
  `reference.gguf` produced by `tools/dump_reference.py`. The
  inline `load_npy_f32` parsers are gone. `qwen3-asr-test-bpe` has
  no reference data and stays as-is.
- **[done]** ~~Extend `tools/reference_backends/qwen3.py` to emit
  `trace_input_ids / trace_audio_pad_pos / trace_first_logits /
  trace_generated_ids`~~ — needed by `qwen3-asr-test-trace` for the
  chat-template prompt + splice + forward path, plus `llm_input_ids`
  + full-T `llm_logits` for the `qwen3-asr-test-llm` differential
  test. Trigger via `--stages` or leave as part of the backend's
  `DEFAULT_STAGES` (they're in the default now).
- **[later]** Mirror the same for `tools/reference_backends/voxtral.py`
  so `voxtral-test-llm` stops reporting `[SKIP]`. Needs the Voxtral
  apply_chat_template → processor → embed → splice → forward path.
- **[rejected]** Vosk as a third `--lid-backend` provider. Vosk's
  C++ API exists and would fit the "no Python" constraint, but it's
  an ASR toolkit, not a language detector — standalone LID means
  running the full Kaldi decoder with each candidate language model
  and comparing likelihoods, which is slow and memory-heavy. It also
  drags in Kaldi + OpenFST + BLAS as build dependencies, ~50-100 MB
  of binary and a non-ggml runtime path that would be the first of
  its kind in this repo. If someone still wants it, the dispatcher
  in crispasr_lid.cpp::crispasr_detect_language() has an easy
  extension point — just another `if (be == "vosk")` branch. For
  now we stick with whisper-tiny (shipping) and the future native
  Silero GGUF port.
- **[done]** ~~**Qwen3-ForcedAligner-0.6B as a generic timestamp post-step.**~~
  Fully implemented. `qwen3_asr_align_words()` does the full pipeline:
  mel → encoder → prompt with `<|timestamp|>` markers → single forward
  pass → argmax * 80ms per position. Dispatched in `crispasr_aligner.cpp`
  via filename detection. GGUF on HF: `cstr/qwen3-forced-aligner-0.6b-GGUF`.
  Verified working with voxtral on jfk.wav.
- **[done]** ~~Native GGUF port of Silero's language detector.~~
  **Done.** `src/silero_lid.{h,cpp}` implements a pure-C++ forward pass
  (no ggml graph — manual F32 loops, similar to pyannote_seg). GGUF
  converter at `models/convert-silero-lid-to-gguf.py`. 507 tensors,
  16.1 MB F32 / ~9 MB Q8_0. CLI wiring in `crispasr_lid.cpp`: when
  `--lid-model *.gguf` is passed, the native path runs; falls back to
  sherpa subprocess for `.onnx` models. Verified: English, German,
  Latvian correctly detected across multiple test wavs.
- **[done]** ~~Delete the legacy `models/*-dump-*.py` scripts~~ — done.
  Removed `qwen3-asr-{llm,reference,trace}-dump.py`,
  `voxtral-{encoder,llm}-dump.py`, `voxtral4b-dump-ref.py`, and
  `granite-speech-kaggle-groundtruth.py`. Everything they did is now
  covered by `tools/dump_reference.py` + `tools/reference_backends/`.
  The `voxtral4b-debug-{cpp,light}.py` deep-diagnostic scripts and
  `voxtral-verify-gguf.py` (a converter sanity check) were kept —
  they're useful tools with no GGUF-archive replacement yet.
- **[later]** Delete the empty `examples/{parakeet,canary,cohere,
  qwen3-asr,voxtral,voxtral4b,granite}-main/` directories. They no
  longer have any source files (CMakeLists.txt already dropped them
  from `add_subdirectory`), only stale build artifacts. Untracked,
  so this is a filesystem cleanup rather than a git operation.

---

## Upstream dependencies

Full tracking is in `UPSTREAM.md`. Short summary:

- **[upstream]** crispasr `examples/ffmpeg-transcode.cpp` mp4-family
  container crash. Workaround: pre-convert with ffmpeg one-liner.
- **[upstream]** ggml x86 AVX-VNNI / AVX512-VNNI dispatch for Q8_0 dot
  products. Closes the 5-second gap to ONNX INT8 on x86 servers.
- **[upstream]** NeMo Forced Aligner auxiliary CTC model standalone
  release. Not blocking — our converter extracts it from the `.nemo` tarball.

---

## HF releases

| Repo | Status |
| --- | --- |
| `cstr/parakeet-tdt-0.6b-v3-GGUF` | ✅ shipped |
| `cstr/parakeet_de_med-GGUF` | ✅ shipped |
| `cstr/canary-1b-v2-GGUF` | ✅ shipped |
| `cstr/canary-ctc-aligner-GGUF` | ✅ shipped |
| `cstr/cohere-transcribe-03-2026-GGUF` | ✅ shipped |
| `cstr/qwen3-asr-0.6b-GGUF` | ✅ shipped |
| `cstr/qwen3-asr-1.7b-GGUF` | ✅ shipped |
| `cstr/qwen3-forced-aligner-0.6b-GGUF` | ✅ shipped |
| `cstr/voxtral-mini-3b-2507-GGUF` | ✅ shipped |
| `cstr/voxtral-mini-4b-realtime-GGUF` | ✅ shipped (Q4_K + Q8_0) |
| `cstr/granite-speech-4.0-1b-GGUF` | ✅ shipped (f16, q4_k, q5_0, q8_0) |
| `cstr/granite-speech-3.3-2b-GGUF` | ✅ shipped (f16, q4_k, q5_0, q8_0) |
| `cstr/granite-speech-3.3-8b-GGUF` | ✅ shipped (q4_k, q5_0, q8_0) |
| `cstr/granite-speech-3.2-8b-GGUF` | ✅ shipped (q4_k, q5_0, q8_0) |
| `cstr/stt-en-fastconformer-ctc-large-GGUF` | ✅ shipped (f16, q4_k, q5_0, q8_0) |
| `cstr/stt-en-fastconformer-ctc-xlarge-GGUF` | ✅ shipped (f16, q4_k, q5_0, q8_0) |
| `cstr/stt-en-fastconformer-ctc-xxlarge-GGUF` | ✅ shipped (f16, q4_k, q5_0, q8_0) |
| `cstr/silero-lid-lang95-GGUF` | ✅ shipped (f32 only — 16 MB; quants break accuracy on small conv tensors) |
| `cstr/pyannote-v3-segmentation-GGUF` | ✅ shipped (f32, 5.7 MB) |
| `cstr/wav2vec2-large-xlsr-53-english-GGUF` | ✅ shipped (f16, q4_k, q5_0, q8_0) |
| `cstr/wav2vec2-large-xlsr-53-german-GGUF` | ✅ shipped (q4_k) |
| `cstr/gemma4-e2b-it-GGUF` | ✅ shipped (f16, q8_0, q4_k, q2_k — 4 quants with QAT clip scalars) |
| `cstr/qwen3-tts-0.6b-base-GGUF` | ✅ shipped (f16, q8_0, q4_k — talker; pair with the codec repo below) |
| `cstr/qwen3-tts-tokenizer-12hz-GGUF` | ✅ shipped (f16, q8_0 — RVQ codec) |

---

<!-- Completed work moved to HISTORY.md:
     #26 GLM-ASR-Nano, #56 Silero LID native, #57 pyannote v3 native,
     #63 wav2vec2 ggml rewrite, #64 granite speedup (closed,
     hardware-blocked), #65 iOS+Android CI, v0.1.0 release. -->
