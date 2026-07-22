# Tiron — multi-speaker meeting ASR (Whisper large-v3 + inline speaker tokens)

Issue: https://github.com/CrispStrobe/CrispASR/issues/295
Model: https://huggingface.co/Trelis/tiron  (Apache-2.0, 2B bf16, `model.safetensors` 3.09 GB)
Harness (ref, Apache-2.0): https://github.com/TrelisResearch/tiron

Branch: `feat/tiron-asr` · worktree `.claude/worktrees/feat-tiron`

---

## NOW — active work (2026-07-22, validated against blueprint reference)

- **Conversion DONE:** `cstr/tiron-GGML` = f16 + q4_k (via `crispasr-legacy-quantize`
  — the whisper-bin quantizer; `crispasr-quantize` is GGUF-only and rc=1s).
- **Reference DONE:** `cstr/crispasr-regression-fixtures/tiron/multispeaker/ref.gguf`
  — dumped from the ACTUAL harness recipe (`engine.py`: `decoder_input_ids=[[sot,
  lang,transcribe]]`, generation config disabled, 0.75 s onset pad, constraint
  grammar). Earlier model-card-guess recipe (`language=`/`task=`) injected
  `<|notimestamps|>` → repeat loop; fixed.
- **Runtime DONE + validated:** decode grammar (`whisper_tiron_apply_grammar`,
  port of `constraints.py`) + fixed 30 s windows + 0.75 s onset pad (timeline-
  shifted). C++ q4_k MATCHES the f32 reference on speaker1 (JFK ×2, correct
  segment structure + timeline). speaker2 residual divergence = q4_k on
  overlapping speech (experimental tier).
- **Wirings DONE:** `tiron` factory alias → whisper; registry row; README row
  marked experimental; `tiron_link.{h,cpp}` cross-window linking runtime
  (+ unit test); `tools/reference_backends/tiron.py` + dump_reference registration.
- **NEXT:** wire `crispasr_tiron_link_speakers` into the CLI output (global
  `SPEAKER_NN`); optional f16 A/B to quantify the q4_k speaker2 gap; live test +
  docs; `crispasr-diff tiron` per-stage branch (mel/encoder already whisper-proven).

---

## Earlier notes

- **Status:** Phase 2 (decode mode) + Phase 3 (cross-window linking runtime)
  IMPLEMENTED + build clean on macOS (Metal, Release). Kaggle convert/validate
  run in flight.
- **Phase 3 DONE:** `src/tiron_link.{h,cpp}` — `crispasr_tiron_link_speakers()`
  promotes window-local `<|speakerN|>` indices to meeting-level ids by
  clustering per-(window,local-speaker) group voiceprints. REUSES the existing
  stack: `crispasr_make_speaker_embedder` (TitaNet/ECAPA) +
  `crispasr_agglomerative_cluster` + `crispasr_cluster_centroids`. Adds the
  within-window must-link (aggregate a group's audio into one clean embedding =
  upstream "spine"), acoustic attach for short groups, temporal fallback,
  no-embedder degradation. Unit test `tests/test-tiron-link.cpp` (fake embedder,
  no model): proves linking follows VOICE not local index — 13 assertions PASS.
  Wired into `src/CMakeLists.txt` + `tests/CMakeLists.txt` (label unit;diarize).
- **Phase 2 recap:** `src/crispasr.cpp` — speaker-token detection on load
- **Done:** `src/crispasr.cpp` — speaker-token detection on load
  (`token_speaker_beg`/`n_speakers`/`has_speakers` + `is_timestamp`/`is_speaker`
  helpers), `tiron` decode flag (auto-on for a speaker vocab, `CRISPASR_WHISPER_TIRON=0`
  reverts to stock for A/B), timestamp heuristics bypassed (pairing / max_initial_ts /
  increasing-ts / ts-vs-text forcing), samplers keep `.tid` to real timestamps only,
  emitter renders `<|speakerN|>` inline + splits segments only on real timestamps,
  seek-window no longer advances ~30 s on a speaker token.
- **Next:** run the Kaggle kernel `tools/kaggle/tiron-convert-f5-ab/` on a CUDA
  box. It does Phase 1 (convert → f16/q8_0/q4_k → validate the tiron decode on
  multispeaker.wav + jfk.wav + `CRISPASR_WHISPER_TIRON=0` A/B → upload to
  `cstr/tiron-GGML`) AND the F5 #294 perf A/B fleet (EMBED_GPU / F16_ACT /
  BATCH_CFG / EPSS 7-step / CFG_INTERVAL / DIT_SKIP + recommended stack, each
  with ode_solve ms + ASR-roundtrip proof-of-work). Rebased onto main
  `5b8ed72b6` so the clone has the full F5 lever set.
- **⚠ recovery note:** the tiron `src/crispasr.cpp` edits were first made in the
  SHARED main tree by mistake ([[isolate-worktree-on-shared-tree]]); moved via
  patch into the worktree, main tree restored clean, real build re-run WITH the
  changes present (the first "builds clean" had compiled the unmodified file).
- **Last push SHA:** (this commit)

---

## What Tiron is

A **drop-in `WhisperForConditionalGeneration`** checkpoint = Whisper **large-v3**
architecture (d_model 1280, 32 enc + 32 dec layers, 20 heads, **128 mel bins**),
fine-tuned to **jointly transcribe + attribute speakers** in one decode pass over
a 30 s window. It emits an inline transcript with `<|speakerN|>` turn markers and
`<|t.tt|>` timestamps at 20 ms, e.g.

```
<|speaker1|><|0.00|> Thanks everyone for joining.<|2.96|><|3.52|> Let's get started.<|4.80|><|speaker2|><|2.98|> Morning!<|3.40|>
```

Speaker indices are **local to each 30 s window** (first talker = `<|speaker1|>`).
Cross-window meeting-level identities come from the separate harness (ECAPA
voiceprints + clustering) — that is a follow-up, not the MVP.

### Exact token layout (verified from `added_tokens.json`)

| range | ids | meaning |
|-------|-----|---------|
| eot | 50257 | `<|endoftext|>` |
| sot | 50258 | `<|startoftranscript|>` |
| langs | 50259–50358 | 100 language tokens (`<|en|>`=50259 … `<|yue|>`=50358) |
| task | 50359–50360 | `<|translate|>`, `<|transcribe|>` |
| solm/prev/nosp/not | 50361–50364 | startoflm, startofprev, nospeech, notimestamps |
| **token_beg** | **50365** | `<|0.00|>` |
| timestamps | 50365–51865 | 1501 tokens, 20 ms, up to `<|30.00|>`=51865 |
| **speakers** | **51866–51873** | `<|speaker1|>`..`<|speaker8|>` |
| (reserved) | 51874–51903 | empty pad → `vocab_size` = 51904 |

The speaker tokens sit **contiguously right above the timestamp block**.
`token_speaker_beg` = 51866 = `token_beg` + 1501.

Canonical decode per the model card: `forced_decoder_ids=None`, `suppress_tokens=[]`,
`begin_suppress_tokens=[]`, no `no_timestamps`, `no_speech_threshold=None`,
`do_sample=False`, `num_beams=1`, `max_new_tokens=444` (greedy, unconstrained).

---

## Why the port is tractable — infra already on `main`

Issue #258 (a Japanese-anime Whisper with a custom 20,480-token vocab) already
landed BOTH pieces we need (commit content is on `main`; the `feat/issue-258-*`
branch is just an un-deleted duplicate):

1. **Converter** `models/convert-h5-to-ggml.py` — merges `added_tokens.json`,
   sizes the table to `vocab_size`, serializes every token at its real id. So
   Tiron's `<|speakerN|>` strings land in the GGML vocab with correct names.
2. **Loader** `src/crispasr.cpp` `whisper_vocab` — special tokens resolved **by
   string lookup** (`<|startoftranscript|>`, `<|translate|>`, `<|0.00|>` → token_beg,
   …), NOT magic n_vocab offsets. A 51904-vocab loads and places all specials
   correctly (the old `n_vocab >= 51865` / `- 51765` arithmetic is the fallback
   only). Large-vocab reject is already commented out.

So model load + vocab are DONE for free. The remaining work is the **decode /
emission** path, which assumes "every id ≥ token_beg is a timestamp."

---

## The one real problem: `id >= token_beg` ⇒ "timestamp" is pervasive

`whisper_process_logits` (`src/crispasr.cpp` ~6650–6810) and the segment emitter
(~8150–8250) all treat the whole `[token_beg, n_logits)` band as timestamps:

- `no_timestamps` masks `[token_beg, n_logits)` → would kill speaker tokens.
- timestamps-in-pairs state machine keys on `id >= token_beg` → a speaker token
  mid-stream corrupts it / is blocked.
- timestamp-vs-text logsumexp sums `[token_beg, n_logits)` → speaker prob leaks in.
- max_initial_ts / increasing-ts loops run to `n_logits`.
- emitter line ~8164 only appends `id < token_eot` to text (speaker tokens dropped
  unless `print_special`); line ~8173 splits a segment on any `id > token_beg`
  (a speaker token would wrongly split + mis-time).

Tiron's card explicitly disables all these heuristics anyway (greedy, no suppress,
no pairing). So the clean fix is a **tiron decode mode** that bypasses the
timestamp heuristics and emits speaker tokens inline — NOT a surgical rewrite of
every loop.

---

## Plan

### Phase 0 — investigation + plan  ✅ (this doc)

### Phase 1 — conversion (unblocks validation)
Convert `Trelis/tiron` → legacy GGML `.bin` (whisper format) f16, then q8_0 + q4_k
via `crispasr-quantize`. Reuse the #258 converter (no code change needed — it
already handles the extended vocab). Venue: TBD (local 16 GB Mac, memory-guarded,
vs Kaggle `issue258` kernel template adapted for Tiron). Upload → `cstr/tiron-GGML`.
Verify the card lands with `license: apache-2.0` + attribution.

### Phase 2 — tiron decode mode (core C++)  ← IN FLIGHT
1. `whisper_vocab`: add `id token_speaker_beg = 0; int n_speakers = 0; bool has_speakers`.
   Detect on load: if `<|speaker1|>` ∈ vocab → set them; `token_ts_end` = speaker_beg.
2. `whisper_full_params`: add `bool tiron` (auto-enabled when `has_speakers`, env
   override `CRISPASR_WHISPER_TIRON`). Default greedy, `no_timestamps=false`,
   `suppress_blank=false`, `suppress_nst=false`.
3. `whisper_process_logits`: when `tiron`, skip the pairing rule, max_initial_ts,
   increasing-ts, and timestamp-vs-text forcing; keep only the structural-special
   suppressions (sot/task/lang/nosp/prev/notimestamps). Leave speakers + timestamps
   + text free → pure greedy argmax.
4. Emitter (~8150): in `tiron`, (a) append speaker-token text inline (render
   `<|speakerN|>` → a stable label), (b) do NOT segment-split on speaker tokens
   (only on true timestamps `token_beg ≤ id < token_speaker_beg`).
5. Gate everything behind the mode flag; standard whisper path untouched (A/B).

### Phase 3 — output rendering + optional harness
MVP: per-window inline `SPEAKER_N`-tagged, timestamped segments (local indices).
Follow-up: cross-window ECAPA linking (reuse existing titanet/ecapa embed backend
+ clustering) to promote local indices → meeting-level `SPEAKER_00`… JSON/SRT/VTT.

### Phase 4 — 12-point checklist
Registry auto-download row (`cstr/tiron-GGML`), CLI/C-ABI parity for the speaker
output, quantize rules, diff-harness stage (optional), bindings docstrings, README,
`docs/architecture.md`, live test + `env-live-tests.sh`, Go LDFLAGS (no new lib →
likely no-op). Detection stays architecture=`whisper` / legacy-magic; a `tiron`
sub-flavour is keyed off `has_speakers` in the vocab, not a new arch string.

---

## Open decisions
- **MVP scope:** single-window inline speaker attribution (DEFAULT) vs full
  meeting-level ECAPA harness (follow-up).
- **Conversion venue:** local memory-guarded vs Kaggle.
- **Speaker label rendering:** keep raw `<|speakerN|>` markers vs friendly
  `[SPEAKER_N]` prefixes in the emitted text.

## Validation gates (HARD RULES)
- Decoded-output roundtrip is the only acceptance test: transcribe a known
  multi-speaker clip, confirm text is right AND speaker turns land at plausible
  boundaries. Test F16 AND q4_k (quant amplifies divergence).
- A/B the tiron path vs stock whisper on a single-speaker clip — text must match
  stock whisper (speaker mode must not regress plain transcription).
