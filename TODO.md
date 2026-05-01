# CrispASR — TODO

Live tracker of pending work across the unified `crispasr` binary and the
shared `src/core/` infrastructure. Items marked **[next]** are the current
session's immediate targets; **[later]** are queued; **[upstream]** are
blocked on external fixes (tracked in detail in `UPSTREAM.md`).

Historical milestones and the per-model port plans are in `HISTORY.md`.
Technical deep-dives (optimisation notes, RoPE lessons, benchmark tables)
are in `LEARNINGS.md`. Full roadmap in `PLAN.md`.

---

## Kokoro / StyleTTS2 (iSTFTNet) backend **[next]**

Plan file: `/Users/christianstrobele/.claude/plans/sprightly-enchanting-dahl.md`.
Detailed handoff: `HANDOFF_NEXT_SESSION.md`. Lessons: `LEARNINGS.md`
"Kokoro / StyleTTS2 lessons".

Status as of 2026-05-01: **M1, M2, M3, M4, M5a, M5b, M6, M7a/b shipped
and stage-validated**. Working tree has uncommitted work (see
`HANDOFF_NEXT_SESSION.md` "Files in flight"). 11 stages exposed via
`kokoro_extract_stage()`: token_ids, bert_pooler_out, bert_proj_out,
text_enc_out, dur_enc_out, durations, pred_lstm_out, align_out,
f0_curve, n_curve, dec_encode_out, dec_decode_3_out — all finite,
shapes validated, magnitudes plausible.

**Remaining:**

1. **Commit** the working state as a clean checkpoint. Use explicit
   paths in `git add` — do NOT `-A` (working tree has unrelated
   pre-existing changes).
2. **M7c — Generator** (~600 LOC). Snake-α activation + AdaINResBlock1
   (different class from `kokoro_adain_resblk`, uses Snake-α not
   LeakyReLU) + CPU-side SineGen+STFT for `har` + 2× ConvTranspose1d
   upsample loop with noise injection + 3-kernel-averaged resblocks.
   Conv_post split: `spec=exp(x[:11])`, `phase=sin(x[11:])`.
   Keep on `gen_sched` (CPU-pinned) until validated; `ups[0]` is the
   same Metal-hang shape (k=20, s=10) as the qwen3-tts codec.
3. **M8 — iSTFT** (~150 LOC C). Direct DFT, n_fft=20 → 11 bins, hop=5,
   hann periodic window. Pure CPU.
4. **M9 — Wire `kokoro_synthesize`** to call BERT → text_enc →
   predictor → align → F0Ntrain → decoder → generator → iSTFT and
   return malloc'd `float*` PCM. The CLI adapter
   (`crispasr_backend_kokoro.cpp`) is already plumbed.
5. **M11 — Reference dumper** at `tools/reference_backends/kokoro.py`.
   Hooks at every stage. Must replicate the 9 corrected behaviours
   from the source-comparison work — pad-wrap, drop-unknown-phoneme,
   banker's rounding, exp/sin on conv_post split, etc. — or the diff
   will spuriously fail.
6. **M12 — Validation** via `crispasr-diff kokoro …`. Need to register
   `kokoro` in `examples/cli/crispasr_diff_main.cpp`. Target
   `cos_min ≥ 0.999` per stage; end-to-end audio xcorr peak ≥ 0.999.

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
- **MiMo-V2.5-ASR** — **[IN PROGRESS]** Converters done, F16+Q4_K on HF.
  Runtime is scaffolded (loads cleanly) but `mimo_asr_transcribe` is
  a stub. Forward pass is PLAN #51. MIT.

  **PLAN #51 status (2026-05-01):**
  Tokenizer scaffold + Python reference dumper landed (commits
  12374d7 + b608b91). `mimo_tokenizer_init_from_file` loads
  `cstr/mimo-tokenizer-GGUF` Q4_K cleanly (569 tensors, 32 enc layers,
  20 RVQ stages on disk; ASR uses the first 8). `mimo_asr_transcribe`
  now lazy-loads the tokenizer context, but the forward path is still
  the stub.

  **What's left (in order):**
  1. **Tokenizer encoder forward pass** — mel → conv stem → 32-layer
     transformer (RoPE θ=10000, head_dim=64; q/v/o have biases, k has
     none; GELU FFN 1280→5120; skip-add from output of layer 2 added
     to output of final layer) → final LayerNorm → down_sample
     (Conv1d k=2 s=2 no-bias) + GELU + LayerNorm → 8-stage Euclidean
     RVQ encode (CPU-side argmin; codebooks F16 → F32 at distance
     compute time, matches upstream `quantizer.float()`). Six
     diff-harness stages already declared in mimo_tokenizer.h.
     Recommendation: gate GPU acceleration behind a flag and start
     CPU-pinned (mirror kokoro `gen_sched`) — the conv stack is the
     same shape that hit the M1 `kernel_conv_transpose_1d` watchdog
     hang on qwen3-tts.
  2. **LM half** — speech_embeddings + speech_group_downcast
     (audio.group_proj) + 6-layer input_local_transformer
     (input_full_attention=true, so bidirectional within a group of
     `group_size=4`) + hidden_states_downcast (audio.hidden_proj) →
     audio_features. 36-layer Qwen2 prefill (hidden=4096, heads=32,
     kv_heads=8, head_dim=128, intermediate=11008, vocab=151680,
     rope_theta=640000, **attention_bias=true** — Qwen2 has Q/K/V
     biases unlike Qwen3; `core_attn::kv_self_attn` does NOT yet
     accept bias inputs and will need either a sibling helper or an
     extension). lm_head not tied. Greedy decode against vocab,
     filtering `<|empty|>`/`<|eot|>`/`<|eostm|>`.
  3. **Prompt format** — chat with thinking. The full Python entry
     point is `MimoAudio.asr_sft` in
     `ref/mimo/github/src/mimo_audio/mimo_audio.py`; key bits to
     port: input_ids shape `[1, audio_channels+1, T]` with channel 0
     = text (mostly `<|empty|>`), channels 1..8 = audio codes; pad
     audio length to a multiple of `group_size=4`; assistant-side
     prefix is `<think>\n\n</think>\n<audio_tag>` (audio_tag is
     `<chinese>`/`<english>`/empty). `MimoStopper` stops on
     `eos_token_id` or `<|im_end|>`.
  4. **Diff-harness backend** — add `mimo-tokenizer` clause to
     `examples/cli/crispasr_diff_main.cpp`; pattern after the
     `qwen3-tts-codec` block (single-stage extract loop with
     cos_min ≥ 0.999 threshold).
  5. **Validation** — running the Python reference dumper requires
     the upstream safetensors which were NOT downloaded in the
     2026-05-01 session (disk pressure on `/Volumes/backups`,
     16 GiB free at session start). The tokenizer safetensors are
     ~2.4 GB; the LM is ~14 GB. Plan: download tokenizer-only
     first, validate the encoder bit-by-bit, defer LM-side ground
     truth until disk frees up.

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
  Base / CustomVoice / VoiceDesign in 0.6B and 1.7B. 10 languages,
  ~97ms end-to-end latency.
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
- **[later] Shaw RPE in graph path — make graph path the default.**
  `GRANITE_ENCODER_GRAPH=1` runs the encoder as a single Metal-accel
  graph (used to be 4× faster than the CPU loop; norm+QKV fusion has
  closed most of that gap). Still missing: query-dependent Shaw RPE
  bias. flash_attn_ext can't handle it; needs manual `Q@K.T + Q@R.T`
  attention. Once added, drop the CPU loop entirely.
- **[later] PLUS / NAR HF uploads.** After the runtimes are
  verified, upload the converted GGUFs to
  `cstr/granite-speech-4.1-2b-plus-GGUF` and
  `cstr/granite-speech-4.1-2b-nar-GGUF` and add registry entries.

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
  dispatch branches. What's still missing is
  `tools/reference_backends/{parakeet,canary,cohere}.py` — those
  backends aren't in REGISTERED_BACKENDS yet because loading NeMo
  checkpoints from PyTorch is non-trivial (needs either
  `nemo_toolkit` or direct .nemo unpacking following the pattern
  in `models/convert-parakeet-to-gguf.py`). Follow-up below.
- **[later]** Write `tools/reference_backends/parakeet.py`,
  `canary.py`, `cohere.py` so the crispasr-diff harness has
  references to compare against. Each follows the pattern already
  in tools/reference_backends/{qwen3,voxtral,voxtral4b,granite}.py.
  Parakeet and canary load .nemo tarballs via torch + tarfile
  (mirroring models/convert-parakeet-to-gguf.py's unpack_nemo()).
  Cohere loads a HF transformers checkpoint.
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
