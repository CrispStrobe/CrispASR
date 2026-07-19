# OmniVoice — issue #254 (voice cloning + RTF)

Branch: `fix/omnivoice-254-voiceclone-rtf` (rebased onto `main` on top of the
stranded GPU commit `feat/omnivoice-gpu` = "run the LLM on GPU").

## NOW — active work

**Status (2026-07-16): OmniVoice RTF #4 — fused stage0 step graph, branch
`feat/omnivoice-rtf-stage0` (worktree `.claude/worktrees/omnivoice-rtf-stage0`).**

Reporter re-benched after single-shot (#254 last comment): CUDA RTF still 0.17
vs omnivoice.cpp 0.144 — gen 3.55 s vs their 3.02 s for ~22 s audio (decode now
free). Structural read of the step loop found the residual ~44 ms/step of host
overhead: per-step ~18 MB audio-embedding readback + single-threaded codebook
sum + ~5 MB embed re-upload, a full-sequence ~39 MB logits readback (only the
target slice is used), single-threaded triple-log-softmax CFG scoring (~13M
`exp`/step), and multi-MB per-step vector allocs (+ a full `u_logits` copy).

- ✅ **Fused per-step graph (`OMNIVOICE_FUSED_STEP`, default ON when persistent):**
  token ids in (→ ~140 KB int32 upload), in-graph `get_rows` + cb-ascending
  chained adds (bitwise == host sum) + concat with a device-resident text-embed
  tensor (own buffer, gallocr-alias-proof), transformer unchanged, `ggml_cont`
  view of ONLY the target logit slices out. Modes: cond-only / uncond-only /
  unified. Embed tables are F16 even in quant GGUFs (quantize rule), so the
  in-graph `get_rows` stays CUDA-graph-capture-safe (fork's
  TAG_GET_ROWS_CUDA_GRAPHS disables capture only for quantized `get_rows`).
- ✅ **Threaded CFG scoring** (`ov_parallel_for` over target positions; the
  position-Gumbel draws are precomputed serially so the rng stream is
  bit-identical; sampled path `class_temp>0` stays serial). Scoring buffers
  hoisted; interval-CFG uses the persistent `u_buf` — no more 18 MB/step copy.
- ✅ **Byte-identity (M1 Metal, q8_0, `OMNIVOICE_DUMP_CODES` + cmp):** legacy vs
  fused BYTE-IDENTICAL on ALL five config classes — (a) reporter's paragraph,
  2-forward path (modes 0+1, 4360 codes); (b) fox + `OMNIVOICE_UNIFIED_CFG=1`
  (mode 2, 520 codes); (c) fox + `OMNIVOICE_CFG_INTERVAL=2` (interval-CFG
  u_buf caching); (d) fox + `OMNIVOICE_GUIDANCE=0` (cond-only, no uncond arm).
  ASR roundtrip on the fused paragraph clean (all #254 words present). 907/907
  unit tests pass.
- ✅ **Default policy:** fused ON for CPU/Metal (proven above); **legacy stays
  the default on CUDA** until the roundtrip runs there (LEARNING 35 — never
  flip a GPU default blind). `OMNIVOICE_FUSED_STEP=1` opts in on CUDA.
- ⏳ **CUDA A/B blocked on quota:** both Kaggle accounts exhausted (30 h/wk,
  ~2 days to reset); alternative: the A1000 box. Kernel is ready:
  `tools/kaggle/omnivoice-fused-step-ab/` (legacy vs fused vs fused+2-forward,
  reporter's paragraph, byte-identity gate + median gen s + per-stage bench;
  `CRISPASR_REF=main`). Local M1 timing is load-noise (loadavg 100–290 all
  day; legacy vs fused gen 370→234 s directional only, decode-stage noise
  3.8× between arms of identical code).
- ✅ **Reference-voice disk cache shipped (2026-07-16):** the reporter's last
  ask (omnivoice.cpp `--ref-rvq` parity). Automatic content-addressed cache of
  the RVQ ref codes (FNV-1a over preprocessed pcm + encoder fingerprint, OVC1
  file, same dir resolution as the pocket_tts latents cache);
  `CRISPASR_OMNIVOICE_VOICE_CACHE=0` disables. Verified: run 2 logs "voice
  codes loaded from cache", codes byte-identical, WAV audio data bit-identical.
- ✅ **M1 matched-load per-stage A/B (load≈29 both arms):** fused vs legacy gen
  is NEUTRAL on Metal (96.3 s vs 93.7 s totals, per-forward medians within
  noise) — unified memory made the legacy host overhead nearly free (embeds
  0.9 s of ~94 s), so the fused win is CUDA-specific (2.3×). Default fused
  everywhere stands (identical output, neutral Metal, big CUDA).
  Same-box omnivoice.cpp: its Metal backend fails to init on macOS 26 (bf16
  Metal-shader compile error in their ggml), CPU-only path is gen 728 s
  (RTF 34) vs our Metal ~90-116 s — CrispASR is the only implementation with
  a working GPU path on this Mac.
- ✅ **CUDA A/B verdict IN (2026-07-16, reporter, RTX 5070 Ti):** `cmp`
  byte-identical on CUDA; gen 3.55 s → **1.53 s (2.3×)**, RTF 0.17 → **0.07**
  (vs omnivoice.cpp 0.144 — CrispASR is now ~2× FASTER than the reference
  implementation); single CUDA-graph warmup. Per-step 45.9 ms = fwd 27.4 +
  score_cfg 11.6 + read_logits 5.2 + sample 1.5. → **CUDA default flipped to
  fused** (this commit). Reporter's remaining ask: reference-voice caching to
  disk (omnivoice.cpp feature parity) — next work item.
- 📣 **Reporter A/B requested (2026-07-16):** asked the #254 reporter (RTX
  5070 Ti — the exact platform) to run `OMNIVOICE_FUSED_STEP=0` vs `=1` with
  `OMNIVOICE_DUMP_CODES` + `cmp` on current main
  (issue comment 4995155233). Whichever lands first — reporter, Kaggle
  quota (~07-18), or the A1000 — gates the CUDA default flip.
- **Next (when GPU access returns):** run the Kaggle/A1000 A/B → if
  byte-identical + faster on CUDA, flip the CUDA default to fused and ask the
  reporter to re-bench. Expected: kills the ~44 ms/step host overhead that is
  the residual RTF 0.17-vs-0.144 gap.

Reporter's residual complaint after the over-length + word-drop fixes: "CrispASR
is still slower than alternative implementations" and "decoding is on cpu, which
is now taking longer." **Profiled → confirmed:** the DAC decode runs 100% on CPU
(`tok.backend = ggml_backend_cpu_init()`, hardcoded) and was the wall — on M1 the
reporter's paragraph decoded in **11.4 s (11.7 s audio) + 6.8 s (2.5 s tail) = 18.2 s**;
the short tail chunk was decode-RTF **2.7** (per-call F16→F32 kernel casts + im2col
copies amortized over few frames).

- ✅ **FASTCONV (`OMNIVOICE_CODEC_FASTCONV`, default ON)** — the dev-guide
  `QWEN3_TTS_CODEC_FASTCONV` pattern applied to `higgs_decode`: (1) **bake F32
  decode conv kernels once at load** (`bake_decode_f32_kernels`) so the fork's
  per-graph `ggml_cast(F16→F32)` inside every `ggml_conv_1d`/`conv_transpose_1d`
  becomes a no-op; (2) **k=1 conv → `ggml_mul_mat`** (skip the pure-copy im2col);
  (3) baked F32 conv_t1 selects the direct `_f32` conv-transpose CPU path.
- ✅ **A/B (M1, q4_k + tokenizer-f16, back-to-back):** decode **10.6 s → 3.6 s ≈ 2.9×**
  (matches the guide's 2.1× CPU / 3× Metal). Short-sentence decode RTF **2.7 → 0.41**.
- ✅ **Equivalence:** output numerically equivalent (max |Δ| ≈ 20/32768, rmse ≈ 1.75
  int16 ≈ −85 dB, inaudible reduction-order drift from the k=1 matmul + `_f32`
  conv-transpose). **ASR roundtrip identical** (both → "The quick brown fox…").
- ✅ **Reporter ask:** `omnivoice_synthesize` now prints a per-stage timing + RTF
  summary at normal verbosity (`gen Xs + decode Ys = Zs for Ws audio (RTF …)`) —
  no more wrapping in `time`.
- ✅ **Landed on `main`** (`6a1b1903b`, rebased past the 0.8.11 release bump).
- ✅ **GPU codec decode (`OMNIVOICE_CODEC_GPU`, default ON for CUDA, opt-in elsewhere):**
  every decode op is Metal-supported (CONV_TRANSPOSE_1D/IM2COL/MUL_MAT/SIN/CAST/
  GET_ROWS), output equivalent (max |Δ| ≈ 54/32768, inaudible). **But on M1 Metal it
  LOSES to CPU-FASTCONV:** decode 1.15→1.73 s (short) and 11.2→19.6 s (437-frame) —
  ~40 modest-channel convs are dispatch-bound on Metal, so Metal remains CPU by
  default. CUDA validation on RTX 5070 Ti reduced a 7.52 s clip's decode from
  ~1.4 s on CPU to ~34 ms with PCM cosine 0.99999986; issue #254 independently
  reported 0.05–0.11 s CUDA decode. `OMNIVOICE_CODEC_GPU=0` restores CPU placement,
  while `=1` opts non-CUDA GPU backends into the existing path.
- ⏭ **Remaining lever:** the LLM `gen` phase is already at per-step parity with
  omnivoice.cpp, so headroom there is kernel-level (their ggml fork). A persistent
  decode graph (build/alloc once, reuse across chunks) could cut the GPU dispatch
  overhead enough to make GPU decode competitive — untested.
- ✅ **`--tts-steps` / `OMNIVOICE_NUM_STEPS` knob** — stage0 (now the dominant cost
  post-FASTCONV) is exactly `num_steps × 2` backbone forwards, so it's the biggest
  remaining speed lever. Wired the codebase-standard `--tts-steps` flag into
  OmniVoice (`omnivoice_set_num_steps` + `crispasr_session_set_tts_steps` dispatch),
  default 32 (quality). Env `OMNIVOICE_NUM_STEPS=N` for quick A/B. Validated ASR
  roundtrip stays clean down to N≈16 (2× fewer forwards); see sweep in this doc.
  Tunable from EVERY consumer (CLI/server-per-request/C-ABI/Python/Go/Dart).

### Single-shot synthesis (#254 reporter: "reduce chunking")
Reporter (CUDA, q8_0, `OMNIVOICE_CODEC_GPU=1`) benched CrispASR at RTF 0.17–0.21 vs
omnivoice.cpp 0.144 on the SAME text/params — a 15–20% gap traced entirely to
**chunking**: CrispASR sentence-split the paragraph into 3 chunks (292+197+54
frames), each a different T, so the CUDA graph re-warmed per chunk (visible
`warmup complete`/`reset` spam) and stage0 ran 3×32 iterations. omnivoice.cpp does
the whole paragraph as one T=544 MaskGIT pass (one warmup, 32 steps). Their decode
was already free on CUDA (0.05 s via `OMNIVOICE_CODEC_GPU=1` — the gate wins on CUDA,
as predicted).

- ✅ **Fix:** added `omnivoice` to the single-shot whitelist in
  `crispasr_tts_plan_chunks_for_backend` (alongside vibevoice/qwen3-tts/tada/dots-tts).
  OmniVoice is masked-iterative — it synthesizes the whole target span in one
  fixed-`num_steps` pass with a single length estimate (no per-token duration head,
  no `MAX_FRAMES` truncation), exactly like omnivoice.cpp. Verified: the reporter's
  paragraph now renders as ONE 410-frame generation (one decode), ASR-complete.
- ⚖️ **Tradeoff + M1 A/B:** single-shot has the SAME linear compute but ~2.7× the
  attention (O(T²)) vs 3 chunks. BUT it also does 1 decode + 1 forward-graph build
  instead of 3 — and that consolidation more than pays for the attention. M1 A/B
  (q8_0, reporter paragraph, interleaved; abs numbers load-garbage — load hit 79 —
  so read the RELATIVE pair only):

  | path | gen | decode | total |
  |------|-----|--------|-------|
  | single-shot | 103.96 s (+6%, O(T²)) | **9.09 s** | **113.0 s** |
  | chunked (×3) | 98.11 s | 23.65 s (3 graph builds) | 121.8 s |

  So single-shot is **net faster even on M1** — the decode consolidation (2.6×
  fewer graph builds, load-independent structural win) outweighs the ~6% gen
  penalty. This flips the earlier worry that it would regress Metal. On CUDA the
  gen graph-reuse is an additional, larger win (the reporter's case).
- 🔌 **Escape hatch:** `CRISPASR_OMNIVOICE_CHUNK=1` forces the legacy sentence-split
  path (also the A/B toggle) — for a Metal user feeding pathologically long text
  where O(T²) attention could dominate.

### stage0 breakdown (M1, clean full-synth log) — what's worth optimizing
`fwd_cond 49.9% + fwd_uncond 44.3% + sampling 4.6% + embeds 1.1%`. So the embed
path is NOT worth folding (1.1%); the only big lever left is the **uncond forward
(44%)**, which interval-CFG attacks. The forwards themselves are compute-bound at
parity with the reference — no free graph-fusion win on Metal (unified CFG is
CUDA-only for that reason).

- ✅ **Interval-CFG (`OMNIVOICE_CFG_INTERVAL=K`, default 1=exact, opt-in APPROX):**
  recompute the uncond forward only every K steps, reuse cached `u_logits` between
  (cond stays fresh every step; first + last steps always recompute). Gated OFF
  (changes output slightly); forces the 2-forward path (unified fuses cond+uncond).
  Env-only like `OMNIVOICE_UNIFIED_CFG` (experimental perf path). **Sweep (M1 q8_0,
  2-sentence, ASR roundtrip):**

  | K | gen | uncond fwds | ASR |
  |---|-----|-------------|-----|
  | 1 (exact) | 9.01 s | 64 | clean |
  | 2 | 6.35 s (−30%) | 34 | clean |
  | 3 | 5.88 s (−35%) | 24 | clean |
  | 4 | 5.50 s (−39%) | 18 | clean |

  K=1 is exact by construction (`step % 1 == 0` ⇒ recompute every step). Content
  preserved at all K; naturalness of the stale-uncond approximation still wants a
  listen, so it stays opt-in. This is the one lever that pushes stage0 *below*
  reference cost (the cpp/torch ports all do full CFG every step).

### Competitive comparison — vs `rockerritesh/omnivoice-tts.cpp` (read their code post-solution)
Their README is candid and **corroborates our design decisions**:
- **Codec stays on CPU in both ports.** They tried codec-on-GPU and measured it
  **~40× slower** (ggml CUDA `conv_transpose_1d` unoptimized: 14 s vs 0.4 s), and
  abandoned it — the exact conclusion behind our `OMNIVOICE_CODEC_GPU` default-OFF.
- **Their fast decode == our FASTCONV.** Their converter: *"CPU ggml_conv_1d/im2col
  REQUIRES f16 conv kernels; CUDA/Metal accept f32."* Their Metal RTF 0.89 comes from
  **f32 codec convs (no cast)**; they expose `--codec-conv-f32` as an opt-in. FASTCONV
  gets the same cast-free decode **automatically from the f16 model** (baked F32
  kernels) — no 2.7 GB f32 GGUF, no ~24 GB CPU RAM — and adds **k=1→matmul** (they
  im2col k=1). CrispASR weights: q4_k 597 MB / q8_0 818 MB vs their f32 2.7 GB.
- **stage0 is the same shape:** 2 separate cond/uncond forwards, 32 steps, persistent
  graph — matched on both. The one stage0 lever they flag as *not done* ("unbatched
  CFG, ~2×") is one **we already have** (`OMNIVOICE_UNIFIED_CFG`, +13% CUDA).
- Their published wins are **M4 Pro / T4** (different silicon than our M1) at f32 —
  not a same-box comparison. A fair head-to-head belongs on **Kaggle CUDA** (guide's
  <1 %-variance box), at matched dtype, where the kernel-quality gap actually lives.

---

**Status (2026-07-15): OmniVoice WORD-DROPPING (#254) — ✅ FIXED + SHIPPED.
Root cause: `llm.token_embd.weight` in shipped `omnivoice-f16.gguf` had 4094
ZEROED rows (ids ~3380–12594; "quick"=3974) from a post-conversion WRITE
corruption (source clean SHA-verified; reconvert clean → converter is fine).
Reconverted f16/q4_k/q8_0 (0 zeroed rows) → fox + reporter paragraph render
EVERY word. Re-uploaded all 3 to `cstr/omnivoice-GGUF`, **SHA-verified server-
side** (f16 670592a5, q8_0 9d8835c8, q4_k a1a9c6fc). Local canonical files
swapped. Issue #254 updated (comment 4977208265). Registry needs no change
(stores URL+size, not SHA; sizes unchanged). DONE.**

- ✅ **FIXED + validated:** reconverted from clean `k2-fsa/OmniVoice/model.safetensors`
  (2.45 GB, sha `730839316de5…` == HF; source token_embd 0 zeroed rows, "quick"
  row norm 1.53). Reconverted f16 = 0 zeroed rows (was 4094). Acceptance:
  fox → "The quick brown fox…" (full); reporter's paragraph → all of "started",
  "major open weights", "TTS architectures", "One build", "pick", "See…TTS side"
  restored. q4_k + q8_0 re-derived (same sizes) also say "quick".
  Fixed-file SHAs: f16 `670592a5…`, q8_0 `9d8835c8…`, q4_k `a1a9c6fc…`.
  The CONVERTER is fine (reconvert clean) — corruption was a file write/upload.

- 🎯 **ROOT CAUSE (bisected via step-0 dumps):** input-embedding compare vs
  omnivoice.cpp `--dump` (`lm-hidden-step0-cond-embed` [83,1024]) — global
  cos 0.995, but positions 1/4/8 cos=**0.0** (our vectors literally 0). Oracle
  `prompt-cond-ids`: pos1/4=4064 "None", pos8=**3974 "quick"**. Scanned our
  `llm.token_embd.weight`: **4094 zeroed rows in [3380,12594]** (first ~1023
  contiguous, then blocky). ALL other tensors clean (audio_embd/output/layers
  ~0% zeros). Deterministic + word-specific + quant-independent (f16 AND q4_k
  share it) — matches every symptom. Same class as [[cohere-arabic-gguf-zeroed-norms]]
  / the tokenizer block.4 zeroing.
- 🔧 **FIX PLAN:** (1) obtain clean `k2-fsa/OmniVoice` LM safetensors (not in
  local HF cache — only config/tokenizer; original convert ran on VPS);
  (2) reconvert f16, verify `token_embd` has 0 zeroed rows + re-run fox ASR
  (must say "quick"); (3) re-derive q4_k/q8_0; (4) re-upload to
  `cstr/omnivoice-GGUF` (SHA-verified) + bump registry SHAs; (5) tell reporter.
- 🧰 Tooling landed: `OMNIVOICE_DUMP_DIR` step-0 embed/logits dump; diagnostic
  temp/guidance env knobs.
- ⚠ The shipped HF `omnivoice-f16.gguf` is almost certainly the corrupt file the
  reporter used (their dropped words match our local file exactly).

- 🐞 **Reporter** ([comment](https://github.com/CrispStrobe/CrispASR/issues/254#issuecomment-4973702610)):
  long English paragraph drops words ("started", "One", "See", "pick" missing;
  "TTS"→"T"). JP length now fine.
- ✅ **Reproduced** on our build. Minimal repro: `--tts "The quick brown fox
  jumps over the lazy dog."` drops **"quick"** (65 frames).
- ✅ **omnivoice.cpp oracle CLEAN** on the SAME text — both default (pos_temp=5)
  AND greedy (repacked its `--maskgit-test` i32 dump → `.rvq` → codec-decode →
  ASR = full "quick"). So it's OUR bug, not the model.
- ✅ **Ruled OUT**: estimator (21.8 s / 16.6 cps normal); sentence-splitting
  (single-shot also drops — my first hypothesis, WRONG, reverted); CFG
  (guidance 0/1/2/3 all drop); sampling (greedy pos_temp=0/class_temp=0 drops);
  seed (1/42/123/999 all drop); quant (f16 — higher precision than oracle's
  Q8_0 — also drops). ⇒ **deterministic forward divergence.**
- ✅ **Decode logic byte-faithful** to omnivoice.cpp `maskgit-tts.h`
  (schedule, log_softmax, CFG, confidence `(max_lp−k·pen)/pos_temp+gumbel`,
  layer-penalty, top-k selection — all identical, verified line-by-line).
  Hparams match (28L, d=1024, 16/8 heads, hd=128, ff=3072, **rope θ=1e6**,
  eps=1e-6, NEOX). Structure matches (full-bidir attn, Qwen3 Q/K-norm,
  scale 1/√128, SwiGLU, separate/untied `audio_output` head, k·1025 codebook
  offsets, `where(audio_mask,...)` merge).
- 🔎 **Prime suspect**: a mis-converted LM weight (our LM forward was only ever
  A/B'd internally, never diffed vs Python/omnivoice.cpp; we already found one
  corrupt converted tensor — the tokenizer block.4 — earlier this project).
- ⏭ **NEXT**: dump omnivoice.cpp step-0 `lm-logits-step0-cond` [K,T,V] (`--dump`)
  + add matching step-0 dump to ours, compare argmax per (k,t) → localize to a
  layer/weight. Diagnostic knobs landed: `OMNIVOICE_GUIDANCE/POS_TEMP/CLASS_TEMP`.

---

**Prior (DONE): duration estimator adopted + `--tts-speed` knob**
on `main` (`224420b8e`, `19151d124`). Reporter confirmed ref-voice scaling works.

- ✅ **Reporter re-verified** ([#254 comment](https://github.com/CrispStrobe/CrispASR/issues/254#issuecomment-4973702610)):
  no-ref JP line = 6.72 s (good); WITH a (slow) ref voice it grew to 11.60 s —
  too slow, unnatural inter-letter pauses. Asked for a duration-multiplier knob.
- ✅ **`--tts-speed X` knob** (`19151d124`): speaking-rate multiplier scaling the
  target-length estimate (>1 faster/shorter, <1 slower/longer). Reuses the
  codebase-standard flag (f5/piper/melotts/fastpitch); new `omnivoice_set_speed`
  + `ctx->speed` → `estimate_target_tokens(...,speed)`. Verified exact 1/speed
  scaling (fox: 1.0→65f, 1.7→38f, 0.7→93f). Told reporter `--tts-speed 1.7`
  brings their 11.6 s → ~6.8 s. Possible next: check if ref_T is measured long.

- ✅ **Reference-relative duration estimator** (`estimate_target_tokens`,
  `src/omnivoice.cpp`). Faithful port of OmniVoice's `RuleDurationEstimator`
  (`omnivoice/utils/duration.py`, Apache-2.0 © Xiaomi/k2-fsa; MIT C++ mirror
  in ServeurpersoCom/omnivoice.cpp). Both licenses MIT-compatible ⇒ direct
  adoption + attribution, no clean-room. Three pieces, all byte-matching the
  reference: (1) per-Unicode-script phonetic weights; (2) reference-RELATIVE
  formula `target = ref_T·w(target)/w(ref_text)` — with a voice prompt the
  anchor IS the reference, so **length now tracks the reference speaker's rate**
  (fixes reporter's "duration doesn't change with ref voice"); no-ref falls back
  to "Nice to meet you." ≈ 25 frames; (3) low-length **power-curve boost**
  (`<50 → 50·(est/50)^(1/3)`) so short clips don't render too fast / skip
  characters (reporter's Japanese complaint); digit weight 3.5 lengthens
  number-heavy text. Validated end-to-end: fox → 65 frames / 2.60 s (exact
  prediction), short clips boosted, JP/numbers longer, duration scales with ref_T.
- 🔎 **Orthogonal follow-up (NOT this change):** whisper-base ASR-roundtrip of
  the fox drops "quick" — reproduces at **f16@65 AND q4_k@92 frames**, so it's
  neither length nor quant, and not a regression (prior estimator gave ~67
  frames too). Likely model under-articulation of a short word, or base-ASR
  mishearing. Track separately if it recurs on cleaner ASR.
- ⏭ **Next:** post #254 update asking reporter to re-verify JP length + ref-voice
  scaling on current main; optional `--verbose` per-stage timings + RTF flag.

---

**Prior status: investigation + blueprint complete; building isolated reference-dump env.**

- ✅ **Encode blueprint pinned** (cross-validated: HF transformers source +
  OmniVoice inference + omnivoice.cpp as spec-oracle). See "Encode blueprint" below.
  Key correction: **codec is 25 Hz, not 75 Hz** (hop=960; `downsample_factor=320`
  is a red herring). Baseline confirms: 366 frames × 960 / 24000 = 14.64 s. ✅
- ✅ **Profiling**: GPU/Metal gen_step ≈ 0.6–1.0 s each × 32 steps; CPU is *far*
  slower (short clip: GPU ~22 s, CPU still running at 3 min) ⇒ **GPU is the right
  backend** here — NOT the "small-model GPU-loses-to-CPU" pattern. RTF gap vs
  omnivoice.cpp is per-step launch/alloc + dual-forward overhead, not backend choice.
- ✅ Downloaded + SHA-verified `k2-fsa/OmniVoice/audio_tokenizer` (806 MB, sha
  `fe7c5e87…` == HF) for the encode reference.
- ⚠ **Env blocker**: `higgs_audio_v2_tokenizer` needs transformers ≥5.3 (base
  conda has 4.57.6). Building an ISOLATED venv (torch-cpu + transformers 5.x) for
  the dumper — must NOT upgrade the shared base (breaks other backends' dumpers).
- 🔎 Noticed (separate quality bug, not #254 scope): `estimate_target_tokens` uses
  a "75 Hz / 6 frames-per-char" heuristic; at the real 25 Hz that's ~3× too many
  target frames ⇒ over-long audio. Doesn't change RTF (numerator+denominator both
  scale) but bloats latency + trailing silence. Track separately.

- ✅ **Encode reference dumped + validated** — isolated venv (torch 2.13,
  transformers 5.13.1) at `/Volumes/backups/ai/crispasr-gguf/.venv-omnivoice-ref`;
  `tools/dump_omnivoice_encode_reference.py` runs the REAL `encode()` with forward
  hooks → `/Volumes/backups/ai/crispasr-gguf/omnivoice-encode-ref.gguf`. jfk.wav
  (11 s) → **275 frames @ 25 Hz** (264000/960), all stage shapes match the blueprint:
  `sem_hidden_mean`(550,768)→`sem_ds`(275,768)→`e_semantic`(275,768),
  `e_acoustic`(275,256), `emb_fc`(275,1024), `codes`(8,275) ∈ [1,1023]. ref.gguf
  (5.7 MB) → upload to `cstr/crispasr-regression-fixtures` (NOT in git).

### ⚠ SHIPPED GGUF BUG (found via bisect) — corrupt tokenizer weight
`omnivoice-tokenizer-f16.gguf` (HF `cstr/omnivoice-GGUF`, SHA-matched) has
`acoustic_encoder.block.4.conv1.weight` with **511 contiguous output channels
(1411–1921) zeroed** — a ~6 MB zeroed block in a 25 MB tensor (file/write
corruption, same class as [[cohere-arabic-gguf-zeroed-norms]]). Every other tensor
is clean; the source safetensors is clean. Latent because the encoder was never
exercised (voice-clone was a stub). This is what dragged the C++ acoustic encoder
to cos 0.934. **Fix: regenerate the tokenizer GGUF from safetensors → verify no
zeroed channels → re-upload to HF + update registry SHA.** The bisect harness
(`OMNIVOICE_ACENC_BISECT`) localized it: blocks 0–3 exact, block4 one channel
mine≈0 vs ref=23.45 → weight compare showed the zeros.

### Encode port — stage status (diff vs omnivoice-encode-ref.gguf, `OMNIVOICE_ENCODE_DIFF`)
| Stage | wav→ | status |
|-------|------|--------|
| DAC acoustic encoder | `e_acoustic` | ✅ cos_min 1.000000 (needs FIXED GGUF) |
| HuBERT feat-extract | `hb_featextract` | ✅ cos_min 1.000000 |
| HuBERT feat-proj | `hb_featproj` | ✅ cos_min 1.000000 |
| HuBERT encoder (pos_conv+12L) | `hb_layer0/11`,`hb_mean13` | ✅ cos_min 1.000000 |
| `encoder_semantic` | `e_semantic` | ✅ cos_min 1.000000 |
| concat+`fc` | `emb_fc` | ✅ cos_min 1.000000 |
| RVQ (8 cb) | `codes` | ✅ 2197/2200 exact (F16 near-tie flips) |
| **FULL wav→codes** | `codes` | ✅ 1975/2200 (89.8%) — resample-limited |

**ENCODE PORT COMPLETE + validated stage-by-stage at cos 1.0.** Full chain 89.8%
(only gap = C++ Kaiser vs torchaudio Hann-sinc 24→16k resample; acoustic branch
exact). `omnivoice_set_voice_prompt` wired (load→resample24k→RMS-norm→clip×960→
higgs_encode), `generate_iterative` adds `<|denoise|>` + ref-text prepend.

### ✅ Voice-clone acceptance PASSED
Closed-loop speaker cosine (Resemblyzer VoiceEncoder), clone jfk→"quick brown fox":
**cosine(clone,ref)=0.7743 vs cosine(baseline,ref)=0.4737, Δ+0.30** → clone pulls
timbre toward the reference. Runs end-to-end: "voice prompt encoded — 275 ref
frames". **Issue #1 (voice cloning doesn't work) RESOLVED.**

### ✅ Shipped to HF `cstr/omnivoice-GGUF` (all SHA-verified)
- `omnivoice-tokenizer-f16.gguf` → **replaced corrupt file** with clean regen (sha 710ef610).
- `omnivoice-q4_k.gguf` → **new** main-model quant, 597 MB, ASR-clean (sha e9ae2a80).
- **Tokenizer stays F16** — quantizing it collapses RVQ code match to 0.6% (q8 AND
  q4_k): the codec's VQ codebook nearest-neighbor is too quant-sensitive. Do NOT ship
  tokenizer quants.
- ✅ `estimate_target_tokens` fixed to 25 Hz (2.5 frames/char): 23 s → 4.4 s, ASR clean.

### RTF (issue #2) — profiled; naive wins disproven by measurement
Per-phase `gen_step` (M1 Metal, `OMNIVOICE_BENCH`, steady state): embeds ~2 ms each,
**fwd_cond ~99 ms + fwd_uncond ~99 ms** = ~198 of 214 ms. So:
- ❌ **embed-cache** — embeds are 2 ms, worthless (my initial hypothesis, disproven).
- ❌ **target-slice audio head** — head is out_dim·d·T ≈ 0.7 % of the forward (28-layer
  body dominates); even for large T_ref the saving is <1 %.
- ✅ **Real lever = unified cond+uncond graph** (seq-concat + block-diagonal mask +
  per-block RoPE positions): fuses 2 forwards → 1 dispatch. Compute-NEUTRAL (same
  T_total+T_target tokens through 28 layers) — the win is GPU dispatch/sync efficiency
  + batching, ~1.2–1.4× expected. Needs: `build_llm_graph` mask support, F16
  block-diagonal mask padded to `GGML_KQ_MASK_PAD`, gate `OMNIVOICE_UNIFIED_CFG`,
  A/B back-to-back + **Kaggle CUDA verdict** before flipping default (perf-change
  discipline). The 3–4× gap vs omnivoice.cpp likely also involves per-op Metal launch
  overhead (~12.5k dispatches/synth) — investigate CP_DIRECT / fewer ops per layer.
  **Deferred as a focused follow-on** — not landed unverified.

### RTF benchmark vs ServeurpersoCom/omnivoice.cpp (M1 Metal, per-step, instrumented)
Built their repo + timed their batched forward directly (`OVCPP_BENCH`, added to
`pipeline-tts.cpp:638`). Clean per-step forward (S≈128):
| impl | per-step fwd | how |
|------|------|-----|
| omnivoice.cpp | **~165 ms** | **B'=2 4D-batch** cond+uncond, persistent graph |
| ours 2-forward | ~198 ms (99+99) | two separate forwards |
| ours unified seq-concat | ~212 ms | one seq (T_tot+T_tgt) + block-diag mask |

**Findings:**
1. **The reporter's "3–4× RTF" was mostly the over-length bug** (3× frames at wrong
   75 Hz) — now fixed. Per-step we're within **~1.2×** of omnivoice.cpp, not 3–4×.
2. **Seq-concat is the WRONG fusion** — block-diag mask still makes flash-attn compute
   the full (T_tot+T_tgt)² then masks half away (that's why ours is 212 > 198). Their
   **4D batch dim (B'=2)** computes only within-batch attention (2×128²) → the win.
   So the real RTF fix is a **B'=2 4D-batched forward**, NOT seq-concat. (Dev-guide's
   "seq-concat beats 4D batch" was a different model's alloc-bug case; here 4D wins.)

### RTF verdict: PARITY reached (load-matched, back-to-back ×3)
Per-step FORWARD (both cond+uncond), M1 Metal, clean:
| impl | per-step fwd |
|------|------|
| ours q8 2-forward (default) | **~170 ms** |
| omnivoice.cpp B'=2 batched | ~167 ms |
| ours unified attention-split (gated) | ~200 ms (WORSE — extra views/conts/concat) |

- **We're at ~2% = PARITY.** The reporter's "3–4×" was 100% the over-length bug (now
  fixed). Not 3–4× behind — neck-and-neck.
- **Fusion is a dead end on M1**: the forward is COMPUTE-bound (238 tokens ≈ 200 ms in
  1 or 2 dispatches), so fusing 2 forwards saves nothing; the attention-split even adds
  overhead. `OMNIVOICE_UNIFIED_CFG` kept gated OFF (may still help CUDA — Kaggle A/B).
- **q4_k is SLOWER than q8** (~97 vs ~86 ms/fwd) — our Metal q4_k dequant path; so a
  smaller model isn't the lever either.
- Remaining edge to omnivoice.cpp = ~9% per-token kernel efficiency (their ggml fork),
  since we already process fewer tokens (238 vs their padded 256). Beating them further
  is kernel-level (port their Metal kernels) — a separate deep effort, uncertain payoff.

### ✅ CUDA A/B verdict (Kaggle chr1s4, P100/T4) — the fusion WINS on CUDA
`tools/kaggle/omnivoice-cfg-cuda-ab/`. Per-step forward, on_cuda=true (CUDA0),
codes byte-identical (proof-of-work ✓):
| config | CUDA ms/step | M1 Metal ms/step |
|--------|------|------|
| 2-forward | 77.2 | ~170 |
| **unified CFG** | **67.1 (−13%)** | ~200 (+3%, worse) |

**So the fusion that LOSES on M1 (compute-bound) WINS ~13% on CUDA** (batching/dispatch
differ — exactly the dev-guide prediction). Codes byte-identical ⇒ pure speed win.
**Landed: `OMNIVOICE_UNIFIED_CFG` now auto-defaults ON for CUDA, OFF for Metal/CPU**
(env override kept). omnivoice.cpp's CUDA build errored on Kaggle → no head-to-head
vs theirs on CUDA (M1 was parity: 170 vs 167).

Kaggle regime lessons burned in (kernel is now the reference impl for HF-on-Kaggle):
downloads via `curl -L` (HF client Xet path strands, CLAUDE.md gotcha #1; NO `-C -` —
signed CAS URL rejects Range → rc=22); clone to `/kaggle/temp` (#22); flushed
`progress.txt` after every phase (#15 — survives hard-kill); `-j2` (OOM #5).

### Remaining (optional)
1. Kernel-level: profile per-op vs their ggml fork for sub-parity RTF on M1.
2. Match torchaudio resample (Hann sinc) to push encode codes >99%.
3. Re-confirm CUDA win with a 2nd run + T4-vs-P100 before over-trusting one box.
3. **Ship the GGUF fix**: `omnivoice-tokenizer-f16-fixed.gguf` (0 zeroed channels)
   → replace corrupt HF `cstr/omnivoice-GGUF` + registry SHA bump.
4. RTF wins (issue #2), gated + A/B'd.

### Next (implementation)
1. **C++ `higgs_encode` port, stage-by-stage vs `omnivoice-encode-ref.gguf`**
   (self-contained `omnivoice_encode_diff` runner, dots-tts pattern). Order:
   DAC acoustic encoder (`e_acoustic`; reuse `core_dac`/`core/conv.h`) →
   `encoder_semantic` bridge → HuBERT semantic (biggest piece) → mean-13/`[::2]` →
   concat+`fc` (`emb_fc`) → RVQ (`core_rvq::encode_euclidean`, `codes`). First
   divergence = the bug.
2. Load encoder-side tokenizer weights in `load_tokenizer` (currently decoder-only).
3. Wire `omnivoice_set_voice_prompt`: resample→24k, RMS-norm, clip ×960, encode →
   `ref_audio_codes`/`ref_T`; fix `generate_iterative` (add `<|denoise|>`, prepend
   ref_text). Validate: closed-loop cosine(C,R) > cosine(B,R).
4. RTF wins, gated + A/B'd.

- ✅ Rebased the stranded `feat/omnivoice-gpu` (single commit `c80328e08`, never
  merged to main) onto current `main`. GPU-backend init + `CRISPASR_OMNIVOICE_CPU`
  gate are the only pre-existing changes.
- ✅ Model SHA verification vs HF `cstr/omnivoice-GGUF` (user requirement):
  - `omnivoice-q8_0.gguf` local == HF `4e0bbc93…` ✅
  - `omnivoice-tokenizer-f16.gguf` local == HF `9cb7741a…` ✅
  - (f16 LLM not yet local; download+verify when F16 A/B is needed.)
- ✅ **Baseline RTF (prove-the-work)**: GPU/Metal, q8_0, 60-char sentence →
  14.64 s audio in **93.3 s wall** (`user` 30 s, so ~63 s is GPU dispatch/sync
  wait). **RTF ≈ 6.4** — far slower than real-time. `real ≫ user` ⇒ launch/sync
  bound, not compute bound. Reporter says omnivoice.cpp is 3–4× faster ⇒ ~RTF 1.8.
## Root causes

### Issue #1 — voice cloning "doesn't work": it's an unimplemented stub
`omnivoice_set_voice_prompt` (`src/omnivoice.cpp`) has a literal
`// TODO: load WAV, encode through audio tokenizer` and never populates
`ref_audio_codes` / `ref_T`. So `generate_iterative` always runs with `T_ref=0`
(no reference conditioning). The `--voice`/`--ref-text` CLI path is wired, but the
encode is missing.

**Feasibility**: all encoder weights ARE in `omnivoice-tokenizer-f16.gguf`
(`acoustic_encoder.*`, `sem.*` HuBERT 12L, `fc/fc1/fc2`, per-quantizer
`project_in`). The tokenizer encode path (per converter docstring) is:
HuBERT semantic encoder (12L, 768d, 16 kHz; 7-conv feat-extractor strides
[5,2,2,2,2,2,2]) + DAC acoustic encoder (downsample [8,5,4,2,3], hop 960, 16 kHz) +
quantizer bridge (semantic VQ + acoustic residual VQ) → 9 code streams, OmniVoice
uses 8. Frame-rate alignment (sem 50 Hz / ac 16.7 Hz / LLM 75 Hz) is the key
ambiguity to pin from the blueprint.

**Reuse (do not reinvent)**: `core/dac_decoder.h` (`snake`, `conv1d`),
`core/conv.h` (strided/transpose conv), `core/rvq.h::encode_euclidean` (argmin
quantize — already used by `mimo_tokenizer` + `kyutai_stt`).

### Issue #2 — RTF 3–4× worse: launch/sync-bound GPU path
`real ≫ user` on the baseline. Suspects (in the "unified graph / caching / sched /
alloc" family):
- **Per-step embedding lookups do gallocr new/alloc/compute/free** twice
  (text+audio) per arm per step (`read_embedding_rows`, 32 steps × 2 arms) — pure
  launch/alloc overhead on Metal.
- **Final `audio_output_w` projection runs over all `T_total` positions** but only
  `T_target` are used — a large wasted matmul (output = n_codebooks·audio_vocab).
- **cond + uncond are two separate graph computes** — candidate for the
  seq-concat + block-diagonal-mask unified graph (dev-guide CFG fusion learning).
- The persistent-graph reuse (#245) is already in for the two LLM arms; good.

## Validation regime (mandatory)
- Per-stage diff vs Python `ref.gguf` (`crispasr-diff`), earliest divergence = bug.
- Decoded-output roundtrip: TTS→ASR overlap; voice-clone closed-loop
  cosine(C,R) > cosine(B,R) (Resemblyzer).
- ServeurpersoCom/omnivoice.cpp = **black-box output oracle only**; do not read its
  code until our own solution exists, then compare to optimize.
- Every perf path env-gated + A/B'd (F16 + Q8), default flipped only on speed AND
  quality win.

## Encode blueprint (HiggsAudioV2TokenizerModel.encode — for voice cloning)

Cross-validated against HF transformers `modeling_higgs_audio_v2_tokenizer.py`,
OmniVoice `omnivoice/models/omnivoice.py`, and ServeurpersoCom/omnivoice.cpp
(spec-oracle only; code not copied).

**Constants (derived, not raw JSON):** hop_length=960, **frame_rate=25 Hz**,
hidden=1024 (256 acoustic + 768 semantic), **num_quantizers=8**,
semantic_downsample_factor=2, pad=480, audio_vocab=1025, audio_mask_id=1024.

**encode(wav @ 24 kHz mono):**
1. Semantic: resample 24k→16k; `F.pad(x,(160,160))` (hard-coded 160, NOT pad=480);
   HuBERT `output_hidden_states=True`; **mean over all 13 hidden states**; then
   `[:, ::2, :]` → (T25, 768).
2. `encoder_semantic(sem.T)` → (768, T25). **Required module, weights present as
   `encoder_semantic.*` (13 tensors).** conv(768,768,k3,pad1,bias=F) + 2 blocks
   {2 res_units [ELU→conv1 k3 dil1→ELU→conv2 k1], block conv k3 pad1 bias=T}.
3. Acoustic: DAC encoder on 24 kHz wav → (256, T25). conv1(1,64,k7,pad3); 5 blocks
   ratios [8,5,4,2,3], ch 64→128→256→512→1024→2048, each = 3 ResUnit(dil 1,3,9)
   [Snake→conv k7→Snake→conv k1] + Snake + strided conv(k=2·s,stride=s,pad=ceil(s/2));
   snake1 + conv2(2048,256,k3,pad1). If conv-len ≠ sem-len, re-run with `F.pad(wav,(480,480))`.
4. `emb = cat([e_acoustic, e_semantic], dim=1)` (ACOUSTIC FIRST) → (1024, T25);
   `fc`(1024→1024) on transposed.
5. RVQ encode, 8 codebooks greedy residual: `idx_k = argmin_dist(project_in_k(res))`
   over `codebook_k.embed` (1024×64, Euclidean); `res -= project_out_k(embed[idx_k])`.
   → codes (8, T25), values 0..1023. Reuse `core_rvq::encode_euclidean`.

**HuBERT (semantic_model, post-norm variant):** 7 conv feat-extract (k[10,3,3,3,3,2,2]
s[5,2,2,2,2,2,2], dim512, bias=F; layer0 GroupNorm); feat_proj LN(512)+Linear(512→768);
pos_conv wnorm Conv1d(768,768,k128,groups16)+SamePad+GELU; `h = h + pos_conv(h)` →
`encoder.layer_norm` (pre-stack) → 12 layers [MHA12→res→LN→FFN(768→3072 GELU→768)→res→
final_LN]; NO trailing LN. 13 hidden_states = input-after-pos_conv+LN + 12 layer outs.

**Dead at inference (skip):** `fc1`(1024→768), `decoder_semantic.*`, RVQ EMA buffers.

**Voice-clone LLM sequence** `[style | text | ref_audio | target]`:
- style adds `<|denoise|>` when ref present, then lang/instruct tags; `.repeat(8,1)`.
- **text = `<|text_start|>` + (ref_text.strip()+" "+target_text.strip()) + `<|text_end|>`**
  — ref transcript PREPENDED into ONE combined text stream (not separate).
- ref_audio = tokenizer codes (8, T_ref), audio_mask=True.
- target = all `audio_mask_id` (8, T_target), audio_mask=True.
- Ref WAV: resample→24k, RMS-normalize (if 0<rms<0.1 scale to 0.1/rms), clip len to
  multiple of 960. Current `omnivoice_set_voice_prompt` does NONE of this (stub).

**Current-code gaps for voice clone (beyond the encode stub):** missing `<|denoise|>`,
missing ref_text prepend. Both in `generate_iterative` §3–4.
