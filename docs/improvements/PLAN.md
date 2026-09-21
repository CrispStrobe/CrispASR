# CrispASR — cross-cutting improvements program

Five high-leverage improvements surfaced by the #257 marathon. Every runtime
change ships **behind an env gate** (A/B without recompile, never delete the
working path — per the dev-guide), with an **A/B method** and **unit tests**.

## NOW — active work

### §I6 — `src/` ships baseline x86-64 on every release leg (NEW, undecided)
- [ ] **Decide whether this is policy or oversight.** `release.yml` hands ggml
      AVX2+FMA+F16C on 15 legs and full `GGML_CPU_ALL_VARIANTS` runtime dispatch
      on 2 more, but contains **zero** `CMAKE_CXX_FLAGS` — so every hand-written
      kernel under `src/` compiles generic x86-64 (SSE2, 4-wide) on every
      artifact we ship, including the legs that deliberately give ggml AVX2.
      Evidence, options and open questions: **`docs/improvements/SRC_ISA_GAP.md`**.
      Measured worth on the one kernel tried (Basic Pitch `contour_conv`):
      **+45% on the convolutions, +32% end-to-end, byte-identical output** — see
      `docs/music-transcription/BASIC_PITCH_CONV_PERF.md`. Cheapest first step is
      extending `CRISPASR_PORTABLE_CPU` to cover `src/` too (it loops over
      `GGML_*` only today), which closes a real trap at zero cost regardless of
      what is decided about enabling ISA.
- [x] **Basic Pitch conv fast path — DEFAULT FLIPPED ON** (CI run 35471451173:
      1.82x/3.81x ubuntu-24.04, 2.39x macos-14, byte-identical on both).
      Follow-up: `core_parallel::for_each_chunk` spawns threads per conv call,
      so `n_threads=4` costs 70% more CPU than 2 for 6% less wall — route
      through `core/worker_pool.h`. Original notes:
- [x] **(superseded) Basic Pitch conv fast path — flip the default?** Implemented and gated
      (`CRISPASR_BASIC_PITCH_FASTCONV`, branch `perf/basic-pitch-conv`), output
      proven byte-identical (`tests/test-basic-pitch-conv.cpp`, plus raw-head
      FNV + norms end-to-end). SIMD half wins on speed AND quality, so it clears
      the dev-guide bar; the THREADED half is unproven because the dev VPS gives
      a single-threaded process 36–54% of one core. `.github/workflows/basic-pitch-conv-ab.yml`
      runs the A/B on a clean runner (one arm per process, cold discarded,
      median of 3, achieved-parallelism reported). Flip once that is green.


### Follow-ups (2026-07-16) — from the session-long-audio arc
- [x] **F1 — reuse `core_repeat` beyond moonshine.** DONE for **firered_asr**
      (evidence-gated — a runaway was DEMONSTRATED, not assumed). Audit: most ASR
      backends already have adapter-level `core_ngram::fix_loops` output cleanup
      (cohere/granite/glm/qwen3/canary-qwen/higgs/moss); only `firered_asr`/
      `kyutai_stt` libs had NO guard. Probed both on the loop-prone 60 s song:
      **firered greedy (`--beam-size 1`) runs away** — segment 1 saturates
      `max_len=150` with a period-1 cycle (`OOH ×35`), the same class of loop
      moonshine had, and firered's CLI adapter has no `fix_loops` so that garbage
      tail reaches the decoded output. Wired `core_repeat::tail_is_repetition` into
      firered's **greedy branch only** (`beam_size==1`), gate
      `CRISPASR_FIRERED_NO_REPEAT_BREAK` (default on, mirrors moonshine). A/B on the
      SAME binary (token count is the load-invariant proof):

      | firered greedy / 60 s song | seg-1 tokens | tail | decode |
      |----------------------------|--------------|------|--------|
      | break OFF (old)            | 150 (saturated) | `OOH ×35` | 58–354 s |
      | **break ON (default)**     | **59** (EOS via break) | `OOH ×4` | 15–17 s |

      Coherent content byte-identical between the two — only the runaway tail is
      trimmed. No-regression: jfk greedy unchanged (28 tokens, clean); **beam=3
      (firered's default) untouched by construction** (break is inside the
      `beam_size==1` branch; beam self-terminated at 59 tokens in the probe, so no
      speculative wiring — Phase 1b lesson); `test-repeat-break` green (13
      assertions). **kyutai_stt NOT wired** — no model available locally to probe,
      and its streaming decoder (`eos_id=-1`, runs to `T_frames`) is a different
      shape; left evidence-gated. Backends touched: firered_asr (lib, all surfaces).
- [x] **F2 — surface-parity in nightly CI.** Added `test-surface-parity.sh` to the
      regression workflow so cross-surface (CLI vs session) parity is a permanent
      guard against the #257 class, not a manual audit.
- [x] **F3 — flip `CRISPASR_SESSION_UNIFIED_DISPATCH` default ON** for parakeet
      (verified byte-identical to the inline path, Phase 1). `=0` still selects
      the legacy inline path for A/B.
- [x] **F4 — per-backend session auto-chunk window.** SHIPPED **opt-in, gated off**
      (default behavior unchanged: flat 30 s for every backend). `transcribe_autochunk`
      now sources its default window from the pure
      `session_default_chunk_seconds(backend, perbackend_enabled)`; with
      `CRISPASR_SESSION_PERBACKEND_CHUNK=1` short-segment models (moonshine /
      moonshine-streaming) chunk at 20 s — the session mirror of the CLI's
      `vad_slice_cap_seconds()`. `CRISPASR_SESSION_CHUNK_SECONDS` stays the direct
      per-call override.

      **Why opt-in, not default:** the hypothesis (short-segment models want a
      sub-30 s window) REGRESSED the one long clip measured — moonshine / 60 s
      tiled-song via `test-surface-parity.sh` (CLI-vs-session total-content
      word-overlap, greedy ⇒ deterministic, reproduced):

      | window | session segs | overlap vs CLI |
      |--------|--------------|----------------|
      | 15 s   | 5            | 0.58           |
      | 20 s   | 4            | 0.56           |
      | **30 s** (shipped default) | 3 | **0.75** |
      | 40 s   | 2            | 0.75           |

      A smaller window strictly *lowers* the overlap here (more slices → more
      chunk-boundary artifacts on hard song audio); 30 s is the plateau and 40 s
      ties it. The "30 s still loops" premise was already resolved by the moonshine
      decode-time repeat-break that shipped on `main` (below). But per the A/B rule
      3a (dev-guide): a plausible path whose first A/B regressed on ONE clip is kept
      **gated off**, not deleted — it may win on other clips/models and flipping the
      default is then a one-liner. No-regression confirmed: gate off ⇒ moonshine
      30 s = 0.75, identical to pre-change; parakeet + qwen3 parity PASS 1.00 on jfk;
      `test-session-autochunk` green (both gate modes covered). Backends touched:
      none (session-only). Gates: `CRISPASR_SESSION_PERBACKEND_CHUNK`,
      `CRISPASR_SESSION_CHUNK_SECONDS`.
- [ ] **F5 — run the two CUDA kernels** (`tools/kaggle/{parakeet-mem-policy-cuda,
      server-workers-cuda}/`) — prepared; user-gated on Kaggle quota. _2026-07-16:
      push attempted, REJECTED pre-flight — `chr1str` account at its 30 h/week GPU
      quota (0 quota consumed; kernels never created, `status` → 404). Scripts
      verified launch-ready (py_compile OK, GPU+internet on, proof-of-work guards);
      re-run `bash <dir>/push.sh` once the weekly window resets._

### Further follow-ups (2026-07-16, session 2)
- [x] **F6 — extend the F1 runaway audit to the other greedy decoders.** AUDITED,
      **nothing more to wire** — only firered looped (done in F1). `glm-asr` and
      `cohere-transcribe` are also greedy AR decoders (EOS + token cap) with
      adapter-level `core_ngram::fix_loops` (output cleaned) but no decode-time
      break, so a runaway would still BURN decode compute even with a clean
      transcript. Probed both on the same loop-prone 60 s song that caught
      moonshine + firered:
      - **glm-asr** (`glm-asr-nano-q4_k`): clean, EOS-terminated, no repetition —
        "It'll take some time to find your heart… Cause I'll be there". No runaway.
      - **cohere-transcribe** (`cohere-transcribe-q4_k`): clean, EOS-terminated, no
        loop (one mild trailing single-phrase hallucination, which `fix_loops`
        covers regardless). No `max_len` saturation.
      Neither shows the runaway symptom, so per the evidence gate (Phase 1b) NOT
      wired. ⚠ Scope: one clip (the established worst-case); a broader clip corpus
      could still surface a runaway, at which point `core_repeat::tail_is_repetition`
      drops in exactly as for firered. Backends touched: none.
- [ ] **F7 — repo hygiene.** ~25 untracked entries in the tree (stray `bark_*.log`,
      `failed_logs.txt`, loose `*.gguf`/`ggml-*.bin`, `.codex-scratch/`). Triage into
      gitignore vs safe-delete; present the list before touching anything.
- [ ] **F8 — F5 auto-retry.** A `/loop` or scheduled agent that re-runs the two
      `push.sh` scripts and fires them the moment the Kaggle 30 h GPU quota resets.
- [ ] **F9 — test-suite health snapshot.** Run unit + live integration tests against
      current `main` for a clean baseline (nothing verified end-to-end since F1
      beyond the manual firered A/B).
- [ ] **F10 — propagate the dev-guide A/B rule 3a** (keep-gated-not-revert) to the
      `~/code/vps-mds/` copies so it isn't Mac-only.

- [ ] **CUDA proofs — kernels prepared, awaiting a run** (2026-07-16). The two
      hardware-gated open items can't run on M1; ready-to-push Kaggle kernels:
      `tools/kaggle/parakeet-mem-policy-cuda/` (Phase 2: estimate-vs-real VRAM +
      budget-policy OOM avoidance, incl. a torch VRAM-hog to simulate the
      reporter's small card) and `tools/kaggle/server-workers-cuda/` (Phase 4b:
      `CRISPASR_SERVER_WORKERS=2` concurrency + transcript parity). Merge to main,
      then `bash <dir>/push.sh` when Kaggle quota is available.
- [x] **Session long-audio fix** (2026-07-16) — DONE. The long-audio audit found
      `crispasr_session_transcribe` did one degraded/hanging pass on long audio
      (the CLI/server chunk it). Fixed: `transcribe_autochunk` now slices long
      audio at energy minima and transcribes each piece, shifting timestamps to
      the absolute timeline — like the CLI. Merged chunks are run through
      `core_ngram::fix_loops` (identity on clean text) since a short-segment
      model (moonshine) can loop on a hard slice. Pure applicability decision in
      `src/session_autochunk.h` (`test-session-autochunk`, 13 assertions). Gate:
      `CRISPASR_SESSION_AUTOCHUNK` (default ON — it fixes a hang; `=0` disables),
      `CRISPASR_SESSION_CHUNK_SECONDS` (window, default 30). Skips self-chunkers
      (parakeet/reazonspeech), the return_logits path, and explicit chunk
      requests. Verified on moonshine/60 s: 1 seg (hung, 104-word ×15 loop) →
      3 segs, 43 words, loop collapsed, completes, timestamps monotonic. Caveat:
      moonshine still spends decode time generating the loop before fix_loops
      cleans it (57 s on this song-worst-case); normal speech doesn't loop. Other
      backends inherit the same safe CLI-style chunking (verified on moonshine;
      chunking-at-silence mirrors the dispatcher). Default-on because the prior
      behaviour was a hang.
      _Follow-up (same day):_ closed the decode-time-waste caveat — added a
      decode-time repetition break to moonshine's greedy loop
      (`core_repeat::tail_is_repetition`: a period-≤8 block repeated ≥4× stops
      generation early). moonshine/60 s went **57.2 s → 11.3 s (5×), identical
      output**. Gate `CRISPASR_MOONSHINE_NO_REPEAT_BREAK=1`; pure detector
      unit-tested (`test-repeat-break`, 13 assertions).
      _Broad long-audio re-verification (2026-07-16):_ ran the parity harness on a
      60 s clip across the locally-available backends to check the auto-chunk
      default-on doesn't regress others. Total-content (not per-segment — CLI and
      session cut at different energy minima, so segment counts legitimately
      differ) results: **qwen3 PASS** (both chunk to 3, content agrees);
      **nemotron** CLI 11 w / session 11 w, **0.91 overlap** (CLI treats it as
      single-pass, session re-segments into 3 — same content, harmless);
      **moonshine** 0.75 overlap on the *song* worst-case (chunk-boundary
      sensitivity on genuinely hard audio, not a dispatch bug — hang resolved,
      session actually got slightly more words). No content regressions. Also
      hardened the harness to compare TOTAL content with a word-overlap threshold
      so segmentation differences don't false-fail long-audio audits.
- [x] **Phase 0** — cross-surface parity harness + test (safety net) — DONE
      (`src/core/asr_parity.h` + `test-asr-parity` 13 assertions; live
      `test-surface-parity.sh` PASS: CLI==session on parakeet-tdt-1.1b/jfk)
- [x] **Phase 1** — collapse the dual dispatch — DONE for **parakeet** (pilot).
      Orchestration hoisted to `src/parakeet_orchestrate.{h,cpp}`
      (`parakeet_transcribe_segments`); the CLI adapter is now a thin wrapper
      (**−310 LOC**, 544→234) and the session C-ABI calls the same code under
      `CRISPASR_SESSION_UNIFIED_DISPATCH=1`. Pure `parakeet_pick_strategy`
      unit-tested (`test-parakeet-strategy`, 11 assertions). A/B: CLI
      byte-identical pre/post refactor; parity harness PASS gate-ON on jfk (short)
      AND the 225 s single-pass clip (where the old inline session diverged).
      **Remaining backends** (canary/cohere/granite/…) follow the same recipe —
      tracked below.
- [x] **Phase 1b** — audited, no further hoist warranted (evidence-driven). Rather
      than speculatively hoist every backend, generalized the parity harness
      (`CRISPASR_PARITY_BACKEND` + a CONTENT check that ignores punctuation/case)
      and ran it CLI-vs-session on the locally-available backends. Result on jfk
      (16 kHz, no resample artifact): **parakeet PASS** (unified); **qwen3,
      moonshine, nemotron all PASS(content)** — CLI and session produce identical
      transcriptions, differing only by punctuation (the harness's
      `--no-punctuation` CLI flag has no session equivalent — cosmetic, not a
      dispatch bug). So the #257-class divergence was parakeet-specific (its
      CAP_INTERNAL_CHUNKING long-audio path); the other backends' simpler session
      paths already agree with their adapters. **No further unification needed
      now.** ⚠ Caveat: audited on SHORT audio only; parakeet's divergence surfaced
      on LONG/chunked audio, so a long-clip audit of any backend with a bespoke
      long-audio session path (qwen3's ~212-line block) is the remaining check —
      the generalized harness makes it a one-liner. The parakeet gate
      (`CRISPASR_SESSION_UNIFIED_DISPATCH`) stays opt-in.

      **Long-audio audit run (2026-07-16).** Ran the harness on a 225 s / 60 s
      clip. Finding: on long audio CLI and session DO diverge — e.g. moonshine
      CLI = 8 segments (dispatcher chunks at 30 s) vs session = 1 segment. Root
      cause (verified in `crispasr_c_api.cpp`): the raw `crispasr_session_transcribe`
      is a LOW-LEVEL "transcribe this buffer" primitive that does NOT auto-chunk —
      `moonshine_transcribe_with_probs(ctx, pcm, n_samples)` runs one pass over
      the whole buffer (degraded + very slow on a short-segment model like
      moonshine), while the CLI/server dispatcher adds chunking on top. **Parakeet
      is the exception** — its session branch has bespoke inline long-audio
      handling (chunked_merge), which is exactly what let #257's divergence exist.
      So this is an ARCHITECTURAL layering fact, not a per-backend bug: session
      callers must chunk long audio themselves or use `transcribe_chunked`.
      A uniform fix (auto-chunk in the session for every backend, or push chunking
      below the session) is a real design decision for the maintainer, not a
      drive-by — deferred, documented here + in bindings docs. Short-audio parity
      (the common session use) is clean (Phase 1b audit above).
- [x] **Phase 2** — proactive encoder memory policy — DONE (opt-in). Pure
      `parakeet_est_singlepass_peak_mb` / `parakeet_singlepass_fits_budget`
      (`test-parakeet-strategy`, +2 cases) proactively pick streamed over
      single-pass when the O(T²) bias would exceed
      `CRISPASR_PARAKEET_VRAM_BUDGET_MB` — before allocating, layered over the
      reactive OOM fallback. A/B: on the reporter's 225 s clip the estimate came
      out **1931 MiB (T=2812, H=8)** — matching the reporter's actual **1911.98
      MiB** — and `budget=1500` correctly switched to streamed (full 267-word
      transcript). Default (no budget) unchanged. **CUDA memory proof pending**
      (can't OOM on M1); the estimate matching the reporter's number is the
      evidence so far.
- [ ] **Phase 2** — unified encoder memory policy (proactive, replaces ad-hoc gates)
- [x] **Phase 3** — diff-harness parity in CI — **ALREADY IMPLEMENTED** (verified,
      no new code). The initial PLAN under-counted the existing harness. On audit,
      `.github/workflows/regression.yml` already runs: the dry-run pin gate on PR,
      a nightly per-backend matrix via `tests/regression/run_one.py`, per-stage
      `cos_min` thresholds, exact-transcript + CER/WER tolerance, AND a full
      TTS→ASR roundtrip gate (21 `tts_backends`, WER≤max) — the exact "cos 0.99 →
      hallucination" / "component cos but audio garbage" coverage this phase
      wanted. The pure functions (`_levenshtein`, `compute_transcript_metrics`,
      `parse_diff_stdout`, `evaluate_stage_thresholds`) are unit-tested in
      `tests/regression/test_driver_smoke.py` (model-free, runs in CI). Nothing to
      add without redundancy. Future extension if desired: a distinct word-OVERLAP
      (present-in) verdict alongside WER for very-noisy roundtrips.
- [~] **Phase 4** — server throughput — primitive DONE, integration scoped.
      Landed the reusable, thread-safe `core_pool::WorkerPool<T>` (RAII lease,
      blocking acquire, `test-worker-pool` 17 assertions incl. a blocking-until-
      release concurrency case) + the `CRISPASR_SERVER_WORKERS` gate design.
      **Full server integration deferred to 4b with a hard constraint I verified
      by reading the code:** `do_transcribe`'s single `model_mutex` guards not
      just the backend but the SHARED, explicitly non-re-entrant post-processing
      contexts (punctuation `fireredpunc`/`pcs`, truecaser — comment at
      server L708). So a correct pool must ALSO pool (or fine-grain-lock) those,
      not just the backend — otherwise concurrent workers race the truecaser.
      That's a real `do_transcribe` locking refactor + a concurrent load test,
      not a drop-in; not shipping it half-verified. 4b plan: (a) split
      `do_transcribe` into ASR (pooled, lock-free) + post-process (own mutex or
      per-worker ctx); (b) build the pool of N workers at startup; (c) A/B N=1 vs
      N=2 throughput with identical transcripts; GPU-worker concurrency
      (per-context Metal queues) validated per-platform.
- [x] **Phase 4b** — WorkerPool wired into the server — DONE (gated, correctness-
      verified; throughput is workload-dependent). `CRISPASR_SERVER_WORKERS=N`
      builds N independent backends; a "pure ASR" request (explicit language, no
      aligner, no punctuation/truecaser) routes to a pooled worker and runs
      concurrently, while anything touching the shared LID/aligner/post-processing
      stays on the primary backend + `model_mutex` (serialized, unchanged). `/load`
      returns 409 when the pool is active (pooled workers hold the startup model).
      Default N=1 → `asr_pool` null → zero behaviour change.

      **A/B (honest):** correctness holds — 2 concurrent requests returned
      identical, correct transcripts, and they demonstrably ran in parallel. But
      **throughput is workload-bound**: on this CPU (M1) with the memory-bandwidth-
      bound parakeet-tdt-1.1b-q4_k, 2 concurrent requests took **62 s vs 16 s
      single** — *worse* than the ~32 s a serial pair would take, i.e. the two
      instances contend for memory bandwidth. So the pool is a **latency/
      throughput win only where a single request under-utilises the box** (spare
      cores, a GPU not saturated by one stream, smaller models, or I/O-bound
      mixes), and a net loss on a saturated memory-bound CPU. Kept **default-off**
      per the "flip only when it wins on speed AND quality" rule; documented as an
      opt-in for deployments with headroom. (GPU-worker concurrency —
      per-context Metal/CUDA queues — still wants a per-platform check.)

Execution order is deliberately **0 before 1**: the parity test is the guard that
proves the dispatch unification changes nothing observable. Each phase merges to
`main` green before the next starts.

---

## Phase 0 — cross-surface parity harness + test

**Why:** #257 needed the identical fix in CLI adapter, server, and session C-ABI
because they are three separate code paths. A test that pins "all surfaces agree"
would have caught the wiring gap on commit 1, and is the precondition for safely
unifying them (Phase 1).

**Deliverable:** `tests/test-surface-parity.sh` (live, needs a model) — runs the
same clip through (a) `crispasr` CLI, (b) an in-process session via the Python
binding, (c) the HTTP server, and asserts identical segment text/offsets. Plus a
pure **unit test** for the params→`whisper_params` marshalling helper Phase 1 adds.

**Env gate:** none (test infra). Gated to run only when `CRISPASR_MODELS_DIR` +
a parakeet model are present (label `live;parity`).

**A/B method:** the test itself is the A/B — CLI vs session vs server on the same
seed/clip; diff the JSON `transcription[]`.

**Acceptance:** three surfaces byte-identical on parakeet (F16 + Q4_K) for a short
clip and a `--chunk-seconds 7` clip; unit test green.

## Phase 1 — collapse the dual dispatch

**Why:** `crispasr_session_transcribe*` reimplements each backend's transcribe
inline (dev-guide HARD RULE #6) instead of calling the `CrispasrBackend` adapter
the CLI/server use. Every fix/feature/default risks landing in one path but not
the other (JA-detection was patched in ~5 places; #257 segmentation in 3).

**Approach:** add a session→`whisper_params` marshaller (sticky `source/target_lang`,
chunk, att-context, hotwords, …) and route `crispasr_session_transcribe*` through
`backend->transcribe()` for backends that have an adapter. Delete the inline
per-backend branch once parity holds. Start with **parakeet** (freshest), then
canary/cohere/granite/etc.

**Env gate:** `CRISPASR_SESSION_UNIFIED_DISPATCH` — `1` routes through the adapter
(new), `0` keeps the inline path (old). Default `0` until parity proven per
backend, then flip to `1` and keep `0` as the A/B escape hatch.

**A/B method:** Phase-0 parity test with the gate `0` vs `1` — must be byte-identical
before flipping. Judge decoded output (HARD RULE #3), F16 + Q4_K.

**Unit tests:** marshaller mapping (session fields → `whisper_params`), incl. the
sticky-language and chunk/att-context/overlap fields; a regression that the gate
default is the proven value.

**Acceptance:** parakeet session output identical gate 0 vs 1 across the parity
matrix; measured LOC deleted from the inline branch reported in this doc.

## Phase 2 — unified encoder memory policy

**Why:** windowed local attention is bit-exact + ~3× faster + lower memory yet
still opt-in (`--att-context` + `CRISPASR_FC_WINDOWED_ATTN`), and we just bolted a
*reactive* single-pass-OOM fallback on top (issue #257). Three scattered gates +
a catch-and-retry instead of one decision.

**Approach:** a `parakeet_pick_encode_strategy(T, backend, free_vram?)` that chooses
single-pass / windowed / streamed proactively (bound the O(T²) bias you can't
afford before allocating it). Keep the reactive fallback as a backstop.

**Env gate:** `CRISPASR_PARAKEET_MEM_POLICY` = `auto` (new default) | `windowed`
| `streamed` | `off` (proactive routing disabled). Never removes the existing
`--att-context` / `--chunk-seconds` / `CRISPASR_FC_WINDOWED_ATTN`, and never
disables the physical-memory guard at the encoder allocation boundary.

**A/B method:** back-to-back on the reporter's 225 s clip + a long clip; decoded
output equality vs single-pass within tolerance + peak-footprint (`phys_footprint`
on macOS) and, where possible, a CUDA cross-check (R5 open item).

**Unit tests:** the pure strategy-selection function (T, caps, vram → strategy)
across boundary cases; no model needed.

**Acceptance:** `auto` never OOMs where single-pass would, output within tolerance,
peak memory ≤ single-pass; policy table documented.

## Phase 3 — diff-harness parity in CI

**Why:** the `dump_reference → crispasr-diff` methodology is the crown jewel but
run by hand. Automating it catches the "cos 0.99 snowballs into a hallucination"
class (and the TTS "component cos=0.999 but audio garbage" case) on every push.

**Approach:** a CI job (nightly / label-gated) that, per backend with a hosted
`ref.gguf` (dataset `cstr/crispasr-regression-fixtures`), runs the per-stage diff
+ the decoded-output roundtrip and reds on cos / word-overlap regression. Extend
the manifest beyond ASR-transcript-only to include a TTS→ASR roundtrip gate.

**Env gate:** n/a (CI infra); thresholds configurable via the manifest.

**A/B method:** self-checking (compares runtime vs hosted reference).

**Unit tests:** the threshold/verdict logic (cos_min, word-overlap → PASS/FAIL)
as a pure function.

**Acceptance:** at least parakeet + one TTS backend gated in CI; a deliberately
broken quant reds the job.

## Phase 4 — server throughput

**Why:** the server serialises on a single `model_mutex` — one request at a time.

**Approach:** request batching / a small worker pool / KV reuse. Bigger lift;
scoped last.

**Env gate:** `CRISPASR_SERVER_WORKERS=N` (default 1 = current behaviour).

**A/B method:** concurrent-request throughput + latency, N=1 vs N>1, identical
transcripts.

**Unit tests:** the queue/dispatch logic (no model).

**Acceptance:** throughput scales with workers, transcripts unchanged vs N=1.
