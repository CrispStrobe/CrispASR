# CrispASR — Current work plan

Updated 2026-11-07. Start with [the maintainer handoff](docs/maintenance-handoff.md).
It contains the completed work, exact evidence, bounded next actions and blockers
needed by a fresh agent; no earlier chat is required. Follow `AGENTS.md` and its
mandatory maintainer-guide instructions before implementation.

**v0.8.51 is published and verified (2026-11-07); v0.8.50 too (lanes R1–R3 complete). Open work: the lanes under NOW (M1 #568 Metal, S1 SCNet follow-ups, V1 optional checks, #577/#579 model work) and the holds.** Claim work by adding owner,
source SHA, missing behavior, acceptance gates and next action under "NOW"
before starting. The full scope of each task (exact runs, inputs, commands and
acceptance) is in [the handoff, "Open tasks"](docs/maintenance-handoff.md#open-tasks--fully-scoped).

## NOW — open lanes, fully scoped (2026-11-03)

### Model documentation and issues #577/#579 — Codex (2026-11-07)

- Source `0c39103756be95576202985e9972479648b4b5c9`. First document shipped Haas-ja usage and the actual Ultra/RE-USE support boundaries against the registry and runtime. Then review draft PR #580 and continue the pinned Ultra local-head and RE-USE work with unit and original-model parity tests; assess Nemotron 3.5 30B conversion, resource and Q3/Q4 quality feasibility. Preserve the #568 Metal CPU fallback and defer physical Metal work as requested.

### v0.8.51 publication — COMPLETE (2026-11-07)

- Tag `v0.8.51` → `008a771c57fe285e86727c80e155d5376a9ee44b`. Release **42318289402** (58 assets), wrappers 42318289418 and Docker 42318289400 succeeded. Native audit **42334728938** and wheel audit **42360358086** are accepted. Receipts: [native](docs/historical/release-v0851-native-audit-2026-11-07.json), [wheel](docs/historical/release-v0851-wheel-audit-2026-11-07.json).
- **What blocked PyPI, and the lesson:** the first wheel run, 42334706334, built 10 of 11 wheels and published the GPU index. Its `macos-arm64` job was cancelled at 0 s on both attempts, because GitHub now rejects the `macos-14` runner label. With that wheel missing, `publish-pypi` was skipped, so PyPI stayed at 0.8.50.
  - A rerun cannot help, because it reuses the workflow file from the original commit.
  - Fix `06a620d59` moved every `macos-14` runner use to `macos-15`. Then a fresh `workflow_dispatch` of `release-python-wheels.yml` with `tag=v0.8.51` (run **42359206396**) rebuilt and published PyPI and the GPU index from one run, and the audit used that run.
  - Rule: if a matrix job is cancelled at 0 s with no steps, suspect a rejected runner label, not starvation.
- Closed #574 with the release link.
### Issue #579 Ultra local-head port — Codex (2026-11-06)

- Source `fbde4c5f2c8c3a2d52dabf47ae9512aafb7fe61a`. Port the optional head only against the pinned upstream reference, beginning with the posterior-mean decoder and duration/overlap repair. The [CPU reference run](https://github.com/CrispStrobe/CrispASR/actions/runs/42313568853) stores 44 endpoint inputs and logits for a real 22-word clip. The CPU head graph, offset decoder, acoustic features and opt-in hidden-state capture match the captured CPU reference. The pinned sidecar converter round-trips all 28 arrays. [End-to-end CPU parity](https://github.com/CrispStrobe/CrispASR/actions/runs/42320992672) passed on one English clip. The experimental CLI, C ABI and Python interfaces are on main and passed a separate full-model CPU interface audit; broader audio/language coverage, other bindings and GPU acceptance remain open.

### Issue #579 RE-USE native parity — Codex (2026-11-07)

- Source `f169cc3a3`. The four independent ports and their licenses are audited in [the port audit](docs/reuse-port-audit.md). The one-second 71-boundary CPU comparison `issue579-reuse-cpu-diff-v4` passed the native waveform and all phase-conditioned checks but failed native raw phase and intermediate gates; see the [receipt](docs/historical/reuse-one-second-diff-20261107.json). The experimental diff supports one-second windows and enforces row cosine. Optional window and complex-STFT diagnostics and strict raw-phase regression controls passed targeted tests on experimental branch `72203acc2`. The opt-in oneMKL research frontend at `4830b8e22` passes all 71 unconditioned local 100 ms intermediate checks; the integrated one-second run also passed all 71 unconditioned checks and all 71 conditioned checks. Kaggle FFT provenance `issue579-reuse-fft-env-v1` identified an AVX2 oneMKL dispatch versus the local AVX-512 dispatch; locally forcing AVX2 reproduces the captured Kaggle FFT on one clip. The five-excerpt `issue579-reuse-fft-env-v2` probe confirmed exact AVX2 complex spectra; the explicit `mkl-avx2` profile at `3f43acce9` passed all 71 unconditioned checks against the independently captured `issue579-reuse-original-stages-v1` CPU reference. `issue579-reuse-cufft-v1` found direct cuFFT matches Torch GPU spectra on one clip; native CUDA frontend `1d889456a` passed six CTest targets, GPU frontend controls at three lengths, and all 71 unconditioned checks against original unfused CUDA stages in `issue579-reuse-cuda-profile-v2`. The one-second independent Kaggle CPU model reference `issue579-reuse-original-stages-v2` passed all 71 unconditioned and 71 conditioned checks against local `mkl-avx2`. The profile remains experimental. `issue579-reuse-fused-source-v1` passed a compiler-only preflight for pinned original Mamba sources and pinned CUB headers; `issue579-reuse-fused-source-v2` built the original fused extensions with source unchanged, but model capture stopped before inference because the installed Transformers lacks a required legacy import. `issue579-reuse-fused-capture-v2` captured 72 original fused CUDA tensors with 120 calls to each fused extension. `issue579-reuse-cuda-fused-v1` passed all 71 unconditioned and 71 conditioned comparisons against that original fused capture on one 100 ms T4 input. The one-second original fused capture `issue579-reuse-fused-capture-1s-v1` and native full-stage comparison `issue579-reuse-cuda-fused-1s-v1` passed all 71 unconditioned and 71 conditioned checks on a T4. Long recordings, additional devices, speech quality, and a full GPU model backend remain open. The [window/FFT audit](docs/historical/reuse-window-fft-diagnosis-20261107.json) isolates cross-environment FFT differences and rejects exact-Hann substitution as a parity fix; all existing gates remain. The experimental CPU port and `crispasr-diff` stage runner are on `work/issue579-reuse-decoder`; no product backend has shipped. Original unfused Mamba on CUDA and an independent ONNX network agree with the corresponding adapter when given the same spectra, and all 71 conditioned native stage checks pass against the unfused CUDA capture. The default native frontend and unconditioned 100 ms waveform still fail the investigative gate; the upstream raw first-frame phase itself varies between CPU environments. Broader fused CUDA and normal-duration samples, rates, long-file chunking and speech-quality regressions need separate qualification. Continue using [the audit](docs/reuse-reference-audit.md), independent C++/Rust/ONNX ports and explicit source hashes; do not treat short-clip cross-environment instability as proof that portable inference is impossible.

### Issue #579 model evaluation and release preparation — Codex (2026-11-06)

- Source `b11617ea7d4a6808c0642ff1d0e8c606057d878f`. Evaluate Qwen3-ForcedAligner-Ultra against the existing aligner and RE-USE against its upstream implementation, using pinned checkpoints and independent reference/parity, meaningful unit and end-to-end tests, and hosted resources. Assess #577 and #568 against their documented architectural and physical-hardware gates. Retry the failed Pages deployment; prepare the next patch from the audited main tip, qualify the exact release source, and verify all publishers and release receipts before calling publication complete.
### Issue #579 Ultra aligner parity — Codex (2026-11-06)

- Source `b11617ea7d4a6808c0642ff1d0e8c606057d878f`. Validate the base Ultra checkpoint through the existing Qwen3 forced-aligner conversion and runtime against the pinned upstream Python implementation, document the reproducible scope and retain comparator controls. The optional local head and broader language/quantization qualification remain separate gates.

### Capture variants, issue #577 triage, and Metal #568 — Codex (2026-11-06)

- Source `f6b91a04e2eca3420d30431275190e9adbbd4c93`. Exercise capture-enabled CustomVoice and VoiceDesign plus HTTP failure/disconnect; inspect #577 against actual code and upstream models; investigate #568 on real Apple Metal if available. Accept capture only with actual model output, manifest/hash checks and failure controls; preserve the Metal CPU fallback unless the existing accuracy gates are met. Issue #577 triage is documented; [hosted Metal comparison](https://github.com/CrispStrobe/CrispASR/actions/runs/42235998518) did not reproduce Q8 silence on a virtual Apple device, but physical M1 and full accuracy gates remain open. The [capture variant live audit](https://github.com/CrispStrobe/CrispASR/actions/runs/42236556427) passed CustomVoice, VoiceDesign, injected failure and actual disconnect on CPU; PR CI remains pending.

### PR #576 capture audit — Codex (2026-11-06)

- Source: `ad596e11aa81e82867b37ef372ddb5a27c189b54`. Audit opt-in Qwen3-TTS capture for accurate provenance, failure handling and cross-surface behavior. Model-free capture and voice-pack tests pass; hosted [capture A/B and buffered/streaming HTTP audit](https://github.com/CrispStrobe/CrispASR/actions/runs/42211681989) passes against `910c4f7ae`. CI remains the integration gate; the Metal workflow moves to `macos-latest` because GitHub rejected its `macos-14` runner configuration.

### Main CI repair — claimed by Codex (2026-11-06)

- Source: `de3ff9cbc6ac0429f057b6598020cf75dde338d8`. Fix clang-format violations and the Breeze session control test harness missing the token-budget helper. Accept when the focused test and formatting check pass; verify hosted CI on the fix.

**v0.8.50 is published and verified (2026-11-03).** Tag `v0.8.50` → `5c96e5521c6ab40ab53b91eeec54ced9c30562b5`. Lanes R1–R3 are COMPLETE (details below, kept as evidence). Open work: M1, S1, V1 and the holds. Main moved on after the tag (#573 CI job selection, docs); none of that needs a release.

Each lane gives the start state, the exact steps, the acceptance and what to do on failure. Claim a lane by adding your owner name under it before starting. Never re-tag `v0.8.50`. Never re-upload under an existing PyPI version.

### Lane R1 — COMPLETE: v0.8.50 native publication

- **Done:** release run 41675510092 succeeded on attempt 2 (52 assets); wrappers 41675510012 and Docker 41675510045 succeeded.


- **Start state:** release run **41675510092** (workflow `release.yml`, triggered by the tag push). Language wrappers **41675510012**; Docker publish **41675510045** (re-run after an accidental cancel). At hand-off all three were queued or in progress.
- **Do:**
  1. `gh run watch 41675510092 -R CrispStrobe/CrispASR`. Poll at most every 5 min; a bursty poll loop hit the API rate limit once.
  2. If a job fails, read its log. For a transient runner or network failure, run `gh run rerun 41675510092 --failed`. Never redispatch the whole release: it rebuilds CPU bytes that then collide with immutable PyPI files.
  3. When it succeeds, confirm the GitHub release `v0.8.50` has every v0.8.49 asset name with `0.8.50` substituted (52 names, same set as v0.8.47–v0.8.49).
- **Progress (2026-11-03 ~19:50 UTC):** wrappers 41675510012 and Docker 41675510045 **succeeded**. Release attempt 1 failed only because `build-macos` was cancelled after about 107 min with no log (likely macOS runner starvation; it took 6 min on v0.8.49). Every other job passed, and the failed job was rerun as attempt 2 of the same run. If `build-macos` is cancelled again, rerun `--failed` once more. If it really hangs (a log exists), inspect the macOS build step; SCNet added an OpenBLAS 0.3.34 source build, though only in its qualification workflow.
- **Accept:** all release jobs `success`; 52 assets; wrappers and Docker runs `success`.
- **Then:** dispatch the native audit (lane R2, step 1).

### Lane R2 — COMPLETE: v0.8.50 publication audits

- **Done:** native audit **41710437654** and wheel audit **41718388489**, both `complete: true, acceptance: true`. Receipts: [native](docs/historical/release-v0850-native-audit-2026-11-03.json), [wheel](docs/historical/release-v0850-wheel-audit-2026-11-03.json).
- **Lesson — two wheel runs for one tag:** Release attempt 1 failing and attempt 2 succeeding fired `Release Python wheels` twice (`workflow_run`): 41706418193 (A) and 41710320363 (B).
  - A published PyPI, then was cancelled by B's concurrency group during `publish-gpu-index`.
  - B rebuilt different bytes, and PyPI kept A's files. The first wheel audit, against B, failed at the PyPI readback (`downloaded == record`).
  - Fix: `gh run rerun 41706418193 --failed`, which completed A's GPU index publish, then audit with `wheel_run=<A>`.
  - Next time: the audit's `wheel_run` must be the run whose `publish-pypi` actually uploaded. Check `upload_time` on PyPI against the runs.


- **Depends on:** R1, and the automatic `Release Python wheels` run, which starts via `workflow_run` when Release completes. Find it with `gh run list -w release-python-wheels.yml -L3`. Pages must also deploy the wheel index commit.
- **Do:**
  1. Native audit: `gh workflow run verify-release-publication.yml --ref main -f tag=v0.8.50 -f source=5c96e5521c6ab40ab53b91eeec54ced9c30562b5 -f release_run=41675510092 -f runner=ubuntu-latest`.
  2. When the wheel run and Pages are green, run the wheel audit: the same command plus `-f wheel_run=<wheels run id> -f native_audit_run=<native audit run id>`.
- **Accept:** each run's `release-publication-<run>` artifact `receipt.json` has `complete: true, acceptance: true`. Commit the receipts under `docs/historical/release-v0850-{native,wheel}-audit-2026-11-0X.json`, following the v0.8.49 receipts.
- **If it fails:** read the traceback. A real defect in a published file needs a new patch release (v0.8.51), never a re-upload. `REVIEWED_TAGS` in `tools/ci-heavy/release_publication_audit.py` already includes `v0.8.50`.

### Lane R3 — COMPLETE: #562, #564, #569, #570 closed with the release link

<!-- original scope kept below -->

- **Depends on:** R1 (the release page exists).
- **Do:** comment with the release link and close **#562** (Mini-BS-RoFormer), **#564** (MOSS-Diarize beam/sampling), **#569** (moss-audio 30 s pad) and **#570** (LID download consent). Each already has a fix-explanation comment; just add "Released in v0.8.50 <link>". Keep **#568** open (lane M1).

### Lane M1 — #568 Mini-BS-RoFormer Metal output (needs Apple hardware)

- **State:** mitigated in v0.8.50. On Metal the backend falls back to CPU (`src/mini_bs_roformer.cpp`, search `#568`); `CRISPASR_MINI_BS_ROFORMER_METAL=1` opts back in. The root cause is unknown; the reporter suspects the per-stem decoder or mask application.
- **Do, on an Apple Silicon Mac:**
  1. Build with Metal. Run `crispasr --separate -m mini-bs-roformer -f <stereo 44.1 kHz speech>` with `CRISPASR_MINI_BS_ROFORMER_METAL=1`, and again with `--no-gpu`.
  2. Bisect with the stage dump or diff tooling (`crispasr-diff` / `tools/` Mini-BS validators) to find the first diverging stage.
  3. Fix it in ggml-metal or in the graph.
- **Accept:** Metal stems match CPU (vocals RMS within 1 %; SDR on the reporter's synthetic band within 0.5 dB of CPU) for F16/Q8_0/Q4_K. Then remove the CPU fallback, keep the env var as a kill switch, and close #568.

### Lane S1 — SCNet follow-ups (status 2026-11-07)

SCNet is on main (v0.8.50+). Review findings from PR #566:
1. **Open — graph cache hit rate.** The random per-chunk shift prefix changes the frame count, so the cached graph probably misses on most chunks. Measure it with an actual model (log cache hits in `src/scnet.cpp` graph reuse over a 60 s file). If it misses, bucket the shape (pad the prefix to a fixed frame count). Any numeric change requires a re-run of `scnet-qualification.yml`.
2. **Fixed and verified** (qualification 42371575575 runs real-model 0.01 s and 100-sample-tail controls). `forward()` no longer aborts on inputs too short for the reflective STFT pad; it zero-extends to `FFT/2+1` samples and trims. Only inputs that previously threw take this path, so qualified results are unchanged. **To verify:** run the CLI `--separate` with an SCNet model on a 0.01 s stereo file, and on a file whose final segment is shorter than 2048 samples (with `--shifts 0`). Both must succeed with finite stems of the input length.
3. **Not applicable.** The server always interleaves to 2 channels before `scnet_separate`, so the ">2 channels → 500" path cannot occur.
4. **Won't fix here.** `crispasr_set_gpu_device_pref` is process-wide by design for every backend (Mini-BS-RoFormer does the same). Changing it is a codebase-wide refactor, not an SCNet bug.
5. **Fixed.** The server separation endpoint (all separators) now returns 400 for empty audio, and 413 above `CRISPASR_SERVER_SEPARATE_MAX_SECONDS` (default 1800; 0 disables).

### Lane V1 — optional hosted checks for #569 / #570

- #569: run moss-audio on a short (8 s) clip with a beep and a hit, as in the issue. Check that the log shows about 100 enc tokens and that event times fall inside the clip. Locally verified only on JFK (138 tokens, was 375).
- #570: check that `--backend moss-audio --prompt ...` prints `skipping language detection` and downloads nothing. This path is untested; the consent path itself was verified with an empty cache.

### Holds (unchanged; do not schedule speculative repairs)

#558 (waiting on reporter), #532 (needs a reproducible failing recording), #483 (GTX1660/MX150 hardware), #461 (Vulkan voxcpm2 hardware), #456 (feature request, unscheduled). Lane E (real-hardware `+vulkan`/`+hip`/aarch64 `+cuda` wheel qualification) still needs hardware.

## Earlier NOW entries (historical, 2026-10-30 to 2026-11-02)

### SCNet / SCNet-large — native port recovery, 2026-11-02 (INTEGRATED 2026-11-03 as `57fb8ff46`; see lane S1)

- Owner: SCNet recovery team (native graph, conversion/reference/validation,
  CLI/Session integration); starting source
  `04200fae791c8d5113fa47d5896ccd13f18f6bad`.
- Recovery: no SCNet branch, commit, pull request, or source files were found in
  the repository at resumption. Earlier chat progress is not validation evidence.
  Rebuild against the current source and checkpoint reviewable work in Git.
- Missing behavior: source-faithful native ggml inference for the author's SCNet
  and SCNet-large checkpoints, GGUF conversion, F16/Q8_0/Q4_K precision policy,
  four-stem stereo separation, CLI and Session/C ABI architecture detection.
- Acceptance: pin the official source and checkpoint hashes; compare every
  required original-Python stage and complete stems with shape, finite-value,
  amplitude and error checks (F16 cosine >0.999, Q8_0 >0.98, Q4_K >0.94).
  Preserve the published normalization, normalized rectangular STFT, recurrent
  FFTs, overlap weighting and per-chunk shift behavior. Measure matched-input,
  matched-device warm inference against the original implementation; each
  qualified precision must at least match its timing. Record failures and
  hardware/fixture limits without weakening these full-stage gates or promoting unqualified
  models. Exercise CLI and actual Session/C ABI, including neutral filenames,
  cold/warm reuse, changed inputs, result ownership and invalid arguments.
- Implemented: native ggml graph with explicit CPU recurrent/FFT operations,
  strict 372-tensor converter, original-method 35-stage capture, quantizer policy,
  CLI/server/Session wiring, bounded graph caching, and complete validation tools.
  Actual local F16/Q8_0/Q4_K synthetic-weight graphs pass finite-output, geometry,
  cold/warm/changed-input reuse checks; mathematical, converter, routing and
  comparator rejection controls pass. See [backend documentation](docs/scnet.md)
  and [bounded local receipt](docs/historical/scnet-local-validation-2026-11-02.json).
- Hosted progress: [preflight 41395094930](https://github.com/CrispStrobe/CrispASR/actions/runs/41395094930)
  built the production native library/probe and quantizer with OpenBLAS; math and
  routing controls passed. Its earlier quantizer-test filename collision is
  corrected in `1ef9331`. The [first qualification run](https://github.com/CrispStrobe/CrispASR/actions/runs/41396683922)
  exposed a NumPy/Accelerate dependency conflict before model execution; retain
  Accelerate 0.32.1 and pin compatible NumPy 1.26.4 for the next run.
- [Run 41397985492](https://github.com/CrispStrobe/CrispASR/actions/runs/41397985492)
  built both production variants and passed generated capabilities, C ABI,
  wiring, and original-observer controls. Its actual CLI generated the checked-in
  145-backend catalogue. Small-model raw F16 passes every full-stage and output
  metric but failed the added angle-only row rule. Independent original-F16
  storage and original-GELU-layout experiments reproduced these failures;
  [the documented row-policy correction](docs/scnet.md#numerical-and-performance-gates)
  retains the global gates and requires fresh acceptance. Large conversion
  identified the author's distinct band ratios `[0.225, 0.372, 0.403]`; the
  native/reference contracts are corrected and the small graph is bit-identical.
- Quantization diagnosis: default Q8_0 and Q4_K failed the quiet bass stem in
  the 0.25-second raw probe. Keeping the four final spectral reconstruction
  matrices in F16 repairs all 35 stage gates for both formats, with every other
  tensor byte-identical and only 45,600 extra bytes per small model. The default
  policy retains 99.4469% Q8_0 and 76.5695% Q4_K matrix-element coverage. Preserve
  the failed defaults and [candidate evidence](docs/historical/scnet-quantization-diagnosis-2026-11-02.json);
  require fresh raw probes for all precisions plus full overlapping-audio fidelity
  and performance before promotion.
  The [genuine large raw probe](docs/historical/scnet-large-raw-2026-11-02.json)
  also passes all 35 stages in all three formats under this default policy,
  with 99.56995% Q8_0 and 97.65234% Q4_K matrix-element coverage.
- [Run 41403856227](https://github.com/CrispStrobe/CrispASR/actions/runs/41403856227)
  at `94ea189` passes all 35 full-pipeline stages and timed-output fidelity for
  small SCNet in F16/Q8_0/Q4_K, but fails performance: native medians are
  46.31/46.53/46.64 seconds versus original 16.43/16.15/15.70 seconds.
  This predates the four-matrix quantization repair and is retained as a
  [failed runtime baseline](docs/historical/scnet-runtime-baseline-2026-11-02.json),
  not current-policy qualification. The [large job](docs/historical/scnet-large-runtime-baseline-2026-11-02.json)
  likewise passes numerical and production-surface gates, but takes about
  146.5–147.1 seconds versus 49.5–50.4 seconds original on its separate worker.
  Graph creation costs only 0.07–0.16 seconds per complete call.
- The [focused profile](docs/historical/scnet-runtime-profile-2026-11-02.json)
  at `b51801b` reproduces the exact full small-model baseline waveform and
  attributes 18.225 seconds to recurrent gates, 3.981 seconds to recurrent GEMMs,
  6.288 seconds to other BLAS operations, and substantial copy/layout/broadcast
  costs. SIMD gate and CPU layout changes pass all 35 native and independent
  original-reference stages for both sizes and all three formats on the short
  fixture. Forced-scalar F16 runs reproduce all 35 earlier native tensor payloads
  exactly, isolating the layout changes from the activation approximation.
  See [optimization controls](docs/historical/scnet-runtime-optimization-controls-2026-11-02.json).
- The [optimized focused run](docs/historical/scnet-runtime-optimized-profile-2026-11-02.json)
  at `4264d2e` passes all four final-waveform gates for small F16 in both warmups
  and all three alternating timing pairs, but native median 21.761 seconds still
  exceeds the same-host original median 16.406 seconds. The instrumented run
  attributes 1.592 seconds to gates, 4.378 seconds to recurrent GEMMs, 6.942 seconds
  to other BLAS operations, and 2.817 seconds to internal FFT nodes. This worker
  uses EPYC 9V74 hardware; earlier cross-host timings are descriptive only.
  This diagnostic checks no intermediate stages or quantized models, and its
  overall acceptance and qualification remain false.
- Exact SIMD FFT, convolution column layout and grouped recurrent bias fusion
  now pass all 35 original-reference stages for both sizes and all three formats
  in fresh genuine short probes. Actual OpenBLAS outer-product routing is
  verified in every case. [The controls](docs/historical/scnet-fft-convolution-controls-2026-11-02.json)
  retain component identities and numerical scope. Dependency [ggml PR #6](https://github.com/CrispStrobe/ggml/pull/6)
  fixes the first outer product's thread budget; its exact source is pinned.
- The [next focused profile](docs/historical/scnet-runtime-fft-convolution-profile-2026-11-02.json)
  at `6a4c805` passes every final-waveform check but still fails speed: native
  median **18.604 seconds** versus original **16.581 seconds** on the same
  two-thread EPYC 7763 worker. Its instrumented call attributes 5.402 seconds
  to recurrence (3.924 GEMM, 1.472 gates), 4.107 seconds to other matrix
  multiplications, 2.471 seconds to convolution outer products, and 1.956
  seconds to internal FFT nodes. The receipt retains all three timing pairs,
  source snapshots, provider/thread identities and the failed performance gate.
- Shared bidirectional input projections and paired real FFTs now pass all 35
  native and independent original-F32 stage checks for both sizes in all three
  formats on the genuine short fixture. The [new controls](docs/historical/scnet-real-fft-projection-controls-2026-11-02.json)
  bind the unchanged model bytes, new binaries, actual fused projection routing,
  exclusive FFT worker ownership and the unchanged numerical gates. The
  component FFT receipt retains its extreme unequal-amplitude conditioning
  limit; these short diagnostics make no full-pipeline or speed claim.
- Full qualification binds the five native entrypoints and their complete
  standalone loader resolutions before and after execution. Packaging
  requires this proof and a fresh dependency check. The [controls](docs/historical/scnet-linkage-controls-2026-11-02.json)
  include 13 rejection-control groups and genuine five-entrypoint/16-library
  snapshots, independently repeated. These are tooling controls, not a model
  qualification result.
- [Run 41469961180](https://github.com/CrispStrobe/CrispASR/actions/runs/41469961180)
  at `0b5639c` passes every final-waveform check but fails speed on its EPYC
  9V45 runner: default native median **18.806 seconds** versus original
  **6.395 seconds**; passive **19.229** versus **6.724 seconds**. All calls are
  retained in the [receipt](docs/historical/scnet-runtime-real-fft-projection-profile-2026-11-02.json).
  OpenBLAS 0.3.20 selects Barcelona kernels on this newer CPU while Torch uses
  AVX512. Prior workers used different CPUs, so their absolute times are not
  a controlled comparison with this run.
- Dependency repair: CI now builds official OpenBLAS 0.3.34 at
  `e0166008be8e466242aa76b2ff75ce3f0fbf574a`, with automatic CPU dispatch,
  LP64 integers, pthreads, and two runtime threads. The tested helper, 12
  independent matrix controls and six ggml thread controls pass. The local
  SCNet archive is byte-identical to the preceding provider build. The
  [provider controls and native-only baseline](docs/historical/scnet-openblas034-build-controls-2026-11-02.json)
  establish the dependency and repeated-output identities, not model or speed
  acceptance.
- Allocation repair: CPU uses one persistent scheduler and
  allocator with fresh metadata when geometry changes. The previous two-entry
  cache misses all three recurring default-fixture shapes. The pool retains its
  largest capacity until model destruction; GPU keeps its existing cache.
  [Local diagnostics](docs/historical/scnet-local-runtime-allocation-diagnostic-2026-11-02.json)
  preserve the failed timing checks and resource counters that prompted this
  investigation. Minor-fault counts do not directly attribute allocation cost.
  The [independently audited controls](docs/historical/scnet-cache-allocator-controls-2026-11-02.json)
  pass all 210 short-stage comparisons and 11 cache lifecycle cases. Two ggml
  allocation-failure bugs were reproduced and repaired; all 12 failure/retry
  groups pass with address and undefined-behavior sanitizers. On the retained
  native-only full fixture, output bytes match, minor faults fall 81.38% and
  peak RSS falls 29.21%, but the single measured call is 3.07% slower. This
  establishes no comparative speed qualification.
- Actual mapped libraries now have independent
  [runtime identity controls](docs/historical/scnet-loaded-runtime-controls-2026-11-02.json),
  separate from standalone loader resolution. Torch's mapped OpenMP runtime
  is retained, and all three precision proofs must pass again at packaging.
- CPU BLAS now materializes 24 immutable recurrent matrices once at model load.
  All 144 tensors across both sizes and three precisions exactly match actual
  ggml casts. All 72 observed ordinary recurrent splits plan and execute one
  ggml worker while OpenBLAS retains two threads. This adds 15 MiB for SCNet or
  60 MiB for SCNet-large and leaves original GGUF tensors and storage types
  intact. The [final candidate controls](docs/historical/scnet-recurrent-weight-controls-2026-11-02.json)
  also pass all 210 original-F32 stage checks with byte-identical stage payloads
  versus the preceding implementation. These controls establish scheduling,
  conversion and short-input fidelity, not performance acceptance.
- Main's newer Metal work is preserved with both SCNet dependency fixes in
  ggml `1b972bb754aa5f83e2d872ccf37a335717f6658d`; CPU SCNet and allocator
  source bytes are unchanged from the final local checks.
- The [fresh full hosted run](https://github.com/CrispStrobe/CrispASR/actions/runs/41489721058)
  stopped before provider or native compilation: `setup-python` supplied
  `LD_LIBRARY_PATH`, which the provider guard correctly rejected. The
  [failure receipt](docs/historical/scnet-hosted-provider-prebuild-failures-2026-11-02.json)
  contains no model result. Both qualification workflows now remove that
  variable before shell/Python startup and verify the selected interpreter
  before installation; [startup controls](docs/historical/scnet-python-clean-start-controls-2026-11-02.json)
  bind the official Python artifact and its embedded RUNPATH. Provider and
  loaded-runtime guards remain unchanged.
- The [normal-runtime local F16 diagnostic](docs/historical/scnet-runtime-normal-access-profile-2026-11-02.json)
  uses ordinary Torch AVX512 dispatch, with no ISA override. All warmup and
  measured waveform checks pass. Native median **17.124176 seconds** narrowly
  beats original **17.301418 seconds** across three alternating pairs on the
  same EPYC 9V74 worker with two threads. This is not complete qualification.
  The [native-only full-output replay](docs/historical/scnet-recurrent-weight-full-output-2026-11-02.json)
  and [exact historical control sources](docs/historical/scnet-recurrent-weight-control-sources-2026-11-02/README.md)
  preserve the earlier observations and their original compiled-source scope.
- The AVX512 gate extension preserves the existing AVX2 equations, constants,
  FMA grouping, tails and fallbacks. Its [component controls](docs/historical/scnet-avx512-gate-controls-2026-11-02.json)
  pass all 12 groups, including bit-identical activation, gate-state and
  recurrent-feedback comparisons. Production model and speed acceptance still
  require the complete campaign. The unrelated adaptive-convolution candidate
  is not part of this implementation.
- Generic CI exposed an inherited Mini-BS Windows `setenv` build error; the
  Metal-only setting is now guarded by `__APPLE__`, preserving dynamic Metal
  behavior. [Baseline CI triage](docs/historical/scnet-generic-ci-baseline-triage-2026-11-02.json)
  also retains unrelated ARM harness failures without treating them as SCNet
  numerical results.
- The [first complete local small campaign](docs/historical/scnet-first-full-small-2026-11-02.json)
  at `a80cbbba` passes all 210 short/full stage checks, all six warmups and 18
  timed waveform checks, every production CLI/Session surface, and all runtime
  identity checks. It fails all three speed gates: native/original medians are
  **15.860326/14.749629 s** F16, **16.363169/16.271565 s** Q8_0, and
  **18.890005/17.151493 s** Q4_K. All calls and the automatic diagnostic are
  retained; no model was packaged. Peak child RSS was 3,573,896 KiB for this
  capture-heavy campaign, not a standalone production-memory measurement.
- The [hosted large-model campaign at the same `a80cbbba` source](docs/historical/scnet-a80-large-qualification-2026-11-02.json)
  passes all numerical, surface, runtime and speed gates in all three formats
  on its EPYC 7763/AVX2 worker. Native/original medians are
  **64.442049/73.001402 s** F16, **64.607757/72.975983 s** Q8_0 and
  **64.819255/73.263762 s** Q4_K. It produced a success-only hosted bundle.
  The independent evidence audit passes; the separate bundle bytes have not
  yet been downloaded/audited. This qualifies that exact large-model CPU build
  and fixture, not subsequent runtime changes or the small model.
- CPU convolution packing now chooses frequency tiles from the actual time and
  kernel dimensions, with a strict **128 MiB per packing tensor** budget.
  Complete time rows, coefficient precision and frequency order are preserved.
  The maximum supported single-frequency column is 45.439 MiB, so valid inputs
  always have a fitting tile; invalid/overflowing geometry raises the existing
  recoverable error. Total graph and process memory are separate from this
  individual-buffer bound. [Component and short-reference controls](docs/historical/scnet-adaptive-tiling-controls-2026-11-02.json)
  bind this candidate. Its [complete local campaign](docs/historical/scnet-c2-full-small-2026-11-02.json)
  at `9724c205` again passes all 210 stage decisions, all 96 warmup/timed stem
  decisions, production surfaces and runtime identities, but fails all three
  speed gates: native/original medians are **16.782511/14.785485 s** F16,
  **16.493169/16.185961 s** Q8_0 and **16.303297/15.609759 s** Q4_K.
  The exact binaries, source, first-attempt timings and diagnostic attribution
  are retained; no model was packaged.
- Current implementation work targets the recurrent matrix loop and gate
  updates with immutable packed F32 weights and independent ggml worker batches.
  This candidate must preserve the original bias/state equations, runtime
  fallback paths, thread budget and all numerical/performance gates. Its new
  accumulation order passes [component and original-reference controls](docs/historical/scnet-packed-recurrence-controls-2026-11-02.json):
  all six component groups, twelve concurrent/clamped callback comparisons,
  and all 210 short model-stage decisions pass. Complete waveform and speed
  qualification remain pending on its exact committed source.
- Next action: qualify both model sizes directly
  through all original-checkpoint, raw, full-pipeline, CLI/Session and paired
  performance gates in `scnet-qualification.yml` and [draft PR #566](https://github.com/CrispStrobe/CrispASR/pull/566).
  The dedicated `ci/scnet-qualification-20261102` branch triggers the expensive
  campaign on relevant source changes. Concurrency is keyed by commit so that
  distinct candidates retain independent runs; ordinary feature-PR checks
  remain in place. The preserved [a80 hosted run](https://github.com/CrispStrobe/CrispASR/actions/runs/41506291896)
  has completed: the small-model campaign fails only performance, while the
  large-model campaign passes as recorded above. The separate hosted
  `9724c205` campaign remains attached to its exact source.
  Complete CPU qualification, GPU qualification and held-out quality remain unproven; do not promote the
  candidate or substitute diagnostic runs for these gates.
- License: upstream [MIT notice](https://github.com/starrytong/SCNet/blob/main/LICENSE)
  plus the [author's explicit checkpoint permission](https://github.com/starrytong/SCNet/issues/35)
  cover the author's released SCNet/SCNet-large weights, including format-converted
  redistribution with attribution. Retain the notice and exact weight provenance.

### Separation and speech frontend candidates — license inventory, 2026-11-02

These are research candidates; this inventory does not claim implemented or
qualified native support. Keep code and checkpoint permission distinct.

| Candidate | Source-code terms | Pretrained-weight evidence / next gate |
|---|---|---|
| SCNet / SCNet-large | [MIT](https://github.com/starrytong/SCNet/blob/main/LICENSE) | [Author confirms MIT and converted-checkpoint redistribution](https://github.com/starrytong/SCNet/issues/35); active port above. |
| Mini-BS-RoFormer-18M | MIT declared in author's model repository | [MIT model card](https://huggingface.co/HiDolen/Mini-BS-RoFormer-18M/blob/main/README.md); separate architecture/checkpoint qualification from shipped V2.5. |
| Mini-BS-RoFormer 8.83M | MIT declared in author's model repository | [MIT model card](https://huggingface.co/HiDolen/Mini-BS-RoFormer/blob/main/README.md); separate architecture/checkpoint qualification from shipped V2.5. |
| MossFormer2_SS_16K | [Apache-2.0](https://github.com/modelscope/ClearerVoice-Studio/blob/main/LICENSE) | [Official weights declare Apache-2.0](https://huggingface.co/alibabasglab/MossFormer2_SS_16K/blob/main/README.md). |
| GTCRN | [MIT](https://github.com/Xiaobin-Rong/gtcrn/blob/main/LICENSE) | Checkpoints are inside the official MIT repository; separate explicit weight-scope confirmation not located. |
| UL-UNAS | [MIT](https://github.com/Xiaobin-Rong/ul-unas/blob/main/LICENSE) | Checkpoints are inside the official MIT repository; separate explicit weight-scope confirmation not located. |
| DBS-TIGER-DnR | [Apache-2.0 with retained third-party MIT notices](https://github.com/JusperLee/Dynamic-Band-Splitting#license) | [Official model card declares Apache-2.0](https://huggingface.co/JusperLee/DBS-TIGER-DnR). |
| ZipEnhancer | [Apache-2.0 in ModelScope](https://github.com/modelscope/modelscope/blob/master/LICENSE) | [Exact official weight listing](https://www.modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base) declares Apache-2.0. |
| DeepFilterNet3 | [MIT or Apache-2.0 for code](https://github.com/Rikorose/DeepFilterNet#license) | Explicit checkpoint-redistribution scope remains unconfirmed; [issue #712](https://github.com/Rikorose/DeepFilterNet/issues/712) has no maintainer clarification. |
| SAM Audio family | [Custom SAM License](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE) | Same custom terms cover trained weights; [official checkpoint access is gated](https://huggingface.co/facebook/sam-audio-base). Requires a separate license review; not a permissive MIT/Apache candidate. |

### Mini-BS-RoFormer V2.5 — CPU qualification complete, 2026-11-01

- Owner: model-port team (native graph, blueprint/converter/reference, integration, quantization/validation); starting source `1dd0bf6fb9e7a8bf5849cb7a14605b2234e6ee93`.
- Delivered behavior: native support for `HiDolen/Mini-BS-RoFormer-V2.5-46.2M` with four stereo stems, original chunking, F16/Q8_0/Q4_K weights, and optimized ggml inference.
- Acceptance: pinned original Python stages and complete stems; strict Q8 cosine >0.98 and Q4_K >0.94, shape/finite/amplitude checks; same-input warm performance at least matching the Python blueprint on each qualified device; CLI/C ABI and separation wiring tests. Preserve sensitive tensors at sufficient precision based on measured errors.
- Implemented: pinned converter/original-reference capture, native cached ggml graphs, precision policy, full 144-backend production build, generated catalogues, and CLI/server/Session integration. Final F16/Q8_0/Q4_K Session/CLI checks pass cold/warm and changed-input reuse, fresh-session agreement, owned results, neutral-filename detection, getter bounds, and selected-stem WAV agreement.
- CPU optimization: supported Linux x86 AVX512 BF16 hosts select packed BF16 matrix operands with F32 accumulation and other neural operations; the portable CPU path stays F32. Shared conversion nodes, 96-row matrix blocks, exact-size frequency attention, and common affine decoder projections reduce runtime. The selected production library is `f862d6bd1079bb787695f44d0c5fad281488bb4c6f391bfe09b3e80e4e60608d` (SHA256), with all acceleration overrides unset.
- Quality: all three variants pass all 25 required stages against the pinned original F32 blueprint with the genuine CrispEmbed comparator. Minimum whole-tensor cosine is F16 `0.999978781`, Q8_0 `0.999848187`, Q4_K `0.985801816`; the F16 gated-row minimum is `0.999764860`. Genuine CrispEmbed and CrispASR quantizers produce byte-identical Q8_0/Q4_K artifacts. All three no-BLAS waveform fallback checks also pass.
- Performance: on the recorded AMD EPYC 9V74 CPU at two threads, all three variants beat the original published BF16 loader's `separate(batch_size=4)` medians across three alternating pairs. Eight-second speedups are F16 `1.014222x`, Q8_0 `1.215138x`, Q4_K `1.142296x`; twelve-second speedups are `1.178146x`, `1.169934x`, `1.071020x`. The twelve-second run's memory-limit interruption, seven retained pairs, two resumed Q4_K pairs, and separate untimed metric checks are explicit in the evidence. No completed timing was replaced.
- Publication and scope: [PR #563](https://github.com/CrispStrobe/CrispASR/pull/563) carries the implementation and exact-head hosted checks. See [the qualification report](docs/mini-bs-roformer-validation.md) and [portable evidence](docs/historical/mini-bs-roformer-v25-cpu-validation-2026-11-01.json) for hashes, retained failed experiments, and recovery details. These measurements qualify the recorded CPU profile on one implementation-parity fixture; GPU speed/parity, other hosts, and held-out source-separation quality remain unmeasured.

### Main integration follow-up — 2026-10-31

- Owner: audit repair team; source `d71788aa769334940d12096b549f5edd2f7770e7`.
- Evidence: current main's complete tree equals the delivered repair tree `1695e0835a492e5e2018bfafffe4ff6ded3ea0ea`; all 79 changed files are present and identical. The 131-case local parity rerun passes.
- Missing behavior: repair the actual `linux-unit` build failure in CI run `41196253916` and explicit CPU-backend-header lint failure in run `41196253931`; retain all numerical gates.
- Completed locally: reproduced and repaired three Omnivoice test calls missing the new thread argument and two missing direct CPU-backend header includes. The complete configured CPU/CBLAS/OpenMP build passes; 133 performance-parity CTests pass; the strengthened discovery guard checks 49 required cases across 14 executables; all 129 header-lint files pass. See [integration follow-up](docs/post-release-audit-integration.md).
- Broader unit result: 2,164 pass, six skip, and three fail at demonstrated unavailable OS facilities; nine additional OS-dependent cases were excluded and remain unqualified. These results do not establish a complete unit-tier pass.
- Next action: apply the follow-up patch, then rerun CI/Lint on its exact source. The separate hosted Moonshine acceptance failure predates this integration but remains internally undiagnosed; obtain the `heavy-41196253946` artifact and failing inner log before proposing a repair. No upstream publication is included.

### Post-v0.8.49 audit repairs — 2026-10-31

- Owner: audit repair team; starting source `26945b343b855dda9b0bce5e653cd5832c60bf7a`.
- Missing behavior: preserve ArkTTS graph outputs, restore bounded and useful CPU parallelism, correct iSTFT scratch growth, RVQ reduction order, cosine self-similarity, MeanVC2 conditioning/interpolation and Haas dispatch; make numerical regression tests discoverable and wire portable CBLAS consistently.
- Acceptance: production-kernel model-free regression tests, CPU scalar/CBLAS checks, actual CTest discovery, explicit external model/platform qualification jobs, and a reproducible downloadable patch/source bundle. GPU, cross-platform and real-model quality results require their respective workers and fixtures.
- Completed locally: production repairs and test/validation infrastructure on `fix/post-v0849-audit`; implementation/test commit `1b5a16c40118a3b15d53c7dd6bbc66c7458bc203`. Full shared library/CLI build, 131 selected CTests, actual Haas entrypoints, 73 registry cases, 738 exact release comparisons, CBLAS/OpenMP variants, and Go dependency-list consistency passed. See [repair report](docs/post-release-audit-repairs.md).
- Remaining qualification: real-model fixtures/reference captures, autoregressive/full-pipeline quality and timing on the target GPU, macOS/Windows/ARM workers, and MeanVC2 speaker extraction/generic Session conditioning. Historical completion claims below do not establish these gates for this repair.
- Next action: review/apply the downloadable source and patch bundle, then run the prepared manual platform/model gates with qualified fixtures. No publication to the upstream repository is part of this task.

### Current handoff — 2026-10-30 18:05 UTC

Updated 2026-10-30 18:05 UTC: Haas-ja-v1.0 Japanese ASR and MeanVC2 streaming zero-shot voice conversion COMPLETE.
- Haas-ja-v1.0: 600M param Japanese ASR (third-intelligence/Haas-ja-v1.0, Apache-2.0). Reused Parakeet runtime with 2-layer KDA predictor (linear RNN + rank-1 state update + gated RMSNorm). 1.000000 cosine parity on `crispasr-diff` (encoder, KDA predictor, joint network), 0 CER verbatim match on Japanese test audio, 11.3x realtime on Apple M1. GGUFs (Q8_0, Q4_K, F16) published to `cstr/Haas-ja-v1.0-GGUF`. Wired in model registry, backend maps, and feature matrix. Issue #561 CLOSED.
- MeanVC2: 5/5 parity stages passed on `crispasr-diff` (>= 0.999998), 0.0% WER on Whisper ASR intelligibility verification, Q4_K/F16 quantizations verified, Catch2 unit tests + standalone convert harness integrated.

| Priority | Open work | Current evidence / next gate |
|---|---|---|
| 0 | COMPLETE: Audio pipeline & model decoders acceleration across FBank, Resampling, Core Mel, Parakeet KDA, SenseVoice, Paraformer, CTC Beam Search, Wav2Vec2, CTC Forced Alignment, IBM Macaron Conformer Shaw Block Attention, CPU LayerNorm, ConvT Permutation, FireRed ASR, F5-TTS, VoxCPM2, Mel-Band RoFormer, Speaker Diarization, Kokoro TTS, CQT, CQT2010v2, Diarization Smoothing, RVQ, and iSTFT | 1. `src/core/kaldi_fbank.cpp`: Precalculated single-pass power spectrum, sparse Mel filter triangular pruning, and multithreaded frame processing (fbank+lfr frontend dropped 9.1x from 20.4ms to 2.24ms on JFK audio). 2. `src/core/lfr.h`: Direct zero-copy output assembly eliminating intermediate padded buffer heap allocations. 3. `src/core/audio_resample.cpp`: FilterKey Kaiser-sinc caching, branch-free tap interval bounding, tap index striding, and parallel chunks (>2770x realtime, 10s audio resampled in 3.61ms). 4. `src/core/mel.cpp`: Direct pre-emphasis computation fused into center-padded buffer eliminating intermediate allocations; NEON/AVX SIMD vectorized frame windowing; multithreaded non-BLAS projection; chunk-parallel `apply_log`; multithreaded chunk-parallel `PerFeatureZ` normalization; and cache-friendly parallel transpose for `Layout::MelsTime` (92 unit test assertions passed across 3 test cases in `test-core-mel-features`, bit-identical output across 100% of floats). 5. `src/parakeet.cpp`: Reusable scratch buffer on `state.scratch` (zero heap allocations in KDA predictor) and `cblas_sgemv` BLAS acceleration. 6. `src/sensevoice.cpp`: Query embedding table pre-dequantization (query_embed time 0.00ms), chunked multithreaded CMVN & log-softmax, zero-alloc single-pass CTC greedy decode (SenseVoice Small: 19.0x-33.2x realtime on Apple M1 Metal GPU, verbatim JFK 0% WER). 7. `src/paraformer.cpp`: Load-time CMVN and CIF predictor weight caching, parallel chunked Conv1d and Linear alpha generation, precalculated PE inverse frequencies. 8. `src/core/ctc.h`: Replaced $O(N)$ linear scans in `prefix_beam_search` with an $O(1)$ custom-hashed prefix lookup map, speeding up large-vocabulary CTC beam search by >10,000x per frame. 9. `src/wav2vec2-ggml.cpp` & `src/core/cpu_attention.h`: Vectorized SIMD 4-way unrolled dot products (`cpu_attn::cpu_dot`) across Conv1d & GroupedConv1d; 4-way vector unrolled `cpu_online_softmax_accumulate` with zero memory allocation; multithreaded chunk-parallel channel and frame loops with `core_parallel::for_each_chunk` for Conv1d, GroupedConv1d, LayerNorm, InstanceNorm1D, LayerNormCF, and 24 layers of encoder self-attention ($n\_heads \times T$). Gated via `CRISPASR_WAV2VEC2_SERIAL=1` fallback. Live model JFK alignment verified bit-identical. 10. `src/align.cpp` & `src/canary_ctc.cpp`: Chunk-parallel log-softmax across CPU cores (`core_parallel::for_each_chunk`), precomputed skip-blank state invariant eliminating 750k DP branch conditions, inductive Viterbi state bounding ($j < \min(S, 2t + 2)$), and unique token log-softmax pruning in Canary CTC forced alignment. Gated via `CRISPASR_CTC_ALIGN_SERIAL=1` fallback. Paired A/B benchmark (`test-align-ab-bench`): 100% bit-identical timestamps across 22 words, Arm A (serial baseline) median 78.27ms vs Arm B (accelerated parallel) median 20.77ms -> 3.77x-4.13x speedup. 11. `src/core/conformer_ibm.h`: Conformer Shaw Block Attention CPU acceleration (used by Granite Speech and Granite NLE) with cache-friendly contiguous V-matrix streaming accumulation, zero per-block heap allocation with reusable scratch buffers, contiguous pointer dot product acceleration, and multithreaded `core_parallel::for_each_chunk` over $n\_blocks \times n\_heads$. Gated via `CRISPASR_CONFORMER_SERIAL=1` fallback. Paired A/B benchmark (`test-conformer-ab-bench`): 100% exact bit-identical output across 524,288 floats (max_diff = 0.0f), Arm A (serial baseline) median 44.00ms vs Arm B (accelerated parallel) median 5.90ms -> 7.46x speedup (38.10ms saved per block attention). 12. `src/core/cpu_ops.h` & `src/core/conv.h`: SIMD 4-way unrolled mean/var and affine projection with frame-parallel chunking in `core_cpu::layernorm` (5.50x faster, median 1.09ms -> 0.20ms, max_diff 2.62e-6), gated via `CRISPASR_CPU_OPS_SERIAL=1`; multithreaded tile chunking in `core_convt::permute_convt1d_host` (1.85x faster, median 6.00ms -> 3.24ms, 100% bit-identical across 4,194,304 floats, max_diff 0.0f), gated via `CRISPASR_CONV_SERIAL=1`. 13. `src/firered_asr.cpp` & `src/f5_tts.cpp`: Parallelized FireRed ASR CPU matmul over output dimension N (M=1 token decoding) and batch dimension M (beam search batch) using `core_parallel::for_each_chunk`, plus Conformer rel-shift self-attention across (nh x T) and decoder cross-attention (gated via `CRISPASR_FIRERED_SERIAL=1`, 1.85x speedup, max_diff = 0.0f). Replaced unsupported/silent Apple Clang `#pragma omp` with `core_parallel::for_each_chunk` over output channels C_out in F5-TTS `cpu_conv1d`, `cpu_conv1d_dil` (HiFi-GAN resblocks), and vocoder transpose conv upsamplers (gated via `CRISPASR_F5_SERIAL=1`, 2.98x speedup, max_diff = 0.0f). 14. `src/voxcpm2_tts.cpp` & `src/mel_band_roformer.cpp`: Accelerated VoxCPM2 VAE weight reconstruction across output channels via `core_parallel::for_each_chunk` (4.47x faster, median 1.41ms -> 0.32ms, max_diff = 7.45e-9), snake1d per-channel chunk parallelization (3.12x faster, median 1.90ms -> 0.61ms, max_diff = 0.0f), grouped & transposed causal conv1d chunking (gated via `CRISPASR_VOXCPM2_SERIAL=1`). Accelerated Mel-Band RoFormer with precalculated RoPE inv_freq caching (3.30x faster, median 1.61ms -> 0.49ms, max_diff = 0.0f) and `core_parallel::for_each_chunk` across time & frequency transformer bands (gated via `CRISPASR_MEL_BAND_SERIAL=1`). 15. `src/core/spectral_diarize.cpp` & `src/kokoro.cpp`: Speaker diarization pairwise cosine similarity and L2 normalization parallelized via `core_parallel::for_each_chunk` (1.71x faster, median 20.42ms -> 11.95ms, max_diff = 0.0f), gated via `CRISPASR_DIARIZE_SERIAL=1`; Kokoro TTS harmonic source generator Direct DFT loop parallelized via `core_parallel::for_each_chunk` across $T\_har$ frames (3.03x faster, median 3.55ms -> 1.17ms, max_diff = 0.0f) and nearest-neighbor f0 upsampling chunk parallelized (max_diff = 0.0f), gated via `CRISPASR_KOKORO_SERIAL=1`. 16. `src/core/cqt.h`, `src/core/cqt2010v2.h`, and `src/core/diarize_smooth.cpp`: Constant-Q Transform magnitude evaluation accelerated via branch-free tap interval bounding and `core_parallel::for_each_chunk` frame parallelization (4.35x faster, median 25.04ms -> 5.75ms, 100% exact bit-identical max_diff = 0.0f), gated via `CRISPASR_CQT_SERIAL=1`; Basic Pitch CQT2010v2 octave correlation parallelized across filter kernels (2.99x faster, median 1.38ms -> 0.46ms, max_diff = 0.0f), gated via `CRISPASR_CQT2010_SERIAL=1`; speaker diarization smoothing centroid estimation accelerated via single-pass inverse row-norm precomputation (2.03x faster, median 0.42ms -> 0.21ms, max_diff = 0.0f), gated via `CRISPASR_DIARIZE_SMOOTH_SERIAL=1`. 17. `src/core/rvq.cpp` & `src/core/istft.h`: Residual vector quantization Euclidean argmin search chunk-parallelized across time frames via `core_parallel::for_each_chunk` (2.46x-3.60x faster, median 10.58ms -> 2.94ms, 100% exact bit-identical code assignments with 0 mismatches), gated via `CRISPASR_RVQ_SERIAL=1`; Inverse Short-Time Fourier Transform (iSTFT) non-BLAS Hermitian IRFFT and BLAS spec_re/im loops chunk-parallelized across frames (3.48x faster, median 1314.67ms -> 378.20ms, max_diff = 1.16e-10), gated via `CRISPASR_ISTFT_SERIAL=1`. Tests: 13,624+ unit test assertions passed (`test-core-istft` [405 assertions], `test-core-rvq` [29 assertions], `test-rvq-istft-ab-bench` [3 assertions], `test-core-cqt` [11,826 assertions], `test-diarize-smooth` [30 assertions], `test-cqt-diarize-ab-bench` [11 assertions], `test-spectral-diarize` [268 assertions], `test-kokoro-diarize-ab-bench` [5 assertions], `test-align-ab-bench` [343 assertions], `test-conformer-ab-bench` [3 assertions], `test-cpu-ops-ab-bench` [5 assertions], `test-voxcpm2-ab-bench` [5 assertions], `test-firered-ab-bench` [4 assertions], `test-core-mel-features` [92 assertions]). Format 18 clean, wiring checks PASS, test_no_secrets PASS. |
| 1 | COMPLETE: Haas-ja-v1.0 Japanese ASR backend port & KDA predictor integration (#561) | Clean-room KDA predictor in `src/parakeet.cpp`. FastConformer encoder + KDA predictor + TDT joint. GGUF converter `models/convert-haas-to-gguf.py`, reference dumper `tools/dump_haas_reference.py`. Parity: `crispasr-diff` encoder 1.000000, decoder_sos 1.000000, joint_sos_t0 1.000000. Accuracy: 0 CER on `samples/ja_test.wav`. Speed: 11.3x realtime on M1 GPU. Quantized Q8_0 (~626 MB) and Q4_K (~368 MB). GGUFs published to `cstr/Haas-ja-v1.0-GGUF`. Registered in `crispasr_model_registry.cpp` (`haas`, `haas-ja`). Tests: `test-arch-backend-map` (609/609 passed), `test-registry` (8/8 passed). Wiring: `check-backend-wiring.py` PASS, format 18 OK, `test_no_secrets.py` OK. Issue #561 closed. |
| 1 | COMPLETE: MeanVC2 streaming zero-shot voice conversion backend port & performance acceleration (6.6x realtime, 4x faster than audio.cpp) | Clean-room C++ implementation in `src/meanvc2.{h,cpp}`. Unified GGUF converter in `models/convert-meanvc2-to-gguf.py`. Parity: `crispasr-diff` 5/5 PASS (`asr_bn` 1.000000, `gtm_k` 1.000000, `gtm_v` 1.000000, `dit_mel` 1.000000, `vocos_pcm` 0.999998). ASR intelligibility: 0.0% WER on Whisper transcription of converted audio. Speed: 242.9 ms for 1.6s audio (RTF 0.1518, 6.59x realtime vs audio.cpp RTF 0.633). Performance optimizations: dual-matrix GEMM ISTFT (114x faster, 4.19 ms), FP32 weight caching, AdaLN precomputation, RoPE inv_freq caching, zero-heap streaming buffers, and Accelerate BLAS ISTFT extended across `fireredtts3`, `lfm2_audio`, `miotts`, `miocodec`. Quantization: F16 (175 MB RAM) & Q4_K (77 MB RAM). Tests: Catch2 unit tests (`tests/test-meanvc2-params.cpp`) and convert harness (`tests/test-meanvc2-convert.cpp`). CLI: `crispasr_backend_meanvc2.cpp`. Wiring: `check-backend-wiring.py` PASS, format 18 OK, `test_no_secrets.py` 5/5 OK. |
| 2 | COMPLETE: Audio8-TTS (ArkTTS 0.6B) backend port, Metal/GPU acceleration & HF release | Full Dual-AR + Snake vocoder architecture, 1.000000 cosine parity on `crispasr-diff` (7/7 PASS), GPU Metal prefix execution for fast AR + dual-transformer in-place KV caching (~55–80 ms/frame on M1 Metal, RTF ~0.67x; CPU ~190–215 ms/frame), chunked causal vocoder (<2.5 GB peak RAM for 3+ min audio), Q4_K/Q8_0/F16 GGUFs published to `cstr/Audio8-TTS-Realistic-0.6b-GGUF`, 0.0% WER on ASR roundtrips. |
| 2 | COMPLETE: v0.8.49 release cut & tagged | `VERSION` bumped to 0.8.49 via `scripts/bump-version.sh`, synced across Python, Rust, Dart, JS. `RELEASE_NOTES_v0.8.49.md` and `HISTORY.md` committed; tag `v0.8.49` pushed. |
| 3 | COMPLETE: Integrate #559, then #560 | `#559` merged at `777e8ff23`; `#560` merged at `641e9bfab` (CI **40900035576**, Lint **40900041428**, Heavy CPU **40900046409**). Both shipped in v0.8.49. |
| 4 | COMPLETE: Kaggle Whisper loader A/B and same-toolchain v4 | Loader-cache A/B v1 passed (`576.68 -> 318.37 ms`, [receipt](docs/historical/whisper-loader-device-ab-t4-diagnostic-2026-10-29.json)). Same-toolchain v4 completed ([receipt](docs/historical/whisper-toolchain-paired-t4-diagnostic-2026-10-29.json)): `crisp` beats `upstream` (`1.142 s` vs `1.292 s`) but is `+12.3%` vs whisper-only `fork` (`1.017 s`, decode `3.23` vs `2.52 ms/call`); `d28102214` stays on `validate/whisper-integration-20261029`. |
| 5 | Follow-up: Whisper per-step decode gap vs whisper-only fork | Per-call threadpool overhead was ruled out (`21–30 us` total across 27 calls); remaining gap vs `fork` is ~`0.70 ms/call` in decode (`3.226` vs `2.524 ms/call`). Isolate per-op CUDA graph/scheduler difference before merging `d28102214`. |
| 6 | COMPLETE: Original and native step / paired measurements | Granite original steps ([receipt](docs/historical/granite5-original-steps-t4-2026-10-29.json)), CSM original steps ([receipt](docs/historical/csm-original-steps-t4-2026-10-29.json)), CSM native steps ([receipt](docs/historical/csm-native-steps-t4-2026-10-29.json)) and Melo paired v1–v3 ([receipt](docs/historical/melo-original-paired-t4-diagnostic-2026-10-29.json)) are archived. |
| 7 | Qualify remaining upstream safety candidates | Short DTW (`d28102214`) passed CI **40903036393**, Lint **40903042314** and full inference **40903516297** ([receipt](docs/historical/whisper-dtw-short-2026-10-29.json)). VAD offset and vocab-bounds remain on validation branches pending rights-cleared model checks. |
| Holds | #557, #558, #532, #483, #461, #456, MUSA | See the handoff for explicit approval, evidence, device and rights limits. Do not close #557 or publish private paths. |


- OpenVoice stage diagnosis owner: current agent, starting from the accepted paired adapter `8a5bb390b`. The warm converter comparison shows a large CPU-flow versus CUDA-original gap but has no native phase breakdown. Prepare a hosted-only run on the same qualified inputs and application, retaining unchanged waveform gates and reporting stage timers without private audio or embeddings. Do not infer a cause or change neural math until measured.

- Recent-model timing: [Nemotron T4 paired receipt](docs/historical/nemotron-original-t4-benchmark-2026-10-29.json) records one public clip, three measured calls per engine, GPU execution and zero WER. Native F16 and original F32 differ in precision; process medians include model load and do not establish a steady-state ranking. A separate native frame-count diagnostic is prepared. Granite's first paired run failed before inference on a truncated hash; the corrected run produced native warmup timings but failed before original inference because its snapshot omitted the chat template. The repaired third run completed with zero WER and GPU execution; see [the T4 receipt](docs/historical/granite5-original-t4-benchmark-2026-10-29.json) for precision and timing-boundary limits. The [CSM T4 receipt](docs/historical/csm-original-t4-benchmark-2026-10-29.json) records the private original-without-watermark comparison with different precision and load-inclusive timing. The [OpenVoice T4 receipt](docs/historical/openvoice-original-t4-benchmark-2026-10-29.json) records two accepted warm conversions: native CPU flows/CUDA vocoder are much slower than the original CUDA converter. Full original Melo synthesis and comparable step/fork measurements remain outstanding.

- #560 full-recording LID owner: delegated implementation agent, starting source `4023a6080`. Missing behavior: standalone `--detect-language` and automatic CLI LID only inspect the first 15 seconds even when the recording is longer. Add an explicit opt-in that scores windows throughout the recording and reports one duration-aware result; preserve the existing 15-second default. Acceptance: model-free window/aggregation controls, actual CLI caller coverage, and hosted model execution when available. Use only light local tests; no local model build/inference, commit, push, or public issue response. Next: implement and hand off the uncommitted change for review.

- Whisper timing diagnosis owner: root agent, starting source `4023a6080`. Missing evidence: paired process timing includes load and differs by build/CUDA libraries, so the stage responsible for the gap is unknown. Add opt-in stage instrumentation and hosted paired profiling with matched workload and explicit device/quality gates. Keep heavy builds and inference hosted; preserve current benchmark receipts.

- Whisper T4 timing diagnostic COMPLETE: [receipt](docs/historical/whisper-timing-t4-diagnostic-2026-10-29.json). Same-output GPU controls passed; median single-token decode is 2.52 ms for source-built CrispASR versus 2.41 ms for the packaged fork, while model loading is 531 ms versus 293 ms. Builds differ; the cause and target performance remain unproven. A same-toolchain comparison including upstream is running on Kaggle.

- Whisper safety repair owner: current agent, source `802cc6a04`. Port upstream MIT change `8631825d41a2712268813981a9550b04a3f225e5` for very short reflective padding; require before/after bounds control and unchanged normal-length reflection. Full build remains hosted.

- Recent-model benchmark owner: current agent, source `c2bd6701a`. User priority is Granite Speech5 Q4_K, Nemotron F16, CSM and recently repaired Melo/OpenVoice, against their pinned original implementations and relevant ggml forks where available. Use matched inputs, model/quant and decode settings where possible, qualify outputs, measure individual stages/steps and warm steady-state separately from loading, and retain source/device/precision differences. All model builds/inference go to suitable hosted workers; check GPU slot availability before Kaggle submission. Preserve earlier accepted parity campaigns; benchmark measurements need new timed adapters rather than repeating parity jobs. The Whisper pilot was outside this requested model scope.

- Whisper fork comparison COMPLETE, bounded process-level result: Kaggle `crispasr-whisper-fork-paired` version 2 passed all ten GPU-PID and JFK WER gates. Four alternating measured calls each: CrispASR v0.8.47 median 1.468 s, OpenWhispr fork v1.0.0 median 1.017 s. Both use identical pinned model/audio bytes; startup and GPU monitoring are included, builds and CUDA libraries differ. This is not per-step timing or broad model acceptance. Full terminal source/logs retained; [receipt](docs/historical/whisper-fork-t4-benchmark-2026-10-29.json). Version 1 failed before inference due to the wrong packaged executable name and is retained. Next: source-matched step timing and original-implementation comparisons need their own hosted adapters and quality controls.

- Paired benchmark collector owner: current agent, source `df4786900`. Add hosted-only alternating process timing with resource checks, input hashing, warmups and retained failure logs; validate with mocked commands. Step-level adapters and real matched GPU/original comparisons remain separate.

- Benchmark comparison owner: current agent, source `5bcb0bbb8`. Add a model-free comparison of retained paired timing records that rejects mismatched inputs, model identities and settings; report per-step timing only when counts exist. Actual GPU measurements remain hosted and require separately qualified workloads.

- H model publication COMPLETE: the qualified converter, original MIT notice, model card and provenance were published at immutable revision `61741391da62c278148a2c75b6c0c5611d447149`. All four files pass anonymous full-byte SHA-256 readback; no validation audio, target embedding or reference capture was published. [Receipt](docs/historical/openvoice2-tcc-publication-2026-10-29.json).

- I owner: current agent. Source/tag `cc114d2b4a5a9413738f390093e9131576743cb6` / v0.8.48 is on main and the remote tag; exact-target CI **40830218588** (13 jobs), Lint **40830221020** (10 jobs), Secret **40830201135**, Rust **40830201115** and Dart **40830201120** pass. Follow native Release **40833098926** and language wrappers **40833098923** without duplicate dispatches. After native completion, verify all 52 assets, run the native auditor, collect the automatic wheel publisher and Pages, then run the wheel auditor with the accepted native audit ID. Publication is not accepted yet. J waits until both audits and all producers pass.

- #560 bounded routing error COMPLETE, owner current agent after Sol implementation/review. Full current CLI run **40807940665** accepts default/auto Whisper and dedicated Silero language detection without ASR model/HF resolution, multiple inputs, failure exits and ordinary automatic-language transcription. Three actual-caller/resolver controls pass after baseline failures; [accepted receipt](docs/historical/issue-560-lid-cli-accepted-2026-10-29.json). Code `bd1e299e1` has CI **40807943181** (all 13 jobs) and Lint **40807945520** accepted. The issue stays open for the separate full-recording feature after I/J; no public reply or closure.

- H owner: current agent. Full declared F16 profile is accepted on CPU **40825101583** and real T4 CUDA kernel **version 5**. Both pass original inputs/activation/duration, 2 × 27 TCC stages, actual Session/C ABI and all six CLI automatic BERT load checks, repeats/rates and all four unchanged speech/speaker cases. Complete source/log/output/checkpoints are retained; both 120-file private envelopes and checkpoint hashes are verified. [CPU receipt](docs/historical/openvoice-full-f16-cpu-accepted-2026-10-29.json), [CUDA receipt](docs/historical/openvoice-full-f16-cuda-accepted-2026-10-29.json). Failed Q4 receipts stay failed.

- H integration owner: current agent, starting source `2d388274b`. Managed Melo V2 now requests the qualified F16 BERT under the canonical discoverable filename, with both model URLs pinned to revision `35522347b1bed9a913d6894803c5cab929ae42a5`. The actual production resolver → CLI init fixture fails before this change and passes after; explicit codec selection is preserved. BERT Apache-2.0 attribution/full license is retained independently of Melo MIT. Melo V3 is outside this qualification. Candidate `500aaaf03` passes CI **40827487259** (13 jobs; 2,105 unit tests), Lint **40827489979** (10 jobs) and Secret scan **40827480637**. H is accepted. Task I has integrated main and published the immutable v0.8.48 tag at `cc114d2b4` after exact-source CI/Lint/secret and Rust/Dart tests passed. Native producer **40833098926** and wrapper producer **40833098923** are in flight; wheel production and both publication audits remain. Do not bump or tag v0.8.48 again. [Qualification receipt](docs/historical/release-v0848-qualification-2026-10-29.json).

- Owner: current maintainer agent, starting source `42d787586`; collect A/B/D terminal receipts, repair actual failures, and preserve existing runs. Large work stays on suitable hosted workers; local load 0.26 and available memory about 4.4 GiB at resumption.
- A complete: corrected auditor 40736324867 passed; all eight distributions and anonymous readbacks accepted. [Receipt](docs/historical/release-v0846-wheel-audit-2026-10-28.json).
- B: native producer 40711745769 completed successfully (38/38 jobs including publish). Native audit 40751444564 PASSED with accepted full readbacks for all 52 assets. All 14 jobs in wheel producer 40751349609 passed, including PyPI and GPU-index publication. Pages 40776528499 deployed index commit `08ffbfaf1`; public HIP/Vulkan and ARM CUDA v0.8.47 links are verified. Final wheel audit **40789883589** is accepted: all 11 distributions and all anonymous readbacks pass ([receipt](docs/historical/release-v0847-wheel-audit-2026-10-29.json)). B is complete. Do not retag.
- D: 40735433388 completed on EPYC 9V74 with all three actual CPU profiles. Repeated raw arrays independently confirm native/AVX512 Q4 drops `A`, AVX2 Q4 retains it, and every F16 control matches the unchanged original reference. Added a one-word Q4 allowance with mandatory zero-WER F16 transcript and unchanged stage gates in both drivers. Targeted production run 40750908085 at `2c733b9c3` PASSED on EPYC 9V74: Q4 WER 1/23, F16 WER zero, CTC-logit cosine 0.999138. [Accepted receipt](docs/historical/data2vec-production-regression-2026-10-28.json). Do not duplicate. No further capability probes.
- Additional error repaired: normal nightly now fails missing explicitly required stages, matching Kaggle; negative caller controls cover partial/absent captures.
- Resources: light local inspection only; large builds/readbacks/inference on suitable Actions workers.
- Same-family repair owner: current agent; Java/Ruby integer speaker setters also swallow native out-of-range `-2`. Shared native range/public-method fixtures failed before the strict zero-success fixes and pass after: Java 95/9 current/older assertions, Ruby 343 assertions. No model changes.
- Error repair owner: current agent, source `140c02f53`; C# `SetPuncModel`, `SetVoice` and `SetSpeakerId` swallow native `-2` failures. Repaired with public-method negative controls: failed before, 101/35 assertions pass after, using a single compiler process with available memory about 4.9 GiB and load 2.54. Strict zero-success for these explicit operations preserves optional tuning-knob compatibility.
- OpenVoice numerical repair owner: current agent. The three original-recipe fixes pass **27/27 on CPU and T4 CUDA** on identical frozen model/reference bytes; raw waveform cosine 0.99999973, norm ratio 1.000042. The full three-case conditioning/shared-noise/cache and ASR/speaker gates also pass in version 4: [accepted receipt](docs/historical/openvoice-conditioning-accepted-2026-10-28.json). The Session composition also passes (G below). Next qualify a full rebuilt-current library, actual CLI and authorized model publication. No default promotion.
- Current-runtime validation owner: current agent, starting source `24178da9d`. Separate frozen replay is accepted against the v0.8.47 SDK and matching `e2ae679a` ggml headers: resolved dependencies, four overlay symbol owners, 27/27 CPU and 27/27 T4 CUDA stages, raw waveform norm ratio 1.000042. [Receipt](docs/historical/openvoice-current-runtime-replay-2026-10-28.json). Full conditioning version 3 passed native checks but hit a PyAV decoder error; repaired version 4 passes quality. No local model inference or SDK rebuild. Both Kaggle jobs are terminal; no replacement is required.
- C: user requests keeping #557 open and preparing a reply for review; do not post or close. E remains hardware-dependent.
- OpenVoice Session integration COMPLETE: kernel version 2 passes architecture detection on a renamed model, codec/voice setters, native output rates, raw/marked synthesis, exact reseeded repeats, actual CUDA and unchanged content/speaker gates. Preset identity remains unknown; raw validation audio stays private. [Accepted receipt](docs/historical/openvoice-session-accepted-2026-10-28.json). This is the shipped-SDK/model-overlay composition; actual CLI and full rebuilt-current-library qualification remain separate.
- Newly opened #558: reproduced and repaired standalone-marker grouping, with model-free regression controls. Reporter audio/NeMo/CUDA acceptance remains pending; see [repair scope](docs/historical/parakeet-word-boundary-558-2026-10-28.md).

- v0.8.46 is **published**. Native and wheel audits pass (task A complete).
- v0.8.47 (#557 Linux GPU wheels) has an **audited public release**, tagged at `5f87a5626`. Release run
  40711745769 completed successfully with exactly 52 public asset names. Native byte audit 40751444564 is accepted; Python wheel producer 40751349609 passed all 14 jobs. Pages 40776528499 and live index links pass; final wheel audit 40789883589 is accepted (task B complete).
- [Publication receipt](docs/historical/release-publication-2026-10-28.md).

## Completed baseline

- v0.8.45 and v0.8.46 are published. v0.8.46 carries the qualified Nemotron F16
  default; its tag/assets and v0.8.45's must not be rewritten.
- #557 code is merged: Linux x86_64 `+vulkan`/`+hip` and aarch64 SBSA `+cuda`
  wheels and producers, and a `whl/hip` index.
- Earlier completed work is unchanged: Qwen3-TTS #555 voice packs, Granite5
  Q4_K, CSM RoPE, OmniVoice processing, the MMQ archive and the binding fixes.
  See [HISTORY.md](HISTORY.md).

## Open tasks — in order (superseded 2026-11-03: the live list is "NOW — open lanes" at the top; rows below are kept for evidence)

| # | Task | Depends on | Done when |
|---|---|---|---|
| A | COMPLETE: v0.8.46 wheel auditor 40736324867 passed. | — | Accepted receipt committed; do not repeat. |
| B | COMPLETE: v0.8.47 publication verified: release jobs, 52 native assets, three new GPU wheel legs, `whl/hip` + aarch64 `whl/cuda` index entries, native and wheel audits. | v0.8.47 release run finished | Both verifier runs pass; IDs recorded. For transient failures rerun failed jobs in the original producer. Whole redispatch risks different CPU bytes colliding with immutable PyPI; see handoff. Never re-tag. |
| C | COMPLETE: [final #557 reply prepared for review](docs/historical/issue-557-reply-review-2026-10-29.md); keep the issue open. | B | Both audits support the draft claims; reviewed wording is ready. No posting or closure authorized. |
| D | COMPLETE: data2vec-base Q4_K nightly diagnosis and paired Q4/F16 production gate. | — | Cause retained in same-host arrays; model-free guards and actual production regression 40750908085 pass. |
| F | COMPLETE: bounded OpenVoice2 conditioned-cloning CUDA proof; separate current-runtime structural replay accepted. | — | All three 27/28-stage cases, waveform/cache/clip-mean, WER <=0.2 and speaker improvement >0.01 pass. Publication and high-level integration remain separate. See handoff follow-up scope. |
| G | COMPLETE: real OpenVoice Python Session/C ABI validation over the qualified SDK/model-overlay composition; kernel version 2 accepted. | F | Architecture detection, voice/codec/rates/repeatability, raw and marked synthesis, actual CUDA and unchanged ASR/speaker gates pass. Preset identity stays unknown; full rebuilt-current library and actual CLI remain separate. |
| H | COMPLETE: full current library/CLI, original BERT conditioning repair and managed V2 default qualified. | F/G and hosted current candidate | Original input boundaries, TCC stages, Session/CLI, unchanged speech quality, CPU and CI/Lint gates all accepted on the final source. |
| K | COMPLETE: #560 documented detect-and-exit CLI routing. Full-recording aggregation stays a later feature. | Before I | Three caller/resolver controls and actual hosted CLI 40807940665 accept bypass, providers, failures and ordinary transcription. CI/Lint pass at `bd1e299e1`. |
| I | COMPLETE: v0.8.48 published and verified (native audit **40856997446**, wheel audit **40895402248**). | B, H and K | Accepted receipts committed ([qualification](docs/historical/release-v0848-qualification-2026-10-29.json), [wheel audit](docs/historical/release-v0848-wheel-audit-2026-10-29.json)). Never rewrite v0.8.47 or v0.8.48. |
| J | COMPLETE: PR #559 and full-recording #560 validated and merged to `main` (`777e8ff23`, `641e9bfab`). | I; errors before features | Exact-tree CI/Lint/secret and hosted model checks accepted ([receipt](docs/historical/issue-560-exact-tree-integration-2026-10-29.json)). |
| L | COMPLETE (2026-11-02): Lint Deep green again; four cppcheck 2.7 findings fixed (ArkTTS virtual call in destructor, Mini-BS-RoFormer lambda, parakeet null check, sampler comparator). | — | Lint Deep passes on `main`. |
| M | COMPLETE (2026-11-02): #564 MOSS-Transcribe-Diarize Session/CLI forward beam size and optional temperature/top-k/top-p/seed sampling (`src/core/logit_sampler.h`). Greedy stays default; beam wins over sampling. | — | Hosted CPU acceptance **41417217786** passed (`moss-diarize-sampling.yml`, now dispatch-only). Ships in the next release. |
| N | COMPLETE (2026-11-02): #567 parakeet-unified GGUFs carried parakeet-rnnt-0.6b's tokenizer. Converter takes the tokenizer from the unified `.nemo` tar and asserts the last piece is `¥`. Both HF files had only `tokenizer.ggml.tokens` rewritten; JFK exact on q4_k and f16; card corrected (base model, RNNT, NVIDIA Open Model License). | — | Published files verified by re-download. |
| O | Replies only: #562 (Mini-BS-RoFormer usage posted; close once a release includes it) and draft PR #566 SCNet (review posted: graph-cache miss per chunk, short-input abort, server >2ch 500, process-wide GPU pref, no server length cap, leftover push trigger). | Next release / PR owner | #562 closed after release; #566 owner addresses review. |
| E | Real-hardware qualification of `+vulkan` (AMD/Intel), `+hip` (ROCm) and aarch64 `+cuda`. | Hardware access | Clean-venv install, Session JFK transcript matches, GPU backend selected and used, CPU/GPU compared. Recorded per class. |

## External holds — do not schedule speculative repairs

| Front | Missing input / condition | Resume only with |
|---|---|---|
| #532 Canary | Failing German recording, preceding assertion, startup device/build and selected model filenames. | An exact reproducible case; existing CPU/T4 probes did not reproduce it. |
| #483 MMQ | GTX1660/MX150 device access. | Matched OFF/ON correctness and cold-PTX versus warm timing on each real target. Preserve the verified archive/draft. |
| #461 B390 | Intel Arc B390 access. | Current ten-step timing, complete logs/WAVs and listening/original-reference acceptance. Eight-step quality is poorer; retain ten. |
| MUSA | Authorized S5000 host and applicable terms/permissions. | Actual runtime packaging/dependency and speech checks; full redistribution review before image publication, and permission for benchmark disclosure where required. No ordinary hosted SDK execution absent an applicable exception. |
| #456 Nyra | Appropriate model/output rights and distribution review. | Resolved production/hosted, derivative/recipient, output/training, WavLM-provenance and dependency obligations. No port/conversion behind a plain NC switch. |

The user explicitly confirmed these inputs/devices/new permissions are unavailable.
Refresh complete issue threads before resuming if new evidence arrives. Seven
issues are open at the 2026-10-29 check: #456, #461, #483, #532, #557
(implemented and publication accepted; reply held for review) and #558/#560. New PR #559 is reviewed
in [a scoped checkpoint](docs/historical/qwen3-context-pr559-review-2026-10-28.md); CI and
independent caller/runtime checks remain. Keep this feature after active error
repairs. None was closed merely to finish this batch.

## Parked work and historical detail

The handoff contains a bounded research/API inventory. It is not an active
assignment or a release blocker. Audit code and current evidence before
activating anything from [PLAN-ARCHIVE.md](PLAN-ARCHIVE.md); its old NOW/OPEN and
queued labels are historical. `HISTORY.md` records completed work, not a new queue.
All actual machine paths, credentials/account details, storage and job recovery
instructions remain in gitignored local Markdown.
