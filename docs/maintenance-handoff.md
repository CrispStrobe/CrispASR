# Fresh-agent maintainer handoff — 2026-10-29

This is the current handoff. No previous chat is needed to understand the work,
what is complete, or what to do next. Read `AGENTS.md` and the full maintainer
development guide it names when present; this document does not replace that
mandatory method. Read gitignored local `CLAUDE.md` for private worktree/storage,
account, credential and recovery bindings. Then claim one bounded task in
[PLAN.md](../PLAN.md) before implementation.

The 2026-10-27 version of this file is in Git history. The prior [October 19 handoff](maintenance-handoff-2026-10-19.md) and
[historical plan](../PLAN-ARCHIVE.md) are archived snapshots. Their old release,
PR, job-queue and ownership claims are not current instructions. Do not resume
them automatically. Public path aliases describe roles; actual bindings stay
in ignored local notes.

## Current state and first actions

### Active continuation — 2026-10-29

The work is not complete until #559 and the full-recording part of #560 are
accepted, all four recent-model comparisons have usable Kaggle results, and
Whisper performance matches the comparison implementation on Kaggle. Do not
close #557; its reply is for review only. Large builds and inference belong on
hosted workers. Check load and available memory before each build or inference,
and check actual GPU session availability before every Kaggle submission.

- **Release:** native producer **40833098926** passed all 38 jobs and the
  release lists 52 assets; Rust/Dart publisher **40833098923** also passed.
  Native audit **40856997446** passed all 52 full anonymous readbacks and
  notice/producer checks. Automatic Python wheel producer **40856452802** is
  queued. Finish wheel publication, its audit and all wheel/index readbacks as
  task I specifies; do not retag v0.8.48.
- **#559:** exact PR head `6b132720090cbb532f79f544c130381ddf1cf6c9` was
  applied to a validation tree, candidate `5eef35019`. CI **40848995923** and
  lint **40848999141** are queued. Hosted short-clip and actual C API prompt
  routing run **40860832787** is queued on test tree `8a12a0ad7`, including exact C ABI, CLI, chunked routing and actual CLI factory exclusions. Earlier run **40855248796** passed; its retained C ABI and CLI traces were rechecked with the stricter parser, including four chunk prompts. Its receipt overstates unrelated coverage, so use only the bounded [Qwen evidence](qwen3-pr559-cpu-diagnostic-2026-10-29.json). Neither a
  queued run nor the contributor's own probes establish independent acceptance.
  Review the current discussion and complete CLI, C API and chunking controls
  before deciding whether to merge.
- **#560:** the standalone detect-and-exit repair is already on main. Opt-in
  full-recording detection candidate `f81fba67a` is on a validation branch;
  Hosted long-clip run **40850814403** failed: after Silero rejected a low-evidence Korean
  window, fallback tried to load the explicit Silero GGUF as a Whisper model.
  Whisper long-clip controls and Silero late-speech detection passed. Fix the
  fallback without lowering the Silero gate. Repair `de2e532dd` passes
  model-free actual-caller controls; CI **40858515815**, lint **40858519622**
  are queued; hosted long-clip rerun **40858523187** has started. Model acceptance and
  main integration remain outstanding. Kaggle CPU workers lack internet, and
  the existing approved datasets do not contain the exact source, models and
  dependencies for the full #559/#560 tests. Keep the hosted Actions jobs;
  a reduced model-free check is not an acceptance substitute.
- **Recent models:** [the Nemotron paired receipt](nemotron-original-t4-benchmark-2026-10-29.json)
  records T4 results for native F16 and original F32, with zero WER and GPU
  execution; precision and operation boundaries differ, and no per-step count
  was captured. A native step-count diagnostic is prepared. Granite Q4_K
  version 1 failed on a truncated hash; version 2 ran native warmups but its
  original lacked the chat template. Version 3 completed with GPU and zero-WER
  controls; see the [T4 receipt](granite5-original-t4-benchmark-2026-10-29.json)
  for the native 26-step decoder and non-equivalent original timing boundaries.
  The Granite original forward-step diagnostic and Nemotron F16 comparison
  against MIT-licensed transcribe.cpp are prepared on validation branch
  `validate/whisper-granite-diagnostics-20261029` at `ca0fb1e05`; neither has
  been submitted. Their READMEs specify GPU, model provenance, context and
  quality gates. Coordinate available Kaggle slots before submission.
  Native CSM opt-in graph-step profiling is on validation branch
  `validate/csm-step-profile-20261029` at `399758238`, with CI **40862657041**
  and lint **40862660572** queued. A reviewer identified partial-frame failures;
  the revised profiler marks their counters incomplete. This is unvalidated instrumentation, not a
  performance result; require hosted build and profile-on/off output identity.
  Native and original CSM step adapters are prepared, not submitted, at
  `validate/csm-native-steps-20261029` commit `51fae70cb`. Each README sets
  separate boundaries, GPU/identity/ASR gates and license limits. The native
  adapter builds the pinned profiler and rejects incomplete frame traces; the
  original adapter checks actual frame and decoder forwards. The native parser
  and original fake-module checks passed locally without inference or builds. Private CSM original-without-watermark version 1 failed on missing
  `bitsandbytes`; version 2 passed GPU and zero-WER controls. The [aggregate
  receipt](csm-original-t4-benchmark-2026-10-29.json) discloses precision,
  watermark and timing boundaries. Gated access and no redistribution apply.
  [OpenVoice paired T4 timing](openvoice-original-t4-benchmark-2026-10-29.json)
  completed on two cases with output parity; native CPU flows/CUDA vocoder
  were substantially slower than the original CUDA converter. Full original
  Melo synthesis remains unproven.
- **Whisper:** the short-audio safety fix `4023a6080` has queued CI
  **40847257799**, lint **40847261708** and secret scan **40847251740**.
  The [T4 timing diagnostic](whisper-timing-t4-diagnostic-2026-10-29.json) of
  `c5513ccfb` passed output and GPU controls. Median decode was 2.52 ms/call
  versus 2.41 ms for the packaged fork, and model load 531 ms versus 293 ms;
  builds differ. Same-toolchain comparison `crispasr-whisper-toolchain-paired`
  version 1 failed before fork compilation: its CMake configuration could not
  find `CUDA::cuda_driver`. Retained output and source document the failure.
  Version 2 is submitted with CUDA VMM disabled for all three builds and checks
  each compile command; its uploaded script SHA-256 is
  `ea0d3c6b3ed61b00927c44c4994be463af79943d7d3629e32cb6af27eb2f883e`.
  The independent `crispasr-whisper-vmm-ab` version 1 tests VMM on and off on
  the same CrispASR source. Collect and check both terminal outputs before
  drawing conclusions or scheduling another run. These results do not yet
  establish the reason for the gap or matched performance. Keep changes on
  validation branches until their own acceptance checks pass. A static upstream
  comparison identified per-call persistent CPU-threadpool setup as a possible
  source of overhead, not a proven cause. Optional timing source `e34ca22e4`
  is on `validate/whisper-threadpool-profile-20261029`, with CI **40864034522**
  and lint **40864037912** requested. The prepared, unsubmitted adapter is on
  `validate/whisper-threadpool-adapter-20261029` at `c6ea7a3a6`. Only run it if
  current comparisons leave that hypothesis relevant; keep CPU ownership and
  synchronization behavior unchanged until correctness and speed are proved.

The validation branch `validate/ordered-model-benchmarks-20261029` contains the
prepared adapters and bounded Nemotron receipt. Later local changes and private
evidence locations are recorded in ignored `CLAUDE.md`; do not assume every
local commit has been pushed. Check actual job status and preserve terminal
evidence before any retry. Never publish private audio, credentials, or local
machine paths.

Snapshot 2026-10-29: **v0.8.46 publication is accepted. v0.8.47 native
publication, both wheel publishers, Pages and live GPU index links pass; final
wheel audit 40789883589 is accepted for all 11 distributions and readbacks.** Keep #557 open and prepare its reply
for review only. The current CLI, BERT, final activation and duration repairs are integrated
on main and tagged **v0.8.48** at `cc114d2b4`. The immutable v0.8.47 release
is unchanged. v0.8.48 publication builds/readbacks remain in flight. The full declared F16 profile is **accepted**:
CPU **40825101583** at `395daf6ae`, artifact **12876647336**; real T4 CUDA
`crispasr-openvoice-full-source` **version 5**, application/controller
`97053abff`, bootstrap `395daf6ae`. Both complete source/log/output/checkpoint
readbacks and 120-file encrypted archives are verified. All four unchanged
speech/speaker cases pass on both devices, with actual adjacent F16 discovery
through Session/C ABI and all six CLI calls. The candidate now changes managed
Melo V2 downloads to the qualified F16 companion and immutable revision;
production resolver → CLI initialization fails before and passes after that
routing repair. Candidate `500aaaf03` passes CI **40827487259** (13 jobs; 2,105 unit tests),
Lint **40827489979** (10 jobs) and Secret scan **40827480637**. H is accepted.
The exact release target `cc114d2b4` also passes CI **40830218588** (13 jobs),
Lint **40830221020** (10 jobs), Secret **40830201135**, Rust **40830201115**
and Dart **40830201120**. Main and the new annotated v0.8.48 tag were pushed
atomically and both remote commit readbacks match. Native Release
**40833098926** and language wrappers **40833098923** have started; no
publication acceptance is inferred from the tag. Continue task I, then J.
[Release qualification receipt](release-v0848-qualification-2026-10-29.json).
[CPU acceptance](openvoice-full-f16-cpu-accepted-2026-10-29.json),
[CUDA acceptance](openvoice-full-f16-cuda-accepted-2026-10-29.json).

The completed BERT comparison isolates Q4 error (relative L2 0.136–0.165 versus
original features) against F16 (about 0.0003). A fixed speech comparison on the
same application passes the unchanged disclosed WER gate with F16 (0.125 /
0.055556); Q4 still fails (0.25 / 0.333333). These are diagnostic receipts,
not complete application acceptance. All terminal source/log/output/checkpoint
and authenticated private archives are retained; do not repeat accepted
component campaigns.

#560's bounded CLI error is qualified: hosted **40807940665** passes on source
`bd1e299e1`, plus CI **40807943181** (all 13 jobs) and Lint **40807945520**.
Later candidate changes are validation tooling/receipts; C++ is unchanged.
The issue remains open for its separate full-recording LID request, after I/J.
#557 also remains open with its prepared reply held for review.

| Item | State | Evidence |
|---|---|---|
| v0.8.46 (Nemotron F16 default, packaging provenance) | Published, latest at the time, 50 native assets. Native audit PASS. Wheels, PyPI and Rust/Dart PASS. Formal wheel audit 40736324867 PASS; task A complete. | [Receipt](release-publication-2026-10-28.md) |
| v0.8.47 (#557 Linux GPU wheels) | Tag `v0.8.47` → `5f87a56265e4bbd7e2adc7052254e576d5774abb`. Release run **40711745769** passes all 38 jobs including publish, with 52 expected public asset names. Native audit 40751444564 is accepted; all 14 jobs in wheel producer 40751349609 pass. Pages 40776528499 passes and live index links are verified; wheel audit 40789883589 is accepted; B complete. | Task B |
| #557 issue | Code merged. Issue still **open**. No public comment posted yet. | Task C |
| data2vec-base nightly gate | COMPLETE: same-host cause, paired Q4/exact-F16 policy and actual production regression 40750908085 accepted. | Task D |
| OpenVoice2 | Bounded conditioning/SDK Session proofs accepted. Full rebuilt CPU library/CLI at `7c158b325` is accepted. Latest BERT conditioning candidate needs its own current CPU/CUDA and CI gates; no model/default promotion. | Task F and follow-up below |

First actions for a fresh agent:

1. Read this handoff and [PLAN.md](../PLAN.md), inspect status and local recovery notes, then fetch remote refs. Update a clean main checkout without rewriting candidate or unrelated work.
   Do not reset, clean or build in the maintainer's protected original checkout
   (identified in the ignored local notes).
2. Refresh open issues and PRs. Read each relevant whole thread (body, all
   comments, linked comments, latest follow-ups) before changing behavior. The
   2026-10-29 check: open #456/#461/#483/#532/#557/#558/#560, PR #559,
   zero comments on #557. The [PR #559 review checkpoint](qwen3-context-pr559-review-2026-10-28.md)
   scopes its remaining gates; no review or merge was submitted.
3. A, B, D, bounded F/G and K are accepted; C is prepared for review with #557 open. H is accepted; continue I → J in order; do not duplicate accepted campaigns. E needs hardware.
4. For any task, record owner, source SHA, exact missing behavior, acceptance gates,
   resources and next action in PLAN.md before starting. Do not recreate a passing campaign.

Post-tag main also contains the C# explicit-setter, Java/Ruby speaker-ID,
Parakeet standalone-boundary and OpenVoice recipe repairs. These changes are
**not in the immutable v0.8.47 tag**. A later patch release requires its own
source/CI/validation checks; do not retag v0.8.47 to include them.

## Completed work — preserve these decisions

| Area | Implemented and accepted | Limits / evidence |
|---|---|---|
| v0.8.46 release | Promotes the qualified Nemotron F16 default with OpenMDW notices, the VoxCPM2 host-permute timing split, and wheel notice/provenance staging. Published 2026-10-28; native and wheel publication pass. The verifier now takes `tag`/`source` inputs, and `REVIEWED_TAGS` lists v0.8.45–v0.8.47. | Wheel audit 40736324867 accepted (task A complete). v0.8.45 tag/assets untouched. [Receipt](release-publication-2026-10-28.md). |
| #557 Linux GPU wheels (code) | Producers `build-libs-linux-x86_64-vulkan` and `build-libs-linux-arm64-cuda`. Wheel legs `linux-x86_64-vulkan`, `linux-x86_64-hip` (`--host-provided libomp.so`) and `linux-arm64-cuda`. A `whl/hip` index. A Linux Vulkan clean-venv loader smoke. Version-gated (≥ 0.8.47) exact auditor expectations. Tagged for v0.8.47; native and wheel/index readback audits are accepted. | Packaging only; no physical-GPU run (task E). Jetson excluded. Reply remains review-only (task C). |
| v0.8.45 release | Native, Python, Rust and Dart publication completed. Native assets, CUDA splits, package notices, registry distributions and live GPU indexes were read back and verified. | Immutable tag source `e58bb48b5`. Native audit 40304161701; wheel audit 40313434985. This earlier receipt predates the separately scoped v0.8.46/v0.8.47 work. [Publication receipt](release-publication-2026-10-26.json). |
| Nemotron default | Managed downloads now select the pinned qualified F16 profile only, retain OpenMDW/origin notices, and reject stale-Q4/missing-Q4 substitution. CPU and actual T4 pass 26/26 original transcript matches, 22/22 VoxCPM prompts, unchanged stage/magnitude gates, CLI, opaque-filename C ABI and fresh/reused-session controls. | Former plain Q4 fails both devices (11/26 and 8/22), so it is not an accepted default. Explicit existing Q4 files remain diagnostic. Existing v0.8.45 binaries still require explicit F16 selection. [Policy and full proof](nemotron-default-policy-2026-10-26.md). |
| Qwen3-TTS #555 | Incompatible Base packs are rejected before prefill; failed replacement preserves the previous pack. Designed clear defaults use the matching original 1024/2048-dimensional encoders for 0.6B/1.7B. CPU/T4 speech/reference controls pass. | Model-specific embeddings/codes are not interchangeable voice packs. Apache/MIT notices, recipes and original declarations are retained; speaker identity/likeness rights and human listening are not inferred from ASR. [Runtime proof](issue555-voice-pack-2026-10-23.json), [designed voices](qwen3-designed-voices.md). |
| Granite5 #552 | Qualified mixed Q4_K is the approximately 4.07 GB default. Bare `crispasr-quantize input-f16.gguf output.gguf q4_k` reproduces the accepted model byte-for-byte. CPU/T4 original-reference and C ABI/default checks pass; full model/recipe notices are retained. | Native context remains 4096; no full-16k or speed claim. Q8/Q6 and earlier failed Q4 recipes remain rejected despite matching words. [Q4_K proof](granite5-q4k-2026-10-24.json). |
| CSM | Original Llama-3 rotary scaling and independent depth theta restored without changing sampling/weights/seed/gates. Nineteen original F16 stages and 442 quantized payloads verified; CPU and actual T4 speech controls pass. | Do not claim full depth-decoder/Mimi stage parity. Preserve the diagnostic legacy switch and original failures. Shipped in v0.8.45. [Qualification](csm-cpu-profile-2026-10-25.json). |
| Bindings and wheels | #549/#551/#556 and contributor PR553/554 fixes integrated, authorship preserved, accepted API/package checks passed. Reporter #557 confirms Windows CUDA sessions now work for their Japanese subtitle pipeline. | CLI filename detection alone is not C ABI proof. Actual shared-header/device/loader surfaces need their own tests. [Earlier closure receipt](release-errors-2026-10-25.json). |
| OmniVoice processing | Corrected CPU campaign includes every integrity-eligible reference; accepted upstream audio/text utilities are merged. Exact encoder cache/Apple ARM CPU pointwise path is accepted within its scope. | Keep hybrid/graph/fused GPU encode off: changed codes are not waived by WER 0. Broader GPU, cloning/speaker and human-listening scope is separate. [Encoder decisions](omnivoice-encoder-2026-10-23.md), [processing method](omnivoice-audio-processing.md). |
| B390 diagnostics | VAE upload timing now separates host permutation. Existing weight/context caches and batched prefill were audited. | No graph/math/default change or new B390 speed claim. Ten inference steps remain the quality default. [Capture procedure](voxcpm2-b390-review-2026-10-26.md). |
| MMQ preservation | Original OFF/ON ZIPs preserved byte-exact with full NVIDIA CUDA 12.6.3/cuBLAS and original notices; both uploaded assets pass full hash readback. Independent sidecar audit reads only 203,920 bytes and rejects three invalid receipts. | Maintainer draft 424870665 remains unpublished; no GPU execution/rebuild or target performance acceptance. [Preservation and reproducible proof](cuda126-mmq-preservation-2026-10-26.md). |
| MUSA compiler candidate | SDK 5.2/architecture 31 compiles unchanged tagged WKV/full CLI; actual Docker recipe selects that profile. Manual workflows enforce physical platform/resource checks; hosted policy controls pass without SDK use. | S5000 runtime/image/speech, redistribution and older architectures are not qualified. Historical hosted compilation is not permission for further non-MUSA SDK use. [Build/terms](musa-build.md). |
| Nyra review | Complete pinned software/model/output terms reviewed. | MIT inference code does not license model/lexicon/configuration assets or outputs. Integration remains deferred; no conversion, registry entry, backend or output dataset was added. [Full review](nyra-license-review-2026-10-26.md). |

Qualified Nemotron F16: model repository `cstr/nemotron-3.5-asr-streaming-GGUF`,
revision `bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2`, file
`nemotron-3.5-asr-streaming-0.6b-f16.gguf`, 1,286,368,480 bytes, SHA-256
`14cbc98df495d8ab452c6a4ac367136d94ab5da55e73afaa96ba4f749dd95d09`.
Use this exact qualified file with published binaries; do not assume an older
managed download or a similarly named Q4 file is equivalent. Do not bake another
voice pack or requantize Granite merely to reproduce these completed results.

## Acceptance and job ledger

- **v0.8.46**, tag source `34a8e436192de19d3e5a872437edfe3fd26ce8c4`:
  - CI **40470378993**, Lint **40477169406** and Secret scan **40470378983** pass.
    The earlier bump commit `1e7d3b735` failed only Lint: the Flutter changelog
    entry was missing. It was fixed before tagging.
  - Release **40649389464** passes all 36 jobs.
  - Native audit **40682766228** PASS: 50 assets fully read back, with notices
    and the CLI version smoke.
  - Wheels **40679618075** and wrappers **40649389411** pass.
  - Wheel audit **40736324867** PASS; task A complete.
- **v0.8.47**, tag source `5f87a56265e4bbd7e2adc7052254e576d5774abb`:
  - CI **40689967257**, Lint **40689967286** and Secret scan **40689967265** pass.
  - Release **40711745769** remains in progress: task B.
  - Wrappers **40711745994** pass all three jobs. Full small registry archive
    readbacks match checksums, version, original manifests, package licenses and
    tagged Rust/Dart sources: [receipt](release-wrappers-readback-2026-10-28.json).
- **#557 dry runs** of the new library producers pass:
  - Vulkan **40460158618**: build, relocated-load verify, dependency gate.
  - ARM64 CUDA **40661942703**: same gates.
  - A wheel staged from the Vulkan artifact passed notice (20 files),
    provenance and manylinux_2_35 floor checks, and loaded with `CDLL` locally.
- Earlier evidence for v0.8.45 and the Nemotron, Qwen3, Granite, CSM, MMQ and
  MUSA work (CI 40403357613, lint 40403361382, preservation 40414804633 and
  related runs) is unchanged. See the Git history of this file and
  [ordered-agenda-2026-10-26.json](ordered-agenda-2026-10-26.json).

Public aliases such as `${CRISPASR_STORAGE}`, `${CRISPASR_WORKTREES}` and
`${CRISPASR_SCRATCH}` denote storage roles, not literal machine layouts. Private
locations and exact recovery commands are in ignored notes, not in this ledger.

## Open tasks — fully scoped

A is complete: **40736324867** passed; its accepted full-distribution readback
receipt is [retained here](release-v0846-wheel-audit-2026-10-28.json).
B native producer **40711745769** has all **38 jobs successful**, including
final publish, and the exact 52 expected public asset names. Native readback
audit **40751444564** passed on `ubuntu-latest` with `complete: true, acceptance: true`, auditor source
`bda9d00ce6209a7811dccc5a996077b8bdc2521d`, pinned tag source
`5f87a56265e4bbd7e2adc7052254e576d5774abb`, release_run `40711745769`,
`repair_notices=false`. Automatic Python wheel producer **40751349609** passes all 14 jobs, including
the 12 setup/build jobs, PyPI and GPU-index publication (`workflow_run`, workflow
source `2c733b9c3`; release inputs remain the frozen tag). Index commit
`08ffbfaf1f9c123740d3807ffa4b07c0af3f29e0` is on main. Pages deployment
**40776528499** succeeded; anonymous HIP/Vulkan and ARM CUDA v0.8.47
index links are verified. Final wheel audit **40789883589** was dispatched
on main with the exact frozen source and accepted producer/native-audit inputs.
It passed with complete/acceptance true, all 11 distributions and five PyPI
readbacks, plus every GPU index/full-file readback. [Accepted wheel receipt](release-v0847-wheel-audit-2026-10-29.json). B is complete; do not repeat either accepted auditor.
The [accepted native receipt](release-v0847-native-audit-2026-10-28.json)
and [full inventory](release-v0847-native-assets-2026-10-28.json) are retained.
Wheel readback still waits for successful
wheel/index/Pages publication and the accepted native inventory.
D production confirmation **40750908085** passed; its
[accepted receipt](data2vec-production-regression-2026-10-28.json) closes this
bounded nightly-gate repair. Diagnostic **40735433388** completed on EPYC 9V74 with all three actual CPU profiles.
Its independently decoded retained arrays establish the quantized profile
sensitivity. No further capability attempts are needed. See task D for the fix
and remaining hosted confirmation. No local background collector remains.
Earlier canceled capability/worker attempts supply no acceptance result.
Before every push run the five privacy guards (see the ignored local notes for
the account binding). Avoid bursts of pushes to main: each push cancels the
pending Pages deploy and in-progress CI/Lint on the earlier commit. That is
what delayed the live wheel index for v0.8.46.

### Task A — complete: v0.8.46 wheel audit accepted

- **Complete:** auditor **40736324867**, artifact **12840629916**, has
  `complete: true, acceptance: true`; all eight distributions and five PyPI
  anonymous readbacks pass, along with GPU index/file readbacks. Auditor source
  `eafae0f48819eaf2a0c75d5d250383335ae70b23`. Do not rerun an accepted task.
  [Portable receipt](release-v0846-wheel-audit-2026-10-28.json).
- The following dispatch details explain the completed rerun, not another job.
- **Why:** wheel audit run **40703639437** failed with HTTP 404 on
  `https://crispstrobe.github.io/CrispASR/whl/hip/crispasr/`. The cause was the
  auditor, not the release: it required a `hip` index for every version, and
  `whl/hip` first exists with 0.8.47. PyPI readbacks and the `cuda`/`vulkan`
  index checks run before that loop. `gpu_index_flavors(version)` in
  `tools/ci-heavy/release_publication_audit.py` now version-gates it, with a
  unit test (`test_linux_gpu_wheel_set_is_version_gated_and_flavors_exact`).
- **Do:** dispatch `.github/workflows/verify-release-publication.yml` on main with
  `tag=v0.8.46`, `source=34a8e436192de19d3e5a872437edfe3fd26ce8c4`,
  `wheel_run=40679618075`, `native_audit_run=40682766228`,
  `runner=ubuntu-latest` (the workflow default remains Ubuntu 22.04;
  pinned Ubuntu 24.04 is also supported).
- **Accept:** the run concludes `success`, and its `release-publication-<run>`
  artifact `receipt.json` has `complete: true, acceptance: true`. Add the run ID
  to [the receipt](release-publication-2026-10-28.md).
- **If it fails:** read the traceback. Do not weaken an assertion to pass it.
  A real defect in a published distribution needs a new patch release, never a
  re-upload under the same version.

### Task B — complete: v0.8.47 publication verified

- **Live:** native producer and native audit **40751444564** are accepted.
  Wheel producer **40751349609** is also successful (all 14 jobs). Pages
  **40776528499** at index commit `08ffbfaf1` and live index links pass.
  Final wheel audit **40789883589** passed. Steps 1–5 below are historical
  completed instructions; do not repeat the producer, Pages or accepted audits. Use `native_audit_run=40751444564`
  for the eventual wheel verifier; do not repeat the accepted native audit.

- **Do, in order:**
  1. Wait for release run **40711745769** (tag `v0.8.47`). Every build job must
     conclude `success`. In particular check the new
     `build-libs-linux-x86_64-vulkan` and `build-libs-linux-arm64-cuda`, which
     had passed only branch dry runs (40460158618, 40661942703).
  2. Confirm the release has every v0.8.46 asset name (with 0.8.47) plus
     `libcrispasr-linux-x86_64-vulkan.tar.gz` and `libcrispasr-linux-arm64-cuda.tar.gz`.
     The native set is exactly the reviewed 50-name baseline plus these two
     bundles (52); the auditor now rejects missing and unexpected names. Its
     19 unit controls pass, including missing-bundle and extra-asset cases.
     The current public v0.8.45/v0.8.46 name sets pass this stricter check.
  3. Wait for the `Release Python wheels` run it triggers (event `workflow_run`)
     and for `Release wrappers` (Rust/Dart). The wheel run must include and pass
     the GPU legs `linux-x86_64-vulkan`, `linux-x86_64-hip` and
     `linux-arm64-cuda`, plus the Linux Vulkan clean-venv smoke step.
     `publish-gpu-index` needs every GPU leg to pass.
  4. Once the index commit `ci: publish GPU wheel index for v0.8.47 [skip ci]`
     is on main, wait for Pages to deploy it. Check that
     `https://crispstrobe.github.io/CrispASR/whl/hip/crispasr/` lists the `+hip`
     wheel and that `whl/cuda/` lists the `manylinux_2_35_aarch64` `+cuda` wheel.
  5. Dispatch the verifier with `tag=v0.8.47`,
     `source=5f87a56265e4bbd7e2adc7052254e576d5774abb`, `runner=ubuntu-latest`
     and `wheel_run=40751349609`, `native_audit_run=40751444564`.
     Dispatch `verify-release-publication.yml` on main; the native audit is
     already accepted and must not be dispatched again.
     `v0.8.47` is already in `REVIEWED_TAGS`. For versions ≥ 0.8.47 the auditor
     requires the three new GPU wheels and the `hip` index.
- **If a new GPU wheel leg fails:** inspect the failing step first. For a
  transient failure, rerun only failed jobs of the same producer run, retaining
  the original successful CPU/sdist artifacts. Do not move/re-push the tag.
  If workflow code needs repair after CPU distributions reached PyPI, do not
  blindly dispatch the full wheel matrix for the same version: fresh archives
  can have different hashes, while `twine --skip-existing` retains the old public
  bytes. The strict auditor then correctly rejects that mismatch. Prepare explicit
  byte-preserving reuse of the original qualified CPU artifacts, or a new patch
  release if published contents are defective. Such cross-run reuse is not yet
  implemented; never weaken byte comparison or overwrite an immutable version.
  Known risks to check first:
  - **HIP:** `stage_libs.py --host-provided libomp.so` must leave nothing
    unresolved (the leg runs on a runner without ROCm).
  - **ARM CUDA:** the `ubuntu-22.04-arm` runner must build the wheel against
    `manylinux_2_35_aarch64`.
- **Accept:** both verifier runs succeed. Record run IDs, asset count and wheel
  filenames in [the receipt](release-publication-2026-10-28.md).

### Task C — complete: #557 reply prepared for review; keep the issue open

The [final reply for review](issue-557-reply-review-2026-10-29.md) is supported
by accepted native audit **40751444564**, wheel audit **40789883589** and live
Pages/index checks. The whole issue was refreshed: open, no comments, no new
reporter requirements. It still explicitly states that physical AMD/Intel/ARM
GPU execution is unverified.

**User decision:** keep #557 open. No comment, closure or public review has
been sent. Preparing the draft does not authorize posting it. Preserve it for
the maintainer's review without asking again or sending it automatically.

### Task D — CPU-profile cause established; confirm the paired regression policy

- Original failed nightly **40481871937** used EPYC 9V74 and native CPU flags.
  Its Q4 transcript omitted `A` from the frozen original 23-word target;
  F16 stage cosine passed at 0.999138. The preceding green workflow skipped
  data2vec and is not a passing control.
- First controlled run **40723799365** on EPYC 7763 passes native/AVX2 but has
  no AVX-512. [First controls](data2vec-avx2-controls-2026-10-28.json).
- Decisive run **40735433388**, source
  `a21687e2918cbb1de9850e94b3350aac7f473afb`, artifact **12841953440**,
  completed on EPYC 9V74. Native, explicit AVX2 and explicit AVX512 use identical
  frozen model/reference/audio inputs and sequential source-matched builds.
  All 18 retained 549x32 raw arrays were independently decoded and checked for
  exact within-profile repeat equality. Native/AVX512 Q4 omit `A`; AVX2 Q4
  retains it. At frame 47, Q4 A-minus-delimiter is -0.1045904 versus +0.0651703.
  F16 stays positive (~0.616), retains the independent target in every profile,
  and passes unchanged stage minima 0.999138/0.999141/0.999138. Explicit-English
  pipeline and exact nightly invocations agree. This rules out punctuation/LID
  as the cause in the controlled reproduction.
  [Portable retained-byte summary](data2vec-avx512-controls-2026-10-28.json).
- Implemented in `tests/regression/run_one.py` and
  `tools/kaggle/crispasr-regression.py`: data2vec Q4 allows **one word error over
  23 original reference words**, with `require_exact_diff_transcript: true`
  requiring the existing pinned F16 `diff_gguf` to pass a separate **zero-WER**
  transcript control. Original expected text, independent CTC reference hashes,
  model pins and F16 stage threshold 0.999 are unchanged. Both drivers verify
  original-reference provenance; missing control configuration fails before
  downloads. Separate `transcript-f16` diagnostics preserve both outputs.
- Also repaired: the normal driver counted missing required stages as skips.
  Missing explicitly named stages now fail, matching Kaggle. A model-free
  production caller control failed before the repair and passes afterward.
- Model-free controls cover one/two Q4 word errors, strict F16 failures, stage
  degradation/absence, incomplete configuration, separate diagnostic retention,
  and replay of all three profiles through the actual Kaggle validation function.
  These checks do not claim a new hosted inference run.
- **COMPLETE:** targeted production regression **40750908085** at committed
  source `2c733b9c3335b2a909d527041fb90ec360d070ff`, `regression.yml --ref main`,
  `diagnostic_backend=data2vec-base`, `granite5_only=false`, passed on EPYC 9V74.
  Q4 WER is 1/23, exact F16 WER is zero, and CTC-logit cosine minimum is
  0.999138 against the unchanged 0.999 threshold (one pass, no fail/skip).
  Artifact 12854280702 and complete raw logs are retained privately;
  [portable accepted receipt](data2vec-production-regression-2026-10-28.json).
  Do not duplicate it. Local validation passed 60 driver, seven diagnostics and
  19 publication-audit tests, plus all five privacy guards. The ad hoc Ruff
  check reports the same seven pre-existing findings as the parent; no new
  findings. Acceptance required actual execution, Q4 and exact F16
  transcript passes, stage pass, and retained host/build/transcript evidence.
  A pending/skipped overall green workflow is not acceptance. Do not repeat
  capability probes, rebake targets, change kernels or use human words as the
  upstream parity oracle. If it fails, investigate retained outputs first.

### Task E — real-hardware qualification of the new GPU wheels

- **Status:** packaging only. No new wheel has run on a physical AMD (Vulkan or
  ROCm), Intel Arc (Vulkan) or ARM NVIDIA GPU. A Kaggle T4 run qualifies none of them.
- **Resume only with:** a reporter or maintainer machine of that class.
- **Accept, per class:**
  - Clean venv: `pip install` from the index.
  - `crispasr.Session` transcription of the regression JFK sample with the
    reference-matched transcript.
  - Log evidence that the GPU backend was selected and the device was used.
  - CPU and GPU transcripts compared.
- **Record:** in HISTORY.md and the receipt.

### Task F — OpenVoice2 conditioned-cloning proof (independent)

- **COMPLETE for the bounded three-case raw C API scope:** conditioning kernel
  version 4, worker `2026-10-28.openvoice-conditioning.6`, source
  `c5287611ef8416704a216d1a91c1a0fea5127a66`, passes 27/27, 27/27 and 28/28
  original-reference stages on T4 CUDA. Exact repeats, shared posterior noise,
  uninstrumented conversion and both clip-mean controls pass. Clone WER is
  0.1111, 0 and 0.1111; speaker improvement is 0.1833, 0.2373 and 0.1494.
  All retained artifact bytes and WER arithmetic were independently checked.
  [Accepted receipt](openvoice-conditioning-accepted-2026-10-28.json).
  Validation was read-only; publication remains false. Do not repeat this
  passing matrix merely to recover the same evidence.
  The full quality matrix used the frozen `0a8620765` runtime; the current
  `e2ae679a` runtime separately passes the exact 27-stage CPU/CUDA replay.
  These scopes must remain distinct. Neither qualifies high-level Session/CLI,
  additional languages/natural references, sensitive-tensor quantization,
  broader devices or a production default. Follow-up scope is below.
- **Accepted structural result:** frozen replay kernel version2 passes **27/27
  on both CPU and T4 CUDA**, unchanged model/reference/gates. Raw waveform cosine
  is0.99999973, norm ratio1.000042. [Verified receipt](openvoice-frozen-replay-accepted-2026-10-28.json).
  Native CPP is `1b874a104`; worker is `d84007612`. Full conditioning kernel
  version 3 finished all three native cases: 27/27, 27/27 and 28/28 T4 CUDA
  stages, exact repeats, raw C API and clip-mean controls. Its independent
  quality process failed because installed PyAV rejects `metadata_errors`;
  no ASR/speaker acceptance is claimed. [Retained proof](openvoice-conditioning-structural-2026-10-28.json).
  Repair decodes the generated PCM WAVs directly into mono float32 at 16 kHz,
  preserving the same ASR model and WER/speaker thresholds. The repaired full
  worker passed as version 4 above. Version 3 source, terminal logs and all
  outputs are retained privately as a failed decoder control.
  **Current-runtime structural replay is also accepted:**
  `tools/kaggle/openvoice-current-replay`, kernel version 1 (slug
  `crispasr-openvoice-current-runtime-replay`), uses the shipped v0.8.47 SDK
  with matching `e2ae679a` ggml headers and only the verified model overlay.
  Dependencies resolve, all four model symbols belong to the overlay, and
  all 27 stages pass on both CPU and T4 CUDA. Raw waveform norm ratio is
  1.000042. [Receipt](openvoice-current-runtime-replay-2026-10-28.json).
  Keep model acceptance false; Session/CLI and speech quality are separate.
  No full SDK rebuild or local model run was needed. Do not repeat this proof.
- **Historical failed control:** frozen replay v1 passes 26/27 on CPU and actual T4 CUDA.
  The bias/flow repairs are verified on identical original/model bytes. Only
  raw decoder amplitude fails (~1.11145 norm ratio). Original final activation
  uses slope 0.01; native used 0.1. The final slope is corrected with a failing-
  before/passing-after graph-call test and an isolated owned-input check
  (norm 1.111307 → 0.999915). Repeat the exact frozen replay with the new CPP
  delta subsequently passed in version 2 above. [Both profiles](openvoice-frozen-replay-2026-10-28.json).
- **Historical failed control:** conditioning version 2 reached actual T4 CUDA work and failed 11/27 stages.
  The original reference/model bytes are retained and verified. The two
  original-recipe corrections (preserve zero-input conditioning bias; couple
  before forward flip) pass tiny production-body negative controls. Exact owned
  input CPU/CUDA replay subsequently passed; full speech acceptance is not claimed.
  [All stages](openvoice-original-parity-failure-2026-10-28.json) and
  [repair method](openvoice2-baking.md#first-pretrained-numerical-failure-and-bounded-repair).
- The frozen replay data attachment is ready: real model/reference data only,
  with MIT notices and synthetic-audio provenance. Model SHA-256
  `483fd233035bb2c7cde69519d4d460742b6a70c96c5293e878fe25e72d944255`;
  original reference SHA-256
  `259993b526944e8e36ccca436e005cbc3935a46636b42c618cfd5c3d19429c8f`.
  The accepted `tools/kaggle/openvoice-fixed-replay` version 2 used these bytes;
  do not duplicate it merely to recover the same structural proof.
  The worker retains clone/header base `6be9fda36` and audited ggml `0a8620765`,
  overlaying only model CPP `1b874a104` (SHA-256 verified). Current main's
  ggml is newer; passing this frozen-runtime repair is not whole-current-ggml
  qualification. No full SDK rebuild is needed.
  The accepted frozen replay bootstrap pins worker `d84007612167105cf24c7af80014d1f0ea1bd8a3`
  (SHA-256 `b1fed1e4518ceeb517c749be67ce3eeee9d3243153b5f50043f6edf9c4584f20`),
  using CPP delta `1b874a104c70bc77e1a64dc1d5f4e924f0a3cfeb`
  (SHA-256 `f658c40176fe1c67584cd9ab80ca9d954433cfe7973f6ba84b43a448d59b8c13`).
- Old bake v2 is a terminal CPU-backend registration abort. The prepared
  dynamic-backend/module-discovery repair was never qualified by that run.
- Conditioning kernel version 1, worker `.2` (2026-10-28), compiled both
  native C++ deltas and converted F32-sensitive weights on a two-T4 allocation,
  then failed Hub repository creation with 403 **before inference**. All five
  retained compute resource checks passed. `gpu_executed` and acceptance remain
  false. [Portable receipt](openvoice-conditioning-attempt-2026-10-28.json).
  Previous source/logs/output are retained privately; do not erase them.
- Worker `.3` defaults to read-only validation: model/reference/notice/hash
  checkpoints remain in kernel output. Only `CRISPASR_OV2_PUBLISH=1` requests
  Hub writes, with separate `publication_requested/publication_complete` fields.
  A denied requested write fails; it never becomes numerical acceptance.
- The accepted quality run retains header base `6be9fda36` and the audited
  original runtime/ggml pins, overlaying only the corrected model CPP `1b874a104`.
  The full worker and bootstrap are source/hash pinned. Its terminal source,
  logs and output are preserved; no replay or replacement is currently needed.
  Never treat allocation/building as CUDA execution.
- **Accept:** original 27/28 stage, magnitude and waveform gates; exact repeated
  uninstrumented C API output; clip-mean conditioning and shared-noise controls;
  actual CUDA vocoder work; independent baseline/original/clone ASR WER <=0.2
  and speaker-similarity improvement >0.01. Preserve rejected controls and
  upstream/model/input hashes and notices. No registry default promotion from
  structural parity alone. See [the full method](openvoice2-baking.md).

#### OpenVoice follow-up scope after task F

**G COMPLETE — real Session integration:** `tools/kaggle/openvoice-session`,
kernel version 2 (`crispasr-openvoice-session`), controller source
`264ff48a5`, bootstrap `1f5b35a85`. The renamed model auto-detects as Melo;
codec/voice setters, baseline 44.1 kHz and clone 22.05 kHz rates, exact reseeded
repeats, raw and normal marked synthesis all pass. Actual CUDA is observed.
Marked-clone WER is 0.1111 and 0; independent speaker improvements are 0.1823
and 0.2731. Raw speech passes too. Exact controller bytes, source/header/model
pins, all resource checks and WER arithmetic were independently verified.
[Accepted Session receipt](openvoice-session-accepted-2026-10-28.json).
Preset identity is explicitly `unknown`: model-generated audio does not prove
whose identity a preset represents. Raw artifacts remain private and labelled;
these tests grant no identity, consent or voice-pack publication rights.
Both Session versions' terminal source, logs and outputs are retained privately.
These accepted Session runs need no replacement. The separate full-source
application qualification below does not replace their bounded evidence.
This qualifies the shipped SDK plus verified model overlay, not a complete
rebuilt-current library or the actual CLI. No model/default promotion or Hub
publication occurred. Remaining scope:

1. **Full rebuilt-current library and CLI — retained attempts and accepted F16 profile:**
   candidate branch `validate/openvoice-interfaces-20261028`; application source
   `91e1e97b399d105924915c0e8d5ea0176359b147`, bootstrap `ae1f243a7`.
   These changes were still unmerged at that attempt; current status is above. The actual CLI adapter needed GPU
   forwarding, native-rate declaration before synthesis, neutral disclosure
   bypass/resampling, fail-closed converter inputs, F32-sensitive discovery and
   unset-speaker normalization. Native Melo rejects out-of-range IDs. Actual
   production-body routing/setter controls failed before and pass after repair.
   Kaggle `crispasr-openvoice-full-source` version 1 is terminal **ERROR**.
   All current application C++ compiled against audited v0.8.47 ggml DSOs and
   exact `e2ae679a` headers; dependencies resolve with no old application DSO or
   model overlay. Both original-reference cases pass 27/27 with converter CUDA.
   Real Session detection/native speaker rejection, raw/marked repeats and rates
   pass; the six expected converter CUDA windows are observed. All actual CLI
   commands, rate checks, neutral bypass and converter GPU controls pass.
   **Overall acceptance remains false:** default-disclosed CLI case 0 has
   tiny.en WER **0.25**, above the unchanged **0.20** gate. Its content-only
   clone passes at 0.111 with speaker gain above 0.01; both Session cases and
   CLI case 1 pass. [Failed quality receipt](openvoice-fullsource-attempt-2026-10-28.json).
   All source/logs, the exact compiled application checkpoint and encrypted
   private speech were retained and the checkpoint hash read back. Do not
   rebuild it merely to repeat this proof; do not change seeds, thresholds or
   the recognition oracle to make the failed gate green.
   **Disclosure diagnostic version 2 is complete:** controller `f2729e429`,
   bootstrap `a455b144f`, exact retained application/checkpoint, no rebuild.
   Neutral speech and extracted prefixes score tiny.en WER 2/7; small.en scores
   them 1/7. Content passes both. The prefix agrees with the neutral baseline
   after resampling (cosine 0.999937, norm ratio 0.999637); conversion/rate
   routing is not the cause. [Diagnostic receipt](openvoice-disclosure-diagnostic-2026-10-29.json).
   Original tiny.en acceptance remains failed; the additional recognizer does
   not replace it. All terminal source/logs/output and encrypted speech are
   retained, including the version-1 transport failure before inference.
   Kaggle rejected the failed full-source run as an input attachment. The
   corrected transport obtains the existing output API's download URL for the
   exact checkpoint and renders its placeholder only into a private push copy;
   never commit that capability URL or copy an account API token into a worker.
   The private-cache alternative reached the account storage limit; no assets
   were deleted or made public.
   **Claimed conditioning repair, runtime acceptance pending:** original Melo
   `209145371cff8fc3bd60d7be902ea69cbdb7965a` exposes two concrete frontend
   mismatches. Native BERT retained punctuation inside whitespace words and
   distributed already-doubled phone counts over WordPieces. The original
   isolates punctuation and distributes raw phones before inserting blanks.
   Actual-function model-free controls fail before and pass after the repair.
   Shared existing Unicode normalization/pretokenization now feeds both callers;
   counts come from the same WordPiece pass, unknown words are atomic, and
   mismatched token/phone shapes fail closed. No neural operators, weights,
   seeds, ASR oracle or quality threshold were changed.
   Next rebuild the full candidate on a hosted CUDA worker. Require exact
   captured native token/phoneme/tone/language/phone-count arrays against pinned
   original Python functions in `tools/validation/melotts_inputs.py`, then both
   original TCC 27-stage cases, Session/C ABI, actual CLI and all four original
   quality cases, including tiny.en default-disclosed WER <=0.20. This is input
   protocol proof plus decoded acceptance, not complete original Melo neural
   stage parity or broad-language acceptance. Before every compile/inference
   check load, memory and storage. Retain terminal source/logs/checkpoints and
   encrypted evidence before any replacement. Full-source CUDA **version 2**
   is terminal from source `9259b9f3e`, bootstrap `379146cb0`. All three
   original-input controls, TCC stages and functional surfaces pass, but
   default-disclosed CLI case 0 still has tiny.en WER 0.25 >0.20.
   [Retained failed attempt](openvoice-conditioning-inputs-attempt-2026-10-29.json).
   The next concrete repair follows original `Generator.forward`: its final
   `F.leaky_relu(x)` uses default 0.01; native Melo used 0.1 there. The actual
   graph-call regression fails before and passes after correcting only that
   final slope. Upsample/residual slopes stay 0.1. Require actual captured
   pre/post tensors against the pinned original operator and an uninstrumented
   waveform comparison before interpreting decoded quality. Full-source
   **version 3** is terminal from application/controller `01a288763`, bootstrap
   `315339f65`. All three exact-input cases pass; actual final activation matches
   the original operator exactly on CUDA0, with norm ratio 1.0 and zero maximum
   error. Instrumented/plain waveform errors are zero. Both TCC 27-stage cases,
   Session/CLI and rates pass, but disclosed CLI WER 0.25 / 0.333333 fails the
   unchanged original 0.20 gate. [Retained failed receipt](openvoice-final-activation-attempt-2026-10-29.json).
   The complete checkpoint and 108-file encrypted envelope were read back,
   hashed and authenticated. Do not claim application acceptance.
   **Decoder diagnostic version 1 is accepted:** all nine boundary comparisons
   across three cases match the original Generator/residual blocks with shared
   converted weights and native latent input. Waveform relative L2 is about
   4e-6 and scale is within 1e-6 of unity. [Receipt](melotts-decoder-parity-2026-10-29.json).
   **Encoder/inverse-flow diagnostic version 2 is accepted:** all fifteen
   comparisons across three cases pass with relative errors below 5e-7,
   using actual native BERT features and sampled latent as shared inputs.
   [Receipt](melotts-upstream-parity-2026-10-29.json). Both terminal runs have
   complete source/log/output readbacks and authenticated private envelopes.
   Do not repeat these accepted diagnostics or infer full-model parity.

   **Concrete duration defect and current repair:** the shipped model declares
   zero SDP flows and zero DDS layers because the converter counts deterministic
   `dp` names instead of `sdp`. Captured SDP outputs for different sentences
   have byte-identical common prefixes: learned conditioning was ignored.
   The candidate corrects converter counts and recovers legacy zero metadata
   from the actual contiguous tensor inventory, rejecting incompatible nonzero
   counts. It restores the original initial reverse Flip and the original
   1e-3 spline minima/normalization. Four production-body tests pass after
   failed before controls; ten of sixteen original scalar spline rows failed
   before and all pass after. [Diagnosis](melotts-sdp-defect-2026-10-29.json).
   New `tools/validation/melotts_duration.py` runs both pinned original
   predictors with actual captured native encoder output and scaled noise;
   RNG equivalence is not assumed. It preserves strict full parameter loading,
   all original operators and shared model bytes. No seeds, recognizer,
   speech threshold or model default changed.

   Full-source CUDA **version 4** is terminal from application/controller
   `9001ff913`, bootstrap `8b9511e79`. All three original deterministic and
   stochastic duration cases pass, SDP relative error about 1.3e-6. Input,
   activation/backend, instrumentation, TCC, Session/CLI and rates pass too;
   disclosed CLI WER 0.25 / 0.333333 still fails 0.20. [Retained CUDA attempt](openvoice-duration-attempt-2026-10-29.json).
   NN CPU **40806522459** finished at `8b9511e79`, likewise passing original
   duration and functional checks while disclosed case 0 WER 0.25 fails.
   [Retained CPU attempt](openvoice-cpu-duration-attempt-2026-10-29.json).
   Both complete artifacts/checkpoints and authenticated 120-file envelopes
   are retained. Do not rerun these unchanged failed default-profile campaigns.

   **Important causal limit:** the shipped `sdp_ratio` is zero. The SDP defect
   is genuine in the optional conditioning path, but repairing it cannot
   change these default clips. Do not claim it caused or fixed their quality.
   **BERT diagnosis is complete:** `crispasr-melotts-decoder` version 3,
   controller `59751b4e2`, bootstrap `c0afba306`, compares pinned original
   Transformers 4.57.1 and Melo phone expansion against actual native features.
   Q4 relative L2 versus original FP32 is 0.1577 / 0.1646 / 0.1364; even the
   shared-dequantized-weight comparison has relative L2 0.0586 / 0.0486 / 0.0386.
   F16 matches shared weights within about 1e-6 and original FP32 within about
   0.0003. Native BERT is CPU; original reference is CUDA. All token IDs/repeats
   match. [BERT boundary receipt](melotts-bert-boundary-2026-10-29.json).
   The version-3 speech phase failed because its diagnostic omitted the
   adjacent converter copy. Complete source/log/output and authenticated
   107-file envelope are retained; this was a harness staging error.

   **Fixed speech diagnostic version 4 is complete:** controller `837e02957`,
   bootstrap `07a12d7b7`, reuses the exact version-4 application checkpoint
   (SHA-256 `5573e9c05f11e0b339660cc82eedf1d4a0ec11475ca0b700d51671b13b13b502`).
   It stages the validated converter and runs only the missing speech phase,
   without repeating the completed BERT component campaign or rebuilding.
   Same seed 42, text, Melo/reference/TCC and tiny.en recognizer: Q4 neutral
   WER 3/7, disclosed 0.25 / 0.333333; F16 neutral 1/7, disclosed 0.125 /
   0.055556. [Precision speech receipt](melotts-bert-precision-speech-2026-10-29.json).
   Complete source/log/output and authenticated 13-file envelope are retained.
   This is diagnostic only; complete speaker/application acceptance is separate.

   **Full F16 profile accepted:** CPU **40825101583** at `395daf6ae`,
   artifact **12876647336**, and CUDA kernel **version 5**, source `97053abff`,
   bootstrap `395daf6ae`, both pass. F16 is from revision
   `35522347b1bed9a913d6894803c5cab929ae42a5`, canonical file
   `bert-base-uncased.gguf`, SHA-256
   `f7437929246c43d7300cbf6ff39b4715669a6053661117df99a456bdc5e8ecba`.
   Actual no-override discovery is observed in both C ABI opens and all six
   CLI calls on each device. Original input/final-activation/duration checks,
   both 27-stage TCC references, rates/repeats and all four unchanged
   speech/speaker cases pass. CPU disclosed WER is 0.0625 / 0.055556; CUDA is
   0.125 / 0.055556. Complete source/log/output/checkpoints and both 120-file
   private archives are authenticated and retained; never repeat accepted
   component campaigns. [CPU receipt](openvoice-full-f16-cpu-accepted-2026-10-29.json),
   [CUDA receipt](openvoice-full-f16-cuda-accepted-2026-10-29.json).

   **Current integration task:** candidate starting `2d388274b` changes only
   managed Melo V2 to that canonical F16 companion and pins both URLs to the
   qualified revision. Real production registry resolution feeds actual CLI
   initialization in a model-free fixture: failure before, pass after, with
   explicit codec precedence preserved. A bundle unit guard covers pinned
   URLs, canonical filename and separate Apache-2.0 BERT attribution. Full
   upstream license remains in THIRD_PARTY_NOTICES.txt. Melo V3 and existing
   explicit Q4 paths are unchanged; no implicit old-Q4 fallback is introduced.
   Existing hand-managed cache files remain subject to existing cache trust
   behavior; immutable default URLs are not a new content-verification claim.
   Candidate `500aaaf03` passes CI **40827487259** (all 13 jobs; 2,105
   unit tests), Lint **40827489979** (all 10 jobs) and Secret scan
   **40827480637**. H is accepted. The integrated v0.8.48 target `cc114d2b4`
   also passes exact CI **40830218588**, Lint **40830221020**, Secret
   **40830201135**, Rust **40830201115** and Dart **40830201120**. Main/tag
   readbacks match; task I now awaits all producers and publication audits. The earlier accepted CI **40807943181** (13 jobs) and Lint
   **40807945520** are at `bd1e299e1`, before this registry change; they do not
   replace final-source qualification.
   Earlier CPU **40783760561** passed its complete four-case matrix at
   `7c158b325`, including default-disclosed WER 0.1875 and zero; its full
   artifact and authenticated encrypted readback are retained.
   [Earlier CPU receipt](openvoice-cpu-pre-conditioning-2026-10-29.json).
   It predates both conditioning repairs and cannot qualify them. The queued
   pre-activation CPU **40795945687** was canceled; old CI/Lint runs were
   superseded by the current candidate. Preserve failed/older controls without
   redispatching them as proof of the new source.
   The current CPU worker checkpoints its compiled application before model
   calls; artifacts are validation checkpoints, not audited release packages.
   Public receipts contain metrics/hashes; speech/transcripts stay encrypted
   to the operator key recovered from ignored local notes. Preset identity
   stays unknown. Merge only after current original-quality, CPU and CI gates;
   prepare the next patch afterward.
2. **Publication complete:** the accepted validation run made no Hub write.
   A separate model-only publication retained the MIT notice and provenance,
   and verified all four anonymous full-byte readbacks. The earlier 403 remains
   a historical failed attempt. No reference captures or audio were published.
   See the current publication section below.
3. **Broader acceptance:** sensitive-tensor quantization, natural references,
   other languages/voices and additional device families need their own
   original-reference and decoded-output checks. None is implied by these
   bounded synthetic English cases. A default change needs an explicit scoped
   acceptance decision and cannot be inferred from stage parity alone.

### Task K — standalone LID routing error #560

The whole [issue #560](https://github.com/CrispStrobe/CrispASR/issues/560) has
been read: no comments, linked issues or follow-ups at this checkpoint.
Help/docs promise `--detect-language` exits after detection, but both old and
unified CLI routes initialize ASR first and continue transcription. The bounded
repair is committed at `bd1e299e1`; all three production caller/resolver controls
pass after baseline failures. Hosted current CLI **40807940665** is accepted:
full CPU build, default/auto/explicit Whisper, dedicated Silero, multiple input,
invalid ASR/HF bypass, unsupported-mode and bad-input failures, and ordinary
`-l auto` transcription all pass. CI **40807943181** (13 jobs) and Lint
**40807945520** pass at that exact source. Checkpoint readback is verified.
[Accepted receipt](issue-560-lid-cli-accepted-2026-10-29.json),
[scope and review](issue-560-lid-cli-review-2026-10-29.md).
K is complete; keep #560 open for its separate full-recording request. No
public reply or closure has been sent.

The requested full-recording mode is a separate later feature, after current
error/release/PR #559 gates. Shared prefix detection caps input at 15 seconds;
individual detectors have different 5/15/30-second bounds and memory costs.
Use bounded windows with a declared aggregation/confidence policy and test
late speech, mixed languages and invalid inputs; removing the cap is not
sufficient. Default detection behavior stays unchanged until separately scoped.

### OpenVoice converter model publication — complete

The qualified F32-sensitive generic converter is published at immutable
[revision `61741391da62c278148a2c75b6c0c5611d447149`](https://huggingface.co/cstr/openvoice2-tcc-GGUF/tree/61741391da62c278148a2c75b6c0c5611d447149). The GGUF, original MIT notice, model
card and pinned provenance all pass anonymous full-byte SHA-256 readback.
No validation audio, target embedding or reference capture was uploaded.
Preset identity and rights in a reference voice remain separate; qualification
is bounded to the documented synthetic English CPU/T4 cases.
[Publication receipt](openvoice2-tcc-publication-2026-10-29.json).

### Matched Whisper CUDA fork benchmark

Kaggle `crispasr-whisper-fork-paired` version 2 completed on T4. The same
pinned base.en and JFK bytes produced zero WER and observed GPU process use
on all ten calls. Four alternating measured calls per engine give medians
1.468 s (CrispASR v0.8.47) and 1.017 s (OpenWhispr fork v1.0.0).
This measures process startup and model loading with concurrent GPU monitoring,
not per-step execution; builds and CUDA libraries differ. It is one input and
not a general model ranking. [Receipt](whisper-fork-t4-benchmark-2026-10-29.json).
All terminal source/log/output retained; version 1's missing-executable failure
preceded inference and is retained separately. Do not repeat accepted runs
without a new question. Per-step and upstream-original comparison remain to be
implemented and validated on hosted hardware.

### Task I — next patch after current error gates

**Current state: steps 1–3 are complete. Do not re-bump, recreate or move the
published v0.8.48 tag.** Source is
`cc114d2b4a5a9413738f390093e9131576743cb6`; annotated tag object is
`f9eb753bbdfd66977e6d6c57f2e6f7b712166cb7`. Exact-source checks all pass and
remote main/peeled-tag readbacks match. Native Release **40833098926**,
language-wrapper Release **40833098923**, and native audit **40856997446**
passed. Automatic wheel producer **40856452802** is queued; do not duplicate
it. After it passes, dispatch the wheel audit using `tag=v0.8.48`, the full
source above, `runner=ubuntu-latest`, `repair_notices=false`,
`release_run=40833098926`, `wheel_run=40856452802`, and
`native_audit_run=40856997446`.
Require all 52 native names, all 11 distributions, all five anonymous PyPI
readbacks, complete GPU index/full-byte readbacks and Pages deployment, plus
Rust/Dart publication. Preserve original producer artifacts; rerun only failed
jobs for transient failures. J starts only after this acceptance.

1. Complete B, H and K first. Refresh main and the candidate; preserve unrelated
   work. Merge the qualified runtime changes only when current-source CPU,
   CUDA, CI, lint and secret checks support them. Documentation-only main
   updates do not mean the candidate runtime changes are merged.
2. Prepare `RELEASE_NOTES_v0.8.48.md` from the actual post-v0.8.47 changes,
   retaining source/validation limits and hardware holds. Use
   `scripts/bump-version.sh 0.8.48`; it synchronizes versions and creates the
   release commit and local annotated tag. Never hand-edit version files or
   rewrite a published tag/PyPI distribution. Check the release auditor's
   reviewed-tag allow-list before the bump, with its existing unit controls.
3. Push the exact release commit to a qualification branch first. Request CI,
   lint and secret checks there; require their `headSha` to equal the local
   annotated tag's peeled commit. Keep the release tag local until those
   checks pass. Then fast-forward main safely and push that new tag once.
4. Collect native, wheel and Rust/Dart publishers. Verify the full expected
   inventory, notices, immutable distribution bytes, indexes and Pages, using
   the existing publication verifiers. For transient producer failures rerun
   only failed jobs in the original run. Retain artifacts before replacement.
5. Record accepted receipts and release source/run IDs. Do not call this task
   complete from allocation, building or a subset of successful publishers.
   Keep physical AMD/Intel/ARM execution claims excluded without hardware.

### Task J — PR #559 after error and release acceptance

Follow [the scoped review](qwen3-context-pr559-review-2026-10-28.md). Refresh
its entire description, comments, reviews, inline comments, linked #218/#488
follow-ups and current head before acting. The earlier reviewed head was
`6b132720090cbb532f79f544c130381ddf1cf6c9`; source review was not independent
runtime acceptance. Review any fork workflow changes before authorizing CI.
Retain actual C ABI/CLI probes for context unset/zero/enabled/changed/cleared,
empty/nonempty context, forced/automatic language, explicit questions, backend
exclusions and chunk propagation. Run the existing short-clip Qwen regression
on a suitable hosted worker with context enabled/cleared, preserving #488's
recognition and language-prefill gates. Require exact-candidate CI/lint/secret
checks. Merge only with independent caller/runtime evidence; a dedicated context
setter across every binding/server/WASM surface is outside this narrow PR.
No public review/comment or merge has been sent for this PR.

### Scope notes kept from #557 review

- GPU wheels never vendor a driver or vendor runtime (`LINUX_HOST_PROVIDED` in
  `python/tools/stage_libs.py`). `--host-provided` is a per-leg opt-in, used only
  for ROCm's `libomp.so` on `+hip`. Do not add it globally.
- The aarch64 CUDA bundle targets SBSA, CUDA 12.8, architectures 80/86/89/90/120.
  Jetson/L4T needs a separate toolchain and is out of scope.

## New #558 — source repair with bounded remaining acceptance

The full report and linked #15 thread were read. A standalone SentencePiece
space is preserved in full token text but was lost as a word-grouping boundary.
The shared range helper is repaired and model-free before/after controls cover
the demonstrated defect. See [repair and remaining evidence](parakeet-word-boundary-558-2026-10-28.md).
This later source fix is outside frozen v0.8.47. Personal natural-speech input,
matched NeMo token traces and device acceptance are still missing; keep #558 open.

## External holds — exact unblock conditions

The user confirmed no additional reproducer or target access/new permissions.
Keep these holds until new evidence changes the stated condition.

| Front | Resume input | Bounded action and acceptance |
|---|---|---|
| #532 Canary | Shareable failing German clip; assertion immediately before the trace; exact startup device/build and selected ASR/alignment filenames. | Reproduce the translation/VAD/diarization/alignment pipeline, then isolate allocation/cache behavior. Existing probes and later Viterbi changes do not prove the private crash fixed. [Assessment](canary-532-reproducer-review-2026-10-26.json). |
| #483 MMQ | GTX1660/MX150 access. | Run the exact preserved OFF/ON pair on the same device/model/audio/settings, verify correctness and separate cold PTX JIT from repeated warm timing. No T4 substitution or release-default change. Preserve original/additive notices. |
| #461 B390 | Real B390 access and complete current capture. | Fixed mixed model/seed/sentence, ten steps, full stage/per-op logs plus WAV/listening. Distinguish process restarts from reused context, and VAE initialization from compute. Do not revive the withdrawn RTF projection or reduce quality to eight steps. |
| MUSA | Authorized S5000 host; applicable SDK terms/permissions for the intended work. | Validate the actual candidate image/dependency closure and original-reference speech/operations. SDK 5.2 architecture 31 compilation alone does not establish runtime or older-target support. Review all redistributed SDK components; benchmark disclosure has a separate written-permission condition. No non-MUSA SDK execution absent an applicable exception. |
| #456 Nyra | Appropriate written rights and completed intended-use/distribution review. | Resolve model/output, production/hosted/API, converted-derivative, recipient-contract and downstream-training obligations; separately establish WavLM provenance/notices and dependency terms. Do not port or convert behind a plain NC flag. |

The MMQ destination remains draft **424870665**, tag
`archive-issue483-mmq-20261008`, target
`04ded753e909da019a2cd391f651f51f5299b6f5`. Both preserved assets and all license
sidecars are verified. To audit them, use the command in
[the preservation README](../tools/ci-heavy/mmq-preservation/README.md), which
reads only bounded metadata/notice ranges. Do not recover the original gigabyte
artifacts again. When editing draft metadata, explicitly retain tag/source and
unpublished flags; a description-only update was observed to rename its tag.
No public promotion is assigned by this handoff.

## Parked API/research inventory — not active claims

These fronts are retained so they are not lost. Their legacy labels are not
proof of missing implementation. Before scheduling one, inspect current code,
latest full issue discussion, exact receipts and any privately recovered terminal
outputs; define a new hypothesis/input and acceptance boundary.

| Front | First bounded next action | Preserve / do not repeat |
|---|---|---|
| Callable wrappers/lifecycle | Audit one genuinely missing exported/wrapped operation or new model/platform lifecycle case using [the coverage ledger](wrapper-callable-coverage.md). | Existing primitive repairs, high-level Go CPU lifecycle and accepted binding/metadata fixes are done. Raw low-level handles remain caller-serialized; API presence does not imply universal concurrent/model acceptance. |
| Alignment #217 | Scope labeled natural-speech boundaries, additional long-form/model/device coverage using [the clean-room audit](alignment-cleanroom-2026-10-18.md). | Character/subword synthetic and bounded Japanese CPU evidence exists; #550 actual T4 memory/reuse is complete. Preserve default-off optimization gates and exact timestamps; do not repeat the accepted 45-minute CUDA matrix or claim Subtimer rankings. |
| CANN | Obtain physical Ascend evidence under [the CANN TODO](cann-todo.md) and [layout procedure](cann-layout-followup-2026-10-20.md). | Host/SDK controls are accepted. Physical full-model placement, fallback, ACL replay and cache-on/off require target hardware; no Kaggle replacement. Keep optional caches off until qualified. |
| OpenVoice2 | Active task F above uses the repaired read-only worker under [the baking method](openvoice2-baking.md). | Old queued/session-cap notes are stale. Synthetic encoder proof is not pretrained quality. Preserve original hooks, 27/28 stage gates, noise/clip-mean controls, WER <=0.2 and speaker improvement >0.01; do not relaunch an old worker blindly. |
| Echo9B quantization | One identical-owned-input projection intervention, beginning with down-projection layer 25; [accepted attribution](index-echo-owned-ffn-audit-2026-10-19.json). | Seven recipes remain rejected. Component improvement is not complete EN/ZH trajectory, speech, cache, GPU or default acceptance. No duplicate attribution campaign. |
| MiMo attention | Capture the actual failing 71-row Q/K/V, mask, scores/probabilities and pre-output projection with OFF/ON/OFF identity; [operator audit](mimo-attention-operator-2026-10-17.md). | Tiny F16-accumulator findings do not repair the F32 71-row path. Old cache-read/BLAS hypotheses were rejected. Preserve existing error gates and PR492's closed/unmerged status. |
| Dia | Require a new operator hypothesis on identical original/native inputs; otherwise leave experimental. [Factorial results](dia-cuda-factorial-results-2026-10-16.json). | Original, precision/recipe, draw-placement and BOS/source sweeps did not recover all speech cases. Keep failed controls, seeds and thresholds; no token overrides or repeated full sweeps. |
| OmniVoice | New GPU semantic-boundary, speaker or human-listening evidence only. | Corrected CPU processing campaign includes all eligible references. Do not outcome-filter references or promote GPU encode modes that change codes merely because ASR words match. |
| Breeze / Dolphin | Only a new reproducible quality/grid hypothesis or the actual failing host/clip. | Accepted API/server/voice fixes and original-reference repairs are not work to redo. Historical grid or platform qualifications are not blanket acceptance. |
| Other older models/API/performance ideas | Audit the named requirement in PLAN-ARCHIVE.md against code and HISTORY.md before claiming a task. | No old OPEN label starts a task. Drama3/custom research-output restrictions remain deferred; a new trained speech reader needs separate data/rights/evaluation scope. |

## Resource, license and completion rules

- Check load and available memory before compilation/inference. Large CPU/model
  work prefers suitable Actions workers while their queue moves. The maintainer's
  current usage-guide override allows Kaggle CPU workers when Actions stalls
  (CPU inputs need datasets because those kernels lack internet); do not spend
  GPU quota on CPU build farms. Actual supported CUDA tests belong
  on Kaggle under the maintainer's usage guide, the single configured account
  and the two-session cap. Do not use GPU slots as SDK/build farms, reinstall
  Torch/CUDA arbitrarily, probe other accounts or guess session IDs. Save terminal
  code/logs/output before replacing any failed job. At most one or two permitted
  Luna/Sol agents may do small work when local headroom allows; no large local job.
- Use convert → quantize → original dump → diff → parity port. Preserve rejected
  controls, source/model/input pins, magnitudes and fixed numerical/speech gates.
  Judge success using actual outputs and actual device work, not idle allocation,
  a filename shortcut, matching words alone or a build-only receipt.
- Review code, weights, codec/tokenizer, voice assets and outputs separately.
  Existing voice packs are model-specific; conversion does not create missing
  rights. Retain complete applicable agreements/notices/provenance. An acceptance
  switch cannot supply production, output, recipient or redistribution rights.
- Before every Git/Kaggle push, stage new public files and run all five privacy
  guards with the actual configured account supplied privately. Never skip its
  account guard or put credentials, private account IDs, host addresses, local
  mount/worktree paths or recovery/session details into public Markdown.
- For a completed task, save small portable receipts, check Markdown anchors and
  changed generated files as applicable, record exact source/test scope, push to
  remote main when authorized, and verify the readback. Do not publish a draft,
  close a hardware/reproducer/rights hold, rewrite a tag or advertise broader
  acceptance to make a task appear complete. Keep private recovery current.
