# Fresh-agent maintainer handoff — 2026-10-27

This is the current handoff. No previous chat is needed to understand the work,
what is complete, or what to do next. Read `AGENTS.md` and the full maintainer
development guide it names when present; this document does not replace that
mandatory method. Read gitignored local `CLAUDE.md` for private worktree/storage,
account, credential and recovery bindings. Then claim one bounded task in
[PLAN.md](../PLAN.md) before implementation.

The prior [October 19 handoff](maintenance-handoff-2026-10-19.md) and
[historical plan](../PLAN-ARCHIVE.md) are archived snapshots. Their old release,
PR, job-queue and ownership claims are not current instructions. Do not resume
them automatically. Public path aliases describe roles; actual bindings stay
in ignored local notes.

## Current state and first actions

Checkpoint before this documentation wrap-up: main `630a1fc1f`. Worktree recovery
and the protected original checkout are identified privately. Do not reset,
clean or build in the maintainer's unrelated dirty checkout. No job or collector
from the completed six-item batch requires resumption.
Older unadopted-job references may exist in local notes. Inspect actual service
state before allocating a new job; do not assume the account is idle.
No implementation claim is transferred to a fresh agent.

1. Read this handoff and the short current plan. Check working-tree status and
   remote main; preserve unrelated work.
2. Refresh open issues, PRs and applicable checks. Read each relevant whole issue,
   all comments, linked comments and latest follow-ups before changing behavior.
   The October 27 check found five open issues (#456/#461/#483/#532/#557), zero
   open PRs, no changes to the 26 existing comments, and zero comments on #557.
3. Choose the proposed correctness-release scope first, or a separately assigned
   #557 packaging task. Do not start either during a documentation-only wrap-up.
4. For implementation, record owner, source, exact missing behavior, acceptance
   gates, resource/hardware needs and next action in PLAN.md. Keep completed
   evidence and rejected controls; do not recreate an already passing campaign.

## Completed work — preserve these decisions

| Area | Implemented and accepted | Limits / evidence |
|---|---|---|
| v0.8.45 release | Native, Python, Rust and Dart publication completed. Native assets, CUDA splits, package notices, registry distributions and live GPU indexes were read back and verified. | Immutable tag source `e58bb48b5`. Native audit 40304161701; wheel audit 40313434985. No release after v0.8.45 was published by this batch. [Publication receipt](release-publication-2026-10-26.json). |
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

- Native source `2506cdd4b4f93994e280040d8f766d6bbbc48b53`: CI **40403357613**
  passes all 13 jobs; lint **40403361382** passes all 10; privacy **40403364971**
  passes. Unit job **121854716196** selects 2,104 CTest entries with zero failures
  and four optional dictionary/G2P fixture skips. All four Nemotron default/C ABI
  controls pass. These are not claims that every selected test executed.
- Deep lint **40401945512** passes at `5c98e2196`, whose native bytes match the
  qualified source above. Documentation checkpoint `630a1fc1f` also retains
  those native bytes; privacy **40423913023** and Pages **40423912257** pass.
  Its redundant native CI/lint/Ruby runs were deliberately canceled, not failed
  or counted as passing. Do not repeat a full compiled matrix for unchanged
  Markdown or file modes.
- Archive worker `45cd1f9494b1906d416ce36a34fecaccaa9aac95`: preservation
  **40414804633** and privacy **40414720651** pass. Eight guard tests pass;
  original/member/runtime hashes and both complete draft-asset readbacks pass.
  Startup load is allowed to settle for at most five minutes without changing
  the original resource limits. The first actual resource failure is retained.
- Hosted Docker smoke **40402386515** passes both jobs and all eight MUSA policy
  controls. It does not use the MUSA SDK or qualify the MUSA runtime image.
- CPU/T4 original-model proof, licenses, inputs and verifier recipes are already
  retained with immutable pins. The [aggregate receipt](ordered-agenda-2026-10-26.json)
  and linked model receipts identify them. Do not redownload large model-bearing
  Actions artifacts to recover small audit files.
- The two temporary validation refs were removed only after proof was pushed.
  Preserve `archive/issue483-mmq-source-20261008` and the matching maintainer
  draft. No required collector or acceptance job remains awaiting completion.

Public aliases such as `${CRISPASR_STORAGE}`, `${CRISPASR_WORKTREES}` and
`${CRISPASR_SCRATCH}` denote storage roles, not literal machine layouts. Private
locations and exact recovery commands are in ignored notes, not in this ledger.

## Next task 1 — correctness-release preparation

This is proposed next work, not a release launched by the documentation task.
The source-only Nemotron fix has not reached the published v0.8.45 registry.
Prepare a new candidate containing accepted fixes; do not retag or overwrite
v0.8.45 to make it appear fixed. Use `scripts/bump-version.sh` for version/tag
propagation, preserve source/model/license pins and freeze the selected source.

Inspect `.github/workflows/release.yml`, `release-python-wheels.yml` and
`release-wrappers.yml`, the notice/provenance helpers, and the existing publication
auditor. Validate the actual resulting CLI/C ABI, relocated native bundles,
wheel ABI/metadata/notices, full/split CUDA payloads and runtime dependencies.
Inspect terminal producer results before publication claims, and verify public
asset/PyPI/index/Rust/Dart readbacks after any authorized publication. Existing
speech evidence can be reused only where native/inference identity is established;
new native behavior requires its own original-reference and actual-device proof.
Keep MUSA and Nyra unqualified scope out of publication acceptance.

## Next tasks 2–4 — #557 Linux GPU wheels

The full thread was read: zero comments/follow-ups at the latest check. See
[the code and actual-asset audit](issue-557-linux-gpu-wheels-review-2026-10-26.md).
No implementation or new wheel publication has started.

1. **Linux x86_64 Vulkan first.** Add a relocatable shared-library native producer
   in `.github/workflows/release.yml`, with source/tag provenance, all notices,
   headers/plugins and loader/dependency closure. The current Vulkan CLI producer
   explicitly uses `BUILD_SHARED_LIBS=OFF`; its archive cannot supply a missing
   shared library. Add the corresponding manylinux wheel matrix leg in
   `.github/workflows/release-python-wheels.yml` using the accepted bundle.
2. **Linux x86_64 HIP next.** A HIP lib bundle already exists. Add the wheel leg,
   explicit `+hip` metadata and `whl/hip` PEP 503 index. Extend
   `tools/ci-heavy/release_publication_audit.py`: native-payload mappings, exact
   output sets, local-version logic and live index checks currently cover only
   three GPU variants and CUDA/Vulkan indexes. Preserve strict sets; do not hide
   the new output from verification.
3. **ARM CUDA afterward.** CPU aarch64 wheels exist, but there is no matching ARM
   CUDA native producer/bundle or GPU wheel. Define the ARM toolchain, CUDA
   architecture/runtime target and actual-device acceptance separately. An x86
   CUDA wheel or T4 execution does not qualify aarch64.

For each wheel, preserve native bytes/provenance, exact dependency notices and
manylinux symbol-floor checks. `python/tools/stage_libs.py` treats absent GPU
runtime/driver families as host-provided. State that contract accurately and
review any additional vendored payload. Test a clean installed environment
through C ABI/Python session, backend/device selection and adjacent plugin
loading. Actual GPU smoke/parity is needed before claiming acceleration; an
import, static CLI build or CPU/software-Vulkan fallback is insufficient.
Do not launch model/SDK builds locally or rebuild unrelated accepted models.

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
| OpenVoice2 | Recover current terminal status/artifact identity privately, then adopt one actual pretrained embedding/conditioning/quality case under [the baking method](openvoice2-baking.md). | Old queued/session-cap notes are stale. Synthetic encoder proof is not pretrained quality. Preserve original hooks, 27/28 stage gates, noise/clip-mean controls, WER <=0.2 and speaker improvement >0.01; do not relaunch an old worker blindly. |
| Echo9B quantization | One identical-owned-input projection intervention, beginning with down-projection layer 25; [accepted attribution](index-echo-owned-ffn-audit-2026-10-19.json). | Seven recipes remain rejected. Component improvement is not complete EN/ZH trajectory, speech, cache, GPU or default acceptance. No duplicate attribution campaign. |
| MiMo attention | Capture the actual failing 71-row Q/K/V, mask, scores/probabilities and pre-output projection with OFF/ON/OFF identity; [operator audit](mimo-attention-operator-2026-10-17.md). | Tiny F16-accumulator findings do not repair the F32 71-row path. Old cache-read/BLAS hypotheses were rejected. Preserve existing error gates and PR492's closed/unmerged status. |
| Dia | Require a new operator hypothesis on identical original/native inputs; otherwise leave experimental. [Factorial results](dia-cuda-factorial-results-2026-10-16.json). | Original, precision/recipe, draw-placement and BOS/source sweeps did not recover all speech cases. Keep failed controls, seeds and thresholds; no token overrides or repeated full sweeps. |
| OmniVoice | New GPU semantic-boundary, speaker or human-listening evidence only. | Corrected CPU processing campaign includes all eligible references. Do not outcome-filter references or promote GPU encode modes that change codes merely because ASR words match. |
| Breeze / Dolphin | Only a new reproducible quality/grid hypothesis or the actual failing host/clip. | Accepted API/server/voice fixes and original-reference repairs are not work to redo. Historical grid or platform qualifications are not blanket acceptance. |
| Other older models/API/performance ideas | Audit the named requirement in PLAN-ARCHIVE.md against code and HISTORY.md before claiming a task. | No old OPEN label starts a task. Drama3/custom research-output restrictions remain deferred; a new trained speech reader needs separate data/rights/evaluation scope. |

## Resource, license and completion rules

- Check load and available memory before compilation/inference. Large CPU/model
  work belongs on suitable Actions workers; actual supported CUDA tests belong
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
