# CrispASR — Current work plan

Updated 2026-10-28. Start with [the maintainer handoff](docs/maintenance-handoff.md).
It contains the completed work, exact evidence, bounded next actions and blockers
needed by a fresh agent; no earlier chat is required. Follow `AGENTS.md` and its
mandatory maintainer-guide instructions before implementation.

**A, D, bounded F and G are complete. B publication remains in flight.** Claim work by adding owner,
source SHA, missing behavior, acceptance gates and next action under "NOW"
before starting. The full scope of each task (exact runs, inputs, commands and
acceptance) is in [the handoff, "Open tasks"](docs/maintenance-handoff.md#open-tasks--fully-scoped).

## NOW — release verification and nightly diagnosis (2026-10-28)

- Current-source/CLI acceptance owner: current agent, candidate `validate/openvoice-interfaces-20261028`. Full-source CUDA version 1 (source `91e1e97b399d105924915c0e8d5ea0176359b147`) is terminal: application build/dependencies, both 27-stage original replays, Session and actual CLI routing/rates/GPU controls pass. Overall quality acceptance is **false**: the first default-disclosed clip has tiny.en WER 0.25 (limit 0.20); its content-only clone passes at 0.111. Both Session cases and the other CLI case pass. All terminal source/logs/checkpoint and encrypted speech are retained; [failed quality receipt](docs/openvoice-fullsource-attempt-2026-10-28.json). Claimed next action: hosted disclosure-prefix versus joined-clip diagnosis with the exact built application, same fixed inputs and additional independently pinned recognizer; retain the original failed gate. Do not rebuild the accepted checkpoint or change NN/seed/threshold/oracle to pass. Actions CPU 40783760561 and candidate CI40787625795/Lint40787628325/Secret40787631487 remain pending. Main stays `08ffbfaf1`; Pages40776528499 and live index links now pass, final wheel audit40789883589 is queued.

- Owner: current maintainer agent, starting source `42d787586`; collect A/B/D terminal receipts, repair actual failures, and preserve existing runs. Large work stays on suitable hosted workers; local load 0.26 and available memory about 4.4 GiB at resumption.
- A complete: corrected auditor 40736324867 passed; all eight distributions and anonymous readbacks accepted. [Receipt](docs/release-v0846-wheel-audit-2026-10-28.json).
- B: native producer 40711745769 completed successfully (38/38 jobs including publish). Native audit 40751444564 PASSED with accepted full readbacks for all 52 assets. All 14 jobs in wheel producer 40751349609 passed, including PyPI and GPU-index publication. Pages 40776528499 deployed index commit `08ffbfaf1`; public HIP/Vulkan and ARM CUDA v0.8.47 links are verified. Final wheel audit **40789883589** is queued; collect its accepted receipt before completing B. Do not retag.
- D: 40735433388 completed on EPYC 9V74 with all three actual CPU profiles. Repeated raw arrays independently confirm native/AVX512 Q4 drops `A`, AVX2 Q4 retains it, and every F16 control matches the unchanged original reference. Added a one-word Q4 allowance with mandatory zero-WER F16 transcript and unchanged stage gates in both drivers. Targeted production run 40750908085 at `2c733b9c3` PASSED on EPYC 9V74: Q4 WER 1/23, F16 WER zero, CTC-logit cosine 0.999138. [Accepted receipt](docs/data2vec-production-regression-2026-10-28.json). Do not duplicate. No further capability probes.
- Additional error repaired: normal nightly now fails missing explicitly required stages, matching Kaggle; negative caller controls cover partial/absent captures.
- Resources: light local inspection only; large builds/readbacks/inference on suitable Actions workers.
- Same-family repair owner: current agent; Java/Ruby integer speaker setters also swallow native out-of-range `-2`. Shared native range/public-method fixtures failed before the strict zero-success fixes and pass after: Java 95/9 current/older assertions, Ruby 343 assertions. No model changes.
- Error repair owner: current agent, source `140c02f53`; C# `SetPuncModel`, `SetVoice` and `SetSpeakerId` swallow native `-2` failures. Repaired with public-method negative controls: failed before, 101/35 assertions pass after, using a single compiler process with available memory about 4.9 GiB and load 2.54. Strict zero-success for these explicit operations preserves optional tuning-knob compatibility.
- OpenVoice numerical repair owner: current agent. The three original-recipe fixes pass **27/27 on CPU and T4 CUDA** on identical frozen model/reference bytes; raw waveform cosine 0.99999973, norm ratio 1.000042. The full three-case conditioning/shared-noise/cache and ASR/speaker gates also pass in version 4: [accepted receipt](docs/openvoice-conditioning-accepted-2026-10-28.json). The Session composition also passes (G below). Next qualify a full rebuilt-current library, actual CLI and authorized model publication. No default promotion.
- Current-runtime validation owner: current agent, starting source `24178da9d`. Separate frozen replay is accepted against the v0.8.47 SDK and matching `e2ae679a` ggml headers: resolved dependencies, four overlay symbol owners, 27/27 CPU and 27/27 T4 CUDA stages, raw waveform norm ratio 1.000042. [Receipt](docs/openvoice-current-runtime-replay-2026-10-28.json). Full conditioning version 3 passed native checks but hit a PyAV decoder error; repaired version 4 passes quality. No local model inference or SDK rebuild. Both Kaggle jobs are terminal; no replacement is required.
- C: user requests keeping #557 open and preparing a reply for review; do not post or close. E remains hardware-dependent.
- OpenVoice Session integration COMPLETE: kernel version 2 passes architecture detection on a renamed model, codec/voice setters, native output rates, raw/marked synthesis, exact reseeded repeats, actual CUDA and unchanged content/speaker gates. Preset identity remains unknown; raw validation audio stays private. [Accepted receipt](docs/openvoice-session-accepted-2026-10-28.json). This is the shipped-SDK/model-overlay composition; actual CLI and full rebuilt-current-library qualification remain separate.
- Newly opened #558: reproduced and repaired standalone-marker grouping, with model-free regression controls. Reporter audio/NeMo/CUDA acceptance remains pending; see [repair scope](docs/parakeet-word-boundary-558-2026-10-28.md).

- v0.8.46 is **published**. Native and wheel audits pass (task A complete).
- v0.8.47 (#557 Linux GPU wheels) has a **partial public release**, tagged at `5f87a5626`. Release run
  40711745769 completed successfully with exactly 52 public asset names. Native byte audit 40751444564 is accepted; Python wheel producer 40751349609 passed all 14 jobs. Pages 40776528499 and live index links pass; final wheel audit 40789883589 remains queued (task B).
- [Publication receipt](docs/release-publication-2026-10-28.md).

## Completed baseline

- v0.8.45 and v0.8.46 are published. v0.8.46 carries the qualified Nemotron F16
  default; its tag/assets and v0.8.45's must not be rewritten.
- #557 code is merged: Linux x86_64 `+vulkan`/`+hip` and aarch64 SBSA `+cuda`
  wheels and producers, and a `whl/hip` index.
- Earlier completed work is unchanged: Qwen3-TTS #555 voice packs, Granite5
  Q4_K, CSM RoPE, OmniVoice processing, the MMQ archive and the binding fixes.
  See [HISTORY.md](HISTORY.md).

## Open tasks — in order

| # | Task | Depends on | Done when |
|---|---|---|---|
| A | COMPLETE: v0.8.46 wheel auditor 40736324867 passed. | — | Accepted receipt committed; do not repeat. |
| B | Verify the v0.8.47 publication: release jobs, 52 native assets, three new GPU wheel legs, `whl/hip` + aarch64 `whl/cuda` index entries, native and wheel audits. | v0.8.47 release run finished | Both verifier runs pass; IDs recorded. For transient failures rerun failed jobs in the original producer. Whole redispatch risks different CPU bytes colliding with immutable PyPI; see handoff. Never re-tag. |
| C | Prepare the #557 final reply for maintainer review; keep the issue open. | B | Both audits support the draft claims; reviewed wording is ready. No posting or closure authorized. |
| D | COMPLETE: data2vec-base Q4_K nightly diagnosis and paired Q4/F16 production gate. | — | Cause retained in same-host arrays; model-free guards and actual production regression 40750908085 pass. |
| F | COMPLETE: bounded OpenVoice2 conditioned-cloning CUDA proof; separate current-runtime structural replay accepted. | — | All three 27/28-stage cases, waveform/cache/clip-mean, WER <=0.2 and speaker improvement >0.01 pass. Publication and high-level integration remain separate. See handoff follow-up scope. |
| G | COMPLETE: real OpenVoice Python Session/C ABI validation over the qualified SDK/model-overlay composition; kernel version 2 accepted. | F | Architecture detection, voice/codec/rates/repeatability, raw and marked synthesis, actual CUDA and unchanged ASR/speaker gates pass. Preset identity stays unknown; full rebuilt-current library and actual CLI remain separate. |
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
Refresh complete issue threads before resuming if new evidence arrives. Six
issues were open at the 2026-10-28 check: #456, #461, #483, #532, #557
(implemented, awaiting publication acceptance) and #558. New PR #559 is reviewed
in [a scoped checkpoint](docs/qwen3-context-pr559-review-2026-10-28.md); CI and
independent caller/runtime checks remain. Keep this feature after active error
repairs. None was closed merely to finish this batch.

## Parked work and historical detail

The handoff contains a bounded research/API inventory. It is not an active
assignment or a release blocker. Audit code and current evidence before
activating anything from [PLAN-ARCHIVE.md](PLAN-ARCHIVE.md); its old NOW/OPEN and
queued labels are historical. `HISTORY.md` records completed work, not a new queue.
All actual machine paths, credentials/account details, storage and job recovery
instructions remain in gitignored local Markdown.
