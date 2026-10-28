# CrispASR — Current work plan

Updated 2026-10-28. Start with [the maintainer handoff](docs/maintenance-handoff.md).
It contains the completed work, exact evidence, bounded next actions and blockers
needed by a fresh agent; no earlier chat is required. Follow `AGENTS.md` and its
mandatory maintainer-guide instructions before implementation.

**A is complete; D has decisive retained CPU-profile evidence and a scoped regression fix. B publication remains in flight.** Claim work by adding owner,
source SHA, missing behavior, acceptance gates and next action under "NOW"
before starting. The full scope of each task (exact runs, inputs, commands and
acceptance) is in [the handoff, "Open tasks"](docs/maintenance-handoff.md#open-tasks--fully-scoped).

## NOW — release verification and nightly diagnosis (2026-10-28)

- Owner: current maintainer agent, starting source `42d787586`; collect A/B/D terminal receipts, repair actual failures, and preserve existing runs. Large work stays on suitable hosted workers; local load 0.26 and available memory about 4.4 GiB at resumption.
- A complete: corrected auditor 40736324867 passed; all eight distributions and anonymous readbacks accepted. [Receipt](docs/release-v0846-wheel-audit-2026-10-28.json).
- B: native producer 40711745769 completed successfully (38/38 jobs including publish). Native audit 40751444564 and automatic Python wheel producer 40751349609 are queued; collect both, then Pages and wheel audit. Do not retag.
- D: 40735433388 completed on EPYC 9V74 with all three actual CPU profiles. Repeated raw arrays independently confirm native/AVX512 Q4 drops `A`, AVX2 Q4 retains it, and every F16 control matches the unchanged original reference. Added a one-word Q4 allowance with mandatory zero-WER F16 transcript and unchanged stage gates in both drivers. Targeted hosted production run 40750908085 at `2c733b9c3` is queued; collect it, do not duplicate. No further capability probes.
- Additional error repaired: normal nightly now fails missing explicitly required stages, matching Kaggle; negative caller controls cover partial/absent captures.
- Resources: light local inspection only; large builds/readbacks/inference on suitable Actions workers.
- Same-family repair owner: current agent; Java/Ruby integer speaker setters also swallow native out-of-range `-2`. Shared native range/public-method fixtures failed before the strict zero-success fixes and pass after: Java 95/9 current/older assertions, Ruby 343 assertions. No model changes.
- Error repair owner: current agent, source `140c02f53`; C# `SetPuncModel`, `SetVoice` and `SetSpeakerId` swallow native `-2` failures. Repaired with public-method negative controls: failed before, 101/35 assertions pass after, using a single compiler process with available memory about 4.9 GiB and load 2.54. Strict zero-success for these explicit operations preserves optional tuning-knob compatibility.
- Independent recovery owner: current agent, source `140c02f53`; recovered OpenVoice2 terminal v2 logs/source: CPU backend registration abort, not model acceptance. Prepared worker now checks headroom before compute. Bounded session metadata then confirmed recent active jobs are CPU-only; a single GPU admission succeeded. OpenVoice conditioning kernel version 1 (worker `.2`) reached terminal ERROR: Hub create_repo 403 before inference, after builds/conversion and five resource guards. Evidence preserved. Worker `.3` separates read-only validation from optional publication; repeat the software-corrected proof once with unchanged acceptance gates. Native source remains `6be9fda36`. Acceptance stays 27/28 strict stages plus speech/speaker controls. Read-only wrapper-contract review delegated to one Sol agent; no local build/inference. Resumption headroom: four CPUs, load 2.52, available RAM about 4.6 GiB.
- C: user requests keeping #557 open and preparing a reply for review; do not post or close. E remains hardware-dependent.
- Newly opened #558: reproduced and repaired standalone-marker grouping, with model-free regression controls. Reporter audio/NeMo/CUDA acceptance remains pending; see [repair scope](docs/parakeet-word-boundary-558-2026-10-28.md).

- v0.8.46 is **published**. Native and wheel audits pass (task A complete).
- v0.8.47 (#557 Linux GPU wheels) has a **partial public release**, tagged at `5f87a5626`. Release run
  40711745769 completed successfully with exactly 52 public asset names. Native byte audit 40751444564 and Python wheel producer 40751349609 remain unverified (task B).
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
| D | Investigate the data2vec-base Q4_K nightly exact-match failure (run 40481871937). | — | Cause explained by retained same-host arrays; paired Q4/F16 policy and missing-stage fix covered by model-free controls. Confirm a hosted production regression on the committed source. |
| F | OpenVoice2 conditioned-cloning CUDA proof using the repaired read-only worker. | Free supported Kaggle session | 27/28 stages, waveform/cache/clip-mean, WER <=0.2 and independent speaker improvement >0.01 pass, with retained licensed inputs and receipts. Publication remains separate. |
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
(implemented, awaiting publication acceptance) and new #558; no PR was open. None was closed merely to finish this batch.

## Parked work and historical detail

The handoff contains a bounded research/API inventory. It is not an active
assignment or a release blocker. Audit code and current evidence before
activating anything from [PLAN-ARCHIVE.md](PLAN-ARCHIVE.md); its old NOW/OPEN and
queued labels are historical. `HISTORY.md` records completed work, not a new queue.
All actual machine paths, credentials/account details, storage and job recovery
instructions remain in gitignored local Markdown.
