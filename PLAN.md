# CrispASR — Current work plan

Updated 2026-10-28. Start with [the maintainer handoff](docs/maintenance-handoff.md).
It contains the completed work, exact evidence, bounded next actions and blockers
needed by a fresh agent; no earlier chat is required. Follow `AGENTS.md` and its
mandatory maintainer-guide instructions before implementation.

**Tasks A/B/D claimed by the current maintainer agent.** Claim other work by adding owner,
source SHA, missing behavior, acceptance gates and next action under "NOW"
before starting. The full scope of each task (exact runs, inputs, commands and
acceptance) is in [the handoff, "Open tasks"](docs/maintenance-handoff.md#open-tasks--fully-scoped).

## NOW — release verification and nightly diagnosis (2026-10-28)

- Owner: current maintainer agent; starting source `b967685fd`.
- A: corrected v0.8.46 wheel auditor 40723794473 is queued on Ubuntu 24.04; accept only successful terminal receipt.
- B: monitor existing v0.8.47 producers, then native/wheel audits and public readbacks; do not retag.
- D: controlled CPU/F16 comparison 40723799365 at bf724a8e5 is queued on Ubuntu 24.04; no gate relaxation without proof. The alleged previous data2vec pass was actually skipped.
- Resources: light local inspection only; large builds/readbacks/inference on suitable Actions workers.
- C: user requests keeping #557 open and preparing a reply for review; do not post or close. E remains hardware-dependent.
- Newly opened #558: reproduced and repaired standalone-marker grouping, with model-free regression controls. Reporter audio/NeMo/CUDA acceptance remains pending; see [repair scope](docs/parakeet-word-boundary-558-2026-10-28.md).

- v0.8.46 is **published**. Native audit passes; the wheel audit receipt is pending (task A).
- v0.8.47 (#557 Linux GPU wheels) has a **partial public release**, tagged at `5f87a5626`. Release run
  40711745769 is still building; its complete publication is unverified (task B).
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
| A | Rerun the v0.8.46 wheel audit after the auditor `hip`-index fix. | — | Verifier run `success` and `receipt.json` acceptance true; run ID in the receipt. |
| B | Verify the v0.8.47 publication: release jobs, 52 native assets, three new GPU wheel legs, `whl/hip` + aarch64 `whl/cuda` index entries, native and wheel audits. | v0.8.47 release run finished | Both verifier runs pass; IDs recorded. If a new wheel leg fails, fix and re-dispatch the wheel workflow for `v0.8.47`. Never re-tag. |
| C | Prepare the #557 final reply for maintainer review; keep the issue open. | B | Both audits support the draft claims; reviewed wording is ready. No posting or closure authorized. |
| D | Investigate the data2vec-base Q4_K nightly exact-match failure (run 40481871937). | — | Controlled profile/raw-CTC/pipeline comparison explains the failure; any gate change requires original-reference evidence. |
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
