# CrispASR — Current work plan

Updated 2026-10-28. Start with [the maintainer handoff](docs/maintenance-handoff.md).
It contains the completed work, exact evidence, bounded next actions and blockers
needed by a fresh agent; no earlier chat is required. Follow `AGENTS.md` and its
mandatory maintainer-guide instructions before implementation.

**No task is claimed.** Every row below is open. Claim one by adding owner,
source SHA, missing behavior, acceptance gates and next action under "NOW"
before starting. The full scope of each task (exact runs, inputs, commands and
acceptance) is in [the handoff, "Open tasks"](docs/maintenance-handoff.md#open-tasks--fully-scoped).

## NOW — nothing claimed (2026-10-28 wrap-up)

- v0.8.46 is **published**. Native audit passes; the wheel audit receipt is pending (task A).
- v0.8.47 (#557 Linux GPU wheels) is **tagged** at `5f87a5626`. Release run
  40711745769 was in flight; nothing about its outputs is verified (task B).
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
| C | Post the approved #557 reply (draft in the handoff) and close #557. | B | Maintainer approved the text; comment posted; issue closed. |
| D | Explain the CPU-sensitive data2vec-base Q4_K nightly exact-match failure (run 40481871937). | — | CPU flags recorded for passing and failing runs, F16 checked on both, and the gate adjusted with evidence. |
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
Refresh complete issue threads before resuming if new evidence arrives. Five
issues were open at the 2026-10-28 check: #456, #461, #483, #532 and #557
(implemented, awaiting tasks B/C); no PR was open. None was closed merely to finish this batch.

## Parked work and historical detail

The handoff contains a bounded research/API inventory. It is not an active
assignment or a release blocker. Audit code and current evidence before
activating anything from [PLAN-ARCHIVE.md](PLAN-ARCHIVE.md); its old NOW/OPEN and
queued labels are historical. `HISTORY.md` records completed work, not a new queue.
All actual machine paths, credentials/account details, storage and job recovery
instructions remain in gitignored local Markdown.
