# CrispASR — Current work plan

Updated 2026-10-27. Start with [the maintainer handoff](docs/maintenance-handoff.md).
It contains the completed work, exact evidence, bounded next actions and blockers
needed by a fresh agent; no earlier chat is required. Follow `AGENTS.md` and its
mandatory maintainer-guide instructions before implementation.

**No implementation task or model/GPU job is assigned by this handoff.** Claim
one bounded task here before starting. This documentation request does not
start a new release, feature build, model campaign or package publication.

## Completed baseline

- v0.8.45 is published and verified; tag source remains `e58bb48b5`.
- The later Nemotron F16 default, notices/cache guards and B390 timing correction
  are on main. Native source `2506cdd4b` passes 13 CI and 10 lint jobs; its
  2,104-test tier has zero failures and four optional fixture skips. Later native
  bytes through checkpoint `630a1fc1f` are identical. Deep lint at `5c98e2196`
  also passes with those same native bytes.
- Qwen3-TTS #555 model-sized voice packs, Granite5 mixed Q4_K, CSM original RoPE,
  accepted OmniVoice processing and the binding/publication fixes are complete
  within their recorded scopes. Do not redo their accepted campaigns.
- Original #483 MMQ packages are preserved with full vendor notices and verified
  readbacks in an unpublished maintainer draft. Original artifacts need not be
  recovered again before their November 7 expiry.
- [Completed six-item review](docs/ordered-agenda-2026-10-26.md),
  [release receipt](docs/release-publication-2026-10-26.json),
  [historical completion log](HISTORY.md).

**Published/source distinction:** v0.8.45 binaries retain the old Nemotron
registry default. Use the qualified F16 file explicitly with those binaries;
the automatic F16 selection requires a build containing the source fix.

## Proposed next execution order — unclaimed

| Priority | Bounded task | Entry point and completion gate |
|---|---|---|
| 1 | Prepare promotion of the qualified correctness fixes in a new release. | Follow the handoff's release scope. Freeze the selected source, use the version script, validate package/wheel payloads, notices, relocation, C ABI and public readbacks. Preserve v0.8.45/tag/assets; do not silently retag or republish them. Publication belongs to a separately scoped implementation/release task. |
| 2 | #557 Linux x86_64 Vulkan wheel. | Add the missing relocatable shared-library producer, then the wheel leg. The existing Vulkan CLI is static and is not a shared-lib bundle. Require clean-environment loader/ABI/notice/provenance checks and honest actual-device scope. |
| 3 | #557 Linux x86_64 HIP wheel. | Reuse the existing pinned HIP lib bundle; add wheel/local-version/index support and expand publication-audit expectations. Preserve the explicit host-runtime contract and carry exact dependency notices. |
| 4 | #557 ARM CUDA, lower priority. | Separate ARM-native producer/toolchain/runtime/architecture and device qualification. An x86_64 T4 run does not qualify an aarch64 wheel. |

Full implementation map and acceptance details:
[#557 code/asset audit](docs/issue-557-linux-gpu-wheels-review-2026-10-26.md).
No #557 implementation or wheel publication has been performed.

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
issues were open at the latest check: #456, #461, #483, #532 and new feature #557;
no PR was open. None was closed merely to finish this batch.

## Parked work and historical detail

The handoff contains a bounded research/API inventory. It is not an active
assignment or a release blocker. Audit code and current evidence before
activating anything from [PLAN-ARCHIVE.md](PLAN-ARCHIVE.md); its old NOW/OPEN and
queued labels are historical. `HISTORY.md` records completed work, not a new queue.
All actual machine paths, credentials/account details, storage and job recovery
instructions remain in gitignored local Markdown.
