# ASR prompt fixes and validation — 2026-10-03

Qwen3 hotwords, MiMo language prompts, and Glint authentication are integrated on
main. MioTTS/Echo source is also integrated; see the separate
[MioTTS/Echo integration report](miotts-echo-integration-2026-10-03.md). Echo Q4
model acceptance remains pending. No new release/tag was cut.

## Integrated fixes

| Change | Behavior and evidence | State |
|---|---|---|
| [Qwen3 #488](https://github.com/CrispStrobe/CrispASR/issues/488) | Shared CLI/streaming/session prompt builder places hotword context in the system turn, preserving forced-language assistant prefill and explicit questions. Three Piper clips pass off/on/clear session calls and CLI: 12 recognitions with exact expected transcripts. | Landed; issue open for reporter retest. |
| [MiMo PR #489](https://github.com/CrispStrobe/CrispASR/pull/489) | Author-preserving merge plus native language-selected default instructions. English/Chinese forced and auto modes match CLI/session output exactly with public Q4_K and matching codec; auto uses no external Whisper LID. Capability tables and binding/architecture docs reflect the behavior. | Merged and integrated. |
| Glint sync | Checkout uses the built-in GitHub token. A real sync push explicitly dispatches CI; manual dry-run mode validates provenance without publishing changes. | Dry run and normal main run pass. |

Native speech proof is [hosted run 37120330648](https://github.com/CrispStrobe/CrispASR/actions/runs/37120330648)
at source `6f27e8d36b73401b1b46fa6c16eed4e0b08dcd6e`. Prompt units pass
17 assertions/3 cases and native MiMo units pass 12 assertions/6 cases.
The [JSON receipt](asr-prompt-validation-2026-10-03.json) records immutable
model, audio and source pins, actual transcripts, and workflow evidence.

Glint [dry run 37120211560](https://github.com/CrispStrobe/CrispASR/actions/runs/37120211560)
passes 48 MP3/AAC provenance assertions across 13 cases. Normal main
[run 37122115577](https://github.com/CrispStrobe/CrispASR/actions/runs/37122115577)
authenticates successfully and confirms upstream
`77738f3ed9b15f627196cc5bbd7f6406814ba2fb` is current. Because upstream is
unchanged, the conditional commit/push/CI-dispatch path was not exercised.

Main [CI 37123144540](https://github.com/CrispStrobe/CrispASR/actions/runs/37123144540)
is green at `6e451b51f2716dde27e59525605a7752a267b9b8`.
[Lint 37122745123](https://github.com/CrispStrobe/CrispASR/actions/runs/37122745123)
is green at the preceding README follow-up
`9fe672f127ef0ae50d6e520c8f48c88720ebf7de`. Feature-source CI and lint also
passed all 13 and 10 jobs respectively before integration.

These speech results are CPU prompt and decoded-output checks. The reporter's
six recordings were unavailable; Windows/Vulkan compile success does not prove
speech behavior on that hardware. No new per-stage cosine or GPU performance
claim follows from these prompt fixes.

## Related work

| Work | Proven so far | Remaining gate |
|---|---|---|
| MioTTS | Feature-branch ARM/x86 tests prove model-derived 44100 Hz WAV output, 3.000-second duration, preset restoration and CLI/session speech readbacks at 0% WER. | Integrated with current ABI fixes and regenerated capabilities; see the integration report. |
| Echo scheduler | Feature-branch two-T4 stage/magnitude/cache/output and roundtrip acceptance; 48 AB/BA timed calls show 2.5–3.6% improvement with the opt-in pipeline-disable gate. | Source integrated; no default flip or broader hardware claim. |
| Echo Q4 | F16 GPU control passes. Corrected hosted CPU preparation produces a 5.05 GB plain Q4 decoder. Recipes preserve all original F32 tensors. | HF upload failed private-storage quota; no candidate has a successful artifact pin or GPU acceptance. Establish transfer, prepare/pin candidates, then run unchanged canonical and roundtrip gates. |

The private HF repository was chosen for unvalidated staging; the runtime does
not require a private repository. GPU-only Q4 validator v2 has not been pushed
and still needs artifact hashes/revisions and an updated preparation-run pin.
See [PLAN.md](../../PLAN.md) for source commits, preparation failures and next steps.

Original prompt proof commits remain reachable through
`archive/asr-prompt-glint-proof-20261003`. Full artifacts/logs and the independent
worktree are under `${CRISPASR_STORAGE}/crispasr/issue488-pr489-glint-20261003/`;
`${CRISPASR_WORKTREES}/wt-488-489-glint` remains a symlink to the cold worktree.
