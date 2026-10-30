# CrispASR — Current work plan

Updated 2026-10-29. Start with [the maintainer handoff](docs/maintenance-handoff.md).
It contains the completed work, exact evidence, bounded next actions and blockers
needed by a fresh agent; no earlier chat is required. Follow `AGENTS.md` and its
mandatory maintainer-guide instructions before implementation.

**A, B, D, bounded F, G, H, I, J (#559) and K + full-recording #560 are complete. C is prepared for review; keep #557 open.** Claim work by adding owner,
source SHA, missing behavior, acceptance gates and next action under "NOW"
before starting. The full scope of each task (exact runs, inputs, commands and
acceptance) is in [the handoff, "Open tasks"](docs/maintenance-handoff.md#open-tasks--fully-scoped).

## NOW — release verification and nightly diagnosis (2026-10-28)

### Current handoff — 2026-10-29 12:20 UTC

Updated 2026-10-29 17:35 UTC: no work is claimed and zero Kaggle GPU jobs remain active. The [continuation snapshot](docs/maintenance-handoff.md#continuation-snapshot--2026-10-29-1220-utc) supersedes older owner/status prose below and records all completed run IDs and public receipts. Keep #557 open for reply review.

| Priority | Open work | Current evidence / next gate |
|---|---|---|
| 1 | COMPLETE: Kaggle Whisper loader A/B and same-toolchain v4 | Loader-cache A/B v1 passed (`576.68 -> 318.37 ms`, [receipt](docs/historical/whisper-loader-device-ab-t4-diagnostic-2026-10-29.json)). Same-toolchain v4 completed ([receipt](docs/historical/whisper-toolchain-paired-t4-diagnostic-2026-10-29.json)): `crisp` beats `upstream` (`1.142 s` vs `1.292 s`) but is `+12.3%` vs whisper-only `fork` (`1.017 s`, decode `3.23` vs `2.52 ms/call`); `d28102214` stays on `validate/whisper-integration-20261029`. |
| 2 | COMPLETE: v0.8.48 wheel publication and audit | Producer **40856452802**, Pages `a17d1957e`, and final wheel audit **40895402248** passed ([receipt](docs/historical/release-v0848-wheel-audit-2026-10-29.json)). |
| 3 | COMPLETE: Integrate #559, then #560 | `#559` merged at `777e8ff23` (CI **40899681646**, Lint **40899687134**); `#560` merged at `641e9bfab` (CI **40900035576**, Lint **40900041428**, Heavy CPU **40900046409**, [receipt](docs/historical/issue-560-exact-tree-integration-2026-10-29.json)). |
| 4 | Follow-up: Whisper per-step decode gap vs whisper-only fork | Per-call threadpool overhead was ruled out (`21–30 us` total across 27 calls); remaining gap vs `fork` is ~`0.70 ms/call` in decode (`3.226` vs `2.524 ms/call`). Isolate per-op CUDA graph/scheduler difference before merging `d28102214`. |
| 5 | COMPLETE: Original and native step / paired measurements | Granite original steps ([receipt](docs/historical/granite5-original-steps-t4-2026-10-29.json)), CSM original steps ([receipt](docs/historical/csm-original-steps-t4-2026-10-29.json)), CSM native steps ([receipt](docs/historical/csm-native-steps-t4-2026-10-29.json)) and Melo paired v1–v3 ([receipt](docs/historical/melo-original-paired-t4-diagnostic-2026-10-29.json)) are archived. |
| 6 | Qualify remaining upstream safety candidates | Short DTW (`d28102214`) passed CI **40903036393**, Lint **40903042314** and full inference **40903516297** ([receipt](docs/historical/whisper-dtw-short-2026-10-29.json)). VAD offset and vocab-bounds remain on validation branches pending rights-cleared model checks. |
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

## Open tasks — in order

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
