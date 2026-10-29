# Release publication receipt — v0.8.46 and v0.8.47 (2026-10-28)

Public run IDs only. Each line says what it verifies and what it does not.

## v0.8.46 — Nemotron qualified F16 default and packaging provenance

| Check | Run | Result |
|---|---|---|
| Tag `v0.8.46` → `34a8e436192de19d3e5a872437edfe3fd26ce8c4` (annotated, VERSION 0.8.46) | — | verified by the auditor's tag/VERSION asserts |
| CI / Lint / Secret scan on tag source | 40470378993 / 40477169406 / 40470378983 | pass |
| Release (native assets) | 40649389464 | 36/36 jobs pass; 50 native assets, same family set as v0.8.45 |
| Native publication audit | 40682766228 | PASS: full anonymous readback of all 50 assets, root notices, CLI runtime-version smoke, CUDA split/runtime payload sets |
| Python wheels (PyPI CPU + sdist, GPU index) | 40679618075 | all 8 distributions built and published |
| Rust / Dart wrappers | 40649389411 | pass |
| Wheel publication audit | 40703639437 | FAILED in the auditor (required a `whl/hip` index that only exists from 0.8.47). Fixed on main; superseded by successful audit 40736324867 |
| Corrected wheel publication audit | 40736324867 | PASS: all eight distributions, five full PyPI and all GPU readbacks; [accepted receipt](release-v0846-wheel-audit-2026-10-28.json) |

Not covered: new speech or device evidence. The v0.8.46 native inference changes
reuse the Nemotron CPU/T4 qualification described in
[nemotron-default-policy-2026-10-26.md](nemotron-default-policy-2026-10-26.md).

A nightly regression run (40481871937) failed data2vec-base Q4_K on exact match
(WER 0.0435, human-accuracy WER 0). The preceding green workflow skipped data2vec, so it supplies no passing
control. The failed job retained AMD EPYC 9V74/native-build evidence and a
passing F16 stage check; the same-host follow-up 40735433388 establishes Q4 native/AVX512 versus AVX2
near-tie sensitivity while every F16 control passes. The paired regression
policy and remaining hosted confirmation are tracked as task D.

## v0.8.47 — Linux GPU Python wheels (#557)

| Check | Run | Result |
|---|---|---|
| Tag `v0.8.47` → `5f87a56265e4bbd7e2adc7052254e576d5774abb` | — | pushed 2026-10-28 |
| CI / Lint / Secret scan on tag source | 40689967257 / 40689967286 / 40689967265 | pass |
| Vulkan lib producer dry run (branch) | 40460158618 | pass: build, relocated dlopen verify, dependency gate (`libvulkan.so.*` host) |
| ARM64 CUDA lib producer dry run (branch) | 40661942703 | pass: same gates (CUDA runtime host) |
| Release | 40711745769 | PASS: 38/38 jobs including final publish; exact 52-name public asset set checked. Full byte audit 40751444564 accepted — task B |
| Rust / Dart wrappers | 40711745994 | PASS: all three publisher jobs; three registry archives fully read back, registry SHA-256/version, original manifests, licenses and packaged source files match frozen tag ([receipt](release-wrappers-readback-2026-10-28.json)) |
| Python wheels | 40751349609 | PASS: all 14 jobs including all builds, PyPI and GPU-index publication. Pages and live index links pass; final wheel audit 40789883589 is accepted |
| Native byte publication audit | 40751444564 | PASS: complete/acceptance true; all 52 full anonymous readbacks, notices/runtime/qualified-library inventory accepted. [Receipt](release-v0847-native-audit-2026-10-28.json), [inventory](release-v0847-native-assets-2026-10-28.json) |
| Pages deployment | 40776528499 | PASS at index commit `08ffbfaf1`; live HIP/Vulkan and ARM CUDA v0.8.47 links verified |
| Wheel publication audit | 40789883589 | PASS: complete/acceptance true; all 11 original distributions, five PyPI anonymous readbacks and all GPU index/full-file readbacks. [Accepted receipt](release-v0847-wheel-audit-2026-10-29.json) |

Not covered: execution on a physical AMD, Intel or ARM NVIDIA GPU (task E). The
`+hip` wheel cannot be load-tested on hosted runners.

## Current follow-up state

- A is complete: v0.8.46 wheel auditor **40736324867** passed. Previous
  unassigned attempts 40723794473 and 40720956336 were canceled; do not resume
  or duplicate them.
- D has decisive same-host CPU-profile evidence from **40735433388** on
  EPYC 9V74: native/AVX512 Q4 drops `A`, AVX2 Q4 retains it, and all F16
  controls exactly match the original transcript. The earlier EPYC 7763 run
  **40723799365** lacked AVX512 and is a retained historical control.
  The source now pairs the one-word Q4 allowance with a mandatory exact F16
  transcript and unchanged stage gates. Production regression **40750908085**
  PASSED: Q4 WER 1/23, exact F16 WER zero and CTC-logit cosine minimum
  0.999138 (threshold 0.999). [Receipt](data2vec-production-regression-2026-10-28.json).
  Do not dispatch another capability probe.
- B's native audit is accepted. All wheel builds passed in producer
  **40751349609**, including both publishers and the three new GPU variants.
  Pages **40776528499** and live index links pass. Final wheel audit
  **40789883589** is accepted against native audit **40751444564**. B is complete. Never retag v0.8.47.
- The user requests keeping #557 open and preparing its reply for review.
  The [final reply](issue-557-reply-review-2026-10-29.md) is prepared for review. No comment was posted and no issue was closed.
- Standalone Parakeet word-boundary repairs are separately documented in
  [#558 scope](parakeet-word-boundary-558-2026-10-28.md). They are outside
  frozen v0.8.47 and do not establish reporter speech/device acceptance.

Recovery limit: a full wheel-workflow rerun can rebuild different archive bytes
while `twine --skip-existing` preserves the original PyPI files. Prefer failed-job
reruns within the same producer run for transient failures. A workflow-code fix
after CPU publication requires qualified original CPU artifact reuse or a new
patch version; byte-preserving cross-run reuse is not implemented. Strict public
byte comparisons remain mandatory.
