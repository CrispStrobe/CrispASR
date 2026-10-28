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
| Release | 40711745769 | PASS: 38/38 jobs including final publish; exact 52-name public asset set checked. Full byte audit 40751444564 queued — task B |
| Rust / Dart wrappers | 40711745994 | PASS: all three publisher jobs; three registry archives fully read back, registry SHA-256/version, original manifests, licenses and packaged source files match frozen tag ([receipt](release-wrappers-readback-2026-10-28.json)) |
| Python wheels | 40751349609 | automatic workflow_run producer queued; no wheel acceptance yet |
| Native byte publication audit | 40751444564 | queued; full 52-asset readback not yet accepted |
| Wheel publication audit | — | waits for wheels/index/Pages and accepted native inventory — task B |

Not covered: execution on a physical AMD, Intel or ARM NVIDIA GPU (task E). The
`+hip` wheel cannot be load-tested on hosted runners.

## Current follow-up attempts

- v0.8.46 corrected wheel auditor: **40736324867**, `ubuntu-latest`, queued.
  Pinned Ubuntu 24.04 attempt 40723794473 was canceled after remaining unassigned.
  Original Ubuntu 22.04 attempt 40720956336 was canceled while queued.
- CPU-profile diagnostic: **40723799365** at `bf724a8e5`, Ubuntu 24.04/EPYC 7763,
  succeeds for native/AVX2 only. Twelve raw controls, eight pipeline controls,
  two F16 stage checks and 26 resource gates independently pass
  ([receipt](data2vec-avx2-controls-2026-10-28.json)). AVX-512 remains unavailable.
  Original attempt 40721782580 was canceled while queued. No acceptance gate changed.
- Native auditor now requires the exact reviewed asset set: 50 legacy names,
  plus the two new native bundles from v0.8.47. Nineteen auditor unit controls
  pass; current public v0.8.45/v0.8.46 names pass, incomplete v0.8.47 is rejected.
- The user explicitly requested keeping #557 open and preparing its final reply
  for review. No comment was posted and no issue was closed.
- Later source repair for standalone Parakeet word boundaries is separately
  documented in [#558 scope](parakeet-word-boundary-558-2026-10-28.md). It is
  outside frozen v0.8.47 and is not new speech/device acceptance.

The genuine older data2vec control is now independently rechecked in
[data2vec-cpu-profile-evidence-2026-10-28.json](data2vec-cpu-profile-evidence-2026-10-28.json):
EPYC 7763/AVX2 (without AVX-512) passes on unchanged relevant runtime objects,
with the same compiler version. This supports testing CPU sensitivity; causal
acceptance and F16 transcript controls remain pending. Manual diagnostic backend
runs now retain successful data2vec host/build evidence as well as CSM evidence.
No transcript or numerical gate was relaxed.

AVX-512 follow-up **40735433388** at `a21687e29` is queued, capability attempt
1/3, using `--require-avx512`. Collect it before dispatching a replacement; a
capability-only skip does not close the numerical investigation. The auditor's
`ubuntu-latest` option changes only its worker label, not release platforms or
tagged bytes. The final worker-label/docs checkpoints skip redundant native CI;
normal code checks on `a21687e29` are separate and are not claimed as passing.

Recovery limit: a full wheel-workflow rerun can rebuild different archive bytes
while `twine --skip-existing` preserves the original PyPI files. Prefer failed-job
reruns within the same producer run for transient failures. A workflow-code fix
after CPU publication requires qualified original CPU artifact reuse or a new
patch version; byte-preserving cross-run reuse is not implemented. Strict public
byte comparisons remain mandatory.
