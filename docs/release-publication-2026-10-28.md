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
| Wheel publication audit | 40703639437 | FAILED in the auditor (required a `whl/hip` index that only exists from 0.8.47). Fixed on main; rerun is task A in the handoff |

Not covered: new speech or device evidence. The v0.8.46 native inference changes
reuse the Nemotron CPU/T4 qualification described in
[nemotron-default-policy-2026-10-26.md](nemotron-default-policy-2026-10-26.md).

A nightly regression run (40481871937) failed data2vec-base Q4_K on exact match
(WER 0.0435, human-accuracy WER 0). The preceding green workflow skipped data2vec, so it supplies no passing
control. The failed job retained AMD EPYC 9V74/native-build evidence and a
passing F16 stage check; CPU near-tie variance remains an unverified hypothesis.
The controlled follow-up is tracked as task D.

## v0.8.47 — Linux GPU Python wheels (#557)

| Check | Run | Result |
|---|---|---|
| Tag `v0.8.47` → `5f87a56265e4bbd7e2adc7052254e576d5774abb` | — | pushed 2026-10-28 |
| CI / Lint / Secret scan on tag source | 40689967257 / 40689967286 / 40689967265 | pass |
| Vulkan lib producer dry run (branch) | 40460158618 | pass: build, relocated dlopen verify, dependency gate (`libvulkan.so.*` host) |
| ARM64 CUDA lib producer dry run (branch) | 40661942703 | pass: same gates (CUDA runtime host) |
| Release | 40711745769 | in flight at hand-off — task B |
| Wheels / wrappers / audits | — | not yet run — task B |

Not covered: execution on a physical AMD, Intel or ARM NVIDIA GPU (task E). The
`+hip` wheel cannot be load-tested on hosted runners.
