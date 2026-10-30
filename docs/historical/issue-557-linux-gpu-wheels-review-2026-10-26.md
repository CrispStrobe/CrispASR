> Publication update (2026-10-29): v0.8.47 native audit 40751444564 and wheel
> audit 40789883589 are accepted; all GPU index links are live. The issue stays
> open by maintainer request. [Final reply for review](issue-557-reply-review-2026-10-29.md).
> Physical AMD/Intel/ARM GPU execution remains unverified.

# New #557 — Linux GPU wheel packaging review

> **Status 2026-10-28:** implemented and shipped in v0.8.47 (all three requested
> wheels). Publication verification, the issue reply and real-GPU qualification
> are open tasks B, C and E in [the handoff](../maintenance-handoff.md). The audit
> below is the original pre-implementation review.

[Issue #557](https://github.com/CrispStrobe/CrispASR/issues/557) arrived during the
final six-item audit. The complete description and all comments were read:
there are currently zero comments, no linked issue comments, and no later
owner/reporter follow-up. It is a new feature request, separate from the six
accepted tasks. No build, package publication or implementation was started here.

The reporter confirms Windows `+cuda` wheels and `crispasr.Session` now work
for their Japanese subtitle pipeline, and requests Linux x86_64 `+vulkan`, then
`+hip`, with aarch64 `+cuda` at lower priority. Current source and the actual
v0.8.45 release assets confirm the gap:

| Requested wheel | Current evidence | Required next work |
|---|---|---|
| Linux x86_64 `+vulkan` | Only the Vulkan CLI archive exists. Its native producer uses `BUILD_SHARED_LIBS=OFF`; there is no `libcrispasr-linux-x86_64-vulkan.tar.gz` in the release. GPU wheel matrix has only Linux CUDA, Windows CUDA and Windows Vulkan. | Add a relocatable Vulkan shared-library producer with complete notices/dependency closure, then its manylinux wheel leg. A matrix-only change cannot consume the static CLI archive. |
| Linux x86_64 `+hip` | `libcrispasr-linux-x86_64-hip.tar.gz` already exists, but there is no HIP wheel leg or `whl/hip` bucket. | Reuse the pinned native bundle; add explicit `+hip` metadata/index handling and expand the publication auditor's payload mapping, expected outputs and index checks. |
| Linux aarch64 `+cuda` | CPU aarch64 wheels exist; no matching ARM CUDA native bundle/producer or GPU wheel is published. | Schedule afterward: add a separately qualified ARM CUDA native toolchain/bundle, architecture/runtime policy, wheel leg and target-device validation. |

Implement Vulkan first and HIP second. Reuse existing staging, native-provenance,
notice and manylinux symbol-floor checks; extend the verifier when adding outputs
rather than weakening its exact expected-artifact set. Validate installed wheels
in a clean environment through the C ABI/Python session, including backend/device
selection and adjacent plugin loading. Retain actual GPU smoke/parity evidence
before claiming acceleration; an import or a driverless build alone does not
establish it. Keep ARM CUDA as the lower-priority separate target requested.

`python/tools/stage_libs.py` explicitly treats Vulkan/ROCm GPU runtime and driver
families as host-provided when absent from the bundle. These wheels therefore
need clear compatible-runtime requirements; they must not promise to supply a
GPU driver. Review the exact additional payload/notice obligations for any new
vendored dependency. The existing `release_publication_audit.py` expects three
GPU distributions and handles only CUDA/Vulkan indexes; HIP needs explicit
acceptance coverage as well as packaging.

Source audit: `.github/workflows/release.yml`,
`.github/workflows/release-python-wheels.yml`, `python/tools/stage_libs.py`, and
`tools/ci-heavy/release_publication_audit.py`. Release asset listing checked via
the [v0.8.45 release](https://github.com/CrispStrobe/CrispASR/releases/tag/v0.8.45).
The issue remains open for this next feature batch; this review does not retag,
replace or publish new packages in v0.8.45.
