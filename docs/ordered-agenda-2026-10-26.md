# Ordered correctness and dependency work — 2026-10-26

The six-item agenda was worked in order. Implementation, preservation and the
full license reviews are complete; the external reproducer, device and rights
dependencies below remain explicit. The final sweep keeps four existing issues open, finds new feature #557, and
finds no open PRs. No issue comments/messages or unsupported closures were made.
[Full receipt](ordered-agenda-2026-10-26.json),
[final issue/PR audit](ordered-agenda-issue-review-2026-10-26.json).

| Item | Result |
|---|---|
| Nemotron default | Pinned F16 passes all 26 original transcript and 22 VoxCPM prompt controls on CPU and actual T4, including unchanged stage/magnitude, CLI, opaque C ABI and session-reuse controls. Former plain Q4 fails on both devices. Managed downloads select qualified F16, retain full OpenMDW/origin notices, and prevent stale-Q4 or missing-Q4 substitution. [Evidence](nemotron-default-policy-2026-10-26.md). |
| Canary #532 | Whole issue/comments and source reviewed. No exact failing clip/assertion/startup details are available; existing probes do not reproduce the private pipeline. Keep open without a speculative allocator patch. [Assessment](canary-532-reproducer-review-2026-10-26.json). |
| CUDA/MMQ #483 | Original matched packages preserved byte-for-byte before expiry, with full vendor/cuBLAS and original notices; both draft assets pass full readback and independent small-sidecar inspection. No GTX1660/MX150 access, so no target acceptance or speed claim. [Preservation](cuda126-mmq-preservation-2026-10-26.md). |
| B390 #461 | Historical log/source audit identifies VAE initialization variance and corrects the timer that conflated upload with host permutation. Ten steps remain the quality default. No B390 access or new speed claim. [Review/capture](voxcpm2-b390-review-2026-10-26.md). |
| MUSA | Current terms and runner status reviewed; load/memory enforced before SDK pull. Hosted policy controls pass without MUSA SDK use. No authorized S5000/new SDK permissions; runtime, redistribution and benchmark-disclosure requirements remain open. [Review](musa-permissions-review-2026-10-26.json). |
| Nyra #456 | Complete pinned software/model/output license reviewed. Production/hosted, derivative, recipient and downstream-output restrictions exceed a normal NC gate. No appropriate rights supplied; keep port/conversion deferred. [Review](nyra-license-review-2026-10-26.md). |

Frozen native source `2506cdd4b` passes all **13 CI jobs, 10 lint jobs and the
privacy check**. The 2,104-test unit tier has zero failures and four optional
fixture skips; all four Nemotron default/C ABI controls pass. The archive worker
at `45cd1f949` passes eight archive/resource/draft guards, actual preservation and
its own privacy check. Native source bytes are unchanged between those commits.
Both hosted Docker smoke jobs pass, including all eight MUSA policy controls;
this does not qualify a MUSA runtime image or GPU speech output.

The original MMQ pair stays in an unpublished maintainer draft. Source fixes
have not been retroactively inserted into published v0.8.45 binaries: those
binaries can use the qualified Nemotron F16 file explicitly, while the new
automatic selection requires a build containing this source fix. No new release,
MUSA image, Nyra model or generated alignment dataset was published here.

The final sweep discovered new [#557](https://github.com/CrispStrobe/CrispASR/issues/557).
Its complete thread (zero comments), current publishing code and actual release
assets were reviewed. Recommend Linux x86_64 Vulkan, then HIP, followed by the
lower-priority ARM CUDA target. [Concrete packaging gaps/next steps](issue-557-linux-gpu-wheels-review-2026-10-26.md).
It remains open for the next feature batch; no new wheel publication was started.
