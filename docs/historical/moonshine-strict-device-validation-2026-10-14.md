# Moonshine strict device validation — 2026-10-14

The implementation tested at `9e2dd0116dd9f9d8b008b22b4d14e9a91757f61d`
is integrated by rebase/fast-forward at `1f71367bbfd1c8768e8e0eae6035362dfda2daf8`.
A subsequent diff reporting correction is `b1cd47fdad39fa8514396cd5c11026f15c3fee22`;
it changes diagnostic row grouping/reporting, with unchanged inference runtime.

The opt-in `CRISPASR_MOONSHINE_REQUIRE_BACKEND` requests an exact ggml registry
(case-insensitive, with metal→MTL alias). `CRISPASR_MOONSHINE_REQUIRE_DEVICE`
selects a nonnegative device ordinal within that registry; default0. Missing,
invalid or unavailable devices and non-CPU requirements with GPU disabled fail
instead of choosing auto/CPU. Shared Moonshine init covers CLI, C ABI Sessions
and public stage APIs without changing ABI structure sizes. Default automatic
selection stays available. The candidate also uses backend type rather than
pointer inequality to identify CPU instances.

In strict mode, the encoder reads scheduler assignments after graph allocation,
requires matmuls on the selected backend and no unassigned nodes, then records
per-backend operation counts and compute success. It adds no eval callbacks,
intermediate output marks or graph changes. CPU fallback for unsupported
individual GPU operations remains legitimate and visible; selecting a GPU with
no encoder matmuls assigned to it cannot pass validation.

The Moonshine diff dispatch activates the requested GPU path, checks exact
reference PCM, refuses missing/empty encoder references, checks full encoder
shape/count and enforces row norm and relative L2 error limits of1%, alongside
the unchanged cosine threshold0.999. Magnitudes are printed for passing stages
too. These are floating-point fixture gates, not accepted quantized-model limits.

Bounded actual native CPU replay passes normal and strict CPU arms. Complete
encoder arrays and both transcripts per arm are byte identical to the accepted
backend proof. Strict CPU logs39 encoder matmuls, zero unassigned nodes and
successful computation. Contradictory GPU-disabled, invalid-index and unavailable
CPU-index model opens fail. Thirteen actual selector/scheduler helper checks pass,
including rejecting an allocated ADD-only graph as model work. The complete diff
translation unit also passes syntax checking with its actual hosted build flags.
This bounded proof does not replace full CLI/C ABI validation.

Hosted full model controls: [38575701613](https://github.com/CrispStrobe/CrispASR/actions/runs/38575701613).
The worker reruns original conversion/dumper/generation and normal CPU checks,
then strict diff/CLI/auto+explicit Sessions and deliberately scaled, nonfinite,
wrong-shape, missing-stage and changed-PCM references. Missing-CANN and invalid
or unavailable device cases must fail through the real entry points.
General CI: [38575706459](https://github.com/CrispStrobe/CrispASR/actions/runs/38575706459).
Lint: [38575711575](https://github.com/CrispStrobe/CrispASR/actions/runs/38575711575).
All thirteen general CI jobs, ten lint jobs and the secret scan pass at the
tested source. The unit job executes2,024 tests, including four new exact-device
selector tests. The hosted model worker passes all13 strict surface/reference/
device controls. No physical NPU, ACL capture/replay, quantized
quality or performance result is claimed. No unexecuted hardware procedure is
published. All build/run/evidence files remain under
`${CRISPASR_STORAGE}/crispasr/cann-model-strict-20261014` or hosted scratch.

The bounded probe also compiles the verbatim actual Moonshine dispatch and
Ref implementation. Good reference passes; doubled scale, NaN and wrong shape
return6; changed PCM and missing encoder reference return4. This uses a raw PCM
loader, so it does not prove the full application audio reader or Session path.
It caught an initially misplaced magnitude edit before the queued model run
executed. The edit is corrected; superseded hosted submissions38574829573,
38574834564 and38574839175 were cancelled, with the observed local failure
retained. No measured model failure was discarded or acceptance gate loosened.

[Complete67-payload bounded proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/1caca74803027ac1b249250b213617970612b061/cann-20261014-model-strict/complete-local-strict-validation.zip) contains actual probes, source, executables, arrays, build/run logs and failed-before/corrected controls. Anonymous full download verifies 6,572,420 bytes, SHA256 `686cc4333f2bd3671846a384c79d0aa2d46b9f4551df15a86b38801e996ff0f0`. CPU libraries/model remain in their already published immutable bundles. [Result receipt](moonshine-strict-device-validation-2026-10-14.json).

## Hosted acceptance and reporting correction

The complete49-payload hosted archive is independently rehashed and checked
against original coefficients, PCM, full encoder arrays and decoded outputs.
Strict and normal CPU arrays are byte identical; all eight automatic/explicit
Session transcripts match the original text byte for byte. Both CLI word
sequences match. All five corrupted reference archives are independently checked
to contain exactly the intended defects, and the real diff binary rejects them.
Missing CANN fails through diff, CLI (exit13) and Session (exit1).

| All457×288 encoder values vs same-host original | Result |
|---|---:|
| Minimum row cosine | 0.999999998826876 |
| Maximum row norm error | 8.589623501698007e-6 |
| Relative L2 error | 8.741357385766627e-6 |
| Maximum absolute error | 2.2140145301818848e-4 |

Original arrays from separate hosts differ at floating-point rounding scale;
they are not byte identical. Shared-library ELF bytes are not included in the
hosted archive, so library hashes remain producer receipts. Complete build,
compilation-command and library-closure logs are retained.

The final reporting correction compares canonical288-value encoder rows and
preserves measured cosine when a magnitude gate fails. The real extracted
Moonshine dispatch/Ref/backend probe passes the good reference and rejects all
five defects again; the doubled-scale control prints `[FAIL]` with actual
`cos_min=1.000000` and row norm error0.5. Full diff source syntax checking also
passes. This correction changes no inference graph, model or decoded output;
its complete bounded controls are separate from the hosted9e2dd01 receipt.
Current composition CI still must settle before any release tag.

[Complete complete-hosted-strict-fixture.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/9cf29ec347a91d7a96177cfdc048bb235f8c3289/cann-20261014-model-strict/complete-hosted-strict-fixture.zip): 108,495,214 bytes, anonymously verified SHA256 `7d63d2a5b90eb77843aff25107b1d1f70bafcfc1c8f73075e2f5a9a3f87d5e84`.

[Complete complete-hosted-strict-fixture-audit.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/9cf29ec347a91d7a96177cfdc048bb235f8c3289/cann-20261014-model-strict/complete-hosted-strict-fixture-audit.zip): 4,301,226 bytes, anonymously verified SHA256 `b257ca3e737ef9bc52b5c36194c161f1a02922f961afadf0722bb9e71f668b09`.

[Complete post-hosted cosine/reporting controls](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/c49233c8753f682cbf3621a5cb2c801fa5eeab1d/cann-20261014-model-strict/complete-cosine-report-controls.zip): 4,337,883 bytes, anonymously verified SHA256 `de5a6448242882c038305caa58d3a409b2b25c51c2625055ad340211d7134a3e`. Includes actual extracted source, executable, references, full logs, method and per-file ledger.

[Official SDK follow-up38598516003](https://github.com/CrispStrobe/CrispASR/actions/runs/38598516003) compiles this composition for x86_64/910B (ACL graphs on) and ARM64/310P (graphs off). Its result is pending; it cannot validate physical NPU work.
