# Post-release audit integration follow-up — 2026-10-31

## Integration identity

The fetched GitHub `main` commit is
`d71788aa769334940d12096b549f5edd2f7770e7`. Its complete Git tree is
`1695e0835a492e5e2018bfafffe4ff6ded3ea0ea`, exactly the tree delivered in
`CrispASR-audit-repairs-20261031.zip` at local commit `fbc0f701`.
All 79 changed files are present and byte-identical, and the complete tree
comparison has no differences. The integration did not omit or alter repairs.

Two omissions in the original repair batch nevertheless broke ordinary CI.
This follow-up repairs those test/build requirements and expands the focused
validation selection. Production inference and numerical code are unchanged.

## Reproduced failures and repairs

### Omnivoice encoder test did not compile

The ordinary Linux unit job in
[CI run 41196253916](https://github.com/CrispStrobe/CrispASR/actions/runs/41196253916)
failed during its all-target build, before discovery or unit execution.
A local complete build of the identical tree independently fails at
`tests/test-omnivoice-encoder.cpp`: three calls to the private
`higgs_rvq_encode` helper omit its newly required `n_threads` argument.
The earlier targeted repair build did not include this pre-existing test.

The three calls now pass an explicit budget of one for their deterministic
fixtures. The production helper remains explicit about its caller budget;
no test assertion or numerical expectation was relaxed. The affected test
compiles, links, and passes its F32/F16 scalar/cache/graph, tie, allocation,
placement, and cleanup checks.

### Two test files failed the header visibility lint rule

The `core_cpu_backend header reachable everywhere` job in
[Lint run 41196253931](https://github.com/CrispStrobe/CrispASR/actions/runs/41196253931)
fails on the ArkTTS cache and MeanVC2 conditioning tests. Each directly calls
`core_cpu_backend::init` while obtaining its header transitively through the
included production implementation.

Both tests now explicitly include `core/ggml_cpu_backend.h`. The unchanged
lint scanner reproduces two failures before the fix and passes all 129
inspected files afterward. The source-policy check is preserved.

### Prevent the focused gate from missing the same caller again

The existing `omnivoice-encoder-parity` test now carries the
`performance-parity` label and its executable is in every explicit numerical
platform build list. The discovery guard checks its built executable, both
unit/parity registrations, and the actual registered command. It does not send
Catch listing arguments to this standalone program.

Six small positive/rejection controls cover a valid standalone registration,
a missing binary, either missing label, a wrong executable, and a missing full
CTest registration. They run in the regular unit/parity tier as well as the
manual workflow's lightweight validation job.

## Local verification

- The complete configured Linux CPU/CBLAS/OpenMP build passes, including the
  shared library, CLI, server, examples, and all test executables. The optional
  ONNX SDK is absent from this local configuration.
- The expanded discovery guard passes and requires 49 numerical/control cases
  across 14 executables plus the explicitly registered controls.
- All 133 selected `performance-parity` CTests pass, without failures or skips.
- The actual original and repaired compiler/lint outcomes, generated CTest
  inventories, commands, and JUnit results are retained with the follow-up
  bundle. The build reuses the existing identical-source dependency build,
  with candidate test/configuration files verified before execution.

The broader local unit selection executes 2,173 CTests: 2,164 pass, six skip,
and three fail before exercising their intended behavior. The subprocess
fixture and VAD export test use an unavailable hardcoded `/tmp` path. The ASR
environment-parity harness uses Bash process substitution through `/dev/fd`,
which is absent in this sandbox; its sample enumeration fails before invoking
the fake ASR command. A minimal process-substitution control reproduces the
same error while an ordinary pipe succeeds.

Nine additional cases were excluded after the initial run: two needing `/proc`
mapping inspection, six Unix-socket cases after that group timed out, and one
using legacy `std::tmpnam` access to `/tmp`. The six skips comprise four absent
G2P model/dictionary fixtures and two packaging checks requiring `patchelf`.
These cases remain unqualified; the full unit tier must run on a normal CI worker. No
assertion, OS-dependent test, or production behavior was changed to make this
environment report success. The commands, exclusions, JUnit results and
diagnostic controls are retained in the follow-up bundle's validation receipt.

## Separate hosted Moonshine gate remains unresolved

[Moonshine acceptance run 41196253946](https://github.com/CrispStrobe/CrispASR/actions/runs/41196253946)
fails inside `tools/ci-heavy/pr515_acceptance.py` after successful setup and
dependency installation. Its explicit target list does not build
`test-omnivoice-encoder`, so the compile repair above does not establish a fix
for this job.

The same outer step already failed on pre-audit ancestor `5aec8487` in
[run 41132345943](https://github.com/CrispStrobe/CrispASR/actions/runs/41132345943)
and on `dd6362b6` in
[run 41191889426](https://github.com/CrispStrobe/CrispASR/actions/runs/41191889426).
The workflow, reusable runner, acceptance script, and SDK fetch script are
unchanged across those revisions and current main. Public metadata alone
does not establish that their internal failure causes are identical.

The anonymous GitHub API refuses job-log downloads with HTTP 403 requiring
repository rights. The useful next evidence is the `heavy-41196253946`
artifact's `acceptance.json` and last failing preparation/build/test/model log.
Some native models in that job use shared helpers touched by the audit, so the
historical failure does not qualify their current behavior. No source change
is guessed from a generic exit code and no hosted model acceptance is claimed.

The broader GPU/model/MeanVC2 qualification limits in
[the repair report](post-release-audit-repairs.md) remain in effect. No external
workflow was dispatched by this follow-up, and no upstream push is included.
