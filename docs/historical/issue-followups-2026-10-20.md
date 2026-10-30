# Issue follow-ups — 2026-10-20

## License disposition

[Policy](../model-license-policy.md): ordinary NC models remain eligible behind
explicit acceptance. Research-only and additional use/output/revenue/hosting
conditions require separate review. Do not silently treat code and weight
licenses as interchangeable. [Source hashes and correction receipt](license-source-audit-2026-10-20.json).

The exact pinned OmniVoice source already states NC main weights and includes
the Boson tokenizer terms. The public converted-model card is corrected; no
weights were modified. The registry now gates its bundle separately, and
OpenRAIL requires explicit acceptance. Registry controls fail in three cases
before the fix and pass 6,224 assertions in 63 cases afterward. Full bundle
promotion and PR521 remain pending license review and unchanged speech gates.

## Issue550

Version2 reached two T4 GPUs but failed before inference: no CPU backend was
registered by the frozen package from a foreign Python cwd. Complete terminal
logs and partial output were retained. Version3 explicitly loads the verified
bundle's plugins, checks CPU and CUDA0 before model download, and keeps identical
runtime/model/input pins and acceptance gates. It completed and passes an independent audit: all 147/8,925/147 units are
retained over 44.52/2,703/44.52 seconds, returning-short output is exact after
offset adjustment, CLI matches public output, and all 95 encoder splits are
on CUDA0. The long-phase sampled device total peaks at 2,735 MiB. Three forged
controls with recomputed hashes (missing unit, CPU placement, changed return
offset) are rejected. [Audit](alignment-cuda-live-2026-10-20.json).
This is not natural conversational alignment accuracy, constant host memory or
RTX3090 certification. Three loader protocol
controls pass. The local Kaggle CLI launcher was also repaired and tested
against a package exposing the CLI without a package `__main__`.

## Shared ggml PR2 review

Exact proposal head: `2ef4ee11c73e3a4186c4e2b1e8b522472de787ed`. The lowering
was ported onto current shared source `e2ae679a`; only CMake insertion points
needed adjustment. A fresh Metal-enabled build with the option ON links.
The actual graph toggles between CONV_2D and the legacy path; 100 CPU and
100 Metal cases pass a scalar convolution oracle, including F16/F32, batch1/3,
kernel1/3/5, stride/dilation1/2 and large activations. Full build and per-case
logs are retained; four additional non-contiguous CPU fallback controls pass
with the option both ON and OFF. [Receipt](ggml-conv1d-pr2-review-2026-10-20.json).
These synthetic controls establish no AMD Vulkan execution,
Chatterbox speech quality or performance result. Incompatible-type and non-CPU layout
fallback, actual Radeon finite/magnitude/cancellation
controls and decoded audio remain acceptance gates. No ggml change is merged.
The reusable source is `tools/ci-heavy/conv1d_pr2_probe.cpp`; it deliberately
rejects a build where the option does not actually alter the graph.

## Issue549

Final token-boundary source `ebf69c437` passes CI39117921894 and
lint39117923785. The complete Linux unit log was independently inspected:
2,033 CTest cases and both real exported-getter metadata suites pass, including
Japanese slice/gap-fill and tied-frame ownership. The actual Japanese Metal Session short/long/short passes with 17/123/17
tokens and exact returned-short metadata. The independent direct native short
route matches every token byte, time and probability. The initial strict time
ordering guard incorrectly rejected a native separator-only token retaining an
earlier frame; it now preserves those times while rejecting content inversions
and missing owned tokens, with three controls. [Live receipt](ja-session-metadata-live-2026-10-20.json).
Generic CLI postprocessing, published wheels and downstream reporter acceptance
remain separate. For CPU use, the worker requires a CPU-only build: the direct
Parakeet wrapper follows its native default device.

## Reporter actions

- [Issue485](https://github.com/CrispStrobe/CrispASR/issues/485#issuecomment-6291292498)
  closed completed: original Index-Echo request shipped in v0.8.41. Rejected
  quantization and unmeasured hardware are not promoted by that closure.
- [Issue545](https://github.com/CrispStrobe/CrispASR/issues/545#issuecomment-6291304859)
  and [issue546](https://github.com/CrispStrobe/CrispASR/issues/546#issuecomment-6291305573)
  closed not planned under the current Fish revenue/hosting conditions.
- [Nyra456](https://github.com/CrispStrobe/CrispASR/issues/456#issuecomment-6291353434)
  remains deferred for separate research/output-license review.
- [PR521](https://github.com/CrispStrobe/CrispASR/pull/521#issuecomment-6291353763),
  [issue550](https://github.com/CrispStrobe/CrispASR/issues/550#issuecomment-6291354113)
  and [ggml PR2](https://github.com/CrispStrobe/ggml/pull/2#issuecomment-6291304497)
  received scoped evidence and remaining-gate updates.
- Issue483 still needs GTX1660/MX150 evidence; issue532 needs the reporter's
  failing recording/assertion/device details. No speculative fixes or duplicate
  requests were posted.

[Issue550 acceptance/closure](https://github.com/CrispStrobe/CrispASR/issues/550#issuecomment-6291711315)
and [issue549 live follow-up](https://github.com/CrispStrobe/CrispASR/issues/549#issuecomment-6291711992)
are posted. A hosted CPU/software-Vulkan worker for PR2 and bounded native
package dry runs are prepared for the frozen follow-up source.

No new release is published by this checkpoint. Release/package acceptance and
remaining model/hardware tests must be completed before advertising them.

## Complete public proof

[Immutable archive](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/ad638b4cd525c5633f831da9eb763864e9827ce9/issue-followups-20261020/complete-proof.zip)
contains 55 files, 420,650 bytes; SHA256
`e446e1a4b2a3c925033598677d3555558f35470d5957d058d0c4c37e9731c70e`.
The full anonymous download is byte-verified. Numerical/word outputs, complete
logs, hashes, negative controls and licensed fixture attribution are retained.
No model weights, runtime binaries or credentials are republished. Text path
prefixes/account identifiers use portable aliases; the manifest records both
original and public-file hashes. The successful worker also appends its terminal
progress event after hashing that file; its original hash covers the verified
prefix, and the public manifest covers the complete retained file.

[Three bounded library dry runs](followup-package-dry-runs-2026-10-20.json) are
dispatched on frozen native source `f0199f54f`: Linux39429480076,
Mac39429485161 and Windows39429489355. Their tag inputs are empty, so they
upload workflow artifacts and cannot publish a release. All three native build jobs pass; Linux and Mac logs also prove relocated
load and Session ABI resolution. Their original library bundles omit the
Session header; the follow-up below repairs that packaging defect. The later receipt/tooling changes do not modify native library
or wheel runtime sources; package claims still require reading actual results
and inspecting staged artifacts.

The hosted PR2 replay also pins the [supplemental applied patch](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/10385d27b50d21906fff6ee407ac8031c1a08e02/issue-followups-20261020/ggml-pr2-applied.patch),
3,146 bytes, SHA256 `7505375a1ef40726da05a0d52c14c07ef88b00128f1f2459cf1fcd6ece3951b2`.
The initial archive includes the proposal and its source hashes; this supplement
retains the exact CMake-adapted patch as well. Full anonymous readback and hash
checks passed before hosted dispatch.

The [hosted PR2 CPU/software-Vulkan review](https://github.com/CrispStrobe/CrispASR/actions/runs/39430148014)
stopped at configuration because the worker omitted `spirv-headers`. Its
[repaired replay](https://github.com/CrispStrobe/CrispASR/actions/runs/39438818205)
at source `306c17c65` passes 100 CPU, four CPU fallback and 100 software-Vulkan
cases. All11 output hashes and per-case logs are independently verified. The
Vulkan device is llvmpipe LLVM20.1.2 / Mesa25.2.8 on a CPU.
It is a read-only counterfactual in a separate ggml worktree, not a merge or
a physical Radeon performance/speech result.

## Installed-wheel and library header checks

The original frozen-source Mac wheel builds, installs into a separate venv,
loads from a foreign cwd, and resolves nine native Session-token/pointer-wrapper
symbols. The real Japanese Metal short/long/short test through that installed
package passes 17/123/17 tokens, exact returning-short metadata and direct-native
short comparison. The worker differs only by removing its repository `sys.path`
injection; both source hashes are retained. This is an unpublished test wheel
still carrying the development tree's 0.8.43 version, not a replacement for the
published release.

Actual staging found that all11 library bundle packagers omit
`include/crispasr_session.h`. The helper compiler rejects that omission and
passes when the header is present; the repaired helper also loads and resolves
all three wrappers. Current native pointer exports remain available, so missing
the optional helper alone does not establish a broken legacy class. Commit
`241b01455` ships both public headers consistently and adds a non-publishing
cross-platform artifact/wheel smoke workflow. It verifies bundle source pins,
header presence, clean installed-package loading and the actual ABI. Running it
against the original bundles is a deliberate negative control; fresh repaired
bundle runs pass on Linux x86_64, Mac arm64 and Windows x86_64. All three
[hosted installed-wheel jobs](https://github.com/CrispStrobe/CrispASR/actions/runs/39442789304)
also pass. Their receipts and wheel contents independently match the native
bundle bytes. The repaired Mac wheel additionally passes locally, including
three helper exports; its native library is byte-identical to the earlier
installed-wheel Japanese live test. The staging warning now correctly describes
its optional helper and native by-reference alternatives.

[Complete package/Vulkan proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/2376f0bd253731bff1639e57d6d1b6c29e8be79a/issue-followups-20261020/package-vulkan-proof.zip):
35 files, 221,589 bytes, SHA256
`51eb9e0cdb04fab690273dd8fa1700fd1480bc953593e6f93aa01667d2664960`.
Full anonymous byte readback passes. It retains native build logs/inventories,
all successful and unsuccessful hosted Vulkan output, the installed Mac wheel
live outputs, worker identities and the compiler control. Private local path
prefixes are replaced with portable aliases with both original/public hashes.
No model weights or native binaries are included.


The final [complete package acceptance archive](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/85bf50e0dfb93745e55441d4798269b6f71720f2/issue-followups-20261020/package-acceptance-proof.zip) retains all three
successful native builds, all three hosted installed-wheel receipts/logs,
independent library/wheel hashes, local repaired Mac checks, compiler controls,
software-Vulkan controls and fixture/license attribution. It contains
58 files, 445,477 bytes; SHA256 `d39af2c6eddd25d68189f2f44ca07103772d834af11791dfe082591291ec5f02`.
Every expanded public-file hash and credential scan passes, and anonymous full
byte readback is verified. No native binaries, wheels or weights are embedded.
The older archives remain immutable snapshots. General main-tip CI/lint remain
separate from these completed package jobs; no release or public wheel update
is published by this work.


## Final main CI and expanded fallback review

Main source `0c355f398400d7f2ef7fd838ddcb91b32dd9089b` passes all 13
[CI jobs](https://github.com/CrispStrobe/CrispASR/actions/runs/39445302976)
and all 10 [lint jobs](https://github.com/CrispStrobe/CrispASR/actions/runs/39445303065).
Each job was checked individually. The retained full Linux unit log confirms
2,039 CTest cases, 189 Python cases and 34 subtests passed. These are completed
frozen-source results; earlier pending checkpoints above are historical.

The unchanged PR2 candidate now passes 108 cases each on Mac CPU, Metal/CPU,
Linux CPU and software Vulkan/CPU. Each matrix retains the 100 ordinary/large-
activation cases and adds four strided-input plus four BF16-weight controls.
All 50 eligible direct convolutions execute on the requested backend. Metal and
software Vulkan each place six strided IM2COL operations on CPU, including
two BF16 cases. The outputs match the scalar oracle. The original direct-only
Metal harness aborted when it forced an operation Metal declares unsupported;
that harness control is retained. The corrected probe preallocates leaves and
uses the normal scheduler, with supported-operation/placement checks.
A buffered log previously interleaved backend stderr into one case record;
line-buffered output now passes strict per-row auditing.
[Expanded hosted run](https://github.com/CrispStrobe/CrispASR/actions/runs/39460690445)
and the machine receipt record exact source/probe/binary/log hashes.
This covers the tested types/layouts; actual Radeon precision and matched
Chatterbox decoded-speech/performance acceptance remain open. PR2 stays unmerged.

## Cache release already captured by the CUDA run

The pinned v3 worker calls public `free_aligner_cache()` before launching its
fresh CLI child. Independent reanalysis follows the one resident NVML device,
index 0: the short-after samples retain 2,735 MiB; the first CLI-phase sample is
125 MiB, followed by fresh CLI model residency peaking at 2,823 MiB. The observed
device-total drop is 2,610 MiB. The parent remains running while the child executes.
The second idle GPU is excluded from the comparison. Six tests reject ambiguous
or wrong-device/phase readings and missing fresh-model residency. The full
alignment audit still passes and reproduces exactly from the public data.
[Receipt](alignment-cache-release-2026-10-20.json).
This is sampled device-total evidence, not per-process allocator accounting,
exact idle return, free-call latency or a same-process reload test. No GPU
session or accepted long-input run was repeated.

[Complete CI/fallback/cache proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/1ed61568906ba43e509cc35f184dd50fb743277c/issue-followups-20261020/fallback-cache-proof.zip): 59 files,
599,965 bytes; SHA256 `0b3dd058be4cecb24c7683371562391622456f7b8e5279632080dc012b34d115`. Expanded hashes and
credential scans pass; full anonymous byte readback is verified. An extracted
archive replays the complete CUDA-data audit and all six telemetry guards
without model weights or native libraries. It includes fixture attribution,
project license, successful logs and earlier harness controls.

## Clean-build Python unit registration

Inspecting the complete Linux unit log for source `8237fd36c` exposed a test
registration gap: all 2,039 selected CTest cases pass, but the Python guard
block ran before `find_package(Python3)`, so it registered no tests in that
clean build. The separate 189-case/34-subtest pytest result also passes; it
does not establish execution of the missing cache telemetry guards.

Python discovery now precedes the first Python test command. The model-free
Japanese metadata validator is named `ja-metadata-validator`, keeping it out
of the `-E live` exclusion. CI reads the actual selected unit inventory and
requires that validator plus `alignment-cuda-loader`,
`alignment-cache-release-samples` and `kaggle-cli-launcher` before compiling.
The exact workflow guard rejects the original clean configuration and passes
a fresh fixed configuration. The fixed inventory adds 29 Python suites.
All 23 non-SDK suites pass through CTest locally, including the six cache
guards, three loader controls, launcher control and three Japanese metadata
controls. The six language SDK suites are separately gated and are not claimed
as executed by this local run. No native runtime or binding source changes.

To reproduce, configure a **new** build directory with
`-DCRISPASR_BUILD_TESTS=ON`, then inspect
`ctest --test-dir BUILD --show-only=json-v1 -L unit -E live`.
The four required names must be present. Run those suites with
`ctest --test-dir BUILD -R '^(alignment-cuda-loader|alignment-cache-release-samples|kaggle-cli-launcher|ja-metadata-validator)$' --no-tests=error --output-on-failure`.
The normal hosted unit tier will execute the wider inventory after building.
Local before/after inventory, configure logs and the 23-suite CTest log are
retained with the follow-up artifacts. Fresh hosted results remain pending.

The8237 macOS job was cancelled without acquiring a runner: its check
annotations explicitly report hosted macOS arm64 capacity constraints and no
job steps ran. This is not a failed source build or a passing macOS check.
