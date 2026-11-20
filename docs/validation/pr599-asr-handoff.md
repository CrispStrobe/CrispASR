# PR #599 ASR handoff

Checkpoint recorded 2026-11-18T17:46:30.680828+00:00.

## State and publication

- Branch: `fix/598-ssm-cuda-fallback`, remote `CrispStrobe/CrispASR`.
- PR: https://github.com/CrispStrobe/CrispASR/pull/599, draft/open; issue #598 remains open.
- Code snapshot: `1a3d7f63dc187496774c44cdff4069d74b33b418`. This handoff commit adds documentation; its exact SHA is recorded in the PR checkpoint description.
- Previously published head: `82d9daa14823db8022b5a8a3c62049d4b99a8926`; retained terminal CI result: **63 PASS / 19 FAIL / 7 SKIP**, no pending checks. These results belong to that older head.
- Final instruction authorizes a normal, non-force checkpoint push. Earlier local-only publication holds in the dated PLAN entries are historical. `[skip ci]` requests no new automatic runs for this checkpoint; exact-head CI remains unrun and required before merge.
- No merge, model/performance/provider run, native backend execution, new SDK probe, or broad experiment is part of this checkpoint.

## Completed ASR work

- PR #590 exact head `534e4703c3a5eb3a26023794433d9413babfc5e6`: 56 passing / 4 skipped checks, then normal merge `3b3dae4817b4ef0dded9dde8f4713178833de8d1`. #589 closed with [merge evidence](https://github.com/CrispStrobe/CrispASR/issues/589#issuecomment-6969737518). #592 and #593 remain closed.
- #598: ggml pin `f906ba0b467a9e2043d76e15f4d6d5f0ea9429fc` restricts CUDA SSM convolution capability to implemented K=3/4/5/9/15 with channels divisible by128. K=8/16/31 falls back through the scheduler to CPU. No new CUDA kernel or model math.
- Regression: production predicate exercised through real ggml scheduling with CPU-backed emulated CUDA capability and scalar oracle; optional CUDA operator/session tests require actual hardware/model availability.

## Commits since the previously published head

| Commit | Change |
|---|---|
| `7f875932a2aec6307b999a46dc2209da5e1819e2` | Declare CLI environment helpers for MinGW Clang |
| `63c7adb5185acbe57c095ab4e7720373f6c0846b` | Use platform C char types for Rust metadata buffers |
| `99030320082e82d3032640dd55fce683310084a2` | Supply NumPy to legacy CI unit-test interpreters |
| `6495d853c044a852226b988233e661cbc7d4f388` | Prepare low-perf CI baseline for observed AVX512 host mismatch |
| `ccd641b6aeae6dae057ba0f4a0e0b843c0841291` | Record independently scoped CI repairs and unqualified ISA candidate |
| `d5912e426c17f2a7d6b0a38529a6f1c45b2e1010` | ci: preserve command failures and stop downstream phases |
| `5fc17ea8356c4fd7bb8d01f4e9c6284e3f02fb1b` | ci: supply NumPy to compiler unit jobs |
| `0425154d26c2ff634a77be2cd7640210327a74d5` | arktts: release backends when initialization fails |
| `1a3d7f63dc187496774c44cdff4069d74b33b418` | tests: preload matching ASan runtime before Python ctypes |

The x64 low-perf AVX512 candidate is limited to that CI invocation. Its actual log reports compiled AVX512F, host missing AVX512F and SIGILL. Product ISA defaults/guards, supported features, performance commands/baselines, crashes and assertions remain intact. This candidate has no runtime/performance qualification and does not establish resolution of all49 failed tests.

## Retained actual checks and limits

| Scope | Actual evidence | Unrun / limitation |
|---|---|---|
| SSM scheduler/CPU fallback | 132/132 cases:40 emulated preferred-device,92 CPU fallback; scalar absolute-error criterion1e-5; old alignment-only predicate fails | Retained CPU ggml libraries; no exact-current full build or CUDA execution |
| CUDA source contract | Two source checks pass; CUDA/live C++ syntax passes | No actual CUDA runtime, xASR GPU inference or reporter Windows/GTX1070Ti proof |
| Rust C-char fix |11 E0308 errors before; unsigned-char ARM target metadata check passes after, DOCS_RS=1/offline | Installed aarch64-linux-android metadata only; no native link, ARM Linux runtime or model run |
| CLI portability | Single-TU syntax/format checks pass | Hosted MinGW result on new repairs unrun |
| Pipeline failure propagation | Frozen14/14 pure mocks pass; old baseline10 failures/2 errors; Bash syntax and whitespace pass | Fake tools only; no real compile, benchmark, model or download; unchanged controls not rerun |
| ArkTTS cleanup | Single-TU syntax passes; old ASan reports832 bytes/8 allocations in null/empty-path init test | New cleanup has no ASan runtime proof |
| ASan Python launch |2 test methods pass:5 isolated mock CMake configurations and3 mocked quantizer preload environments; old nonempty preload chains fail2 controls | No sanitizer/native execution; real ASan/TSan/runtime outcome unrun |
| Privacy before earlier local work |5/5 unskipped retained | Final staged checkpoint privacy result is recorded in the PR checkpoint description |

Source files and helper code are not covered by the historical published63 passing checks. New exact-head workflow results must be inspected independently. No unchanged control rerun is used to imply broader validation.

## All19 retained failed jobs

Primary causes remain distinct from downstream missing binaries and aggregate CTest failure. Every row is still a failed published check. Links identify exact existing jobs; no logs need refetching for these findings.

| Failed job | Exact job evidence | Cause / remaining scope |
|---|---|---|
| linux-unit | [134607521208](https://github.com/CrispStrobe/CrispASR/actions/runs/44246740104/job/134607521208) | SDK1.30.0 download HTTP404 before compilation |
| ubuntu-22-gcc (Debug) | [134607560260](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560260) | 20 failed unit tests; missing NumPy demonstrated in exact terminal job log |
| ubuntu-22-gcc (Release) | [134607560131](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560131) | NumPy absent;20 unit failures |
| ubuntu-22-gcc-arm64 (Debug) | [134607560299](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560299) | NumPy absent; 11 Rust c_char type errors |
| ubuntu-22-gcc-arm64 (Release) | [134607559661](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607559661) | NumPy absent; 11 Rust c_char type errors |
| ubuntu-22-clang (Debug, amd64) | [134607560240](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560240) | 20 failed unit tests; missing NumPy demonstrated in exact terminal job log |
| ubuntu-22-clang (Debug, arm64) | [134607560402](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560402) | NumPy absent; 11 Rust c_char type errors |
| ubuntu-22-clang (Release, amd64) | [134607560141](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560141) | NumPy absent;20 unit failures |
| ubuntu-22-clang (Release, arm64) | [134607560451](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560451) | NumPy absent;11 Rust C-char errors; Conformer exact-bit19.30389404296875% !=100.1048576 successful assertions establish finite outputs only; numerical cause unknown. |
| ubuntu-22-gcc-sanitized (ADDRESS, linux/amd64) | [134607560339](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560339) | NumPy absent; LeakSanitizer 832 bytes in 8 allocations in arktts null/empty path test; CPU backend init stacks; ASan runtime not first in initial library list in dynamically loaded Python/native tests; ArkTTS ownership cleanup and scoped ASan Python launch repairs committed; runtime validation unrun. |
| ubuntu-22-gcc-sanitized (THREAD, linux/amd64) | [134607560102](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560102) | NumPy absent; libtsan.so.0 cannot allocate static TLS block in dynamic Python/native probes |
| ubuntu-22-gcc-sanitized (UNDEFINED, linux/amd64) | [134607560191](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560191) | NumPy absent; 20 unit failures |
| windows-msys2 (CLANG64, clang-x86_64, Release) | [134607560133](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560133) | CLI getenv/atoi declarations missing |
| vad | [134607560307](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560307) | test-vad segfault after Silero6.2 model load |
| ggml-ci-x64-cpu-low-perf | [134607560295](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560295) | compiled AVX512F vs host missing AVX512F/SIGILL; NumPy absent |
| ggml-ci-arm64-cpu-low-perf | [134607560398](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560398) | NumPy absent; 11 Rust c_char type errors |
| ggml-ci-x64-cpu-high-perf | [134607560162](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560162) | NumPy absent;20 unit failures |
| ggml-ci-arm64-cpu-high-perf | [134607560350](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560350) | NumPy absent; 11 Rust c_char type errors |
| ggml-ci-arm64-cpu-high-perf-sve | [134607560230](https://github.com/CrispStrobe/CrispASR/actions/runs/44246739793/job/134607560230) | NumPy absent; 11 Rust c_char type errors |

Legacy ggml Debug failures skip Release, then the old unconditional benchmark encounters a missing binary and masks exit status. The local pipeline repair captures producer/logger status and stops subsequent phases. It preserves successful performance commands and baselines and fixes no primary unit-test failure by itself.

## Exact remaining blockers

1. **External SDK byte gap.** Pinned official ref `v1.30.0`, asset `onnxruntime-linux-x64-1.30.0.tgz`, [official asset route](https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz). Expected SHA256: `a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd`. Retained route response is HTTP404,9-byte Not Found,0 redirects. Error-body SHA256 `0019dfc4b32d63c1392aa264aed2253c1e0c2fb09216f8e2cc269bbfb8bb49b5` is not archive evidence. No verified exact archive bytes obtained. Related ref/release routes also404 in existing evidence; permanent asset deletion is not proven. Downloader, version, checksum and validation unchanged. No unchecked mirror or unchanged retry.
2. **VAD fault site unknown.** Retained Silero6.2.0 job reaches model-size reporting, then SegFault before context compute-buffer success logging. Source interval includes backend setup, state tensor/buffer allocation and scheduler graph initialization. No backtrace or exact faulting instruction supports a safe repair.
3. **TSan startup.** Retained Python child trace reaches `ctypes.CDLL` -> `_dlopen`: `libtsan.so.0: cannot allocate memory in static TLS block`. No safe startup remedy is proven. Scoped ASan preload repair leaves TSan unchanged; no leak/assertion/runtime suppression.
4. **Conformer numerical cause unknown.** Clang ARM Release exact-bit parity fails before timing. The1048576 passing assertions check finiteness only. Baseline/accelerated loop layout differs, but no compiler contraction, reassociation or ISA culprit is established.100% equality and zero-max-difference gates unchanged.
5. **Runtime and exact-head proof outstanding.** ASan preload/ArkTTS cleanup, CI prerequisite/portability/ISA candidates and CUDA fallback require appropriate actual exact-head checks. No CUDA runtime success is claimed. New local source has no green hosted CI evidence.

## Handoff steps

1. Preserve this branch and existing fixes. Check the PR head against this checkpoint commit; keep draft/#598 open.
2. Obtain only verified expected SDK1.30.0 bytes through an authorized route and retain hash/availability proof before expecting linux-unit to compile. Existing GitHub source restriction is CrispStrobe-only; further external SDK/ref probes require changed authorization.
3. Under compatible execution authorization, inspect new normal exact-head CI outcomes. Do not reuse82d9daa results as current-head proof, retry unchanged failures, dispatch ci:full, run new model matrices or waive actual failures.
4. When authorized, validate existing ArkTTS null/empty-path regression under ASan and the affected Python launch tests on the actual sanitizer runtime. Use new retained fault stacks or explicitly authorized bounded diagnostics for VAD, TSan and Conformer; keep gates unchanged.
5. CUDA/operator/xASR tests run only with real hardware and supplied model. Record actual execution separately from emulated scheduling/CPU proof.
6. Before any build, require fresh user.slice memory.current <=3.8GiB, project at least1GiB headroom under5GiB, and at least12GiB fast-disk free for heavy builds. No heavy build while other owners are active; host MemAvailable is insufficient.
7. Merge normally only after final exact-head required checks are green and scope evidence is complete; close #598 with actual merge/evidence. No force push or supervisor merge.

The current owner stops after checkpoint publication. No duplicate owner or watcher was created.
