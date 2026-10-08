# Issue and PR triage — 2026-10-07

Pulled `main` at `3f4ca9372`. The review snapshot contains 11 open issues and
4 open PRs, with descriptions, all issue comments, PR reviews and inline
comments archived in `/mnt/storage/crispasr/triage-20261007/`.
The [receipt](issue-pr-triage-2026-10-07.json) distinguishes offline checks,
previous PR CI and queued fresh hosted checks. No release or new GPU
performance claim is made here.

## Integrated source changes

- **[#491](https://github.com/CrispStrobe/CrispASR/issues/491): Orukeet short-name
  routing.** The registry already contained the correct Q4 model. The filename
  pass did not recognize `orukeet`, so the CLI chose the legacy Whisper path
  before reaching the registry. Orukeet names now select the Parakeet runtime.
  This also covers the cached filename, case variants and Windows paths.
  An actual C++ factory probe passes 14 cases; the old source fails the six
  Orukeet cases. Directory-name controls preserve existing detection. The
  graph, weights and download/licence policy are unchanged.
- **[PR #496](https://github.com/CrispStrobe/CrispASR/pull/496): browser note
  transcription.** Retains the author's `sessionPianoNotes` and
  `sessionPianoSampleRate` Embind functions. Added API documentation, export
  checks in all five WASM builds, and a real Node/Embind no-session smoke check
  in the single-thread build. Existing PR CI compiled all five variants; the
  newly added smoke check passes on main. This does not claim model/audio
  acceptance in a browser.
- **[PR #514](https://github.com/CrispStrobe/CrispASR/pull/514): Hikari CUDA
  benchmark.** Retains the author's kernel and model pin. Corrected success
  reporting: download/benchmark exceptions, nonzero inference exits, missing
  timing, empty or wrong short-clip output now fail the kernel. Long input must
  produce proportionally more words. Read the actual transcript file, discard
  stale files before each invocation, and save structured results with source
  and model revisions. Offline fault injection passes eight cases, including
  a good control and stale-output rejection. No fresh Kaggle run was launched;
  this validates failure handling, not CUDA speed or new long-clip parity.
- **CI package setup.** PR #515's failed Moonshine-streaming/Canary/Kokoro/
  Moonshine-base regression jobs exhausted their 45-minute limit while fetching
  Ubuntu indexes/packages from the Azure mirror, before compilation or inference.
  The retained Moonshine-streaming and Canary logs show the failure. Native CI
  and regression now use `tools/ci-apt.sh`: replace that failed mirror URL with
  the official Ubuntu HTTPS URL and bound network timeouts/retries. Ports URLs
  remain untouched. Offline checks verify URL substitution, argument forwarding
  and propagation of apt's failure exit code. Native CI and regression pass on integrated main.

For unattended first download, use:

```sh
crispasr -m orukeet --auto-download --accept-license cc-by-sa-4.0 -f audio.wav
# Once cached, the same short name works without a download flag:
crispasr -m orukeet -f audio.wav
```

The first download also requires explicit CC-BY-SA licence acceptance. This
policy is unchanged; CC-BY-SA is not a noncommercial licence. Without
`--auto-download`, an accepted uncached named model retains the normal
interactive download prompt; noninteractive callers must opt in. The documented
manual-path workaround also continues to work.

## Hosted results and 2026-10-08 follow-up

On integrated main `d754aa10a`, [native CI](https://github.com/CrispStrobe/CrispASR/actions/runs/37704656972),
[regression](https://github.com/CrispStrobe/CrispASR/actions/runs/37704657102),
[lint](https://github.com/CrispStrobe/CrispASR/actions/runs/37704656959) and
[all five WASM builds](https://github.com/CrispStrobe/CrispASR/actions/runs/37704657094)
passed. The single-thread job emits `WASM piano bindings: 3 assertions passed`.
The original branch native CI `37704036313`, lint `37704039045` and WASM
`37703785629` also passed.

[Lint Deep](https://github.com/CrispStrobe/CrispASR/actions/runs/37704657127)
found uninitialized Hikari token probability in the C ABI. The follow-up sets
it to the existing unavailable-confidence convention, 1.0; Hikari exposes
emission times but no token probability API.

[Orukeet acceptance](https://github.com/CrispStrobe/CrispASR/actions/runs/37703780788)
built successfully and passed 5,988 registry assertions, then correctly refused
the unaccepted CC-BY-SA licence. The test command had omitted the acceptance
flag. It did not reach model inference. The CLI additionally mislabeled every
licence requiring acceptance as noncommercial; corrected notices distinguish
CC-BY-NC from CC-BY-SA/custom terms while preserving the acceptance policy.
Four actual resolver cases pass (CC-BY-SA/CC-BY-NC, cached/uncached), and the
old source reproduces the incorrect CC-BY-SA notice.

[Revised Orukeet CPU acceptance](https://github.com/CrispStrobe/CrispASR/actions/runs/37728843835)
passed at `4b45fec39ff603ad68bad0435c733e41695bfe10`. Fresh download,
cached short name, filename, explicit path and anonymous C ABI all return
identical JFK text. The pinned Q4 model is revision
`2a93474c2771a6ca2228a7e001bd06ce0c97f880`, SHA-256
`769f346e3960de76afabcb82670b9d0d78c91051bc030be82e77eac45a7075be`.
Released v0.8.41 reproduces the original failure. Missing licence acceptance
refuses without downloading. This proves routing and speech acceptance across
those surfaces, not new numerical parity or speed.

PR #515 validation branch `f43dfd5eb5f9915e2c54b42da1ce5fa3fa00318d`
preserves author commits. Review fixed missing/wrong companions on named-model
downloads, failure propagation and cached repair. Offline actual CLI/resolver
probes pass six isolated bundles. Hosted CPU `37730765885` built successfully,
loaded all six deployments, and passed five German choices plus Qwen streaming.
It found Small int8 final punctuation drift versus batch and Nemotron C ABI batch
language-tag leakage. Neither failure gate was relaxed. Final ONNX flush now
recomputes whole-utterance encoder output because dynamic int8 scales depend on
window shape; Nemotron cleanup is shared with the CLI and also filters C ABI
word output.

Added an optional independent Python ONNX driver and `crispasr-diff` dispatch:
exact PCM and shapes, cosine >= 0.99999, relative L2 <= 1e-4 and final text via
Python's standard tokenizer. A deliberately scaled reference must fail. This
measures native wrapper/graph execution parity; it does not certify the exporter
against original source checkpoints. C++ runtime syntax, formatting and Python
syntax pass locally; regression-driver smoke passed 56 tests.

[New CPU acceptance](https://github.com/CrispStrobe/CrispASR/actions/runs/37733187037)
is running at `91496680a` with identical runtime plus automatic ONNX/stream CI
and speech-content checks for every deployment. ARM64 run
[37734757849](https://github.com/CrispStrobe/CrispASR/actions/runs/37734757849)
is queued at `eb2ebe30a` with the same runtime and strict gates. Earlier
[native CI](https://github.com/CrispStrobe/CrispASR/actions/runs/37730684549)
passes all 13 jobs;
[numerical regression](https://github.com/CrispStrobe/CrispASR/actions/runs/37730687001)
are pending on the earlier runtime source. #515 remains unmerged. #491 is closed
with final acceptance linked.

The earlier native CI dispatch `37703783152` was superseded by the apt-fixed
source dispatch. Original proof source and queued-run commits remain reachable
through `archive/triage-source-20261007`. Author commits for #496 and #514 remain
ancestors of the integration. The rebase changed integration hashes; it did not
change their implementation.

## Remaining issue and PR work

| Thread | Finding and next action |
|---|---|
| [#490](https://github.com/CrispStrobe/CrispASR/issues/490) Arabic character alignment | Feasible. `src/align.cpp` already walks UTF-8 codepoints and traces per-label Viterbi positions, then collapses them to word times. Preserve those spans through the aligner and JSON surface. Test repeated letters, Arabic codepoints/diacritics, blanks, punctuation and OOV handling; do not estimate character times by dividing word durations. Implemented in draft [PR #516](https://github.com/CrispStrobe/CrispASR/pull/516): 41 known-path assertions and ASAN/UBSAN pass; C# builds cleanly. First hosted Arabic Q4 output has valid JSON (15 words / 85 measured characters), but a UTF-8 diagnostic preview crashed strict log reading before reference/binding checks. Corrected combined-source acceptance `37751687697` is queued. Still unmerged. |
| [PR #492](https://github.com/CrispStrobe/CrispASR/pull/492) MiMo/CANN performance | Existing CI is green and the contributor reports five exact transcripts on Ascend. Shared attention/mel code and device placement deserve stage/magnitude and decoded-output validation on default and non-flash paths, F16 and shipped quant, with the current ggml pin. The cited timing combines this PR with the still-open [ggml #4](https://github.com/CrispStrobe/ggml/pull/4); it cannot be attributed to this PR alone. Retained unmerged. |
| [PR #515](https://github.com/CrispStrobe/CrispASR/pull/515) streaming bindings / German ONNX Moonshine | Integrated with author ancestry preserved after final x86 `37744837589` and ARM `37744841616` each pass all eight cases, graph parity and cap/stream checks. Final main-tip CI remains a release gate. |
| [#483](https://github.com/CrispStrobe/CrispASR/issues/483) CUDA 12.6 | Reporter confirms the mismatch warning is gone. Latest follow-up requests an experimental forced-MMQ/no-tensor build for GTX1660 comparisons. Both matched Windows packages pass, and their actual uploaded manifests pair locally. Packages and proof are published; reporter GPU model/output/timing comparisons remain pending. No forced-MMQ default change. |
| [#488](https://github.com/CrispStrobe/CrispASR/issues/488) Qwen3 hotwords | Fixed and validated on main. Reporter Windows/Vulkan six-clip retest remains external. |
| [#485](https://github.com/CrispStrobe/CrispASR/issues/485) Index-Echo | Both sizes shipped; 9B Q4 candidate acceptance remains open in PLAN, with transfer quota constraints. |
| [#484](https://github.com/CrispStrobe/CrispASR/issues/484) Intel Mac regression | Correct SIMD packaging shipped in v0.8.40/41. Reporter Russian-recording hardware retest remains external. |
| [#482](https://github.com/CrispStrobe/CrispASR/issues/482) AudioSeal graph capacity | Fixed and tested on CPU/T4. Exact reporter RTX5060 Ti/CUDA13 retest remains external. |
| [#481](https://github.com/CrispStrobe/CrispASR/issues/481) Phonon-2 | Native port shipped. The upstream 164 MB artifact and headline timing remain distinct from native GGUF sizes/performance. |
| [#478](https://github.com/CrispStrobe/CrispASR/issues/478), [#461](https://github.com/CrispStrobe/CrispASR/issues/461) VoxCPM2 | Existing prefill and mixed-head fixes shipped. Windows/Khmer remains external. The 2026-10-08 B390 retest confirms active batched prefill, but RTF is 1.019, VAE setup is slower, and eight-step audio quality is poorer than ten. Earlier RTF estimates are not accepted measurements; no newly proven Vulkan optimization. |
| [#456](https://github.com/CrispStrobe/CrispASR/issues/456) nyra | Explicitly deferred in PLAN; model/output licence restrictions and GMM-HMM integration remain. |

## #483 experimental package follow-up — 2026-10-08

The full issue includes the reporter confirming that matched CUDA 12.6 removed
the warning, then requesting a forced-MMQ build for GTX1660 tests. Branch
`04ded753e909da019a2cd391f651f51f5299b6f5` prepares matched ON/OFF packages
with CUDA 12.6.3, identical PTX 61/80 targets, source, CPU floor and runtime.
[Hosted package/pair checks](https://github.com/CrispStrobe/CrispASR/actions/runs/37733919241)
have started both Windows jobs. Compile definitions, staged runtime version and DLL hashes are
checked, followed by packaged CLI driverless startup and a pair comparison.
Actionlint passes. No release defaults change; no GTX1660 speed/output verdict
exists until the reporter exercises the actual GPU. The packaging workflow is now merged; the original tested source is retained in `archive/issue483-mmq-source-20261008`.

Final-source native CI [37735374748](https://github.com/CrispStrobe/CrispASR/actions/runs/37735374748)
and lint [37735377165](https://github.com/CrispStrobe/CrispASR/actions/runs/37735377165)
are queued at `eb2ebe30ab42378a0b16ecb302b0c266145e9f7c`.

## Strict harness and binding completion — 2026-10-08

Strict CPU run [37733187037](https://github.com/CrispStrobe/CrispASR/actions/runs/37733187037)
found matching Python/native graph values and text, but failed the shape gate:
`Ref::shape` removes singleton axes after dimension zero. Latest branch
`a1a3d4a25` applies the same normalization, retaining cosine >= 0.99999 and
relative L2 <= 1e-4. This run separately proves Nemotron batch tag cleanup and
packet-size-independent streaming, legacy Moonshine deployments and Qwen.
ONNX streaming acceptance was blocked by the earlier shape check and remains
pending. No success is inferred for tests that did not execute.

The new streaming-kind query is now exposed in all seven wrappers, WASM/JS and
the WebSocket ready event. C# compiles with no warnings/errors; WebSocket source
syntax passes. Rust format checking reports pre-existing formatting elsewhere,
so those unrelated lines were left intact. Fresh x86 `37735853879`, ARM
`37735856512`, native CI `37735908557`, lint `37735910817`, WASM `37735912990`,
Go `37735851685` and Rust `37735851828` validate the latest branch. Replaced
queued jobs were cancelled; #515 remains unmerged.

## Hikari deep-lint completion follow-up — 2026-10-08

[Deep lint 37729063238](https://github.com/CrispStrobe/CrispASR/actions/runs/37729063238)
finished with `virtualCallInConstructor` in the Hikari realtime adapter. The
probability initialization fix remains intact. Source `f55c7bc8c` replaces the
constructor's virtual call with a private reset helper shared by `reset()`.
Focused cppcheck 2.17.1 and C++17 syntax pass; full pinned cppcheck 2.7
[37737578258](https://github.com/CrispStrobe/CrispASR/actions/runs/37737578258)
is running. No inference behavior or performance default changes.

## ARM acceptance and MMQ packages — 2026-10-08

ARM64 acceptance `37735856512` passes all eight cases on `a1a3d4a25`: six German
deployments plus Nemotron and Qwen. All four five-graph variants pass 15 stages
at reported cosine 1.000000 / relative L2 0, matching Python text. Small int8
final/batch parity passes; the scale-negative control rejects its encoder tensor.
Replacement x86 `37738992713` uses Ubuntu 22.04 at `68b849eb9` after the older
latest-image run waited over 30 minutes without a runner. The runtime is
unchanged; the negative control now preserves text metadata, so only scaling
may fail. Rust checks pass; other native/Go/WASM/lint checks remain in flight.

Both Windows MMQ packages pass and their uploaded manifests pair locally.
[Downloads, pinned source and scope](cuda126-mmq-experiment-2026-10-08.md).
The final hosted pairing job also passes; the complete workflow is green. #483 remains open for actual GPU tests.


## x86 frontend rounding and corrected final flush — 2026-10-08

The ARM pass above was insufficient to establish x86 final/batch identity.
A narrow local x86 replay of Small int8 on the same pinned German clip still
failed after encoder-only recomputation: batch starts “Guten Morgen! Die”,
while streaming final starts “Guten Morgen, die”. Whole-utterance versus
40 ms frontend processing differs by maximum 4.76837e-7 (relative L2
3.10646e-7), enough to alter this greedy int8 decode.

Runtime `cdd1b80d0df90279e26489b94fa20c8a73abb2ca` retains original PCM and
replays the actual batch frontend, encoder and decoder on final flush. Partial
updates retain incremental processing. PCM history adds 64 kB per second.
The saved pre-fix binary fails; the fixed runtime passes all four combinations
of 1/4 threads and 1777/5120-sample packets, including exact final/batch text,
repeat flush and feed-after-flush rejection. The feature branch contains
`docs/moonshine-onnx-x86-flush-2026-10-08.json` with pins, hashes and scope.

Fresh full CLI/C ABI acceptance at that source is queued on
[x86 Ubuntu 22.04](https://github.com/CrispStrobe/CrispASR/actions/runs/37740556768)
and [ARM64](https://github.com/CrispStrobe/CrispASR/actions/runs/37740559460).
The older queued x86 run `37738992713` was cancelled because its runtime still
had this defect. #515 remains unmerged. Go, Rust, all five WASM jobs and the
Windows CUDA13 check pass; native CI has ten of 13 jobs passing and three in
progress. Deep lint rerun `37737578258` is running. No release tag is cut.


## Explicit generation-cap wiring — 2026-10-08

The final contributing checklist found ignored ONNX generation caps. Source
`c46e44e82` forwards explicit CLI/session limits to both legacy and five-graph
decoders, including persistent streams and retained draft prefixes. Clearing
the session override restores its existing duration-derived default. Local
Small int8 x86 at 1/4 threads passes one-token “Guten” output, capped final/batch
equality and full-text reset; all four normal packet cases remain exact.

Hosted acceptance now checks CLI/C ABI capped output, persistent capped streams
and reset on all five ONNX deployments. Replacement
[x86 37741562051](https://github.com/CrispStrobe/CrispASR/actions/runs/37741562051)
and [ARM64 37741564942](https://github.com/CrispStrobe/CrispASR/actions/runs/37741564942)
are queued; prior flush-only runs were superseded before starting. Native CI
`37735908557` passes all 13 jobs on the earlier binding-completion source.
This does not replace optional-SDK acceptance for the newest runtime.

Comprehensive next-release notes are drafted in `RELEASE_NOTES_v0.8.42.md`.
They cover landed changes since v0.8.41 and exclude #515/#492 from shipped
features while those PRs remain unmerged. Version/tag remain unchanged.

## Final validation checkpoint — 2026-10-08

Lint `37735910817` passes all 10 jobs; native CI `37735908557` passes all 13
on binding-completion source `a1a3d4a25`. Corrected optional-ONNX runtime
`c46e44e82` is running full x86 `37741562051` and ARM `37741564942` acceptance.
MMQ `37733919241` passes both Windows packages and the final hosted pair job.
Draft #516 adds measured character output; real Arabic-audio proof remains queued.

## Generic ONNX path routing — 2026-10-08

Both final acceptance runs at `c46e44e82` passed all four deployed graph diffs
and the three native speech cases, then failed the generic-path capped CLI
checks: `encoder*.onnx` was incorrectly sent to Whisper. C ABI detection was
already correct. Source `a9c312cb1` reuses its config detector in the CLI.
The saved baseline reproduces the failure; four positive/negative factory
checks pass. [Local receipt](moonshine-onnx-routing-2026-10-08.json). Fresh full
x86 `37744837589` and ARM `37744841616` runs are queued. Neither failed run
proves the later ONNX cap/flush checks, which were not reached.

## Character-binding validation follow-up — 2026-10-08

PR #516 is now `d57b0f7c7` with the same core alignment implementation.
Go `37743605844`, Rust `37743605797` and Linux C# in `37743740252` pass at
`5a8471309`. Dart CI built the native library and exposed a format failure;
the actual Dart 3.13.5 formatter corrects it, all nine lib/test files pass,
and static analysis reports no issues. Java wrapper/driver compilation emits
15 class files locally. Fresh Arabic Q4 acceptance `37746296873` adds exact
Java/JNA text/times/offset comparisons; runtime proof remains queued. The
workflow now covers future C ABI/binding/runtime/test edits. Unicode fixtures
compile explicitly as UTF-8 under MSVC. Final ONNX ARM acceptance is running.

## Corrected final ARM acceptance — 2026-10-08

[37744841616](https://github.com/CrispStrobe/CrispASR/actions/runs/37744841616)
passes all eight cases at `a9c312cb1`, including direct ONNX CLI path routing,
one-token caps and reset for five ONNX choices, capped persistent final output,
both packet sizes, and native Nemotron/Qwen speech. Four deployed graph variants
each pass 15 stages (reported cosine 1.000000, relative L2 0) and exact decoded
text; scaling fails exactly one numerical check while text still passes.
[Receipt](moonshine-onnx-arm-2026-10-08.json). Prepared integration `c0f10937b`
adds main documentation only and preserves original author ancestry. Corrected
x86 acceptance `37744837589` remains queued; #515 is still unmerged.

## Hikari deep lint accepted — 2026-10-08

Full pinned cppcheck 2.7 run [37737578258](https://github.com/CrispStrobe/CrispASR/actions/runs/37737578258)
passes at `f55c7bc8c`, covering the probability initialization and nonvirtual
constructor-reset fixes. Corrected final ONNX x86 acceptance is now running.

## PR #515 accepted and integrated — 2026-10-08

Final x86 `37744837589` passes the same complete eight-case suite as ARM
`37744841616`, including all five ONNX caps/reset, capped persistent final
output and both packet sizes. Four graph variants each pass 15 stages at
reported cosine 1.000000 / relative L2 0 plus exact text; the scale-negative
control rejects only the scaled tensor. [X86 receipt](moonshine-onnx-x86-2026-10-08.json).
Integration preserves original author commit `2bf39c58a` and the validated runtime
`a9c312cb1`; newer main changes are documentation only. #515 is integrated.
Hikari deep lint also passes. #516 real Arabic acceptance and final main-tip
checks remain pending; no release/version tag has been created.

## Character alignment integration and UTF-8 fixes — 2026-10-08

PR #516 is rebased onto merged #515, latest `8c373c4a9`. Both CMake test
targets survive the append conflict. Integrated Dart formatting (11 files),
Java compilation, C# build (zero warnings/errors) and Python parsing pass.
Arabic Q4 run [37746296873](https://github.com/CrispStrobe/CrispASR/actions/runs/37746296873)
built, passed 41 new / 91 existing assertions, and produced valid word JSON
with 15 words and 85 measured characters. Its diagnostic preview split an
Arabic codepoint and crashed strict log decoding; reference and binding runtime
checks were not reached. The CLI now uses the existing UTF-8 prefix helper.
Strict log decoding remains enabled. A separate saved JNA guard fails under
US-ASCII before the explicit UTF-8 binding option and passes afterward.
[37751687697](https://github.com/CrispStrobe/CrispASR/actions/runs/37751687697)
reruns real Arabic CLI/reference/Python/Java acceptance with that host encoding
forced. #516 remains unmerged pending full acceptance.

Cold-file migration recovered 1.43 GB on root and 0.53 GB on `/mnt/volume1`.
Each source was unused for over five hours and had no open descriptor; SHA-256
was verified before atomically replacing its path with a symlink. Native
executables/shared libraries were retained. Receipts live under
`/mnt/storage/crispasr/storage-cleanup-20261008/`.

## PR #492 numerical acceptance started — 2026-10-08

Author-preserving candidate `fae7e2241` merges #492 onto current main with the
same ggml pin. Shared mel projection passes 16 local A/B cases byte-for-byte,
including the 64-frame branch boundary, both filterbank layouts, float/double
accumulators and 1/4 threads. A frozen independent Python LM archive was found
and checksum-verified; it contains five LM stages and explicitly skips generation.

[Q4 acceptance](https://github.com/CrispStrobe/CrispASR/actions/runs/37754319857)
is queued; [F16 acceptance](https://github.com/CrispStrobe/CrispASR/actions/runs/37754323202)
waits in the script's serial concurrency group. Both check baseline/default,
candidate/default and candidate/non-flash LM values with norms/relative L2,
tokenizer stages/RVQ codes, and English/Chinese CLI/session decoded text. F16
means the LM; the codec is the same shipped Q4_K in both jobs. CPU results do
not establish CANN/CUDA correctness or the PR's combined NZ-weight speedup.
#492 is still unmerged. #516 has progressed to the Java input fix below.

## Arabic reference accepted; Java input encoding fixed — 2026-10-08

[Arabic Q4 run 37751687697](https://github.com/CrispStrobe/CrispASR/actions/runs/37751687697)
passes CLI word/segment JSON, exact independent full-sequence Viterbi for all
85 measured characters, Python/C ABI offsets and invalid accessors, then fails
at Java alignment. JNA 5.13's scalar String arguments still use the global
encoding despite the per-library UTF-8 option. The earlier local guard checked
only returned text; it missed incoming Arabic being replaced by question marks.

A strict native guard now validates input UTF-8 bytes, a Unicode model path,
returned Arabic characters and centisecond offsets. It rejects the old wrapper
and passes after explicit per-library input conversion under US-ASCII,
ISO-8859-1 and UTF-8 defaults. The global JNA setting is not changed. This guard
runs before the expensive work. [Fresh acceptance 37756933124](https://github.com/CrispStrobe/CrispASR/actions/runs/37756933124)
at `d9fcfcc90` is queued; #516 remains draft and unmerged.

Metal cache test source `c849ba214` repairs the ggml device API call, direct
Metal linkage and Objective-C++17. A compiler negative control rejects the old
call and accepts the fixed one. [Shared/static macOS coverage](https://github.com/CrispStrobe/CrispASR/actions/runs/37755267314)
is queued; no-device runtime checks explicitly skip, and no PSO serialization
or GPU performance proof is claimed.

## MiMo C ABI settings and F16 memory follow-up — 2026-10-08

Candidate `c50c50061` also forwards session device preference and verbosity.
Previously only flash attention was forwarded; CPU-only hosts hid the device
omission because init_best selected another CPU backend. That different backend
instance takes the split-loader weight-copy path, defeating mmap and risking a
16 GB F16 allocation. [Actual-library parameter guard](https://github.com/CrispStrobe/CrispASR/actions/runs/37758333664)
at `7e1806060` is queued: eight combinations plus failed-open default restoration,
with an incremental negative-control rebuild removing the two assignments.
The probe compiles and scripts parse locally; hosted proof remains required.

The original Q4 numerical run continues at its unchanged `fae7e2241` source.
The unstarted F16 job was cancelled for the memory defect above. Replacement
[combined Q4/F16 acceptance](https://github.com/CrispStrobe/CrispASR/actions/runs/37758607763)
at `c50c50061` waits in the same serial group. All numerical arms explicitly force
CPU, including the legacy baseline, and session/CLI model contexts are released
between speech checks. This is CPU-only acceptance, not a GPU claim.

The first Metal shared job compiled the Objective-C++ test but failed linking
Objective-C runtime symbols. `7b44257d3` adds explicit `objc`; [fresh shared/static
coverage](https://github.com/CrispStrobe/CrispASR/actions/runs/37757366299) is queued.

## Non-flash numerical gate and Metal discovery — 2026-10-08

[MiMo Q4 run 37754319857](https://github.com/CrispStrobe/CrispASR/actions/runs/37754319857)
failed the non-flash numerical gate. All 11 default LM/tokenizer stage arrays
are byte-identical to baseline. English/Chinese text matches for all three arms
through both CLI and C ABI (12 outputs). Non-flash final hidden-state cosine
is 0.998144631 with relative L2 6.1097%; logits are 0.998423708 / 7.4700%.
Tokenizer pooling relative L2 reaches 8.3401%, and discrete RVQ codes agree
on 73.5960%. The thresholds remain unchanged; this is not numerical acceptance.
The archive also preserves independent Python-reference metrics, including
Q4 error already present in the baseline. #492 remains unmerged.

The queued combined run was cancelled before execution because the known Q4
gate would prevent reaching F16. [F16-only replacement 37761799417](https://github.com/CrispStrobe/CrispASR/actions/runs/37761799417)
uses the corrected memory/CPU harness at `c50c50061`. The actual-library ABI
guard and Arabic/Java acceptance are now executing.

[Metal static job 113245123775](https://github.com/CrispStrobe/CrispASR/actions/runs/37757366299/job/113245123775)
compiles and links the actual Objective-C++ test, then fails with no tests
matching the label. A minimal Catch 3.7.1 CMake reproduction confirms that
semicolon-valued PROPERTIES split the labels, leaving only `unit`. The focused
workflow now selects the actual test-name prefix, requires exactly two discovered
cases and keeps the unit label. Corrected discovery passes locally; that proof
uses a listing-only dummy and makes no GPU/cache-runtime claim. Fresh macOS
shared/static coverage is required.

## Complete Arabic and MiMo ABI proof — 2026-10-08

[Arabic acceptance 37756933124](https://github.com/CrispStrobe/CrispASR/actions/runs/37756933124)
passes all runtime checks: 15 words / 85 measured character spans against exact
independent full-sequence Viterbi, CLI word/segment JSON, Python/C ABI offsets
and invalid accessors, plus real Java/JNA equality under a US-ASCII host default.
Artifact 11542736536 tests merge `a9c1afead`, PR head `d9fcfcc90`. Branch
`f0a40990b` incorporates main `608e60122` and records proof; only the Metal
test/workflow and maintainer docs changed after accepted runtime. Final platform
checks remain pending; #516/#490 stay unmerged/open.

[MiMo ABI guard 37758333664](https://github.com/CrispStrobe/CrispASR/actions/runs/37758333664)
passes eight device/verbosity/attention combinations and eight failed-open
default resets. Removing the two forwarding assignments fails with actual
GPU=true/verbosity=1 instead of requested CPU/quiet; restoring them passes
all 16 rows. Artifact 11543240521 is archived. This is actual-library parameter
proof, not model/GPU inference. Q4 non-flash numerical acceptance remains failed,
and F16-only 37761799417 is queued at `c50c50061`. Branch documentation
`18b69d580` preserves all stage metrics, norms and categorical code agreement.

Metal discovery repair `608e60122` is in fresh [shared/static validation
37761910908](https://github.com/CrispStrobe/CrispASR/actions/runs/37761910908).
Both actual macOS jobs are queued; discovery-only local proof does not establish
cache runtime behavior.
