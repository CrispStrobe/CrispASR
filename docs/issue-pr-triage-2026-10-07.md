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
| [#490](https://github.com/CrispStrobe/CrispASR/issues/490) Arabic character alignment | Feasible. `src/align.cpp` already walks UTF-8 codepoints and traces per-label Viterbi positions, then collapses them to word times. Preserve those spans through the aligner and JSON surface. Test repeated letters, Arabic codepoints/diacritics, blanks, punctuation and OOV handling; do not estimate character times by dividing word durations. Not implemented in this batch. |
| [PR #492](https://github.com/CrispStrobe/CrispASR/pull/492) MiMo/CANN performance | Existing CI is green and the contributor reports five exact transcripts on Ascend. Shared attention/mel code and device placement deserve stage/magnitude and decoded-output validation on default and non-flash paths, F16 and shipped quant, with the current ggml pin. The cited timing combines this PR with the still-open [ggml #4](https://github.com/CrispStrobe/ggml/pull/4); it cannot be attributed to this PR alone. Retained unmerged. |
| [PR #515](https://github.com/CrispStrobe/CrispASR/pull/515) streaming bindings / German ONNX Moonshine | Separate, substantial runtime change. Its regression failures above are setup failures; two native CI jobs also remained in progress. Re-run against bounded package setup and review model-dependent packet/flush, tokenizer, licence and ONNX companion behavior before merging. Retained unmerged. |
| [#483](https://github.com/CrispStrobe/CrispASR/issues/483) CUDA 12.6 | Reporter confirms the mismatch warning is gone. Latest follow-up requests an experimental forced-MMQ/no-tensor build for GTX1660 comparisons. That A/B package is still doable; it needs matched binaries, runtime and model/output checks. No forced-MMQ default change. |
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
exists until the reporter exercises the actual GPU. This branch is unmerged.

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
