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
  newly added smoke check is still queued. This does not claim model/audio
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
  and propagation of apt's failure exit code. Hosted validation remains pending.

For unattended first download, use:

```sh
crispasr -m orukeet --auto-download -f audio.wav
# Once cached, the same short name works without a download flag:
crispasr -m orukeet -f audio.wav
```

Without `--auto-download`, an uncached named model retains the normal
interactive download prompt; noninteractive callers must opt in. The documented
manual-path workaround also continues to work.

## Fresh hosted acceptance still pending

At this checkpoint GitHub had not assigned runners to the new jobs. They are
queued, not passed:

- [Orukeet CPU download/cache/C ABI acceptance](https://github.com/CrispStrobe/CrispASR/actions/runs/37703780788)
  at `5bf3e80dd`: actual download of the 402,226,496-byte public Q4 file,
  SHA-256 verification, cached short name, filename, explicit path and anonymous
  C ABI comparisons on JFK; released v0.8.41 negative control.
- [Native CI with the apt fix](https://github.com/CrispStrobe/CrispASR/actions/runs/37704036313)
  and [lint](https://github.com/CrispStrobe/CrispASR/actions/runs/37704039045)
  at `061460b57`.
- [Five WASM builds and new note smoke check](https://github.com/CrispStrobe/CrispASR/actions/runs/37703785629)
  at `5bf3e80dd`.

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
| [#478](https://github.com/CrispStrobe/CrispASR/issues/478), [#461](https://github.com/CrispStrobe/CrispASR/issues/461) VoxCPM2 | Existing prefill and mixed-head fixes shipped. Windows/Khmer and Arc B390 remeasurements remain external; no newly proven Vulkan optimization. |
| [#456](https://github.com/CrispStrobe/CrispASR/issues/456) nyra | Explicitly deferred in PLAN; model/output licence restrictions and GMM-HMM integration remain. |
