# Historical plan snapshots — archived 2026-10-27

This file preserves earlier plans and dated receipts. Its NOW/OPEN/queued labels
are historical, not current assignments or job status. Start with
[the current plan](PLAN.md) and [the current handoff](docs/maintenance-handoff.md).
Audit code, complete issue threads and exact-source receipts before reviving an
item. No task in this archive is automatically active.

---

# CrispASR — Pending work

## Completed — Ordered correctness and dependency review (2026-10-26)

Owner: Codex. All six items were worked in the authorized order, followed by
final source acceptance and the closure/issue/PR audit. Available implementation,
preservation and full license reviews are complete. Four existing issues retain explicit
external dependencies confirmed by the user. The final sweep also found new
feature #557; none was closed speculatively.
[Complete result](docs/historical/ordered-agenda-2026-10-26.md),
[source/check receipt](docs/historical/ordered-agenda-2026-10-26.json).

1. **Nemotron fixed and qualified:** CPU and actual T4 accept pinned F16 at all
   26 original transcript/22 VoxCPM controls and unchanged stage/magnitude gates;
   former Q4 fails on both. Managed F16-only selection and full OpenMDW/origin
   notices ship in source, with cache/filename and opaque C ABI coverage. Existing
   v0.8.45 binaries need the qualified F16 file explicitly; no new release here.
2. **Canary #532 held:** all issue/comments/source reviewed; exact clip/assertion/
   startup evidence remains unavailable. No speculative allocator change.
3. **MMQ #483 preserved; hardware held:** run 40414804633 preserves both original
   ZIPs byte-exact in unpublished maintainer draft 424870665, with full vendor/
   cuBLAS and original notices. Full asset readbacks and independent 203,920-byte
   sidecar audit pass. Resource startup wait retains the original limits. No
   GTX1660/MX150 access or target correctness/performance claim.
   [Preservation proof](docs/historical/cuda126-mmq-preservation-2026-10-26.md).
4. **B390 #461 held:** source/log audit and upload/host-permutation timer split
   complete; ten-step quality retained. No target access or new speed claim.
5. **MUSA held:** complete current terms review and pre-SDK resource enforcement;
   eight hosted policy controls pass without SDK use. No authorized S5000/new
   permissions; runtime, redistribution and benchmark-disclosure conditions open.
6. **Nyra #456 deferred:** complete pinned Parts A–C review; model/outputs,
   production/hosted, derivative/recipient and training conditions remain. No
   appropriate rights, conversion or port.

Frozen native source `2506cdd4b` passes CI 40403357613 (13 jobs), lint 40403361382
(10 jobs) and privacy 40403364971. Unit tier: 2,104 entries, zero failures, four
optional fixture skips, all four Nemotron default/C ABI controls passed. Archive
worker `45cd1f949` passes eight guards, actual preservation and privacy
40414720651; native source is unchanged. Final audit finds five open issues,
zero open PRs: all 26 existing comments are unchanged, and new #557 has no comments. No issue closure
or external comment/message was justified. Inputs/hardware/rights, rather than
speculative feature work, are the remaining next steps.

## NEXT — New Linux GPU wheel request #557 (unclaimed)

The final sweep found [#557](https://github.com/CrispStrobe/CrispASR/issues/557),
created during this agenda. Whole description/all zero comments reviewed and
checked against current code and actual release assets. Recommend x86_64 Vulkan
first (needs a shared-library producer; current CLI is static), HIP next (bundle
exists, wheel/index/publication-audit support missing), and ARM CUDA afterward
(separate native toolchain/bundle/device target). Preserve runtime-dependency,
notice, ABI and actual-device acceptance. No implementation or publication was
started outside the completed six-item batch.
[Code/asset audit and next steps](docs/historical/issue-557-linux-gpu-wheels-review-2026-10-26.md).

## Published and verified — 0.8.45 (2026-10-26)

Owner: Codex. The authorized native/Python/Rust/Dart publication is complete.
[Permanent publication receipt](docs/historical/release-publication-2026-10-26.json).

- The original annotated tag and source `e58bb48b5` are preserved. Native producer
  40284721476 and wrapper publisher 40284721364 passed; anonymous registry reads
  matched the tagged source files.
- Native audit 40304161701 verified all 50 public assets, CUDA runtime checksums
  and full/split payloads, notices, portable CLI version and C ABI loading.
  Four notice-only repairs preserve all 130 original members, modes and links.
  The full member inventory and repair proofs are retained in the repository.
- Wheel producer 40303514758 passed all eight distributions and both publishers.
  Final audit 40313434985 verified the original four CPU wheels and sdist against
  full PyPI reads, all three GPU wheel downloads and live CUDA/Vulkan indexes,
  tagged package sources, notice manifests and qualified native library hashes.
  The first audit correctly rejected the stale index before Pages deployment;
  no package rebuild or re-upload was needed. Pages 40306957371 passed.
- Release-body readback matches. The current root release-note copy was removed
  after publication; the immutable tag retains its original notes. Thirty
  packaging/repair fault controls passed. No local compilation or inference ran
  during publication; existing Kaggle evidence qualifies the unchanged NN source.

## OPEN — MUSA runtime qualification and older SDK targets (2026-10-26)

Owner: Codex. **Source fix prepared:** the actual Docker recipe uses the tested
SDK 5.2 digest, official package versions and architecture 31, with matching
SDK build/runtime bases. MUSA is removed from automatic registry publication.
A manual self-hosted image validator checks packaging and dependency closure;
eight policy controls reject unpinned SDKs, widened targets, mismatched runtime
bases, hosted SDK execution and unreviewed publication. No matching runner is
registered, so the new runtime image is not yet validated.

The vendor's general software agreement section II.2(7) restricts non-Moore-Threads
platform use absent written authorization. SDK workflows now require a MUSA
self-hosted runner with a physical platform check until an applicable exception
is established. The earlier hosted compile result remains historical technical
evidence, not license clearance. See [build status and terms](docs/musa-build.md).

Owner: Codex. **Compile experiment complete:** SDK 5.2 with architecture 31
compiled the unchanged tagged WKV object, linked the full CLI through all 767
remaining build steps and produced the MUSA shared library. Run 40330966977
passed; root and ggml compiled-source diffs are empty. The first attempt failed
on CMake's generated source-tree files before WKV was tested; the corrected mount
retains source-diff guards. [Permanent evidence](docs/historical/musa-sdk52-2026-10-26.json).

The image digest, compiler revision, installed package versions and copyright
metadata are retained. These metadata files mostly reference proprietary terms;
redistribution permission is not established. No image/binaries were published.
Actual MUSA operations and speech output, runtime dependency closure and full
redistribution terms still need validation. Architecture 31/S5000 is the only
compiled target; this does not qualify 21/22 or isolate SDK from target selection.

Six Docker flavours published. MUSA is excluded from publication acceptance.
Adding pinned official muAlg headers and complete MIT/BSD notices cleared the
missing CUB header, but both MCC 4.2 (40303383587) and MCC 4.3 (40305189068) crash
in `rwkv_wkv7_f32<128>` with `MTGPUISD::FRAME_LOAD`. Neither recovery published an
image. The production recipe now selects SDK 5.2/architecture 31; the obsolete
4.3 recovery workflow is removed. The older targets and runtime/image
qualification remain open. No kernel changes or MUSA hardware inference claims.

## NOW — 0.8.45 prepared; external evidence pending (2026-10-25)

Owner: Codex. The complete authorized agenda is integrated on main. All fourteen
final-source workflows passed before the closure review and fresh issue/PR sweep.
[Release-preparation receipt](docs/historical/csm-release-preparation-2026-10-25.json).

1. **Complete: recurring CSM speech failure.** Conversion and runtime now preserve
   original Llama-3 rotary scaling for both transformer stacks and the independent
   depth theta. CLI and C ABI defaults use the corrected path; explicit
   `CRISPASR_CSM_ORIGINAL_ROPE=0` retains the diagnostic legacy control. Sampling,
   published weights, seed and the 0.15 speech limit were preserved.
2. **Complete: original and actual-device qualification.** Original HF F32
   backbone-prefill hooks supply 19 stages; fresh F16 passes their magnitude and
   relative-error gates. Separate original rotary operators qualify both stacks.
   Actual quantization with the three existing Q8 embedding tables reproduces
   all 442 published tensor payloads. The controlled AVX512 native case scores
   WER 0 and AVX2 scores WER 1/9. Final-source bare/explicit-ON CPU controls pass
   all eight speech cells with exact PCM identity. Actual Tesla T4 version 1
   passes rotary controls and both speech repeats at WER 0 with exact PCM;
   synthesis GPU peaks are 1696/1590 MiB. Full depth-decoder and Mimi model-stage
   parity is not claimed. [Qualification ledger](docs/historical/csm-cpu-profile-2026-10-25.json).
3. **Complete: external dependencies reviewed in full.** Descriptions, all
   comments, timelines and latest owner/reporter requirements were reviewed.
   Keep #532 open for the German crash reproducer, #483 for GTX 1660/MX150 MMQ
   measurements, #461 for B390 Vulkan quality/performance evidence, and #456 for
   its separate research/output license review. The paired #483 artifacts remain
   available through 2026-11-07. No speculative resolution or license grant.
4. **Complete: integration and 0.8.45 preparation.** The repository version script
   prepared version metadata and local annotated tag `v0.8.45`, targeting
   `e58bb48b52cf1fe957c19586297c83a2ed0d9fe4`. Publication is recorded in the section above.
   All fourteen workflows on that source pass: compiler CI, lint, deep analysis,
   regression, WASM, six binding ABI contracts, secret scan, Docker smoke, Ruby,
   Rust, Dart, CTC alignment, Metal cache and Pages. CI selects 2,100 CTest cases
   with zero failures and four dictionary skips. Final ordinary CSM scores WER 0
   at the unchanged 0.15 limit; Granite's published-profile regression passes.
   Compiled sources and tests match qualified NN source `2c1183840`, preserved in
   main history. CUDA builder 40220671948 supplies the independently verified
   69-file runtime with required notices and exact Debian runtime source packets;
   NVIDIA runtime libraries are host-provided.
5. **Complete: closure review, then fresh sweep.** After all final checks passed,
   the closure review found no additional supported resolution. The prior six
   resolved issues and two integrated PRs were already closed after the preceding
   agenda. The 21:03 UTC fresh sweep found exactly the four dependencies above,
   no new issues or follow-ups, and no open PRs. Each description and every
   comment is byte-identical to the full dependency review; there are no linked
   issue comments to follow.

Large CPU work ran on Actions; actual CUDA ran only on an available Kaggle slot.
No full model was loaded locally. Parent wrote NN/math changes; one Sol agent
handled metadata, capture/build plumbing and read-only review. Load and memory
checks guard compilation/inference; all five privacy guards run before pushes.
Public receipts use portable aliases and public Actions IDs; raw proof remains
private. Further issue work awaits the listed external evidence or a new report.

## Release and error follow-up — completed (2026-10-25)

The reviewed agenda was processed through the final license item before any
closure. Six resolved issues (#556, #555, #552, #551, #549, #518) and two
integrated PRs (#553, #554) are closed. The subsequent 08:26 UTC sweep found
no new issues, PRs or actionable follow-ups; only the four dependencies below
remain open. [Complete receipt](docs/historical/release-errors-2026-10-25.json).

- Original 0.8.44 CPU wheels and sdist are on PyPI, with all five full anonymous
  readbacks matching original producer hashes. Recovery 40171262926 reused the
  successful artifacts. Artifact selection excludes ABI receipts; seven checks
  guard metadata, completeness and Python 3.10 compatibility. No new tag or
  rebuilt release artifacts were introduced.
- Contributor Dart/Rust patches retain original authorship in `1a9ff6889` and
  `2fdc45923`. Actual hosted builds and all six ABI contracts pass, including
  malformed UTF-8 and native synthesis-error/null-fallback coverage.
- Corrected OmniVoice CPU campaign 40171264744 passes all nine arms and both
  eligible references/targets at unchanged WER <= 0.2 (maximum 1/9). Same-code
  DSP matches upstream exactly; independent PCM/code/hash/coverage audit passes.
  Historical M1 judge disagreement remains recorded. This establishes the
  upstream-processing port's CPU scope, not every GPU, speaker similarity or
  universal click removal. Full raw proof remains private with exact notices;
  no CC-BY-NC version or wider fixture-rights grant is invented.
- Nightly investigation 40171767752 passes pinned sources `8bc09441a` and
  `96aeb4269` on one AMD EPYC 7763 host. Their data2vec CTC arrays and all four
  CSM audio PCM streams are byte-identical, with CSM WER 0. Original nightly
  failures remain unreproduced; no NN fix or gate relaxation is claimed. Keep
  the old failures and new CPU/compiler/raw evidence for a future failing host,
  and do not repeat the same passing pair without a new hypothesis/input.
- Source `008b421e7` passes all 13 CI jobs, 2,094 selected CTest tests (zero
  failures; four dictionary skips), 189 Python tests and 34 subtests. All10 lint
  checks, deep analysis, secret scan, six ABI jobs, Dart and Rust pass. Native
  sources remain unchanged from qualified `96aeb4269`. A later scheduled
  DeepLint 40173354661 is optional background verification of unchanged C/C++.

External dependencies remain open:

- #532: the reported assertion/device/model/clip reproducer is still missing;
  CPU/T4 allocation controls did not reproduce the crash.
- #483: actual GTX 1660/MX150 MMQ results remain external; paired packages are
  available through 2026-11-07. Matching CUDA 12.6 packages already resolve the
  reporter's runtime-version warning.
- #461: B390 Vulkan profiling must preserve the reported ten-step quality.
- #456: retain the separate license-review hold; upstream terms still cover
  generated alignments. No conversion or registry support was added.

Heavy CPU work ran on Actions. Local model-free compilation was stopped when
unrelated load exceeded capacity; no large local inference was run. Any future
CUDA work requires a free Kaggle slot under the existing usage guide.

## Granite Speech5 Q4_K (#552) — completed (2026-10-24)

The `granite` default is the qualified4.07 GB mixed Q4_K model. Bare
`crispasr-quantize input-f16.gguf output.gguf q4_k` reproduces the CPU/T4-tested
file byte-for-byte. F16 remains explicit; Q8/Q6 and earlier failed recipes remain
excluded. Apache model/MIT recipe licenses, notices, original publisher card
and prior F16 provenance are preserved. All model/metadata readbacks pass.

Frozen source96aeb4269 passes13 CI jobs,10 lint jobs, cppcheck, secret scan and
full CPU regression40078090078 for BOTH published F16/Q4 models. Independent
raw-array replay and all44-file anonymous proof readback pass at immutable
validation revision3df6d8a66f94036d4773ee2424e42ab2ece9b046. Native2093 tests
include actual C ABI/default-bundle and CLI resolver checks. Cached F16-era
provenance upgrades to the new pinned Q4 release through the native resolver.
[Complete source/model/legal/CPU/CUDA proof](docs/historical/granite5-q4k-2026-10-24.json).

The lint pipeline now propagates build failures through `tee`, proven with
successful and failing command controls. Granite-only replay skips unrelated
legacy preflight and cannot advance the core-model regression baseline.
Inference/quantizer/test bytes remain identical to the qualified source.
Native context4096; English JFK qualification only, with no full16k or speed claim.

## Granite Speech5.0 F16 (#552) — completed (2026-10-23)

Owner: Codex, isolated feature worktree. Native YaRN conversion/runtime support
and shared CLI/C ABI prompts are implemented. The initial release qualified F16:
full original F32 stage/magnitude, prompt IDs, both27-token decodes, repeated
renamed-model C ABI Sessions and CLI checks pass on hosted CPU39984477044 and
actual T4 CUDA v4. Native context remains4096; rotary testing through16383 does
not qualify full16k inference. Legacy non-YaRN and NAR defaults retain their
original parameters.

Hosted publication40009546796 succeeded. Model repo
`cstr/granite-speech-5.0-2b-GGUF` revision
`701f5529d8459412cc1b6053b5880c630125b138` contains the5.58 GB F16 model, full
Apache model license, separate MIT recipe license, NOTICE, original publisher
card and selected-profile validation provenance. Anonymous full-byte model and
metadata readbacks pass. The failed parent matrix remains unchanged in public
evidence; only its fully accepted F16 profile is released. Corrected CUDA
capsules retain the actual SDK CUB2.3.2 BSD notice and corresponding sources for
the external runtimes; all old capsule ELF payloads remain byte-identical.

The initial integration set the `granite` default to pinned F16, with legal extras.
The2026-10-24 Q4_K recipe above supersedes that default and its Q4 rejection.
Initially unqualified Q8/Q6/Q4 downloads and silent filename-to-default substitutions are
rejected. Explicit legacy4.0 filenames and4.1 defaults are preserved. Registry
checks pass6938 assertions/68 cases. A dedicated hosted nightly regression
repeats the full accepted F16 gates; generic Granite frontend diff is insufficient.
Final source `5b5278a0f` passes all13 CI jobs (40012494853), all10 lint jobs
(40012494794), regression40012494531 and secret scan40012334127. The dedicated
F16 regression repeats the full original stage/magnitude, prompt, both decodes,
renamed-model C ABI and CLI acceptance on that final source. Its21 output files
and manifest are permanently checkpointed in the public validation dataset at
`0fa4dbaf1041069dbf5fecacbb41ac1149af8767`; the portable
[receipt](docs/historical/granite5-2026-10-23.json) pins the exact prefix and hash.
Rebasing onto current main preserves the validated tree; subsequent changes only
record these results. Integration is complete. No release tag, PR or issue
comments are part of this task.

Open precision follow-up: Q8 retaining F16 embeddings passes CPU but fails the
actual CUDA pipeline-logit norm gate at1.010822821 (maximum1.01). Keeping the
head at F16 also fails at1.010823834 in the controlled GPU comparison. Both
retain exact words; that does not waive the magnitude gate. Q6 and every Q4
head/embedding arm are also unaccepted. The2026-10-24 value/down F16 recipe
separately passes full CPU/CUDA acceptance; the earlier failed recipes stay rejected.

Resource policy: check load and available memory immediately before local
compile/inference. Large CPU work stays on GitHub; actual GPU work stays on
Kaggle, using a free slot on the single configured account. Up to two Luna/Sol
agents may do small tasks when local resources permit; no NN math delegation.


## Issue555 model-sized designed defaults — completed (2026-10-23)

Owner: Codex, branch `fix/new-issues-20261023`. Fresh original VoiceDesign
bake39938969713 passes license/array audit. Raw default reference fails full
transcription and is rejected; clear is exact. Select clear as default with
unchanged original 1024/2048-d Base encoder arrays. CPU39942586420 and actual
Tesla T4 CUDA `issue555-designed-voices-cuda-v1` each pass13 reference/speech
checks at WER0; wrong legacy pack exits15. Model-specific defaults are byte
identical across CPU/CUDA. No Vulkan-hardware or performance claim.

[Asset revision9ab96c5](https://huggingface.co/cstr/qwen3-tts-voices-GGUF/tree/9ab96c5ba3022bd2ed46306ded0489a24096f8b5)
passes anonymous readback of all37 files, actual card license and unchanged
legacy bytes. New defaults include embedded/readable Apache and MIT notices,
complete source declarations, reference, arrays and exact recipes. Registry
uses separate immutable URLs and includes notices/provenance. Speaker identity
stays unknown; human listening feedback is pending, not inferred as approval.
Publisher grants do not independently clear upstream training or likeness rights.
[Method](docs/qwen3-designed-voices.md), [receipt](docs/historical/qwen3-designed-voices-2026-10-23.json).
Granite5 F16 integration and final accepted checks are recorded above.

## OmniVoice listened Metal default promotion — completed (2026-10-23)

Owner: Codex. Maintainer listened to all station WAV comparisons and accepts
minor differences. Eligible F16/F32 Higgs on Apple ARM macOS/Metal now defaults
to precise hybrid+graph+fused encoding, preserving explicit old paths. No
CPU/CUDA/iOS/quantized-profile promotion. Fresh encoder39–47% faster; combined
13–16% faster than precise hybrid; all48code arrays exact. Fresh native-process
median2.16→1.98s; native tokenizer peakRSS median adds223MB. 16 broad Session
renders and final linked Session/marked CLI preserve codes/audio. Prior scored
Anna omission and older fox/Linux limits remain. Five CTest cases, ASan/UBSan,
wrong-tie control, fresh-context cache hit and no-GPU checks pass. Injected
failures at all3GPU stages recover exact CPU codes and stay CPU; failed attempts
skip reference-cache publication. HYBRID=0 restores prior CPU encoding.
[Eligibility, switches and complete receipt](docs/historical/omnivoice-metal-defaults-2026-10-23.md).
Hosted checks remain separately pending; no release/tag or model/audio upload.

## OmniVoice GPU semantic divergence — completed (2026-10-23)

Owner: Codex. Runtime935b31eca adds independent default-off semantic F32
matrix hints and actual truncated-graph/mixed-stage probe. Station's close
stage3/frame55 codebook choice is flipped by Metal reduced-precision matrix
arithmetic; hints reduce full semantic maxabs0.003507→0.00000787 and preserve
CPU codes in all56 stage combinations across seven references (en/de/fr,
60–315frames). Corrected precise/combined GPU paths remain34–50% faster than
shipping CPU in48 paired fresh encodes, all exact codes. Full Session24renders
pass16 candidate/CPU byte-exact code/PCM comparisons; both ASR judges score
seven distinct cases0 and preserve Anna's CPU omission of “home” (1/12).
Normally marked CLI matches target codes; both judges0 including disclosure,
C2PA present. Five CTest cases, ASan/UBSan and negative controls pass.
Local original and corrected listening WAVs prepared; corrected copies equal
CPU WAV bytes. GPU/precision defaults stay off; broader fox/Linux acceptance
and cross-device/quant evidence remain open.
[Recipes, stage/margin evidence and raw receipt](docs/historical/omnivoice-semantic-precision-2026-10-23.md).

## OmniVoice encoder A/B — completed (2026-10-23)

Owner: Codex. All five independent encoder paths implemented with old paths
retained (checkpoint48aa6d20a). Promote exact immutable cache and Apple ARM CPU
pointwise fastconv only: three uncached synthetic references improve5–8% with
exact frozen-baseline codes and five exact cloned-code/PCM pairs through Session.
Extra default cached payload55.6 MB; full281 MB bake remains explicit mode2.
Hybrid/all GPU encode is about39–54% faster but station changes4/488 reference
codes, so hybrid/graph/fused stay off despite WER0 in both frozen judges on all
new controls. Unit/default/mode-switch/allocation/tie tests, five CTest cases,
ASan/UBSan, wrong-tie mutant and full lib/CLI checks pass. Broader fox-cloning,
hosted acceptance and GPU semantic boundary parity remain open.
[Parameters, commands, decisions and receipt](docs/historical/omnivoice-encoder-2026-10-23.md).

## Issue555 Base voice-pack crash — runtime validated (2026-10-23)

Owner: Codex. Runtime fix `de83cc6a5` rejects incompatible speaker/code tensors
before prefill, preserves the previous pack on failed replacement, resets stale
selection after successful replacement, and removes the incompatible0.6B extra
from1.7B auto-downloads. Exact pre-fix bounds abort is reproduced without weights.
The actual public loader/prefill test passes38 checks; all11 selected local
CTest cases pass. Hosted unit39911120897 passes2081 CTest cases and189 Python
tests/34 subtests, including the actual lifecycle test.

Hosted package/CPU run39910096656 at `682b54af2` passes. One actual Tesla T4
CUDA kernel, `issue555-cuda-v1` / `crispasr-qwen3-voice-pack-555` version1,
passes the same10 cases: both wrong-pack CLI calls exit15, both public loader
probes pass11 checks each, and all three0.6B-Q8/1.7B-F16 speech controls have
WER0 on CPU and CUDA. Native CUDA graph logs and per-child GPU-residency peaks
are retained. No idle-GPU denominator or new default/performance claim.
[Receipt](docs/historical/issue555-voice-pack-2026-10-23.json) and the public60-file proof
include all logs/WAVs, source/model/binary pins and an independent saved-data
replay with three rejected negative controls. Anonymous archive/expanded hashes
and replay pass. Full platform CI/lint on the final documentation tip remain
to be collected; the direct-header lint finding is fixed and passes locally.
1.7B requires a matching voice pack or reference WAV/transcript. No release tag
or Vulkan-hardware acceptance. Issue551's13 CI/10 lint jobs pass; issue552
Granite5 F16 support is implemented; see the current integration task above.

## M1 OmniVoice longer inputs and parameter reference — completed (2026-10-23)

Owner: Codex. Runtime `50196b2cd` hosted CI39899280705 and Lint39899280570
pass; Ruby/WASM/Docker/regression/secret scan pass. New 18-second / 450-frame
32-step control: views request 28.30→26.11 s; packing 26.23→25.97 s remains
inconclusive. All 16 CrispASR PCM/code arrays are exact. Same-active-weight
omnivoice.cpp median 35.28 s (sequential, shared-host scope); both ASR judges
score all three longer controls zero. Full marked CLI emits identical codes.
TTS guide now documents defaults/interactions, sampling, commands and costs;
[long-input report](docs/historical/omnivoice-m1-long-input-2026-10-23.md) includes source/
model/judge pins, exact recipes/builds, individual timing receipt and limits.
A public isolated matmul probe tests actual Q8 versus derived F16 weights at
150/450 columns with exact outputs, norm/CPU/wall checks. Its modest hot-op
benefit does not validate a full-model F16 cache or artifact. No runtime/default,
model or ggml changes. Fox-cloning and full-model F16/Q4/CUDA coverage remain open.

## M1 OmniVoice generation and startup — completed (2026-10-23)

Owner: Codex. Hosted harness repairs `1ae585389` address the allocator stub and
CPU-header lint (hosted CI39897381899 and Lint39897381791 now pass). The
earlier 2,077 Linux CTest cases already passed. New opt-in
attention views remove 84 copies per forward: paired generation 8.10→7.45 s,
request 8.60→7.99 s, exact code/PCM parity. A precompiled repeat confirms
8.49→7.62 s requests. Packed projections stay opt-in
(~312 MB extra, no convincing timing win); unified CFG stays off on Metal.
Same internal-file startup medians 0.503/0.490 s embedded/precompiled, warm OS
caches; historical external I/O is not a cold-start control. Full precompiled
Release CLI runs with normal marking; full embedded shared-library Session
checks unconditioned, both synthetic reference voices and combined gates.
Both pinned ASR judges score these new walk-sentence controls and marked CLI
output zero. Actual F32/F16/Q8_0/Q4_K graph parity, allocation checks, four CTest
cases, ASan/UBSan and a deliberately wrong K-offset negative control pass.
[Samples and scope](docs/historical/omnivoice-m1-startup-generation-2026-10-23.md).
All new gates stay off by default; broader fox-cloning acceptance stays open.
Next target is the dominant matrix kernels/longer inputs and cross-device
coverage before a default change. No ggml bump or release is implied.

## M1 OmniVoice comparison and resident decoder — completed (2026-10-23)

Owner: Codex. Gated resident decoder, loader/lifetime regressions and reproducible
native comparison tools are implemented. Quiet paired request 8.84→8.08 s;
matched-value omnivoice.cpp bracketing medians 9.57/10.01 s. Both ASR judges pass
the unconditioned comparison; clone transcripts retain existing disagreements.
Final 48 fixed-code decodes and fresh uncached CPU encodes preserve baseline
bytes/codes; four selected CTest cases, ASan/UBSan and negative control pass.
The full Release CLI/shared library builds; live Python Session C ABI outputs
match the scored native code/PCM bytes for CPU and resident modes.
Resident weight/cache buffers 124 MB. Split/resident/decomposition remain off
by default. [Scope, commands and receipt](docs/historical/omnivoice-m1-comparison-2026-10-23.md).
Next optimization target is the dominant generation stage and expensive
initialization; these measurements do not validate a ggml bump or a
platform-wide default. Broad cloning acceptance is still open.

## NOW — OmniVoice codec and acceptance audit (2026-10-22)

Owner: Codex. Repairs landed on main at runtime source `5acff9f21`. Audit baseline main
`22e1b6ada` and codec branch `f76d4b190`. Repairs cover split-off placement,
backend-aware reference caches, complete reload cleanup, transactional
permutation publication and a 40 MB unused F32 cache. Decomposition and split
both remain off by default on all backends; the counterbalanced CPU test did
not establish a consistent wall-time gain.
Reference selection now includes every integrity-eligible reference. Historical
subset acceptance is rejected; failed WER gates persist aggregate receipts.
Local native/sanitizer tests and four selected CTest registrations pass; fixed-code
live pairs preserve both ASR transcripts and differ by at most one PCM16 step.
On M1, `station` fails the established Nemotron gate (2/9) while CTC scores it
zero. Linux CPU campaign39857487336 on first repair `b608bcdd1` instead fails
three `window` cloning surfaces at2/9; station targets pass. Its utility/DSP
oracles pass, including exact waveform parity. Complete campaign39861560201 at harness source `f87af58aa`
(runtime code `5acff9f21`) also failed WER 2/9 on the three window-cloning
surfaces; station and unconditioned arms passed. Generated codes are retained
for every surface.
Superseded39860613583 was cancelled before acceptance. CI/lint follow the latest
main tip; earlier partial green jobs do not certify the tip. No full hosted or
cloning acceptance is claimed.
See `docs/historical/omnivoice-audit-2026-10-22.md` and its live receipt.

## Issue551 schema rejection — validation complete (2026-10-23)

Owner: Codex, branch `fix/issue551-routing`, base `95e454b49`.
Implemented and locally validated through source `17de1a7`: shared schema
rejection, positive CTC detection, CLI filename/alias/legacy-Whisper protection,
and C ABI/Python rejection. Actual external Unified metadata is RNN-T; full
Unified inference/streaming support is separate. All 24 CLI controls,113 schema
assertions,64 registry cases and existing architecture/parameter guards pass.
CTest selects and executes all 5 routing tests; CI now requires that registration.
[Scoped evidence](docs/historical/issue551-routing-2026-10-22.json).

Hosted CI39722231798 passes all13 jobs and lint39722234064 passes all10
jobs, checked individually on2026-10-23. Both target immutable source `17de1a7`;
this validates the explicit unsupported-layout handling, not Unified inference. NVIDIA and OmniVoice acceptance now explicitly
affirm the user's eligibility/compliance obligations; project redistribution
obligations remain ours. Five corrected model-card/notice documents are published
and anonymously verified, with no weight changes.
[Notice receipt](docs/historical/omnivoice-license-notices-2026-10-22.json).
PR #521 has merged; the complete corrected cloning gate remains pending.

## NOW — v0.8.44 publication and wheel compatibility (2026-10-21)

Owner: Codex, branch `fix/issue-followups-20261020`. User authorized publication.
Runtime tag `v0.8.44` is public and immutable at `46506786d`: CI13/lint10/WASM5,
Dart/Ruby and all6 SDK ABI jobs pass. Three versioned installed-wheel controls
and actual Japanese Metal17/123/17 metadata reuse pass. Published Mac runtime
libraries are byte-identical to that test; shipped Linux/Windows metadata and
cache-release exports are independently inspected. Rust/sys/pub.dev0.8.44 are
published; the native Release workflow is still completing its platform builds.

The artifact audit found incorrect wheel compatibility labels before PyPI upload:
Linux importsGLIBC2.34/GLIBCXX3.4.30, requiring the pinned manylinux2_35 policy;
the Mac core targets11.0 but the optional helper was accidentally built for14.0.
Python publication is temporarily disabled. The fix keeps Mac11.0 by compiling
its helper for that target and checks every finished wheel's OS/stdlib floor.
Packaging QA follows the workflow revision; runtime/Python sources remain at the
requested tag. All15 guard tests pass; actual old Linux/Mac artifacts are rejected,
corrected wheels pass, and the rebuilt Mac helper loads all3 bridge exports.

Finish corrected publisher checks, push/re-enable it, and dispatch from updated
main with `tag=v0.8.44` if the automatic event was missed. Inspect all4 CPU wheels
plus sdist, all3 GPU wheels/index links, all53 expected GitHub assets (explicitly
record the Linux GPU filename's corrected platform tag), and registry versions.
Then update issue549/PR544, keep549open for WhisperJAV retest, and archive the
final receipt/history. Optional NZ caching stays off; shared-ggml PR2 and
OmniVoice/custom-license promotion stay held. Hardware/private reporter inputs
remain external. No runtime retagging or model conversion is required.

## NOW — issue follow-ups and license policy (2026-10-20)

Owner: Codex, branch `fix/issue-followups-20261020`, base `069365a42`.

Active follow-up: exact-tip Linux CI at8237 passes 2,039 CTest cases, but its
full log exposes missing Python test registration in a clean configuration:
the interpreter is discovered after the guarded declarations. Discovery ordering
is repaired, the model-free Japanese validator no longer matches `-E live`, and
CI asserts all four follow-up validators are selected before compilation.
The exact CI guard rejects the original clean configuration and accepts a new
fixed configuration; 29 Python tests become eligible. All 23 non-SDK suites pass
through CTest locally; six language SDK probes retain separate acceptance.
Completed source883549696 passes all13 CI39467977527 jobs and all10
lint39467977612 jobs, plus Ruby, Metal-cache, CTC-alignment and secret checks.
The full unit log confirms 2,068 selected CTest tests with zero failures and four
dictionary-data skips, plus189 Python tests and34 subtests. All four required
validators actually execute. Conditional SDK skips are not complete SDK coverage.
The same run's macOS job never acquired a hosted runner (capacity annotation),
so its cancellation is not a source build result.

Completed checkpoint: ordinary NC support retained; missing OmniVoice/OpenRAIL
acceptance gates repaired and the public OmniVoice license card corrected.
Issue550 actual T4 long-input and issue549 actual Metal metadata controls pass
independent review. Shared ggml PR2 passes CPU/Metal and hosted CPU/software-
Vulkan controls and remains unmerged. Repaired Linux/Mac/Windows native bundles
and all three hosted installed-wheel checks pass; library bytes match the
frozen bundles. [Package acceptance](docs/historical/followup-package-dry-runs-2026-10-20.json).
Main `0c355f398` now passes all 13 CI39445302976 jobs and all 10 lint39445303065
jobs. The remaining host review now passes 108 cases each on Mac CPU,
Metal/CPU, Linux CPU and software Vulkan/CPU, including BF16-weight and
strided-input fallback with actual placement checks. The candidate is unchanged.
The retained T4 trace also proves a sampled 2,735 → 125 MiB transition after the
public cache-free call and before fresh CLI model residency; six guards reject
wrong-device/phase interpretations. No new Kaggle run is needed.
[Cache-release audit](docs/historical/alignment-cache-release-2026-10-20.json).
Future release checks, custom-license review and actual Radeon/Ascend gates
remain open. No tag or release is published here.
Checkpoint: [follow-up evidence and reporter actions](docs/historical/issue-followups-2026-10-20.md).
NC models remain eligible behind explicit license opt-in; research-only and
additional use/revenue/hosting restrictions require separate review and are not
automatically accepted by a generic switch. Audit original revision and codec
licenses for OmniVoice; do not remove ordinary NC support. The existing issue550 CUDA run and issue549 exported-getter/live metadata
proofs are now audited; do not dispatch duplicates. Review shared ggml PR2 with
precision and fallback controls before considering integration. Reconcile
fulfilled issue485 and keep hardware/reporter dependencies explicit.


## CANN weight layout — remaining device and CI acceptance

Owner: maintainer; optional implementation integrated 2026-10-20.

NOW (2026-10-20), owner Codex: lifecycle/telemetry and pinned physical runner
are implemented at18b540bd5 (ggmle2ae679a).53 host lifecycle checks and576
independently audited CPU probe computes pass. The real Mac CPU/Metal F32
reference/CLI/Session/short-long-short matrix passes unchanged gates, with
22/44/22 words on each device. [Complete receipt and pinned device procedure](docs/historical/cann-layout-followup-2026-10-20.md).
SDK39401353196 (both device families), all13 CI39401352847 jobs and all10
lint39401352899 jobs pass for this exact source. Code and evidence are on remote main; verified scope and the optional
bounded request are posted on [PR544](https://github.com/CrispStrobe/CrispASR/pull/544#issuecomment-6288185477)
and [shared-ggml PR4](https://github.com/CrispStrobe/ggml/pull/4#issuecomment-6288187900).
Next: optional physical Ascend proof; completed host/SDK controls need no rerun.
No Kaggle work is needed. Physical Ascend execution remains external.

Implemented explicit usage hints across normal/filtered/split/fit/repack loads,
audited MiMo/Moonshine policies and shared-ggml optional NZ cache. The ordinary
ND representation is retained for reads/copies/fallback, with chunk coverage,
write invalidation and captured-graph pointer invalidation. Published artifacts
remain unchanged; the cache is off by default and the old path remains for A/B.

Host proof: [receipt](docs/historical/cann-weight-layout-host-2026-10-20.json):83 usage
assertions in each mmap mode,188 split/11 bounds/28 release assertions,26 SDK-free
policy checks and432 actual CPU probe computes over48 cases pass. The actual
CANN request correctly rejects absent hardware. MiMo/Moonshine/core/shared ABI
and probe targets link. This is not physical NPU/model or performance acceptance.

Shared candidate4d734595 includes completion synchronization before publishing
cache readiness. The first SDK/CI/lint jobs target146392764 /ggmldc3bc0bb;
they are superseded for final-source acceptance by source3d808b76d, pinning
ggml4d734595: [SDK39391430100](https://github.com/CrispStrobe/CrispASR/actions/runs/39391430100),
[CI39391432699](https://github.com/CrispStrobe/CrispASR/actions/runs/39391432699) and
[lint39391434462](https://github.com/CrispStrobe/CrispASR/actions/runs/39391434462).
These jobs were submitted and pending/running at this checkpoint, not accepted
as passing. Physical310P3 and910B3 cache-on/off, ACL replay, full-model
diff/magnitudes/CLI/Session and memory/timing gates remain open under
[the procedure](docs/cann-weight-layout.md). This checkpoint changes only tracking;
the compiled implementation code is unchanged from3d808b76d. Integration uses
shared-fork merge6b4bd833, whose source tree is exactly identical to4d734595
(tree30cc056e19599bb7667013c02be3cd75bb6f48e3). The newer fork history is
preserved on its canonical `crispstrobe-ops` branch. The cache remains off by
default; this integration does not accept pending CI or physical performance.

## NOW — OpenVoice2 actual-model and CUDA acceptance

Owner: unassigned; ready for a fresh agent (handoff 2026-10-19). Implementation, exact source
speaker dispatch, clip means/noise replay and the nine-stage synthetic CPU
encoder proof are archived in HISTORY.md. Method and scopes:
[OpenVoice baking](docs/openvoice2-baking.md). The synthetic F32 arithmetic
receipt does not establish pretrained embedding/conversion or GPU quality.

External state and next actions:

1. CUDA packaging [run39266520251](https://github.com/CrispStrobe/CrispASR/actions/runs/39266520251)
   succeeded on source0b4213c39d9ae335df32fb8d09f80febd2f2d8ae.
   Artifact12193662775 is698,965,729 bytes (Actions ZIP digest
   bce0538953a84094fc86c63729c37e2bc8dc41e27d25337a27f5de793bbc5030).
   Next: download to external storage, independently inspect provenance/member
   hashes and dependency closure; retain or mirror before its14-day expiry.
   Logs are retained locally. This is compile/link/package proof only, not
   actual GPU/model acceptance. Earlier failed build attempts are in HISTORY.md.
2. Kaggle `openvoice-bake` version2 is queued on old native source71e329ccf.
   Version1 terminal logs/output are retained after its pre-inference loader
   failure. Version2 preserves host loader paths but lacks the later dynamic
   backend ABI flags/discovery repair; it cannot qualify current conditioning.
   Retain logs/output before superseding it. Do not guess session IDs, launch
   hardware-fishing retries or exceed the two-session GPU cap.
3. Prepared `tools/kaggle/openvoice-conditioning/` pins a shared worker and
   native source6be9fda36. Bootstrap commit0b4213c39 pins shared worker5f9ad8b36
   by SHA256, including DL repair and early producer checkpoints.
   Its submission was rejected by the two-session batch-GPU cap: both the old
   OpenVoice and needed alignment jobs are queued. Cancellation needs an exact
   session ID, which the public API cannot resolve from a kernel slug; the
   maintainer has been asked to stop the obsolete run or supply its session ID.
   Submit once with tools/kaggle/kpush.py after a slot is confirmed free;
   check the actual response (CLI exit0 can still report rejection). The prebuilt
   base191815511 uses GGML_BACKEND_DL and exact ggml0a862076; compile
   the small model/driver delta with that macro, discover modules from the
   bundle directory and isolate Torch with host loader paths/different cwd.
4. Independently audit27/28 required stages, original pristine/captured replay,
   exact PCM and actual posterior noise, both norms/errors, clip means and
   mean-conditioned waveforms, uninstrumented C API/cache reuse and actual
   CUDA scheduler splits. Retain original/model/GGUF/input/library/source hashes.
   Publish vetted immutable reference artifacts. Keep F32-sensitive and any
   quant variants explicitly identified; no silent threshold relaxation.
5. Require baseline/original/clone ASR WER<=0.2 and independent speaker cosine
   improvement>0.01 in the bounded synthetic matrix. Then extend only new
   scope: full shared Session/CLI actual-model paths, raw mixed-rate processing,
   automatic Silero selection, natural references/languages, free-running noise
   and additional devices/quantization. GPU vocoder remains gated by
   CRISPASR_OV2_GPU=1; reference encoder/flows remain CPU. No release claim yet.

## Character-alignment acceptance qualification

Arabic CTC/Java39214206020 is complete and its retained output is independently
audited; see HISTORY.md and docs/arabic-characters-java-audit-2026-10-19.json.
Do not rerun the repaired Java preflight or accepted output matrix without a
new scope. Separate future oracle/DP replay needs retained vocabulary/binaries;
this older worker did not preserve those, or a separate raw Python output.

## Issue550 — completed actual CUDA acceptance

Closed after the independently audited T4 short/long/short matrix: complete
147/8,925/147 units, exact returning-short offset reuse, fresh CLI agreement
and 95 actual CUDA encoder splits. Long-phase sampled device total peaks at
2,735 MiB. The complete former procedure and failed controls are archived in
HISTORY.md; do not submit another run to recreate this acceptance.
[Receipt](docs/historical/alignment-cuda-live-2026-10-20.json),
[closure and scope](https://github.com/CrispStrobe/CrispASR/issues/550#issuecomment-6291711315).
Host memory still grows with duration; natural-speech boundary accuracy and
RTX3090-specific behavior are not certified.

## Go lifecycle — remaining scope after actual CPU model acceptance

High-level Model implementation,51 Go race cases/35 Rust cases/five doctests,
and the real tiny.en/JFK Go model proof are complete; see HISTORY.md and
[the independent audit](docs/historical/go-model-lifecycle-audit-2026-10-19.json).
Run39228691810/source5aba585ce verifies actual callback close/re-entry,
concurrent rejection, cancellation/recovery, parallel decoding and owned results.
Do not rerun the same accepted fixture merely because an older note says pending.
Next extend only new scope: beam/partial UTF-8, actual VAD model, DTW model mode,
other supported platforms/devices and other wrapper languages. Raw low-level
Go pointers and generic speaker handles remain caller-serialized; assess an
owned adapter before promising concurrent safety. High-level Contexts share the
latest native result state. All six earlier binding contracts39215435420 pass;
current-tip CI remains a separate integration gate.

## Generic speaker primitives — remaining model/platform scope

Python/Dart and C#/Java operation review is now complete alongside Go/Rust and
Ruby/WASM; repairs and their scoped local probes are archived in HISTORY.md and
mapped in docs/wrapper-callable-coverage.md. Do not reimplement these methods.
Remaining: real speaker-model/device quality, additional target platforms,
owned concurrent adapters and callback/concurrent-close behavior. Generic
handles remain caller-serialized; static/callable closure is not model or
universal concurrency acceptance. Broader API inventory remains below.

## NOW — finish the callable wrapper inventory and remaining adapters

Owner: maintainer coordinator (active). Whisper CPU pointer/Python model proof
and Japanese/Echo artifact audits are now complete; see HISTORY.md and receipts.
Issue550 actual CUDA memory/reuse is completed and independently audited.
Continue wrapper lifecycle/callable gaps, updated PR521 and release acceptance.
Go high-level parallel access and Python CI expectation repairs are complete;
see HISTORY.md. Do not infer model acceptance from the local glue regressions.
The Rust scalar-helper family now has compatible public controls and parallel
forwarding; completed repairs and measured probes are archived in HISTORY.md.
All 2,033 native unit tests passed; rerun hosted CI after the Parakeet
metadata fixture optional-signature repair (see HISTORY.md).
Continue the cross-language wrapper task; the bounded external agent tasks have
ended. Start with the incremental
[callable ledger](docs/wrapper-callable-coverage.md), then inspect actual code.
The [initial audit](docs/historical/binding-gaps-2026-10-18.md) is a historical snapshot.
Do not reimplement rows merely because an older note says they are missing.

The shared census covers **297 unique `crispasr_*` exports** across the two
shared headers, plus **14 separate chat exports**. It does not include every
upstream `whisper_*` declaration. Source declarations/mentions are not proof
of a public operation. Full parity and real-model acceptance remain open.

Recommended implementation order and concrete acceptance:

1. **Complete the callable ledger.** For each export, identify the actual
   public method/result in Python, C#, Java, Go, Rust, Dart, Ruby and WASM.
   Record bulk/superset equivalents, retired refusing stubs and deliberate
   platform exclusions separately. TitaNet and roster DB now have reviewed
   public paths in all eight; extend the ledger beyond those families.
2. **Legacy Whisper controls.** Rust legacy pointer-symbol routing and Java
   context/parameter ownership and option forwarding now have operation probes;
   completed repairs are in HISTORY.md. Validate actual Whisper models and
   supported native platforms, including callback threads and failed reinitialization.
   Java's legacy full-params struct is still 64-bit only: 32-bit decode refuses
   explicitly until size_t/pointer layouts are repaired and exercised on a target
   runtime. Preserve the existing Java long-valued public options when adding a
   pointer-width-aware native layout. Continue auditing mutable parameter objects
   and caller-owned pointer arrays/strings during callbacks and concurrent calls.
   C#/Python/Dart have pointer-safe parameter, DTW and parallel probes; inspect Go's compiled parameter routes and finish
   Java/Rust/Ruby/WASM callable gaps. Cover scalar setters, language detection,
   native progress/parallel, token IDs/raw bytes/alternatives/DTW and disposal.
   Use `tests/build_legacy_wrapper_probe.py`; a bare ABI declaration is not an
   implementation. Direct Parakeet uses an integer centisecond offset.
   Go's legacy parameter-string ownership and five scalar decode controls now
   have compiled operation probes. Rust now has parallel decoding and PCM language
   detection, and Go parallel decoding uses the progress-tracked pointer route
   with optional per-call progress callbacks. Verify actual model behavior and
   progress frequency. Continue DTW,
   remaining token introspection and legacy context/callback lifecycle auditing.
   Review higher-level APIs as well as the low-level bindings. Go's
   higher-level `ProcessParallel` optional capability now reaches the native
   pointer/progress adapter, with compiled operation tests. The bounded Go CPU
   model acceptance is complete (see its audit above); other modes/devices remain
   open. The built-in high-level
   Model's ownership/close guards now have compiled race/pthread probes.
   Rust/Go built-in DTW constructors, validated scalar DTW getters and Go's
   high-level model constructor now have operation probes. The shared helper
   disables incompatible flash attention. Owned Rust/Go token snapshots now expose
   IDs/raw bytes, timing/probability, DTW and alternatives; Go has a compatible
   high-level optional accessor without changing the older `Token` struct.
   Validate actual model output and candidate quality, including greedy/beam
   behavior, partial UTF-8 and alignment with the Python wrapper/reference.
   Custom-head configuration remains pending: own the head storage for model
   lifetime and validate model dimensions if exposing it. Audit the remaining
   Java/Ruby/WASM token operations against actual public methods; do not infer
   coverage from declarations. Preserve unavailable sentinels and split UTF-8.
   Extend
   `tests/fixtures/go-rust-wrapper-abi.c` and the actual Rust/Go public-method
   probes, including invalid indices/counts, null/empty native output and cleanup.
   Native declarations alone do not satisfy these rows. Go callback registries
   now synchronize concurrent contexts without holding locks through user
   callbacks, with a hosted race-detector gate. The built-in high-level Model
   now rejects overlapping native operations/close, including foreign callback
   threads. Raw low-level pointers still require caller serialization; audit an
   owned adapter and the remaining other-language lifecycle separately.
   Rust's 25 scalar-helper routes now have compiled controls/old-options probes;
   validate language/prompt/thread/temperature and alternative behavior with
   actual models, including native parallel chunk merging. Audit raw full-params
   fields and legacy callback routes outside that helper family separately.
   Reuse the owned controls in `crispasr/src/whisper_controls.rs`, common decode
   path `crispasr/src/lib.rs::transcribe_pcm_impl` and the owned token reader in
   `crispasr/src/whisper_tokens.rs`; do not reimplement completed setters. Go's owned
   reader lives in `bindings/go/whisper_tokens.go`; higher-level detailed access
   is in `bindings/go/pkg/whisper/token_snapshots.go`.

### Native parallel follow-up: remaining timeline/model-platform scope

CPU actual-model metadata acceptance is complete: run39211685024 passes all12
native and six Python cases, independently audited against exact source/library/
probe/audio identities and raw timestamps. See
[the receipt](docs/historical/whisper-wrapper-model-audit-2026-10-19.json) and HISTORY.md.
The tiny suites (22 parallel,13 VAD/entry, six emission) remain CTest/CI gates.
Do not rerun the same tiny.en/JFK matrix merely because an old note says failed.
Extend coverage to beam candidates, partial UTF-8, actual VAD-model detections,
foreign callback threads, GPU/supported platforms and model-specific behavior.
The scoped CPU proof is not original-reference/WER/speed or whole-wrapper parity.

Still audit these distinct semantics before claiming complete timeline parity:

- Available token/DTW getter values now use the same original-time map as
  segment getters, with copy-only mapping and sentinel preservation. Plain
  sequential/parallel/direct-state decoding clears a previous VAD map; failed
  VAD clears results. Nine literal native timeline/stitching cases cover these
  repairs. Validate actual VAD detections, overlap/clipping boundaries and
  VAD-on -> plain -> VAD-on model reuse through each public wrapper. Also test
  bounded offsets with Tiron's fixed 75-centisecond onset pad before claiming
  those model-specific window semantics; do not
  assume the controlled detector fixture establishes real alignment quality.
- Global parallel duration now clips PCM once before splitting, with exact
  sample geometry and forwarding checked by seven additional regressions.
  VAD offsets/duration still select in processed-audio units after the prepass;
  establish whether a future original-time selection API is needed without
  silently changing that existing contract.
- The first chunk may have delivered callbacks before another worker fails;
  failure clears the stored result but cannot retract callbacks. Exercise
  foreign-thread encoder/abort callbacks and actual context/device allocation
  failures. Same-context concurrent decode/close is unsupported.

Source audit found no double filtering: `whisper_full_with_state` does not call
VAD. Preserve the single prepass and parent mapping rather than disabling a flag
based on the earlier hypothesis. The parallel worker fixture checks mapping retention;
`tests/test_whisper_vad_timeline.py` additionally executes the real stitcher,
interpolation, token getters and scalar adapters with controlled detections.
The decoder body remains controlled, so model quality is still unmeasured. Use Kaggle for genuine GPU behavior
when needed, after hosted CPU correctness; do not wait on hours of idle CI.

3. **Ruby remaining resource/legacy audit.** Session/stream/chat borrows,
   opaque tickets, child-close ordering and raw text processor ownership now
   have compiled regressions; see HISTORY.md. Audit microphone ring/close and
   the old Whisper object/parameter/result path, including all allocation-returning
   audio/container/scoring helpers on Ruby allocation exceptions. Microphone
   ticket/pump ownership now has compiled native-thread regressions, including
   callback-close and interrupted/error cleanup; verify actual device drivers
   on supported platforms. Extend the
   callable ledger rather than treating 155 guarded operations as API coverage.
4. **WASM remaining adapters and dispatch.** Registry/cache and v2 load-options adapters now have compiled Node
   probes; validate their full-module size and browser filesystem/download behavior.
   Generic diarization/turns, language-aware VAD and owned logits results now
   have compiled operation probes; include them in real-model/browser acceptance.
   Finish remaining legacy controls. Retained ASR callback setters and session
   stream adapters now have public paths; validate actual native backend behavior.
   Generic speaker/clustering/pyannote adapters now have operation probes;
   include them in full-module/native/model validation. Finish compute/servicer scheduling. Chat remains opt-in/single-thread until
   work and callback delivery are safely proxied. Measure complete default
   and opt-in module sizes and instantiate actual models in a browser.
   Concrete remaining WASM acceptance:
   - Validate retained ASR segment/token/progress events with actual models and
     streams, including which backends emit each native hook. Copied-event transport,
     raw token bytes, generations, replacement/close, bounded backlog and JS/native
     allocation cleanup have component probes; they do not establish event support
     on every backend. Keep registration idle and before retained stream creation.
   - Validate retained stream adapters with actual models: feed/flush/text/live
     decode, backend stream-kind semantics, finalization and browser responsiveness.
     The compiled probes cover native transport and child-before-parent ownership,
     including async replacement/close. Threaded callers must use async stream
     methods and async parent replacement/close while a child exists. Check actual
     backend cleanup and module instantiation in hosted full builds.
   - Dispatch now has real pthread Node/browser operation probes, including
     native/encoder/callback errors, overlap guards and progress responsiveness.
     Extend acceptance to full models and module variants. Audit remaining ASR/
     synchronous result encoders for intermediate emval/scratch leaks when JS
     constructors/getters throw; JS throws do not unwind default C++ EH frames.
     Extend borrow guards through argument coercion: a JS getter can submit
     async work or close an owner after an operation's initial guard.
   - Async v2 ASR/TTS load options now have literal-POD and worker-thread probes;
     validate real backend/model replacement and memory limits.
   - Token polling copies before JS constructors run; count/drain calls now
     reject during active async compute. Retained ASR delivery uses a separate copied
     queue that can drain during compute. Audit coercion/lifetime guards across
     remaining adapters.
5. **Residual public options and polling ownership.** Go/Rust now have load
   options, standalone truecase/PCS and owned segment polling; Rust text scoring
   and Go capability queries have compiled probes. Review remaining ledger
   rows rather than treating the old missing-family list as current. Rust's raw
   token polling is explicitly unsafe: a future safe adapter needs ownership
   beyond the borrowed buffer selected globally by the C ABI. Do not permit
   another thread to close/free it while copying.
   Ruby and WASM now expose v2 session-open options; verify them with actual models.
   Finish per-call VAD settings in languages beyond the C#/Java/Dart/Python
   paths already implemented; probe actual option fields and language.
6. **Platform and model acceptance.** Run full-library/compiler checks and
   representative real tasks on pinned source/models/inputs, including TTS→ASR
   roundtrips. Exercise 32-bit Java/Dart diarization marshaling on actual target
   runtimes; the new pointer-aware layouts and 64-bit probes are not that proof.
   Review real CI logs before acceptance. No release from fixture success or
   queued jobs.

The ASR event checkpoint `6eda4a9da2c867fcbda2262f2cd867dd0c6e5915` passed all
six hosted [binding contract jobs](https://github.com/CrispStrobe/CrispASR/actions/runs/39151650142).
The [full WASM build](https://github.com/CrispStrobe/CrispASR/actions/runs/39151650113)
and actual-model/platform gates still require review. Subsequent legacy
Rust/Java changes require their own hosted results for the actual tip.
Older source checkpoints establish only their recorded component contracts.
Heavy CPU/native builds belong in GitHub CI; genuine GPU validation uses the
maintainer's external Kaggle guide. Keep private paths/toolchains/handoff data
only in ignored maintainer notes.

## Python shared-header expansion — hosted/model acceptance pending

Owner: maintainer coordinator. Source inventory and operation contracts are in
[Python API coverage](docs/python-api-coverage.md); completed implementation is
archived in HISTORY.md. The census now reads both shared public headers: 297 unique functions
(292 bound, three equivalent/compiled adapters, two retired native stubs).
The eight additional audio/container/channel functions were implemented and
probed; completed work is in HISTORY.md. Separate chat, other-language callable
coverage and real-model acceptance remain independent gates.

New operation/metadata probes pass with GCC and Clang. They exercise compiled C
ABI transport, actual native metadata functions, allocated-buffer ownership,
strided inputs, successful empty music results, replay-noise dimensions,
callback exceptions/re-entry, options layout, and compiled legacy adapters.
The combined local selection reports 63 passed, 92 subtests and 166 skips:
164 export-resolution cases await an explicitly supplied current library, and
two optional native-library checks await that build. GCC/Clang focused operation
and metadata suites each report 39 passed, 64 subtests and one native-library
skip. These skips are not hosted acceptance. The real-header helper also compiles locally. No full local library/model build
was attempted. Hosted CI checks the newly built library's exports/getters and
loads a helper compiled against that build. A fixture library is never the
native model acceptance gate.

Frozen implementation source `c12707299c5dcbca7bafe60a073688250238cdd9`
is on immutable `diag/python-api-20261018`. CI39120101432 finished with12
successful jobs and a Linux metadata-fixture failure after all2,033 native
unit tests passed; lint39120101602 passed all10 jobs. The failure was untyped
optional Parakeet word-alternative getters truncating a pointer, repaired by
ff3aeeefd; its observed failure/repair is in HISTORY.md. Do not rerun that old
source as though it were queued. Refreshed PR521 headab6c92cae's Linux unit job
39223238352 passes the full metadata/getter/export/helper step, including
297-function inventory and189 pytest cases/34 subtests. That is PR-source
integration evidence, not current-main release or real-model acceptance.

Next acceptance work, in order:

1. Check exact current-main full-library CI after subsequent wrapper changes,
   including legacy helper link/load, export smoke and native metadata. Preserve
   the corrected terminal outcomes above; do not repeat the old failed fixture. Do not accept a stale
   build discovered from another checkout. Local symbol-resolution checks skip
   when no current native library was explicitly supplied.
2. Exercise representative real models through Python: piano (including empty
   notes and MT3 programs), chords, beats at native/mismatched rates, tab
   emissions/mapping, conversion with pinned ContentVec/F0 and supplied replay
   noise, text processors, and TTS reference/streaming/disclosure. Compare
   structured outputs against the same native C ABI/CLI where that surface
   exists; TTS still requires ASR roundtrips and conversion its pinned reference
   recipe. Retain model/revision/source/input pins. Use hosted CPU/GPU capacity
   for heavy work; no performance/accuracy claim from transport fixtures.
3. #549 real Japanese Metal multi-window metadata acceptance now passes; keep
   other-platform/package/downstream validation open;
   the separate #217 forced-alignment worker does not validate that route.
4. Audit other language bindings independently, including the two new replay
   dimension getters. Python's complete census does not establish Go/Rust/Dart/
   Java/Ruby/C#/WASM parity. Do not ship a release from queued CI alone.

## #217 — hosted character CTC synthetic benchmark accepted; full/model gates pending

Owner: maintainer coordinator. Work is isolated on
`perf/217-alignment-cleanroom`; no third-party source or fitted data is copied
into this tree. The external review is pinned to Subtimer
`4a894d0e166191d41ebc734b61d68f5c531fb55b`; project-wide licensing is absent
from inspected metadata, while native/vendor notices have their own scope.
Read [the full comparison and clean-room record](docs/historical/alignment-cleanroom-2026-10-18.md).
No matched Subtimer/current-CrispASR accuracy/runtime benchmark exists here.
A research agent's unsupported DP-bottleneck and small-banding-quality-cost
claims were rejected; do not promote them as measurements.

A fresh implementation agent authored an independent opt-in character-CTC
workspace candidate using only our source and sanitized requirements. The first
T*S emission design was rejected on measured memory regression. Corrected
source SHA256 `d089c141e03fa61970a5136d1c8f9b92a05e306ae36bab78d6b8b8597b68cfd5`
uses unique-token emission columns, flat traceback storage and character-derived
word bounds. Only `CRISPASR_CTC_ALIGN_OPT=1` enables it; default stays unchanged.
Six acceptance tests pass: 240 randomized cases, 60 unique-optimum exhaustive
path checks, ties/repeats/Unicode, invalid/nonfinite cases, and memory acceptance.
Local synthetic measurements and their noisy timing ranges are retained in
`docs/historical/alignment-ctc-local-2026-10-18.json`; they do not measure encoder/model
or Japanese Parakeet/Canary subword performance.

Frozen character-performance source `567fc431e333d3cf1e2cb895f3d2895364e54328`
is on `diag/217-ctc-workspace-20261018`. Heavy CPU `39112822169` passes
all seven `alignment_ctc_bench.py --large` shapes on EPYC 7763/GCC13.3.
Independent audit verifies terminal source/hash pins, artifact/log equality,
all ratios/ranges and lower allocations/heap. Worker exact-output assertions
pass; this aggregate receipt retains no raw per-call output/timing arrays.
Gate-off/on median ratios are 1.124–1.215×; frozen-reference/on ratios are
1.089–1.388×. Largest T4096/V16384/W256: gate-off339.608ms → on288.506ms,
tracked heap275,986,668 → 7,556,460 bytes (36.52×). See
`docs/historical/alignment-ctc-hosted-2026-10-18.json`; this is synthetic DP, not encoder,
model, RSS/VRAM or Japanese/subword acceptance. CI `39112823566`
was also dispatched for that source and is retained while progressing.
Queued lint `39112825190` was superseded/cancelled in favor of combined-source
lint below. Audit terminal results and verify source pins before accepting any
hosted result. It compiles only our model-free kernel
and frozen own baseline; no external models/code are needed. Audit all receipt
output parity, kernel-time distributions, actual allocation counts and tracked
peak heap, source hashes, compiler/CPU/flags, and the independent test suite.
Audit the retained character CI and later frozen integrated production
CI/lint separately; this synthetic job is not full-library acceptance. Keep
those branches immutable while documentation checkpoints advance. No default promotion until
those gates and representative model alignment acceptance are satisfied.

## #217 subword CTC failure contract — implemented; full CI pending

The tiny production-tokenizer/DP test was first observed returning success with
zero times for a two-label transcript and one frame. The independent CTC
feasibility assertion failed before the fix. `canary_ctc_align_words` now rejects
unreachable/nonfinite terminal states and invalid ABI/token-ID inputs. All four
contract tests pass, including repeated-label minimum frames and known valid
centisecond bounds. No encoder, tokenizer or valid-path DP arithmetic changes;
this is a correctness fix, not a Japanese Parakeet speedup. CTest/ordinary CI
run this suite. The independent character-kernel source remains unchanged.

Frozen combined source `deaf9c010acce1410f470d6df421d0f77ff16be2` is on
`diag/217-alignment-contract-20261018`: CI `39113844983` and lint
`39113846749` are dispatched and queued. Audit this combined source separately
from the immutable character-only performance source. Do not treat a synthetic token fixture as
proof of long-form model accuracy. Character-path nonfinite-input policy still
follows its frozen baseline; any future policy unification needs separate
valid-output and error-contract acceptance, not a concealed speed change.

## #217 subword workspace — implemented; hosted/model acceptance pending

Independent own-code candidate uses `CRISPASR_SUBWORD_CTC_ALIGN_OPT=1`, default
off. It normalizes one vocabulary row at a time and flattens traceback while
retaining the frozen `deaf9c010` tokenizer/DP return/text/timestamp behavior.
Five tests pass under GCC and Clang (240 seeded random, 60 unique-optimum
exhaustive tiny paths plus edge/memory cases); four contracts exercise both
gates and six character tests remain green. CTest/CI run all suites. The local
receipt and scope are in the [comparison](docs/historical/alignment-cleanroom-2026-10-18.md#independent-subword-workspace-experiment).
Frozen source `a1434c9b2d2484e7af572b577cb1f53127662775` is on
`diag/217-subword-workspace-20261018`. Heavy CPU `39115492999` runs
`tools/ci-heavy/alignment_subword_bench.py --large` (NumPy; timeout30), CI is
`39115494689`, lint is `39115496152`; all three are queued at dispatch audit.
Audit source and baseline hashes, exact return/text/bounds, successful complete alignments,
allocation/heap reductions and timing distributions. Never promote from these
synthetic results alone. Keep input/encoder memory and model quality separate.

## #217 Japanese alignment — CPU proof audited; long-form/hardware pending

Completed CPU evidence is archived in HISTORY.md and the
[verified audit](docs/historical/alignment-japanese-audit-2026-10-19.json). Keep both workspace
gates default-off. The measured DP saving is0.055ms beside4591ms encoder time;
do not prioritize another DP micro-optimization as an end-to-end speed fix.
Issue550 long-window CUDA worker is completed; do not repeat it. Next: inspect release bundles
[run39212307554](https://github.com/CrispStrobe/CrispASR/actions/runs/39212307554)
at source62e598ca0, then long-input CUDA memory, seams and labeled boundaries.
Short-clip parity does not establish long-form acoustic accuracy, Japanese ASR
multiwindow ownership (#549), original encoder parity or Subtimer superiority.

## #217 clean-room accuracy follow-ups — bounded next tasks

The actual Japanese model profile is audited (see HISTORY.md). Preserve the
measured encoder/postprocessor split when choosing another optimization. Preserve native centisecond-output/tokenizer parity and
keep original behavior available for A/B. The unchanged per-word path scans
remain a separate candidate only if profiling justifies them; do not conflate
invalid-path semantic repair with a speed change.

Long-form accuracy requires audio/text anchors before changing the current
proportional Qwen chunk assignment or extending the new bounded CTC encoder windows. Build a small
licensed Japanese boundary-labeled corpus with long pauses, music, repeated
phrases and subtitle omissions. Acceptance: no dropped/reordered lines, correct
absolute offsets, reported median/p95 boundary error and coverage, and bounded
RAM/VRAM on a long clip. Evaluate optional VAD/vocal signals and independently
trained boundary refinement against a raw-audio baseline. No external fitted
parameters, exact feature recipes or source may be reused. The comparison and
licensing/provenance constraints are in
[the audit](docs/historical/alignment-cleanroom-2026-10-18.md).

## #549 — token ownership and real Metal metadata accepted; packaging pending

Frozen source `741141118335643adc7d29b61bc95e1d77c7f43a` passes all 13 CI jobs
(`39104662258`) and all 10 lint jobs (`39104663518`). Independent audit checked
terminal metadata, exact source and the complete Linux unit log: 2,033 CTest
unit cases pass; the exported-C-ABI/Python metadata suite passes all 11 tests,
including all 30 legacy/shared splitter fixture/window combinations. Later main
commits through the initial #217 claim alter documentation only relative to
this validated source. Preserve this frozen proof; do not repeat it merely
because documentation tips differ. The API fixes remain unreleased in v0.8.43.

The remaining native packaging gaps are now implemented: legacy overlap
splice/dedup and Japanese slice/gap-fill carry owned token blocks with retained
words. Shared non-JA gap repair uses native grouping indices instead of inclusive
timestamp matching, which duplicated both words' tokens when frames were tied.
The grouping rules live in `src/core/parakeet_word_ranges.h` and are used by
native word construction, so no word-string/timestamp reconstruction is needed.
No public POD layout, decoder, window decision or Python constructor changed.

Six new tests pass under GCC and Clang, after first reproducing empty tokens
on JA/legacy routes and tied-frame duplicate tokens in shared repair. They
exercise production grouping/window/repair functions, opaque getters and the
Python result reader: equal frames, UTF-8 fragments, exact F32 probabilities,
BOS/empty ownership, offset slices and token-less native words. Grouping also
matches frozen own `08aea3537` word text/times/probabilities on 2,000 cases.
The expanded metadata suite now has 12 tests, including a native-grouped
continuation token sharing the next output segment's first frame. Both legacy
and shared output splitters keep it with its own word, without changing word
segments; all 18 metadata/window tests pass under GCC and Clang. CTest registers the new suite;
ordinary CI runs it with `CRISPASR_LIB` against actual exported getters.
`docs/contributing.md` now requires explicit result-field and multi-route wrapper
parity and corrects the claim that Dart symbol checks auto-discover new APIs.

Keep the issue open for full-library/platform follow-up and reporter validation.
The first window-only source `aaa5f21a9` has CI39117671481/lint39117673618
pending. Final integrated source `ebf69c43721e56a43b7c8f637873902098f523ba`
is frozen on `diag/549-token-boundaries-20261018`, CI `39117921894` and lint
`39117923785` now both pass. The actual exported-getter step for both suites
is independently audited. The real licensed Japanese Metal short/long/short
Session matrix also passes 17/123/17 tokens, exact returning-short metadata and
a distinct native short-token byte/time/probability comparison.
[Live receipt](docs/historical/ja-session-metadata-live-2026-10-20.json). The CTC-aligner profile for #217 is a different model
route and cannot stand in for this transcription validation. Generic CLI
postprocessing (`examples/cli/crispasr_gap_fill.h`) retains timestamp-based token
filtering for multiple model families; audit that separately before claiming
full long-form CLI/session token identity, without applying Parakeet-specific
grouping to unrelated tokenizers. Record token-less native routes explicitly.

See [session metadata](docs/python-session-metadata.md) for `None` timing,
byte-fragment handling, cache-release lifetime and older-library compatibility.
The process-wide aligner cache wrapper is tested, but no real VRAM-counter
measurement has been made. No release/tag is implied by the CI acceptance.

Updated 2026-10-18. This file contains current actions, dependencies and a
legacy audit inventory. Completed work and experiment receipts live in
[HISTORY.md](HISTORY.md#plan-reconciliation-archive--2026-10-15).
The archived snapshot preserves detailed older requirements without treating
its stale status labels as current facts.

## Working protocol

Read the sibling development guide and AGENTS.md. Work in a separate worktree;
large models, builds and artifacts belong on the maintainer storage volume.
Read ignored CLAUDE.md for actual local paths; never commit machine-specific
operational details. Tracked receipts use portable location aliases. Before coding,
add a dated CLAIMED block naming the owner, worktree, scope and acceptance
criteria, and push that claim to main. Remove the claim when the work lands.
Read complete issue/PR threads before acting. Audit old notes against code.

Use pinned original Python → conversion/quantization → dump-reference → diff
with cosine, norms and relative error → actual decoded-output acceptance.
Only then profile or promote defaults. Preserve failures and source/input
hashes; use distinct run directories so failed evidence cannot be overwritten.
Follow docs/contributing.md for all affected surfaces, including the C ABI.

Historical checkpointad2998ab passes all13 general CI jobs (38855803220), Ruby,
secret scan and Pages. Native-fork worker5417ed6d passes all10 lint jobs
(38855559850) and heavy38855559740; independent auditc0fac6c1 passes642
payloads. These are exact-source checks. The diagnostic integration and future
release tips require fresh CI; older documentation-tip queues are superseded.
Last published release is v0.8.43; v0.8.44 remains untagged. Dia sampling controls are integrated from candidate7d0f68f3 (all13 CI and10
lint jobs passed); all non-document source bytes are preserved. Main f9eef29e passes all13 CI38974250081 jobs. Later documentation tips
require applicable fresh CI. Echo observer integration preserves the complete non-document
tree of3a13e0c6 (all13 combined CI and10 lint jobs passed). Routing controls do
not establish native speech acceptance.


## Handoff and immediate priorities

See [the maintainer handoff](docs/maintenance-handoff.md) for every active front's
entry points, bounded next action, acceptance gates and external dependencies.
No task remains assigned to a departed agent. Follow the approved ordered
roadmap below:550 GPU completed, wrapper/package closure, PR521 license review,
release, then bounded numerical
work. Preserve accepted Echo audits and current-main547 fixes; research and
hardware waits remain separate.

Issue #547 now has the streaming utterance-open throttle fix on main in
`dd14099c3`, including literal boundary tests and the dedicated
`tools/ci-heavy/issue547_nemotron_multiturn.py` validation worker. Do not repeat
the old stale-cache hypothesis or reimplement that fix. Audit the worker's
existing runs/results and fresh/reused-session/multi-turn acceptance before
claiming full hardware validation; this #549 task did not execute that worker.

## Ordered roadmap — maintainer-approved execution order (2026-10-19)

1. **Issue550 GPU acceptance is completed.** Preserve the audited T4 proof and
   closed issue; no repeat dispatch is needed. Continue wrapper/package work.
2. **Close wrapper acceptance and implementation gaps.** Audit green Arabic
   CTC/Java39214206020 and binding contracts39215435420; bounded Go CPU model
   lifetime/re-entry is also audited. Generic speaker operations in all eight
   wrappers are reviewed. Continue owned low-level adapters, other model modes,
   language lifecycle/platform checks and the remaining callable ledger. Validate
   Japanese Metal multi-window metadata for549 is accepted separately from
   CTC forced alignment; complete published-wheel/downstream validation next.
3. **PR521 is held for bundle license review and speech acceptance.** NC main
   weights are distinct from the bundled Boson tokenizer conditions. Do not
   merge full runtime support merely because CI passes. Review the complete diff,
   repair genuine CI failures and preserve contributor history. Merge only
   accepted scope; keep original/native clone failures and quality gates intact.
4. **Release accepted work.** Require exact-tip CI/lint and package/wheel
   relocation/load checks, reconcile comprehensive notes, then use
   scripts/bump-version.sh, publish and inspect assets. Do not hold accepted
   release scope indefinitely for unrelated research; do not advertise failed
   or hardware-unvalidated model paths.
5. **Resume bounded numerical work.** Echo: one identical-input projection
   precision counterfactual. MiMo: actual failing71-row Q/K/V, masks, scores,
   probabilities and pre-output capture. Dia remains experimental until real
   speech acceptance; do not repeat completed sweeps without a new hypothesis.
6. **Maintain the handoff continuously.** Archive completed work in HISTORY.md;
   give every PLAN.md item an entry point, exact next action and acceptance gate.
   Separate reporter/hardware/gated-asset waits; keep machine details ignored.
   Drama3 stays deferred outside active/release scope.

Documentation-tip CI runs can be superseded by newer pushes. Inspect Actions
for the actual main SHA; green component workflows or older cancelled runs do
not replace exact-tip general CI. Hardware waits do not justify
repeating accepted diagnostics. GH handles heavy CPU/builds; Kaggle handles
actual supported GPU tests per the external guide. Neither supplies Ascend.

## Dia correctness — current dependencies

The completed reduction/cache/RNG, broader matrix and recognition investigations
are in HISTORY.md and their pinned machine receipts. Current active claims carry
the remaining source-quality counterfactuals.

1. Direct-original CPU construction/recognition is audited and fails speech.
   Original README F16 CUDA is independently audited: both fox seeds pass,
   legacy and both hello seeds fail; both recognizer positives pass. Its81 payloads
   and82 anonymous public downloads are verified. The subsequent trajectory
   experiment is fully audited (201 payloads/202 anonymous checks): no invalid
   codes were clamped, and all three quiet paths use one common nine-book vector
   for over99% of frames. Both hello and fox cases pass exact observer identity;
   first legacy OFF0 differs slightly, so the global observer gate remains failed.
   The matched CUDA precision/recipe experiment is independently audited:261
   payloads and262 anonymous file checks. Precision/recipe alone did not recover
   the three quiet cases; see docs/dia-cuda-factorial-results-2026-10-16.json.
   Categorical draw placement is isolated and independently audited: CUDA/CPU/
   CUDA on the same model changes fox quality0/1/0 WER, with neural compute
   fixed. First applied feedback differences follow byte-identical probability
   inputs. All other cases still fail speech. The final-BOS source counterfactual is independently rejected for speech
   recovery (260payloads/261anonymous downloads): same three failures, fox controls
   pass. Preserve original behavior. No new full-model sweep is justified by
   those hypotheses; advance independent Echo owned-input quantization attribution.
   See HISTORY.md and both result receipts.
   Existing full-F32 broader quality fails
   on both recognizer architectures; numerical/code equality alone is insufficient.
   Once bounded fixed cases pass unchanged speech gates, test the existing
   heads-only-F32/F16 neural export with full-F32 positive controls before GPU/Q4
   and profiling. Preserve source/physical audits and all earlier failures.
   Do not add token overrides, change seeds or relax thresholds as a fix.
2. Earlier module Q differs6.90e-7 from actual full-model Q despite exact keys
   and normalized input. Preserve that qualification. A new intervention must
   use the actual original tensors and native identity controls, with cosine,
   magnitudes, relative errors and decision-margin effects. Do not repeat the
   rejected qualified-Q, norm/head/late-layer/source-RoPE sweeps unchanged.
3. Any production candidate must run current runtime without reference injection,
   preserve controls, and pass real speech on bounded additional prompts/seeds,
   then actual GPU/Q4 and performance. Single-seed diagnostic recovery is not
   production acceptance. If no new operator hypothesis is justified, advance
   independent CANN preparation instead of repeating accepted diagnostics.

## CANN — hardware follow-ups after merged PR544

Implementation and official SDK builds are integrated. The portable Moonshine
validator and its original/CPU/CLI/Session proof are ready. Remaining actions:
1. Run the fixed full-model fixture on physical310P/910B, preserving actual
   placement/offload, original stage/magnitude metrics and decoded transcripts.
2. Prove composed scheduler fallback: supported CANN work plus rejected BF16
   CPU operations, with actual output checks; physical310P required.
3. Prove actual ACL capture/replay, repeated calls and changing shapes on910B.
4. After floating-point acceptance, validate quantization and then profile.

Maintainer owns the preparation; contributor hardware help is optional. No
further rerun of the already accepted rotary repair is requested. Details and
commands: [CANN TODO](docs/cann-todo.md) and
[portable validator](docs/historical/cann-model-validator-preparation-2026-10-14.md).

## Remaining PRs — scoped salvage and acceptance

PR521 review branch now incorporates current main without rewriting contributor
history. Python helper conflict is resolved with main's tested matrix; ggml
retains main's newer CANN fixes. The signed128-minimum formatter overflow is
fixed after an observed UBSan failure, with a permanent CI regression. Inspect
fresh branch checks; unchanged original/native clone speech failures still
prevent full-runtime/default acceptance. Do not repeat accepted utility proof.

- **PR492 / MiMo:** author KokerZhou closed the PR unmerged on2026-10-15;
  it is no longer an open merge candidate. Independent opt-in mel/cached-decode pieces are merged.
  Dynamic Q8 activation amplification is measured; the existing F32 cache-read
  hypothesis is independently rejected and archived in HISTORY.md.
  [Four-arm result](docs/historical/mimo-kvread-ab-2026-10-15.md): hidden/logit error remains
  5.73%/5.13%, with a0.5% gate. The default Q4 original-stage gap also remains.
  The completed64-cell tiny attention audit proves a scalar F16 value-accumulator
  error (0.460% on constant V, exact repeated-half prediction). The old71-row
  prefill takes the F32 tiled path, so this is not its repair. Diagnostic
  source02bd2aee remains frozen on its branch. All10 lint39003053402 and13 general CI39003202215 jobs pass;
  the unchanged diagnostic source is integrated on main. Actual matching Q/K/V,
  mask, scores/probabilities and pre-output-projection capture with OFF/ON/OFF
  identity is next under a fresh claim; the old full capture has no Q/K/V.
  See docs/mimo-attention-operator-2026-10-17.md. Do not repeat old BLAS/cache-read sweeps.
  CANN RVQ/F16 embedding and ggml NZ need physical Ascend and PR-only/NZ-only/composed attribution. The measured
  full tokenizer path was7–10% slower; do not enable it by default. Review the
  complete latest thread before proposing any further split or merge.
- OmniVoice diagnostic data recovery is complete: the recorded HF pin returns
  404, but exact original GH artifact11600052124 has been recovered, independently
  array-audited and permanently republished with a verified anonymous download.
  See HISTORY.md and docs/omnivoice-clone-diagnostic-2026-10-09.json. Remaining
  work is runtime/input/precision/quality diagnosis, not another proof search.
- **Issue518 / PR521 / OmniVoice — MERGED 2026-10-22 (`0cd80cba2`):** audio and
  text utilities match upstream `08be0b4`, behind env gates with
  `CRISPASR_OMNIVOICE_AUDIO_LEGACY=1` restoring the prior pipeline. The2/9 clone
  WER campaign reported seven passing `window` arms. Audit subsequently found
  outcome-based exclusion of eligible `station`; that subset is not complete
  cloning acceptance. The corrected harness includes all integrity-eligible
  references, and a fresh full run remains required. ASR agreement on `fox`
  raises a fixture concern without proving the cause of all cloning errors.
  Merge scope also closed the helper-test conflict, the stale ggml pin, the
  signed128-minimum UBSan overflow and the CLANG64 `<cstdlib>` include. Receipt
  `docs/historical/omnivoice-clone-reference-integrity-2026-10-22.json`, method in
  `docs/omnivoice-audio-processing.md`. The pre-merge head passed CI, Lint,
  Build WASM, Docker Smoke, Binding ABI contracts, Bindings (Ruby) and
  Regression (nightly) on `ed6c21ecb`; contributor history is preserved in the
  merge commit.
  The onset/click is NOT proven repaired — fades attenuate a transient but do
  not remove every click. Do not reuse `fox.wav` as a cloning reference and do
  not relax the clone gate.
- **Breeze:** accepted guided/voice/server fixes are merged. Investigate canonical
  BF16 grid equality (0/8 despite passing scoped speech controls) and unresolved
  model-quality reports from merged contributor PR530 only with a new hypothesis.

## Quantization — no unproven model promotion

- **Index-Echo / issue485:** higher-precision accepted paths remain available.
  All four plain recipes and three calibrated recipes fail complete acceptance.
  Collection covers all96 FFN statistics and sensitive source F32 tensors are
  preserved. Archived actual logs are now inventoried: plain first crosses the
  global0.999 summary floor at block0; guarded at11/12/15 and middle at12/12/15
  for JFK/JFK-tail/ZH. These differ from native per-stage quantized gates: JFK's
  literal FAIL remains teacherforced_logits (.967303/.983031/.979347).
  Block4 middle cosine0.999908 passes0.999. Completed owned-input attribution is
  audited; see HISTORY.md and [the receipt](docs/historical/index-echo-owned-ffn-audit-2026-10-19.json).
  Next evaluate one bounded projection intervention on identical saved inputs
  before a full decoder: isolate down-projection layer25 (worst owned-down-input
  error,8.00% on Mandarin), then inspect gate/up error if MLP output remains
  outside the existing gate. Cumulative first crossings are not causal layers.
  Audit previous guarded Q8-down/F16 recipes before choosing another override;
  avoid repeating an already rejected recipe. Use a fresh public prefix and
  retain original/converted/physical-type/input pins and exact byte-size cost.
  Only after component improvement run complete EN/ZH stages/magnitudes, cache,
  CLI/subtitles/session and TTS-ASR plus actual GPU validation. No accepted Q4
  recipe or runtime arithmetic repair follows from attribution.
- **Echo performance:** scheduler opt-in has narrow T4 evidence; new CPU/Metal/
  CUDA A/B must establish output and timing before any default change.
- **Nemotron:** accepted experimental hybrid T4 controls are not compact-Q4
  acceptance. Hybrid is1,179,413,408 bytes, only14,155,776 Q4 bytes and8.3%
  smaller than F16. Audit promotion/package scope separately; seek meaningful
  compression with preserved original stage/speech/session/streaming behavior.
  No speed or production pin is established. Evidence:
  docs/nemotron-position-q4-gpu-2026-10-10.json.

## Reporter, hardware and asset dependencies

- **Issue532 / Canary:** CPU/T4 reuse/allocation controls did not reproduce the
  crash; await original recording/device details. Do not apply a speculative fix.
- **Issue483:** original matched CUDA/MMQ packages and complete vendor notices
  are [preserved in a verified maintainer draft](docs/historical/cuda126-mmq-preservation-2026-10-26.md).
  Actual GTX1660/MX150 correctness and timing remain external.
- **Issue461 / VoxCPM2:** profile VAE setup and Vulkan LocDiT on relevant hardware.
  Eight-step RTF1.019 and poorer reported quality do not justify changing the
  ten-step quality default. Issue478 Windows/Khmer retest, Q4 batching and
  other-ISA/GPU coverage remain separate; verify current issue state first.
- **Issue456 / Nyra:** deferred. The [published model/output terms](https://huggingface.co/nyralabs/nyra_forced_aligner_en/blob/9c8e724072261c30641a91ca365ce4f7b4ae7f83/LICENSE.md)
  restrict production/operational and hosted/API uses, not just paid use, and
  cover lexicon/configuration assets and generated outputs. MIT code does not
  remove those restrictions. A production port requires appropriate written
  rights; no such agreement was found or obtained. Research-only redistribution
  also requires separate review of recipient/output obligations and unresolved
  bundled WavLM provenance. No port, conversion or registry entry is queued;
  espeak-ng/phonemizer bundling requires its own GPL compliance review.
- **Dolphin:** merged original-reference fix does not prove reporter Kylin/GCC9,
  private clips or GPU behavior. Obtain those inputs if further work is needed.

## Historical release preparation — superseded by published v0.8.45

The old v0.8.44 preparation checklist is no longer pending work. The prepared
successor v0.8.45 was published and its native/Python/Rust/Dart packages and
notices verified on October 26; see the
[publication receipt](docs/historical/release-publication-2026-10-26.json) and the current
published section above. Historical v0.8.44 notes remain in
RELEASE_NOTES_v0.8.44.md. Later source fixes, including Nemotron default selection,
are not retroactively present in the published tag or binaries.


## Deferred outside active/release scope — Drama3 (issues545/546)

Maintainer decision (2026-10-19): exclude this backend from the active roadmap
and next release because its custom licence is unsuitable for the intended
scope. Do not start conversion, download weights or claim support. Reconsider
only after an explicit scope/licence decision; the technical notes below are
retained for handoff, not an active task.

Issue546 requests the same backend, explicitly including GGUF conversion and
zero-shot voice cloning; track those requirements together with545. Cloning
must cover the original reference-audio processing and actual CLI/C ABI paths,
with closed-loop decoded-speech/voice validation before claiming support.

Metadata/source/licence triage is complete; no weights or implementation started.
Pin fishaudio/drama-3 e0a9218aa48888270f06ef1ed9ea67622c0e17d2 and read the
actual inference/codec blueprint before defining a port. Scope includes40 hybrid
layers,256-expert/8-selected MoE, shared experts, zero-centered norms, partial
RoPE, a4-layer ten-codebook MTP decoder and codec, with FP8-block/BF16 weights.
Fish Audio Community Licence is custom, not Apache2; establish redistribution
terms before conversion/public artifacts. Audit ggml operation/backend coverage,
then original conversion/dump/diff/decoded-speech pipeline on suitable hosted
hardware, sensitive-tensor quantization and full contributing checklist. Keep
behind correctness work and avoid an unbounded35B download on this VPS.

## Research backlog — independently trained EmbeddingGemma speech reader

User requested a concrete task for an independently trained reader, word bank
and reproducible decoder. This is a research backlog task, not implemented
support or a claim to reproduce the author's unpublished reader. Feasibility:
`docs/historical/embeddinggemma2-asr-feasibility-2026-10-13.md`.

1. Pin and read the official encoder implementation and preprocessing. Compare
   temporal audio-tower features with features after the shared backbone; do
   not substitute pooled recording embeddings for time-resolved features.
2. Freeze the encoder and train a small temporal CTC reader. Compare ordinary
   subword CTC with embedding-matching word CTC on the same features/data so
   the shared word space's contribution is measurable. CTC is our proposed
   recipe, not an assertion about Shahir's implementation.
3. Build a licensed, versioned word bank with normalization, stable IDs and
   pinned text embeddings. Start around 30–50k words; define handling for
   names/numbers/unseen words. Exclude held-out evaluation transcripts from
   vocabulary construction. A 229k-by-768 FP16 bank alone is 351,744,000 bytes;
   measure dense scoring cost before choosing compression or approximate search.
4. Specify the exact decoding contract: sample rate, feature layer, chunking,
   overlap, score normalization/temperature, blank score, CTC repeat/blank
   handling, boundary merging and optional beam search. Begin with greedy CTC;
   keep punctuation separate from the initial acoustic-quality experiment.
5. Pilot on LibriSpeech train-clean-100 with separate development/evaluation
   sets, then expand training only after measurable held-out progress. Report
   WER on clean/other speech, silence and out-of-vocabulary behavior, memory
   and feature-extraction/training throughput. Do not promise the author's
   ~5% WER, 288x speed or eight-H100 training time on a Kaggle GPU.
6. Use Kaggle for heavy encoder extraction/training and GH CI for conversion,
   reproducibility and integration checks. Keep datasets/features/checkpoints
   under `${CRISPASR_STORAGE}`; checkpoint produced artifacts before later risky work.
7. After a useful Python prototype, port via convert → reference dump → stage
   diff with cosine AND magnitudes → decoded transcripts. Establish F32/BF16
   and Q8 baselines before lower quants; validate CPU/Metal/actual CUDA and all
   `docs/contributing.md` integration points, including C ABI and bindings.
   Reuse audited CrispEmbed components where applicable; its typed-decision
   backend is not already a temporal audio encoder or speech reader.

Planning estimate, not a delivery promise: 1–2 engineering weeks for a research
prototype, then several more for training iterations and a validated ggml port.
First decision gate: useful held-out transcripts from a small frozen-feature
reader versus its subword baseline. Reader/checkpoint/bank provenance and
redistribution terms must be documented. Matching advertised quality/speed is
an experimental target, not a release acceptance assumption.
Sources: https://arxiv.org/abs/2306.11473,
https://huggingface.co/google/embeddinggemma-2/blob/main/config.json,
https://www.openslr.org/12. Upstream ggml support and base Apache-2.0 weights
already exist; the app's own reader assets were not located in our source check.


## Older backlog — audit against current code before scheduling

These are preserved requirements, not assertions that an old OPEN label still
means missing implementation. None are active claims. Consult the complete
[archived requirements](HISTORY.md#plan-reconciliation-archive--2026-10-15),
search the heading below, audit code/tests and current issue state, then either
schedule the remaining action here or record completion in HISTORY.md. Never
repeat completed work solely because the archived note says OPEN.

- Audit remaining scope: live transcribe + translate: follow-ups
- Audit remaining scope: §247 shared scheduler profiler, then hidden-F16 audit
- Audit remaining scope: (optional) — #416 Sidon quant: CUDA MMQ arm
- Audit remaining scope: #387 Quds follow-up
- Audit remaining scope: canary follow-ups (after #375, record in HISTORY)
- Audit remaining scope: vibevoice-asr 7B answers "[Silence]" on time-stretched audio
- Audit remaining scope: vibevoice-bitnet advertises caps its default output cannot support
- Audit remaining scope: TQ2_0 has no Metal kernels: BitNet models are silent on GPU
- Audit remaining scope: carried out of the #316 round-2 work
- Audit remaining scope: follow-ups from #332 (Rust/Dart diarize ABI, landed 2026-08-05)
- Audit remaining scope: #326 diarization: the count estimator is the last accuracy item
- Audit remaining scope: stamp the kartoffel repos (needs ~15 GB free; this box has 11)
- Audit remaining scope: Why not the hash-chained log the sibling projects built
- Audit remaining scope: follow-ups from #300 / #308 (landed 2026-07-27, see HISTORY)
- Audit remaining scope: #316 Kokoro G2P English residue
- Audit remaining scope: VAD + mel front-end parallelization campaign (#305 → fleet-wide)
- Audit remaining scope: #296 mel-band-roformer — follow-ups PROMISED on the issue (do not drop)
- Audit remaining scope: #266 follow-up — hoist speaker orchestration into the library (PARKED, LOW)
- Audit remaining scope: Untrusted-input parser hardening — follow-ups
- Audit remaining scope: Diff-harness extensions (detail for "Ready to take" #5 and #6)
- Audit remaining scope: CometBeat handoff — singing-voice-conversion vocoders (OPEN, NOT STARTED)
- Audit remaining scope: Gemma-4 12B (gemma4_unified) ASR support (OPEN)
- Audit remaining scope: crispasr-diff harness extensions — catch decode-policy / quality bugs (OPEN)
- Audit remaining scope: #201 follow-up — generate a TADA voice ref from audio+transcript at query time (C-ABI + server DONE gated; roundtrip pending)
- Audit remaining scope: §192 follow-up — native-Vulkan TADA garbled output FIXED (codec on CPU)
- Audit remaining scope: moss-transcribe follow-ups (OPEN, LOW)
- Audit remaining scope: #218 qwen3-asr long-audio root cause — quantized audio tower + prompt contract
- Audit remaining scope: #218 glm-asr long-form blueprint parity — instruction + multi-window
- Audit remaining scope: #218 qwen3-family rebake + CUDA validation — kernel results
- Audit remaining scope: Priority ordering
- Audit remaining scope: §214 follow-up — chatterbox T3 batched-CFG (B=2) — deliver the win (OPEN)
- Audit remaining scope: §215 — batched-CFG (B=2) for the remaining TTS backends (OPEN)
- Audit remaining scope: §210 follow-up — shape-stable bucketed decode for remaining LLM/AR backends (CUDA-graph capture) (OPEN, CONDITIONAL)
- Audit remaining scope: §219 — more permissive audio input formats for crispasr_audio_load
- Audit remaining scope: §166 follow-up — WASM `asr*` session surface needs a build-verify
- Audit remaining scope: §175 Surgical DRY — share pure helpers across the CLI/library boundary
- Audit remaining scope: §177 VibeVoice #171 — remaining layer: RDNA4/RADV quantized-matmul path (OPEN)
- Audit remaining scope: 40. More Moonshine model variants
- Audit remaining scope: 97. More Parakeet variants
- Audit remaining scope: 98. Hotwords / contextual biasing (OPEN)
- Audit remaining scope: funasr — perf follow-ups (LOW priority, not blocking)
- Audit remaining scope: §187 Cross-runtime embed fast path sweep
- Audit remaining scope: §201 Kaggle CUDA backend failures — full sweep 2026-06-20 (OPEN)
- Audit remaining scope: 51c. MiMo-V2.5-ASR F16 step decode — open (validation only)
- Audit remaining scope: 52. Qwen3-TTS
- Audit remaining scope: 54-follow-up. granite-speech-4.1 plus speaker labels + word timestamps — open
- Audit remaining scope: 56. Kokoro multilingual phonemizer (espeak-ng)
- Audit remaining scope: 57. Commercial-friendly TTS backend expansion (OPEN — Phase 1 done)
- Audit remaining scope: Ecosystem expansion (lower priority)
- Audit remaining scope: 59. Cross-binding C-ABI parity
- Audit remaining scope: 65-residual. JS / emscripten word-accessor surface — open
- Audit remaining scope: 67. Deferred follow-ups carry-over (mid-May 2026 session)
- Audit remaining scope: 70. Streaming TTS via chunked VAE decode (latency win, vibevoice / qwen3-tts)
- Audit remaining scope: 75-followups. /v1/audio/speech OpenAI round 2 — open
- Audit remaining scope: 168. GPU scheduler migration — gallocr-only backends
- Audit remaining scope: 91. CrispASR CLI features missing from CrisperWeaver (OPEN — parity gaps)
- Audit remaining scope: 92. All-backend regression suite (nightly CI)
- Audit remaining scope: 95. IndexTTS-1.5 Chinese TN — binary alternative to the Python `wetext` hook (OPEN)
- Audit remaining scope: 102. RapidTP-Aligns — dedicated NN timestamp predictor (survey)
- Audit remaining scope: 104. Stateful frame-streaming TDT decode for parakeet long-form (issue #89)
- Audit remaining scope: 105. WhisperX word alignment models — wav2vec2 CTC zoo
- Audit remaining scope: 106. TEN-VAD — low-latency cross-platform VAD
- Audit remaining scope: 125. Issue #125 — multi-backend bug sweep from montvid (mostly DONE — validation + longer-term threads open)
- Audit remaining scope: 127. Coverage gaps from the 2026-05-26 overlap-save sweep close-out
- Audit remaining scope: §136 — funasr CUDA !-loop fix (issue #125)
- Audit remaining scope: §139 Beam search — remaining ASR backends (issue #136 follow-up)
- Audit remaining scope: §TTS-PROV: TTS AI-provenance compliance (watermark, C2PA, consent, disclaimer)
- Audit remaining scope: 156. Permissive G2P phonemizer
- Audit remaining scope: §167 — Beam search, MAES, and KV caching expansion
- Audit remaining scope: §176 Runtime optimization pass — 18/20 DONE, 2 open
- Audit remaining scope: §ARK — ARK-ASR-3B support (⚠️ EXPERIMENTAL / WIP; branch feat/arkasr-3b)
- Audit remaining scope: llama.cpp comparison — perf-tricks to adopt (ANALYSIS / OPEN)
- Audit remaining scope: 221. Issue #89 hardening + v0.8.8 release
- Audit remaining scope: §222 Aligner model expansion — permissive-license fleet (OPEN)
- Audit remaining scope: §224 Issue #222 follow-ups — silero LID ggml graph (DONE) + diarize/firered CPU perf (OPEN)
- Audit remaining scope: §226 irodori-tts GPU + codec GGUF fix
- Audit remaining scope: §227 starling comparison — CUDA decode-loop optimization options (OPEN)
- Audit remaining scope: §229 transcribe.cpp perf audit — vs handy-computer/transcribe.cpp (OPEN)
- Audit remaining scope: §232 qwen3-tts code predictor perf — fuse 15 graph dispatches into 1 (#245, OPEN)
- Audit remaining scope: §235 next perf targets — triage after the §232 qwen3-tts sweep (2026-07-11)
- Audit remaining scope: §234 omnivoice — persistent step graphs + silence root cause
- Audit remaining scope: §232 transcribe.cpp parity — close the RTF gap (mostly CLOSED; residual items OPEN)
- Audit remaining scope: Runtime speedup roadmap (2026-07-11 cross-repo sweep)
- Audit remaining scope: §247 roll #81 techniques out to the other runtimes (OPEN)
- Audit remaining scope: §248 — audio.cpp competitive benchmark + gap analysis (IN PROGRESS)
- Audit remaining scope: §249 WebRTC VAD — remaining follow-ups (backend DONE 2026-07-19, see HISTORY)
- Audit remaining scope: §250 — Music transcription backends (piano_transcription + Basic Pitch)
- Audit remaining scope: §251 — Music-analysis infrastructure + the remaining CometBeat roster
- Audit remaining scope: §252 — CosyVoice3 native-Vulkan: fp32-accumulation lever (LOW priority, likely dead end)

Legacy publishing/bootstrap claims require checking current registry/workflow
state; do not assume the old404s or missing credentials remain true. Legacy
MioTTS/Glint integration and completed native sample-rate/binding/UTF-8 fixes
are historical, not work to reimplement. Consumer-requested binding expansion
can be scheduled after checking actual exported and wrapped surfaces.


## Echo9B owned-input FFN follow-up

The completed v2 worker/source/payload/tensor/metric audit is archived in
HISTORY.md. Do not dispatch it again to recreate accepted attribution. Next
precision task and all quality gates are in the quantization section above.
Keep all seven rejected recipes rejected and production defaults unchanged.
Component proof does not reproduce a whole original model trajectory or
validate speech, Q4 quality, GPU portability or speed.

Final-source `2c1183840` CPU qualification completes: all 13 CI jobs 40220615289,
all 10 lint jobs 40220617460, ordinary CSM40220623142 and default-vs-explicit
matrix40220619349 pass. The latter is a portable host, with all8 corrected
cells atWER1/9 and bare/ON PCM identity; the earlier AVX512 opt-in comparison
remains separately scoped. Original port40220621471 passes all19 F16 stages
and all442 published coefficient payloads reproduce at that exact source.
The licensed CUDA archive passes an independent streaming69-file hash audit.

Prelaunch transport check found the documented GitHub token absent from the
Kaggle credential dataset, before allocating a GPU. Verified inputs are staged
privately using existing HF credentials, with immutable revision/hash pins;
the mounted Kaggle HF credential successfully reads the actual archive.
The source-pinned kernel is prepared without adding new credentials. Actual
GPU acceptance, integration and release preparation remain pending.
