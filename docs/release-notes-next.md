# Draft release notes — changes after v0.8.43

The published release is documented in [v0.8.43 release notes](../RELEASE_NOTES_v0.8.43.md)
and [the tagged validation receipt](release-tag-v0.8.43-2026-10-12.json). No later
tag has been cut.

## Dia DAC precision conversion

The explicit [converter](../models/convert-dia-dac-to-gguf.py) reproduces the
original critical F32 output-weight candidate with a 1,312-byte size increase,
leaving 131 unrelated payloads unchanged. It verifies published-template and
original-checkpoint hashes and writes all 134 physical tensor audit rows.
Seven tiny actual-original-module/CLI contracts pass, including observed
wrong-template failure before the guard and early rejection after it.
[Instructions and limits](dia-dac-conversion.md) link the unchanged export's
48/48 startup and 40/40 full-history hosted numerical proof. The dedicated
path-filtered conversion workflow requires the same candidate bytes and tensor
audit. Its [fresh integration result](dia-dac-converter-hosted-2026-10-13.json) passes; runtime short-speech/Q4
and GPU acceptance are not established, and registry pins remain unchanged.

## Dolphin non-streaming decoding — PR #541 merged

Honor the checkpoint causal flag in both depthwise convolution sites and preserve
the original causal/noncausal LayerNorm/padding order. Register Dolphin in the
shared architecture table so automatic binding sessions can open its GGUFs.

Both non-streaming and streaming models pass Chinese/JFK original references:
64 stage comparisons, exact CTC frame argmax, exact original raw transcripts
and matching automatic C ABI/Python sessions. Independent checks cover all
128 complete native/reference arrays. The 12 original-module/native controls
and causal-only/normalization-order negative controls are also retained.
See [the full proof receipt](dolphin-noncausal-2026-10-12.json). All 45 applicable general CI checks pass, one intentional skip; the fix is
merged on main. Reporter Kylin/GCC9/private-clip and GPU validation are not
established.

## Breeze direct GPU initialization — PR #542 merged

Register dynamic backend plugins before selecting the Breeze compute device.
Direct library initialization now finds CUDA; verbose initialization reports the
actual compute backend. All 36 applicable PR checks pass, with two intentional
skips. The composition package exercises the identical initialization prefix on
Tesla T4: Breeze uses 6482/6924 MiB in stage/guided dumps, 107 original stage
comparisons pass, and all 20 speech/interface outputs remain byte-identical to
the prior real-CUDA run. The later branch-cache code differs between composition
and standalone builds; this is not a complete standalone-binary GPU proof.
See [the CUDA dispatch receipt](breeze-cuda-dispatch-2026-10-12.json).

## TTS session ownership, conversion rates and speaker boundaries — PR #531 merged

Rejected Breeze reference updates preserve the previous reference; replaced Moss
reference resources are released. Melo/OpenVoice conversions use the actual
input rate and return owned PCM at the converter model's output rate. Supertonic
direct speaker getters now safely reject null contexts and invalid indices.
Contributor history from #530 is preserved. All 48 applicable CI checks pass,
one intentional skip; the hosted native tier passes all 2,009 unit tests and
all 140 ONNX-enabled CLI/C ABI capability rows. Separate model-quality reports
in #530 remain unresolved.

## Breeze instruction, reference and server voice handling — PR #543 merged

The TTS ownership/rate/speaker repairs now pass the complete ONNX capability
catalogue and native C ABI checks, plus all 2009 weight-free native tests.
Breeze's refreshed package additionally passes all 107 original stage/magnitude
comparisons and all 20 actual-T4 speech/interface roundtrips, with byte-identical
speech against the previous GPU run. Maximum ASR WER is 0.111111; speaker clone
similarity is 0.894997 versus the 0.527019 baseline. See [the refreshed package
receipt](breeze-cuda-integration-package-2026-10-12.json). Canonical complete
BF16 code equality remains 0/8 and is retained as a separate diagnostic;
fixtures, recipes, defaults and its comparator are unchanged. PR #543 merged
as `8dfa0d72cbe5cc24d7853d4631fa758a4134b927` after 51 passing checks and
one intentional skip, preserving contributor history from #529/#536. Instructions
and reference audio now reach synthesis together through the C ABI; server
voice caches include transcript-sidecar changes and reject unknown voices while
preserving subsequent valid requests. See [the merge receipt](breeze-merge-2026-10-12.json).

## CANN backend/device selection — PR #544 merged

The CANN follow-up exposes CLI backend/device selection and adds a separate
opt-in x86_64/910B and arm64/310P compile matrix for CLI, shared library, diff
and quantizer. Ten production-source routing controls pass, with four failing
against the previous helper. Current-source x86_64/910B (ACL graph on) and
ARM64/310P (ACL graph off) builds both pass all four targets and ggml-cann.
See [the complete compile receipt](cann-pr544-compile-2026-10-12.json). PR #544
merged as `a86e366043f4a23251f10b67d982778cd89f83dd` after 46 passing
checks and one intentional skip. See [the merge receipt](cann-merge-2026-10-12.json).
Hosted compilation cannot establish Ascend inference or speed.

The operator-validation wrapper is merged on main. Hosted ARM CPU runs one
real positive ADD case and rejects three actual upstream false-success cases:
missing backend, empty parameter selection and unknown operator. It preserves
full logs, hashes and incremental receipts. The [physical-device procedure](cann-hardware-validation-2026-10-12.md)
has been sent to the contributor; physical Ascend results remain pending.

## Dia reference and short-speech fixes — validation in progress

The active Dia branch preserves aligned audio frames during delay reversal,
zeroing invalid individual codebook entries to match the pinned original
method. Eight complete code-grid regressions pass; five fail before the fix.
The reference/diff integration observes original generation and real DAC calls,
checks exact input boundaries plus stage cosine and magnitude, and rejects
missing or uniformly scaled reference stages. Input validation rejects
unaligned whitespace/NUL before model loading; five tiny-original generation
controls and two observer contracts pass. These small tests establish boundary and capture behavior.

[Actual-weight numerical/codec validation](https://github.com/CrispStrobe/CrispASR/actions/runs/38278884937)
and [complete native F16/Q8/Q4 speech validation](https://github.com/CrispStrobe/CrispASR/actions/runs/38272642504)
are separate gates. The numerical/codec run completed with failure: F16 neural
stages pass, but isolated DAC replay fails cosine/magnitude; Q4_K also fails
neural boundaries. Complete native speech also finished with failure: long
prompts pass, but the reported short sentence is unrecognizable through both
CLI and C ABI: F16/Q8 are quiet, while Q4 produces louder non-speech. No threshold is
relaxed. The reported short-prompt silence is not yet declared
resolved, and the runtime changes are not merged or included in release
acceptance. Decoder-history and DAC-code replay do not establish free-running
sampler equality. The eight-boundary follow-up at `24468e487` stopped before native comparison on an unmapped published DAC input-projection bias; its actual original captures and full failure logs are retained in [the harness-failure receipt](dia-codec-trace-harness-failure-2026-10-12.json). The mapper repair at `778dac9cc` passes three tiny-original contracts covering all 134 parameters, code-replay independence and unknown-name rejection. The fresh comparison has started: [original/published/F32 codec A/B with eight boundaries](https://github.com/CrispStrobe/CrispASR/actions/runs/38294496529)
and [complete original short-speech control](https://github.com/CrispStrobe/CrispASR/actions/runs/38285566623).
The complete original control has finished: seed 42, 1,210 DAC frames,
14.048 seconds, RMS 0.073534 and verbatim ASR (WER 0). Native F16 replaying
this full original history passes all 40 strict comparisons, including isolated
DAC PCM (cosine 0.9999999, norm error 0.00004443983). Q4_K passes only 13/40
and first fails encoder layer 6. [The full receipt](dia-full-original-short-2026-10-12.json)
retains all stage metrics and 21 artifact hashes. The workflow remains failed
on Q4_K. This distinct full-utterance result does not supersede the 128-step
startup-codec failure or accept native free-running short speech. The separately
labelled original-model/native-sampler control isolates stochastic sampling;
canonical original PCM, defaults and thresholds remain unchanged. That control
has now finished with failure: original neural/DAC using the compiled native
sampler also emits quiet short audio with WER 1, as does native free generation.
They share 37 exact input steps with minimum neural cosine 0.9999993818 and
maximum norm error 0.0006661525. The first sampled difference is step 36/head 8.
[The full sampler receipt](dia-native-sampler-control-2026-10-12.json) retains
all inputs, logits, draws, PCM and log hashes. The native sampling path is
sufficient to trigger the failure with original modules; pure RNG versus
probability/filter replay passes on pinned Torch 2.6 with matching support and maximum error below 3.61e-7; RNG compatibility remains unproven. No gate waiver.
The eight-boundary codec worker has now completed successfully. The published
startup failure first appears at the output convolution; native code matches
actual original modules using those same rounded weights. Three original F32
final-output tensors pass all 48 canonical startup comparisons, with PCM cosine
0.9999927 and norm error 0.002027489. The export adds only 1,312 bytes.
[The full codec receipt](dia-codec-precision-2026-10-12.json) preserves the
published failure, all four variants, four failing negative controls and 47
artifact hashes. Actual physical audits verify three original F32 payloads and
131 unchanged unrelated payloads/types. Standalone export now reproduces the validated bytes and passes fresh startup/full-history comparisons; native free-speech acceptance remains failed and production pins are unchanged.
The generic reference/diff additionally observes eight DAC boundaries, preserving
canonical PCM/codes and requiring every declared stage. Seven local capture
contracts plus three parameter contracts pass; this remains unmerged diagnostic work until hosted acceptance.
The [complete rejected numerical and speech proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/43b643898356a14c9d7a8aa0168ef5ea0d64261d/dia-20261012/rejected-codec-and-short-speech-proof.zip)
is public and immutable: 54,845,734 bytes, anonymously downloaded and fully
SHA256 verified. [The receipt](dia-rejected-proof-2026-10-12.json) retains every
artifact file hash, both failed runs and the follow-up source provenance.

## Work excluded from release acceptance

PRs #492 and #521 remain pending their documented model, quantization or
hardware gates. Contributor PRs #529 and #536 are incorporated in the validated
Breeze composition merged through #543. The additional model-quality
reports recorded in merged contributor PR #530 remain unresolved. Breeze's actual-CUDA strict
guided generation remains 0/8 against canonical BF16. Original-only arithmetic
controls reproduce some complete native grids: joint F32 matches 1/8 CPU and
3/8 CUDA; F16 KV rounding matches 2/8 CPU and 3/8 CUDA. These diagnostics do not
change fixtures or acceptance gates. The completed full-inference F32 original
control matches 5/8 CPU and 5/8 CUDA grids; canonical BF16 remains 0/8.
See [the full-inference arithmetic receipt](breeze-guided-full-inference-arithmetic-2026-10-12.json).
See [the KV arithmetic receipt](breeze-guided-kv-arithmetic-2026-10-12.json).
Canary CUDA allocation/reuse controls do not reproduce #532 and do not establish
original Q4 parity. No new default, quantization format or performance claim
follows from these diagnostics.

Dia's unmerged [portable sampler replay](dia-portable-random-hosted-2026-10-13.json) reproduces all 11,034 original draws and 1,210 codec frames with native filtering. In the [real speech control](dia-portable-speech-hosted-2026-10-13.json), the original neural model with this sampler produces byte-identical canonical PCM and WER 0, while native F16 CLI/C-ABI speech still fails with empty transcripts. The first differing draw follows a nucleus-support boundary change despite close logits. [Completed original-head/full-F32 controls](dia-neural-precision-hosted-2026-10-13.json) both pass 40/40 replay stages but reject real CLI/C-ABI speech with empty transcripts and WER 1. Full F32 extends exact shared input history to 865 steps; [Opt-in cross-cache precision control](dia-cross-cache-controls-2026-10-13.json) is queued with fixed F32 weights and same-host OFF/ON/OFF guards. The runtime stays opt-in and unmerged; production defaults/model pins and speech/Q4/GPU gates remain unchanged.

[Durable Dia proof retrieval](dia-proof-archives.md) preserves all 77 canonical/component/converter/rejected-speech files in immutable public archives. The stdlib helper verifies full archive and payload hashes, rejects invalid extraction and preserves existing paths; eight contracts and real archive retrieval pass. Future validation input transport also passes complete [original/native-filter replay](dia-public-input-replay-2026-10-13.json). These are reproducibility fixes, with native speech still rejected and cache precision under investigation.

[Complete rejected neural precision and tiny cache proofs](dia-neural-public-proof-2026-10-13.json) are also durably public, with full anonymous archive verification and unchanged acceptance scopes. The TTS documentation records the short-prompt failure; the architecture guide corrects the original checkpoint’s CFG equation.
