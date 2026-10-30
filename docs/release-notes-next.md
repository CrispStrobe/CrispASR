# Draft release notes — changes after v0.8.43

The published release is documented in [v0.8.43 release notes](../RELEASE_NOTES_v0.8.43.md)
and [the tagged validation receipt](historical/release-tag-v0.8.43-2026-10-12.json). No later
tag has been cut.

**Current 2026-10-21 release preparation:** source883549696 passed all13
CI39467977527 and10 lint39467977612 jobs, with2,068 selected CTest tests,
zero failures/four dictionary-data skips, plus189 Python tests/34 subtests.
The user authorized preparing and publishing0.8.44. The concise canonical notes are
[RELEASE_NOTES_v0.8.44.md](../RELEASE_NOTES_v0.8.44.md); remaining release gates
are in [PLAN.md](../PLAN.md) and [the handoff](maintenance-handoff.md).
OpenVoice reference/conditioning integration and bounded synthetic encoder
proof are complete; CUDA package39266520251 passes but actual pretrained/GPU
and speech quality remain unaccepted. Generic speaker operation review covers
all eight wrappers; real-model/platform/concurrency scope remains separate.
Dia's broader speech gates failed despite numerical evidence; older dated
investigations below are historical, not a pending validation claim. Preserve
all failed gates and do not advertise defaults, GPU/Q4 or performance promotion.
Completed work is archived in [HISTORY.md](../HISTORY.md).

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
audit. Its [fresh integration result](historical/dia-dac-converter-hosted-2026-10-13.json) passes; runtime short-speech/Q4
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
See [the full proof receipt](historical/dolphin-noncausal-2026-10-12.json). All 45 applicable general CI checks pass, one intentional skip; the fix is
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
See [the CUDA dispatch receipt](historical/breeze-cuda-dispatch-2026-10-12.json).

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
receipt](historical/breeze-cuda-integration-package-2026-10-12.json). Canonical complete
BF16 code equality remains 0/8 and is retained as a separate diagnostic;
fixtures, recipes, defaults and its comparator are unchanged. PR #543 merged
as `8dfa0d72cbe5cc24d7853d4631fa758a4134b927` after 51 passing checks and
one intentional skip, preserving contributor history from #529/#536. Instructions
and reference audio now reach synthesis together through the C ABI; server
voice caches include transcript-sidecar changes and reject unknown voices while
preserving subsequent valid requests. See [the merge receipt](historical/breeze-merge-2026-10-12.json).

## CANN backend/device selection — PR #544 merged

The CANN follow-up exposes CLI backend/device selection and adds a separate
opt-in x86_64/910B and arm64/310P compile matrix for CLI, shared library, diff
and quantizer. Ten production-source routing controls pass, with four failing
against the previous helper. Current-source x86_64/910B (ACL graph on) and
ARM64/310P (ACL graph off) builds both pass all four targets and ggml-cann.
See [the complete compile receipt](historical/cann-pr544-compile-2026-10-12.json). PR #544
merged as `a86e366043f4a23251f10b67d982778cd89f83dd` after 46 passing
checks and one intentional skip. See [the merge receipt](historical/cann-merge-2026-10-12.json).
Hosted compilation cannot establish Ascend inference or speed.

The operator-validation wrapper is merged on main. Hosted ARM CPU runs one
real positive ADD case and rejects three actual upstream false-success cases:
missing backend, empty parameter selection and unknown operator. It preserves
full logs, hashes and incremental receipts. The [physical-device procedure](historical/cann-hardware-validation-2026-10-12.md)
has returned physical 310P3 results (SDK 9.1.0, ARM64, ACL graphs off).
RMS_NORM 51/51, SOFT_MAX 107/107 and ROPE 109/109 pass, with unsupported
cases retained. Restricted F32 ADD and F32/F16-weight matrix controls pass;
the original broad ADD/MUL_MAT suites aborted on unsupported BF16 inputs.
The integrated repair rerun passes all 924 executed cases with unsupported
cases retained; both SDK architectures and all 13 general CI jobs pass.
Full-model speech and speed are unvalidated.
See [the complete hardware receipt](historical/cann-310p3-hardware-2026-10-13.json).

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
sampler equality. The eight-boundary follow-up at `24468e487` stopped before native comparison on an unmapped published DAC input-projection bias; its actual original captures and full failure logs are retained in [the harness-failure receipt](historical/dia-codec-trace-harness-failure-2026-10-12.json). The mapper repair at `778dac9cc` passes three tiny-original contracts covering all 134 parameters, code-replay independence and unknown-name rejection. The fresh comparison has started: [original/published/F32 codec A/B with eight boundaries](https://github.com/CrispStrobe/CrispASR/actions/runs/38294496529)
and [complete original short-speech control](https://github.com/CrispStrobe/CrispASR/actions/runs/38285566623).
The complete original control has finished: seed 42, 1,210 DAC frames,
14.048 seconds, RMS 0.073534 and verbatim ASR (WER 0). Native F16 replaying
this full original history passes all 40 strict comparisons, including isolated
DAC PCM (cosine 0.9999999, norm error 0.00004443983). Q4_K passes only 13/40
and first fails encoder layer 6. [The full receipt](historical/dia-full-original-short-2026-10-12.json)
retains all stage metrics and 21 artifact hashes. The workflow remains failed
on Q4_K. This distinct full-utterance result does not supersede the 128-step
startup-codec failure or accept native free-running short speech. The separately
labelled original-model/native-sampler control isolates stochastic sampling;
canonical original PCM, defaults and thresholds remain unchanged. That control
has now finished with failure: original neural/DAC using the compiled native
sampler also emits quiet short audio with WER 1, as does native free generation.
They share 37 exact input steps with minimum neural cosine 0.9999993818 and
maximum norm error 0.0006661525. The first sampled difference is step 36/head 8.
[The full sampler receipt](historical/dia-native-sampler-control-2026-10-12.json) retains
all inputs, logits, draws, PCM and log hashes. The native sampling path is
sufficient to trigger the failure with original modules; pure RNG versus
probability/filter replay passes on pinned Torch 2.6 with matching support and maximum error below 3.61e-7; RNG compatibility remains unproven. No gate waiver.
The eight-boundary codec worker has now completed successfully. The published
startup failure first appears at the output convolution; native code matches
actual original modules using those same rounded weights. Three original F32
final-output tensors pass all 48 canonical startup comparisons, with PCM cosine
0.9999927 and norm error 0.002027489. The export adds only 1,312 bytes.
[The full codec receipt](historical/dia-codec-precision-2026-10-12.json) preserves the
published failure, all four variants, four failing negative controls and 47
artifact hashes. Actual physical audits verify three original F32 payloads and
131 unchanged unrelated payloads/types. Standalone export now reproduces the validated bytes and passes fresh startup/full-history comparisons; native free-speech acceptance remains failed and production pins are unchanged.
The generic reference/diff additionally observes eight DAC boundaries, preserving
canonical PCM/codes and requiring every declared stage. Seven local capture
contracts plus three parameter contracts pass; this remains unmerged diagnostic work until hosted acceptance.
The [complete rejected numerical and speech proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/43b643898356a14c9d7a8aa0168ef5ea0d64261d/dia-20261012/rejected-codec-and-short-speech-proof.zip)
is public and immutable: 54,845,734 bytes, anonymously downloaded and fully
SHA256 verified. [The receipt](historical/dia-rejected-proof-2026-10-12.json) retains every
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
See [the full-inference arithmetic receipt](historical/breeze-guided-full-inference-arithmetic-2026-10-12.json).
See [the KV arithmetic receipt](historical/breeze-guided-kv-arithmetic-2026-10-12.json).
Canary CUDA allocation/reuse controls do not reproduce #532 and do not establish
original Q4 parity. No new default, quantization format or performance claim
follows from these diagnostics.

Dia's unmerged [portable sampler replay](historical/dia-portable-random-hosted-2026-10-13.json) reproduces all 11,034 original draws and 1,210 codec frames with native filtering. In the [real speech control](historical/dia-portable-speech-hosted-2026-10-13.json), the original neural model with this sampler produces byte-identical canonical PCM and WER 0, while native F16 CLI/C-ABI speech still fails with empty transcripts. The first differing draw follows a nucleus-support boundary change despite close logits. [Completed original-head/full-F32 controls](historical/dia-neural-precision-hosted-2026-10-13.json) both pass 40/40 replay stages but reject real CLI/C-ABI speech with empty transcripts and WER 1. Full F32 extends exact shared input history to 865 steps; [Opt-in cross-cache precision control](historical/dia-cross-cache-controls-2026-10-13.json) is complete and rejected, preserving fixed F32 weights and same-host OFF/ON/OFF guards. The runtime stays opt-in and unmerged; production defaults/model pins and speech/Q4/GPU gates remain unchanged.

[Durable Dia proof retrieval](dia-proof-archives.md) preserves all 77 canonical/component/converter/rejected-speech files in immutable public archives. The stdlib helper verifies full archive and payload hashes, rejects invalid extraction and preserves existing paths; eight contracts and real archive retrieval pass. Future validation input transport also passes complete [original/native-filter replay](historical/dia-public-input-replay-2026-10-13.json). These are reproducibility fixes, with native speech still rejected and cache precision under investigation.

[Complete rejected neural precision and tiny cache proofs](historical/dia-neural-public-proof-2026-10-13.json) are also durably public, with full anonymous archive verification and unchanged acceptance scopes. The TTS documentation records the short-prompt failure; the architecture guide corrects the original checkpoint’s CFG equation.


Completed fixed-cache control (GH 38357576078, source 2205d920d): all three
F16/F32/F16-repeat arms pass 40/40 canonical stages, but all six CLI/C ABI
speech controls remain empty/WER 1. OFF/repeat inputs, logits and PCM match
exactly. Actual cross-cache vectors double from 11,501,568 to 23,003,136 bytes;
F32 extends shared input history from 865 to 979 steps and reduces maximum
aligned logit norm error from 2.40e-5 to 5.01e-6. At step 978/head 2, tiny neural
changes include token 904 across the nucleus boundary; original/native filters
agree on identical logits and reproduce the recorded draws. This is a rejected
speech control, with no sampling-policy or quality-gate waiver.
Complete [cache controls](historical/dia-cross-cache-hosted-2026-10-13.json) and [first-draw replay](historical/dia-cross-cache-boundary-2026-10-13.json).


The scoped 310P BF16 capability candidate also passes both SDK9.2 compile
architectures, including the upstream operator binary; its physical rerun now
passes all 924 executed cases and the production ggml pin now uses the exact physically validated repair after the composed CI gate passed. The first Dia T4 attempt is retained as a
rejected validation path (CPU diff, zero observed GPU allocation/splits); the
corrected request is proven by the complete version-2 actual GPU controls.
Neither result establishes full-model NPU/CUDA speech or speed acceptance.


The source-faithful rotary-timescale CPU OFF/ON/OFF control is complete and
rejects native speech; the separate actual-GPU control also completes and rejects. Actual-source tiny coefficients at the remaining late boundary
improve from maximum error 1.20e-4 to 5.96e-8 through existing ggml frequency
factors; native speech remains rejected on CPU and actual T4. The auxiliary table is
physically audited and existing coefficient bytes are preserved. Original and
native late-step capture is tested for transparency; the first hosted attempt
is retained as a logging/count rejection, with complete failed evidence and a
corrected retry now passes all repeat/negative diagnostic controls.
The runtime experiments remain unmerged and do not change release
model pins, native quality gates or GPU defaults.


The subsequent physical Ascend310P3/SDK9.1 candidate rerun is verified: all
five broad operator suites complete with 924/924 executed cases passing,
unsupported cases retained, and both previous BF16 aborts resolved. Exact
source/compiler/archive/log provenance is recorded; all 13 composed CI jobs pass and the exact repair is integrated. Operator correctness does not establish model-quality or speed.


Actual T4 CUDA cache controls (2026-10-13) are complete: all 120 canonical
replay stages pass with real scheduler splits and sampled process GPU memory
8,528–8,734 MiB. All six native CLI/C-ABI speech checks still reject with empty
transcripts/WER 1. F16 cache shares 353 original input steps; F32 shares 979
and reaches the same step-978/head-2 divergence as CPU. OFF/repeat full arrays
match exactly. These are correctness diagnostics, not accepted native speech
or performance measurements. [Complete physical CUDA proof](historical/dia-cuda-cache-physical-2026-10-13.json).


The complete CPU/CUDA cache, partial late-layer and causal diagnostic proofs
are preserved as five immutable public archives. All payload and full anonymous
remote size/SHA checks pass; rejected speech remains rejected. [Publication
receipt](historical/dia-cache-public-proof-2026-10-13.json).


The completed original-timescale CPU control passes 120 replay rows but does
not change generated token histories or PCM; all six speech checks remain
empty/WER 1. The +608-byte auxiliary model is a rejected control. The corrected
late-capture experiment independently passes default/978/default complete-array
identity and all three rejection controls. These establish diagnostic fidelity,
not accepted native speech or speed.
[Rotary control](historical/dia-original-rope-hosted-2026-10-13.json), [complete late capture](historical/dia-late-stage-hosted-2026-10-13.json).


The scoped Ascend310P BF16 capability repair is integrated at the exact
physically tested ggml revision. It resolves both broad ADD/MUL_MAT aborts;
924/924 executed physical operator cases pass, with unsupported cases retained.
Both SDK compile architectures and all 13 composed general CI jobs pass.
The shared fork retains contributor history. Full NPU model quality and speed
remain separate validation. [Integration receipt](historical/cann-310p-bf16-integration-2026-10-13.json).


The GPU rotary ON arm shortens matching original input history from 979 to
865 steps; all six speech checks remain WER 1 despite 120 passing replay rows.
Canonical original audio passes the same GPU recognizer at WER 0. Complete
terminal logs and 61 output files are retained and hashed; the experiment stays
unmerged. [Physical rotary receipt](historical/dia-original-rope-cuda-physical-2026-10-13.json).


The exact-source CUDA v3 bundle is fully audited (27 members and seven source
files); its actual T4 speech rejection is retained separately. A weight-free
RMSNorm audit on all retained decoder boundaries isolates F32 versus double-sum
rounding in nine of 36 comparisons, with no runtime/default change or speed
claim. [CUDA package](historical/dia-cuda-package-v3-2026-10-13.json), [RMSNorm primitive](historical/dia-rmsnorm-primitive-2026-10-13.json).


The exact-shape output-head probe reproduces every retained native step-978
logit byte-for-byte. Combining the heads changes nothing; flattening batch
shape changes rounding but does not restore the differing draw. All 16
original-filter controls choose token 877 with original hidden states and 904
with native hidden states. The cause at this boundary precedes the output
head. This is isolated numerical evidence, not accepted speech or speed; the
local original projection is not byte-identical to the hosted original one.
[Projection audit](historical/dia-head-projection-isolation-2026-10-13.json).


The learned final norm is byte-identical on retained original/native layer-17
inputs and reproduces both saved hidden states exactly. Decoder embedding
captures also match exactly at four checkpoints, including step 978. Repeats
and changed-feedback/scaled-output negatives pass. The remaining hidden
variation is inside the decoder blocks or their attention inputs; these
isolated controls do not accept native speech, Q4 or speed.
[Norm/input audit](historical/dia-norm-embedding-isolation-2026-10-13.json).


Block-0 pre-attention norm is byte-identical on all four verified embedding
inputs. Q/K/V projections introduce small rounding differences (minimum cosine
0.9999999999999603, maximum row-norm error 9.19345e-8). Flattened branch batches
worsen relative L2 in all 12 controlled cases, so no runtime candidate is
introduced. Repeats and a magnitude negative pass. This is primitive isolation,
with native speech still unaccepted.
[Block-0 audit](historical/dia-block0-projection-isolation-2026-10-13.json).


The actual Q/K rotary replay confirms local error on audited first-block
projections. At step 978, composed relative L2 falls from 5.37020e-5 to
1.57296e-7 for Q and 3.55324e-5 to 2.28756e-7 for K using original timescales.
All 48 comparisons, repeats and eight negative controls are retained. This is
CPU primitive evidence; full-model source-timescale speech remains rejected,
with no runtime/default promotion or performance claim.
[Rotary audit](historical/dia-block0-rotary-isolation-2026-10-13.json).


A bounded actual-source block-zero self-attention replay on canonical steps
0–3 verifies byte-identical same-input K/V cache histories. Attention minimum
head cosine is 0.9999999999999742 and maximum norm error 1.46145e-7. Missing
cache reorder, tiled GQA heads, wrong sqrt-head scaling and doubled output all
strongly reject. New step-1/3 norm rounding differences and the stopped initial
expectation are retained. This does not establish late-history, higher-layer,
GPU or full speech acceptance; production defaults remain unchanged.
[Attention/cache audit](historical/dia-block0-attention-isolation-2026-10-13.json).

The pinned-config 3072-slot original cache also reproduces all short-prefix
Q/K/V/attention outputs and cache history bytes exactly, with exact repeats.
This checks production cache stride, without claiming late-history acceptance.

Late-history self-attention capture tooling is implemented on diagnostic source
42b111c5e. Tiny actual-original two-layer and compiled replay guards pass; GH
38438623888 queues full-model capture-transparency controls and step-978 snapshots
for all 18 layers. Production main receives tracking only. This is pending
numerical evidence, not a runtime fix, speech acceptance or performance result.
[Capture protocol](historical/dia-attention-history-capture-2026-10-13.md).

The weight-free same-input attention consumer reproduces audited prefix 0–3
original snapshots with production cache stride and native snapshots at one/four
CPU threads, byte-for-byte. Its 24 comparison rows and 12 strong negatives are
retained. Hosted snapshots require verified file hashes and capture transparency;
late-history results, cross-host identity and native speech remain unaccepted.
[Same-input replay audit](historical/dia-attention-input-replay-2026-10-13.json).

The matching CUDA 12.4 SM60/75 capture package is queued on GH at source
2328f15ce; all six native runtime/ABI/binding files match CPU capture source
42b111c5e byte-for-byte. Eight source guard tests pass. The CPU capture worker
is now running. Both hosted results and actual GPU/model acceptance remain
pending; production main receives tracking only.
[CUDA package protocol](historical/dia-attention-cuda-package-2026-10-13.md).

### Dia diagnostics: retained CPU/T4 evidence and receipt repair

Preserved and independently verified completed GPU work after a diagnostic
receipt serialization failure: 120 stage comparisons, 72 late self-attention
snapshots, complete OFF/ON/OFF capture transparency and original-audio GPU
ASR WER 0. Fixed integer shape serialization and atomic checkpoint writes;
eleven diagnostic worker guards pass. Complete evidence is public.

Same-input actual-original attention and block-zero cross-K/V replays now
show relative L2 differences below 7.21e-7 on retained CPU/T4 outputs;
scale and layout negatives remain effective. The original GPU job remains
ERROR, and the original reference operators in these bounded replays run CPU.
Native free-running speech and Q4 remain unaccepted. Opt-in decoder
residual/cross-Q/fused-MLP observation on diagnostic source 8d0ad886 now passes
full hosted capture transparency, sixteen actual-original hook tests and 120
ordinary stages in GH 38472367538. Independently verified boundary differences
are small (maximum relative L2 5.204e-6), and complete proof is public. These
rows include input drift. Actual fused-original/split-native MLP replay now
reproduces hosted native stages exactly and differs on matching inputs by at
most relative L2 2.399e-7, with scale/wrong-split controls rejecting. Native
speech/Q4 remain unaccepted. No production default, model or release promotion
follows. See
`docs/historical/dia-cross-cache-input-replay-2026-10-13.md` and
`docs/historical/dia-decoder-boundary-results-2026-10-14.md` and
`docs/historical/dia-mlp-owned-input-results-2026-10-14.md`.

## UPDATE 2026-10-14 — Owned self-output and cross-attention verified

Actual original projections and SDPA/mask/cache expressions versus literal
native CPU graph yield 24 rows and 12 rejecting scale/missing-mask controls.
Minimum same-input cosine 0.9999999999998742, maximum row norm error
3.122e-7, maximum relative L2 2.580e-7. All four native outputs exactly
reproduce hosted native snapshots; repeat/thread controls are exact. Actual
mask is conditional-derived for BOTH CFG branches and permits the 39-token
prefix of the 1,024-token source context. Synthetic finite masked tails
verify invariance only, not original unretained padded caches. Complete
40-payload proof is public and anonymously verified. No runtime math change,
full speech, GPU, Q4, speed or release acceptance. Evidence:
`docs/historical/dia-cross-owned-input-results-2026-10-14.md` and `.json`.

## Ascend 910B3 physical follow-up — 2026-10-14

New current-main/ggml21acf3162 physical 910B3, ARM64, SDK9.1, ACL-graph-on
archive is retained; all 25 member hashes and seven unchanged-wrapper
assessments verify. Broad ADD154, RMS_NORM51, SOFT_MAX107 and MUL_MAT927
pass. RoPE fails 450/452 and repeats the same two F32 NEOX failures.
This is not broad 910B acceptance and does not change the earlier scoped
310P result. Native Git/compiler provenance verifies; the binary hash is
producer-reported because no executable is included. No full-model,
quantized speech or performance proof. Cache-width allocation is a candidate
explanation awaiting regression/device proof. Evidence and next step:
`docs/historical/cann-hardware-validation-2026-10-12.md` and
`docs/historical/cann-910b3-hardware-2026-10-14.json`.

## CANN rotary requested-capacity defect reproduced — 2026-10-14

A bounded pre-fix regression compiles the actual tested allocation block
and cache metadata. Constant-512-position widths64/64/256/128 leave
131,072-byte sin/cos requests when 524,288 and 262,144 bytes are required.
Width growth with fewer positions also fails. Four tests: two expected
failures, two positive controls, complete logs retained. This is allocation
coverage only, not RoPE math/device validation or a repair. Immutable test
source7b77e0b remains diagnostic; production pin/runtime unchanged.
Public 14-payload proof is anonymously verified. Evidence:
`docs/historical/cann-rope-capacity-2026-10-14.md` and `.json`.

## Dia full free-running transparency verified — 2026-10-14

Run38527734866 completes; independent whole-delivery/source/history/PCM/WER
audit passes. Ten unforced arms match full PCM within CLI/C ABI and all
1,115-step logged model histories match across surfaces. Common/extended
step978 marks are transparent for the fixed CPU F32 diagnostic recipe.
Baseline speech remains rejected on both surfaces: empty ASR, WER1.0.
Public raw262-payload and complete audit archives are anonymously verified.
No speech/Q4/GPU/performance/default or release acceptance follows.
Evidence: docs/historical/dia-free-capture-results-2026-10-14.md and .json.

## CANN rotary allocation candidate prepared — 2026-10-14

Shared ggml0a8620765 records sin/cos allocated bytes independently of
last-call dimensions. Four unchanged capacity regressions now pass; old
two failing controls remain public. Thirteen BF16 routing cases per chip
pass. SDK sourceca4c12c4 pins the candidate; run38536737288 is submitted
on 910B/graph-on and 310P/graph-off. SDK result and physical 910B3 rerun
remain pending. Production ggml21acf3162 is unchanged; no numerical
NPU/model/quantized quality/speed or release acceptance. Immutable
fresh-directory commands and public candidate proof:
`docs/historical/cann-rope-capacity-candidate-2026-10-14.md` and .json.

### October14 physical candidate and query-isolation checkpoint

Physical910B3/SDK9.1 capacity candidate passes1691/1691 broad cases plus
452/452 RoPE repeat, preserving unsupported cases and original failures.
SDK/composed CI remain pending. See historical/cann-910b3-rope-physical-2026-10-14.md.
Dia query/rotary diagnosis adds14rows/16negative controls and public complete
proof; native speech still fails. See historical/dia-query-owned-input-results-2026-10-14.md.

Both rotary candidate SDK9.2 architecture builds are independently verified
with complete logs and actual target/link evidence. General integration CI
remains active. See historical/cann-rope-sdk-2026-10-14.md and cann-todo.md.

Rotary byte-capacity repair is integrated at exact tested ggml0a8620765:
all13general CI jobs, both SDK architectures and physical1691/1691 broad
plus452/452 full RoPE repeat pass. Regression remains wired into both manual
SDK jobs. See historical/cann-rope-integration-2026-10-14.md. Full model/quantized speech,
actual ACL replay and performance remain future validation.

### Strict Moonshine device validation

Opt-in exact backend/device requirements reject missing or invalid devices and
GPU-disabled contradictions instead of silently running on another backend.
Encoder scheduler assignments and successful computation are recorded without
altering the graph. Moonshine diff verifies exact reference PCM, complete
shape/count, row magnitude and relative error alongside cosine, and refuses
missing encoder references. Magnitude failures preserve the actual measured
cosine. Full hosted CPU CLI/auto+explicit C ABI Session checks and13 deliberate
failure controls pass; all13 general CI jobs and2,024 unit tests pass on the
runtime candidate. The final diagnostic row/reporting correction also passes
bounded real-reference controls. Physical NPU, ACL replay, quantized model
quality and performance remain pending; no release is cut by this result.

### Portable full-model CANN validation interface

A standalone F32 Moonshine fixture runner verifies immutable model/reference
bytes, runs genuine CPU controls before requested CANN, and requires repeated
encoder, diff, CLI and auto/explicit C ABI Session output with real selected-
device work. It retains incremental failure receipts and actual binary/library
snapshots. The complete CPU interface and unavailable-CANN/GPU-argument controls
pass in hosted run38602252235, with all171 returned payloads independently audited.
Eight false-success/archive/argument guards are wired into Linux CI. Diagnostic
encoder capture now explicitly forwards GPU selection; defaults preserve CPU.
Both official SDK architectures link all6 requested targets in38598516003.
Compiler cache/command artifact retention is added; execution follow-up38605386237 passes both chips; actual
compiler chip/ACL definitions and all six links per chip are independently audited. These results establish preparation and CPU controls; physical NPU,
ACL replay, quantized-model quality and performance remain separate gates.


## Dia F32 sampling boundary diagnosed — 2026-10-14

Late block-zero self K/V replay passes10 metric rows and14 controls; native
last cache slots match hosted values exactly. The complete unforced F32
trajectory matches original inputs through979 steps, then the step978/head2
nucleus support swaps tokens46/904, causing original877/native904. Actual
original filtering with both retained noise streams reproduces that difference.
All963 aligned prefix code frames agree; both PCM prefixes are nearly silent
and agree at cosine0.999963626/norm ratio0.996749141. The original only reaches
speech later; the native ASR failure is not cleared. Complete independently
audited public proofs and scopes are in docs/dia-self-kv-owned-input-results-
2026-10-14.md and docs/dia-f32-first-divergence-2026-10-14.md. No runtime, filter,
seed, default or quality-gate change; no performance or release acceptance.


## Dia complete layer17 diagnosis — 2026-10-14

Whole original/native layer17 replay passes39 metric rows and32 scale controls;
eight inherited-input/cache interventions and two real wrong-cache controls
are independently audited. Native output and one/four-thread results reproduce
hosted bytes exactly. Matching-input output relative L2 is6.590e-7; inherited
hidden-input drift has the largest measured individual effect in this block.
Complete coefficients, sources, arrays, methods and binary proof are public:
docs/historical/dia-layer17-owned-input-results-2026-10-14.md and .json. Native speech
remains rejected; no Q4/GPU/performance/default or release acceptance.


## Integrated Dia top-k correctness repair (2026-10-15)

The shared sampler now selects exactly k temperature-scaled logit indices
instead of a rounded-probability threshold. The actual old header loses a
unique top-k1 maximum in34/64 trials; the corrected sampler loses0/64 and matches four
pinned original Python corner fixtures. All18,288 captured probability/sample
rows remain byte-exact to baseline. All13 candidate CI jobs,10 lint checks and the secret scan pass; Linux unit
testing passes2,026 tests. Header/test bytes are unchanged after rebase. No speech repair,
Q4/GPU/performance or release acceptance is inferred. Details and public proof:
[Dia top-k selection](historical/dia-topk-selection-2026-10-14.md).

## Dia layer16 and earlier hidden-state drift (2026-10-15)

Bounded original/native layer16 replay reproduces the hosted native output
byte-exact, with same-input final relative L2 3.636e-7. Independent audit covers
39 rows,32 scale negatives,8 current context interventions and2 real cache
negatives. The report retains an unresolved original cross-process exactness
failure and the initial-array retention limitation. Both complete component
and all18-layer scan archives are public and anonymously SHA-verified.
Cumulative drift peaks earlier at layer3 (3.806e-6); no layer16 repair,
full-speech/Q4/GPU/performance or release acceptance is inferred. Details:
[Dia layer16 and scan](historical/dia-layer16-owned-input-results-2026-10-15.md).

## Dia early layer and actual self-cache history (2026-10-15)

Layer0 reproduces native hosted output and repeated one/four-thread runs
byte-exact. The incoming hidden state is identical; split original-module
interventions localize the larger context effect to actual self-cache keys
(3.366e-6 relative L2) rather than values(1.397e-7). Independent audits cover
39 main rows,32 scale negatives,8 owned contexts,64 split-stage rows and2 real
wrong-cache controls. Complete283-payload evidence is public and anonymously
SHA-verified. No new runtime repair, speech/Q4/GPU/performance/default acceptance.
See [layer0 proof](historical/dia-layer0-owned-input-results-2026-10-15.md).

## Dia joint Q/K and arithmetic diagnostics completed (2026-10-15)

Integrated standalone query producers, tiny arithmetic controls, exact-input
attention decomposition, scalar reduction and CFG margin explanation. The
full-decoder Q/K factorial passes all739 raw payload/control checks and all
applicable hosted/lint gates. Joint replacement improves early attention but
does not restore original token ordering or support. Full public proof and
independent audits are retained; no speech/GPU/Q4/default promotion follows.
See [the complete result](historical/dia-query-key-results-2026-10-15.md).

## Dia one-step diagnostic replay

A validated one-step replay reproduces all127 native stages and logits, then
all18 native replacement controls exactly. Independent audit checks1,323
payloads,37 arms,666 layer metric rows and the actual original sampling filter.
Five original-output substitutions restore the reference support with
non-monotonic layer sensitivity. Complete raw proof and lossless reconstruction
methods are public. This adds diagnostic tooling, not a native speech repair,
new default, Q4/GPU acceptance or performance gain. [Results](historical/dia-suffix-replay-results-2026-10-15.md).

## Dia hosted terminal diagnosis

A five-arm hosted replay and independent277-payload audit retain exact native
controls and original final-hidden consumption. All90 layer metric rows and
original sampler outputs are recalculated. Conditional margin differences
separate post-layer17, final-normalization and head-projection effects; F64 CFG
reproduces every observed margin exactly. Complete raw proof is public. These
are integrated diagnostic tools, not a native speech fix, new default or
Q4/GPU/performance acceptance. [Result](historical/dia-terminal-replay-results-2026-10-15.md).

## Dia actual original trace and decision sensitivity

All18 layers now have simultaneous actual original self Q/K/V/attention,
pre-RoPE projections and decoder boundaries with exact OFF/ON/OFF controls.
An independent549-payload audit recalculates125 comparisons and checks complete
prefix/RNG provenance. Changing only one sampled token in the original takes
this prompt/seed from WER0 to quiet audio and empty ASR/WER1 despite stage cosine
at least0.9999999961. Complete raw proof is public and anonymously SHA-verified.
This explains a severe sensitivity mechanism; it adds no native speech repair,
new default or Q4/GPU/performance acceptance. [Results](historical/dia-original-trace-results-2026-10-16.md).

## Dia reciprocal decision diagnosis

A642-payload independent audit proves exact native controls and unchanged RNG.
Restoring one sampled decision recovers the expected sentence through native
and original DAC (WER1→0), all1,226 original input rows and1,210 code frames.
Complete public evidence is anonymously SHA-verified. This is diagnostic
recovery for one prompt/seed, not a production token override, native speech
repair or GPU/Q4/performance acceptance. [Results](historical/dia-native-fork-results-2026-10-16.md).


Dia diagnostic update (2026-10-16): the matched original CUDA precision/recipe
matrix completed all25 outputs and27 recognitions. Independent audit verifies261
payloads and262 anonymous public files. Both F16 recipes and F32/native pass the
two fox cases, but legacy and both hello cases remain quiet in every arm;
F32/source additionally fails fox42. This does not establish broader native
speech or Q4 acceptance, and changes no defaults. The repeated source-F16 anchor
preserves quality; legacy PCM has a retained small nonzero difference. Full
receipts and failed-auditor provenance are archived in HISTORY.md and
[the matrix result](historical/dia-cuda-factorial-results-2026-10-16.json).


Echo FFN diagnostic capture is integrated. The final composition preserves all
3591 non-document entries of the combined source that passed13 CI and10 lint
jobs. The1946-payload2B F16 observer proof includes original/native intermediate
identity, cached generation, CLI subtitles and C-ABI segments. These controls
support the diagnostic tooling;9B Q4 acceptance and performance remain pending.


Dia diagnosis now isolates draw placement on the same original F32 CUDA model:
both fox cases pass with CUDA draws, fail with CPU draws, then pass after restoring
CUDA draws. Independent audit verifies288 payloads/289 public files, and actual
CPU replay matches2880 captured IDs. Legacy and both hello cases still fail
speech across all arms. These results correct the earlier cross-device causal
interpretation; they do not change defaults or establish broader native/Q4
acceptance. Complete evidence and limitations are archived in HISTORY.md and
docs/dia-sampler-device-results-2026-10-17.json.

## License declarations and validation follow-up (2026-10-20)

Ordinary NC models remain supported behind explicit acceptance. OmniVoice now
uses a distinct bundle gate because its NC main weights and Boson tokenizer
terms differ; OpenRAIL models also require explicit acceptance. The public
OmniVoice card's blanket Apache-2.0 label was corrected without modifying weights.
These gates do not approve additional custom restrictions or grant new rights.
See [policy and source audit](model-license-policy.md).

The issue550 45-minute repeated-speech CUDA memory/reuse run passes independent
output, hash and placement audits on T4. A 2,703-second input retains all 8,925
units; the returning short input is exact after offset adjustment. Its observed
long-phase device peak is 2,735 MiB (sampled device total, not allocator maximum).
VAD remains unused in align-only, and host alignment memory grows with duration.

The actual Japanese Session metadata short/long/short sequence passes on Metal
with 17/123/17 tokens and exact native short-token bytes/times/probabilities.
This complements the exported-getter tests; it does not certify generic CLI
postprocessing, all long-form models or original-model acoustic accuracy.

Shared ggml PR2 remains under review; no Conv1D default or new shader is merged.
CANN NZ caching also remains off by default despite green hosted compilation
and Mac controls. Physical Ascend cache/model/performance acceptance is pending.
