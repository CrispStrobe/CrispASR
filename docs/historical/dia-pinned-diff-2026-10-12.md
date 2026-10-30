# Dia generic reference and diff wiring

The repaired `tools/reference_backends/dia_tts_reference.py` is registered as
`dia` in `tools/dump_reference.py`. `crispasr-diff dia model.gguf ref.gguf`
dispatches before the audio-argument gate. The native model may use an embedded
DAC or `DIA_CODEC`; otherwise the companion is `dac-44khz.gguf` beside the model.

Original code is pinned to `4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8`, matching
the old `nari-labs/Dia-1.6B` checkpoint at
`257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd`. `DIA_SOURCE` must point to that
checkout; HEAD and every top-level Python source are verified against Git.
The remote checkpoint is revision-pinned. An explicit local directory is
identified as local, with its config/weight SHA256, rather than falsely claiming
the remote revision. DAC checkpoint and source hashes are recorded too.

The reference observes the original `generate()` and codec calls. It preserves
the original encoder masks, padding and unconditional/conditional order.
Captured encoder and cross-cache views trim padding to the native sequence
length. The actual tokenized input determines that length. Text must already
have `[S1]`/`[S2]` at its actual start and punctuation at its actual end,
with no NUL bytes, matching native preparation. Leading/trailing whitespace
is rejected before source or model loading rather than silently normalizing
the original input. Interior padding is rejected. Captures own storage; hooks/methods are restored on failure.
The decoder input/logit arrays preserve every captured step and all nine heads
in both CFG branches. Original decoded PCM comes from the real DAC.

The native comparison verifies exact encoder tokens and decoder input history,
encoder embedding/output, all twelve encoder layers and eighteen first-step
decoder layers, layer-zero rotated cross K/V, first cross-attention
output, final decoder hidden states, every decoder-step logit and original-code
DAC output. It rejects missing tensors and unequal complete element counts.
Every numerical row must meet the existing F16 source-parity gates: cosine
at least 0.9995 and norm error below 0.02. Integer input boundaries are exact.
Diagnostics print both norms. Q4_K failures remain failures under those gates.

Decoder history is teacher-forced and the DAC separately replays the original
post-delay code grid. This does not test free-running sampler equality or native
delay reversal. The latter has eight original-method regression cases; actual
complete CLI/C ABI speech and ASR acceptance use `dia_delay_speech.py`.
Per-layer hidden states are captured at each complete block boundary; decoder
block snapshots cover the first step, while logits cover every captured step.

Hosted worker `tools/ci-heavy/dia_pinned_diff.py` exercises the real reference
dispatcher, compiles the native diff, and runs F16 and Q4_K sequentially. Its
negative archives omit encoder output or multiply it by two: the diff must
reject both even when other stages, or cosine alone, would pass. Its default
128-step probe is bounded numerical/codec evidence, not full short-speech
acceptance. `--tokens 3072` requests a full-capacity original generation.

Local validation: five tiny-original generation/capture tests and two codec observer
ownership/failure tests, production boundary
regression, C++ syntax checks and Python compilation. Only the tiny test's final
DAC is a dummy code projection; it provides no PCM/quality evidence. Hosted
run `38278884937` rejects published DAC PCM (cosine 0.9976444, norm error
0.04271824) despite passing F16 neural stages. Complete native speech run
`38272642504` rejects the reported short sentence through CLI/C ABI for all
three quants. Long controls pass. These results do not accept the runtime.

The generic dumper includes all eight DAC boundaries in its default stages and
labels their required schema in archive metadata. The codec-precision worker
additionally calls actual original `from_codes()`
and decoder modules to observe `dac_latent`, `dac_input_conv`, `dac_block_0`
through `dac_block_3`, `dac_output_snake` and `dac_output_conv`. Original
weight-normalization removal and capture must preserve the actual PCM exactly.
Canonical PCM/code arrays in the preserved archive remain unchanged; additional
boundaries supplement them. Precision-matched original-module archives remain
separately labelled controls. Each trace is complete channel/time data `(C,T)`;
the native dump transposes its channel-fast layout to the same ordering.
The existing cosine and magnitude thresholds apply per complete channel.

`CRISPASR_DIA_DAC_TRACE_DIR` gates native tracing. Captured graph tensors are
marked as outputs before allocation; missing tensors or incomplete files fail.
Diff enables the trace only for archives explicitly declaring
`dia_dac_stage_trace=rvq+decoder-v1`, requires every declared stage and restores
the caller's environment. The worker rejects a missing codec block and a latent
scaled by two; preserved encoder missing/scale controls remain separate.
The two small module tests cover storage ownership, unchanged calls and observer
cleanup; they do not execute the real DAC checkpoint or establish codec parity.
Fresh real-weight boundary/precision results are pending.

The input guard was tested before fixing it: five whitespace-altered prompts
and a NUL-containing prompt reached generation, and an invalid dump reached
source loading. All seven regression assertions fail before the guard and pass
after it; unchanged valid-prompt generation remains array-identical. These
checks establish input/capture behavior, not real-weight speech quality.


The eight-boundary worker at `24468e487` stopped before native comparison
when mapping published `quantizer.quantizers.0.in_proj.bias`. Its complete
original-only captures are retained; the original checkpoint reproduction and
weight-norm removal checks completed before that exception. The diagnostic
mapper now includes both input/output projections and the codebook for every
quantizer. Three tiny-original-module contracts cover all 134 quantizer/decoder
Parameters, exact code-replay independence from input-projection bias, and
unknown-name rejection. The previous mapper produces 19 errors (18 missing
input-projection parameters plus the replay contract). The heavy worker runs
these contracts before downloads and actual-weight controls. This fixes
diagnostic coverage; no production codec/math/default or gate changes.

Full-capacity original run `38285566623` has completed. The actual pinned
Torch F32 original generates 1,210 DAC frames (14.048 seconds, RMS 0.073534)
for the reported short sentence, recognized verbatim with WER 0. Native F16
replaying this exact original history passes all 40 comparisons; minimum logit
cosine is 0.9999989 and maximum logit norm error is 0.0008548312. Isolated
full-utterance DAC replay passes at cosine 0.9999999 and norm error 0.00004443983.
Q4_K passes only 13/40; the earliest failure is `encoder_layer_6`, cosine
0.9993770 and norm error 0.009455288. Missing/scaled encoder controls correctly
fail. The workflow remains failed on Q4_K. These full-utterance results do not
erase the different 128-step startup-codec failure or accept native free speech.
Main retains all 21 proof-file hashes and full stage metrics in
`docs/historical/dia-full-original-short-2026-10-12.json`.

`tools/ci-heavy/dia_native_sampler_control.py` isolates the random stream with
a separately labelled original-model control. A small compiled adapter calls
the actual production `dia_sample_token` function in `src/dia_sampling.h`;
it replaces only the original `_sample_next_token` during this control and
restores it on exit. The canonical Torch generation is preserved separately,
with its reference and PCM hashes verified. The fixed original/native seed is
42, temperature 1.2, top-p 0.95, top-k 45, CFG 3 and capacity 3,072. No seed or
prompt selection, production change, forced native history or quality waiver.

The hosted worker generates original-control and native C ABI speech in
separate processes, recognizes both plus the canonical original, and preserves
complete input histories, raw conditional/unconditional logits, pre-override
samples, code grids and PCM. It reports the first discrete divergence and
per-row cosine and magnitude over the identical input prefix under the existing
gates. The trace parser rejects missing, reordered and malformed sample rows.
Tiny tests cover actual-header RNG continuity, invalid-input state preservation,
observer lifetime, unchanged actual tiny original generation and trace rejection.
Their dummy DAC is excluded from speech-quality evidence. Actual heavy results
remain pending; a single successful prompt would still not accept the full model.

The repaired codec worker `38294496529` completed successfully. All eight
original/native boundaries localize the published startup failure to
`dac_output_conv` and PCM: published weights pass against the actual original
modules loaded with those same rounded weights. Original all-F32 weights pass
48/48 canonical comparisons. Preserving only `decoder.snake1.alpha` and
`decoder.conv2.weight`/`bias` as original F32 also passes 48/48, at PCM cosine
0.9999927 and norm error 0.002027489. Published PCM remains a recorded failure
(cosine 0.9976444, norm error 0.04271824). All 134 published coefficients exactly
match rounded original values; four missing/scaled negatives correctly fail.
The narrow candidate is 108,697,184 bytes versus 108,695,872 published:
only 1,312 extra bytes, including its diagnostic metadata/alignment.

`models/convert-dia-dac-to-gguf.py` makes the original critical-output export
reproducible, using `descript-audio-codec==1.0.0` and the exact original checkpoint
SHA256. It requires an explicit precision policy and a new output/receipt path.
The shared lookup covers all 134 actual Parameters and rejects unknown names.
Torch's original weight-normalization removal supplies effective weights;
rounded published weights are never cast back to F32 as a substitute. Physical
audits require exact original promoted payloads and unchanged unrelated tensor
types/bytes. Reader mappings close deterministically, including on CIFS errors.
Six tiny real-module/physical-export contracts pass, without actual-checkpoint
speech evidence.

The hosted `dia_dac_export_validate.py` worker must reproduce the already tested
candidate byte for byte, then run fresh strict startup and full-utterance
original-history comparisons. This is an explicit diagnostic conversion;
published models, pins, defaults, original references and thresholds remain
unchanged. Native free speech, Q4 neural parity and physical GPU behavior are
separate unresolved acceptance gates.

Sampler control `38299826144` finished with speech rejection: canonical Torch
original WER 0; original neural/DAC using the compiled native sampler and
native free generation both WER 1. Exact input prefix is 37 steps; minimum
neural cosine 0.9999993818 and maximum norm error 0.0006661525 pass. First
pre-override sample difference is step 36/head 8 (226 versus 221), with first
input difference at step 37. Original/native-sampler PCM is 16.521 seconds,
RMS 0.000170839, versus native 13.305 seconds, RMS 0.000384941. This establishes
that the native sampling path can trigger quiet short output independently of
the native neural graph or codec. Exact probability/filter and random-stream
effects still need isolation; failed original controls never waive native gates.
The complete proof is retained in `docs/historical/dia-native-sampler-control-2026-10-12.json`.

The real-logit probability observer at `2cdf631b3` executes the actual pinned
CFG/masking/sampler source and reads the native distribution through a
mechanically instrumented actual header. Local weight-free replay reproduces
all 333 retained native samples in both arms, with zero support mismatches and
maximum probability error 3.6076092e-7. Local Torch is actually 2.11.0+cu130;
this does not establish the pinned hosted 2.6.0 result. Complete probability,
support and variation arrays plus source/observer hashes are retained in
`docs/historical/dia-sampling-probability-local-2026-10-12.json`. Hosted run `38307628341`
repeats the component check under Torch 2.6. No model weights are loaded;
speech, native RNG compatibility and production acceptance remain separate.


Completed fixed-cache control (GH 38357576078, source 2205d920d): all three
F16/F32/F16-repeat arms pass 40/40 canonical stages, but all six CLI/C ABI
speech controls remain empty/WER 1. OFF/repeat inputs, logits and PCM match
exactly. Actual cross-cache vectors double from 11,501,568 to 23,003,136 bytes;
F32 extends shared input history from 865 to 979 steps and reduces maximum
aligned logit norm error from 2.40e-5 to 5.01e-6. At step 978/head 2, tiny neural
changes include token 904 across the nucleus boundary; original/native filters
agree on identical logits and reproduce the recorded draws. This is a rejected
speech control, with no sampling-policy or quality-gate waiver.
Complete [cache controls](dia-cross-cache-hosted-2026-10-13.json) and [first-draw replay](dia-cross-cache-boundary-2026-10-13.json).
