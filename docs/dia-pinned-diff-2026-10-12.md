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
