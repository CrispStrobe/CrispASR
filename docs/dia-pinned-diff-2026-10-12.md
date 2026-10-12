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
have `[S1]`/`[S2]` and final punctuation, matching native preparation. Interior
padding is rejected. Captures own storage; hooks/methods are restored on failure.
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

Local validation: three tiny-original neural capture tests, production boundary
regression, C++ syntax checks and Python compilation. Only the tiny test's final
DAC is a dummy code projection; it provides no PCM/quality evidence. Hosted
real-weight/DAC numerical and negative-control results are pending.
