# Dia block-zero cross-cache: owned-input replay

The existing CPU and actual T4 captures now have an independent cross-K/V
check using the original Python producer on each capture's own encoder output.
This isolates upstream input drift from projection/rotation differences without
a new full-model or GPU run. The original GPU job remains ERROR from its later
receipt serialization failure; this replay does not change that status.

Original source is `4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8`. Actual
DenseGeneral, RotaryEmbedding, KVCache and Decoder.precompute_cross_attn_cache
ASTs execute unchanged under Torch 2.6.0 CPU with four threads. Both source
files match immutable Git objects. The two complete original F32 weights were
fetched through strictly validated immutable HTTP ranges, and both mapped
layouts match the full physical F32 export audit. Only 16,777,216 coefficient
bytes were fetched; the complete checkpoint was not loaded or locally rehashed.

Each encoder input has shape `(2, 39, 1024)`, and cross-K/V has shape
`(2, 39, 2048)` with 16 heads of 128 values. Positions use the original F32
arange/expansion convention. The trimmed real-token prefix is valid for this
per-position producer; this replay makes no cross-attention mask claim.
Every input file is checked against its retained producer ledger. Metrics
compare 128-value head rows, including magnitudes.

| Native observed output vs original CPU producer on the SAME input | Minimum row cosine | Maximum row norm error | Relative L2 |
| --- | ---: | ---: | ---: |
| CPU K | 0.999999999982435 | 5.124e-7 | 3.392e-7 |
| CPU V | 0.999999999999491 | 2.228e-7 | 2.754e-7 |
| Actual T4 K | 0.999999999980864 | 1.458e-6 | 7.201e-7 |
| Actual T4 V | 0.999999999999357 | 7.564e-7 | 5.554e-7 |

The actual original producer on the native T4 encoder input differs from the
same producer on the original encoder input by relative L2 7.925e-7 for K
and 1.042e-6 for V. Thus encoder-input drift contributes to the recorded
cross-cache difference. Local original vs hosted original is also measured
explicitly: relative L2 1.610e-7 K and 2.307e-7 V, rather than assuming
cross-host byte equality. All local repeats match exactly.

Sixteen metric rows and eight negatives are retained. Uniform double scaling
preserves cosine but gives norm error/relative L2 1; head reversal rejects
for CPU/T4 K/V. This is a bounded correctness diagnostic, not evidence of
accepted native speech, Q4, full cross-attention or performance. No numerical
change follows merely from these small measured differences.

Next capture self-output projection/residual, pre-cross norm, cross-Q,
cross-attention pre/post-output, and pre-MLP/fused gate-up/SiLU/down/residual
at the fixed late step. Original MlpBlock uses one fused gate/up contraction;
compare that actual operator with the converter's audited split weights.
Additional hooks require complete OFF/ON/OFF logits, feedback and PCM controls
and unchanged quality negatives before interpreting their snapshots.

Full metrics, coefficient/source/input hashes and scope are in
`dia-cross-cache-input-replay-2026-10-13.json`.

Complete replay, original/mapped coefficients, actual source expressions,
receipts and controls are public and anonymously SHA256-verified:
[owned-input cross-cache proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/da448a0c531e7cac29a044fcfae213112248d3bd/dia-20261013-attention/owned-input-cross-cache.zip)
(33,615,876 bytes, 14 payloads; SHA256
`9968f67d6c7c187000cd734bdf4e1ccc6fe995bd9b5675eed4b342594b10496c`).
