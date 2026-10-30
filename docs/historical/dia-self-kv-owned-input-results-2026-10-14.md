# Dia late self-key/value producer: owned-input results

At step978/block0, the actual pinned original DenseGeneral/RotaryEmbedding
and Attention.forward key/value statements run on each producer's captured
pre-self-norm input. The inputs are byte-identical here. Both complete original
F32 tensors and mapped native layouts match the physical model audit; only
8,388,608 coefficient bytes are fetched, without loading the full checkpoint.

| Same-input operation | Minimum row cosine | Maximum row norm error | Relative L2 |
|---|---:|---:|---:|
| Key projection | 0.999999999999963 | 8.365e-8 | 2.290e-7 |
| Key including source-RoPE-off rotation | 0.999999999079992 | 7.562e-8 | 3.553e-5 |
| Value projection | 0.999999999999980 | 5.575e-8 | 1.935e-7 |

Both native last K/V slots reproduce the accepted hosted full-cache snapshots
byte-for-byte. Repeats and one/four threads are exact. Local original versus
hosted original is measured separately: key relative L2 6.851e-7, value6.200e-7.
Ten comparison rows and fourteen scale/head-order/position negatives are
independently recalculated from retained arrays. Earlier cache-reorder and
all-layer attention-consumption proofs are reused rather than rerun.

The default Python interpreter is now3.13, while the retained Torch2.6 CPU
wheel targets3.12. Its initial ABI rejection is retained; the successful
probe uses /usr/bin/python3.12 and the existing private dependency prefixes.
An initial audit path-construction error is also retained and corrected before
acceptance. These tooling failures are separate from model numerical results.

[Complete 51-payload proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/1aae877cd70f1d3b67ba1d2b6de60f267575563a/dia-20261014-decoder-boundaries/complete-owned-self-kv.zip)
is public and anonymously verified: 17,646,336 bytes, SHA256
`ba6b41aa43c3e5f90ae94185d839388db9db0a898af5ff85042e9e3e2d88b62e`. It includes coefficients, owned inputs, outputs, actual source
expressions, compiled replay, verified CPU libraries, methods, licenses and
controls. [Full structured result](dia-self-kv-owned-input-results-2026-10-14.json).

This closes the observed late key/value producer gap. It supplies no numerical
repair, full-model speech, Q4, GPU, speed or release acceptance. Native speech
still fails the ASR roundtrip. See the separate first-F32-divergence diagnosis.
