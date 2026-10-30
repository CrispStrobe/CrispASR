# Dia owned-input query and rotary isolation — 2026-10-14

Actual original DenseGeneral and RotaryEmbedding classes, with the original
Attention.forward query-driving statements, were executed on both producers’
owned block-zero step978 pre-self/cross norm outputs. Native replay uses the
literal pinned query projections and rotary helper. All original coefficients
and converted tensors match the physically audited F32 model.

| Matching-input operation | Maximum relative L2 | Minimum row cosine | Maximum row norm error |
|---|---:|---:|---:|
| Self query projection | 1.481e-7 | 0.9999999999999826 | 6.989e-8 |
| Self query including rotary | 5.371e-5 | 0.9999999981255697 | 6.716e-8 |
| Cross query projection | 1.251e-7 | 0.9999999999999878 | 5.726e-8 |
| Cross query including rotary | 5.167e-6 | 0.9999999998529738 | 6.345e-8 |

Native cross query reproduces the complete hosted native snapshot byte for
byte. One/four CPU threads and repeats agree byte for byte. There is no hosted
self-query snapshot; its matching-input comparison is bounded primitive proof.
Local original cross query versus the hosted original differs by relative L2
6.237e-7, a separately recorded cross-host effect. The self-query inputs from
both producers are identical; cross-query input drift contributes 3.948e-7
relative L2 through the actual original operator.

Fourteen metric rows and sixteen negative controls are retained. Independent
recalculation verifies all eight matching-input rows and both complete native
thread/repeat streams. Double-scale controls fail the magnitude checks despite
cosine1; shifted positions and swapped rotary halves also fail. No tolerance
or generation recipe was changed.

[Complete public 43-payload proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/ae150c8ebb57a3474371b493e2a0aae6b1ae2b1c/dia-20261014-decoder-boundaries/complete-owned-query.zip),
SHA-256 `2895afedee72ccae36f1b81ea06408ddf5de70cdd7a1900d852fa8e6a30bdf90`, 65,258,290 bytes, was anonymously downloaded
and verified in full. It includes original coefficients, native mappings,
source expressions, owned inputs/outputs, replay executable and CPU libraries,
receipts and audit methods. [Full result ledger](dia-query-owned-input-results-2026-10-14.json).
Local artifacts remain under `${CRISPASR_STORAGE}/crispasr/dia-query-owned-replay-20261014`.

This separates query projection error from the larger source-RoPE-off rotary
error. Earlier source-RoPE-on full runs improved primitive parity but failed
speech; this result supplies no new reason to promote that option. Native
speech remains rejected with empty ASR and WER1.0. No full-model speech, Q4,
GPU, performance, or release acceptance follows from this bounded CPU replay.
