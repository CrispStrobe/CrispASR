# Dia actual full-model query versus earlier module replay — 2026-10-16

The newly audited full-original capture confirms that the earlier Q/K factorial
used a qualified module-replayed query, not byte-identical full-model Q. This
read-only comparison verifies each input's SHA against its retained manifest
and checks the full-original/factorial receipts against their official ZIP
entries. The module receipt is also checked against its complete public archive,
whose SHA256 is independently verified as
`33c26a38ca567b0872b90615c25e903c30679db4d7b2a35f6ac144687795c6c3`. It executes no neural model, attention graph or sampler intervention.

| Earlier capture versus actual full original | Byte-exact | Relative L2 | Maximum absolute error |
|---|---|---:|---:|
| hosted-factorial-query | no | 6.902965433e-07 | 3.576278687e-06 |
| hosted-factorial-keys | yes | 0.000000000e+00 | 0.000000000e+00 |
| module-pre_sa_norm | yes | 0.000000000e+00 | 0.000000000e+00 |
| module-pre_rope_query | no | 6.906207657e-07 | 4.053115845e-06 |
| module-query | no | 6.902965433e-07 | 3.576278687e-06 |
| module-sa_attention | no | 9.529530817e-07 | 1.102685928e-06 |
| module-sa_output | no | 1.075296292e-06 | 1.192092896e-06 |

The actual original keys and pre-attention normalized input match the earlier
captures exactly. The standalone query projection and rotated query differ
slightly; attention/output differences are also retained. Identical captured
input values do not by themselves guarantee component/full-model floating-point
byte reproducibility. This does not identify a mathematical runtime defect or
isolate its physical cause; no operator or precision setting was changed.

The earlier factorial's qualification remains necessary. Its result must not
be relabeled as an actual-full-Q intervention. The [full original trace](dia-original-trace-results-2026-10-16.md)
now supplies actual full-model tensors for any justified later intervention.
The reciprocal native sampled-decision experiment remains the current pending
speech test; this small comparison does not justify repeating a sweep unchanged
or establish native speech/GPU/Q4/performance acceptance.

[Complete seven-row receipt with input hashes, cosine, RMS and source pins](dia-query-capture-reconciliation-2026-10-16.json).
Local reproduction script/receipt live under
${CRISPASR_STORAGE}/crispasr/dia-native-fork-20261016/query-provenance-v1.

The [earlier complete module proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/dce9d23fc048c78e57729e403af04ed97dd78adb/dia-20261015-query-key/complete-original-query-replay-v1.zip)
and [factorial result/proof](dia-query-key-results-2026-10-15.md) retain the old
inputs; the full-original result linked above retains the new captures.
