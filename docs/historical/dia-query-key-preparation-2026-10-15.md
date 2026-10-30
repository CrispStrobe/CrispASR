# Dia original self-query and paired query/key diagnostic

**Completed and integrated:** tested diagnostic tools are integrated at 63780005. Hosted and applicable lint checks pass. The [completed Q/K result](dia-query-key-results-2026-10-15.md) remains negative for restoring the disputed ordering. Earlier preparation checkpoints below are historical; they are not current pending jobs.

Candidate `ff7e07c0f5ec846ff7de569c117c1d0b30af404d` captures actual original self-attention Q and adds a
scratch-only five-arm full-decoder diagnostic. It is now integrated as diagnostic tooling. Production
runtime and defaults remain unchanged.

## Original-module query provenance

The producer executes unchanged original Attention/DenseGeneral/RotaryEmbedding
classes at source `4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8`, observing Q before
and after RoPE at step978. It loads only five physically verified F32 layer0
coefficient tensors (41,951,232 bytes), plus retained owned inputs and caches.
It never loads the full neural model on the VPS. The retained full-model GGUF
contains cross-Q only; no earlier full-model self-Q is relabeled or invented.

All retained original standalone norm/attention/output stages are byte-exact.
Against the separate full-model reference:

| Stage | Relative L2 | Maximum row norm error | Minimum cosine |
| --- | ---: | ---: | ---: |
| Pre-SA norm | 0 | 0 | 1 |
| Self-attention | 9.529530817e-7 | 1.286218893e-6 | 0.999999999990153 |
| Self-attention output | 1.075296292e-6 | 1.164986831e-7 | 0.999999999999429 |

These pass the unchanged strict original key-reconstruction gate (relative L2
and row norm error at most2e-6; minimum cosine0.9999999999). The last two stages
are not byte-exact. The observed query belongs to actual original-module replay,
not to an earlier full-model query capture. That limitation is carried into the
hosted receipt and forbids a canonical full-model or speech acceptance claim.

Native query versus replayed-original query has relative L2 5.370203745e-5,
minimum cosine0.999999998126 and maximum row norm error6.715343792e-8. Fresh
repeats are exact, and using the original3072-slot cache instead of the earlier
979-slot component cache leaves the query unchanged. Doubling Q or reversing
its heads gives output relative errors0.08069 and0.80562; both reject.

An independent raw-input replay rehashes all49 payloads, independently executes
the original Q projection/RoPE, recomputes all full-reference metrics and exactly
reproduces every baseline/negative attention array. It does not import the
producer's capture implementation.

The [complete producer proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/dce9d23fc048c78e57729e403af04ed97dd78adb/dia-20261015-query-key/complete-original-query-replay-v1.zip) is
46884243 bytes, SHA256 `33c26a38ca567b0872b90615c25e903c30679db4d7b2a35f6ac144687795c6c3`.
It retains59 payloads including consumed coefficients/source/inputs, all traces,
controls, audit and notices. The [compact runtime input ZIP](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/dce9d23fc048c78e57729e403af04ed97dd78adb/dia-20261015-query-key/query-inputs-v1.zip) is
26217 bytes, SHA256 `7b20c5f07250621ae9fde8e4aec9e8ba1ab114f420dd95b30d70aea27381affb`.
Both anonymous full downloads match. Their immutable revision is
`dce9d23fc048c78e57729e403af04ed97dd78adb`.

## Hosted factorial and controls

The [heavy run38764924112](https://github.com/CrispStrobe/CrispASR/actions/runs/38764924112)
is running at the exact candidate source; its worker step started16:57 UTC
and remains active at the17:15 UTC checkpoint. The internal build/export/replay
phase is not exposed by the current API response. Latest
[lint38767820341](https://github.com/CrispStrobe/CrispASR/actions/runs/38767820341)
is submitted at `b1de56284867dfcd45bc90289aadbb34a189f4e2`, superseding the earlier
manual lint38764928533. This commit adds the completed
[native query-producer isolation](dia-native-query-producer-2026-10-15.md);
the full-model worker, patch, selftest and original-query producer are unchanged.
No hosted numerical pass or integration is claimed.

The arms are baseline, native Q/K substitution, replayed-original Q/native K,
native Q/captured-original K, and replayed-original Q/captured-original K.
Both native controls must preserve all979 input/logit rows and captured stages
exactly. Every requested Q/K tensor must be the actual consumed attention
input; earlier978 logits and layer0 input/norm/V/cross inputs stay exact.
All18 layer outputs, final hidden/logits, norms/errors and original Torch2.6
sampler46/904 margins/ranks/support survive every arm and failure.

Local factorial ownership guards reject six malformed arrays. Actual compiled
Q/K readers reject missing, truncated, extra and nonfinite files. Full patched
runtime syntax passes with exact c36dab89 headers. A small graph linked to the
actual prior hosted ggml libraries proves renamed Q/K leaves survive, the old
query producer is pruned, consumed inputs are exact and dot products are exact.
The first local compile used an incorrect include directory; its failure is
retained beside the corrected command/output in local-guards-v2. No source or
acceptance gate was changed to bypass it.

The [candidate method](https://github.com/CrispStrobe/CrispASR/blob/ff7e07c0f5ec846ff7de569c117c1d0b30af404d/tools/ci-heavy/dia-query-key-intervention/README.md)
records the exact pipeline and dispatch. Full hosted artifacts still require
independent source/hash/CRC/array audit before interpretation or integration.
This conditional diagnostic cannot promote speech, GPU, Q4, speed or defaults.
All large files remain under ${CRISPASR_STORAGE}/crispasr/dia-query-key-20261015.

## Hosted audit preparation checkpoint — 17:15 UTC

The independent audit verifies official archive size/SHA/CRC/safe paths, complete
payload sets, methods against immutable git objects, input manifests, native
controls, actual consumed Q/K, all979 shared input/logit rows, all18 stage metrics
and actual original sampler supports/ranks. It additionally cross-checks native
and original-key-only arms against the prior accepted key experiment.

Its shared checks pass a fresh calibration against the previous real successful
key-only artifact:720 payloads, both exact native controls, all reported stages
and original sampling supports reproduce. This does not execute the new paired
Q/K-specific branches or accept the pending factorial. The prepared auditor and
calibration logs/results remain under the owned storage directory; the new raw
artifact is still unavailable. Latest candidate lint has4/10 successful jobs;
the standalone attention-tool lint has1/10, with the rest queued.
