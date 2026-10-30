# Dia hosted terminal hidden-state preparation — 2026-10-15

**Completed:** [accepted result](dia-terminal-replay-results-2026-10-15.md) supersedes this historical preparation checkpoint.

The [accepted37-arm suffix result](dia-suffix-replay-results-2026-10-15.md)
restores original support after output17 replacement, but leaves margin
+2.09808349609375e-5 versus original+4.6253204345703125e-5. The replacement
is the actual original captured post-layer activation. This new experiment
isolates final-hidden injection in the actual hosted graph, preserving the
known original-layer17 anchor and all native controls.

Candidate: [85be03e353cd2aa4da5529acfc41585eecf87116](https://github.com/CrispStrobe/CrispASR/commit/85be03e353cd2aa4da5529acfc41585eecf87116).
[Method](https://github.com/CrispStrobe/CrispASR/blob/85be03e353cd2aa4da5529acfc41585eecf87116/tools/ci-heavy/dia-terminal-replay/README.md).
[Heavy38805728288](https://github.com/CrispStrobe/CrispASR/actions/runs/38805728288)
and [lint38805731957](https://github.com/CrispStrobe/CrispASR/actions/runs/38805731957)
are queued at this checkpoint; no terminal numerical result is available.

The worker pins8d0ad886/C36 and prior public archive revision
049d93d4d97ce7a542a030b4a20366df8b372699, SHA
`162e4e25317e60b5b9ee703feb51aa7112c8e73e0ba89a49c55da493c56aeb61`.
Baseline, native layer17 and native final hidden must reproduce all127 stages
and logits exactly. Original layer17 must reproduce all accepted prior anchor
bytes. Only then is actual original final hidden substituted after final RMSNorm
and before all9 heads. Every earlier stage and cache must remain native exact.
Both computed and consumed hidden remain graph outputs; the discarded producer
is an explicit graph root. Simultaneous layer/hidden replacements reject.

The original Python decoder source was checked directly: all18 blocks, final
F32 RMSNorm and DenseGeneral head run in that order. Full patched C++ source
passes syntax against exact pinned headers. Accepted finite file/prefix tests,
single-use patch tests and all4 unskipped secret guards pass locally. The
actual ggml ownership guard is reused and will be rebuilt/executed on GH.
Local preflight receipt remains under
`${CRISPASR_STORAGE}/crispasr/dia-terminal-20261015/local-v1/preflight.json`.

The worker records actual logits, all18 layer metrics with norms/errors, hidden
metrics, all18 cross-cache identity checks and original Torch2.6 filter support,
ranks and margin. It retains complete byte hashes and lossless reconstruction
recipes. Independent artifact audit is mandatory before interpreting results.
No production/default change, sampled trajectory, speech, Q4, GPU, speed or
release acceptance is established. Original full-model self-Q provenance remains
a separate follow-up; unchanged local late-layer/norm/head probes are not rerun.

## Independent auditor preparation

[Auditor63468e54](https://github.com/CrispStrobe/CrispASR/blob/63468e54/tools/ci-heavy/dia-terminal-replay/audit.py)
is prepared separately from the unchanged submitted worker85be03e3. It requires
official terminal run/archive proof, all payload/source hashes, exact composed
scratch patch, actual ELF/library identities and a fresh retained ggml guard.
All five arms, both exact native controls and the complete accepted original
layer17 anchor must pass. It reconstructs every logical capture and checks
computed/consumed hidden, all upstream stages, split logits,90 layer metric
rows, hidden/logit errors and the actual pinned original filter.

Separate observed F32 CFG and exact-input F64 CFG sequences retain all four raw
logits. Baseline→original post17→original final hidden→original logits receives
signed conditional differences; those must telescope exactly. These are not
global additive causal percentages. Unknown hosted values are not filled from
local component reconstructions.

Local and Torch2.6 reference checks pass the existing24 calibrated reconstruction/
metric negatives and12 new activation/hash/nonfinite negatives. Signed binary
fractions have hand-derived exact differences. Two accepted actual native/
original layer17 captures validate the ownership checks, the composed runtime
patch reproduces exact preflight bytes, and three real prior CFG cases agree
with exact rational arithmetic. [Calibration receipt](dia-terminal-auditor-calibration-2026-10-15.json).
Raw receipts remain under
`${CRISPASR_STORAGE}/crispasr/dia-terminal-20261015/auditor-calibration-v1/` and
`auditor-calibration-v2/`. No new terminal hidden-state or speech result follows;
both GH numerical and lint jobs remain queued at this checkpoint.
