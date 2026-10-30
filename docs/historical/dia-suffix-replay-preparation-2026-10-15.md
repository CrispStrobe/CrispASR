# Dia one-step suffix replay preparation — 2026-10-15

The candidate is now accepted and integrated. [Completed result](dia-suffix-replay-results-2026-10-15.md) supersedes this preparation checkpoint.
No production runtime or default changes are proposed by this diagnostic.

- Immutable candidate: [70f2d9de678b31f6b55f957e3a721aa91bf0af88](https://github.com/CrispStrobe/CrispASR/commit/70f2d9de678b31f6b55f957e3a721aa91bf0af88).
- [Full worker/method](https://github.com/CrispStrobe/CrispASR/blob/70f2d9de678b31f6b55f957e3a721aa91bf0af88/tools/ci-heavy/dia-suffix-replay/README.md).
- [Hosted numerical job38793672943](https://github.com/CrispStrobe/CrispASR/actions/runs/38793672943): completed successfully; independent artifact audit passes.
- [Lint38793677203](https://github.com/CrispStrobe/CrispASR/actions/runs/38793677203): completed successfully, all10 jobs.

The pinned runtime8d0ad886/ggmlC36 imports native self-cache positions0..977
from the independently audited full-decoder Q/K artifact. It computes only
position978 with the original forced token row and native encoder/cross producer.
Baseline must reproduce all127 captured stages and final logits byte-for-byte;
all18 native post-layer replacement controls must pass the same gate before
all18 original captured post-layer replacements run. Each original arm changes
one activation independently while holding the historical self prefixes native.

Both computed and consumed activations remain actual graph outputs. An explicit
root preserves the replaced producer; the replacement input is retained by
pointer after downstream naming. Every arm captures all18 cross K/V pairs and
must match the baseline bytes. Earlier layers and current-layer self Q/K/V must
stay exact; all18 stage cosines/norms/errors, hidden and logits plus original
Torch2.6 sampler margin/ranks/support are reported. Original sampler source and
actual full-model original activation captures come from the verified prior
artifact. The full original model is not instantiated.

Large captures retain complete SHA/byte identities with a lossless encoding:
unchanged files reference baseline bytes; changed self K/V stores only the
current2×512 row after asserting all978 past rows exact. Reconstructing the full
array must reproduce its original SHA before the full redundant file is removed.
The baseline, current rows, references and complete reconstruction recipe remain
in the output. All cross caches retain one baseline plus per-arm exact hashes.

Local preparation passes the actual ggml prefix and graph ownership guard,
finite exact-size loader rejection controls, lossless cache reconstruction with
changed-prefix rejection, ZIP path/hash controls and single-use pinned patch.
The complete patched runtime passes C++17 syntax against exactC36 headers;
format18 and all4 unskipped secret guards pass. Local command receipts and logs
are under `${CRISPASR_STORAGE}/crispasr/dia-suffix-20261015/local-v1/`, including
`local-validation.json`. These are source/component controls, not new numerical
or decoded-output evidence. The candidate branch stays separate until hosted
validation and independent artifact audit complete.

Skipping978 iterations does not reproduce prefix RNG state. The worker observes
the actual original filter before multinomial drawing; it cannot claim a sampled
trajectory, speech repair, Q4, performance or GPU acceptance. Missing simultaneous
full-model original self-Q capture remains a separate follow-up. The earlier
[Q/K factorial result](dia-query-key-results-2026-10-15.md) remains definitive
for that unchanged experiment; do not repeat it while waiting for this new probe.

## Independent auditor preparation

[Auditor5cb83887](https://github.com/CrispStrobe/CrispASR/blob/5cb83887/tools/ci-heavy/dia-suffix-replay/audit.py)
and its weight-free calibration are prepared separately from the unchanged
submitted worker70f2d9de. The actual hosted artifact still must pass this audit.
Independent byte slicing reconstructs all3 capture encodings;24 broken metadata,
path, hash, row and metric controls reject, and hand-derived3-4-5 metric values
pass in both local and Torch2.6 reference environments. The actual original
sampler reproduces prior original/native margins +4.6253204345703125e-5 and
−2.6702880859375e-5, ranks32/33 and33/32, and both exact retained support lists.
This is calibration on already audited logits, not a new numerical experiment.

The audit checks official terminal run/artifact identity, SHA/size/CRC, safe
unique paths, extracted root receipt and all payload hashes. It checks immutable
method git objects, reproduces the scratch patch, checks ELF/library identities
and reruns the retained actual ggml guard. It then reconstructs all captures,
verifies native controls, prefixes, computed/consumed activations, earlier block
boundaries and split logits, and recalculates all666 layer metrics, hidden/logit
errors and the actual pinned filter. Calibration logs/receipt remain under
`${CRISPASR_STORAGE}/crispasr/dia-suffix-20261015/auditor-calibration-v2/`.

The independent numerical auditor additionally reproduces90 actual layer rows,
five hidden-state rows and five logit rows from the completed Q/K factorial,
and all3 encodings reconstruct exact real native/query-intervention captures.
[Full calibration receipt](dia-suffix-auditor-calibration-2026-10-15.json).
The already audited input source isff7e07c0; these checks validate the auditor,
not the running suffix experiment. Raw calibration data/receipt remain in
`${CRISPASR_STORAGE}/crispasr/dia-suffix-20261015/auditor-real-calibration-v1/`.
