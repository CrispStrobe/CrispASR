# Dia reciprocal native token fork — 2026-10-16

Changing only sampled token904→877 at step978/codebook2 restores the expected
sentence in the pinned native trajectory. The complete independent audit passes
642 payloads, exact pristine/OFF/identity controls, earlier fork prefix and RNG,
physical F32 exports, retained runtime/library identities and actual DAC/ASR
provenance. All1,226 decoder input rows then match the actual original, and all
1,210 original code frames match exactly. Native8d0ad886 retains one additional
output frame under its historical reversal protocol; that difference is explicit.

| Arm | Decoder calls | Native DAC WER | Original DAC WER | Native PCM RMS |
|---|---:|---:|---:|---:|
| Pristine | 1,115 | 1 | not run | 0.0001556675 |
| Observer OFF | 1,115 | 1 | 1 | 0.0001556675 |
| Identity904 | 1,115 | 1 | 1 | 0.0001556675 |
| Restore877 | 1,226 | 0 | 0 | 0.0734399214 |

All failed native controls have an empty transcript. Both restore decoders
produce **“We are committed to making it secure.”** Original canonical PCM and
its actual original DAC reconstruction both score WER0 through the same final
recognizer library. Original DAC reconstruction relative L2 is5.57170474e-7.
Native restoration lasts14.05968254 seconds versus12.77097506 for controls.

## Controls and interpretation

The unchanged actual nine-head sampler runs before substitution. Only the
completed sample at978/codebook2 changes. All979 earlier input/logit rows,
all127 selected native stage captures, the first963 code frames and the full
serialized pre/post MT state remain exact. Pristine, observer and identity
complete inputs/logits/PCM are byte-identical; observer/identity code grids and
final RNG are also identical. The prior accepted native prefix reproduces exactly.
The fresh actual pinned sampler guard and29 rejecting audit controls pass.

Together with the [original-only reverse intervention](dia-original-trace-results-2026-10-16.md),
this establishes bidirectional single-decision sensitivity for this prompt/seed.
It connects the previously measured discontinuous nucleus-support change to the
quiet trajectory. It does not identify a unique erroneous arithmetic operator,
prove general behavior across prompts/seeds, or justify a hardcoded replacement.
The retained [terminal margin decomposition](dia-terminal-replay-results-2026-10-15.md)
and [query provenance qualification](dia-query-capture-reconciliation-2026-10-16.md)
still apply. Ordinary floating-point differences remain distinct from a newly
proven algebraic defect.

Runtime8d0ad886/C36, full physically verified F32 neural weights, F32 cross
caches, portable RNG, source-RoPE off, four CPU threads, seed42 and the original
3,072-token capacity/temperature1.2/top_p0.95/top_k45/CFG3 recipe stay fixed.
This is the earlier diagnostic runtime, not a claim about current main defaults.
Later production top-k/output-protocol repairs remain independently scoped.
No production override/default change, native speech acceptance, GPU/Q4,
performance or release promotion follows.

## Evidence and reproduction

Heavy [38855559740](https://github.com/CrispStrobe/CrispASR/actions/runs/38855559740)
passes at worker5417ed6df0d69f69f8bda8a6ea6ce5f3e924e3da; all10 checks in
[lint38855559850](https://github.com/CrispStrobe/CrispASR/actions/runs/38855559850)
pass. Auditorc0fac6c1 corrects a debug-only log assumption and disables bytecode
writes into retained evidence; both earlier failed audit logs remain public.
No numerical/quality gate or worker bytes changed. V3 uses a fresh extraction,
regenerates the scratch patch and freshly compiles the actual sampler guard.
Rebased integrated methodfc3f620c preserves all8 accepted method files exactly.

Complete official raw archive:904,039,288 bytes, SHA256
`0cdd76a38bdae5be82b7a854b7d0c3f1194e47237afee4b68fb1164baca2ad47`.
[Public raw evidence](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/f0a412c02c23bf2625ec91011ed962493db66763/dia-20261016-native-fork/complete-hosted-native-fork-v1/raw.zip),
[public manifest](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/f0a412c02c23bf2625ec91011ed962493db66763/dia-20261016-native-fork/complete-hosted-native-fork-v1/manifest.json).
All24 published files pass full anonymous download, size and SHA verification at
immutable revisionf0a412c02c23bf2625ec91011ed962493db66763. The bundle includes
both failed audit logs, complete official logs, independent methods/results and
licenses. Local proof remains under the owned storage root's
hosted-38855559740-v1; failed audit extraction is retained separately.

[Structured results](dia-native-fork-results-2026-10-16.json) retain all ASR
scores, model/library/audio hashes, independent trajectory comparison and full
publication ledger. [Runnable method](../../tools/ci-heavy/dia-native-fork/README.md)
describes exact reproduction. No full neural model was run on the local VPS.
