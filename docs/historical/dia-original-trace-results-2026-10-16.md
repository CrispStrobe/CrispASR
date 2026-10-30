# Dia actual original trace and single-decision sensitivity — 2026-10-16

Follow-up: the [reciprocal native experiment](dia-native-fork-results-2026-10-16.md)
is complete and independently audited. The one-token restoration recovers WER0
and the original input/code trajectory for this prompt/seed. Earlier pending
statements below describe the historical preparation checkpoint. No production
speech/default/GPU/Q4/performance acceptance follows.

Changing only one sampled decision is sufficient to destroy this original
Python model's speech for the pinned prompt/seed. This is now an independently
audited conditional result, not a claimed native arithmetic repair.

The unchanged original transcribes “We are committed to making it secure.”
with WER0. The same original with only step978/codebook2 sampled877 replaced
by904 produces an empty transcript, WER1. The actual original sampler still
runs once per step; cloning/replacing that entry adds no random draw. Inputs
and raw logits through step978 are byte-exact, input979 differs only in that
entry, and the first963 undelayed code frames remain exact. Other generation
settings, weights, source, DAC and recognizer remain fixed.

| Original arm | Duration | PCM RMS | Peak | ASR WER |
|---|---:|---:|---:|---:|
| Unchanged | 14.048073 s | 0.0735338613 | 0.993742466 | 0 |
| One sampled decision replaced | 12.759365 s | 0.0001576508 | 0.009593331 | 1 (empty transcript) |

The fork ends after1115 decoder calls versus1226 unchanged. It is a quiet,
different trajectory. This demonstrates sufficiency for this prompt/seed; it
does not prove that the native runtime has only one cause, or that reversing
that token in the native trajectory would restore speech. A native reciprocal
intervention must be validated separately, with exact controls and roundtrip.
Never ship a token override as a production fix.

## Actual original intermediates

All18 self Q/K/V/attention, pre-RoPE Q/K/V projections,16 decoder boundaries
per layer and actual returned cross-cache valid39-position prefixes are now
captured simultaneously from the running full original. No shadow/module replay
is relabeled as a full-model observation. Full OFF/ON/OFF trajectories, logits,
codes, PCM and RNG witnesses are exact. Original token/code grids reproduce the
canonical archive. All18 captured original layer outputs plus final hidden
remain byte-exact to the previously accepted original captures.

The independent audit verifies549 payloads,41/491/41/41 arm arrays, complete
ZIP/CRC/SHA/source provenance and lossless array references. It recalculates125
comparisons against the accepted native raw captures from pinned8d0ad886/C36,
with full F32 neural weights, F32 cross caches, portable RNG enabled and source
RoPE disabled. Those retained position978 captures use an aligned history; they
are diagnostic runtime/configuration results, not current-default speech. Every comparison has
cosine at least0.9999999960597721; maximum relative L2 is6.284418e-5.
Cosine alone therefore does not establish decoded-output parity.

| Layer | Minimum cosine (Q/K/V/attention/output) | Q rel L2 | K rel L2 | V rel L2 | Attention rel L2 | Output rel L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.999999998114 | 5.374407e-05 | 1.573064e-05 | 6.787689e-07 | 4.711227e-06 | 1.436099e-06 |
| 1 | 0.999999996816 | 6.284418e-05 | 2.070144e-05 | 1.429572e-06 | 1.743098e-06 | 1.450769e-06 |
| 2 | 0.999999997598 | 4.340841e-05 | 1.800439e-05 | 2.126236e-06 | 5.978566e-06 | 2.453093e-06 |
| 3 | 0.999999998276 | 4.314648e-05 | 1.696211e-05 | 4.650754e-06 | 1.062243e-05 | 3.806002e-06 |
| 4 | 0.999999998597 | 3.270692e-05 | 1.259614e-05 | 6.444500e-06 | 3.532966e-06 | 3.166903e-06 |
| 5 | 0.999999998691 | 3.394416e-05 | 1.246532e-05 | 4.568066e-06 | 2.732186e-06 | 2.534920e-06 |
| 6 | 0.999999997915 | 3.779093e-05 | 1.921819e-05 | 3.422581e-06 | 9.120348e-06 | 2.799527e-06 |
| 7 | 0.999999998461 | 4.058995e-05 | 1.860262e-05 | 3.737950e-06 | 6.067856e-06 | 2.531559e-06 |
| 8 | 0.999999998020 | 5.395486e-05 | 1.473672e-05 | 2.548122e-06 | 4.160506e-06 | 2.467320e-06 |
| 9 | 0.999999997182 | 5.280091e-05 | 1.893510e-05 | 3.030614e-06 | 6.063091e-06 | 2.959207e-06 |
| 10 | 0.999999997605 | 5.431305e-05 | 1.724584e-05 | 5.824430e-06 | 8.391866e-06 | 2.572805e-06 |
| 11 | 0.999999996882 | 5.666565e-05 | 1.960325e-05 | 3.376455e-06 | 3.725995e-06 | 2.182034e-06 |
| 12 | 0.999999996922 | 5.506546e-05 | 1.822813e-05 | 2.826027e-06 | 4.526211e-06 | 2.254909e-06 |
| 13 | 0.999999997674 | 4.915910e-05 | 1.656531e-05 | 3.760173e-06 | 4.811864e-06 | 2.045642e-06 |
| 14 | 0.999999997571 | 5.515456e-05 | 1.821092e-05 | 2.735907e-06 | 3.051448e-06 | 1.998791e-06 |
| 15 | 0.999999996060 | 5.937625e-05 | 2.712398e-05 | 2.548583e-06 | 3.167518e-06 | 1.970837e-06 |
| 16 | 0.999999997398 | 5.486568e-05 | 1.872366e-05 | 2.575558e-06 | 2.615295e-06 | 2.126875e-06 |
| 17 | 0.999999997129 | 5.815125e-05 | 1.830551e-05 | 2.366480e-06 | 3.265524e-06 | 2.347026e-06 |

Each cosine is the minimum row cosine, with Q/K/V/attention rows preserving
head dimension128; layer-output rows preserve dimension2048. Relative L2 is
computed over the complete stage. Native and original RMS, maximum absolute
errors and all125 individual cosines are in the [complete metric receipt](dia-original-trace-results-2026-10-16.json).
These are position978 comparisons on the aligned prefix, not full-trajectory
native speech acceptance.

## Reproducibility and limits

Heavy [38818571466](https://github.com/CrispStrobe/CrispASR/actions/runs/38818571466)
and all applicable [lint checks38818570959](https://github.com/CrispStrobe/CrispASR/actions/runs/38818570959)
pass source9ed6cff6f12627fddb61ea0ff7410b50950fd2b3. Independent auditor4b0df07b
and its15 rejecting controls pass locally; integrated tooling6f001996 preserves
those accepted source bytes after rebase. Original4a9e29b1/caba weights and
reference8d0ad886 run Torch2.6 CPU/four threads, seed42,3072-token capacity,
temperature1.2, top_p.95, top_k45 and CFG3. The same actual C ABI Nemotron
recognizer and resampler score both audio files.

Official artifact12014433279 is338692054 bytes, SHA256
`c57ddb8fba5fd6c2c96eca39192c3cb38f4079924a5efbc788c60a7462c8ced4`.
The [method and independent audit](../../tools/ci-heavy/dia-original-trace/README.md)
describe source pins, complete-array reconstruction and reproduction commands.
Complete [raw proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/bed84796a4f829f87b017ad765a2f3a6f8cd335e/dia-20261016-original-trace/complete-hosted-original-trace-v1/raw.zip) and [publication manifest](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/bed84796a4f829f87b017ad765a2f3a6f8cd335e/dia-20261016-original-trace/complete-hosted-original-trace-v1/manifest.json)
are public at immutable revisionbed84796a4f829f87b017ad765a2f3a6f8cd335e.
Every published file was downloaded anonymously and verified by full byte count
and SHA256, including the complete raw archive.

No native math/default changes, unique runtime defect, native speech repair,
GPU/Q4/performance promotion or release acceptance follows. The result links
the earlier nucleus-boundary discrepancy to severe trajectory sensitivity in
the original itself, while the [terminal diagnosis](dia-terminal-replay-results-2026-10-15.md)
continues to constrain where the small native logit difference arises.
