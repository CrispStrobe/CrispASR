# Dia one-step suffix replay — 2026-10-15

The one-step replay is validated. Hosted38793672943 passes at70f2d9de,
lint38793677203 passes all10 checks, and independent audit verifies all1,323
payloads,37 arms and666 layer comparisons. The baseline reproduces every
one of127 retained stages and the final logits byte-for-byte. All18 native
post-layer replacement controls also reproduce those bytes exactly.

Each original arm substitutes one actual captured original post-layer output
at position978, retaining native positions0..977 in all18 self caches, the
same forced token row and the native encoder/cross producer. All18 cross K/V
pairs stay exact. Computed and consumed activations are retained separately.
All compacted captures reconstruct to their original complete SHA/byte counts;
the actual retained ggml guard also passes a fresh independent execution.

The original46/904 CFG margin is+4.6253204345703125e-5, ranks32/33.
Native baseline is−2.6702880859375e-5, ranks33/32. The actual pinned
Torch2.6 filter is observed before drawing; all support lists are recalculated.

| Replaced output layer | Baseline output cosine min | Baseline output relative L2 | Baseline norm mine/ref | Resulting CFG46−904 | Original support restored |
|---:|---:|---:|---:|---:|:---:|
| 0 | 0.999999999998972 | 1.436099e-06 | 0.999999972 | -3.242492676e-05 | no |
| 1 | 0.999999999998942 | 1.450769e-06 | 1.000000014 | -4.434585571e-05 | no |
| 2 | 0.999999999996969 | 2.453093e-06 | 1.000000180 | -3.671646118e-05 | no |
| 3 | 0.999999999992772 | 3.806002e-06 | 1.000000275 | -2.384185791e-05 | no |
| 4 | 0.999999999994875 | 3.166903e-06 | 1.000000180 | -2.002716064e-05 | no |
| 5 | 0.999999999996782 | 2.534920e-06 | 1.000000034 | -3.576278687e-05 | no |
| 6 | 0.999999999996075 | 2.799527e-06 | 1.000000085 | -2.098083496e-05 | no |
| 7 | 0.999999999996789 | 2.531559e-06 | 1.000000001 | -2.765655518e-05 | no |
| 8 | 0.999999999996874 | 2.467320e-06 | 0.999999992 | -1.382827759e-05 | no |
| 9 | 0.999999999995535 | 2.959207e-06 | 0.999999985 | -8.106231689e-06 | no |
| 10 | 0.999999999996523 | 2.572805e-06 | 1.000000004 | +5.245208740e-06 | yes |
| 11 | 0.999999999997609 | 2.182034e-06 | 1.000000053 | +1.001358032e-05 | yes |
| 12 | 0.999999999997439 | 2.254909e-06 | 1.000000084 | +2.861022949e-06 | yes |
| 13 | 0.999999999997807 | 2.045642e-06 | 1.000000024 | -5.722045898e-06 | no |
| 14 | 0.999999999997949 | 1.998791e-06 | 0.999999996 | +9.536743164e-06 | yes |
| 15 | 0.999999999997987 | 1.970837e-06 | 0.999999901 | -2.384185791e-06 | no |
| 16 | 0.999999999997661 | 2.126875e-06 | 0.999999927 | -3.337860107e-06 | no |
| 17 | 0.999999999996986 | 2.347026e-06 | 0.999999547 | +2.098083496e-05 | yes |

Layers10,11,12,14 and17 restore the exact original retained support.
The effect is non-monotonic:13,15 and16 reverse it again. This is suffix
sensitivity under fixed native prefixes, not a binary search locating one
defective block or additive causal attribution.

Replacing layer17 output reduces final-hidden relative L2 from2.2802e-6
to7.3802e-8, with margin+2.09808349609375e-5. Its consumed post-layer
activation is byte-identical to the actual original capture. The remaining
margin discrepancy therefore arises downstream through native final norm
and output heads in this intervention. Earlier local norm/head isolation
does not guarantee identical full-runtime/hosted floating-point reductions.
A bounded hosted final-hidden replacement can separate those terminal paths
without rerunning the978-position prefix or repeating the accepted late-layer
owned-input replays.

No sampling trajectory is reproduced: skipping the prefix does not advance
its RNG stream. Native speech remains rejected; no runtime/default fix,
Q4, GPU, speed or release acceptance follows. Full original self-Q provenance
remains a separate follow-up. Do not repeat these37 unchanged arms.

[Complete public raw proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/049d93d4d97ce7a542a030b4a20366df8b372699/dia-20261015-suffix-replay/complete-hosted-suffix-replay-v1/raw.zip):328,470,169bytes, SHA256
`162e4e25317e60b5b9ee703feb51aa7112c8e73e0ba89a49c55da493c56aeb61`. All15 published files were fully
downloaded anonymously and SHA/size-verified. They include official metadata,
raw captures, lossless recipes, actual sources/ELF/libs, complete logs, original
sampler and independent audit. [Structured results](dia-suffix-replay-results-2026-10-15.json).
The [integrated method](../../tools/ci-heavy/dia-suffix-replay/README.md) documents
the replay and independent reconstruction procedure.
