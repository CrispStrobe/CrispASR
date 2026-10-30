# Dia complete layer17: owned-input replay

The complete actual original DecoderLayer at step978 is replayed against the
literal native layer graph. Only layer17 coefficients are fetched:13 original
F32 tensors,293,625,856 bytes; all14 native mappings match the physical full-model
export audit. The full checkpoint is not loaded or locally rehashed. Original
classes/config/source pins and the previously audited CPU libraries are verified.

Each producer's captured layer16 output, real978-position self-cache prefix and
captured encoder output are retained. Original and native cross-cache producers
are reconstructed and compared on identical encoder inputs. The39 real text
positions use the original shared mask convention. Padded cross-cache tails
are not reconstructed; earlier padding/trimmed-cache controls remain separate.

Both owners are compared at all16 layer boundaries, including self attention,
residuals, cross query/attention, fused-original/split-native MLP and final output.
There is no large same-input defect. Native final output relative L2 is6.590e-7
against the original module on identical native inputs and caches; its own
hosted final output is reproduced byte-for-byte. One/four threads agree exactly.
All39 metric rows and32 scale-sensitive negatives are independently recalculated.
Local original versus hosted original is recorded separately from operator error.

Eight original-module interventions vary three inherited inputs independently:

| Native input substituted into original context | Final output relative L2 |
|---|---:|
| Hidden input only | 1.760e-6 |
| Self-cache prefix only | 6.210e-7 |
| Cross cache only | 1.150e-7 |
| All three | 1.985e-6 |

The hidden-input intervention has the largest individual effect in this block.
These nonlinear effects are not additive percentages or proof of the global
cause. All eight outputs and independent repeats are retained; baseline contexts
reproduce the earlier original-module outputs exactly. Real wrong-cache controls
reverse self heads and swap cross batches; they fail with relative L2 26.36% and
3.54%. Their complete input payloads are retained and independently checked for
the intended permutations. No source forward arithmetic is changed.

[Complete127-payload proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/2da60e115a9c555cfc633b0676a6ee8cce014879/dia-20261014-decoder-boundaries/complete-owned-layer17.zip) is
public and anonymously verified: 592,159,424 bytes, SHA256
`331bc7ba81aec51a684aedba9e5fdf1adf360a883c9aa04a3df9cb2ad516a127`. It includes all original/mapped coefficients, source expressions,
owned inputs/caches, outputs, repeats, wrong-cache fixtures, replay executable,
verified CPU libraries, methods and notices. Two explicit ledgers cover the
original payloads and added CrispASR notice; the earlier immutable publication
is preserved. [Structured result](dia-layer17-owned-input-results-2026-10-14.json).

Native speech still fails ASR; this is bounded CPU diagnosis, not full-model,
Q4, GPU, performance or release acceptance. The sampling cutoff diagnosis is
in [first F32 divergence](dia-f32-first-divergence-2026-10-14.md). Next trace the
inherited hidden drift earlier in the decoder without changing the original
seed, filter, reported phrase or quality gate.
