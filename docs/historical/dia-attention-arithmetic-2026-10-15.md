# Dia tiny tensors and exact-input attention arithmetic

**Completed and integrated:** tested diagnostic tools are integrated at f38980ce. Hosted and applicable lint checks pass. The [completed Q/K result](dia-query-key-results-2026-10-15.md) remains negative for restoring the disputed ordering. Earlier preparation checkpoints below are historical; they are not current pending jobs.

The component probe is complete. Tiny analytic fixtures pass the intended head,
batch and scale laws, and four cancellation terms expose a reduction-order
difference. Actual original Q/K/V reproduce retained original-module attention
exactly; native Q/K/V through the bridge reproduce hosted native attention
byte-for-byte. No full model or new speech generation is run on the VPS.

Tested source is `e6be594f27788d8bfcbab2b60682f64e7ea977fc`. Rebased standalone
candidate `95ade29094e81c8a9c7437fd0caafedca76e3e1e` preserves all tested
scripts/C++ bytes; only README context changed to avoid importing the separate
unvalidated Q/K factorial. These diagnostic tools are now integrated after all10 lint checks pass.
Runtime expressions are pinned to `8d0ad8862d2cad5d305a48a57f2a99b5024bc39a`,
actual ggml headers/libraries to `c36dab89b662838f0f5d4826c399198c0b90bbfc`,
and original Python classes to `4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8`.

## Tiny mathematical controls

The uniform fixture has batch2, query-heads4, KV-heads2, four positions and
head dimension8. Zero queries give scores0 and probabilities exactly1/4;
values uniquely identify batch, KV head and component. Analytic expected
weighted values match native and original SDPA exactly. Tiled heads and swapped
batches disagree. A nonuniform fixture with scores[-2,-1,1,2] matches a float64
scale=1 softmax calculation. Actual incorrect sqrt scaling, reversed KV heads
and doubled values give relative errors0.0138654,0.123118 and1; all reject.
The doubled-value control retains cosine1 while doubling its norm.

Four scalar values are sufficient to expose numerical cancellation:

    probabilities = [1/4, 1/4, 1/4, 1/4]
    values        = [2^24, 1, -2^24, 1]
    exact sum     = 0.5
    actual Python = 0.5
    actual ggml   = 0.25

The retained c36 CPU dot implementation uses float accumulators and a float
leftover loop. For this small reduction, adding0.25 to4194304 rounds back to
4194304; cancellation then leaves only the final0.25. Independent explicit
float32 accumulation reproduces this result. This is a numerical diagnostic
against an exact answer, not a relaxed tolerance or evidence of a head-layout
bug. Different kernel shapes and reduction orders need not round identically.

## Original tensors fed through ggml

The graph follows the literal native sequence: consecutive GQA expansion,
QK dot products, scale=1 softmax, then probability-weighted values. Inputs are
copied before execution, consumed replacement tensors are checked exactly,
and all intermediates are retained. Original noncompact3072-slot cache strides
are preserved for the actual original `Attention.forward` SDPA call.

For every retained case, an explicit Python matmul/softmax/value shadow matches
actual original SDPA byte-for-byte. It is still a shadow, not a captured internal
fused-kernel probability tensor, and this equality is specific to these fixtures.
The original-module replay uses its reconstructed current K/V plus retained
prefix; a separate arm preserves the earlier full-model captured complete K/V.
Neither is described as a previously captured full-model self-query.

On original-module replay inputs:

| Comparison | Relative L2 | Maximum absolute error |
| --- | ---: | ---: |
| Native QK scores versus Python shadow | 7.617002734e-8 | 1.525878906e-05 |
| Complete native attention versus actual original SDPA | 4.802293154e-7 | 5.811452866e-07 |
| Original shadow scores through native softmax/value multiplication | 4.744261217e-7 | 5.811452866e-07 |
| Original shadow probabilities through native value multiplication | 4.673265751e-7 | 5.513429642e-07 |
| Native fixed-probability value multiplication versus float64 | 8.198455963e-8 | 6.030920385e-8 |
| Original fixed-probability value multiplication versus float64 | 4.893739597e-7 | 5.942231036e-7 |

Injecting original probabilities isolates a remaining reduction difference.
Here native output is closer to the float64 calculation than the original;
matching Python is not equivalent to increasing arithmetic accuracy. These
nonlinear/error comparisons are not additive causal percentages.

## A real disagreement reduced to one dot

The largest fixed-probability difference is batch0, query-head3, component124,
using KV-head0. Retaining only979 scalar probabilities and values, one native
weighted-value output reproduces that full native attention component exactly:

| Execution | Scalar result |
| --- | ---: |
| Original full matrix / SDPA shadow | -0.19854898750782013 |
| Full native attention and isolated native dot | -0.19854843616485596 |
| Isolated Torch dot | -0.19854839146137238 |
| Isolated Torch matrix multiplication | -0.19854877889156342 |
| Float64 products summed with `math.fsum` | -0.19854839328471657 |

Native absolute error against the high-precision sum is4.2880e-8, versus
5.9422e-7 for the original full matrix. The Torch scalar/matrix results are
reported separately: changing the shape changes its reduction dispatch, so
neither isolated Torch result replaces the actual original full-matrix result.

## Exact rank-margin explanation

The four disputed incoming logits are retained and independently checked.
CFG is `g = c + 3*(c-u)`, so the token46-minus904 margin is
`4*(c46-c904)-3*(u46-u904)`.

| Contribution | Value |
| --- | ---: |
| Original margin | +4.6253204345703125e-5 |
| Conditional-logit error contribution | -5.7220458984375e-5 |
| Unconditional-logit error contribution | -1.5735626220703125e-5 |
| Native margin | -2.6702880859375e-5 |

The equation closes exactly. Float32 and float64 CFG produce the same margin;
rounding inside CFG does not cause this particular swap. The original ranks
are32/33 and native33/32. Earlier original filtering/noise replay establishes
that this swap changes nucleus support and the resulting sampled token.
This explains incoming-logit sensitivity, not an attribution of the entire
upstream logit error to one attention operator or proof of speech repair.
The earlier [head-isolation proof](dia-head-projection-isolation-2026-10-13.json)
is reused rather than rerunning an unchanged head experiment; it already
separates inherited hidden direction from an isolated output-head repair.

## Independent proof and remaining gates

The independent auditor checks565 raw payloads, recomputes99 metric rows,
reexecutes actual original SDPA and the explicit shadow, verifies all one/four
thread repeats and native/original exact controls, then freshly invokes all18
case/mode ELF combinations with loader overrides removed. A separate auditor
reproduces the tiny cancellation, full-component-to-one-dot equality and four
logit CFG equation. C++ format18, document/workflow checks and all four unskipped
secret tests pass. A full production build, GPU or speech pass is not claimed.

The [complete public proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/ee71525c57d8a709020bee3df6463d53d7bdb96b/dia-20261015-attention-arithmetic/complete-attention-arithmetic-v1.zip) contains816 payloads: arrays, actual ELF/libraries,
headers/source, commands/logs, initial-run receipt, fresh independent outputs,
auditors, scalar reduction, boundary logits and notices. Size62,259,676 bytes;
SHA256 `19efcd3073914fce5f090edfbd0e6a6bead3eae979b3c646d8d29412cf691532`.
A fresh anonymous download verifies the full hash, every ZIP CRC and815 manifest
payloads. Both independent audits also pass from a fresh public extraction.
On filesystems that do not preserve executable modes when unzipping, first run
`chmod +x run-v2/bridge`. Then, in the pinned CPU reference environment:

```sh
python methods/audit.py --proof run-v2 --output fresh-audit.json
python methods/audit_small.py --proof run-v2 \
  --reduction worst-reduction-cli-v2 --margin cfg-margin-v1 \
  --output fresh-small-audit.json
```

[Structured results](dia-attention-arithmetic-2026-10-15.json) retain exact
metrics, source/library hashes, audits, public delivery and CFG inputs.
Standalone tools await [lint38771099343](https://github.com/CrispStrobe/CrispASR/actions/runs/38771099343)
before integration; that exact-source run is queued. The separate
[full-decoder Q/K factorial](https://github.com/CrispStrobe/CrispASR/actions/runs/38764924112)
is running at ff7e07c0 and is not restarted. Native speech remains rejected;
this probe establishes no CUDA/Q4/speed/default acceptance.
