# Dia numerical cause: serial F32 versus fused CPU reduction

The reference's output projection is reproduced exactly by a serial F32
multiply-then-add recurrence. The native CPU head uses four eight-lane FMA
accumulators and a tree reduction. On the same captured native hidden state,
this arithmetic choice reverses the disputed nucleus-cutoff ordering and
changes the unchanged original sampler's draw904→877. No token is substituted.

## Controlled arithmetic result

The actual pinned original DenseGeneral runs with identical inputs, weights,
shapes, strides and four threads in separate processes. Changing only MKL CNR
mode gives:

| Projection path | Original-hidden CFG46−904 | Native-hidden CFG46−904 | Native-hidden draw |
|---|---:|---:|---:|
| Original MKL COMPATIBLE | +4.625320434570e-5 | +2.050399780273e-5 | 877 |
| Original MKL SSE4_2 | +4.625320434570e-5 | +2.050399780273e-5 | 877 |
| Original MKL AUTO/AVX512 | +2.479553222656e-5 | -3.385543823242e-5 | 904 |
| Original MKL AVX2 | +1.335144042969e-5 | -3.480911254883e-5 | 904 |
| Native captured head | see terminal intervention | -2.670288085938e-5 | 904 |
| High-precision product sum | +1.906230906279e-5 | -2.748825236409e-5 | not used for generation |

COMPATIBLE and SSE4_2 reproduce all4,096 actual full-model pre-RoPE query
values and all18,504 actual full-model original-head logits **byte-for-byte**.
AUTO/AVX512 reproduce the earlier isolated-query discrepancy instead:
relative L2 6.906207657e-7, max absolute4.053115845e-6. Six environment requests
were tested; the AVX request actually reports CNR:SSE4_2, so it is explicitly an
alias in this run, not evidence of a distinct AVX kernel. Actual MKL verbose
SGEMM calls, instruction branches, thread counts and exact repeats are retained.

The earlier conclusion that a changed head cannot recover this decision was
too broad: its local Python replay used AUTO/AVX512, which differs from the
actual full-model reference. The measured old results remain valid for those
local kernels; they did not test the matching serial/COMPATIBLE recurrence.
This also resolves the [full-query/module replay mismatch](dia-query-capture-reconciliation-2026-10-16.md).

## Exact arithmetic mechanism

For each output, the reference-matching recurrence is:

```text
s = float32(0)
for k in input order:
    product = float32(weight[k] * hidden[k])
    s = float32(s + product)
```

This matches every head logit on **both** captured hidden states. A separate
auditor using rounded products followed by NumPy cumulative F32 sums reproduces
all outputs independently; it also reconstructs the full captured query.

The actual retained c36 ggml CPU library instead executes AVX2 FMA instructions
into4×8 accumulators, followed by `(sum0+sum2)+(sum1+sum3)` and horizontal adds.
An explicit scalar `fmaf` reconstruction reproduces all eight disputed dots
exactly. On native hidden, these also match the actual hosted native logits,
not merely a standalone projection. The n=1 decoder matrix shape rejects the
llamafile SGEMM path and reaches this vector dot implementation.

Fusion alone has a tiny exact witness: with `x=1+2^-23`, `y=1-2^-23`, `s=-1`,
separate F32 multiply/add returns0, while `fmaf(x,y,s)` returns−2^-46. Multiple
accumulators additionally change addition order. Thus the two kernels compute
the same real-valued dot product with different intermediate rounding.

CFG magnifies contrast differences: `g46-g904 = 4*(c46-c904)-3*(u46-u904)`.
The native margin−2.670288085938e-5 becomes+2.050399780273e-5 with the serial
head on unchanged native hidden. The original filter and verified original
exponential noise then choose877. No seed, noise, top_k/top_p, threshold,
reference rescaling or forced token changes.

Increasing precision is a different experiment: exact F32 products summed
with `math.fsum` give a **negative** native-hidden margin. The less accurate
serial reference rounding happens to preserve the desired support here.
Do not describe this as a generic accuracy improvement from FP64 or claim
all native matrix kernels are mathematically wrong.

## What follows and what does not

The [reciprocal full-generation diagnostic](dia-native-fork-results-2026-10-16.md)
already proves that selecting877 at this boundary restores the original input/
code trajectory and WER0. This new evidence supplies an actual arithmetic path
that selects877 without injecting that token at the fixed boundary. It is a
pointwise sufficient numerical cause, not yet an all-step runtime speech repair.
Earlier propagated hidden differences still exist; this does not assign all
upstream error to the head or establish one unique bad layer.

Next test a generic opt-in serial-F32 head path at **every** generation step,
using only loaded native weights/current hidden, no reference tensors or
step/token-specific branch. Require pristine/OFF/identity controls, original
sampler/RNG and full native/original DAC speech checks. Keep it diagnostic until
current-main integration, additional prompts/seeds and GPU/Q4 behavior pass;
profile only afterward. A graph CPU fallback and its performance cost must be
explicit. No production default or acceptance threshold has changed.

## Evidence

Method71d5a6ec runs only small audited coefficients and captured arrays on the
VPS; no full neural model is loaded. Independent saved-array, nine physical
head-mapping, exact cumulative-sum, tiny fusion and actual-library/hosted-dot
checks pass. The first order-audit attempt exposed an unrelated system
libggml-base loader collision; the corrected probe explicitly preloads and
hash-verifies its pinned base/CPU pair. Both failed logs remain retained.
A failed independent audit assumed AVX request meant CNR:AVX; actual verbose
output identifies its SSE4_2 alias, now recorded explicitly without a waiver.

Complete public proof has79 payloads plus its manifest,170,121,121 bytes, SHA256
`b092ed027e26480ae4ad6c59d30d216662c7c5afe949f5300c9afafe86163397`.
[Immutable archive](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/81d18136133e7bad61961813f45afa58301fe32e/dia-20261016-reduction-cause/complete-reduction-cause-v1.zip)
passes a full anonymous download and SHA/size verification. Sources, original
provenance, raw arrays, actual libraries, dispatch logs, independent methods,
failed logs and licenses are retained. [Structured results](dia-reduction-cause-2026-10-16.json).
Fresh hosted replay of the matching actual original projection is being prepared;
local component proof is distinct from full model/GPU/Q4/performance acceptance.

MKL branch selection is documented in [Intel's CNR guide](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-1/get-started-with-conditional-num-reproducibility.html).
