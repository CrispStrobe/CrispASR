# Dia full-decoder query/key factorial: complete negative result

Hosted [run38764924112](https://github.com/CrispStrobe/CrispASR/actions/runs/38764924112) passes at scientific source
`ff7e07c0f5ec846ff7de569c117c1d0b30af404d`. All10
[query-producer lint checks](https://github.com/CrispStrobe/CrispASR/actions/runs/38767820341) pass at b1de5628;
all10 [standalone attention-tool lint checks](https://github.com/CrispStrobe/CrispASR/actions/runs/38771099343) pass at95ade290.
Tool integrations `f38980ce07b4a98b6275d25781f047d75329b090` and `637800051f7a2498d9830828f2dc435a372465f5` preserve tested code bytes after rebase;
only README merge context is reconciled. Production inference, model pins and defaults do not change.

## Independent controls and original provenance

The official ZIP SHA256/size/CRC/safe-path checks and all739 payload hashes pass.
Methods match immutable git source; actual ELF/libraries and input manifests verify.
All979 forced input/logit rows and all captured stages are byte-exact in baseline
and native Q/K substitution controls. Every requested Q/K is consumed exactly;
earlier978 logits, layer0 input/norm/V and cross-cache inputs remain unchanged.
All18 stage metrics, hidden/logit norms/errors and actual original Torch2.6
sampler ranks/supports are independently recomputed. Baseline, native control and
original-key-only outputs also reproduce the previous hosted key experiment exactly.

Original Q is actual pinned original-module replay, qualified by exact standalone
controls and unchanged2e-6 full-reference reconstruction gates. It is not a
previously captured full-model self-Q; full-reference attention/output are not
byte-exact. See [query provenance](dia-query-key-preparation-2026-10-15.md).
This conditional experiment cannot rule out every possible full-model original
Q/K discrepancy or establish a speech repair.

## Final margin and support

Original token46-minus904 CFG margin is **+4.6253204345703125e-5**, ranks32/33.
Every tested native arm retains ranks33/32 and exactly the native32-token support:
46 is excluded and904 included.

| Arm | CFG46-minus904 | Hidden relative L2 | Logit relative L2 |
| --- | ---: | ---: | ---: |
| baseline | -2.67028808594e-05 | 2.28021266e-06 | 1.4132039e-06 |
| native-control | -2.67028808594e-05 | 2.28021266e-06 | 1.4132039e-06 |
| original-query | -1.43051147461e-05 | 2.41936214e-06 | 1.49969627e-06 |
| original-keys | -2.14576721191e-05 | 2.25007358e-06 | 1.36786545e-06 |
| original-query-and-keys | -2.67028808594e-05 | 2.27593765e-06 | 1.41108871e-06 |

Joint replacement moves both disputed guided logits down by the same
9.5367431640625e-7, leaving their margin exactly equal to baseline. Individual
query/key margin changes do not add to the joint change. No sampling filter,
seed, weights or generation recipe is modified. The observer retains original
multinomial input probabilities; this experiment does not generate new free speech.

## Early attention improves, final ordering does not

An additional independently checked25-row comparison uses actual block0 arrays
from the canonical full-model original GGUF. Cosine and row-norm comparisons
use the last tensor dimension (128 per attention head, 2048 for block output).
It preserves reference/native arrays
and their source hashes; this comparison is separate from same-owned-input operator replay.

| Arm | SA relative L2 | SA minimum cosine | Block0 output relative L2 |
| --- | ---: | ---: | ---: |
| baseline | 4.71122677e-06 | 0.999999999943487 | 1.43609871e-06 |
| native-control | 4.71122677e-06 | 0.999999999943487 | 1.43609871e-06 |
| original-query | 2.01974759e-05 | 0.999999997439016 | 3.80797669e-06 |
| original-keys | 2.18493657e-05 | 0.999999997171082 | 4.12286067e-06 |
| original-query-and-keys | 1.4222975e-06 | 0.999999999976956 | 1.21106571e-06 |

Both native Q/K share rotary behavior, so replacing only one can worsen
attention agreement while replacing both improves it. The measured joint
correction still does not restore the final disputed margin. This is evidence
against the tested layer0 substitution being sufficient, not additive causal
percentages or proof that all later operators are correct.

## Complete public proof

The [official raw artifact](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/6fa7de84ebc1700c841cf155827ba326b8e889e1/dia-20261015-query-key/complete-hosted-query-key-v1/raw.zip) is 1177798269 bytes, SHA256
`4e444752df59e76f965822e826282e07dbfac893215bc12e223facae4ba520de`.
The [immutable manifest](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/6fa7de84ebc1700c841cf155827ba326b8e889e1/dia-20261015-query-key/complete-hosted-query-key-v1/manifest.json) pins all22 public files,
including terminal logs/APIs, independent audit/results, compact query provenance
and notices. Every anonymous full download matches; raw archive retains all739
payloads, actual executable/libraries, patched runtime, all arms and original sampler.

The [complete block0 stage proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/82b9605fd7b8836dd738f9bc7e97def3cf47fa72/dia-20261015-query-key/block0-factorial-stage-analysis-v1.zip) contains38 payloads,
299688 bytes, SHA256 `98cbd36684db6ace8f548cf90e91d3cc53ad827e9906e03bcb82d6f534c85681`.
Its31 input/source payloads and25 rows are independently checked; the full
anonymous ZIP hash and CRC pass. It retains actual original stage arrays, all
native arm arrays, methods, provenance and notices.

[Structured result](dia-query-key-results-2026-10-15.json) preserves every
18-layer metric, sampler support, exact-source CI result and publication ledger.

## Next useful work

Do not repeat unchanged key-only/joint-QK or all-layer source-RoPE speech runs.
Trace the final margin through a controlled one-step suffix replay using actual
captured original layer outputs, fixed native prefixes/cross caches, and exact
native activation-substitution controls. Prefer reusing the shared-prefix state
over recomputing979 steps for every arm. Also close the missing simultaneous
full-model original self-Q capture with unchanged-blueprint hook transparency.
Component improvements must still pass fixed-recipe CLI/C-ABI speech,
predetermined additional prompts/speakers/lengths/reuse, original-stage diff,
DAC/ASR and actual CUDA before Q4/performance/default promotion.
