# Dia block-zero owned output projections and cross-attention

Bounded replay at step 978 uses pinned original DenseGeneral, the actual SDPA
call/transpose/output-projection expressions, KVCache, text preparation and
mask construction. The original conditional text produces a 1,024-token mask
shared by both CFG branches: exactly the 39 retained prefix tokens are valid.
An unconditional all-zero token row must not supply its own padding mask.

Both complete original output tensors (33,554,432 bytes) were fetched with
strict immutable HTTP ranges. Both mapped transposes match the physical F32
export audit. Original source is 4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8;
native expressions are extracted from 8d0ad8862d2cad5d305a48a57f2a99b5024bc39a.
Byte-verified c36dab89 CPU libraries and their loaded closure are checked.

Matching owned original/native post-rotary Q/K/V and pre-output tensors give:

| Stage | Minimum row cosine | Maximum row norm error | Maximum relative L2 |
| --- | ---: | ---: | ---: |
| sa_output | 0.999999999999928 | 1.2042e-07 | 2.5794e-07 |
| cross_attention | 0.999999999999874 | 3.1212e-07 | 1.4700e-07 |
| cross_output | 0.999999999999970 | 1.7810e-07 | 1.4798e-07 |
| cross_projection | 0.999999999999968 | 1.6568e-07 | 1.4305e-07 |

`cross_output` includes SDPA plus projection; `cross_projection` isolates the
projection on independently retained pre-output inputs. All four native outputs
on native inputs match their hosted snapshots byte-for-byte. Repeats and
one/four-thread controls are exact. Original repeats are exact. Twenty-four
metric rows separate input drift and local-original/hosted-original differences
from same-input native operator differences. Twelve negative controls reject
eight doubled outputs and four omitted masks.

The original full padded caches were not retained. Finite synthetic nonzero
tails exercise padding invariance only: masked full-length versus trimmed
relative L2 is at most 8.752e-8; omitting the mask fails. This does not claim
reconstruction of actual full padded caches. The replay also does not recompute
upstream Q projection/RoPE or validate other layers, full speech, Q4, GPU or
performance. Native free-running speech remains rejected; no numerical change
or release follows. Next test unmarked versus debug-marked free-running
CLI/C-ABI execution.

The initial AST probe selected the method prologue rather than the mask
assignments and stopped with a missing local-variable error. It was corrected
by selecting the actual three named position/mask assignments; the initial log
is retained. No original operator was altered.

[Complete 40-payload proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/c0fec68e9c03f03f5fad13166aed13eea53070dc/dia-20261014-decoder-boundaries/complete-owned-cross.zip)
is public and fully anonymously SHA256-verified (69,532,658 bytes, SHA256
`b1dd08c3e66f268684ab6bd93a78ba89203d5f08c102f9b62ae91acdf73de49d`).
The archive includes original/mapped coefficients, owned arrays, extracted
expressions, bridge, verified libraries, full receipts and controls. Scripts
use maintainer paths; parent captures/source pins are publicly recorded in
`dia-decoder-boundary-results-2026-10-14.md`.
