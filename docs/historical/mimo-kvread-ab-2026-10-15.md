# MiMo F32 cache-read hypothesis rejected — 2026-10-15

The existing `CRISPASR_KV_READ_F32=1` option does not fix Q4 flash/eager
parity on the pinned original five-stage fixture. All four hosted processes
complete successfully; the workflow deliberately fails its unchanged numerical
acceptance gate. Preserve this rejected result rather than repeating the same
switch, changing the gate or promoting a default.

## Four-arm result

All arms use the same immutable Q4 model, original reference, executable and
CPU configuration. Each arm runs in a fresh process to isolate cached env flags.
Stored KV remains F16 (576 MiB); the option changes its read representation.

| Eager vs matching flash | F16 reads | F32 reads | Required |
| --- | ---: | ---: | ---: |
| Hidden relative L2 | 0.0604836 | 0.0572553 | ≤0.005 |
| Hidden cosine | 0.9981740 | 0.9983611 | ≥0.9999 |
| Logit relative L2 | 0.0401054 | 0.0513234 | ≤0.005 |
| Logit cosine | 0.9991978 | 0.9989766 | ≥0.9999 |
| Logit norm error | 0.0013787 | 0.0252777 | ≤0.005 |

All three input stages are byte-identical across the four arms. Flash outputs
are byte-identical with the option OFF/ON at all five stages. Eager outputs do
change: the override has a real effect, but the result remains rejected.
Original vs Q4 absolute gates also fail, including the pre-existing default
arm: its hidden/logit relative L2 is0.273699/0.227217 against original Python.
This experiment demonstrates neither a new regression nor original Q4 stage
acceptance. F16 cache storage, Q4 weights and F32 cache reads are separate
precision choices; this result must not be described as an original-weight
F32 model test.

## Independent audit

The raw11891130-byte GitHub artifact matches its API SHA256 digest:
`b6f8af43667aa32c60af5219235274b44781a67e3e21c2e60f109ef576219548`.
CRC, unique/safe members, all reference input IDs and the retained original
reference SHA verify. A separate implementation recalculates20 original-stage
comparisons and20 attention/read-effect comparisons, including norms and errors.
Twenty scale×2 negatives reject; the worker's scale/layout/shape/nonfinite/
zero-norm controls pass. All four actual process IDs differ. CPU-only compiler
configuration and live CPU/F16-cache reports are retained. Full model SHA is
the producer's checked-download receipt; the whole model is not delivered here
or independently rehashed locally. The retained reference and actual binaries
are independently rehashed.

Source `f242ba80eccbc801a44bebe134486667cf6b127d` adds only the diagnostic
worker and its documentation to runtime baseline `ecd7de6e`. Retained worker
bytes match that immutable Git object. Actual tested wrapper ELF SHA256:
`d89161918553c0ae316df2126c3b3361dba9ea1f18f6501e1eed7d78b45d1358`.
The full dynamic libraries, loader report, compiler commands and actual arrays
are retained; an executable hash alone is not treated as source provenance.
An initial local audit wrongly assumed every disabled CMake entry was BOOL;
its failed log and corrected audit survive without numerical gate changes.

[Full59-payload proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/4f18b5bea0c5891aeb0ea77ee377803109bc2f51/mimo-20261015-kvread/complete-rejected-cache-read-v1.zip),24,103,492 bytes, SHA256
`08e36b54c194a42277ea5f833cd6296912a6e81a73fdd3cf3ffc81f90c7e9e9a`, is public and fully anonymously downloaded/rehashed. It includes the
original raw GitHub ZIP, every returned payload, full job log, both audit
versions, source and notices. [Machine receipt](mimo-kvread-ab-2026-10-15.json).
[Rejected hosted run](https://github.com/CrispStrobe/CrispASR/actions/runs/38703457282).

## Remaining work

The preceding [Q4 projection diagnosis](mimo-q4-projection-2026-10-15.md)
confirms amplification by dynamic Q8 activation rounding. This cache-read
experiment does not remove it. Select a specific attention/activation arithmetic
hypothesis on owned inputs before another run; do not repeat the old BLAS or
cache-read sweeps. Full original-layer magnitude gates and actual fixed-recipe
CLI/C-ABI speech remain required. No GPU, speed, complete PR492 or default
acceptance follows from these CPU diagnostic timings.
