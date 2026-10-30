# CANN RoPE cache: failing pre-fix capacity regression

The literal sin/cos allocation block and actual cache metadata class reproduce
a requested-capacity defect before any repair. They match the physically
tested ggml21acf3162 sources (Git blobs 902d2eda693837c7ea636840facb6725ab0f7746
and 1c6e685c38cac5de85bf70914a95ce3afc90488b).

The standalone CPU probe instruments malloc/free requests, calls the actual
metadata setter after each step, and checks `allocated bytes >= required
bytes` for both buffers before any sin/cos write. It performs no rotary
arithmetic or ACL device operation. Actual NPU allocator rounding and device
results are not simulated.

| Reported width sequence | Positions | Required bytes per buffer | Requested allocation |
| ---: | ---: | ---: | ---: |
| 64 | 512 | 131,072 | 131,072 |
| 64 | 512 | 131,072 | 131,072 |
| 256 | 512 | 524,288 | **131,072: insufficient** |
| 128 | 512 | 262,144 | **131,072: insufficient** |

A separate width-64/512-position to width-256/256-position transition
also leaves 131,072 bytes for a 262,144-byte request. Ordinary position
growth and changed shapes with the same byte requirement pass. Thus the
original suite runs four tests with two expected failures and two successful
controls. The raw nonzero-exit unittest log is retained. It is not described
as a passing test run.

[Test source](https://github.com/CrispStrobe/CrispASR/blob/7b77e0b0a29f2e754a64a5eae06c6911cda371c2/tests/test_cann_rope_capacity.py)
lives in the isolated diagnostic branch. It must not land on the current
main pin as a falsely green guard. All four secret tests pass. No candidate
was applied; the production runtime/pin remains unchanged.

[Complete 14-payload pre-fix proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/7e683151a7eab8d6b115fb305ff3d7a0fe0dd0a9/cann-20261014-rope-capacity/complete-pre-fix-capacity.zip)
is public and fully anonymously verified: 69,043 bytes, SHA256
`372b92c65abf9bcc854b0af2e07cc8c9d207ac83a9903f3fdec97714d4f792f8`.
It retains both original source files, extracted probe, executable, unit
source, all per-case rows, failure/method/source receipts and ggml license.
[Structured result](cann-rope-capacity-2026-10-14.json).

This proves the cache-capacity defect, not that a proposed repair resolves
the two physical 910B3 numerical failures. Next maintain independent
allocated-byte capacity instead of using last-call position metadata as
capacity, rerun the same regression, compile both SDK architectures and
request a complete physical operator rerun with upstream tolerances intact.
ACL-graph capture compatibility and the earlier 310P repair must be preserved.
Full-model diff, quantized speech and performance remain separate. Per the
requested order, runtime repair follows the active Dia transparency gate.
