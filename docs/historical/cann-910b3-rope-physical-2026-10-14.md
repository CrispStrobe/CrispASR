# Physical 910B3 rotary-capacity candidate validation — 2026-10-14

The candidate resolves both reported F32 NEOX failures on the contributor’s
910B3/ARM64, SDK9.1.0 setup. The complete five-operator suite passes
**1,691/1,691 executed cases**, retaining **1,474 unsupported cases**.
A separate fresh-process complete RoPE repeat passes **452/452**, retaining
18 unsupported cases. These counts exclude the repeat from the broad total.

| Operator | Executed pass | Unsupported |
|---|---:|---:|
| ADD | 154/154 | 0 |
| RMS_NORM | 51/51 | 0 |
| SOFT_MAX | 107/107 | 107 |
| ROPE | 452/452 | 18 |
| MUL_MAT | 927/927 | 1,349 |

The [contributor attachment](https://github.com/CrispStrobe/CrispASR/pull/544#issuecomment-6164970290)
SHA-256 is `b200279b56ab5289fb7ceefe8574c03e14c576b6f7e70258df7e23082a0e83e6`.
All 20 regular archive members were safely extracted and rehashed. The
unchanged wrapper independently reproduces all six operator assessments from
complete logs. Both exact previously failing shapes remain present and pass
in both runs: `[256,8,512,1]`, n_dims256, ff0; and `[512,8,512,1]`, n_dims128,
ff1; both F32, NEOX mode2, v0, inplace0. No parameter filter or tolerance
change was used. Original failed evidence remains retained.

Native Git receipts identify clean detached CrispASR
`ca4c12c4cfec475c0009f9718a18ffb42d2cda6a` and ggml
`0a862076567afd8468822ccc512c97ac6a69b340`. Actual CANN compiler commands
contain ASCEND_910B and USE_ACL_GRAPH; CMake enables Release/CANN/Ascend910B
and ACL graphs. The complete build log links the actual operator executable.
Four capacity controls and the existing chip-routing test pass on the device.
The binary SHA in the producer receipt was not locally rehashed: the archive
contains build receipts and logs, rather than the executable itself.

[Complete independent receipt](cann-910b3-rope-physical-2026-10-14.json) retains
all member/log hashes, commands, assessments and exact repaired case lines.
Local raw archive and audit methods are under
`${CRISPASR_STORAGE}/crispasr/cann-910b3-rope-candidate-20261014`.

The [SDK9.2 two-architecture builds](https://github.com/CrispStrobe/CrispASR/actions/runs/38536737288)
remain in progress at this checkpoint. Compose the tested source with current
main, retain the newer shared-fork history with an identical candidate tree,
and require SDK and composed general CI gates before updating production.
This is operator repair evidence. Full NPU model inference, quantized speech,
performance, and actual execution of captured ACL graphs remain separate gates.
