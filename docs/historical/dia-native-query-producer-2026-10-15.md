# Dia native self-query projection and RoPE isolation

**Completed and integrated:** tested diagnostic tools are integrated at 63780005. Hosted and applicable lint checks pass. The [completed Q/K result](dia-query-key-results-2026-10-15.md) remains negative for restoring the disputed ordering. Earlier preparation checkpoints below are historical; they are not current pending jobs.

The one-step probe reproduces the captured native layer0 self-query at step978
byte-for-byte. Projection relative error is 1.48e-7; applying native RoPE to the
original pre-RoPE query still gives 5.37e-5 error. This localizes the query
arithmetic discrepancy to RoPE at this boundary. Its effect on full-decoder
sampling and speech remains unproven.

## Source and owned-input controls

Candidate `b1de56284867dfcd45bc90289aadbb34a189f4e2` adds the standalone
producer and runner. The literal rotary helper is checked against runtime
`8d0ad8862d2cad5d305a48a57f2a99b5024bc39a`; headers and actual previously
hosted libraries are pinned to ggml `c36dab89b662838f0f5d4826c399198c0b90bbfc`.
Only the physically audited 16 MiB query matrix and small owned arrays are
loaded. The native normalized input equals the original input exactly, and
transposed Q weights match their physical GGUF mapping SHA256
`17d07c13004926d7c7f99c1df2ee57e33b584ae01105dfef77ec08462e7e7575`.

Native projection followed by native RoPE reproduces the captured hosted query
exactly. Supplying its owned native pre-RoPE result also reproduces that query.
Both modes repeat exactly with one and four threads. Actual scale and head-order
controls reject: doubling pre-Q preserves cosine but doubles magnitude, with
relative L2 and row norm error both 1. Reversing heads has relative L2 1.45084.

## Measured boundaries

Each reference is the corresponding actual original-module replay array,
qualified in the [original-query receipt](dia-query-key-preparation-2026-10-15.md).

| Boundary | Relative L2 | Minimum cosine | Native norm | Reference norm | Maximum row norm error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q projection | 1.480960317e-7 | 0.999999999999983 | 31.515238452 | 31.515237853 | 6.988511236e-8 |
| Native projection and RoPE | 5.370203745e-5 | 0.999999998125570 | 31.515238325 | 31.515237859 | 6.715343792e-8 |
| Original pre-Q through native RoPE | 5.371474382e-5 | 0.999999998125268 | 31.515237747 | 31.515237859 | 2.783383157e-8 |

These errors are conditional comparisons, not additive causal percentages.
The original Q comes from actual module replay; no earlier full-model self-Q
capture exists. No runtime repair or full speech acceptance follows from this
operator probe.

## Independent audit and complete public proof

The independent auditor rehashes all87 producer payloads, verifies actual
ELF/library hashes and physical weight mapping, recomputes all metrics from
raw arrays, checks every repeat/control, and freshly executes both retained
ELF modes from an unrelated directory with loader overrides removed. Both
fresh outputs reproduce the retained arrays exactly. C++ format18 and local
shell/document checks pass. The producer is actually compiled and run; a full
production CMake build or cross-platform validation is not claimed.

The [complete public proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/2a9a269913d6e4a2fa5df06b29fb14cec45d7db0/dia-20261015-query-key/complete-native-query-producer-v1.zip) contains98 payloads, including all raw
inputs/outputs, source/headers, binary/libraries, commands/logs, receipts,
independent auditor and notices. It is 32,621,191 bytes, SHA256
`3940c5510551ce3bc708093280da8bab994a99c71bc38a27e8267201154d0adc`,
at immutable revision `2a9a269913d6e4a2fa5df06b29fb14cec45d7db0`.
A fresh complete anonymous download matches that hash and every ZIP CRC.
Extracting it into a fresh directory and running the retained auditor passes
all87 payload checks and both fresh binary modes. Unpack into a fresh directory
and run its retained `audit_native_query.py` with NumPy to repeat that audit.
The [machine receipt](dia-native-query-producer-2026-10-15.json) retains all
producer metrics, payload hashes and publication metadata.

## Remaining acceptance

[Full Q/K factorial38764924112](https://github.com/CrispStrobe/CrispASR/actions/runs/38764924112)
is running at source `ff7e07c0f5ec846ff7de569c117c1d0b30af404d`.
The new commit leaves that worker, original-query producer, patch, graph guard
and selftest byte-for-byte unchanged. Only a standalone target and its docs
are added to their CMake/method directory.
[Latest lint38767820341](https://github.com/CrispStrobe/CrispASR/actions/runs/38767820341)
is queued at the new candidate. Independently audit full hosted raw artifacts
and exact native controls before interpretation or integration. These tools
are integrated; speech, CUDA, Q4 and speed acceptance remain separate.
