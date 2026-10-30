# Full-model CANN validator preparation — 2026-10-14

Candidate `168384d47ba462ae72858042ffc13188b143cca1`, branch
`diag/cann-model-validator-20261014`, adds `tools/validate_cann_model.py`.
Its full CPU interface and unavailable-CANN controls pass in hosted run38602252235;
returned171 payloads are independently audited. It is integrated by rebase/fast-
forward at `8f6aaca14f36bb2ba58e090b189f8f8a10bf2d01`.

The runner consumes the immutable accepted108,495,214-byte Moonshine fixture
archive (SHA256 `7d63d2a5b90eb77843aff25107b1d1f70bafcfc1c8f73075e2f5a9a3f87d5e84`)
and a supplied Linux build. It checks archive identity and extraction paths,
original PCM/reference shape and arrays, snapshots actual native binaries and
libraries by SHA, retains CMake cache/compilation commands and ldd output, then
runs a real explicit CPU control before requested CANN execution. Each arm
requires encoder diff, repeated automatic/explicit C ABI Sessions, repeated
encoder arrays and two CLI transcriptions. All encoder values must pass the
existing cosine/row-norm/relative-L2 gates; CANN also compares directly to the
paired CPU output. Transcripts must reproduce the original words, with raw
text and byte-equality results retained. Output directories must be fresh;
partial receipts, complete logs and snapshots survive failure.

CANN acceptance requires successful exact-device initialization and allocated
encoder matmuls on that same device, followed by successful computation. Legitimate
unsupported-operation CPU fallback remains visible. Missing/invalid devices,
zero or partial execution evidence, mismatched devices, bad output or uncompiled
CANN cannot pass. Device metadata from npu-smi is retained when available;
checkout identity is explicitly the validator identity and is not asserted to
prove that supplied ELF bytes were compiled from it. Build logs/source provenance
remain a separate requirement when auditing returned hardware results.

The original native dumper forced CPU for its public encoder call even when
Sessions used GPU. New explicit GPU and encoder-only flags let the public
stage call reach the selected NPU; defaults preserve the previous CPU behavior.
Flags requiring native mode and incomplete native argument sets now fail at
argument parsing. Eight weight-free false-success/archive/argument guards pass,
and are wired into ordinary Linux CI.

The bounded actual unchanged Moonshine backend shared probe passes both CPU
GPU-flag settings, returning complete repeated457×288 arrays byte identical to
the accepted CPU fixture. With a CANN requirement, GPU=false rejects the
GPU-disabled contradiction; GPU=true reaches actual device lookup and rejects
the unavailable CANN registry. This proves argument forwarding, not NPU work.
It does not exercise the complete shared C ABI Session/CLI runner.

[Complete30-payload native flag proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/8b16f924c00fc285001f12060b7a24945708d9b8/cann-20261014-model-validator/complete-native-gpu-flag-controls.zip) includes actual source,
shared probe, full build/library-closure logs, arrays, guard results and method
ledgers. Anonymous full download verifies 2,208,646 bytes, SHA256
`de8965041d6e8cf88659c75ff676b243f09581fe31adc698b7e03defdaa0a7ea`. [Structured preparation receipt](cann-model-validator-preparation-2026-10-14.json).

[Hosted full interface run38602252235](https://github.com/CrispStrobe/CrispASR/actions/runs/38602252235)
exercises the exact command through genuine CPU diff/CLI/auto+explicit
Sessions, require CANN rejection after CPU controls, and independently exercise
GPU=true public encoder initialization on the CPU host.
[General CI38602256898](https://github.com/CrispStrobe/CrispASR/actions/runs/38602256898)
is also submitted. The model command passes and all returned payloads, original coefficients,
encoder values, transcripts and actual binary snapshots are independently
rehashed. General CI remains queued; no C++ inference runtime changes are made.
[Official SDK matrix38598516003](https://github.com/CrispStrobe/CrispASR/actions/runs/38598516003)
passes both architectures with all six requested target links audited in full
logs. The legacy workflow retained no cache/compiler-command artifacts, so
actual final compiler definitions are not independently claimed from it. Metadata
retention is added and SDK follow-up38605386237 passes both chips with
actual compiler definitions independently audited. Physical
NPU, quantized speech quality, ACL capture/replay and performance remain
unvalidated. All heavy local files stay under
`${CRISPASR_STORAGE}/crispasr/cann-model-validator-20261014`.

## Accepted full interface result

The exact complete CPU command passes encoder diff, four repeated automatic/
explicit Session transcripts, two native encoder arrays and two CLI outputs.
The CANN request runs and accepts those same CPU controls, then exits1 with
`Supplied build does not enable CANN`; it never converts that into CANN success.
The real GPU=true public encoder request also rejects the absent registry.
Eight unit guards execute in hosted output. All encoder arrays match the normal
accepted build byte for byte, and all Session text matches the original exactly.
CPU minimum row cosine0.999999998826876 and relative L2 error8.741357385766627e-6.

Actual ELF snapshots are included this time: CLI, diff, shared API and ggml
library bytes are fully rehashed against their receipts for both command arms.
This establishes actual retained binary identity; the validator still does not
assert that a supplied binary was compiled from its own checkout revision.
Full build/compile-command/ldd evidence is retained. These checks accept a single
pinned F32 JFK fixture, not a general corpus or quantized model.

[Complete complete-hosted-portable-validator.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/e53a3693849d07ba0d3d234f2a6b9345dd8cbbfa/cann-20261014-model-validator/complete-hosted-portable-validator.zip): 350,235,628 bytes, anonymously verified SHA256 `254a0036250ba07704e1427feab9d7f280b90060225a4a68a3e1c4cfa5a71bba`.

[Complete complete-hosted-portable-validator-audit.zip](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/e53a3693849d07ba0d3d234f2a6b9345dd8cbbfa/cann-20261014-model-validator/complete-hosted-portable-validator-audit.zip): 34,743 bytes, anonymously verified SHA256 `3b86ca79e573e34693e551f3954cae32fe49e011c85e5cc781306d6bf7978f84`.

[Complete two-chip SDK target-link evidence](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/1c6a10cf554a54344294c6e734d313cf93ae3b1d/cann-20261014-model-validator/complete-strict-routing-sdk-logs.zip): 401,056 bytes, anonymously verified SHA256 `49b201145a16c7b8ff96b4366e75823d1b0de82d37462eeb94a3a4173c88b6a3`. No NPU execution or missing compiler metadata is inferred from these logs.


## Compiler metadata follow-up accepted

[Official SDK run38605386237](https://github.com/CrispStrobe/CrispASR/actions/runs/38605386237)
passes at source `230d7e490b243e1ecd47606467782e47bd254fe3` on both architectures. Independent inspection
of each retained CMake cache and all566 compile commands confirms all three
CANN translation units carry `ASCEND_310P` without `USE_ACL_GRAPH` on ARM64,
and `ASCEND_910B` with `USE_ACL_GRAPH` on x86_64. Complete checkout/job logs
confirm source identity, all six target links per chip, and actual routing/
rotary CPU controls. Both metadata archives match official GitHub SHA256/size
and pass CRC, safe-member and complete per-file hash checks. Two positive and
sixteen rejecting synthetic audit controls also pass.

[Complete SDK metadata and independent audit proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/dc70b6f1abc67addaf0fae824cbcd2872ab39809/cann-20261014-model-validator/complete-sdk-compiler-metadata-proof.zip):
455,549 bytes, anonymously verified SHA256 `0eaa653ab99d7eb47101b17dfeaa2038cae4e86563e1aa27e24da943b7c2e42c`.
Includes both raw official metadata archives and complete job logs. This
closes the compiler-definition evidence gap; it retains no SDK ELF bytes and
does not establish physical NPU inference, ACL execution, quant quality or speed.
