# Moonshine Tiny native CPU backend fixture — 2026-10-14

The unchanged actual Moonshine backend reproduces the original JFK transcript
exactly on both calls. It uses the complete pinned F32 model and original PCM;
no teacher forcing, transcript rewriting, generation fishing or runtime change
was used. [Original coefficient/recipe/dumper proof](moonshine-original-fixture-audit-2026-10-14.md).

| Actual encoder comparison, all457×288 values | Result |
|---|---:|
| Minimum row cosine | 0.9999999997296947 |
| Maximum row norm error | 5.323259081535525e-6 |
| Relative L2 | 2.1074627915260832e-6 |
| Maximum absolute error | 4.9054622650146484e-5 |

Two encoder calls agree byte for byte; both complete native transcripts equal
the original text byte for byte. The actual predeclared producer guard passes
the complete arrays and rejects double-scale, nonfinite and wrong-shape
controls. Cosine alone was not the acceptance criterion.

A bounded standalone probe compiles the actual unchanged backend, tokenizer
and core GGUF loader from sourcea149749. Full source/build/symbol/library
closure evidence is retained. Existing ggml CPU libraries are rehashed against
the prior complete receipt. Their c36dab89 CPU/core source tree is identical
to current0a8620765: only three CANN files differ. Explicit CPU init avoids
claiming GPU work. The initial missing-loader link and private-header include
errors remain preserved; the final executable links, exposes the actual public
backend functions and completes both model runs successfully.

[Complete public33-payload native proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/5afafd2e40d3941b49979c41a3960b175bfe71a4/cann-20261014-model-fixture/complete-native-moonshine-backend.zip) includes the actual source,
probe source/executable, build logs, verified CPU libraries, tokenizer, PCM,
full encoder arrays, transcripts and method/result receipts. Anonymous full
download verifies `5b99d8ca092ac86db19b0c652d7ba5323e90052974dacfd49052c1f2a1f6613c`, 3,649,899 bytes.
[Complete result ledger](moonshine-native-backend-fixture-2026-10-14.json).
All compiled fixtures and heavy artifacts stay under
`${CRISPASR_STORAGE}/crispasr/cann-model-fixture-native-backend-20261014`.

This is actual CPU backend stage and decoded-output proof. It does not replace
full CLI/C ABI dispatch checks. [Hosted run38555394740](https://github.com/CrispStrobe/CrispASR/actions/runs/38555394740)
passes fresh conversion, original dumper/generation, crispasr-diff, CLI and
repeated auto/explicit Python Session checks through the freshly built shared
library. [Its returned artifact is independently accepted](moonshine-hosted-cpu-fixture-2026-10-14.md). NPU, quantized model,
actual ACL replay and performance acceptance remain separate gates.
