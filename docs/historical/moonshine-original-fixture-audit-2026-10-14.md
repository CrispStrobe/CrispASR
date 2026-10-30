# Moonshine Tiny original fixture audit — 2026-10-14

The independent source check passes: all **160 tensors / 27,092,736 F32
values** in the pinned published GGUF match the original checkpoint byte
for byte through the actual converter’s mapping. Both complete files were
locally rehashed against their immutable Hub records; this extends the earlier
header-only dtype evidence.

| Input | Bytes | Complete SHA-256 |
|---|---:|---|
| Original checkpoint, revision390624ed | 108,389,192 | `867cd2215804859c55aa972d740bd5002be149b4e7526328c895d2408848c736` |
| Published GGUF, revision28281de8 | 108,383,136 | `64518aaf5535085075b081c8c1c89532e7cb5160efe90272c73937f179cf5f78` |

The actual torch2.6.0 CPU / transformers4.57.6 original model-card recipe
reproduces the existing JFK transcript **exactly**, including punctuation.
The processor passes input PCM unchanged with an all-valid attention mask;
the original6.5tokens/second rule gives maximum length71. Full token IDs,
original source/config files and the owned encoder array are retained.
The first local import failed on an incompatible system torchvision; its
complete failure log remains retained. Installing matching torchvision0.21.0
CPU in the isolated prefix fixed the environment without changing model,
inputs, generation recipe or gates. Shared packages were not modified.

Actual `tools/dump_reference.py` captures both original intermediates byte
for byte against the separate owned captures: raw audio `[176000]` and
encoder output `[457,288]`, both F32. Encoder norm is200.93036392366048.
The1,231,136-byte reference GGUF SHA-256 is
`8084853dff68e8d7f9bbfca2a1c91f3554fc7b27d49dbfec88dee9393df03fd0`.

[Complete public26-payload source/oracle proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/8d3500c572ce6f32bf0934357a45eb2934c4b320/cann-20261014-model-fixture/complete-original-moonshine-audit.zip) includes full160-row
coefficient ledger, actual mapping/modeling source, methods, original config/
card/tokenizer, transcript/token IDs, reference GGUF, owned arrays and raw logs.
Model weights remain at the two recorded immutable public Hub objects.
Full anonymous download verifies `cc84efd19a97c276118bcf3f3750bca46d045e13f3c55c430e67a9535df7cba5`,
2,395,093 bytes. [Full result ledger](moonshine-original-fixture-audit-2026-10-14.json).
All local files and isolated packages remain on ${CRISPASR_STORAGE}.

This establishes original coefficients, generation and reference capture.
Native stage/CLI/Session acceptance remains pending in hosted run38555394740;
NPU, quantized model and performance acceptance remain separate. No native
quality or speed conclusion follows from the original-only result.
