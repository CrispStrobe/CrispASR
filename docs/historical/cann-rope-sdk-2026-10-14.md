# Rotary-capacity candidate SDK compilation — 2026-10-14

[Run38536737288](https://github.com/CrispStrobe/CrispASR/actions/runs/38536737288)
passes both official SDK9.2 beta2 builds. Complete log ZIP SHA-256:
`6c9add26bf34c4e51ddd6d34a690eb4b6b41f25e3ab942192c68442669464af6`,
385,361 bytes; logs and native run/job metadata remain under
`${CRISPASR_STORAGE}/crispasr/cann-rope-sdk-20261014`.

| Architecture / target | ACL graph compilation | Actual build targets | Controls |
|---|---|---|---|
| ARM64 / 310P | Off | ggml-cann, CLI, shared ABI, diff, quantizer, operator test | 4 capacity tests; 13 routing cases per chip |
| x86_64 / 910B | On | Same six targets | Same controls |

The independently inspected complete logs contain actual target completions
and linker lines, original ggml checkout pin, SDK path and container digests.
Exact CrispASR source is `ca4c12c4cfec475c0009f9718a18ffb42d2cda6a`,
ggml `0a862076567afd8468822ccc512c97ac6a69b340`.
[Full compile receipt](cann-rope-sdk-2026-10-14.json).

The separate [physical SDK9.1 proof](cann-910b3-rope-physical-2026-10-14.md)
passes1691/1691 broad operator cases plus452/452 full RoPE repeat.
[Composed general CI](https://github.com/CrispStrobe/CrispASR/actions/runs/38541851377)
is still active. Production remains at ggml21acf3162 until that gate passes.
Compilation does not establish full-model NPU inference, quantized speech,
executed ACL graph capture or performance. [Remaining work](../cann-todo.md).

[Complete public SDK proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/2b1a7c30a1ed50a6147ca141715d81b4fe6e5ba7/cann-20261014-rope-capacity/complete-sdk-proof.zip)
contains all native run/job metadata, full raw ZIP/logs, independent target
receipt and the exact executed workflow/control sources. All nine payloads
were hashed and ZIP-verified; anonymous full download confirms SHA-256
`2f2709468f2ffc7a083f51e8977e33c4789a29cf430a5e87e45c6df1c7916832`, 824,796 bytes.
