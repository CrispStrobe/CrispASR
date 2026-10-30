# CANN rotary-capacity repair integration — 2026-10-14

The production ggml pin advances to the exact physically tested candidate
`0a862076567afd8468822ccc512c97ac6a69b340`. The repair tracks allocated
sin/cos byte capacity separately from the last invocation’s rotary shape,
so constant-position wider rotations and width growth with fewer positions
receive sufficiently sized buffers. Rotary arithmetic and tolerances are unchanged.

[All 13 composed general CI jobs pass](https://github.com/CrispStrobe/CrispASR/actions/runs/38541851377)
on `1a36c871d76b162dcf0512a0fec450ce61ed4840`. Complete logs independently
verify each checkout, build and required step. The unit tier passes 2,020 tests;
audio sanitizers pass 121 assertions in 17 cases; Windows speaker tests pass
42 cases. Metal, Vulkan, dynamic backend discovery, fuzzing, AMR/Opus,
iOS and Android jobs also pass. The only skipped step uploads a fuzz crash
input when no crash occurred. All 142 log members are hashed and the full
ZIP is CRC-verified. Native terminal metadata and full logs are retained under
`${CRISPASR_STORAGE}/crispasr/cann-rope-integration-20261014`.

- [Both official SDK9.2 beta2 builds pass](cann-rope-sdk-2026-10-14.md),
  building CANN, CLI, shared ABI, diff, quantizer and the real operator test;
  four capacity regressions and thirteen routing cases per chip pass.
- [Physical 910B3/SDK9.1 proof](cann-910b3-rope-physical-2026-10-14.md)
  passes 1,691/1,691 broad cases with 1,474 unsupported retained, and the
  independent full RoPE repeat passes 452/452 with eighteen unsupported.
  Both formerly failing F32 NEOX shapes pass at unchanged tolerances.
- [Shared-fork merge ea526e120](https://github.com/CrispStrobe/ggml/commit/ea526e120deff8333fc42decc315137d7e297fca)
  preserves newer fork history on crispstrobe-ops. Its entire source tree is
  byte-identical to the physically tested pin, verified by Git tree identity
  and an empty full diff. CrispASR retains the exact tested pin.

Final composition `f693bfe254cdc3e7156a2ff3214aa45aa4a76899` merges current
main’s documentation into the tested source. Every subsequent difference
from CI source is documentation; runtime and workflow/test sources are unchanged.
[Full integration receipt](cann-rope-integration-2026-10-14.json) records
job steps, build/test evidence, all log member hashes and source identity.

The lightweight capacity regression remains in tests and both manual CANN
SDK jobs execute it. All four secret tests pass before publishing the fork
and CrispASR changes. Original failed controls and device archives remain retained.
Full NPU model inference, quantized speech quality, actual captured ACL graph
execution and performance remain separate follow-ups in [the TODO table](../cann-todo.md).
No release tag is cut by this integration.

Integrated on main in [d17c5d8](https://github.com/CrispStrobe/CrispASR/commit/d17c5d8b90fe941ed823580b3bf9234610347b59).
[Complete public integration CI proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/805706c285182f008b04207e4e93ef9ae689fc34/cann-20261014-rope-capacity/complete-integration-proof.zip)
retains nine payloads: native run/job metadata, full raw ZIP and logs,
all 142 log hashes, independent audit and collector/publication methods.
Full anonymous download verifies `536f1c7ef82412b46febbf4c54f869b800432fee01207d6fd8c00fce9452b657`,
4,069,034 bytes.
