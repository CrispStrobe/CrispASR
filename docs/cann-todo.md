# CANN remaining work

Updated 2026-10-15. PR544 is merged; its requested integration is complete.
Two subsequent device-reported defects have narrow repairs with retained
before/after evidence. The 310P BF16 capability repair is integrated. The
910B rotary-capacity repair is also integrated after physical, SDK and
all thirteen general CI gates pass.

The maintainer owns these follow-ups. Contributor hardware help is optional;
there is no outstanding rerun request for the now-passing rotary candidate.
The rotary integration result is posted on PR544. Contributors can
unsubscribe from the merged PR; this document is the durable follow-up list.

| Order | Task | Current state / acceptance | Owner / hardware need |
|---:|---|---|---|
| 0 | Integrate rotary byte-capacity repair | Physical 910B3/SDK9.1:1691/1691 broad plus452/452 repeat. Both SDK9.2 builds pass. All thirteen general CI jobs pass; exact tested pin0a8620765 is integrated. | Maintainer; no further device rerun currently requested |
| 1 | Prepare a small full-model ASR validation bundle | Moonshine Tiny producer submitted in GH run38555394740, sourcea149749. Independent full coefficient/original-dumper audit and actual CPU backend stage/exact transcript proof pass. Hosted full diff/CLI/Session checks pass and returned bytes/arrays are independently audited; NPU remains pending. Prove CPU controls before publishing hardware commands. | Maintainer prepares; optional 310P3/910B3 execution afterward |
| 2 | Prove NPU full-model parity and decoded output | Same model/input/config on CPU and CANN; crispasr-diff stages include magnitude/error as well as cosine. Actual CLI and Python Session transcripts must pass declared quality gates. Preserve scheduler/device traces to prove real NPU work. Strict exact-device/encoder-placement implementation is integrated; hosted full-surface checks38575701613 and all13 general CI jobs pass, with returned bytes/arrays independently audited. Physical NPU not yet validated. | Maintainer audits; physical NPU needed |
| 3 | Verify actual scheduler fallback for rejected 310P BF16 cases | Current predicate/operator proof excludes unsupported cases; demonstrate supported CANN plus rejected CPU operations in a composed graph with correct output. Not yet validated. | Maintainer prepares CPU controls; physical 310P needed |
| 4 | Verify ACL graph execution and reuse | Current 910B build enables ACL graphs; operator logs do not establish executed capture/replay. Fixed-output graph-on/off comparison, repeated calls and growing/changing shapes; retain capture/replay evidence. Not yet validated. | Maintainer prepares; physical 910B needed |
| 5 | Validate shipped quantized inference | After floating-point model acceptance, test relevant F16/Q8/Q4_K recipes against reference stages and decoded speech. Keep sensitive tensors where measured necessary; no tolerance waivers. Not yet validated. | Maintainer prepares; NPU model runs needed |
| 6 | Profile and optimize | Only after output acceptance: prove offload/fallback, time load/encoder/decode separately, warm up and use repeated medians under identical load. Gate changes; compare quality and speed before changing defaults. No NPU performance claim yet. | Maintainer analyses; physical NPU timing needed |
| 7 | Maintain coverage and release documentation | Keep lightweight regressions, manual official SDK matrix and complete device receipts. Add model/device evidence when available; release only after applicable CI and acceptance gates pass. | Maintainer; hosted CI covers compilation only |

Evidence: [310P BF16 integration](historical/cann-310p-bf16-integration-2026-10-13.json),
[910B rotary physical proof](historical/cann-910b3-rope-physical-2026-10-14.md),
[rotary SDK proof](historical/cann-rope-sdk-2026-10-14.md),
[hardware protocol](historical/cann-hardware-validation-2026-10-12.md).
Physical tests preserve unsupported cases, prior failures and unchanged
upstream tolerances. Passing operator suites alone cannot accept a model.

[Completed rotary integration receipt](historical/cann-rope-integration-2026-10-14.md).

[Completed CPU fixture preparation](historical/cann-model-fixture-preparation-2026-10-14.md).

[Actual native CPU backend proof](historical/moonshine-native-backend-fixture-2026-10-14.md).

[Completed full CPU fixture proof](historical/moonshine-hosted-cpu-fixture-2026-10-14.md).

[Integrated strict device validation and bounded controls](historical/moonshine-strict-device-validation-2026-10-14.md).

[Portable full-model validator preparation and live GPU-argument controls](historical/cann-model-validator-preparation-2026-10-14.md); hosted interface and actual binary-snapshot proof pass and are integrated; physical NPU success remains pending.

Official compiler-metadata follow-up38605386237 is accepted on both chips: actual
ASCEND/ACL definitions and all six target links are independently audited from
complete retained artifacts/logs. Physical model/fallback/ACL/quant/timing tasks
above remain open. See the portable validator evidence linked above.

Current priorities and maintainer claims are in [PLAN.md](../PLAN.md).
Completed preparation/compiler/operator work is historical evidence; the
remaining tasks above require full-model or composed-graph device validation.

[Composed graph preparation](historical/cann-composed-preparation-2026-10-15.md) is
integrated after independently audited hosted CPU and both official SDK final
executable links,13 general CI jobs,10 lint checks and the secret scan. The
[actual mixed-graph probe](../tools/ci-heavy/cann-composed/README.md) checks18
computes over three shapes and rejects false backend/device/fallback claims.
The static consumer SDK search-directory repair is included; initial failed
links and complete public proofs remain retained. Physical310P composed
fallback and910B ACL capture/replay remain unvalidated.


[Weight usage and optional NZ cache implementation](cann-weight-layout.md): explicit
pre-allocation model hints, bounded name compatibility including fused QKV,
and an ordinary-byte-preserving optional cache are integrated, off by default. Host controls and SDK compilation do not establish physical/model or
performance acceptance. Existing published artifacts need no rename.
