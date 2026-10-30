# CANN weight usage and NZ cache

Optional implementation integrated 2026-10-20. Physical Ascend acceptance is pending.
No model files are renamed or republished.

## Contract

`core_gguf::WeightUsagePolicy` is an optional model-supplied classifier, applied
before sizing, allocation and upload in ordinary, filtered, split and fit loads.
`MatmulOnly` describes immutable two-dimensional weights used exclusively as
`src[0]` of `MUL_MAT`. `General` vetoes layout repacking; use it for embeddings,
tied embedding/output tensors, convolutions, views and unclassified weights.
`Unspecified`, or no classifier, leaves legacy recognition available.

These hints use two new bits in ggml's existing tensor flags field, without
changing the tensor structure size. They are not written to GGUF. Names and
lookup keys remain unchanged. MiMo and Moonshine supply audited policies;
other models retain unspecified usage until their consumers are audited.
The CPU repack loader marks the same usage bits from its existing model
predicate. Neither PARAM nor a tensor being two-dimensional is sufficient.

CANN's cache compatibility policy recognizes bounded legacy/dot-separated
attention and FFN suffixes, fused `attn.qkv.weight` / `attn_qkv.weight`, and
`lm_head.weight`. It excludes BF16, quantized, batched, noncontiguous, view and
non-leaf tensors. An explicit General hint takes precedence over every name.
This fallback is not a complete architecture-independent usage audit.

## Execution modes

| Environment | Behavior |
|---|---|
| `GGML_CANN_WEIGHT_NZ=off` | Ordinary ND weights; no NZ conversion/cache |
| `GGML_CANN_WEIGHT_NZ=on`, cache unset/off | Existing in-place NZ implementation, retained for A/B; legacy recognition only |
| `GGML_CANN_WEIGHT_NZ=on GGML_CANN_WEIGHT_NZ_CACHE=on` | New optional cache with explicit usage and extended compatibility recognition |

Environment settings are read once per process. Use a separate process per arm.
The cache is off by default until physical correctness and performance gates
pass. The legacy path retains its historical read/copy restrictions; this
candidate does not claim those restrictions are repaired outside cache mode.

In cache mode, `tensor->data` remains ordinary ND data. A separate NZ allocation
belongs to the CANN weight buffer and is built during synchronous upload, before
ACL graph capture. Upload coverage uses a union of byte ranges, so duplicate,
overlapping and out-of-order chunks cannot declare completion while holes remain.
A synchronous partial update after initialization rebuilds the NZ representation.
Async writes, copies, memset and clear invalidate overlapping caches; execution
uses ND until a later synchronous upload rebuilds them. Views always use ND.

Matmul selects the recorded ready cache, not a fresh name-based interpretation.
Cache state changes invalidate captured ACL graphs to refresh their ND/NZ pointer
selection. Ordinary reads and copies, including CPU transfers, see ND bytes.
The registry procedure `ggml_backend_cann_nz_cache_size(const ggml_tensor*)`
returns ready-cache bytes (zero for ordinary or invalidated weights), allowing
physical probes to reject an optimization that silently did no NZ work.

The optional cache can nearly double floating-point weight device memory, plus
padding/workspace. Cache allocation failure keeps the ordinary ND weight usable;
other SDK failures remain errors. This does not guarantee sufficient remaining
memory for a model's compute graph. Measure load time, peak memory and steady-state
latency before adopting it for a model/device. Quantized model recipes are unchanged.

## Follow-up lifecycle and live controls

The lifecycle host test executes the production cache function bodies with
mocked SDK memory calls. It covers overlapping uploads, alias/range invalidation,
copy destinations, allocation failure/recovery, destruction and zero-length
writes. It is not an Ascend conversion or numerical test. A zero-length alias
write failed before the follow-up repair; it now preserves cache readiness and
the layout epoch.

Model validation can require actual cache selection with `--nz-cache on`.
Each encoder receipt records cached matmul nodes, unique weights/bytes and host
matmul encodes that selected the NZ pointer. These selection counts are not
physical kernel execution counts. Ordinary ND controls require zero NZ usage.
CANN placement and successful output are checked separately. The `--exercise-reuse`
option adds public Session short/double-length/short calls; CPU/device transcripts
must agree, short output must survive the shape change and the longer input must
establish proportional work. The double-length input is a CPU/device regression
comparison, not a newly dumped independent original-model oracle.

The model validator supports macOS CPU/Metal as local regression controls and
Linux CPU/CANN for the physical test. Metal cannot satisfy an NZ-on request.

## Host controls

The [retained host receipt](historical/cann-weight-layout-host-2026-10-20.json) records both
mmap modes, loader regressions, the432-compute CPU execution control and rejection
of an unavailable CANN device. It establishes no NPU result.


Build `test-weight-usage`, `moonshine` and `mimo_asr` with the ordinary repository
build. Run `test-weight-usage` and `python tests/test_cann_weight_layout.py -v`.
The Python test compiles the production SDK-free policy; it does not simulate NPU
arithmetic. Run GGUF loader regressions with mmap enabled and disabled.

Build the portable execution probe with
`-DCRISPASR_BUILD_CANN_LAYOUT_PROBE=ON`, target `cann-weight-layout`.
Its CPU control is:

```sh
build/bin/cann-weight-layout --backend cpu --device 0 --out proof-cpu --require-nd
```

The extended probe executes576 computes in48 cases. It verifies the independent scalar oracle, actual ordinary read/copy bytes,
F32/F16, aligned/irregular widths, repeated graph execution and changing input
shapes. A CANN request fails if CANN or the requested device is unavailable;
a CPU run cannot satisfy `--require-nz`.

## Physical acceptance

The manual CANN SDK workflow compiles the probe on both supported architectures.
SDK compilation alone cannot accept the implementation. On the same physical
310P3 and 910B3 setups, retain full configure/build logs, compiler definitions,
SDK/chip details, native Git/submodule revisions and binary/input/output hashes.
Use fresh proof directories and fresh processes:

```sh
GGML_CANN_WEIGHT_NZ=off GGML_CANN_WEIGHT_NZ_CACHE=on \
  build/bin/cann-weight-layout --backend cann --device 0 --out proof-cann-nd --require-nd
GGML_CANN_WEIGHT_NZ=on GGML_CANN_WEIGHT_NZ_CACHE=on \
  build/bin/cann-weight-layout --backend cann --device 0 --out proof-cann-cache --require-nz
```

Run both with `GGML_CANN_ACL_GRAPH=off`, and repeat on 910B with it `on` and
`--require-acl-replay`. The latter requires positive actual capture and replay
counters, not merely an enabled build flag.
Keep all payloads and logs. Require every supported case, cache readiness check,
exact ordinary-byte check and independent arithmetic comparison to pass.
No unsupported case or zero-test run is accepted as a pass.

Then run the existing Moonshine full-model validator through diff, CLI and
Session on identical pinned models/audio/configuration, first ND then cache.
Require stage cosine **and magnitudes/errors**, decoded speech and actual CANN
placement. Repeat with MiMo F16 where hardware capacity permits. Quantized cases
remain quality controls rather than NZ-benefit claims. Use the existing composed
probe for actual rejected-BF16 scheduler fallback; a plain byte transfer alone
is not full scheduler-fallback acceptance. Retain original failed evidence.

Only after quality acceptance, profile load/encoder/decode and repeated warmed
medians under identical load, with memory measurements. Leave the cache optional
until both quality and resource/performance evidence justify a default change.

## Integration scope

Shared-fork integration6b4bd833 preserves its newer canonical history with
exactly the same source tree as tested candidate4d734595. CrispASR pins that
integration commit. The new cache remains off by default, with the legacy path
retained for A/B. Host controls are accepted; final-source
[SDK compilation](https://github.com/CrispStrobe/CrispASR/actions/runs/39391430100),
[general CI](https://github.com/CrispStrobe/CrispASR/actions/runs/39391432699) and
[lint](https://github.com/CrispStrobe/CrispASR/actions/runs/39391434462) were still
pending/running at integration. No physical NPU, model-quality or speed claim
is established by merging this optional implementation.

## Bounded device runner

`tools/ci-heavy/cann-weight-layout/run_device.py` accepts an exact40-hex source
revision, requires a clean matching checkout/submodule and puts all build/proof
files outside it. Source the installed CANN SDK environment first, then run:

```sh
python3 tools/ci-heavy/cann-weight-layout/run_device.py \
  --source EXACT_40_HEX_COMMIT --chip 310p --device 0 --out ../proof-310p-layout
```

Use `--chip 910b` for the910B eager plus capture/replay matrix. The first request
is probe-only and needs no speech-model download. The runner checks actual
compiler chip/ACL definitions, exact device selection and source/binary hashes,
retains full logs/payloads and independently recomputes all576 products per arm.
It rejects missing/duplicate cases, zero work, absent NZ selections, absent
payloads and absent required replay. Every failed run leaves its proof intact.

After the small probe passes, `--with-model` adds the immutable108,495,214-byte
Moonshine fixture (SHA2567d63d2a5b90eb77843aff25107b1d1f70bafcfc1c8f73075e2f5a9a3f87d5e84)
and actual diff/CLI/Session/reuse checks with NZ off/on in eager mode. This needs
NumPy and the repository GGUF Python module, but no Torch or model reconversion.
Full-model ACL capture and performance are still separate acceptance tasks.
Return the complete proof directory, including failures, rather than just the
final PASS line. Review private machine paths before publishing raw logs.

[Completed follow-up host/Mac controls and immutable device procedure](historical/cann-layout-followup-2026-10-20.md).
