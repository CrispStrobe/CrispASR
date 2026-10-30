# CANN layout lifecycle and Mac live follow-up — 2026-10-20

Implementation source: `18b540bd5c3e3abe2a3d51e797bd40d81daabc9f`,
shared ggml `e2ae679a`. The NZ cache remains off by default.

## Accepted host and Mac evidence

- The production cache bodies pass 53 lifecycle checks using mocked SDK memory
  calls. Aliases, neighboring allocations, cached copy destinations, allocator
  failure/recovery, byte coverage and cleanup are covered. A zero-length alias
  write failed before its repair; the failed-control identity is retained.
- The registered CTest passes; 83 weight-usage assertions, 26 production policy
  checks, 10 model-validator tests and six device-receipt guard tests pass.
- The expanded CPU probe completes 576 real computes in 48 cases. The device
  runner independently reconstructs every matrix product from saved inputs.
- The immutable F32 Moonshine fixture passes the original-reference stage,
  magnitude and decoded-text gates on this Mac's CPU and Metal. CLI and automatic
  and explicit Python Session dispatch pass. Strict assignment evidence identifies
  encoder matmuls on `MTL0`; CPU cannot silently substitute for Metal.
- A public Session keeps one handle through short/double-length/short inputs.
  Both devices produce 22/44/22 words and identical normalized transcripts.
  This is input-length/cache reuse coverage. The longer input uses the accepted
  CPU arm as its regression comparison, not a new original-model reference dump.

Metal vs the original reference: minimum row cosine 0.999982347,
maximum row-norm relative error 0.000772651, relative L2 error 0.000573751.
The declared thresholds are unchanged. These results establish no NPU execution,
NZ arithmetic, CANN full-model quality, ACL replay or performance claim.

[Machine-readable receipt](cann-layout-followup-2026-10-20.json).

[Complete sanitized host proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/a4cfe1799f153f86591ba6956d0da442b376c032/cann-layout-20261020/mac-host-proof.zip): 3,005,130 bytes,
SHA256 `820ec04e9d69507975e249ba050f19b707ca65194ae07003bff6409e963b5a32`.
Its full anonymous download is verified. The archive retains numerical arrays,
576 compute payload sets, complete normalized logs/build metadata and a per-file
manifest with original and public-text hashes. Private path prefixes are replaced
by portable aliases. Runtime binary hashes are retained; speech-model weights
and local credentials are not republished.

## Next physical experiment

The [refreshed two-architecture SDK matrix](https://github.com/CrispStrobe/CrispASR/actions/runs/39401353196),
[general CI](https://github.com/CrispStrobe/CrispASR/actions/runs/39401352847) and
[lint](https://github.com/CrispStrobe/CrispASR/actions/runs/39401352899) target the
exact implementation source above. Both refreshed SDK legs (310P/off and
910B/ACL-on), all13 general CI jobs and all10 lint jobs pass. These are the
refreshed-source results; no earlier compilation is substituted.

With SDK compilation passed, optional hardware help starts with the bounded,
model-free probe on 310P. Source the host's installed CANN SDK environment, then:

```sh
git clone --filter=blob:none --no-checkout https://github.com/CrispStrobe/CrispASR.git CrispASR-layout-proof
cd CrispASR-layout-proof
git checkout --detach 18b540bd5c3e3abe2a3d51e797bd40d81daabc9f
git submodule update --init --recursive
python3 tools/ci-heavy/cann-weight-layout/run_device.py \
  --source 18b540bd5c3e3abe2a3d51e797bd40d81daabc9f \
  --chip 310p --device 0 --out ../proof-310p-layout
```

The output must be fresh and outside the clean checkout. The runner builds the
probe, checks actual compiler chip/ACL definitions, verifies source/submodule
and binary hashes, runs ND/cache arms on the exact CANN device and independently
audits all 576 computes per arm. It retains full success/failure logs and payloads.
Return the entire proof directory plus the device/SDK details; do not discard a
failed run or send only the final PASS line.

For 910B, change `--chip` to `910b` and use a new output directory. It additionally
requires real positive ACL capture **and replay** counters in graph-on runs.
Once the small probe passes, `--with-model` adds the pinned F32 Moonshine eager
ND/cache diff, CLI, Session and reuse checks. The NZ-on model arm must report
positive actual NZ pointer-selection evidence, not just NPU placement.

MiMo F16, full-model ACL capture, quantized quality, load/compute memory and fair
warmed timing remain later acceptance tasks. Kaggle CUDA cannot establish CANN
NZ correctness; CPU preparation belongs on hosted CPU or the local Mac.

## Published follow-up

The implementation and this receipt are on remote `main`; shared ggml is on
its canonical `crispstrobe-ops` branch. The reporter has been updated on
[CrispASR PR544](https://github.com/CrispStrobe/CrispASR/pull/544#issuecomment-6288185477)
and [shared ggml PR4](https://github.com/CrispStrobe/ggml/pull/4#issuecomment-6288187900).
The physical probe is optional and should follow successful refreshed SDK builds.
