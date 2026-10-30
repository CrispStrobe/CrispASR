# Dia hosted terminal interventions — 2026-10-15

Follow-up: the [actual full-original trace and one-decision fork](dia-original-trace-results-2026-10-16.md)
are complete. All18 actual self-attention captures pass exact OFF/ON/OFF
controls; replacing only original sampled877→904 reproduces quiet audio and
empty ASR. The terminal findings below remain valid; the reciprocal native
decision intervention is the next pending sensitivity test.

The actual pinned hosted graph now separates terminal normalization and head
projection effects. Heavy38805728288 passes at85be03e3; lint38805731957 passes
all10 jobs. Independent audit checks277 payloads,5 arms,90 layer comparisons,
complete lossless reconstruction, source/method/ELF/library identities and a
fresh retained ggml guard. Native layer17 and native final-hidden controls
reproduce all127 baseline stages and logits byte-for-byte. Original layer17
reproduces the earlier37-arm anchor exactly.

| Input to native terminal path | CFG46−904 margin | Ranks46/904 | Final-hidden relative L2 |
|---|---:|:---:|---:|
| baseline | -2.670288085938e-05 | 33/32 | 2.280212657e-06 |
| native-layer-17 | -2.670288085938e-05 | 33/32 | 2.280212657e-06 |
| native-final-hidden | -2.670288085938e-05 | 33/32 | 2.280212657e-06 |
| original-layer-17 | +2.098083496094e-05 | 32/33 | 7.380180909e-08 |
| original-final-hidden | +2.479553222656e-05 | 32/33 | 0.000000000e+00 |
| Original full reference | +4.625320434570e-5 | 32/33 | 0 |

Both original-input arms restore the exact original retained sampling support.
Every earlier stage/cache remains native exact for final-hidden injection;
computed and consumed hidden are retained separately. Original final hidden is
consumed byte-for-byte, with no local module reconstruction substituted for it.

The signed sequential differences are:

- Native→original post-layer17: +4.76837158203125e-5.
- Original post-layer17→original final hidden: +3.814697265625e-6.
- Original final hidden→original full logits: +2.1457672119140625e-5.

They sum exactly to the7.2956085205078125e-5 margin gap. These are conditional
intervention differences, not global additive causal percentages. The second
change isolates the propagated final-normalization difference on the actual
original post-layer17 input; the last isolates the remaining head-projection
path on the exact captured original hidden. Earlier local component bit matches
do not imply full-runtime/hosted byte reproducibility.

For codebook2, the actual original CFG expression gives

```
margin = 4*(c46-c904) - 3*(u46-u904)
       = 0.9638805389404297 - 0.963834285736084
       = 0.000046253204345703125
```

Thus a small difference between two approximately0.964 contrasts determines
the rank cutoff. With original hidden supplied exactly, remaining conditional
and unconditional contrast errors are−1.71661376953125e-5 and
−4.291534423828125e-6. All four raw logits and every contrast contribution are
retained in the structured receipt. Re-evaluating CFG in F64 gives exactly the
same margins and sequential differences for all five arms: CFG rounding does
not explain this measured boundary. The original/native floating-point neural
outputs differ before the discontinuous nucleus-support decision.

This does not identify a new algebraic runtime defect or establish speech
acceptance. Same-input norm/head/late-layer component probes are already done;
do not repeat them unchanged. The next useful full-model reference task is
simultaneous actual self-Q/projection/cache capture with OFF/ON/OFF transparency.
A further original-only single-token fork can test whether the first native
support change is itself sufficient to send the original into a quiet trajectory,
without changing the production sampler or quality gate.

Skipping the prefix does not reproduce its RNG progression. Native speech
remains rejected; no Q4, GPU, speed, default or release acceptance is claimed.

[Complete public raw proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/6b0b9fdf167e4a01bcdc2a397f27f246f0eadaf7/dia-20261015-terminal-replay/complete-hosted-terminal-replay-v1/raw.zip):315,247,244bytes, SHA256
`a2d4c27e7ef6351e9b0026c1389ec9fd409c33421694ac1e596b090a0ec2cc21`. All18 published files
pass full anonymous SHA/size verification. The publication retains a runnable
audit-method folder layout, full raw captures and reconstruction recipes,
actual sources/ELF/libs, original sampler, logs, licenses and independent audit.
[Structured result](dia-terminal-replay-results-2026-10-15.json);
[integrated method](../../tools/ci-heavy/dia-terminal-replay/README.md).
