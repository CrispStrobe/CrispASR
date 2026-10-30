# MiMo Q4 activation-rounding diagnosis — 2026-10-15

The captured layer0 flash/eager input difference is amplified by real Q8_K
activation quantization. This bounded projection experiment establishes an
amplification mechanism; it does not accept the remaining PR492 changes,
whole-model parity, speech, speed or a default change.

## Measured result

Both arms use the same immutable Q4_K gate matrix, 4096 inputs ×11008 outputs,
and the original 71-column FFN-normalized inputs from hosted run37855479658.

| Flash/eager comparison | Relative L2 |
| --- | ---: |
| Captured F32 input | 0.0001377272 |
| Actual Q8_K activation shadow | 0.0011844056 |
| Same-weight F64 projection on original F32 input | 0.0001362483 |
| Actual default CPU Q4_K projection | 0.0007318077 |
| Same-weight F64 projection on actual Q8_K shadow | 0.0007318077 |

486 of290816 activation codes change. The actual output difference is5.3714×
the same-weight floating-input projection difference. The Q8 shadow predicts
the output delta to0.000122 relative L2 globally; nearly zero delta rows make
its minimum per-row cosine only0.7145, so that cosine is not an acceptance gate.
The final outputs themselves match the independent same-weight/Q8 oracle at
relative L2 7.85–7.86e-8, minimum row cosine >0.99999999999999 and maximum
row-norm error <1.05e-8. Scale×2 and reversed-channel negative controls reject.
The floating-input oracle differs by0.006237 relative L2: high cosine alone
would obscure this activation-quantization error.

## Controls and retained failures

The native Q4 dequantizer matches the independent GGUF decoder exactly. Q8
payloads decode independently and their16-element block sums match. A second
NumPy F64 matrix calculation matches the Torch2.6 oracle within4.45e-16.
An explicit ggml reference graph is byte-identical to the real scalar-row
dot routine. The default tiled graph differs from that reference by1.88e-7
relative L2, while matching the Q8 mathematical oracle more closely.

The first method wrongly required the default tiled graph to equal the scalar
routine byte-for-byte. That failed run and its raw arrays remain preserved;
the corrected method reproduces its default outputs exactly. Earlier failed
compile/codec/name assumptions and a license-metadata assertion also remain in
the proof. These failures are not model-quality gate changes.

Current ggml source0a862076 and capture-era c36dab89 have identical CPU kernel
and quantization source trees. Actual tested ELF SHA256:
`e0b96f82ad64fa75eb1b49bbc27bff2775cb96e3a3ef9a02ced230e51a9c1589`.
Source identity and actual executable identity are recorded separately.

[Machine receipt](mimo-q4-projection-2026-10-15.json) retains full metrics,
input/matrix hashes, exact tensor range, compiler command and independent audit.
Only the25,362,432-byte coefficient range was fetched from the4,517,988,832-byte
model; the full model was not downloaded or rehashed locally. Dequantizing
these Q4 coefficients does not recover original-checkpoint precision.

## Completed follow-up

The existing F32 cache-read OFF/ON experiment is complete and rejected:
[full four-arm result](mimo-kvread-ab-2026-10-15.md). It does not fix Q4
flash/eager parity, and no production default changed. Select a new specific
attention/activation arithmetic hypothesis before further whole-model work.
Actual CLI/C-ABI speech and controlled timing remain required before promotion.

## Public retained proof

[Proof ZIP](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/cb2326740f03ad1b32b952740ddcd485459a8d19/mimo-20261015-q4-projection/projection-proof-v3.zip):71 payloads,61,407,951 bytes, SHA256
`3537bef5468dfe87e6921b6ad5d02b2c8d59b2962430beacef9295a110138f68`; anonymously downloaded and fully rehashed.
Includes all raw projection/activation/oracle arrays, tested executable, scripts,
compiler/build metadata and failed attempts. Model coefficients/metadata remain
at the pinned public source rather than being redistributed; the archive records
all omitted model payload hashes and the exact bounded acquisition recipe.
Full coefficient copies are also retained locally under the storage task path.
