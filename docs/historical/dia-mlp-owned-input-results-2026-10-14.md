# Dia block-zero MLP: actual fused original versus native split graph

A bounded replay now separates operator differences from input drift at the
late decoder step 978. It runs actual original DenseGeneral/MlpBlock ASTs
from pinned source `4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8` under Torch
2.6.0 CPU. Gate/up observations come from the real fused projection; hidden
comes from the actual wo input. The native graph uses the literal two MLP
expressions from source `8d0ad8862d2cad5d305a48a57f2a99b5024bc39a`,
3D CFG batch layout and byte-verified c36dab89 CPU libraries. Actual library
loading is checked. This is a bounded primitive, not a full-model scheduler
or GPU/performance test.

Two original F32 tensors, 201,326,592 bytes in total, were fetched through
strict immutable HTTP ranges. Both original coefficient shapes and converter
source are verified; all three gate/up/down mapped hashes match the full
physical F32 export audit. The GGUF down projection is `.wo`; `wo_mlp` is
the C++ field name. An initial probe assertion using the latter as a GGUF
name failed and was corrected from the actual converter/loader; that initial
log/method is retained. No coefficient or runtime change followed.

Both original and native pre-MLP-normalized inputs come from the independently
verified hosted capture. Every local native gate/up/hidden/output on native
input reproduces its hosted native snapshot byte-for-byte. Native repeats
and one-versus-four-thread comparisons are exact; original repeats are exact.

On matching inputs, maxima below are taken over both input owners:

| Stage | Minimum row cosine | Maximum row norm error | Relative L2 |
| --- | ---: | ---: | ---: |
| gate | 0.999999999999991 | 3.783e-08 | 1.095e-07 |
| up | 0.999999999999972 | 1.101e-07 | 1.672e-07 |
| hidden | 0.999999999999918 | 4.312e-07 | 2.399e-07 |
| output | 0.999999999999959 | 1.373e-07 | 1.911e-07 |

The source-owned operator comparisons are much smaller than the full pipeline
MLP-output difference. Local original versus hosted original on identical
original input has output relative L2 1.432e-6; this cross-host difference is
measured explicitly, rather than treated as byte equality. Changing only the
input to the actual original local operator gives output relative L2 8.269e-7.
Neither row is a same-input native operator defect. The same-input native
output difference is at most 1.912e-7.

Twenty metric rows and ten negatives are retained. Eight double-scale
controls preserve cosine but give norm error/relative L2 1. Two swapped
gate/up controls give output relative L2 about 3.363 and reject. All file,
coefficient, source, operator, library and input hashes are in
`dia-mlp-owned-input-results-2026-10-14.json`.

This excludes a large block-zero split-layout/SiLU/MLP primitive defect at
these owned inputs. Native free-running speech and Q4 remain unaccepted.
No numerical/default change or release follows. Next isolate self-output
projection and cross-attention on matching inputs with the actual original
mask/cache recipe. Separately, teacher-history capture transparency does not
prove default free-running execution is unaffected by debug output marks;
keep that full CLI/C-ABI control in the correctness roadmap.

Complete coefficients, source expressions, owned arrays, bridge, CPU libraries,
receipts and controls are public and anonymously SHA256-verified:
[complete same-input MLP proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/1bc7d487abad8e14bf4c9ef20ac9ac04b9d0baab/dia-20261014-decoder-boundaries/complete-owned-mlp.zip)
(382,167,825 bytes, 40 payloads; SHA256
`676d043bcd66e94285aefc143d89891552a8d306dba84520fa038940e401ed24`). The parent full hosted capture is linked in
`dia-decoder-boundary-results-2026-10-14.md`.
