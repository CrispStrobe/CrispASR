# CPU attention accumulator isolated with tiny tensors — 2026-10-17

Sixty-four actual ggml CPU operator cells and an independent F64 audit establish
a shape-dependent accumulator effect. Constant V rows must produce V for any
normalized attention weights. The scalar F16 value accumulator instead yields
relative L2 **0.004595963669287369** (about0.460%), cosine0.9999895055406776 and
norms0.48396438873580994 /0.48379232021915236 for one query. Independently
rounding the sum to F16 after every addition predicts all12 constant-V scalar/
reference F16 cells byte for byte. The same stored values in the default64/71-row
tiled F32 accumulator satisfy the invariant.

Inputs cover32 channels,71 keys,two heads, two patterns,1/63/64/71 query rows,
one/four threads, F16/F32 K/V storage and default/scalar-reference execution.
Every query and mask row within a head is identical, so query count changes
dispatch without changing the mathematical problem. Both storage arms own the
same F16-representable K/V values. Scale×2, channel-order, shape and changed-input
negatives reject. Nonuniform worst score/probability/eager relative L2 is
4.510e-8 /1.379e-7 /1.819e-7; scalar F16 flash reaches0.0014147470788592978.
The actual patched F16 CPU dot trait consumes F32 operands; assuming another
Q-to-F16 conversion would misidentify the mechanism.

## Scope and next step

This is a source-backed operator attribution, not a MiMo repair. The old71-row
MiMo prefill selects the tiled F32 path, so scalar F16 accumulation cannot alone
explain that earlier flash/eager gap. The retained full capture has no raw Q/K/V.
Actual same-input Q/K/V, mask, scores/probabilities and pre-output-projection
capture with OFF/ON/OFF controls is still needed. Existing failed Q4/original
stage and decoded-speech gates remain unchanged. No production source, defaults,
GPU, speed or release acceptance follows.

Diagnostic source02bd2aeed3a1461280daaa89756136530ace9676 remains on its frozen
branch. All10 lint39003053402 and13 general CI39003202215 jobs pass; the
unchanged diagnostic is integrated on main. The small probe links previously
hosted/audited CPU libraries from
ggml0a862076, with all three physical hashes checked anew. No model weights or
full ggml build ran locally. Native compile uses unchanged-Wall/-Wextra/-Werror
and clang-format18 passes. Initial deprecated-setter compile and CIFS alias
failures remain recorded. Source, executable and library provenance are distinct.

## Permanent proof

[Public proof and reproduction](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/2ffac96dd901bb34e55755eef93094cfa8f8c14d/mimo-20261017-owned-attention/operator-proof-v1/README.md) includes all512 raw arrays, independent
auditor, actual executable/libraries, source/licenses, compiler/loader metadata
and negative controls. The ZIP is1183435bytes, SHA256
`5bf0b034a37fd0a1ba9b21d56733e622c15c5a07e1d5c396f8b932c2a8161dfb`; ZIP and README are fully
anonymously downloaded and hash-verified. Revision`2ffac96dd901bb34e55755eef93094cfa8f8c14d`.
See [machine receipt](mimo-attention-operator-2026-10-17.json).

The first old-artifact transfer failed its API hash and five member checks and
is retained as untrusted. An independent second transfer matches GitHub's API
hash and all463 member checks; both known layer0 FFN inputs match the earlier
independent projection proof. Neither archive contains Q/K/V. The failed
transfer was never used for numerical acceptance; its failure does not invalidate
the separately retained older projection proof.
