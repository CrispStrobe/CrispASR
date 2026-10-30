# Dia layer0 keys: projection and rotary history error

The actual native producer reconstructs all979 captured layer0 key-cache rows
byte-for-byte. On the same shared input history, most measured original/native
key error is introduced by rotary arithmetic. This is a component diagnosis,
not a new runtime or speech repair. The earlier source-RoPE full-speech attempt
remains rejected; this result does not justify repeating it unchanged.

Only11 original F32 tensors are loaded: nine embeddings, the pre-attention norm
and key projection, about80MB total. Immutable HF byte ranges and every physical
native F32 tensor hash match the previously accepted export audit. The full
checkpoint is neither loaded nor claimed to be locally rehashed. All979 input
rows match the independently retained original/native histories. Configuration,
original source4a9e29b1, native source8d0ad886 and ggml0a8620765 are pinned.

The actual original embedding loop, RMSNorm, DenseGeneral and RotaryEmbedding
run one step at a time on Torch2.6.0+cpu. Original cache reconstruction passes
its predefined2e-6 error/norm gate at relative L2 7.720519e-7, minimum head-row
cosine0.999999999998802 and maximum norm error9.507806e-7. It is not byte-exact
across the retained/current processes. The last original input embedding is
byte-exact, and repeated original calculations are exact within this process.
Batched embeddings/norms stay exact versus one-step; batching projection changes
it by9.854700e-8 relative L2, so batched math is kept separate from the gate.

| Same-history native producer stage | Relative L2 vs original |
| --- | ---: |
| Summed embeddings | 0 (byte-exact) |
| Pre-attention normalization | 4.872930e-8 |
| Key projection, before RoPE | 2.240087e-7 |
| Keys after native RoPE | 1.571270e-5 |
| Original normalized inputs → native projection/RoPE | 1.571259e-5 |
| Original pre-RoPE keys → native RoPE | 1.571131e-5 |

The native full producer matches the actual captured cache byte-for-byte, and
all four native stages are identical with1/4 threads. Feeding exact original
pre-RoPE keys through native RoPE leaves almost the same aggregate discrepancy;
normalization/projection are much smaller effects here. Full and RoPE-only error
vectors have cosine0.999896517, with their difference norm0.01438597 times the
full error norm. These are controlled numerical interventions, not additive
percentages of a speech failure.

At position0 pre/post-RoPE error is1.842e-7. Post-RoPE error grows to1.072e-5 at
position512 and3.553e-5 at978, while pre-RoPE error remains about2e-7. All complete
arrays and selected-position cosine/norm/error metrics are retained. Scale,
reversed heads and shifted positions reject; changing one real input token
at step978 rejects with48.4% relative error. Native output files and original
NPY arrays use explicitly documented axis mappings.

The independent auditor rehashes all18 prepared input files, checks all11
physical coefficient mappings and the canonical config, recomputes16 stage
rows/27 selected-position rows, verifies byte-exact native/captured and thread
controls, and recalculates rejecting controls from their actual arrays. The
fresh actual CPU executable SHA256 is
`ddae9dd904881c621f60d8fb174ff596f982e8850f6f9bcbe3ad1817cf52c7ba`.
Its original compile command, runtime library closure and the verified shared
libraries are retained separately from source identity.

[Complete98-payload proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/c5fb5fd2a70d6c37126c688e2dbdb373fd22e586/dia-20261015-key-history/complete-key-history-producer-v1.zip)
is258,062,906 bytes, SHA256
`099e5150548f9c9908bd94267cfc2c35af5f865b34ca5d77594529641c12a50c`,
fully anonymously downloaded/rehashed. It includes required coefficients,
original/native captured keys and tokens, complete stage/intervention arrays,
actual source/ELF/libraries, method scripts/logs, independent audit and notices.
Candidate source matches the tested/published bytes.

[Reusable tool and producer](https://github.com/CrispStrobe/CrispASR/tree/c4a9a78e643df6ee59bdc95fe36bae39e06adccd/tools/ci-heavy/dia-key-history)
are pushed atc4a9a78e. [Applicable lint](https://github.com/CrispStrobe/CrispASR/actions/runs/38723508075)
is submitted; tooling is not yet integrated. No production/default, full speech,
CUDA, Q4, performance or release acceptance follows. A useful next distinct test
is a fixed shared-history full-decoder intervention with original layer0 key
prefix/current-key controls, measuring the46/904 CFG margin. It must first
reproduce baseline input/logit hashes and preserve all failed arms.

[Structured receipt](dia-key-history-producer-2026-10-15.json).

## Diagnostic tool integration

The [lint run38723508075](https://github.com/CrispStrobe/CrispASR/actions/runs/38723508075)
passes all10 jobs on c4a9a78e643df6ee59bdc95fe36bae39e06adccd. Rebased/ff
integrationc1be30229bcd180753ef6923077f4423a1287e89 preserves both tested
producer sources and the reproduction README byte-for-byte. The full independent
component proof predates integration and remains public. This accepts the
component diagnostic tools only; production math/defaults and speech status do
not change.
