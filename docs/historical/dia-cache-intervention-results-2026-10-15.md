# Dia full-decoder key intervention: complete negative result

Hosted [run38738218813](https://github.com/CrispStrobe/CrispASR/actions/runs/38738218813) and all 10 [lint38738222779](https://github.com/CrispStrobe/CrispASR/actions/runs/38738222779) checks pass at
`fc8f652fac1a602a2a9e7fbc645dc6795cd8f0e4`. Integration `e34d551b186cbd07fb07d9d81ae166adc30b0ce7` preserves
the tested diagnostic code after rebase. Production runtime and defaults do not change.

The independent audit checks the complete official raw ZIP SHA256, CRC, unique
safe paths and every one of its 720 payload hashes. All 979 forced input/logit
rows are byte-exact in both baseline and native-key substitution controls.
Every arm consumes the requested key array exactly; the earlier 978 logit rows,
layer0 input/norm/Q/V and cross inputs remain exact. All 18 decoder-stage
cosines, both norms, relative errors and final-stage metrics are independently
recomputed. The pinned original Torch2.6 sampler independently reproduces every
reported filtering support and rank.

Original head2 CFG46-minus904 is **+4.6253204345703125e-5**, with ranks32/33.
All native and original-key arms retain the opposite ranks33/32:

| Arm | CFG46-minus904 | Final hidden relative L2 | Logit relative L2 |
| --- | ---: | ---: | ---: |
| baseline | -2.67028808594e-05 | 2.28021266e-06 | 1.4132039e-06 |
| native-control | -2.67028808594e-05 | 2.28021266e-06 | 1.4132039e-06 |
| original-prefix | -2.47955322266e-05 | 2.25676895e-06 | 1.3754009e-06 |
| original-current | -2.5749206543e-05 | 2.27512444e-06 | 1.4096455e-06 |
| original-both | -2.14576721191e-05 | 2.25007358e-06 | 1.36786545e-06 |

Original keys move the logits slightly toward the original, but neither prefix,
current key nor both restore the original disputed ordering. Key-only
replacement is insufficient in this fixed-history fixture. This conditional
result does not settle correlated query/key phase error or establish a speech
repair. The earlier failed plumbing attempt remains retained in the
[preparation record](dia-cache-intervention-preparation-2026-10-15.md).

The next bounded hypothesis is a matched layer0 current-query/key intervention.
First verify original/current-query provenance, axes, pre/post-RoPE conventions
and physically audited coefficients. The retained original GGUF has only
decoder_block0_cross_q, not a self-attention query. Capture the actual original
self-attention query with a hook in the pinned original layer0 module; first
reproduce its retained original self-attention output as a positive control.
Do not substitute the cross-attention query or an unvalidated reconstruction.
Require baseline and native Q/K controls
to remain exact, then hold all other caches, inputs and sampling fixed. Do not
repeat key-only or all-layer source-RoPE speech runs unchanged. Any demonstrated
runtime repair still needs fixed-recipe CLI/C-ABI speech, multiple predetermined
prompts/speakers/lengths/reuse, original-stage diff, DAC/ASR and real CUDA gates.

The [complete raw hosted artifact](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/02eac70ef6025a299bc8bbae8976346c346e0687/dia-20261015-cache-intervention/complete-hosted-intervention-v2/raw.zip) is
1177709408 bytes, SHA256 `83db2618e48dd329f58da66dfd1ff2e3f493adfb6177cee1e1104c6838efe02c`.
The [immutable public manifest](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/02eac70ef6025a299bc8bbae8976346c346e0687/dia-20261015-cache-intervention/complete-hosted-intervention-v2/manifest.json) pins all
12 published files: raw archive, terminal APIs/logs, auditor/result and
license notices. Anonymous full downloads of every file match their hashes.
The archive retains all five arms, actual executable/libraries, source patch,
input fixture and original sampler. It establishes no speech, GPU, Q4 or speed
acceptance. [Structured result](dia-cache-intervention-results-2026-10-15.json).
