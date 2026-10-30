# Dia current-runtime cache/sampler factorial — 2026-10-16

The disabled-by-default controls are integrated at ad0d1546 with executable
source bytes exact to tested5e04955e. Heavy38892690744, all13 CI and10 lint
jobs pass. Independent v3 verifies535 payloads and all30 initial stage metrics,
actual original/native DAC provenance, CLI/Session equality and ASR results.

Holding generic serial-head arithmetic fixed, the original prompt and seed42:

| Cross cache | Sampler | First differing original input | Frames | Native/original DAC/CLI WER |
|---|---|---:|---:|---|
| F16 | Standard | 1 | 1423 | 1 / 1 / 1 |
| F16 | Portable | 865 | 1099 | 1 / 1 / 1 |
| F32 | Standard | 1 | 1423 | 1 / 1 / 1 |
| F32 | Portable | None: all1226 rows exact | 1210 | 0 / 0 / 0 |

The passing arm restores all original code frames. Actual transcript from
both DACs and the CLI: "We are committed to making it secure." Native PCM RMS
is0.0735281; failed controls are around0.00016-0.00018 with empty transcripts.
Native decode replay is exact, and CLI/Session inputs/logits are byte-identical.
Original is the positive control. Pristine/OFF/identity controls remain exact.

Re-executing the actual original Python sampler on the actual native logits
reproduces every10035 F16-portable and11034 F32-portable raw head decision.
Thus F16's later divergence is neural numerical drift, not a sampler mismatch.
The different standard sampling recipe explains the first feedback difference
on original as well as native logits. The earlier head-reduction near-tie repair
remains a distinct required part of this scoped passing recipe.

Auditor v1 caught the producer excluding a nested tiny-test receipt from its
inventory. v2 encountered CIFS ENOMEM while re-extracting the same official ZIP.
v3 reuses the complete v1 extraction only after exact ZIP/local file-set and
size checks; verifies every indexed SHA, the extra receipt against fresh ZIP
bytes and its complete tiny test arrays/state, then all scientific results.
All failed logs and extraction attempts are preserved. No quality gate was
relaxed. The raw official ZIP SHA256 is
`d4753346f6f98088e28d098fe014c12b960a05ff8acbbed1648d90b7854ad25c`.

This accepts one CPU prompt/seed with opt-in serial head, F32 cross-cache and
portable RNG. Production defaults remain unchanged. Additional texts/seeds,
GPU/Q4 and profiling are separate tasks in PLAN.md. [Full audit](dia-cache-sampler-results-2026-10-16.json).

The first F16-portable fork is now traced mathematically on the actual captured
step864/head6 logits. The CFG logit margin of token55 over token1008 changes
from +1.8119812e-5 (original) to -1.0728836e-5 (F16), versus +4.0054321e-5
(F32). Token55 moves from rank42 to43. Cumulative top45-softmax probability
BEFORE it moves from0.9418578 to0.9565715, crossing top_p0.95. The actual
shifted nucleus filter therefore changes token55's probability from0.0153854
to exactly0, and the unchanged exponential race selects938 instead of55.
This is a filter-support discontinuity: the original race score margin is
about0.84936, not a race near-tie. Identical input histories through step864
and the cache-only intervention bound this attribution; this does not isolate
an individual cross-attention layer or prove broad F32 speech acceptance.

A fresh standalone reproduction executes the actual original sampler and
production C++ sampler/noise headers on these small arrays. All27 decisions
(9 heads for original/F16/F32) match, including native zeroing of token55.
No model weights or reference feedback are injected. The archive includes the
actual logits/noise, complete original source, native headers, observers,
executed scripts and a portable reproduction command.

[Complete factorial proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/b1e02825d4fdb91b8b702accdcfb252798d475ee/dia-20261016-cache-sampler/complete-factorial-v1/complete-cache-sampler-factorial-v1.zip) is 1248008904 bytes, SHA256
`3d11b3c3a5d9e35862c1f8a2507d5da2b15e8367602c897fb61a027f37d7ede7`.
[Small actual-logit proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/c0ccddf211ad0e5d9b9c5108def4d5062b4f4342/dia-20261016-cache-sampler/actual-logit-nucleus-v1/dia-actual-logit-nucleus-proof-v1.zip) is 501619 bytes, SHA256
`d3207e3cc5b0798bb834172fcf85e76b54d762336418e893cfd82e61862fee38`.
Both immutable public archives pass fresh full anonymous downloads and hashes.
