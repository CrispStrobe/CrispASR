# Dia generic serial F32 head — unforced speech recovery

Applying the same serial F32 head projection at every step restores the exact
expected sentence in the pinned native runtime, without a sampled-token override
or reference input. The independent audit passes655 complete payloads. The
sampler, noise, loaded weights, current hidden and recipe remain native/fixed.
This demonstrates a sufficient arithmetic repair for the observed failure;
it does not establish general robustness or current-main/default acceptance.

| Arm | Decoder calls | Native DAC WER | Original DAC WER | Native RMS |
|---|---:|---:|---:|---:|
| Pristine | 1115 | 1 | not run | 0.0001556675 |
| OFF | 1115 | 1 | 1 | 0.0001556675 |
| Identity | 1115 | 1 | 1 | 0.0001556675 |
| Serial | 1226 | 0 | 0 | 0.0734399214 |

Both serial decoders transcribe **“We are committed to making it secure.”**
Both original positive controls score WER0 through the same final recognizer.
Original-codec reconstruction relative L2 is5.570250055e-7. Serial native PCM
lasts14.05968254 seconds; failed controls last12.77097506 seconds.

## Arithmetic and complete controls

The CPU ggml custom node multiplies actual loaded F32 weights by current native
hidden, rounds each product separately and adds products sequentially in F32.
There is no token-ID or step-specific numerical branch. The ordinary matmul
remains a dependency to provide an exact identity topology; this deliberately
computes both paths and is not an optimization. The measured AVX2/FMA versus
reference serial rounding cause is [documented separately](dia-reduction-cause-2026-10-16.md).

Pristine/OFF/identity complete inputs, logits, all127 captured stages and PCM
are byte-identical. OFF/identity code grids and complete final MT state match.
The pristine baseline reproduces previously accepted complete free generation
and its earlier prefix/stages exactly. Fresh compilation against retained actual
C36 libraries reproduces both saved hidden-state head projections and the
selected serial hidden's actual recorded logits. Seed42 MT state advanced once
per decoder step reproduces each complete final RNG state exactly.

The first changed serial input row is979, following the disputed step978 draw.
All1226 serial decoder input rows match the actual original. All1210 original
code frames match; historical8d0ad886 reversal produces1211 frames. Serial
native codes, inputs and PCM are byte-exact to the earlier forced-decision
restoration. Original DAC PCM is not byte-exact between those jobs; both
separately pass original-source/checkpoint, positive reconstruction and ASR
checks. No cross-job bitwise codec claim is made.

## Proven scope and outstanding validation

Runtime8d0ad886/C36, physically audited381-tensor F32 neural export, F32 cross
caches, portable RNG, source-RoPE off, CPU4, seed42, capacity3072,
temperature1.2/top_p0.95/top_k45/CFG3 stay fixed. Those diagnostic cache/RNG paths
are absent from current main. Current top-k/output-protocol corrections also
postdate this historical runtime. Its one-extra-frame behavior is not copied
back into production.

Current-main graph compilation/replay is independently accepted, but actual
current-main CLI/Session speech, bounded additional prompts/seeds, GPU/Q4 and
performance remain pending. No default, hardcoded decision, release tag or
production quality promotion follows from this single-prompt causal result.

## Complete evidence

Heavy [38870499727](https://github.com/CrispStrobe/CrispASR/actions/runs/38870499727)
and all10 checks in [lint38870502430](https://github.com/CrispStrobe/CrispASR/actions/runs/38870502430)
pass at immutable worker3c2fbb5a9c8c73225677ce6c9eabb6b31b21a92b. Auditor
eadfb145 passes655 payloads, complete source/physical/library/array/ASR checks
and26 rejecting calibration controls. Rebased303a7bb3 preserves all8 method
files byte-exact. Local evidence remains under
${CRISPASR_STORAGE}/crispasr/dia-serial-head-20261016/hosted-38870499727-v1.

Official artifact12035991525 is904974837 bytes, SHA256
`1f7fd84771ce7ce08dcf7f5a3ae8c9fa379f1f4315d4bb4b836f739839916b15`.
[Public raw archive](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/990ff4cf7ee440ecdc239203f93f4093b84c58a8/dia-20261016-serial-head/complete-hosted-serial-head-v1/raw.zip)
and [complete manifest](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/990ff4cf7ee440ecdc239203f93f4093b84c58a8/dia-20261016-serial-head/complete-hosted-serial-head-v1/manifest.json)
include official metadata/logs, independent audit/source, trajectory comparison
and licenses. All22 files pass complete anonymous size/SHA verification at that
immutable HF revision. [Structured results](dia-serial-head-results-2026-10-16.json)
retain every ASR/PCM/library hash and publication ledger. No full neural model
ran on the local VPS.
