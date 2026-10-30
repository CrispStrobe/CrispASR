# Dia complete native-code reconstruction and codec probe

The final emitted native row is now recoverable from retained raw sampler
predictions and the actual producer stopping/BOS logic. Complete CLI/C-ABI
code grids agree. Original-DAC decoding/ASR and independent audit are now
complete locally. Native speech still fails. Initial hosted dependency failure
is preserved. Hosted retry38723274160 now passes, with independent full
artifact/PCM/ASR audit. The diagnostic worker is integrated as7a9345b4.

## Complete reconstruction

The source from native run38527734866, `eb94d724`, is verified against its
immutable Git object. The literal stopping/reversal functions and literal
BOS/output-update blocks compile in a small standalone replay adapter. Every
one of1115 captured feedback rows matches; the actual first EOS is step1100,
with the final stop at1115. The resulting1100 aligned frames predict563200
PCM samples, exactly matching the retained float C-ABI and CLI sample counts.
The CLI/C-ABI aligned grids are byte-identical. A deliberately changed raw
prediction rejects at the next observed feedback row. No stopping row is
invented, and no neural model is rerun.

Selected raw inputs, sampler records, PCM, logs, native source and original
reference are rehashed against their previously accepted delivery ledgers.
The99,803,232-byte original GGUF is rehashed in full. Its original1210-frame
code grid and619520-sample PCM are retained as positive controls; the latter
is byte-identical to the separately delivered original waveform. Pinned
original source files match their recorded hashes at `4a9e29b1`.

## Actual original return-slice comparison

On Torch2.6.0+cpu, the literal original `_prepare_audio_prompt` and `generate`
methods run with the actual original `DecoderOutput` class. Neural prediction
is replaced with the retained native sampler rows; every original feedback
input matches native exactly. This tests the generation protocol, not neural
math or a new original-model generation.

| Same retained predictions | Native | Original return slice |
| --- | ---: | ---: |
| Feedback steps consumed | 1115 | 1115 |
| Returned delayed rows | 1115 | 1114 |
| Aligned code frames | 1100 | 1099 |
| Expected PCM samples (512/frame) | 563200 | 562688 |

All1114 shared emitted rows match exactly. Native retains one extra final row;
original `generated_tokens[prefill_step:dec_step+1]` excludes the newly written
stopping row. This is a measured11.61ms tail mismatch. A repair must preserve
forced-input/output diagnostics and cover natural EOS, capacity and short runs;
it is a separate runtime task. Existing free-speech failure remains unaccepted.

## Hosted original DAC/ASR test

Candidate `7b42071f41e5814f957d7397d8c73615b2cd7c9a` adds a diagnostic worker
without production changes. [Hosted run38710473906](https://github.com/CrispStrobe/CrispASR/actions/runs/38710473906)
recompiles/replays the native adapter, verifies original sources and replays the
original protocol. It downloads the hash-pinned actual DAC0.0.1 checkpoint,
checks the three actual DAC neural source files against the retained originals,
and decodes native complete codes, the original return slice and original codes.

The original positive PCM requires cosine≥0.999999, relative L2≤1e-4 and norm
error≤1e-4. The same original Whisper tiny recipe checks every decoded waveform;
the original-code positive control must retain zero WER. Complete arrays,
float WAVs, ASR inputs/results, source, binaries and receipts survive failures.
No generation seed, top-p/top-k, model/default, GPU or timing promotion follows.
The local metric controls reject scale/order/shape/nonfinite/zero-norm errors.

## Complete original-DAC diagnosis

The unchanged worker at7b42071f completed on the local VPS with isolated
Torch2.6.0+cpu, DAC1.0.0, audiotools0.7.2 and Whisper20250625. All models,
installed reference dependencies and outputs are under ${CRISPASR_STORAGE}. This is a
local CPU result; hosted numerical/reference success is not claimed.

| Codes decoded by actual original DAC | Samples | RMS | ASR WER |
| --- | ---: | ---: | ---: |
| Original positive | 619520 | 0.0735338684 | 0 |
| Native complete | 563200 | 0.0001561470 | 1 |
| Native with original return slice | 562688 | 0.0001576506 | 1 |

Original positive PCM matches the retained original waveform at cosine
0.9999999999998493, relative L2 5.572414e-7 and norm error9.646689e-8, passing
all unchanged strict gates. Its ASR is exactly "We are committed to making it
secure." Both native-code variants transcribe as "You", WER1. The original DAC
does not restore speech from these native codes. Native-complete PCM through
the original versus native DAC has cosine0.999964985, relative L2 0.008929458
and norm error0.003080364; native-code RMS is0.00212347 times the positive RMS.
This localizes the missing speech to generated code content in this fixture,
without claiming that every codec implementation or precision choice is exact.

An independent auditor rehashes all77 raw worker payloads, both checkpoint
files and the input bundle; recomputes PCM/magnitude metrics and WER; verifies
float WAV/array equality and actual ffmpeg resampling against all retained ASR
inputs; checks source identity, recipe and raw segments.
The first965 aligned code frames match original exactly (11.203628s), and
99.9996567% of original waveform energy follows that prefix. A matched mostly
silent prefix therefore does not establish successful spoken output.

The repaired production protocol is also compiled/replayed on both real retained
CLI/C-ABI sampler captures. All1115 feedback rows,1114 returned rows and1099
aligned codec frames match actual original methods exactly. This establishes
protocol correctness on those captures, without a new neural generation or
speech acceptance. The scoped runtime protocol repair is now integrated after
all13 general CI and10 lint jobs; its tested source bytes remain unchanged.

[Complete117-payload diagnosis and retained-protocol proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/668196e7450b1ce35bded6355ec58464235deb1e/dia-20261015-full-codec/complete-original-dac-diagnosis-v1.zip)
is23,699,070 bytes, SHA256
`9ff7dda447d1d8cbd75dcbbbb2316db70d6ad84ba9ad0cc12fed88a74db2862c`,
fully anonymously downloaded and rehashed. It includes original/native complete
PCM and WAVs, ASR inputs/results, source, raw logs, method/environment setup,
independent audit, actual protocol executable and all retained arrays. The
[pinned upstream Dia license](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/949b03fd276ecf968ba589e336c2788f80a4e445/dia-20261015-full-codec/Dia-LICENSE)
accompanies the proof alongside the included DAC/CrispASR notices.

## Hosted dependency failure and corrected dispatch

Initial hosted38710473906 passes installation, source pinning and native replay,
then fails importing the original model before reference inference because its
package list omitted huggingface_hub. This is separate from the complete local
numerical diagnosis. The raw10,351,853-byte artifact matches GitHub SHA256
`a13b5dd6a3c78472f8d23c82917ac128775188651feff2d2d3cdee42031f7709`;
its CRC/safe unique members and the full job log are verified and preserved.
[Complete failed-attempt proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/f972cead2a3695c421e14cef9032292d92c3c2ea/dia-20261015-full-codec/complete-hosted-dependency-failure-v1.zip)
is10,326,196 bytes, SHA256
`2683740004fa6c32093f4f4edd6b9642c094cc9b46744cf91063d332737706bc`,
fully anonymously downloaded/rehashed.

[Corrected hosted dispatch38716091962](https://github.com/CrispStrobe/CrispASR/actions/runs/38716091962)
adds huggingface_hub1.10.1, matching the successful local environment. Worker
source/math remain unchanged at7b42071f; a separate README command correction
is pushed as87a11e3d. That rerun passed strict positive PCM (relative L2
5.571705e-7) and produced all three complete decoded waveforms, then failed
before ASR because ffmpeg was absent. Independent partial PCM/WAV/array audit
passes; hosted ASR did not run. Its raw artifact22,828,110 bytes matches SHA256
`e74d6a025bf0a89fcdcf4e1bfe1544fc9443ae87f538a77a70bca61657fa6bb0`.
[Complete failed ffmpeg-attempt proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/0c3be77b4ed30830bce1f949d7f71dc014e0b2cf/dia-20261015-full-codec/complete-hosted-ffmpeg-failure-v1.zip)
is22,807,538 bytes, SHA256
`2e5afdfbe19754de03ac18b7e4e0791bb74fe8c594c61b0f1ba21cc0ded87761`,
anonymously downloaded/rehashed with the raw log, artifact and partial audit.

Prerequisite-only sourcea40755b7 now checks ffmpeg before expensive decoding
and explicitly installs it only on GitHub-hosted Linux if missing.
[Fresh hosted38723274160](https://github.com/CrispStrobe/CrispASR/actions/runs/38723274160)
is submitted with the same pinned packages, actual model/source checks and
unchanged strict PCM/ASR gates. It has no hosted success claim yet.

## Public preparation proof

[Complete protocol/preparation proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/ea07987236fd47882f62c348dad2f80d764229b1/dia-20261015-full-codec/complete-protocol-preparation-v1.zip):14 outer payloads,5,261,390 bytes,
SHA256 `63baa2f12bc1da0775e511d274fbadbd0d2aa0cf059ed2299dcd0817e3bb3b9d`, fully anonymously downloaded/rehashed. It includes the complete
42-payload [input bundle](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/e22cb7451fa0b073e3b497273266338913243831/dia-20261015-full-codec/retained-inputs-v1.zip), actual original-method replay source/results,
all native reconstruction inputs/adapter/source, positive waveform/code controls,
parent hashes and notices. The input bundle is5,220,208 bytes, SHA256
`b711e33d4bb60f6c421397a3ca2eaf8600facd196a23c03e2bce20bafbee6126`.
[Machine receipt](dia-full-codec-preparation-2026-10-15.json).

## Hosted completion and tool integration

[Hosted38723274160](https://github.com/CrispStrobe/CrispASR/actions/runs/38723274160)
at a40755b7 succeeds after the ffmpeg prerequisite repair. Raw artifact11967984300
is23318376 bytes, SHA256
`6601d0b809eaa022e11fe1ddff74dc285b38a593bac1477393cdedc60c96ba06`;
full digest, CRCs, unique safe paths and80 payload hashes pass. Independent audit
recomputes every PCM/WAV equality, ffmpeg ASR input, WER/segments, strict original
PCM gate and native/original waveform metrics. Original positive relative L2 is
5.5702501e-7, WER0; both native variants still have WER1. The diagnosis is complete,
and native speech remains rejected.

The [complete hosted proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/6ac1bbd6072c031d546a07354c9b79c832406162/dia-20261015-full-codec/complete-hosted-codec-diagnosis-v1.zip) is
23296826 bytes, SHA256
`3a01150147bfad341d8383600dcd5a68379d4d100851b85ae85c67150ef604d9`,
anonymously full-SHA verified. It includes the complete raw artifact, API/logs,
auditor, full result and notices. Both earlier dependency failures remain public.

Rebased/ff integration7a9345b442708f4dbe84784a536736c7a674cc50 preserves the
tested worker bytes. Its five scale/order/shape/nonfinite/zero-norm rejecting
controls pass after rebase. This integrates diagnostic tooling and documentation;
production protocol repair and full speech acceptance are separate.
