# Nemotron default qualification, 2026-10-26

The managed Nemotron default is pinned F16, 1,286,368,480 bytes (about 1.3 GB).
The former default Q4 file fails the fixed original-model controls on both CPU
and a real T4. Managed quant overrides accept only F16. Explicit existing Q4
files remain usable for diagnostics; a missing Q4 filename never silently
resolves to F16.

| Fixed controls | F16 CPU | F16 T4 | Former default Q4 CPU | Former default Q4 T4 |
| --- | --- | --- | --- | --- |
| Matches original ASR transcript words | 26/26 | 26/26 | 11/26 | 11/26 |
| Matches VoxCPM prompt words | 22/22 | 22/22 | 8/22 | 8/22 |
| Fresh/reused transcript changes | 0 | 0 | 0 | 0 |
| All three stage and magnitude gates | pass | pass | fail | fail |
| CLI original-transcript control | pass | pass | fail | fail |
| Opaque-filename C ABI detection | pass | pass | pass | pass |

All 26 saved PCMs and their hashes are retained in the pinned original-reference
archive. The oracle is NVIDIA's actual F32 ASR transcript, not the TTS input
prompt: four OmniVoice recordings have substitutions that the original ASR
also hears. No cases were excluded. Batch decoding uses `en-US`, context preset
0 (left 56/right 3), the existing Session attention default, and reversed reused
sessions. Native diagnostic stages retain their existing attention default.
Cosine minimum 0.999 and relative L2 maximum 0.01 remain fixed. Independent
audits recompute transcript scores, state changes, complete arrays and gates,
and reject three corrupt receipts per device.

The first divergent Q4 stage is pre-encode. F16 T4 encoder cosine minimum is
0.99999763 with relative L2 0.00028004; Q4 is 0.96630924 with relative L2
0.07358670. A successful script exit means diagnosis completed; it does not
approve Q4. CUDA evidence includes exact released runtime hashes, host-library
loading, own-process allocation and positive SM samples during reused decoding
after model initialization (13 F16 samples, 12 Q4). This does not establish that
every graph operator runs on CUDA, performance, streaming parity, or accuracy
across the model's full language roster.

The accepted F16 file has SHA-256
`14cbc98df495d8ab452c6a4ac367136d94ab5da55e73afaa96ba4f749dd95d09`
at conversion revision `bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2`.
The rejected former default is the 408,557,984-byte file from the separate
legacy repository, SHA-256
`22480dd283a3bf1b39e9d2ab86c048b0c1654b2fc280e46627a4cbfded4b46fd`.
It is not the distinct historical 479 MB Q4 snapshot or the earlier 1.18 GB hybrid.
Neither is promoted by this change.

The original and conversion model cards specify OpenMDW-1.1. Managed downloads
include the [full agreement](../nemotron-model-notices/LICENSE-OpenMDW-1.1.txt) and
[pinned origin notice](../nemotron-model-notices/NOTICE.txt). These retain the
published terms without certifying upstream training-data or third-party rights.

The source default takes effect in newly built binaries. Existing v0.8.45
installations can use the [pinned F16 file](https://huggingface.co/cstr/nemotron-3.5-asr-streaming-GGUF/resolve/bbd95a9ca5fa0dfca3312a122dfc45a2b578b9c2/nemotron-3.5-asr-streaming-0.6b-f16.gguf)
explicitly after downloading it to `models/`:

```sh
crispasr --backend nemotron -m models/nemotron-3.5-asr-streaming-0.6b-f16.gguf -f audio.wav -l en-US
```

Evidence: [CPU run 40377479500](https://github.com/CrispStrobe/CrispASR/actions/runs/40377479500),
[CPU receipt](nemotron-default-replay-2026-10-26-cpu.json),
[CPU independent audit](nemotron-default-replay-2026-10-26-cpu-audit.json),
[T4 receipt](nemotron-default-replay-2026-10-26-cuda.json),
[T4 independent audit](nemotron-default-replay-2026-10-26-cuda-audit.json).
CUDA worker source is `cc191e718595aa4be90d62fcd66b66360d44e426`, version 2,
entry SHA-256 `8e46703e834cdda8768ed0724e9136915b49f46e26f310907e38bdf4e3c79399`.
The tested native source is unchanged v0.8.45
(`e58bb48b52cf1fe957c19586297c83a2ed0d9fe4`). The source fix changes registry
selection and notices, not model tensors or inference kernels.

Two harness corrections are explicit. The first CPU run accidentally uploaded
opaque model symlinks; bounded recovery preserves every non-model member and
later workers keep such links in scratch. CUDA version 1 fully passes F16 but
stops before Q4 inference because its arbitrary 512 MiB allocation floor exceeds
the compact model's 492 MiB allocation. The
[partial version 1 receipt](nemotron-default-replay-2026-10-26-cuda-v1.json)
is retained. Version 2 uses a model-size allocation bound and completes both
profiles with the original numerical gates.

The [permanent 79-file proof](https://huggingface.co/cstr/crispasr-regression-fixtures/resolve/5fccbdd17f5640385576d7d5a8b06d41c3f5db08/tts-asr/nemotron-default-replay-20261026/proof.zip)
retains numerical members byte-for-byte, tested source recipes, original
lineage, licenses and the independent auditor. Anonymous full readback verifies
its SHA-256 `ff2f7c5f1280462d2c64a435a76b447d603c0e663851df2d596ee100222f5aff`
and all 79 members. Only the private account identifier in two console logs is
redacted; the manifest records this. Full CPU recovery on Actions verifies the
original archive SHA and matches every non-model member to independent range
reads. The accidental 1.59 GB duplicate is removed after durable preservation.
[Publication receipt](nemotron-default-replay-2026-10-26-proof.json),
[recovery receipt](nemotron-default-replay-2026-10-26-recovery.json), and
[cleanup verification](nemotron-default-replay-2026-10-26-artifact-cleanup.json).

The [corrected converted-model card](https://huggingface.co/cstr/nemotron-3.5-asr-streaming-GGUF/blob/f300f4f36b388c5f22f810c4e8503359fb9c1ce7/README.md)
removes the unsupported blanket Q4 recommendation, fixes its broken license
link and includes the full agreement and origin notice. Both model file hashes
and byte sizes remain unchanged; the source default retains its tested F16 pin.
[Anonymous card/notice/weight-identity verification](nemotron-default-replay-2026-10-26-model-card.json).
