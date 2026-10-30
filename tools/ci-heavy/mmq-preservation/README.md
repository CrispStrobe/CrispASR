# Original MMQ experiment preservation

The manual `preserve-mmq-483.yml` workflow archives the original OFF/ON artifacts
from run 37733919241 before their November 7, 2026 expiry. Large downloads,
archive inspection and full release-asset readback run on Actions. It installs
no SDK and does not build or execute the archived native payloads.

Each preservation ZIP retains the original GitHub artifact ZIP byte-for-byte,
plus its complete member inventory, original project notices and additive CUDA
terms/origin. NVIDIA's full CUDA 12.6.3 agreement, supplement and attachments are
retained as text with source hashes in `nvidia-origin.json`. Attachment B
contains cuBLAS third-party notices missing from the historical project notice.
The full agreement copy is a way to carry the complete applicable attachments;
the main agreement does not state a universal full-EULA-copy requirement.
NVIDIA runtime DLLs retain their vendor terms, separately from application MIT.

The release stays a maintainer draft, targeted at original source `04ded753e`.
No publication operation exists in the workflow. This preserves the pair
durably while keeping public promotion separate from archive recovery and
actual hardware acceptance. Receipts record both original and preservation ZIP
hashes, CUDA DLL hashes, matched manifests and authenticated full asset readback.
The small Actions artifact contains audit data/notices only, never the binaries.

Actual GTX1660 and MX150 runs remain required. Use OFF and ON from this same
pair on the same selected GPU, explicit identical model/audio files and settings,
and retain startup/device/runtime logs, model/audio hashes, transcripts and
correctness checks. Record cold PTX-JIT startup separately from multiple warm
runs. Comparing only with a production release also changes architecture setup;
T4 execution does not establish these devices' performance. Release defaults are
unchanged.

## Verified preservation and independent sidecar audit

[Completed preservation and proof](../../../docs/historical/cuda126-mmq-preservation-2026-10-26.md).
The worker checks resource headroom before transfers and waits at most five
minutes for startup load/memory pressure to settle, retaining every sample.
The thresholds stay unchanged; insufficient disk stops immediately.

The small worker artifact is retained byte-for-byte with scoped Git attributes
that prevent line-ending normalization of the proof inputs, so the
independent audit does not depend on Actions log/artifact retention. With
`requests` installed and `gh` authenticated for access to the maintainer draft,
run from the repository root:

```sh
python tools/ci-heavy/mmq-preservation/verify_sidecars.py \
  docs/cuda126-mmq-preservation-2026-10-26 mmq-sidecar-check.json .
```

The auditor bounds each request to 1 MiB and each asset's total reads to 4 MiB,
requires explicit HTTP 206 responses, and checks only ZIP metadata and small
notice/inventory members. Full responses are closed before reading their body.
Original native payload integrity comes from the worker's full artifact hash
and authenticated full draft-asset readback; it is not recomputed locally.

When updating the draft description, include its intended `tag_name` explicitly
and verify tag/source/draft flags afterward. A description-only API update was
observed to replace a not-yet-created tag with an `untagged-*` name; the intended
archive tag was restored and verified before the final audit. No publication is
part of this workflow.
