# Original MMQ experiment preservation

The manual `preserve-mmq-483.yml` workflow archives the original OFF/ON artifacts
from run 37733919241 before their November 7, 2026 expiry. Large downloads,
archive inspection and full release-asset readback run on Actions. It installs
no SDK and builds or executes no native program.

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
