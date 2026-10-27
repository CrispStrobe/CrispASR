# Original CUDA 12.6 MMQ pair preserved — 2026-10-26

The original OFF/ON packages from [run 37733919241](https://github.com/CrispStrobe/CrispASR/actions/runs/37733919241)
are preserved in maintainer draft **424870665**, tag
`archive-issue483-mmq-20261008`, targeted at original source
`04ded753e909da019a2cd391f651f51f5299b6f5`. The draft remains unpublished.
The original artifacts expire November 7; the draft assets have no Actions
artifact expiry. Source remains at `archive/issue483-mmq-source-20261008`.

[Preservation run 40414804633](https://github.com/CrispStrobe/CrispASR/actions/runs/40414804633)
passes all eight archive/draft/resource controls, original artifact size/hash
checks, every inner-member hash/CRC, pinned manifests, three matching CUDA DLL
hashes, and authenticated full SHA-256 readback of both uploaded assets. The
archives and all binaries are unchanged. Each new wrapper adds complete NVIDIA
CUDA 12.6.3 agreement/supplement/attachments, original project notices, provenance
and inventories. The full agreement carries the cuBLAS third-party notices
missing from the original project notice; application MIT does not relicense
NVIDIA DLLs. Retain original and additive notices when extracting binaries.

| Mode | Preserved wrapper bytes | Wrapper SHA-256 |
|---|---:|---|
| OFF | 520722110 | `c3fd68c6c9d6702208da5f998d58d6e8b823da85208b4852c159b250d76efcfc` |
| ON | 520761342 | `d6f6e957a29d0a525d4b33e24a64ee50e4663e1ffeef03c74ec3542d3daac4d0` |

The [receipt](cuda126-mmq-preservation-2026-10-26.json) links the byte-exact small
worker proof, full original/package member inventories and the independent
sidecar audit. The latter reads only 203,920 bytes through explicit HTTP 206
ranges, verifies every notice/inventory against its expected bytes and hashes,
and never reads the nested native ZIP payloads. Three invalid-receipt controls
are rejected before payload access. Both project binaries and the three NVIDIA
runtime DLLs are accounted for; no driver or SDK installer is included.

The first actual Mac attempt stopped before transfers because startup load
exceeded the existing limit. The corrected worker waits at most five minutes
without changing its 2 GiB available-memory, load below twice CPU count, or
8 GiB scratch gates. The successful job records all 24 samples and begins work
only after load falls to 5.18 on three CPUs. No local large download, native
build, SDK installation or GPU execution ran for this preservation.

This is preservation and notice verification, not GTX1660/MX150 acceptance.
The user confirms no device access yet. Keep #483 open for actual correctness
and cold-PTX versus repeated-warm measurements on each target; no performance
benefit or production-default change is claimed.
