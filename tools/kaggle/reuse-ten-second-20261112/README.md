This Kaggle research candidate is based on `7b4f0935a`; version 1 completed.
Runtime source is frozen to `a5a46b6c5b616b7556d3bfa27335d33a95c74794`, which
includes the explicit `chunked(..., max_window_samples=80000)` interface.
That source must be remotely available when submitting. Do not use a moving
branch in its place.
The uploaded script hashes itself and records its runtime source SHA, submodule
SHAs, dependency versions, oneMKL library hash, upstream Python source hashes,
input hashes, and converted-model hash in `/kaggle/working/provenance.json`.

The input is the first 160,000 samples of the pinned source's `samples/jfk.wav`,
a speech proxy with seeded Gaussian noise at 5 dB SNR. The research overlap-add
helper processes starts 0, 40,000 and 80,000 with an 80,000-sample window and
40,000-sample hop: three 5 s windows with 2.5 s overlap. This resembles the
upstream window lengths; the helper's Hann weighting is an experimental policy,
not a claim of official stitching or streaming equivalence. No whole-ten-second
model baseline is run, so stitched quality does not measure chunking degradation
against whole-file inference.

Each window captures every original Python reference stage (including frontend
FFT diagnostics), checkpoints the complete GGUF in `/kaggle/working`, and writes
an atomic `reference-manifest.json` containing archive and individual tensor
hashes and shapes **before** native comparison. The original checkpoint and
config hashes remain mandatory. Python model sources are downloaded from
immutable upstream revision `31f82f2efad1095034492f236e8dac9239ba713b`.
`reuse-long-diff`, the bounded five-second entry point using `reuse_diff.h`,
must return zero and emit exactly all 71 unique unconditioned and 71 unique
conditioned PASS rows per window; existing numerical gates are unchanged.
`window-*-diff.log` preserves magnitudes, cosine and relative error. A failed
comparison leaves its full reference, manifest and log available for recovery.

The original model and audited non-fused Mamba adapter also run on CUDA for each
window. This exercises actual GPU work; it is not a native GPU or fused-kernel
qualification. Unsupported Torch GPU draws exit inconclusive before downloads
and builds, without a retry loop. The scalar native comparison builds CPU-only
with the harness toolchain, cache/link flags, two build jobs and heartbeat.
The GPU build flags are deliberately inapplicable to that CPU-only target.
This is an internet-enabled GPU session. It has no CPU/offline fallback: CPU
workers need properly attached source/model datasets and must not clone/pip
online. No new tooling/cache dataset or GitHub CI workflow is introduced.

Outputs include clean-proxy/noisy WAVs; stitched CPU-reference, native and
unfused-GPU-reference WAVs; SNR, SI-SDR and STOI against the clean proxy; native
versus CPU-reference error at stitch transitions; and an artifact hash manifest.
These are objective diagnostics on one noisy public recording, with no
perceptual-quality threshold or production claim. Listen to the saved WAVs and
validate a representative corpus and real product interfaces separately.
The reference and native outputs use the same research helper, not the product
C ABI; the C ABI still rejects this research model.

Only one capture/model is held at a time. A five-second guard checks 10 GiB peak
RSS for the live Linux process tree and completed-process peaks, 8 GiB total output, and a 2 GiB free-space reserve. It records
budget failure before exiting. Shared resident pages count once per process,
which is conservative. Process creation/exits can race the snapshot; this
polling guard is not an OS-enforced memory cap, and a fast allocator spike can
still OOM between polls. GPU memory is outside the RSS accounting.
Reference output is never deleted. Scratch, dependencies and build/cache stay
under `/kaggle/temp`; ccache is capped at 512 MiB. Kaggle output transfer can
return empty placeholders for individual artifacts above 1 GiB: inspect each
manifest and rehash recovered bytes before treating them as preserved evidence.

Model-free local controls: `python tests/test_reuse_ten_second_kernel.py -q`.
The kernel additionally runs existing reference, retention, chunking, quality,
comparison and oneMKL controls on the frozen worker source. No run or model
result is implied by those local tests. The completed run's [receipt](../../../docs/historical/reuse-ten-second-20261112.json)
records all stage comparisons and a significant quality regression on the
latter portion of its single speech proxy. Fused upstream comparison and
broader quality evaluation remain open.
