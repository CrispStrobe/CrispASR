# Prepared Dia CUDA attention observation

Diagnostic worker `tools/ci-heavy/dia_attention_cuda_control.py` and Kaggle
bootstrap `tools/kaggle/dia-attention-cuda-control/` prepare the actual GPU
follow-up to CPU capture 38438623888 and CUDA package 38446095504. Neither
artifact has completed at preparation time; no kernel has been submitted.

The worker requires independently verified capture and package ledgers with
immutable SHA256 pins. It checks the full package payload, embedded provenance
and all six runtime files before model downloads. Those runtime bytes match
both capture source 42b111c5e and package source 2328f15ce. No neural code or
default changes. Eight input guards and eight existing actual-source guards
pass; these are small fixtures/source contracts, not GPU or speech evidence.
The exact check and source ledger is
`dia-attention-cuda-worker-checks-2026-10-13.json`.

On actual SM60/75 hardware, the prepared worker requires original-audio GPU ASR
positive, all 40 normal diff stages, native complete OFF/ON/OFF trajectory
identity and exactly 72 ON-only step-978 snapshots. Every native capture arm
must show CUDA scheduler work and at least the existing 6,000 MiB lower-bound
allocation witness. Scaled-hidden, invalid-index and out-of-history negatives
remain strict. Cosine, row magnitudes, maximum error and relative L2 are
recorded across all 18 layers. Failures preserve logs and a partial receipt.
This observes teacher replay; it does not accept native free speech or Q4.

The bootstrap consumes a SHA256-pinned manifest from the existing public
`cstr/crispasr-regression-fixtures` dataset at an immutable revision. Manifest
entries `capture`, `package` and `package_ledger` each specify `file`, `bytes`
and `sha256`; `capture` additionally specifies its extracted `root`, relative
`ledger` and `ledger_sha256`. A complete audited capture ZIP, delivered runtime
tar and independent package ledger must be published before rendering the
three bootstrap pins: worker source, input revision and manifest SHA256.
Unrendered code fails before hardware checks/downloads. Archives reject parent
and absolute paths, ZIP symlinks and escaping tar aliases.

The existing kernel slug is reused only after its previous terminal logs and
outputs are preserved. No new account, private model/proof repository or
Kaggle tooling dataset. There is no Torch/DAC installation, compiler build or
original CPU generation on Kaggle. The audited DAC converter proof is reused;
F32 coefficients are exported from pinned original weights and must retain
the exact whole-model SHA. `GGML_BACKEND_PATH` points to the CUDA library file,
as required by the audited ggml loader, rather than a directory.

Initial fixture checks exposed CIFS symlink creation failure and a missing
TMPDIR in additional test commands. Tests were rerun with one explicit local
small-fixture environment; no checks were skipped or thresholds changed.
Models and delivered artifacts remain on storage/hosted scratch. The prepared
bootstrap and full worker still require end-to-end hardware validation.
