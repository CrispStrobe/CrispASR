# Dia T4 attention capture: independently verified postmortem

Actual Kaggle version 5 remains **ERROR**: all GPU work finished, then JSON
serialization failed on a NumPy uint64 GGUF shape. The original terminal state,
receipt, 232 outputs and full raw/decoded logs remain preserved. Independent
analysis verifies completed computations; it does not rewrite the run green.

The driver repair worked on two assigned T4 devices. Actual driver/plugin
preflight passed. Original audio ASR ran 25 CUDA scheduler splits and recognized
“We are committed to making it secure.” at WER 0. Each of the three native
replays ran 11,037 CUDA splits and recorded 8,736 MiB peak allocation. All
120 normal stage comparisons pass, including magnitude checks. Complete
OFF/ON/OFF logits, feedback and decoded audio match byte-for-byte; exactly 72
ON-only step-978 snapshots exist. Scale/index/history negatives still fail.
Neural/DAC hashes and physical audits match the CPU controls.

The actual GPU attention output versus the actual original attention expression
on identical captured GPU Q/K/V inputs has minimum row cosine
0.9999999999974468, maximum row norm error 7.674099771204368e-7 and maximum
relative L2 4.813245892395258e-7 across all 18 layers. This reference operator
runs pinned Torch 2.6 CPU; captured native outputs come from actual T4. All
repeats and compact/3072-slot cache comparisons are exact. All 18 uniform
2x-scale negatives expose the magnitude error despite cosine remaining one.
The full 72 snapshot metrics, 120 ordinary stage rows, 18 same-input rows and
18 scale controls are in `dia-attention-cuda-postmortem-2026-10-13.json`.

The worker now converts shape dimensions explicitly to Python integers and
writes receipts atomically. Eleven input tests pass, reproducing the original
uint64 failure and checking JSON roundtrip, scale negatives and preservation
of the previous checkpoint on serialization failure. No new GPU inference was
run solely to regenerate metadata. Six runtime files, package/model/reference
bytes and quality gates remain unchanged. Repair source is 36c4bbdff.

This excludes a large self-attention cache/GQA/operator defect at the observed
boundary on CPU and T4. It does not accept native free-running speech, Q4 or
performance, and does not justify the previously rejected rotary default.
Existing encoder-output, cross-K/V and layer-zero cross-attention snapshots
are available for the next same-input investigation. Follow self-output
projection, cross-attention and feed-forward norm/gating/output boundaries
before proposing another full-model candidate. No release/default/model change.

The complete 261-payload archive is public and anonymously hash-verified:
[complete T4 proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/39a4e7652a3007778501bf01fd47e18dd0aba8c7/dia-20261013-attention/complete-t4-attention-v5.zip)
(403,853,797 bytes; SHA256
`1ac6b213bd0d75f89b0a0ffb19fb3cb5609ac2d250697e2c1fcf2129c72e0642`).
Original private logs remain intact. Source and acceptance scopes
must remain explicit when interpreting a failed worker's retained computations.
