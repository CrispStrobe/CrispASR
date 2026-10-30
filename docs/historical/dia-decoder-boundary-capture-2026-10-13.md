# Dia late decoder boundary capture

The diagnostic source adds opt-in observation of 16 block-zero boundaries
at the fixed original feedback step 978. It follows the pinned original
DecoderLayer, Attention and MlpBlock calls; no inference expression is replaced.

The stages are input, pre-SA norm, SA pre/output projection, SA residual,
pre-cross norm, rotated cross-Q, cross pre/output projection, cross residual,
pre-MLP norm, gate, up, actual SiLU-times-up hidden, MLP output and final residual.
Python reads the actual fused wi_fused output and actual wo input. Native
graphs retain existing tensor pointers and mark outputs only; tensor names
and arithmetic expressions remain unchanged. Retained pointers handle the
final residual also being named by the existing layer-output capture.

Python uses `DIA_CAPTURE_BOUNDARY_LAYERS=0`; native uses
`CRISPASR_DIA_DUMP_DECODER_BOUNDARIES=1` plus the existing selected-step dump
directory. The diff replay preserves/restores both modern and legacy flags
and forwards only an explicit modern request. Both gates default off.

The hosted worker `tools/ci-heavy/dia_decoder_boundary_control.py` must prove:

- Actual original tiny-model hooks preserve all common arrays, observe the
  requested call and fused MLP values, own snapshots and clean up after errors.
- Original full-model OFF/ON/OFF all common arrays match exactly, and all three
  original input/code histories match the canonical reference.
- Native OFF/ON/OFF logits, feedback and decoded PCM match exactly, with
  40 ordinary stage passes per arm and exactly 16 ON-only boundary dumps.
- Existing doubled-hidden and malformed/out-of-history step negatives reject.
- Every boundary records cosine, magnitude, relative L2, shape and hash;
  double-scale controls preserve cosine while exposing norm/L2 errors.

Original source `4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8`, model revision
`257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd`, Torch 2.6 CPU, seed 42,
capacity 3072, temperature 1.2, top-p .95, top-k 45 and CFG 3 remain fixed.
The same whole F32 neural export and critical-F32 DAC hashes are required.
Atomic JSON receipts and always-uploaded CI artifacts preserve partial results.

Local checks: three native parser/scratch/environment guards, two C++ syntax
checks and Python compilation pass. Local original-model tests were interrupted
by unavailable CIFS reference libraries and are not claimed green. They are
mandatory in the hosted worker before model inference. Source-only temporary
worktrees on ${CRISPASR_WORKTREES} use about 230 MiB plus ggml; all weights, builds and
full captures stay on storage/CI. The shared main checkout remains untouched.

Full hosted controls have passed and are independently audited; see
`dia-decoder-boundary-results-2026-10-14.md`. This is observation code, not a numerical
repair or acceptance of native free-running speech, Q4, performance or release.
Only interpret boundary metrics after capture transparency passes.

Submitted source `8d0ad8862d2cad5d305a48a57f2a99b5024bc39a` succeeded in
[GH heavy replay 38472367538](https://github.com/CrispStrobe/CrispASR/actions/runs/38472367538).
Exact source-file hashes, local checks, inputs and required controls are in
`dia-decoder-boundary-submission-2026-10-13.json`.
