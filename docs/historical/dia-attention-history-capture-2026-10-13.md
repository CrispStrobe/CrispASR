# Dia late self-attention capture control

Source lives on diagnostic branch `diag/dia-attention-capture-20261013`, exact
commit `42b111c5e6006e7dcc82afff2b1b43da7c15861c`; the worker/runtime changes are
not integrated on production main. [Hosted run](https://github.com/CrispStrobe/CrispASR/actions/runs/38438623888)
is queued. [Submission ledger](dia-attention-history-submission-2026-10-13.json).

Diagnostic source is based on 2954c5a7c, preserving full F32 neural weights,
F32 cross cache and the default geometric rotary path. No production default,
model/source pin, seed, filter or quality threshold changes.

Original `DIA_CAPTURE_ATTENTION_LAYERS=0,1,...` observes the actual pinned
self-attention modules at `DIA_CAPTURE_STEP`. It copies post-rotation queries,
complete updated K/V history and the input to the output projection. It does
not replace the original attention function. Stage arrays use `(batch, time,
heads, head_dim)` and preserve `[unconditional, conditional]` order. Invalid or
duplicate layer indices reject before generation; malformed environment syntax
rejects before model loading. Every hook is removed on success and failure.

Native `CRISPASR_DIA_DUMP_SELF_ATTN=1` with an existing dump directory and exact
capture step marks and saves matching Q/K/V-history and attention tensors for
all layers. The diff dispatcher captures this explicit request before its
environment isolation guard and restores both prefixed and legacy parent values.
Default diagnostic output and neural graph math stay unchanged when the flag
is absent. Marking outputs can perturb graph allocation, so complete trajectory
OFF/ON/OFF equality is mandatory before interpreting snapshots.

`tools/ci-heavy/dia_attention_history_control.py` runs on GH heavy-cpu:

1. Fetch the immutable canonical original archive and verify its whole hash.
2. Verify the pinned original checkout, run tiny actual-source hook/device/index
   contracts, then run original capture OFF/ON/OFF at step 978. Require canonical
   input/code grids and byte-identical common arrays across the three runs.
3. Export and physically audit unchanged full F32 neural/DAC tensors, build the
   actual native CPU library/diff and require the original audio ASR positive.
4. Run native teacher replay OFF/ON/OFF against the same original reference.
   Require 40/40 existing stages, exactly 72 additional files only in the ON arm
   and byte-identical complete feedback/logits/PCM across all three arms.
5. Record all 72 attention/history cosine, norm, absolute and relative errors
   with actual shapes and file hashes. Retain negative capture-index/scaled-hidden
   controls, all logs, arrays and sources, including any failing job.

These observations stop before self-output projection and do not independently
validate full native free speech, Q4, CUDA, performance or a release. Hosted
results are pending until the job produces and verifies its evidence.
