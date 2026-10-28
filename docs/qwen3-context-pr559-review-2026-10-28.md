# Qwen3 context PR #559 — review checkpoint

Reviewed [PR #559](https://github.com/CrispStrobe/CrispASR/pull/559) at head
`6b132720090cbb532f79f544c130381ddf1cf6c9`, against base `24178da9d`.
The description and complete nine-file diff were read. At this checkpoint
there were no issue-style comments, reviews or inline review comments.
The complete #218 and #488 threads were also read, including the maintainer's
shipped fix and reporter follow-ups. No public review, approval or merge was
submitted. Error repairs and publication acceptance retain priority.

The change adds CLI `--context` routing to Qwen3's system message and an opt-in
environment switch for treating Session hotwords as raw context. The original
[Qwen3-ASR prompt builder](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/qwen3_asr.py)
places context in the system message and appends a forced-language assistant
prefill. The proposed routing follows that structure. Inspection found no
change to the default prompt when context is empty, or to the language-prefill
decision that #488 repaired. Mega-ASR and Raon remain excluded intentionally;
the realtime prefix-rollback path is explicitly outside this PR's scope.

No source-level blocker was identified in this bounded review. This is not
runtime acceptance: the contributor reports builds, 26 prompt assertions,
31 caller-routing checks and a three-slice chunk test, but those receipts were
not independently reproduced here. The committed tests exercise the prompt
helper; the reported caller probes are not part of the diff. Fork Actions runs
show `action_required`, which is neither a passing nor a failing code result.

Next steps, after the active error/publication gates:

1. Refresh the PR head, discussion and all review comments before acting.
2. Review the fork workflow diff before authorizing its Actions runs. Require
   CI, lint and secret checks on the exact candidate head or merge tree.
3. Retain executable model-free C API and CLI routing probes for flag unset,
   zero, enabled, changed and cleared; nonempty/empty context; forced/automatic
   language; explicit question; backend exclusions and chunk propagation.
   Test the actual callers, since helper tests alone cannot prove forwarding.
4. Run the existing short-clip Qwen3 regression through the C API and CLI on a
   suitable hosted CPU worker, adding context enabled/cleared controls. Keep
   #488's exact recognition and language-prefill gates. No local model run is
   needed while this host lacks headroom.
5. Decide whether to retain the documented hotwords opt-in API. A dedicated
   context setter is a separate API change requiring all binding/server/WASM
   surfaces; it should not be inferred from this narrowly scoped PR.

Do not claim the author's measurements as independent validation or merge the
feature merely because this inspection found no blocker.
