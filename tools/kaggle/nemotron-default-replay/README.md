# Nemotron current-default replay

This script uses the published v0.8.45 CUDA CLI/library archive for inference.
It verifies the complete release member inventory and host CUDA loader before
loading either model. It builds nothing and installs no Torch or CUDA runtime.

Do not dispatch while `SOURCE` or `CPU_SHA256` is `PENDING`. First review the
complete CPU run 40377479500 and commit its unmodified `replay.json` at the path
specified by `CPU_RECEIPT`. Both model profiles must complete and the pinned F16
control must pass before CUDA diagnosis. Pin the receipt hash and the immutable
worker commit, then follow the sibling Kaggle operational guide and normal
account/slot/privacy preflight. Never use a second account or retry for a
preferred GPU allocation.

The replay retains all 26 original-model controls, fresh and reused C ABI
sessions, opaque-filename architecture detection, a CLI control and complete
native stage arrays. Original ASR transcripts are the oracle; the four
ambiguous OmniVoice inputs are not scored against their TTS input prompts.
Stage cosine and magnitude gates remain fixed. CUDA evidence requires actual
backend selection, the worker's own model allocation and process-specific SM
activity during the reused decode sequence, after model initialization.
This does not prove that every graph operator executes on CUDA, isolated MMQ
behavior, or performance on other hardware.

A complete diagnosis can reject the Q4 default. Read `current_default_accepted`
and each profile's `accepted`; successful script termination alone does not
approve Q4. This worker never changes defaults or publishes model weights.
Preserve terminal outputs and logs before any corrected rerun.
