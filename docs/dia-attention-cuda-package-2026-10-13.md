# Matching CUDA package for Dia attention capture

Compile-only prerequisite for actual GPU validation. This worktree is based on
capture source `42b111c5e6006e7dcc82afff2b1b43da7c15861c`. Package guards require
exact runtime/dispatcher/sampler/RNG/C ABI/binding bytes from that CPU capture
source; only builder guards, provenance, this procedure and guard tests change.
No neural math, model/source pin, quality threshold or default is changed.

The existing `dia-cuda-speech-validation.yml` builds CUDA 12.4 SM60/75 on GH,
including the shared ABI, CLI, diff, quantizer, server and same-source bindings.
It verifies dependency closure, preserves aliases and keeps the driver outside
the delivered package. Tests reject changed cross-cache precision, missing
sampler bytes and unconditional capture dispatch before compilation.

GH CPU run 38438623888 remains the independent full-model capture-transparency
control. A compiler/package pass does not establish GPU execution or accept
speech/Q4/performance. After delivery, verify the entire archive, every member,
source provenance and closure before any actual Kaggle use. Preserve previous
terminal Kaggle logs/outputs before repushing; never use Kaggle as a CPU build
farm or reinstall its preinstalled Torch. Actual GPU capture must show device
allocation/scheduler work and unchanged complete OFF/ON/OFF trajectories.
