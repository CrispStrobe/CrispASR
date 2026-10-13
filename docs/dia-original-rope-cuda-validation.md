# Original rotary buffer on actual CUDA hardware

This isolated diagnostic composes CPU source `ead5a5d5d` with the validated
explicit CUDA request from `35945f527`. Neural/cache/sampler code and the
original-source export helper remain byte-identical to the CPU candidate.
The diff combines actual device selection, bounded late capture and stdout
flushing; source-compiled contracts cover each boundary.

The completed CPU rotary control remains rejected: all 120 replay stages pass,
but all six speech checks fail empty/WER 1. ON changes logits without changing
complete input histories or PCM. It shares 979 original input steps and retains
the step-978/head-2 draw difference. GPU results must be assessed separately.

Kaggle uses the original Torch2.6-generated timescale buffer from the immutable
public primitive proof, never a constructor from its installed Torch version.
Whole ZIP, every payload, source/head/dtype/size/hash and original provenance
are checked before use. Its bytes and ordered metadata match the actual CPU
constructor. The full exported model must match CPU SHA256
`9a2c0df72586b3f325bbbf42473e2156140870f48ce57474f2c577943008e9b1`;
381 learned tensors, the auxiliary table and all 134 codec tensors are audited.
The model is 6,444,669,600 bytes, +608 over the unchanged all-F32 export; the
actual table payload is 256 bytes. No model/default promotion follows.

GPU runs five version-independent physical export contracts. All seven original
constructor/export contracts and ten actual-source capture contracts pass on
pinned Torch2.6 CPU. No Torch reinstall or new private model/proof repo is used.
The existing private compiler cache is separate from public model/proof assets.

Run the fixed actual-GPU worker with `--cross-cache-control
--source-rope-control --cuda-control --prebuilt <build>`. Legacy/source/legacy
repeat all use F32 cross caches, the same immutable model/source/reference,
seed 42, top-p .95, top-k 45, temperature 1.2 and CFG 3. Native diff, C ABI and
CLI must show actual CUDA splits and at least 6,000 MiB sampled process GPU
allocation. The unchanged canonical original audio must also pass actual
CUDA ASR as a positive control before evaluating native speech. Preserve strict stage metrics, full inputs/logits/PCM, exact OFF
repeat, legacy negative speech controls and every failure. No quality-gate
waiver, seed selection, profiling or broad acceptance is included.

Before replacing the prior Kaggle version, its complete terminal logs/output
were saved and all files verified; its public proof is anonymously hash checked.
Deploy only committed/pushed immutable source through the existing push helper.
