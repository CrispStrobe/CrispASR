# VoxCPM2 current Vulkan profile

Pinned source/model files; actual NVIDIA Vulkan, ten steps and seed 2 throughout.
Compare Q8 and published Q8/F16-LocDiT cohorts, short/long text, first-context
setup versus repeated seeded synthesis (two warmups, six measured calls).
VAE phase logs distinguish weight reconstruction, permutation/upload, allocation
and compute. All seeded repeats must produce identical PCM within a cohort.
Actual CUDA ASR must return each input sentence exactly, with no quality-gate
relaxation. VAE CPU fallback fails the profile. NVIDIA hardware results cannot
certify Intel B390 speed or listening quality. No runtime/default/model change.
Archive terminal logs/output before repush; refresh exported real build cache.

A separate process captures `GGML_VK_PERF_LOGGER` operation timings after the
accepted timing loop, so instrumentation does not contaminate those samples.
The capture must contain LocDiT matmul and VAE sine-operation evidence.
