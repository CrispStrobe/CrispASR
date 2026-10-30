# Dia cache precision and sampling isolation — 2026-10-16

Candidate [5e04955e](https://github.com/CrispStrobe/CrispASR/commit/5e04955e580b798b3cd45d870d407a3dd7600c4e)
is a diagnostic experiment awaiting full-model acceptance. It composes the
pending reference/diff wiring with the accepted generic serial head and adds
two opt-in controls. Defaults, corrected top-k indices and stopping/crop rules
remain unchanged. No reference decisions are injected into speech generation.

`CRISPASR_DIA_CROSS_CACHE_F32=1` preserves projected cross K/V as F32, matching
storage, graph-input types and upload byte counts. Default remains F16.
`CRISPASR_DIA_PORTABLE_RNG=1` uses the pinned PyTorch2.6 MKL CPU exponential
sampling recipe with one MT word for the complete heads × vocabulary batch.
Default remains std::discrete_distribution. Only an exact `1` enables either
control; explicit `0` preserves defaults. Neither claims arbitrary PyTorch/GPU
RNG compatibility. Scalar/MKL logarithms can differ in final floating bits.

The actual original sampler passes all11034 decisions from1226 original
canonical logit batches against the production portable function. Tiny tests
also compare full noise arrays and original MT state advancement, including
masked vocabulary entries, dominant tokens, top-k/nucleus and greedy bypass.
On original and both native initial logit sets, standard sampling chooses568
and portable/Python530 for head zero, with all9 decisions agreeing by recipe.
Thus the first feedback difference is sampling-driven even with original logits.
Later neural arithmetic and speech quality must still be tested independently.

Fresh compilation at this exact candidate passes2388 complete synthetic C++
boundary comparisons across capture/head/cache/sampler controls. Original IDs,
OFF/ON/OFF/identity/explicit-zero logits remain exact. With identical forced
inputs, changing RNG does not change neural logits at either cache precision.
These are neural diagnostic tests; the synthetic fixture has no codec and
establishes no speech acceptance. [Compact local receipts](dia-cache-sampler-local-2026-10-16.json).

[Heavy38892690744](https://github.com/CrispStrobe/CrispASR/actions/runs/38892690744),
[general CI38892693594](https://github.com/CrispStrobe/CrispASR/actions/runs/38892693594)
and [lint38892696276](https://github.com/CrispStrobe/CrispASR/actions/runs/38892696276)
are queued. The worker holds serial-head arithmetic fixed and runs all four
F16/F32 × standard/portable combinations, plus pristine/OFF/identity controls.
Each factorial arm requires actual CLI/Session agreement, code-to-PCM replay,
native/original DAC decoding and ASR; the original is the positive control.
All results are saved before stage and WER-zero assertions. Original-prefix
counts distinguish the first sampling difference from later numerical drift.

Single collectors and independent auditors retain official raw ZIP digests,
all source/physical pins and complete arrays under
${CRISPASR_STORAGE}/crispasr/dia-cache-sampler-20261016. Audit code is prepared, not yet
accepted against future artifacts. Broader prompts/seeds, real GPU/Q4, profiling
and default promotion remain blocked on unforced speech acceptance. Per the
sibling kaggle_usage.md this CPU recipe belongs on GitHub, with Kaggle reserved
for actual supported CUDA work.
