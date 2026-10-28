# OpenVoice clip conditioning and shared-noise proof

Prepared follow-up to `openvoice-bake`; the original submission was rejected
because both batch-GPU slots were occupied. Those October 19 queue notes are
historical; check current capacity before submitting. No accepted conditioning
run is recorded. The recovered bake v2 failed CPU backend registration; it
provides no model acceptance. Save the previous
kernel's terminal log/output before replacing or superseding a live job, and
respect the two-GPU-session cap. The bootstrap pins the shared worker by commit
and SHA256; it enables conditioning mode before loading it. Only the bootstrap
is uploaded by Kaggle; the shared worker is fetched by immutable URL.

Native source6be9fda36 supplies the exact original posterior-noise replay and
explicit equal-weight clip averaging APIs. Three cases include tau0 and0.3,
27/28 strict stages, uninstrumented C API/cache output, clip-mean embedding and
waveform controls, independent ASR/speaker similarity and retained model/input
hashes. No ggml/CUDA/full-SDK build or Torch reinstall. Actual VAD selection and
mixed-rate resampler equivalence remain separate scope.

See [OpenVoice baking](../../../docs/openvoice2-baking.md) for gates, limitations
and source identities. A successful submission/build is not model acceptance.

The resource-guarded worker version is `2026-10-28.openvoice-conditioning.3`.
It records available memory/load, scratch and GPU headroom before compilation
and inference, and stops on insufficient resources. The bootstrap pins the
exact worker source/hash; its model-free contract rejects stale delivery pins.
The native model source and 27/28 stage, waveform and quality gates are unchanged.

Kernel version 1 used worker `.2`: delta builds/conversion passed on two T4s,
then Hub repository creation returned 403 before inference. It is not model
acceptance. Worker `.3` retains model/reference/notice checkpoints locally by
default. Publication requires `CRISPASR_OV2_PUBLISH=1` plus suitable Hub rights,
and its completion is recorded separately from numerical acceptance. Preserve
the previous terminal log/output before submitting the software repair.
