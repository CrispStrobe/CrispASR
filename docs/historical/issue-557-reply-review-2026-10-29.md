# Issue #557 — final reply for review

Prepared after native audit 40751444564 and wheel audit 40789883589 passed.
The complete issue was refreshed: it is open, with no comments. This draft has
not been posted; the maintainer requested keeping the issue open.

---

Linux GPU wheels are available in v0.8.47:

| Wheel | Platform | Index |
|---|---|---|
| `+vulkan` | Linux x86_64, AMD/Intel GPUs | [Vulkan index](https://crispstrobe.github.io/CrispASR/whl/vulkan/) |
| `+hip` | Linux x86_64, ROCm 6.x | [HIP index](https://crispstrobe.github.io/CrispASR/whl/hip/) |
| `+cuda` | Linux aarch64 SBSA servers, including Grace/GH200; excludes Jetson | [CUDA index](https://crispstrobe.github.io/CrispASR/whl/cuda/) |

The host supplies the GPU driver and matching vendor runtime: the Vulkan
loader; ROCm HIP/hipBLAS/rocBLAS and ROCm's `libomp.so`; or CUDA 12 runtime and
cuBLAS. The Linux wheels use `manylinux_2_35` tags.

Packaging, dependency, notice and full anonymous download checks passed. The
Linux Vulkan wheel also loads in a clean environment. Physical AMD, Intel and
ARM NVIDIA GPU execution remains unverified, so reports from your hardware
with `crispasr.Session` and Parakeet ja would be welcome. We are keeping this
issue open for those reports.

---

Evidence: [publication receipt](release-publication-2026-10-28.md),
[native audit](release-v0847-native-audit-2026-10-28.json),
[wheel audit](release-v0847-wheel-audit-2026-10-29.json).
