# v0.8.47 — Linux GPU Python wheels (#557)

This release adds Linux GPU builds of the `crispasr` Python wheel. Native
inference code is unchanged from `v0.8.46`.

## New wheels

| Wheel | Index | Host must provide |
|---|---|---|
| Linux x86_64 `+vulkan` | `https://crispstrobe.github.io/CrispASR/whl/vulkan/` | Vulkan loader (`libvulkan.so.1`) and a Vulkan GPU driver |
| Linux x86_64 `+hip` | `https://crispstrobe.github.io/CrispASR/whl/hip/` | ROCm 6.x: HIP runtime, hipBLAS/rocBLAS and ROCm's `libomp`, on the loader path (e.g. `/opt/rocm/lib`, `/opt/rocm/lib/llvm/lib`) |
| Linux aarch64 `+cuda` (SBSA server ARM, e.g. Grace/GH200) | `https://crispstrobe.github.io/CrispASR/whl/cuda/` | NVIDIA driver, CUDA 12 runtime and cuBLAS. Jetson/Tegra (L4T) is not covered |

```bash
pip install crispasr --extra-index-url https://crispstrobe.github.io/CrispASR/whl/vulkan/
pip install crispasr --extra-index-url https://crispstrobe.github.io/CrispASR/whl/hip/
```

GPU wheels never ship a GPU driver or vendor runtime.

## New native bundles

- `libcrispasr-linux-x86_64-vulkan.tar.gz`: relocatable shared libraries and
  headers. The existing Vulkan CLI archive is static and stays unchanged.
- `libcrispasr-linux-arm64-cuda.tar.gz`: shared libraries for SBSA ARM with
  CUDA architectures 8.0–12.0.

## Validation scope

Every new bundle passes the relocated-load and dependency-closure gates. Every
new wheel passes notice, provenance and manylinux_2_35 symbol-floor checks. The
Linux Vulkan wheel is load-tested in a clean environment against a software
Vulkan driver. That proves packaging, not GPU acceleration. No new wheel has yet
been run on a physical AMD, Intel or ARM NVIDIA GPU. The `+hip` wheel cannot be
load-tested on hosted runners, which have no ROCm. Reports from real hardware
are welcome on #557.
