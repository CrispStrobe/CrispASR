# v0.8.53 — ROCm 7 HIP tarball

## Packaging

- New `crispasr-linux-x86_64-hip7.tar.gz` for AMD GPUs on **ROCm 7.x** (Arch and other rolling distros). ROCm 7 renamed its library sonames (`libhipblas.so.2` → `.so.3`), so the ROCm 6 HIP tarball could not load its GPU module there and ran on the CPU. The new tarball is built against ROCm 7.2 and, like the other HIP tarball, starts and falls back to CPU when ROCm is absent. Not yet run on ROCm 7 hardware; reports welcome (#586).
