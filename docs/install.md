# Install & build

This page covers the full build matrix. For a quick sanity build, see
the **Quick install** section in the [README](../README.md).

## Prebuilt Linux tarballs — which one to download (#355)

The GitHub releases carry several Linux x86-64 tarballs.

| tarball | GPU acceleration | falls back to CPU? |
|---|---|---|
| `crispasr-linux-x86_64.tar.gz` | none (CPU only) | n/a — it *is* the CPU build |
| `crispasr-linux-x86_64-avx512.tar.gz` | none (AVX-512 CPU) | n/a |
| `crispasr-linux-x86_64-cuda.tar.gz` | NVIDIA GPU (CUDA 12) | **yes** — runs on CPU if CUDA libs are absent |
| `crispasr-linux-x86_64-cuda13.tar.gz` | NVIDIA GPU (CUDA 13) | **yes** — runs on CPU if CUDA libs are absent |
| `crispasr-linux-x86_64-hip.tar.gz` | AMD GPU (ROCm) | **no** |
| `crispasr-linux-x86_64-vulkan.tar.gz` | Vulkan GPU | **no** |

Since v0.8.30 the CUDA tarballs use dynamic backend loading
(`GGML_BACKEND_DL`): the CUDA backend is a separate shared library
(`ggml-cuda.so`) that the main binary loads via `dlopen` at runtime. If the
NVIDIA driver (`libcuda.so.1`) or CUDA runtime (`libcudart.so.12`) are absent,
the CUDA backend simply does not load and the CPU backend is used instead —
the "auto-select the best backend, fall back to CPU" behavior works as
advertised. No wrapper script is needed.

For GPU acceleration, the NVIDIA driver and CUDA toolkit must be installed on
the host (or injected into the container via `--gpus` / `runtime: nvidia`).

The HIP and Vulkan tarballs still require their respective runtimes at
link time (no CPU fallback). Use the plain CPU tarball if you need a
GPU-less fallback for those.

Verify what a given tarball actually requires before deploying it:

```bash
readelf -d crispasr | grep NEEDED
```

> **Can the GPU build degrade gracefully instead?** Yes, if you build it
> yourself: `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON` makes each backend a
> `dlopen`-ed module, so a missing CUDA driver leaves the CUDA backend
> unregistered instead of killing the process. This works and is verified on
> Metal and CPU — see [the #355 section](#graceful-degradation-via-ggml_backend_dl-355)
> for the measured transcripts and the two throughput caveats. The prebuilt
> tarballs below are still statically linked, because flipping the release leg
> needs an A/B on real CUDA, HIP and Vulkan hardware first.

## Windows CPU: which zip? (#380)

`crispasr-windows-x86_64-cpu.zip` targets an **AVX2 + FMA** baseline (Intel
Haswell 2013+ / AMD Excavator 2015+). On an older CPU — e.g. Sandy/Ivy Bridge,
which have AVX but not AVX2 — the first compute instruction raises an
illegal-instruction fault that the Windows console swallows: the program
prints its banner and exits with no output and no error (issue #380). Since
that report the CLI checks at startup and prints an explicit error instead
(`CRISPASR_IGNORE_CPU_ISA=1` overrides).

> Hitting that "banner, then nothing" symptom on a machine that *does* have
> AVX2? It is still a crash, and the exit code says which kind — see
> [troubleshooting.md](troubleshooting.md#it-printed-the-banner-then-nothing-happened).

For pre-AVX2 CPUs use **`crispasr-windows-x86_64-cpu-legacy.zip`** — a generic
x86-64/SSE2-floor build (`CRISPASR_PORTABLE_CPU=ON`) that runs on anything
from Westmere up, just slower per core. Run the AVX2 build once: its startup
check tells you exactly which required feature is missing. The CUDA and Vulkan
Windows archives use the same AVX2 + FMA CPU baseline; the legacy archive is
CPU-only.

## Windows CUDA: split downloads (#342)

The Windows CUDA packages bundle three NVIDIA runtime DLLs — `cudart64_*.dll`,
`cublas64_*.dll` and `cublasLt64_*.dll` — so they work on a host that never
added the CUDA `bin` directory to `PATH`. They also dominate the download, and
they change far less often than we cut releases. Re-fetching them for every
version is wasted bandwidth, which is painful if github.com is slow for you.

So each release ships both forms:

| asset | contains |
|---|---|
| `crispasr-windows-x86_64-cuda.zip` | CLI, self-contained |
| `crispasr-windows-x86_64-cuda-non-cuda.zip` | CLI, **without** the three DLLs |
| `libcrispasr-windows-x86_64-cuda.tar.gz` | shared libs + headers, self-contained |
| `libcrispasr-windows-x86_64-cuda-non-cuda.tar.gz` | shared libs + headers, **without** the three DLLs |
| `cudart64_*.dll`, `cublas64_*.dll`, `cublasLt64_*.dll` | the three DLLs, on their own |
| `crispasr-windows-x86_64-cuda-runtime-sha256.txt` | SHA-256 of each of the three |

To upgrade without re-downloading the runtime, unpack the `-non-cuda` archive
into a **new empty directory**. After checking the SHA-256 manifest, copy only
the three named NVIDIA runtime DLLs you already have next to `crispasr.exe`
(for the libs package, into `bin\`). Do not copy an old `ggml-cpu.dll`,
`ggml-cuda.dll`, `crispasr.dll`, or executable into the new directory.

The three DLLs are published **once** per release and shared by both packages —
sound only because every CUDA-bundling job pins the same toolkit (12.8.0).
`tools/check-cuda-split-packaging.py` enforces that, along with the rule that
any job bundling them must also ship a split archive and attach it to the
release. Check the SHA-256 manifest before reusing DLLs from an older download;
a CUDA version bump changes the filenames, which is your signal to re-fetch.

## Prerequisites

- C++17 compiler (GCC 10+, Clang 12+, MSVC 19.30+)
- CMake 3.14+
- `curl` or `wget` on `$PATH` if you want to use `-m auto` auto-download

Optional:
- `libavformat` / `libavcodec` / `libavutil` / `libswresample` for
  Opus / M4A / WebM ingestion (`-DCRISPASR_FFMPEG=ON`).
- `libopenblas` / MKL / Accelerate — speeds up CPU-side matmuls for
  Conformer-based encoders (parakeet, canary, cohere, granite,
  fastconformer-ctc). The ggml CPU backend picks BLAS up automatically
  when present at build time; no CrispASR flag is needed.
- CUDA / Metal / Vulkan / MUSA / SYCL toolchains for GPU acceleration —
  enabled via ggml's standard flags (`-DGGML_CUDA=ON`,
  `-DGGML_METAL=ON`, `-DGGML_VULKAN=ON`, `-DGGML_MUSA=ON`,
  `-DGGML_SYCL=ON`). On CUDA, set
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` at runtime to allow swapping
  to system RAM when VRAM is exhausted.
- `sherpa-onnx` binaries on `$PATH` if you want
  `--diarize-method sherpa` with ONNX models.

No Python, PyTorch, or pip is required at runtime.

## Linux / macOS

```bash
git clone --recursive https://github.com/CrispStrobe/CrispASR
cd CrispASR
# if you already cloned without --recursive:
#   git submodule update --init --recursive

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

> The bundled `ggml/` submodule is required. A non-recursive clone leaves it
> empty; CMake now stops early with a message telling you to run
> `git submodule update --init --recursive`.

The default build produces every CLI target. Binaries land in
`build/bin/`:

| Binary | Purpose |
|---|---|
| `crispasr` | Main CLI (transcribe / TTS / server) |
| `crispasr-quantize` | Re-quantize any GGUF model — see [quantize.md](quantize.md) |
| `crispasr-diff` | Per-stage cosine-similarity diff vs Python reference |

To build only the library (faster CI builds), pass
`--target crispasr-lib`:

```bash
cmake --build build -j$(nproc) --target crispasr-lib
```

### CMake presets

The repo ships a `CMakePresets.json` with sensible defaults
(Release + tests off + ccache friction off):

```bash
cmake --preset default      # Release
cmake --preset debug        # Debug
cmake --preset linux        # Release + OpenMP
cmake --build build -j$(nproc)
```

### Convenience build script

`scripts/dev-build.sh` wraps the configure + build with platform-aware
defaults (Ninja, ccache, OpenMP, mold linker on Linux):

```bash
scripts/dev-build.sh                                  # default target
scripts/dev-build.sh --target crispasr-quantize       # build a different target
scripts/dev-build.sh --reconfigure -DGGML_VULKAN=ON   # extra cmake args
```

## Windows (convenience scripts)

Two batch scripts handle the Windows build without requiring a
pre-opened Developer Command Prompt. They use `vswhere.exe` to locate
Visual Studio 2022 automatically, call `vcvars64.bat`, then drive
CMake + Ninja.

### `build-windows.bat` — CPU build

```cmd
build-windows.bat
```

Produces `build\bin\crispasr.exe`. Extra CMake flags can be appended:

```cmd
build-windows.bat -DCRISPASR_CURL=ON   :: enable libcurl fallback
build-windows.bat -DGGML_CUDA=ON       :: NVIDIA GPU (CUDA must be installed)
build-windows.bat -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60  :: Tesla P100
```

Tesla P100 is a Pascal GPU with compute capability 6.0 (`sm_60`). Use a
CUDA 12.x toolkit when building for it; CUDA 13 no longer generates code
for Pascal GPUs.

What it does:
1. Locates `vswhere.exe` under `%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\`.
2. Finds the latest VS 2022 installation that includes the VC++ toolchain.
3. Calls `vcvars64.bat` to initialize the 64-bit MSVC environment.
4. Runs `cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release [extra flags]`.
5. Builds the `crispasr` target → `build\bin\crispasr.exe`.

### `build-vulkan.bat` — Vulkan GPU build

```cmd
build-vulkan.bat
```

Produces `build-vulkan\bin\crispasr.exe` with the Vulkan compute backend
enabled. In addition to the VS detection above, it:

1. Checks `%VULKAN_SDK%`. If unset, scans `C:\VulkanSDK\` for the
   newest installed version and sets `VULKAN_SDK` accordingly.
2. Adds `-DGGML_VULKAN=ON -DGGML_CUDA=OFF` so CUDA is not accidentally
   pulled in if the CUDA toolkit is also installed.
3. Writes the build into a separate `build-vulkan\` directory so it
   coexists with a CPU build.

```cmd
:: Typical usage — VULKAN_SDK is picked up automatically
build-vulkan.bat

:: Override Vulkan SDK location explicitly
set VULKAN_SDK=C:\VulkanSDK\1.4.304.1
build-vulkan.bat

:: Run on Vulkan, pinned to GPU 1 (NVIDIA on a hybrid laptop)
build-vulkan\bin\crispasr.exe --gpu-backend vulkan -dev 1 -m model.gguf -f audio.wav
```

Important:
- `build-windows.bat -DGGML_CUDA=ON` produces a CUDA build, **not** a
  Vulkan build.
- `--gpu-backend vulkan` only works if the binary was actually built
  with Vulkan support.
- On hybrid laptops, Vulkan device `0` may be the integrated GPU. Use
  `-dev N` to pin the discrete GPU if needed.

Both scripts exit with a non-zero code and a `[ERROR]` message if any
step fails (VS not found, CMake configure error, build error).

### Consuming `libcrispasr` from a language binding (Rust / Go / …)

These scripts build the **CLI**, but adding `-DBUILD_SHARED_LIBS=ON` also
produces `libcrispasr` (`build\src\crispasr.lib` + the DLL). Three ways to link
it from a binding:

- **This local build** — `build-windows.bat -DBUILD_SHARED_LIBS=ON [-DGGML_CUDA=ON]`,
  then point the binding's `CRISPASR_SYS_LIB_DIR` at the `build\` dir (its
  `build.rs` now finds the single-config `build\src\crispasr.lib`); put the
  directory holding `crispasr.dll` on `PATH`.
- **Prebuilt bundle** — download `libcrispasr-windows-x86_64[-cuda].tar.gz` from
  [Releases](https://github.com/CrispStrobe/CrispASR/releases), extract, set
  `CRISPASR_SYS_LIB_DIR` to the bundle root + put its `bin\` on `PATH`.
- **git dependency** — the binding's `build.rs` runs cmake for you (needs VS 2022
  + CMake; honours the `cuda`/`vulkan` features).

Exact per-OS environment is in the Rust
[`crispasr-sys` README](../crispasr-sys/README.md#prebuilt-release-bundle-no-build).

## GPU backends

CrispASR builds against ggml's GPU backends. Pick the one matching
your hardware at configure time:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON     # NVIDIA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON    # Apple Silicon
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON   # cross-vendor
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_MUSA=ON     # Moore Threads
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON     # Intel oneAPI
```

You can compile multiple backends into one binary; ggml will pick the
highest-priority compiled backend at runtime
(CUDA > Metal > Vulkan > MUSA > SYCL > CPU). Force a specific backend
with `--gpu-backend <name>`, and pin a device with `-dev N`:

```bash
crispasr --gpu-backend vulkan -dev 1 -m model.gguf -f audio.wav
crispasr --gpu-backend cpu -m model.gguf -f audio.wav        # benchmarking
```

## Opus / AAC support (default, no ffmpeg, no libopus)

`.opus` (Ogg/Opus, **including WebM/Matroska Opus**) and raw ADTS `.aac`
(AAC-LC) decode natively through the in-tree clean-room
[glint](https://github.com/CrispStrobe/glint) decoder — **no ffmpeg and no
libopus needed** — via the library `crispasr_audio_load` API used by the
bindings. glint is RFC-conformant for Opus (all 12 RFC 6716/8251 vectors), so
this works on every platform including WASM, out of the box. A build with
`-DCRISPASR_OPUS=OFF` (no libopus linked at all) still decodes `.opus` and
WebM/Opus.

libopus + opusfile remains available as an optional Opus fallback, selected with
`CRISPASR_OPUS_DECODER=libopus`. It's on by default (`CRISPASR_OPUS`) when the
system `opusfile` is found via pkg-config (e.g. `apt install libopusfile-dev`,
`brew install opusfile`), but is no longer required for any input format. On
platforms without system libs (Windows / iOS / Android / WASM), build it
statically:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS_FETCH=ON
```

Container AAC (`.m4a` / `.alac` / `.caf`) decodes natively on **Apple**
(macOS/iOS) via AudioToolbox, or via libfdk-aac (`dlopen`) on Linux/Windows —
also no ffmpeg. See [cli.md](cli.md#audio-formats) for the full format matrix.

## AMR-NB / AMR-WB support (telephony + voicemail recordings)

`.amr` (AMR-NB 8 kHz) and AMR-WB 16 kHz decode via
[opencore-amr](https://github.com/CrispStrobe/opencore-amr) (Apache-2.0) — the
standard codecs for mobile voice recordings and voicemail. On by default
(`CRISPASR_AMR`) when the system libraries are found via pkg-config
(`apt install libopencore-amrnb-dev libopencore-amrwb-dev`,
`brew install opencore-amr`). Without them, AMR is simply skipped and the build
succeeds — the CMake status line tells you which path was taken.

To build the decoder statically instead (no system packages — Windows, Android /
Termux, iOS, WASM):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCRISPASR_AMR_FETCH=ON
```

> **`register` build errors on clang ≥ 16 (fixed in v0.8.24).** opencore-amr is
> 2000s-era code that declares locals `register`, a storage class **C++17
> removed**. On a toolchain whose default is C++17 — clang 16 and newer,
> including Termux's — every one of those is a hard error:
> `error: ISO C++17 does not allow 'register' storage class specifier
> [-Wregister]` (issue #314). Older defaults only warn, which is why this
> surfaced as a sudden break rather than a long-standing one; it is
> toolchain-dependent, not architecture-dependent.
>
> v0.8.24 scopes `-Wno-register` (and clang's `-Wno-deprecated-register`) to the
> two vendored codec targets, so no flag of your own is needed. On an older
> CrispASR the workaround is `export CXXFLAGS="$CXXFLAGS -Wno-register"` —
> effective, but it silences the same mistake in *your* code too, so prefer
> upgrading.

## ffmpeg ingestion (container AAC/M4A off-Apple, WMA, …) — optional fallback

For formats with no permissive native decoder (container `.m4a` without
libfdk-aac, WMA, exotic containers, …), build with the optional ffmpeg fallback:

```bash
# Install ffmpeg dev libs first:
#   apt install libavformat-dev libavcodec-dev libavutil-dev libswresample-dev

cmake -B build-ffmpeg -DCMAKE_BUILD_TYPE=Release -DCRISPASR_FFMPEG=ON
cmake --build build-ffmpeg -j$(nproc) --target crispasr-lib
```

> **Upstream bug warning.** `.m4a` / `.mp4` / `.webm` containers
> currently crash CrispASR's ffmpeg integration. For those formats,
> pre-convert to WAV (or, on Apple, `.m4a`/AAC work natively without ffmpeg):
> ```bash
> ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le -y /tmp/audio.wav
> ```

## Older glibc systems

The pre-built binaries on some HuggingFace model cards (e.g.
`bin/cohere-quantize`) were built against glibc 2.38 and fail on
Ubuntu 22.04 (glibc 2.35) with:

```
./bin/cohere-quantize: /lib/x86_64-linux-gnu/libc.so.6:
    version 'GLIBC_2.38' not found
```

The fix is to build from source — CrispASR has no glibc minimum
version of its own, so it builds cleanly against whatever glibc your
distro ships.

```bash
git clone --recursive https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ls build/bin/crispasr-quantize
```

## Android / Termux

CrispASR builds natively under [Termux](https://termux.dev) on aarch64
Android devices. Use a **static build** to avoid linker conflicts with
system-installed `libggml.so` from the `whisper-cli` package (#137):

```bash
pkg install build-essential cmake git
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DCRISPASR_BUILD_TESTS=OFF
cmake --build build -j$(nproc)
```

**Why static?** Termux's `whisper-cli` package installs an older
`libggml.so` into `$PREFIX/lib`. The dynamic linker finds it before the
locally built version, causing `cannot locate symbol` errors at runtime.
Static linking (`-DBUILD_SHARED_LIBS=OFF`) embeds all ggml code directly
into the binary, eliminating the conflict entirely.

Strip debug symbols to reduce binary size:

```bash
strip build/bin/crispasr*
```

**Termux clang is new enough to default to C++17**, which matters if you enable
the statically-built AMR decoder (`-DCRISPASR_AMR_FETCH=ON`): the vendored
opencore-amr sources use the `register` storage class that C++17 removed. Fixed
in v0.8.24 — see [AMR-NB / AMR-WB support](#amr-nb--amr-wb-support-telephony--voicemail-recordings)
if you are building an older tag (#314).

### Cross-compiling for Android (NDK)

To cross-compile from a Linux or macOS host for Android deployment
(e.g. embedding `libcrispasr.so` in an Android app), use the provided
`build-android.sh` script. This requires the
[Android NDK](https://developer.android.com/ndk) installed on the host:

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk
./build-android.sh                      # all ABIs (arm64-v8a, armeabi-v7a, x86_64)
./build-android.sh --abi arm64-v8a      # single ABI
./build-android.sh --vulkan             # with Vulkan GPU support
```

Output lands in `build-android/<ABI>/src/libcrispasr.so`.

**This is not the same as building inside Termux.** The NDK
cross-compiler produces binaries linked against Android's bionic libc,
suitable for embedding in Android apps via JNI. Termux uses its own
linker and packages — use the native Termux build above instead.

### Graceful degradation via `GGML_BACKEND_DL` (#355)

`-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON` now **configures, links and runs**.
It is opt-in: the default build is unchanged, and the release tarballs still ship
statically linked (see "flipping the release leg" below).

Three things blocked it, all now resolved:

1. **CMake** — every per-model library linked `ggml-cuda` / `ggml-metal`
   explicitly (125 sites), and under DL those are `MODULE` targets that cannot
   be linked. They now route through `crispasr_link_ggml_*` interface targets
   that are empty in a DL build and the real backend otherwise. ✅
2. **CPU-backend symbols** — under DL even the CPU backend is a module, so
   `ggml_backend_cpu_init`, `ggml_backend_is_cpu`, `ggml_backend_cpu_set_n_threads`,
   `ggml_backend_cpu_buffer_type`, `ggml_backend_cpu_reg`, `ggml_backend_is_metal`
   and `ggml_backend_cpu_set_threadpool` are not linkable. All 426 call sites go
   through `src/core/ggml_cpu_backend.h`, whose non-DL branch is the identical
   direct call. ✅
3. **Direct CPU graph execution** — `ggml_graph_compute`,
   `ggml_graph_compute_with_ctx`, `ggml_graph_plan`, `ggml_threadpool_new` /
   `_free` and `ggml_get_type_traits_cpu`, across ~23 sites. These now go
   through the same header, which under DL routes them to a `thread_local` CPU
   backend obtained from the registry (`thread_local` because a ggml backend is
   not safe for concurrent use and several of these sites run on worker
   threads). ✅

`tests/test-cpu-backend-shim.cpp` is the equivalence gate: the same 7 cases
compile unchanged in both modes and assert arithmetic, not just return codes.
Both report *27 assertions in 7 test cases*.

Measured on an M1 with `parakeet-tdt-0.6b-v3-q4_k` + Silero VAD over a 300 s
FLEURS clip, all four runs producing 608 words:

| build | device | transcript |
| --- | --- | --- |
| default | Metal | **X** |
| DL, `libggml-metal.so` present | Metal | **X** — byte-identical |
| default, `-ng` | CPU | **Y** |
| DL, `libggml-metal.so` removed | CPU | **Y** — byte-identical |

X and Y differ in two tokens. That is the ordinary CPU-vs-Metal reduction-order
divergence — it is present in the default build too, which is exactly what the
`-ng` row is there to show. DL itself changes nothing.

Removing the module is the #355 scenario in miniature: the process does not die,
it logs `load_backend: loaded CPU backend` and transcribes.

**Two limitations keep this opt-in rather than default.** ggml's CPU registry
does not expose the threadpool setter through `get_proc_address` (only
`set_n_threads` — see `ggml_backend_cpu_get_proc_address`), so a DL build cannot
install the shared worker pool and falls back to ggml's own per-call threading.
For the same reason `ggml_graph_plan` has no DL equivalent, so the hot paths
that size a work buffer once and reuse it across frames re-plan per call
instead. Neither is a correctness issue and neither has been measured; both are
throughput risks on the per-frame VAD and wav2vec2 paths.

`tests/test_ggml_audio_ops_*` are not built under DL. They call
`ggml_backend_metal_init()` by symbol on purpose — naming one exact backend with
no scheduler fallback is the point of those tests — and routing them through the
registry would pick "best available" instead.

Flipping the `-cuda` release leg to `BUILD_SHARED_LIBS=ON` still needs an A/B on
real CUDA, HIP and Vulkan hardware; only Metal and CPU are verified above.
