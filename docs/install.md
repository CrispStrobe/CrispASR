# Install & build

This page covers the full build matrix. For a quick sanity build, see
the **Quick install** section in the [README](../README.md).

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
