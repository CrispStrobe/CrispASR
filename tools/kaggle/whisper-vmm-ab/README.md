# Whisper VMM A/B diagnostic (prepared, not submitted)

This adapter uses one pinned CrispASR source, one pinned `ggml-base.en.bin`, and
the public JFK clip. It builds two CUDA variants on a Tesla T4. The only requested
CMake difference is `GGML_CUDA_NO_VMM=ON` versus `OFF` with an explicit link to
the verified host CUDA driver. Each variant runs profiling off and on with the
same binary, in alternating order, after a warmup. Every run must have the same
27 decoder calls, zero WER, identical transcript bytes, CUDA model placement,
CUDA execution, and a sampled GPU process PID. The receipt is diagnostic; it
does not claim general performance or recognition quality.

The earlier packaged-fork comparison showed VMM enabled in the fork and
`NO_VMM=1` in the source-built CrispASR. Its build log reports that `libcuda`
could not be found at link time. The builds also differed in other ways, so
that comparison cannot establish a VMM effect. ggml allocates Whisper model
weights with `cudaMalloc`; VMM is used for its CUDA scratch pool. This A/B tests
the actual effect on the whole process and measured phases without assuming it
will close the loader gap.

The adapter resolves `libcuda.so.1` from the running host driver, calls `cuInit`
and `cuDeviceGetCount`, checks its ELF SONAME, and passes that exact host path to
CMake. It rejects a toolkit stub as the runtime driver. It checks the effective
CUDA compile definition, the CUDA library's `DT_NEEDED` entry and `ldd`
resolution, plus the inference log's VMM state. It does not copy the driver or
any toolkit stub into the output. Logs and a machine-readable receipt remain
under `/kaggle/working`; model and builds stay under `/kaggle/temp`.

A statically linked VMM-on CUDA build may fail to launch on a host without an
NVIDIA driver. A driverless-host launch and CPU-fallback check is a separate
shipping gate; passing this GPU A/B does not authorize changing release defaults.
