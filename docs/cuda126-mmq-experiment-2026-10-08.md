# CUDA 12.6 MMQ experiment for #483

The reporter confirmed that the matched CUDA 12.6 package removes the runtime
mismatch warning, and requested a forced-MMQ build for GTX1660 comparison.

`Windows CUDA Smoke` has an opt-in `package_mmq_pair` dispatch input. It builds
two self-contained CLI/quantizer packages with the same source, CUDA 12.6.3,
PTX `61-virtual;80-virtual`, and AVX2/FMA/F16C CPU floor. The two arms differ
only in `GGML_CUDA_FORCE_MMQ=OFF/ON`. They are workflow artifacts, and the
workflow does not publish a release or change release defaults.

Each arm verifies actual CUDA compiler definitions, CMake settings, the staged
runtime API version, and packaged CLI startup with toolkit/build paths removed
from PATH. It includes a configuration/source manifest and runtime DLL hashes.
A final pairing job requires identical source, ggml pin, CPU floor, architecture
list and three runtime DLL hashes between arms.

The OFF arm is the control for this PTX experiment; it is not a stock release
build with native Turing cubins. Hosted runners do not have the reporter's
GTX1660/MX150. Compilation and driverless startup cannot establish GPU output
correctness or speed. The reporter must compare both arms on the same GPU,
model, quant, audio and settings, retain transcripts, and separate cold PTX JIT
startup from warm inference. No performance benefit is claimed before that
hardware evidence exists.

Hosted packaging validation is pending. Download/test instructions and exact
artifact links will be supplied after the packages pass their checks.
