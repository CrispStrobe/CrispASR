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

## Validated packages

Both Windows package jobs pass in
[run 37733919241](https://github.com/CrispStrobe/CrispASR/actions/runs/37733919241):
144 CUDA compile commands per arm, correct MMQ definitions, matching CUDA 12.6
runtime API 12060 and staged `--version` / `--diagnostics` startup. The manifests
extracted from the actual uploaded artifacts pass the same pair comparison
locally. The final hosted pairing job `113183674377` also passes, including
comparison of the actual uploaded manifests. The complete workflow is green.

- [MMQ OFF control](https://github.com/CrispStrobe/CrispASR/actions/runs/37733919241/artifacts/11532751916)
- [MMQ ON experiment](https://github.com/CrispStrobe/CrispASR/actions/runs/37733919241/artifacts/11532004516)

GitHub artifacts require a signed-in account and are retained for 30 days.
Each artifact contains a self-contained package ZIP and its manifest. Extract
the two packages into separate directories, retaining their included DLLs.
Use the same already-working model/audio/settings invocation in each package,
on the same selected GPU. Record the complete startup log, transcript, cold
startup time and repeated warm transcription times. Compare both packages;
comparing only with the release package would change the PTX/native-code setup
as well as the MMQ switch.

The exact built source is `04ded753e909da019a2cd391f651f51f5299b6f5`, archived
under `archive/issue483-mmq-source-20261008`; ggml is
`c36dab89b662838f0f5d4826c399198c0b90bbfc`. Source/configuration/runtime hashes:
[receipt](cuda126-mmq-experiment-2026-10-08.json). Actual GPU correctness and
performance remain unmeasured; the issue stays open for the reporter's results.
