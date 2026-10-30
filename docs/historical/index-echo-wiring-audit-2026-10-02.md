# Index-Echo integration audit — 2026-10-02

Both 2B and 9B satisfy the twelve-point maintainer checklist associated with
[contributing.md](../contributing.md). The 9B checkpoint is a variant of the
existing `index-echo` backend, with a projection connector and a larger Qwen3.5
decoder. It adds no backend identifier, native library or public session method.

| Point | Code and evidence | Result |
|---|---|---|
| 1. C runtime | `src/index_echo.{h,cpp}`; metadata selects residual or projection connector and validates dimensions. `INDEX_ECHO_BENCH=1` times pipeline stages. | PASS |
| 2. CLI adapter | `examples/cli/crispasr_backend_index_echo.cpp`; bilingual native timestamps, supported target languages, temperature/seed, glossary/instruction, GPU/thread/flash-attention settings and explicit generation cap. The unset cap preserves 2000 tokens. | PASS |
| 3. Factory and detection | Factory, roster and filename detection in `examples/cli/crispasr_backend.cpp`; `index_echo` in shared `src/core/arch_backend_map.h`. Actual anonymous-filename C ABI opens and decodes successfully on CPU and CUDA. | PASS |
| 4. CLI CMake | `examples/cli/CMakeLists.txt` includes the adapter; actual CLI binaries decode the accepted clips and full-file cases. | PASS |
| 5. Library CMake | `src/CMakeLists.txt` builds and links `index_echo` into `crispasr-lib`. The shared-library symbol audit runs with `--require-lib`; no skipped library gate. | PASS |
| 6. Session ABI | `src/crispasr_c_api.cpp` has include/flag, context, open, transcribe, available-backends and free dispatch. Target language, temperature/seed, max tokens and instruction are forwarded independently of the CLI adapter. Unsupported source-language hints warn on both paths. Existing generic session setters reach all wrappers. | PASS |
| 7. Registry | `src/crispasr_model_registry.cpp` lists 2B Q8, 2B F16 and explicit 9B F16 primary/decoder pairs, plus the Silero VAD companion. The first/default pair remains 2B Q8. Public 9B weight hashes and anonymous downloads match the tested artifacts. | PASS |
| 8. Quantization | The producer invokes `crispasr-quantize` with its existing precision guards and explicit per-tensor overrides for controlled experiments; frontend constants remain F32. 2B F16/Q8 pass. Plain/selective/FFN-only 9B Q8 controls fail complete decoded acceptance and remain private; only 9B F16 is published and registered. No blanket rule is imposed on other Qwen3.5 users. | PASS |
| 9. Reference and diff | `tools/reference_backends/index_echo.py`, `tools/dump_reference.py`, and the `index-echo` dispatch in `examples/cli/crispasr_diff_main.cpp`; dynamic decoder layer count covers all 32 9B blocks. Independent source pins, parameter dtypes, magnitudes, prompt and cached greedy IDs are gated. Fixtures live separately from public model weights. | PASS |
| 10. Binding documentation | Python `Session`, Go session and Flutter docstrings describe bilingual output and companion loading; `python/README.md` includes usage. The 9B variant uses the same generic session ABI, so no new wrapper entry points or struct mirrors are needed. | PASS |
| 11. Go static linkage | `python tools/sync_go_cgo_ldflags.py --check` passes locally and in the hosted built-library wiring audit. | PASS |
| 12. Docs and live tests | README model table, architecture, CLI and generated feature matrix; `tests/test_index_echo_live.cpp`, model-free window/batch/connector and acceptance guards; `tests/env-live-tests.sh` has an overridable primary-model variable. Pinned 2B/9B regression manifest entries, daily protected 2B and separate weekly large 9B regression are enrolled and actually executed. TTS-only tables and speaker/audio-output APIs do not apply to this translation backend. | PASS |

## Executed gates

- [Main full ARM CPU acceptance](https://github.com/CrispStrobe/CrispASR/actions/runs/37016734351): built shared library, generated feature matrix and `check-backend-wiring.py --require-lib` all pass. The audit reports every canonical runtime present and required factory/ABI/capability wiring complete; its unrelated advisory gaps contain no Index-Echo entry.
- [Main selected regressions](https://github.com/CrispStrobe/CrispASR/actions/runs/37016733584): all eleven jobs pass, including protected Index-Echo 2B and six other live backends.
- [Main CI](https://github.com/CrispStrobe/CrispASR/actions/runs/37016734092), [lint](https://github.com/CrispStrobe/CrispASR/actions/runs/37016733637) and [WASM builds](https://github.com/CrispStrobe/CrispASR/actions/runs/37016733853): 13/13, 10/10 and 5/5 pass respectively.
- [Canonical 9B acceptance receipt](index-echo-9b-acceptance-2026-10-02.json): main CPU and two physical T4 GPUs test the same public F16 model pin. Three direct clips cover 75 numerical rows each, exact prompt/cached decoding and actual CLI/session output; five complete file cases cover context/VAD/timestamps; three real Piper round trips have zero word errors on each platform. Retained rejected controls remain visible.
- [Resident GPU profile](index-echo-9b-profile-2026-10-02.json): practical warm speedups of about 1.33× for JFK and 1.36× for Chinese against the original resident BF16 Python blueprint, with precision and device-placement differences recorded.

CPU/CUDA runtime acceptance is established. WASM build success does not establish
9B browser execution, and physical Metal/Vulkan 9B acceptance is not claimed.
The 17.914 GiB F16 pair also exceeds ordinary WASM32 address space.

## Release decision

Index-Echo integration is ready for v0.8.41. At this audit, `VERSION` remains
0.8.40 and the release is claimed in `PLAN.md` by `${CRISPASR_WORKTREES}/wt-483`.
The latest [deterministic Windows Piper live check](https://github.com/CrispStrobe/CrispASR/actions/runs/37022213369)
and [tip lint](https://github.com/CrispStrobe/CrispASR/actions/runs/37022213402)
both finished successfully after the initial audit. The unchanged native source
also passes tip CI (`37022213223`). The separate Ruby binding run
(`37022213108`) remains in progress at this final check. The release owner is
updating its packaging receipt; once its final checks and notes are complete,
use `scripts/bump-version.sh 0.8.41`, push the annotated tag and publish the
prepared release notes. This audit does not create a competing release tag.
