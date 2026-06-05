**Title:** `ci/run.sh: add GG_BUILD_NO_AVX512 knob to pin an AVX2 baseline on heterogeneous GitHub-hosted x86_64 runners`

---

## Background

`ci/run.sh` is the canonical entry point for ggml / whisper.cpp / llama.cpp CI on GitHub-hosted runners (and the recommended way to reproduce CI failures locally). It already exposes `GG_BUILD_*` knobs for build-time CPU feature gating:

- `GG_BUILD_LOW_PERF=1` — limit memory + thread fan-out for small runners
- `GG_BUILD_NO_SVE=1` — disable ARM SVE (drops to `armv8.5-a+fp16+i8mm`)
- `GG_BUILD_NO_BF16=1` — disable bf16
- `GG_BUILD_EXTRA_TESTS_0=1` — extra test slice

For x86_64 there is no symmetric knob. The default behaviour is `GGML_NATIVE=ON`, which picks up whatever CPU features the build host advertises. On GitHub-hosted Linux runners this is fragile because the `ubuntu-22.04` pool is heterogeneous: some VMs have AVX512 / AVX512_VBMI / AVX512_VNNI, some don't. The runner that runs the build step may have those features; the runner that runs the bench step (same workflow, same `runs-on`, but a different physical VM under Azure's allocator) may not. Result: build succeeds, bench SIGILLs (exit 132) on its first AVX512 instruction.

## Repro (observed in CrispASR's downstream sync of `build.yml`)

`.github/workflows/build.yml` job `ggml-ci-x64-cpu-high-perf` on a recent push:

```
+ ./build-ci-release/bin/crispasr-bench -m .../ggml-tiny.en.bin -t 4 -nfa
…
system_info: AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1
…
+ res=1   # exit 132 = 128 + SIGILL
+ echo 'Benchmark failed for model: tiny.en'
```

The build clearly compiled in AVX512 (`system_info` confirms), then the same binary SIGILL'd on first use — the only way that's possible with a single-runner job is the well-documented Azure / GitHub-hosted pool variance ([GitHub issue 7842](https://github.com/actions/runner-images/issues/7842), [GitHub issue 8842](https://github.com/actions/runner-images/issues/8842), recurring reports since Aug 2023).

## Proposed change

Add a `GG_BUILD_NO_AVX512` knob to `ci/run.sh`, symmetric with the existing `GG_BUILD_NO_SVE` for ARM, that forces a uniform AVX2 + FMA baseline:

```bash
if [ ! -z ${GG_BUILD_NO_AVX512} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_NATIVE=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX2=ON -DGGML_FMA=ON"
fi
```

Then opt in from the affected workflow job:

```yaml
- name: Test
  run: |
    LLAMA_ARG_THREADS=$(nproc) GG_BUILD_NO_AVX512=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

This is bit-identical to upstream behaviour out of the box: the knob defaults OFF, so anyone who already runs `ci/run.sh` without it sees the same flags as before.

## Why AVX2 + FMA rather than "leave it to runner detection"

Every GitHub-hosted Linux runner in the `ubuntu-22.04` pool supports AVX2 + FMA (the floor is Skylake-class). The heterogeneity is at the AVX512 family, not below it. Pinning AVX2 ON also avoids the symmetric risk — if a future runner generation drops AVX2, the build will fail loudly at cmake time instead of silently SIGILLing at runtime.

## Risk

None. Default-OFF, opt-in. The flag combination is the same one `cmake -DGGML_NATIVE=OFF -DGGML_AVX2=ON` already supports across all backends and architectures; we're not adding new build configurations, only making one explicitly addressable from `ci/run.sh`.

## Files changed

- `ci/run.sh` — +10 lines (env-var stanza next to `GG_BUILD_NO_SVE`)

## Tested in

CrispASR's downstream `build.yml` ran the new path on its `ggml-ci-x64-cpu-high-perf` job and the SIGILL is gone. We're also keeping the in-tree `GG_BUILD_NO_AVX512=0` (i.e. omitting it) on `ggml-ci-x64-cpu-low-perf` and the arm64 jobs, so the existing matrix coverage is unchanged.
