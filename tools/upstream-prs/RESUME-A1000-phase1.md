# RESUME — issue #81 A1000 Phase 1 kernel work (2026-05-11)

This document is the complete pickup guide if the session context that
started this work is lost. It assumes you've read:

- `PERFORMANCE.md` → the four issue #81 subsections in chronological
  order (`onnx-asr cross-comparison` → `jason-ni/parakeet.cpp` →
  `A1000 Ampere CUDA A/B (sm_86)` → the two follow-up addenda)
- `tools/upstream-prs/README.md` for the existing patch-workflow
  conventions (`// CrispASR patch` comments, `NN-<area>.{md,patch}`)
- `tools/upstream-prs/05-cuda-unary-row-contig.md`
- `tools/upstream-prs/06-cuda-fa-perhead-mask.md`

## Where we left off

Three commits are on `main` covering the A1000 work that's already
landed:

- `d211409` — A1000 baseline (PRE/POST verdict, flash-attn-ext win at boost)
- `300a112` — `-DGGML_CUDA_GRAPHS=ON` closes 2.12× → **1.99×** behind onnx-fp32
- `01d50b1` — Phase 0/1 root-cause: two ggml-cuda support gates fall back to CPU

Phase 1 kernel work is **in progress, uncommitted, gated default-OFF**.
Specifically:

| file | change | gate |
|---|---|---|
| `ggml/CMakeLists.txt` | adds 2 CMake options | `GGML_CUDA_CRISPASR_UNARY_ROWS` (OFF), `GGML_CUDA_CRISPASR_FA_PERHEAD_MASK` (OFF) |
| `ggml/src/ggml-cuda/CMakeLists.txt` | wires both options to `add_compile_definitions` | — |
| `ggml/src/ggml-cuda/unary.cu` | adds `unary_op_kernel_strided` + dispatch fork | `#ifdef GGML_CUDA_CRISPASR_UNARY_ROWS` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | UNARY support gate uses `ggml_is_contiguous_rows` | same `#ifdef` |

Both default-OFF means anyone building with the existing `release.yml`
flags gets bit-identical DLLs to before the changes. **Don't revert
without first benching with the flag ON** — the whole point of the
gate is so we can A/B fairly.

`tools/upstream-prs/05-cuda-unary-row-contig.md` and
`06-cuda-fa-perhead-mask.md` exist as untracked draft narratives.
`.patch` files for #05 and #06 do **not** exist yet (generate via
`git diff` after committing — see Step 7 below).

#06 (FA per-head mask) is design-only — the .md documents the call
chain that needs editing but the code change is not started. It needs
real kernel-signature changes across 4-6 `.cuh` files plus
`test-backend-ops` verification on at least sm_75 + sm_86 + sm_89.
Out of scope until #05 lands clean.

## Resume sequence (Phase 1, #05 UNARY only)

Sets a few env constants used throughout. Set these once per
PowerShell session that you're working in:

```powershell
$WORK    = "C:\Users\stc\Downloads\code\bench-issue81"
$BUILD   = "$WORK\build-cg-uar"
$DLL_OUT = "$WORK\dll-cg-uar"
$RESULTS = "$WORK\results"
$MODELS_GGUF = "$WORK\models-gguf"
$MODELS_ONNX = "$WORK\models-onnx"
$env:HF_HOME = "$WORK\hf-cache"
# CUDA-12 deps for ONNX EP (already installed in .venv)
$venv_sp = ".\.venv\Lib\site-packages"
$nvbin = @(
  "$venv_sp\nvidia\cublas\bin","$venv_sp\nvidia\cudnn\bin",
  "$venv_sp\nvidia\cuda_runtime\bin","$venv_sp\nvidia\cuda_nvrtc\bin",
  "$venv_sp\nvidia\cufft\bin"
) | ForEach-Object { (Resolve-Path $_).Path }
$cleaned = ($env:PATH -split ';' | Where-Object { $_ -and $_ -notmatch 'CUDA\\v13' }) -join ';'
$env:PATH = ($nvbin -join ';') + ';' + $cleaned
$py = ".\.venv\Scripts\python.exe"
```

### Step 1. Make sure the NVAPI `PreferredPState=1` is still set

The 3.063 s baseline was measured with global `PreferredPState=1`
(via `nvidiaProfileInspector`). If you cold-rebooted in between or
re-installed the NVIDIA driver, re-set it:

```powershell
# elevated PowerShell (UAC prompt expected)
$NPI = "$WORK\npi\nvidiaProfileInspector.exe"
Start-Process -FilePath $NPI `
  -ArgumentList @("-setProfileSetting","_GLOBAL_DRIVER_PROFILE,0x1057EB71,1") `
  -Verb RunAs -Wait
nvidia-smi --query-gpu=pstate,clocks.current.graphics --format=csv,noheader
# expect: P0, 1140 MHz (or higher) at idle
```

If `nvidia-smi` still reports P8/210 MHz idle, the setting didn't
stick — diagnose before benching, or your numbers will be
unrepresentative (we already saw POST go from 3.06 s to 73 s when the
GPU is parked at P8).

### Step 2. Finish the UAR build (incremental)

The build was paused at ~117/~600 .obj. The cache survived; resuming
is incremental:

```powershell
cmake --build $BUILD --config Release --target crispasr -j $env:NUMBER_OF_PROCESSORS
```

Expected: ~5-10 more minutes (the bulk of the remaining work is mma
and mmq template instances). Watch for compile errors in the new
`unary_op_kernel_strided` — the template+typename combo is the only
non-trivial part. If nvcc rejects it, re-check `unary.cu` lines
~118-178 against this document.

Sanity-check the produced DLL:

```powershell
mkdir $DLL_OUT -Force | Out-Null
Copy-Item "$WORK\dll-post-cg\*.dll" $DLL_OUT -Force     # base bundle (cublas + ggml-cpu + ggml-base + whisper)
Copy-Item "$BUILD\bin\Release\crispasr.dll"   "$DLL_OUT\crispasr.dll"   -Force
Copy-Item "$BUILD\bin\Release\ggml-cuda.dll"  "$DLL_OUT\ggml-cuda.dll"  -Force
Copy-Item "$BUILD\bin\Release\ggml.dll"       "$DLL_OUT\ggml.dll"       -Force
Copy-Item "$BUILD\bin\Release\whisper.dll"    "$DLL_OUT\whisper.dll"    -Force
& $py -c "import ctypes; ctypes.CDLL(r'$DLL_OUT\crispasr.dll'); print('UAR bundle: OK')"
```

`cublas64_13.dll` + `cublasLt64_13.dll` from the base bundle must be
co-located with the new `crispasr.dll` (we did this once already in
`dll-post-cg/`). The `Copy-Item "$WORK\dll-post-cg\*.dll" $DLL_OUT`
line carries them across.

### Step 3. Sanity-check the new path activates

Run with `GGML_SCHED_DEBUG=2` on the short clip; confirm the 24
`GGML_OP_UNARY` ops are now on `[CUDA0]` instead of `[CPU]`:

```powershell
$env:GGML_SCHED_DEBUG = "2"
& $py tools\benchmark_asr_engines.py `
  --engine crispasr --crispasr-lib "$DLL_OUT\crispasr.dll" `
  --gguf-quants q8_0 --gguf-dir $MODELS_GGUF --onnx-dir $MODELS_ONNX `
  --mode chunked --window-s 4 --warmups 0 --runs 1 `
  --gpu-backend cuda `
  --json "$RESULTS\sched-debug-uar.json" --audio short 2>&1 |
  Tee-Object "$WORK\sched-debug-uar.log" | Select-Object -Last 3
Remove-Item Env:GGML_SCHED_DEBUG
```

Compare against `bench-issue81\sched-debug.log` (from the baseline
run). Specifically:

```powershell
"baseline CPU splits: $((Select-String -Path $WORK\sched-debug.log     '## SPLIT #.*CPU').Count)"
"UAR      CPU splits: $((Select-String -Path $WORK\sched-debug-uar.log '## SPLIT #.*CPU').Count)"
```

Expected: baseline shows 144 CPU splits per chunk, UAR shows
**72** (the 24 UNARY splits per chunk × 3 split-pattern entries went
away; the 24 FLASH_ATTN_EXT splits remain). If UAR shows the same
144 the patch isn't actually firing — debug before going further.

Also confirm WER is still 0.000:

```powershell
& $py -c "import json; r=json.load(open(r'$RESULTS\sched-debug-uar.json'))['results'][0]; print('WER:', r.get('wer')); print('transcript:', r.get('transcript_sample',''))"
```

If WER != 0.000, the strided kernel has a correctness bug. Revert
and debug `unary_op_kernel_strided`'s index math (most likely
suspect: `i00 = i % ne00` vs `i00 = i / (...)` shuffle).

### Step 4. Wallclock A/B vs the 3.063 s baseline

```powershell
& $py tools\benchmark_asr_engines.py `
  --engine crispasr --crispasr-lib "$DLL_OUT\crispasr.dll" `
  --gguf-quants q8_0 --gguf-dir $MODELS_GGUF --onnx-dir $MODELS_ONNX `
  --mode chunked --window-s 4 --warmups 1 --runs 10 `
  --gpu-backend cuda --prewarm `
  --json "$RESULTS\a1000-post-cg-uar.json" --audio long
```

Expected: long-clip mean somewhere between **2.8 s** (conservative,
matches the "9 % wallclock from UNARY splits" estimate) and
**2.5 s** (optimistic, if the split reduction also lets CUDA Graphs
capture more of the per-chunk encoder pass).

Compare:
- baseline (POST+CG q8_0, no UAR): `a1000-post-cg-q8_0.json` → 3.063 s
- UAR: `a1000-post-cg-uar.json` → target < 3.0 s

### Step 5. Decision tree

| outcome | action |
|---|---|
| UAR ≥ 2.8 s **and** WER 0.000 **and** sched-debug shows 72 CPU splits | **Win** — proceed to Step 6. |
| UAR > 2.95 s but WER 0.000 | Marginal — proceed to Step 6 anyway, document that the win is small. The patch is still correct and the upstream-ggml PR is still worth filing. |
| WER ≠ 0.000 | **Correctness bug in `unary_op_kernel_strided`.** Revert ggml changes (`git checkout ggml/`); document in #05's .md what went wrong; do NOT commit. |
| UAR regresses (mean > 3.2 s) | Surprising. Diagnose with full nsys profile and `cuda_api_sum`. Most likely cause: the extra arithmetic in the strided kernel slows down even the fully-contiguous path. If so, verify the `if (ggml_is_contiguous(src0))` dispatch fork actually fires for the encoder's other UNARY ops (silu in FF, gelu_quick in attention). Could need to also gate the contiguous-fast-path dispatch behind `#ifdef`. |

### Step 6. Decide the default

If the win is real **and clean**, flip the CMake option default:

```cmake
option(GGML_CUDA_CRISPASR_UNARY_ROWS        "CrispASR: allow strided-row unary ops on CUDA"   ON)
```

The `release.yml` Windows-CUDA shared-libs slot doesn't need any
change — the option will simply default ON for our internal builds.
**Do not flip it on for non-Ampere arches without measuring** — the
strided kernel's extra arithmetic might cost more on smaller GPUs;
defer to a conditional default (`set(GGML_CUDA_CRISPASR_UNARY_ROWS_DEFAULT ON)`
under a CMake `if (CMAKE_CUDA_ARCHITECTURES MATCHES "86|89|90|120")`
check would be the conservative form).

If the win is marginal or the patch fits the upstream criteria
cleanly, keep the default **OFF** in our `ggml/CMakeLists.txt` and
opt-in via our own `release.yml` Windows-CUDA matrix slot:

```yaml
# release.yml :: build-libs-windows-x86_64-cuda
cmake -B build `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_SHARED_LIBS=ON `
  -DGGML_CUDA=ON `
  -DGGML_CUDA_GRAPHS=ON `
  -DGGML_CUDA_CRISPASR_UNARY_ROWS=ON `
  -DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;89-real;120-real;120-virtual" `
  ...
```

This keeps `ggml/` mainline-shaped while activating our patch in
our shipped DLLs. Same pattern as `GGML_CUDA_GRAPHS` (a knob whose
default we already want to flip ON for our builds per the previous
PERFORMANCE.md addendum).

### Step 7. Generate .patch files + update README

```powershell
cd C:\Users\stc\Downloads\code\CrispASR
git diff ggml/CMakeLists.txt ggml/src/ggml-cuda/CMakeLists.txt `
         ggml/src/ggml-cuda/unary.cu ggml/src/ggml-cuda/ggml-cuda.cu `
  > tools\upstream-prs\05-cuda-unary-row-contig.patch
```

Then update `tools/upstream-prs/README.md`'s table to add a row
between #04 and #05 (or whatever number is correct after any other
filed upstream PRs):

```markdown
| 05 | `ggml-cuda : support row-contiguous (strided) unary ops` | yours (CrispASR Phase 1) | drafted |
```

And update `tools/upstream-prs/MASTER-AUDIT.md` to add #05 to the
"vulnerable" table with the line numbers in master where the support
gate is still strict.

### Step 8. clang-format only for CrispASR-authored code, NOT vendored ggml

**Per project memory (`feedback_clang_format_18.md`):** the v18 rule
applies to **CrispASR-authored** sources (`src/`, `examples/`, …).
**Vendored ggml subtree** (`ggml/`) has its own upstream style; the
.clang-format at root produces enormous whole-file churn when run on
ggml-cuda mainline (verified during this session — `clang-format
-i ggml/src/ggml-cuda/unary.cu` produced 1 798 insertions / 1 726
deletions of whitespace because mainline ggml-cuda is not v18-clean).

**Rule:** match the surrounding upstream-ggml style for new code
inside `ggml/`. The existing in-tree patches
(`tools/upstream-prs/02-cuda-im2col.patch`, `03-cuda-cpy.patch`) all
do this and never run clang-format over the vendored subtree.

For this PR's files specifically:

- ✅ `ggml/CMakeLists.txt` — not formatted by clang-format anyway.
- ✅ `ggml/src/ggml-cuda/CMakeLists.txt` — same.
- ✅ `ggml/src/ggml-cuda/unary.cu` — **do NOT run clang-format**.
  Match mainline ggml indent (4 spaces, no break-before-brace, etc.).
- ✅ `ggml/src/ggml-cuda/ggml-cuda.cu` — same.

`clang-format-18` is named just `clang-format` on this Windows host
and lives at `C:\Users\stc\AppData\Roaming\Python\Python312\Scripts\clang-format`
(`clang-format --version` reports `18.1.8`). It IS v18 — the name
just doesn't have the `-18` suffix on this install.

**Do not substitute** clang-format-17 or -19 if you ever do format
CrispASR-side files (`src/`, `examples/`, …) in a future change.

Sanity-check before committing: `git diff --stat ggml/` should show
small +N/-M numbers (this PR: 92 insertions, 0 deletions). If the
diff is in the thousands of lines, you've accidentally run
clang-format over the vendored subtree — `git checkout -- ggml/` and
re-apply only your edits manually.

### Step 9. Commit + push (also adds the .md / .patch artifacts)

```powershell
cd C:\Users\stc\Downloads\code\CrispASR
git fetch origin main
git pull --rebase   # (stash first if there are unstaged changes)
git add ggml/CMakeLists.txt ggml/src/ggml-cuda/CMakeLists.txt `
        ggml/src/ggml-cuda/unary.cu ggml/src/ggml-cuda/ggml-cuda.cu `
        tools/upstream-prs/05-cuda-unary-row-contig.md `
        tools/upstream-prs/05-cuda-unary-row-contig.patch `
        tools/upstream-prs/06-cuda-fa-perhead-mask.md `
        tools/upstream-prs/RESUME-A1000-phase1.md `
        tools/upstream-prs/README.md `
        tools/upstream-prs/MASTER-AUDIT.md
git status   # double-check only those files
git commit -m "ggml-cuda: support row-contiguous (strided) unary ops (#81 Phase 1 #05)

<one-paragraph summary of the patch + measured speedup vs 3.063s baseline>

Gated behind -DGGML_CUDA_CRISPASR_UNARY_ROWS=<ON|OFF, depending on
Step 6 decision> in ggml/CMakeLists.txt; contiguous fast-path
preserved bit-identical to mainline.

Adds tools/upstream-prs/05-cuda-unary-row-contig.{md,patch} for the
eventual upstream PR. Companion patch #06 (FA per-head mask) is
documented as the next target but not yet implemented because the
kernel-signature change spans 4-6 .cuh files and needs full
test-backend-ops verification across sm_75 / sm_86 / sm_89 before
landing."
git push origin main
```

**No `Co-Authored-By: Claude` trailer per project memory.** The user
adds attribution locally if they want it.

### Step 10. Update PERFORMANCE.md

Append a short paragraph to the "Closing the 2.12× gap" addendum
(commit `300a112`) noting the UAR measured result, where it sits in
the wallclock breakdown, and link to `tools/upstream-prs/05-…`. Keep
it brief — the .md in tools/upstream-prs/ is the authoritative narrative.

---

## Resume sequence (Phase 1, #06 FA per-head mask — when ready)

This is the **next** PR to ship after #05 lands clean. It is design-
only as of 2026-05-11; the .md `06-cuda-fa-perhead-mask.md` has the
call-chain analysis you need to start the kernel-signature change.

Rough scope:

1. Add `stride_mask_head` parameter to the FA kernel signatures
   (start with `fattn-mma-f16.cuh` — the kernel selected on
   Ampere/Ada in `get_best_fattn_kernel`; expand to `wmma-f16` and
   `tile` only after MMA-F16 is verified).
2. Plumb the param through `launch_fattn_*` / `flash_attn_ext_f16_case`
   in `fattn-common.cuh`.
3. Inside the kernel, advance `mask_h` by `head_idx * stride_mask_head`
   per head iteration (the kernel already has a per-head loop).
4. Loosen `get_best_fattn_kernel`'s `mask->ne[2] != 1` guard for the
   MMA-F16 path (keep it for kernels you haven't extended).
5. `test-backend-ops` case for per-head mask (mask shape `(T_kv,
   T_q, n_heads, 1)`, n_heads ∈ {1, 2, 4, 8, 16}).

Verify on parakeet-tdt-0.6b-v3 on A1000 (n_heads=8) and at least one
other model with multi-head mask if you have one (whisper does not
have per-head masks; canary uses the same FastConformer encoder as
parakeet so it gets the same speedup).

Expected post-#06 wallclock: ~2.5 s on long-clip, RT ~24×, ~1.6×
behind onnx-fp32. After that, the structural cuDNN-conv gap
discussed in PERFORMANCE.md takes over.

---

## State if you abandon halfway

If you decide not to land #05:

```powershell
cd C:\Users\stc\Downloads\code\CrispASR
git checkout -- ggml/CMakeLists.txt ggml/src/ggml-cuda/CMakeLists.txt `
                 ggml/src/ggml-cuda/unary.cu ggml/src/ggml-cuda/ggml-cuda.cu
Remove-Item tools/upstream-prs/05-cuda-unary-row-contig.md `
            tools/upstream-prs/06-cuda-fa-perhead-mask.md `
            tools/upstream-prs/RESUME-A1000-phase1.md
```

The repo returns to the `01d50b1` state (Phase 0/1 verdict
committed, no code change to ggml). Nothing in `main` is broken
because the CMake flags defaulted OFF anyway, but the untracked
.md files clutter the worktree until removed.

Also clean up local build dirs that are no longer needed:

```powershell
Remove-Item -Recurse -Force C:\Users\stc\Downloads\code\bench-issue81\build-cg-uar
Remove-Item -Recurse -Force C:\Users\stc\Downloads\code\bench-issue81\dll-cg-uar
Remove-Item -Recurse -Force C:\Users\stc\Downloads\code\bench-issue81\dll-post-cg-cudaonly
Remove-Item -Recurse -Force C:\Users\stc\Downloads\code\bench-issue81\dll-post-cg-offload
Remove-Item -Recurse -Force C:\Users\stc\Downloads\code\bench-issue81\dll-post-cg-glucont
```

The `dll-post-cg/` bundle is the one that backs the published
3.063 s baseline; keep it.

---

## Where everything lives

| artifact | path |
|---|---|
| **baseline 3.063 s DLL bundle** | `bench-issue81\dll-post-cg\` (q8_0 + CUDA Graphs ON, default UNARY/FA gates OFF) |
| baseline JSON | `bench-issue81\results\a1000-post-cg-q8_0.json` (also copied to `handover-prompts\a1000-post.json` with rename) |
| baseline nsys | `bench-issue81\results\nsys-crispasr-post-cg.nsys-rep` (~2 MB) |
| baseline sched-debug | `bench-issue81\sched-debug.log` (~658 KB, GGML_SCHED_DEBUG=2 dump) |
| F16 GGUF (downloaded) | `bench-issue81\models-gguf\parakeet-tdt-0.6b-v3.gguf` |
| Q4_K GGUF (downloaded) | `bench-issue81\models-gguf\parakeet-tdt-0.6b-v3-q4_k.gguf` |
| ONNX snapshot | `bench-issue81\models-onnx\` (3.07 GB, encoder+decoder+joint int8/fp32) |
| nvidiaProfileInspector | `bench-issue81\npi\nvidiaProfileInspector.exe` |
| Python venv | `.venv\` (CrispASR root; has onnxruntime-gpu + nvidia-* CUDA-12 wheels) |

The `bench-issue81\` directory is **not** under version control. If
you nuke the laptop you have to redo the model downloads + venv
setup; the build dirs themselves are reconstructible from the
documented cmake invocations.

## Memory + permissions context

- Project memory file `feedback_clang_format_18.md` documents the
  clang-format-18 rule. Re-read at session start.
- Project memory file `project_windows_download_fixes.md` documents
  unrelated earlier session work; the NPI/HF_HOME setup pattern from
  this session is **not** in memory and won't auto-load (was a
  one-time fix for this machine's `HF_HOME=E:\…` env-var leftover).
- The `Co-Authored-By: Claude` git trailer is **not** wanted (per
  project memory).
- The `/handover-prompts/` dir is gitignored; commits should only
  touch tracked files (the existing pattern from `d211409` /
  `300a112` / `01d50b1`).
