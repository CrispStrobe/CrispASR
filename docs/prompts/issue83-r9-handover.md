# Issue #83 round 9 chatterbox S3Gen UNet Metal GPU drift — handover

You're picking up the open thread on PLAN #83 in
`/Users/christianstrobele/code/CrispASR`. The production fix has
shipped (weight residency split). The kernel-level fix that would
let the UNet run on GPU bit-equivalent to CPU has not — it's stuck
on a specific repro that works in one call context and produces NaN
in another with the same model and seed. This document is the
complete picture for someone to pick up and finish.

## TL;DR — what's wrong, what's done, what's open

**Symptom**: chatterbox S3Gen UNet1D denoiser run on the M1 Metal
backend produces `s3gen_mel cos_min = 0.940` vs the CPU reference (and
~0.86 on CUDA P100 per Round 8). The audio is unintelligible without
mitigation. CPU produces bit-perfect output.

**Production fix (shipped, commit `b84af324`)**: load the 910
`s3.fd.*` UNet weight tensors on the CPU backend buffer in
`chatterbox_s3gen_init_from_file`, so the ggml scheduler routes
the UNet sub-graph to CPU based on weight residency. Encoder
(`s3.fe.*`), flow front-end (`s3.flow.*`), tokenizer (`s3.tok.*`),
speaker encoder (`s3.se.*`), and HiFT vocoder (`s3.v.*`) stay on
GPU. M1: `cos_min 0.940 → 0.999980` in diff harness, intelligible
audio at all T in smoke. Wall-time comparable to pure CPU on M1.

**What's still open**: making the UNet run **on GPU** bit-equivalent
to CPU. The bisect this session found one path that does it — but
only in one call context.

The cleanest empirical fix discovered: `ggml_set_output` on **all 62**
UNet sub-block intermediates restores `cos_min = 1.000, max_abs = 0`
(literally bit-perfect, max diff = 0.0e+00) in the **diff-harness call
context** at `T_mel = 102`. The exact same code NaNs in the **smoke
call context** at every `T_mel` tested (38, 58, 66, 68, 244).

Same model. Same UNet graph (`build_graph_unet1d`). Same CFM solver
loop. Same `--seed 42`. The diff-vs-smoke divergence is invariant
against: random seed, T_mel value, S3-tokenizer presence (.wav vs
precomputed .gguf voice), Metal concurrency (`GGML_METAL_CONCURRENCY_DISABLE=1`
no effect), in-place reuse (`GGML_NO_INPLACE=1` makes it worse), and
F32-tile Q in `kernel_flash_attn_ext` (also makes it worse).

The remaining question is structural in `ggml_gallocr` state across
multi-graph sched invocations. The diff-harness call path runs fewer
ggml graphs into `c->s3gen_ctx->sched` before the UNet than the smoke
call path does. Something about that prior allocator state determines
whether `set_output` on 62 intermediates produces a clean fix or a
NaN.

## Where everything lives

| What | Path |
|---|---|
| Repo root | `/Users/christianstrobele/code/CrispASR` |
| Chatterbox S3Gen runtime | `src/chatterbox_s3gen.cpp` |
| Chatterbox top-level | `src/chatterbox.cpp` |
| Vendored ggml | `ggml/src/ggml-metal/*` (`ggml-metal.metal`, `ggml-metal-ops.cpp`, `ggml-metal-device.cpp`) |
| ggml allocator | `ggml/src/ggml-alloc.c` |
| ggml scheduler | `ggml/src/ggml-backend.cpp` |
| Diff harness | `examples/cli/crispasr_diff_main.cpp` |
| Build dir | `build/` (cmake ninja) |
| Upstream PR drafts | `tools/upstream-prs/{09,10,11}*.{md,patch}` |
| Models (local) | `/Volumes/backups/ai/crispasr/chatterbox-{t3,s3gen}-q8_0.gguf` |
| Models (reference dump) | `/Volumes/backups/ai/chatterbox-ref.gguf` |
| Sample wav | `samples/jfk.wav` |

Linux CPU validation host (no GPU): `ssh root@168.119.190.252`,
source at `/mnt/storage/whisper.cpp`, build dir
`/root/crispasr-build`. The commits here are pushed there on
branch `plan-83-r9-s3gen-gpu-prec-hints` (commit `bd8b98cf`).

## Build

```
cmake --build build --target crispasr-lib crispasr-diff -j 4
```

CMake target names: the binary is `crispasr-cli` but ships as
`bin/crispasr` (per `examples/cli/CMakeLists.txt`); diff harness is
`crispasr-diff`. Both should already exist in `build/bin/`.

## Reproducers

All run from repo root. Both rely on the GPU-residency env knob to
skip the production fix:

```
CRISPASR_CHATTERBOX_FORCE_GPU=1          # use Metal backend for chatterbox
CRISPASR_S3GEN_UNET_GPU_RESIDENCY=1      # opt out of the weight-residency
                                         # production fix (forces UNet to GPU)
CRISPASR_DIFF_USE_GPU=1                  # diff harness only: use GPU
CRISPASR_S3GEN_UNET_PRESERVE_INTERMEDIATES=1  # this session's debug knob:
                                              # set_output on 14 block outputs
CRISPASR_S3GEN_DUMP_UNET=<tag>           # set_output on all 62 intermediates +
                                         # dump them to /tmp/cb-unet-dump-<tag>-*.bin
CRISPASR_S3GEN_UNET_PIN_CPU_OP=<op>      # pin all ops of type <op> to CPU
                                         # (e.g. mul_mat, norm, flash_attn_ext)
GGML_METAL_CONCURRENCY_DISABLE=1         # existing ggml knob — serial Metal exec
GGML_NO_INPLACE=1                        # not committed — was a session experiment
```

### Repro A — diff harness (THE PASSING CASE)

```
CRISPASR_DIFF_USE_GPU=1 CRISPASR_CHATTERBOX_FORCE_GPU=1 \
CRISPASR_S3GEN_UNET_GPU_RESIDENCY=1 CRISPASR_S3GEN_DUMP_UNET=test \
  build/bin/crispasr-diff chatterbox \
  /Volumes/backups/ai/crispasr/chatterbox-t3-q8_0.gguf \
  /Volumes/backups/ai/chatterbox-ref.gguf \
  samples/jfk.wav 2>&1 | grep s3gen_mel
```

Expected: `[PASS] s3gen_mel ... cos_min=1.000000 ... max_abs=0.00e+00`.
Bit-perfect. T_mel=102.

### Repro B — smoke (THE FAILING CASE)

```
CRISPASR_CHATTERBOX_FORCE_GPU=1 CRISPASR_S3GEN_UNET_GPU_RESIDENCY=1 \
CRISPASR_S3GEN_DUMP_UNET=test \
  build/bin/crispasr --backend chatterbox --tts "Hello." \
  --voice samples/jfk.wav --tts-output /tmp/cb-smoke.wav --seed 42 \
  2>&1 | grep "vocoder mel"
```

Expected: `s3gen: vocoder mel T=68 rms=nan min=1e30 max=-1e30`. NaN.
T_mel=68.

**Both use `c->s3gen_ctx` with the same `--voice samples/jfk.wav`,
same chatterbox-t3-q8_0.gguf + chatterbox-s3gen-q8_0.gguf, same
seed.** Only the call entry point differs.

### Repro C — production fix (the working baseline)

Drop `CRISPASR_S3GEN_UNET_GPU_RESIDENCY=1` from either of the above.
Both produce intelligible audio (`cos_min 0.999980` in diff harness,
`rms ~5.1` in smoke matching ref).

## What the diff harness does that smoke doesn't

Both paths end up in `chatterbox_s3gen_compute_gen_mel`
(`src/chatterbox_s3gen.cpp:2878`) which calls `cfm_euler_solve` →
10× `run_denoiser` → 10× `build_graph_unet1d`. **The UNet graph
itself is built identically.**

The diff-harness entry point is `chatterbox_synthesize_mel_from_tokens_with_noise`
(`src/chatterbox.cpp:2966`) which calls
`chatterbox_s3gen_synthesize_mel_with_noise` (`src/chatterbox_s3gen.cpp:3054`)
which calls `chatterbox_s3gen_compute_gen_mel` with an
externally-supplied `init_noise_cf`.

The smoke entry point is `chatterbox_synthesize` → eventually
`chatterbox_s3gen_synthesize` (`src/chatterbox_s3gen.cpp:3088`) →
`chatterbox_s3gen_compute_gen_mel(..., init_noise_cf=nullptr, ...)`.
Smoke generates noise from `c->noise_rng` instead.

So functionally:

- diff harness: encoder graph + 10× UNet graph
- smoke (`--voice .wav`): S3 tokenizer graph + encoder graph + 10× UNet graph
- smoke (`--voice .gguf`): encoder graph + 10× UNet graph (still NaNs)

The smoke-with-`.gguf`-voice case still NaNs, so S3 tokenizer
involvement is **not** the cause. Something else about the
`crispasr` CLI invocation path vs the `crispasr-diff` invocation
path is.

Look at:
- `examples/cli/crispasr_diff_main.cpp:1232` (calls
  `chatterbox_mel_from_tokens_with_noise_r`)
- `examples/cli/crispasr_run.cpp` (the smoke path through chatterbox)

The chatterbox_s3gen_context is freshly constructed in each
invocation (it's destroyed when the CLI exits), so cross-invocation
state isn't the issue. Within a single invocation, what graphs run
through `c->s3gen_ctx->sched` before the UNet?

## The dump infrastructure

`CRISPASR_S3GEN_DUMP_UNET=<tag>` (in `src/chatterbox_s3gen.cpp`
around line 1700 and 1944) does two things:

1. In `build_graph_unet1d`, calls `ggml_set_output` on 62 named
   intermediate tensors (`dump_db_resnet`, `dump_db_tb_0..3`,
   `dump_db_out`, `dump_mb_{0..11}_resnet`, `dump_mb_{0..11}_out`).
   The `set_output` disables `ggml_gallocr`'s in-place buffer reuse
   for those tensors. *This is what restores cos=1.000 in the diff
   harness.*

2. In `run_denoiser`, after `ggml_backend_sched_graph_compute` at
   step 0, walks the graph and dumps every `dump_*` node via
   `ggml_backend_tensor_get` to `/tmp/cb-unet-dump-<tag>-<name>.bin`.

To compare two runs:

```bash
# CPU run (UNet on CPU via default weight residency)
CRISPASR_DIFF_USE_GPU=1 CRISPASR_CHATTERBOX_FORCE_GPU=1 \
  CRISPASR_S3GEN_DUMP_UNET=cpu \
  CRISPASR_CHATTERBOX_SEED=42 \
  build/bin/crispasr-diff chatterbox \
  /Volumes/backups/ai/crispasr/chatterbox-t3-q8_0.gguf \
  /Volumes/backups/ai/chatterbox-ref.gguf samples/jfk.wav

# GPU run (UNet on GPU via opt-out)
CRISPASR_DIFF_USE_GPU=1 CRISPASR_CHATTERBOX_FORCE_GPU=1 \
  CRISPASR_S3GEN_UNET_GPU_RESIDENCY=1 \
  CRISPASR_S3GEN_DUMP_UNET=gpu \
  CRISPASR_CHATTERBOX_SEED=42 \
  build/bin/crispasr-diff chatterbox \
  /Volumes/backups/ai/crispasr/chatterbox-t3-q8_0.gguf \
  /Volumes/backups/ai/chatterbox-ref.gguf samples/jfk.wav
```

Then compare with a small python script (use `python`, not
`python3` — there's an alias):

```python
import numpy as np, glob, os, re
keep = lambda p: "(transposed)" not in p and "(cont)" not in p
cpu_files = sorted([p for p in glob.glob("/tmp/cb-unet-dump-cpu-*.bin") if keep(p)])
gpu_files = sorted([p for p in glob.glob("/tmp/cb-unet-dump-gpu-*.bin") if keep(p)])
# ... per-file cos, max_abs, rms
```

Note: with `DUMP_UNET` set, the diff harness's `cos=1.000` reflects
the modified graph (with set_outputs), not the production
configuration. Read accordingly.

## What's already been ruled out

All 11 from the bisect (LEARNINGS Round 9):

1. Bit-match Q8_0 mul_mat kernel — committed (`752baecf`) but no
   help with the UNet drift.
2. PIN_CPU_OP=mul_mat at T=102 → cos=1.000, but NaN at T=200 smoke.
3. PIN bisect at T=102: any frequent op restores parity, sparse ops
   don't. *Sync-barrier density signal, not op-identity.*
4. set_output on 1 intermediate → 0.879 (worse than baseline 0.940).
5. set_output on 14 (block outputs only) → 0.907 at diff, ~rms 14 at
   smoke (degraded, no NaN).
6. set_output on 62 (all intermediates) → **1.000 at diff, NaN at
   smoke**. The repro that this handover is for.
7. GGML_NO_INPLACE=1 (global allocator knob skipping in-place reuse)
   → cos=-0.97 (sign-flipped garbage; some downstream code expects
   in-place semantics).
8. GGML_METAL_CONCURRENCY_DISABLE=1 → no effect.
9. kernel_norm audit → F32-clean.
10. kernel_flash_attn audit: F32 K/V family downcasts Q to F16 (line
    6430). Patched FA_TYPES_F32 to F32 Q tile → cos went 0.940 →
    0.860 (worse).
11. Multiple seeds (1, 7, 42, 100, 1234) + smoke with .gguf voice
    (no S3 tokenizer): all NaN.

## Suggested next steps (in priority order)

**1. Trace ggml_gallocr's per-tensor address decisions.** Both
runs (diff harness and smoke) build the same UNet graph and call
`ggml_backend_sched_alloc_graph(c->s3gen_ctx->sched, gf)`. Instrument
`ggml_gallocr_allocate_node` in `ggml/src/ggml-alloc.c:622` to log:

- Op name + index in graph
- For each src parent: name, n_children, n_views, addr
- Allocation decision: which `buffer_id`, which `addr`, whether
  in-place reuse fired

Run both repros, diff the logs. The first node where allocations
differ is where the divergence starts. With set_output on 62 tensors,
the allocator's reuse decisions cascade — small differences upstream
flip many decisions downstream.

**2. Test the "diff harness with extra graph" hypothesis.** Add a
no-op extra ggml graph compute to `chatterbox_synthesize_mel_from_tokens_with_noise`
*before* the UNet (e.g., one redundant encoder call). If this makes
the diff harness ALSO NaN, the divergence is from the cumulative
graph count seen by `c->s3gen_ctx->sched` before the UNet runs.
If not, the divergence is from something more specific.

**3. Bisect the smoke→diff path.** The smoke entry point is
`chatterbox_synthesize` (`src/chatterbox.cpp:2827`). It calls into
`chatterbox_s3gen_synthesize`. Skip steps incrementally and see
when the NaN goes away:
   - Skip vocoder (it's CPU and runs AFTER UNet so unlikely cause —
     but verify by running smoke and checking NaN is in UNet output
     not vocoder).
   - Skip CAMPPlus speaker embedding processing.
   - Skip the S3 tokenizer (already tested via `.gguf` voice — still
     NaN, so it's not this).
   - Skip whatever else runs through `c->s3gen_ctx->sched` before the
     CFM solver.

**4. If (1) reveals the divergent allocation point**, that tells you
which tensor address change to fix. The fix may be in
`ggml_gallocr_allocate_node`'s heuristic, or in how
`ggml_backend_sched_reset` clears state between graph computes.

**5. If (1–3) all fail to isolate the difference**, file PR 10
upstream as a bug report (it's drafted in
`tools/upstream-prs/10-metal-sched-buffer-reuse-drift.md`). The
reduced repro for upstream is "set_output on intermediates of a
moderately complex graph produces bit-perfect output in one call
context, NaN in another; both use the same scheduler, same model,
same seed."

## Quick repro of "what works today in production"

Smoke with the production fix (the WORKING path):

```
CRISPASR_CHATTERBOX_FORCE_GPU=1 \
  build/bin/crispasr --backend chatterbox \
  --tts "Ask not what your country can do for you." \
  --voice samples/jfk.wav --tts-output /tmp/cb.wav --seed 42

build/bin/crispasr --backend parakeet \
  --model /Volumes/backups/ai/crispasr/parakeet-tdt-0.6b-v3-q8_0.gguf \
  -f /tmp/cb.wav
```

Should give "Ask not what your country cando for yo." (intelligible).
If that breaks, something earlier than this handover regressed.

## Don't waste time on

- F16/bf16 audit of kernel_norm (already done — F32-clean).
- Q audit of kernel_flash_attn_ext (already done — F16 downcast
  identified; patching to F32 made it worse, not better).
- Concurrency bugs (`GGML_METAL_CONCURRENCY_DISABLE=1` had no
  effect).
- Global no-inplace (tested, breaks other things via cos=-0.97
  sign flip).
- Op-identity bisect ("which op is buggy") — there isn't one. The
  drift is distributed; the fix is sync-barrier density.
- Different random seeds (all 5 tested produce same NaN).
- Skipping the S3 tokenizer (already tested via `.gguf` voice).

## Per the standing repo rules

- No Claude attribution in commit messages (see
  `~/.claude/projects/-Users-christianstrobele-code-CrispASR/memory/feedback_strip_local_markers.md`).
- Outbound artifacts (upstream PRs) strip the `// CrispASR patch`
  markers and `(#83)` refs. The 3 drafts in `tools/upstream-prs/` are
  already cleaned; keep them that way if you add commits.
- Use `python` not `python3` (alias). Explicit venv paths fine.
- Use `clang-format` v18 (`./tools/format.sh` enforces).
- Don't park work on feature branches — rebase into main when ready.
- For commit-bound parallel work, use a git worktree. The VPS already
  has `plan-83-r9-s3gen-gpu-prec-hints` checked out at
  `/mnt/storage/whisper.cpp`.

## Commits this session (chronological)

| Hash | Subject |
|---|---|
| `b84af324` | fix(#83): S3Gen UNet weight residency split + per-op GGML_PREC_F32 hints |
| `752baecf` | metal(#83): Q8_0 x F32 bit-match mul_mat under GGML_PREC_F32 |
| `c00c1493` | debug(#83): per-segment dump hook for UNet diff-bisect |
| `d7d859a2` | docs(upstream-prs): drafts 09-11 from chatterbox UNet debug session |
| `2daf2a19` | debug(#83): UNet PRESERVE_INTERMEDIATES bisect knob |
| `b6a0b610` | docs(upstream-prs/10): expand bisect findings |
| `7dccd202` | docs(#83): write up Round 9 in PLAN/HISTORY/LEARNINGS |

Read `LEARNINGS.md` § "Chatterbox #83 Round 9" for the full
narrative. This handover is the operational version.
