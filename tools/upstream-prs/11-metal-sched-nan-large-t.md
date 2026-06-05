**Title:** `metal/sched : mixed CPU+GPU op pinning produces NaN at large input dimensions`

---

Bug report, no patch. Filing this as a separate issue from the
buffer-reuse drift report (see related issue on `ggml-alloc` drift in
long F32 GPU graphs), since the failure mode is different even though
both surface in the same chatterbox-tts S3Gen UNet.

## Symptom

A graph that runs cleanly on Metal at a smaller input dimension
produces all-NaN output at a larger one when an op type is pinned to
the CPU backend via `ggml_backend_sched_set_tensor_backend`.

Concretely, in our UNet graph (described in the buffer-reuse drift
report), with all weights GPU-resident and the scheduler set to
`[backend_metal, backend_cpu]`:

| T_mel | PIN_CPU_OP=mul_mat result | All-GPU result |
| - | - | - |
| 102 (diff-harness shape) | `cos_min = 1.000` (clean) | `cos_min = 0.940` |
| 200 (production shape) | **NaN everywhere** | `cos_min = 0.940` |

The all-GPU path is consistent (same drift level) across T. Only the
mixed-backend execution fails at the larger size.

Failure mode is dramatic: `rms = nan`, `min = +1e30`, `max = -1e30` at
the very first step of the iterative solver, indicating uninitialised
or corrupted memory rather than a numerical blow-up.

## Bisect

Tested pinning each of `unary_gelu, flash_attn_ext, concat, permute,
reshape, cont, norm, mul, add` to CPU at the production T (= 200).
All except `permute` produce the same NaN. `permute` produces finite
but garbage output (cos vs ref near 0).

The breaking point is somewhere between T = 102 (works) and T = 200
(NaN); we haven't bisected the exact threshold. Activation tensor
size at the breaking T is ~70 KB per channel × 320 channels = 22 MB
per tensor, well within Metal allocation limits.

## What this looks like from the application side

In normal use we wanted to combine the GPU compute path (fast) with a
CPU pin on `mul_mat` (per the related buffer-reuse drift report, this
restores `cos_min = 1.000` at any T). On the larger production
graphs the mixed schedule corrupts memory instead.

The fallback we're shipping is to load the entire UNet's weight
tensors on the CPU backend up-front so the scheduler routes the whole
sub-graph to CPU based on weight residency (no per-op pin needed).
That works at all T because the sub-graph is internally homogeneous
— no GPU↔CPU sync points inside the UNet.

## Investigation pointers

The two paths the scheduler exercises differently between the
working-small case and the failing-large case:

- `ggml-backend.cpp:1664` `iface.cpy_tensor_async` — the async GPU↔CPU
  tensor copy used when feeding a pinned op's inputs. The
  `ggml_backend_synchronize` + event-record dance after it. Could be a
  command-buffer fence that's not waited on for large allocations.

- `ggml-metal-ops.cpp:147` `ggml_metal_op_concurrency_reset` — barrier
  insertion at the boundary between Metal command groups. With many
  small allocations the barrier pattern differs from few large ones,
  and a missing barrier could leave a downstream read picking up
  uninitialised memory.

The "all-NaN with +1e30 ranges" signature suggests reading
uninitialised buffer contents rather than NaN-producing arithmetic.

## Reproducer status

Same as the related drift report — repro currently lives in our
diff harness against the chatterbox model. We can extract a minimal
`test-backend-ops` case showing a graph with mul_mat pinned to CPU
producing NaN at large T but not small T if the maintainers flag
this issue.
