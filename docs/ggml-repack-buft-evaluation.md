# Does ggml's repack buffer type make quantised matmul faster here?

Settles item 1 of `ggml-optimisation-playbook.md` §8 — *"whether selecting
ggml's repack extra buffer type actually speeds anything up here"* — and
resolves the disagreement between the hFT-Transformer and Onsets & Frames
quantisation numbers that §4 and `ONSETS_AND_FRAMES_PERF.md` both flag as
unexplained.

**Short answer: yes, it is offered, and yes it pays — a lot — but not for the
quantisation every model in this tree actually ships.** On x86 ggml has no
repacked q8_0 kernel at all, so the models that exist today gain nothing. q4_0
and q4_K gain 1.4–3.4×, which is enough to turn quantisation from a throughput
*loss* into a throughput *win* against f32.

Everything below was measured with `crispasr-repack-probe`
(`examples/cli/crispasr_repack_probe.cpp`) and, for the end-to-end arm, the
`crispasr --piano` hFT backend. **Every number names its machine.** A result on
a CPU without an int8 dot-product instruction does not transfer to one that has
one, in either direction.

---

## 0. The machines, and which numbers to trust

| | VPS (`crispasr-dev`) | Kaggle CPU worker | GitHub `ubuntu-24.04` / `ubuntu-24.04-arm` / `macos-14` |
| --- | --- | --- | --- |
| CPU | Intel Xeon Skylake-SP (IBRS, no TSX) | Intel Xeon @ 2.20 GHz (GCE) | see §3c — printed per run |
| cores | 4 vCPU, **shared** | 4 vCPU, shared | 4, dedicated |
| load during measurement | **6.4 one-minute, rising to 40 later in the night** | ~0 | ~0 |
| AVX2 | yes | yes | yes |
| AVX-512F/DQ/CD/BW/VL | yes | no | printed per run |
| AVX-512 VNNI | **no** | **no** | printed per run |
| AMX-INT8 | **no** | **no** | printed per run |
| ARM dotprod / i8mm | n/a | n/a | the two arm64 legs |

⚠ **Read the load row before the numbers.** The VPS is a shared 4-vCPU box that
was carrying a load average of 6 when §3a was taken and reached 40 later the
same night. Interleaving the arms — which every measurement here does — removes
*some* of that error, but not memory pressure and not cache thrash, and no
number taken there is fit to quote on its own.
`BASIC_PITCH_CONV_PERF.md` records the same trap from the other side: a
threading win that was invisible on this VPS measured 3.6–3.8× on a clean
runner. **The Kaggle and CI numbers are the ones to cite; the VPS numbers are
kept because they agree, and are labelled so nobody quotes them as primary.**

### 0a. What the runners actually are — and the ISA result that came free

`ubuntu-24.04` was expected to be Ice Lake / Cascade Lake class and therefore to
carry AVX-512 VNNI, the instruction this whole question turns on. **It is not.**
Read straight out of the job log:

| runner | CPU | avx2 | avx512f | avx512_vnni | amx_int8 | asimddp | i8mm | sve |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ubuntu-24.04` | **AMD EPYC 7763** (Zen 3) | 1 | 0 | 0 | 0 | — | — | — |
| `ubuntu-24.04-arm` | (arm64) | — | — | — | — | **1** | **1** | **1** |

So **four** machines have now been checked — the VPS (Skylake-SP), a Kaggle GCE
Xeon, and a GitHub EPYC 7763 — and **not one x86 machine reachable from here has
an int8 dot-product instruction.** The VNNI/AMX question is not "untested
because nobody ran it"; it is untestable from this project's available hardware.
Say that rather than implying a negative.

The arm64 leg is the more valuable one in any case, and is the reason the
workflow has three legs rather than one. `asimddp` + `i8mm` is exactly where
ggml's repack table gives **q8_0** a kernel (§2) — and q8_0 is the quantisation
every GGUF in this tree actually ships. **x86 has already answered "this lever
cannot help the models as they exist"; arm64 is where it might.**

---

## 1. The mechanism, verified

ggml's repacked GEMM (`ggml/src/ggml-cpu/repack.cpp`) is reached by putting the
**weight** in the CPU device's *extra* buffer type. Four facts, each checked
against this tree rather than assumed:

**1.1 The dispatch needs no scheduler.** `ggml_compute_forward`
(`ggml-cpu.c:1752`) calls `ggml_cpu_extra_compute_forward`, which walks the
extra buffer types and asks each whether it owns `op->src[0]->buffer->buft`
(`traits.cpp:12`). It is keyed purely on the weight's buffer type. A model
driving a raw `gallocr` — `src/hft_transformer.cpp`, `src/crepe.cpp` — gets the
fast path exactly as a `ggml_backend_sched` model does. The playbook did not
claim otherwise, but it is worth stating, because it is why a loader change is
sufficient and no model needs restructuring.

**1.2 A repack buffer type IS offered on both machines.** §4 raised the
possibility that `ggml_backend_dev_get_extra_bufts` would return nothing
without VNNI, making the question moot. It does not: `CPU_REPACK` is offered on
both, because `ggml_backend_cpu_get_extra_buffer_types()`
(`ggml-cpu.cpp:42`) gates it on the *compile-time* `GGML_USE_CPU_REPACK`
(ON here) and nothing else. That branch of the question is closed.

**1.3 The buffer type supports `MUL_MAT` and `MUL_MAT_ID` — not `GET_ROWS`.**
`repack::extra_buffer_type::supports_op` (`repack.cpp:4774`) accepts only those
two ops, with a 2-D `src[0]` and an F32 `src[1]`. §4 and the comment at
`src/crispasr.cpp:1945` both say "`MUL_MAT` and `GET_ROWS`"; against this ggml
version that is stale. It does not change the conclusion — it narrows it.

**1.4 It is incompatible with the zero-copy mmap path — verified, not
inferred.** §4 marked this as inference. It is now checked. `gguf_loader.cpp:657`
wraps the file mapping in a backend buffer with a custom iface and binds each
`tensor->data` straight at an offset into the map; **it never calls
`set_tensor` at all**. Repacking *is* a `set_tensor` that rewrites the bytes
into an interleaved layout in a buffer the buffer type owns. The two cannot
both hold for one tensor. A repacked tensor costs a real copy at load and a
resident private page for every weight byte.

---

## 2. The measurement that matters: which types get a kernel

`ggml_repack_get_optimal_repack_type` (`repack.cpp:4528`) is a table over
(quant type, ISA, `ne[1]` divisibility). Read out on x86:

| GGUF type | x86 repack kernel? | gate |
| --- | --- | --- |
| **q8_0** | **NO** | NEON+dotprod / NEON+i8mm / RISC-V only — no AVX2 or AVX-512 branch exists |
| q4_0 | yes | `ggml_cpu_has_avx2() && ne[1] % 8 == 0` → `q4_0_8x8_q8_0` |
| q4_K | yes | `ggml_cpu_has_avx2() && ne[1] % 8 == 0` → `q4_K_8x8_q8_K` |
| q5_K, q6_K | **NO** | NEON only |
| iq4_nl, mxfp4 | yes | AVX2 |

**This is the finding that decides the whole question for this tree today.**
Every quantised model here ships q8_0, and q8_0 has no repacked kernel on x86.
The AVX2 kernels that do exist (`arch/x86/repack.cpp:2026`) use
`gemm_q4_b32_8x8_q8_0_lut_avx` and do **not** need VNNI — they fall back to
`maddubs`-style accumulation — which is why the win below shows up on machines
that have no int8 instruction at all. The playbook's reasoning ("no VNNI, so
there is no int8 dot-product for the repacked path to reach") was too
pessimistic: the *layout* pays on its own.

On arm64 the table is the other way round — q8_0 *does* have a kernel under
`dotprod`/`i8mm`, while q4_K's AVX2 branch obviously does not apply. **Nothing
in this document should be carried to a phone build.**

### 2a. A trap for anyone wiring this up

`ggml_backend_cpu_repack_buffer_set_tensor` (`repack.cpp:4733`) dereferences
`tensor->extra` unconditionally. When `init_tensor` found no kernel for that
(type, shape, ISA) it leaves `extra` null, so writing such a tensor into the
repack buffer type is a **null dereference, not a graceful fallback**. Verified
by crashing it. Tensors must be classified before they are written.

---

## 3. Kernel-level A/B

Single `MUL_MAT`, weight `[K, N]`, activation `[K, M]` F32. Arms interleaved
round-robin with the leading arm alternating, so contention perturbs both
equally (playbook §6.9). `MKL_NUM_THREADS=1` pinned. Best-of-N reported,
because on a shared box the minimum is the closest thing to an uncontended
sample; medians are given in the raw output.

Reproduce: `crispasr-repack-probe --threads 1 --reps 40`.

### 3a. VPS — Skylake-SP, AVX-512F, no VNNI — ⚠ load average 6.4, corroborating only

| shape (K,N,M) | type | generic vs f32 | repacked vs f32 | **repack vs generic** |
| --- | --- | --- | --- | --- |
| 256,256,128 | q8_0 | 1.31× slower | *no kernel* | — |
| | q4_0 | 1.44× slower | **0.85× (faster)** | **1.69×** |
| | q4_K | 3.02× slower | **0.54× (faster)** | **5.62×** |
| | q6_K | 2.27× slower | *no kernel* | — |
| 512,2048,256 | q8_0 | 1.08× slower | *no kernel* | — |
| | q4_0 | 1.22× slower | **0.76×** | **1.61×** |
| | q4_K | 1.83× slower | **0.48×** | **3.83×** |
| | q6_K | 1.71× slower | *no kernel* | — |
| 2048,512,256 | q8_0 | 1.28× slower | *no kernel* | — |
| | q4_0 | 1.39× slower | **0.97×** | **1.44×** |
| | q4_K | 2.14× slower | **0.66×** | **3.23×** |
| | q6_K | 1.81× slower | *no kernel* | — |

### 3b. Kaggle — AVX2-only Xeon, no AVX-512, no VNNI, quiet machine, best of 25

| shape | type | repack vs generic, 1 thread | 4 threads |
| --- | --- | --- | --- |
| 256,256,128 | q4_0 | 2.57× | 2.97× |
| | q4_K | 3.43× | 3.31× |
| 512,2048,256 | q4_0 | 2.36× | 2.84× |
| | q4_K | 2.42× | 2.48× |
| 2048,512,256 | q4_0 | 1.90× | 2.53× |
| | q4_K | 1.98× | 2.04× |

q8_0 and q6_K declined on Kaggle too, confirming §2 is an x86 property and not
a VPS quirk. The Kaggle spread between best and median is under 1%; the VPS's
is 20–50%, which is the load average showing up and is why the two tables do
not agree to the decimal.

Numerical agreement between arms is ~1e-7 relative on the output sum — the
repacked kernel quantises the activation the same way, so this is rounding
order, not a different answer.

---

## 4. The hFT / Onsets & Frames contradiction, resolved

The two measurements were:

* hFT-Transformer: q8_0 at **1.29× the CPU of f32** (6.25 vs 4.84 CPU-s per
  audio-second), `HFT_TRANSFORMER.md` §Cost.
* Onsets & Frames: q8_0 at **0.96× the CPU of f32** (0.425 vs 0.442),
  `ONSETS_AND_FRAMES_PERF.md`.

**The hFT number is real and is now reproduced at the kernel level.** §3 shows
generic-path q8_0 costing 1.08–1.31× f32 for transformer-shaped GEMMs on this
exact box, with no model, no decoder and no front end involved. hFT is 83.5%
weight GEMM, so a 1.1–1.3× kernel penalty across 83.5% of the work is a
1.08–1.26× whole-model penalty. The measured 1.29× sits at the top of that
range. It is a mechanism, not noise: quantised weights force ggml's generic
path to re-quantise the activation to Q8_0 on every GEMM and then run a
`vec_dot`, and on this ISA that costs more than the f32 GEMM it replaces.

**The O&F number is not a contradiction. It is a null result being read as a
sign.** O&F's op mix is 46% convolution, 29% LSTM, 19% dense. Convolution goes
through `ggml_conv_2d`'s im2col and the LSTM is a scalar C++ recurrence;
**neither touches a quantised weight GEMM**, so at most 19% of the work is even
eligible for the q8_0 penalty. Propagating §3's kernel penalty through that mix
predicts a whole-model change of roughly **+2% to +6%** — call it ±0.01 on a
0.44 CPU-s-per-audio-second figure. The measured difference was **0.017, i.e.
3.8%**, on a box whose run-to-run spread for the *same* arm is 20–50% (§3).
The margin is inside the noise floor by a wide margin, and its sign carries no
information.

So the two numbers are not in conflict: one model is dominated by the op that
quantisation penalises and shows the penalty; the other spends 81% of its time
on ops quantisation does not touch and shows nothing, which is exactly what the
op mixes predict.

**What this costs the reader:** `ONSETS_AND_FRAMES_PERF.md`'s "q8_0 is slightly
FASTER than f32 here" should be read as "q8_0 and f32 are indistinguishable
here, as the op mix predicts". Its own caution — "small, one box, one model" —
was the right instinct. §4 of the playbook needs no correction on this point;
its "quantise for size, and measure" advice survives, with the addition that
**which** ops the model spends its time in tells you in advance how much the
measurement can possibly move.

---

## 5. What was changed in the tree

`core_gguf::load_weights_repack()` (`src/core/gguf_loader.{h,cpp}`) — loads a
model with matmul weights in the repack buffer type and everything else in the
default one, and `core_gguf::repack_buft_accepts()`, which asks ggml whether a
given (type, shape) has a kernel on this host rather than duplicating ggml's
dispatch table.

Three constraints from §1 and §2a are handled and not assumed away:

1. **Unsupported ops.** The loader cannot know which tensors are used as
   `MUL_MAT` `src[0]`; only the model can. So the entry point takes a
   predicate, exactly as `src/crispasr.cpp:1945`'s `weight_buft_supported` does
   for the whisper backend by building a probe op. **This is why the playbook's
   "would apply tree-wide in one change" is not right** — every adopting model
   must supply the predicate, and must be checked to make sure no other op
   touches those tensors.
2. **Declined tensors.** Each candidate is classified through
   `repack_buft_accepts()` before it is written, so the null dereference in §2a
   cannot be reached. The answer is cached per (type, ne0, ne1).
3. **mmap.** This path gives up the zero-copy mmap, and says so at the call
   site. When no tensor is accepted — the q8_0-on-x86 case, i.e. every shipping
   quantised model today — it falls back to plain `load_weights()` and keeps
   mmap, so adopting it costs nothing where it cannot help.

`CRISPASR_GGUF_REPACK=0` disables it without a rebuild.

`src/hft_transformer.cpp` is the first adopter: 83.5% weight GEMM, every
`hft_linear::w` used exactly once as `ggml_mul_mat` src[0] with an F32
activation, nothing else eligible.

---

## 6. What remains untestable here, and why

1. **Whether a CPU with an int8 dot-product instruction changes the picture.**
   Neither machine has VNNI, AVX-VNNI or AMX, and neither is arm64. The Kaggle
   arm was pushed specifically to move this variable and drew an AVX2-only
   Xeon. Expect the repacked path to widen its lead where VNNI exists — ggml's
   `mul_sum_i8_pairs` has a `_mm512_dpbusd_epi32` branch
   (`arch/x86/repack.cpp:124`) that neither machine took — but that is a
   prediction, not a measurement.
2. **arm64.** The type table inverts there (§2): q8_0 gains a kernel, and it is
   the quantisation every model in this tree already ships. **On a phone this
   lever may well pay for the models as they exist today, where on x86 it does
   not.** That is the single most valuable untested case and it needs a device
   or a CI runner on `macos-14`/`ubuntu-24.04-arm`.
3. **Whether q4_0 or q4_K is accurate enough to adopt.** This document is about
   throughput. hFT's own table already records q4_0 at the same throughput as
   q8_0 and does not report its F1. Switching a shipping model from q8_0 to
   q4_K to reach this fast path is an accuracy decision that needs the
   note-level F1 harness, not this one.
4. **Nothing in §3 was taken on a clean machine.** The VPS is a shared 4-vCPU
   box carrying a load average of 4–7. The Kaggle worker is quieter but is
   still shared infrastructure. Both arms of every comparison were interleaved
   so contention cannot masquerade as a result, which is the most that can be
   done without a dedicated machine.
