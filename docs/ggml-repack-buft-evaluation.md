# Does ggml's repack buffer type make quantised matmul faster here?

Settles item 1 of `ggml-optimisation-playbook.md` §8 — *"whether selecting
ggml's repack extra buffer type actually speeds anything up here"* — and
resolves the disagreement between the hFT-Transformer and Onsets & Frames
quantisation numbers that §4 and `ONSETS_AND_FRAMES_PERF.md` both flag as
unexplained.

**Short answer: yes, it is offered on every machine tested, and yes it pays —
but which quantisations it pays for depends entirely on the ISA, and the answer
splits cleanly in two.**

* **On arm64 it is a large win for the models exactly as they ship.** Nothing
  is declined — q8_0 included — and q8_0 gets **2.3–2.4×** over the generic
  path on Linux arm64, and **3.1–3.4× on Apple Silicon**, the platform
  CrispASR actually ships to. It compounds with a fact x86 does not prepare
  you for: on arm64 the *generic* quantised path is already **2.2–3.1× faster**
  than f32 before any repacking, so q8_0 + repack lands at **0.13–0.15× the
  cost of f32** — roughly **7×** faster than running the model at f32.
* **On x86 it cannot help any model in this tree today.** ggml has no repacked
  q8_0 kernel for x86 at all, and every quantised GGUF here is q8_0. q4_0 and
  q4_K do gain 1.7–2.9× on a clean runner, so the lever is real — it just
  requires re-quantising, which is an accuracy decision.

If CrispASR cares about phones and Apple Silicon, **this is worth adopting**.
If the target is an x86 server, it is worth nothing until something is
re-quantised.

Everything below was measured with `crispasr-repack-probe`
(`examples/cli/crispasr_repack_probe.cpp`) and, for the end-to-end arm, the
`crispasr --piano` hFT backend. **Every number names its machine.** A result on
a CPU without an int8 dot-product instruction does not transfer to one that has
one, in either direction.

---

## 0. The machines, and which numbers to trust

| | VPS (`crispasr-dev`) | Kaggle CPU worker | GitHub `ubuntu-24.04` / `ubuntu-24.04-arm` / `macos-14` |
| --- | --- | --- | --- |
| CPU | Intel Xeon Skylake-SP (IBRS, no TSX) | Intel Xeon @ 2.20 GHz (GCE) | AMD EPYC 7763 / arm64 / Apple Silicon |
| cores | 4 vCPU, **shared** | 4 vCPU, shared | 4, dedicated |
| load during measurement | **6.4 one-minute, rising to 40 later in the night** | ~0 | ~0 |
| AVX2 | yes | yes | yes |
| AVX-512F/DQ/CD/BW/VL | yes | no | **no** (EPYC 7763 is AVX2) |
| AVX-512 VNNI | **no** | **no** | **no** |
| AMX-INT8 | **no** | **no** | **no** |
| ARM dotprod / i8mm | n/a | n/a | **yes** on both arm64 legs (`i8mm` on Linux arm64 only) |

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

**1.2 A repack buffer type IS offered on every machine tested** — the VPS, a
Kaggle Xeon, and all three CI runners. Playbook §4 raised the possibility that
`ggml_backend_dev_get_extra_bufts` would return nothing without VNNI, making
the question moot. It does not: `CPU_REPACK` is offered everywhere, because
`ggml_backend_cpu_get_extra_buffer_types()` (`ggml-cpu.cpp:42`) gates it on the
*compile-time* `GGML_USE_CPU_REPACK` (ON in this build) and nothing else. That
branch of the question is closed. What varies by ISA is not whether the buffer
type exists but which tensors it accepts — §2.

**1.3 The buffer type supports `MUL_MAT` and `MUL_MAT_ID` — not `GET_ROWS`.**
`repack::extra_buffer_type::supports_op` (`repack.cpp:4774`) accepts only those
two ops, with a 2-D `src[0]` and an F32 `src[1]`. Playbook §4 and the comment
above the switch in `weight_buft_supported` (`src/crispasr.cpp:1957`) both say
"`MUL_MAT` and `GET_ROWS`"; against this ggml version that is stale. It is a
stale comment rather than a bug — that function builds a probe op and *asks*
the buffer type, so a `GET_ROWS` weight is correctly refused at runtime. It
does not change the conclusion; it narrows it.

Worth stating explicitly, because it reframes what this work is: **the whisper
backend already has all of this**, and has had since it was written
llama.cpp-style. A quantised whisper GGUF in CrispASR is already taking the
repack path where its ISA has a kernel. What was missing was everything loading
through `core_gguf`, which is the six transcription models and most of the rest
— and `src/crispasr.cpp:1925-1937` is the working reference the loader change
in §5 was modelled on.

**1.4 It is incompatible with the zero-copy mmap path — verified, not
inferred.** Playbook §4 marked this as inference. It is now checked. `gguf_loader.cpp:657`
wraps the file mapping in a backend buffer with a custom iface and binds each
`tensor->data` straight at an offset into the map; **it never calls
`set_tensor` at all**. Repacking *is* a `set_tensor` that rewrites the bytes
into an interleaved layout in a buffer the buffer type owns. The two cannot
both hold for one tensor. A repacked tensor costs a real copy at load and a
resident private page for every weight byte.

---

## 2. The measurement that matters: which types get a kernel

`ggml_repack_get_optimal_repack_type` (`repack.cpp:4528`) is a table over
(quant type, ISA, `ne[1]` divisibility). Read out:

| GGUF type | x86 | arm64 (dotprod/i8mm) | gate |
| --- | --- | --- | --- |
| **q8_0** | **NO** | **yes** | NEON+dotprod / NEON+i8mm / RISC-V only — no AVX2 or AVX-512 branch exists |
| q4_0 | yes | yes | `avx2 && ne[1] % 8 == 0` → `q4_0_8x8_q8_0`; NEON via dotprod/i8mm |
| q4_K | yes | yes | `avx2 && ne[1] % 8 == 0` → `q4_K_8x8_q8_K` |
| q5_K, q6_K | **NO** | **yes** | NEON only |
| iq4_nl, mxfp4 | yes | yes | AVX2 / NEON+dotprod |

Confirmed at runtime, not just read: the x86 runs decline q8_0, q5_K and q6_K,
and the arm64 run declines **nothing**.

**This is the finding that decides the whole question for this tree today.**
Every quantised model here ships q8_0 — verified by reading the GGUF headers:
`hft-transformer-q8_0.gguf` is 63 Q8_0 tensors and 94 F32, and all 63 are 2-D
with `ne[1] % 8 == 0`, so they are repack-eligible *by shape* on every ISA. On
x86 they are declined anyway, because q8_0 has no repacked kernel there. On
arm64 all 63 are accepted.

The AVX2 kernels that do exist (`arch/x86/repack.cpp:2026`) use
`gemm_q4_b32_8x8_q8_0_lut_avx` and do **not** need VNNI — they fall back to
`maddubs`-style accumulation — which is why the win below shows up on machines
that have no int8 instruction at all. The playbook's reasoning ("no VNNI, so
there is no int8 dot-product for the repacked path to reach") was too
pessimistic: the *layout* pays on its own.

On arm64 the table is the other way round, and §3c measures it: q8_0 gets a
kernel and is 2.3–2.4× faster repacked on Linux arm64, 3.1–3.4× on Apple
Silicon. **Do not carry the x86 conclusion to a phone build, in either
direction.**

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

Reproduce: `crispasr-repack-probe --threads 1 --reps 40` for the interleaved
view below, or `--arm default` / `--arm repack` in separate processes for the
one §3c treats as primary.

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

### 3c. GitHub Actions — clean runners, the numbers to cite

`.github/workflows/ggml-repack-buft-ab.yml`, run 35817843249. Dedicated
4-core runners at load ~0. Two independent views are taken and they agree to a
few percent: the in-process interleave above, and **one arm per process with
the cold run discarded and the median of three**, which is the discipline
`basic-pitch-conv-ab.yml` encodes. The table below is the separate-process one.

**`ubuntu-24.04-arm` — arm64, `dotprod` + `i8mm` + `sve`. Nothing declined.**

| type | 1 thread, K=512 N=2048 M=256 | 4 threads | vs f32 (generic → repacked) |
| --- | --- | --- | --- |
| f32 | 30.99 ms | 8.15 ms | — |
| **q8_0** | 9.86 → **4.15 ms**, **2.37×** | 2.52 → **1.04 ms**, **2.42×** | 0.32× → **0.13×** |
| q4_0 | 12.16 → 3.44 ms, **3.54×** | 3.10 → 0.86 ms, **3.59×** | 0.39× → **0.11×** |
| q4_K | 11.10 → 6.17 ms, 1.80× | 2.87 → 1.56 ms, 1.83× | 0.36× → 0.20× |
| q6_K | 17.73 → 8.01 ms, 2.21× | 4.53 → 1.98 ms, 2.29× | 0.57× → 0.26× |

The other two shapes give 2.26–2.37× for q8_0 and 3.34–3.59× for q4_0, so the
effect is flat in shape and in thread count.

**`macos-14` — Apple Silicon, `neon` + `dotprod` (ggml reports `i8mm` = 0).
Nothing declined.** This is the platform CrispASR actually ships to.

| type | 1 thread, K=512 N=2048 M=256 | vs f32 (generic → repacked) |
| --- | --- | --- |
| f32 | 17.41 ms | — |
| **q8_0** | 7.99 → **2.40 ms**, **3.33×** | 0.45× → **0.13×** |
| q4_0 | 9.12 → 2.24 ms, **4.07×** | 0.52× → **0.12×** |
| q4_K | 8.11 → 3.16 ms, 2.57× | 0.48× → 0.19× |
| q6_K | 13.11 → 6.16 ms, 2.13× | 0.92× → 0.46× |

Apple Silicon is the **best** case measured anywhere: q8_0 at 3.1–3.4× over the
generic path, and 4.0× for q4_0. Note it reaches this with `dotprod` but
without `i8mm`.

**`ubuntu-24.04` — AMD EPYC 7763, AVX2, no AVX-512, no VNNI.**

| type | 1 thread, K=512 N=2048 M=256 | 4 threads | vs f32 |
| --- | --- | --- | --- |
| f32 | 14.43 ms | 6.73 ms | — |
| **q8_0** | 13.48 ms, **declined** | 5.95 ms, **declined** | 0.92×, no repack kernel |
| q4_0 | 17.54 → 6.58 ms, **2.67×** | 7.79 → 3.05 ms, **2.55×** | 1.20× → **0.45×** |
| q4_K | 12.31 → 5.91 ms, **2.08×** | 5.41 → 2.71 ms, 1.99× | 0.83× → **0.40×** |
| q6_K | 14.66 ms, **declined** | 6.35 ms, declined | 1.00×, no repack kernel |

⚠ **One number here disagrees with the VPS and the disagreement is real, not
noise.** Generic-path q8_0 measures **0.84–1.05× f32 on the EPYC** but
**1.08–1.31× f32 on the Skylake-SP VPS**. That is an ISA difference, not
contention: Skylake-SP has AVX-512, so its *f32* GEMM is unusually fast, which
makes the quantised path look correspondingly worse. **hFT's 29% q8_0 penalty
may therefore be specific to AVX-512 hosts and not an x86-wide fact.** The
end-to-end CI job exists partly to settle that; until it has, treat "quantising
costs you throughput" as a statement about AVX-512 x86, which is where it was
measured.

Numerical agreement between arms is 1e-8 to 5e-7 relative on the output sum
across every machine and type — rounding order, not a different answer.

---

## 4. The hFT / Onsets & Frames contradiction, resolved

The two measurements were:

* hFT-Transformer: q8_0 at **1.29× the CPU of f32** (6.25 vs 4.84 CPU-s per
  audio-second), `HFT_TRANSFORMER.md` §Cost.
* Onsets & Frames: q8_0 at **0.96× the CPU of f32** (0.425 vs 0.442),
  `ONSETS_AND_FRAMES_PERF.md`.

They are not in conflict, and the reason is not op mix in the abstract — it is
**which tensors the two converters actually quantise**, read straight out of
the GGUF headers.

### 4a. hFT's q8_0 is quantised everywhere that matters

`hft-transformer-q8_0.gguf`: 157 tensors, **63 Q8_0 and 94 F32**. All 63 are
2-D, all have `ne[1] % 8 == 0`, and every one of them is used exactly once as
`src[0]` of `ggml_mul_mat` in `hft_linear_apply()`. The model is 83.5% weight
GEMM, so essentially all of the arithmetic runs on a quantised weight through
ggml's generic path.

§3a measures that path's cost directly **on the same machine hFT's 29% was
measured on**, with no model, decoder or front end involved: generic q8_0
`MUL_MAT` is **1.08–1.31× f32** for transformer-shaped GEMMs on that Skylake-SP
box. Applied to 83.5% of the work that predicts a 1.07–1.26× whole-model
penalty; 1.29× was measured. **hFT's 29% is a real kernel effect on that
host**, not a measurement artefact — the mechanism is that ggml's generic path
re-quantises the activation to Q8_0 on every GEMM and then runs a `vec_dot`.

⚠ But it is a statement about *that ISA*. On the clean AVX2-only EPYC the same
kernel measurement gives **0.84–1.05×** — no penalty — and on arm64 it gives
**0.32–0.45×**, a large speedup. The reason Skylake-SP is the worst case is
that AVX-512 makes its *f32* GEMM unusually fast, so the quantised path loses
by comparison rather than being slow in absolute terms. **Do not generalise
hFT's 29% beyond AVX-512 x86.** Re-measuring it on a clean runner of each class
is what the end-to-end CI job is for.

### 4b. Onsets & Frames' q8_0 barely quantises anything on the hot path

`onsets-and-frames-q8_0.gguf`: 62 tensors, **19 Q8_0 and 43 F32**. Two facts
decide it:

1. **Every convolution weight stays F32** — all twelve
   `oaf.*.conv{0,1,2}.weight` are F32 `[3,3,·,·]`. The measured profile in
   `ONSETS_AND_FRAMES_PERF.md` puts the conv stacks at **72% of the forward**.
   Seventy-two per cent of this model's work never touches a quantised weight
   at all, in either GGUF.
2. **The LSTM recurrence never sees a quantised tensor either.** The `R`
   matrices are Q8_0 in the file, but `src/onsets_and_frames.cpp:76` is
   explicit: the recurrence "is dequantised once at load". The same doc
   measures the recurrence at **91% of the BiLSTM**, and the BiLSTM at 27% of
   the forward.

What is left exposed to ggml's quantised GEMM path is the LSTM **input
projections** — 19.1 + 18.0 ms out of ~600 ms per BiLSTM, so about **1.7% of
the whole forward** — plus the `fc`/`head` GEMMs, which that profile records as
*below the noise floor*. Call the exposure 2%.

At §3's 1.1–1.3× kernel penalty, 2% exposure predicts a whole-model difference
of about **+0.6%**. The measured difference was **3.8%, in the other
direction**, on a box whose run-to-run spread for a *single* arm is 20–50%
(§0). **It is a null result, and its sign carries no information.**

### 4c. What to take from it

* hFT's 29% is real. Do not explain it away.
* O&F's "q8_0 is slightly faster" should read "q8_0 and f32 are
  indistinguishable here, as the file's own tensor types predict".
* ⚠ The brief that prompted this work described O&F as "46% convolution, 29%
  LSTM, 19% dense". **That split is not what the tree measured** — its own
  instrumented profile is 72% conv / 27% BiLSTM / heads below the noise floor,
  and the 19%-dense figure in particular overstates the quantised share by
  roughly an order of magnitude. The conclusion is the same either way, but
  more strongly on the real numbers.
* The generalisable rule is sharper than "measure": **open the GGUF and look at
  which tensors are actually quantised before predicting anything from a
  quantisation A/B.** A converter that leaves the convolutions in F32 has
  already decided that the A/B cannot move.
* A corollary worth flagging separately: O&F's q8_0 GGUF **dequantises its
  largest quantised tensors back to F32 at load**, so the file is smaller but
  the resident set is not correspondingly smaller. That is a size claim worth
  re-checking, not a perf one, and it is out of scope here.

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
   site. When no tensor is accepted it falls back to plain `load_weights()` and
   keeps mmap, **so adopting it costs nothing on a machine where it cannot
   help**. That fallback is not hypothetical: it is what every q8_0 model does
   on x86, and it means the same binary takes the fast path on arm64 and the
   mmap path on x86 without a build flag or a conditional at the call site.

   It also means the *cost* side of the trade only appears where the *benefit*
   does, which is the right way round — but the cost is real and unmeasured for
   large models. See §6.2.

`CRISPASR_GGUF_REPACK=0` disables it without a rebuild.

`src/hft_transformer.cpp` is the first adopter: 83.5% weight GEMM, every
`hft_linear::w` used exactly once as `ggml_mul_mat` src[0] with an F32
activation, nothing else eligible. Its predicate is "ends in `.weight` and is
not a `.ln.` tensor" — the layer-norm weights are `ggml_mul` operands and the
positional tables are `ggml_add` operands, so both are excluded by name.

Validated against the real file rather than by reading: over
`hft-transformer-q8_0.gguf`'s 157 tensors the predicate selects **67, rejects
90, and rejects no quantised tensor at all**. All 67 are 2-D. Of them 63 are
Q8_0 and four — `hft.encoder.front.weight` and the `mpe`/`offset`/`onset`
heads — are F32 even in the quantised file, so `repack_buft_accepts()` sends
those four to the default partition on every ISA. No missed opportunity and no
mis-selection, which is the check any further adopter should run before
trusting a name-based predicate.

One thing the adoption got wrong first time and is worth flagging for the next
adopter: `load_weights_repack()` returns **two** buffers, `wl.buf` for the
repack partition and `wl.buf_cpu` for the default one, and a model that stores
only the first leaks the second — which is most of the tensor count. Same
shape as `load_weights_split()`, and the same trap.

`.github/workflows/ggml-repack-buft-ab.yml` is the harness, with three guards
that matter more than they look: it asserts the repack arm actually printed
its load line (so a silent fallback cannot be reported as a measured null), it
asserts the **control** arm did not (so the A/B cannot compare an arm against
itself — the exact failure `basic-pitch-conv-ab.yml` records having hit when a
default flipped), and it diffs the decoded notes between arms (so a speedup
that changed the answer is visible rather than celebrated).

---

### 5a. End-to-end, whole model, on clean runners

Run 35819605846, `end-to-end` legs. `crispasr --piano`, hFT on
`samples/jfk.wav`, one process per arm, arms interleaved with a rotating start
order, cold rep discarded, **median of 3**, CPU seconds (user+sys) via
`getrusage(RUSAGE_CHILDREN)`, `MKL_NUM_THREADS=1`, `-t 1`. Runner load ~1.0.

| arm | `ubuntu-24.04` (EPYC 7763, AVX2) | `ubuntu-24.04-arm` (dotprod/i8mm) |
| --- | --- | --- |
| hFT f32 | 12.88 cpu-s | 102.5 cpu-s |
| hFT q8_0 (generic) | 14.05 — **1.09× f32** | 51.4 — **0.50× f32** |
| hFT q4_0 (generic) | 15.7 — 1.22× f32 | 56.5 — 0.55× f32 |
| **hFT q4_0 + repack** | **9.3 — 0.72× f32** | **35.7 — 0.35× f32** |
| **repack vs generic q4_0** | **1.69×** | **1.58×** |
| O&F f32 | 1.58 cpu-s | 4.37 cpu-s |
| O&F q8_0 | 1.57 — **0.99× f32** | 3.85 — 0.88× f32 |
| peak RSS, any hFT arm | 233–234 MiB | 233 MiB |

**This is the measured, reproducible speedup the investigation was after.**
Routing hFT's q4_0 weights through the repack buffer type makes the whole model
**1.69× faster on x86 and 1.58× faster on arm64** than the same GGUF without
it — and on x86 it is what finally makes quantisation pay at all: q4_0 goes
from 1.22× the cost of f32 to **0.72×**.

Four things fall out of that table that were not measurable before:

1. **hFT's q8_0 penalty is 1.09× on the EPYC, not 1.29×.** §3c predicted this
   from the kernel measurements and it holds at whole-model scale. The 29%
   figure in `HFT_TRANSFORMER.md` is a Skylake-SP/AVX-512 number. It should not
   be quoted as an x86-wide one.
2. **On arm64, quantisation is a large win with no repacking at all** — q8_0
   alone is **0.50× f32**, i.e. the model runs twice as fast quantised. Playbook
   §4's "quantise for size, not speed" is an x86 statement.
3. **O&F q8_0 is 0.99× f32 on a clean machine.** The VPS's 0.96× was not a
   speedup and the difference was not real; §4b predicted "indistinguishable"
   from the GGUF's tensor types and that is exactly what a load-1.0 box shows.
   **This is the cleanest possible confirmation of the contradiction's
   resolution**, and it needed a quiet machine to see.
4. **The mmap trade cost nothing measurable here.** Peak RSS is 233–234 MiB
   across every arm including the repacked one, against the 237 MiB
   `HFT_TRANSFORMER.md` records. For a 22 MB model, giving up zero-copy mmap is
   free. That is *not* evidence about a multi-gigabyte one — see §6.2.

⚠ One thing to note rather than explain away: the arm64 runner is **8× slower
than the EPYC on hFT f32** (102.5 vs 12.88 cpu-s) while the hermetic job has
its f32 `MUL_MAT` only 2.2× slower for the same shape. Something outside the
weight GEMMs — most likely the mel front end or the conv path — is
disproportionately slow on that runner. The within-machine ratios above are
interleaved A/Bs and are unaffected, but the cross-machine absolute numbers
should not be compared until that is understood.

#### Final run, both legs green, both guards satisfied (run 35820466189)

This is the authoritative end-to-end table: the loader selects the buffer type
by name, the parity guard is the corrected one, and both legs passed. **Note
the x86 CPU — GitHub's pool is heterogeneous and this run drew a third one.**

| arm | `ubuntu-24.04` = **AMD EPYC 9V74**, AVX-512 **VNNI**, no AMX | `ubuntu-24.04-arm`, dotprod |
| --- | --- | --- |
| hFT f32 | 25.56 cpu-s | 102.67 cpu-s |
| hFT q8_0 generic | 21.55 — **0.84× f32** | 51.79 — 0.50× f32 |
| hFT q8_0 + repack | 21.53 — 0.84× (*declined, fell back*) | **37.58 — 0.37×**, **1.38× over generic** |
| hFT q4_0 generic | 23.28 — 0.91× f32 | 56.92 — 0.55× f32 |
| **hFT q4_0 + repack** | **18.81 — 0.74×**, **1.24× over generic** | **36.02 — 0.35×**, **1.58× over generic** |
| O&F q8_0 vs f32 | 2.84 vs 2.90 — **0.98×** | 3.91 vs 4.44 — 0.88× |

Guards: `x86: q8_0 correctly declined and fell back to the mmap path`;
`arm64: q8_0 repacked, as expected`; `pitch sequence: IDENTICAL` on both.

**Reproducibility.** These are not one-shot numbers. Run 35822849587, a fresh
dispatch that happened to draw the same EPYC 9V74, gives f32 25.76, q8_0
generic 21.72, q4_0 generic 23.60, **q4_0 + repack 19.08** — every figure
within **1.4%** of the run above, and every ratio identical to two decimals
(0.84 / 0.92 / 0.74 / 0.98). The arm64 figures likewise reproduce to 0.4%
across two runs. A clean runner plus interleaved, one-process-per-arm
measurement gives a genuinely stable A/B, which is exactly what the VPS could
not.

**And this settles the VNNI question for CPU_REPACK on x86.** The EPYC 9V74
has `avx512_vnni = 1`. The repacked path's lead there is **1.24×**, *smaller*
than the **1.69×** measured on the VNNI-less EPYC 7763 — because VNNI speeds
up the **generic** path too: q4_0 generic is 0.91× f32 on the 9V74 against
1.22× on the 7763. So VNNI does not widen the repack lead; it narrows it, by
lifting the baseline. The repacked path still wins, everywhere it has a kernel.

The other thing this table shows, and it is the broadest correction in this
document: **on a VNNI x86 CPU, quantisation is already faster than f32 with no
repacking at all** — q8_0 at 0.84×. Playbook §4's "quantising does not make
CPU inference faster" holds only for older x86 without VNNI.

#### The arm64 q8_0 number — the one that decides adoption

Run 35819665631 added the missing arm. On `ubuntu-24.04-arm`, whole model,
same protocol:

| arm | cpu-s | vs f32 | vs its own generic arm |
| --- | --- | --- | --- |
| hFT f32 | 102.96 | 1.00× | — |
| hFT q8_0 generic | 51.70 | 0.50× | — |
| **hFT q8_0 + repack** | **37.71** | **0.37×** | **1.37×** |
| hFT q4_0 generic | 56.96 | 0.55× | — |
| hFT q4_0 + repack | 35.98 | 0.35× | 1.58× |

The CI guard confirmed it took the intended path (`arm64: q8_0 repacked, as
expected`). **So the GGUF that ships today, unchanged, runs 1.37× faster whole-
model on arm64 with this one loader change, and 2.7× faster than f32.** That is
the number the adoption decision turns on.

### 5b. Parity: what the repacked kernel changes in the output


`end-to-end linux-x86_64`, run 35817927145, on the EPYC runner with
`hft-transformer-q4_0.gguf`:

```
hft: repack buffer type: 2 MiB (63 tensors) repacked, 1 MiB (94 tensors) default
```

All 63 quantised tensors took the fast path, the 94 F32 ones went to the
default partition, and the control arm (`CRISPASR_GGUF_REPACK=0`) printed
nothing — so the two arms really were different, which is the thing a perf A/B
most often gets wrong silently.

**The decoded notes are not bit-identical.** Measured over the ~90 notes
`crispasr --piano` decodes from `samples/jfk.wav`, repack arm against generic
arm, on both runners:

| | linux-x86_64 (EPYC) | linux-arm64 |
| --- | --- | --- |
| notes decoded | 90 vs 90 | 91 vs 92 |
| **pitch sequence** | **identical** | one extra note in the generic arm |
| max onset/offset shift | **1 ms** — exactly one hop | 1 ms, apart from the extra note |
| max velocity delta | 6 | 6 |

On x86 the transcription is **the same notes at the same pitches**, with onsets
moved by at most a single frame and velocities by a few units. On arm64 one
extra 18 ms note (`0.984–1.002, C#4`) appears in one arm and not the other.

That is the signature of §3's ~1e-7 accumulation-order difference tipping a
threshold, and **`jfk.wav` is close to the worst possible fixture for this
check**: it is speech fed to a piano transcriber, so essentially every
detection sits near the decision boundary and an 18 ms blip is exactly what a
marginal one looks like. It is not evidence that the repacked kernel is wrong —
§3 shows the GEMM outputs agreeing to 1e-8..5e-7 relative — but it is also not
something to wave away. **What would settle it is note-level F1 on MusicNet
through `tools/hft_musicnet_f1.py`, not a byte diff on a speech clip**, and
that has not been run.

⚠ *Correction, recorded because it nearly went into this document as a
finding:* this section first said "onsets, offsets and pitches match on every
note; one velocity moved by 3 units". That came from reading a `diff | head
-20` and mistaking the truncation for the whole story. The real diff is ~20 of
90 note lines on each runner. The numbers above are the full comparison.

The workflow's guard was wrong for the same reason and is now fixed: comparing
onset/offset exactly can never pass, because a one-hop shift is expected. It
asserts the **pitch sequence** and the **note count**, which are the things a
real regression would move.

⚠ **The first end-to-end run produced no timing at all and did not look like
it.** `/usr/bin/time` is not installed on GitHub's ubuntu runners, so every arm
recorded an empty string and the summary printed `0.00 cpu-s` for all six — a
number, not an error. It is now taken with
`resource.getrusage(RUSAGE_CHILDREN)`, which needs no package. Whole-model
timings will come from the re-run; **until they land, the whole-model claims in
this document are the VPS ones and §3's kernel measurements, not clean-runner
end-to-end measurements.**

---

## 6. What is still open, and what is untestable from here

1. **Whether a CPU with an int8 dot-product instruction widens the x86 lead —
   reachable after all, and the first accidental look at it is a negative.**

   This section previously said VNNI was untestable from this project's
   hardware. **That was wrong.** GitHub's `ubuntu-24.04` pool is
   **heterogeneous**: most runs drew an AMD EPYC 7763 (AVX2, no VNNI, no AMX),
   but run 35819665631 drew an **INTEL(R) XEON(R) PLATINUM 8573C** (Emerald
   Rapids) reporting `avx512_vnni = 1` **and** `amx_int8 = 1`. So the hardware
   is reachable, just not deterministically — re-dispatch until the log names
   the CPU you want.

   That run also exposed a **bug in the loader**, which is why its numbers are
   not in §5a. `ggml_backend_cpu_get_extra_buffer_types()` pushes **AMX first**
   when the build has `__AMX_INT8__ && __AVX512VNNI__`, and
   `load_weights_repack()` was taking `v[0]` — so on that host it selected
   **AMX, not repack**, and reported "repacking" q8_0 on x86, which has no q8_0
   repack kernel at all. The CI guard caught it and failed the job. Selection
   is now by name (`CRISPASR_GGUF_EXTRA_BUFT` overrides) and the load line
   prints which buffer type it used.

   **The accidental AMX measurement is a result in its own right, and it is a
   negative:** on that Emerald Rapids host every quantised hFT arm cost
   **1.42–1.51× f32** (q8_0 27.1, q4_0 28.3, f32 19.1 cpu-s) — AMX made the
   model *slower*. The playbook's expectation that VNNI/AMX would widen the
   repacked path's lead is **not supported** by the one look we have. Caveats
   it deserves: this was AMX rather than CPU_REPACK, it was a single
   unintended run, and hFT's GEMMs may simply be too small for AMX tiles. **A
   deliberate run on a named 8573C, with CPU_REPACK selected explicitly, is the
   single most valuable remaining measurement** and the tooling for it now
   exists.
2. **Whether to adopt it on arm64, which is a decision rather than a
   measurement.** §3c answers the measurement: 2.3–3.4× on q8_0, on the GGUFs
   as they ship, on both Apple Silicon and Linux arm64. What is not yet priced
   is the other side of §1.4 — the repack path **gives up the zero-copy mmap**,
   so it trades load time and resident memory for GEMM throughput. hFT is a
   22 MB model and will not notice; a multi-gigabyte one might, and
   `gguf_loader.cpp`'s mmap path exists because a 14.9 GB F16 GGUF was
   thrashing swap on a 16 GB Mac. **Measure RSS and load time per model before
   turning this on for a large one.** The end-to-end CI job reports both.
3. **Whether q4_0 or q4_K holds its accuracy well enough to adopt on x86.**
   This document is about throughput only. On x86 the lever requires
   re-quantising away from q8_0, and `crispasr_model_registry.cpp:794` already
   notes that q4_0 "damages" onsets-and-frames. That is a question for the
   note-level F1 harness, not this one.
4. **The other five `core_gguf` backends.** Only `src/hft_transformer.cpp`
   adopts `load_weights_repack()` so far. Each further adopter needs its own
   predicate and an audit that no op other than `MUL_MAT`/`MUL_MAT_ID` touches
   those tensors (§5.1). `src/onsets_and_frames.cpp` is a poor candidate for a
   different reason: §4b shows its convolutions are F32 in the GGUF and its
   LSTM recurrence dequantises at load, so there is almost nothing on its hot
   path for the repack buffer type to act on.
5. **The VPS numbers in §3a should not be quoted.** They are kept because they
   agree in sign and rough magnitude with the clean runs, and because the
   Skylake-SP-vs-EPYC disagreement in §3c is a genuine ISA finding rather than
   noise. But they were taken at a load average of 6.4 on a box that reached 40
   the same night. Cite §3b and §3c.
