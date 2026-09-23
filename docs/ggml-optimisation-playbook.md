# The ggml optimisation playbook

**What this is.** An inventory of the techniques this tree *already* uses to make
ggml graphs fast, written so that a rewrite of a slow backend can be done by
copying what works here rather than by rediscovering it. Every claim cites a
file and a line. Where something is unproven or I could not verify it, it says
so — a playbook that is confidently wrong is worse than one with gaps marked.

**What it is not.** General ggml advice. Everything below is specific to this
repository as of 2026-09-22.

**Read order.** §0 corrects two premises. The decision table (§1) is the
lookup. §2 (graph lifecycle), §3 (threading) and §4 (quantisation) are the
three patterns you must get right. §5 is the verification you must pass. §6 is
the list of things that went wrong here before. §7 applies all of it to the six
note/piano/transcription backends. §8 lists what is *not* established.

**Related documents, all still authoritative:**

| doc | what it settles |
| --- | --- |
| `../crispasr-crispembed-dev.md` (sibling of the repo root, untracked) | the method: HARD RULES, the port pipeline, A/B discipline. Where it and this file disagree, **it wins**. |
| `docs/LEARNINGS-INDEX.md` → `LEARNINGS.md` | 44 lessons under "ggml graphs, allocation & caching" alone. Grep the heading, then read. |
| `docs/music-transcription/BASIC_PITCH_CONV_PERF.md` | the im2col-vs-direct-SIMD screening test, measured |
| `docs/music-transcription/HFT_TRANSFORMER.md` | the quantisation result (§4 below) and a GEMM-vs-ORT comparison |
| `docs/improvements/SRC_ISA_GAP.md` | `src/` ships baseline x86-64 on every release leg |
| `docs/concurrency.md` | the three layers of parallelism and how they contend |

---

## 0. Two corrections to the brief that prompted this document

The table that motivated this work counted `ggml_build_forward_expand` in one
file per model. That undercounts in two directions, and both matter for the
rewrite plan.

**`src/mt3.cpp` is not a graph-less backend.** It builds seven graphs
(`mt3.cpp:708, 728, 771, 839, 851, 852, 888, 931, 932, 987`), drives them
through a persistent `ggml_backend_sched_t`
(`mt3.cpp:861, 1008, 1046`), and maintains **two** device-resident caches — a
4-D decoder KV cache (`mt3.cpp:788-795`) and a per-layer cross-attention KV
cache built once per encoder length (`mt3.cpp:802-821`). It is one of the
*better* graph backends in the tree, not one of the worst.

**`src/basic_pitch.cpp` is not unthreaded.** Its convolutions — 70% of the
network's arithmetic by `contour_conv` alone — run SIMD-dispatched and threaded
out of `src/core/basic_pitch_conv.h`, and
`docs/music-transcription/BASIC_PITCH_CONV_PERF.md` measures **1.82× from SIMD
and 3.81× at four threads** on a clean CI runner, byte-identical to the
reference loop.

**The general trap: per-file op counting reads the file, not the call graph.**
Work that has been hoisted into `src/core/` — which is exactly what the DRY
refactors were *for* — disappears from a per-file grep. Before concluding a
backend is unoptimised, grep its `core/` includes and follow them. This also
applies to `src/piano_transcription.cpp` (threading lives in
`core/parallel_for.h`), to any backend using `core/mel.h` (already threaded),
and to the whole `*_simdconv.h` family.

---

## 1. Decision table — what this tree uses for each layer type

| layer | what to use | header / reference | notes |
| --- | --- | --- | --- |
| **2-D conv** | *Screen first.* im2col GEMM (`ggml_conv_2d`) **or** direct SIMD | screening test: `docs/music-transcription/BASIC_PITCH_CONV_PERF.md` §7.2; direct-SIMD reference `src/core/basic_pitch_conv.h` | compute `(H·W_out)·(IC·KH·KW)·4` bytes and compare with `OC`. Big matrix + small `OC` ⇒ direct SIMD. Modest matrix + `OC ≥ 32` ⇒ measure `ggml_conv_2d`. |
| **1-D conv, K>1** | `ggml_im2col` + `ggml_mul_mat`, keeping the channel-fastest `(OC, OL, N)` layout | `src/crepe.cpp:167` (`crepe_conv1d_cf`, rationale at `crepe.cpp:150-164`) | **Do not reach for `ggml_conv_1d` reflexively.** It permutes back to `(OL, OC, N)`, materialising the activation every layer, and its `N > 1` batch path is *wrong* in this ggml (`crepe.cpp:21-26`, measured). |
| **1-D conv, K==1** | `ggml_mul_mat(reshape_2d(w), x)` directly — never im2col | `LEARNINGS.md` L14428 item 1 | a K=1 im2col is a pure copy; measured ~300 MB of F32 intermediates at ~75 ms each, ×12 sites, on the qwen3-tts codec. |
| **causal 1-D conv** | `ggml_conv_1d` with symmetric `p = K-1`, then `ggml_view_2d` crop — **never an explicit `ggml_pad_ext` node** | `LEARNINGS.md` L14428 item 2; `src/voxcpm2_tts.cpp` `causal_conv1d_ggml` | Metal rejects left/asymmetric `GGML_OP_PAD`, so a pad node forces a CPU graph split and two cross-backend copies per conv per call. Crop is bit-identical. |
| **asymmetric pad** | convolve symmetric and drop the extra output column | `src/crepe.cpp:12-16` | same Metal constraint, stated for CREPE's `(31,32)` pad. |
| **hand-SIMD conv1d (packed weights)** | `src/core/cpu_packed_conv1d.h` + the `*_simdconv.h` family | see §1a | runtime `__builtin_cpu_supports` dispatch; no `-march` required. |
| **dense / linear** | `ggml_mul_mat` — remember `ggml_mul_mat(A,B) = B × Aᵀ` | `../crispasr-crispembed-dev.md` §Parity | flatten position-wise GEMMs into one `mul_mat` where the positions are independent (done for hFT at `8f76e054`). |
| **BiLSTM / BiGRU** | `src/core/lstm.h` (`lstm_unidir`, `lstm.h:60`) / `src/core/gru.h` — **with the caveat in §1b** | `src/kokoro.cpp`, `src/dots_tts.cpp`, `src/bananamind_tts.cpp` | the input projection is hoisted out of the timestep loop (`lstm.h:23-28, 66-68`); the recurrence is unrolled into the graph, ~12 nodes per timestep. **Node count scales with T — see §1b before adopting for a long-sequence model.** |
| **self-attention (F32-critical)** | `core_sdpa::attn(..., use_flash=false)` | `src/core/sdpa.h:47` | manual `mul_mat`/`soft_max`/`mul_mat` in F32 is the **default**, fused flash is opt-in. Reason at `sdpa.h:4-12`: `ggml_flash_attn_ext` accumulates QKᵀ in F16 and `GGML_PREC_F32` is silently ignored on some kernels (P100/sm_60). |
| **self-attention (robust encoder)** | `ggml_flash_attn_ext`, via `core_attn::encoder_self_attn` (`attention.h:517`) | ~90 call sites across ~50 files | the default everywhere robustness allows it. |
| **self-attention with KV cache** | `core_attn::kv_self_attn` (`attention.h:705`) | `src/canary_qwen.cpp`, `src/voxtral.cpp` | three selectable GQA strategies (`attention.h:609-613`), eager-F32 escape hatch via `CRISPASR_CORE_ATTN_EAGER_F32` (`attention.h:964-994`). |
| **cross-attention** | write it inline against your own cached K/V; **`src/core/cross_attn.h` is dead code** | `src/mt3.cpp:802-821` (cross-KV built once per encoder length) | `core_cross_attn::` has **zero** callers tree-wide. Its own header names `t5_translate.cpp` and `moonshine_streaming.cpp` as adopters; both have independent implementations. Do not treat it as the blessed path. |
| **softmax** | `ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f)` | `src/btc_chords.cpp:195` | the `scale` argument applies to the logits, not Q. Check which one your reference scales. |
| **LayerNorm** | `ggml_norm` **only if** the reference is biased-variance with eps inside the sqrt | `src/btc_chords.cpp:11-13` | BTC's norm is unbiased std with eps *outside* the sqrt, so `ggml_norm` **cannot** be used and it is built from primitives. Pocket-TTS's flow net stayed manual for the same reason (`LEARNINGS.md` L12837). Check before assuming. |
| **BatchNorm** | fold into the preceding conv at load time — **branching on the conv tensor's dtype** | `src/piano_transcription.cpp:229` (`fuse_bn`); the trap is `LEARNINGS.md` L1778 | `ggml_backend_tensor_get/set` with an F16-sized buffer against an F32 tensor silently corrupts the layer. Also: if a ReLU sits *between* conv and BN, the fold is invalid — CREPE ships BN as a per-channel affine instead (`crepe.cpp:10-12`). |
| **mel / STFT front end** | `src/core/mel.h` (already threaded), `src/core/fft.h` | `src/piano_transcription.cpp`, `src/onsets_and_frames.cpp`, `src/mt3.cpp` | ⚠ issue **#453** (open): *"core_mel: threaded MKL silently multiplies the upper mel bins by the thread count."* The in-tree evidence is `tools/oaf_parity.py:84-102`, which pins `MKL_NUM_THREADS=1` for exactly this. See §5.4. |
| **CQT front end** | `src/core/cqt.h`, `src/core/cqt2010v2.h` | `src/btc_chords.cpp`, `src/basic_pitch.cpp` | HARD RULE 2b: a CQT missing librosa's `scale=True` passed correlation 0.9999 while every bin was low by up to 152×. Assert the algebraic invariant (`L1 == sqrt(N_k)`), not a tolerance. |
| **ISTFT** | `src/core/istft.h` | — | CPU. |

### 1a. The SIMD / packed-conv family, and the bridge back into a graph

There are **three** distinct CPU-conv families here, and one of them dissolves
the apparent "ggml graph *or* hand-written SIMD" dichotomy.

| family | header(s) | ggml? |
| --- | --- | --- |
| graph convs (depthwise / grouped / transposed, which ggml has no `groups` argument for — `conv.h:3-7`) | `src/core/conv.h` (303 lines) | pure ggml graph; 18 TTS/codec `.cpp` callers |
| the packing + ISA-dispatch engine | `src/core/cpu_packed_conv1d.h` (396 lines), `struct PackedConv1d` at `:117` | **no ggml include at all**; plain `float*` in, `float*` out; parallelised with `core_parallel::for_each_chunk` at `:200-203` |
| the Basic Pitch 2-D path | `src/core/basic_pitch_conv.h` (366 lines) | no ggml; reuses only `core_cpu_conv1d::Isa` / `isa_available` (`:11, :124, :133-140`) |

**The bridge — `ggml_map_custom1`.** `src/core/hift_simdconv.h` (96 lines) runs
the hand-SIMD `PackedConv::run_rows` kernel *as a single node inside a larger
ggml graph*: `conv_op` (`hift_simdconv.h:77`) and `conv_tm` (`:90`) wrap it in
`ggml_map_custom1`, with `to_time_major`/`from_time_major` (`:69-75`) handling
the ggml channel-major ↔ packed-conv time-major layout mismatch, and `Packer`
(`:18-64`) doing load-time pack/validate/rollback. `src/core/hift_packed_conv1d.h`
(`struct PackedConv`, `:14-40`) adds the square-conv causal/symmetric padding
policy, and `chatterbox_hift_simdconv.h` / `cosyvoice3_hift_simdconv.h` are
23-line model adapters that each fix that policy.

**This matters for the rewrites.** "This layer is faster as a hand-written SIMD
loop" and "the model should be one ggml graph" are not in conflict. Keep the
graph for allocation, scheduling and the ops ggml is good at, and drop the one
hot kernel in as a custom op. The cost is that a `map_custom` node is CPU-only
and opaque to the scheduler, so it pins that subgraph to the CPU backend —
acceptable for a CPU-target model, disqualifying if you want GPU offload.

**Two traps carried from the Basic Pitch work
(`docs/music-transcription/BASIC_PITCH_CONV_PERF.md` §4):**

- **`cpu_packed_conv1d.h` `#undef`s `CRISPASR_CPU_PACKED_CONV1D_X86` at the end
  of the header**, so `#if CRISPASR_CPU_PACKED_CONV1D_X86` in a consumer
  silently evaluates to 0 and compiles every AVX kernel out. *Consumers must
  carry their own probe.* The first "AVX2" measurements in that work were pure
  scalar because of this.
- **Never copy `__attribute__((optimize("fp-contract=off")))`** out of
  `cpu_packed_conv1d.h`, where it is correct. On GCC the `optimize` attribute
  *replaces* the function's optimisation options rather than adding to them, so
  the annotated function loses `-O3`. Measured cost: 226 ms → 1230 ms, a 3.7×
  regression. Keep contraction off by construction instead (no `-mfma` in the TU).

### 1a-bis. Other shared primitives worth knowing about

| header | lines | what it gives you | who uses it |
| --- | --- | --- | --- |
| `core/ffn.h` | 132 | ggml-graph SwiGLU / GeGLU / SiLU-FFN / GELU-erf-FFN builders (`swiglu:39`, `swiglu_fused_gate_up:54`, `geglu:86`, `silu_ffn:100`, `gelu_erf_ffn:120`); "a one-line call instead of a 6-8 line block" (`ffn.h:1-20`) | 32 `.cpp` files |
| `core/cpu_ops.h` | 135 | CPU helpers shared by the granite family: **dtype-safe dequant `to_f32` (`:47`)**, `layernorm` (`:78`), one-op `matmul` via a sched (`:108`) | 18 files |
| `core/fft.h` | 202 | shared radix-2 + odd-N mixed-radix FFT, replacing per-model recursive copies | 19 `.cpp` files |
| `core/mel.h` / `mel.cpp` | 267 | log-mel replacing **9** copy-pasted implementations; NeMo and Whisper/HF parameter clusters via `Params` (`:116`); caller supplies its own `FftR2C` (`:20-24`) | 32 `.cpp` files |
| `core/istft.h` | 246 | CPU overlap-add inverse STFT, hoisted from 3 vocoders; carries a per-source **adoption verdict** including "`cosyvoice3_tts.cpp` — DIVERGENT, do NOT adopt as-is" (`:10-24`) | 5 files |
| `core/rnnt_ggml.h` | 317 | RNNT/TDT predictor + joint as ggml graphs so the **GPU is not idle during decode** (`:1-14`); persistent-graph `Decoder` (`:154`, init `:187`). Explicitly "P100-only win; M1 is neutral" | `parakeet.cpp`, `nemotron.cpp` |
| `core/quant_bcast.h` | 102 | `mul_mat_fold_batch` (`:39`) folds the batch dim into the token dim so a quantised `src0` never hits ggml's broadcast path — which produced **all-zero output** on a GTX 1660 SUPER (`:1-16`). Plus `audit()` (`:72`, `CRISPASR_AUDIT_QUANT_BCAST=1`), a graph walker, because "static greps for this pattern undercount badly" | `sidon.cpp`, `beat_this.cpp`, `cosyvoice3_tts.cpp`, `funasr.cpp`, `qwen3_asr.cpp` |
| `core/worker_pool.h` | 116 | bounded blocking RAII resource pool (`WorkerPool:23`, `acquire:82`) | **no `src/` caller**; only `examples/cli/crispasr_server.cpp` |

**Two primitives are written but unadopted**, which is worth knowing before you
assume they are load-bearing: `core/gru.h` (140 lines) has **zero callers** —
only `tests/test-core-gru.cpp` exercises it; and `core/cross_attn.h` (226 lines)
has zero callers. Both are correct-looking and both have never run in
production here. Adopting either is a port, not a drop-in.

### 1b. ⚠ The BiLSTM caveat — read this before adopting `core/lstm.h`

`core/lstm.h` is a real ggml graph and it does the important thing: the input
projection `W_ih @ X + b_ih` is computed once for all T
(`lstm.h:66-68`), leaving only the genuinely sequential `W_hh @ h_{t-1}` per
step. That is strictly better than recomputing the projection per timestep.

But `lstm_unidir` **unrolls the recurrence into the graph**: the loop at
`lstm.h:79` emits roughly a dozen nodes per timestep (`view`, `mul_mat`, three
`add`s, four gate views, `sigmoid`×3, `tanh`×2, `mul`×3, `cpy`).

Its three current callers — `src/kokoro.cpp`, `src/dots_tts.cpp`,
`src/bananamind_tts.cpp` — are all TTS models where T is a few hundred
phoneme/frame steps. **Onsets & Frames sees T ≈ 3000 frames for 30 s of audio.**
At ~12 nodes per step per direction, three BiLSTMs × two directions is on the
order of 200,000 graph nodes.

**I have not measured this, and I am not asserting that the ggml BiLSTM beats
the scalar recurrence at that T.** Graph-build cost, `ggml_tensor_overhead()`
per node, and gallocr planning all scale linearly in node count, and the
per-step work (one `384×1536` mat-vec) is small enough that node overhead could
dominate. What I *am* asserting is that the current O&F arrangement — a ggml
graph for the projection with a **fresh allocator per call**
(`onsets_and_frames.cpp:507-556`), then a scalar recurrence — pays the graph
overhead without getting the reuse. **Fix the allocator first (§2), measure,
and only then decide whether the recurrence should move into the graph.** If it
should, chunking T into segments with carried `(h, c)` is the obvious way to
bound node count, and it is bit-identical because the recurrence is exact.

---

## 2. The graph lifecycle pattern

### The rule

`ggml_gallocr_t`, `ggml_backend_sched_t`, the graph-metadata arena and any
persistent state buffer are **per-context**, created at init and freed at
teardown. The only per-call work is: a cheap `ggml_init` over the *existing*
arena with `no_alloc=true`, building the ops, allocating, computing.

### Two strategies, and when each applies

The dev guide states the split (`../crispasr-crispembed-dev.md` §"ggml graph
allocation: sched vs gallocr"):

- **`ggml_backend_sched`** for full weighted model inference. It handles the
  weight buffer and compute buffer together and gives you CPU fallback for ops
  the GPU backend cannot run. Use this by default for a weighted model.
- **`ggml_gallocr`** for hot paths and single-backend graphs where you have
  already proved every op is supported. There is no fallback, so you must gate
  on capability (see `core_step_cache::supports_graph`,
  `src/core/step_graph_cache.h:101`).

### Reference A — scheduler, with an explicit measure pass: `src/crispasr.cpp`

The cleanest sched lifecycle in the tree. Four independent schedulers for four
graph shapes rather than one oversized worst-case graph.

```cpp
struct whisper_sched {                       // crispasr.cpp:1038-1042
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t>  meta;
};
```

Init, once per context, per graph shape (`crispasr.cpp:1054-1073`, call sites
`4041, 4056, 4071, 4086`):

```cpp
sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(),
                               CRISPASR_MAX_NODES, false, true);          // :1059
meta.resize(ggml_tensor_overhead() * CRISPASR_MAX_NODES
            + ggml_graph_overhead());                                     // :1062
// since there are dependencies between the different graphs, we need to
// allocate them instead of only reserving to get the correct buffer size
if (!ggml_backend_sched_alloc_graph(sched, get_graph())) { ... }           // :1066
ggml_backend_sched_reset(sched);                                          // :1071
```

Per call, the context is cheap because the arena is reused
(`crispasr.cpp:2753-2761`):

```cpp
struct ggml_init_params params = {
    wstate.sched_encode.meta.size(),   // the arena sized once, above
    wstate.sched_encode.meta.data(),
    /*.no_alloc =*/ true,
};
ggml_context* ctx0 = ggml_init(params);
ggml_cgraph*  gf   = ggml_new_graph_custom(ctx0, CRISPASR_MAX_NODES, false);
```

Then `ggml_backend_sched_alloc_graph` + compute (`crispasr.cpp:3072-3080`).
`ggml_free(ctx0)` at `:2933` is safe and cheap *because* `mem_buffer` is an
externally-owned vector — it releases only the small context bookkeeping
object, not the arena the graph tensors live in.

And the multi-call safety net (`crispasr.cpp:7637-7646`), at the top of every
`whisper_full_with_state`:

```cpp
// Reset all schedulers so a second whisper_full() call on the same context
// doesn't hit GGML_ASSERT(!sched->is_alloc) ...
ggml_backend_sched_reset(state->sched_conv.sched);
ggml_backend_sched_reset(state->sched_encode.sched);
ggml_backend_sched_reset(state->sched_cross.sched);
ggml_backend_sched_reset(state->sched_decode.sched);
```

### Reference B — persistent build-once graph on a raw gallocr: `src/crepe.cpp`

**This is the reference to copy for an encoder-only, fixed-shape model**, which
is what all six transcription backends are. It applies four separate lessons in
one 524-line file:

1. `ggml_gallocr_t galloc` is a context member (`crepe.cpp:135`), created and
   `ggml_gallocr_alloc_graph`'d **once** at init (`crepe.cpp:452-453`), freed
   only at teardown (`crepe.cpp:470`).
2. The graph itself is built once at init and reused; per frame it is
   `tensor_set → compute → tensor_get` (`crepe.cpp:28-32`).
3. **Every input is re-set on every compute** (`crepe.cpp:308`), because
   gallocr may hand an input's buffer to a later intermediate. This is not
   optional — see §6.
4. F32 copies of F16 conv kernels are baked once at load
   (`crepe.cpp:386-400`), because `ggml_conv_1d` casts the kernel *inside* the
   graph when activations are F32; in a persistent graph that cast re-runs
   every compute. Measured: it re-cast 44 MB of weights per 10 ms frame, RTF 31
   on M1. Gated for A/B by `CRISPASR_CREPE_NO_BAKE_F32=1`.

### Reference C — one allocator per graph *shape*: `src/hft_transformer.cpp`

When a model has a small number of distinct graph shapes, keep one allocator
each. The comment says why, and names the bug this playbook exists to fix
(`hft_transformer.cpp:108-115`):

```
// One allocator per graph shape, kept for the life of the context.
// §36.4 of the flutter_tuner benchmark named "a fresh ggml allocator per
// convolution chunk" as one of the two unfixed causes of the Onsets &
// Frames arm's throughput gap; reusing them costs nothing (ggml_gallocr
// keeps its buffer when the next graph fits) and this model runs four
// encoder chunks and a time decoder for every 2.048 s of audio.
ggml_gallocr_t alloc_enc  = nullptr;
ggml_gallocr_t alloc_time = nullptr;
```

Its arena grows monotonically and is never shrunk (`hft_graph_ctx`,
`hft_transformer.cpp:479-484`) — a two-line pattern worth copying verbatim.

### Reference D — bucketed graph cache for autoregressive steps

`src/core/step_graph_cache.h` (233 lines) is the shared helper for AR decode
steps: `Cache::get_or_build` (`:163-222`) builds a `no_alloc` arena per `Lk`
bucket, checks `supports_graph`, then `ggml_gallocr_new` + `ggml_gallocr_reserve`
**once** (`:210-212`), freeing only on LRU eviction or teardown (`:141-149`).

Read its header comment in full before using it: it documents the trap (a
single `max_ctx` graph is a **loss**, +69% measured on funasr), that the width
must be narrow, and — importantly — that **it measured no win on voxtral-3B**
(`step_graph_cache.h:22-39`). Adoption is partial: only `src/voxtral.cpp` uses
it (`:26, 244, 1236, 1252`); `src/funasr.cpp` still carries its own copy
(`:245, 1287-1293, 1328-1361`).

Encoder-only models do not need this. See §3's note on what KV machinery
transplants.

### Persistent CPU threadpool

Independent of the graph, and part of the same lifecycle discipline
(`crispasr.cpp:197-201`):

```
// Persistent threadpool for CPU backends.  Without this, every
// ggml_backend_cpu_graph_compute call creates a disposable threadpool,
// spawning n_threads-1 pthreads on non-OpenMP builds.  After hundreds of
// encoder/decoder calls the accumulated mmap/munmap for thread stacks
// fragments memory and degrades perf 2-5× — the #132 pattern.
```

Each CPU backend gets its own pool, keyed in a map (`crispasr.cpp:205-213`), so
multiple states do not race. Only `crispasr.cpp` does this today; any backend
making hundreds of graph computes per file should.

### Memory-mapped weights

`core_gguf::load_weights` binds tensors directly into an mmap of the GGUF rather
than allocating a backend-side copy (`core/gguf_loader.cpp:609-660`), with
`posix_madvise(..., WILLNEED)` readahead (`:645`) and optional preload/mlock
(`:651-656`). Default on since issue #94; opt out with `CRISPASR_GGUF_MMAP=0`.
This is free and you get it by using the shared loader.

---

## 3. The threading model

### The layers, and the rule

`docs/concurrency.md` names three layers: intra-op ggml threads, server
workers, and process-level parallelism. For a single backend's forward pass,
only the first is yours to set, plus whatever hand-threading sits *outside* the
graph.

**The rule:**

> **Inside a ggml graph, set `n_threads` and let ggml do it.** Hand-thread only
> work that is *outside* any graph — a hand-written conv loop, a scalar
> recurrence, a front end — and when you do, make sure the two are not live at
> the same time.

### Setting ggml's threads correctly

Always through the shim, never the raw call:

```cpp
core_cpu_backend::set_n_threads(ctx->backend, n);   // core/ggml_cpu_backend.h
```

`src/core/ggml_cpu_backend.h` (331 lines) exists because under
`GGML_BACKEND_DL` the CPU backend is a dlopen'd module and
`ggml_backend_cpu_init` / `ggml_backend_is_cpu` / `ggml_backend_cpu_set_n_threads`
are not linkable — this tree calls them at ~424 sites across 104 files
(`ggml_cpu_backend.h:1-24`). Without DL every wrapper compiles to the direct
call, so the shipped build is unchanged. The shim is also null-tolerant in both
branches (`:52-55`), which matters because the raw
`ggml_backend_cpu_set_n_threads` **asserts CPU** and aborts on any GPU backend
(`LEARNINGS.md` L3321 — this crashed silero-LID on every Metal/CUDA load, #165).

`src/hft_transformer.cpp:537` and `:607` show the right shape: call
`set_n_threads` immediately before each `ggml_backend_graph_compute`, from a
`hft_nthreads(ctx)` helper that reads `ctx->params.n_threads`
(`hft_transformer.cpp:171-172`).

### `core/parallel_for.h` — for work outside the graph

`src/core/parallel_for.h` (118 lines) exists because **five copies of the same
loop had accumulated**. Its header says so, and names them (`parallel_for.h:1-25`):

```
// core/parallel_for.h — one place for "run this loop across the cores".
//
// Five copies of this had accumulated, each written for its own file:
//   firered_vad.cpp        firered_parallel_for   (static template)
//   marblenet_vad.cpp      marblenet_parallel_for (static template, identical)
//   omnivoice.cpp          ov_parallel_for        (std::function variant)
//   core/mel.cpp           inlined, no helper
//   core/foxnose_pipeline.cpp  inlined, no helper
//
// They come in TWO genuinely different shapes ...
//   for_each_chunk — splits [0, n) into one contiguous block per thread.
//     Right when every item costs about the same (audio frames, feature rows)
//   for_each_task — hands out item indices from an atomic counter, so threads
//     take more work as they finish. Right when per-item cost varies a lot ...
//     Also carries a SLOT index for callers that own per-thread resources
```

- `core_parallel::for_each_chunk(n_items, n_threads, fn(begin, end))` —
  `parallel_for.h:66`. Used at `piano_transcription.cpp:283` (over output
  channels) and `:458` (over time).
- `core_parallel::for_each_task(n_tasks, n_threads, fn(task, slot))` —
  `parallel_for.h:91`, pulling indices off a `std::atomic<int>` (`:100`). Used
  at `piano_transcription.cpp:422` for the two GRU directions.

It is neither `std::execution::par` (libc++ has no PSTL backend; libstdc++'s
needs TBB) nor OpenMP-only (absent on the stock macOS toolchain) —
`parallel_for.h:32-42`.

**The determinism contract, `parallel_for.h:27-30`** — this is what makes the
bit-identity argument in §7 work:

```
// `fn` must write only to storage owned by its own index/range. Any
// order-sensitive reduction — argmin, argmax, accumulation — belongs in a
// SERIAL pass afterwards, so results do not depend on thread timing.
```

`resolve_threads` (`:55-63`) turns `n_threads <= 0` into
`hardware_concurrency()`, clamped to `[1, n_items]`.

Both degrade to serial, **in the original order**, at `nthreads == 1` — stated
at `piano_transcription.cpp:419-421`. That property is what makes the
bit-identity argument in §7 work.

Note the known cost, measured: both shapes spawn raw `std::thread`s per call
with **no pool** (`parallel_for.h:75, 83-85, 109-115`) — they reuse the calling
thread as one worker, so `n_threads == 1` is essentially free, but every call
creates and destroys `nt-1` threads.
so for a six-conv-per-window model the spawn is paid ~84 times for a 22 s file,
and 4 threads costs ~75% more CPU than 2 for ~9% less wall
(`docs/music-transcription/BASIC_PITCH_CONV_PERF.md` §5). `src/core/worker_pool.h`
is the intended fix and has not been wired in.

### Thread-count policy

`piano_nthreads()` (`piano_transcription.cpp:215-226`) is the pattern to copy:
cache in a function-local static, honour an opt-out env var, clamp to hardware:

```cpp
if (crispasr_env::get("CRISPASR_PIANO_SERIAL") != nullptr) v = 1;
else { unsigned hw = std::thread::hardware_concurrency();
       v = (hw == 0) ? 1 : (int)std::min(hw, 8u); }
```

The clamp at 8 is deliberate. Note that it does **not** read
`params.n_threads` — for a new backend, prefer `hft_nthreads`'s shape (read the
params first, fall back to a default) so `-t` actually reaches the work.

### Oversubscription — three documented forms

This tree has hit all three. Do not treat this as theoretical.

1. **Hand-threading around a ggml graph.** If you wrap `for_each_chunk` around a
   region that itself calls `ggml_backend_graph_compute`, you multiply
   `n_outer × n_ggml` threads on a box that has neither. ggml's CPU threadpool
   spin-waits at barriers, so the loss is worse than linear: hFT's f32 arm goes
   **4.84 → 7.35 CPU-seconds per audio-second between one thread and four**
   (`docs/music-transcription/HFT_TRANSFORMER.md`). I have **not** found a
   comment in this tree explicitly forbidding the nesting; the evidence above is
   the closest thing, and I am stating the rule as a conclusion from it rather
   than quoting one.
2. **BLAS nested under OpenMP** — `src/mel_band_roformer.cpp:47`, issue **#296**.
3. **Two OpenMP runtimes in one process corrupting output** — commit
   `00822744`. Relevant to BLAS selection at build time.

4. **Threaded MKL corrupting a mel front end** — issue **#453**, open:
   *core_mel: threaded MKL silently multiplies the upper mel bins by the thread
   count*. This is a **correctness** bug caused by threading, in a front end
   three of the six models use. The in-tree evidence predates the issue:
   `tools/oaf_parity.py:84-102` already pins `MKL_NUM_THREADS=1` and its comment
   calls it "a host defect, not a model one". Any timing or parity run of a
   mel-front-end model on a box with threaded MKL must pin the BLAS thread
   count first — see §5.4.

---

## 4. Quantisation on the compute path — the biggest single finding

### The claim

**Quantising a model in this tree makes it smaller. With one exception it does
not make it faster on CPU, and it frequently makes it slower.** hFT measured
**q8_0 at 29% more CPU than f32**, and q4_0 the same again
(`docs/music-transcription/HFT_TRANSFORMER.md`).

### Why, verified

ggml's fast repacked int8 GEMM lives in `ggml/src/ggml-cpu/repack.cpp`. It is
reached **only** through the CPU device's *extra* buffer types, exposed via the
registry proc address `ggml_backend_dev_get_extra_bufts`
(`ggml/src/ggml-cpu/ggml-cpu.cpp:656-658`). `GGML_CPU_REPACK:BOOL=ON` is set in
this build (`build/CMakeCache.txt`), so the code is compiled in.

I verified where that entry point is requested in this tree. **Exactly one place:**

```cpp
auto* cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
auto get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
    ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
if (get_extra_bufts_fn) {
    ggml_backend_buffer_type_t* extra_bufts = get_extra_bufts_fn(cpu_dev);
    while (extra_bufts && *extra_bufts) { buft_list.emplace_back(cpu_dev, *extra_bufts); ++extra_bufts; }
}
```
— `src/crispasr.cpp:1925-1937`, the whisper backend's llama.cpp-style
`buft_list`, guarded by `weight_buft_supported` (`crispasr.cpp:1945+`) which
notes the extra buffer types support only `GGML_OP_MUL_MAT` and
`GGML_OP_GET_ROWS`.

Every other backend loads weights through `core_gguf::load_weights`, which uses
`ggml_backend_alloc_ctx_tensors` (`core/gguf_loader.cpp:852`) or
`ggml_backend_get_default_buffer_type` (`:860, :1140`) or wraps the mmap in
`core_cpu_backend::buffer_type()` (`:657`) — **the default buffer type in every
path, never an extra one.** So their quantised matmuls take ggml's generic
route, which re-quantises the activation operand to Q8_0 on every GEMM and then
runs a `vec_dot`.

**Correction to how this was reported to me:** it is not true that the repack
path is unreached *anywhere* in the tree. It is reached by `src/crispasr.cpp`.
It is unreachable for every backend that loads through `core_gguf`, which is
all six transcription models and most of the rest.

### What is and is not the conclusion

**This section was written before the lever had been measured. It has now been
measured — `ggml-repack-buft-evaluation.md` is the authority, and it corrects
three things below. Summary of what changed:**

- ✅ **The lever is real and it is in `core/gguf_loader.cpp`, not in any model
  file.** Still true. `core_gguf::load_weights_repack()` is now that path.
- ❌ **"…would apply tree-wide in one change" was wrong.** The loader cannot
  know which tensors are used as `MUL_MAT` `src[0]`; only the model can. Every
  adopting backend must supply a predicate and be checked for other uses of
  those tensors, exactly as `crispasr.cpp:1945` does with a probe op. It is a
  one-line change *per model*, on top of one shared loader.
- ❌ **"no VNNI, so there is nothing for the repacked path to reach" was too
  pessimistic.** The AVX2 kernels do not use VNNI — the interleaved *layout*
  pays on its own. Measured on this very box (Skylake-SP, no VNNI, no AMX),
  single thread, interleaved arms, best of 40:
  `q4_0` generic 1.22–1.44× f32 → repacked **0.76–0.97×** f32; `q4_K` generic
  1.83–3.02× f32 → repacked **0.48–0.66×** f32. A Kaggle AVX2-only Xeon agrees
  at 1.9–3.4× over the generic path, with a far tighter spread.
- 🔑 **Measured end-to-end, whole model, on clean CI runners** (run
  35819605846; `crispasr --piano`, hFT, one process per arm, interleaved,
  median of 3, `MKL_NUM_THREADS=1`):

  | arm | EPYC 7763 (AVX2) | arm64 (dotprod/i8mm) |
  | --- | --- | --- |
  | hFT f32 | 12.88 cpu-s | 102.5 cpu-s |
  | hFT q8_0 | 14.05 (**1.09×** f32) | 51.4 (**0.50×** f32) |
  | hFT q4_0 generic | 15.7 (1.22×) | 56.5 (0.55×) |
  | **hFT q4_0 + repack** | **9.3 (0.72×)** | **35.7 (0.35×)** |
  | **repack vs generic** | **1.69×** | **1.58×** |
  | O&F q8_0 vs f32 | **0.99×** | 0.88× |

  Two corrections to this section's headline claims fall straight out of it.
  **hFT's 29% q8_0 penalty is 1.09× on the EPYC** — it is a Skylake-SP/AVX-512
  number, not an x86-wide one. And **on arm64 q8_0 alone is 0.50× f32**: the
  model runs twice as fast quantised with no repacking at all.
- 🔑 **The answer splits by ISA, and the arm64 half is a large win for the
  models exactly as they ship.** Measured on clean GitHub runners
  (run 35817843249, three legs green), q8_0 `MUL_MAT`, repacked vs generic:

  | runner | ISA | q8_0 | q4_0 | declined |
  | --- | --- | --- | --- | --- |
  | `macos-14` | neon + dotprod | **3.1–3.4×** | 4.0× | *nothing* |
  | `ubuntu-24.04-arm` | dotprod + i8mm + sve | **2.3–2.4×** | 3.3–3.6× | *nothing* |
  | `ubuntu-24.04` | AMD EPYC 7763, AVX2 | **no kernel** | 2.3–2.7× | q8_0, q5_K, q6_K |

  `ggml_repack_get_optimal_repack_type` (`repack.cpp:4528`) has **no q8_0
  branch for x86 at all** — it is gated on NEON+dotprod/i8mm or RISC-V. Every
  quantised GGUF in this tree is q8_0. So on x86 this lever is worth nothing
  without re-quantising to q4_0/q4_K, which is an accuracy decision; on arm64,
  including Apple Silicon, it is worth 2.3–3.4× on the files as they exist.
- ⚠ **"Quantisation does not make CPU inference faster" is a statement about
  OLD x86 — pre-VNNI — and nothing else.** Measured whole-model, hFT q8_0
  against f32, one process per arm, interleaved, median of 3: **1.29×** on the
  Skylake-SP VPS (AVX-512, no VNNI), **1.09×** on an AMD EPYC 7763 (AVX2),
  **0.84×** on an AMD EPYC 9V74 (AVX-512 **VNNI**) and **0.50×** on arm64. On
  anything with an int8 dot-product instruction, quantisation is a *speedup*
  before any repacking at all.
  At the kernel level the same story: the *generic* quantised `MUL_MAT` is
  already 2.2–3.1× faster than f32 on arm64 before any repacking, and q8_0 +
  repack lands at **0.12–0.15× the cost of f32**. Why Skylake-SP is the worst
  case: its AVX-512 makes the *f32* GEMM unusually fast, so the quantised path
  loses by comparison rather than being slow in absolute terms.
- ✅ **The mmap incompatibility is now verified, not inferred.** The path at
  `gguf_loader.cpp:657` binds `tensor->data` into the file map and **never
  calls `set_tensor`**, while repacking *is* a `set_tensor` that rewrites the
  bytes. They cannot both hold for one tensor.
- ⚠ **Correction: the buft supports `MUL_MAT` and `MUL_MAT_ID`, not
  `GET_ROWS`** (`repack.cpp:4774`). The comment at `crispasr.cpp:1945` is stale
  against this ggml version.
- ⚠ **A trap.** `ggml_backend_cpu_repack_buffer_set_tensor` dereferences
  `tensor->extra` unconditionally, and `init_tensor` leaves it null when it has
  no kernel for that (type, shape, ISA). Writing a declined tensor into the
  repack buft is a **null dereference, not a fallback**. Classify first;
  `core_gguf::repack_buft_accepts()` does that by asking ggml.
- ❌ **Do not quantise a compute-bound CPU model expecting speed today.**
  Still the right default. Generic-path q8_0 measures 1.08–1.31× the cost of
  f32 for transformer-shaped GEMMs on this box — which is the mechanism behind
  hFT's 29%, reproduced without a model. Quantise for size, and measure.

Run `crispasr-repack-probe` on any new machine before assuming any of this
transfers to it.

### The narrower quantisation lever that *is* proven

`core_conformer::repack_conv_pw_q8` (`src/core/fastconformer.h:335-374`, issue
#81) repacks F16 pointwise-conv weights to Q8_0 at load, because "the ggml CPU
F16 mul_mat has no repack fast path" (`fastconformer.h:339`). Used by
`parakeet.cpp:2979`, `canary_ctc.cpp:775`, `canary_qwen.cpp:1315`,
`gigaam.cpp:1105`, `lfm2_audio.cpp:624`. This is a *targeted* dtype change on
specific tensors, not a blanket quant, and it is the shape of quantisation work
that has paid off here.

`src/core/quant_bcast.h` — see §1a of the primitives inventory below; its role
is making broadcast ops safe against non-F32 weights, which is a **correctness**
guard (`LEARNINGS.md` L18623: "a binary-broadcast op with a non-F32 weight is a
latent abort on EVERY ggml backend"), not a speed lever.

### Build flags

`docs/improvements/SRC_ISA_GAP.md` is the authority and its finding is
unresolved: release builds hand **ggml** `-DGGML_AVX2=ON -DGGML_FMA=ON
-DGGML_F16C=ON` on 15 legs and full runtime dispatch on 2 more, while
`release.yml` contains **zero** occurrences of `CMAKE_CXX_FLAGS` — so every
hand-written kernel under `src/` ships generic x86-64, SSE2, 4-wide, on every
leg. Objdump evidence: `libggml-cpu.so` has 19886 `%ymm` refs and 5427 AVX/FMA
instructions; `basic_pitch.cpp.o` has 0 `%ymm`, 1460 `%xmm`, 0 `vfmadd`.

The consequence for this playbook: **a hand-written kernel that wants SIMD must
dispatch at runtime** (`__builtin_cpu_supports` via
`core_cpu_conv1d::isa_available`), because it will not be given `-march`.
`basic_pitch_conv.h` is the worked example.

⚠ `GGML_AVX2:BOOL=OFF` in a local `CMakeCache.txt` does **not** mean ggml is
baseline — `GGML_NATIVE=ON` supersedes those switches. That misreading was the
first conclusion reached in the Basic Pitch work and it was wrong.

---

## 5. The parity procedure

A perf change is done when its **output is proven equal to a trusted
reference**, not when it compiles and is fast
(`../crispasr-crispembed-dev.md`, HARD RULE and the A/B section). This section
is how that proof is produced in this tree today.

### 5.0 The shape of it

Three pieces, in order: a **Python/ONNX reference dumper** writes per-stage
intermediates; the **C++ side** reproduces the same stages; a **diff tool**
reports per-stage cosine plus magnitudes and gates on a threshold. Debug from
the **earliest failing stage** — first divergence is the bug.

**Cosine is scale-blind. Always read the `|mine|` vs `|ref|` columns.** HARD
RULE 2b: a stage wrong by a uniform factor passes cosine, Pearson, argmax
agreement and SNR-vs-normalised-reference alike. This has bitten this tree three
times, including a CQT missing librosa's `scale=True` that reported correlation
0.9999 and 97.6% peak match while every bin was low by up to **152×**.

**Mark intermediates with `ggml_set_name()` *and* `ggml_set_output()`.** Without
the latter ggml reuses the memory and a later layer overwrites the value you
meant to read. Reading an unmarked *input* tensor after compute is the same bug
from the other side (§6.6).

### 5.1 The generic path — `dump_reference.py` + `crispasr-diff`

```bash
python tools/dump_reference.py --backend <name> --model-dir <HF id|path>     --audio samples/jfk.wav --output ref.gguf
build/bin/crispasr-diff <backend> model.gguf ref.gguf audio.wav
```

`tools/dump_reference.py` args (`:520-545`): `--backend`, `--model-dir`,
`--audio`, `--output`, `--stages` (comma-separated; empty = the backend's
`DEFAULT_STAGES`), `--max-new-tokens`, `--list-backends`. Audio load is stdlib
`wave` only and requires **16-bit PCM at exactly 16 kHz** (`:405-427`).

**Registering a backend** (4 steps, `../crispasr-crispembed-dev.md`):
1. `tools/reference_backends/<name>.py` exposing
   `dump(model_dir, audio, stages, max_new_tokens) -> dict[str, ndarray]` and
   `DEFAULT_STAGES` (contract at `dump_reference.py:78-90`);
2. a line in `REGISTERED_BACKENDS` (`dump_reference.py:113-396`);
3. an arm in `examples/cli/crispasr_diff_main.cpp`;
4. either stage APIs on the C++ side, or the self-contained
   `<name>_diff(model_gguf, ref_gguf, ...)` pattern — which is what all the
   music backends use.

**Metrics.** `crispasr_diff::Report` (`examples/cli/crispasr_diff.h:81-102`)
carries `max_abs`, `mean_abs`, `rms`, `rms_data`, `rms_ref`, `norm_ratio`,
`cos_min`, `cos_mean`, `top1_match/top1_total`, `n_nonfinite`.
`is_pass(cos_threshold = 0.999f)` (`.h:101`) = `found && n_nonfinite == 0 &&
cos_min >= cos_threshold`. The global gate is `COS_THRESHOLD = 0.999f`
(`crispasr_diff_main.cpp:1835`); the summary line and exit code are at `:8690`
(`0` on pass, `6` on failure).

### 5.2 Which of the six is wired to what — read this before planning a rewrite

| model | reference dumper | `crispasr-diff` arm | in `tests/regression/manifest.json`? |
| --- | --- | --- | --- |
| `basic_pitch` | `tools/reference_backends/basic_pitch.py` — **standalone script, not the `dump()` contract** (see 5.3) | `:1772-1786` → `basic_pitch_diff` (`src/basic_pitch.cpp:824-922`), `COS_MIN = 0.999` at `:840` | no |
| `crepe` | `tools/reference_backends/crepe.py` — proper `dump()` + `DEFAULT_STAGES` (`:83-89, :128-224`) | inline in `main()`, `:8580-8672` | no |
| `btc_chords` | `tools/btc_torch_parity.py` (4th arg writes the ref GGUF) | `:1706-1710` → `btc_chords_diff` | no |
| `piano_transcription` | — (diff replays; mel recomputed) | `:1755-1770` → `piano_transcription_diff`, chaining `piano_transcription_mel_spectrogram()` then `_acoustic_model()` | no |
| `mt3` | — | `:1788-1803` → `mt3_diff` | no |
| **`onsets_and_frames`** | **none** | **none** | no |

**None of the six is in the nightly CI regression matrix.**
`.github/workflows/regression.yml:150-186` is entirely ASR/TTS backends. The
only CI these models get is `basic-pitch-conv-ab.yml`, which gates the SIMD
conv path's **byte-equality and speed**, not model parity.

**Two gaps worth stating plainly:**

- **`onsets_and_frames` has no per-layer parity path at all** — see 5.4 for
  what it does have.
- **`crepe`'s per-layer stages are dumped but not compared.**
  `crispasr_diff_main.cpp:8663-8669` skips `frames` / `conv1_out..conv6_out` /
  `embedding` with the comment that `src/crepe.h` exposes no per-layer stage
  API; only `activation` is truly diffed. Its real per-layer gate is
  `tools/crepe_numpy_parity.py`, which `crepe.cpp:3-5` says must be changed
  first if the graph changes.

### 5.3 Worked recipe — per-layer cosine parity for `basic_pitch`

⚠ `basic_pitch` is listed in `REGISTERED_BACKENDS` but its module has **no
`dump()` and no `DEFAULT_STAGES`** — it is a standalone script with its own
argparse (`basic_pitch.py:312-321`) and its own `write_ref_gguf`
(`:296-309`). Invoking it through `tools/dump_reference.py` will fail. The
usage example in its own docstring (`:39-42`) is stale. **This is the actual
sequence:**

```bash
PY=/mnt/volume1/miniconda/bin/python     # torch 2.11, onnxruntime 1.23.2, librosa 0.11
                                         # needs: $PY -m pip install gguf

# 1. convert the upstream ONNX to GGUF
$PY models/convert-basic-pitch-to-gguf.py     --input  /mnt/storage/gguf-models/basic-pitch-src/nmp.onnx     --output /mnt/storage/gguf-models/basic-pitch-f16.gguf

# 2. build the diff driver
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target crispasr-diff -j$(nproc)

# 3. dump the reference (standalone script, NOT dump_reference.py)
$PY tools/reference_backends/basic_pitch.py     --model      /mnt/storage/gguf-models/basic-pitch-src/nmp.onnx     --audio      samples/jfk.wav     --output-dir /mnt/storage/parity/basic-pitch-ref

# 4. per-stage cosine diff
build/bin/crispasr-diff basic-pitch     /mnt/storage/gguf-models/basic-pitch-f16.gguf     /mnt/storage/parity/basic-pitch-ref/ref.gguf     samples/jfk.wav
```

Stages compared, in order (`src/basic_pitch.cpp:857-916`): `audio_window0`,
`cqt_magnitude`, `normalized_log`, `harmonic_stack`, `head_contour`,
`head_note`, `head_onset`, `unwrapped_{contour,note,onset}`.

A pass prints one line per stage with `cos=` and `max_abs=` and the element
counts, then `basic-pitch diff: PASS (0 failing stages)`, exit code 0. Any
stage below `cos = 0.999`, or a size mismatch, flips that line to `FAIL` and
the process exits 1.

*(Keep parity artefacts on `/mnt/storage`, not `/mnt/volume1`.)*

### 5.4 The Onsets & Frames path — what exists, and its two limits

O&F compares against **native onnxruntime**, not a PyTorch dump, through a
bespoke pair of tools.

```bash
# C++ side: writes mel + all five heads + decoded notes to <prefix>.*
cmake --build build --target oaf-parity-dump -j$(nproc)
build/bin/oaf-parity-dump <model.gguf> <audio.wav> <out-prefix> [n_threads]

# Python driver: runs the above, then ORT, and reports
python tools/oaf_parity.py     --model build/.../onsets-and-frames-f32.gguf     --audio some.wav     --dump  build/bin/oaf-parity-dump     --onnx  /mnt/storage/tuner-bench/onnx/onsets_and_frames.onnx     [--threads 4] [--seconds 0] [--workdir DIR]

# Note-level F1 on MusicNet, N arms through one decoder
python tools/oaf_musicnet_f1.py     --musicnet /mnt/storage/tuner-bench/datasets/musicnet/musicnet     --arm onnx=/mnt/storage/tuner-bench/onnx/onsets_and_frames.onnx     --arm f32=/path/onsets-and-frames-f32.gguf     --arm q4_0=/path/onsets-and-frames-q4_0.gguf     --dump build/bin/oaf-parity-dump --workdir /mnt/storage/oaf-f1
```

`oaf-parity-dump` (`tests/oaf_parity_dump.cpp`) is a plain `main()`, built as a
standalone binary (`tests/CMakeLists.txt:5684-5686`) and deliberately **not**
registered with CTest because it needs a model, a WAV and onnxruntime
(`:5681-5683`). It writes `<prefix>.mel.f32` (T×229), the five heads as T×88
post-sigmoid, a `.meta.txt` with frame counts and wall/CPU realtime factors,
and a `.notes.tsv`.

`tools/oaf_parity.py` compares in two stages — front end, then heads on the
**same** mel, so a model bug cannot hide behind a front-end one (`:5-11`) — and
prints `max_abs / mean_abs / rms / cos / |mine| / |ref|` (`:105-125`), plus
raw-decision agreement at onset 0.5 / frame 0.5 (`:190-199`).

**Two limits, both important for the rewrite:**

1. **It is not per-layer.** Five heads and a mel is end-of-pipeline. A
   regression introduced inside a ConvStack or a BiLSTM shows up as "the onset
   head moved", not as a layer. Building an O&F dumper on the `crepe.py`
   contract is the cheapest thing you can do before touching the file.
2. **There is no PASS/FAIL gate.** The script always exits 0 unless the C++
   dump subprocess itself fails (`:154-156`). It is a diagnostic report. Any CI
   use needs a threshold added.

It also documents a real ONNX export bug you will trip over: the graph's
`output_names` were assigned to a five-output forward with a **one-slot shift**
(`oaf_parity.py:44-50`) — `onset→onset, offset→offset, activation→frame,
frame→velocity, velocity→679`.

**And it contains the in-tree evidence for issue #453.** `blas_safe_env()`
(`oaf_parity.py:84-102`) pins `MKL_NUM_THREADS=1`, because Debian's threaded
MKL `sgemm`, linked alongside `libgomp`, silently corrupts the mel projection's
upper bins by a factor tied to the thread count. The comment calls it "a host
defect, not a model one." Issue **#453** ("core_mel: threaded MKL silently
multiplies the upper mel bins by the thread count", open) is the same
phenomenon filed against `core/mel.h`. **Any parity run of a mel-front-end
model on this box must pin the BLAS thread count, or the reference and the
runtime will disagree for reasons that have nothing to do with your change.**

### 5.5 The other parity tools in the tree

| tool | what it does | invocation |
| --- | --- | --- |
| `tools/btc_torch_parity.py` | an *executable spec*: reimplements BTC's forward in raw numpy from the GGUF weights and checks it against the real PyTorch `BTC_model`. Gate: `cos < 0.9999 or argmax_agree < 1.0` → `sys.exit("FAIL")` (`:226-227`) | `python tools/btc_torch_parity.py <btc-chords.gguf> <btc_model.pt> <BTC-ISMIR19 dir> [ref-dump.gguf]` — the 4th arg writes the per-stage ref GGUF for `crispasr-diff btc` |
| `tools/beatrice_torch_parity.py` | dumps Beatrice per-stage refs, and **gates on relative max-abs (`TOL = 1e-6`), not cosine** — the file documents a wrong-GELU bug that still scored `cos = 0.9999996` (`:326-333`). Plus a hard non-finite check before any tolerance check | `python tools/beatrice_torch_parity.py --component pitch_estimator --model <ckpt.pt> --audio samples/jfk.wav --output beatrice-pitch-ref.gguf --trainer-path <dir>` |
| `tools/cb_turbo_perlayer_dump_pyref.py` + `tools/cb_turbo_perlayer_diff.py` | Chatterbox-turbo GPT-2 per-layer CPU-vs-PyTorch (#94). The dumper patches `GPT2Block.forward` to capture post-attn/post-FFN at T==1 AR steps; the differ globs `/tmp/{cb,py}_gpt2_step_*` and prints `cos / rms_cpp / rms_py / rms_diff / max_abs_diff` per layer | the C++ side is driven by `CRISPASR_CHATTERBOX_DUMP_GPT2_LAYERS=1` with `CRISPASR_CHATTERBOX_T3_SEED=0` and **`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`** on the Python side (full command in the dumper's docstring, `:17-34`) |
| `tools/compare_probe_dumps.py` | Chatterbox S3Gen UNet CPU-vs-GPU bisector. No args; reads fixed `/tmp/cb-unet-dump-{cpu,gpu}-probe-*.bin`, per-stage cosine, then a column-level drill-down for any stage below `cos 0.99` (`:114-160`) | `python tools/compare_probe_dumps.py`, after two `crispasr` runs with `CRISPASR_S3GEN_UNET_PROBE_BLOCK1=<N>` |

**The `beatrice_torch_parity.py` lesson generalises**: for a stage whose error
is a small rotation rather than a scaling, relative max-abs is the stronger
gate. Pick the statistic that the defect you fear would actually fail.

### 5.6 C++ dump knobs for these models

| var | file:line | effect |
| --- | --- | --- |
| `CRISPASR_BTC_DUMP_FEAT=<path>` | `btc_chords.cpp:542-551` | **writes a file**: raw-binary dump of the computed log-CQT feature matrix. Exists because the per-stage diff replays the reference's own `input_feat` and therefore *cannot* catch a CQT mismatch |
| `CRISPASR_BTC_DEBUG` | `btc_chords.cpp:140` | stderr debug |
| `CRISPASR_CREPE_DEBUG` | `crepe.cpp:330` | stderr prints at `:434, :460` |
| `CRISPASR_CREPE_NO_BAKE_F32=1` | `crepe.cpp:392` | A/B the in-graph F16→F32 kernel cast |
| `CRISPASR_BASIC_PITCH_FASTCONV=0` | `basic_pitch_conv.h:338-341` | revert to the reference conv loop |
| `CRISPASR_BASIC_PITCH_CONV_ISA=scalar\|avx2\|avx2fma\|avx512` | same | force a kernel for A/B |
| `CRISPASR_BASIC_PITCH_TIMING=1` | `basic_pitch.cpp` | per-window cqt/hstack/conv/activation split to stderr |
| `CRISPASR_PIANO_SERIAL=1` | `piano_transcription.cpp:218` | force `n_threads = 1`, recovering the exact serial order |
| `CRISPASR_AUDIT_QUANT_BCAST=1` | `quant_bcast.h:63-72` | walk the graph for quantised `MUL_MAT` nodes whose src1 batch dims exceed src0's |

⚠ **`basic_pitch.cpp`, `piano_transcription.cpp` and `mt3.cpp` have no `getenv`
calls at all.** Any new path in those files needs its gate added — env-var
gating is mandatory (`../crispasr-crispembed-dev.md` §"Env-var gating").

### 5.7 The minimum bar for a rewrite in this family

1. **Before touching anything**: capture the current output and a timing
   baseline, one arm per process, CPU time primary.
2. **Add the per-layer dumper if the model lacks one** — O&F does; crepe's is
   half-wired. This is the step people skip and then pay for.
3. **Gate the new path behind an env var and keep the old one.** Never delete
   the working path.
4. **Diff per stage, from the earliest.** Read `|mine|` and `|ref|`, not just
   cosine.
5. **Validate with a repeated-call test**, not a single run — second-use
   allocator corruption is invisible to a one-shot CLI invocation (§6.7).
6. **Judge the decoded output too**, not only the tensors: note events, F1.
   `tools/oaf_musicnet_f1.py` is the task-level gate for the piano models.
7. **Flip the default only when the new path wins on speed *and* quality.**
   Otherwise it stays opt-in — the inverse-default rule.


---

## 6. Anti-patterns, with real examples from this tree

### 6.1 A fresh allocator per call

**`src/onsets_and_frames.cpp` does this in three places, one of them inside a
loop.** Verified counts for the file: three `ggml_gallocr_new` sites, nine
`ggml_gallocr_free` sites, and **zero** `ggml_gallocr_reserve` calls.

| function | lines | allocator new/free | called |
| --- | --- | --- | --- |
| `oaf_conv_stack_chunk` | 403-471 | new `:445`, free `:449, 459, 467` | **inside** `for (t0 = 0; t0 < T; t0 += OAF_CONV_CHUNK)` at `:483`, call site `:490` |
| `oaf_lstm_input_proj` | 507-556 | new `:525`, free `:529, 537, 543` | twice per BiLSTM direction, `:602`, `:606` |
| `oaf_apply_head` | 615-660 | new `:633`, free `:636, 644, 650` | five times per forward, `:682, 690, 696, 702, 716` |

`src/btc_chords.cpp` has the same shape at smaller scale: `ggml_gallocr_new` at
`:432`, `ggml_gallocr_free` at `:472`, inside `btc_forward_block` (`:366`),
which is called per block from the loop at `:576`. T is fixed by the chunk
geometry, so a single reserved allocator is a direct substitution.

**Fix:** the `hft_transformer.cpp:108-115` pattern — one allocator per graph
shape as a context member. Its own comment names the O&F bug as the reason.

**Smell test** for finding more: asymmetric `new`/`free` counts with zero
`reserve`. Other files flagged by that heuristic but *not* verified:
`irodori_tts.cpp`, `dots_tts.cpp`, `htdemucs.cpp`, `confucius4_tts.cpp`,
`omnivoice.cpp`.

### 6.2 Hand-rolled recurrence where a shared header exists

`src/onsets_and_frames.cpp:558-591` (`oaf_lstm_recurrence`) is a scalar
LSTM recurrence with its own `oaf_sigmoid` (`:179`) and `std::tanh`, while
`src/core/lstm.h` provides a ggml-graph bidirectional LSTM with the projection
hoisted. **But read §1b before treating this as an unconditional bug** — the
unrolled-graph node count at O&F's T is a real risk and the honest order is:
fix the allocator, measure, then decide.

`src/core/gru.h` (140 lines) is a real ggml-graph GRU with the same hoisted
input projection (`gru.h:70-72`) and a bidirectional wrapper (`gru_bidir:132`),
and it documents two traps a naive port gets wrong (`gru.h:14-25`): the reset
gate multiplies the *recurrent term* `r*(W_hn h + b_hn)`, not `h_{t-1}`; and
`b_ih`/`b_hh` **cannot** be pre-summed the way LSTM's can. But it has **zero
callers** — only `tests/test-core-gru.cpp` exercises it. It has never run in
production. `piano_transcription.cpp` hand-writes its GRU (`gru_forward`, called
at `:423-427`). Adopting `gru.h` is a port, not a drop-in, and the §1b node-count
caveat applies to it identically.

### 6.3 ggml as a weight container

`src/basic_pitch.cpp` (922 lines) and `src/piano_transcription.cpp` (1461
lines) build **no** compute graph — zero `ggml_build_forward_expand`, zero
`ggml_graph_compute`. ggml holds the GGUF weights and everything runs in
hand-written C++.

**This is not automatically wrong.** For Basic Pitch it is the *measured* right
answer (§1 decision table, and §7 below). For piano-transcription the case is
argued in the file itself. State the argument and measure it; do not port on
principle. The dev guide's inverse-default rule
(`../crispasr-crispembed-dev.md`) exists for exactly this: a verified-correct
but net-slower path stays opt-in.

### 6.4 `n_threads` stored and never applied

Three backends have shipped this bug: the context records `n_threads`, nothing
ever calls `ggml_backend_cpu_set_n_threads`, and the model silently runs at
`GGML_DEFAULT_N_THREADS` regardless of `-t`. `src/wespeaker.cpp:564-568`
carries the fixed version and its own postmortem:

```
// n_threads used to be stored and never applied, so every context silently
// ran at ggml's default no matter what the caller asked for — and any
// attempt to run several contexts at once oversubscribed the machine by
// that default factor.
```

Chatterbox had the same (HISTORY §210/§212). Basic Pitch had a variant of it:
`n_threads` reached only GGUF loading, never the conv loops, which is why 2 and
4 threads measured identical (`BASIC_PITCH_CONV_PERF.md` §2.2).

**How to check, one line per backend:**

```bash
grep -n "n_threads" src/<backend>.cpp | grep -v "set_n_threads" 
grep -c "core_cpu_backend::set_n_threads\|ggml_backend_cpu_set_n_threads" src/<backend>.cpp
```
A non-zero first result with a zero second is the bug.

### 6.5 Hand-threading that fights ggml's threadpool

See §3. Do not wrap `core_parallel::for_each_*` around a region that computes a
ggml graph.

### 6.6 Not re-setting inputs of a persistent graph

`ggml_gallocr` may alias an input-flagged tensor's slot with a later
intermediate. Symptom: **step 0 bitwise-equal, step 1 onward diverged**
(`LEARNINGS.md` L14367, omnivoice). Treat *every* `ggml_set_input` tensor as
volatile, including "constant" ones like positions or a fixed mask, and re-set
it before every compute. `crepe.cpp:308` does this and says why.

This also bites the *diff harness*: reading an unmarked input tensor after
compute returns whatever intermediate reused the buffer. Dump from a
`ggml_set_output`-marked tensor or from the raw input buffer, never from an
unmarked input post-compute (`../crispasr-crispembed-dev.md`, voxtral-tts case).

### 6.7 Caching a `cgraph` across calls on a shared scheduler

Do not. `ggml_backend_sched_reset` + `sched_alloc_graph` on the *same* cgraph
object does not clear the cached tensors' `buffer`/`data` from the previous
allocation; the **second and every later** call reads stale memory. The failure
is not a crash — Parakeet's encoder output merely *shifted* (std 0.020 → 0.014)
and the decoder emitted only blanks (`LEARNINGS.md` L2458, #208). If you want a
cached graph, drive it on its own reserved `gallocr` with **no sched**
(`step_graph_cache.h:66-70` explains the reasoning).

Corollary: **validate any perf cache with a repeated-call test.** A one-shot CLI
invocation can never surface a second-use corruption.

### 6.8 An unused graph output still runs

ggml executes every graph output. An opt-in experiment whose baseline still
*constructs* the experimental output tensors is not measuring the baseline
(`LEARNINGS.md` L18863, #81). Gate graph *construction*, or use a separate graph.

### 6.9 Benchmarking mistakes that have fabricated results here

- A gate's default is part of every harness that reads it. When the Basic Pitch
  default flipped, the "reference" arm — which passed no env var — silently
  became the fast path and the CI A/B reported `1.00x` for every arm, green
  (`BASIC_PITCH_CONV_PERF.md` §4, run 35471993867). Assert the arm name the
  binary prints against what was requested.
- **RTF is a lie until you prove the work.** A crashed or no-op run mints a fake
  win; the tell is that the "55 s" clip took the same wall time as the "11 s"
  one (`../crispasr-crispembed-dev.md` A/B rule 4a).
- On a contended box, judge by CPU time, not wall — and know that CPU time
  *understates* a SIMD win, because contention hurts the faster kernel more
  (`BASIC_PITCH_CONV_PERF.md` §5: the clean runner showed 1.7× where the VPS
  showed 1.45×).
- Run each arm as a **separate process**, discard the cold run, take a median of
  ≥3, and hold load constant.

---

## 7. Which of the six models gets which treatment

### Summary

| model | lines | graph today | verdict |
| --- | --- | --- | --- |
| `src/onsets_and_frames.cpp` | 862 | 3 graphs, allocator per call | **Fix the lifecycle. Do not rewrite.** Highest value per unit of work in the set. |
| `src/btc_chords.cpp` | 756 | full graph transformer, allocator per block | **One-line-shaped fix.** Hoist the allocator; otherwise leave alone. |
| `src/basic_pitch.cpp` | 922 | none, by decision | **Leave it.** The graph-free design is measured-correct. One narrow follow-up. |
| `src/piano_transcription.cpp` | 1461 | none, by decision | **Leave the loops. Screen one layer.** The #305 argument holds. |
| `src/crepe.cpp` | 524 | persistent build-once graph | **Leave it — and copy it.** This is a reference implementation, not a problem. |
| `src/mt3.cpp` | 1815 | 7 graphs, sched, 2 KV caches | **Leave it.** The premise that it has no graph is wrong. |

So: **one backend needs real work, one needs a small fix, and four are either
already right or right for reasons that were argued and measured.** That is a
different conclusion from "three of six never build a graph", and the
difference is the call-graph point in §0.

---

### `src/onsets_and_frames.cpp` — fix the lifecycle, in that order

This is the one with a measured gap: **0.671 CPU-seconds per audio-second
single-threaded against native ONNX Runtime's 0.103**, 6.5×
(`docs/music-transcription/HFT_TRANSFORMER.md` §"the gap to ORT"). The same doc
names the two causes: *"the O&F port is a scalar C++ recurrence and a fresh
allocator per chunk."*

**Do these in order, measuring between each. The order is the point.**

1. **Hoist the allocators.** Three `ggml_gallocr_new` sites, one of them inside
   the per-chunk loop; zero `ggml_gallocr_reserve` calls anywhere in the file
   (§6.1 has the line numbers). Replace with one allocator per graph shape as a
   context member, `ggml_gallocr_reserve`'d once, on the
   `src/hft_transformer.cpp:108-115` pattern — whose comment names this exact
   bug as its reason for existing. This is mechanical, low-risk, and it is the
   change most likely to move the number.
2. **Apply `n_threads`.** Check for §6.4 (`n_threads` stored, never applied).
   `core_cpu_backend::set_n_threads` immediately before each
   `ggml_backend_graph_compute`, as `hft_transformer.cpp:537, 607` does.
3. **Fuse the three conv-stack chunk graphs and the five head graphs into
   fewer computes.** Eight `gallocr_alloc` + compute round trips per forward is
   a lot of dispatch for a small model. The voxcpm2 result is the precedent:
   folding many tiny graphs into one per-call graph gave **2.3× on CPU before
   any other change** (`LEARNINGS.md` L9079), and the win was amortising
   graph-build overhead, not the matmul work.
4. **Only then consider the BiLSTM.** §1b is the argument: `core/lstm.h` unrolls
   ~12 nodes per timestep, and at O&F's T ≈ 3000 frames per 30 s that is a
   ~200k-node graph across three BiLSTMs. **I have not measured it and I am not
   claiming the graph BiLSTM wins at that T.** If steps 1–3 close most of the
   gap, this may not be worth doing at all. If it is attempted, chunk T with
   carried `(h, c)` to bound the node count — exact, because the recurrence is
   exact.

**Before any of it, build the parity harness.** O&F is the one model of the six
with **no `tools/reference_backends/` dumper and no `crispasr-diff` arm** — see
§5. What it has — `tests/oaf_parity_dump.cpp` + `tools/oaf_parity.py` against
native onnxruntime — is end-of-pipeline (mel plus five heads), **not per-layer,
and not a gate**: the script always exits 0 unless the dump subprocess itself
fails (`oaf_parity.py:154-156`). A per-layer cosine check is what localises a
regression in step 3 or 4 to a layer instead of to "the output moved". Adding
an O&F dumper on the `crepe.py` contract (§5.1) and a threshold to
`oaf_parity.py` is cheap next to debugging without them. Also pin
`MKL_NUM_THREADS=1` for every run — see §5.4 and issue #453.

**Do not skip the gate.** `../crispasr-crispembed-dev.md` A/B rule 1: keep both
paths behind an env var, never delete the working one.

---

### `src/btc_chords.cpp` — hoist the allocator, stop there

Already a full ggml graph: transformer with two attention blocks per layer,
`ggml_soft_max_ext` with the correct mask, `mul_mat` throughout
(`btc_chords.cpp:185-200, 393-424`). The only lifecycle defect is
`ggml_gallocr_new` at `:432` / `ggml_gallocr_free` at `:472` inside
`btc_forward_block` (`:366`), which is called per block from the loop at `:576`.

T is fixed by the chunk geometry (`:513`), so this is the easy case: one
context-member allocator, reserved once, is a direct substitution. Everything
else in the file — including the hand-built LayerNorm, which `ggml_norm`
genuinely **cannot** express (`btc_chords.cpp:11-13`) — should be left alone.
It has a parity spec (`tools/btc_torch_parity.py`, held at cos ≥ 0.99999,
`btc_chords.cpp:3-5`), so the change is cheaply verifiable.

---

### `src/basic_pitch.cpp` — leave it graph-free

The graph-free design is not an oversight; it is the measured answer, and the
reasoning generalises. `docs/music-transcription/BASIC_PITCH_CONV_PERF.md` §7
prices `ggml_conv_2d` for the dominant layer:

> The im2col matrix for `contour_conv` is `45408 × 936` floats = **170.0 MB per
> window**, against a current largest activation of 1.45 MB … But the decisive
> argument is that the blow-up buys nothing. `contour_conv` has **OC = 8**. …
> pay 170 MB of writes and 170 MB of reads to save nothing.

The direct-SIMD path already ships as the default and is **byte-identical** to
the reference loop at 1 and 4 threads, at **1.82× / 3.81× on x86-64 and 2.39× on
arm64** on clean CI runners.

Two narrow follow-ups, both already scoped in that document, neither a rewrite:

- **`onset_conv` is the one layer where `ggml_conv_2d` could plausibly win** —
  im2col is 12.1 MB, `OC = 32`, so there is real amortisation. 20% of the
  network, currently 1.86 GMAC/s, untouched by the SIMD work because
  `stride_w == 3` gives a gathered inner loop. Route *only that layer*, measure,
  keep it gated.
- **`core_parallel::for_each_chunk` spawns threads per call**, so 4 threads
  costs ~75% more CPU than 2 for ~9% less wall. `n_threads = 2` is the better
  operating point today; `core/worker_pool.h` would remove the trade.

If anything here gets a ggml treatment, the `ggml_map_custom1` bridge (§1a) is
the shape to consider — but there is no evidence it would help, because nothing
in this model needs a scheduler.

---

### `src/piano_transcription.cpp` — the #305 argument holds; leave the loops

The brief asked whether "rewrite everything as graphs" is correct here. **It is
not, and the file says why.** `piano_transcription.cpp:205-214`:

```
// #305: everything below the mel front-end (which core_mel already threads) ran
// on one core — the 3x3 convs, the dense layers and the two GRU directions.
// Each is split along an axis whose iterations write disjoint outputs and only
// READ shared inputs, and the innermost accumulation order is untouched, so the
// result is bit-identical to the serial path. Opt out with
// CRISPASR_PIANO_SERIAL=1.
//
// Deliberately NOT parallelized: the four acoustic_model_forward calls in
// piano_transcription_transcribe. Running those concurrently would hold four
// sets of conv activations at once, and the box this ships on has 8 GB.
```

Three things make this a good design rather than an unfinished one:

1. **The bit-identity claim is structural, not a tolerance.** The parallel axis
   writes disjoint outputs and the innermost accumulation order is unchanged.
   `core_parallel`'s own contract (`parallel_for.h:27-30`) requires exactly
   this, and `for_each_task` at `nthreads == 1` runs tasks in the original
   order (`piano_transcription.cpp:419-421`), so the serial path is recoverable
   exactly.
2. **The thing it declines to parallelise, it declines for a stated resource
   reason**, not from neglect.
3. **The threading is not fighting a ggml threadpool**, because there is no
   graph to fight. Converting to graphs would introduce exactly the nesting
   hazard §3 warns about, unless the hand-threading came out at the same time.

What *would* be worth doing, as a screened experiment and not a rewrite:

- **Run the im2col arithmetic on its conv layers.** They are wide-channel 2-D
  convs into GRUs, which `BASIC_PITCH_CONV_PERF.md` §7.3 predicts is on the
  *favourable* side of the line. One multiplication tells you; if it is
  favourable, route one layer and measure.
- **Note that it already uses `core/parallel_for.h`, `core/mel.h` and
  `core/fft.h`** — it is one of the better-integrated files in this set, and
  `fuse_bn` (`:229`) is the shared-pattern BatchNorm fold.
- The GRU has the same `core/gru.h` question as O&F's LSTM, with the same
  §1b caveat and the additional fact that `gru.h` has never had a production
  caller. Low priority.

---

### `src/crepe.cpp` — a reference implementation, not a problem

524 lines, one persistent build-once graph, a context-member `gallocr`
(`:135`) allocated once (`:452-453`) and freed only at teardown (`:470`). It
independently applies four separate lessons this tree learned the hard way:
deliberate avoidance of `ggml_conv_1d` for both correctness and cost reasons
(`:150-164`), symmetric-pad-then-drop-a-column for Metal's asymmetric-PAD
restriction (`:12-16`), re-setting every input on every compute (`:308`), and
baking F32 conv kernels once at load because the in-graph cast re-runs per
compute (`:386-400`, measured at RTF 31 on M1 before the fix).

It is also validated against `tools/crepe_numpy_parity.py` at cos = 1.0 on both
capacities, with the file header instructing that the script changes first
(`:3-5`).

**Recommendation: change nothing, and use it as the template for the O&F
rewrite.** The one open item its own header names is that batch > 1 is
unavailable because ggml's `ggml_conv_1d` batch path is broken here
(`:21-26`) — if that is ever fixed upstream, CREPE gets batching for free.

---

### `src/mt3.cpp` — the premise is wrong

The brief listed mt3 as "0 ggml graph ops, no threading, a transformer doing
that across 1815 lines". That is not what the file does:

| | where |
| --- | --- |
| encoder graph | `mt3.cpp:708, 728, 771` |
| cross-KV projection graph | `:839-861` |
| decoder step graph with in-graph KV writes | `:888, 931-932, 987` |
| persistent scheduler | `ggml_backend_sched_graph_compute` at `:861, 1008, 1046` |
| decoder KV cache, 4-D, own context + backend buffer | `:788-795` |
| cross-attention KV cache, per layer, rebuilt only when `T_enc` changes | `:802-821` (`:802` short-circuits on an unchanged length) |
| a comment costing out `sched_reset`/`alloc_graph` per step against graph size | `:680` |

That is the standard pattern §2/§3 describe, correctly applied. Its cross-KV
caching in particular — build once per encoder length, reuse across every
decode step — is the piece of KV machinery that *does* transplant to a
non-AR setting, and worth reading if another model needs it.

**Recommendation: no rewrite.** If mt3 is slow, the honest next step is a
per-node trace, not a port — `LEARNINGS.md` L14428's three graph-construction
wastes (K=1 convs through im2col, CPU-placed pad nodes, per-graph F16→F32
kernel casts) are all things a trace finds in one run and a rewrite would
reproduce. And L14605's rule applies: **the big wins are bugs, not micro-opts.**

---

### What to measure before and after, for all of them

Per `docs/music-transcription/BASIC_PITCH_CONV_PERF.md` §5 and the dev guide's
A/B rules: **CPU time** (`CLOCK_PROCESS_CPUTIME_ID`, min of ≥8 in-process
iterations, ≥3 separate processes per arm, cold process discarded) as the
primary metric, wall reported alongside as a lower bound. One arm per process.
Assert the arm name the binary prints against the arm you asked for. And prove
the work happened — an exit code is not proof (HARD RULE 8).

This box is not a measurement instrument: a single-threaded process here gets
less than half of one core, and wall-clock swings 2–4× run to run. Use CI
runners (`.github/workflows/basic-pitch-conv-ab.yml` is the template; it
reproduced to 0.05% variance) or a Kaggle box for any number that will be
quoted.



---

## 8. What this document does not establish

Stated explicitly so nobody builds on a gap thinking it is a finding.

1. ~~**Whether selecting ggml's repack extra buffer type actually speeds
   anything up here.**~~ **SETTLED** — `ggml-repack-buft-evaluation.md`, and §4
   above is updated. It is offered on every machine tested. It pays **2.3–3.4×
   for q8_0 on arm64 — the models exactly as they ship, no re-quantisation** —
   and 2.1–2.7× for q4_0/q4_K on x86, where q8_0 gets no repack kernel at all.
   None of that needs VNNI. The mmap incompatibility is now verified rather
   than inferred. What remains open is narrower: whether a CPU with VNNI or AMX
   changes the x86 ratios — **no x86 machine reachable from this project has
   one**, the VPS, a Kaggle Xeon and GitHub's EPYC 7763 all lack it — and
   whether q4_0/q4_K hold their accuracy well enough to be worth adopting on
   x86, which is a question for the F1 harness and not this one.
2. **Whether a ggml-graph BiLSTM beats a scalar recurrence at O&F's sequence
   length.** §1b sets out the node-count concern. Nobody has measured it. The
   recommendation in §7 is deliberately ordered so this question is answered
   *after* the cheap fixes, on real numbers.
3. **Whether hand-threading around a ggml graph is explicitly forbidden
   anywhere in this tree.** I found no such comment. §3's rule is a conclusion
   drawn from the hFT thread-scaling number, the three documented
   oversubscription cases, and how ggml's spin-wait barriers work — not a quote.
4. **How much of the 6.5× O&F gap each of the four steps in §7 recovers.**
   `docs/music-transcription/HFT_TRANSFORMER.md` names the allocator and the
   scalar recurrence as the two causes; it does not apportion them.
5. **Whether `core/gru.h` and `core/cross_attn.h` are correct.** Both are
   written, both have zero production callers, and `gru.h` has a unit test while
   `cross_attn.h` has nothing. Treat adopting either as a port.
6. **Anything about GPU backends for these six models.** Everything here is the
   CPU path. The GPU portability traps in `../crispasr-crispembed-dev.md`
   (Metal asymmetric PAD, no k-quant CPY, the weight-less-first-op sched bug)
   apply and are not restated.
7. **The benchmark numbers in this document were not taken on this box.** Every
   quoted figure comes from a CI runner, a Kaggle box, or a cited document.
   This VPS gives a single-threaded process less than half a core; nothing timed
   here should be quoted.
