# F5-TTS performance — issue #294

## NOW — active work

Branch `perf/f5-speedups` (worktree `CrispASR-f5perf`). Reporter: F5-TTS slow on
RTX 5060 Ti (sm_120) + Ryzen 5 2600, F16 model, 32 ODE steps.

**Hard constraint:** F16 is the ONLY viable format for F5 (flow-matching:
every weight used 1408×/synth, q8's ~0.5% error compounds → unintelligible;
see `hf_readmes/f5-tts-GGUF.md`). So **no quantization** — all wins come from
compute efficiency and fewer/smaller forward passes.

**Methodology (mandatory, dev doc §"A/B every perf optimization"):** every change
gated + default OFF; negative control = gate-off must stay byte-identical
(md5) to baseline; judge the ON path by TTS→ASR roundtrip, not cosine. M1 Metal
GPU timing is dispatch-bound/noisy → **speed verdict + any default-flip needs a
CUDA A/B (reporter box or Kaggle)**, not M1.

### Baseline (M1 Metal, F16, jfk ref + "quick brown fox…", seed 42, 32 steps)

- ref_T=1032 duration=1662 (ref = 62% of T)
- text_embed 87 ms
- **ode_solve 60.9 s** = host_embed (CPU) **15.6 s / 26%** + dit_graph (GPU) **45.3 s / 74%**
- vocos 0.4 s (**negligible** — vocos only decodes the generated frames)
- md5 `e249b19c8822b5b061d302839ef65678`; roundtrip = exact ("The quick brown fox
  jumps over the lazy dog and then it ran away.")

### Where time goes → priorities

1. **dit_graph 74%** → F16 activations (#4). [IN PROGRESS]
2. **host_embed 26%** (bigger on reporter's slow CPU) → move input-embed into the
   GPU graph (omnivoice-style). [TODO]
3. **NFE reduction** (deterministic, box-independent): fewer steps (`--tts-steps`,
   shipped), interval-CFG (`CRISPASR_F5_CFG_INTERVAL`, shipped), higher-order ODE
   solver (new). [TODO]
4. Shorter reference clip → smaller T on every forward (user-side, free). [TELL REPORTER]
5. ~~Vocos GPU/FASTCONV~~ — DROPPED, vocos is <1%.

### Validated NFE levers (M1 Metal, roundtrip intact — DETERMINISTIC, carries to CUDA)

| Config | ode_solve | Speedup | Roundtrip |
|--------|-----------|---------|-----------|
| baseline (32 steps) | 60.9 s | 1.0× | perfect |
| `--tts-steps 16` | 30.3 s | **2.01×** | perfect ✅ |
| `CRISPASR_F5_CFG_INTERVAL=2` | 46.6 s | **1.31×** | perfect ✅ |
| 16 steps + interval 2 | 23.5 s | **2.59×** | perfect ✅ |

These use existing knobs — the win is validating + recommending them. Fewer/skipped
forward passes ⇒ speedup is box-independent (unlike the GPU-compute changes below,
which need a CUDA verdict).

### Ecosystem research (how F5 runs elsewhere) — reshapes priorities

- Upstream (SWivid) = torchdiffeq **Euler**, nfe **32** (16 offered), **CFG as ONE
  2×-batch forward** (our `F5_BATCH_CFG`, i.e. our *default two-forward path is the
  non-standard one*), sway −1.0, Vocos. cfg_strength 2.0.
- **EPSS (arXiv 2505.19931)** training-free non-uniform step pruning → **~4× at 7 NFE**,
  quality stable to 7–12 NFE. **We already ship these schedules** in
  `get_epss_timesteps` (n=5/6/7/10/12/16). So the headline win is already coded —
  just needs validation + recommendation, no new kernel.
- **Guidance-free / interval CFG** halves per-step cost; interval form is portable and
  we already have `CRISPASR_F5_CFG_INTERVAL`. Paper RTF 0.31→0.17 by dropping uncond.
- **Layer caching across steps** (DiTReducio 2509.09748) — training-free, NEW lever we
  lack. Complex + quality-risky. Future direction.
- Reference length is a real lever (joint ref+gen DiT sequence). Confirmed.
- No verified Candle/burn F5 port, no vLLM/SGLang. MLX port ~8× RT on M3 Max.

**Conclusion: F5 is already well-optimized; the real wins are configuration
(EPSS low steps + interval-CFG + short ref + batched-CFG on CUDA), not new kernels.**

### Changes tried

| # | Change | Gate | Status |
|---|--------|------|--------|
| 4 | F16 activations in DiT matmuls | `CRISPASR_F5_F16_ACT` | built + gated. **Metal: byte-identical + ~17% SLOWER** (ggml already casts RHS to F16 internally). Committed, default OFF, CUDA-A/B-only. |
| 6 | stable-alloc (skip per-step re-alloc for CUDA-graph replay) | `CRISPASR_F5_STABLE_ALLOC` | **REVERTED — correctness bug**: pos_in clobbered after step 0 → garbage ("(wind blowing)"). Proper fix needs persistent input tensors on a dedicated buffer (omnivoice §245 pattern); CUDA-only value, unverifiable on Metal. Not worth it now. |
| 2 | host-embed → GPU graph | — | NOT DONE — ggml has no grouped conv1d (F5 conv-pos groups=16 ⇒ 16 sliced convs + concat in-graph), invasive + correctness-risky + only a CUDA win. Deprioritized vs config levers. |
| batched CFG default (CUDA) | `CRISPASR_F5_BATCH_CFG` | exists; validating correctness. Matches upstream. Candidate CUDA default. |
| EPSS low-NFE + interval | (knobs) | validating quality at n=7/10/12 (+interval). Primary recommendation. |

### Reporter comms
- Posted knobs (`--tts-steps 16`, `CRISPASR_F5_CFG_INTERVAL=2`), `-nfa`-is-a-no-op,
  and `CRISPASR_F5_BENCH=1` request. issue #294 comment.
- TODO: follow up with the validated 2.6× numbers + shorter-reference tip.
