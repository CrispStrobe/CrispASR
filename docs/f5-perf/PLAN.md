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

### Changes

| # | Change | Gate | Status |
|---|--------|------|--------|
| 4 | F16 activations in DiT matmuls | `CRISPASR_F5_F16_ACT` | built + gated. **Metal: byte-identical + ~17% SLOWER** (ggml already casts RHS to F16 internally). Kept gated OFF for a CUDA-only A/B; do NOT default on. |
| 2 | host-embed → GPU graph | tbd | TODO — biggest new-code win for reporter's slow CPU; correctness-verifiable locally, speed verdict needs CUDA (M1 GPU already the bottleneck ⇒ expect M1 regression / reporter win). |
| 6 | persistent inputs for CUDA-graph replay | tbd | TODO (CUDA-only verdict) |
| — | higher-order ODE solver | tbd | LOW priority — 16-step Euler already holds quality here. |

### Reporter comms
- Posted knobs (`--tts-steps 16`, `CRISPASR_F5_CFG_INTERVAL=2`), `-nfa`-is-a-no-op,
  and `CRISPASR_F5_BENCH=1` request. issue #294 comment.
- TODO: follow up with the validated 2.6× numbers + shorter-reference tip.
