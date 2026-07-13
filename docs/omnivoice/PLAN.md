# OmniVoice — issue #254 (voice cloning + RTF)

Branch: `fix/omnivoice-254-voiceclone-rtf` (rebased onto `main` on top of the
stranded GPU commit `feat/omnivoice-gpu` = "run the LLM on GPU").

## NOW — active work

**Status: investigation complete, root causes confirmed, harness setup in flight.**

- ✅ Rebased the stranded `feat/omnivoice-gpu` (single commit `c80328e08`, never
  merged to main) onto current `main`. GPU-backend init + `CRISPASR_OMNIVOICE_CPU`
  gate are the only pre-existing changes.
- ✅ Model SHA verification vs HF `cstr/omnivoice-GGUF` (user requirement):
  - `omnivoice-q8_0.gguf` local == HF `4e0bbc93…` ✅
  - `omnivoice-tokenizer-f16.gguf` local == HF `9cb7741a…` ✅
  - (f16 LLM not yet local; download+verify when F16 A/B is needed.)
- ✅ **Baseline RTF (prove-the-work)**: GPU/Metal, q8_0, 60-char sentence →
  14.64 s audio in **93.3 s wall** (`user` 30 s, so ~63 s is GPU dispatch/sync
  wait). **RTF ≈ 6.4** — far slower than real-time. `real ≫ user` ⇒ launch/sync
  bound, not compute bound. Reporter says omnivoice.cpp is 3–4× faster ⇒ ~RTF 1.8.
- 🔄 **In flight**: downloading `k2-fsa/OmniVoice/audio_tokenizer` (806 MB) for the
  encode-path reference dump; blueprint-research agent pinning the HiggsAudioV2
  encode() call graph.

### Next
1. Extend `tools/reference_backends/omnivoice.py` to dump **encode-path** stages
   (semantic feats → VQ, acoustic latent → residual VQ, final codes) + existing
   forward stages → `ref.gguf` for `crispasr-diff` (mandatory per dev-guide regime).
2. Implement C++ `higgs_encode` (WAV→codes) + wire `omnivoice_set_voice_prompt`.
3. Land RTF wins (gated + A/B'd against decoded output).

## Root causes

### Issue #1 — voice cloning "doesn't work": it's an unimplemented stub
`omnivoice_set_voice_prompt` (`src/omnivoice.cpp`) has a literal
`// TODO: load WAV, encode through audio tokenizer` and never populates
`ref_audio_codes` / `ref_T`. So `generate_iterative` always runs with `T_ref=0`
(no reference conditioning). The `--voice`/`--ref-text` CLI path is wired, but the
encode is missing.

**Feasibility**: all encoder weights ARE in `omnivoice-tokenizer-f16.gguf`
(`acoustic_encoder.*`, `sem.*` HuBERT 12L, `fc/fc1/fc2`, per-quantizer
`project_in`). The tokenizer encode path (per converter docstring) is:
HuBERT semantic encoder (12L, 768d, 16 kHz; 7-conv feat-extractor strides
[5,2,2,2,2,2,2]) + DAC acoustic encoder (downsample [8,5,4,2,3], hop 960, 16 kHz) +
quantizer bridge (semantic VQ + acoustic residual VQ) → 9 code streams, OmniVoice
uses 8. Frame-rate alignment (sem 50 Hz / ac 16.7 Hz / LLM 75 Hz) is the key
ambiguity to pin from the blueprint.

**Reuse (do not reinvent)**: `core/dac_decoder.h` (`snake`, `conv1d`),
`core/conv.h` (strided/transpose conv), `core/rvq.h::encode_euclidean` (argmin
quantize — already used by `mimo_tokenizer` + `kyutai_stt`).

### Issue #2 — RTF 3–4× worse: launch/sync-bound GPU path
`real ≫ user` on the baseline. Suspects (in the "unified graph / caching / sched /
alloc" family):
- **Per-step embedding lookups do gallocr new/alloc/compute/free** twice
  (text+audio) per arm per step (`read_embedding_rows`, 32 steps × 2 arms) — pure
  launch/alloc overhead on Metal.
- **Final `audio_output_w` projection runs over all `T_total` positions** but only
  `T_target` are used — a large wasted matmul (output = n_codebooks·audio_vocab).
- **cond + uncond are two separate graph computes** — candidate for the
  seq-concat + block-diagonal-mask unified graph (dev-guide CFG fusion learning).
- The persistent-graph reuse (#245) is already in for the two LLM arms; good.

## Validation regime (mandatory)
- Per-stage diff vs Python `ref.gguf` (`crispasr-diff`), earliest divergence = bug.
- Decoded-output roundtrip: TTS→ASR overlap; voice-clone closed-loop
  cosine(C,R) > cosine(B,R) (Resemblyzer).
- ServeurpersoCom/omnivoice.cpp = **black-box output oracle only**; do not read its
  code until our own solution exists, then compare to optimize.
- Every perf path env-gated + A/B'd (F16 + Q8), default flipped only on speed AND
  quality win.
