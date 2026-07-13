# OmniVoice — issue #254 (voice cloning + RTF)

Branch: `fix/omnivoice-254-voiceclone-rtf` (rebased onto `main` on top of the
stranded GPU commit `feat/omnivoice-gpu` = "run the LLM on GPU").

## NOW — active work

**Status: investigation + blueprint complete; building isolated reference-dump env.**

- ✅ **Encode blueprint pinned** (cross-validated: HF transformers source +
  OmniVoice inference + omnivoice.cpp as spec-oracle). See "Encode blueprint" below.
  Key correction: **codec is 25 Hz, not 75 Hz** (hop=960; `downsample_factor=320`
  is a red herring). Baseline confirms: 366 frames × 960 / 24000 = 14.64 s. ✅
- ✅ **Profiling**: GPU/Metal gen_step ≈ 0.6–1.0 s each × 32 steps; CPU is *far*
  slower (short clip: GPU ~22 s, CPU still running at 3 min) ⇒ **GPU is the right
  backend** here — NOT the "small-model GPU-loses-to-CPU" pattern. RTF gap vs
  omnivoice.cpp is per-step launch/alloc + dual-forward overhead, not backend choice.
- ✅ Downloaded + SHA-verified `k2-fsa/OmniVoice/audio_tokenizer` (806 MB, sha
  `fe7c5e87…` == HF) for the encode reference.
- ⚠ **Env blocker**: `higgs_audio_v2_tokenizer` needs transformers ≥5.3 (base
  conda has 4.57.6). Building an ISOLATED venv (torch-cpu + transformers 5.x) for
  the dumper — must NOT upgrade the shared base (breaks other backends' dumpers).
- 🔎 Noticed (separate quality bug, not #254 scope): `estimate_target_tokens` uses
  a "75 Hz / 6 frames-per-char" heuristic; at the real 25 Hz that's ~3× too many
  target frames ⇒ over-long audio. Doesn't change RTF (numerator+denominator both
  scale) but bloats latency + trailing silence. Track separately.

### Next
1. Finish isolated venv → extend `tools/reference_backends/omnivoice.py` to dump
   encode stages (sem feats pre/post `[::2]`, `encoder_semantic` out, `e_acoustic`,
   post-`fc` emb, per-quantizer residual + codes) → `ref.gguf`.
2. Implement C++ `higgs_encode` (reuse `core_dac`, `core/conv.h`, `core_rvq`) +
   port HuBERT encoder; diff each stage vs `ref.gguf`.
3. Wire `omnivoice_set_voice_prompt`: resample→24k, clip to ×960, encode; and fix
   `generate_iterative` layout (add `<|denoise|>`, prepend ref_text to target text).
4. RTF wins, gated + A/B'd.

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

## Encode blueprint (HiggsAudioV2TokenizerModel.encode — for voice cloning)

Cross-validated against HF transformers `modeling_higgs_audio_v2_tokenizer.py`,
OmniVoice `omnivoice/models/omnivoice.py`, and ServeurpersoCom/omnivoice.cpp
(spec-oracle only; code not copied).

**Constants (derived, not raw JSON):** hop_length=960, **frame_rate=25 Hz**,
hidden=1024 (256 acoustic + 768 semantic), **num_quantizers=8**,
semantic_downsample_factor=2, pad=480, audio_vocab=1025, audio_mask_id=1024.

**encode(wav @ 24 kHz mono):**
1. Semantic: resample 24k→16k; `F.pad(x,(160,160))` (hard-coded 160, NOT pad=480);
   HuBERT `output_hidden_states=True`; **mean over all 13 hidden states**; then
   `[:, ::2, :]` → (T25, 768).
2. `encoder_semantic(sem.T)` → (768, T25). **Required module, weights present as
   `encoder_semantic.*` (13 tensors).** conv(768,768,k3,pad1,bias=F) + 2 blocks
   {2 res_units [ELU→conv1 k3 dil1→ELU→conv2 k1], block conv k3 pad1 bias=T}.
3. Acoustic: DAC encoder on 24 kHz wav → (256, T25). conv1(1,64,k7,pad3); 5 blocks
   ratios [8,5,4,2,3], ch 64→128→256→512→1024→2048, each = 3 ResUnit(dil 1,3,9)
   [Snake→conv k7→Snake→conv k1] + Snake + strided conv(k=2·s,stride=s,pad=ceil(s/2));
   snake1 + conv2(2048,256,k3,pad1). If conv-len ≠ sem-len, re-run with `F.pad(wav,(480,480))`.
4. `emb = cat([e_acoustic, e_semantic], dim=1)` (ACOUSTIC FIRST) → (1024, T25);
   `fc`(1024→1024) on transposed.
5. RVQ encode, 8 codebooks greedy residual: `idx_k = argmin_dist(project_in_k(res))`
   over `codebook_k.embed` (1024×64, Euclidean); `res -= project_out_k(embed[idx_k])`.
   → codes (8, T25), values 0..1023. Reuse `core_rvq::encode_euclidean`.

**HuBERT (semantic_model, post-norm variant):** 7 conv feat-extract (k[10,3,3,3,3,2,2]
s[5,2,2,2,2,2,2], dim512, bias=F; layer0 GroupNorm); feat_proj LN(512)+Linear(512→768);
pos_conv wnorm Conv1d(768,768,k128,groups16)+SamePad+GELU; `h = h + pos_conv(h)` →
`encoder.layer_norm` (pre-stack) → 12 layers [MHA12→res→LN→FFN(768→3072 GELU→768)→res→
final_LN]; NO trailing LN. 13 hidden_states = input-after-pos_conv+LN + 12 layer outs.

**Dead at inference (skip):** `fc1`(1024→768), `decoder_semantic.*`, RVQ EMA buffers.

**Voice-clone LLM sequence** `[style | text | ref_audio | target]`:
- style adds `<|denoise|>` when ref present, then lang/instruct tags; `.repeat(8,1)`.
- **text = `<|text_start|>` + (ref_text.strip()+" "+target_text.strip()) + `<|text_end|>`**
  — ref transcript PREPENDED into ONE combined text stream (not separate).
- ref_audio = tokenizer codes (8, T_ref), audio_mask=True.
- target = all `audio_mask_id` (8, T_target), audio_mask=True.
- Ref WAV: resample→24k, RMS-normalize (if 0<rms<0.1 scale to 0.1/rms), clip len to
  multiple of 960. Current `omnivoice_set_voice_prompt` does NONE of this (stub).

**Current-code gaps for voice clone (beyond the encode stub):** missing `<|denoise|>`,
missing ref_text prepend. Both in `generate_iterative` §3–4.
