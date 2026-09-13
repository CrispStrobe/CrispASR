# Supertonic-3 TTS port (#434)

## NOW — active work

- [x] Licence OpenRAIL-M verified from HF card (not gated).
- [x] Blueprint read line-by-line (py/helper.py) + all 4 ONNX graphs reversed.
- [x] Converter → GGUF (200 MB f16, 723 tensors) — runs locally.
- [x] Reference dumper (ORT intermediates, seeded noise).
- [x] C++ runtime + full 12-point wiring (src, CLI adapter, factory, arch map,
  CMake, C-ABI session ×10 points, registry, quantizer, diff harness, tests,
  README/docs/tts.md, Go LDFLAGS). All syntax-checked with g++ -fsyntax-only.
- [~] Kaggle validate: kernel `chr1str/crispasr-supertonic-434`.
  - v1 ERROR: ref dumper missing from branch (bash-cwd trap). Fixed e58b06da.
  - v2 ERROR, two findings: (a) te_* stages cos~0 with |mine|==|ref| to 1e-6 =
    the pre-e32ca6f2 reference layout transpose; (b) GGML_ASSERT(can_repeat)
    at ggml.c:2274 — ConvNeXt gamma ships (1,C,1) and cannot broadcast against
    (C,T); killed the vf graph AND both synth arms. Fixed c084a6d6.
    POSITIVES: text_ids byte-identical, dur PASS (whole CPU dp path correct),
    control_overlap 1.0 (ASR arm + upstream reference both sound).
  - v3 ERROR, one finding: everything cos=1.000000 through xt_8; ONLY "audio"
    failed (cos~0, |mine|/|ref|=0.613 — NOT a transpose signature). Localised
    locally with a numpy op-for-op sim of the C++ vocoder vs ORT with promoted
    Pad outputs: the VOCODER convs are CAUSAL — edge pad (K-1)*dil entirely on
    the LEFT (embed 6/0, convnext (K-1)*dil/0, head 2/0) while vf/dp/te are
    symmetric. Causal numpy sim == ORT end-to-end at cos 0.999999. Fixed
    5cf3c481; v4 in flight.
- [ ] v4: per-stage diff ALL PASS + TTS→ASR roundtrip (CPU + GPU) + control.
- [x] HF: f16 checkpointed; model card with license: openrail VERIFIED landed
  (api.model_info cardData). ref.gguf uploads on validation pass.

## VERIFIED vs ASSUMED (honest ledger)
- VERIFIED: upstream ORT pipeline reproduced locally (3.1 s wav, M1). Converter
  asserts every baked constant (CFG 4/3, rotary theta, time freqs, attn scales,
  edge-pad amounts) against the graph, so those are checked, not assumed.
- ASSUMED until the Kaggle diff returns: that the C++ ggml graph reproduces the
  ORT per-stage tensors. The VITS relative attention, GST tanh-key cross-attn,
  length-normalised rotary, and the CFG-fused Euler update are hand-derived from
  the graph and NOT yet validated against a reference dump. This is the whole
  point of the diff kernel; do not claim parity before it is green + roundtrip.

## Model facts (from onnx/tts.json + graph inspection, NOT guessed)

- 4 ONNX components, opset 19, all feed-forward/deterministic:
  `duration_predictor` (4 MB), `text_encoder` (36 MB), `vector_estimator`
  (257 MB), `vocoder` (101 MB). 44.1 kHz output, 31 languages, 10 preset
  voices (`voice_styles/{F1..F5,M1..M5}.json`: style_ttl [1,50,256],
  style_dp [1,8,16]).
- Text processing: NFKD → emoji strip → punctuation replacements → append "."
  if no trailing punctuation → wrap `<lang>`…`</lang>` → per-CHARACTER unicode
  codepoint → `unicode_indexer.json` (list[65536] → id, vocab 8322).
- Chunking: 300 chars max (120 for ko/ja), 0.3 s silence joins.

### duration_predictor
char_emb(8322×64) *mask → prepend learned sentence_token (CLS at pos 0, mask
extended with 1) → 6× ConvNeXt-1D (k5 dil 1..1, masked, LN eps per graph,
GELU-erf, gamma) → VITS attn encoder ×2 layers (2 heads, head 32,
emb_rel_k/v [1,9,32] window 4, post-LN, ReLU FFN k1 convs, masked) with
GLOBAL residual: out = attn_out + convnext_out → slice CLS pos → 1×1 conv
(64→64, no bias) → concat [sent(64) | style_dp.flat(128)] → Gemm 192→128 →
PReLU → Gemm 128→1 → Exp = seconds. Then dur /= speed (default 1.05).

### text_encoder
char_emb(8322×256) → 6× ConvNeXt (dil 1,1,2,2,4,4) → VITS attn ×4 (4 heads,
head 64, emb_rel [1,9,64]) → global residual add → *mask →
speech_prompted_text_encoder: 2× GST cross-attn (2 heads, split axis2 →
stack axis0): Q=text(256→256), K=tanh(W_k·style_key_prototype[1,50,256]),
V=W_v·style_ttl, scores/scale, softmax over 50, out_fc, *mask, residual;
then LayerNorm at end. Output text_emb [B,256,L].

### vector_estimator (flow matching, CFG INSIDE the graph)
- Batch doubled: cond half + uncond half (text→text_special_token bcast,
  style k/v→style_{key,value}_special_token).
- t = current_step/total_step; time emb: sin/cos(t·1000·freqs[32]),
  freqs_i = 10000^(−i/31) → MLP 64→256 →Mish→ 64.
- proj_in conv1x1 144→512 (no bias), *mask.
- 4 blocks, flat module list per block:
  [convnext×4 (dil 1,2,4,8) | time: x+=Linear64→512(t_emb) | convnext×1 |
   attn: rotary cross-attn text (8 heads, head 64, Q=x·mask, K/V=text_emb,
   angle=(arange(L)/actual_len)·theta_i, theta_i=10·10000^(−i/32),
   rotate-half 32|32, scores/16, mask=−inf, out_fc, *mask, residual,
   post-LN eps 1e-6) | convnext×1 | attention: GST style cross-attn
   (2 heads, Q=x·mask 512→256, K=tanh(style_key), V=style_value, out 512,
   residual, post-LN)]
- last_convnext ×4 (dil 1,1,1,1) → proj_out conv1x1 512→144 (no bias), *mask.
- Update (Euler, INSIDE graph): v = 4·v_cond − 3·v_uncond;
  xt ← (xt + (1/total_step)·v)·latent_mask.
- Noise: xt0 = randn(B,144,ceil(dur·sr/3072)) · latent_mask (chunk 512·6).

### vocoder
latent [B,144,L] → denorm (normalizer scale 0.25, ae.latent_mean/std
[1,24,1]) + decompress 144→24 ch × 6L frames → conv k7 24→512 → 10×
ConvNeXt (k7, dil 1,2,4,1,2,4,1,1,1,1, sym pad) → BatchNorm(512) →
head: conv k3 512→2048 → PReLU → conv k1 2048→512 (no bias) → reshape:
each frame = 512 samples → wav [B, 6L·512]. Trim to sr·dur.

### Constants verified from the graphs
- CFG: 4.0 / 3.0. Rotary theta[32] = 10·10000^(−i/32). Attn score /16.
- Time freqs[32] = 10000^(−i/31), t scale 1000. Mish in time MLP.
- ConvNeXt dwconv pads symmetric 2·dil (k5) / 3·dil (k7) — EDGE (replicate) mode in ALL four graphs, incl. vocoder embed + head layer1; zero-mode pads exist only inside VITS relative attention.
- softmax mask value −inf; post-softmax re-mask with 0 (VITS style).

## Design decisions
- ONE GGUF: all 4 nets + unicode indexer (i32[65536]) + NFKD decomposition
  tables (generated from Python unicodedata at convert time) + all 10 voices
  (`voice.<name>.{ttl,dp}`). arch = "supertonic-tts".
- CFG = two vector-field passes per step (cond/uncond share xt); 8 steps
  default ⇒ 16 VF passes.
- Diff harness: reference = onnxruntime with intermediates promoted to
  graph outputs; fixed np.random noise saved in ref.gguf and injected into
  the C++ side for parity (production uses its own RNG; CRISPASR_SEED-able).
- Acceptance: TTS→ASR roundtrip (HARD RULE #3), on Kaggle CUDA + CPU.

## Worktree
`.claude/worktrees/feat-434-supertonic`, branch `feat/434-supertonic`.
Kaggle account: chr1str (chr1s4 taken by a parallel agent).
