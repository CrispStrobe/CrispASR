# cohere.cpp — Bug Investigation Notes

## Status
**WORKING ✓ — SHIPPED**

- GitHub: https://github.com/CrispStrobe/cohere-whisper.cpp (main=clean, debug=with DBG prints)
- HuggingFace: https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF
- optimize.md: full roadmap for FFT / BLAS / ggml / GPU speedups

Root cause found and fixed: **mel per-feature normalization formula was wrong** (std was sqrt(T) times too large).
All prior issues traced back to this single bug in both C++ and Python.

---

## What We Know

### Model dims (from GGUF metadata)
- Encoder: 48 layers, d=1280, heads=8, head_dim=160, ffn=5120, conv_kernel=9
- Decoder: 8 layers, d=1024, heads=8, head_dim=128, ffn=4096, max_ctx=1024
- Vocab: 16384, n_mels=128, sr=16000, hop=160, win=400, n_fft=512

### GGUF tensor storage
- All tensors stored as F16 (GGML type 1)
- GGUF shape dims are reversed vs PyTorch: GGUF [d0_fast, d1_slow] = PyTorch [d1, d0]
- `ct_to_f32` uses `ggml_get_f32_1d` which correctly converts F16→F32
- Weight values verified to match safetensors (emb_w[0,:5] matches)

### Prompt tokens (verified correct)
```
[7, 4, 16, 62, 62, 5, 9, 11, 13]
= <|startofcontext|>, <|startoftranscript|>, <|emo:undefined|>,
  <|en|>, <|en|>, <|pnc|>, <|noitn|>, <|notimestamp|>, <|nodiarize|>
```
From `build_prompt()` in modeling_cohere_asr.py (line 985):
`<|startofcontext|><|startoftranscript|><|emo:undefined|><|{lang}|><|{lang}|>{pnc}{task}<|notimestamp|><|nodiarize|>`

### Python reference result
Running the exact C++ algorithm in Python (GGUF weights, batch prompt, NO cross-attention):
→ Top-5 tokens: [16, 4, 7, 290, 6167] with logits [81.7, 24.8, 14.1, 10.8, 9.4]
→ Token 13764 logit: **-3.1** (near bottom)

Our C++ outputs: **13764 (▁) at every step** — something fundamentally wrong.

### Decoder architecture (from modeling_cohere_asr.py + decoder.rs)
Per layer (pre-norm everywhere):
1. `norm1` → self-attention → residual
2. `norm2` → cross-attention → residual
3. `norm3` → FFN (ReLU) → residual

Final: `final_layer_norm` → head linear

Embedding: `layer_norm(token_emb[tok] + pos_enc[pos])` ← LN AFTER sum

Decoder activation: **ReLU** (confirmed from config `"hidden_act": "relu"`)
This was already fixed (was Swish before).

### Rust inference style (inference.rs)
- Feeds prompt tokens **one at a time** to the decoder
- Each step: single token → full decoder forward → update KV cache
- After all prompt tokens: argmax of last logits → first generated token
- Generated tokens also fed one at a time

Our C++: feeds ALL prompt tokens at once (batch). Should be equivalent but needs verification.

### Cross-attention pre-computation (from decoder.rs + kaggle ONNX script)
The Rust pre-computes cross-attn K/V once from encoder output:
```rust
pub fn precompute_cross_kv(&self, encoder_hs: &Tensor) -> Vec<(Tensor, Tensor)>
```
Each layer's cross K/V = `encoder_hs @ cross_k_w.T + cross_k_b`

Our C++ recomputes them on every decode step — correct but inefficient.

### Encoder→decoder projection
From encoder.rs and cohere.cpp: encoder output is projected from enc_d=1280 to dec_d=1024
via `enc.proj.weight [1280, 1024]` GGUF = PyTorch [1024, 1280].
This is done in `cohere_encode()` and the result is passed as `enc_out` to `cohere_decode_step`.

### ROOT CAUSE: Python was reading F32 tensors as F16
The GGUF file stores 1432 tensors as **F32** (dtype=0) and 672 as **F16** (dtype=1).
All Python comparison scripts used `np.frombuffer(t.data, dtype=np.float16, count=n)` unconditionally,
which **misread all F32 tensors** (biases, norms, BN params, window, pos_bias) as F16.

Effect: F32 bytes read 2-at-a-time as F16 → alternating zeros in the low bytes of each float.
This caused every LayerNorm, bias, BN, and Hann window to have corrupted values, making
ALL prior Python encoder comparisons meaningless.

**F32 tensors** (biases, norms, pos_bias, BN, fe.window): 1432 total
**F16 tensors** (weight matrices only): 672 total

The C++ `ct_to_f32` correctly checks `t->type` → reads F32 or F16 appropriately. C++ is fine.

**Fix for Python**: check `t.tensor_type == gguf.GGMLQuantizationType.F32` and use `dtype=np.float32`.
Script `/tmp/compare_enc3.py` implements this fix.

---

### Numerical ground truth (ONNX reference)
```
ONNX encoder output shape: cross_k (8, 1, 69, 1024) — T_sub=69 frames
ONNX ck[0,0,0,:5] = [-0.7066, 2.5989, -0.3135, -0.5355, -1.0828]
```
The ONNX model takes raw audio in, handles mel internally, outputs correct cross K/V.
Expected transcription: "If not, there will be a big crisis between you and the European Parliament."

### enc_in: Python = C++ ✓ (after F32 fix)
With correct F32 loading:
```
Python enc_in[0,:10] = [0.7069, 0.5949, -2.0885, -0.0812, 1.5088, 0.3594, -1.5633, 1.2182, -1.3948, 2.1012]
C++    enc_in[0,:10] = [0.7070, 0.5950, -2.0885, -0.0812, 1.5088, 0.3594, -1.5633, 1.2182, -1.3948, 2.1012]
```
Max diff < 0.001 — only F16 quantization noise. **Mel + conv subsampling are correct in C++.**

### enc_out: Python = C++ ✓ (after F32 fix)
```
Python enc_out[0,:5] = [1.5059, 1.5365, 0.3384, 0.1271, 0.3907]
C++    enc_out[0,:5] = [1.5058, 1.5365, 0.3384, 0.1269, 0.3909]
```
Max diff < 0.001. **All 48 conformer layers + projection are correct in C++.**
Layer values no longer show zeros: layer 48 h[0,:5]=[-0.065, 0.010, -0.389, -0.681, 0.046].

### ONNX ck0 still differs from Python/C++
```
Python ck0[0,:5] = [0.982, -0.799, 1.424, 2.142, 1.784]
ONNX   ck0[0,:5] = [-0.707, 2.599, -0.314, -0.535, -1.083]
```
This is expected: ONNX uses original F32 weights while GGUF has F16 weights. After 48 layers
the F16 quantization error accumulates. Whether this causes wrong transcription TBD.

### Encoder conv subsampling — Python vs C++ diverge (OLD — now resolved)
Running the complete Python encoder (GGUF weights, C++ mel computation):
```
Python enc_in[0,:10] = [0.4541, 0.9227, -1.5847, 0.7808, 1.0622, -1.0564, -1.2567, -0.9503, -0.1755, 2.3541]
C++    enc_in[0,:10] = [0.7070, 0.5950, -2.0885, -0.0812, 1.5088, 0.3594, -1.5633, 1.2182, -1.3948, 2.1012]
Max diff: 2.1685
```
The conv subsampling produces different values in Python vs C++ despite using same mel and weights.
**Neither has been confirmed correct against ONNX** — we need a way to extract intermediate values
from the ONNX model to find the ground truth enc_in.

### Python encoder also diverges from ONNX
Running 48 conformer layers in Python (GGUF weights, C++ mel):
```
Python ck0[0,:5] = [-6.249, -2.590, 1.867, 1.444, -1.039]
ONNX   ck0[0,:5] = [-0.706, 2.599, -0.314, -0.535, -1.083]
```
Very large divergence — indicates a bug in the Python encoder implementation too.
Suspicious symptom: h[0,:3] values at layer boundaries are [0.0, X, 0.0] (always zero at positions 0 and 2),
suggesting dead neurons or wrong weight tensor mapping.

### Mel computation — verified consistent
- C++ uses: pre-emphasis (0.97), center-pad (n_fft/2=256 each side), FFT, mel-fb, per-feature norm
- Python replication gives same T_mel=545, T_sub=69 (matching ONNX output shape)
- Per-feature norm formula: `std = sqrt(var * T/(T-1) + 1e-5)` — matches C++
- **BUT: unknown if ONNX uses same pre-processing** (ONNX is a black box)

### GGUF weight shapes verified
| Tensor | GGUF shape | PyTorch reshape | Notes |
|--------|-----------|-----------------|-------|
| fe.mel_fb | [257, 128, 1] | [1, 128, 257] | Access: `[m*257+f]` ✓ |
| enc.pre.conv.0.weight | [3, 3, 1, 256] | (256, 1, 3, 3) | ✓ |
| enc.pre.conv.2.weight | [3, 3, 1, 256] | (256, 1, 3, 3) | depthwise ✓ |
| enc.pre.conv.3.weight | [1, 1, 256, 256] | (256, 256, 1, 1) | pointwise ✓ |
| enc.pre.out.weight | [4096, 1280] | (1280, 4096) | ✓ |

### Conformer layer changes (rel_shift and scale)
Applied these fixes to `ct_rel_pos_mha`:
1. **rel_shift**: changed `rel = i - j + T - 1` → `rel = j - i + T - 1` (BD[tq,tk] uses pos enc at tk-tq+T-1)
2. **Scale**: removed Q pre-scaling, now applies `(AC+BD)*scale` after both matrices (matching Python)

These may or may not be correct — need numerical validation against ONNX.

---

## Known Fixes Already Applied

1. **Decoder FFN ReLU** (was ct_swish_inplace, now ReLU): ✅ fixed
2. **OpenMP parallelism** in ct_linear and attention head loops: ✅ added
3. **-O3 -march=native** compile flags: ✅ added
4. **Encoder conv orientation** (was H=n_mels/W=T_mel, now H=T_mel/W=n_mels): ✅ fixed
5. **Conv subsampling activations** (was SiLU everywhere, now ReLU after c0, c3, c6): ✅ fixed
6. **Conv subsampling flatten** (was flat=ch*H3/T_sub=W3, now flat=ch*W3=4096/T_sub=H3): ✅ fixed
7. **rel_shift sign**: changed `rel = i-j+T-1` → `rel = j-i+T-1`: ✅ fixed (matching Python formula)
8. **Attention scale**: removed Q pre-scaling, now `(AC+BD)*scale`: ✅ fixed (matching Python)

---

## What Must Still Be Checked

### HIGH PRIORITY — likely decoder bug causes

1. **Batch vs token-by-token prompt processing**
   - Rust/Python feed prompt one token at a time; our C++ feeds all at once
   - Are the KV cache writes + causal masking correct for batch mode?
   - Specifically: in batch mode, KV cache for ALL tokens is written BEFORE attention
     is computed. For tq=0, causal_end=1 so it only reads cache[0]. ✓ looks correct
     but needs end-to-end numerical verification.

2. **Cross-attention correctness**
   - Does the encoder output have correct shape/values when passed to the decoder?
   - Is `enc_out` (T_enc=69, d=1024) correctly used in cross-attention?
   - Check: is `d` in `cohere_decode_step` = dec_d_model=1024 everywhere? ✓
   - Check cross-attention uses the pre-norm `h_cross_norm`, not `h` directly ✓

3. **Self-attention scale**
   - scale = 1/sqrt(head_dim=128) = ~0.0884
   - Python model: `self.scale = self.head_dim**-0.5` ✓ matches

4. **DEBUG fprintfs to remove**
   - Added to `cohere_decode_step` around embedding and logit computation
   - Must remove before final version

5. **Verify the decoder produces correct results WITHOUT cross-attention**
   - Need a test that bypasses the 5-minute encoder
   - Create test with zeros for enc_out and compare to Python reference
   - If decoder-alone gives wrong results → bug in self-attn/FFN
   - If decoder+encoder gives wrong results → bug in cross-attn or encoder output

6. **Check encoder output values**
   - Could the encoder be producing NaN/Inf/garbage that messes up cross-attention?
   - Add a check: print first 5 values of enc_out after encoder finishes

7. **Check decoder KV cache dimensions**
   - C++: `kv_n = H * max_ctx * hd = 8 * 1024 * 128 = 1,048,576` per layer ✓
   - Cache write: `ck[(h_idx * max_ctx + pos) * hd + j]` ✓
   - Cache read: same indexing ✓

8. **Check whether prompt token IDs are all found (none -1)**
   - The C++ filters -1 tokens; if some are missing, prompt is shorter
   - Verify all 9 tokens exist in the vocabulary

### MEDIUM PRIORITY

9. **Performance: encoder is too slow (~5 min for 5.4s audio)**
   - 48 layers × d=1280 × ffn=5120: huge compute
   - OpenMP on 4 cores helps but still slow
   - Consider: pre-saving encoder output for testing, or using BLAS
   - Check if OpenMP is actually being used (libgomp.so linked ✓)

10. **Cross-attention layer normalization placement**
    - Python model layer 2: `norm2` → cross-attn Q (decoder side)
    - C++ applies `cross_ln` to h before computing CQ ✓
    - But CK, CV are from enc_out WITHOUT any norm — is this correct?
    - In Rust: `cross_k = encoder_hs @ cross_k_w.T + cross_k_b` (no norm on encoder side) ✓

11. **Decoder output position indexing**
    - When reading logits for first generated token after prompt:
      C++ step=0: `last_logits = logits.data() + (prompt.size()-1) * vocab`
      This reads position `n_tok-1` = last prompt token's logits ✓

### LOW PRIORITY

12. **Positional encoding range**
    - Decoder positions: prompt tokens at 0..8, generated at 9, 10, ...
    - max_ctx=1024, so positions are valid ✓

13. **EOS detection**
    - eos_id = token 3 (`<|endoftext|>`)
    - Also check nospeech (token 1)
    - Currently only checks eos; should also check nospeech

---

## Files to Reference

| File | Purpose |
|------|---------|
| `modeling_cohere_asr.py` | Python reference, decoder architecture |
| `decoder.rs` | Rust reference decoder (single-token stepping) |
| `inference.rs` | Rust greedy decode loop (prompt handling) |
| `encoder.rs` | Rust encoder reference |
| `weights.rs` | Rust weight loading (tensor name mapping) |
| `kaggle_full_script.py` | ONNX export + decoder architecture reference |
| `cohere.cpp` | Our C++ implementation |
| `cohere-arch.h` | GGUF tensor name constants |

---

---

## Investigation Plan

### ONNX model structure (confirmed from graph inspection)
**Pre-processing** (all confirmed present in ONNX graph):
- Pre-emphasis: `Slice → Mul(×0.97) → Sub` → pre-emphasized audio
- Center-pad: `Pad` node
- STFT: implemented as `Conv` (DFT via convolution with sin/cos kernels)
- Log-mel: `MatMul(mel_fb) → Add(log_grd) → Log`
- Per-feature norm: `ReduceMean → Sub → Pow → ReduceMean → Sqrt → Div`

**Conv subsampling** (`/encoder/pre_encode/conv/`):
```
conv.0 → Relu (conv.1) → conv.2(DW) → conv.3(PW) → Relu (conv.4) → conv.5(DW) → conv.6(PW) → Relu (conv.7)
```
This **matches C++ ReLU placements** (after conv0, conv3, conv6). ✓

**Key ONNX intermediate tensor names** (FP32, extractable):
```
/features/Log_output_0                              log-mel before normalization     shape (1, 128, T_mel)
/features/Div_1_output_0                            normalized mel                   shape (1, 128, T_mel)
/encoder/pre_encode/conv/conv.7/Relu_output_0       after all 3 stride-2 convs      shape (1, 256, T_sub, 16)
/encoder/pre_encode/out/MatMul_output_0_output_quantized  enc_in (quantized, int8 model)
```
Note: the ONNX encoder is int8 quantized, so enc_in comes from `MatMulInteger` after DynamicQuantizeLinear.
The mel tensors and conv output (before the linear) should still be FP32 and directly extractable.

---

### Step-by-step plan (in order)

#### STEP 1: Extract ONNX ground truth for mel and conv subsampling output
Write `/tmp/extract_onnx.py`:
```python
import onnx, onnxruntime as ort, numpy as np, soundfile as sf

ENC_ONNX = '/mnt/akademie_storage/test_cohere/cohere-int8-tristan/cohere-encoder.int8.onnx'
AUDIO_PATH = '/mnt/akademie_storage/akademie_backup/local_cohere_model/demo/voxpopuli_test_en_demo.wav'

m = onnx.load(ENC_ONNX)
# Add FP32 intermediate outputs
for name in [
    '/features/Log_output_0',       # log-mel
    '/features/Div_1_output_0',     # normalized mel
    '/encoder/pre_encode/conv/conv.7/Relu_output_0',  # after all convs
]:
    m.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))

audio, _ = sf.read(AUDIO_PATH)
sess = ort.InferenceSession(m.SerializeToString())
results = sess.run(None, {'audio': audio.reshape(1,-1).astype(np.float32)})

onnx_ck, onnx_cv = results[0], results[1]
onnx_log_mel = results[2]    # (1, 128, T_mel)
onnx_norm_mel = results[3]   # (1, 128, T_mel)
onnx_conv_out = results[4]   # (1, 256, T_sub, 16)

print(f'log_mel[0,0,:5]   = {onnx_log_mel[0,0,:5].tolist()}')
print(f'norm_mel[0,0,:5]  = {onnx_norm_mel[0,0,:5].tolist()}')
print(f'conv_out[0,0,0,:5]= {onnx_conv_out[0,0,0,:5].tolist()}')
```
**Goal**: Ground truth values for mel and conv subsampling output.

#### STEP 2: Compare Python mel vs ONNX mel
In Python (C++ mel computation):
```
mel shape: (128, T_mel)  — Python has mel[mel_bin, time_frame]
ONNX has:  (1, 128, T_mel)
```
Compare `Python log_mel[:,0]` vs `onnx_log_mel[0,:,0]` (first time frame, all mel bins).
Compare `Python norm_mel[:,0]` vs `onnx_norm_mel[0,:,0]`.
- If they match → mel computation is correct, bug is in conv subsampling or conformer
- If they don't match → diagnose: window shape? pre-emphasis coefficient? STFT padding?

#### STEP 3: Compare Python conv output vs ONNX conv output
After Python's c6 (before linear projection), shape is (1, 256, T_sub, 16).
Compare against `onnx_conv_out[0, 0, 0, :5]`.
Note: ONNX shape is `(1, 256, T_sub, 16)`, Python c6 has same shape.
- If conv output matches → bug is in linear projection or conformer
- If not → find which of the 3 conv pairs diverges first

#### STEP 4: Fix Python conv subsampling to match ONNX
Since the ONNX model is int8-quantized, there will be slight numerical differences.
A match within ~0.1 per element is acceptable; large differences indicate a bug.

#### STEP 5: Diagnose Python conformer divergence
Once conv subsampling is validated, find where the 48 conformer layers diverge.
The `h[0,:3]=[0.0, X, 0.0]` pattern suggests systematic zeroing.

Diagnosis approach:
1. Print ALL GGUF tensor names for layer 0 (`gguf.GGUFReader` → filter by `enc.blk.0`)
2. Compare against tensor names used in Python (`enc.blk.{li}.ff1.up.weight`, etc.)
3. Check if any tensor is missing (would cause KeyError, or silent 0 if handled as default)
4. Run layer 0 only, print intermediate outputs: ff1_out, attn_out, conv_out, ff2_out

#### STEP 6: Use Python as reference to fix C++
Once Python matches ONNX:
- Compare `Python enc_in[0,:10]` vs `C++ enc_in[0,:10]` (already printed by C++ debug)
- Add C++ debug prints after each conv layer: `c0[0:5]`, `c2[0:5]`, `flat_in[0:5]`
- Fix first divergence

#### STEP 7: Clean up
- Remove all `fprintf(stderr, "DBG...")` from cohere.cpp
- Remove the debug prints added around line 856

---

### Known values for comparison
```
ONNX final output:   ck[0,0,0,:5] = [-0.7066, 2.5989, -0.3135, -0.5355, -1.0828]
C++ enc_in[0,:10]  = [0.7070, 0.5950, -2.0885, -0.0812, 1.5088, 0.3594, -1.5633, 1.2182, -1.3948, 2.1012]
Py  enc_in[0,:10]  = [0.4541, 0.9227, -1.5847, 0.7808, 1.0622, -1.0564, -1.2567, -0.9503, -0.1755, 2.3541]
```

(Enc_in values above are OLD — from broken F32-as-F16 Python. After fix: Python = C++ to within F16 noise.)

---

## Session 2026-03-31: Decoder Bug Isolation

### Key finding: Python GGUF decoder is CORRECT

Script: `test_cohere/tests/debug_decoder.py`
Audio:  `test_cohere/tests/sample2_16k.wav` (T_enc=53)
F16/F32 GGUF reading: correct (checked tensor_type before reading)

ONNX ground truth (sample2_16k.wav):
```
ONNX encoder output: cross_k (8, 1, 53, 1024)
ONNX top-5 after prompt: [749, 527, 1198, 1805, 583]
ONNX generates:           749 2981 10194 1424 13809 606 679 1446 990 527 624 897 5288 13785 3
```

Python GGUF decoder + ONNX cross KV:
```
GGUF+ONNX-CK top-5: [749, 527, 1805, 1198, 583]  ← MATCHES ONNX top-1
h after emb+ln (tok7, pos0): [-1.2141, 1.0966, -0.2632, 0.2215, 0.5057]
h after layer0 self-attn:    [-0.2794, -0.4906,  0.5317, 0.2661, 1.0790]
h after layer0 cross-attn:   [ 0.1510, -1.0254, -0.3650, 0.0604, 0.4592]
```

**Conclusion: the decoder algorithm and tensor names are correct in Python.**
**The C++ implementation has a bug — same algorithm, different code.**

### Python decoder reference values (ground truth for C++ comparison)
Audio: sample2_16k.wav, prompt = [7,4,16,62,62,5,9,11,13], offset=0

| Checkpoint | h[0,:5] (tok=7, pos=0) |
|-----------|------------------------|
| After emb+LN | [-1.2141, 1.0966, -0.2632, 0.2215, 0.5057] |
| After layer0 self-attn+residual | [-0.2794, -0.4906, 0.5317, 0.2661, 1.0790] |
| After layer0 cross-attn+residual | [0.1510, -1.0254, -0.3650, 0.0604, 0.4592] |

Final logits at last prompt token (pos 8):
- Top token: 749 (logit ~26.21)
- Tok 13764: logit ~12.45 (NOT the top token)

### C++ debug output (sample2_16k.wav) — comparison with Python reference

```
DBG dec h[0]_emb+ln=-1.2141 1.0966 -0.2632 0.2215 0.5057   tok=7  ← MATCHES Python ✓
DBG dec h[0]_sa_li0=-0.2794 -0.4906 0.5317 0.2662 1.0790           ← MATCHES Python ✓
DBG dec h[0]_ca_li0=1.4219 -0.6793 1.6433 1.5130 0.1316            ← DIFFERENT from Python (ONNX-CK)
DBG dec logits: top5 = 13764(12.03) 563(10.28) 1691(9.91) 749(9.53) 884(9.22)
```

Python STEP 3 (GGUF enc_out → GGUF cross KV):
```
GGUF+GGUF-CK top-5: [13764, 563, 1691, 749, 884]  logit 13764=12.03, 749=9.53
```

**The C++ and Python GGUF+GGUF-CK outputs are IDENTICAL.** C++ decoder implementation is correct.

---

## ROOT CAUSE FOUND AND FIXED (2026-03-31)

### The bug: wrong mel per-feature normalization formula

Both C++ and Python were computing:
```
std = sqrt(S * T/(T-1) + 1e-5)   where S = Σ(xi - x̄)²
```

The ONNX model computes:
```
std = sqrt(mean(diff²) + eps) = sqrt(S/T + eps)    (biased variance)
```

The ratio: `sqrt(S * T/(T-1)) / sqrt(S/T)` = `sqrt(T²/(T-1))` ≈ `sqrt(T)`

For T=417 frames: factor ≈ 20.4 — the std was ~20 times too large,
making the normalized mel features ~20 times too small.
This propagated through all 48 Conformer layers → completely wrong enc_out → wrong cross KV.

### Confirmation
- F16 vs F32 encoder weights: `max_diff=0.0000` — quantization was NOT the issue
- After fix: `norm_mel diff (biased): max=0.52, mean=0.003` — matches ONNX (residual from log_mel float diff)
- Cross-K diff after fix: `max=0.68, mean=0.076` vs before `max>>1, mean≈0.8`
- **C++ cohere-main output: "The quick brown fox jumps over the lazy dog."** ✓

### The fix applied
**C++ cohere.cpp** (line ~524):
```cpp
// WRONG: float std = sqrtf((float)(var * T / (T - 1.0) + 1e-5));
float std = sqrtf((float)(var / T + 1e-5));   // biased variance, matches ONNX
```
(where `var = Σ(xi - x̄)²` without any prior division)

**Python (save_enc_out.py, test_enc_f32.py)**:
```python
# WRONG: var = diff.pow(2).sum() * n_frames / (n_frames - 1)
var = diff.pow(2).mean(dim=1, keepdim=True)   # biased, matches ONNX
```

### C++ debug values after fix (sample2_16k.wav)
```
enc_in[0,:5]  = [-0.5358, 0.3357, -0.5151, 1.0951, 1.9463]
enc_out[0,:5] = [0.2121, 0.8887, 0.0922, 1.1125, 0.6852]
h[0] emb+LN  = [-1.2141, 1.0966, -0.2632, 0.2215, 0.5057]  (tok=7)
h[0] sa_li0  = [-0.2794, -0.4906, 0.5317, 0.2662, 1.0790]
h[0] ca_li0  = [0.0089, -0.9987, -0.4031, 0.0697, 0.4858]
Top token: 749 (logit 26.15) ✓
```
Full transcript matches ONNX: "The quick brown fox jumps over the lazy dog."

### Next: compare C++ debug prints against these values
Added debug prints to cohere.cpp (cohere_decode_step):
- `DBG dec h[0]_emb+ln=` after embedding+LN step
- `DBG dec h[0]_sa_li0=` after layer 0 self-attn
- `DBG dec h[0]_ca_li0=` after layer 0 cross-attn
- `DBG dec logits last_pos top5:` final top-5 at last prompt position

Running C++ on sample2_16k.wav to compare. (enc_out.bin being saved by save_enc_out.py)

### GGUF tensor naming (from cohere-arch.h / export_gguf.py)
Decoder tensors use these names:
- `dec.blk.{i}.attn_ln.*`  (not `attn.norm`)
- `dec.blk.{i}.attn_q/k/v/o.*`
- `dec.blk.{i}.cross_ln.*`, `cross_q/k/v/o.*`
- `dec.blk.{i}.ffn_ln.*`, `ffn_up.*`, `ffn_down.*`
- ALL decoder biases/norms = F32; weight matrices = F16
