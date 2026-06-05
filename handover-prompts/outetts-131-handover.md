# OuteTTS (§131) — Handover Prompt

> **Status: DONE — speech output confirmed via ASR roundtrip ("Hello world.")**
> Full pipeline: text + speaker JSON → LLM → codes → WavTokenizer → PCM → WAV.
> WavTokenizer decoder validated cos ≥ 0.999 all stages vs Python library.
> Audio amplitude matches Python reference (RMS ~0.14). 8 bugs fixed.
> Model registry, GGUF detection, docs, CLI wiring all complete.

## What this is

Native C++ ggml runtime for [OuteAI/OuteTTS-0.3-1B](https://huggingface.co/OuteAI/OuteTTS-0.3-1B).
OLMo 1B LLM generates interleaved text + audio tokens; WavTokenizer single-codebook
VQ-GAN decodes audio tokens to 24 kHz PCM. CC BY 4.0 license.

## Architecture findings (verified against safetensors)

### OLMo 1B (the LLM)
- **NOT Llama-compatible** in one critical way: **parameter-free LayerNorm**
  (no weight or bias tensors for any norm). `OlmoLayerNorm` = `F.layer_norm(..., None, None, eps=1e-5)`.
- 16 layers, 2048 hidden, 16 MHA heads (not GQA), head_dim=128
- SwiGLU FFN (intermediate=8192), RoPE theta=10000
- Tied embeddings (`model.embed_tokens.weight` is the only top-level tensor)
- No `model.norm.weight`, no `lm_head.weight`, no `input_layernorm`, no `post_attention_layernorm`
- Total: 113 tensors (7 per layer × 16 layers + 1 embed)
- The C++ runtime correctly uses `ggml_norm(ctx, cur, eps)` without any scale/bias multiply
- **Tokenizer**: GPT-NeoX byte-level BPE. Newline is `Ċ` (U+010A = token 187), not `\n`.

### WavTokenizer decoder (VALIDATED — cos ≥ 0.999926)
- Codebook: (4096, 512) single VQ
- conv_pre: Conv1d(512, 768, k=7, pad=3) — weight shape (768, 512, 7)
- pos_net: 4 ResNet blocks (indices 0,1,3,4) + 1 self-attention (index 2) + GroupNorm (index 5)
  - ResNet: LN → Conv1d(768,768,k=3,p=1) → GELU → LN → Conv1d → GELU → residual
  - SelfAttn: LN → Q/K/V 1x1 conv → scaled dot-product → proj_out 1x1 → residual
  - GroupNorm: groups=1 (= LayerNorm)
- 12× ConvNeXtBlock: DW Conv1d(768,k=7,p=3) → AdaNorm(scale,shift using bw_id=0) → PW up(768→2304) → GELU → PW down(2304→768) → gamma gate → residual
  - `gamma` is (768,) applied AFTER pw_down (residual gate, not GRN)
  - AdaNorm: `norm.scale.weight` is (4, 768), `norm.shift.weight` is (4, 768) — 4 bandwidth embeddings, use row 0
- Final LayerNorm(768)
- ISTFTHead: Linear(768, 1282) → split [641 mag, 641 phase] → exp(mag) → iSTFT(n_fft=1280, hop=320, Hann window stored in GGUF) → 24kHz PCM
- Total: 162 tensors, 130 MB F16

### Validation results (codes [0..9])

| Stage | cos | max_abs_err |
|-------|-----|-------------|
| 01_codebook | 1.000000 | 0.000244 |
| 02_conv_pre | 1.000000 | 0.002739 |
| 03_backbone_adanorm | 1.000000 | 0.000463 |
| 04_posnet_resnet0 | 1.000000 | 0.007258 |
| 04_posnet_resnet1 | 1.000000 | 0.010736 |
| 04_posnet_selfattn2 | 1.000000 | 0.010067 |
| 04_posnet_resnet3 | 1.000000 | 0.012850 |
| 04_posnet_resnet4 | 1.000000 | 0.021334 |
| 05_posnet_gnorm | 1.000000 | 0.000247 |
| 06_convnext_0 | 1.000000 | 0.001902 |
| 06_convnext_11 | 0.999997 | 0.142540 |
| 07_final_norm | 0.999995 | 0.021624 |
| 08_istft_head | 0.999999 | 0.083273 |
| 09_mag | 0.999992 | 0.016240 |
| 09_phase | 0.999996 | 0.083273 |
| 10_audio | 0.999926 | 0.000223 |

### Prompt format (V2)
```
<|im_start|>\n<|text_start|>word1<|space|>word2<|text_end|>\n<|audio_start|>\n
```
Audio tokens `<|0|>` through `<|4099|>` → codebook indices 0-4095.
`audio_token_offset = 50307` (the token ID of `<|0|>`).
Generate until `<|audio_end|>`. Non-audio tokens are interspersed (word text, duration markers); filter for audio codes only.

**Critical**: text MUST be lowercased. The model was trained on lowercase input.

## Files

```
models/convert-outetts-to-gguf.py       — OLMo safetensors → GGUF (WORKING, 113/113 tensors)
models/convert-wavtokenizer-to-gguf.py  — WavTokenizer decoder → GGUF (WORKING, 162/162 tensors)
src/outetts.h                           — public C ABI header
src/outetts.cpp                         — LLM runtime (WORKING — generates audio codes)
src/outetts_wavtok.h                    — WavTokenizer decoder C ABI
src/outetts_wavtok.cpp                  — WavTokenizer decoder (VALIDATED — cos ≥ 0.999926)
examples/cli/crispasr_backend_outetts.cpp — CLI adapter (COMPLETE)
examples/cli/crispasr_backend.cpp       — dispatch entry added
src/CMakeLists.txt                      — outetts library added
examples/cli/CMakeLists.txt             — backend source added
tools/reference_backends/outetts_tts.py — Python reference TTS
tools/reference_backends/outetts_wavtok_diff.py — Diff harness (WORKING)
```

## GGUFs on disk

```
/mnt/storage/outetts/outetts-0.3-1b-f16.gguf       — 2.38 GB, 113 tensors
/mnt/storage/outetts/wavtokenizer-decoder-f16.gguf  — 130 MB, 162 tensors
```

## What works

```bash
cd /mnt/storage/whisper.cpp
cmake --build build -j$(nproc) --target crispasr-cli  # builds clean
build/bin/crispasr --backend outetts \
    -m /mnt/storage/outetts/outetts-0.3-1b-f16.gguf \
    --codec-model /mnt/storage/outetts/wavtokenizer-decoder-f16.gguf \
    --tts "Hello world" --tts-output /mnt/storage/outetts/test.wav --seed 42
# Output:
#   outetts: prompt 10 tokens (max_audio=4096)
#   outetts: AR emitted 111 audio codes (~1.5 s at 75 tok/s)
#   wavtok: decoded 111 codes -> 35200 samples (1.47 s at 24000 Hz)
```

## Diff testing

```bash
# Dump C++ stages with fixed codes (matching Python reference):
WAVTOK_DUMP_DIR=/mnt/storage/outetts/cpp_stages \
WAVTOK_FIXED_CODES=0,1,2,3,4,5,6,7,8,9 \
build/bin/crispasr --backend outetts ... --tts "Hi" --tts-output /dev/null

# Compare against Python reference:
python3 tools/reference_backends/outetts_wavtok_diff.py compare \
    --ref /mnt/storage/outetts/ref_stages/ \
    --cpp /mnt/storage/outetts/cpp_stages/

# Re-dump Python reference if needed:
python3 tools/reference_backends/outetts_wavtok_diff.py dump \
    --decoder-dir /mnt/akademie_storage/huggingface/hub/models--OuteAI--wavtokenizer-large-75token-interface/snapshots/83598100ca3fdbef3e055ef44d6f50f550f34f33/decoder \
    --codes "0,1,2,3,4,5,6,7,8,9" \
    --out /mnt/storage/outetts/ref_stages/
```

## Bugs found and fixed (8 total)

1. **pos_net normalization** (CRITICAL): Uses **GroupNorm(32) + SiLU**, not LayerNorm + GELU.
   The ResnetBlock in VocosBackbone uses `Normalize(in_channels)` = `GroupNorm(num_groups=32)`
   and `nonlinearity(x)` = `x * sigmoid(x)` (SiLU/Swish). Wrong norms caused ~2x amplitude
   reduction in the final audio.

2. **AdaNorm/pos_net order** (CRITICAL): VocosBackbone.forward does `embed → pos_net → AdaNorm
   → ConvNeXt`, but C++ had `embed → AdaNorm → pos_net → ConvNeXt`. Combined with #1,
   this caused the backbone output to have wrong value ranges.

3. **Magnitude clipping** (CRITICAL): ISTFTHead clips `exp(mag)` to `max=1e2` to prevent
   overflow from large stft values (up to exp(23) ≈ 1e10). Without this safeguard,
   the iSTFT overlap-add normalization was corrupted.

4. **iSTFT radix-2 zero-padding**: n_fft=1280 is not a power of 2. Replaced zero-padded
   radix-2 FFT with direct inverse RFFT for exact N-point transform.

5. **iSTFT padding="same"**: WavTokenizer uses `padding="same"` which trims
   `(win_length - hop_length)/2 = 480` from each end (not center=True's `n_fft/2 = 640`).

6. **Newline token**: GPT-NeoX byte-level BPE encodes '\n' as 'Ċ' (U+010A, token 187).

7. **Text lowercasing**: Model trained on lowercase input.

8. **Norm epsilon**: All norms use eps=1e-6 (matching the Python library), not 1e-5.

Additionally: repetition penalty 1.1 added, clamp relaxed to ±1.0.

## Speaker prompt support (DONE)

Speaker profiles loaded via `--voice speaker.json`. Format:
```json
{"text": "...", "words": [{"word": "hello", "duration": 0.53, "codes": [123, 456, ...]}, ...]}
```

Created by `tools/reference_backends/outetts_create_speaker.py` (WavTokenizer encoder +
whisper word timestamps). The model interleaves word text + duration markers with audio codes:
```
hello <|t_1.03|> [audio codes...] <|space|> \n world <|t_0.48|> [audio codes...] \n <|audio_end|>
```

### Remaining polish

- CTC forced alignment for better word-level code boundaries (currently uses whisper timestamps → approximate)
- `top_p` / `min_p` sampling (CrispTTS handler uses top_p=0.9, min_p=0.05)
- Upload GGUFs to HuggingFace (`cstr/outetts-0.3-1b-GGUF`)
- Consider OuteTTS 1.0 support (may have better zero-shot capability)

## Quick resume commands

```bash
# Build
cd /mnt/storage/whisper.cpp
cmake --build build -j$(nproc) --target crispasr-cli

# Test
build/bin/crispasr --backend outetts \
    -m /mnt/storage/outetts/outetts-0.3-1b-f16.gguf \
    --codec-model /mnt/storage/outetts/wavtokenizer-decoder-f16.gguf \
    --tts "Hello world" --tts-output /mnt/storage/outetts/test.wav --seed 42

# Rebuild just wavtok (fast iteration)
cmake --build build -j$(nproc) --target outetts
```
