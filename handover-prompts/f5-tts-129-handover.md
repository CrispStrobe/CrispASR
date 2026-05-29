# F5-TTS (#129) — Handover Prompt

## What this is

Native C++ ggml runtime for [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS).
MIT-licensed. DiT (Diffusion Transformer) with flow matching for high-quality
zero-shot voice cloning. ~330M params, ~660 MB F16. 24 kHz mono output via
Vocos vocoder.

## Non-negotiable methodology

Read the piper #128 post-mortem (`handover-prompts/piper-tts-128-handover.md`)
in full before starting. The methodology section there is the result of
finding 14 bugs the hard way. The key lesson: **build the diff harness first,
before writing any runtime code**. Every bug in piper was found in seconds
once the harness existed; without it, the same bugs consumed hours of
speculation and wrong hypotheses.

### Required workflow

1. **Download the reference model** — get `F5TTS_v1_Base.safetensors` (or
   ONNX export) and `vocos.pt` (or ONNX export). Store at
   `/mnt/storage/f5-tts/`.

2. **Write the Python reference runner FIRST** — a script that takes text +
   reference WAV and produces audio via the PyTorch/ONNX model. This is the
   ground truth. If using PyTorch directly, use `torch.no_grad()` and
   `model.eval()`. If using ONNX, use `onnxruntime.InferenceSession`.

3. **Write the diff harness** — dump every intermediate tensor from the
   reference at these boundaries (add more as needed):

   ```
   text_tokens              — character-level token IDs
   text_enc_output          — ConvNeXt text encoder output
   ref_mel                  — mel spectrogram of reference audio
   conditioning_input       — concatenated ref_mel + text_enc (masked infilling input)
   dit_layer{0..21}_output  — after each DiT layer
   dit_output               — final DiT output (denoised mel prediction)
   ode_step{0..N}           — ODE solver state after each step (or at least steps 0, N/4, N/2, N)
   vocos_input              — mel going into vocoder
   vocos_output             — reconstructed waveform
   audio                    — final PCM
   ```

   Save as `.npz`. Write the comparison script that handles the data layout
   transpose (see piper #128 lesson — C++ stores (T,C), ONNX stores (C,T)).

4. **Write the GGUF converter** — read the safetensors/ONNX and export
   all tensors. **Check every weight name mapping** — the piper port had a
   critical bug where anonymous ONNX weights were mapped to the wrong blocks
   because the ONNX export ordered them in reverse.

5. **Implement the runtime stage by stage**, running the diff harness after
   each stage to verify before moving on:
   - Stage A: text encoder (ConvNeXt blocks) — verify cos ≥ 0.999
   - Stage B: reference mel extraction — verify exact match
   - Stage C: conditioning concatenation — verify exact match
   - Stage D: one DiT layer — verify cos ≥ 0.999 BEFORE adding more layers
   - Stage E: full DiT (22 layers) — verify cos ≥ 0.999
   - Stage F: ODE solver (Euler) — verify cos ≥ 0.999 at each step
   - Stage G: Vocos vocoder — verify cos ≥ 0.99
   - Stage H: ASR roundtrip

   **DO NOT proceed to stage N+1 until stage N passes the diff.**

6. **Wire into CLI/C API/registry** — follow the piper pattern exactly.

## Architecture details (from paper + source)

### Pipeline

```
Text → char tokenize → ConvNeXt text encoder → text_emb (D, T_text)
                                                    ↓
Ref WAV → mel → concat with text_emb as conditioning → (D, T_total)
                                                    ↓
                    Noise z ~ N(0,1) same shape → (D, T_total)
                                                    ↓
                    ODE solver: N steps of DiT forward
                    x_t = (1-t)*z + t*x_1 (where x_1 = target mel)
                    v = DiT(x_t, t, conditioning)
                    x_{t+dt} = x_t + v * dt
                                                    ↓
                    Predicted mel = ODE output → (D, T_total)
                                                    ↓
                    Extract generated portion (mask out ref mel)
                                                    ↓
                    Vocos vocoder: mel → magnitude + phase → iSTFT → 24 kHz PCM
```

### DiT layer structure

Each of 22 layers:
1. **AdaLN-Zero** conditioning from timestep embedding: `(1+scale) * norm(x) + shift`
2. **Self-attention** (standard multi-head)
3. **AdaLN-Zero** conditioning again
4. **FFN** (GELU activation, standard or SwiGLU — check the source)
5. **Gated residual** with learnable gate initialized to zero

Key: AdaLN-Zero has a **zero-initialized gate** that modulates the residual
connection. This is different from standard residual — the gate starts at 0
and learns to let signal through. Get this wrong and the DiT won't learn
anything (all layers would be identity).

### ConvNeXt text encoder

- Character-level tokenization (no BPE, no phonemizer)
- Embedding + positional encoding
- N ConvNeXt V2 blocks: depthwise conv → LayerNorm → pointwise up → GELU →
  pointwise down → residual
- Each block has GRN (Global Response Normalization) in ConvNeXt V2

### Vocos vocoder

- Input: mel spectrogram (80 or 100 bins — check the model config)
- ConvNeXt stack → STFT magnitudes + phases
- iSTFT → 24 kHz waveform
- **No HiFi-GAN upsampling** — Vocos reconstructs the full-resolution
  waveform directly from the STFT representation

### ODE solver

F5-TTS uses **conditional OT flow matching**:
- Forward process: `x_t = (1-t)*x_0 + t*x_1` (linear interpolation)
- The model predicts the velocity `v = dx/dt ≈ x_1 - x_0`
- Euler: `x_{t+dt} = x_t + v * dt`
- Default 32 steps. Midpoint variant exists for fewer steps.

This is structurally identical to `chatterbox_s3gen.cpp`'s `cfm_euler_solve`.
The same Euler loop works — just adapt the conditioning injection.

## Specific things to check against the ONNX/PyTorch source

**Before writing any code**, verify each of these by reading the source:

1. **Attention mask**: F5-TTS uses a causal mask? bidirectional? padding mask?
   Check `attention_mask` in the DiT forward — it controls whether ref and
   generated positions can attend to each other.

2. **Timestep embedding**: How is the diffusion timestep `t ∈ [0,1]` embedded?
   Sinusoidal? Learned MLP? Check `time_embed` or `t_embedder` in the source.

3. **Conditioning scheme**: How exactly are text embeddings and ref mel
   combined? Concatenated along time axis? Added? Cross-attention? Check
   the `forward()` method of the main model class.

4. **Normalization**: LayerNorm, RMSNorm, or GroupNorm? Pre-norm or post-norm?
   Check EVERY norm placement by reading the ONNX graph attributes.

5. **FFN variant**: GELU or SwiGLU? Hidden dim ratio? Check `ff_mult` and
   the FFN class.

6. **Rotary position encoding**: Does the DiT use RoPE? If so, what's the
   base frequency? Check for `rotary_emb` or `rope` in the model.

7. **Vocos mel config**: Number of mel bins, FFT size, hop length, window
   function. Must match the training config exactly.

## Reuse from existing code

| Component | Reuse source | Confidence |
|---|---|---|
| Self-attention | `core/attention.h` | HIGH — standard multi-head |
| FFN | `core/ffn.h` | HIGH — GELU or SwiGLU both exist |
| ODE solver | `chatterbox_s3gen.cpp` `cfm_euler_solve` | HIGH — same Euler loop |
| iSTFT | `core/fft.h` | HIGH — proven in chatterbox |
| Mel spectrogram | `core/mel.h` | HIGH — proven in multiple backends |
| GGUF loader | `core/gguf_loader.h` | HIGH |
| AdaLN-Zero | **NEW** — `core/adaln.h` | Write from scratch, ~100 LOC |
| ConvNeXt block | **NEW** — write inline or `core/convnext.h` | Write from scratch, ~150 LOC |
| GRN | **NEW** — trivial (~20 LOC) | |

## New core primitives to create

### `core/adaln.h` — Adaptive Layer Norm Zero

```
// AdaLN-Zero: scale, shift, gate = linear(condition)
// out = gate * sublayer((1 + scale) * layernorm(x) + shift)
//
// The "zero" refers to initializing the gate projection with zeros,
// so all layers start as identity and learn to activate.
```

This MUST go in core/ because Zonos (#130) will also need it.

### ConvNeXt V2 block

```
// depthwise_conv(k=7) → LayerNorm → pointwise_up(4x) → GELU → GRN → pointwise_down → residual
```

Small enough to inline in `f5_tts.cpp` unless another model needs it.

## Effort estimate

~2-3 weeks. The DiT is 22 layers of standard attention+FFN with AdaLN
wrapping. The ODE solver reuses chatterbox CFM code. Main work:
1. Converter (3-4 days — safetensors parsing, weight name mapping, validation)
2. ConvNeXt text encoder (1-2 days)
3. DiT forward + AdaLN (3-4 days — 22 layers, careful per-layer validation)
4. ODE solver adaptation (1 day — Euler loop is trivial, conditioning is the work)
5. Vocos vocoder (2-3 days — ConvNeXt stack + iSTFT)
6. CLI wiring + end-to-end validation (1-2 days)

## Test model

Download and store at `/mnt/storage/f5-tts/`:
- Main model: `SWivid/F5-TTS` from HuggingFace
  (safetensors or export to ONNX with `torch.onnx.export`)
- Vocos: `charactr/vocos-mel-24khz` from HuggingFace

Test utterance: "Hello world" with a 5-second reference clip from
`/mnt/storage/piper/ref_hello_world.wav` or any clean speech sample.

## Files to create

```
NEW  models/convert-f5-tts-to-gguf.py
NEW  src/f5_tts.h
NEW  src/f5_tts.cpp
NEW  tests/test-f5-tts.cpp
NEW  tools/reference_backends/f5_tts.py
NEW  tools/reference_backends/f5_tts_diff.py
NEW  examples/cli/crispasr_backend_f5_tts.cpp
MOD  src/CMakeLists.txt
MOD  tests/CMakeLists.txt
MOD  examples/cli/CMakeLists.txt
MOD  examples/cli/crispasr_backend.cpp
MOD  src/crispasr_c_api.cpp
MOD  src/crispasr_model_registry.cpp
MOD  README.md
MOD  docs/tts.md
```

## Critical lessons from piper #128 (DO NOT SKIP)

1. **Never assume dilation patterns** — always read ONNX conv attributes.
   Piper had THREE different dilation bugs (DDSConv `K^i` not `2^i`, flow
   WaveNet all-1s not `2^i`, decoder resblocks per-kernel patterns).

2. **Never assume Flip = swap halves** — `torch.flip(x,[1])` reverses ALL
   channels. This single bug caused the flow WaveNet output to be 10x wrong.

3. **Check weight mapping in converter** — anonymous `onnx::Conv_NNNN`
   tensors appear in whatever order the ONNX exporter chose (often reversed).
   Verify EVERY mapping by tracing which ONNX Conv node references which
   initializer name.

4. **Check norm placement** — pre-norm vs post-norm changes the output
   magnitude by 15x. Don't trust the paper; read the actual code/ONNX.

5. **Check operation order** — `dp.proj` before or after DDSConv? The
   ONNX graph is the source of truth, not the paper or a code comment.

6. **Check scaling factors** — `1/sqrt(filter_channels)` on spline params
   was undocumented and caused NaN. Look for `Div` nodes in the ONNX.

7. **Use exact activation functions** — tanh GELU approximation diverges
   from erf GELU enough to accumulate 0.014 max error across 6 applications.

8. **Build the diff harness before the runtime** — the harness is 200 LOC
   and saves 20+ hours of debugging. It's not optional.
