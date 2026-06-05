# Dia + SpeechT5 TTS Port Handover

## What's done

### SpeechT5
- Encoder validated cos > 0.999 (all 12 layers)
- Full pipeline: encoder → decoder (KV cache) → postnet → HiFi-GAN → WAV
- GGUF: `/mnt/storage/speecht5/speecht5-tts-f16.gguf` (300 MB)
- Speaker embedding: `/mnt/storage/speecht5/speaker_torch42.bin`
- Decoder produces audio but wrong content — needs per-layer decoder validation

### Dia 1.6B
- Encoder cos = 1.000000 (all 12 layers, bit-perfect)
- Decoder layer 0 cos = 0.999 (validated standalone)
- Decoder step 0 argmax matches Python (ch0 = 568)
- Full pipeline: encoder → cross-attn → AR decoder (18L, GQA, CFG) → DAC → WAV
- GGUF: `/mnt/storage/dia/dia-1.6b-f16.gguf` (3.2 GB)
- DAC: `/mnt/storage/dia/dac-44khz.gguf` (104 MB)
- Audio produced but ASR says "music" — 18-layer decoder precision issue

## Key remaining bugs

### Dia decoder: audio quality
The decoder produces codes but they decode to noise/music. Root causes:
1. **F16 precision + scale=1.0**: Each decoder layer loses ~0.001 cos. Over 18 layers this compounds. The encoder survived 12 layers because it uses the same attention weights for all positions (static), while the decoder is autoregressive (error compounds through KV cache).
2. **Potential fix**: Try F32 GGUF (converter already supports it — just didn't finish converting due to OOM kill). Command: `python models/convert-dia-to-gguf.py --input /mnt/storage/huggingface/local/nari-labs--Dia-1.6B --output /mnt/storage/dia/dia-1.6b-f32.gguf`
3. **Or**: Debug decoder layers 1-17 individually (the approach that found RoPE, GQA, scale bugs)

### Python reference intermediates saved
- `/mnt/storage/dia/ref_intermediates/enc_layer*.npy` — per-layer encoder output
- `/mnt/storage/dia/ref_intermediates/dec_f16_embed.npy` — decoder embedding
- `/mnt/storage/dia/ref_intermediates/dec_f16_layer{0,1,2}.npy` — decoder per-layer output
- `/mnt/storage/dia/ref_intermediates/dec_hidden_step0.npy` — decoder hidden before logits
- `/mnt/storage/dia/ref_intermediates/cross_k_layer0.npy` — cross-attn K
- `/mnt/storage/dia/ref_intermediates/cross_v_layer0.npy` — cross-attn V

### 11 bugs found (all fixed in committed code)
See LEARNINGS.md Round 10 for the full list.

## Test commands
```bash
# SpeechT5
./build/bin/crispasr --backend speecht5 -m /mnt/storage/speecht5/speecht5-tts-f16.gguf \
  --voice /mnt/storage/speecht5/speaker_torch42.bin --tts "Hello" \
  --tts-output /mnt/storage/speecht5/test.wav

# Dia (with DAC codec)
./build/bin/crispasr --backend dia -m /mnt/storage/dia/dia-1.6b-f16.gguf \
  --codec-model /mnt/storage/dia/dac-44khz.gguf \
  --tts "[S1] Hello, how are you today?" --tts-output /mnt/storage/dia/test.wav

# ASR roundtrip
./build/bin/crispasr -m models/ggml-base.en.bin -f /mnt/storage/dia/test.wav
```

## Next steps (on MacBook with more RAM)
1. Convert Dia to F32 GGUF and test — if speech works, F16 precision is the issue
2. If F32 works: implement crispasr-quantize support for Dia, test Q8_0
3. Validate decoder layers 1-17 against Python (need 16GB+ for full Python decoder)
4. SpeechT5 decoder validation (same approach)
5. Upload working GGUFs to HuggingFace with READMEs

## Key architectural facts
- Dia uses `scale=1.0` attention (NO 1/sqrt(d)) — extremely precision-sensitive
- Dia uses RoPE mode=2 (NeoX half-split), NOT mode=0
- Dia CFG: use CFG logits for top-k selection only, sample from CONDITIONAL logits
- Dia GQA: use repeat_interleave pattern (insert unit dim, repeat, flatten)
- HiFi-GAN conv_1d expects (T, C_in) — NOT channel-first
- ggml conv_transpose_1d requires padding=0 — trim output manually
