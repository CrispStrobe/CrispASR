# Raon-OpenTTS port (F5-TTS variant) — DRY audit + blueprint findings

Branch `feat/raon-opentts`. Model: KRAFTON/Raon-OpenTTS-{0.3B,1B}, CC-BY-NC-4.0,
English, F5-TTS DiT + flow matching + HiFi-GAN vocoder, 16 kHz / 80-mel.
Reference code cloned to /mnt/volume1/tmp-overflow/raon-ref/Raon-OpenTTS
(fork of the F5-TTS repo, Apache-2.0 for the vocoder).

## NOW — converter DONE + validated locally; Kaggle ref-dump kernel written.

0.3B converted locally (mmap, <2 GB RSS): 364 DiT + 156 HiFi-GAN + shipped
slaney fb(513,80)+window(1024), 5559 vocab → raon-opentts-0.3b-f16.gguf 880 MB
(the size is the f5 converter's f32-AdaLN conditioning protection, inherited).

**Runtime approach settled (a DRY correction):** f5's vocoder is a PURE-CPU
implementation (`vocos_decode` = cpu_conv1d/cpu_layer_norm, NOT a ggml graph),
so `core/hifigan.h` (ggml primitive) is NOT the drop-in first scoped. The
DRY-consistent path is a compact **CPU HiFi-GAN** in the f5 idiom (reuse
`cpu_conv1d`; add cpu conv_transpose_1d + MRF resblocks + leaky_relu + tanh),
weights extracted to a CPU cache like `voc_cache` (weight-norm already fused
in the converter). Delta #1 is therefore a ~120-line CPU vocoder, not a
core_hifigan wiring. Delta #2 (mel) unchanged: shipped-fb + center=False +
16 kHz, gated on `f5.vocoder`/`f5.mel_spec_type`, in compute_mel_spectrogram.

Ref-dump: `tools/kaggle/raon-ref-dump/` runs the reference Raon end-to-end on
Kaggle, dumps ref_mel/gen_mel/vocoder_audio (crispasr-diff fixtures) + the
raon-ref.wav (TTS→ASR roundtrip target) + reruns our converter on-box. Torch
can't load the .pt on the VPS, so this is the ONLY source of validation
fixtures. Launch it (RAON_SIZE=0.3B) before the runtime pass.

### Runtime integration checklist (next focused pass)
- hparams: add vocoder/mel_spec_type/mel_center strings+bool; HiFi-GAN
  voc hp (upsample_rates/kernels, resblock kernels/dilations, init_ch);
  shipped `f5.mel_fb`/`f5.mel_window` → CPU vectors.
- compute_mel_spectrogram: param `sr`; branch shipped-fb + center=false.
- new cpu_hifigan_decode (mirror vocos_decode structure); dispatch at
  f5_tts.cpp:2505 on hp.vocoder=="hifigan".
- validate: crispasr-diff mel/vocoder vs ref.gguf; then TTS→ASR roundtrip
  of raon-ref target text (HARD RULE #3).
- registry (raon/raon-1b) NC-gated; backend alias → f5-tts; live test.

Checkpoint facts (peeked via mmap): model_225000.pt is a training checkpoint
{model_state_dict, optimizer_state_dict (the 4.9 GB bulk), ema_model_state_dict,
scheduler, update}. Use `ema_model_state_dict` (keys `ema_model.transformer.*`,
364 tensors) — exactly what map_f5tts_name expects; skip the rest; mmap keeps
RSS <2 GB. Vocoder NOT in the checkpoint: speechbrain/tts-hifigan-libritts-16kHz
generator.ckpt (234 weight-normed tensors, `{conv_pre,conv_post,ups.N,
resblocks.N.convs1/2.N}.conv.{weight_g,weight_v,bias}`; conv_pre in=80 ✓,
conv_post 32->1 ✓). vocab.txt = 5559 chars → text_num_embeds 5560.
Converter reuses map_f5tts_name (verbatim) + fuse_weight_norm (speecht5) +
choose_dtype; ships slaney mel fb + Hann window as f5.mel_fb/f5.mel_window.

## Blueprint reading (HARD RULE #1) — the scope collapsed

Read `src/f5_tts/model/{backbones/dit.py,modules.py,vocoder.py}` + both
config.yaml. The intimidating delta evaporated on contact with the code:

- **`norm_type: rmsnorm` is DEAD METADATA.** RMSNorm is only instantiated
  inside `DiTBlock.__init__` when `post_norm=True` (Gemma2-style post-norms).
  Both Raon configs set **`post_norm: False`** → `attn_post_norm`/
  `ff_post_norm` are `None` → the block forward is byte-for-byte our existing
  f5_tts.cpp DiT (AdaLN LayerNorm-eaf=False pre-norm + modulation + gated
  residual). Had I trusted the config key and swapped the pre-norms to
  RMSNorm, it would have been a multi-hour wrong port producing plausible
  garbage. `qk_norm: null` too → no q/k RMSNorm. `attn_mask_enabled: False`
  = our nullptr-mask flash attn. `ff_mult` (4 for 1B, 2 for 0.3B) is a dim,
  converter-read.

## Reuse map (DRY)

| piece | reuse | note |
|---|---|---|
| DiT backbone, AdaLN, RoPE, gated residuals | `src/f5_tts.cpp` **verbatim** | post_norm=False ⇒ identical |
| text-encoder ConvNeXt, time-embed, input-embed conv-pos | `f5_tts.cpp` verbatim | same layout |
| flow-matching ODE (Euler/CFG/sway) | `f5_tts.cpp` verbatim | arch-independent |
| DiT weight name map | converter `map_f5tts_name()` verbatim | Raon IS F5 DiT |
| HiFi-GAN vocoder | `core/hifigan.h::forward()` **drop-in** | config below is standard v1 |
| RMSNorm (if ever needed) | `ggml_rms_norm` | not needed for these configs |
| mel STFT/FFT scaffolding | `f5_tts.cpp` mel + `core/mel.h` | params change |

## The TWO real deltas (both outside the hot DiT loop)

1. **Vocoder: Vocos → HiFi-GAN.** `vocoder.py` HifiganGenerator config maps
   directly onto `core_hifigan::hparams`:
   in=80, out=1, resblock type 1, `upsample_initial_channel=512`,
   `upsample_factors=[8,8,2,2]`, `upsample_kernel_sizes=[16,16,4,4]`,
   `resblock_kernel_sizes=[3,7,11]`, `resblock_dilation_sizes=[[1,3,5]×3]`,
   conv_post_bias=True. 8·8·2·2 = 256 = hop ✓. Weights: the standalone
   generator ckpt (speechbrain/tts-hifigan-libritts-16kHz lineage). Converter
   adds a `map_hifigan_name()` (state_dict → `voc.*` core_hifigan names) —
   crib the tensor conventions from speecht5_tts / fastpitch_tts GGUFs.

2. **Mel front-end: 24 kHz/100-bin/HTK/center=True → 16 kHz/80-bin/SLANEY/
   center=False.** `get_sb_hifigan_mel_spectrogram`: n_fft=1024 win=1024
   hop=256 power=1 **center=False** (frames=(len−n_fft)//hop+1; tail cropped
   by `crop_waveform_to_hop_aligned_length`, no reflect pad actually applied),
   MelScale **norm="slaney" mel_scale="slaney"** f_min=0 f_max=8000, then
   `log(clamp(x,1e-5))`. The slaney filterbank is the scale-blind trap
   (dev-doc CQT/htdemucs class) — so per the "copy the shipped filterbank"
   rule, **the converter computes the fb+window with torchaudio and ships
   them as GGUF buffers** (`preprocessor.fb`, `preprocessor.window`); the
   runtime does STFT + matmul with the shipped fb (center=False), never
   rebuilds slaney in C++. f5's mel hardcodes `sr=24000.0f` (f5_tts.cpp:858)
   → make it read `hp.sample_rate`; add a shipped-fb + center=False path
   gated on a new `f5.mel_spec_type`/`f5.vocoder` GGUF key.

Also verify (likely benign for single-utterance inference): `text_mask_padding:
True` — text padding mask; our runtime feeds exact-length text so no padding.

## Plan (checkpoints are 5.4 GB/16.7 GB torch .pt → convert on Kaggle)

1. **Converter** `models/convert-raon-opentts-to-gguf.py`: reuse map_f5tts_name;
   read dims from config.yaml; add HiFi-GAN state_dict map; compute+ship the
   slaney mel fb + window via torchaudio; new KV `f5.vocoder=hifigan`,
   `f5.mel_spec_type=sbhifigan16k`, `f5.sample_rate=16000`, `f5.mel_dim=80`,
   `f5.norm_type` (informational). **Start with 0.3B** (5.4 GB, same arch,
   cheaper to iterate).
2. **Runtime** `f5_tts.cpp`: (a) mel reads sample_rate/mel_dim, shipped-fb +
   center=False path; (b) `core_hifigan::forward` branch gated on f5.vocoder;
   (c) load HiFi-GAN `voc.*` tensors + hparams. DiT untouched.
3. **Kaggle** kernel: convert 0.3B + dump F5 reference intermediates
   (mel, per-block DiT, velocity, vocoder output) under the F5-TTS lib →
   HF fixtures. Torch can't load the .pt here.
4. **Local**: crispasr-diff per-stage parity vs the ref, then the mandatory
   **TTS→ASR roundtrip** (synthesize → whisper/parakeet → text recognizable).
5. Registry entry (raon / raon-1b) behind the CC-BY-NC-4.0 acceptance gate;
   backend alias → f5-tts runtime; HF upload with attribution; live test.
6. 1B is a converter+kernel re-run once 0.3B proves the runtime.
