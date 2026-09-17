"""Reference backend for HojoAI/Hojo-ASR-Multi-V1 (crispasr-diff), issue #438.

Drives the REAL `hojo-asr` package (PyPI, Apache-2.0 — the same code the
model card and the HF Space run) and taps its modules with forward hooks.
Nothing here re-implements the forward pass: a hand-rolled "reference" would
only prove that two of my readings of the paper agree with each other.

    pip install hojo-asr        # pulls torch, transformers>=4.57.3, omegaconf

Stages
------
  mel_spectrogram        (128, T_mel)      WhisperFeatureExtractor, padding=False
  encoder_output         (T_enc, 2048)     ModifyQwen3OmniMoeAudioEncoder output
  adapter_output         (T_enc, 2560)     ConformerEncoder (bottleneck) output
  speech_embeds          (T_enc, 2560)     after ln_speech — what the LM consumes
  prefill_inputs_embeds  (1+T_enc, 2560)   [embed(<|im_start|>)] ++ speech
  prefill_logits_step0   (vocab,)          logits at the last prefill position
  prefill_argmax_step0   (1,)              argmax of the above
  generated_text         str               the package's own beam-4 output (the recipe)
  generated_text_greedy  str               the same pipeline at num_beams=1

Why both: the C++ runtime defaults to GREEDY, because `core_beam_decode`
replays each beam's whole suffix every step (O(B*T^2) vs greedy's O(T)) and
beam 4 on this 4.4 B decoder is hours per utterance. Comparing C++ greedy
against a beam-4 reference would conflate decode strategy with port
correctness, so the reference emits both and the roundtrip compares like with
like.

CPU dtype adaptation (the one place this file deviates from upstream)
--------------------------------------------------------------------
The encoder and adapter are stored F32; the Qwen3-4B decoder is BF16. Upstream
only ever runs on CUDA, where `HOJO_ASR.autocast_context()` opens an fp16
autocast and the mismatch never surfaces. On CPU that method returns a
`nullcontext`, so `torch.cat([bos_embeds (bf16), speech_embeddings (f32)])`
feeds a float tensor into a BFloat16 Linear and torch raises

    RuntimeError: expected m1 and m2 to have the same dtype,
                  but got: float != c10::BFloat16

`bind_lm_dtype()` reconciles the two. It has two strategies and the choice
matters for WALL TIME, not just correctness:

  * **f32 decoder (default on CPU).** Casts the decoder's parameters to f32 in
    place, one at a time so the peak is ~f32 size rather than f32 + bf16 at
    once. ~17.6 GB for the decoder, ~21 GB with the encoder and adapter — it
    fits a ~30 GB box. torch's CPU f32 gemm goes through oneDNN.
  * **cast the speech embeddings to bf16** (`prefer_f32_lm=False`, and the only
    option on GPU). Cheap in memory and what CUDA autocast effectively does.

The first version of this file used the second strategy on CPU "to avoid
17.6 GB" — and cost a Kaggle run 90 minutes, because PyTorch has no fast CPU
path for bf16 gemm on a machine without AMX. Measured against the real shapes:
1.11 TFLOP of prefill plus 7.06 TFLOP of beam-4 decode is ~2 minutes at
~60 GFLOPS (f32/oneDNN) and ~45 minutes at ~3 GFLOPS (bf16 fallback), per
utterance. Checking that an arm RUNS is not the same as checking it is
affordable.

With the f32 decoder the whole reference is f32 end to end, which is also the
cleanest possible parity target: bf16 is the checkpoint's storage dtype, not a
semantic choice, and the C++ runs f16 weights with f32 accumulation.

Memory
------
The merged checkpoint is 11.96 GB. `HOJO_ASR.load_model` peaks around 24 GB
(the freshly-constructed model plus the state dict) before `assign=True`
releases the duplicates, so it needs a ~30 GB box. This dumper never clones
the model. Set HOJO_ASR_DEVICE=cuda for the upstream GPU path.

Env
---
  HOJO_ASR_DIR        model dir / HF id (default: the --model-dir argument)
  HOJO_ASR_DEVICE     torch device (default "cpu")
  HOJO_ASR_MAX_NEW    override the generated_text token cap
  HOJO_ASR_NUM_BEAMS  override generate.num_beams (default: config.yaml's 4)
"""

import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "mel_spectrogram",
    "encoder_output",
    "adapter_output",
    "speech_embeds",
    "prefill_inputs_embeds",
    "prefill_logits_step0",
    "prefill_argmax_step0",
    "generated_text",
    "generated_text_greedy",
]


def _np(t):
    return t.detach().to("cpu").float().numpy()


def _rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("nan")


def bind_lm_dtype(model, prefer_f32_lm=None):
    """Make the speech embeddings and the decoder agree on a dtype.

    See the module note. `prefer_f32_lm` defaults to True on CPU (fast oneDNN
    f32 gemm, ~21 GB) and False elsewhere (GPU autocast already handles it).
    Idempotent, and a no-op when the dtypes already agree.
    """
    if getattr(model, "_crispasr_lm_dtype_bound", False):
        return model
    import torch

    lm_dtype = next(model.decoder_model.parameters()).dtype
    if lm_dtype == torch.float32:
        model._crispasr_lm_dtype_bound = True
        return model

    on_cpu = model.device.type == "cpu"
    if prefer_f32_lm is None:
        prefer_f32_lm = on_cpu
    if os.environ.get("HOJO_ASR_LM_F32"):
        prefer_f32_lm = os.environ["HOJO_ASR_LM_F32"] not in ("0", "", "false")

    if prefer_f32_lm:
        before = _rss_gb()
        n = 0
        for prm in model.decoder_model.parameters():
            if prm.dtype != torch.float32:
                prm.data = prm.data.float()
                n += prm.numel()
        for buf in model.decoder_model.buffers():
            if buf.is_floating_point() and buf.dtype != torch.float32:
                buf.data = buf.data.float()
        print(f"  [ref] decoder cast {lm_dtype} -> float32 ({n/1e9:.2f} B params); "
              f"RSS {before:.1f} -> {_rss_gb():.1f} GB. bf16 gemm has no fast CPU "
              f"path; f32 is ~20x quicker here.")
    else:
        inner = model.encode_speech

        def wrapped(*args, **kwargs):
            emb, attn = inner(*args, **kwargs)
            if emb.dtype != lm_dtype:
                emb = emb.to(lm_dtype)
            return emb, attn

        model.encode_speech = wrapped
        print(f"  [ref] encode_speech output cast to {lm_dtype} for the LM "
              f"(upstream relies on CUDA autocast for this)")
    model._crispasr_lm_dtype_bound = True
    return model


def decode_rate_probe(model, spectrogram, spectrogram_lens, max_new):
    """Time one prefill + one decode step and PROJECT the full decode cost.

    Exists because a 90-minute Kaggle run was spent discovering that a arm
    which ran correctly ran 20x too slowly. The projection is printed before
    the expensive call, so the log explains its own wall time instead of
    leaving it to be inferred afterwards.
    """
    import time

    import torch

    with torch.no_grad():
        speech, _ = model.encode_speech(spectrogram, spectrogram_lens)
        bos = torch.ones(1, 1, dtype=torch.int32, device=model.device) * model.bos_token_id
        emb = model.decoder_model.model.embed_tokens(bos)
        inp = torch.cat([emb, speech.to(emb.dtype)], dim=1)

        t0 = time.perf_counter()
        out = model.decoder_model(inputs_embeds=inp, use_cache=True)
        t_prefill = time.perf_counter() - t0

        nxt = out.logits[:, -1:, :].argmax(-1)
        step_emb = model.decoder_model.model.embed_tokens(nxt)
        t0 = time.perf_counter()
        model.decoder_model(inputs_embeds=step_emb, past_key_values=out.past_key_values, use_cache=True)
        t_step = time.perf_counter() - t0

    beams = int(model.config.generate.get("num_beams", 4))
    projected = t_prefill + t_step * beams * max_new
    print(f"  [ref] prefill {t_prefill:.1f}s over {inp.shape[1]} positions; "
          f"decode {t_step:.2f}s/step; beams={beams}, max_new={max_new}")
    print(f"  [ref] PROJECTED worst-case decode: {projected/60:.1f} min for this utterance")
    if projected > 15 * 60:
        print("  [ref] !! that is slow enough to suspect a dtype without a fast "
              "kernel — check the cast above before blaming the build", flush=True)
    return projected


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch

    try:
        from hojo_asr import HOJO_ASR
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "hojo-asr is not installed. `pip install hojo-asr` — the reference "
            "MUST be the upstream package, not a re-implementation."
        ) from exc

    src = os.environ.get("HOJO_ASR_DIR", str(model_dir))
    device = os.environ.get("HOJO_ASR_DEVICE", "cpu")

    model = HOJO_ASR.load_model(src, device=device)
    model.eval()
    # NOTE: hooks below tap the UNCAST f32 activations — bind_lm_dtype only
    # changes what reaches the LM, so speech_embeds stays a clean f32 reference.
    bind_lm_dtype(model)

    out: Dict[str, np.ndarray] = {}

    # ---- 1. mel (the package's own feature extractor, padding=False) ----
    wav = np.asarray(audio, dtype=np.float32)
    feats = model.feat_extractor(wav, sampling_rate=16000, return_tensors="pt",
                                 padding=False).input_features
    # (1, 128, T) -> (T, 128), which is what dataset.padding_for_batch produces.
    feat = feats.squeeze(0).transpose(0, 1)
    if "mel_spectrogram" in stages:
        # C++ emits (n_mels, T_mel) mel-major; transpose back to match.
        out["mel_spectrogram"] = np.ascontiguousarray(_np(feat.transpose(0, 1)))

    spectrogram = feat.unsqueeze(0).to(model.device)
    spectrogram_lens = torch.tensor([feat.size(0)], dtype=torch.int64, device=model.device)

    # ---- 2. encoder / adapter / ln_speech, via hooks on encode_speech ----
    taps: Dict[str, np.ndarray] = {}
    handles = []
    handles.append(model.speech_encoder.register_forward_hook(
        lambda m, i, o: taps.__setitem__("encoder_output", _np(o.last_hidden_state))))
    handles.append(model.bottleneck.register_forward_hook(
        lambda m, i, o: taps.__setitem__("adapter_output", _np(o[0]))))
    handles.append(model.ln_speech.register_forward_hook(
        lambda m, i, o: taps.__setitem__("speech_embeds", _np(o))))

    with torch.no_grad():
        speech_embeddings, speech_attn = model.encode_speech(spectrogram, spectrogram_lens)
    for h in handles:
        h.remove()

    for key in ("encoder_output", "adapter_output", "speech_embeds"):
        if key in stages and key in taps:
            a = taps[key]
            out[key] = np.ascontiguousarray(a.reshape(-1, a.shape[-1]))

    # The hook on ln_speech and encode_speech's return value must agree; if they
    # ever don't, the hook is tapping the wrong module and every later stage is
    # being compared against the wrong tensor. The hook sees F32 while the
    # return value may have been cast by bind_lm_dtype, so the hook value is
    # rounded through the SAME dtype before comparing — otherwise this check
    # would measure the cast (~4e-3 for bf16) instead of the wiring, and a
    # tolerance wide enough to pass it would be wide enough to hide a
    # wrong-module hook. A real mis-hook moves this by O(1).
    ref_speech = _np(speech_embeddings).reshape(-1, speech_embeddings.shape[-1])
    if "speech_embeds" in out:
        hook_rounded = _np(torch.from_numpy(out["speech_embeds"]).to(speech_embeddings.dtype))
        delta = float(np.max(np.abs(hook_rounded - ref_speech)))
        print(f"  [ref] ln_speech hook vs encode_speech() return: max|delta| = {delta:.3e} "
              f"(dtype {speech_embeddings.dtype})")
        if delta > 1e-6:
            raise SystemExit(f"ln_speech hook disagrees with encode_speech() by {delta:.3e}")
    elif "speech_embeds" in stages:
        out["speech_embeds"] = np.ascontiguousarray(ref_speech)

    # ---- 3. LM prefill ----
    bos_id = model.bos_token_id
    bos_ids = torch.ones(1, 1, dtype=torch.int32, device=model.device) * bos_id
    with torch.no_grad():
        bos_embeds = model.decoder_model.model.embed_tokens(bos_ids)
        inputs_embeds = torch.cat([bos_embeds, speech_embeddings.to(bos_embeds.dtype)], dim=1)
    if "prefill_inputs_embeds" in stages:
        a = _np(inputs_embeds)
        out["prefill_inputs_embeds"] = np.ascontiguousarray(a.reshape(-1, a.shape[-1]))

    if "prefill_logits_step0" in stages or "prefill_argmax_step0" in stages:
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=model.device)
        with torch.no_grad():
            lm_out = model.decoder_model(inputs_embeds=inputs_embeds, attention_mask=attn)
        logits = lm_out.logits[0, -1, :]
        if "prefill_logits_step0" in stages:
            out["prefill_logits_step0"] = np.ascontiguousarray(_np(logits))
        if "prefill_argmax_step0" in stages:
            out["prefill_argmax_step0"] = np.array([int(torch.argmax(logits).item())], dtype=np.float32)

    # ---- 4. the package's own decode, recipe untouched ----
    if "generated_text" in stages:
        decode_rate_probe(model, spectrogram, spectrogram_lens,
                          max(10, min(int(model.config.generate.get("max_new_tokens", 200)),
                                      speech_embeddings.shape[1] * 2 + 10)))
        gen_cfg = dict(model.config.generate)
        if os.environ.get("HOJO_ASR_MAX_NEW"):
            gen_cfg["max_new_tokens"] = int(os.environ["HOJO_ASR_MAX_NEW"])
        elif max_new_tokens > 0:
            gen_cfg["max_new_tokens"] = int(max_new_tokens)
        if os.environ.get("HOJO_ASR_NUM_BEAMS"):
            gen_cfg["num_beams"] = int(os.environ["HOJO_ASR_NUM_BEAMS"])
        batch = {"spectrogram": spectrogram, "spectrogram_lens": spectrogram_lens}
        with torch.no_grad():
            texts = model.infer(batch, gen_cfg)
        text = texts[0].replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        out["generated_text"] = text
        print(f"  generated_text (beams={gen_cfg.get('num_beams')}): {text!r}")

    if "generated_text_greedy" in stages:
        # Matched arm for the C++ default. Same recipe otherwise -- only
        # num_beams changes -- so a difference between this and the C++ output
        # is the port, not the search.
        g_cfg = dict(model.config.generate)
        g_cfg["num_beams"] = 1
        if os.environ.get("HOJO_ASR_MAX_NEW"):
            g_cfg["max_new_tokens"] = int(os.environ["HOJO_ASR_MAX_NEW"])
        elif max_new_tokens > 0:
            g_cfg["max_new_tokens"] = int(max_new_tokens)
        batch = {"spectrogram": spectrogram, "spectrogram_lens": spectrogram_lens}
        with torch.no_grad():
            g_texts = model.infer(batch, g_cfg)
        g_text = g_texts[0].replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        out["generated_text_greedy"] = g_text
        print(f"  generated_text_greedy: {g_text!r}")

    for name, arr in out.items():
        if isinstance(arr, np.ndarray):
            print(f"  {name:24s} {str(arr.shape):20s} "
                  f"|x|={float(np.linalg.norm(arr.astype(np.float64))):.4f}")
    return out
