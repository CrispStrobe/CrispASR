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
  generated_text         str               the package's own beam-search output

CPU dtype adaptation (the one place this file deviates from upstream)
--------------------------------------------------------------------
The encoder and adapter are stored F32; the Qwen3-4B decoder is BF16. Upstream
only ever runs on CUDA, where `HOJO_ASR.autocast_context()` opens an fp16
autocast and the mismatch never surfaces. On CPU that method returns a
`nullcontext`, so `torch.cat([bos_embeds (bf16), speech_embeddings (f32)])`
feeds a float tensor into a BFloat16 Linear and torch raises

    RuntimeError: expected m1 and m2 to have the same dtype,
                  but got: float != c10::BFloat16

`bind_lm_dtype()` below wraps `encode_speech` so its output is cast to the
decoder's own dtype, which is what autocast would have produced anyway.
Casting the DECODER to f32 instead would be 17.6 GB and OOM a Kaggle box.

The consequence is honest and worth stating: every stage up to and including
`speech_embeds` is pure F32 and should reach ~1.0 cosine, while
`prefill_logits_step0` is computed in BF16 and will not. Judge the LM stage by
argmax agreement and by the decoded text, not by the fourth decimal of a
cosine.

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
]


def _np(t):
    return t.detach().to("cpu").float().numpy()


def bind_lm_dtype(model):
    """Cast encode_speech's output to the decoder's dtype (see the module note).

    Idempotent, and a no-op when the dtypes already agree — so the GPU path,
    where autocast handles it, is untouched. Shared with the Kaggle control
    arm rather than duplicated there: two copies of an adaptation like this
    drift, and then the control arm stops being a control.
    """
    if getattr(model, "_crispasr_lm_dtype_bound", False):
        return model
    import torch

    lm_dtype = next(model.decoder_model.parameters()).dtype
    inner = model.encode_speech

    def wrapped(*args, **kwargs):
        emb, attn = inner(*args, **kwargs)
        if emb.dtype != lm_dtype:
            emb = emb.to(lm_dtype)
        return emb, attn

    if lm_dtype != torch.float32:
        model.encode_speech = wrapped
        print(f"  [ref] encode_speech output cast to {lm_dtype} for the LM "
              f"(upstream relies on CUDA autocast for this)")
    model._crispasr_lm_dtype_bound = True
    return model


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
        print(f"  generated_text: {text!r}")

    for name, arr in out.items():
        if isinstance(arr, np.ndarray):
            print(f"  {name:24s} {str(arr.shape):20s} "
                  f"|x|={float(np.linalg.norm(arr.astype(np.float64))):.4f}")
    return out
