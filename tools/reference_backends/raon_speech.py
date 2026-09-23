"""KRAFTON/Raon-Speech-9B speech-to-text reference dump backend (#455).

Runs the model's own remote code (modeling_raon.py; needs
transformers==4.57.x, the version its config names) through the
RaonPipeline.stt path: RaonProcessor builds the prompt and the 8 s / 24 kHz
audio chunks, RaonModel.get_audio_input_embeds runs the audio tower and the
input adaptor.

Two passes:
  1. generated_text: the whole model in bf16 (as shipped), greedy
     (do_sample=False) so the C++ greedy decode has an exact target (the
     official stt() samples at temperature 0.2, top-k 20, top-p 0.8).
  2. Audio stages: audio_encoder + input_adaptor cast to fp32 and fed fp32
     audio, so the stages are not bf16-rounded:

  raon_mel_chunk{c}     (n_mels, T_c)  each 8 s chunk's log-mel (encoder input)
  raon_encoder_output   (N, 2048)      12.5 Hz encoder frames kept by the mask
  raon_adaptor_output   (N, 4096)      LLM-ready audio embeddings

The input audio is 16 kHz (the CrispASR CLI's rate); the processor resamples
it to 24 kHz exactly as it would a 16 kHz file.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = ["raw_audio", "raon_mel_chunk0", "raon_encoder_output", "raon_adaptor_output", "generated_text"]



def _own(t):
    # a hook's capture must own its storage (see _safe_capture on the aliasing
    # audit branch): later in-place ops would otherwise rewrite it
    return None if t is None else t.detach().clone()


def _classes(model_dir: str):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return (get_class_from_dynamic_module("modeling_raon.RaonModel", model_dir),
            get_class_from_dynamic_module("modeling_raon.RaonProcessor", model_dir),
            get_class_from_dynamic_module("modeling_raon.get_default_stt_prompt", model_dir))


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str], max_new_tokens: int = 0) -> Dict[str, np.ndarray]:
    import soundfile as sf
    import torch

    md = str(model_dir)
    RaonModel, RaonProcessor, get_default_stt_prompt = _classes(md)
    processor = RaonProcessor.from_pretrained(md)
    model = RaonModel.from_pretrained(md, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    with tempfile.TemporaryDirectory(dir=os.environ.get("RAON_TMP")) as td:
        wav = str(Path(td) / "in.wav")
        sf.write(wav, audio.astype(np.float32), 16000, subtype="FLOAT")
        messages = [{"role": "user", "content": [{"type": "audio", "audio": wav},
                                                  {"type": "text", "text": get_default_stt_prompt()}]}]

        def inputs_for(dtype):
            return processor(messages, add_generation_prompt=True, force_audio_output=False, device="cpu",
                             dtype=dtype, max_audio_chunk_length=192000)

        # ---- pass 1: greedy text, bf16 model as shipped ----
        if "generated_text" in stages:
            inp = inputs_for(torch.bfloat16)
            n_in = int(inp["attention_mask"].sum().item())
            with torch.no_grad():
                gen = model.generate(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                                     audio_input=inp.get("audio_input"),
                                     audio_input_lengths=inp.get("audio_input_lengths"),
                                     max_new_tokens=max_new_tokens or 512, do_sample=False,
                                     force_audio_output=False, force_text_output=True, disable_tqdm=True)
            out["generated_text"] = processor.tokenizer.decode(gen["sequences"][0, n_in:], skip_special_tokens=True)
            out["llm_input_ids"] = inp["input_ids"][0].numpy().astype(np.int32)
            print(f"  raon greedy: {out['generated_text']!r}")

        # ---- pass 2: fp32 audio tower + adaptor stages ----
        model.audio_encoder.float()
        model.input_adaptor.float()
        cap: Dict[str, object] = {}
        enc = model.audio_encoder.encoder

        def enc_pre(_m, args, kwargs):
            feats = kwargs.get("input_features", args[0] if args else None)
            lens = kwargs.get("feature_lens", args[1] if len(args) > 1 else None)
            if "mel" not in cap:
                cap["mel"] = [_own(feats[b, :, : int(lens[b])]) for b in range(feats.shape[0])]

        def ad_pre(_m, args, kwargs):
            cap["ad_in"] = _own(args[0] if args else kwargs["inputs"])
            cap["ad_mask"] = _own(kwargs.get("mask", args[1] if len(args) > 1 else None))

        def ad_post(_m, _args, _kwargs, output):
            cap["ad_out"] = _own(output.outputs_embeds)
            cap["ad_out_mask"] = _own(output.mask)

        hs = [enc.register_forward_pre_hook(enc_pre, with_kwargs=True),
              model.input_adaptor.register_forward_pre_hook(ad_pre, with_kwargs=True),
              model.input_adaptor.register_forward_hook(ad_post, with_kwargs=True)]
        inp = inputs_for(torch.float32)
        with torch.no_grad():
            model.get_audio_input_embeds(audio=inp["audio_input"].float(), audio_lengths=inp["audio_input_lengths"])
        for h in hs:
            h.remove()

    m_in = cap["ad_mask"].bool()
    m_out = cap["ad_out_mask"].bool()
    if "raon_mel_chunk0" in stages:
        for c, m in enumerate(cap["mel"]):  # every 8 s chunk; the diff reads chunk 0 and the last
            out[f"raon_mel_chunk{c}"] = m.float().numpy()
    if "raon_encoder_output" in stages:
        out["raon_encoder_output"] = cap["ad_in"][m_in].float().numpy()
    if "raon_adaptor_output" in stages:
        out["raon_adaptor_output"] = cap["ad_out"][m_out].float().numpy()
    print(f"  raon stages: mels {[tuple(m.shape) for m in cap['mel']]} enc {int(m_in.sum())} frames "
          f"adaptor {int(m_out.sum())} frames, placeholders {int((out.get('llm_input_ids', np.zeros(0)) == 151676).sum())}")
    return out
