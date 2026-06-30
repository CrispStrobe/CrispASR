"""Reference backend for MOSS-Transcribe-preview-2B (crispasr-diff).

Captures stage-by-stage activations from the official PyTorch model to diff
against the C++ ggml runtime. Unlike the sibling moss_audio backend, the
MOSS-Transcribe HF repo ships its modeling + processing code, so we import
those directly (no separate GitHub clone) and rely on the stock transformers
Qwen3OmniMoeAudioEncoder + Qwen3Model classes.

Stages: mel_spectrogram, enc_layer_0, encoder_output, adapter_output,
        prefill_inputs_embeds, prefill_last_hidden, prefill_logits_step0,
        prefill_argmax_step0, generated_text.
"""

import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

# Two-phase low-memory dump (fits a 16 GB box; never loads the full 2.4B model
# at once — that OOMs >12 GB). Phase A loads ONLY the audio encoder + adapter
# (~0.6B, ~2.6 GB f32); Phase B frees those and loads ONLY the Qwen3-1.7B LM
# standalone (bf16, ~3.4 GB) to dump the LM prefill stages + a manual greedy
# decode for the ground-truth transcript.
DEFAULT_STAGES = [
    "mel_spectrogram",
    "enc_layer_0",
    "encoder_output",
    "adapter_output",
    "prefill_inputs_embeds",
    "prefill_lm_layer_0",
    "prefill_lm_layer_mid",
    "prefill_lm_layer_last",
    "prefill_last_hidden",
    "prefill_logits_step0",
    "prefill_argmax_step0",
    "generated_text",
]


def _import_from_file(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch

    model_dir = Path(os.environ.get("MOSS_TRANSCRIBE_DIR", str(model_dir)))
    max_new = int(os.environ.get("MOSS_TRANSCRIBE_MAX_NEW", str(max_new_tokens)))

    # ---- Import the repo's modeling + processing code directly ----
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    modeling = _import_from_file("modeling_Moss", model_dir / "modeling_Moss.py")
    processing = _import_from_file("processing_Moss", model_dir / "processing_Moss.py")

    MossForCausalLM = modeling.MossForCausalLM
    MossConfig = modeling.MossConfig
    MossProcessor = processing.MossProcessor
    MelConfig = processing.MelConfig

    device = "cpu"
    dtype = torch.bfloat16

    with open(model_dir / "config.json") as f:
        cfg_dict = json.load(f)
    config = MossConfig(
        audio_config=cfg_dict.get("audio_config"),
        language_config=cfg_dict.get("language_config"),
        adapter_hidden_size=cfg_dict.get("adapter_hidden_size", 8192),
    )

    want_lm = any(s in stages for s in [
        "prefill_inputs_embeds", "prefill_last_hidden", "prefill_lm_layer_0",
        "prefill_lm_layer_mid", "prefill_lm_layer_last",
        "prefill_logits_step0", "prefill_argmax_step0", "generated_text"])

    # ---- Processor (README MelConfig: 128 bins / n_fft 400 / hop 160) ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    mel_cfg = MelConfig(mel_sr=16000, mel_dim=128, mel_n_fft=400, mel_hop_length=160,
                        mel_dtype=torch.float32)
    processor = MossProcessor(tokenizer=tokenizer, config=mel_cfg, enable_time_marker=False)
    # Canonical inference loads the chat template (user/assistant ChatML framing);
    # the bare legacy layout makes the model emit garbage.
    tmpl = model_dir / "chat_template_default.py"
    if tmpl.exists():
        processor.load_template(str(tmpl))
        print(f"  loaded chat template: {tmpl.name}")
    inputs = processor(audio, return_tensors="pt")
    audio_data = inputs["audio_data"].to(device=device, dtype=torch.float32)
    audio_data_seqlens = inputs["audio_data_seqlens"].to(device)
    audio_input_mask = inputs["audio_input_mask"].to(device)
    input_ids = inputs["input_ids"].to(device)
    print(f"  input_ids: {input_ids.shape[1]} tokens, "
          f"audio frames: {int(audio_input_mask.sum().item())}, "
          f"mel: {tuple(audio_data.shape)}")

    out: Dict[str, np.ndarray] = {}
    if "mel_spectrogram" in stages:
        m = audio_data[0] if audio_data.dim() == 3 else audio_data
        out["mel_spectrogram"] = m.detach().cpu().float().numpy()  # (128, T)

    from safetensors import safe_open
    import glob as _glob
    st_files = sorted(_glob.glob(str(model_dir / "model-*.safetensors"))) or \
        sorted(_glob.glob(str(model_dir / "*.safetensors")))

    # ================= PHASE A: audio encoder + adapter (~2.6 GB f32) ========
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder)
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig)
    audio_cfg = Qwen3OmniMoeAudioEncoderConfig(**cfg_dict["audio_config"])
    audio_cfg._attn_implementation = "eager"
    encoder = Qwen3OmniMoeAudioEncoder(audio_cfg).to(device).eval()
    lang_hidden = cfg_dict["language_config"]["hidden_size"]
    adapter = modeling.MossGatedMLP(audio_cfg.output_dim,
                                    cfg_dict.get("adapter_hidden_size", 8192),
                                    lang_hidden).to(device).eval()
    enc_sd, adp_sd = encoder.state_dict(), adapter.state_dict()
    n_enc = n_adp = 0
    with torch.no_grad():
        for sf in st_files:
            with safe_open(sf, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("model.audio_model."):
                        kk = k[len("model.audio_model."):]
                        if kk in enc_sd:
                            enc_sd[kk].copy_(f.get_tensor(k).float()); n_enc += 1
                    elif k.startswith("model.audio_adapter."):
                        kk = k[len("model.audio_adapter."):]
                        if kk in adp_sd:
                            adp_sd[kk].copy_(f.get_tensor(k).float()); n_adp += 1
    print(f"  [A] encoder({n_enc}) + adapter({n_adp}) tensors, f32")

    adapter_out_np = None
    with torch.no_grad():
        captures, handles = {}, []
        if "enc_layer_0" in stages:
            def _l0(m, i, o):
                captures["enc_layer_0"] = (o[0] if isinstance(o, tuple) else o).detach().clone()
            handles.append(encoder.layers[0].register_forward_hook(_l0))
        enc_last = encoder(input_features=audio_data, feature_lens=audio_data_seqlens).last_hidden_state
        for h in handles:
            h.remove()
        if "enc_layer_0" in stages and "enc_layer_0" in captures:
            t = captures["enc_layer_0"]; out["enc_layer_0"] = (t[0] if t.dim() == 3 else t).cpu().float().numpy()
        if "encoder_output" in stages:
            out["encoder_output"] = (enc_last[0] if enc_last.dim() == 3 else enc_last).cpu().float().numpy()
        adapted = adapter(enc_last)
        adapted2d = adapted[0] if adapted.dim() == 3 else adapted
        adapter_out_np = adapted2d.cpu().float().numpy()  # (T_enc, lang_hidden)
        if "adapter_output" in stages:
            out["adapter_output"] = adapter_out_np

    if not want_lm:
        return out

    # Free Phase-A modules before loading the LM.
    del encoder, adapter, enc_sd, adp_sd
    gc.collect()

    # ================= PHASE B: standalone Qwen3-1.7B LM (bf16, ~3.4 GB) =====
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    lang_cfg = Qwen3Config(**cfg_dict["language_config"])
    lang_cfg._attn_implementation = "eager"
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        lm = Qwen3Model(lang_cfg).to(device).eval()
    finally:
        torch.set_default_dtype(prev)
    lm_sd = lm.state_dict()
    n_lm = 0
    with torch.no_grad():
        for sf in st_files:
            with safe_open(sf, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("model.language_model."):
                        kk = k[len("model.language_model."):]
                        if kk in lm_sd:
                            lm_sd[kk].copy_(f.get_tensor(k).to(torch.bfloat16)); n_lm += 1
    gc.collect()
    embed_w = lm.embed_tokens.weight  # tied: lm_head.weight == embed_tokens.weight
    print(f"  [B] Qwen3 LM ({n_lm}) tensors, bf16; lm_head tied to embed")

    import torch.nn.functional as F
    bdt = torch.bfloat16
    adapter_t = torch.from_numpy(adapter_out_np).to(device=device, dtype=bdt)  # (T_enc, H)

    def embed_and_splice(ids):
        emb = lm.embed_tokens(ids)  # (1, L, H) bf16
        m = (audio_input_mask.unsqueeze(-1)).expand_as(emb)
        emb = emb.clone()
        emb.masked_scatter_(m, adapter_t.reshape(-1).to(emb.dtype))
        return emb

    with torch.no_grad():
        layer_caps, handles = {}, []
        mid = lang_cfg.num_hidden_layers // 2
        taps = {0: "prefill_lm_layer_0", mid: "prefill_lm_layer_mid",
                lang_cfg.num_hidden_layers - 1: "prefill_lm_layer_last"}
        for li, nm in taps.items():
            if nm in stages:
                def mk(name):
                    def _h(m, i, o):
                        layer_caps[name] = (o[0] if isinstance(o, tuple) else o).detach().clone()
                    return _h
                handles.append(lm.layers[li].register_forward_hook(mk(nm)))

        emb = embed_and_splice(input_ids)
        if "prefill_inputs_embeds" in stages:
            out["prefill_inputs_embeds"] = emb[0].cpu().float().numpy()
        lm_out = lm(inputs_embeds=emb, use_cache=True, return_dict=True)
        hidden = lm_out.last_hidden_state  # (1, L, H)
        past = lm_out.past_key_values
        for h in handles:
            h.remove()
        for nm, t in layer_caps.items():
            out[nm] = (t[0] if t.dim() == 3 else t).cpu().float().numpy()
        logits_last = F.linear(hidden[:, -1], embed_w)  # (1, vocab)
        if "prefill_last_hidden" in stages:
            out["prefill_last_hidden"] = hidden[0, -1].cpu().float().numpy()
        if "prefill_logits_step0" in stages:
            out["prefill_logits_step0"] = logits_last[0].cpu().float().numpy()
        first_id = int(logits_last[0].argmax().item())
        if "prefill_argmax_step0" in stages:
            out["prefill_argmax_step0"] = np.array([first_id], dtype=np.int32)

        if "generated_text" in stages:
            eos = int(processor.end_token_id)
            gen_ids = []
            nxt = first_id
            cache_pos = int(input_ids.shape[1])
            for _ in range(max_new):
                if nxt == eos:
                    break
                gen_ids.append(nxt)
                step_emb = lm.embed_tokens(torch.tensor([[nxt]], device=device))
                step_out = lm(inputs_embeds=step_emb, past_key_values=past, use_cache=True,
                              cache_position=torch.tensor([cache_pos], device=device),
                              return_dict=True)
                past = step_out.past_key_values
                nxt = int(F.linear(step_out.last_hidden_state[:, -1], embed_w)[0].argmax().item())
                cache_pos += 1
            out["generated_text"] = tokenizer.decode(gen_ids, skip_special_tokens=True)
            print(f"  [B] greedy transcript: {out['generated_text'][:200]}")
            print(f"  [B] first token id={first_id} ('{tokenizer.decode([first_id])}')")
    return out
