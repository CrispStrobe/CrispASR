"""HF-transformers ParakeetForTDT reference dump backend (#454).

For Parakeet checkpoints published in transformers' format rather than as a
.nemo: moondream/parakeet-ultra (F16/F32 weights) and moondream/parakeet-redux
(base-3 packed ternary encoder weights, dequantised here with the converter's
own dequant_ternary, then loaded into the same ParakeetForTDT). Needs a
transformers with parakeet_tdt (5.6.0.dev0 or later).

Stage names and layouts match reference_backends/parakeet.py (NeMo), so
`crispasr-diff parakeet` compares the same boundaries:

  raw_audio          (N,)
  mel_spectrogram    (T_mel, n_mels)   ParakeetFeatureExtractor output (normalised)
  pre_encode_output  (T_enc, d_model)  encoder.subsampling
  encoder_layer_K    (T_enc, d_model)
  encoder_output     (T_enc, d_model)
  generated_text     str               greedy TDT (model.generate)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = ["raw_audio", "mel_spectrogram", "pre_encode_output", "encoder_output", "generated_text"] + [
    f"encoder_layer_{i}" for i in range(24)
]


def _converter():
    path = Path(__file__).resolve().parents[2] / "models" / "convert-parakeet-to-gguf.py"
    spec = importlib.util.spec_from_file_location("convert_parakeet", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(model_dir: Path):
    import torch
    from transformers import AutoConfig, ParakeetForTDT

    md = Path(model_dir)
    if not md.is_dir():
        from huggingface_hub import snapshot_download
        md = Path(snapshot_download(str(model_dir)))
    if not (md / "ternary.json").exists():
        return ParakeetForTDT.from_pretrained(str(md), torch_dtype=torch.float32).eval(), md
    # parakeet-redux: dequantise qweight/scales into ordinary .weight tensors
    from safetensors.torch import load_file

    conv = _converter()
    cfg = json.loads((md / "config.json").read_text())
    raw = load_file(str(md / "model.safetensors"))
    enc = cfg["encoder_config"]
    d, ffd, group = enc["hidden_size"], enc["intermediate_size"], int(cfg.get("ternary_group_size", 128))
    sd = {}
    for k, t in raw.items():
        if k.endswith(".scales"):
            continue
        if k.endswith(".qweight"):
            base = k[: -len(".qweight")]
            w = conv.dequant_ternary(t, raw[base + ".scales"], ffd if base.endswith("linear2") else d, group)
            if ".conv.pointwise_conv" in base:
                w = w.unsqueeze(-1)
            sd[base + ".weight"] = w
        else:
            sd[k] = t
    model = ParakeetForTDT(AutoConfig.from_pretrained(str(md)))
    missing, unexpected = model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()},
                                                strict=False)
    missing = [m for m in missing if not m.startswith("vad_head.")]
    unexpected = [u for u in unexpected if not u.startswith("vad_head.")]
    if missing or unexpected:
        raise RuntimeError(f"redux load: missing={missing[:6]} unexpected={unexpected[:6]}")
    return model.float().eval(), md


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str], max_new_tokens: int = 0) -> Dict[str, np.ndarray]:
    import torch
    from transformers import AutoTokenizer, ParakeetFeatureExtractor

    from . import _hooks

    model, md = _load(model_dir)
    # The moondream repos ship no generation_config.json; set what transformers'
    # convert_nemo_to_hf.py writes for a TDT model (start = blank, the duration
    # logits suppressed from the token argmax).
    gc = model.generation_config
    if gc.decoder_start_token_id is None:
        gc.decoder_start_token_id = model.config.blank_token_id
        durs = getattr(model.config, "durations", None)
        if durs and not gc.suppress_tokens:
            gc.suppress_tokens = list(range(model.config.vocab_size, model.config.vocab_size + len(durs)))
    # moondream ships no preprocessor_config.json; the default extractor is 80
    # mels, these checkpoints take encoder_config.num_mel_bins (128)
    if (md / "preprocessor_config.json").exists():
        fe = ParakeetFeatureExtractor.from_pretrained(str(md))
    else:
        fe = ParakeetFeatureExtractor(feature_size=int(model.config.encoder_config.num_mel_bins))
    tok = AutoTokenizer.from_pretrained(str(md))
    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    captured: Dict = {}
    mods = []
    enc = model.encoder
    if "pre_encode_output" in stages:
        mods.append(("pre_encode_output", enc.subsampling))
    for i, layer in enumerate(enc.layers):
        if f"encoder_layer_{i}" in stages:
            mods.append((f"encoder_layer_{i}", layer))
    handles = _hooks.capture_modules(captured, mods)

    inputs = fe(audio.astype(np.float32), sampling_rate=16000, return_tensors="pt")
    feats, mask = inputs["input_features"], inputs["attention_mask"]
    with torch.no_grad():
        T_valid = int(mask[0].sum().item())
        if "mel_spectrogram" in stages:
            out["mel_spectrogram"] = feats[0, :T_valid].detach().cpu().float().numpy().copy()
        eo = enc(input_features=feats, attention_mask=mask)
        h = eo.last_hidden_state if hasattr(eo, "last_hidden_state") else eo[0]
        T_enc = int(model._get_output_attention_mask(mask).sum().item()) if hasattr(
            model, "_get_output_attention_mask") else h.shape[1]
        if "encoder_output" in stages:
            out["encoder_output"] = h[0, :T_enc].detach().cpu().float().numpy().copy()
        if "generated_text" in stages:
            gen = model.generate(input_features=feats, attention_mask=mask)
            seq = gen.sequences if hasattr(gen, "sequences") else gen
            out["generated_text"] = tok.batch_decode(seq, skip_special_tokens=True)[0]
    _hooks.drop_hooks(handles)
    out.update(_hooks.finalize(captured, T_max=T_enc))
    return out
