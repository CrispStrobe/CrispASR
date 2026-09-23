"""Dolphin (DataoceanAI) reference dump backend — #436.

Drives the upstream `dolphin` package (github.com/DataoceanAI/Dolphin) on one
16 kHz clip and captures the stages crispasr-diff compares. model_dir is a
directory holding `<model_name>.pt`, `train.yaml`, `units.txt`, `global_cmvn`;
DOLPHIN_MODEL_NAME selects the registry name (default small.cn.streaming).

One deliberate departure from upstream: features are computed with dither=0.
Upstream passes train.yaml's fbank_conf (dither 0.1) straight into
torchaudio's kaldi fbank at inference, so its own output is random run to run;
a reference must be reproducible.

Stages:
  raw_audio        (N,)
  fbank            (T, 80)       kaldi fbank of waveform*32768, before CMVN
  subsample_out    (T', d)       encoder.embed output (after the √d scale)
  enc_blk_NN       (T', d)       every E-Branchformer block output
  encoder_output   (T', d)       after encoder.after_norm
  ctc_logprobs     (T', V)
  lang_region      (4,)          [lang, region, <asr>, <notimestamp>] ids
  ctc_nbest_top    (L,)          best CTC prefix-beam hypothesis
  rescore_tokens   (L',)         attention-rescoring output tokens
  text             str
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = ["raw_audio", "fbank", "subsample_out", "encoder_output", "ctc_logprobs",
                  "lang_region", "ctc_nbest_top", "rescore_tokens", "text"] + [f"enc_blk_{i:02d}" for i in range(12)]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str], max_new_tokens: int = 0) -> Dict[str, np.ndarray]:
    import torch
    import torchaudio
    from dolphin.transcribe import load_model
    from dolphin.tokenizer import init_tokenizer  # noqa: F401  (import check)
    from dolphin import search as dsearch

    name = os.environ.get("DOLPHIN_MODEL_NAME", "small.cn.streaming")
    model = load_model(name, str(model_dir), "cpu")
    model.eval()
    cfg = model.model_configs
    out: Dict[str, np.ndarray] = {}

    wav = torch.from_numpy(audio.astype(np.float32))[None]
    fb = dict(cfg["dataset_conf"]["fbank_conf"])
    fb["dither"] = 0.0
    feats = torchaudio.compliance.kaldi.fbank(waveform=wav * (1 << 15), **fb)
    out["raw_audio"] = audio.astype(np.float32)
    out["fbank"] = feats.numpy().astype(np.float32)

    caps: Dict[str, np.ndarray] = {}
    hooks = []
    enc = model.encoder
    hooks.append(enc.embed.register_forward_hook(lambda m, i, o: caps.__setitem__("subsample_out", o[0][0].detach().float().numpy().copy())))
    for i, layer in enumerate(enc.encoders):
        hooks.append(layer.register_forward_hook(lambda m, inp, o, i=i: caps.__setitem__(f"enc_blk_{i:02d}", o[0][0].detach().float().numpy().copy())))

    from dolphin.tokenizer import init_tokenizer as _init_tok
    tokenizer = _init_tok(cfg)
    speech = feats[None]
    lens = torch.tensor([feats.shape[0]])
    with torch.no_grad():
        encoder_out, encoder_mask = model._forward_encoder(speech, lens, -1, -1, False)
        for h in hooks:
            h.remove()
        out.update(caps)
        out["encoder_output"] = encoder_out[0].float().numpy()
        ctc_probs = model.ctc_logprobs(encoder_out, 0.0, 0)
        out["ctc_logprobs"] = ctc_probs[0].float().numpy()
        lr = model.predict_lang_region_timestamp(tokenizer, encoder_out, need_timestamp=False)
        out["lang_region"] = lr[0].numpy().astype(np.int32)
        res = model.decode(methods=["ctc_prefix_beam_search", "attention_rescoring"], speech=speech,
                           speech_lengths=lens, beam_size=10, infos={"tokenizer": tokenizer})
    out["ctc_nbest_top"] = np.asarray(res["ctc_prefix_beam_search"][0].tokens, np.int32)
    toks = list(res["attention_rescoring"][0].tokens)
    out["rescore_tokens"] = np.asarray(toks, np.int32)
    out["text"] = tokenizer.detokenize(toks)[0]
    print(f"  dolphin[{name}] text: {out['text']}")
    return {k: v for k, v in out.items() if (k in stages or not stages)}
