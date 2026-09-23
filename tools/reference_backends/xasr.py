"""X-ASR (icefall streaming Zipformer2 transducer) reference dump backend — #436.

Runs the upstream icefall PyTorch modules in their streaming path, chunk by
chunk, exactly as sherpa-onnx drives the exported encoder
(`export-onnx-streaming.py` OnnxEncoder.forward + sherpa's chunk pump and
greedy search; docs/xasr/PLAN.md). The icefall sources are not a pip package:
XASR_ICEFALL_DIR must hold egs/librispeech/ASR/zipformer/{zipformer,scaling,
subsampling,decoder,joiner}.py. `k2`, `icefall.utils` and `encoder_interface`
are stubbed — at inference k2 only supplies the Swoosh activations, which are
replaced by the same formulas the ONNX export uses.

model_dir is upstream's `deployment/models` (chunk-*ms-model/ with the
sherpa-onnx encoder/decoder/joiner + tokens.txt). Weights come from the ONNX
export through models/convert-xasr-to-gguf.py:onnx_state_dict — the same
extraction the converter uses; `streaming_exp/pretrained.pt` is a different
(earlier) checkpoint and is not used.
  XASR_CHUNK_MS     160 / 480 (default) / 960 / 1920 — picks the export and geometry
  XASR_TAIL_PAD_MS  silence appended before input_finished; default T*10+1000
                    (the runtime's default: every real frame lands in a
                    decoded window, and the transducer gets 1 s to emit)

Features are kaldi-native-fbank with sherpa-onnx's FeatureExtractorConfig
defaults (the `kaldi_native_fbank` package sherpa uses).

Stages (every per-chunk stage concatenated over chunks, time-major):
  raw_audio        (N,)
  fbank            (F, 80)       all frames, tail padding included
  embed_out        (C*chunk, 192)  encoder_embed.streaming_forward
  stack_S          (C*chunk, dim_S) Zipformer2 stack S output at 50 Hz
  enc_full         (C*chunk/2, 768) after _get_full_dim_output + downsample_output
  encoder_out      (C*chunk/2, 512) after joiner.encoder_proj
  first_logits     (V,)          joiner logits of the first encoder frame, blank context
  tokens           (L,)          greedy tokens (sherpa: <=1 symbol per frame, skip blank/unk)
  text             str           rendered as sherpa-onnx does (RemoveSpaceBetweenCjk + leading-space strip)
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import types
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = ["raw_audio", "fbank", "embed_out", "enc_full", "encoder_out", "first_logits", "tokens", "text"] + [
    f"stack_{i}" for i in range(6)
]

def _stub_icefall(icefall_dir: str):
    import torch

    k2 = types.ModuleType("k2")

    def swoosh_l(x):
        return torch.logaddexp(torch.zeros_like(x), x - 4.0) - 0.08 * x - 0.035

    def swoosh_r(x):
        return torch.logaddexp(torch.zeros_like(x), x - 1.0) - 0.08 * x - 0.313261687

    k2.swoosh_l_forward = k2.swoosh_l = swoosh_l
    k2.swoosh_r_forward = k2.swoosh_r = swoosh_r
    sys.modules["k2"] = k2
    ic = types.ModuleType("icefall")
    icu = types.ModuleType("icefall.utils")
    icu.torch_autocast = lambda *a, **k: contextlib.nullcontext()
    ic.utils = icu
    sys.modules["icefall"] = ic
    sys.modules["icefall.utils"] = icu
    ei = types.ModuleType("encoder_interface")
    ei.EncoderInterface = torch.nn.Module
    sys.modules["encoder_interface"] = ei
    sys.path.insert(0, icefall_dir)


def _ints(v):
    return [int(x) for x in str(v).split(",")]


def _converter():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "models" / "convert-xasr-to-gguf.py"
    spec = importlib.util.spec_from_file_location("convert_xasr", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(model_dir: Path, chunk_ms: int):
    import torch

    _stub_icefall(os.environ["XASR_ICEFALL_DIR"])
    from zipformer import Zipformer2
    from subsampling import Conv2dSubsampling
    from decoder import Decoder
    from joiner import Joiner

    chunk_dir = Path(model_dir) / f"chunk-{chunk_ms}ms-model"
    sd, hp = _converter().onnx_state_dict(chunk_dir)
    sd = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}
    dims = hp["dims"]
    chunk = hp["decode_chunk_len"] // 2  # 50 Hz frames after encoder_embed
    left = hp["left_context_frames"]
    enc = Zipformer2(
        output_downsampling_factor=2, downsampling_factor=tuple(hp["downsample"]), num_encoder_layers=hp["n_layers"],
        encoder_dim=dims, encoder_unmasked_dim=dims, query_head_dim=hp["qd"], pos_head_dim=hp["pd"],
        value_head_dim=hp["vd"], pos_dim=hp["pos_dim"], num_heads=hp["n_heads"], feedforward_dim=hp["ffn_dim"],
        cnn_module_kernel=hp["kernel"], dropout=0.0, warmup_batches=4000.0, causal=True,
        chunk_size=[chunk], left_context_frames=[left])
    embed = Conv2dSubsampling(in_channels=hp["feature_dim"], out_channels=dims[0], dropout=0.0)
    dec = Decoder(vocab_size=hp["vocab_size"], decoder_dim=hp["decoder_dim"], blank_id=0, context_size=hp["context_size"])
    joi = Joiner(encoder_dim=max(dims), decoder_dim=hp["decoder_dim"], joiner_dim=hp["joiner_dim"],
                 vocab_size=hp["vocab_size"])

    def load(mod, prefix):
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = mod.load_state_dict(sub, strict=False)
        # the per-layer `bypass_scale` ("TODO: remove it") is never read, so the export drops it
        missing = [k for k in missing if not re.fullmatch(r".*layers\.\d+\.bypass_scale", k)]
        if missing or unexpected:
            raise RuntimeError(f"{prefix}: missing={missing[:8]} unexpected={unexpected[:8]}")

    load(enc, "encoder.")
    load(embed, "encoder_embed.")
    load(dec, "decoder.")
    load(joi, "joiner.")
    for m in (enc, embed, dec, joi):
        m.eval()
    return hp, enc, embed, dec, joi, chunk, left


def fbank(audio: np.ndarray, tail_pad_ms: int) -> np.ndarray:
    import kaldi_native_fbank as knf

    o = knf.FbankOptions()
    o.frame_opts.samp_freq = 16000
    o.frame_opts.dither = 0.0
    o.frame_opts.snip_edges = False
    o.frame_opts.remove_dc_offset = True
    o.frame_opts.preemph_coeff = 0.97
    o.frame_opts.window_type = "povey"
    o.mel_opts.num_bins = 80
    o.mel_opts.low_freq = 20.0
    o.mel_opts.high_freq = -400.0
    fb = knf.OnlineFbank(o)
    fb.accept_waveform(16000, audio.astype(np.float32).tolist())
    fb.accept_waveform(16000, np.zeros(16 * tail_pad_ms, dtype=np.float32).tolist())
    fb.input_finished()
    return np.stack([np.asarray(fb.get_frame(i)) for i in range(fb.num_frames_ready)]).astype(np.float32)


def run(enc, embed, dec, joi, feats: np.ndarray, chunk: int, left: int, caps=None):
    """sherpa-onnx OnlineRecognizer greedy search over `feats` with the torch modules."""
    import torch

    decode_chunk_len = 2 * chunk
    T = decode_chunk_len + 13
    states = enc.get_init_states(1)
    embed_cache = embed.get_init_states(1)
    processed_lens = torch.zeros(1, dtype=torch.int64)
    ctx = int(dec.context_size)
    hyp = [-1] * (ctx - 1) + [0]
    unk = getattr(run, "unk_id", -1)

    def dec_out():
        y = torch.tensor([hyp[-ctx:]], dtype=torch.int64)
        return joi.decoder_proj(dec(y, need_pad=False))  # (1, 1, dj)

    d = dec_out()
    processed = 0
    stack_outs = []
    orig = [m.streaming_forward for m in enc.encoders]
    for i, m in enumerate(enc.encoders):
        def wrap(*a, _f=orig[i], _i=i, **k):
            r = _f(*a, **k)
            if caps is not None:
                caps.setdefault(f"stack_{_i}", []).append(r[0][:, 0].detach().numpy().copy())
            return r
        m.streaming_forward = wrap
    ds_orig = enc.downsample_output.forward

    def ds_wrap(x):
        y = ds_orig(x)
        if caps is not None:
            caps.setdefault("enc_full", []).append(y[:, 0].detach().numpy().copy())
        return y
    enc.downsample_output.forward = ds_wrap
    first_logits = None
    with torch.no_grad():
        while processed + T < feats.shape[0]:
            x = torch.from_numpy(feats[processed:processed + T])[None]
            processed += decode_chunk_len
            x_lens = torch.tensor([T])
            xe, xl, embed_cache = embed.streaming_forward(x=x, x_lens=x_lens, cached_left_pad=embed_cache)
            assert xe.size(1) == chunk
            if caps is not None:
                caps.setdefault("embed_out", []).append(xe[0].numpy().copy())
            pm = torch.arange(left).expand(1, left)
            pm = (processed_lens.unsqueeze(1) <= pm).flip(1)
            mask = torch.cat([pm, torch.zeros(1, chunk, dtype=torch.bool)], dim=1)
            processed_lens = processed_lens + xl
            eo, _, states = enc.streaming_forward(x=xe.permute(1, 0, 2), x_lens=xl, states=states,
                                                  src_key_padding_mask=mask)
            eo = joi.encoder_proj(eo.permute(1, 0, 2))[0]  # (chunk/2, dj)
            if caps is not None:
                caps.setdefault("encoder_out", []).append(eo.numpy().copy())
            for t in range(eo.shape[0]):
                logit = joi.output_linear(torch.tanh(eo[t] + d[0, 0]))
                if first_logits is None:
                    first_logits = logit.numpy().copy()
                y = int(torch.argmax(logit))
                if y != 0 and y != unk:
                    hyp.append(y)
                    d = dec_out()
    for m, f in zip(enc.encoders, orig):
        m.streaming_forward = f
    enc.downsample_output.forward = ds_orig
    return hyp[ctx:], first_logits


def _is_cjk(cp):
    return (0x1100 <= cp <= 0x11FF or 0x2E80 <= cp <= 0xA4CF or 0xA840 <= cp <= 0xD7AF or 0xF900 <= cp <= 0xFAFF
            or 0xFE30 <= cp <= 0xFE4F or 0xFF65 <= cp <= 0xFFDC or 0x20000 <= cp <= 0x2FFFF)


def _is_punct(cp):
    return (0x21 <= cp <= 0x2F or 0x3A <= cp <= 0x40 or 0x5B <= cp <= 0x60 or 0x7B <= cp <= 0x7E
            or 0x3000 <= cp <= 0x303F or 0xFF01 <= cp <= 0xFF0F or 0xFF1A <= cp <= 0xFF20 or 0xFF3B <= cp <= 0xFF40
            or 0xFF5B <= cp <= 0xFF65)


def tokens_to_text(toks, table):
    """sherpa-onnx Convert(): join symbols (leading \u2581 -> space, <unk> dropped),
    RemoveSpaceBetweenCjk, then RemoveLeadingSpaces."""
    s = ""
    for t in toks:
        sym = table[t]
        if sym == "<unk>":
            continue
        if len(sym.encode()) >= 3 and sym.startswith("\u2581"):
            sym = " " + sym[1:]
        s += sym
    if len(s) >= 2:
        out = [s[0]]
        for i in range(1, len(s)):
            if s[i] == " " and i + 1 < len(s) and (
                    (_is_cjk(ord(s[i - 1])) and _is_cjk(ord(s[i + 1]))) or _is_punct(ord(s[i + 1]))):
                continue
            out.append(s[i])
        s = "".join(out)
    return s.lstrip(" ")


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str], max_new_tokens: int = 0) -> Dict[str, np.ndarray]:
    chunk_ms = int(os.environ.get("XASR_CHUNK_MS", "480"))
    hp, enc, embed, dec, joi, chunk, left = build(Path(model_dir), chunk_ms)
    tok_path = Path(model_dir) / f"chunk-{chunk_ms}ms-model" / "tokens.txt"
    table = [ln.rstrip("\n").rsplit(" ", 1)[0] for ln in open(tok_path, encoding="utf-8") if ln.strip()]
    run.unk_id = table.index("<unk>") if "<unk>" in table else -1
    T = 2 * chunk + 13
    tail = int(os.environ.get("XASR_TAIL_PAD_MS", str(T * 10 + 1000)))
    feats = fbank(audio, tail)
    caps: Dict[str, list] = {}
    toks, first = run(enc, embed, dec, joi, feats, chunk, left, caps)
    out: Dict[str, np.ndarray] = {"raw_audio": audio.astype(np.float32), "fbank": feats}
    for k, v in caps.items():
        out[k] = np.concatenate(v, axis=0).astype(np.float32)
    out["first_logits"] = first.astype(np.float32)
    out["tokens"] = np.asarray(toks, dtype=np.float32)
    out["text"] = tokens_to_text(toks, table)
    out["chunk_ms"] = str(chunk_ms)
    out["tail_pad_ms"] = str(tail)
    return {k: v for k, v in out.items() if k in stages or k in ("chunk_ms", "tail_pad_ms") or k.startswith("stack_")}
