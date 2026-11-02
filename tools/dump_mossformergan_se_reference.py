#!/usr/bin/env python3
"""Dump PyTorch reference stages for ClearerVoice-Studio MossFormerGAN_SE_16K.

Runs the ORIGINAL upstream generator (modelscope/ClearerVoice-Studio,
``clearvoice/models/mossformer_gan_se``) and the ORIGINAL upstream decode
wrapper (``decode_one_audio_mossformergan_se_16k`` from
``clearvoice/utils/decode.py`` plus the STFT/power-compression helpers from
``clearvoice/utils/misc.py``). The decode functions are lifted verbatim from
the upstream source with ``ast`` so their heavy module-level imports
(torchaudio, joblib, the dataloader) are not needed.

Written ``.npy`` files (float32, torch layout, batch dimension dropped):

    input_wav            [N]          the audio fed to the decoder
    wave                 [N]          decode_one_audio_mossformergan_se_16k()
    -- network stages of ONE unsegmented forward on the first
    -- min(N, 10 s) samples (the same window the decoder uses first):
    stage_input_spec     [2, T, 201]  power-compressed STFT (model input)
    stage_enc            [64, T, 101] dense encoder output
    stage_enc_conv1      [64, T, 201] dense encoder conv_1 (1x1+IN+PReLU)
    stage_enc_dense      [64, T, 201] dense encoder dilated dense net
    stage_block{i}       [64, T, 101] SyncANet block i output (i = 0..5)
    stage_b0_*           block-0 internals (intra/inter branches)
    stage_mask           [T, 201]     mask decoder output
    stage_complex        [2, T, 201]  complex decoder output
    stage_pred           [2, T, 201]  final real/imag (compressed domain)
    stage_wave           [n]          waveform of that single forward

Usage (run from outside the upstream checkout; it is only imported):

    python -I tools/dump_mossformergan_se_reference.py \
        --cvs-root /path/to/ClearerVoice-Studio \
        --checkpoint /path/to/MossFormerGAN_SE_16K/last_best_checkpoint.pt \
        --wav noisy.wav --out-dir ref/

Requires torch, numpy, soundfile, einops, rotary-embedding-torch==0.8.3
(the version pinned by upstream requirements.txt).
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
import types
from pathlib import Path

import numpy as np


def lift_functions(src_path: Path, names: list[str], namespace: dict) -> None:
    """Exec only the named top-level functions of an upstream module."""
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in wanted}
    if missing:
        raise SystemExit(f"{src_path}: missing functions {sorted(missing)}")
    module = ast.Module(body=wanted, type_ignores=[])
    exec(compile(module, str(src_path), "exec"), namespace)


def load_upstream(cvs_root: Path):
    import torch

    pkg_root = cvs_root / "clearvoice" / "clearvoice"
    models_dir = pkg_root / "models"
    if not (models_dir / "mossformer_gan_se" / "generator.py").is_file():
        raise SystemExit(f"{models_dir}/mossformer_gan_se/generator.py not found")
    sys.path.insert(0, str(models_dir))
    from mossformer_gan_se.generator import MossFormerGAN_SE_16K  # noqa: E402

    ns: dict = {"torch": torch, "np": np}
    lift_functions(pkg_root / "utils" / "misc.py", ["power_compress", "power_uncompress", "stft", "istft"], ns)
    lift_functions(
        pkg_root / "utils" / "decode.py",
        ["decode_one_audio_mossformergan_se_16k", "_decode_one_audio_mossformergan_se_16k"],
        ns,
    )
    return MossFormerGAN_SE_16K, ns


def make_args():
    # clearvoice/config/inference/MossFormerGAN_SE_16K.yaml
    return types.SimpleNamespace(
        mode="inference",
        sampling_rate=16000,
        one_time_decode_length=10,
        decode_window=10,
        win_type="hamming",
        win_len=400,
        win_inc=100,
        fft_len=400,
    )


def build_model(cls, checkpoint: Path, args):
    import torch

    model = cls(args).model  # networks.py: CLS_MossFormerGAN_SE_16K uses .model (SyncANet)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    # Mirror networks.py _load_model: copy matching keys (optionally 'module.'-prefixed).
    state = model.state_dict()
    missing = []
    for key in state:
        for cand in (key, key.replace("module.", ""), "module." + key):
            if cand in sd and sd[cand].shape == state[key].shape:
                state[key] = sd[cand]
                break
        else:
            missing.append(key)
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:3]}")
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cvs-root", type=Path, required=True, help="ClearerVoice-Studio checkout (read-only)")
    ap.add_argument("--checkpoint", type=Path, required=True, help="MossFormerGAN_SE_16K last_best_checkpoint.pt")
    ap.add_argument("--wav", type=Path, required=True, help="mono 16 kHz input")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="crop the input (0 = whole file)")
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = default)")
    ap.add_argument("--no-stages", action="store_true", help="only dump input_wav and wave")
    ap.add_argument("--time", action="store_true", help="print decode wall time")
    args_cli = ap.parse_args()

    import soundfile as sf
    import torch

    if args_cli.threads > 0:
        torch.set_num_threads(args_cli.threads)
    cls, ns = load_upstream(args_cli.cvs_root)
    args = make_args()
    model = build_model(cls, args_cli.checkpoint, args)

    audio, sr = sf.read(str(args_cli.wav), dtype="float32", always_2d=True)
    if sr != 16000:
        raise SystemExit(f"{args_cli.wav}: expected 16 kHz, got {sr}")
    audio = audio[:, 0]
    if args_cli.max_seconds > 0:
        audio = audio[: int(args_cli.max_seconds * sr)]
    out = args_cli.out_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "input_wav.npy", audio.astype(np.float32))

    import time

    t0 = time.perf_counter()
    with torch.no_grad():
        wave = ns["decode_one_audio_mossformergan_se_16k"](model, torch.device("cpu"), audio[None, :], args)
    dt = time.perf_counter() - t0
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    np.save(out / "wave.npy", wave)
    print(f"wave: {wave.shape[0]} samples ({audio.shape[0]} in), decode {dt:.2f} s")
    if args_cli.time:
        print(f"TIME_DECODE {dt:.3f}")

    if args_cli.no_stages:
        return 0

    # One unsegmented forward on the first window, with hooks.
    n = min(audio.shape[0], args.sampling_rate * args.decode_window)
    seg = torch.from_numpy(audio[None, :n].copy())
    norm_factor = torch.sqrt(seg.size(-1) / torch.sum(seg**2.0, dim=-1))
    stages: dict[str, np.ndarray] = {}

    def save(name, t):
        a = t.detach().float().cpu().numpy()
        stages[name] = a[0] if a.shape[0] == 1 else a  # drop batch; keep folded B*T / B*Q rows

    hooks = []

    def hook(name):
        def fn(_m, _inp, outp):
            save(name, outp[0] if isinstance(outp, (list, tuple)) else outp)

        return fn

    def hook_in(name):
        def fn(_m, inp):
            save(name, inp[0])

        return fn

    enc = model.dense_encoder
    hooks.append(model.register_forward_pre_hook(hook_in("stage_input_spec")))
    hooks.append(enc.conv_1.register_forward_hook(hook("stage_enc_conv1")))
    hooks.append(enc.dilated_dense.register_forward_hook(hook("stage_enc_dense")))
    hooks.append(enc.register_forward_hook(hook("stage_enc")))
    for i, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(hook(f"stage_block{i}")))
    b0 = model.blocks[0]
    for name in ("intra_norm", "intra_to_u", "intra_to_v", "intra_rnn", "intra_linear", "intra_mossformer",
                 "intra_se", "inter_norm", "inter_to_u", "inter_rnn", "inter_linear", "inter_mossformer",
                 "inter_se", "attn_concat_proj"):
        hooks.append(getattr(b0, name).register_forward_hook(hook(f"stage_b0_{name}")))
    hooks.append(b0.intra_mossformer.to_hidden.register_forward_hook(hook("stage_b0_intra_mf_hidden")))
    hooks.append(b0.intra_mossformer.to_qk.register_forward_hook(hook("stage_b0_intra_mf_qk")))
    hooks.append(model.mask_decoder.register_forward_hook(hook("stage_mask")))
    hooks.append(model.mask_decoder.dense_block.register_forward_hook(hook("stage_mask_dense")))
    hooks.append(model.complex_decoder.register_forward_hook(hook("stage_complex")))

    with torch.no_grad():
        sw = ns["_decode_one_audio_mossformergan_se_16k"](model, torch.device("cpu"), seg, norm_factor, args)
        # Re-run the model once more to capture its (real, imag) output pair.
        spec = torch.from_numpy(stages["stage_input_spec"][None])
        real, imag = model(spec)
    for h in hooks:
        h.remove()
    stages["stage_pred"] = torch.cat([real, imag], dim=1)[0].numpy()
    stages["stage_mask"] = stages["stage_mask"][0]  # [1, T, F] -> [T, F]
    stages["stage_wave"] = np.asarray(sw, dtype=np.float32).reshape(-1)
    for k, v in stages.items():
        np.save(out / f"{k}.npy", np.ascontiguousarray(v, dtype=np.float32))
        print(f"{k:28s} {tuple(v.shape)}")
    T = stages["stage_input_spec"].shape[1]
    assert T == math.ceil(n / args.win_inc) + 1, (T, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
