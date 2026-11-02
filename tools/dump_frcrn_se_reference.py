#!/usr/bin/env python3
"""Dump named stage tensors of the original ClearerVoice-Studio FRCRN_SE_16K.

The reference is the upstream PyTorch model (modelscope/ClearerVoice-Studio,
clearvoice/models/frcrn_se) driven exactly like upstream's
``decode_one_audio_frcrn_se_16k`` (with the DataReader's ``audio_norm`` input
normalisation and the final ``* scalar`` renormalisation that
``networks.SpeechModel.process`` applies for FRCRN_SE_16K).

Stage tensors come from the FIRST ``model.inference`` call (the whole padded
input when the clip is not segmented; the first 1 s window otherwise) and are
written to one ``.npz``.  Every UNet stage is stored in the layout the native
port uses, numpy shape ``(2, C, F, T)`` (re/im, channel, frequency, frame), so
``tools/frcrn_se_parity.py`` can compare without knowing PyTorch's layout.

Stage names (prefix ``u1.`` for the first UNet, ``u2.`` for the second):
  input             normalised + padded model input           (L,)
  stft              ConvSTFT output                           (T, 2*(fft/2+1))
  uN.enc{i}         encoder i (conv + BN + LeakyReLU)         (2, 128, F_i, T+i+1)
  uN.se_enc{i}      SE output kept for the skip connection
  uN.fsmn_enc{i}    frequency FSMN before encoder i (i >= 1)
  uN.fsmn           bottleneck complex FSMN (time-causal)
  uN.dec{i}         decoder i (transposed conv + BN + LeakyReLU)
  uN.fsmn_dec{i}    frequency FSMN after decoder i (i <= 5)
  uN.se_dec{i}      SE after decoder i (i <= 4)
  uN.out            final 1x1 complex conv (UNet output)      (2, 1, F, T)
  mask              tanh(u2.out) + tanh(u1.out)                (2, 1, F, T)
  est_spec          masked complex spectrum                    (T, 2*(fft/2+1))
  wav               ConviSTFT output of this inference call    (n,)
  output            final decode output (all segments, * scalar, truncated)
  scalar            audio_norm de-normalisation factor          ()

Usage (the upstream tree is untrusted data: it is only imported from, never
executed as a script; run with ``python -I``):

  python -I tools/dump_frcrn_se_reference.py \
      --upstream /path/to/ClearerVoice-Studio/clearvoice/clearvoice/models \
      --checkpoint /path/to/FRCRN_SE_16K/last_best_checkpoint.pt \
      --audio noisy.wav --out ref.npz [--seconds 3] [--force-segment]
"""

import argparse
import sys
import types

import numpy as np

EPS = 1e-6  # clearvoice/dataloader/dataloader.py


def audio_norm(x):
    """Verbatim port of clearvoice.dataloader.dataloader.audio_norm (float64)."""
    rms = (x**2).mean() ** 0.5
    scalar = 10 ** (-25 / 20) / (rms + EPS)
    x = x * scalar
    pow_x = x**2
    avg_pow_x = pow_x.mean()
    rmsx = pow_x[pow_x > avg_pow_x].mean() ** 0.5
    scalarx = 10 ** (-25 / 20) / (rmsx + EPS)
    x = x * scalarx
    return x, 1 / (scalar * scalarx + EPS)


def load_model(upstream_models_dir, checkpoint):
    import torch

    sys.path.insert(0, upstream_models_dir)
    from frcrn_se.frcrn import DCCRN  # noqa: E402  (upstream package)

    model = DCCRN(
        complex=True,
        model_complexity=45,
        model_depth=14,
        log_amp=False,
        padding_mode="zeros",
        win_len=640,
        win_inc=320,
        fft_len=640,
        win_type="hanning",
    )
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    # networks.SpeechModel._load_model: copy every key whose shape matches,
    # tolerating a "module." prefix.
    state = model.state_dict()
    missing = []
    for key in state.keys():
        for cand in (key, key.replace("module.", ""), "module." + key):
            if cand in sd and state[key].shape == sd[cand].shape:
                state[key] = sd[cand]
                break
        else:
            missing.append(key)
    if missing:
        raise SystemExit(f"checkpoint misses {len(missing)} keys, e.g. {missing[:3]}")
    model.load_state_dict(state)
    model.eval()
    return model


def pad_like_upstream(inputs, window, stride):
    """Padding rule of decode_one_audio_frcrn_se_16k (kept bug-for-bug)."""
    t = inputs.shape[1]
    if t < window:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - t))], axis=1)
    elif t < window + stride:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window + stride - t))], axis=1)
    elif (t - window) % stride != 0:
        padding = t - (t - window) // stride * stride
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], axis=1)
    return inputs


def decode(model, x, sampling_rate=16000, decode_window=1, one_time_decode_length=120, force_segment=False,
           on_first=None):
    """decode_one_audio_frcrn_se_16k, with an optional first-call callback."""
    import torch

    window = int(sampling_rate * decode_window)
    stride = int(window * 0.75)
    t = x.shape[1]
    do_segment = force_segment or t > sampling_rate * one_time_decode_length
    x = pad_like_upstream(x, window, stride)
    inputs = torch.from_numpy(np.float32(x))
    t = inputs.shape[1]
    first = [True]

    def infer(seg):
        out = model.inference(seg)
        if first[0] and on_first is not None:
            on_first(seg)
        first[0] = False
        return out.detach().cpu().numpy()

    with torch.no_grad():
        if do_segment:
            outputs = np.zeros(t)
            give_up = (window - stride) // 2
            idx = 0
            while idx + window <= t:
                o = infer(inputs[:, idx : idx + window])
                if idx == 0:
                    outputs[idx : idx + window - give_up] = o[:-give_up]
                else:
                    outputs[idx + give_up : idx + window - give_up] = o[give_up:-give_up]
                idx += stride
        else:
            outputs = infer(inputs)
    return outputs


def to_port_layout(t):
    """PyTorch [1, C, F, T, 2] -> numpy (2, C, F, T)."""
    a = t.detach().cpu().numpy()[0]
    return np.ascontiguousarray(np.moveaxis(a, -1, 0)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, help="ClearerVoice-Studio clearvoice/clearvoice/models directory")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--audio", required=True, help="mono 16 kHz audio (soundfile-readable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=0, help="truncate the input to this many seconds")
    ap.add_argument("--offset", type=float, default=0, help="start offset in seconds")
    ap.add_argument("--force-segment", action="store_true",
                    help="use the >120 s segmented path regardless of length (for short parity clips)")
    ap.add_argument("--no-stages", action="store_true", help="only dump input/output")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    import torch

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    audio, sr = sf.read(args.audio, dtype="float64", always_2d=True)
    if sr != 16000:
        raise SystemExit(f"expected 16 kHz audio, got {sr}")
    audio = audio[:, 0]
    o = int(args.offset * sr)
    audio = audio[o:]
    if args.seconds > 0:
        audio = audio[: int(args.seconds * sr)]
    # Upstream reads 16-bit PCM through pydub and divides by 32768; a float
    # WAV read by soundfile at float64 is the same values for 16-bit input.
    x, scalar = audio_norm(audio)
    x = x.astype(np.float32).reshape(1, -1)

    model = load_model(args.upstream, args.checkpoint)
    stages = {}
    hooks = []

    if not args.no_stages:
        def hook(name, conv=to_port_layout):
            def fn(_m, _i, out):
                if name not in stages:
                    stages[name] = conv(out)
            return fn

        for tag, unet in (("u1", model.unet), ("u2", model.unet2)):
            for i in range(unet.model_length):
                hooks.append(unet.encoders[i].register_forward_hook(hook(f"{tag}.enc{i}")))
                hooks.append(unet.se_layers_enc[i].register_forward_hook(hook(f"{tag}.se_enc{i}")))
                if i > 0:
                    hooks.append(unet.fsmn_enc[i].register_forward_hook(hook(f"{tag}.fsmn_enc{i}")))
                hooks.append(unet.decoders[i].register_forward_hook(hook(f"{tag}.dec{i}")))
                if i < unet.model_length - 1:
                    hooks.append(unet.fsmn_dec[i].register_forward_hook(hook(f"{tag}.fsmn_dec{i}")))
                if i < unet.model_length - 2:
                    hooks.append(unet.se_layers_dec[i].register_forward_hook(hook(f"{tag}.se_dec{i}")))
            hooks.append(unet.fsmn.register_forward_hook(hook(f"{tag}.fsmn")))
            hooks.append(unet.linear.register_forward_hook(hook(f"{tag}.out")))

        def stft_hook(_m, _i, out):
            if "stft" not in stages:
                stages["stft"] = out.detach().cpu().numpy()[0].T.astype(np.float32)

        def istft_hook(_m, inp, out):
            if "wav" not in stages:
                stages["est_spec"] = inp[0].detach().cpu().numpy()[0].T.astype(np.float32)
                stages["wav"] = out.detach().cpu().numpy().reshape(-1).astype(np.float32)

        hooks.append(model.stft.register_forward_hook(stft_hook))
        hooks.append(model.istft.register_forward_hook(istft_hook))

        # apply_mask receives the combined mask; capture it once.
        orig_apply = model.apply_mask

        def apply_mask(cmp_spec, cmp_mask):
            if "mask" not in stages:
                stages["mask"] = to_port_layout(cmp_mask)
            return orig_apply(cmp_spec, cmp_mask)

        model.apply_mask = apply_mask

    first_input = {}

    def on_first(seg):
        first_input["x"] = seg.numpy().reshape(-1).astype(np.float32)

    out = decode(model, x, force_segment=args.force_segment, on_first=on_first)
    out = out.reshape(-1)[: x.shape[1]] * scalar
    for h in hooks:
        h.remove()
    stages["input"] = first_input["x"]
    stages["output"] = out.astype(np.float32)
    stages["scalar"] = np.array(scalar, dtype=np.float64)
    stages["audio"] = audio.astype(np.float32)
    np.savez(args.out, **stages)
    print(f"wrote {len(stages)} stages to {args.out} ({len(audio)} samples, scalar={scalar:.6f})")


if __name__ == "__main__":
    main()
