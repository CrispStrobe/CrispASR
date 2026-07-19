"""CREPE reference backend — torchcrepe ground truth for crispasr-diff.

CREPE (Kim et al. 2018, MIT) is a monophonic F0 estimator: a 6-layer 1-D CNN
over raw 16 kHz audio producing a 360-bin pitch activation per frame. There is
no STFT, no attention and no autoregression, so this dumper is a plain forward
pass — no `_hooks.py` machinery needed.

Usage:

    python tools/dump_reference.py --backend crepe \\
        --model-dir tiny \\
        --audio samples/jfk.wav \\
        --output /tmp/crepe-ref.gguf

    build/bin/crispasr-diff crepe crepe-tiny-f16.gguf \\
        /tmp/crepe-ref.gguf samples/jfk.wav

`--model-dir` is the *capacity*, not a directory: pass `tiny` or `full` (or a
path to a `<capacity>.pth`). The weights ship inside the torchcrepe package at
`<torchcrepe>/assets/{tiny,full}.pth`.

Geometry (see docs/music-transcription/PLAN.md for the full trace):

    input   1024-sample frame @ 16 kHz, per-frame mean-centred and divided by
            the UNBIASED (n-1) std, hop 10 ms, zero-padded WINDOW_SIZE//2 at
            both edges (torchcrepe `pad=True`)
    body    6 x [F.pad -> conv -> F.relu -> batch_norm -> max_pool2d(2)]
    head    permute to (T, C) — C is the FAST axis — flatten, Linear -> 360,
            sigmoid

    layer | K   | stride | pad (l, r) | out ch (full / tiny) | T
    ------+-----+--------+------------+----------------------+-----------
    conv1 | 512 |   4    |  254, 254  |  1024 / 128          | 1024->256->128
    conv2 |  64 |   1    |   31,  32  |   128 /  16          | 128 -> 64
    conv3 |  64 |   1    |   31,  32  |   128 /  16          |  64 -> 32
    conv4 |  64 |   1    |   31,  32  |   128 /  16          |  32 -> 16
    conv5 |  64 |   1    |   31,  32  |   256 /  32          |  16 ->  8
    conv6 |  64 |   1    |   31,  32  |   512 /  64          |   8 ->  4

Two traps this file deliberately does NOT step in:

  * **The ReLU is BEFORE the BatchNorm.** Do not reorder, and do not fold BN
    into the conv — the fold is invalid through a rectifier. (The first
    CrispASR converter got this wrong from a fetched *summary* of
    `torchcrepe/model.py`; the real source has relu first.)
  * **`torchcrepe.convert.bins_to_cents` applies triangular DITHERING**, i.e.
    it is random. Nothing decoded through it is reproducible, so this dumper
    emits the RAW 360-bin activation only. Never dump decoded cents/Hz here —
    the fixture would never reproduce.

Stages
------
    activation      (n_frames, 360)   sigmoid pitch activation  <- THE gate
    frames          (n_frames, 1024)  normalized model input
    conv1_out ..
    conv6_out       (F, C, T)         per-layer output, after the maxpool
    embedding       (F, in_features)  channel-fastest flatten fed to classifier

`activation` is the only stage `crispasr-diff crepe` can compare today —
`src/crepe.h` exposes `crepe_compute_activation()` and no per-layer stage API,
so the per-layer captures are diagnostic: run this dumper twice (once per
capacity, or against a modified graph) and diff the `.gguf` archives, or use
them from Python. They are capped to the first `CREPE_REF_LAYER_FRAMES` frames
(default 4) because `conv1_out` for `full` is 1024 x 128 floats per frame.

Env knobs
---------
    CREPE_CAPACITY            override the capacity when --model-dir is unclear
    CREPE_REF_MAX_FRAMES      cap the frame count (0 = all; default 0)
    CREPE_REF_LAYER_FRAMES    frames captured for the per-layer stages (4)
    CREPE_REF_BATCH           forward batch size (128)
"""

import os
import pathlib

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SIZE = 1024
PITCH_BINS = 360

DEFAULT_STAGES = [
    "activation",
    "frames",
    "conv1_out", "conv2_out", "conv3_out",
    "conv4_out", "conv5_out", "conv6_out",
    "embedding",
]


def _capacity(model_dir) -> str:
    """Resolve 'tiny' / 'full' from --model-dir (a capacity, not a directory)."""
    env = os.environ.get("CREPE_CAPACITY", "").strip().lower()
    if env in ("tiny", "full"):
        return env
    name = pathlib.Path(str(model_dir)).name.lower()
    if "tiny" in name:
        return "tiny"
    if "full" in name:
        return "full"
    raise SystemExit(
        f"crepe ref: cannot tell capacity from --model-dir '{model_dir}'. "
        f"Pass --model-dir tiny (or full), or set CREPE_CAPACITY.")


def _frame(audio: np.ndarray, hop: int) -> np.ndarray:
    """torchcrepe.preprocess with pad=True, as numpy.

    Returns (n_frames, 1024) float32, mean-centred and scaled by the UNBIASED
    (n-1) std — torch.std's default correction=1, which src/crepe.cpp mirrors.
    """
    n_frames = 1 + int(len(audio) // hop)
    padded = np.pad(audio.astype(np.float32), (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    idx = np.arange(WINDOW_SIZE)[None, :] + (np.arange(n_frames) * hop)[:, None]
    # The tail frame can reach past the padded buffer for short inputs; clamp
    # by zero-extending rather than truncating, so the frame count matches
    # crepe_n_frames() exactly.
    if idx.max() >= len(padded):
        padded = np.pad(padded, (0, int(idx.max()) + 1 - len(padded)))
    frames = padded[idx].astype(np.float32)
    frames -= frames.mean(axis=1, keepdims=True)
    std = frames.std(axis=1, ddof=1, keepdims=True)
    frames /= np.maximum(std, 1e-10)
    return frames


def dump(model_dir, audio, stages, max_new_tokens=None, **kwargs):
    """Run torchcrepe forward and return per-stage intermediates.

    Args:
        model_dir: the capacity — "tiny" or "full" (or a path to a .pth).
        audio:     float32 numpy array, 16 kHz mono PCM (from the harness).
        stages:    set of stage names to capture.
        max_new_tokens: unused (no autoregressive decoding).

    Returns:
        dict of {stage_name: numpy_array}
    """
    import torch
    import torch.nn.functional as F
    import torchcrepe

    capacity = _capacity(model_dir)
    hop = SAMPLE_RATE // 100  # 10 ms, CREPE's reference hop
    max_frames = int(os.environ.get("CREPE_REF_MAX_FRAMES", "0"))
    layer_frames = int(os.environ.get("CREPE_REF_LAYER_FRAMES", "4"))
    batch = int(os.environ.get("CREPE_REF_BATCH", "128"))

    model = torchcrepe.Crepe(capacity).eval()
    weights = pathlib.Path(str(model_dir))
    if weights.suffix != ".pth" or not weights.is_file():
        weights = pathlib.Path(torchcrepe.__file__).parent / "assets" / f"{capacity}.pth"
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    print(f"crepe ref: capacity={capacity} weights={weights} "
          f"in_features={model.in_features}")

    frames = _frame(np.asarray(audio, dtype=np.float32), hop)
    if max_frames > 0:
        frames = frames[:max_frames]
    n_frames = len(frames)
    print(f"crepe ref: {n_frames} frames (hop {hop} samples = 10 ms, pad=True)")

    captures = {}
    if "frames" in stages:
        captures["frames"] = frames.copy()

    convs = [(model.conv1, model.conv1_BN, (0, 0, 254, 254)),
             (model.conv2, model.conv2_BN, (0, 0, 31, 32)),
             (model.conv3, model.conv3_BN, (0, 0, 31, 32)),
             (model.conv4, model.conv4_BN, (0, 0, 31, 32)),
             (model.conv5, model.conv5_BN, (0, 0, 31, 32)),
             (model.conv6, model.conv6_BN, (0, 0, 31, 32))]

    def forward(x, capture_layers):
        """model.embed + model.layer(conv6) + classifier, with taps.

        Mirrors torchcrepe.Crepe.layer() exactly: pad -> conv -> RELU ->
        batch_norm -> max_pool. The relu is before the BN; do not reorder.
        """
        out = {}
        h = x[:, None, :, None]  # (B, 1, 1024, 1)
        for i, (conv, bn, pad) in enumerate(convs):
            h = F.pad(h, pad)
            h = conv(h)
            h = F.relu(h)
            h = bn(h)
            h = F.max_pool2d(h, (2, 1), (2, 1))
            if capture_layers:
                # (B, C, T, 1) -> (B, C, T)
                out[f"conv{i + 1}_out"] = h[:, :, :, 0].numpy().astype(np.float32)
        emb = h.permute(0, 2, 1, 3).reshape(-1, model.in_features)
        if capture_layers:
            out["embedding"] = emb.numpy().astype(np.float32)
        out["activation"] = torch.sigmoid(model.classifier(emb)).numpy().astype(np.float32)
        return out

    activation = np.empty((n_frames, PITCH_BINS), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_frames, batch):
            chunk = torch.from_numpy(frames[start:start + batch])
            out = forward(chunk, capture_layers=False)
            activation[start:start + len(chunk)] = out["activation"]

        # Per-layer diagnostics on a small head slice — conv1_out for `full`
        # is 1024 x 128 floats per frame, so capturing every frame would be
        # hundreds of MB for a 10 s clip.
        want_layers = [s for s in stages if s.startswith("conv") or s == "embedding"]
        if want_layers and layer_frames > 0:
            head = torch.from_numpy(frames[:min(layer_frames, n_frames)])
            for name, arr in forward(head, capture_layers=True).items():
                if name in stages:
                    captures[name] = arr

    if "activation" in stages:
        captures["activation"] = activation

    # Sanity print: the argmax bin of the most confident frame, undecoded.
    # (No Hz here — torchcrepe.convert.bins_to_cents dithers, so any decoded
    # value is non-deterministic and would make this fixture irreproducible.)
    peak = int(activation.max(axis=1).argmax())
    print(f"crepe ref: captured {len(captures)} stages; peak frame {peak} "
          f"bin={int(activation[peak].argmax())} conf={activation[peak].max():.4f}")
    return captures
