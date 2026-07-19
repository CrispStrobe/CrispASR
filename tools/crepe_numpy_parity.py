"""Executable spec for the CREPE ggml graph — validates it against torchcrepe.

Usage: python tools/crepe_numpy_parity.py <crepe.gguf> <tiny|full>

Reimplements the forward pass in numpy EXACTLY as src/crepe.cpp must build it,
reading weights back out of the converted GGUF. Keep the two in lockstep: if
the C++ graph changes, change this first and re-run it.

Two geometry decisions it pins down:
  * layer order is pad -> conv -> RELU -> BN affine -> maxpool. The relu is
    BEFORE the BN, so BN is a standalone per-channel affine, never folded into
    the conv.
  * conv2..6 use a SYMMETRIC p=32 conv + a leading-column crop instead of the
    blueprint's asymmetric (31,32) pad, because Metal rejects an asymmetric
    GGML_OP_PAD. Left-pad 32 shifts each output window one sample early vs
    left-pad 31, so output j is symmetric-output j+1 -> drop column 0.

Validated: cos=1.0 vs torchcrepe on both capacities, max_abs ~2e-5 (tiny) /
~4e-6 (full), which is f16 weight rounding.
"""
import numpy as np, torch, torchcrepe, sys
from gguf import GGUFReader

GGUF, CAP = sys.argv[1], sys.argv[2]
LAYERS = [("conv1", 512, 4, 254, 254)] + [(f"conv{i}", 64, 1, 31, 32) for i in range(2, 7)]

r = GGUFReader(GGUF)
T = {t.name: np.array(t.data, dtype=np.float32) for t in r.tensors}
S = {t.name: tuple(int(x) for x in t.shape) for t in r.tensors}


def get_conv(name):
    # gguf ne = (K, in, out); numpy view is the reverse -> (out, in, K)
    k, ic, oc = S[name + ".weight"]
    return T[name + ".weight"].reshape(oc, ic, k), T[name + ".bias"]


def conv1d(x, w, b, stride, pad_l, pad_r, crop_lead=0):
    """x: (C_in, T). w: (out, in, K). Symmetric-pad + crop, as the graph will."""
    p = max(pad_l, pad_r)
    xp = np.pad(x, ((0, 0), (p, p)))
    oc, ic, k = w.shape
    tout = (xp.shape[1] - k) // stride + 1
    # im2col
    idx = np.arange(k)[None, :] + (np.arange(tout) * stride)[:, None]   # (T,K)
    cols = xp[:, idx]                                    # (C_in, T, K)
    cols = cols.transpose(1, 0, 2).reshape(tout, ic * k)  # (T, C_in*K)
    y = cols @ w.reshape(oc, ic * k).T + b               # (T, out)
    y = y.T                                              # (out, T)
    return y[:, crop_lead:crop_lead + (x.shape[1] if stride == 1 else tout)]


def maxpool2(x):
    c, t = x.shape
    return x[:, : t - t % 2].reshape(c, t // 2, 2).max(axis=2)


def forward_np(frame):
    x = frame[None, :]  # (1, 1024)
    for i, (name, k, stride, pl, pr) in enumerate(LAYERS):
        w, b = get_conv(name)
        # asymmetric (31,32): symmetric p=32 shifts the window one left -> crop 1
        crop = 1 if pl != pr else 0
        x = conv1d(x, w, b, stride, pl, pr, crop_lead=crop)
        # blueprint order: conv -> relu -> BN affine -> maxpool
        x = np.maximum(x, 0.0)
        x = x * T[name + "_BN.scale"][:, None] + T[name + "_BN.offset"][:, None]
        x = maxpool2(x)
    # torch permutes to (batch, T, C) then flattens -> C is the FAST axis
    flat = x.T.reshape(-1)
    cw = T["classifier.weight"].reshape(360, -1)
    return 1.0 / (1.0 + np.exp(-(cw @ flat + T["classifier.bias"])))


rng = np.random.default_rng(0)
frames = rng.standard_normal((4, 1024)).astype(np.float32)
frames = (frames - frames.mean(1, keepdims=True)) / frames.std(1, keepdims=True)

model = torchcrepe.Crepe(CAP).eval()
model.load_state_dict(torch.load(
    __import__("pathlib").Path(torchcrepe.__file__).parent / "assets" / f"{CAP}.pth",
    map_location="cpu", weights_only=True))
with torch.no_grad():
    ref = model(torch.from_numpy(frames)).numpy()

ok = True
for i in range(len(frames)):
    mine = forward_np(frames[i])
    cos = float(mine @ ref[i] / (np.linalg.norm(mine) * np.linalg.norm(ref[i])))
    mx = float(np.abs(mine - ref[i]).max())
    # the thing that actually matters: same pitch bin
    same = int(mine.argmax()) == int(ref[i].argmax())
    print(f"frame{i}: cos={cos:.8f} max_abs={mx:.3e} |mine|={np.linalg.norm(mine):.4f} "
          f"|ref|={np.linalg.norm(ref[i]):.4f} argmax {mine.argmax()} vs {ref[i].argmax()} "
          f"{'OK' if cos > 0.9999 and same else 'FAIL'}")
    ok &= cos > 0.9999 and same
print("PASS" if ok else "FAIL")
