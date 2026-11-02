#!/usr/bin/env python3
"""Model-free tests for models/convert-frcrn-se-to-gguf.py.

Checks the exact rewrites the native frcrn-se runtime depends on, against
independent numpy references (no torch, no checkpoint):

* ConvSTFT / ConviSTFT kernels: forward kernel == windowed rfft of a frame,
  inverse kernel + window-power OLA reconstructs the signal.
* Polyphase rewrite of ConvTranspose2d(stride (2,1), padding (0,1)) equals a
  direct transposed convolution for every decoder kernel height.
* Conv-bias + eval-BatchNorm folding equals the unfused computation.
* A synthetic full-size state dict converts to the expected tensor set and a
  GGUF that carries the runtime's metadata (when the gguf package exists).
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("frcrn_converter", ROOT / "models" / "convert-frcrn-se-to-gguf.py")
CONV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONV)

try:
    import scipy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False

try:
    import gguf  # noqa: F401

    HAVE_GGUF = True
except ImportError:  # pragma: no cover
    HAVE_GGUF = False


def conv_transpose2d_ref(x, w, stride=(2, 1), padding=(0, 1)):
    """Naive ConvTranspose2d. x (Cin, H, W), w (Cin, Cout, kh, kw)."""
    cin, h, wd = x.shape
    _, cout, kh, kw = w.shape
    full = np.zeros((cout, (h - 1) * stride[0] + kh, (wd - 1) * stride[1] + kw))
    for i in range(h):
        for j in range(wd):
            full[:, i * stride[0] : i * stride[0] + kh, j * stride[1] : j * stride[1] + kw] += np.einsum(
                "c,cokl->okl", x[:, i, j], w
            )
    ph, pw = padding
    return full[:, ph : full.shape[1] - ph, pw : full.shape[2] - pw]


def conv2d_ref(x, w, pad_h):
    """Naive stride-1 Conv2d. x (Cin, H, W), w (Cout, Cin, kh, kw)."""
    x = np.pad(x, ((0, 0), (pad_h, pad_h), (0, 0)))
    cout, cin, kh, kw = w.shape
    h, wd = x.shape[1] - kh + 1, x.shape[2] - kw + 1
    y = np.zeros((cout, h, wd))
    for a in range(kh):
        for b in range(kw):
            y += np.einsum("oc,chw->ohw", w[:, :, a, b], x[:, a : a + h, b : b + wd])
    return y


def synthetic_state_dict(rng):
    sd = {}

    def put(name, shape):
        sd[name] = rng.standard_normal(shape).astype(np.float32) * 0.1

    def bn(prefix, c):
        for part in ("re", "im"):
            put(f"{prefix}.bn_{part}.weight", (c,))
            put(f"{prefix}.bn_{part}.bias", (c,))
            put(f"{prefix}.bn_{part}.running_mean", (c,))
            sd[f"{prefix}.bn_{part}.running_var"] = rng.uniform(0.5, 2.0, c).astype(np.float32)

    def fsmn(prefix):
        put(f"{prefix}.linear.weight", (128, 128))
        put(f"{prefix}.linear.bias", (128,))
        put(f"{prefix}.project.weight", (128, 128))
        put(f"{prefix}.conv1.weight", (128, 1, 20, 1))

    def se(prefix):
        for tag in ("fc_r", "fc_i"):
            put(f"{prefix}.{tag}.0.weight", (16, 128))
            put(f"{prefix}.{tag}.0.bias", (16,))
            put(f"{prefix}.{tag}.2.weight", (128, 16))
            put(f"{prefix}.{tag}.2.bias", (128,))

    for unet in ("unet", "unet2"):
        for i in range(7):
            cin = 1 if i == 0 else 128
            for part in ("re", "im"):
                put(f"{unet}.encoders.{i}.conv.conv_{part}.weight", (128, cin, CONV.ENC_KF[i], 2))
                put(f"{unet}.encoders.{i}.conv.conv_{part}.bias", (128,))
            bn(f"{unet}.encoders.{i}.bn", 128)
            se(f"{unet}.se_layers_enc.{i}")
            dcin = 128 if i == 0 else 256
            dcout = 1 if i == 6 else 128
            for part in ("re", "im"):
                put(f"{unet}.decoders.{i}.transconv.tconv_{part}.weight", (dcin, dcout, CONV.DEC_KF[i], 2))
                put(f"{unet}.decoders.{i}.transconv.tconv_{part}.bias", (dcout,))
            bn(f"{unet}.decoders.{i}.bn", dcout)
            for part in ("re", "im"):
                fsmn(f"{unet}.fsmn_enc.{i}.fsmn_{part}_L1")
                fsmn(f"{unet}.fsmn_dec.{i}.fsmn_{part}_L1")
            if i < 6:
                se(f"{unet}.se_layers_dec.{i}")
        for unit in ("re_L1", "im_L1", "re_L2", "im_L2"):
            fsmn(f"{unet}.fsmn.fsmn_{unit}")
        for part in ("re", "im"):
            put(f"{unet}.linear.conv_{part}.weight", (1, 1, 1, 1))
            put(f"{unet}.linear.conv_{part}.bias", (1,))
    return sd


class StftKernels(unittest.TestCase):
    @unittest.skipUnless(HAVE_SCIPY, "scipy missing")
    def test_forward_kernel_is_windowed_rfft(self):
        fwd, window = CONV.init_kernels(640, 320, 640, "hanning")
        self.assertEqual(fwd.shape, (642, 640))
        n = np.arange(640)
        np.testing.assert_allclose(window**2, 0.5 - 0.5 * np.cos(2 * np.pi * n / 640), atol=1e-6)
        frame = np.random.default_rng(0).standard_normal(640)
        spec = np.fft.rfft(frame * window)
        got = fwd.astype(np.float64) @ frame
        np.testing.assert_allclose(got[:321], spec.real, atol=1e-4)
        np.testing.assert_allclose(got[321:], spec.imag, atol=1e-4)

    @unittest.skipUnless(HAVE_SCIPY, "scipy missing")
    def test_inverse_overlap_add_round_trip(self):
        fwd, window = CONV.init_kernels(640, 320, 640, "hanning")
        inv, _ = CONV.init_kernels(640, 320, 640, "hanning", invers=True)
        x = np.random.default_rng(1).standard_normal(320 * 20 + 320)
        T = (len(x) - 640) // 320 + 1
        frames = np.stack([x[t * 320 : t * 320 + 640] for t in range(T)], 1)  # (640, T)
        spec = fwd.astype(np.float64) @ frames  # (642, T)
        out_frames = inv.T.astype(np.float64) @ spec  # (640, T) — stft.inv is stored as inv.T
        y = np.zeros(len(x))
        coff = np.zeros(len(x))
        for t in range(T):
            y[t * 320 : t * 320 + 640] += out_frames[:, t]
            coff[t * 320 : t * 320 + 640] += window.astype(np.float64) ** 2
        y = y / (coff + 1e-8)
        # every sample covered by two frames is reconstructed exactly
        np.testing.assert_allclose(y[320:-320], x[320:-320], atol=1e-4)


class Rewrites(unittest.TestCase):
    def test_polyphase_transposed_conv(self):
        rng = np.random.default_rng(2)
        for kf in (2, 5, 6):
            w = rng.standard_normal((4, 3, kf, 2))
            x = rng.standard_normal((4, 7, 9))
            ref = conv_transpose2d_ref(x, w)
            pw = CONV.polyphase_tconv(w)
            q = (kf + 1) // 2
            self.assertEqual(pw.shape, (6, 4, q, 2))
            y = conv2d_ref(x, pw, q - 1)  # (2*Cout, M, W-1)
            m = y.shape[1]
            y = y.reshape(2, 3, m, -1).transpose(1, 2, 0, 3).reshape(3, 2 * m, -1)[:, : 2 * (7 - 1) + kf]
            self.assertEqual(y.shape, ref.shape)
            np.testing.assert_allclose(y, ref, atol=1e-9)

    def test_bias_batchnorm_folding(self):
        rng = np.random.default_rng(3)
        sd = {}
        for part in ("re", "im"):
            sd[f"c.conv_{part}.bias"] = rng.standard_normal(5)
            sd[f"b.bn_{part}.weight"] = rng.standard_normal(5)
            sd[f"b.bn_{part}.bias"] = rng.standard_normal(5)
            sd[f"b.bn_{part}.running_mean"] = rng.standard_normal(5)
            sd[f"b.bn_{part}.running_var"] = rng.uniform(0.1, 2, 5)
        scale, shift = CONV.fold_affine(sd, "c.conv", "b")
        z = rng.standard_normal((2, 5))  # bias-free complex conv output (re, im)
        pre = [z[0] + sd["c.conv_re.bias"] - sd["c.conv_im.bias"], z[1] + sd["c.conv_re.bias"] + sd["c.conv_im.bias"]]
        for p, part in enumerate(("re", "im")):
            ref = (pre[p] - sd[f"b.bn_{part}.running_mean"]) / np.sqrt(sd[f"b.bn_{part}.running_var"] + 1e-5)
            ref = ref * sd[f"b.bn_{part}.weight"] + sd[f"b.bn_{part}.bias"]
            np.testing.assert_allclose(z[p] * scale[p] + shift[p], ref, rtol=1e-5, atol=1e-5)


class FullConversion(unittest.TestCase):
    def test_tensor_set(self):
        sd = synthetic_state_dict(np.random.default_rng(4))
        tensors = {name: (arr, kind) for name, arr, kind in CONV.build_tensors(sd)}
        self.assertEqual(tensors["u1.enc.0.w_re"][0].shape, (128, 10))
        self.assertEqual(tensors["u1.enc.6.w_im"][0].shape, (128, 128 * 2 * 2))
        self.assertEqual(tensors["u2.dec.0.w_re"][0].shape, (256, 128 * 1 * 2))
        self.assertEqual(tensors["u2.dec.4.w_re"][0].shape, (256, 128 * 3 * 2))
        self.assertEqual(tensors["u2.dec.4.w_im_skip"][0].shape, (256, 128 * 3 * 2))
        self.assertEqual(tensors["u2.dec.6.w_re"][0].shape, (2, 128 * 3 * 2))
        self.assertNotIn("u2.dec.0.w_re_skip", tensors)
        # the split halves are the polyphase kernel's input-channel halves
        w = CONV.polyphase_tconv(sd["unet2.decoders.4.transconv.tconv_re.weight"])
        np.testing.assert_array_equal(tensors["u2.dec.4.w_re"][0], w[:, :128].reshape(256, -1))
        np.testing.assert_array_equal(tensors["u2.dec.4.w_re_skip"][0], w[:, 128:].reshape(256, -1))
        self.assertEqual(tensors["u1.dec.6.scale"][0].shape, (2, 1))
        self.assertEqual(tensors["u1.fsmn_enc.3.re.mem"][0].shape, (128, 20))
        self.assertEqual(tensors["u1.fsmn.iml2.proj.w"][1], "big")
        self.assertEqual(tensors["u1.out.w"][0].shape, (4,))
        self.assertNotIn("u1.fsmn_enc.0.re.lin.w", tensors)  # never used upstream
        self.assertNotIn("u1.fsmn_dec.6.re.lin.w", tensors)
        self.assertNotIn("u1.se_dec.5.r1.w", tensors)
        self.assertEqual(sum(1 for n in tensors if n.startswith("u1.")), sum(1 for n in tensors if n.startswith("u2.")))

    def test_alias_mismatch_is_rejected(self):
        sd = synthetic_state_dict(np.random.default_rng(5))
        sd["unet.encoder0.conv.conv_re.weight"] = sd["unet.encoders.0.conv.conv_re.weight"] + 1
        with self.assertRaises(ValueError):
            CONV.build_tensors(sd)

    @unittest.skipUnless(HAVE_GGUF and HAVE_SCIPY, "gguf/scipy missing")
    def test_gguf_metadata_and_dtypes(self):
        from gguf import GGUFReader

        sd = synthetic_state_dict(np.random.default_rng(6))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "frcrn.gguf"
            CONV.write_gguf(sd, path, "f16")
            r = GGUFReader(str(path))
            kv = {f.name: f for f in r.fields.values()}
            arch = kv["general.architecture"]
            self.assertEqual(bytes(arch.parts[arch.data[0]]).decode(), "frcrn-se")
            for key in ("frcrn_se.win_len", "frcrn_se.win_inc", "frcrn_se.n_levels", "frcrn_se.enc_kernel_f",
                        "frcrn_se.dec_kernel_f", "frcrn_se.one_time_decode_length_s"):
                self.assertIn(key, kv)
            types = {t.name: t.tensor_type.name for t in r.tensors}
            self.assertEqual(types["stft.fwd"], "F32")
            self.assertEqual(types["stft.inv"], "F32")
            self.assertEqual(types["u1.enc.1.w_re"], "F16")
            self.assertEqual(types["u1.enc.1.scale"], "F32")
            self.assertEqual(types["u1.fsmn_enc.1.re.mem"], "F32")
            self.assertEqual(types["u1.se_enc.0.r1.w"], "F32")


if __name__ == "__main__":
    unittest.main()
