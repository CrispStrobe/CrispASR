#!/usr/bin/env python3
"""Model-free checks for models/convert-mossformer2-sr-to-gguf.py.

Tiny synthetic state dicts with the upstream MossFormer2_SR_48K key layouts
(Mossformer.mossformer = MossFormer_MaskNet with one speaker, reusing the SE
converter's name map; Generator with weight-normed convolutions and Snake
alphas) exercise the generator name map, the weight-norm fold, the per-tap /
transposed-conv matrix layouts (checked against torch convolutions), the
F16/F32 precision policy and a GGUF round trip.
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("conv_mf2sr", ROOT / "models" / "convert-mossformer2-sr-to-gguf.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)
spec_t = importlib.util.spec_from_file_location("test_mf2se_conv", ROOT / "tests" / "test-mossformer2-se-converter.py")
se_test = importlib.util.module_from_spec(spec_t)
spec_t.loader.exec_module(se_test)

TINY = dict(conv.HPARAMS, num_mels=4, mossformer_sequence_dim=8, num_mossformer_layer=2, query_key_dim=4,
            hidden_dim=16, fsmn_inner_channels=4, fsmn_lorder=3, conv_module_kernel=5, rotary_dim=4,
            upsample_initial_channel=16, gen_conv_kernel=3)
TINY_ARR = {"upsample_rates": [2, 2], "upsample_kernel_sizes": [4, 4], "resblock_kernel_sizes": [3, 5],
            "resblock_dilations": [1, 3]}


def tiny_mossformer(h=TINY):
    se_h = dict(se_test.TINY, num_mels=h["num_mels"], in_channels=h["num_mels"], mask_bins=h["num_mels"],
                num_spks=1, mossformer_sequence_dim=h["mossformer_sequence_dim"],
                num_mossformer_layer=h["num_mossformer_layer"], query_key_dim=h["query_key_dim"],
                hidden_dim=h["hidden_dim"], fsmn_inner_channels=h["fsmn_inner_channels"],
                fsmn_lorder=h["fsmn_lorder"], conv_module_kernel=h["conv_module_kernel"], rotary_dim=h["rotary_dim"])
    return se_test.tiny_state(se_h)


def tiny_generator(h=TINY, arr=TINY_ARR, seed=1):
    rng = np.random.default_rng(seed)
    r = lambda *s: rng.standard_normal(s).astype(np.float32)
    ch, k = h["upsample_initial_channel"], h["gen_conv_kernel"]
    sd = {}

    def wn(prefix, shape):
        sd[prefix + ".weight_v"] = r(*shape)
        sd[prefix + ".weight_g"] = np.abs(r(shape[0], 1, 1)) + 0.5
        sd[prefix + ".bias"] = r(shape[1] if prefix.startswith("ups") else shape[0])

    wn("conv_pre", (ch, h["num_mels"], k))
    n_rb = len(arr["resblock_kernel_sizes"])
    for i, (u, uk) in enumerate(zip(arr["upsample_rates"], arr["upsample_kernel_sizes"])):
        cin, cout = ch >> i, ch >> (i + 1)
        sd[f"snakes.{i}.alpha"] = r(1, cin, 1)
        wn(f"ups.{i}", (cin, cout, uk))
        for j, rk in enumerate(arr["resblock_kernel_sizes"]):
            for d in range(len(arr["resblock_dilations"])):
                for s in ("1", "2"):
                    wn(f"resblocks.{i * n_rb + j}.convs{s}.{d}", (cout, cout, rk))
                    sd[f"resblocks.{i * n_rb + j}.convs{s}_activates.{d}.alpha"] = r(1, cout, 1)
    last = ch >> len(arr["upsample_rates"])
    sd["snake_post.alpha"] = r(1, last, 1)
    wn("conv_post", (1, last, k))
    return sd


class ConverterTest(unittest.TestCase):
    def test_generator_names(self):
        self.assertEqual(conv.gen_base("conv_pre.weight_v"), ("gen.conv_pre", "conv"))
        self.assertEqual(conv.gen_base("resblocks.11.convs2.2.bias"), ("gen.rb.11.c2.2", "conv"))
        self.assertEqual(conv.gen_base("ups.3.weight_g"), ("gen.ups.3", "ups"))
        self.assertEqual(conv.gen_base("snakes.0.alpha"), ("gen.snakes.0.alpha", "alpha"))
        self.assertEqual(conv.gen_base("resblocks.4.convs1_activates.1.alpha"), ("gen.rb.4.a1.1", "alpha"))
        self.assertEqual(conv.gen_base("snake_post.alpha"), ("gen.snake_post.alpha", "alpha"))
        self.assertIsNone(conv.gen_base("mpd.discriminators.0.convs.0.weight"))
        # The release layout: 311 generator tensors -> 233 GGUF tensors, all GGML_MAX_NAME-safe.
        full = tiny_generator(dict(TINY, upsample_initial_channel=1024, gen_conv_kernel=7, num_mels=2),
                              {k: v for k, v in conv.ARRAYS.items()})
        self.assertEqual(len(full), 311)

    def test_layouts_match_torch(self):
        import torch

        sd = tiny_generator()
        out = conv.build_generator(sd, "f32")
        # Conv1d: per-tap rows k * Cout + o.
        w = conv.fold_weight_norm(sd["resblocks.1.convs1.1.weight_v"], sd["resblocks.1.convs1.1.weight_g"])
        x = np.random.default_rng(3).standard_normal((1, w.shape[1], 11)).astype(np.float32)
        dil, kk = 3, w.shape[2]
        ref = torch.nn.functional.conv1d(torch.from_numpy(x), torch.from_numpy(w), padding=dil * (kk - 1) // 2,
                                         dilation=dil).numpy()[0]
        m = out["gen.rb.1.c1.1.weight"]
        cout = w.shape[0]
        xp = np.pad(x[0], ((0, 0), (dil * (kk - 1) // 2,) * 2))
        y = sum(m[t * cout:(t + 1) * cout] @ xp[:, t * dil:t * dil + 11] for t in range(kk))
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)
        # ConvTranspose1d: rows o * K + k, overlap-add at stride u, crop (K - u) / 2.
        wt = conv.fold_weight_norm(sd["ups.0.weight_v"], sd["ups.0.weight_g"])
        cin, cout, kt = wt.shape
        u = TINY_ARR["upsample_rates"][0]
        xt = np.random.default_rng(4).standard_normal((1, cin, 6)).astype(np.float32)
        ref = torch.nn.functional.conv_transpose1d(torch.from_numpy(xt), torch.from_numpy(wt), stride=u,
                                                   padding=(kt - u) // 2).numpy()[0]
        col = out["gen.ups.0.weight"] @ xt[0]  # [Cout * K, T]
        full = np.zeros((cout, (6 - 1) * u + kt), np.float32)
        for t in range(6):
            full[:, t * u:t * u + kt] += col[:, t].reshape(cout, kt)
        p = (kt - u) // 2
        np.testing.assert_allclose(full[:, p:full.shape[1] - p], ref, rtol=1e-5, atol=1e-5)
        # weight norm: torch's own parametrisation
        lin = torch.nn.utils.weight_norm(torch.nn.Conv1d(w.shape[1], w.shape[0], kk))
        with torch.no_grad():
            lin.weight_v.copy_(torch.from_numpy(sd["resblocks.1.convs1.1.weight_v"]))
            lin.weight_g.copy_(torch.from_numpy(sd["resblocks.1.convs1.1.weight_g"]))
        torch.nn.utils.remove_weight_norm(lin)
        np.testing.assert_array_equal(lin.weight.detach().numpy(), w)

    def test_precision_and_count(self):
        t = conv.build_tensors(tiny_mossformer(), tiny_generator(), "f16", TINY, TINY_ARR)
        self.assertEqual(t["gen.conv_pre.weight"].dtype, np.float16)
        self.assertEqual(t["gen.ups.1.weight"].dtype, np.float16)
        self.assertEqual(t["gen.rb.0.a1.0"].dtype, np.float32)
        self.assertEqual(t["gen.rb.0.c1.0.bias"].dtype, np.float32)
        self.assertEqual(t["mn.conv_dec.weight"].dtype, np.float16)
        self.assertEqual(t["mn.conv_dec.weight"].shape, (TINY["num_mels"], TINY["mossformer_sequence_dim"]))
        self.assertEqual(t["blk.0.fsmn.mem.weight"].dtype, np.float32)
        self.assertTrue(all(len(n) < 64 for n in t))
        t32 = conv.build_tensors(tiny_mossformer(), tiny_generator(), "f32", TINY, TINY_ARR)
        self.assertTrue(all(a.dtype == np.float32 for a in t32.values()))
        bad = tiny_generator()
        del bad["snake_post.alpha"]
        with self.assertRaises(ValueError):
            conv.build_tensors(tiny_mossformer(), bad, "f16", TINY, TINY_ARR)

    def test_gguf_round_trip(self):
        try:
            from gguf import GGUFReader
        except ImportError:
            self.skipTest("gguf package not installed")
        t = conv.build_tensors(tiny_mossformer(), tiny_generator(), "f16", TINY, TINY_ARR)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tiny.gguf"
            conv.write_gguf(t, path, "f16", hparams=TINY, arrays=TINY_ARR)
            r = GGUFReader(str(path))
            arch = r.fields["general.architecture"]
            self.assertEqual(bytes(arch.parts[arch.data[0]]).decode(), "mossformer2-sr")
            rates = r.fields["mossformer2_sr.upsample_rates"]
            self.assertEqual([int(rates.parts[i][0]) for i in rates.data], TINY_ARR["upsample_rates"])
            names = {x.name for x in r.tensors}
            self.assertEqual(names, set(t))


if __name__ == "__main__":
    unittest.main()
