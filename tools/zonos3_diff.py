#!/usr/bin/env python3
"""Stage-by-stage diff of ZONOS3-Mini (issue #603): CrispASR vs Zyphra's upstream ggml runtime.

Reference side (upstream servers/ggml zonos3-cli, unmodified):
    zonos3-cli backbone-ptq.gguf --cpu --build-prompt "<text>" ref/input_ids.npy --speaker-embedding emb.npy
    zonos3-cli backbone-ptq.gguf --cpu --validate ref/input_ids.npy ref/
    zonos3-cli backbone-ptq.gguf --cpu --tts "<text>" ref/codes.npy --speaker-embedding emb.npy --greedy
CrispASR side (real CLI, env-gated dump hook in src/zonos3_tts.cpp):
    CRISPASR_ZONOS3_DUMP_DIR=ours CRISPASR_ZONOS3_GREEDY=1 crispasr --backend zonos3 -m backbone-ptq.gguf \
        --tts "<text>" --voice emb.npy --tts-output ours.wav
Then:  python tools/zonos3_diff.py ref/ ours/ [--ref-wav ref.wav --wav ours.wav]

Prints cos / max_abs / |mine| / |ref| per stage (cosine is scale-blind: read the norms), the first failing
stage in graph order, and exact equality of the input ids and the generated code matrix.
"""
import argparse
import os
import re
import sys

import numpy as np

ORDER = ["input_ids", "emb_sum", "emb_cond", "emb_norm"]


def stage_key(name):
    if name in ORDER:
        return (ORDER.index(name), -1, "")
    m = re.match(r"(.*)_(\d+)$", name)
    if m:
        sub = ["attnres_attn", "attn_in", "gdn_conv", "gdn_q", "gdn_k", "gdn_v", "gdn_decay", "gdn_beta",
               "gdn_recurrent", "gdn_state", "gdn_conv_state", "gdn_gated", "attn_out", "attnres_ffn", "ffn_in",
               "router_logits", "ffn_out", "layer_x"]
        base = m.group(1)
        return (10, int(m.group(2)), "%02d" % (sub.index(base) if base in sub else 99) + base)
    tail = ["attnres_final", "out_norm", "mout", "logits"]
    return (20, 0, "%02d" % (tail.index(name) if name in tail else 99))


def load_wav(path):
    import wave
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
        return x, w.getframerate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("ours")
    ap.add_argument("--tol", type=float, default=0.9999, help="per-stage cosine threshold")
    ap.add_argument("--ref-wav")
    ap.add_argument("--wav")
    a = ap.parse_args()

    names = sorted({f[:-4] for f in os.listdir(a.ref) if f.endswith(".npy") and not f.startswith("codes")}
                   & {f[:-4] for f in os.listdir(a.ours) if f.endswith(".npy") and not f.startswith("codes")},
                   key=stage_key)
    if not names:
        print("no common stages")
        return 2
    fails = []
    print("%-28s %10s %12s %12s %12s  %s" % ("stage", "cos", "max_abs", "|mine|", "|ref|", "shape"))
    for n in names:
        r = np.load(os.path.join(a.ref, n + ".npy")).astype(np.float64).ravel()
        o = np.load(os.path.join(a.ours, n + ".npy")).astype(np.float64).ravel()
        if r.shape != o.shape:
            print("%-28s SHAPE MISMATCH ref %s ours %s" % (n, r.shape, o.shape))
            fails.append(n)
            continue
        nr, no = np.linalg.norm(r), np.linalg.norm(o)
        cos = float(r @ o / (nr * no)) if nr > 0 and no > 0 else (1.0 if nr == no else 0.0)
        mx = float(np.max(np.abs(r - o))) if r.size else 0.0
        exact = n == "input_ids"
        ok = (mx == 0.0) if exact else (cos >= a.tol and abs(no - nr) <= 1e-3 * max(nr, 1e-12))
        if not ok:
            fails.append(n)
        print("%-28s %10.6f %12.3e %12.4e %12.4e  %s%s" % (n, cos, mx, no, nr, np.load(os.path.join(a.ref, n + ".npy")).shape,
                                                         "" if ok else "  FAIL"))
    print("stages: %d compared, %d fail%s" % (len(names), len(fails), (", first: " + fails[0]) if fails else ""))

    rc = 1 if fails else 0
    rcodes, ocodes = os.path.join(a.ref, "codes.npy"), os.path.join(a.ours, "codes.npy")
    if os.path.exists(rcodes) and os.path.exists(ocodes):
        r, o = np.load(rcodes), np.load(ocodes)
        re_, oe = np.load(rcodes + ".eos.npy"), np.load(ocodes + ".eos.npy")
        same = r.shape == o.shape and np.array_equal(r, o)
        agree = 0
        if r.shape[1:] == o.shape[1:]:
            k = min(len(r), len(o))
            agree = int(np.all(r[:k] == o[:k], axis=1).sum())
        print("codes: ref %s eos %s | ours %s eos %s | identical=%s, matching frames %d" %
              (r.shape, re_.tolist(), o.shape, oe.tolist(), same, agree))
        rc = rc or (0 if same and np.array_equal(re_, oe) else 1)
    if a.ref_wav and a.wav:
        x, sx = load_wav(a.ref_wav)
        y, sy = load_wav(a.wav)
        k = min(len(x), len(y))
        d = float(np.max(np.abs(x[:k] - y[:k]))) if k else 0.0
        c = float(x[:k] @ y[:k] / (np.linalg.norm(x[:k]) * np.linalg.norm(y[:k]) + 1e-30)) if k else 0.0
        print("wav: ref %d @%d | ours %d @%d | cos %.6f max_abs %.3e rms ref %.4f ours %.4f" %
              (len(x), sx, len(y), sy, c, d, float(np.sqrt(np.mean(x ** 2))), float(np.sqrt(np.mean(y ** 2)))))
    return rc


if __name__ == "__main__":
    sys.exit(main())
