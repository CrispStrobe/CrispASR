#!/usr/bin/env python3
"""Silero VAD v6.3 vs v6.2 before switching the --vad default (#612).

1. Parity: our per-frame probabilities (source context on, as upstream) against
   the silero-vad Python package of the same version. Proves the v6.3 weights
   load and run correctly in our port. The product path (whisper.cpp-style, no
   64-sample context) is reported alongside, since that is what --vad runs.
2. Segmentation and ASR on long files built from LibriSpeech utterances with
   known pauses (80 ms .. 1.5 s) over a low noise floor: speech recall, false
   speech, how many pauses of each length split segments, and parakeet WER
   with `--vad` per model and threshold. Everything goes through the real CLI.

  gh workflow run heavy-cpu.yml --ref <sha> -f script=tools/ci-heavy/vad_v63_eval.py \\
     -f pip="huggingface_hub soundfile pyarrow" -f timeout_minutes=180
"""
import ctypes
import io
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_controls_common as C  # noqa: E402

NT = str(os.cpu_count() or 4)
SR = 16000
GAPS = [0.08, 0.15, 0.25, 0.4, 0.7, 1.5]
MODELS = {
    "v6.2": ("ggml-org/whisper-vad", "ggml-silero-v6.2.0.bin", "9ffd54a1e1ee413ddf265af9913beaf518d1639b", "6.2.0"),
    "v6.3": ("cstr/silero-vad-GGUF", "ggml-silero-v6.3.0.bin", "baa429e63d02402131ff3ed4369e9b40d4981627", "6.3"),
}
ARMS = [("v6.2", 0.5), ("v6.3", 0.5), ("v6.3", 0.45), ("v6.3", 0.4), ("v6.3", 0.35)]


def build_shared():
    bd = C.SCRATCH / "build-shared"
    C.run(["cmake", "-S", C.ROOT, "-B", bd, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
           "-DBUILD_SHARED_LIBS=ON"], C.OUT / "cmake.log")
    C.run(["cmake", "--build", bd, "-j", NT, "--target", "crispasr-cli", "crispasr-lib"], C.OUT / "build.log")
    return bd / "bin" / "crispasr", next(bd.rglob("libcrispasr.so"))


def py_silero(version):
    """Install silero-vad==version (+ CPU torch once) into its own dir."""
    tgt = C.SCRATCH / f"silero-{version}"
    if not tgt.exists():
        C.run([sys.executable, "-m", "pip", "install", "-q", "--target", str(tgt), "--no-deps",
               f"silero-vad=={version}"], C.OUT / f"pip-silero-{version}.log")
    return tgt


def py_probs(version, wav):
    code = f"""
import sys, json, numpy as np, torch
sys.path.insert(0, {str(py_silero(version))!r})
from silero_vad import load_silero_vad
m = load_silero_vad()
x = torch.from_numpy(np.load(sys.argv[1]))
with torch.no_grad():
    p = m.audio_forward(x, 16000)[0].numpy()
np.save(sys.argv[2], p)
"""
    src = C.SCRATCH / "pyprobs.py"
    src.write_text(code)
    npin, npout = C.SCRATCH / "in.npy", C.SCRATCH / f"py-{version}.npy"
    np.save(npin, wav.astype(np.float32))
    C.run([sys.executable, src, npin, npout], C.OUT / f"pyprobs-{version}.log")
    return np.load(npout)


class Lib:
    def __init__(self, path):
        class P(ctypes.Structure):
            _fields_ = [("n_threads", ctypes.c_int), ("use_gpu", ctypes.c_bool), ("gpu_device", ctypes.c_int)]
        self.P = P
        L = self.L = ctypes.CDLL(str(path))
        L.whisper_vad_init_from_file_with_params.argtypes = [ctypes.c_char_p, P]
        L.whisper_vad_init_from_file_with_params.restype = ctypes.c_void_p
        L.whisper_vad_detect_speech.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        L.whisper_vad_detect_speech.restype = ctypes.c_bool
        L.whisper_vad_n_probs.argtypes = [ctypes.c_void_p]
        L.whisper_vad_n_probs.restype = ctypes.c_int
        L.whisper_vad_probs.argtypes = [ctypes.c_void_p]
        L.whisper_vad_probs.restype = ctypes.POINTER(ctypes.c_float)
        L.whisper_vad_free.argtypes = [ctypes.c_void_p]
        L.crispasr_silero_enable_context.argtypes = [ctypes.c_void_p]
        L.crispasr_silero_enable_context.restype = ctypes.c_bool

    def probs(self, model, wav, source_context):
        ctx = self.L.whisper_vad_init_from_file_with_params(str(model).encode(), self.P(1, False, 0))
        assert ctx, model
        if source_context:
            assert self.L.crispasr_silero_enable_context(ctx)
        x = np.ascontiguousarray(wav, dtype=np.float32)
        assert self.L.whisper_vad_detect_speech(ctx, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x))
        n = self.L.whisper_vad_n_probs(ctx)
        p = np.ctypeslib.as_array(self.L.whisper_vad_probs(ctx), shape=(n,)).copy()
        self.L.whisper_vad_free(ctx)
        return p


def librispeech():
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    import soundfile as sf
    repo = "hf-internal-testing/librispeech_asr_dummy"
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet") and "clean" in f]
    utts = []
    for f in sorted(files):
        t = pq.read_table(hf_hub_download(repo, f, repo_type="dataset", cache_dir=str(C.SCRATCH / "hf"))).to_pylist()
        for row in t:
            a, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            assert sr == SR
            utts.append((a, row["text"]))
    return utts


def trim(a):
    """Speech extent of one utterance: first/last 10 ms frame within 35 dB of its peak."""
    f = len(a) // 160
    e = np.array([np.sqrt(np.mean(a[i * 160:(i + 1) * 160] ** 2)) + 1e-9 for i in range(f)])
    on = np.where(20 * np.log10(e / e.max()) > -35)[0]
    return on[0] * 160, (on[-1] + 1) * 160


def make_files(utts, n_files=12, per_file=6, seed=0):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    out = []
    order = list(range(len(utts)))
    rng.shuffle(order)
    k = 0
    for fi in range(n_files):
        pieces, truth, text, gaps = [np.zeros(int(0.5 * SR), np.float32)], [], [], []
        pos = len(pieces[0])
        for j in range(per_file):
            a, t = utts[order[k % len(order)]]
            k += 1
            s, e = trim(a)
            a = a[s:e]
            truth.append((pos, pos + len(a)))
            text.append(t)
            pieces.append(a)
            pos += len(a)
            if j < per_file - 1:
                g = GAPS[(fi + j) % len(GAPS)]
                gaps.append((pos, pos + int(g * SR), g))
                pieces.append(np.zeros(int(g * SR), np.float32))
                pos += int(g * SR)
        pieces.append(np.zeros(int(0.5 * SR), np.float32))
        wav = np.concatenate(pieces)
        wav = wav + nrng.normal(0, 10 ** (-50 / 20), len(wav)).astype(np.float32)  # -50 dBFS floor
        path = C.SCRATCH / f"long{fi:02d}.wav"
        import soundfile as sf
        sf.write(path, wav, SR, subtype="PCM_16")
        out.append({"path": path, "wav": wav, "truth": truth, "gaps": gaps, "text": " ".join(text)})
    return out


def seg_metrics(segs, f):
    n = len(f["wav"])
    sp = np.zeros(n, bool)
    for s, e in f["truth"]:
        sp[s:e] = True
    hit = np.zeros(n, bool)
    for s, e in segs:
        hit[max(0, s):min(n, e)] = True
    split = {}
    for s, e, g in f["gaps"]:
        mid = (s + e) // 2
        split.setdefault(g, []).append(not hit[mid] or any(a > s and a < e for a, _ in segs))
    return {"recall": float((hit & sp).sum() / sp.sum()), "false": float((hit & ~sp).sum() / (~sp).sum()),
            "split": split}


def main():
    C.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchaudio", "packaging", "--index-url",
           "https://download.pytorch.org/whl/cpu", "--extra-index-url", "https://pypi.org/simple"],
          C.OUT / "pip-torch.log")
    b, lib_path = build_shared()
    lib = Lib(lib_path)
    asr = C.hf("cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf", None)
    vad = {k: C.hf(r, f, rev) for k, (r, f, rev, _) in MODELS.items()}
    files = make_files(librispeech())
    res = {"parity": {}, "arms": {}}

    # 1. parity on the first three files
    for k, (_, _, _, pyver) in MODELS.items():
        rows = []
        for f in files[:3]:
            ref = py_probs(pyver, f["wav"])
            ours = lib.probs(vad[k], f["wav"], True)
            prod = lib.probs(vad[k], f["wav"], False)
            n = min(len(ref), len(ours))
            rows.append({"n": n, "len_ref": len(ref), "len_ours": len(ours),
                         "max_abs_ctx": float(np.abs(ref[:n] - ours[:n]).max()),
                         "max_abs_product": float(np.abs(ref[:n] - prod[:n]).max()),
                         "mean_abs_product": float(np.abs(ref[:n] - prod[:n]).mean())})
        res["parity"][k] = rows
        C.save(res)

    # 2. product CLI per arm
    for k, th in ARMS:
        name = f"{k}@{th}"
        agg = {"recall": [], "false": [], "split": {}, "wer_ref": [], "wer_hyp": []}
        for f in files:
            exp = C.SCRATCH / f"{f['path'].stem}-{name}.json"
            # --vad-export-raw exits after writing the segments, so the
            # transcript needs its own run with the same VAD settings.
            C.run([b, "-m", asr, "-f", f["path"], "--vad", "-vm", vad[k], "-vt", str(th), "--vad-export-raw",
                   exp, "-np", "-t", NT], C.OUT / f"vad-{name}-{f['path'].stem}.log", check=False, timeout=1800)
            r = C.run([b, "-m", asr, "-f", f["path"], "--vad", "-vm", vad[k], "-vt", str(th), "-np", "-nt", "-t", NT],
                      C.OUT / f"cli-{name}-{f['path'].stem}.log", check=False, timeout=1800)
            sl = json.loads(exp.read_text())["crispasr_vad"]["slices"] if exp.exists() else []
            m = seg_metrics([(s["start"], s["end"]) for s in sl], f)
            agg["recall"].append(m["recall"])
            agg["false"].append(m["false"])
            for g, v in m["split"].items():
                agg["split"].setdefault(g, []).extend(v)
            agg["wer_ref"].append(f["text"])
            agg["wer_hyp"].append(" ".join(ln.strip() for ln in r.stdout.splitlines() if ln.strip()))
        res["arms"][name] = {
            "recall": float(np.mean(agg["recall"])), "false": float(np.mean(agg["false"])),
            "split": {str(g): f"{sum(v)}/{len(v)}" for g, v in sorted(agg["split"].items())},
            "wer": C.wer(" ".join(agg["wer_ref"]), " ".join(agg["wer_hyp"])),
        }
        C.save(res)

    if all(a["wer"] > 0.9 for a in res["arms"].values()):
        raise SystemExit("every arm has WER > 0.9: the transcript capture is broken, not the VAD")
    lines = ["## Silero VAD v6.3 vs v6.2", "", "### Parity vs silero-vad Python (max |Δp| per file)", "",
             "| model | ours, source context | product path (no context) |", "|---|---|---|"]
    for k, rows in res["parity"].items():
        lines.append(f"| {k} | {', '.join(f'{r['max_abs_ctx']:.2e}' for r in rows)} | "
                     f"{', '.join(f'{r['max_abs_product']:.3f}' for r in rows)} |")
    lines += ["", "### Product `--vad` on 12 long files (72 utterances, pauses 0.08–1.5 s)", "",
              "| model@threshold | speech recall | false speech | WER | pauses split: " +
              " / ".join(f"{g}s" for g in GAPS) + " |", "|---|---|---|---|---|"]
    for name, a in res["arms"].items():
        lines.append(f"| {name} | {a['recall']:.3f} | {a['false']:.3f} | {a['wer']:.3f} | "
                     f"{' / '.join(a['split'].get(str(g), '-') for g in GAPS)} |")
    C.summary(lines)


if __name__ == "__main__":
    main()
