#!/usr/bin/env python3
"""VoxCPM2 across C++ implementations: quality, speed, RAM, VRAM, file size.

Implementations (each built from source at its current default branch):
  - crispasr          (this repo)                 --backend voxcpm2
  - llama.cpp-omni    (tc-mb/llama.cpp-omni)       tools/omni/voxcpm2 voxcpm2-cli
  - VoxCPM.cpp        (bluryar/VoxCPM.cpp)         examples/voxcpm_tts (no seed option)

Weights are each project's own published GGUFs; the crispasr "q4_k mixed" arm
is built from voxcpm2-f16.gguf (stop/fsq/ralm F16, rest Q4_K). Every run is a
fresh process (load + synthesis), timed with /usr/bin/time (wall, peak RSS);
on CUDA a poller records peak device memory. Every WAV is transcribed with
whisper base.en through crispasr and scored exact / WER against the text.

  CPU (GH):  gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/voxcpm2_impl_bench.py \
                -f pip="huggingface_hub" -f args="device=cpu runs=5"
  CUDA:      tools/kaggle/voxcpm2-impl-bench (clones this repo, runs device=cuda)
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

ARG = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
DEVICE = ARG.get("device", "cpu")
RUNS = int(ARG.get("runs", "5"))
ONLY = set(ARG["impls"].split(",")) if "impls" in ARG else None
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("HEAVY_OUT", ROOT / "heavy-out"))
SCR = Path(os.environ.get("HEAVY_SCRATCH", ROOT / "heavy-scratch"))
OUT.mkdir(parents=True, exist_ok=True)
SCR.mkdir(parents=True, exist_ok=True)
NT = str(os.cpu_count() or 4)
CUDA_ARCH = ARG.get("cuda_arch", "")
TEXTS = {
    "short": "Hello, this is a short test sentence.",
    "long": "The quick brown fox jumps over the lazy dog. Every morning, she walks to the station "
            "and buys a newspaper before the train arrives.",
}
res = {"device": DEVICE, "runs": RUNS, "texts": TEXTS, "builds": {}, "arms": {}}


def save():
    (OUT / "result.json").write_text(json.dumps(res, indent=1))


def run(cmd, log=None, cwd=None, env=None, timeout=None, check=True):
    print("+", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=cwd, env=env, timeout=timeout)
    if log:
        Path(log).write_text(f"rc={r.returncode}\n--stdout--\n{r.stdout}\n--stderr--\n{r.stderr}")
    if check and r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"failed: {cmd[0]}")
    return r


def hf(repo, f):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, f, cache_dir=str(SCR / "hf")))


def cmake_build(src, bdir, flags, targets, tag):
    gen = ["-G", "Ninja"] if subprocess.run(["which", "ninja"], capture_output=True).returncode == 0 else []
    run(["cmake", "-S", src, "-B", bdir, *gen, "-DCMAKE_BUILD_TYPE=Release", *flags], OUT / f"cmake-{tag}.log")
    run(["cmake", "--build", bdir, "-j", NT, "--target", *targets], OUT / f"build-{tag}.log", timeout=7200)


def clone(url, d):
    if not d.exists():
        run(["git", "clone", "--depth", "1", "--recursive", url, d])
    return run(["git", "rev-parse", "HEAD"], cwd=d).stdout.strip()


def cuda_flags(key):
    if DEVICE != "cuda":
        return []
    f = {"crispasr": ["-DGGML_CUDA=ON"], "omni": ["-DGGML_CUDA=ON"], "vcpp": ["-DVOXCPM_CUDA=ON"]}[key]
    return f + ([f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}"] if CUDA_ARCH else [])


class VramPeak:
    def __init__(self):
        self.peak, self.base, self.stop = 0, self.read(), False

    @staticmethod
    def read():
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        try:
            return int(r.stdout.split()[0])
        except (IndexError, ValueError):
            return 0

    def __enter__(self):
        def poll():
            while not self.stop:
                self.peak = max(self.peak, self.read())
                time.sleep(0.1)
        self.t = threading.Thread(target=poll, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join()


def wav_seconds(p):
    try:
        with wave.open(str(p)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:  # noqa: BLE001
        return None


def norm(s):
    return re.sub(r"[^a-z ]", " ", (s or "").lower()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return round(d[len(h)] / max(1, len(r)), 3)


def main():
    if not Path("/usr/bin/time").exists():
        sudo = [] if os.geteuid() == 0 else ["sudo"]
        run(sudo + ["apt-get", "update", "-qq"], check=False)
        run(sudo + ["apt-get", "install", "-y", "-qq", "time"], check=False)
    # --- builds -------------------------------------------------------------
    cb = SCR / "b-crispasr"
    cmake_build(ROOT, cb, ["-DGGML_NATIVE=OFF", "-DBUILD_SHARED_LIBS=OFF", *cuda_flags("crispasr")],
                ["crispasr", "crispasr-quantize"], "crispasr")
    cli, quant = cb / "bin" / "crispasr", cb / "bin" / "crispasr-quantize"
    res["builds"]["crispasr"] = run(["git", "rev-parse", "HEAD"], cwd=ROOT).stdout.strip()
    impls = {}
    impls["crispasr"] = True
    if not ONLY or "omni" in ONLY:
        od = SCR / "llama.cpp-omni"
        res["builds"]["omni"] = clone("https://github.com/tc-mb/llama.cpp-omni.git", od)
        cmake_build(od, od / "build", ["-DLLAMA_CURL=OFF", "-DGGML_NATIVE=OFF", *cuda_flags("omni")],
                    ["voxcpm2-cli"], "omni")
        impls["omni"] = od / "build" / "bin" / "voxcpm2-cli"
    if not ONLY or "vcpp" in ONLY:
        vd = SCR / "VoxCPM.cpp"
        res["builds"]["vcpp"] = clone("https://github.com/bluryar/VoxCPM.cpp.git", vd)
        cmake_build(vd, vd / "build", ["-DVOXCPM_BUILD_BENCHMARK=OFF", "-DVOXCPM_BUILD_TESTS=OFF",
                                       "-DGGML_NATIVE=OFF", *cuda_flags("vcpp")], ["voxcpm_tts"], "vcpp")
        cand = [p for p in (vd / "build").rglob("voxcpm_tts") if p.is_file() and os.access(p, os.X_OK)]
        impls["vcpp"] = cand[0]
    save()
    wm = hf("ggerganov/whisper.cpp", "ggml-base.en.bin")

    # --- arms: (impl, name) -> (files, argv builder) -------------------------
    def crisp(files):
        return lambda text, wav, seed: [cli, "--backend", "voxcpm2", "-m", files[0], "--tts", text,
                                        "--tts-output", wav, "--seed", str(seed), "-t", NT]

    def omni(files):
        return lambda text, wav, seed: [impls["omni"], "-t", text, "-o", wav, "--seed", str(seed),
                                        *(["--cpu"] if DEVICE == "cpu" else []), files[0], files[1]]

    def vcpp(files):
        return lambda text, wav, seed: [impls["vcpp"], "--model-path", files[0], "--text", text, "--output", wav,
                                        "--backend", DEVICE, "--threads", NT]

    arms = [
        ("crispasr", "f16", lambda: [hf("cstr/voxcpm2-GGUF", "voxcpm2-f16.gguf")], crisp),
        ("crispasr", "q8_0", lambda: [hf("cstr/voxcpm2-GGUF", "voxcpm2-q8_0.gguf")], crisp),
        ("crispasr", "q4_k", lambda: [hf("cstr/voxcpm2-GGUF", "voxcpm2-q4_k.gguf")], crisp),
        ("crispasr", "q4_k mixed", None, crisp),
        ("omni", "f16", lambda: [hf("DennisHuang648/VoxCPM2-GGUF", "VoxCPM2-BaseLM-F16.gguf"),
                                 hf("DennisHuang648/VoxCPM2-GGUF", "VoxCPM2-Acoustic-F16.gguf")], omni),
        ("omni", "baselm q8_0", lambda: [hf("DennisHuang648/VoxCPM2-GGUF", "VoxCPM2-BaseLM-Q8_0.gguf"),
                                         hf("DennisHuang648/VoxCPM2-GGUF", "VoxCPM2-Acoustic-F16.gguf")], omni),
        ("vcpp", "f16", lambda: [hf("bluryar/VoxCPM-GGUF", "voxcpm2-f16.gguf")], vcpp),
        ("vcpp", "q8_0", lambda: [hf("bluryar/VoxCPM-GGUF", "voxcpm2-q8_0.gguf")], vcpp),
        ("vcpp", "q4_k", lambda: [hf("bluryar/VoxCPM-GGUF", "voxcpm2-q4_k.gguf")], vcpp),
    ]

    def mixed_files():
        f16 = hf("cstr/voxcpm2-GGUF", "voxcpm2-f16.gguf")
        m = SCR / "voxcpm2-q4_k-mixed.gguf"
        if not m.exists():
            run([quant, f16, m, "q4_k", "--tensor-type", r"^(stop|fsq|ralm)\.=f16"], OUT / "quant-mixed.log")
        return [m]

    for impl, name, getf, mk in arms:
        if impl not in impls:
            continue
        key = f"{impl} {name}"
        try:
            files = mixed_files() if getf is None else getf()
        except Exception as e:  # noqa: BLE001
            res["arms"][key] = {"error": f"download: {e}"}
            save()
            continue
        size = round(sum(Path(f).stat().st_size for f in files) / 1e9, 2)
        argv = mk(files)
        rows = []
        for tname, text in TEXTS.items():
            for seed in range(1, RUNS + 1):
                wav = SCR / "out.wav"
                wav.unlink(missing_ok=True)
                tf = SCR / "time.txt"
                cmd = ["/usr/bin/time", "-f", "%e %M", "-o", tf] + argv(text, wav, seed)
                vp = VramPeak() if DEVICE == "cuda" else None
                if vp:
                    with vp:
                        r = run(cmd, OUT / f"run-{impl}-{name.replace(' ', '_')}-{tname}-{seed}.log",
                                check=False, timeout=3600)
                else:
                    r = run(cmd, OUT / f"run-{impl}-{name.replace(' ', '_')}-{tname}-{seed}.log",
                            check=False, timeout=3600)
                try:
                    wall, rss_kb = tf.read_text().split()[-2:]
                    wall, rss = float(wall), round(int(rss_kb) / 1024)
                except Exception:  # noqa: BLE001
                    wall, rss = None, None
                secs = wav_seconds(wav) if wav.exists() else None
                heard = None
                if secs:
                    heard = run([cli, "-m", wm, "-f", wav, "-np", "-nt"], check=False, timeout=600).stdout.strip()
                rows.append({"text": tname, "seed": seed, "rc": r.returncode, "wall_s": wall, "rss_mb": rss,
                             "vram_mb": (vp.peak - vp.base) if vp else None, "audio_s": secs,
                             "rtf": round(wall / secs, 3) if wall and secs else None, "asr": heard,
                             "exact": norm(heard) == norm(text), "wer": wer(text, heard) if heard else 1.0,
                             "err": r.stderr[-400:] if r.returncode else ""})
                print(key, rows[-1], flush=True)
                res["arms"][key] = {"size_gb": size, "rows": rows}
                save()
        for f in files:  # keep disk bounded
            if "whisper" not in str(f):
                Path(f).unlink(missing_ok=True)

    # --- summary ------------------------------------------------------------
    def med(v):
        v = sorted(x for x in v if x is not None)
        return v[len(v) // 2] if v else None

    lines = [f"## VoxCPM2 implementations ({DEVICE}, {RUNS} runs per text)", "",
             "| impl / weights | size GB | exact | mean WER | median RTF short / long | peak RSS MB | peak VRAM MB |",
             "|---|---|---|---|---|---|---|"]
    for key, d in res["arms"].items():
        if "rows" not in d:
            lines.append(f"| {key} | — | {d.get('error')} | | | | |")
            continue
        rs = d["rows"]
        ok = [r for r in rs if r["rc"] == 0 and r["audio_s"]]
        lines.append(
            f"| {key} | {d['size_gb']} | {sum(r['exact'] for r in rs)}/{len(rs)} | "
            f"{round(sum(r['wer'] for r in rs) / len(rs), 3)} | "
            f"{med([r['rtf'] for r in ok if r['text'] == 'short'])} / {med([r['rtf'] for r in ok if r['text'] == 'long'])} | "
            f"{max((r['rss_mb'] or 0) for r in rs)} | "
            f"{max((r['vram_mb'] or 0) for r in rs) if DEVICE == 'cuda' else '—'} |")
    bad = [f"- {k} {r['text']} s{r['seed']}: rc={r['rc']} `{r['asr']}`" for k, d in res["arms"].items()
           for r in d.get("rows", []) if not r["exact"]]
    lines += ["", "RTF includes model load (fresh process per run).", "", "### Non-exact runs"] + bad
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
