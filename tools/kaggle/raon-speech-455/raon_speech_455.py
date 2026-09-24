#!/usr/bin/env python3
"""#455 — KRAFTON/Raon-Speech-9B speech-to-text: convert, verify, publish.

1. Build CrispASR (CUDA) with crispasr, crispasr-diff, crispasr-quantize and
   the resampler unit test.
2. Reference: tools/reference_backends/raon_speech.py in a virtualenv pinned
   to transformers 4.57.3 (the remote code's version), on jfk (11 s, two 8 s
   chunks) and ko-369 (3.3 s, one chunk): greedy bf16 text + fp32 stages.
3. convert-qwen3-asr-to-gguf.py -> F16; crispasr-diff raon-speech on CPU
   (F16 does not fit a 16 GB GPU); Q8_0 + Q4_K; CLI transcripts (F16 on CPU,
   quants on GPU).
Upload gate: every diff stage passes on both clips and the F16 transcript
equals the reference greedy text on both clips.
"""
import json, os, shutil, subprocess, sys, traceback
from pathlib import Path

WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; BUILD = REPO / "build"
res = {"errors": [], "disk": {}, "diff": {}, "cli": {}, "ref": {}}
# DIAG: reference stages + F16 diff only (no greedy text, quants, CLI or upload)
DIAG = False
res["diag"] = DIAG


def save():
    (OUT / "r455.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))


def run(cmd, log, env=None, timeout=7200, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
    (OUT / log).write_text(r.stdout[-40000:] + "\n--- stderr ---\n" + r.stderr[-20000:])
    return r.returncode, r.stdout, r.stderr


def free_gb(p):
    try:
        return round(shutil.disk_usage(p).free / 1e9, 1)
    except Exception:
        return 0.0


try:
    for p in ("/kaggle/working", "/kaggle/tmp", "/tmp", "/root"):
        res["disk"][p] = free_gb(p)
    big = Path(max(("/kaggle/tmp", "/tmp", "/kaggle/working"), key=free_gb)) / "raon"
    big.mkdir(parents=True, exist_ok=True)
    res["big"] = str(big); save()

    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b",
                           "feat/455-raon-speech", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    res["head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    kh.install_build_toolchain()
    arch = None if DIAG else kh.detect_cuda_arch()
    gpu_flags = [] if DIAG else kh.cuda_build_flags(arch)
    kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF "
          f"-DCRISPASR_BUILD_TESTS=ON " + " ".join(gpu_flags + kh.cache_and_link_flags()))
    with kh.build_heartbeat("build"):
        kh.sh(f"cmake --build {BUILD} -j{kh.safe_build_jobs(gpu=not DIAG)} "
              f"--target crispasr crispasr-diff crispasr-quantize test-torchaudio-resample")
    rc, out, err = run([str(BUILD / "bin/test-torchaudio-resample")], "unit-resample.log")
    res["unit_resample"] = {"rc": rc, "tail": out[-300:]}; save()

    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    md = snapshot_download("KRAFTON/Raon-Speech-9B", local_dir=str(big / "hf"))
    res["disk"]["after_download"] = free_gb(str(big)); save()

    # ---- reference (transformers 4.57.3 in its own environment) ----
    venv = Path("/tmp/raonenv")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "virtualenv", "gguf"], check=True)
    subprocess.run([sys.executable, "-m", "virtualenv", "-q", "--system-site-packages", str(venv)], check=True)
    run([str(venv / "bin/pip"), "install", "-q", "transformers==4.57.3", "speechbrain", "soundfile", "gguf", "pyyaml"],
        "pip-raon.log")
    wav = {"jfk": str(REPO / "samples/jfk.wav"), "ko": str(REPO / "samples/ko-369.wav")}
    env = dict(os.environ, RAON_TMP=str(big), OMP_NUM_THREADS=str(os.cpu_count() or 4))
    for c, w in wav.items():
        ref = big / f"raon-{c}-ref.gguf"
        rc, out, err = run([str(venv / "bin/python"), str(REPO / "tools/dump_reference.py"), "--backend", "raon-speech",
                            "--model-dir", md, "--audio", w, "--output", str(ref), "--max-new-tokens", "256"]
                           + (["--stages", "raw_audio,raon_mel_chunk0,raon_encoder_output,raon_adaptor_output"] if DIAG else []),
                           f"dump-{c}.log", env=env, cwd=str(REPO / "tools"), timeout=10800)
        import gguf
        R = res["ref"].setdefault(c, {"rc": rc})
        if rc == 0:
            rd = gguf.GGUFReader(str(ref))
            R["text"] = next((bytes(f.parts[f.data[0]]).decode() for k, f in rd.fields.items()
                              if k == "crispasr.ref.generated_text"), "")
            R["stdout"] = "\n".join(l for l in out.splitlines() if "raon" in l)[-800:]
        else:
            R["err"] = err[-2500:]
        save()

    # ---- convert + diff (F16, CPU) ----
    f16 = big / "raon-speech-9b-f16.gguf"
    rc, out, err = run([sys.executable, str(REPO / "models/convert-qwen3-asr-to-gguf.py"), "--input", md,
                        "--output", str(f16)], "convert.log", timeout=10800,
                       env=dict(os.environ, TMPDIR=str(big)))  # GGUFWriter spools to a temp file
    res["convert_rc"] = rc
    if rc != 0:
        res["convert_err"] = err[-2500:]; save(); raise SystemExit("convert failed")
    shutil.rmtree(big / "hf", ignore_errors=True)  # 18 GB back before quantizing
    ok = all(res["ref"].get(c, {}).get("rc") == 0 for c in wav)
    cpu_env = dict(os.environ, CRISPASR_DIFF_NO_GPU="1")
    for c, w in wav.items():
        ref = big / f"raon-{c}-ref.gguf"
        if not ref.exists():
            continue
        rc, out, err = run([str(BUILD / "bin/crispasr-diff"), "raon-speech", str(f16), str(ref), w], f"diff-{c}.log",
                           env=cpu_env)
        rows = [l for l in out.splitlines() if l.startswith("[") or "raon frames" in l]
        res["diff"][c] = {"rc": rc, "rows": rows, "n_fail": sum(1 for l in rows if l.startswith(("[FAIL", "[ERR")))}
        if rc != 0 or res["diff"][c]["n_fail"]:
            ok = False
        save()

    if DIAG:
        # the reflect-pad fix changes every crisp_audio mel: re-check the
        # qwen3-asr family fixtures (Confucius4-R2T2, F16) still pass
        import gguf, numpy as np, soundfile as sf
        from huggingface_hub import hf_hub_download
        r2 = hf_hub_download("cstr/confucius4-r2t2-GGUF", "confucius4-r2t2-f16.gguf", local_dir=str(big / "r2t2"))
        res["r2t2"] = {}
        for c in ("jfk", "zh"):
            fx = hf_hub_download("cstr/crispasr-regression-fixtures", f"r2t2/{c}/ref.gguf", repo_type="dataset",
                                 local_dir=str(big / "fx"))
            ra = [t for t in gguf.GGUFReader(fx).tensors if t.name == "raw_audio"][0]
            wv = str(big / f"r2t2-{c}.wav")
            sf.write(wv, np.array(ra.data, dtype=np.float32).reshape(-1), 16000)
            rc, out, err = run([str(BUILD / "bin/crispasr-diff"), "qwen3", r2, fx, wv], f"diff-r2t2-{c}.log", env=cpu_env)
            res["r2t2"][c] = {"rc": rc, "rows": [l for l in out.splitlines() if l.startswith("[")]}
            save()
        raise SystemExit(0)
    ggufs = {"f16": f16}
    for qt in ("q8_0", "q4_k"):
        p = big / f"raon-speech-9b-{qt}.gguf"
        if subprocess.run([str(BUILD / "bin/crispasr-quantize"), str(f16), str(p), qt], capture_output=True).returncode == 0:
            ggufs[qt] = p
    res["sizes"] = {q: p.stat().st_size for q, p in ggufs.items()}; save()

    for q, g in ggufs.items():
        for c, w in wav.items():
            cmd = [str(BUILD / "bin/crispasr"), "--backend", "raon-speech", "-m", str(g), "-f", w, "-np", "-nt"]
            if q == "f16":
                cmd.append("-ng")  # 17.5 GB does not fit the GPU
            rc, out, err = run(cmd, f"cli-{q}-{c}.log", timeout=5400)
            res["cli"][f"{q}/{c}"] = out.strip() if rc == 0 else f"rc={rc}: {err[-400:]}"
            save()
    for c in wav:
        if res["cli"].get(f"f16/{c}", "").strip() != res["ref"].get(c, {}).get("text", "").strip():
            ok = False
    res["upload_gate"] = ok; save()

    if ok:
        target = "cstr/raon-speech-9b-GGUF"
        api.create_repo(target, repo_type="model", exist_ok=True)
        for q in ("q4_k", "q8_0", "f16"):
            if q in ggufs:
                api.upload_file(path_or_fileobj=str(ggufs[q]), path_in_repo=ggufs[q].name, repo_id=target,
                                repo_type="model")
        for c in wav:
            api.upload_file(path_or_fileobj=str(big / f"raon-{c}-ref.gguf"), path_in_repo=f"raon-speech/{c}/ref.gguf",
                            repo_id="cstr/crispasr-regression-fixtures", repo_type="dataset")
        res["uploaded"] = target
    save()
except SystemExit:
    pass
except BaseException:
    res["errors"].append(traceback.format_exc()[-4000:])
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
