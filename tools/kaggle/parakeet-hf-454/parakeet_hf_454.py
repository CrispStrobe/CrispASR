#!/usr/bin/env python3
"""#454 — moondream/parakeet-ultra and parakeet-redux: convert, verify, publish.

Per model: convert-parakeet-to-gguf.py --hf -> F16 (+ Q8_0, Q4_K); reference
dumps with transformers ParakeetForTDT (tools/reference_backends/parakeet_hf.py)
on jfk + a German clip; crispasr-diff parakeet; CLI transcripts at every
quant; moondream Photon transcripts as the independent check (for redux it is
the only check of the ternary dequantisation that is not our own code).
Uploads are gated: every diff stage passes and the F16 text equals the
transformers text on both clips.
"""
import json, os, shutil, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; BUILD = REPO / "build"; M = Path("/tmp/m")
res = {"models": {}, "errors": []}
def save(): (OUT / "p454.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
def run(cmd, log, env=None, timeout=5400, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
    (OUT / log).write_text(r.stdout[-40000:] + "\n--- stderr ---\n" + r.stderr[-20000:])
    return r.returncode, r.stdout, r.stderr
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b", "feat/454-parakeet-hf",
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    res["head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    kh.install_build_toolchain()
    kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF " + " ".join(kh.cache_and_link_flags()))
    with kh.build_heartbeat("build"):
        kh.sh(f"cmake --build {BUILD} -j$(nproc) --target crispasr crispasr-diff crispasr-quantize")
    run([sys.executable, "-m", "pip", "install", "-q", "gguf", "librosa", "safetensors", "virtualenv",
         "git+https://github.com/huggingface/transformers.git"], "pip.log")
    res["transformers"] = subprocess.run([sys.executable, "-c", "import transformers;print(transformers.__version__)"],
                                         capture_output=True, text=True).stdout.strip()
    # Photon in its own environment (it may pin its own dependencies)
    subprocess.run([sys.executable, "-m", "virtualenv", "-q", "--system-site-packages", "/tmp/photon"], check=True)
    run(["/tmp/photon/bin/pip", "install", "-q", "moondream>=2.4.0"], "pip-photon.log")
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    # German clip: the raw_audio of the orukeet/de fixture
    import gguf, numpy as np, soundfile as sf
    de = hf_hub_download("cstr/crispasr-regression-fixtures", "orukeet/de/ref.gguf", repo_type="dataset", local_dir=str(M / "fx"))
    ra = [t for t in gguf.GGUFReader(de).tensors if t.name == "raw_audio"][0]
    sf.write("/tmp/de.wav", np.array(ra.data, dtype=np.float32).reshape(-1), 16000)
    wav = {"jfk": str(REPO / "samples/jfk.wav"), "de": "/tmp/de.wav"}
    for repo, short in (("moondream/parakeet-ultra", "parakeet-ultra"), ("moondream/parakeet-redux", "parakeet-redux")):
        R = res["models"].setdefault(short, {"diff": {}, "cli": {}, "photon": {}})
        try:
            md = snapshot_download(repo, local_dir=str(M / short))
            f16 = M / f"{short}-f16.gguf"
            rc, out, err = run([sys.executable, str(REPO / "models/convert-parakeet-to-gguf.py"), "--hf", md, "--output", str(f16)],
                               f"convert-{short}.log")
            R["convert_rc"] = rc
            if rc != 0:
                R["convert_err"] = err[-1500:]; save(); continue
            ggufs = {"f16": f16}
            for qt in ("q8_0", "q4_k"):
                p = M / f"{short}-{qt}.gguf"
                if subprocess.run([str(BUILD / "bin/crispasr-quantize"), str(f16), str(p), qt], capture_output=True).returncode == 0:
                    ggufs[qt] = p
            R["sizes"] = {q: p.stat().st_size for q, p in ggufs.items()}
            ok = True
            for c, w in wav.items():
                ref = M / f"{short}-{c}-ref.gguf"
                rc, out, err = run([sys.executable, str(REPO / "tools/dump_reference.py"), "--backend", "parakeet-hf",
                                    "--model-dir", md, "--audio", w, "--output", str(ref)], f"dump-{short}-{c}.log", cwd=str(REPO / "tools"))
                D = R["diff"].setdefault(c, {"dump_rc": rc})
                if rc != 0:
                    D["dump_err"] = err[-1500:]; ok = False; save(); continue
                rd = gguf.GGUFReader(str(ref))
                D["hf_text"] = next((bytes(f.parts[f.data[0]]).decode() for k, f in rd.fields.items() if k == "crispasr.ref.generated_text"), "")
                D["aliasing"] = next((bytes(f.parts[f.data[0]]).decode() for k, f in rd.fields.items() if k == "crispasr.ref.aliasing_after_capture"), "n/a")
                rc, out, err = run([str(BUILD / "bin/crispasr-diff"), "parakeet", str(f16), str(ref), w], f"diff-{short}-{c}.log")
                rows = [l for l in out.splitlines() if l.startswith("[")]
                D.update({"rc": rc, "n_fail": sum(1 for l in rows if l.startswith("[FAIL")), "rows": rows})
                if rc != 0 or D["n_fail"]:
                    ok = False
                for q, g in ggufs.items():
                    rc, out, err = run([str(BUILD / "bin/crispasr"), "-m", str(g), "-f", w, "-np", "-nt"], f"cli-{short}-{q}-{c}.log")
                    R["cli"][f"{q}/{c}"] = out.strip() if rc == 0 else f"rc={rc}: {err[-300:]}"
                if R["cli"].get(f"f16/{c}", "").strip() != D["hf_text"].strip():
                    ok = False
                rc, out, err = run(["/tmp/photon/bin/python", "-c",
                                    f"import moondream as md\nwith md.photon({repo!r}, device='cpu') as s:\n    print(s.transcribe(audio={w!r})['text'])"],
                                   f"photon-{short}-{c}.log", timeout=1800)
                R["photon"][c] = out.strip()[-600:] if rc == 0 else f"rc={rc}: {err[-400:]}"
                save()
            R["upload_gate"] = ok
            if ok:
                target = f"cstr/{short}-GGUF"
                api.create_repo(target, repo_type="model", exist_ok=True)
                for q, g in ggufs.items():
                    api.upload_file(path_or_fileobj=str(g), path_in_repo=g.name, repo_id=target, repo_type="model")
                for c in wav:
                    api.upload_file(path_or_fileobj=str(M / f"{short}-{c}-ref.gguf"), path_in_repo=f"{short}/{c}/ref.gguf",
                                    repo_id="cstr/crispasr-regression-fixtures", repo_type="dataset")
                R["uploaded"] = target
            save()
        except BaseException:
            R["error"] = traceback.format_exc()[-2500:]
        save()
        shutil.rmtree(M / short, ignore_errors=True)
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
