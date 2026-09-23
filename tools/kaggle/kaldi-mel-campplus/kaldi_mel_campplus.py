#!/usr/bin/env python3
"""Mel-domain fbank triangles, CAM++ leg: rebake chatterbox's campplus_fbank +
campplus_xvector reference (upstream ChatterboxMultilingualTTS), then compare
the CONTROL build (main) and the FIX build with crispasr-diff chatterbox.

Kept apart from kaldi-mel-recheck because chatterbox-tts pins its own torch.
"""
import json, os, shutil, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; M = Path("/tmp/m")
res = {"clips": {}, "errors": []}
def save(): (OUT / "campplus.json").write_text(json.dumps(res, indent=1, default=str))
def run(cmd, log, timeout=5400, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    (OUT / log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-30000:])
    return r.returncode, r.stdout, r.stderr
CHANGED = ["src/core/kaldi_fbank.h", "src/firered_asr.cpp", "src/firered_vad.cpp"]
try:
    subprocess.check_call(["git", "clone", "--depth", "2", "--recurse-submodules", "--shallow-submodules", "-b", "fix/kaldi-mel-domain",
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    kh.install_build_toolchain()
    flags = " ".join(kh.cache_and_link_flags())
    fixed = {f: (REPO / f).read_text() for f in CHANGED}
    subprocess.check_call(["git", "-C", str(REPO), "fetch", "-q", "--depth", "1", "origin", "main"])
    builds = {}
    for tag in ("ctrl", "fix"):
        for f in CHANGED:
            (REPO / f).write_text(subprocess.check_output(["git", "-C", str(REPO), "show", f"FETCH_HEAD:{f}"], text=True)
                                  if tag == "ctrl" else fixed[f])
        b = WORK / f"build-{tag}"
        kh.sh(f"cmake -S {REPO} -B {b} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF {flags}")
        with kh.build_heartbeat(f"build.{tag}"):
            kh.sh(f"cmake --build {b} -j$(nproc) --target crispasr-diff")
        builds[tag] = b / "bin"
    run([sys.executable, "-m", "pip", "install", "-q", "chatterbox-tts", "gguf"], "pip.log")
    # chatterbox-tts pins torch 2.6 but leaves Kaggle's newer torchvision, whose
    # compiled ops then fail to register (torchvision::nms) and break
    # transformers' import chain. Nothing here uses torchvision.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], "pip-uninstall-tv.log")
    res["torch"] = subprocess.run([sys.executable, "-c", "import torch,torchaudio,transformers;print(torch.__version__,torchaudio.__version__,transformers.__version__)"],
                                  capture_output=True, text=True).stdout.strip()
    # PyPI's chatterbox-tts predates the V3 t3_model option the dumper uses;
    # run the upstream GitHub source (the dumper honours RESEMBLE_CHATTERBOX_SRC).
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/resemble-ai/chatterbox", "/tmp/chatterbox-src"])
    os.environ["RESEMBLE_CHATTERBOX_SRC"] = "/tmp/chatterbox-src/src"
    res["chatterbox_src"] = subprocess.check_output(["git", "-C", "/tmp/chatterbox-src", "rev-parse", "HEAD"], text=True).strip()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    md = snapshot_download("ResembleAI/chatterbox", local_dir=str(M / "chatterbox"))
    t3 = hf_hub_download("cstr/chatterbox-GGUF", "chatterbox-v3-t3-f16.gguf", local_dir=str(M / "gguf"))
    hf_hub_download("cstr/chatterbox-GGUF", "chatterbox-v3-s3gen-f16.gguf", local_dir=str(M / "gguf"))
    wav = {"jfk": str(REPO / "samples/jfk.wav"), "zh": str(REPO / "samples/paraformer_zh.wav")}
    for c, w in wav.items():
        C = res["clips"].setdefault(c, {})
        ref = M / f"campplus-{c}-ref.gguf"
        rc, out, err = run([sys.executable, str(REPO / "tools/dump_reference.py"), "--backend", "chatterbox", "--model-dir", md,
                            "--audio", w, "--output", str(ref), "--stages", "campplus_fbank,campplus_xvector"],
                           f"dump-{c}.log", cwd=str(REPO))
        C["dump_rc"] = rc
        if rc != 0:
            C["dump_err"] = err[-1500:]; save(); continue
        api.upload_file(path_or_fileobj=str(ref), path_in_repo=f"chatterbox-campplus/{c}/ref.gguf",
                        repo_id="cstr/crispasr-regression-fixtures", repo_type="dataset",
                        commit_message=f"chatterbox-campplus/{c}: rebake (mel-domain fbank recheck)")
        for tag, bindir in builds.items():
            rc, out, err = run([str(bindir / "crispasr-diff"), "chatterbox", t3, str(ref), w], f"diff-{c}-{tag}.log")
            C[tag] = {"rc": rc, "rows": [l for l in out.splitlines() if "campplus" in l]}
        save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
    for b in ("build-ctrl", "build-fix"):
        shutil.rmtree(WORK / b, ignore_errors=True)
