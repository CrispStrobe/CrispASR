#!/usr/bin/env python3
"""SenseVoice CMVN + corrected FunASR references.

1. Build CONTROL (main's C/C++ sources) and FIX (this branch).
2. Convert SenseVoiceSmall -> F16 GGUF with the am.mvn CMVN tensors.
3. Rebake sensevoice/{jfk,zh} and paraformer/{zh,jfk} references with the
   dumpers that now use AutoModel's real front-end (CMVN, dither 0); upload.
4. crispasr-diff: sensevoice CONTROL + old GGUF vs FIX + new GGUF; paraformer
   CONTROL vs FIX.
5. Only if the FIX passes every sensevoice stage and the rich-tag text on
   both clips: quantize Q8_0 / Q4_K and upload all three SenseVoice GGUFs.
"""
import json, os, shutil, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; M = Path("/tmp/m")
res = {"errors": [], "diff": {}, "cli": {}}
def save(): (OUT / "sensevoice_cmvn.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
def run(cmd, log, timeout=5400, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    (OUT / log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-30000:])
    return r.returncode, r.stdout, r.stderr
try:
    subprocess.check_call(["git", "clone", "--depth", "30", "--recurse-submodules", "--shallow-submodules", "-b", "fix/kaldi-mel-domain",
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    kh.install_build_toolchain()
    flags = " ".join(kh.cache_and_link_flags())
    subprocess.check_call(["git", "-C", str(REPO), "fetch", "-q", "--depth", "30", "origin", "main"])
    base = subprocess.check_output(["git", "-C", str(REPO), "merge-base", "HEAD", "FETCH_HEAD"], text=True).strip()
    changed = [f for f in subprocess.check_output(["git", "-C", str(REPO), "diff", "--name-only", base, "HEAD"], text=True).split()
               if f.endswith((".cpp", ".h", ".c")) and f.startswith(("src/", "examples/"))]
    res["control_base"] = base; res["changed_cpp"] = changed
    fixed = {f: (REPO / f).read_text() for f in changed}
    builds = {}
    for tag in ("ctrl", "fix"):
        for f in changed:
            (REPO / f).write_text(subprocess.check_output(["git", "-C", str(REPO), "show", f"{base}:{f}"], text=True)
                                  if tag == "ctrl" else fixed[f])
        b = WORK / f"build-{tag}"
        kh.sh(f"cmake -S {REPO} -B {b} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF {flags}")
        with kh.build_heartbeat(f"build.{tag}"):
            kh.sh(f"cmake --build {b} -j$(nproc) --target crispasr crispasr-diff crispasr-quantize")
        builds[tag] = b / "bin"
    run([sys.executable, "-m", "pip", "install", "-q", "gguf", "funasr", "modelscope", "sentencepiece"], "pip.log")
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    sv = snapshot_download("FunAudioLLM/SenseVoiceSmall", local_dir=str(M / "sv"))
    pf = snapshot_download("funasr/paraformer-zh", local_dir=str(M / "pf"))
    new16 = M / "sensevoice-small-f16.gguf"
    rc, out, err = run([sys.executable, str(REPO / "models/convert-sensevoice-to-gguf.py"), "--input", sv, "--bpemodel",
                        str(Path(sv) / "chn_jpn_yue_eng_ko_spectok.bpe.model"), "--output", str(new16)], "convert.log")
    res["convert_rc"] = rc; save()
    old16 = hf_hub_download("cstr/sensevoice-small-GGUF", "sensevoice-small-f16.gguf", local_dir=str(M / "old"))
    pf16 = hf_hub_download("cstr/paraformer-zh-GGUF", "paraformer-zh-f16.gguf", local_dir=str(M / "pfg"))
    wav = {"jfk": str(REPO / "samples/jfk.wav"), "zh": str(REPO / "samples/paraformer_zh.wav")}
    ok = rc == 0
    for name, md, clips in (("sensevoice", sv, ["jfk", "zh"]), ("paraformer", pf, ["zh", "jfk"])):
        for c in clips:
            ref = M / f"{name}-{c}-ref.gguf"
            rc, out, err = run([sys.executable, str(REPO / "tools/dump_reference.py"), "--backend", name, "--model-dir", md,
                                "--audio", wav[c], "--output", str(ref)], f"dump-{name}-{c}.log", cwd=str(REPO))
            D = res["diff"].setdefault(f"{name}/{c}", {"dump_rc": rc})
            if rc != 0:
                D["dump_err"] = err[-1500:]; ok = False; save(); continue
            api.upload_file(path_or_fileobj=str(ref), path_in_repo=f"{name}/{c}/ref.gguf", repo_id="cstr/crispasr-regression-fixtures",
                            repo_type="dataset", commit_message=f"{name}/{c}: rebake with AutoModel's real front-end (am.mvn CMVN)")
            arms = {"ctrl": (builds["ctrl"], old16 if name == "sensevoice" else pf16),
                    "fix": (builds["fix"], str(new16) if name == "sensevoice" else pf16)}
            for tag, (bindir, g) in arms.items():
                rc, out, err = run([str(bindir / "crispasr-diff"), name, g, str(ref), wav[c]], f"diff-{name}-{c}-{tag}.log")
                rows = [l for l in out.splitlines() if l.startswith("[")]
                D[tag] = {"rc": rc, "n_fail": sum(1 for l in rows if l.startswith("[FAIL")), "rows": rows[:12] + rows[-6:]}
                if tag == "fix" and name == "sensevoice" and (rc != 0 or D[tag]["n_fail"]):
                    ok = False
                rc, out, err = run([str(bindir / "crispasr"), "--backend", name, "-m", g, "-f", wav[c], "-np", "-nt"], f"cli-{name}-{c}-{tag}.log")
                res["cli"][f"{name}/{c}/{tag}"] = out.strip() if rc == 0 else f"rc={rc}"
            save()
    res["upload_gate"] = ok
    if ok:
        files = [new16]
        for qt in ("q8_0", "q4_k"):
            p = M / f"sensevoice-small-{qt}.gguf"
            subprocess.check_call([str(builds["fix"] / "crispasr-quantize"), str(new16), str(p), qt])
            files.append(p)
        for p in files:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id="cstr/sensevoice-small-GGUF", repo_type="model",
                            commit_message="Re-convert with the am.mvn CMVN tensors (sensevoice.cmvn_shift/scale)")
            res.setdefault("uploaded", []).append([p.name, p.stat().st_size])
        save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
    for b in ("build-ctrl", "build-fix"):
        shutil.rmtree(WORK / b, ignore_errors=True)
