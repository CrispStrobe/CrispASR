#!/usr/bin/env python3
"""FINAL verification of fix/kaldi-mel-domain: every Kaldi-fbank backend, fix build only.

Rebakes each reference with the corrected dumpers (real FunASR front-end, own-
storage captures, knf dither 0), converts SenseVoice fresh (with CMVN), and runs
crispasr-diff + CLI with the branch head. Supersedes the control-vs-fix run.

Original notes:

For every backend on a Kaldi-style front-end:
  1. rebake its crispasr-diff reference from the upstream package
     (tools/dump_reference.py) and upload it to
     cstr/crispasr-regression-fixtures/<backend>/<clip>/ref.gguf;
  2. run crispasr-diff with the CONTROL build (main: Hz-linear triangles) and
     the FIX build (mel-linear triangles);
  3. run the CLI with both builds for the end-to-end transcript.
Every log goes to out/, the summary to out/recheck.json. Backends run one
after another and each saves as it finishes; a failure in one does not stop
the others.
"""
import json, os, shutil, subprocess, sys, traceback
from pathlib import Path
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"; M = Path("/tmp/m")
res = {"backends": {}, "errors": []}
def save(): (OUT / "recheck.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
def run(cmd, log, env=None, timeout=5400, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
    (OUT / log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-30000:])
    return r.returncode, r.stdout, r.stderr
CHANGED = ["src/core/kaldi_fbank.h", "src/firered_asr.cpp", "src/firered_vad.cpp"]
# backend: (diff name, upstream HF id, GGUF repo, GGUF file, clips, extra env)
BACKENDS = [
    ("wespeaker", "wespeaker", "Wespeaker/wespeaker-voxceleb-resnet34-LM", "cstr/wespeaker-resnet34-lm-GGUF", "wespeaker-resnet34-lm-f32.gguf", ["jfk"]),
    ("sensevoice", "sensevoice", "FunAudioLLM/SenseVoiceSmall", "cstr/sensevoice-small-GGUF", "sensevoice-small-f16.gguf", ["jfk", "zh"]),
    ("paraformer", "paraformer", "funasr/paraformer-zh", "cstr/paraformer-zh-GGUF", "paraformer-zh-f16.gguf", ["zh", "jfk"]),
    ("firered-asr", "firered-asr", "FireRedTeam/FireRedASR2-AED", "cstr/firered-asr2-aed-GGUF", "firered-asr2-aed.gguf", ["jfk", "zh"]),
    ("funasr", "funasr", "FunAudioLLM/Fun-ASR-Nano-2512", "cstr/funasr-nano-GGUF", "funasr-nano-2512-f16.gguf", ["jfk", "zh"]),
    ("dolphin", "dolphin", None, "cstr/dolphin-cn-dialect-small-streaming-GGUF", "dolphin-cn-dialect-small-streaming-f16.gguf", ["zh", "jfk"]),
]
try:
    subprocess.check_call(["git", "clone", "--depth", "2", "--recurse-submodules", "--shallow-submodules", "-b", "fix/kaldi-mel-domain",
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    res["head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.resolve_hf_token(); os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    kh.install_build_toolchain()
    flags = " ".join(kh.cache_and_link_flags())
    fixed = {f: (REPO / f).read_text() for f in CHANGED}
    subprocess.check_call(["git", "-C", str(REPO), "fetch", "-q", "--depth", "1", "origin", "main"])
    res["control_main"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "FETCH_HEAD"], text=True).strip()
    builds = {}
    for tag in ("fix",):
        for f in CHANGED:
            (REPO / f).write_text(subprocess.check_output(["git", "-C", str(REPO), "show", f"FETCH_HEAD:{f}"], text=True)
                                  if tag == "ctrl" else fixed[f])
        b = WORK / f"build-{tag}"
        kh.sh(f"cmake -S {REPO} -B {b} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF -DCRISPASR_AMR=OFF {flags}")
        with kh.build_heartbeat(f"build.{tag}"):
            kh.sh(f"cmake --build {b} -j$(nproc) --target crispasr crispasr-diff")
        builds[tag] = b / "bin"
    res["built"] = True; save()
    # reference environments
    run([sys.executable, "-m", "pip", "install", "-q", "gguf", "kaldi-native-fbank", "funasr", "modelscope"], "pip-funasr.log")
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "fireredasr"], "pip-firered.log")
    run([sys.executable, "-m", "pip", "install", "-q", "kaldiio", "sentencepiece"], "pip-firered-deps.log")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/wenet-e2e/wespeaker", "/tmp/wespeaker"], capture_output=True)
    os.environ["WESPEAKER_REPO"] = "/tmp/wespeaker"
    fr = subprocess.run([sys.executable, "-c", "import fireredasr,os;print(os.path.dirname(fireredasr.__file__))"], capture_output=True, text=True)
    if fr.returncode == 0:  # record the upstream inference dither
        d = fr.stdout.strip()
        res["fireredasr_dither_lines"] = subprocess.run(["grep", "-rn", "dither", d], capture_output=True, text=True).stdout[-2000:]
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    wav = {"jfk": str(REPO / "samples/jfk.wav"), "zh": str(REPO / "samples/paraformer_zh.wav")}
    for name, diffname, upstream, grepo, gfile, clips in BACKENDS:
        R = res["backends"].setdefault(name, {"clips": {}})
        try:
            md = snapshot_download(upstream, local_dir=str(M / name)) if upstream and name != "firered-asr" else upstream
            if name == "sensevoice":
                gguf = str(M / "gguf" / "sensevoice-small-f16-cmvn.gguf"); (M / "gguf").mkdir(parents=True, exist_ok=True)
                rc, out, err = run([sys.executable, str(REPO / "models/convert-sensevoice-to-gguf.py"), "--input", md, "--bpemodel",
                                    str(Path(md) / "chn_jpn_yue_eng_ko_spectok.bpe.model"), "--output", gguf], "convert-sensevoice.log")
                R["convert_rc"] = rc
            else:
                gguf = hf_hub_download(grepo, gfile, local_dir=str(M / "gguf"))
            for c in clips:
                C = R["clips"].setdefault(c, {})
                if upstream:
                    ref = M / f"{name}-{c}-ref.gguf"
                    rc, out, err = run([sys.executable, str(REPO / "tools/dump_reference.py"), "--backend", diffname, "--model-dir", str(md),
                                        "--audio", wav[c], "--output", str(ref)], f"dump-{name}-{c}.log", cwd=str(REPO))
                    C["dump_rc"] = rc
                    if rc != 0:
                        C["dump_err"] = err[-1500:]; save(); continue
                    api.upload_file(path_or_fileobj=str(ref), path_in_repo=f"{name}/{c}/ref.gguf",
                                    repo_id="cstr/crispasr-regression-fixtures", repo_type="dataset",
                                    commit_message=f"{name}/{c}: rebake (final: mel-domain fbank, real FunASR front-end, own-storage captures)")
                    C["uploaded"] = True
                else:
                    ref = hf_hub_download("cstr/crispasr-regression-fixtures", f"{name}/{c}/ref.gguf", repo_type="dataset",
                                          local_dir=str(M / "fx"))
                for tag, bindir in builds.items():
                    rc, out, err = run([str(bindir / "crispasr-diff"), diffname, gguf, str(ref), wav[c]], f"diff-{name}-{c}-{tag}.log")
                    C[f"diff_{tag}"] = {"rc": rc, "rows": [l for l in out.splitlines() if l.startswith(("[", "  "))][:120]}
                    rc, out, err = run([str(bindir / "crispasr"), "--backend", diffname, "-m", gguf, "-f", wav[c], "-np", "-nt"],
                                       f"cli-{name}-{c}-{tag}.log")
                    C[f"cli_{tag}"] = out.strip() if rc == 0 else f"rc={rc}: {err[-300:]}"
                save()
        except BaseException:
            R["error"] = traceback.format_exc()[-2000:]
        save()
        shutil.rmtree(M / name, ignore_errors=True)
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
    for b in ("build-ctrl", "build-fix"):
        shutil.rmtree(WORK / b, ignore_errors=True)
