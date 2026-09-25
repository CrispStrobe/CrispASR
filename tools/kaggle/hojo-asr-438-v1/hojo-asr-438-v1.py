#!/usr/bin/env python3
"""Hojo-ASR (#438) follow-up: the English-capable Hojo-ASR-V1, and long audio.

Two things the reporter raised after Multi-V1 shipped:

1. HojoAI/Hojo-ASR-V1 (Apache-2.0, same Qwen3-Omni tower + adapter + Qwen3-4B
   layout) works better on English. Pipeline as for Multi-V1: control arm
   (upstream package) -> convert f16 -> upload -> build -> reference dump ->
   diff f16 -> quantize q4_k -> upload -> diff q4_k -> CLI transcripts.
   Uploaded to cstr/Hojo-ASR-V1-GGUF the moment each file exists.

2. "Output appears to be short" on a 28 s clip (the reporter's tt.wav).
   Hypothesis: config.yaml's flat max_new_tokens = 200, which upstream applies
   too, cuts the transcript. Readouts that can fail:
     - upstream control transcript of tt.wav (does upstream truncate as well?)
     - C++ with CRISPASR_HOJO_TOKENS_PER_SEC=0 (the old flat cap) must print the
       new "stopped at the N-token limit" warning if the hypothesis holds;
     - C++ default (duration-scaled cap) must end on <|im_end|> (no warning),
       and its text must START with the capped run's text (same greedy prefix).
   Run for both Multi-V1 (from HF) and V1.
"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BRANCH = os.environ.get("CRISPASR_REF", "fix/438-hojo-long")
SRC_REPO = "HojoAI/Hojo-ASR-V1"
HF_REPO = "cstr/Hojo-ASR-V1-GGUF"
MULTI_Q4 = ("cstr/Hojo-ASR-Multi-V1-GGUF", "hojo-asr-multi-v1-q4_k.gguf")
TT_URL = "https://github.com/user-attachments/files/32574766/tt.wav.zip"
res = {"steps": {}, "errors": [], "long": {}}


def save():
    (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def sec(t):
    print(f"\n{'=' * 78}\n== {t}\n{'=' * 78}", flush=True)


try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recursive", "--shallow-submodules", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    # every pip install before kaggle_harness imports huggingface_hub
    os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
    subprocess.check_call("pip install -q gguf safetensors huggingface_hub omegaconf soundfile "
                          "'transformers>=4.57.3,<5.0.0' openai-whisper && pip install -q --no-deps hojo-asr",
                          shell=True)
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok:
        os.environ["HF_TOKEN"] = tok
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(token=tok)

    def upload(local, name, repo=HF_REPO, rtype="model"):
        api.create_repo(repo_id=repo, repo_type=rtype, exist_ok=True, private=False)
        api.upload_file(path_or_fileobj=str(local), path_in_repo=name, repo_id=repo, repo_type=rtype)
        res["steps"][f"upload {name}"] = "ok"; save()

    sec("audio")
    import zipfile, io
    import numpy as np, soundfile as sf
    AUD = TEMP / "audio"; AUD.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(TT_URL, timeout=60).read())).extractall(AUD)
    tt = AUD / "tt.wav"
    d, sr = sf.read(tt)
    res["tt"] = {"sr": sr, "seconds": round(len(d) / sr, 2)}
    arms = [("jfk", REPO / "samples" / "jfk.wav"), ("tt", tt)]
    save()

    sec("download V1 + control arm (upstream package)")
    src = snapshot_download(repo_id=SRC_REPO, cache_dir=str(TEMP / "hojo-src"), token=tok)
    ctrl = TEMP / "control.py"
    ctrl.write_text(f'''
import json, sys
sys.path.insert(0, {str(REPO / "tools")!r})
from reference_backends.hojo_asr import bind_lm_dtype
from hojo_asr import HOJO_ASR
model = HOJO_ASR.load_model({src!r}, device="cpu"); model.eval(); bind_lm_dtype(model)
out = {{}}
for name, path in {[(n, str(p)) for n, p in arms]!r}:
    with open(path, "rb") as f:
        out[name] = model.run_infer([f.read()], batch_size=1, cuda_enabled=False)[0]["text"]
    print("CONTROL", name, repr(out[name]), flush=True)
json.dump(out, open({str(TEMP / "control.json")!r}, "w"), ensure_ascii=False)
''')
    rc = subprocess.run([sys.executable, str(ctrl)]).returncode
    res["control"] = json.load(open(TEMP / "control.json")) if rc == 0 else f"FAILED rc={rc}"
    save()

    sec("convert f16 + upload")
    F16 = TEMP / "hojo-asr-v1-f16.gguf"
    kh.sh_with_progress(f"{sys.executable} {REPO}/models/convert-hojo-asr-to-gguf.py --input {src} --output {F16}")
    res["steps"]["f16_gb"] = round(F16.stat().st_size / 1024**3, 2)
    upload(F16, F16.name)

    sec("build")
    kh.install_build_toolchain()
    flags = kh.cache_and_link_flags()
    kh.sh_with_progress(f"cmake -G Ninja -B {REPO}/build -S {REPO} -DCMAKE_BUILD_TYPE=Release " + " ".join(flags))
    kh.sh_with_progress(f"cmake --build {REPO}/build -j{kh.safe_build_jobs(False)} "
                        f"--target crispasr-quantize crispasr-diff crispasr-cli")
    BIN = REPO / "build" / "bin"
    res["steps"]["build"] = "ok"; save()

    sec("reference dump (jfk) + diff f16")
    ref = TEMP / "hojo-asr-v1-jfk-ref.gguf"
    rd = subprocess.run([sys.executable, str(REPO / "tools/dump_reference.py"), "--backend", "hojo-asr",
                         "--model-dir", src, "--audio", str(arms[0][1]), "--output", str(ref),
                         "--max-new-tokens", "200"], capture_output=True, text=True)
    (OUT / "ref-jfk.log").write_text(rd.stdout[-20000:] + rd.stderr[-5000:])
    res["ref_greedy"] = [l.split(":", 1)[1].strip() for l in rd.stdout.splitlines() if "generated_text_greedy:" in l]

    def diff(label, gguf):
        r = subprocess.run([str(BIN / "crispasr-diff"), "hojo-asr", str(gguf), str(ref), str(arms[0][1])],
                           capture_output=True, text=True)
        (OUT / f"diff-{label}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-5000:])
        res["diff_" + label] = {"rc": r.returncode, "rows": [l[:130] for l in r.stdout.splitlines() if l.startswith("[")]}
        save()

    if ref.exists():
        diff("f16", F16)

    sec("quantize q4_k + upload + diff")
    Q4 = TEMP / "hojo-asr-v1-q4_k.gguf"
    rc = subprocess.run([str(BIN / "crispasr-quantize"), str(F16), str(Q4), "q4_k"]).returncode
    if rc == 0 and Q4.exists():
        res["steps"]["q4_gb"] = round(Q4.stat().st_size / 1024**3, 2)
        upload(Q4, Q4.name)
        if ref.exists():
            diff("q4_k", Q4)

    sec("CLI transcripts + long-audio token cap A/B")
    multi_q4 = hf_hub_download(MULTI_Q4[0], MULTI_Q4[1], cache_dir=str(TEMP / "multi"), token=tok)
    models = {"v1-f16": F16, "v1-q4_k": Q4, "multi-q4_k": Path(multi_q4)}
    for mname, gguf in models.items():
        if not Path(gguf).exists():
            continue
        for aname, wav in arms:
            for cap, env_extra in (("flat200", {"CRISPASR_HOJO_TOKENS_PER_SEC": "0"}), ("scaled", {})):
                if aname == "jfk" and cap == "flat200":
                    continue
                env = dict(os.environ); env.update(env_extra)
                t0 = time.time()
                r = subprocess.run([str(BIN / "crispasr"), "-m", str(gguf), "--backend", "hojo-asr", "-f", str(wav),
                                    "-bs", "1", "-t", str(os.cpu_count() or 4)],
                                   capture_output=True, text=True, env=env)
                text = r.stdout.strip()
                res["long"][f"{mname}/{aname}/{cap}"] = {
                    "rc": r.returncode, "s": round(time.time() - t0, 1), "chars": len(text), "text": text[-2500:],
                    "hit_cap_warning": "token limit" in r.stderr}
                save()
except BaseException:
    import traceback
    res["errors"].append(traceback.format_exc())
finally:
    save()
