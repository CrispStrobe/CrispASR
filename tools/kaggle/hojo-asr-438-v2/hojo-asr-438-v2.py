#!/usr/bin/env python3
"""Hojo-ASR (#438) v2: HF-exact KV-cached beam search, repetition-penalty fix, V1 F32 gap.

For Hojo-ASR-Multi-V1 and Hojo-ASR-V1, on jfk / the reporter's tt.wav / the demo
Space's german + french clips:
  - upstream reference (tools/dump_reference.py, fp32 decoder on CPU): the
    package's beam-4 text (generate() as HOJO_ASR.infer calls it) and greedy text;
  - C++ F16 (from cstr/*-GGUF): default decode (now config.yaml's num_beams=4)
    and -bs 1 greedy, each compared EXACTLY with its upstream counterpart,
    with wall time;
  - C++ q4_k default decode (information only);
  - V1 only: an F32 conversion diffed stage by stage on jfk next to the F16 one -
    does the encoder per-channel cos_min 0.9969 (F16) close at F32?
The readout can fail: every text pair is printed side by side with a MATCH /
DIFF verdict, and a missing reference is reported as missing, not as a match.
"""
import json, os, re, shutil, subprocess, sys, time, urllib.request, zipfile, io
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR")  # NOT /kaggle/working: keeps the output download small
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BRANCH = os.environ.get("CRISPASR_REF", "fix/438-hojo-long")
MODELS = {
    "multi": ("HojoAI/Hojo-ASR-Multi-V1", "cstr/Hojo-ASR-Multi-V1-GGUF", "hojo-asr-multi-v1"),
    "v1": ("HojoAI/Hojo-ASR-V1", "cstr/Hojo-ASR-V1-GGUF", "hojo-asr-v1"),
}
DEMO_SPACE = "hugging-apps/hojo-asr-multi-v1-demo"
TT_URL = "https://github.com/user-attachments/files/32574766/tt.wav.zip"
res = {"errors": [], "texts": {}, "diff": {}, "steps": {}}


def save():
    (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def norm(t):
    t = (t or "").replace("<|im_end|>", "").replace("<|endoftext|>", "")
    t = re.sub(r"^\[[^\]]*\]\s*", "", t.strip(), flags=re.M)  # CLI timestamps, if any
    return " ".join(t.split())


try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recursive", "--shallow-submodules", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
    subprocess.check_call("pip install -q gguf safetensors huggingface_hub omegaconf soundfile "
                          "'transformers>=4.57.3,<5.0.0' openai-whisper && pip install -q --no-deps hojo-asr",
                          shell=True)
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok:
        os.environ["HF_TOKEN"] = tok
    from huggingface_hub import hf_hub_download, snapshot_download
    import numpy as np, soundfile as sf

    # ---- audio (normalised once to 16 kHz mono PCM16, shared by both sides)
    AUD = TEMP / "audio"; AUD.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(TT_URL, timeout=60).read())).extractall(AUD)
    arms = [("jfk", REPO / "samples" / "jfk.wav"), ("tt", AUD / "tt.wav")]
    for name, rel in (("german", "examples/german.wav"), ("french", "examples/french.wav")):
        try:
            p = hf_hub_download(repo_id=DEMO_SPACE, repo_type="space", filename=rel, cache_dir=str(TEMP / "demo"),
                                token=tok)
            w, sr = sf.read(p, dtype="float32", always_2d=True)
            m = w.mean(axis=1)
            if sr != 16000:
                import torch, torchaudio
                m = torchaudio.functional.resample(torch.from_numpy(m), sr, 16000).numpy()
            pk = float(np.abs(m).max()) if m.size else 0.0
            if pk > 1.0:
                m = m / pk
            dst = AUD / f"{name}.wav"; sf.write(dst, m, 16000, subtype="PCM_16"); arms.append((name, dst))
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"audio {name}: {e}")
    res["arms"] = {n: round(sf.info(str(p)).duration, 2) for n, p in arms}
    save()

    # ---- build
    kh.install_build_toolchain()
    flags = kh.cache_and_link_flags()
    kh.sh_with_progress(f"cmake -G Ninja -B {REPO}/build -S {REPO} -DCMAKE_BUILD_TYPE=Release " + " ".join(flags))
    kh.sh_with_progress(f"cmake --build {REPO}/build -j{kh.safe_build_jobs(False)} "
                        f"--target crispasr-diff crispasr-cli test-hojo-asr-frames")
    BIN = REPO / "build" / "bin"
    ut = subprocess.run([str(BIN / "test-hojo-asr-frames")], capture_output=True, text=True)
    res["steps"]["unit"] = {"rc": ut.returncode, "tail": ut.stdout[-800:]}
    save()

    def cli(gguf, wav, extra):
        t0 = time.time()
        r = subprocess.run([str(BIN / "crispasr"), "-m", str(gguf), "--backend", "hojo-asr", "-f", str(wav), "-np",
                            "-t", str(os.cpu_count() or 4)] + extra, capture_output=True, text=True)
        return norm(r.stdout), round(time.time() - t0, 1), r.returncode, r.stderr[-1500:]

    for mkey, (src_repo, gguf_repo, stem) in MODELS.items():
        src = snapshot_download(repo_id=src_repo, cache_dir=str(TEMP / f"src-{mkey}"), token=tok)
        f16 = hf_hub_download(gguf_repo, f"{stem}-f16.gguf", cache_dir=str(TEMP / f"g-{mkey}"), token=tok)
        q4 = hf_hub_download(gguf_repo, f"{stem}-q4_k.gguf", cache_dir=str(TEMP / f"g-{mkey}"), token=tok)
        for aname, wav in arms:
            key = f"{mkey}/{aname}"
            ref = TEMP / f"ref-{mkey}-{aname}.gguf"
            rd = subprocess.run([sys.executable, str(REPO / "tools/dump_reference.py"), "--backend", "hojo-asr",
                                 "--model-dir", src, "--audio", str(wav), "--output", str(ref),
                                 "--max-new-tokens", "200"], capture_output=True, text=True)
            (OUT / f"ref-{mkey}-{aname}.log").write_text(rd.stdout[-20000:] + rd.stderr[-4000:])
            rb = [l.split("):", 1)[1].strip() for l in rd.stdout.splitlines() if "generated_text (beams=" in l]
            rg = [l.split(":", 1)[1].strip() for l in rd.stdout.splitlines() if "generated_text_greedy:" in l]
            ref_beam = norm(rb[0].strip("'\"")) if rb else None
            ref_greedy = norm(rg[0].strip("'\"")) if rg else None
            e = {"ref_beam4": ref_beam, "ref_greedy": ref_greedy}
            for label, gguf, extra in (("f16_beam", f16, []), ("f16_greedy", f16, ["-bs", "1"]), ("q4k_beam", q4, [])):
                text, s_, rc, err = cli(gguf, wav, extra)
                e[label] = {"text": text, "s": s_, "rc": rc}
                if rc:
                    e[label]["stderr"] = err
            e["verdict_beam"] = "NO REF" if ref_beam is None else ("MATCH" if e["f16_beam"]["text"] == ref_beam else "DIFF")
            e["verdict_greedy"] = "NO REF" if ref_greedy is None else ("MATCH" if e["f16_greedy"]["text"] == ref_greedy else "DIFF")
            res["texts"][key] = e
            print(f"[{key}] beam {e['verdict_beam']} greedy {e['verdict_greedy']}", flush=True)
            if mkey == "v1" and aname == "jfk" and ref.exists():
                dargs = [str(BIN / "crispasr-diff"), "hojo-asr"]
                r = subprocess.run(dargs + [str(f16), str(ref), str(wav)], capture_output=True, text=True)
                res["diff"]["v1-f16/jfk"] = [l[:140] for l in r.stdout.splitlines() if l.startswith(("[", "  "))][:30]
                f32 = TEMP / "hojo-asr-v1-f32.gguf"
                kh.sh_with_progress(f"{sys.executable} {REPO}/models/convert-hojo-asr-to-gguf.py --input {src} "
                                    f"--output {f32} --outtype f32")
                r = subprocess.run(dargs + [str(f32), str(ref), str(wav)], capture_output=True, text=True)
                res["diff"]["v1-f32/jfk"] = [l[:140] for l in r.stdout.splitlines() if l.startswith(("[", "  "))][:30]
                text, s_, rc, _ = cli(f32, wav, [])
                e["f32_beam"] = {"text": text, "s": s_, "rc": rc}
                f32.unlink(missing_ok=True)
            save()
        shutil.rmtree(TEMP / f"src-{mkey}", ignore_errors=True)
        shutil.rmtree(TEMP / f"g-{mkey}", ignore_errors=True)
except BaseException:
    import traceback
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
