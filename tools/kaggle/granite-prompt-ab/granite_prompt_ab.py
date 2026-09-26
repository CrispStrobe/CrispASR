#!/usr/bin/env python3
"""granite-4.0-1b-speech prompt-token fix: main vs fix/granite-prompt-tokens vs upstream generate().

The default prompt's hard-coded ids had "?" + "\\n" where apply_chat_template gives
the single token "?\\n" (5380); non-default prompts lost the "\\n" entirely. The
beam A/B saw our output punctuated ("mr. quilter ... gospel.") where upstream is
"mister quilter ... gospel". Readout per clip: HF greedy / HF beam4 (fp32) vs
C++ F16 greedy (-bs 1) and beam (-bs 4) on both builds, exact text match; plus
the -l en path (runtime tokenizer) on the fix build vs HF with that prompt.
"""
import json, os, re, shutil, subprocess, sys, time, traceback
from pathlib import Path

WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR"); T = Path("/tmp/g"); T.mkdir(parents=True, exist_ok=True)
res = {"errors": [], "builds": {}, "cases": {}, "summary": {}}


def save():
    (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def norm(t):
    return " ".join(re.sub(r"^\[[^\]]*\]\s*", "", (t or "").strip(), flags=re.M).split())


try:
    subprocess.check_call(["git", "clone", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.check_call(["git", "fetch", "origin", "fix/granite-prompt-tokens:fixg"], cwd=str(REPO))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.3", "soundfile", "datasets<4"])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok:
        os.environ["HF_TOKEN"] = tok
    import numpy as np, soundfile as sf, torch
    from huggingface_hub import hf_hub_download
    kh.install_build_toolchain()
    flags = " ".join(kh.cache_and_link_flags())
    bins = {}
    for name, ref in (("main", "origin/main"), ("fix", "fixg")):
        wt = Path(f"/tmp/wt-{name}")
        subprocess.check_call(["git", "worktree", "add", "-f", str(wt), ref], cwd=str(REPO))
        subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=str(wt))
        res["builds"][name] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(wt), text=True).strip()
        with kh.build_heartbeat(f"build.{name}"):
            kh.sh(f"cmake -S {wt} -B {wt}/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF "
                  f"-DCRISPASR_AMR=OFF {flags}")
            kh.sh(f"cmake --build {wt}/build -j$(nproc) --target crispasr-cli")
        bins[name] = wt / "build" / "bin" / "crispasr"
        save()
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration
    rid = "ibm-granite/granite-4.0-1b-speech"
    g = hf_hub_download("cstr/granite-speech-4.0-1b-GGUF", "granite-speech-4.0-1b-f16.gguf", cache_dir=str(T))
    pr = AutoProcessor.from_pretrained(rid)
    md = GraniteSpeechForConditionalGeneration.from_pretrained(rid, torch_dtype=torch.float32).eval()
    prompts = {"default": "can you transcribe the speech into a written format?", "lang_en": "can you transcribe the speech into English?"}
    torch.set_num_threads(os.cpu_count() or 4)
    for i in range(8):
        a = ds[i]["audio"]; wav = T / f"ls{i}.wav"
        x = np.asarray(a["array"], dtype=np.float32)
        sf.write(wav, x, a["sampling_rate"], subtype="PCM_16")
        for pname, ptxt in prompts.items():
            chat = [{"role": "user", "content": "<|audio|>" + ptxt}]
            prompt = pr.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inp = pr(prompt, torch.from_numpy(x).unsqueeze(0), device="cpu", return_tensors="pt")
            refs = {}
            with torch.no_grad():
                for nb in (1, 4):
                    gen = md.generate(**inp, num_beams=nb, max_new_tokens=200, do_sample=False)
                    refs[nb] = norm(pr.tokenizer.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True))
            e = {"ref_greedy": refs[1], "ref_beam4": refs[4], "truth": ds[i]["text"]}
            builds = ("main", "fix") if pname == "default" else ("fix", "main")
            for b in builds:
                for nb in (1, 4):
                    args = [str(bins[b]), "--backend", "granite", "-m", g, "-f", str(wav), "-np", "-bs", str(nb),
                            "-t", str(os.cpu_count() or 4)] + (["-l", "en"] if pname == "lang_en" else [])
                    r = subprocess.run(args, capture_output=True, text=True)
                    txt = norm(r.stdout) if r.returncode == 0 else f"<rc={r.returncode}> {r.stderr[-300:]}"
                    e[f"{b}_bs{nb}"] = txt
                    k = f"{pname}/{b}/bs{nb}"
                    s = res["summary"].setdefault(k, {"n": 0, "exact": 0})
                    s["n"] += 1; s["exact"] += txt == refs[nb]
            res["cases"][f"ls{i}/{pname}"] = e
            save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
