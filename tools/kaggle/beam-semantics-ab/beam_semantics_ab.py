#!/usr/bin/env python3
"""A/B of core_beam_decode's search semantics against upstream transformers generate().

Arms: CRISPASR_BEAM_SEMANTICS=legacy (the original loop) vs =hf (transformers
4.57 _beam_search). Reference: the upstream checkpoint's own generate() with
the same beam width, length cap and generation_config, fp32 on CPU. The C++
side uses the least-quantised GGUF on HF (F16/F32) so precision is not the
variable. Readout per case: normalised text of legacy / hf / reference, and an
EXACT-match verdict per arm - a case with no reference says so.

  m2m100-418m  text en->de/fr, DEFAULT beam 5 (generation_config: early_stopping true)
  madlad-3b    text en->de, -bs 4
  moonshine-tiny  librispeech_asr_dummy x8, -bs 4 (max_new = ceil(s * 6.5), as the C++)
  granite-speech-4.0-1b  librispeech_asr_dummy x8, -bs 4, 200 tokens
Also runs tests/test-beam-hf (the toy-model fixtures) on the build.
"""
import json, math, os, re, shutil, subprocess, sys, time, traceback
from pathlib import Path

WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/tmp/CrispASR")
TEMP = Path("/tmp/ab"); TEMP.mkdir(parents=True, exist_ok=True)
BRANCH = os.environ.get("CRISPASR_REF", "feat/beam-hf-semantics")
res = {"errors": [], "cases": {}, "summary": {}}
# run 1 settled m2m100 (hf 6/6 vs legacy 5/6) and madlad (4/4 vs 3/4); run 2 = the two arms the harness broke
GROUPS = set(os.environ.get("AB_GROUPS", "moonshine,granite").split(","))
TEXTS = [
    ("en", "de", "Hello world, how are you today?"),
    ("en", "de", "The president said he would not attend the meeting on Thursday."),
    ("en", "fr", "Machine translation has improved a lot over the last ten years."),
    ("en", "de", "Please close the window before you leave the house."),
    ("en", "fr", "The weather will be sunny tomorrow, with a light breeze in the afternoon."),
    ("en", "de", "I have to take those sheep back out, because the buyer changed his mind."),
]


def save():
    (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def norm(t):
    t = re.sub(r"^\[[^\]]*\]\s*", "", (t or "").strip(), flags=re.M)
    return " ".join(t.split())


def record(group, key, ref, legacy, hf, extra=None):
    e = {"ref": ref, "legacy": legacy["text"], "hf": hf["text"], "t_legacy": legacy["s"], "t_hf": hf["s"],
         "legacy_match": None if ref is None else legacy["text"] == ref,
         "hf_match": None if ref is None else hf["text"] == ref}
    if extra:
        e.update(extra)
    res["cases"][f"{group}/{key}"] = e
    s = res["summary"].setdefault(group, {"n": 0, "legacy_exact": 0, "hf_exact": 0, "arms_differ": 0})
    s["n"] += 1
    s["legacy_exact"] += bool(e["legacy_match"])
    s["hf_exact"] += bool(e["hf_match"])
    s["arms_differ"] += legacy["text"] != hf["text"]
    print(f"[{group}/{key}] legacy={e['legacy_match']} hf={e['hf_match']}", flush=True)
    save()


try:
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    res["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.3", "sentencepiece",
                           "soundfile", "datasets<4"])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    tok = kh.resolve_hf_token()
    if tok:
        os.environ["HF_TOKEN"] = tok
    import numpy as np, soundfile as sf, torch
    import transformers
    from huggingface_hub import hf_hub_download
    res["transformers"] = transformers.__version__
    torch.set_num_threads(os.cpu_count() or 4)

    kh.install_build_toolchain()
    flags = kh.cache_and_link_flags()
    with kh.build_heartbeat("build"):
        kh.sh(f"cmake -S {REPO} -B {REPO}/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPASR_OPUS=OFF "
              f"-DCRISPASR_AMR=OFF " + " ".join(flags))
        kh.sh(f"cmake --build {REPO}/build -j$(nproc) --target crispasr-cli test-beam-hf")
    BIN = REPO / "build" / "bin"
    t = subprocess.run([str(BIN / "test-beam-hf")], capture_output=True, text=True)
    res["unit_test"] = {"rc": t.returncode, "tail": t.stdout[-400:]}
    save()

    def cli(args, arm):
        env = dict(os.environ); env["CRISPASR_BEAM_SEMANTICS"] = arm
        t0 = time.time()
        r = subprocess.run([str(BIN / "crispasr"), "-np", "-t", str(os.cpu_count() or 4)] + args,
                           capture_output=True, text=True, env=env)
        out = {"text": norm(r.stdout), "s": round(time.time() - t0, 1)}
        if r.returncode:
            out["text"] = f"<rc={r.returncode}> " + r.stderr[-400:]
        return out

    # ---- librispeech dummy clips (8)
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    clips = []
    for i in range(8):
        a = ds[i]["audio"]
        p = TEMP / f"ls{i}.wav"
        sf.write(p, np.asarray(a["array"], dtype=np.float32), a["sampling_rate"], subtype="PCM_16")
        clips.append((f"ls{i}", p, ds[i]["text"]))

    # ---- m2m100-418m: default beam 5
    try:
        if "m2m100" not in GROUPS:
            raise StopIteration
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        g = hf_hub_download("cstr/m2m100-418m-GGUF", "m2m100-418m-f16.gguf", cache_dir=str(TEMP / "g"))
        tk = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        md = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", torch_dtype=torch.float32).eval()
        for i, (sl, tl, text) in enumerate(TEXTS):
            tk.src_lang = sl
            with torch.no_grad():
                gen = md.generate(**tk(text, return_tensors="pt"), forced_bos_token_id=tk.get_lang_id(tl))
            ref = norm(tk.batch_decode(gen, skip_special_tokens=True)[0])
            args = ["--backend", "m2m100", "-m", g, "--text", text, "-sl", sl, "-tl", tl]
            record("m2m100", f"t{i}", ref, cli(args, "legacy"), cli(args, "hf"))
        del md
    except StopIteration:
        pass
    except Exception:
        res["errors"].append("m2m100: " + traceback.format_exc()[-1500:]); save()

    # ---- moonshine-tiny: -bs 4
    try:
        if "moonshine" not in GROUPS:
            raise StopIteration
        from transformers import AutoProcessor, MoonshineForConditionalGeneration
        g = hf_hub_download("cstr/moonshine-tiny-GGUF", "moonshine-tiny.gguf", cache_dir=str(TEMP / "g"))
        hf_hub_download("cstr/moonshine-tiny-GGUF", "tokenizer.bin", cache_dir=str(TEMP / "g"))  # read next to the GGUF
        pr = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")
        md = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-tiny",
                                                               torch_dtype=torch.float32).eval()
        for key, p, truth in clips:
            wav, sr = sf.read(p, dtype="float32")
            mx = int(math.ceil(len(wav) / 16000 * 6.5))
            with torch.no_grad():
                gen = md.generate(**pr(wav, sampling_rate=16000, return_tensors="pt"), num_beams=4, max_new_tokens=mx)
            ref = norm(pr.batch_decode(gen, skip_special_tokens=True)[0])
            args = ["--backend", "moonshine", "-m", g, "-f", str(p), "-bs", "4"]
            record("moonshine-tiny", key, ref, cli(args, "legacy"), cli(args, "hf"), {"truth": truth})
        del md
    except StopIteration:
        pass
    except Exception:
        res["errors"].append("moonshine: " + traceback.format_exc()[-1500:]); save()

    # ---- granite-speech-4.0-1b: -bs 4
    try:
        if "granite" not in GROUPS:
            raise StopIteration
        from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration
        g = hf_hub_download("cstr/granite-speech-4.0-1b-GGUF", "granite-speech-4.0-1b-f16.gguf",
                            cache_dir=str(TEMP / "g"))
        rid = "ibm-granite/granite-4.0-1b-speech"
        pr = AutoProcessor.from_pretrained(rid)
        md = GraniteSpeechForConditionalGeneration.from_pretrained(rid, torch_dtype=torch.float32).eval()
        chat = [{"role": "user", "content": "<|audio|>can you transcribe the speech into a written format?"}]
        prompt = pr.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        for key, p, truth in clips:
            wav, sr = sf.read(p, dtype="float32")
            inp = pr(prompt, torch.from_numpy(wav).unsqueeze(0), device="cpu", return_tensors="pt")
            with torch.no_grad():
                gen = md.generate(**inp, num_beams=4, max_new_tokens=200, do_sample=False)
            ref = norm(pr.tokenizer.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True))
            args = ["--backend", "granite", "-m", g, "-f", str(p), "-bs", "4"]
            record("granite-4.0-1b", key, ref, cli(args, "legacy"), cli(args, "hf"), {"truth": truth})
        del md
    except StopIteration:
        pass
    except Exception:
        res["errors"].append("granite: " + traceback.format_exc()[-1500:]); save()

    # ---- madlad-3b: -bs 4, 64 tokens (heaviest last)
    try:
        if "madlad" not in GROUPS:
            raise StopIteration
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        g = hf_hub_download("cstr/madlad400-3b-mt-GGUF", "madlad400-3b-mt-f16.gguf", cache_dir=str(TEMP / "g"))
        tk = T5Tokenizer.from_pretrained("google/madlad400-3b-mt")
        md = T5ForConditionalGeneration.from_pretrained("google/madlad400-3b-mt", torch_dtype=torch.float32).eval()
        for i, (sl, tl, text) in enumerate(TEXTS[:4]):
            with torch.no_grad():
                gen = md.generate(**tk(f"<2{tl}> {text}", return_tensors="pt"), num_beams=4, max_new_tokens=64)
            ref = norm(tk.decode(gen[0], skip_special_tokens=True))
            args = ["--backend", "madlad", "-m", g, "--text", text, "-sl", sl, "-tl", tl, "-bs", "4",
                    "--translate-max-tokens", "64"]
            record("madlad-3b", f"t{i}", ref, cli(args, "legacy"), cli(args, "hf"))
        del md
    except StopIteration:
        pass
    except Exception:
        res["errors"].append("madlad: " + traceback.format_exc()[-1500:]); save()
except BaseException:
    res["errors"].append(traceback.format_exc())
finally:
    save()
    shutil.rmtree(REPO, ignore_errors=True)
