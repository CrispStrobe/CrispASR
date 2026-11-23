#!/usr/bin/env python3
"""
#609, second kernel: does CFG 4 stop the instruction recitation, and does the
UPSTREAM Python (breezeblue-ai/breeze-tts) run away on the same inputs?

Upstream's decode loop has no stopping rule beyond a sampled EOS and
max_new_tokens, like ours. Its README pairs every instruction with
`--cfg-scale 4`; the reporter runs cfg_scale 1.0.

Arms, on the #609 payloads (000002 control, 000206, 000214, 000384; the
reporter's 15-line instruction), voices Piedmont / mixed, seeds 87, 42, 0..N:
  ours_cfg4      CrispASR bt2-tts q4_k, CUDA, --tts-cfg-scale 4
  up_cfg1        upstream bf16 eager, cfg 1  (reporter's setting)
  up_cfg4        upstream bf16 eager, cfg 4  (upstream's documented setting)
Upstream runs with max_new_tokens 750 (its model default, 60 s). Every
output goes through the same parakeet roundtrip.

Results: /kaggle/working/bt2_609_cfg.json, rewritten after every run.
"""
import glob, json, os, re, shutil, subprocess, sys, time, wave, zipfile
from pathlib import Path

WORK = Path("/kaggle/working"); TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
RES = WORK / "bt2_609_cfg.json"
OUT = TMP / "wav"; OUT.mkdir(exist_ok=True)
SEEDS = [87, 42] + list(range(int(os.environ.get("N_SEEDS", "4"))))
UP_CAP = 750
T_END = time.time() + 11 * 3600


def sh(cmd, timeout=None, env=None):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout, env=env)


def clone(url, dst, branch=None):
    for _ in range(4):
        if dst.exists():
            shutil.rmtree(dst)
        cmd = ["git", "clone", "--depth", "1"] + (["--branch", branch, "--recurse-submodules", "--shallow-submodules"] if branch else []) + [url, str(dst)]
        if subprocess.run(cmd, timeout=1800).returncode == 0:
            return
        time.sleep(15)
    raise SystemExit("clone failed " + url)


CR = TMP / "CrispASR"; UP = TMP / "breeze-tts"; CKPT = TMP / "breeze-tts-2"
if not CR.exists():
    clone("https://github.com/CrispStrobe/CrispASR.git", CR, os.environ.get("CRISPASR_BRANCH", "test/609"))
if not UP.exists():
    clone("https://github.com/breezeblue-ai/breeze-tts.git", UP)
sys.path.insert(0, str(CR / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
kh.install_build_toolchain()
res = {"started": time.strftime("%F %T"), "seeds": SEEDS, "runs": []}
save = lambda: RES.write_text(json.dumps(res, indent=1))
res["crispasr_head"] = sh(f"git -C {CR} rev-parse HEAD").stdout.strip()
res["upstream_head"] = sh(f"git -C {UP} rev-parse HEAD").stdout.strip()
save()

DS = Path(next(p for p in glob.glob("/kaggle/input/**/crispasr-609-repro*", recursive=True) if os.path.isdir(p)))
DATA = TMP / "data"
if not DATA.exists():
    shutil.copytree(DS, DATA)
    for z in DATA.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.with_suffix(""))
find = lambda pat: sorted(DATA.rglob(pat))[0]
PAYLOADS = [json.loads(find(f"payloads_r1/{n}.json").read_text()) | {"id": n} for n in ["000002", "000206", "000214", "000384"]]
VOICES = {v: (str(find(f"{v}.wav")), find(f"{v}.txt").read_text().strip()) for v in ["Piedmont", "mixed"]}

# ---- build ours ----
B = TMP / "b"
flags = ["-G Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DCRISPASR_OPUS_FETCH=ON"] + \
    kh.cuda_build_flags() + kh.cache_and_link_flags()
r = sh(f"cmake -S {CR} -B {B} " + " ".join(flags), timeout=1800)
if r.returncode:
    res["configure_error"] = r.stderr[-3000:]; save(); raise SystemExit(1)
with kh.build_heartbeat("build", interval_s=120):
    kh.sh_with_progress(f"cmake --build {B} -j{kh.safe_build_jobs(gpu=True)} --target crispasr-cli")
CLI = str(B / "bin" / "crispasr")

sh(f"{sys.executable} -m pip install -q transformers==4.57.3 qwen-tts==0.1.1 soundfile huggingface_hub safetensors")
from huggingface_hub import hf_hub_download, snapshot_download
HF = kh.resolve_hf_token() or None
M = TMP / "models"
BREEZE = hf_hub_download("cstr/breeze-tts-2-GGUF", "breeze-tts-2-q4_k.gguf", local_dir=str(M), token=HF)
CODEC = hf_hub_download("cstr/qwen3-tts-tokenizer-12hz-GGUF", "qwen3-tts-tokenizer-12hz.gguf", local_dir=str(M), token=HF)
ASR = hf_hub_download("cstr/parakeet-tdt-0.6b-v3-GGUF", "parakeet-tdt-0.6b-v3-q4_k.gguf", local_dir=str(M), token=HF)


def norm(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def record(arm, p, voice, seed, wav, secs, extra=None):
    dur = None
    if os.path.exists(wav):
        with wave.open(wav) as w:
            dur = w.getnframes() / w.getframerate()
    text = ""
    if dur:
        r = sh([CLI, "-m", ASR, "-f", wav, "-nt", "-np"], timeout=900)
        text = " ".join(l.strip() for l in r.stdout.splitlines() if l.strip())
    rec = {"arm": arm, "id": p["id"], "voice": voice, "seed": seed, "s": round(secs, 1), "dur": dur,
           "asr": text, "extra_words": max(0, len(norm(text)) - len(norm(p["input"]))), **(extra or {})}
    res["runs"].append(rec); save()
    print(arm, p["id"], voice, seed, dur, rec["extra_words"], flush=True)
    # One progress line per run, so the HF mirror shows how far a long arm has got.
    kh.step("run", arm=arm, id=p["id"], voice=voice, seed=seed, dur=dur, extra=rec["extra_words"], s=rec["s"])


# ---- arm 1: ours, cfg 4 ----
kh.step("ours_cfg4")
COMMON = ["--backend", "bt2-tts", "-m", BREEZE, "--codec-model", CODEC, "--accept-license", "other",
          "--i-have-rights", "--speaker-identity", "synthetic", "--no-watermark",
          "--accept-marking-responsibility", "--no-spoken-disclaimer", "-n", "1500"]
for p in PAYLOADS:
    for voice, (vw, vt) in VOICES.items():
        for s in SEEDS:
            wav = str(OUT / f"ours4_{p['id']}_{voice}_{s}.wav")
            t0 = time.time()
            r = sh([CLI] + COMMON + ["--tts", p["input"], "--instruct", p["instructions"], "--voice", vw,
                                     "--ref-text", vt, "--seed", str(s), "--tts-cfg-scale", "4", "--tts-output", wav],
                   timeout=3600)
            record("ours_cfg4", p, voice, s, wav, time.time() - t0, {"rc": r.returncode, "err": r.stderr[-500:] if r.returncode else ""})

# ---- arms 2+3: upstream, cfg 1 and 4 ----
kh.step("upstream_load")
snapshot_download("BreezeBlue/Breeze-TTS-2", local_dir=str(CKPT), token=HF,
                  allow_patterns=["*.json", "*.safetensors", "LICENSE", "audio_tokenizer/*", "*.model", "*.txt"])
sys.path.insert(0, str(UP))
os.chdir(UP)
import soundfile as sf
import torch
from breeze_infer.runtime import load_runtime, resolve_device, set_all_seeds, update_generation_config_for_breeze
from breeze_infer.templates import get_template, prepare_inputs, select_template_name
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

torch.set_grad_enabled(False)
res["gpu"] = torch.cuda.get_device_name(0); save()
tok, model, atok = load_runtime(CKPT, device=resolve_device(), attn_implementation="eager")
update_generation_config_for_breeze(model)
rt = FastBreezeStreamingRuntime(model, atok, FastStreamingConfig(max_new_tokens=UP_CAP, max_seq_len=2048,
                                                                  repetition_penalty=1.1), tokenizer=tok)
kh.step("upstream_runs")
for p in PAYLOADS:
    for voice, (vw, vt) in VOICES.items():
        for s in SEEDS:
            for cfg in (1.0, 4.0):
                if time.time() > T_END:
                    break
                req = {"id": "r", "text": p["input"], "speaker": "S0", "instruction": p["instructions"].strip(),
                       "ref_audio_path": vw, "ref_text": vt}
                wav = str(OUT / f"up{int(cfg)}_{p['id']}_{voice}_{s}.wav")
                t0 = time.time()
                try:
                    set_all_seeds(s)
                    inputs = prepare_inputs(tok, atok, model, [req], get_template(select_template_name(req)),
                                            guidance_scale=cfg, guidance_scale_ref=None, guidance_scale_ins=None)
                    with sf.SoundFile(wav, mode="w", samplerate=rt.sample_rate, channels=1, subtype="PCM_16") as f:
                        for ch in rt.iter_audio_chunks(inputs, request_id="r", seed=s):
                            f.write(ch.audio)
                    err = ""
                except Exception as e:
                    err = repr(e)[-500:]
                record(f"up_cfg{int(cfg)}", p, voice, s, wav, time.time() - t0, {"err": err})

# ---- summary ----
summ = {}
for r in res["runs"]:
    a = summ.setdefault(r["arm"], {"n": 0, "runaway": 0, "by_id": {}})
    a["n"] += 1
    bad = r["extra_words"] >= 4
    a["runaway"] += bad
    a["by_id"].setdefault(r["id"], [0, 0])
    a["by_id"][r["id"]][0] += bad; a["by_id"][r["id"]][1] += 1
res["summary"] = summ
res["finished"] = time.strftime("%F %T")
save()
kh.step("done")
print(json.dumps(summ, indent=1))
