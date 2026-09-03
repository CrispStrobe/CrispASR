#!/usr/bin/env python3
"""Raon 1B real-cond ODE diff (#387-adj) — the decisive, controlled run.

Everything in isolation is byte-exact to the reference (DiT, euler, ref-mel).
Real synthesis still diverges. The one thing no probe exercised: the ODE under
the REAL zero-padded cond ([ref_mel ; zeros] over the generated region). Test it
with two controls (per crispasr-dc):
  ARM 1  CONTROL (must converge): reference euler, real cond, duration=2*ref_T
         (the ref-dump's known-good setting). If this NaNs → harness bug, not a
         finding.
  ARM 2  reference euler, real cond, duration = C++'s formula (ref_T+rate*text).
  ARM 3  C++ euler (ODE probe), SAME cond + SAME x0 + SAME tokens as ARM 2.
x0 is FIXED across ARM2/ARM3 so the only variable is the code path; ARM1 vs ARM2
isolates the duration formula; ARM2 vs ARM3 isolates C++ euler vs reference.
RAON_SIZE=1B; chr1s4.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

WORK = Path("/kaggle/working"); TMP = Path("/kaggle/temp"); TMP.mkdir(exist_ok=True)
PROBE = TMP / "ode"; PROBE.mkdir(exist_ok=True)
SIZE = os.environ.get("RAON_SIZE", "1B")
REPO = f"KRAFTON/Raon-OpenTTS-{SIZE}"
CKPT_FILE = {"0.3B": "model_225000.pt", "1B": "model_520000.pt"}[SIZE]
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/raon-opentts-1b")
RAON_URL = "https://github.com/krafton-ai/Raon-OpenTTS.git"
CLONE = TMP / "CrispASR"; RAON = TMP / "Raon-OpenTTS"


def sh(cmd, **kw): return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
def step(name, **kv): print(f"[{time.strftime('%H:%M:%S')}] {name} " + json.dumps(kv), flush=True)


for url, dst, ref in ((CRISPASR_URL, CLONE, CRISPASR_REF), (RAON_URL, RAON, None)):
    if not dst.exists():
        for _ in range(4):
            if subprocess.run(["git", "clone", "--depth", "1"] + (["--branch", ref] if ref else []) + [url, str(dst)]).returncode == 0:
                break
            time.sleep(15)
for _ in range(4):
    if subprocess.run(["git", "submodule", "update", "--init", "--recursive", "ggml", "third_party/c2pa-audio"],
                      cwd=str(CLONE)).returncode == 0 or (CLONE / "ggml" / "CMakeLists.txt").exists():
        break
    time.sleep(15)
sys.path.insert(0, str(CLONE / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress()
HF_TOKEN = kh.resolve_hf_token()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "x_transformers", "torchdiffeq", "ema_pytorch",
                "loguru", "einops", "jieba", "pypinyin", "hydra-core", "omegaconf", "vocos", "pyyaml",
                "soundfile", "huggingface_hub"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "gguf"], check=False)
sys.path.insert(0, str(RAON / "src"))
from huggingface_hub import hf_hub_download  # noqa: E402
import torch, yaml  # noqa: E402

MODELS = TMP / "models"; MODELS.mkdir(exist_ok=True)
def dl(repo, f, sub=""):
    d = MODELS / sub if sub else MODELS; d.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(repo, f, local_dir=str(d), token=HF_TOKEN or None)


kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
flags = ["-DCMAKE_BUILD_TYPE=Release"] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
r = sh(f"cd {CLONE} && cmake -G Ninja -B build " + " ".join(flags), timeout=1200)
if r.returncode != 0:
    step("cmake_FAIL", err=r.stderr[-1500:]); sys.exit(1)
with kh.build_heartbeat("build", interval_s=30):
    kh.sh_with_progress(f"cmake --build build -j{kh.safe_build_jobs(gpu=True)} --target crispasr-cli", cwd=str(CLONE))
CLI = CLONE / "build" / "bin" / "crispasr"
if not CLI.exists():
    c = [p for p in (CLONE / "build").rglob("crispasr") if p.is_file() and os.access(p, os.X_OK)]
    CLI = c[0] if c else None
os.environ["LD_LIBRARY_PATH"] = str(CLI.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
gguf = hf_hub_download(f"cstr/raon-opentts-{SIZE.lower()}-GGUF", f"raon-opentts-{SIZE.lower()}-f16.gguf",
                       local_dir=str(MODELS), token=HF_TOKEN or None)
ref_wav = CLONE / "samples" / "jfk.wav"
step("built", gguf=os.path.basename(gguf))

# ── reference model + real ref-mel of jfk ──────────────────────────────────
ckpt = dl(REPO, CKPT_FILE, SIZE); cfg = dl(REPO, "config.yaml", SIZE); vocab = dl(REPO, "vocab.txt", SIZE)
from f5_tts.model.backbones.dit import DiT  # noqa: E402
from f5_tts.model.modules import MelSpec  # noqa: E402
from f5_tts.model.utils import get_tokenizer, list_str_to_idx  # noqa: E402
conf = yaml.safe_load(open(cfg)); a = conf["model"]["arch"]; mspec = conf["model"]["mel_spec"]
n_mel = mspec["n_mel_channels"]
sd = torch.load(ckpt, map_location="cpu", weights_only=True)["ema_model_state_dict"]
sd = {k.replace("ema_model.transformer.", ""): v for k, v in sd.items() if k.startswith("ema_model.transformer.")}
rows = int(sd["text_embed.text_embed.weight"].shape[0])
vmap, _ = get_tokenizer(vocab, "custom"); vmap = {c: i for c, i in vmap.items() if i < rows - 1}
dit = DiT(**{k: a[k] for k in a if k != "name"}, mel_dim=n_mel, text_num_embeds=rows - 1)
dit.load_state_dict(sd, strict=False); dit.eval()
# DiT has no stft, so it runs on P100 even though the mel extractor doesn't;
# probe a tiny forward and fall back to CPU if the GPU lacks a kernel.
DEV = "cpu"
if torch.cuda.is_available():
    try:
        _d = dit.cuda()
        with torch.no_grad():
            _d(torch.randn(1, 8, n_mel, device="cuda"), torch.randn(1, 8, n_mel, device="cuda"),
               torch.zeros(1, 4, dtype=torch.long, device="cuda"), torch.tensor([0.5], device="cuda"),
               drop_audio_cond=False, drop_text=False, cfg_infer=False)
        DEV = "cuda"; dit = _d
    except Exception as e:
        print("GPU DiT probe failed, using CPU:", str(e)[:200], flush=True)
        dit = dit.cpu()
step("device", dev=DEV)

# jfk ref audio (16k) + its real sbhifigan mel
import wave as _w
_wf = _w.open(str(ref_wav), "rb"); _sr = _wf.getframerate(); _n = _wf.getnframes()
_a = np.frombuffer(_wf.readframes(_n), dtype=np.int16).astype(np.float32) / 32768.0; _wf.close()
wav = torch.from_numpy(_a).unsqueeze(0)
ms = MelSpec(**{**mspec})
with torch.no_grad():
    ref_mel = ms(wav)[0].T.contiguous()   # (ref_T, n_mel)
ref_T = ref_mel.shape[0]
ref_text = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
gen_text = "The quick brown fox jumps over the lazy dog."
full_text = ref_text + gen_text
tok_ids = list_str_to_idx([list(full_text)], vmap)                # (1, nt)
tok = tok_ids[0].numpy().astype(np.int32)
# C++ duration: ref_T + round(rate * gen_len), rate = ref_T / ref_text_len
rate = ref_T / max(1, len(ref_text))
genT_cpp = int(rate * len(gen_text))
T_cpp = ref_T + genT_cpp
T_ctrl = 2 * ref_T
STEPS, SWAY, CFG = 32, -1.0, 2.0
sched = torch.linspace(0, 1, STEPS + 1); sched = sched + SWAY * (torch.cos(torch.pi / 2 * sched) - 1 + sched)
step("setup", ref_T=ref_T, genT_cpp=genT_cpp, T_cpp=T_cpp, T_ctrl=T_ctrl, nt=int(tok.size))


tok_dev = tok_ids.to(DEV)


def make_cond(T):
    c = torch.zeros(1, T, n_mel)
    c[0, :min(ref_T, T)] = ref_mel[:min(ref_T, T)]
    return c.to(DEV)


def ref_euler(T, cond, x0, label):
    x = x0.clone().to(DEV); stds = [round(float(x.std()), 3)]
    with torch.no_grad():
        for k in range(STEPS):
            tv = sched[k].item(); dt = (sched[k + 1] - sched[k]).item(); tt = torch.tensor([tv], device=DEV)
            vc = dit(x, cond, tok_dev, tt, drop_audio_cond=False, drop_text=False, cfg_infer=False)
            vu = dit(x, cond, tok_dev, tt, drop_audio_cond=True, drop_text=True, cfg_infer=False)
            x = x + (vc + CFG * (vc - vu)) * dt
            s = float(x.std()) if torch.isfinite(x).all() else float("nan")
            stds.append(round(s, 3) if s == s else None)
            if s != s:
                break
    step(label, T=T, n=len(stds), final_std=stds[-1])
    return x.cpu(), stds


torch.manual_seed(387)
x0_ctrl = torch.randn(1, T_ctrl, n_mel)
x0_test = torch.randn(1, T_cpp, n_mel)      # FIXED, shared by ARM2 (ref) and ARM3 (C++)

# ARM 1 — control (must converge)
_, ctrl_std = ref_euler(T_ctrl, make_cond(T_ctrl), x0_ctrl, "ARM1_control")
# ARM 2 — reference, real cond, C++'s duration
_, arm2_std = ref_euler(T_cpp, make_cond(T_cpp), x0_test, "ARM2_ref_cppdur")

# ARM 3 — C++ euler, identical cond + x0 + tokens
(PROBE / "shape.txt").write_text(f"{T_cpp} {tok.size}")
x0_test[0].contiguous().numpy().astype(np.float32).tofile(PROBE / "x0.bin")
make_cond(T_cpp)[0].contiguous().numpy().astype(np.float32).tofile(PROBE / "cond.bin")
tok.tofile(PROBE / "tokens.bin")
env = dict(os.environ); env["CRISPASR_F5_ODE_PROBE"] = str(PROBE)
rc = sh(f"{CLI} --backend raon-1b -m {gguf} --voice {ref_wav} --ref-text x --tts probe "
        f"--tts-output {WORK/'o.wav'} -t 4 --seed 42 --i-have-rights -v", env=env, timeout=1800)
print(rc.stderr[-1500:], flush=True)

def cpp_std(k):
    p = PROBE / f"ode_step_{k}.bin"
    if not p.exists():
        return None
    v = np.fromfile(p, dtype=np.float32)
    return round(float(v.std()), 3) if np.isfinite(v).all() else None

arm3_std = [cpp_std(k) for k in range(STEPS + 1)]
# per-step cos ARM2 vs ARM3
def arm2_x(): pass
result = {"size": SIZE, "ref_T": ref_T, "T_ctrl": T_ctrl, "T_cpp": T_cpp, "genT_cpp": genT_cpp,
          "ARM1_control_std": ctrl_std, "ARM2_ref_cppdur_std": arm2_std, "ARM3_cpp_std": arm3_std,
          "control_converged": (ctrl_std[-1] is not None and ctrl_std[-1] < 1.5),
          "arm2_diverged": (arm2_std[-1] is None or (arm2_std[-1] or 9) > 1.5),
          "arm3_diverged": (arm3_std[-1] is None or (arm3_std[-1] or 9) > 1.5)}
(WORK / "raon_realcond.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
step("DONE", control_ok=result["control_converged"], arm2_div=result["arm2_diverged"], arm3_div=result["arm3_diverged"])
