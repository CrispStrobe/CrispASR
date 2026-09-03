#!/usr/bin/env python3
"""Raon 1B ODE trajectory diff (#387-adj) — localize the compounding divergence.

The free-running trajectory grows (std 1.0→1.6, NaN by step 20) while the
reference converges. Matched per-stage snapshots are blind to this (the model
eats its own output). So: give BOTH the SAME x0/cond/tokens/schedule, run each
euler loop FREE, and diff x at every step. The first step whose cos/norm_ratio
departs from 1.0 is where our per-step velocity is wrong for the evolving
(partially-denoised) x distribution the snapshot probes never exercised.
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

# ── reference model ────────────────────────────────────────────────────────
ckpt = dl(REPO, CKPT_FILE, SIZE); cfg = dl(REPO, "config.yaml", SIZE); vocab = dl(REPO, "vocab.txt", SIZE)
from f5_tts.model.backbones.dit import DiT  # noqa: E402
from f5_tts.model.utils import get_tokenizer, list_str_to_idx  # noqa: E402
conf = yaml.safe_load(open(cfg)); a = conf["model"]["arch"]; mspec = conf["model"]["mel_spec"]
n_mel = mspec["n_mel_channels"]
sd = torch.load(ckpt, map_location="cpu", weights_only=True)["ema_model_state_dict"]
sd = {k.replace("ema_model.transformer.", ""): v for k, v in sd.items() if k.startswith("ema_model.transformer.")}
rows = int(sd["text_embed.text_embed.weight"].shape[0])
vmap, _ = get_tokenizer(vocab, "custom"); vmap = {c: i for c, i in vmap.items() if i < rows - 1}
dit = DiT(**{k: a[k] for k in a if k != "name"}, mel_dim=n_mel, text_num_embeds=rows - 1)
dit.load_state_dict(sd, strict=False); dit.eval()

# ── shared inputs: x0, cond (ref-mel-like), tokens, schedule ────────────────
T = 200
torch.manual_seed(387)
x0 = torch.randn(1, T, n_mel)
cond = torch.randn(1, T, n_mel) * 0.3 - 2.24
gen_text = "the quick brown fox jumps over the lazy dog"
text_ids = list_str_to_idx([list(gen_text)], vmap)
tok = text_ids[0].numpy().astype(np.int32)
STEPS = 32; SWAY = -1.0; CFG = 2.0
t = torch.linspace(0, 1, STEPS + 1)
t = t + SWAY * (torch.cos(torch.pi / 2 * t) - 1 + t)   # matches C++ get_epss_timesteps(32)+sway

# dump the shared inputs for the C++ ODE probe
(PROBE / "shape.txt").write_text(f"{T} {tok.size}")
x0[0].contiguous().numpy().astype(np.float32).tofile(PROBE / "x0.bin")
cond[0].contiguous().numpy().astype(np.float32).tofile(PROBE / "cond.bin")
tok.tofile(PROBE / "tokens.bin")

# ── reference euler (same loop as C++ euler_solve) ─────────────────────────
ref_traj = [x0[0].numpy().astype(np.float32).copy()]  # step 0 = x0
x = x0.clone()
with torch.no_grad():
    for k in range(STEPS):
        tv = t[k].item(); dt = (t[k + 1] - t[k]).item()
        tt = torch.tensor([tv])
        v_cond = dit(x, cond, text_ids, tt, drop_audio_cond=False, drop_text=False, cfg_infer=False)
        v_unc = dit(x, cond, text_ids, tt, drop_audio_cond=True, drop_text=True, cfg_infer=False)
        v = v_cond + CFG * (v_cond - v_unc)
        x = x + v * dt
        ref_traj.append(x[0].numpy().astype(np.float32).copy())
        if not np.isfinite(ref_traj[-1]).all():
            step("ref_NAN", at_step=k + 1); break
step("ref_done", n=len(ref_traj), final_std=round(float(ref_traj[-1].std()), 3))

# ── C++ euler (same x0/cond/tokens via ODE probe) ──────────────────────────
env = dict(os.environ); env["CRISPASR_F5_ODE_PROBE"] = str(PROBE)
rc = sh(f"{CLI} --backend raon-1b -m {gguf} --voice {ref_wav} --ref-text x --tts probe "
        f"--tts-output {WORK/'o.wav'} -t 4 --seed 42 --i-have-rights -v", env=env, timeout=1800)
print(rc.stdout[-800:], rc.stderr[-1500:], flush=True)

def load(k):
    p = PROBE / f"ode_step_{k}.bin"
    return np.fromfile(p, dtype=np.float32) if p.exists() else None

def cmp(u, v):
    if v is None:
        return None
    u = np.asarray(u).reshape(-1).astype(np.float64); v = np.asarray(v).reshape(-1).astype(np.float64)
    if u.size != v.size:
        return {"size": [int(u.size), int(v.size)]}
    fin = np.isfinite(v).all()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return {"cos": round(float(u.dot(v) / (nu * nv)), 5) if fin and nu > 0 and nv > 0 else None,
            "norm_ratio": round(float(nv / nu), 4) if fin and nu > 0 else None,
            "cpp_std": round(float(v.std()), 3) if fin else None, "cpp_finite": bool(fin)}

rows = []
first_div = None
for k in range(len(ref_traj)):
    c = cmp(ref_traj[k], load(k))
    rows.append({"step": k, **(c or {"missing": True})})
    if c and c.get("cos") is not None and c["cos"] < 0.9999 and first_div is None:
        first_div = k
result = {"size": SIZE, "T": T, "first_divergent_step": first_div,
          "ref_final_std": round(float(ref_traj[-1].std()), 3),
          "steps": rows}
(WORK / "raon_ode_diff.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
step("DONE", first_divergent_step=first_div)
