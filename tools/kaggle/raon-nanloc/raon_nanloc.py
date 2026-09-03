#!/usr/bin/env python3
"""Raon 1B NaN localization (#387-adj) — name the first non-finite tensor.

Converging control established (v4): C++ tracks the oracle then goes non-finite
at ~step 22-23 on a HEALTHY (std 0.62) trajectory. This run, per crispasr-dc:
  (1) DRIFT: report per-step max|x_cpp - x_oracle| (2-dp std agreement is NOT
      tensor byte-exactness) — if it's ~0 through the last finite step, the NaN
      is C++'s fault on identical input; if it drifts, THAT is the finding.
  (2) NaN vs Inf, and the offending tensor's stats one step earlier.
  (3) BLOCK LOCALIZE: at the last finite step, run the DiT with per-block taps
      on the oracle's (finite) latent and name the first block that goes
      non-finite. t-driven vs x-driven falls out of which op it is.
No mechanism named until the dump speaks. RAON_SIZE=1B; chr1s4.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

WORK = Path("/kaggle/working"); TMP = Path("/kaggle/temp"); TMP.mkdir(exist_ok=True)
PROBE = TMP / "ode"; PROBE.mkdir(exist_ok=True); DIT = TMP / "dit"; DIT.mkdir(exist_ok=True)
SIZE = os.environ.get("RAON_SIZE", "1B")
REPO = f"KRAFTON/Raon-OpenTTS-{SIZE}"; CKPT_FILE = {"0.3B": "model_225000.pt", "1B": "model_520000.pt"}[SIZE]
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
kh.init_progress(); HF_TOKEN = kh.resolve_hf_token()
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


kh.install_build_toolchain(); arch = kh.detect_cuda_arch()
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
step("built")

ckpt = dl(REPO, CKPT_FILE, SIZE); cfg = dl(REPO, "config.yaml", SIZE); vocab = dl(REPO, "vocab.txt", SIZE)
from f5_tts.model.cfm import CFM  # noqa: E402
from f5_tts.model.backbones.dit import DiT  # noqa: E402
from f5_tts.model.modules import MelSpec  # noqa: E402
from f5_tts.model.utils import get_tokenizer, list_str_to_idx  # noqa: E402
conf = yaml.safe_load(open(cfg)); a = conf["model"]["arch"]; mspec = conf["model"]["mel_spec"]; n_mel = mspec["n_mel_channels"]
DEV = "cpu"
sd = torch.load(ckpt, map_location="cpu", weights_only=True)["ema_model_state_dict"]
sd = {k.replace("ema_model.", ""): v for k, v in sd.items() if k.startswith("ema_model.") and "mel_spec" not in k}
vocab_map, _ = get_tokenizer(vocab, "custom"); ckpt_text = int(sd["transformer.text_embed.text_embed.weight"].shape[0])
model = CFM(transformer=DiT(**{k: a[k] for k in a if k != "name"}, mel_dim=n_mel, text_num_embeds=ckpt_text - 1),
            mel_spec_kwargs=mspec, vocab_char_map={c: i for c, i in vocab_map.items() if i < ckpt_text - 1})
model.load_state_dict(sd, strict=False); model = model.to(DEV).eval(); torch.set_num_threads(os.cpu_count() or 4)

# converging control setup: random-noise ref + "reference", duration=2*ref_T
torch.manual_seed(0); wav = torch.randn(1, mspec["target_sample_rate"] * 3) * 0.1
with torch.no_grad():
    ref_mel = MelSpec(**mspec)(wav)[0].T.contiguous()
ref_T = ref_mel.shape[0]
ref_text = "reference"; gen_text = "The quick brown fox jumps over the lazy dog."
full_text = ref_text + " " + gen_text; D = 2 * ref_T
tok = list_str_to_idx([list(full_text)], model.vocab_char_map)[0].numpy().astype(np.int32)
STEPS, SWAY, CFG = 32, -1.0, 2.0
sched = torch.linspace(0, 1, STEPS + 1); sched = sched + SWAY * (torch.cos(torch.pi / 2 * sched) - 1 + sched)
step("setup", ref_T=ref_T, D=D, nt=int(tok.size))

with torch.no_grad():
    out, traj = model.sample(cond=ref_mel.unsqueeze(0), text=[full_text], duration=D, steps=STEPS,
                             cfg_strength=CFG, sway_sampling_coef=SWAY, seed=387, use_epss=True)
traj = traj.cpu().float()  # (STEPS+1, 1, T, n_mel)
T = traj.shape[2]
step("oracle", T=T, gen_final=round(float(traj[-1, 0, ref_T:].std()), 3), all_finite=bool(torch.isfinite(traj).all()))

# ── C++ euler on the oracle's exact y0/cond/tokens (ODE_PROBE dumps ode_step_N) ──
y0 = traj[0, 0].numpy().astype(np.float32)
stepcond = np.zeros((T, n_mel), np.float32); stepcond[:min(ref_T, T)] = ref_mel[:min(ref_T, T)].numpy()
(PROBE / "shape.txt").write_text(f"{T} {tok.size}")
y0.tofile(PROBE / "x0.bin"); stepcond.tofile(PROBE / "cond.bin"); tok.tofile(PROBE / "tokens.bin")
env = dict(os.environ); env["CRISPASR_F5_ODE_PROBE"] = str(PROBE)
sh(f"{CLI} --backend raon-1b -m {gguf} --voice {ref_wav} --ref-text x --tts probe --tts-output {WORK/'o.wav'} "
   f"-t 4 --seed 42 --i-have-rights", env=env, timeout=1800)

def load(d, name):
    p = Path(d) / name
    return np.fromfile(p, dtype=np.float32) if p.exists() else None

# (1) DRIFT: per-step max|x_cpp - x_oracle| on IDENTICAL injected input
drift = []; first_drift = None; first_nonfinite = None; nan_or_inf = None
for k in range(STEPS + 1):
    c = load(PROBE, f"ode_step_{k}.bin")
    if c is None or c.size != T * n_mel:
        drift.append(None); continue
    fin = np.isfinite(c).all()
    if not fin and first_nonfinite is None:
        first_nonfinite = k
        nan_or_inf = "NaN" if np.isnan(c).any() else "Inf"
    o = traj[k, 0].numpy().reshape(-1)
    dmax = float(np.max(np.abs(c - o))) if fin else None
    drift.append(None if dmax is None else round(dmax, 8))
    if dmax is not None and dmax > 1e-6 and first_drift is None:
        first_drift = k
step("drift", first_step_gt_1e6=first_drift, first_nonfinite_step=first_nonfinite, kind=nan_or_inf,
     drift_head=[d for d in drift[:6]], drift_at_fail=(drift[first_nonfinite - 1] if first_nonfinite else None))

# (2)+(3) BLOCK LOCALIZE at the last finite step k* = first_nonfinite - 1
result = {"size": SIZE, "T": T, "ref_T": ref_T, "oracle_gen_final": round(float(traj[-1, 0, ref_T:].std()), 3),
          "drift_per_step": drift, "first_drift_step_gt_1e6": first_drift,
          "first_nonfinite_step": first_nonfinite, "nan_or_inf": nan_or_inf}
if first_nonfinite is not None and first_nonfinite >= 1:
    kstar = first_nonfinite - 1
    x_prev = traj[kstar, 0].numpy().astype(np.float32)   # oracle's finite latent feeding the failing DiT
    tval = float(sched[kstar].item())
    # INPUT_PROBE: C++'s hidden + temb at x_prev, t=kstar
    (DIT / "shape.txt").write_text(f"{T} {tok.size}"); (DIT / "t.txt").write_text(str(tval))
    x_prev.tofile(DIT / "x.bin"); stepcond.tofile(DIT / "cond.bin"); tok.tofile(DIT / "tokens.bin")
    e1 = dict(os.environ); e1["CRISPASR_F5_INPUT_PROBE"] = str(DIT)
    sh(f"{CLI} --backend raon-1b -m {gguf} --voice {ref_wav} --ref-text x --tts probe --tts-output {WORK/'o.wav'} "
       f"-t 4 --i-have-rights", env=e1, timeout=1800)
    hid = load(DIT, "cpp_hidden.bin"); tem = load(DIT, "cpp_temb.bin")
    hfin = hid is not None and np.isfinite(hid).all(); tfin = tem is not None and np.isfinite(tem).all()
    result["kstar"] = kstar; result["t_at_kstar"] = round(tval, 5)
    result["hidden_finite"] = bool(hfin); result["temb_finite"] = bool(tfin)
    if hid is not None and tem is not None:
        hid.tofile(DIT / "hidden.bin"); tem.tofile(DIT / "temb.bin")
        e2 = dict(os.environ); e2["CRISPASR_F5_DIT_PROBE"] = str(DIT)
        sh(f"{CLI} --backend raon-1b -m {gguf} --voice {ref_wav} --ref-text x --tts probe --tts-output {WORK/'o.wav'} "
           f"-t 4 --i-have-rights", env=e2, timeout=1800)
        blocks = []
        first_bad = None
        for kb in range(a["depth"]):
            b = load(DIT, f"cpp_block_{kb}.bin")
            if b is None:
                blocks.append({"k": kb, "missing": True}); continue
            fin = bool(np.isfinite(b).all())
            info = {"k": kb, "finite": fin, "max": round(float(np.abs(b[np.isfinite(b)]).max()), 2) if np.isfinite(b).any() else None}
            if not fin:
                info["kind"] = "NaN" if np.isnan(b).any() else "Inf"
                if first_bad is None:
                    first_bad = kb
            blocks.append(info)
        vel = load(DIT, "cpp_velocity.bin")
        result["first_nonfinite_block"] = first_bad
        result["velocity_finite"] = bool(vel is not None and np.isfinite(vel).all())
        result["blocks"] = blocks
        step("BLOCK_LOC", kstar=kstar, hidden_finite=hfin, temb_finite=tfin, first_nonfinite_block=first_bad,
             kind=(blocks[first_bad].get("kind") if first_bad is not None else None))
(WORK / "raon_nanloc.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2)[:3000], flush=True)
step("DONE", first_nonfinite_step=first_nonfinite, first_nonfinite_block=result.get("first_nonfinite_block"))
