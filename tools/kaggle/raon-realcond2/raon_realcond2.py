#!/usr/bin/env python3
"""Raon 1B: C++ euler vs the KNOWN-GOOD sampler (#387-adj), gen-region metric.

Fixes two harness faults from raon-realcond: (1) my hand-rolled reference euler
diverged where the library's own CFM.sample (odeint euler) converges, so the
reference arm must BE CFM.sample; (2) full-x std conflates the ref-region
overshoot (evolves freely, discarded via out=where(cond_mask,cond,out)) with
real divergence, so the metric must be the GEN-REGION std (frames >= ref_T,
what actually becomes audio).

CONTROL = CFM.sample's gen-region std per step — MUST converge (~0.9). Then
inject CFM.sample's EXACT y0 + step_cond + tokens into C++'s euler_solve and
compare its gen-region std. If C++ diverges where CFM.sample converges, C++'s
sampler is the 1B bug. RAON_SIZE=1B; chr1s4.
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

# ── build CFM exactly like the (converging) ref-dump ───────────────────────
ckpt = dl(REPO, CKPT_FILE, SIZE); cfg = dl(REPO, "config.yaml", SIZE); vocab = dl(REPO, "vocab.txt", SIZE)
from f5_tts.model.cfm import CFM  # noqa: E402
from f5_tts.model.backbones.dit import DiT  # noqa: E402
from f5_tts.model.modules import MelSpec  # noqa: E402
from f5_tts.model.utils import get_tokenizer, list_str_to_idx  # noqa: E402
conf = yaml.safe_load(open(cfg)); a = conf["model"]["arch"]; mspec = conf["model"]["mel_spec"]
n_mel = mspec["n_mel_channels"]
# FORCE CPU for torch: Kaggle's PyTorch has no sm_60 kernels, so CFM.sample
# CUDA-errors on a P100 (the ref-dump forced CPU for the same reason). C++/ggml
# stays on GPU independently (ggml-cuda ships sm_60/75). CPU is why we cap ref_T.
DEV = "cpu"
sd = torch.load(ckpt, map_location="cpu", weights_only=True)["ema_model_state_dict"]
sd = {k.replace("ema_model.", ""): v for k, v in sd.items() if k.startswith("ema_model.") and "mel_spec" not in k}
vocab_map, _ = get_tokenizer(vocab, "custom")
ckpt_text = int(sd["transformer.text_embed.text_embed.weight"].shape[0])
model = CFM(
    transformer=DiT(**{k: a[k] for k in a if k != "name"}, mel_dim=n_mel, text_num_embeds=ckpt_text - 1),
    mel_spec_kwargs=mspec,
    vocab_char_map={c: i for c, i in vocab_map.items() if i < ckpt_text - 1},
)
model.load_state_dict(sd, strict=False); model = model.to(DEV).eval()
torch.set_num_threads(os.cpu_count() or 4)
step("cfm_built", dev=DEV, threads=os.cpu_count())

# jfk ref-mel via the model's own extractor (CPU stft) then to DEV
import wave as _w
_wf = _w.open(str(ref_wav), "rb"); _n = _wf.getnframes()
_a = np.frombuffer(_wf.readframes(_n), dtype=np.int16).astype(np.float32) / 32768.0; _wf.close()
wav = torch.from_numpy(_a).unsqueeze(0)
with torch.no_grad():
    ref_mel = MelSpec(**mspec)(wav)[0].T.contiguous()  # (full_ref_T, n_mel), CPU
# Cap ref length so the CPU CFM.sample (no GPU on P100) is tractable. A ~2.4 s
# ref is a valid voice prompt, and the sampler bug (if any) is per-step, not
# scale-specific — it must show at T~300 as much as T~965.
REF_CAP = int(os.environ.get("RAON_REF_CAP", "150"))
ref_mel = ref_mel[:REF_CAP].contiguous()
ref_T = ref_mel.shape[0]
# ref_text MUST match the (capped) ref audio, else text/audio disagree and even
# the oracle CFM.sample diverges. The first ~2.4 s of jfk is its opening phrase.
# Duration = 2*ref_T is the ref-dump's known-good setting (it converged).
ref_text = "And so my fellow Americans"
gen_text = "Hello there my friend."
full_text = ref_text + " " + gen_text
D = 2 * ref_T
tok = list_str_to_idx([list(full_text)], model.vocab_char_map)[0].numpy().astype(np.int32)
step("setup", ref_T=ref_T, D=D, nt=int(tok.size))

# ── CONTROL: CFM.sample (the known-good sampler), capture trajectory ───────
with torch.no_grad():
    out, traj = model.sample(cond=ref_mel.unsqueeze(0).to(DEV), text=[full_text], duration=D,
                             steps=32, cfg_strength=2.0, sway_sampling_coef=-1.0, seed=387, use_epss=True)
traj = traj.cpu().float()              # (steps+1, 1, T, n_mel)
T = traj.shape[2]
ref_gen_std = [round(float(traj[i, 0, ref_T:].std()), 3) if torch.isfinite(traj[i]).all() else None
               for i in range(traj.shape[0])]
ref_full_std = [round(float(traj[i, 0].std()), 3) if torch.isfinite(traj[i]).all() else None for i in range(traj.shape[0])]
step("CONTROL_cfm_sample", T=T, gen_final=ref_gen_std[-1], full_final=ref_full_std[-1])

# ── C++ euler with CFM.sample's EXACT y0 + step_cond + tokens ──────────────
y0 = traj[0, 0].numpy().astype(np.float32)                 # (T, n_mel) — the exact init noise CFM used
stepcond = np.zeros((T, n_mel), np.float32); stepcond[:min(ref_T, T)] = ref_mel[:min(ref_T, T)].numpy()
(PROBE / "shape.txt").write_text(f"{T} {tok.size}")
y0.tofile(PROBE / "x0.bin"); stepcond.tofile(PROBE / "cond.bin"); tok.tofile(PROBE / "tokens.bin")
env = dict(os.environ); env["CRISPASR_F5_ODE_PROBE"] = str(PROBE)
rc = sh(f"{CLI} --backend raon-1b -m {gguf} --voice {ref_wav} --ref-text x --tts probe "
        f"--tts-output {WORK/'o.wav'} -t 4 --seed 42 --i-have-rights -v", env=env, timeout=1800)
print(rc.stderr[-1200:], flush=True)

def gen_std_from(dirp, k):
    p = Path(dirp) / f"ode_step_{k}.bin"
    if not p.exists():
        return None
    v = np.fromfile(p, dtype=np.float32)
    if v.size != T * n_mel or not np.isfinite(v).all():
        return None
    return round(float(v.reshape(T, n_mel)[ref_T:].std()), 3)

cpp_gen = [gen_std_from(PROBE, k) for k in range(33)]

# ── GUARD 1 (against FALSE POSITIVE): the injection MUST have landed. C++'s
#    ode_step_0 is the init x = the injected y0. If it isn't bitwise y0, C++ ran
#    its OWN noise and any divergence is meaningless — void the run. ──
s0p = PROBE / "ode_step_0.bin"
inj_cos, inj_norm = None, 0.0
if s0p.exists():
    s0 = np.fromfile(s0p, dtype=np.float32)
    if s0.size == y0.size:
        a, b = y0.reshape(-1).astype(np.float64), s0.astype(np.float64)
        inj_norm = float(np.linalg.norm(b))
        if np.linalg.norm(a) > 0 and inj_norm > 0:
            inj_cos = float(a.dot(b) / (np.linalg.norm(a) * inj_norm))
injection_landed = (inj_cos is not None and abs(inj_cos - 1.0) < 1e-5 and inj_norm > 0
                    and float(np.abs(y0.reshape(-1) - np.fromfile(s0p, dtype=np.float32)).max()) < 1e-4)
step("GUARD1_injection", cos=None if inj_cos is None else round(inj_cos, 6), norm=round(inj_norm, 2),
     landed=bool(injection_landed))

# ── GUARD 2 (against FABRICATED POSITIVE): perturb y0 by a known epsilon, run
#    C++ again, and confirm the gen-region-std metric actually MOVES. If a
#    deliberately corrupted injection still reads identical, the metric is blind
#    and neither branch of the clean-read table means anything. ──
PROBE2 = TMP / "ode_perturb"; PROBE2.mkdir(exist_ok=True)
y0p = y0 + (0.5 * np.random.RandomState(1).standard_normal(y0.shape)).astype(np.float32)
(PROBE2 / "shape.txt").write_text(f"{T} {tok.size}")
y0p.tofile(PROBE2 / "x0.bin"); stepcond.tofile(PROBE2 / "cond.bin"); tok.tofile(PROBE2 / "tokens.bin")
env2 = dict(os.environ); env2["CRISPASR_F5_ODE_PROBE"] = str(PROBE2)
sh(f"{CLI} --backend raon-1b -m {gguf} --voice {ref_wav} --ref-text x --tts probe "
   f"--tts-output {WORK/'o2.wav'} -t 4 --seed 42 --i-have-rights", env=env2, timeout=1800)
cpp_gen_perturbed = [gen_std_from(PROBE2, k) for k in range(33)]
# the metric must distinguish the perturbed run at some step (>0.02 abs on std, or a NaN-vs-finite split)
metric_sensitive = any(
    (g is None) != (p is None) or (g is not None and p is not None and abs(g - p) > 0.02)
    for g, p in zip(cpp_gen, cpp_gen_perturbed))
step("GUARD2_metric_sensitive", sensitive=bool(metric_sensitive))

result = {"size": SIZE, "ref_T": ref_T, "T": T,
          "GUARD1_injection_landed": bool(injection_landed), "injection_cos": None if inj_cos is None else round(inj_cos, 6),
          "GUARD2_metric_sensitive": bool(metric_sensitive),
          "VALID": bool(injection_landed and metric_sensitive),
          "CONTROL_ref_gen_std": ref_gen_std, "CONTROL_ref_full_std": ref_full_std,
          "CPP_gen_std": cpp_gen, "CPP_gen_std_perturbed": cpp_gen_perturbed,
          "control_converged": (ref_gen_std[-1] is not None and ref_gen_std[-1] < 1.3),
          "cpp_diverged": (cpp_gen[-1] is None or (cpp_gen[-1] or 9) > 1.3)}
(WORK / "raon_realcond2.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
step("DONE", VALID=result["VALID"], control_ok=result["control_converged"], cpp_diverged=result["cpp_diverged"], g1=result["GUARD1_injection_landed"], g2=result["GUARD2_metric_sensitive"])
