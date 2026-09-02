#!/usr/bin/env python3
"""Raon 1B DiT per-stage diff (#387-adj) — localize the C++ DiT divergence.

The 1B converts+loads+runs but emits a wrong mel (non-speech). Python reference
is perfect and the GGUF is provably correct, so the bug is in our C++ DiT. This
kernel captures ONE reference DiT forward (torch, CPU) — the input-embed output
(hidden), the timestep embedding, every transformer-block output, and the final
velocity — then injects the SAME hidden + t_emb into our C++ via
CRISPASR_F5_DIT_PROBE and diffs each stage. The first block whose cosine drops
is where our port diverges (rope/attn/norm/ffn/adaln all live inside the block).

POSITIVE CONTROL (a green wall must not be the comparator failing to fire):
  - identity  cos(ref_b0, ref_b0) == 1
  - sensitivity cos(ref_b0, ref_b1) < 0.999   (comparator can report MISMATCH)
  - injection  the C++ MUST at least reproduce block 0 from the injected hidden;
    if cpp_block_0 already ~0 or NaN the harness (not the model) is broken.

CPU torch for the reference (P100 lacks torch stft kernels); C++ runs on GPU
(CUDA) to match the failing roundtrip config. RAON_SIZE=1B. chr1s4 datasets.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

WORK = Path("/kaggle/working"); TMP = Path("/kaggle/temp"); TMP.mkdir(exist_ok=True)
PROBE = TMP / "probe"; PROBE.mkdir(exist_ok=True)
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
            cmd = ["git", "clone", "--depth", "1"] + (["--branch", ref] if ref else []) + [url, str(dst)]
            if subprocess.run(cmd).returncode == 0:
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


# ── build crispasr (GPU, to match the failing config) ──────────────────────
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
if not CLI:
    step("no_binary"); sys.exit(1)
os.environ["LD_LIBRARY_PATH"] = str(CLI.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
gguf = hf_hub_download(f"cstr/raon-opentts-{SIZE.lower()}-GGUF", f"raon-opentts-{SIZE.lower()}-f16.gguf",
                       local_dir=str(MODELS), token=HF_TOKEN or None)
ref_wav = CLONE / "samples" / "jfk.wav"
step("built", cli=str(CLI), gguf=os.path.basename(gguf))

# ── reference capture: one DiT forward (torch, CPU, b=1, no cfg) ───────────
ckpt = dl(REPO, CKPT_FILE, SIZE); cfg = dl(REPO, "config.yaml", SIZE); vocab = dl(REPO, "vocab.txt", SIZE)
from f5_tts.model.backbones.dit import DiT  # noqa: E402
from f5_tts.model.utils import get_tokenizer, list_str_to_idx  # noqa: E402
conf = yaml.safe_load(open(cfg)); a = conf["model"]["arch"]; mspec = conf["model"]["mel_spec"]
n_mel = mspec["n_mel_channels"]
sd = torch.load(ckpt, map_location="cpu", weights_only=True)["ema_model_state_dict"]
sd = {k.replace("ema_model.transformer.", ""): v for k, v in sd.items()
      if k.startswith("ema_model.transformer.")}
rows = int(sd["text_embed.text_embed.weight"].shape[0])
vmap, _ = get_tokenizer(vocab, "custom")
vmap = {c: i for c, i in vmap.items() if i < rows - 1}
dit = DiT(**{k: a[k] for k in a if k != "name"}, mel_dim=n_mel, text_num_embeds=rows - 1)
missing, unexpected = dit.load_state_dict(sd, strict=False)
dit.eval()
step("dit_loaded", missing=len(missing), unexpected=len(unexpected), dim=a["dim"], depth=a["depth"], heads=a["heads"])

T = 200
torch.manual_seed(387)
x = torch.randn(1, T, n_mel) * 0.3
cond = torch.randn(1, T, n_mel) * 0.3
gen_text = "the quick brown fox jumps over the lazy dog"
text_ids = list_str_to_idx([list(gen_text)], vmap)  # (1, nt)
time_t = torch.tensor([0.5])

cap = {}
h1 = dit.input_embed.register_forward_hook(lambda m, i, o: cap.__setitem__("hidden", o.detach()))
h2 = dit.time_embed.register_forward_hook(lambda m, i, o: cap.__setitem__("temb", o.detach()))
h3 = dit.proj_out.register_forward_hook(lambda m, i, o: cap.__setitem__("velocity", o.detach()))
blk_out = {}
bh = [dit.transformer_blocks[k].register_forward_hook(
        (lambda kk: (lambda m, i, o: blk_out.__setitem__(kk, o.detach())))(k))
      for k in range(len(dit.transformer_blocks))]
with torch.no_grad():
    _ = dit(x, cond, text_ids, time_t, drop_audio_cond=False, drop_text=False, cfg_infer=False)
for h in [h1, h2, h3] + bh:
    h.remove()

hidden = cap["hidden"][0].contiguous().numpy().astype(np.float32)   # (T, dim)
temb = cap["temb"].reshape(-1).numpy().astype(np.float32)           # (dim,)
velocity = cap["velocity"][0].contiguous().numpy().astype(np.float32)  # (T, mel)
depth = len(blk_out)
(PROBE / "shape.txt").write_text(str(T))
hidden.tofile(PROBE / "hidden.bin"); temb.tofile(PROBE / "temb.bin")
velocity.tofile(PROBE / "ref_velocity.bin")
ref_blocks = []
for k in range(depth):
    b = blk_out[k][0].contiguous().numpy().astype(np.float32)
    b.tofile(PROBE / f"ref_block_{k}.bin"); ref_blocks.append(b)
step("captured", T=T, dim=hidden.shape[1], depth=depth, hidden_std=float(hidden.std()),
     temb_std=float(temb.std()), vel_std=float(velocity.std()))

# ── run our C++ DiT probe (CUDA) ───────────────────────────────────────────
env = dict(os.environ); env["CRISPASR_F5_DIT_PROBE"] = str(PROBE)
rc = sh(f"{CLI} --backend raon-1b -m {gguf} --voice {ref_wav} --ref-text x --tts probe "
        f"--tts-output {WORK/'probe.wav'} -t 4 --i-have-rights -v", env=env, timeout=1200)
print(rc.stdout[-1500:], rc.stderr[-2500:], flush=True)

def cos(u, v):
    u = u.reshape(-1).astype(np.float64); v = v.reshape(-1).astype(np.float64)
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u.dot(v) / d) if d > 0 else float("nan")

def load(name, n):
    p = PROBE / name
    if not p.exists():
        return None
    a = np.fromfile(p, dtype=np.float32)
    return a if a.size == n else a

# positive control
ctrl = {"identity_cos": cos(ref_blocks[0], ref_blocks[0]),
        "sensitivity_cos_b0_b1": cos(ref_blocks[0], ref_blocks[1]) if depth > 1 else None}
cpp_v = load("cpp_velocity.bin", T * n_mel)
inj_ok = cpp_v is not None and np.isfinite(cpp_v).all() and float(np.abs(cpp_v).sum()) > 0
step("positive_control", **ctrl, cpp_velocity_present=cpp_v is not None,
     cpp_velocity_finite_nonzero=bool(inj_ok))

# per-stage diff
stages = []
first_div = None
for k in range(depth):
    cb = load(f"cpp_block_{k}.bin", T * hidden.shape[1])
    c = cos(ref_blocks[k], cb) if cb is not None else None
    stages.append({"block": k, "cos": None if c is None else round(c, 5)})
    if c is not None and c < 0.99 and first_div is None:
        first_div = k
vel_cos = cos(velocity, cpp_v) if cpp_v is not None else None
result = {"size": SIZE, "T": T, "depth": depth,
          "positive_control": {k: (None if v is None else round(v, 6)) for k, v in ctrl.items()},
          "injection_ok": bool(inj_ok),
          "first_divergent_block": first_div,
          "velocity_cos": None if vel_cos is None else round(vel_cos, 5),
          "block_cos": stages}
(WORK / "raon_dit_diff.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
step("DONE", first_divergent_block=first_div, velocity_cos=result["velocity_cos"])
