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

dim = a["dim"]; text_dim = a["text_dim"]
cap = {}; blk_out = {}
h1 = dit.input_embed.register_forward_hook(lambda m, i, o: cap.__setitem__("hidden", o.detach()))
h2 = dit.time_embed.register_forward_hook(lambda m, i, o: cap.__setitem__("temb", o.detach()))
h4 = dit.text_embed.register_forward_hook(lambda m, i, o: cap.__setitem__("text_embed", o.detach()))
h3 = dit.proj_out.register_forward_hook(lambda m, i, o: cap.__setitem__("velocity", o.detach()))
bh = [dit.transformer_blocks[k].register_forward_hook(
        (lambda kk: (lambda m, i, o: blk_out.__setitem__(kk, o.detach())))(k))
      for k in range(len(dit.transformer_blocks))]
with torch.no_grad():
    _ = dit(x, cond, text_ids, time_t, drop_audio_cond=False, drop_text=False, cfg_infer=False)
for h in [h1, h2, h4, h3] + bh:
    h.remove()

hidden = cap["hidden"][0].contiguous().numpy().astype(np.float32)          # (T, dim)
temb = cap["temb"].reshape(-1).numpy().astype(np.float32)                  # (dim,)
text_embed = cap["text_embed"][0].contiguous().numpy().astype(np.float32)  # (T, text_dim)
velocity = cap["velocity"][0].contiguous().numpy().astype(np.float32)      # (T, mel)
depth = len(blk_out)
ref_blocks = [blk_out[k][0].contiguous().numpy().astype(np.float32) for k in range(depth)]

# ── UNCOND arm capture (CFG null: drop_audio_cond + drop_text) ─────────────
capu = {}
u1 = dit.input_embed.register_forward_hook(lambda m, i, o: capu.__setitem__("hidden", o.detach()))
u4 = dit.text_embed.register_forward_hook(lambda m, i, o: capu.__setitem__("text_embed", o.detach()))
u3 = dit.proj_out.register_forward_hook(lambda m, i, o: capu.__setitem__("velocity", o.detach()))
dit.clear_cache()
with torch.no_grad():
    _ = dit(x, cond, text_ids, time_t, drop_audio_cond=True, drop_text=True, cfg_infer=False)
for h in [u1, u4, u3]:
    h.remove()
u_text_embed = capu["text_embed"][0].contiguous().numpy().astype(np.float32)
u_hidden = capu["hidden"][0].contiguous().numpy().astype(np.float32)
u_velocity = capu["velocity"][0].contiguous().numpy().astype(np.float32)
tok = text_ids[0].numpy().astype(np.int32)
(PROBE / "shape.txt").write_text(f"{T} {tok.size}")
(PROBE / "t.txt").write_text(str(float(time_t.item())))
x[0].contiguous().numpy().astype(np.float32).tofile(PROBE / "x.bin")
cond[0].contiguous().numpy().astype(np.float32).tofile(PROBE / "cond.bin")
tok.tofile(PROBE / "tokens.bin")
hidden.tofile(PROBE / "hidden.bin"); temb.tofile(PROBE / "temb.bin")
step("captured", T=T, nt=int(tok.size), dim=dim, depth=depth,
     hidden_std=float(hidden.std()), velocity_std=float(velocity.std()))

def run(mode):
    env = dict(os.environ); env[mode] = str(PROBE)
    r = sh(f"{CLI} --backend raon-1b -m {gguf} --voice {ref_wav} --ref-text x --tts probe "
           f"--tts-output {WORK/'probe.wav'} -t 4 --i-have-rights -v", env=env, timeout=1200)
    print(f"[{mode}]", r.stdout[-500:], r.stderr[-1500:], flush=True)

run("CRISPASR_F5_INPUT_PROBE")   # -> cpp_temb / cpp_text_embed / cpp_hidden
run("CRISPASR_F5_DIT_PROBE")     # -> cpp_velocity / cpp_block_<k> (injected hidden+temb)

def load(name):
    p = PROBE / name
    return np.fromfile(p, dtype=np.float32) if p.exists() else None

def compare(ref, cpp):
    if cpp is None:
        return None
    u = np.asarray(ref).reshape(-1).astype(np.float64); v = np.asarray(cpp).reshape(-1).astype(np.float64)
    if u.size != v.size:
        return {"size_mismatch": [int(u.size), int(v.size)]}
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    cosv = float(u.dot(v) / (nu * nv)) if nu > 0 and nv > 0 else None
    return {"cos": None if cosv is None else round(cosv, 5),
            "norm_ratio": round(float(nv / nu), 4) if nu > 0 else None,       # ||cpp|| / ||ref||  (1.0 = same scale)
            "maxabs_ratio": round(float(np.abs(v).max() / (np.abs(u).max() + 1e-12)), 4)}

# positive control on magnitude too: a 1.7x-scaled ref must read norm_ratio≈1.7, cos≈1
pc = compare(hidden, (hidden * 1.7).reshape(-1))
result = {"size": SIZE, "T": T, "depth": depth,
          "positive_control_scale1.7": pc,
          "cond_input_path": {"temb": compare(temb, load("cpp_temb.bin")),
                              "text_embed": compare(text_embed, load("cpp_text_embed.bin")),
                              "hidden": compare(hidden, load("cpp_hidden.bin"))},
          "UNCOND_arm": {"text_embed": compare(u_text_embed, load("cpp_text_embed_uncond.bin")),
                         "hidden": compare(u_hidden, load("cpp_hidden_uncond.bin")),
                         "velocity": compare(u_velocity, load("cpp_velocity_uncond.bin"))},
          "cond_velocity": compare(velocity, load("cpp_velocity.bin")),
          "blocks": [{"k": k, **(compare(ref_blocks[k], load(f"cpp_block_{k}.bin")) or {})} for k in range(depth)]}
(WORK / "raon_dit_diff.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
nr = [b.get("norm_ratio") for b in result["blocks"] if b.get("norm_ratio")]
step("DONE", cond_velocity=result["cond_velocity"], uncond=result["UNCOND_arm"]["velocity"], block0_nr=result["blocks"][0].get("norm_ratio"),
     blocklast_nr=result["blocks"][-1].get("norm_ratio"),
     norm_ratio_span=[min(nr), max(nr)] if nr else None)
