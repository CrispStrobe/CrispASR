#!/usr/bin/env python3
"""Raon 1B per-forward drift root (#387-adj) — where does the ~1.4e-4 originate.

The drift check showed C++ differs from the oracle by ~1.4e-4 on the FIRST
velocity eval (identical x0), compounding to NaN. This localizes that single
forward with ABSOLUTE max|cpp - oracle| (NOT cos/norm, which rounded it away):
  - time-embed:  max|cpp_temb  - oracle_temb|
  - input-embed: max|cpp_hidden - oracle_hidden|
  - per BLOCK:   inject the ORACLE's hidden+temb into C++ so blocks are compared
                 on identical input; max|cpp_block_k - oracle_block_k|.
Uniform small growth across blocks => distributed f16 rounding; a JUMP at one
block => that op. No mechanism named until the numbers say so. RAON_SIZE=1B.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

WORK = Path("/kaggle/working"); TMP = Path("/kaggle/temp"); TMP.mkdir(exist_ok=True)
IN = TMP / "inp"; IN.mkdir(exist_ok=True); DT = TMP / "dit"; DT.mkdir(exist_ok=True)
SIZE = os.environ.get("RAON_SIZE", "0.3B"); REPO = f"KRAFTON/Raon-OpenTTS-{SIZE}"
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
if sh(f"cd {CLONE} && cmake -G Ninja -B build " + " ".join(flags), timeout=1200).returncode != 0:
    step("cmake_FAIL"); sys.exit(1)
with kh.build_heartbeat("build", interval_s=30):
    kh.sh_with_progress(f"cmake --build build -j{kh.safe_build_jobs(gpu=True)} --target crispasr-cli", cwd=str(CLONE))
CLI = CLONE / "build" / "bin" / "crispasr"
if not CLI.exists():
    c = [p for p in (CLONE / "build").rglob("crispasr") if p.is_file() and os.access(p, os.X_OK)]
    CLI = c[0] if c else None
os.environ["LD_LIBRARY_PATH"] = str(CLI.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
# Manual-SDPA A/B: the flash kernel ignores GGML_PREC_F32 on P100/sm_60, so isolate
# the flash op with the hint-independent manual attention path (default on for this
# diagnostic; set CRISPASR_F5_NO_FLASH=0 to measure the flash baseline instead).
os.environ.setdefault("CRISPASR_F5_NO_FLASH", "1")
step("attn_mode", no_flash=os.environ["CRISPASR_F5_NO_FLASH"])
gguf = hf_hub_download(f"cstr/raon-opentts-{SIZE.lower()}-GGUF", f"raon-opentts-{SIZE.lower()}-f16.gguf",
                       local_dir=str(MODELS), token=HF_TOKEN or None)
ref_wav = CLONE / "samples" / "jfk.wav"
BACKEND = {"0.3B": "raon", "1B": "raon-1b"}[SIZE]
step("built", backend=BACKEND)

ckpt = dl(REPO, CKPT_FILE, SIZE); cfg = dl(REPO, "config.yaml", SIZE); vocab = dl(REPO, "vocab.txt", SIZE)
from f5_tts.model.backbones.dit import DiT  # noqa: E402
from f5_tts.model.utils import get_tokenizer, list_str_to_idx  # noqa: E402
conf = yaml.safe_load(open(cfg)); a = conf["model"]["arch"]; mspec = conf["model"]["mel_spec"]; n_mel = mspec["n_mel_channels"]
sd = torch.load(ckpt, map_location="cpu", weights_only=True)["ema_model_state_dict"]
sd = {k.replace("ema_model.transformer.", ""): v for k, v in sd.items() if k.startswith("ema_model.transformer.")}
rows = int(sd["text_embed.text_embed.weight"].shape[0])
vmap, _ = get_tokenizer(vocab, "custom"); vmap = {c: i for c, i in vmap.items() if i < rows - 1}
dit = DiT(**{k: a[k] for k in a if k != "name"}, mel_dim=n_mel, text_num_embeds=rows - 1)
dit.load_state_dict(sd, strict=False); dit.eval()   # CPU f32 (P100 lacks sm_60 torch)

T = 200
torch.manual_seed(387)
x0 = torch.randn(1, T, n_mel)
cond = torch.randn(1, T, n_mel) * 0.3 - 2.24
gen_text = "the quick brown fox jumps over the lazy dog"
tok_ids = list_str_to_idx([list(gen_text)], vmap); tok = tok_ids[0].numpy().astype(np.int32)
tval = 0.0   # first ODE step t (linspace+sway both give 0 at index 0)

cap = {}; blk = {}
h1 = dit.input_embed.register_forward_hook(lambda m, i, o: cap.__setitem__("hidden", o.detach()))
h2 = dit.time_embed.register_forward_hook(lambda m, i, o: cap.__setitem__("temb", o.detach()))
h3 = dit.proj_out.register_forward_hook(lambda m, i, o: cap.__setitem__("vel", o.detach()))
bh = [dit.transformer_blocks[k].register_forward_hook(
        (lambda kk: (lambda m, i, o: blk.__setitem__(kk, o.detach())))(k)) for k in range(len(dit.transformer_blocks))]
with torch.no_grad():
    _ = dit(x0, cond, tok_ids, torch.tensor([tval]), drop_audio_cond=False, drop_text=False, cfg_infer=False)
for h in [h1, h2, h3] + bh:
    h.remove()
o_hidden = cap["hidden"][0].contiguous().numpy().astype(np.float32)
o_temb = cap["temb"].reshape(-1).numpy().astype(np.float32)
o_vel = cap["vel"][0].contiguous().numpy().astype(np.float32)
o_blocks = [blk[k][0].contiguous().numpy().astype(np.float32) for k in range(len(blk))]
depth = len(o_blocks)
step("oracle_forward", depth=depth, hidden_std=round(float(o_hidden.std()), 4), vel_std=round(float(o_vel.std()), 4))

# ── f16-MIRROR oracle (peer's substitution): round the SAME weights C++ stores
# as f16 through f16 precision (keep f32 dtype so arithmetic stays f32, matching
# C++'s f16-weights/f32-activations path), then run the blocks on the SAME f32
# o_hidden/o_temb C++ is given. If torch-f16 drift == C++ drift, ~1e-4 is just
# f16 and C++ is exonerated; if C++ drifts materially MORE, the excess is C++'s.
import copy  # noqa: E402
_A_F32 = ("text_emb", "freqs_cis", "inv_freq", "time_", "conv_pos", "input_proj",
          "input_embed.proj", "final_adaln", "final_proj", "adaln", ".layer_scale")
def gg(nm):  # mirror convert-raon-opentts map_f5tts_name for a DiT.state_dict key
    n = nm.replace("transformer_blocks.", "blk.").replace(".attn.to_q.", ".attn_q.").replace(".attn.to_k.", ".attn_k.")
    n = n.replace(".attn.to_v.", ".attn_v.").replace(".attn.to_out.0.", ".attn_o.").replace(".attn_norm.linear.", ".adaln.")
    n = n.replace(".ff.ff.0.0.", ".ffn_up.").replace(".ff.ff.2.", ".ffn_down.")
    n = n.replace("text_embed.text_embed.", "text_emb.").replace("text_embed.text_blocks.", "text_blk.")
    n = n.replace(".dwconv.", ".dw.").replace(".pwconv1.", ".pw_up.").replace(".pwconv2.", ".pw_down.")
    n = n.replace("time_embed.time_mlp.0.", "time_mlp_0.").replace("time_embed.time_mlp.2.", "time_mlp_1.")
    n = n.replace("input_embed.proj.", "input_proj.").replace("norm_out.linear.", "final_adaln.").replace("proj_out.", "final_proj.")
    n = n.replace("input_embed.conv_pos_embed.conv1d.0.", "conv_pos_0.").replace("input_embed.conv_pos_embed.conv1d.2.", "conv_pos_1.")
    return "f5." + n
dit16 = copy.deepcopy(dit)
n_f16 = 0
with torch.no_grad():
    for nm, p in dit16.named_parameters():
        if p.ndim >= 2 and p.numel() >= 256 and not any(s in gg(nm) for s in _A_F32):
            p.data = p.data.half().float(); n_f16 += 1
# run torch-f16 BLOCKS on the injected f32 o_hidden/o_temb (same input as C++ DIT_PROBE)
xb = torch.from_numpy(o_hidden).unsqueeze(0); tb = torch.from_numpy(o_temb).unsqueeze(0)
rope = dit16.rotary_embed.forward_from_seq_len(T)
f16_blocks = []
with torch.no_grad():
    for b in dit16.transformer_blocks:
        xb = b(xb, tb, mask=None, rope=rope); f16_blocks.append(xb[0].contiguous().numpy().astype(np.float32))
    f16_vel = dit16.proj_out(dit16.norm_out(xb, tb))[0].contiguous().numpy().astype(np.float32)
step("f16_mirror", n_f16_weights=n_f16)

def maxdiff(u, v):
    if v is None or u.size != v.size:
        return None
    return float(np.max(np.abs(u.reshape(-1).astype(np.float64) - v.reshape(-1).astype(np.float64))))

def load(d, n):
    p = Path(d) / n
    return np.fromfile(p, dtype=np.float32) if p.exists() else None

# C++ INPUT_PROBE on identical x0 -> cpp hidden + temb
(IN / "shape.txt").write_text(f"{T} {tok.size}"); (IN / "t.txt").write_text(str(tval))
x0[0].contiguous().numpy().astype(np.float32).tofile(IN / "x.bin")
cond[0].contiguous().numpy().astype(np.float32).tofile(IN / "cond.bin"); tok.tofile(IN / "tokens.bin")
e1 = dict(os.environ); e1["CRISPASR_F5_INPUT_PROBE"] = str(IN)
sh(f"{CLI} --backend {BACKEND} -m {gguf} --voice {ref_wav} --ref-text x --tts probe --tts-output {WORK/'o.wav'} -t 4 --i-have-rights", env=e1, timeout=1800)
cpp_hidden = load(IN, "cpp_hidden.bin"); cpp_temb = load(IN, "cpp_temb.bin")

# C++ DIT_PROBE with the ORACLE's hidden+temb -> isolates the blocks on identical input
o_hidden.tofile(DT / "hidden.bin"); o_temb.tofile(DT / "temb.bin"); (DT / "shape.txt").write_text(f"{T}")
e2 = dict(os.environ); e2["CRISPASR_F5_DIT_PROBE"] = str(DT)
sh(f"{CLI} --backend {BACKEND} -m {gguf} --voice {ref_wav} --ref-text x --tts probe --tts-output {WORK/'o.wav'} -t 4 --i-have-rights", env=e2, timeout=1800)

def relerr(md, ref):
    # max|diff| normalized by the block's RMS magnitude (calibration #2): a
    # uniform relative rate is f16 rounding; absolute grows just because
    # activations grow through the stack, masking a genuine single-op jump.
    if md is None:
        return None
    rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    return round(md / rms, 7) if rms > 0 else None

cpp_blocks = [load(DT, f"cpp_block_{k}.bin") for k in range(depth)]
cpp_vel_inj = load(DT, "cpp_velocity.bin")
# both C++ and torch-f16 blocks were fed the SAME f32 o_hidden/o_temb; compare each to the f32 oracle
cpp_bmd = [maxdiff(o_blocks[k], cpp_blocks[k]) for k in range(depth)]
f16_bmd = [maxdiff(o_blocks[k], f16_blocks[k]) for k in range(depth)]
result = {"size": SIZE, "T": T, "depth": depth, "t": tval, "n_f16_weights": n_f16,
          "input_embed_maxdiff_cpp": maxdiff(o_hidden, cpp_hidden),
          "time_embed_maxdiff_cpp": maxdiff(o_temb, cpp_temb),
          "velocity_maxdiff_CPP": maxdiff(o_vel, cpp_vel_inj),
          "velocity_maxdiff_TORCH_f16": maxdiff(o_vel, f16_vel),
          # THE decisive pair: per-block drift, C++ vs torch-f16, both on identical f32 input
          "block_maxdiff_CPP": [None if x is None else round(x, 7) for x in cpp_bmd],
          "block_maxdiff_TORCH_f16": [None if x is None else round(x, 7) for x in f16_bmd],
          "block_relerr_CPP": [relerr(cpp_bmd[k], o_blocks[k]) for k in range(depth)],
          "block_relerr_TORCH_f16": [relerr(f16_bmd[k], o_blocks[k]) for k in range(depth)]}
(WORK / "raon_drift.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
step("DONE", vel_cpp=result["velocity_maxdiff_CPP"], vel_torch_f16=result["velocity_maxdiff_TORCH_f16"],
     blocklast_cpp=cpp_bmd[-1], blocklast_torch_f16=f16_bmd[-1])
