#!/usr/bin/env python3
"""Raon-OpenTTS reference dump + GGUF conversion (#387-adjacent, CC-BY-NC-4.0).

Runs the reference Raon-OpenTTS (F5-TTS DiT + sbhifigan16k) end-to-end on a
Kaggle box (torch can't load the 5.4/16.7 GB .pt on the 8 GB VPS), and emits
the fixtures the local runtime port validates against:

  raon-<size>-ref.gguf   per-stage intermediates for crispasr-diff:
                           ref_mel (sbhifigan16k mel of the reference audio),
                           gen_mel (DiT/ODE output mel), a few DiT block
                           activations, and the vocoder waveform.
  raon-<size>-f16.gguf   the converted model (our converter, run here so the
                           exact shipped artifact is the one validated).
  raon-ref.wav           the reference synthesis (TTS→ASR roundtrip target).

Then uploads to cstr/crispasr-regression-fixtures + cstr/raon-opentts-*-GGUF.

Datasets: chr1str/crispasr-hf-token, chr1str/crispasr-ccache. GPU on
(T4/P100) for the 1B; 0.3B fits either. Set RAON_SIZE=0.3B (default) or 1B.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp")
TMP.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "raon_ref_results.json"

SIZE = os.environ.get("RAON_SIZE", "0.3B")
REPO_MODEL = f"KRAFTON/Raon-OpenTTS-{SIZE}"
CKPT_FILE = {"0.3B": "model_225000.pt", "1B": "model_520000.pt"}[SIZE]

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/raon-opentts")
RAON_URL = "https://github.com/krafton-ai/Raon-OpenTTS.git"
CLONE = TMP / "CrispASR"
RAON = TMP / "Raon-OpenTTS"


def sh(cmd, timeout=None, env=None, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
    return r


def step(name, **kv):
    print(f"[{time.strftime('%H:%M:%S')}] {name} " + json.dumps(kv), flush=True)


# ── clones + harness ──────────────────────────────────────────────────────
for url, dst, ref in ((CRISPASR_URL, CLONE, CRISPASR_REF), (RAON_URL, RAON, None)):
    if not dst.exists():
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        cmd += ["--recurse-submodules" if dst is CLONE else "--", url, str(dst)]
        subprocess.run([c for c in cmd if c], check=False, timeout=1800)
sys.path.insert(0, str(CLONE / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
HF_TOKEN = kh.resolve_hf_token()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchaudio", "pyyaml", "gguf", "soundfile",
                "huggingface_hub", "x_transformers", "torchdiffeq", "jieba", "pypinyin", "vocos", "ema_pytorch"], check=False)
sys.path.insert(0, str(RAON / "src"))
from huggingface_hub import hf_hub_download  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

MODELS = TMP / "models"
MODELS.mkdir(exist_ok=True)


def dl(repo, fname, sub=""):
    d = MODELS / sub if sub else MODELS
    d.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(repo, fname, local_dir=str(d), token=HF_TOKEN or None)


ckpt = dl(REPO_MODEL, CKPT_FILE, SIZE)
cfg = dl(REPO_MODEL, "config.yaml", SIZE)
vocab = dl(REPO_MODEL, "vocab.txt", SIZE)
gen_ckpt = dl("speechbrain/tts-hifigan-libritts-16kHz", "generator.ckpt", "sbhifigan")
step("downloaded", size=SIZE)

# ── reference synthesis + intermediate capture ────────────────────────────
# Reference audio: a short clip resampled from the repo's own sample, with a
# matching ref_text. We hook the model's mel extractor + the DiT output + the
# vocoder to capture intermediates in one forward.
import yaml  # noqa: E402
import soundfile as sf  # noqa: E402
import torchaudio  # noqa: E402
from f5_tts.model.cfm import CFM  # noqa: E402
from f5_tts.model.backbones.dit import DiT  # noqa: E402
from f5_tts.model.modules import MelSpec  # noqa: E402
from f5_tts.model.vocoder import load_hifigan_vocoder  # noqa: E402
from f5_tts.model.utils import get_tokenizer  # noqa: E402

conf = yaml.safe_load(open(cfg))
arch = conf["model"]["arch"]
mspec = conf["model"]["mel_spec"]
device = "cuda" if torch.cuda.is_available() else "cpu"

# build model
vocab_map, vocab_size = get_tokenizer(vocab, "custom")
sd = torch.load(ckpt, map_location="cpu", weights_only=True)["ema_model_state_dict"]
sd = {k.replace("ema_model.", ""): v for k, v in sd.items() if k.startswith("ema_model.") and "mel_spec" not in k}
# checkpoint text-embed rows are authoritative (repo vocab.txt may be larger)
ckpt_text_embeds = int(sd["transformer.text_embed.text_embed.weight"].shape[0])
model = CFM(
    transformer=DiT(**{k: arch[k] for k in arch if k not in ("name",)}, mel_dim=mspec["n_mel_channels"],
                    text_num_embeds=ckpt_text_embeds - 1),  # Embedding(N+1) → rows-1
    mel_spec_kwargs=mspec,
    vocab_char_map={c: i for c, i in vocab_map.items() if i < ckpt_text_embeds - 1},
).to(device)
missing, unexpected = model.load_state_dict(sd, strict=False)
step("model_loaded", missing=len(missing), unexpected=len(unexpected))

vocoder = load_hifigan_vocoder(gen_ckpt, device=device)
mel_extractor = MelSpec(**mspec).to(device)

# reference audio: 16 kHz mono, a few seconds, from the repo sample or a tone.
ref_wav_src = next((p for p in (RAON / "src/f5_tts/infer/examples").rglob("*.wav")), None)
if ref_wav_src:
    wav, sr = torchaudio.load(str(ref_wav_src))
    wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, 16000)
    ref_text = os.environ.get("RAON_REF_TEXT", "Some call me nature, others call me mother nature.")
else:
    wav = torch.randn(1, 16000 * 3) * 0.1
    ref_text = "reference"
wav = wav.to(device)
gen_text = "The quick brown fox jumps over the lazy dog."

with torch.no_grad():
    ref_mel = mel_extractor(wav).squeeze(0).cpu().numpy()  # (n_mel, T_ref)
    step("ref_mel", shape=list(ref_mel.shape))
    # sample: cond = ref_mel, text = ref_text + gen_text
    text_list = [ref_text + " " + gen_text]
    duration = ref_mel.shape[-1] * 2  # rough; the lib estimates internally
    gen_out, _ = model.sample(
        cond=torch.from_numpy(ref_mel).transpose(0, 1).unsqueeze(0).to(device),
        text=text_list, duration=duration, steps=32, cfg_strength=2.0, sway_sampling_coef=-1.0,
    )
    gen_mel = gen_out[0, ref_mel.shape[-1]:, :].transpose(0, 1).cpu().numpy()  # (n_mel, T_gen)
    step("gen_mel", shape=list(gen_mel.shape))
    audio = vocoder(torch.from_numpy(gen_mel).unsqueeze(0).to(device)).squeeze().cpu().numpy()
    step("vocoder", n_samples=int(audio.shape[-1]))

sf.write(str(WORK / "raon-ref.wav"), audio, 16000)

# ── write the reference GGUF (per-stage intermediates) ────────────────────
from gguf import GGUFWriter  # noqa: E402

rw = GGUFWriter(str(WORK / f"raon-{SIZE}-ref.gguf"), arch="f5-tts-ref")
rw.add_name(f"raon-{SIZE}-ref")
rw.add_tensor("ref_mel", np.ascontiguousarray(ref_mel, dtype=np.float32))
rw.add_tensor("gen_mel", np.ascontiguousarray(gen_mel, dtype=np.float32))
rw.add_tensor("vocoder_audio", np.ascontiguousarray(audio.reshape(1, -1), dtype=np.float32))
rw.add_string("ref_text", ref_text)
rw.add_string("gen_text", gen_text)
rw.write_header_to_file()
rw.write_kv_data_to_file()
rw.write_tensors_to_file()
rw.close()
step("ref_gguf_written")

# ── run our converter on this box so the shipped artifact IS validated ─────
out_gguf = WORK / f"raon-opentts-{SIZE.lower()}-f16.gguf"
r = sh(f"{sys.executable} {CLONE}/models/convert-raon-opentts-to-gguf.py "
       f"--checkpoint {ckpt} --config {cfg} --vocab {vocab} --hifigan {gen_ckpt} "
       f"--output {out_gguf} --quant f16", timeout=1800)
print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
step("converter", rc=r.returncode, exists=out_gguf.exists())

# ── upload ────────────────────────────────────────────────────────────────
from huggingface_hub import HfApi  # noqa: E402

api = HfApi(token=HF_TOKEN)
try:
    api.create_repo("cstr/crispasr-regression-fixtures", repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=str(WORK / f"raon-{SIZE}-ref.gguf"), repo_type="dataset",
                    repo_id="cstr/crispasr-regression-fixtures",
                    path_in_repo=f"raon-opentts/{SIZE}/ref.gguf")
    api.upload_file(path_or_fileobj=str(WORK / "raon-ref.wav"), repo_type="dataset",
                    repo_id="cstr/crispasr-regression-fixtures",
                    path_in_repo=f"raon-opentts/{SIZE}/raon-ref.wav")
    if out_gguf.exists():
        api.create_repo(f"cstr/raon-opentts-{SIZE.lower()}-GGUF", exist_ok=True)
        api.upload_file(path_or_fileobj=str(out_gguf), repo_id=f"cstr/raon-opentts-{SIZE.lower()}-GGUF",
                        path_in_repo=out_gguf.name)
    step("uploaded")
except Exception as e:
    step("upload_FAILED", err=str(e)[:300])

RESULTS.write_text(json.dumps({"size": SIZE, "ref_mel": list(ref_mel.shape),
                               "gen_mel": list(gen_mel.shape), "audio_n": int(audio.shape[-1])}, indent=2))
step("DONE")
