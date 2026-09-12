"""FireRedTTS3 (#377) reference dump backend for the crispasr-diff harness.

Runs the REAL upstream inference path (fireredtts3.llm.fireredtts3_base
FireRedTTS3BaseCore.generate + RedAE + CamppEmbedding) end-to-end on CPU,
capturing every pipeline boundary plus the Gaussian noise draws so the C++
runtime can replay them bit-for-bit.

⚠ MEMORY: this loads base (8.5 GB fp32) + redae (3.8 GB fp32) — it does NOT
fit the 8 GB VPS. Run it on a Kaggle CPU kernel (~30 GB RAM); see
tools/kaggle/fireredtts3-refdump/.

Env:
  FIREREDTTS3_UPSTREAM     path to a clone of FireRedTeam/FireRedTTS3 (the
                           python package; REQUIRED)
  FIREREDTTS3_SYN_TEXT     target text  (default "Hello there, how are you today?")
  FIREREDTTS3_PROMPT_TEXT  transcript of the prompt WAV (default: JFK line)
  FIREREDTTS3_LANG         language tag (default English)
  FIREREDTTS3_SEED         seed (default 1234, upstream default)
  FIREREDTTS3_NT           flow timesteps (default 10)
  FIREREDTTS3_CFG          inference cfg (default 2.0)

The --audio argument of dump_reference.py is the PROMPT wav (16 kHz mono
float; e.g. samples/jfk.wav).

Stages:
  text_tokens      (T_text,)          token ids as f32 (prompt-token parity)
  spk_emb          (512,)             CAM++ x-vector
  spk_llm          (2048,)            spk_proj_llm(spk_emb)
  spk_dit          (512,)             spk_proj_dit(spk_emb)
  campp_fbank      (T_fb, 80)         mean-subtracted kaldi fbank
  prompt_latents   (T_lat, 64)        RedAE encode of padded 24 kHz prompt
  enc_hidden       (T50, 896)         RedAE encoder qwen3 out (pre-downsample)
  penc_prompt      (N_patch, 2048)    PatchEncoder over prompt latents
  prefill_embeds   (T_seq, 2048)      [spk_llm, text embeds, penc_prompt]
  llm_prefill_out  (T_seq, 2048)      backbone last_hidden_state (prefill)
  stop_scores      (n_steps+1,)       sigmoid stop score per AR step
  dit_cond_s0      (3, 1024)          dit_head(backbone_cond[:, -3:]) step 0
  noise_all        (n_steps, 4, 64)   the x0 randn draw of every AR step
  latents_steps    (n_steps, 4, 64)   denoised patch of every AR step
  latents_gen      (T_all, 64)        full generated latents (prompt removed)
  dec_hidden       (T_dec, 896)       RedAE decoder qwen3 out
  gen_audio        (n,)               final 24 kHz audio, prompt trimmed
  meta             (8,)               [seed, n_timesteps, cfg, sr, n_steps,
                                       T_lat, stop_threshold, prompt_pad_samples]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "text_tokens", "spk_emb", "spk_llm", "spk_dit", "campp_fbank",
    "prompt_latents", "enc_hidden", "penc_prompt", "prefill_embeds",
    "llm_prefill_out", "stop_scores", "dit_cond_s0", "noise_all",
    "latents_steps", "latents_gen", "dec_hidden", "gen_audio", "meta",
]

JFK_TEXT = ("And so my fellow Americans ask not what your country can do "
            "for you, ask what you can do for your country.")


def required_packages() -> list[str]:
    return ["torch", "torchaudio", "transformers",
            "FireRedTTS3 repo on FIREREDTTS3_UPSTREAM"]


def _force_sdpa(model) -> int:
    n = 0
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "sdpa"
            n += 1
    return n


def dump(model_dir: Path, audio: np.ndarray, stages: Set[str], **kwargs) -> Dict[str, np.ndarray]:
    up = os.environ.get("FIREREDTTS3_UPSTREAM")
    if not up or not Path(up).is_dir():
        raise SystemExit("FIREREDTTS3_UPSTREAM must point at a clone of "
                         "github.com/FireRedTeam/FireRedTTS3")
    sys.path.insert(0, up)

    import torch
    import torchaudio

    from fireredtts3.llm.fireredtts3_base import FireRedTTS3BaseCore
    from fireredtts3.redae.redae import RedAE
    from fireredtts3.campp.campp import CamppEmbedding, extract_kaldi_mel
    from fireredtts3.utils.text_tokenizer import load_text_tokenizer
    from fireredtts3.utils.utils import fix_seed

    text = os.environ.get("FIREREDTTS3_SYN_TEXT", "Hello there, how are you today?")
    prompt_text = os.environ.get("FIREREDTTS3_PROMPT_TEXT", JFK_TEXT)
    lang = os.environ.get("FIREREDTTS3_LANG", "English")
    seed = int(os.environ.get("FIREREDTTS3_SEED", "1234"))
    n_timesteps = int(os.environ.get("FIREREDTTS3_NT", "10"))
    cfg = float(os.environ.get("FIREREDTTS3_CFG", "2.0"))
    stop_threshold = 0.5

    md = Path(model_dir)
    caps: Dict[str, np.ndarray] = {}

    # ── RedAE encode of the prompt ────────────────────────────────────
    print("[frt-ref] loading RedAE ...", flush=True)
    redae = RedAE.from_pretrained(str(md / "redae"))
    redae.eval()
    _force_sdpa(redae)

    wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # (1, T) @16k
    wav24 = torchaudio.functional.resample(wav, 16000, redae.sample_rate)
    wav24 = RedAE.pad_to_multiple_of(wav24, redae.downsample_rate * 4)

    enc_hidden = {}

    def _enc_hook(mod, args, kwargs, out):
        enc_hidden["v"] = out.last_hidden_state.detach().float()

    h1 = redae.encoder.qwen3.register_forward_hook(_enc_hook, with_kwargs=True)
    with torch.no_grad():
        prompt_latents = redae.encode(wav24, redae.sample_rate).float()
    h1.remove()
    caps["prompt_latents"] = prompt_latents[0].numpy()
    caps["enc_hidden"] = enc_hidden["v"][0].numpy()
    print(f"[frt-ref] prompt_latents {tuple(prompt_latents.shape)}", flush=True)

    # ── CAM++ speaker embedding ──────────────────────────────────────
    campp = CamppEmbedding(str(md / "campp" / "campplus_voxceleb.bin"))
    with torch.no_grad():
        fbank = extract_kaldi_mel(wav24, redae.sample_rate)
        spk_emb = campp.forward(wav24, redae.sample_rate).float()
    caps["campp_fbank"] = fbank.numpy()
    caps["spk_emb"] = spk_emb[0].numpy()
    del campp

    # ── Tokenize (exact base-class template; no TN frontend) ─────────
    tok = load_text_tokenizer(str(md / "text_tokenizer"))
    input_text = f"<|{lang}|><|sot|>{prompt_text}{text}<|eot|>"
    ids = tok(input_text, truncation=False, padding=False,
              add_special_tokens=False)["input_ids"]
    text_tokens = torch.tensor([ids], dtype=torch.long)
    caps["text_tokens"] = np.asarray(ids, dtype=np.float32)
    print(f"[frt-ref] {len(ids)} text tokens", flush=True)

    # ── Core LLM-DiT ─────────────────────────────────────────────────
    print("[frt-ref] loading FireRedTTS3BaseCore ...", flush=True)
    core = FireRedTTS3BaseCore.from_pretrained(str(md / "fireredtts3_base"))
    core.eval()
    _force_sdpa(core)

    with torch.no_grad():
        caps["spk_llm"] = core.spk_proj_llm(spk_emb)[0].float().numpy()
        caps["spk_dit"] = core.spk_proj_dit(spk_emb)[0].float().numpy()
        caps["penc_prompt"] = core.patch_encoder(prompt_latents)[0].float().numpy()

    # Capture hooks / wrappers around the REAL generate() loop.
    noises: list[np.ndarray] = []
    step_latents: list[np.ndarray] = []
    stop_scores: list[float] = []
    prefill: Dict[str, np.ndarray] = {}
    dit_cond0: Dict[str, np.ndarray] = {}

    real_randn = torch.randn

    def randn_tee(*a, **k):
        t = real_randn(*a, **k)
        if tuple(t.shape) == (1, core.patch_size, core.redae_dim):
            noises.append(t[0].detach().float().numpy().copy())
        return t

    real_backbone = core._backbone_one_step

    def backbone_tee(input_embeds, cache=None):
        if cache is None and "embeds" not in prefill:
            prefill["embeds"] = input_embeds[0].detach().float().numpy().copy()
        out, new_cache = real_backbone(input_embeds, cache=cache)
        if "out" not in prefill:
            prefill["out"] = out[0].detach().float().numpy().copy()
        return out, new_cache

    real_flow = core._flow_one_step

    def flow_tee(hist_latents, backbone_cond, spk_cond, t_span, inference_cfg):
        if "v" not in dit_cond0:
            dit_cond0["v"] = backbone_cond[0].detach().float().numpy().copy()
        x1 = real_flow(hist_latents, backbone_cond, spk_cond, t_span, inference_cfg)
        step_latents.append(x1[0].detach().float().numpy().copy())
        return x1

    real_stop = core.stop_head

    class StopTee(torch.nn.Module):
        def forward(self, x):
            out = real_stop(x)
            if x.dim() == 2:  # called on backbone_out[:, -1] → (1, 2048)
                stop_scores.append(float(torch.sigmoid(out.squeeze(-1))[0]))
            return out

    core._backbone_one_step = backbone_tee
    core._flow_one_step = flow_tee
    core.stop_head = StopTee()
    torch.randn = randn_tee

    fix_seed(seed)
    print("[frt-ref] generating ...", flush=True)
    with torch.no_grad():
        gen_latents = core.generate(
            spk_emb=spk_emb,
            text_tokens=text_tokens,
            prompt_latents=prompt_latents,
            n_timesteps=n_timesteps,
            inference_cfg=cfg,
            stop_threshold=stop_threshold,
            min_gen_steps=6,
            max_gen_steps=None,
        )
    torch.randn = real_randn
    core.stop_head = real_stop

    n_steps = len(step_latents)
    print(f"[frt-ref] {n_steps} AR steps, latents {tuple(gen_latents.shape)}",
          flush=True)
    caps["prefill_embeds"] = prefill["embeds"]
    caps["llm_prefill_out"] = prefill["out"]
    caps["stop_scores"] = np.asarray(stop_scores, dtype=np.float32)
    caps["dit_cond_s0"] = dit_cond0["v"]
    caps["noise_all"] = np.stack(noises) if noises else np.zeros((0, 4, 64), np.float32)
    caps["latents_steps"] = np.stack(step_latents)
    caps["latents_gen"] = gen_latents[0].float().numpy()
    del core
    import gc
    gc.collect()

    # ── RedAE decode (prompt latents + generated, then trim prompt) ──
    dec_hidden = {}

    def _dec_hook(mod, args, kwargs, out):
        dec_hidden["v"] = out.last_hidden_state.detach().float()

    h2 = redae.decoder.qwen3.register_forward_hook(_dec_hook, with_kwargs=True)
    with torch.no_grad():
        gen_audio, sr = redae.decode(gen_latents)
    h2.remove()
    gen_audio = gen_audio[:, wav24.shape[1]:]
    caps["dec_hidden"] = dec_hidden["v"][0].numpy()
    caps["gen_audio"] = gen_audio[0].float().numpy()
    caps["meta"] = np.asarray(
        [seed, n_timesteps, cfg, sr, n_steps, prompt_latents.shape[1],
         stop_threshold, wav24.shape[1]], dtype=np.float32)
    print(f"[frt-ref] audio {gen_audio.shape[1]} samples @ {sr} Hz", flush=True)

    return {k: v for k, v in caps.items() if k in stages}
