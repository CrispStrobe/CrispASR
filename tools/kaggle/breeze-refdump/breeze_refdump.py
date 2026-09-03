#!/usr/bin/env python3
"""Breeze TTS 2 reference-oracle dump on a Kaggle GPU (#412, PHASE 1).

WHY KAGGLE: the upstream needs Linux + CUDA and ~7.7 GiB VRAM in eager mode
(README), and the checkpoint is 6.97 GB bf16 — it does not fit the 8 GB VPS.
Everything here is the reference half of the diff harness; the C++ half lands
in phase 2 and is compared with `tools/reference_backends/breeze_tts_2.py`.

WHAT IT DOES
  1. clones CrispASR (for samples/jfk.wav + the kaggle harness) and
     breezeblue-ai/breeze-tts (Apache-2.0 inference code);
  2. loads BreezeBlue/Breeze-TTS-2 in bf16 with attn_implementation="eager"
     (breeze.py:55 resolver — SDPA/flash both drop the additive mask we need
     to reproduce bit-for-bit);
  3. runs ONE Voice-Clone synthesis on a fixed seed / fixed text / the repo's
     own samples/jfk.wav as the reference clip, template "ref_edit_tata"
     restricted to the clone branch (templates.py:74-84);
  4. dumps per-stage .npy at every architectural boundary;
  5. uploads them to cstr/crispasr-regression-fixtures/breeze-tts-2/.

DETERMINISM
  BREEZE_GREEDY=1 (the default) forces temperature -> 0 / do_sample -> False /
  cfg_scale -> 1.0 on both the backbone and the depth decoder, so the token
  IDs are reproducible and the C++ port can be held to exact-ID equality
  (validation plan step 3). Set BREEZE_GREEDY=0 to dump the shipped sampling
  defaults (temp 0.9) instead — useful only for listening tests.

STAGES DUMPED (all float32, row-major, batch dim dropped)
  ref_audio                  (N,)            24 kHz mono PCM fed to the codec
  ref_codes                  (T_ref, 16)     int32 codec codes of the ref clip
  prompt_input_ids           (L,)            int32 flattened prompt token ids
  prompt_text_ids_mask       (L,)            int32 1 where a TEXT token sits
  prompt_text_ids_len        (S,)            int32 per-segment text lengths
  te_seg{K}_hidden           (len_K, 1152)   text-encoder last_hidden_state,
                                             ONE ROW PER SEGMENT (segments are
                                             encoded independently —
                                             breeze.py:1418-1436)
  te_seg{K}_layer{J}         (len_K, 1152)   per-layer text-encoder hidden
                                             states (J = 0..26; 0 is the
                                             scaled embedding). Guarded by
                                             BREEZE_DUMP_TE_LAYERS=1.
  te_proj_out                (n_text, 2048)  text_encoder_proj output, the
                                             rows written into inputs_embeds
                                             at text positions (breeze.py:1458)
  backbone_inputs_embeds     (L, 2048)       assembled prefill embeddings
  backbone_hidden_frame0     (2048,)         last_hidden_state of the LAST
                                             prefill position = the depth
                                             decoder's frame-0 conditioning
  backbone_layer{J}_frame0   (2048,)         per-layer hidden at that position
  backbone_logits_frame0     (2052,)         lm_head output, frame 0
  dd_codes_frame0_stepwise   (16,)           int32 frame-0 codes from the
                                             hand-rolled per-codebook loop
  dd_codes_frame{F}          (16,)           int32 codes from generate(), F = 0..4
                                             (must equal *_stepwise at F=0)
  dd_logits_frame0_cb{C}     (2051,)         depth-decoder logits per codebook
                                             step, C = 1..15, frame 0
  codes                      (T, 16)         int32 full generated code grid
  codec_audio                (N_out,)        24 kHz PCM from the Qwen3-TTS
                                             tokenizer decode
  meta.json                  the constants + shapes + the exact prompt

ENV
  BREEZE_TEXT        target text (default: a short EN line)
  BREEZE_REF_TEXT    transcript of samples/jfk.wav
  BREEZE_SEED        default 42
  BREEZE_GREEDY      1 (default) / 0
  BREEZE_MAX_FRAMES  cap the AR loop (default 24; enough for 5 dumped frames)
  BREEZE_DUMP_TE_LAYERS  1 to also dump all 27 text-encoder layer states
  CRISPASR_REF       CrispASR branch to clone (default main)

DO NOT run/push this from an agent session — the maintainer pushes it.
  kaggle kernels push -p tools/kaggle/breeze-refdump
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp")
TMP.mkdir(parents=True, exist_ok=True)
DUMPS = WORK / "breeze-dumps"
DUMPS.mkdir(parents=True, exist_ok=True)

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
BREEZE_URL = "https://github.com/breezeblue-ai/breeze-tts.git"
CLONE = TMP / "CrispASR"
BREEZE = TMP / "breeze-tts"
CKPT = TMP / "breeze-tts-2"

HF_MODEL = "BreezeBlue/Breeze-TTS-2"
HF_FIXTURES = "cstr/crispasr-regression-fixtures"
FIXTURE_PREFIX = "breeze-tts-2"

SEED = int(os.environ.get("BREEZE_SEED", "42"))
GREEDY = os.environ.get("BREEZE_GREEDY", "1") == "1"
MAX_FRAMES = int(os.environ.get("BREEZE_MAX_FRAMES", "24"))
DUMP_TE_LAYERS = os.environ.get("BREEZE_DUMP_TE_LAYERS", "0") == "1"
N_DUMP_FRAMES = 5

SYN_TEXT = os.environ.get(
    "BREEZE_TEXT",
    "The quick brown fox jumps over the lazy dog.",
)
REF_TEXT = os.environ.get(
    "BREEZE_REF_TEXT",
    "And so my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country.",
)


def step(name, **kv):
    print(f"[{time.strftime('%H:%M:%S')}] {name} " + json.dumps(kv, default=str),
          flush=True)


# ── clones + harness ──────────────────────────────────────────────────────
# Kaggle workers have flaky GitHub access (gotcha #18; run 5 here died with
# "could not read Username for 'https://github.com'" on a clean clone), so
# retry. This retry CANNOT live in kaggle_harness.py: the harness is imported
# *from* the CrispASR clone, so this bootstrap runs before it exists. Same
# shape as the raon-roundtrip fix (d921bf2d).
for url, dst, ref in ((CRISPASR_URL, CLONE, CRISPASR_REF), (BREEZE_URL, BREEZE, None)):
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [url, str(dst)]
    for attempt in range(4):
        if dst.exists():
            shutil.rmtree(dst)
        r = subprocess.run(cmd, timeout=1800)
        if r.returncode == 0:
            break
        print(f"clone {url} attempt {attempt + 1} failed rc={r.returncode}; retrying", flush=True)
        time.sleep(15)
    else:
        raise SystemExit(f"clone failed after 4 attempts: {url}")

sys.path.insert(0, str(CLONE / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
# Bump when the arms, capture or predicate change (see kh.provenance).
SCRIPT_VERSION = "2026-09-03.1"
kh.provenance(SCRIPT_VERSION, clone_dir=CLONE)
HF_TOKEN = kh.resolve_hf_token()
step("cloned", crispasr_ref=CRISPASR_REF, hf_token_ok=bool(HF_TOKEN))

# ── deps ──────────────────────────────────────────────────────────────────
# Kaggle's torch is pre-built for its GPU arch — NEVER reinstall it (a fresh
# torch/torchaudio drags a mismatched CUDA kernel image). Only the small
# pure-python pieces the upstream needs are installed here. transformers is
# pinned to the version the checkpoint was saved with (requirements.txt).
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.57.3", "qwen-tts==0.1.1", "soundfile",
                "huggingface_hub", "safetensors"], check=False)
step("deps_installed")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import soundfile as sf  # noqa: E402

torch.set_grad_enabled(False)
assert torch.cuda.is_available(), "this kernel needs enable_gpu=true"

# GPU LOTTERY GUARD. Kaggle assigns P100 (sm_60) or T4 (sm_75) and the
# `machine_shape` metadata field does NOT reliably select one (verified
# 2026-09-02: a kernel carrying GPU_T4_X2 still drew a P100). Kaggle's OWN
# preinstalled torch is now built for sm_70/75/90 only, so a P100 draw is
# FATAL, not merely slow — it dies with "no kernel image is available for
# execution on the device" AFTER the ~7 GB checkpoint download and model
# load (observed here, run 3, ~20 wasted minutes). Check first and exit
# cheaply so a redraw costs ~1 minute.
_cap = torch.cuda.get_device_capability(0)
_name = torch.cuda.get_device_name(0)
print(f"[gpu] {_name} sm_{_cap[0]}{_cap[1]}", flush=True)
if _cap < (7, 0):
    print(
        f"P100_LOTTERY_RETRY: drew {_name} (sm_{_cap[0]}{_cap[1]}); this torch "
        f"build has no kernels below sm_70. Nothing was computed — re-push to redraw.",
        flush=True,
    )
    Path("/kaggle/working/lottery_retry.json").write_text(
        json.dumps({"conclusive": False, "reason": "gpu_too_old", "gpu": _name,
                    "capability": f"sm_{_cap[0]}{_cap[1]}"}, indent=1))
    raise SystemExit(0)

DEVICE = "cuda:0"

# ── checkpoint ────────────────────────────────────────────────────────────
from huggingface_hub import snapshot_download  # noqa: E402

snapshot_download(HF_MODEL, local_dir=str(CKPT), token=HF_TOKEN or None,
                  allow_patterns=["*.json", "*.safetensors", "LICENSE",
                                  "audio_tokenizer/*"])
step("checkpoint_downloaded", gb=round(sum(
    p.stat().st_size for p in CKPT.rglob("*") if p.is_file()) / 1e9, 2))

sys.path.insert(0, str(BREEZE))
from breeze_infer.runtime import (  # noqa: E402
    load_runtime, set_all_seeds, update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs  # noqa: E402

tokenizer, model, audio_tokenizer = load_runtime(
    CKPT, device=DEVICE, attn_implementation="eager",
)
update_generation_config_for_breeze(model)
if GREEDY:
    # Exact-ID equality target. transformers ignores temperature when
    # do_sample=False, but zeroing it too keeps the intent explicit.
    for gc in (model.generation_config, model.depth_decoder.generation_config):
        gc.do_sample = False
        gc.temperature = 1.0
        gc.top_k = 0
        gc.top_p = 1.0
model.generation_config.max_new_tokens = MAX_FRAMES
step("model_loaded", greedy=GREEDY, max_frames=MAX_FRAMES,
     dtype=str(next(model.parameters()).dtype))


def save(name, arr):
    a = np.ascontiguousarray(arr)
    np.save(DUMPS / f"{name}.npy", a)
    print(f"  dump {name:34s} {a.shape} {a.dtype}", flush=True)


def f32(t):
    return t.detach().float().cpu().numpy()


# ── reference audio → codec codes ─────────────────────────────────────────
REF_WAV = CLONE / "samples" / "jfk.wav"
wav, sr = sf.read(str(REF_WAV), always_2d=True, dtype="float32")
wav = np.mean(wav, axis=1)
enc = audio_tokenizer.encode(wav, sr=sr)
ref_codes = np.asarray(enc["audio_codes"][0], dtype=np.int32)   # (T_ref, 16)
save("ref_audio", wav.astype(np.float32))
save("ref_codes", ref_codes)
step("ref_encoded", sr=sr, n_samples=int(wav.shape[0]), ref_frames=int(ref_codes.shape[0]))

# ── prompt assembly (Voice Clone) ─────────────────────────────────────────
# `ref_edit_tata` is the only template that carries a reference clip; the
# clone branch is exactly `_ref_clone_tata_segments` (templates.py:74-84):
#   [S0]{ref_text}  <|AUDIO|>*T_ref <|audio_eos|>  [S0]{text}
# Passing guidance_scale=1.0 with both dual scales None keeps it single-branch
# (templates.py:292-297) — the CFG multi-branch dumps are a phase-2 follow-up.
request = {
    "id": "breeze-ref",
    "text": SYN_TEXT,
    "instruction": "Speak clearly and naturally.",
    "speaker": "S0",
    "ref_audio_path": str(REF_WAV),
    "ref_text": REF_TEXT,
}
set_all_seeds(SEED)
inputs = prepare_inputs(
    tokenizer, audio_tokenizer, model, [request],
    get_template("ref_edit_tata"),
    guidance_scale=1.0, guidance_scale_ref=None, guidance_scale_ins=None,
)
save("prompt_input_ids", f32(inputs["input_ids"][0]).astype(np.int32))
save("prompt_text_ids_mask", f32(inputs["text_ids_mask"][0]).astype(np.int32))
save("prompt_text_ids_len", f32(inputs["text_ids_len"]).astype(np.int32))
step("prompt_built", L=int(inputs["input_ids"].shape[1]),
     n_segments=int(inputs["text_ids_len"].numel()))

# ── stage 1+2: text encoder + projection ──────────────────────────────────
# Re-run the model's own segment splitter so the dumped segments line up
# exactly with what convert_input_ids_to_embeds feeds the encoder.
text_ids = inputs["input_ids"][0][inputs["text_ids_mask"][0]]
seg_lens = [int(x) for x in inputs["text_ids_len"].reshape(-1).tolist()]
segments = list(torch.split(text_ids, seg_lens, dim=0))

seg_hs, seg_layer_hs = model._batched_text_encoder_forward(
    segments, output_hidden_states=DUMP_TE_LAYERS,
)
for k, hs in enumerate(seg_hs):
    save(f"te_seg{k}_hidden", f32(hs))
if DUMP_TE_LAYERS and seg_layer_hs:
    for k, layers in enumerate(seg_layer_hs):
        if layers is None:
            continue
        for j, lhs in enumerate(layers):
            save(f"te_seg{k}_layer{j}", f32(lhs))

inputs_embeds, _ = model.convert_input_ids_to_embeds(
    input_ids=inputs["input_ids"],
    text_ids_mask=inputs["text_ids_mask"],
    text_ids_len=inputs["text_ids_len"],
    attention_mask=inputs["attention_mask"],
)
save("te_proj_out", f32(inputs_embeds[0][inputs["text_ids_mask"][0]]))
step("text_encoder_done", n_segments=len(seg_hs),
     seg_shapes=[list(h.shape) for h in seg_hs])

# ── stage 3: backbone prefill ─────────────────────────────────────────────
# _merge_input_ids_with_input_values folds the ref-audio codebook embeddings
# into the same inputs_embeds at the <|AUDIO|> positions (breeze.py:1544+).
merged = model._merge_input_ids_with_input_values(
    input_ids=inputs["input_ids"],
    input_values=inputs["input_values"],
    text_ids_mask=inputs["text_ids_mask"],
    text_ids_len=inputs["text_ids_len"],
    attention_mask=inputs["attention_mask"],
)
# It returns a DICT (breeze.py, end of _merge_input_ids_with_input_values):
# {"inputs_embeds", "labels", "text_encoder_layer_hidden_states", "text_ids_mask"}
merged_embeds = merged["inputs_embeds"] if isinstance(merged, dict) else merged
save("backbone_inputs_embeds", f32(merged_embeds[0]))

bb_out = model.backbone_model(
    inputs_embeds=merged_embeds,
    attention_mask=inputs["attention_mask"],
    use_cache=False,
    output_hidden_states=True,
    return_dict=True,
)
h_last = bb_out.last_hidden_state[0, -1]              # (2048,)
save("backbone_hidden_frame0", f32(h_last))
for j, hs in enumerate(bb_out.hidden_states):
    save(f"backbone_layer{j}_frame0", f32(hs[0, -1]))
logits0 = model.lm_head(h_last)                       # (2052,)
save("backbone_logits_frame0", f32(logits0))
step("backbone_prefill_done", L=int(merged_embeds.shape[1]),
     n_hidden_states=len(bb_out.hidden_states),
     argmax_cb0=int(logits0.argmax().item()))

# ── stage 4: depth decoder, frame 0, per-codebook logits ──────────────────
# The backbone emits codebook 0; the depth decoder then runs num_codebooks-1
# = 15 steps, one head per codebook (breeze.py:604-628). Reproduced here step
# by step so every head gets its own dump.
RESERVED = model._reserved_codec_token_ids()   # range(2048, 2051)


def pick(lg):
    """argmax with the reserved codec ids suppressed, matching
    `suppress_tokens=self._reserved_codec_token_ids()` on the shipped path
    (generation_breeze.py:976-980) and `_mask_reserved_codec_logits` (:125-131).
    Without this the greedy pick can land on 2048-2050, which the codec
    cannot decode."""
    m = lg.clone()
    m[RESERVED] = float("-inf")
    return int(m.argmax().item())


# Codebook 0 comes from the backbone head; only [0, audio_vocab_size) are
# codes (index 2051 is the EOS class).
cb0 = pick(logits0[: model.config.audio_vocab_size])
dd = model.depth_decoder
seq = torch.tensor([[0, cb0]], device=DEVICE, dtype=torch.long)  # [placeholder, cb0]
frame0 = [cb0]
for c in range(1, model.config.num_codebooks):
    out = dd(input_ids=seq, backbone_last_hidden_state=h_last.unsqueeze(0),
             use_cache=False, return_dict=True)
    lg = out.logits[0, -1].float()
    save(f"dd_logits_frame0_cb{c}", f32(lg))
    nxt = pick(lg)
    frame0.append(nxt)
    seq = torch.cat([seq, torch.tensor([[nxt]], device=DEVICE, dtype=torch.long)], dim=1)
save("dd_codes_frame0_stepwise", np.asarray(frame0, dtype=np.int32))
step("depth_decoder_frame0_done", codes=frame0)

# ── stage 5: full generate (codes) + codec decode ─────────────────────────
set_all_seeds(SEED)
gen = model.generate(**{k: v for k, v in inputs.items() if v is not None},
                     output_audio=True, return_dict_in_generate=True)
codes = gen.sequences                                      # (B, T, 16)
if codes.ndim == 3:
    codes = codes[0]
codes = codes.detach().cpu().numpy().astype(np.int32)      # (T, 16)
# Truncate at the first all-pad frame, exactly like the codec path does
# (generation_breeze.py:1238-1248) — EOS frames are stored as all
# codebook_pad_token_id and must not reach the codec.
pad = int(model.config.codebook_pad_token_id)
is_pad = (codes == pad).all(axis=-1)
cut = int(np.argmax(is_pad)) if is_pad.any() else codes.shape[0]
if cut != codes.shape[0]:
    print(f"  truncating codes at first pad frame: {codes.shape[0]} -> {cut}", flush=True)
codes = codes[:cut]
save("codes", codes)
for f in range(min(N_DUMP_FRAMES, codes.shape[0])):
    save(f"dd_codes_frame{f}", codes[f])

audio = gen.audio[0] if getattr(gen, "audio", None) else None
if audio is None:
    dec = audio_tokenizer.decode(torch.as_tensor(codes)[None].to(DEVICE))
    audio = dec["audio"][0] if isinstance(dec, dict) else dec
audio = np.asarray(audio.detach().float().cpu() if torch.is_tensor(audio) else audio,
                   dtype=np.float32).reshape(-1)
save("codec_audio", audio)
sf.write(str(WORK / "breeze-ref.wav"), audio, 24000, subtype="PCM_16")
step("generate_done", frames=int(codes.shape[0]), n_samples=int(audio.shape[0]))

# ── meta ──────────────────────────────────────────────────────────────────
cfg = model.config
meta = {
    "model": HF_MODEL,
    "seed": SEED,
    "greedy": GREEDY,
    "max_frames": MAX_FRAMES,
    "syn_text": SYN_TEXT,
    "ref_text": REF_TEXT,
    "ref_wav": "samples/jfk.wav",
    "template": "ref_edit_tata (clone branch, cfg_scale=1.0)",
    "attn_implementation": "eager",
    "dtype": "bfloat16",
    "num_codebooks": int(cfg.num_codebooks),
    "audio_vocab_size": int(cfg.audio_vocab_size),
    "audio_token_id": int(cfg.audio_token_id),
    "audio_eos_token_id": int(cfg.audio_eos_token_id),
    "codebook_pad_token_id": int(cfg.codebook_pad_token_id),
    "backbone_eos_token_id": int(cfg.audio_vocab_size),
    "te_layers": int(cfg.text_encoder_config.num_hidden_layers),
    "bb_layers": int(cfg.backbone_config["num_hidden_layers"]
                     if isinstance(cfg.backbone_config, dict)
                     else cfg.backbone_config.num_hidden_layers),
    "dd_layers": int(cfg.depth_decoder_config.num_hidden_layers),
    "shapes": {p.stem: list(np.load(p).shape) for p in sorted(DUMPS.glob("*.npy"))},
}
(DUMPS / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
step("meta_written", n_dumps=len(meta["shapes"]))

# ── upload ────────────────────────────────────────────────────────────────
if HF_TOKEN:
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_FIXTURES, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(DUMPS),
        path_in_repo=FIXTURE_PREFIX,
        repo_id=HF_FIXTURES,
        repo_type="dataset",
        commit_message="Add Breeze-TTS-2 per-stage reference dumps (#412 phase 1)",
    )
    api.upload_file(
        path_or_fileobj=str(WORK / "breeze-ref.wav"),
        path_in_repo=f"{FIXTURE_PREFIX}/breeze-ref.wav",
        repo_id=HF_FIXTURES, repo_type="dataset",
        commit_message="Add Breeze-TTS-2 reference synthesis (ASR-roundtrip target)",
    )
    step("uploaded", repo=HF_FIXTURES, prefix=FIXTURE_PREFIX)
else:
    step("no_hf_token_staged_locally", dir=str(DUMPS))

step("done")
print("[DONE]", flush=True)
