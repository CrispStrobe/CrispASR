"""Hikari (sbintuitions/hikari-medium) reference dump backend.

Runs the UPSTREAM model code (github.com/sbintuitions/hikari,
src/hikari/models/*) on CPU in float32 and replays the server's streaming
policy (src/hikari/server/model_wrapper.py ModelWrapper.get_one_token) one
80 ms chunk at a time, exactly as hikari-server does for every chunk a
hikari-client sends:

  * the audio window is the last `decoder_context` * 1280 samples,
  * its log-mel (openai-whisper log_mel_spectrogram, window-global max-8 clip)
    is zero-padded to 3000 frames and run through the FULL causal encoder,
  * the decoder runs on the token list (zero-padded to the window),
    logits at position pos = min(window-1, j+3),
  * repetition penalty on the arg-max if it is one of the last 5 tokens
    (and not WAIT=93), wait penalty subtracted from logit 93, arg-max,
  * the wait penalty is boosted after 10 WAITs during speech and decays
    toward the baseline on every emitted token.

Speech detection: the upstream stateful Silero TorchScript model (from the
`silero-vad` pip package, data/silero_vad.jit — what torch.hub returns),
called once per step on the newest 512 samples. It feeds the boost, and the
boost is what makes the model emit during speech: with HIKARI_VAD=0 jfk 0-4 s
is 48/48 WAIT. Per-step probabilities are dumped (stream_speech_probs) so
crispasr-diff can separate VAD drift from model drift.

Two deliberate, semantics-preserving departures from the server, both for CPU
time, both opt-out:
  * HIKARI_REF_TRIM=1 (default) passes only the first pos+1 decoder ids instead
    of zero-padding to the window: the decoder self-attention is causal, so the
    logits at pos are the same function of the same inputs.
  * the server casts to fp16 on CUDA; this runs fp32 on CPU.

Env knobs (defaults = hikari-client UI defaults):
  HIKARI_TASK=translate|transcribe   HIKARI_TGT=de|ja|ru|en
  HIKARI_WP_BASE=0.0 HIKARI_WP_BOOST=0.6 HIKARI_WP_DECAY=0.3 HIKARI_REP=40
  HIKARI_CTX=337 (decoder_context)   HIKARI_TAIL_MS=0 (zeros appended)
  HIKARI_SRC=<path to hikari/src>    HIKARI_THREADS=4

Stages (all from the FINAL step):
  mel_spectrogram   (80, 3000)        normalised, zero-padded log-mel
  encoder_output    (1500, 1024)
  decoder_input_ids (n,)  int32       the decoder ids of that step
  decoder_logits    (n, 51865)        logits for every position of that step
  stream_tokens     (n_steps,) int32  token chosen at each step (93 = WAIT)
"""

from __future__ import annotations

import os
import sys
import time
import types
import importlib
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = ["mel_spectrogram", "encoder_output", "decoder_logits", "stream_tokens"]

WAIT = 93
CHUNK = 1280  # 80 ms @ 16 kHz = 160 * 2 * decoder_time_dilation
LANG_TOK = {"en": 50259, "de": 50261, "ru": 50263, "ja": 50266, "zh": 50260}


def _whisper_log_mel():
    """openai-whisper's log_mel_spectrogram without running whisper/__init__
    (which drags in numba/tiktoken)."""
    spec = importlib.util.find_spec("whisper")
    if spec is None:
        raise SystemExit("pip install openai-whisper (only whisper/audio.py + its assets are used)")
    if "whisper" not in sys.modules:
        pkg = types.ModuleType("whisper")
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules["whisper"] = pkg
    return importlib.import_module("whisper.audio").log_mel_spectrogram


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


def dump(model_dir: Path, audio: np.ndarray, stages: Set[str], max_new_tokens: int = 0, **kw) -> Dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    src = _env("HIKARI_SRC", "")
    if src:
        sys.path.insert(0, src)
    from hikari.models import HikariConfig, HikariForConditionalGeneration

    torch.set_num_threads(_env("HIKARI_THREADS", 4, int))
    log_mel = _whisper_log_mel()

    task = _env("HIKARI_TASK", "translate")
    tgt = _env("HIKARI_TGT", "de")
    wp_base = _env("HIKARI_WP_BASE", 0.0, float)
    wp_boost = _env("HIKARI_WP_BOOST", 0.6, float)
    wp_decay = _env("HIKARI_WP_DECAY", 0.3, float)
    rep = _env("HIKARI_REP", 40.0, float)
    tail_ms = _env("HIKARI_TAIL_MS", 0, int)
    trim = _env("HIKARI_REF_TRIM", 1, int) != 0

    cfg = HikariConfig.from_pretrained(str(model_dir))
    cfg.use_cache = False
    model = HikariForConditionalGeneration.from_pretrained(
        str(model_dir), config=cfg, attn_implementation="sdpa", dtype=torch.float32).eval()
    W = min(max(_env("HIKARI_CTX", 337, int), 50), cfg.max_target_positions)
    dil = cfg.decoder_time_dilation

    if task == "transcribe":
        ids = [50258, 50259, 50359, 50363]
    else:
        ids = [50258, LANG_TOK[tgt], 50358, 50363]
    print(f"  hikari ref: task={task} tgt={tgt} W={W} wp=({wp_base},{wp_boost},{wp_decay}) rep={rep} "
          f"tail={tail_ms}ms trim={trim}")

    # The client zero-pads the last chunk (client/app.py make_audio).
    a = audio.astype(np.float32)
    if tail_ms > 0:
        a = np.concatenate([a, np.zeros(tail_ms * 16, np.float32)])
    if len(a) % CHUNK:
        a = np.concatenate([a, np.zeros(CHUNK - len(a) % CHUNK, np.float32)])

    # Silero VAD, as model_wrapper.py: stateful TorchScript model called once
    # per step on the newest 512 samples, speech = prob > 0.8. HIKARI_VAD=0
    # (speech always False) disables the wait-penalty boost.
    vad = None
    if _env("HIKARI_VAD", 1, int):
        spec = importlib.util.find_spec("silero_vad")
        if spec is None:
            raise SystemExit("pip install silero-vad (or HIKARI_VAD=0)")
        vad = torch.jit.load(os.path.join(list(spec.submodule_search_locations)[0], "data", "silero_vad.jit"))
        vad.eval()
    speech_thr = 0.8

    wp = wp_base
    j = -1
    toks = []
    probs = []
    boost_candidates = 0
    snap = {}
    t0 = time.time()
    with torch.no_grad():
        for t in range(CHUNK, len(a) + 1, CHUNK):
            if t < 160 * 2 * dil * 3:
                continue
            j += 1
            pos = min(W - 1, j + 3)
            st = max(0, t - W * 160 * 2 * dil)
            window = a[st:t]
            mel = log_mel(torch.from_numpy(window)).unsqueeze(0)
            mel = F.pad(mel, (0, 3000 - mel.shape[-1]), mode="constant", value=0)
            enc = model.model.encoder(mel).last_hidden_state
            sp = vad(torch.from_numpy(window[-512:].copy()), 16000).item() if vad is not None else 0.0
            probs.append(sp)
            speech = sp > speech_thr
            if len(ids) > W:
                ids.pop(4)
            n_in = pos + 1 if trim else W
            dec_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            dec_ids = F.pad(dec_ids, (0, n_in - dec_ids.shape[1]))
            out = model(encoder_outputs=enc, decoder_input_ids=dec_ids, use_cache=False, return_dict=True)
            logits = out.logits.clone()
            if not torch.isfinite(logits[0, : pos + 1]).all():
                raise SystemExit(f"non-finite logits at step {j}")
            mx = logits.argmax(dim=-1)[0][pos].item()
            if mx != WAIT and mx in ids[-5:]:
                logits[0, pos, mx] -= rep
            logits[:, pos, WAIT] -= wp
            tok = logits.argmax(dim=-1)[0][pos].item()
            ids.append(tok)
            if tok == WAIT:
                if len(ids) > 10 and all(i == WAIT for i in ids[-10:]):
                    boost_candidates += 1
                    if speech:
                        wp += wp_boost
            else:
                wp -= wp_decay * (wp - wp_base)
            toks.append(tok)
            snap = {"mel": mel[0].numpy(), "enc": enc[0].numpy(),
                    "ids": dec_ids[0, : pos + 1].numpy().astype(np.int32),
                    "logits": out.logits[0, : pos + 1].numpy()}
            if j % 10 == 0:
                print(f"    step {j:4d} t={t / 16000:6.2f}s tok={tok:5d} ({time.time() - t0:.0f}s)", flush=True)

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(model_dir))
    text = tk.decode([x for x in toks if x != WAIT and x < 50257])
    print(f"  steps={len(toks)} emitted={sum(1 for x in toks if x != WAIT)} boost_candidates={boost_candidates}")
    print(f"  text: {text}")

    caps: Dict[str, np.ndarray] = {}
    if "mel_spectrogram" in stages:
        caps["mel_spectrogram"] = snap["mel"]
    if "encoder_output" in stages:
        caps["encoder_output"] = snap["enc"]
    if "decoder_logits" in stages:
        caps["decoder_logits"] = snap["logits"]
        caps["decoder_input_ids"] = snap["ids"]
    if "stream_tokens" in stages:
        caps["stream_tokens"] = np.asarray(toks, dtype=np.int32)
        caps["stream_speech_probs"] = np.asarray(probs, dtype=np.float32)
    caps["generated_text"] = text
    caps["hikari_settings"] = f"task={task};tgt={tgt};ctx={W};wp_base={wp_base};wp_boost={wp_boost};" \
                              f"wp_decay={wp_decay};rep={rep};tail_ms={tail_ms};boost_candidates={boost_candidates};" \
                              f"vad={1 if vad is not None else 0}"
    return caps
