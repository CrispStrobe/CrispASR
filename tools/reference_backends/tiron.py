"""Tiron (Trelis/tiron) reference backend — issue #295.

Tiron is a drop-in `WhisperForConditionalGeneration` (Whisper large-v3, 128-mel,
32 enc + 32 dec layers) fine-tuned to emit inline `<|speakerN|>` speaker markers
+ 20 ms timestamps in one decode pass. The port's acceptance target is the
DECODED OUTPUT under Tiron's constrained-decoding grammar — NOT plain greedy,
which the harness says loses ~5 cpWER (see tiron/constraints.py). So this
reference runs `model.generate()` with the upstream
`TironConstraintLogitsProcessor` and dumps the grammar-constrained token ids +
the specials-preserving transcript, alongside the mel + encoder output for
structural (input-alignment) parity.

Patterned after reference_backends/cohere.py. The upstream harness supplies the
grammar; install it with `pip install git+https://github.com/TrelisResearch/tiron`
(pure torch + transformers, already on Kaggle). If it is unavailable the backend
falls back to plain greedy and warns — usable for the mel/encoder stages, but the
decoded-output stage is then NOT a faithful acceptance reference.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "encoder_output",   # final Whisper encoder hidden state
    "llm_argmax",       # grammar-constrained decoded token ids (int32)
    "generated_text",   # transcript with <|speakerN|> + <|t.tt|> preserved
]

# whisper-large-v3 vocab offsets (== Tiron's base; verified from added_tokens.json)
EOS_ID = 50257
SOT_ID = 50258
NOSPEECH_ID = 50363
NOTS_ID = 50364
TS_BEGIN_ID = 50365
TS_END_ID = 51865
SPEAKER_IDS = tuple(range(51866, 51874))  # <|speaker1|>..<|speaker8|>


def _resolve_speaker_ids(tokenizer):
    """Resolve <|speakerN|> ids from the live tokenizer; fall back to defaults."""
    ids = []
    for i in range(1, 9):
        tid = tokenizer.convert_tokens_to_ids(f"<|speaker{i}|>")
        if tid is None or tid < 0:
            break
        ids.append(int(tid))
    return tuple(ids) if ids else SPEAKER_IDS


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "transformers required for the tiron reference backend.\n"
            "Install: pip install 'transformers>=4.45'\n"
            f"(import error: {e})")

    print(f"  loading Tiron (WhisperForConditionalGeneration) from {model_dir}")
    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()

    # Drive decoding exactly as the model card / harness engine.py do: Tiron
    # supplies its own [sot, lang, transcribe] prefix via decoder_input_ids and
    # ALL of Whisper's default forcing/suppression must be OFF. Passing
    # language=/task= instead makes HF inject <|notimestamps|> (prompt_len 4),
    # which desyncs the constraint grammar (prompt_len 3) and degenerates the
    # decode into a repeat loop.
    #
    # ⚠ Newer transformers REJECT controlling generation via model.config
    # ("...not supported anymore..."), so — unlike the harness/model-card code,
    # which predates that change and sets model.config.suppress_tokens=[] — put
    # EVERYTHING on generation_config only and never touch model.config.
    gc = model.generation_config
    gc.forced_decoder_ids = None
    gc.suppress_tokens = None
    gc.begin_suppress_tokens = None
    gc.language = None
    gc.task = None
    if hasattr(gc, "no_timestamps_token_id"):
        delattr(gc, "no_timestamps_token_id")
    gc.no_speech_threshold = None

    # ---- Onset guardrail: prepend 0.75 s of silence once (harness
    # apply_onset_pad / config.PAD_START_SEC). A full-energy mid-word onset makes
    # the model defer output; the benchmark config uses this pad, so the faithful
    # reference must too (and the C++ tiron path must match). ----
    PAD_START_SEC = 0.75
    audio = np.concatenate([np.zeros(int(PAD_START_SEC * 16000), dtype=np.float32),
                            np.asarray(audio, dtype=np.float32)])

    # ---- Feature extraction (128-mel WhisperFeatureExtractor) ----
    inputs = processor(
        audio, sampling_rate=16000, return_tensors="pt")
    feats = inputs["input_features"].to(torch.float32)

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = np.asarray(audio, dtype=np.float32)
    if "mel_spectrogram" in stages:
        mel = feats[0]                                   # (n_mels, T)
        out["mel_spectrogram"] = mel.transpose(0, 1).contiguous().cpu().float().numpy()  # (T, n_mels)

    # ---- Encoder output via a forward hook ----
    enc_holder: Dict[str, torch.Tensor] = {}

    def _enc_hook(_m, _i, o):
        t = o.last_hidden_state if hasattr(o, "last_hidden_state") else (o[0] if isinstance(o, tuple) else o)
        enc_holder["h"] = t.detach().clone()

    eh = model.model.encoder.register_forward_hook(_enc_hook)

    # ---- Build the constrained-decoding grammar (the acceptance target) ----
    speaker_ids = _resolve_speaker_ids(processor.tokenizer)
    logits_processor = None
    try:
        from tiron.constraints import TironConstraintLogitsProcessor
        logits_processor = TironConstraintLogitsProcessor(
            prompt_len=3,  # [sot, lang, transcribe]
            speaker_token_ids=speaker_ids,
            ts_begin_id=TS_BEGIN_ID, ts_end_id=TS_END_ID,
            nots_token_id=NOTS_ID, nospeech_token_id=NOSPEECH_ID,
            eos_token_id=EOS_ID,
            target_mode="speaker_blocks",
            allow_initial_nospeech=True,
        )
        print("  using upstream TironConstraintLogitsProcessor (speaker_blocks)")
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: tiron harness unavailable ({type(e).__name__}: {e}); "
              "falling back to PLAIN GREEDY — decoded-output stage is NOT a "
              "faithful acceptance reference. pip install "
              "git+https://github.com/TrelisResearch/tiron")

    from transformers import LogitsProcessorList

    tok = processor.tokenizer
    prefix = [
        tok.convert_tokens_to_ids("<|startoftranscript|>"),
        tok.convert_tokens_to_ids("<|en|>"),
        tok.convert_tokens_to_ids("<|transcribe|>"),
    ]
    decoder_input_ids = torch.tensor([prefix], dtype=torch.long)

    gen_kwargs = dict(
        input_features=feats,
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=max(max_new_tokens, 444),
        do_sample=False, num_beams=1,
    )
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

    print(f"  running constrained generate() (prefix={prefix}) for the reference transcript")
    with torch.no_grad():
        gen = model.generate(**gen_kwargs)
    eh.remove()

    if "encoder_output" in stages and "h" in enc_holder:
        out["encoder_output"] = enc_holder["h"][0].cpu().float().numpy()

    gen_ids = gen[0].detach().cpu().int().numpy().astype(np.int32)
    if "llm_argmax" in stages:
        out["llm_argmax"] = gen_ids

    if "generated_text" in stages:
        # Preserve <|speakerN|> + <|t.tt|> controls in the reference text: use the
        # harness decoder if present, else the tokenizer WITHOUT stripping specials.
        text = None
        try:
            from tiron.decode import decode_with_specials
            spk_map = {tid: i + 1 for i, tid in enumerate(speaker_ids)}
            # strip the [sot, lang, transcribe] prompt prefix before decoding
            body = [int(t) for t in gen_ids.tolist()][3:]
            text = decode_with_specials(
                body,
                tokenizer_decode=lambda ids: processor.tokenizer.decode(ids, skip_special_tokens=True),
                eos_token_id=EOS_ID, nospeech_token_id=NOSPEECH_ID,
                ts_begin_id=TS_BEGIN_ID, ts_end_id=TS_END_ID,
                speaker_id_to_idx=spk_map,
            )
        except Exception:
            text = processor.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False)
        out["generated_text"] = text
        print(f"  reference transcript: {text[:200]}")

    return out
