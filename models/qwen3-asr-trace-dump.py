#!/usr/bin/env python3
"""Dump Qwen3-ASR full-transcribe trace for end-to-end diff testing.

For a given audio file, captures:
  trace_input_ids.npy        (T,) int32   - prompt token IDs (with audio_pad placeholders)
  trace_audio_pad_pos.npy    (N,) int32   - positions of audio_pad in input_ids
  trace_inputs_embeds.npy    (T, 1024) f32 - inputs_embeds AFTER splice (audio frames inserted)
  trace_first_logits.npy     (vocab,) f32  - logits at the LAST prompt position (next-token distribution)
  trace_generated_ids.npy    (G,) int32    - greedy-generated token IDs from the wrapper
  trace_generated_text.txt   - decoded text
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch


def load_wav_16k(path):
    import wave
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate(); nchan = w.getnchannels(); sampw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sampw != 2: raise SystemExit('only 16-bit PCM')
    pcm = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    if nchan > 1: pcm = pcm.reshape(-1, nchan).mean(axis=1)
    if sr != 16000: raise SystemExit(f'expected 16k, got {sr}')
    return pcm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True, type=Path)
    p.add_argument('--audio',     required=True, type=Path)
    p.add_argument('--out-dir',   required=True, type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from qwen_asr import Qwen3ASRModel
    print(f'Loading {args.model_dir} ...')
    wrapper = Qwen3ASRModel.from_pretrained(args.model_dir, dtype='float32', device_map='cpu')
    model = wrapper.model
    model.eval()
    proc = wrapper.processor
    thinker = model.thinker
    audio_tower = thinker.audio_tower
    text_model = thinker.model
    lm_head = thinker.lm_head

    # Use the processor itself (which has its own chat template) instead of the tokenizer
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio", "audio": ""}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f'chat template: {text!r}')

    # Process audio + text together via the processor
    audio = load_wav_16k(args.audio)
    inputs = proc(text=text, audio=audio, sampling_rate=16000, return_tensors='pt')
    print('processor outputs:')
    for k, v in inputs.items():
        if hasattr(v, 'shape'): print(f'  {k}: {tuple(v.shape)} {v.dtype}')

    input_ids = inputs['input_ids']
    feat = inputs['input_features']
    feat_mask = inputs.get('feature_attention_mask')
    print(f'input_ids[0]: {input_ids[0].tolist()}')
    np.save(args.out_dir / 'trace_input_ids.npy', input_ids[0].numpy().astype(np.int32))

    # Find audio_pad positions
    AUDIO_PAD = 151676
    pad_pos = (input_ids[0] == AUDIO_PAD).nonzero(as_tuple=True)[0].numpy().astype(np.int32)
    print(f'audio_pad positions: {len(pad_pos)} (first 5: {pad_pos[:5].tolist()})')
    np.save(args.out_dir / 'trace_audio_pad_pos.npy', pad_pos)

    # Run encoder to get audio embeddings (N, 1024)
    feature_lens = feat_mask.sum(-1) if feat_mask is not None else torch.tensor([feat.shape[-1]])
    # Convert audio sample lengths to mel frame lengths if needed
    if feat_mask is not None and feature_lens[0] > 10000:
        # mask is over raw samples, convert to mel frames
        feature_lens = feature_lens // 160  # hop_length
    print(f'feature_lens: {feature_lens.tolist()}')

    with torch.no_grad():
        # The encoder needs (n_mels, T) — squeeze batch
        mel_2d = feat.squeeze(0)
        enc_out = audio_tower(mel_2d, feature_lens=feature_lens)
        audio_embeds = enc_out.last_hidden_state  # (N, 1024)
    print(f'audio_embeds: {tuple(audio_embeds.shape)}')

    if audio_embeds.shape[0] != len(pad_pos):
        print(f'WARN: encoder gave {audio_embeds.shape[0]} frames but {len(pad_pos)} audio_pad positions')

    # Build inputs_embeds = embed(input_ids), then splice audio
    with torch.no_grad():
        inputs_embeds = text_model.embed_tokens(input_ids)  # (1, T, 1024)
        # Splice
        inputs_embeds_spliced = inputs_embeds.clone()
        n_to_use = min(len(pad_pos), audio_embeds.shape[0])
        inputs_embeds_spliced[0, pad_pos[:n_to_use]] = audio_embeds[:n_to_use]
    print(f'inputs_embeds_spliced: {tuple(inputs_embeds_spliced.shape)}')
    np.save(args.out_dir / 'trace_inputs_embeds.npy',
            inputs_embeds_spliced[0].detach().cpu().float().numpy())

    # Run LLM forward on the spliced embeds (no KV cache for diff testing)
    with torch.no_grad():
        out = text_model(inputs_embeds=inputs_embeds_spliced, use_cache=False)
        h = out.last_hidden_state
        logits = lm_head(h)  # (1, T, V)
    print(f'logits: {tuple(logits.shape)}')
    last_logits = logits[0, -1].detach().cpu().float().numpy()
    np.save(args.out_dir / 'trace_first_logits.npy', last_logits)
    print(f'next-token argmax (top-5): {np.argsort(-last_logits)[:5].tolist()}')

    # Run the full wrapper transcribe to capture the generated token sequence
    print('\nRunning full wrapper.transcribe ...')
    result = wrapper.transcribe(audio=str(args.audio))
    if isinstance(result, list): result = result[0]
    print(f'language: {result.language}')
    print(f'text: {result.text}')
    (args.out_dir / 'trace_generated_text.txt').write_text(
        f'{result.language}\n{result.text}\n')

    # Tokenize the generated text to get token IDs (rough — not exactly what
    # the wrapper produced internally, but useful for the C++ comparison)
    gen_ids = proc.tokenizer(result.text, return_tensors='pt', add_special_tokens=False).input_ids[0]
    np.save(args.out_dir / 'trace_generated_ids.npy', gen_ids.numpy().astype(np.int32))
    print(f'tokenized text → {len(gen_ids)} ids')

    print(f'\nDone: {args.out_dir}')


if __name__ == '__main__':
    main()
