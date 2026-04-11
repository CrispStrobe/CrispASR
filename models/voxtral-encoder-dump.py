#!/usr/bin/env python3

# LEGACY: kept to feed the examples/{qwen3,voxtral}-test-*/main.cpp
# differential-test drivers which still consume .npy files with the
# specific names this script produces. New backend ports should use
# tools/dump_reference.py + tools/reference_backends/<name>.py, which
# write a single GGUF tensor archive that crispasr-diff can load via
# core_gguf::load_weights. See ARCHITECTURE.md for the overall picture.

"""Dump Voxtral audio encoder + projector reference activations.

Captures ground-truth tensors at every architectural boundary for the
C++ port's differential testing (Stage V2).

Outputs to /tmp/voxtral-ref/encoder/:
  mel_input.npy              (1, 128, T_mel) f32
  conv1_out.npy              after conv1 + GELU
  conv2_out.npy              after conv2 + GELU
  pos_embed_out.npy          after adding positional embedding
  enc_blk00_out.npy          after encoder block 0
  enc_blk31_out.npy          after encoder block 31 (last)
  ln_post_out.npy            after final layer_norm
  proj1_out.npy              after projector linear_1 (pre-GELU)
  proj2_out.npy              after projector linear_2 (final audio embeds)
  transcription.txt          end-to-end ASR output on jfk.wav
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

    print(f'Loading {args.model_dir} ...')
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map='cpu', low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_dir)

    audio_tower = model.audio_tower
    projector   = model.multi_modal_projector

    # Inspect conv strides (Voxtral conv1 may differ from Whisper)
    print(f'  conv1: {audio_tower.conv1}')
    print(f'  conv2: {audio_tower.conv2}')

    # Load audio + compute mel via the feature extractor
    audio = load_wav_16k(args.audio)
    print(f'audio: {len(audio)} samples ({len(audio)/16000:.1f}s)')
    feat = processor.feature_extractor(audio, sampling_rate=16000,
                                        return_tensors='pt', padding=True)
    mel = feat['input_features']  # (1, 128, T_mel)
    # Voxtral encoder expects exactly 3000 mel frames (30s padded), matching
    # WhisperFeatureExtractor's n_samples=480000. Pad with zeros if shorter.
    if mel.shape[-1] < 3000:
        pad = torch.zeros(mel.shape[0], mel.shape[1], 3000 - mel.shape[-1],
                          dtype=mel.dtype, device=mel.device)
        mel = torch.cat([mel, pad], dim=-1)
    print(f'mel: {tuple(mel.shape)} {mel.dtype}')
    np.save(args.out_dir / 'mel_input.npy', mel.detach().cpu().float().numpy())

    # Hooks
    captures = {}
    def cap(name):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captures[name] = t.detach().cpu().float().numpy()
        return hook

    handles = []
    handles.append(audio_tower.conv1.register_forward_hook(cap('conv1_out')))
    handles.append(audio_tower.conv2.register_forward_hook(cap('conv2_out')))
    handles.append(audio_tower.layers[0].register_forward_hook(cap('enc_blk00_out')))
    handles.append(audio_tower.layers[-1].register_forward_hook(cap('enc_blk31_out')))
    if hasattr(audio_tower, 'layer_norm'):
        handles.append(audio_tower.layer_norm.register_forward_hook(cap('ln_post_out')))
    # Projector hooks
    handles.append(projector.linear_1.register_forward_hook(cap('proj1_out')))
    handles.append(projector.linear_2.register_forward_hook(cap('proj2_out')))

    # Run the audio tower + projector
    print('Running audio tower ...')
    with torch.no_grad():
        # The audio_tower expects (B, n_mels, T_mel) and returns BaseModelOutput
        enc_out = audio_tower(mel.to(torch.bfloat16))
        enc_hidden = enc_out.last_hidden_state if hasattr(enc_out, 'last_hidden_state') else enc_out
        print(f'  encoder output: {tuple(enc_hidden.shape)} {enc_hidden.dtype}')

        # The projector expects input reshaped to (-1, intermediate_size=5120).
        # This is the "4-frame stacking": encoder output (1, 1500, 1280) gets
        # reshaped to (375, 5120) = 4 consecutive 1280-vectors per row.
        # See VoxtralForConditionalGeneration.get_audio_features().
        intermediate_size = 5120  # audio_config.intermediate_size
        stacked = enc_hidden.reshape(-1, intermediate_size)
        print(f'  stacked for projector: {tuple(stacked.shape)}')
        proj_out = projector(stacked)
        print(f'  projector output: {tuple(proj_out.shape)} {proj_out.dtype}')

    for h in handles: h.remove()

    # Save all captures
    print(f'\ncaptured: {sorted(captures.keys())}')
    for name, arr in sorted(captures.items()):
        path = args.out_dir / f'{name}.npy'
        # Squeeze batch dim if present
        if arr.ndim >= 3 and arr.shape[0] == 1:
            arr = arr[0]
        np.save(path, arr)
        print(f'  {name}: {arr.shape}  → {path.name}')

    # Save encoder + projector raw outputs too
    enc_arr = enc_hidden.detach().cpu().float().numpy()
    if enc_arr.ndim == 3: enc_arr = enc_arr[0]
    np.save(args.out_dir / 'encoder_final.npy', enc_arr)
    proj_arr = proj_out.detach().cpu().float().numpy()
    if proj_arr.ndim == 3: proj_arr = proj_arr[0]
    np.save(args.out_dir / 'projector_final.npy', proj_arr)
    print(f'  encoder_final: {enc_hidden.shape[1:]}')
    print(f'  projector_final: {proj_out.shape[1:]}')

    # End-to-end transcription
    print('\nRunning end-to-end transcription ...')
    try:
        inputs = processor.apply_transcription_request(
            language='en', audio=str(args.audio),
            model_id=str(args.model_dir))
        inputs = inputs.to('cpu', dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        text = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                                       skip_special_tokens=True)[0]
        print(f'  transcription: {text}')
        (args.out_dir / 'transcription.txt').write_text(text)
    except Exception as e:
        print(f'  transcription failed (non-fatal): {e}')

    print(f'\nDone: {args.out_dir}')


if __name__ == '__main__':
    main()
