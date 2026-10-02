#!/usr/bin/env python3
"""Convert the Index-Echo AuT tower and residual/projection connector to GGUF.

The Qwen3.5 text decoder is a separate standard llama.cpp GGUF. No vision
weights are needed by the speech path. Load safetensors one tensor at a time.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open


def convert(root, output, decoder_name):
    cfg = json.loads((root / 'audio_config.json').read_text())
    text = json.loads((root / 'llm/config.json').read_text())
    text = text.get('text_config', text)
    if cfg['model_type'] != 'qwen3_omni_moe_audio_encoder' or text['model_type'] != 'qwen3_5_text':
        raise ValueError('Expected Index-Echo Qwen-Omni tower and Qwen3.5 text decoder')
    with safe_open(root / 'connector.safetensors', framework='pt', device='cpu') as source:
        projection = 'proj.weight' in source.keys()
        if projection:
            if list(source.get_slice('proj.weight').get_shape()) != [text['hidden_size'], cfg['output_dim']]:
                raise ValueError('Projection connector dimensions differ from audio/decoder')
        elif cfg['output_dim'] != text['hidden_size']:
            raise ValueError('Residual connector requires equal audio/decoder dimensions')
    size = '9b' if projection else '2b'
    decoder_name = decoder_name or f'index-echo-{size}-decoder-f16.gguf'
    spec = importlib.util.spec_from_file_location('qwen_converter', Path(__file__).with_name('convert-qwen3-asr-to-gguf.py'))
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    remap = shared.build_remap('thinker')
    writer = gguf.GGUFWriter(str(output), 'index_echo', use_temp_file=True)
    writer.add_name(f'Index-Echo S2TT {size.upper()}')
    writer.add_string('general.license', 'apache-2.0')
    writer.add_string('index_echo.decoder_file', decoder_name)
    writer.add_string('index_echo.connector_type', 'projection' if projection else 'residual')
    writer.add_uint32('index_echo.embedding_length', text['hidden_size'])
    writer.add_string('crisp_audio.dialect', 'qwen_omni')
    scalars = {'sample_rate': 16000, 'n_mels': cfg['num_mel_bins'], 'n_fft': 400,
               'win_length': 400, 'hop_length': 160, 'n_layers': cfg['encoder_layers'],
               'd_model': cfg['d_model'], 'n_heads': cfg['encoder_attention_heads'],
               'head_dim': cfg['d_model'] // cfg['encoder_attention_heads'],
               'ff_dim': cfg['encoder_ffn_dim'], 'conv_channels': cfg['downsample_hidden_size'],
               'output_dim': cfg['output_dim'], 'max_source_pos': cfg['max_source_positions'],
               'n_window': cfg['n_window'], 'n_window_infer': cfg['n_window_infer'],
               # v5.6.0 forward does not pass _prepare_attention_mask to layers.
               # Its CPU SDPA/eager path attends over all compacted frames.
               'attn_window_mode': 0}
    for key, value in scalars.items():
        writer.add_uint32('crisp_audio.' + key, value)
    # The shared tower owns the frontend constants as well as learned weights.
    # Keep these at F32 in every quant cohort (neither is a .weight tensor).
    writer.add_tensor('audio.mel_filters', shared._compute_mel_filters(sr=16000, n_fft=400, n_mels=128))
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    writer.add_tensor('audio.mel_window', np.ascontiguousarray(window))
    count = 0
    for filename in ['audio_tower.safetensors', 'connector.safetensors']:
        with safe_open(root / filename, framework='pt', device='cpu') as source:
            for name in source.keys():
                target = remap('thinker.audio_tower.' + name) if filename.startswith('audio') else 'connector.' + name
                if target is None:
                    raise ValueError('Unmapped required audio tensor: ' + name)
                value = source.get_tensor(name).float().numpy()
                if value.ndim == 0:
                    value = value.reshape(1)
                dtype = np.float32 if value.ndim <= 1 or 'norm' in target or 'ln_post' in target else np.float16
                writer.add_tensor(target, np.ascontiguousarray(value, dtype=dtype))
                count += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print('Converted', count, 'tower/connector tensors to', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decoder-name', help='Matching decoder companion; inferred from connector by default')
    args = parser.parse_args()
    convert(args.model, args.output, args.decoder_name)
