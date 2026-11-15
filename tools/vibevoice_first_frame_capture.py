#!/usr/bin/env python3
"""Run explicit prefill-only or first-frame diagnostic capture; no audio decode.

Use a binary built from this diagnostic worktree. Nonzero CLI exit is expected;
only a validated completion marker makes the capture successful. No quality test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess

EXPECTED = 'ff68a234c5a1069e379e4d3fdbf29bb07ec4981d0ad8841bb19bc442991413f3'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_prefill_blocks(dump, prompt_tokens, hidden):
    import struct
    marker = dump / 'tts_prefill_blocks_complete.bin'
    if not marker.is_file() or marker.stat().st_size != 12:
        raise RuntimeError('missing/truncated prefill block marker')
    layers, tokens, width = struct.unpack('<iii', marker.read_bytes())
    row_bytes = 39 * hidden * 4
    if tokens != 39 or prompt_tokens != 39 or width != hidden or not 0 < layers <= 128 or row_bytes * layers > 64 << 20:
        raise RuntimeError('invalid prefill block dimensions or payload limit')
    required = {f'tts_prefill_block_{index:03}.bin' for index in range(layers)}
    actual = {path.name for path in dump.glob('tts_prefill_block_*.bin')}
    if actual != required:
        raise RuntimeError('missing/extra prefill block files')
    if any((dump / name).stat().st_size != row_bytes for name in required):
        raise RuntimeError('truncated prefill block payload')
    return layers


def validate_prefill_only(dump):
    import struct
    marker = dump / 'prefill_capture_complete.bin'
    if not marker.is_file() or marker.stat().st_size != 16:
        raise RuntimeError('missing/truncated prefill-only completion marker')
    layers, tokens, hidden, vocab = struct.unpack('<iiii', marker.read_bytes())
    if tokens != 39 or not 0 < hidden <= 8192 or not 0 < vocab <= 262144:
        raise RuntimeError('invalid prefill-only dimensions')
    expected = {'tts_prompt_ids':39*4, 'tts_prompt_embeds':39*hidden*4,
                'tts_prefill_hidden':hidden*4, 'tts_prefill_logits':vocab*4,
                'tts_prefill_dimensions':8}
    for name, size in expected.items():
        path = dump / (name + '.bin')
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError('missing/truncated prefill-only stage: ' + name)
    if struct.unpack('<ii', (dump/'tts_prefill_dimensions.bin').read_bytes()) != (39,vocab):
        raise RuntimeError('inconsistent prefill-only dimensions')
    if validate_prefill_blocks(dump,tokens,hidden) != layers:
        raise RuntimeError('inconsistent prefill-only block count')
    allowed = {name+'.bin' for name in expected} | {'prefill_capture_complete.bin','tts_prefill_blocks_complete.bin'}
    allowed |= {f'tts_prefill_block_{index:03}.bin' for index in range(layers)}
    files = list(dump.glob('*.bin'))
    if {path.name for path in files} != allowed:
        raise RuntimeError('unexpected prefill-only payload; diffusion/frame data forbidden')
    if sum(path.stat().st_size for path in files) > 64 << 20:
        raise RuntimeError('prefill-only payload exceeds64MiB')
    return layers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--text-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--prefill-blocks', action='store_true', help='capture every raw39-token prefill block')
    p.add_argument('--prefill-only', action='store_true', help='stop after39-token prefill; implies --prefill-blocks')
    args = p.parse_args()
    if sha(args.gguf) != EXPECTED:
        p.error('expected original 7B Q4_K weight SHA-256')
    text = args.text_file.read_text().strip()
    if not text or len(text.encode()) > 4096 or not 0 < args.timeout <= 1800:
        p.error('require 1..4096 text bytes and timeout 1..1800 seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    dump = args.output / 'native'
    dump.mkdir()
    env = {k: v for k, v in os.environ.items() if not (k.startswith('CRISPASR_VIBEVOICE') and ('DUMP' in k or 'CAPTURE' in k))}
    env.update(CRISPASR_VIBEVOICE_TTS_CAPTURE_FIRST_FRAME=str(dump.resolve()),
               CRISPASR_VIBEVOICE_TTS_SEMANTIC_FEEDBACK='1',
               CRISPASR_VIBEVOICE_TTS_REFRESH_NEGATIVE='1',
               CRISPASR_VIBEVOICE_SEMANTIC_F16='1',
               CRISPASR_VIBEVOICE_TTS_TOKEN_CONSTRAINT='0')
    if args.prefill_blocks or args.prefill_only:
        env['CRISPASR_VIBEVOICE_TTS_CAPTURE_PREFILL_BLOCKS'] = '1'
    if args.prefill_only:
        env['CRISPASR_VIBEVOICE_TTS_CAPTURE_PREFILL_ONLY'] = '1'
    command = [str(args.binary.resolve()), '-m', str(args.gguf.resolve()), '--tts', text,
               '--backend', 'vibevoice-1.5b', '--seed', '42', '--tts-steps', '20', '--tts-cfg-scale', '1.3',
               '--tts-output', str(args.output.resolve() / 'unused.wav')]
    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    with (args.output / 'native.log').open('wb') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=args.timeout, preexec_fn=limits, check=False)
    if args.prefill_only:
        layers = validate_prefill_only(dump)
        root = Path(__file__).resolve().parents[1]
        git = lambda *a: subprocess.check_output(['git', '-C', str(root), *a])
        files = sorted(dump.glob('*.bin'))
        receipt = dict(scope='prefill-only diagnostic; no diffusion/audio or quality claim',
                       source_revision=git('rev-parse','HEAD').decode().strip(),
                       base_source='a8151a2e8d7914571139dff1fd0a03620eaca051',
                       instrumentation_diff_sha256=hashlib.sha256(git('diff','a8151a2e8d7914571139dff1fd0a03620eaca051','--','src/vibevoice.cpp')).hexdigest(),
                       binary_sha256=sha(args.binary),weights_sha256=EXPECTED,seed=42,prefill_block_layers=layers,
                       text_sha256=hashlib.sha256(text.encode()).hexdigest(),diffusion_executed=False,
                       cli_returncode=result.returncode,files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files})
        (args.output/'capture.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('Complete prefill-only capture:',args.output)
        return
    marker = dump / 'capture_complete.bin'
    if not marker.exists() or marker.stat().st_size != 12:
        raise RuntimeError('no complete first-frame marker; inspect native.log')
    import struct
    vae, hidden, steps = struct.unpack('<iii', marker.read_bytes())
    if not (0 < vae <= 1024 and 0 < hidden <= 8192 and steps == 20):
        raise RuntimeError('invalid dimensions/completion')
    def i32(name, count):
        path = dump / (name + '.bin')
        if not path.is_file() or path.stat().st_size != count * 4:
            raise RuntimeError('missing/truncated capture stage: ' + name)
        return struct.unpack('<' + 'i' * count, path.read_bytes())
    prompt_tokens, vocab = i32('tts_prefill_dimensions', 2)
    if not (0 < prompt_tokens <= 2048 and 0 < vocab <= 262144):
        raise RuntimeError('invalid prefill dimensions')
    step, positive_position, negative_position = i32('tts_frame0_positions', 3)
    if not (0 <= step < 64 and positive_position == prompt_tokens + step
            and 0 <= negative_position <= step + 1):
        raise RuntimeError('invalid first-frame positions')
    expected = {'tts_prompt_ids': prompt_tokens, 'tts_prompt_embeds': prompt_tokens * hidden,
                'tts_prefill_logits': vocab, 'tts_frame0_selected_tokens': step + 1,
                'tts_prefill_hidden': hidden, 'tts_pos_condition_frame0': hidden,
                'tts_neg_condition_frame0': hidden, 'tts_negative_input_frame0': hidden,
                'tts_noise_frame0': vae, 'tts_latent_frame0': vae,
                'tts_frame0_positions': 3, 'tts_frame0_timesteps': steps}
    for i in range(steps):
        for label, count in [('v_both', 2 * vae), ('v_cfg', vae), ('x0', vae), ('z', vae)]:
            expected[f'tts_frame0_{label}_s{i:03}'] = count
    for name, count in expected.items():
        path = dump / (name + '.bin')
        if not path.is_file() or path.stat().st_size != count * 4:
            raise RuntimeError('missing/truncated capture stage: ' + name)
    block_layers = validate_prefill_blocks(dump, prompt_tokens, hidden) if args.prefill_blocks else 0
    files = sorted(dump.glob('*.bin'))
    if sum(f.stat().st_size for f in files) > 64 << 20 or len(files) > 412:
        raise RuntimeError('capture exceeds bounded artifact contract')
    root = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.check_output(['git', '-C', str(root), *a])
    receipt = dict(scope='diagnostic stop after first latent; no audio or prefill parity claim',
                   source_revision=git('rev-parse', 'HEAD').decode().strip(),
                   base_source='a8151a2e8d7914571139dff1fd0a03620eaca051',
                   instrumentation_diff_sha256=hashlib.sha256(git('diff', 'a8151a2e8d7914571139dff1fd0a03620eaca051', '--', 'src/vibevoice.cpp')).hexdigest(),
                   binary_sha256=sha(args.binary), weights_sha256=EXPECTED, seed=42,
                   text_sha256=hashlib.sha256(text.encode()).hexdigest(), cfg=1.3, steps=20,
                   prefill_block_layers=block_layers, semantic_feedback=1, negative_history=1, semantic_f16=1, token_constraint=0,
                   cli_returncode=result.returncode, files={f.name: dict(bytes=f.stat().st_size, sha256=sha(f)) for f in files})
    (args.output / 'capture.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Complete first-frame capture:', args.output)


if __name__ == '__main__':
    main()
