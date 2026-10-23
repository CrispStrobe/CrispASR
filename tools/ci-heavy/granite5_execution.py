#!/usr/bin/env python3
"""Shared Granite5 CPU/CUDA execution gates; no build, upload or model download."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from granite5_result_audit import metrics, token_comparison, words

ROOT = Path(__file__).resolve().parents[2]


def gpu_matmul_nodes(text):
    # Match the NODE assignment before use=, not a later GPU source tensor
    # belonging to a CPU-assigned operation on the same debug line.
    return len(re.findall(
        r'node\s+#\s*\d+\s+\(\s*MUL_MAT(?:_ID)?\s*\):[^\n]*'
        r'\[\s*CUDA0\b[^\]]*\]\s+use=', text))


def gpu_decoder_matmul_nodes(text):
    # The LM head runs in both dynamic Session/CLI and bucketed decoding.
    # A bucket-only profile message cannot prove which device the public
    # Session path used. Require actual CUDA-assigned language-head nodes.
    return sum(bool(gpu_matmul_nodes(line)) and
               bool(re.search(r'\boutput\.weight\b', line.split('use=', 1)[-1]))
               for line in text.splitlines())


def resources(device):
    meminfo = Path('/proc/meminfo').read_text()
    available = next(int(line.split()[1]) for line in meminfo.splitlines() if line.startswith('MemAvailable:'))
    result = dict(load_average=os.getloadavg(), cpus=os.cpu_count(), memory_available_kib=available)
    if device == 'cuda':
        result['gpu0'] = subprocess.check_output([
            'nvidia-smi', '--id=0', '--query-gpu=uuid,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
    return result


def run(command, label, out, environment, device, timeout=3600):
    """Fresh process, retained resource evidence, and GPU0 process residency."""
    out.mkdir(parents=True, exist_ok=True)
    argv = [str(v) for v in command]
    record = dict(command=[Path(v).name if v.startswith('/') else v for v in argv],
                  resources=resources(device), peak_gpu_mib=0)
    report = out / (label + '-execution.json')
    report.write_text(json.dumps(record, indent=2) + '\n')
    if (record['resources']['memory_available_kib'] < 5 * 1024 * 1024 or
            record['resources']['load_average'][0] > 2 * record['resources']['cpus']):
        raise RuntimeError(f'Insufficient execution headroom before {label}')
    if device == 'cuda':
        free = int(record['resources']['gpu0'].split(',')[1].strip())
        if free < 7000:
            raise RuntimeError(f'Insufficient GPU0 memory before {label}')
    print(f"{label}: starting {device}, available RAM "
          f"{record['resources']['memory_available_kib'] // 1024} MiB, "
          f"load {record['resources']['load_average'][0]:.2f}", flush=True)
    log_path = out / (label + '.log')
    deadline = time.monotonic() + timeout
    child = None
    try:
        with log_path.open('w') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=environment)
            while child.poll() is None:
                if device == 'cuda':
                    rows = subprocess.check_output([
                        'nvidia-smi', '--id=0', '--query-compute-apps=pid,used_gpu_memory',
                        '--format=csv,noheader,nounits'], text=True)
                    for row in rows.splitlines():
                        fields = [v.strip() for v in row.split(',')]
                        if len(fields) == 2 and fields[0] == str(child.pid) and fields[1].isdigit():
                            record['peak_gpu_mib'] = max(record['peak_gpu_mib'], int(fields[1]))
                if time.monotonic() > deadline:
                    raise TimeoutError(label)
                time.sleep(.25)
        record['returncode'] = child.returncode
        print(f"{label}: exit {child.returncode}, peak GPU0 {record['peak_gpu_mib']} MiB", flush=True)
        if child.returncode:
            raise RuntimeError(f'{label} failed; see retained log')
        text = log_path.read_text(errors='replace')
        if device == 'cuda':
            record['gpu_matmul_nodes'] = gpu_matmul_nodes(text)
            record['gpu_decoder_matmul_nodes'] = gpu_decoder_matmul_nodes(text)
            if (record['peak_gpu_mib'] < 512 or record['gpu_matmul_nodes'] == 0 or
                    record['gpu_decoder_matmul_nodes'] == 0):
                raise RuntimeError(f'{label} lacks actual GPU0 residency/matmul/backend evidence')
        return record
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        # Receipts are portable; exact native command/output is kept in logs.
        record['command'] = [Path(v).name if v.startswith('/') else v for v in record['command']]
        report.write_text(json.dumps(record, indent=2) + '\n')


def session_child(args):
    sys.path.insert(0, str(args.python_root))
    from crispasr import Session
    pcm = np.load(args.pcm, allow_pickle=False)
    with Session(str(args.model), lib_path=str(args.library), use_gpu=args.device == 'cuda',
                 n_threads=4, verbosity=2) as session:
        if session.backend != 'granite':
            raise RuntimeError(f'Architecture autodetection selected {session.backend}')
        session.set_ask(args.instruction)
        outputs = dict(official=[' '.join(s.text for s in session.transcribe(pcm, language='en')) for _ in range(2)])
        session.set_ask('')
        outputs['legacy_default'] = ' '.join(s.text for s in session.transcribe(pcm))
    args.output.write_text(json.dumps(outputs, indent=2) + '\n')


def validate_case(model, reference, instruction, bundle, python_root, wav, out, device, environment, quant):
    """All stages, teacher/pipeline logits, prompt IDs, two decodes, Session and CLI."""
    from gguf import GGUFReader
    out.mkdir(parents=True, exist_ok=True)
    if device == 'cuda':
        environment = dict(environment, CRISPASR_GRANITE_DEC_PROFILE='1', GGML_SCHED_DEBUG='2')
    receipt = dict(complete=False, accepted=False, device=device, stages={}, executions={})
    def save():
        (out / 'acceptance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    ref = GGUFReader(reference)
    arrays = {t.name: t.data.reshape(-1).copy() for t in ref.tensors}
    expected_text = ref.fields['crispasr.ref.generated_text'].contents()
    prompt = arrays['prompt_ids']
    if not np.array_equal(arrays['llm_argmax'][:len(prompt)], prompt):
        raise ValueError('Original generation/prompt mismatch')
    expected_ids = arrays['llm_argmax'][len(prompt):]
    if not len(expected_ids) or not words(expected_text):
        raise ValueError('Empty original reference output')
    receipt['executions']['stages'] = run([
        bundle / 'granite5-stage-probe', model, reference, out, instruction, device],
        'stages', out, environment, device)
    save()
    for name in ('mel_spectrogram', 'encoder_out', 'projector_out', 'llm_inputs_embeds', 'llm_prefill_logits',
                 'pipeline_prefill_logits'):
        key = 'llm_prefill_logits' if name == 'pipeline_prefill_logits' else name
        values = np.fromfile(out / (name + '.bin'), dtype='<f4')
        result = metrics(arrays[key], values)
        if name == 'llm_inputs_embeds' and quant == 'f16':
            result['pass'] = result['pass'] and result['cosine'] >= .99999
        receipt['stages'][name] = result
        save()
        if not result['pass']:
            raise ValueError(f'{quant} {name} fails fixed numerical gates')
    actual_prompt = np.fromfile(out / 'prompt_ids.bin', dtype='<i4')
    receipt['prompt_ids_exact'] = bool(np.array_equal(prompt, actual_prompt))
    bucket = np.fromfile(out / 'generated_ids.bin', dtype='<i4')
    dynamic = np.fromfile(out / 'dynamic_generated_ids.bin', dtype='<i4')
    receipt['bucketed'] = token_comparison(expected_ids, bucket)
    receipt['dynamic'] = token_comparison(expected_ids, dynamic)
    receipt['native_text'] = (out / 'generated_text.txt').read_text()
    save()
    if not (receipt['prompt_ids_exact'] and receipt['bucketed']['exact'] and receipt['dynamic']['exact'] and
            receipt['native_text'].strip() == expected_text.strip()):
        raise ValueError(f'{quant} prompt, decoder or native text mismatch')
    pcm = out / 'original-audio.npy'
    np.save(pcm, arrays['raw_audio'])
    alias = out / 'renamed-model.gguf'
    alias.symlink_to(Path(model).resolve())
    try:
        receipt['executions']['session'] = run([
            sys.executable, Path(__file__).resolve(), '--session-child', '--model', alias,
            '--library', bundle / 'libcrispasr.so', '--python-root', python_root,
            '--pcm', pcm, '--instruction', instruction, '--output', out / 'session.json', '--device', device],
            'session', out, environment, device)
    finally:
        alias.unlink()
    sessions = json.loads((out / 'session.json').read_text())
    receipt['session'] = sessions
    save()
    if not (len(sessions['official']) == 2 and
            all(t.strip() == expected_text.strip() for t in sessions['official']) and
            words(sessions['legacy_default']) == words(expected_text)):
        raise ValueError(f'{quant} Session transcript mismatch')
    cli_base = out / 'cli-official'
    command = [bundle / 'crispasr', '-m', model, '-f', wav, '-l', 'en', '--ask', instruction,
               '--max-new-tokens', '96', '-otxt', '-of', cli_base, '-t', '4']
    if device == 'cpu':
        command.append('--no-gpu')
    receipt['executions']['cli'] = run(command, 'cli', out, environment, device)
    receipt['cli_text'] = Path(str(cli_base) + '.txt').read_text()
    save()
    if receipt['cli_text'].strip() != expected_text.strip():
        raise ValueError(f'{quant} CLI transcript mismatch')
    receipt.update(complete=True, accepted=True)
    save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-child', action='store_true', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    for name in ('model', 'library', 'python-root', 'pcm', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--instruction', required=True)
    session_child(parser.parse_args())


if __name__ == '__main__':
    main()
