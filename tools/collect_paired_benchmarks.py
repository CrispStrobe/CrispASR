#!/usr/bin/env python3
"""Sequential hosted process benchmarks; timings include startup and loading.

Adapters must independently verify output quality and actual GPU execution.
Output logs may contain private data and must be retained appropriately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from compare_benchmark_records import compare


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def guard(output):
    if os.environ.get('GITHUB_ACTIONS') != 'true' and not (Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir()):
        raise RuntimeError('Benchmark collection requires a hosted worker')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    state = {'available_bytes': int(memory['MemAvailable'].split()[0]) * 1024,
             'load': list(os.getloadavg()), 'cpus': os.cpu_count() or 1}
    if state['available_bytes'] < 3 * 1024**3 or state['load'][0] > 2 * state['cpus'] or shutil.disk_usage(output).free < 2 * 1024**3:
        raise RuntimeError('Insufficient worker headroom')
    return state


def collect(config, output, runner=subprocess.run, resource_guard=guard):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    identity = dict(config['identity'])
    if identity.get('input_sha256') != sha(Path(config['input_file'])):
        raise ValueError('Input SHA-256 does not match the actual file')
    probe = {**identity, 'samples': [{'seconds': 1}] * 3}
    compare(probe, probe)  # Reject invalid identity before running either command.
    commands = config['commands']
    if set(commands) != {'left', 'right'} or any(not isinstance(v, list) or not v or not all(isinstance(x, str) and x for x in v) for v in commands.values()):
        raise ValueError('Each implementation requires a nonempty argv array')
    warmups, rounds = config.get('warmups', 1), config.get('rounds', 4)
    if type(warmups) is not int or not 1 <= warmups <= 10 or type(rounds) is not int or not 3 <= rounds <= 100:
        raise ValueError('Require 1–10 warmups and 3–100 measured rounds')
    timeout = config.get('timeout_seconds', 600)
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= 7200:
        raise ValueError('Invalid timeout')
    records = {name: {**identity, 'samples': []} for name in commands}
    ledger = {'complete': False, 'scope': 'process wall time including startup and model load', 'events': []}
    def save():
        (output / 'ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    save()
    for phase, count in [('warmup', warmups), ('measured', rounds)]:
        for index in range(count):
            order = ('left', 'right') if index % 2 == 0 else ('right', 'left')
            for name in order:
                label = f'{phase}-{index}-{name}'
                event = {'phase': phase, 'round': index, 'implementation': name, 'resources': resource_guard(output)}
                ledger['events'].append(event)
                save()
                started = time.monotonic()
                try:
                    with (output / (label + '.stdout')).open('wb') as stdout, (output / (label + '.stderr')).open('wb') as stderr:
                        completed = runner(commands[name], stdout=stdout, stderr=stderr, timeout=timeout, check=False)
                    event['seconds'] = time.monotonic() - started
                    event['returncode'] = completed.returncode
                    event['stdout_sha256'] = sha(output / (label + '.stdout'))
                    event['stderr_sha256'] = sha(output / (label + '.stderr'))
                    if completed.returncode:
                        raise RuntimeError(f'{label} exited with status {completed.returncode}')
                    if phase == 'measured':
                        records[name]['samples'].append({'seconds': event['seconds']})
                except Exception as exc:
                    event['error_type'] = type(exc).__name__
                    save()
                    raise
                save()
    summary = compare(records['left'], records['right'])
    for name, record in records.items():
        (output / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    ledger['complete'] = True
    save()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    print(json.dumps(collect(json.loads(args.config.read_text()), args.output), indent=2))


if __name__ == '__main__':
    main()
