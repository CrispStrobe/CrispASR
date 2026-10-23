#!/usr/bin/env python3
"""Model-free Kaggle CUDA loader preflight; this is not model acceptance."""
import argparse
import csv
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _nvidia(name):
    return name.startswith(('libcuda.', 'libcudart.', 'libcublas', 'libnv'))


def _not_stub(path):
    path = Path(path)
    for candidate in (path, path.resolve(strict=True)):
        if any('stub' in part.lower() for part in candidate.parts):
            raise RuntimeError(f'CUDA stub path rejected: {candidate}')
    return path.resolve(strict=True)


def _host_library_path(bundle):
    candidates = [bundle]
    candidates.extend(Path(p) for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if p)
    candidates.extend([Path('/usr/local/cuda/lib64'), Path('/usr/local/cuda/targets/x86_64-linux/lib')])
    candidates.extend(sorted(Path('/usr/local').glob('cuda-12*/lib64'), reverse=True))
    candidates.extend(sorted(Path('/usr/local').glob('cuda-12*/targets/x86_64-linux/lib'), reverse=True))
    # Wheels can install CUDA libraries outside ld.so.cache. Inspect filenames
    # only; importing Torch would change the process and load its own runtime.
    for root in sorted(set(sysconfig.get_paths()[key] for key in ('purelib', 'platlib'))):
        candidates.extend(sorted((Path(root) / 'nvidia').glob('*/lib')))
    paths = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        path = _not_stub(candidate)
        if str(path) not in paths:
            paths.append(str(path))
    return ':'.join(paths)


def _gpu0(text):
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) != 1 or len(rows[0]) != 8:
        raise RuntimeError(f'Unexpected GPU0 metadata: {text!r}')
    fields = [value.strip() for value in rows[0]]
    if (fields[0] != '0' or not fields[1].startswith('GPU-') or
            not all(fields) or not re.fullmatch(r'[0-9]+\.[0-9]+', fields[3]) or
            not re.fullmatch(r'[0-9.]+', fields[4]) or
            not all(value.isdigit() for value in fields[5:]) or int(fields[5]) <= 0 or
            int(fields[6]) > int(fields[5]) or int(fields[7]) > 100):
        raise RuntimeError(f'Actual GPU0 metadata required: {fields}')
    return dict(zip(('index', 'uuid', 'name', 'compute_capability', 'driver_version',
                     'memory_total_mib', 'memory_free_mib', 'utilization_gpu_percent'), fields))


def _ldd_paths(text):
    if re.search(r'not found|undefined symbol|version .* not found', text, re.IGNORECASE):
        raise RuntimeError(f'Unresolved CUDA backend dependencies:\n{text}')
    paths = {}
    for line in text.splitlines():
        match = re.match(r'\s*(\S+)\s+=>\s+(/\S+)\s+\(', line)
        if match:
            name, value = match.groups()
            if _nvidia(name):
                paths[name] = str(_not_stub(value))
    if not paths:
        raise RuntimeError('ldd did not resolve any NVIDIA backend dependencies')
    return paths


def _load_child(backend):
    # A fresh exec is essential: the dynamic loader reads LD_LIBRARY_PATH at
    # process startup, and no parent-process library can satisfy this test.
    handle = ctypes.CDLL(str(backend), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    if not handle._handle:
        raise RuntimeError('CUDA backend returned no loader handle')
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/') and _nvidia(Path(fields[5]).name):
            if fields[5].endswith(' (deleted)'):
                raise RuntimeError(f'Loaded NVIDIA library was deleted: {fields[5]}')
            paths.add(str(_not_stub(fields[5])))
    print('GRANITE5_CUDA_LOAD=' + json.dumps(sorted(paths)), flush=True)


def run_preflight(bundle, output):
    """Return/write loader evidence; reuse receipt LD_LIBRARY_PATH for workers.

    No CUDA kernels, models, Torch imports, compiler, installer or network are
    involved. NVIDIA constructors may run when the actual backend is loaded.
    """
    bundle, output = Path(bundle).resolve(), Path(output)
    receipt = dict(schema=1, runtime_preflight_pass=False, gpu_executed=False,
                   model_acceptance=False, scope='GPU0 metadata and model-free CUDA runtime loader only')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for variable in ('LD_PRELOAD', 'LD_AUDIT'):
            if os.environ.get(variable, '').strip():
                raise RuntimeError(f'{variable} must be empty for an uninterposed loader preflight')
        meminfo = Path('/proc/meminfo').read_text()
        available = next(int(line.split()[1]) for line in meminfo.splitlines()
                         if line.startswith('MemAvailable:'))
        receipt['resources'] = dict(load_average=os.getloadavg(), cpus=os.cpu_count(),
                                   memory_available_kib=available, meminfo=meminfo)
        backend = bundle / 'libggml-cuda.so'
        if not backend.is_file() or not backend.resolve().is_relative_to(bundle):
            raise RuntimeError('Actual capsule libggml-cuda.so is required')
        forbidden = [str(p) for p in bundle.iterdir() if _nvidia(p.name) and '.so' in p.name]
        if forbidden:
            raise RuntimeError(f'NVIDIA libraries must be host-provided: {forbidden}')
        gpu_command = ['nvidia-smi', '--id=0',
                       '--query-gpu=index,uuid,name,compute_cap,driver_version,memory.total,memory.free,utilization.gpu',
                       '--format=csv,noheader,nounits']
        gpu = subprocess.run(gpu_command, capture_output=True, text=True, check=True, timeout=30)
        receipt['gpu0'] = _gpu0(gpu.stdout)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is not None and visible.split(',')[0].strip() not in {'0', receipt['gpu0']['uuid']}:
            raise RuntimeError('CUDA_VISIBLE_DEVICES does not select the queried GPU0')
        environment = dict(os.environ, LD_LIBRARY_PATH=_host_library_path(bundle))
        receipt['ld_library_path'] = environment['LD_LIBRARY_PATH']
        receipt['cuda_visible_devices'] = visible
        receipt['backend'] = dict(path=str(backend), sha256=_sha(backend))
        ldd = subprocess.run(['ldd', str(backend)], env=environment, capture_output=True,
                             text=True, check=True, timeout=30)
        receipt['ldd_stdout'], receipt['ldd_stderr'] = ldd.stdout, ldd.stderr
        resolved = _ldd_paths(ldd.stdout + '\n' + ldd.stderr)
        receipt['ldd_nvidia_paths'] = resolved
        child = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                '--load-child', str(backend)], env=environment,
                               capture_output=True, text=True, timeout=120)
        receipt['loader_child'] = dict(returncode=child.returncode, stdout=child.stdout, stderr=child.stderr)
        if child.returncode:
            raise RuntimeError('RTLD_NOW CUDA backend load failed; inspect loader_child evidence')
        records = [line.removeprefix('GRANITE5_CUDA_LOAD=') for line in child.stdout.splitlines()
                   if line.startswith('GRANITE5_CUDA_LOAD=')]
        if len(records) != 1:
            raise RuntimeError('Loader child did not return exactly one loaded-library inventory')
        loaded = json.loads(records[0])
        if not isinstance(loaded, list) or not all(isinstance(p, str) for p in loaded):
            raise RuntimeError('Invalid loaded-library inventory')
        paths = {_not_stub(p) for p in loaded}
        if any(p.is_relative_to(bundle) for p in paths):
            raise RuntimeError('Loaded NVIDIA library came from the capsule')
        for required in ('libcuda.so.', 'libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12'):
            if not any(p.name.startswith(required) for p in paths):
                raise RuntimeError(f'Required host runtime was not actually loaded: {required}')
        if not set(resolved.values()).issubset({str(p) for p in paths}):
            raise RuntimeError('Actual NVIDIA library mappings differ from ldd resolution')
        receipt['loaded_nvidia_libraries'] = [dict(path=str(p), bytes=p.stat().st_size, sha256=_sha(p))
                                               for p in sorted(paths)]
        receipt['runtime_preflight_pass'] = True
        return receipt
    except Exception as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        if isinstance(exc, subprocess.CalledProcessError):
            receipt['failed_command'] = dict(command=exc.cmd, returncode=exc.returncode,
                                             stdout=exc.stdout, stderr=exc.stderr)
        raise
    finally:
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--load-child', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.load_child:
        _load_child(args.load_child)
    elif args.bundle and args.out:
        run_preflight(args.bundle, args.out)
    else:
        parser.error('--bundle and --out are required')


if __name__ == '__main__':
    main()
