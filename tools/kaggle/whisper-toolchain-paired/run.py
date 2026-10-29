#!/usr/bin/env python3
"""Same-input CUDA process comparison; records quality and GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tarfile
import time
import threading
import urllib.request
import zipfile

OUT = Path('/kaggle/working')
WORK = Path('/kaggle/temp/whisper-toolchain-paired')
FORK_SOURCE = '35026a0bd1983b875c0176f47a5d0145fce340f4'
UPSTREAM_SOURCE = '284029c07012d2000941a08d38a31ffb71f4734e'
SOURCE = '0a3a97c1e1c43ba3a1a56f1923ef6a74fb362f89'
REFERENCE = 'and so my fellow americans ask not what your country can do for you ask what you can do for your country'
PINS = {
    'model.bin': ('https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.en.bin', 'a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002'),
    'input.wav': ('https://raw.githubusercontent.com/CrispStrobe/CrispASR/cc114d2b4a5a9413738f390093e9131576743cb6/samples/jfk.wav', '59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e'),
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def guard():
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    state = {'available': int(mem['MemAvailable'].split()[0]) * 1024, 'load': list(os.getloadavg()), 'cpus': os.cpu_count() or 1, 'disk': shutil.disk_usage(WORK).free}
    if state['available'] < 3 * 1024**3 or state['load'][0] > 2 * state['cpus'] or state['disk'] < 6 * 1024**3:
        raise RuntimeError('Insufficient headroom')
    return state


def wer(reference, hypothesis):
    a = re.findall(r"[a-z0-9]+", reference.lower()); b = re.findall(r"[a-z0-9]+", hypothesis.lower())
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1): nxt.append(min(nxt[-1]+1, row[j]+1, row[j-1]+(x != y)))
        row = nxt
    return row[-1] / len(a)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); WORK.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    receipt = {'complete': False, 'acceptance': False, 'scope': 'T4 same-toolchain comparison: same model/audio, compiler, CUDA toolkit and target SM; all three builds disable CUDA VMM; source trees, ggml revisions and other feature defaults may differ. Profiling off/on use the same binary. Process time includes startup and GPU monitoring.', 'runs': [], 'pins': {k: v[1] for k, v in PINS.items()}}
    def save(): (OUT/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    save(); guard()
    hardware = subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap','--format=csv,noheader'], text=True).strip()
    receipt['hardware'] = hardware; save()
    caps = [float(line.split(',')[-1].strip()) for line in hardware.splitlines()]
    if not caps or caps[0] < 7.5: raise RuntimeError('Pinned fork binary requires compute capability >= 7.5')
    for name, (url, expected) in PINS.items():
        p = WORK/name
        with urllib.request.urlopen(url, timeout=180) as source, p.open('wb') as dest: shutil.copyfileobj(source, dest, 1024*1024)
        if sha(p) != expected: raise ValueError('Input hash mismatch: '+name)
    crisp = WORK/'crisp'; fork = WORK/'fork'
    subprocess.run(['git', 'clone', '--branch', 'validate/whisper-loader-device-cache-20261029', '--single-branch',
                    'https://github.com/CrispStrobe/CrispASR.git', str(crisp)], check=True, timeout=300)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=crisp, text=True).strip() != SOURCE:
        raise RuntimeError('Candidate source pin mismatch')
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=crisp, check=True, timeout=300)
    guard()
    build = WORK/'build'
    subprocess.run(['cmake', '-S', str(crisp), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                    '-DGGML_CUDA=ON', '-DGGML_CUDA_NO_VMM=ON', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DGGML_NATIVE=OFF', '-DCMAKE_CUDA_ARCHITECTURES=75',
                    '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF'], check=True, timeout=300)
    subprocess.run(['cmake', '--build', str(build), '--target', 'crispasr-cli', '-j', '2'], check=True, timeout=5400)
    # Build the pinned fork with the same compiler, CUDA toolkit and GPU
    # architecture as the candidate; ggml revisions still differ by project.
    subprocess.run(['git', 'init', str(fork)], check=True, timeout=60)
    subprocess.run(['git', '-C', str(fork), 'remote', 'add', 'origin',
                    'https://github.com/sjoerdteunisse/whisper.cpp.git'], check=True, timeout=60)
    subprocess.run(['git', '-C', str(fork), 'fetch', '--depth=1', 'origin', FORK_SOURCE], check=True, timeout=300)
    subprocess.run(['git', '-C', str(fork), 'checkout', 'FETCH_HEAD'], check=True, timeout=60)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=fork, text=True).strip() != FORK_SOURCE:
        raise RuntimeError('Fork source pin mismatch')
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=fork, check=True, timeout=300)
    guard()
    fork_build = WORK/'fork-build'
    subprocess.run(['cmake', '-S', str(fork), '-B', str(fork_build), '-DCMAKE_BUILD_TYPE=Release',
                    '-DGGML_CUDA=ON', '-DGGML_CUDA_NO_VMM=ON', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DGGML_NATIVE=OFF', '-DCMAKE_CUDA_ARCHITECTURES=75',
                    '-DWHISPER_BUILD_TESTS=OFF', '-DWHISPER_BUILD_SERVER=OFF'], check=True, timeout=300)
    guard()
    subprocess.run(['cmake', '--build', str(fork_build), '--target', 'whisper-cli', '-j', '2'], check=True, timeout=5400)
    # Also build upstream whisper.cpp at the audited current master pin; the
    # earlier packaged comparison used an independent fork, not upstream.
    upstream, upstream_build = WORK/'upstream', WORK/'upstream-build'
    subprocess.run(['git', 'init', str(upstream)], check=True, timeout=60)
    subprocess.run(['git', '-C', str(upstream), 'remote', 'add', 'origin',
                    'https://github.com/ggml-org/whisper.cpp.git'], check=True, timeout=60)
    subprocess.run(['git', '-C', str(upstream), 'fetch', '--depth=1', 'origin', UPSTREAM_SOURCE], check=True, timeout=300)
    subprocess.run(['git', '-C', str(upstream), 'checkout', 'FETCH_HEAD'], check=True, timeout=60)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip() != UPSTREAM_SOURCE:
        raise RuntimeError('Upstream source pin mismatch')
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=upstream, check=True, timeout=300)
    guard()
    subprocess.run(['cmake', '-S', str(upstream), '-B', str(upstream_build), '-DCMAKE_BUILD_TYPE=Release',
                    '-DGGML_CUDA=ON', '-DGGML_CUDA_NO_VMM=ON', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DGGML_NATIVE=OFF', '-DCMAKE_CUDA_ARCHITECTURES=75',
                    '-DWHISPER_BUILD_TESTS=OFF', '-DWHISPER_BUILD_SERVER=OFF'], check=True, timeout=300)
    guard()
    subprocess.run(['cmake', '--build', str(upstream_build), '--target', 'whisper-cli', '-j', '2'], check=True, timeout=5400)
    for tree in (build, fork_build, upstream_build):
        commands = json.loads((tree/'compile_commands.json').read_text())
        cuda = [row['command'] for row in commands if row['file'].endswith('ggml-cuda.cu')]
        if len(cuda) != 1 or 'GGML_CUDA_NO_VMM' not in cuda[0]:
            raise RuntimeError('CUDA VMM-off compile proof missing: ' + tree.name)
    receipt['cuda_vmm'] = 'disabled for all three builds; compile commands verified'
    save()
    c = [p for p in build.rglob('crispasr') if p.is_file()]
    f = [p for p in fork_build.rglob('whisper-cli') if p.is_file()]
    u = [p for p in upstream_build.rglob('whisper-cli') if p.is_file()]
    if len(c)!=1 or len(f)!=1 or len(u)!=1: raise RuntimeError('Expected one executable per source tree')
    for binary in [c[0], f[0], u[0]]: binary.chmod(binary.stat().st_mode | 0o111)
    receipt['binaries'] = {'crisp': sha(c[0]), 'fork': sha(f[0]), 'upstream': sha(u[0])}; save()
    for phase, rounds in [('warmup',1),('measured',4)]:
        for i in range(rounds):
            for name, binary in ([('crisp',c[0]),('crisp-profile',c[0]),('fork',f[0]),('upstream',u[0])] if i%2==0 else [('upstream',u[0]),('fork',f[0]),('crisp-profile',c[0]),('crisp',c[0])]):
                state = guard(); prefix = WORK/f'{phase}-{i}-{name}'
                cmd = [str(binary),'-m',str(WORK/'model.bin'),'-f',str(WORK/'input.wav'),'-l','en','-t','2','-bs','1','-otxt','-of',str(prefix)]
                env = dict(os.environ, LD_LIBRARY_PATH=str(binary.parent)+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''))
                if name == 'crisp-profile': env['CRISPASR_WHISPER_TIMING_PROFILE'] = '1'; env['CRISPASR_WHISPER_PROFILE_LOAD'] = '1'
                start = time.monotonic(); log = OUT/(prefix.name+'.log')
                with log.open('wb') as stream:
                    proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
                    observed = [False]
                    stop = threading.Event()
                    def sample_gpu():
                        while not stop.is_set():
                            result = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
                            observed[0] |= str(proc.pid) in result.stdout.split()
                            stop.wait(.1)
                    monitor = threading.Thread(target=sample_gpu, daemon=True); monitor.start()
                    try:
                        proc.wait(timeout=180)
                        elapsed = time.monotonic()-start
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait(); raise TimeoutError(name)
                    finally:
                        stop.set(); monitor.join()
                    rc = proc.returncode
                transcript = prefix.with_suffix('.txt')
                row = {'phase':phase,'round':i,'engine':name,'seconds':elapsed,'rc':rc,'cuda_process_observed':observed[0],'resources':state}
                if transcript.exists(): row['wer'] = wer(REFERENCE, transcript.read_text())
                contents = log.read_text(errors='replace')
                if 'CUDA0 total size' not in contents or 'using CUDA0 backend' not in contents:
                    raise RuntimeError('CUDA model placement or execution backend missing: '+prefix.name)
                timing = re.search(r'decode time\s*=\s*([0-9.]+) ms\s*/\s*([0-9]+) runs', contents)
                loading = re.search(r'load time\s*=\s*([0-9.]+) ms', contents)
                if not timing or not loading or int(timing.group(2)) <= 0:
                    raise RuntimeError('Missing load or per-decode timing: '+prefix.name)
                row['load_ms'] = float(loading.group(1))
                row['decode_ms'] = float(timing.group(1))
                row['decode_calls'] = int(timing.group(2))
                row['decode_ms_per_call'] = row['decode_ms'] / row['decode_calls']
                if name == 'crisp-profile':
                    loads = re.findall(r'whisper timing profile: load total_us=([0-9]+)', contents)
                    decodes = re.findall(r'whisper timing profile: decode count=([0-9]+)', contents)
                    if len(loads) != 1 or len(decodes) != 1 or int(decodes[0]) <= 0:
                        raise RuntimeError('Missing loader or positive decoder profile')
                    row['profile'] = [x for x in contents.splitlines() if 'whisper timing profile:' in x]
                if transcript.exists():
                    row['transcript_sha256'] = sha(transcript)
                    row['transcript_text_stripped'] = transcript.read_text().strip()
                receipt['runs'].append(row); save()
                if rc or not observed[0] or row.get('wer',1)>0.2: raise RuntimeError('GPU, output or transcript quality gate failed: '+prefix.name)
    import statistics
    engines = ['crisp','crisp-profile','fork','upstream']
    receipt['median_seconds'] = {name: statistics.median(r['seconds'] for r in receipt['runs'] if r['phase']=='measured' and r['engine']==name) for name in engines}
    receipt['median_load_ms'] = {name: statistics.median(r['load_ms'] for r in receipt['runs'] if r['phase']=='measured' and r['engine']==name) for name in engines}
    receipt['median_decode_ms_per_call'] = {name: statistics.median(r['decode_ms_per_call'] for r in receipt['runs'] if r['phase']=='measured' and r['engine']==name) for name in engines}
    crisp_fork_hashes = {r['transcript_sha256'] for r in receipt['runs'] if r['engine'] in ('crisp', 'crisp-profile', 'fork')}
    if len(crisp_fork_hashes) != 1: raise RuntimeError('Transcript SHA-256 differs across crisp, crisp-profile, or fork')
    if len({r['transcript_text_stripped'] for r in receipt['runs']}) != 1:
        raise RuntimeError('Stripped transcript text differs across engines')
    if len({r['decode_calls'] for r in receipt['runs']}) != 1:
        raise RuntimeError('Decoder call counts differ across engines')
    receipt['source'] = SOURCE
    receipt['fork_source'] = FORK_SOURCE
    receipt['upstream_source'] = UPSTREAM_SOURCE
    receipt['complete'] = True
    receipt['acceptance'] = bool(
        receipt['median_seconds']['crisp'] <= receipt['median_seconds']['fork'] * 1.05
        and receipt['median_seconds']['crisp'] <= receipt['median_seconds']['upstream']
    )
    save()


if __name__ == '__main__': main()
