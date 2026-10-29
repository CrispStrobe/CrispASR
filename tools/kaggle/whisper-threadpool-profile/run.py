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
WORK = Path('/kaggle/temp/whisper-threadpool-profile')
SOURCE = 'e34ca22e4efd1841f788737d8528ab0621ac2706'
REFERENCE = 'and so my fellow americans ask not what your country can do for you ask what you can do for your country'
PINS = {
    'fork.zip': ('https://github.com/sjoerdteunisse/whisper.cpp/releases/download/v1.0.0/whisper-cpp-linux-x64-cuda.zip', '286ad85d70ef9dd753b313f5d82c2ee3ce9015a1286e523f029fe215fdd7da61'),
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
    if state['available'] < 3 * 1024**3 or state['load'][0] > 2 * state['cpus'] or state['disk'] < 2 * 1024**3:
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
    receipt = {'complete': False, 'acceptance': False, 'scope': 'T4 diagnostic; same model/audio; profiling off/on use the same binary; fork build and CUDA libraries differ; phase durations are host wall times and compute includes scheduler synchronization', 'runs': [], 'pins': {k: v[1] for k, v in PINS.items()}}
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
    crisp = WORK/'crisp'; fork = WORK/'fork'; fork.mkdir()
    subprocess.run(['git', 'clone', '--branch', 'validate/whisper-threadpool-profile-20261029', '--single-branch',
                    'https://github.com/CrispStrobe/CrispASR.git', str(crisp)], check=True, timeout=300)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=crisp, text=True).strip() != SOURCE:
        raise RuntimeError('Candidate source pin mismatch')
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=crisp, check=True, timeout=300)
    guard()
    build = WORK/'build'
    subprocess.run(['cmake', '-S', str(crisp), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                    '-DGGML_CUDA=ON', '-DGGML_NATIVE=OFF', '-DCMAKE_CUDA_ARCHITECTURES=75',
                    '-DCRISPASR_BUILD_TESTS=OFF', '-DCRISPASR_BUILD_SERVER=OFF'], check=True, timeout=300)
    subprocess.run(['cmake', '--build', str(build), '--target', 'crispasr-cli', '-j', '2'], check=True, timeout=2400)
    with zipfile.ZipFile(WORK/'fork.zip') as a:
        for item in a.infolist():
            if Path(item.filename).is_absolute() or '..' in Path(item.filename).parts: raise ValueError('Unsafe archive path')
        a.extractall(fork)
    c = [p for p in build.rglob('crispasr') if p.is_file()]
    f = [p for p in fork.rglob('whisper-cpp-linux-x64-cuda') if p.is_file()]
    if len(c)!=1 or len(f)!=1: raise RuntimeError('Expected one executable per archive')
    for binary in [c[0], f[0]]: binary.chmod(binary.stat().st_mode | 0o111)
    receipt['binaries'] = {'crisp': sha(c[0]), 'fork': sha(f[0])}; save()
    for phase, rounds in [('warmup',1),('measured',4)]:
        for i in range(rounds):
            for name, binary in ([('crisp',c[0]),('crisp-profile',c[0]),('fork',f[0])] if i%2==0 else [('fork',f[0]),('crisp-profile',c[0]),('crisp',c[0])]):
                state = guard(); prefix = WORK/f'{phase}-{i}-{name}'
                cmd = [str(binary),'-m',str(WORK/'model.bin'),'-f',str(WORK/'input.wav'),'-l','en','-t','2','-bs','1','-otxt','-of',str(prefix)]
                env = dict(os.environ, LD_LIBRARY_PATH=str(binary.parent)+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''))
                env.pop('CRISPASR_WHISPER_TIMING_PROFILE', None)
                if name == 'crisp-profile': env['CRISPASR_WHISPER_TIMING_PROFILE'] = '1'
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
                    raise RuntimeError('CUDA model or execution backend missing: ' + prefix.name)
                decode = re.search(r'decode time\s*=\s*([0-9.]+) ms\s*/\s*([0-9]+) runs', contents)
                if not decode or int(decode.group(2)) != 27:
                    raise RuntimeError('Expected 27 decode calls: ' + prefix.name)
                row['decode_calls'] = int(decode.group(2))
                if name == 'crisp-profile':
                    lines = [x for x in log.read_text(errors='replace').splitlines() if 'whisper timing profile:' in x]
                    if len(lines) < 3: raise RuntimeError('Missing loader or decoder profile')
                    pools = [re.fullmatch(r'whisper timing profile: decode threadpool_us=([0-9]+)', x.strip()) for x in lines]
                    pools = [m for m in pools if m]
                    if len(pools) != 1: raise RuntimeError('Missing unique threadpool timing')
                    row['threadpool_us'] = int(pools[0].group(1))
                    phases = [re.fullmatch(r'whisper timing profile: decode count=([0-9]+) graph_us=([0-9]+) input_us=([0-9]+) compute_us=([0-9]+) read_us=([0-9]+)', x.strip()) for x in lines]
                    phases = [m for m in phases if m]
                    if len(phases) != 1 or int(phases[0].group(1)) != 27:
                        raise RuntimeError('Missing unique 27-call decoder phase profile')
                    if row['threadpool_us'] > int(phases[0].group(4)):
                        raise RuntimeError('Threadpool time exceeds containing compute time')
                    row['profile'] = lines
                elif 'whisper timing profile:' in contents:
                    raise RuntimeError('Profile emitted while disabled: ' + prefix.name)
                if transcript.exists(): row['transcript_sha256'] = sha(transcript)
                receipt['runs'].append(row); save()
                if rc or not observed[0] or row.get('wer',1)>0.2: raise RuntimeError('GPU, output or transcript quality gate failed: '+prefix.name)
    import statistics
    receipt['median_seconds'] = {name: statistics.median(r['seconds'] for r in receipt['runs'] if r['phase']=='measured' and r['engine']==name) for name in ['crisp','crisp-profile','fork']}
    hashes = {r['transcript_sha256'] for r in receipt['runs']}
    if len(hashes) != 1: raise RuntimeError('Transcript differs across profiling controls')
    receipt['source'] = SOURCE
    receipt['complete'] = True
    receipt['acceptance'] = False  # A diagnostic, not proof of matched performance.
    save()


if __name__ == '__main__': main()
