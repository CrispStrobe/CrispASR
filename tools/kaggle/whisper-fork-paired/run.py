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
WORK = Path('/kaggle/temp/whisper-fork-paired')
REFERENCE = 'and so my fellow americans ask not what your country can do for you ask what you can do for your country'
PINS = {
    'crisp.tar.gz': ('https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.47/crispasr-linux-x86_64-cuda.tar.gz', '90e23cffb44f4a80d929bf88cf3abb2d526ec4c0b95d88a82d1a8bf33135722d'),
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
    receipt = {'complete': False, 'acceptance': False, 'scope': 'same input/model bytes and requested settings; process time includes loading and concurrent GPU monitoring; not per-step; differing builds and CUDA libraries', 'runs': [], 'pins': {k: v[1] for k, v in PINS.items()}}
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
    crisp = WORK/'crisp'; fork = WORK/'fork'; crisp.mkdir(); fork.mkdir()
    with tarfile.open(WORK/'crisp.tar.gz') as a: a.extractall(crisp, filter='data')
    with zipfile.ZipFile(WORK/'fork.zip') as a:
        for item in a.infolist():
            if Path(item.filename).is_absolute() or '..' in Path(item.filename).parts: raise ValueError('Unsafe archive path')
        a.extractall(fork)
    c = [p for p in crisp.rglob('crispasr') if p.is_file()]
    f = [p for p in fork.rglob('whisper-cpp-linux-x64-cuda') if p.is_file()]
    if len(c)!=1 or len(f)!=1: raise RuntimeError('Expected one executable per archive')
    for binary in [c[0], f[0]]: binary.chmod(binary.stat().st_mode | 0o111)
    receipt['binaries'] = {'crisp': sha(c[0]), 'fork': sha(f[0])}; save()
    for phase, rounds in [('warmup',1),('measured',4)]:
        for i in range(rounds):
            for name, binary in ([('crisp',c[0]),('fork',f[0])] if i%2==0 else [('fork',f[0]),('crisp',c[0])]):
                state = guard(); prefix = WORK/f'{phase}-{i}-{name}'
                cmd = [str(binary),'-m',str(WORK/'model.bin'),'-f',str(WORK/'input.wav'),'-l','en','-t','2','-bs','1','-otxt','-of',str(prefix)]
                env = dict(os.environ, LD_LIBRARY_PATH=str(binary.parent)+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''))
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
                receipt['runs'].append(row); save()
                if rc or not observed[0] or row.get('wer',1)>0.2: raise RuntimeError('GPU, output or transcript quality gate failed: '+prefix.name)
    import statistics
    receipt['median_seconds'] = {name: statistics.median(r['seconds'] for r in receipt['runs'] if r['phase']=='measured' and r['engine']==name) for name in ['crisp','fork']}
    receipt['complete'] = receipt['acceptance'] = True; save()


if __name__ == '__main__': main()
