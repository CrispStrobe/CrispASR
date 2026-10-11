#!/usr/bin/env python3
"""Issue #532: actual CTC graph allocation/reuse on pinned public weights.

This is an allocation and same-build replay diagnostic, not original-model
parity or a reproduction of the unavailable German recording/GPU environment.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'canary-ctc-allocation'
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)


def run(command, name, **kwargs):
    with (OUT / (name + '.log')).open('w') as log:
        r = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, timeout=3600, **kwargs)
    print(name, r.returncode, (OUT / (name + '.log')).read_text()[-2000:], flush=True)
    return r.returncode


repo = 'cstr/canary-ctc-aligner-GGUF'
filename = 'canary-ctc-aligner-q4_k.gguf'
revision = HfApi().model_info(repo).sha
model = Path(hf_hub_download(repo, filename, revision=revision))
with wave.open(str(ROOT / 'samples/jfk.wav')) as wav:
    assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
pcm.tofile(SCRATCH / 'input.f32')
receipt = dict(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               ggml=subprocess.check_output(['git', '-C', 'ggml', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               model=dict(repo=repo, revision=revision, filename=filename,
                          bytes=model.stat().st_size, sha256=hashlib.sha256(model.read_bytes()).hexdigest()),
               sample_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
               original_parity=False, reporter_recording_reproduced=False,
               hardware='GitHub CPU, GPU backends disabled', results=[])
(OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
runner = ROOT / 'canary-allocation-probe.cpp'
runner.write_text(r'''
#include "canary_ctc.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    std::ifstream file(argv[2], std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    std::vector<float> base(bytes.size() / sizeof(float));
    std::memcpy(base.data(), bytes.data(), bytes.size());
    auto p = canary_ctc_context_default_params(); p.n_threads = 4;
    auto* reused = canary_ctc_init_from_file(argv[1], p);
    if (!reused || base.empty()) return 2;
    int seq = 0;
    for (int seconds : {11, 4, 33, 90, 11}) {
        std::vector<float> pcm(seconds * 16000);
        for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = base[i % base.size()];
        float* logits = nullptr; int frames = 0, vocab = 0;
        std::fprintf(stderr, "probe seq=%d seconds=%d reused\n", seq, seconds);
        int rc = canary_ctc_compute_logits(reused, pcm.data(), (int)pcm.size(), &logits, &frames, &vocab);
        if (rc || !logits) return 3;
        std::string path = std::string(argv[3]) + "/case-" + std::to_string(seq);
        std::ofstream out(path + ".f32", std::ios::binary);
        out.write((char*)logits, (size_t)frames * vocab * sizeof(float)); out.close();
        std::ofstream shape(path + ".json"); shape << "{\"seconds\":" << seconds << ",\"frames\":" << frames << ",\"vocab\":" << vocab << "}\n";
        // Exact fresh/reused control for the returning short shape. Long shapes
        // remain allocation probes rather than a promise of recognizer quality.
        if (seq == 0 || seq == 4) {
            auto* fresh = canary_ctc_init_from_file(argv[1], p);
            if (!fresh) return 4;
            float* ref = nullptr; int rf = 0, rv = 0;
            rc = canary_ctc_compute_logits(fresh, pcm.data(), (int)pcm.size(), &ref, &rf, &rv);
            bool equal = !rc && ref && rf == frames && rv == vocab &&
                !std::memcmp(ref, logits, (size_t)frames * vocab * sizeof(float));
            std::free(ref); canary_ctc_free(fresh);
            if (!equal) return 5;
        }
        std::free(logits); ++seq;
    }
    canary_ctc_free(reused); return 0;
}
'''.replace('#include <cstdio>', '#include <cstdio>\n#include <cstring>'))
with (ROOT / 'CMakeLists.txt').open('a') as f:
    f.write('\nadd_executable(canary-allocation-probe canary-allocation-probe.cpp)\n'
            'target_link_libraries(canary-allocation-probe PRIVATE canary_ctc)\n')
build = SCRATCH / 'build'
assert run(['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
            '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_METAL=OFF',
            '-DCRISPASR_BUILD_SERVER=OFF', '-DCRISPASR_BUILD_TESTS=OFF'], 'configure') == 0
assert run(['cmake', '--build', build, '--target', 'canary-allocation-probe', '-j4'], 'build') == 0
failed = []
for bucket in (0, 128):
    directory = OUT / f'bucket-{bucket}'
    directory.mkdir(exist_ok=True)
    env = dict(os.environ, CRISPASR_FC_BUCKET=str(bucket))
    rc = run([build / 'bin/canary-allocation-probe', model, SCRATCH / 'input.f32', directory],
             f'probe-{bucket}', env=env)
    rows = []
    for shape in sorted(directory.glob('case-*.json')):
        row = json.loads(shape.read_text())
        values = np.fromfile(shape.with_suffix('.f32'), dtype=np.float32)
        row.update(file=shape.with_suffix('.f32').name, finite=bool(np.isfinite(values).all()),
                   size=int(values.size), sha256=hashlib.sha256(values.tobytes()).hexdigest())
        assert row['finite'] and row['size'] == row['frames'] * row['vocab']
        rows.append(row)
    receipt['results'].append(dict(bucket=bucket, returncode=rc, cases=rows))
    (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if rc or len(rows) != 5:
        failed.append(bucket)
(OUT / 'summary.md').write_text(f'Canary CTC allocation probe: failed buckets {failed}.\n'
                               'CPU replay diagnostic only; no reporter GPU/original parity claim.\n')
raise SystemExit(bool(failed))
