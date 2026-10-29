#!/usr/bin/env python3
"""Hosted actual-function before/after controls for short DTW median windows."""
import json
import os
from pathlib import Path
import resource
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = 'bfcaa759754764bdcd22b43b9ba5cfb5de3cfb0e'


def guard():
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(memory['MemAvailable'].split()[0]) < 3 * 1024**2 or os.getloadavg()[0] > 2 * (os.cpu_count() or 1):
        raise RuntimeError('Insufficient hosted resources')


def extract(source):
    start = source.index('struct median_filter_user_data {')
    end = source.index('static void whisper_exp_compute_token_level_timestamps_dtw(', start)
    return source[start:end]


def main():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('Hosted-only compiler control')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = Path(os.environ['HEAVY_SCRATCH']) / 'whisper-dtw-short'
    root.mkdir(parents=True, exist_ok=True)
    out = Path(os.environ['HEAVY_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'fetch', '--depth=1', 'origin', BASE], cwd=ROOT, check=True, timeout=120)
    base = subprocess.check_output(['git', 'show', BASE + ':src/crispasr.cpp'], cwd=ROOT, text=True)
    current = (ROOT / 'src/crispasr.cpp').read_text()
    fixture = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>
#define GGML_TYPE_F32 0
#define CRISPASR_ASSERT(x) assert(x)
struct ggml_tensor { int64_t ne[4]; int type; std::vector<float> data; };
static int ggml_n_dims(const ggml_tensor*) { return 3; }
static float whisper_get_f32_nd(const ggml_tensor* t, int64_t, int64_t, int64_t k, int64_t) {
    assert(k >= 0 && k < t->ne[2]); return t->data.at(k);
}
static void whisper_set_f32_nd(ggml_tensor* t, int64_t, int64_t, int64_t k, int64_t, float v) {
    assert(k >= 0 && k < t->ne[2]); t->data.at(k) = v;
}
'''
    tail = r'''
int main(int argc, char** argv) {
    assert(argc == 3);
    const int n = std::atoi(argv[1]), width = std::atoi(argv[2]);
    ggml_tensor a{{1,1,n,1}, GGML_TYPE_F32, {}}, out{{1,1,n,1}, GGML_TYPE_F32, {}};
    const float values[] = {3, 1, 4, 2, 5};
    for (int i = 0; i < n; ++i) { a.data.push_back(values[i]); out.data.push_back(0); }
    median_filter_user_data config{width}; median_filter(&out, &a, 0, 1, &config);
    const float normal[] = {1, 3, 2, 4, 2};
    for (int i = 0; i < n; ++i) {
        const float expected = ((n == 5 && width == 3) || (n == 4 && width == 7)) ? normal[i] : values[i];
        if (out.data[i] != expected) return 5;
    }
    return 0;
}
'''
    outcomes = {}
    for label, source in [('before', base), ('after', current)]:
        guard()
        cpp, binary = root / (label + '.cpp'), root / label
        cpp.write_text(fixture + extract(source) + tail)
        subprocess.run(['c++', '-std=c++17', '-O0', str(cpp), '-o', str(binary)], check=True, timeout=120)
        for n, width in [(1, 7), (2, 7), (3, 7), (4, 7), (5, 3)]:
            guard()
            proc = subprocess.run([str(binary), str(n), str(width)], capture_output=True, timeout=30)
            expected_success = label == 'after' or n == 5
            outcomes[f'{label}-{n}-{width}'] = proc.returncode
            if (proc.returncode == 0) != expected_success:
                raise RuntimeError('Unexpected result for ' + label + f' n={n} width={width}')
    (out / 'receipt.json').write_text(json.dumps({'complete': True, 'acceptance': True,
        'base': BASE, 'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'Actual extracted median_filter with tensor access stubs; hosted CPU control, not full DTW inference',
        'results': outcomes}, indent=2) + '\n')


if __name__ == '__main__':
    main()
