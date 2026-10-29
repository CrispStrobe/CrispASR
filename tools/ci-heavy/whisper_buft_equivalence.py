#!/usr/bin/env python3
"""Hosted, model-free controls for the actual Whisper weight-placement change.

Requires GitHub Actions, HEAVY_SCRATCH, HEAVY_OUT, CMake, Ninja, a C++17
compiler, and the initialized pinned ggml submodule. Run from a checkout whose
src/crispasr.cpp exactly matches CANDIDATE below. Only ggml-base and extracted
placement functions are compiled; no model, inference, CUDA SDK or GPU is used.
This establishes placement/control-flow equivalence, not CUDA speed or parity.
"""

import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import traceback


ROOT = Path(__file__).resolve().parents[2]
BASELINE = '9df3071a933b2c85ceb3e278596c219c5d00c32e'
CANDIDATE = '0a3a97c1e1c43ba3a1a56f1923ef6a74fb362f89'
GGML = 'e2ae679a563014f46c546ce6c6448e8601e4dea6'

# A different operation, shape or dtype on successive calls must be rechecked.
TENSORS = [
    ('MUL_MAT', 'F16', 32, 1),
    ('MUL_MAT', 'F32', 64, 1),
    ('GET_ROWS', 'F32', 32, 1),
    ('GET_ROWS', 'F16', 64, 1),
    ('ADD', 'F32', 32, 1),
    ('MUL_MAT', 'F32', 32, 2),
    ('GET_ROWS', 'F16', 32, 1),
    ('MUL_MAT', 'F16', 32, 1),
    ('MUL_MAT', 'Q8_0', 32, 1),
]

# Buffer ids: GPU=0, IGPU=1, canonical CPU=2, CPU extras=3/4/5/6,
# ACCEL=7, META=8, second GPU=9, noncanonical CPU buffer=10.
# Expected choices are explicit acceptance oracles, independent of the selector.
SCENARIOS = [
    ('gpu-first', [0, 6, 3, 2], [0] * 9),
    ('igpu-first', [1, 0, 2], [1] * 9),
    ('cpu-default-first', [2, 6, 3], [2] * 9),
    ('cpu-extras', [6, 3, 4, 5, 2], [3, 5, 4, 2, 2, 2, 4, 3, 2]),
    ('repeated-buffer-and-device', [6, 6, 3, 4, 2], [3, 2, 4, 2, 2, 2, 4, 3, 2]),
    ('cpu-before-gpu', [6, 3, 0, 2], [3, 0, 0, 0, 0, 0, 0, 3, 0]),
    ('multiple-gpus', [9, 0, 1, 2], [9] * 9),
    ('mixed-devices', [6, 7, 8, 5, 3, 4, 2], [3, 5, 4, 8, 2, 7, 8, 3, 2]),
    ('accel-meta-fallback', [8, 7, 2], [2, 2, 2, 8, 2, 7, 8, 2, 2]),
    ('noncanonical-cpu', [10, 5, 2], [2, 5, 2, 2, 2, 2, 2, 2, 2]),
    ('no-compatible-buffer', [6, 10], [-1] * 9),
]
BUFFER_DEVICES = [0, 1, 2, 2, 2, 3, 2, 4, 5, 6, 3]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, timeout=120)


def guard():
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(memory['MemAvailable'].split()[0]) < 2 * 1024**2:
        raise RuntimeError('Insufficient hosted available memory')
    if os.getloadavg()[0] > 2 * (os.cpu_count() or 1):
        raise RuntimeError('Hosted load exceeds the build/run admission threshold')


def extract(source, candidate):
    """Keep native function bodies and cache construction byte-for-byte."""
    start = source.index('struct whisper_weight_buft {') if candidate else source.index(
        'static bool weight_buft_supported(const whisper_hparams&')
    end = source.index('// load the model from a ggml file', start)
    functions = source[start:end]
    if functions.count('static bool weight_buft_supported(') != 1:
        raise RuntimeError('ASR predicate extraction is ambiguous')
    if functions.count('static ggml_backend_buffer_type_t select_weight_buft(') != 1:
        raise RuntimeError('ASR selector extraction is ambiguous')
    if candidate:
        anchor = '    std::unordered_map<ggml_backend_dev_t, enum ggml_backend_dev_type> device_types;'
        if source.count(anchor) != 1:
            raise RuntimeError('Candidate load-local cache anchor changed')
        begin = source.index(anchor)
        cache = source[begin:source.index('    if (profile_load)', begin)]
        prepare = 'static auto prepare(const buft_list_t& buft_list) {\n' + cache + '\nreturn weight_bufts;\n}\n'
    else:
        cache = ''
        prepare = 'static buft_list_t prepare(const buft_list_t& buft_list) { return buft_list; }\n'
    return functions, cache, prepare


FIXTURE = r'''
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend-impl.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#define CRISPASR_LOG_INFO(...) std::fprintf(stderr, __VA_ARGS__)
struct whisper_hparams { int n_audio_ctx = 17; };
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;
static ggml_backend_buffer_type buffers[11];
static ggml_backend_device devices[7];
static int queries[7] = {};
static const int buffer_device[] = {0,1,2,2,2,3,2,4,5,6,3};
static const enum ggml_backend_dev_type device_types[] = {
    GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU,
    GGML_BACKEND_DEVICE_TYPE_CPU, GGML_BACKEND_DEVICE_TYPE_CPU,
    GGML_BACKEND_DEVICE_TYPE_ACCEL, GGML_BACKEND_DEVICE_TYPE_META,
    GGML_BACKEND_DEVICE_TYPE_GPU};
static const char* case_name;
static int active_tensor, active_load;
static ggml_tensor* active_weight;
namespace core_cpu_backend {
static ggml_backend_buffer_type_t buffer_type() { return &buffers[2]; }
}
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
// Deliberately identical names: identities must remain distinct pointers.
static const char* device_name(ggml_backend_dev_t) { return "same-device-name"; }
static const char* buffer_name(ggml_backend_buffer_type_t) { return "same-buffer-name"; }
static enum ggml_backend_dev_type get_type(ggml_backend_dev_t dev) {
    const auto id = dev - devices;
    require(id >= 0 && id < 7, "invalid device");
    ++queries[id]; return device_types[id];
}
static bool supports(ggml_backend_dev_t dev, const ggml_tensor* op) {
    require(op && op->src[0] == active_weight, "support probe changed weight identity");
    auto* w = op->src[0];
    require(w->buffer && ggml_backend_buffer_get_size(w->buffer) == 0,
            "support probe missing zero-sized dummy buffer");
    auto* buft = ggml_backend_buffer_get_type(w->buffer);
    const auto id = buft - buffers;
    require(id >= 0 && id < 11 && buft->device == dev, "support probe changed buffer/device");
    const auto* b = op->src[1];
    require(b && b->buffer == nullptr && b->data == nullptr, "support probe allocated input data");
    require(op->type == GGML_TYPE_F32, "support probe output dtype changed");
    if (op->op == GGML_OP_MUL_MAT) {
        require(b->type == GGML_TYPE_F32 && b->ne[0] == w->ne[0] && b->ne[1] == 17 &&
                b->ne[2] == w->ne[2] && b->ne[3] == w->ne[3], "matmul probe shape/type changed");
        require(op->ne[0] == w->ne[1] && op->ne[1] == 17 && op->ne[2] == w->ne[2] &&
                op->ne[3] == w->ne[3], "matmul output metadata changed");
    } else {
        require(op->op == GGML_OP_GET_ROWS && b->type == GGML_TYPE_I32 && b->ne[0] == 8 &&
                b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1, "get_rows probe shape/type changed");
        require(op->ne[0] == w->ne[0] && op->ne[1] == 8, "get_rows output metadata changed");
    }
    bool accepted = false;
    if (id == 3) accepted = op->op == GGML_OP_MUL_MAT && w->ne[0] == 32 && w->type == GGML_TYPE_F16;
    if (id == 4) accepted = op->op == GGML_OP_GET_ROWS && w->ne[0] == 32;
    if (id == 5) accepted = op->op == GGML_OP_MUL_MAT && w->ne[0] == 64 && w->type == GGML_TYPE_F32;
    if (id == 7) accepted = op->op == GGML_OP_MUL_MAT && w->ne[2] == 2;
    if (id == 8) accepted = op->op == GGML_OP_GET_ROWS && w->type == GGML_TYPE_F16;
    std::printf("support|%s|%d|%d|%td|%td|%d\n", case_name, active_load, active_tensor,
                dev - devices, id, int(accepted));
    return accepted;
}
'''

TAIL = r'''
struct TensorSpec { ggml_op op; ggml_type type; int width, batches; };
static const TensorSpec tensor_specs[] = {
@TENSORS@
};
struct Scenario { const char* name; std::vector<int> buffers; };
static const Scenario scenarios[] = {
@SCENARIOS@
};
int main() {
    try {
        for (int i = 0; i < 7; ++i) {
            devices[i].iface.get_name = device_name;
            devices[i].iface.get_type = get_type;
            devices[i].iface.supports_op = supports;
        }
        for (int i = 0; i < 11; ++i) {
            buffers[i].device = &devices[buffer_device[i]];
            buffers[i].iface.get_name = buffer_name;
        }
        whisper_hparams hparams;
        for (const auto& scenario : scenarios) {
            case_name = scenario.name;
            // Reuse identical backend pointers across independent load scopes.
            for (active_load = 0; active_load < 2; ++active_load) {
                for (int& n : queries) n = 0;
                buft_list_t buft_list;
                for (int id : scenario.buffers)
                    buft_list.emplace_back(buffers[id].device, &buffers[id]);
                auto prepared = prepare(buft_list);
                for (active_tensor = 0; active_tensor < 9; ++active_tensor) {
                    const auto& spec = tensor_specs[active_tensor];
                    ggml_init_params ip{ggml_tensor_overhead(), nullptr, true};
                    ggml_context_ptr ctx{ggml_init(ip)};
                    require(bool(ctx), "weight metadata context allocation failed");
                    active_weight = ggml_new_tensor_4d(ctx.get(), spec.type, spec.width, 4, spec.batches, 1);
                    auto* selected = select_weight_buft(hparams, active_weight, spec.op, prepared, "fixture");
                    require(active_weight->buffer == nullptr && active_weight->data == nullptr,
                            "support probe leaked dummy buffer or allocated weights");
                    std::printf("choice|%s|%d|%d|%td\n", case_name, active_load, active_tensor,
                                selected ? selected - buffers : std::ptrdiff_t(-1));
                }
                for (int i = 0; i < 7; ++i)
                    std::printf("type|%s|%d|%d|%d\n", case_name, active_load, i, queries[i]);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "fixture failure: %s\n", error.what()); return 1;
    }
}
'''


def expected(candidate):
    rows = []
    for name, buffers, choices in SCENARIOS:
        for load in range(2):
            counts = [0] * 7
            if candidate:
                for dev in {BUFFER_DEVICES[b] for b in buffers}:
                    counts[dev] = 1
            for tensor, (spec, chosen) in enumerate(zip(TENSORS, choices)):
                for buft in buffers:
                    dev = BUFFER_DEVICES[buft]
                    if not candidate:
                        counts[dev] += 1 if dev in (0, 6) else 2 if dev == 1 else 3
                    if buft not in (0, 1, 2, 9) and spec[0] in ('MUL_MAT', 'GET_ROWS'):
                        rows.append(f'support|{name}|{load}|{tensor}|{dev}|{buft}|{int(buft == chosen)}')
                    if buft == chosen:
                        break
                rows.append(f'choice|{name}|{load}|{tensor}|{chosen}')
            rows.extend(f'type|{name}|{load}|{dev}|{count}' for dev, count in enumerate(counts))
    return rows


def main():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('Hosted-only compiler control; local compilation is prohibited')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'whisper-buft-equivalence'
    scratch.mkdir(parents=True, exist_ok=False)
    out = Path(os.environ['HEAVY_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    receipt = {'complete': False, 'acceptance': False, 'baseline': BASELINE, 'candidate': CANDIDATE,
               'ggml': GGML, 'scope': __doc__, 'results': {}}

    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def run(label, argv, timeout=120):
        guard()
        with (out / f'{label}.log').open('w') as stream:
            subprocess.run(list(map(str, argv)), check=True, timeout=timeout, stdout=stream,
                           stderr=subprocess.STDOUT)

    save()
    try:
        sources = {}
        for label, commit in [('baseline', BASELINE), ('candidate', CANDIDATE)]:
            try:
                git('cat-file', '-e', commit + '^{commit}')
            except subprocess.CalledProcessError:
                run('fetch-' + label, ['git', '-C', ROOT, 'fetch', '--depth=1', 'origin', commit])
            sources[label] = git('show', commit + ':src/crispasr.cpp').decode()
            if git('rev-parse', commit + ':ggml').decode().strip() != GGML:
                raise RuntimeError('Pinned sources do not share the required ggml revision')
        if (ROOT / 'src/crispasr.cpp').read_bytes() != sources['candidate'].encode():
            raise RuntimeError('Current native source differs from the pinned candidate')
        if git('rev-parse', 'HEAD', cwd=ROOT / 'ggml').decode().strip() != GGML:
            raise RuntimeError('Initialize the exact pinned ggml submodule before running')
        if git('status', '--porcelain', '--untracked-files=no', cwd=ROOT / 'ggml').strip():
            raise RuntimeError('Pinned ggml submodule has tracked modifications')
        receipt['driver_source'] = git('rev-parse', 'HEAD').decode().strip()
        receipt['script_sha256'] = digest(Path(__file__).read_bytes())
        receipt['source_sha256'] = {label: digest(source.encode()) for label, source in sources.items()}
        tensor_specs = '\n'.join(f'{{GGML_OP_{op}, GGML_TYPE_{dtype}, {width}, {batch}}},'
                                 for op, dtype, width, batch in TENSORS)
        scenarios = '\n'.join('{' + json.dumps(name) + ', {' + ','.join(map(str, buffers)) + '}},'
                              for name, buffers, _ in SCENARIOS)
        tail = TAIL.replace('@TENSORS@', tensor_specs).replace('@SCENARIOS@', scenarios)
        bodies = {}
        for label, source in sources.items():
            functions, cache, prepare = extract(source, label == 'candidate')
            bodies[label] = FIXTURE + functions + prepare + tail
            receipt.setdefault('extracted_sha256', {})[label] = {
                'functions': digest(functions.encode()), 'cache': digest(cache.encode())}
        # Prove that the oracles catch classification collapse and skipped checks.
        old = 'weight_bufts.push_back({entry.first, entry.second, it->second});'
        new = 'weight_bufts.push_back({entry.first, entry.second, GGML_BACKEND_DEVICE_TYPE_GPU});'
        probe = 'op_supported = ggml_backend_dev_supports_op(dev, op_tensor);'
        if bodies['candidate'].count(old) != 1 or bodies['candidate'].count(probe) != 1:
            raise RuntimeError('Negative-control source anchors changed')
        bodies['negative-all-gpu'] = bodies['candidate'].replace(old, new)
        bodies['negative-skip-support'] = bodies['candidate'].replace(probe, 'op_supported = true;')
        for label, body in bodies.items():
            (scratch / (label + '.cpp')).write_text(body)
            (out / (label + '.cpp')).write_text(body)
        cmake = '''cmake_minimum_required(VERSION 3.16)
project(whisper_buft_equivalence LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_subdirectory("@GGML@" ggml EXCLUDE_FROM_ALL)
'''.replace('@GGML@', (ROOT / 'ggml').as_posix())
        for label in bodies:
            cmake += f'add_executable({label} {label}.cpp)\n'
            cmake += f'target_link_libraries({label} PRIVATE ggml-base)\n'
            cmake += f'target_include_directories({label} PRIVATE "{ROOT / "ggml/src"}")\n'
        (scratch / 'CMakeLists.txt').write_text(cmake)
        build = scratch / 'build'
        run('configure', ['cmake', '-S', scratch, '-B', build, '-G', 'Ninja',
                          '-DCMAKE_BUILD_TYPE=Debug', '-DBUILD_SHARED_LIBS=OFF', '-DGGML_CPU=OFF',
                          '-DGGML_NATIVE=OFF', '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF',
                          '-DGGML_VULKAN=OFF', '-DGGML_BLAS=OFF', '-DGGML_OPENMP=OFF',
                          '-DGGML_BACKEND_DL=OFF', '-DGGML_BUILD_TESTS=OFF', '-DGGML_BUILD_EXAMPLES=OFF'])
        run('build', ['cmake', '--build', build, '-j2', '--target', *bodies], timeout=600)
        observed = {}
        for label in bodies:
            guard()
            binary = build / label
            proc = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
            (out / (label + '.stdout')).write_text(proc.stdout)
            (out / (label + '.stderr')).write_text(proc.stderr)
            rows = proc.stdout.splitlines()
            is_candidate = label != 'baseline'
            oracle = expected(is_candidate)
            matches = proc.returncode == 0 and rows == oracle
            receipt['results'][label] = {'returncode': proc.returncode, 'matches_oracle': matches,
                                         'rows': len(rows), 'expected_rows': len(oracle),
                                         'binary_sha256': digest(binary.read_bytes())}
            if not matches:
                mismatch = next((i for i, pair in enumerate(zip(rows, oracle)) if pair[0] != pair[1]),
                                min(len(rows), len(oracle)))
                receipt['results'][label]['first_mismatch'] = {
                    'row': mismatch, 'actual': rows[mismatch] if mismatch < len(rows) else None,
                    'expected': oracle[mismatch] if mismatch < len(oracle) else None}
            save()
            if matches != (not label.startswith('negative-')):
                raise RuntimeError('Unexpected acceptance/negative-control result: ' + label)
            # Negative controls must run successfully and be rejected by the oracle.
            if proc.returncode:
                raise RuntimeError('Fixture failed internally: ' + label)
            observed[label] = [row for row in rows if not row.startswith('type|')]
        if observed['baseline'] != observed['candidate']:
            raise RuntimeError('Ordered placement or per-tensor supports_op behavior differs')
        receipt.update(complete=True, acceptance=True, scenarios=len(SCENARIOS),
                       tensors_per_load=len(TENSORS), loads_per_scenario=2,
                       choices_per_arm=len(SCENARIOS) * len(TENSORS) * 2)
        save()
    except BaseException:
        receipt['error'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
