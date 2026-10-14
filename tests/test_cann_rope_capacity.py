"""Compile actual CANN allocation/cache code; test byte capacity, not RoPE math.

CPU malloc/free instrument only requested buffer sizes. No ACL device operation
or numerical reference is simulated. Run with TMPDIR on the storage volume.
CANN_CAPACITY_OUT retains complete source, executable and per-case JSON rows.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CannRopeCapacityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ.get('CANN_ROPE_SOURCE_ROOT', ROOT / 'ggml'))
        operations = source / 'src/ggml-cann/aclnn_ops.cpp'
        common = source / 'src/ggml-cann/common.h'
        text = operations.read_text()
        start = text.index('    // Step5: calculate sin cos.', text.index('static void aclnn_rope_cache_init('))
        end = text.index('    // sin/cos\n', start)
        allocation = text[start:end]
        text = common.read_text()
        start = text.index('struct ggml_cann_rope_cache {')
        end = text.index('\n};', start) + 3
        metadata = text[start:end]
        cls.temporary = None
        if os.environ.get('CANN_CAPACITY_OUT'):
            cls.out = Path(os.environ['CANN_CAPACITY_OUT'])
            cls.out.mkdir(parents=True, exist_ok=True)
        else:
            cls.temporary = tempfile.TemporaryDirectory(prefix='cann-capacity-')
            cls.out = Path(cls.temporary.name)
        preamble = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
static std::unordered_map<void *, size_t> capacity;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
static int aclrtMalloc(void **ptr, size_t bytes, int) {
    *ptr = std::malloc(bytes);
    if (!*ptr) return 1;
    capacity[*ptr] = bytes;
    return 0;
}
static int aclrtFree(void *ptr) {
    if (capacity.erase(ptr) != 1) return 1;
    std::free(ptr);
    return 0;
}
#define ACL_CHECK(expr) assert((expr) == 0)
'''
        driver = r'''
struct Context { ggml_cann_rope_cache rope_cache; };
static void allocate(Context &ctx, int64_t theta_scale_length, int64_t position_length) {
'''+allocation+r'''
}
int main(int argc, char **argv) {
    if (argc < 3 || argc % 2 != 1) return 2;
    Context ctx;
    int sections[4] = {0, 0, 0, 0};
    for (int i = 1; i < argc; i += 2) {
        const int64_t dims = std::strtoll(argv[i], nullptr, 10);
        const int64_t positions = std::strtoll(argv[i+1], nullptr, 10);
        if (dims <= 0 || dims % 2 || positions <= 0) return 3;
        allocate(ctx, dims / 2, positions);
        const size_t required = size_t(dims) * size_t(positions) * sizeof(float);
        const size_t sin_bytes = capacity.at(ctx.rope_cache.sin_cache);
        const size_t cos_bytes = capacity.at(ctx.rope_cache.cos_cache);
        const bool safe = sin_bytes >= required && cos_bytes >= required;
        std::printf("{\"frame\":%d,\"rotary_dims\":%lld,\"positions\":%lld,"
                    "\"required_bytes_each\":%zu,\"sin_bytes\":%zu,\"cos_bytes\":%zu,\"safe\":%s}\n",
                    (i-1)/2, (long long)dims, (long long)positions,
                    required, sin_bytes, cos_bytes, safe ? "true" : "false");
        // Execute the actual post-computation metadata update. No sin/cos writes.
        ctx.rope_cache.set(dims/2, positions, 0, .99f, 1, 1, true, false, false, false, sections);
    }
    return 0;
}
'''
        cls.binary = cls.out / 'capacity-probe'
        cpp = cls.out / 'actual-capacity.cpp'
        cpp.write_text(preamble + metadata + driver)
        command = ['c++', '-std=c++17', '-O0', '-Wall', '-Wextra', '-Werror', str(cpp), '-o', str(cls.binary)]
        subprocess.run(command, check=True)
        receipt = dict(scope=__doc__, command=command,
                       operations_sha256=hashlib.sha256(operations.read_bytes()).hexdigest(),
                       common_sha256=hashlib.sha256(common.read_bytes()).hexdigest(),
                       allocation_sha256=hashlib.sha256(allocation.encode()).hexdigest(),
                       metadata_sha256=hashlib.sha256(metadata.encode()).hexdigest(),
                       probe_source_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
                       probe_binary_sha256=hashlib.sha256(cls.binary.read_bytes()).hexdigest())
        (cls.out / 'method.json').write_text(json.dumps(receipt, indent=2)+'\n')

    @classmethod
    def tearDownClass(cls):
        if cls.temporary:
            cls.temporary.cleanup()

    def check_sequence(self, name, frames):
        command = [str(self.binary)] + [str(value) for pair in frames for value in pair]
        output = subprocess.check_output(command, text=True)
        (self.out / (name+'.jsonl')).write_text(output)
        rows = [json.loads(line) for line in output.splitlines()]
        self.assertEqual(len(rows), len(frames))
        for row, (dims, positions) in zip(rows, frames):
            self.assertEqual(row['required_bytes_each'], dims*positions*4)
            self.assertTrue(row['safe'], row)

    def test_reported_constant_position_width_sequence(self):
        self.check_sequence('reported-width-sequence', [(64, 512), (64, 512), (256, 512), (128, 512)])

    def test_width_growth_with_fewer_positions(self):
        self.check_sequence('width-growth-fewer-positions', [(64, 512), (256, 256)])

    def test_ordinary_position_growth(self):
        self.check_sequence('position-growth', [(128, 1), (128, 512)])

    def test_same_byte_requirement_at_different_shapes(self):
        self.check_sequence('same-bytes-different-shapes', [(64, 512), (128, 256)])


if __name__ == '__main__':
    unittest.main()
