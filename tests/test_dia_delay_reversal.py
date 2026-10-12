#!/usr/bin/env python3
"""Compile the production delay reversal and compare actual official DAC inputs."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/dia_tts.cpp'


class DiaDelayReversal(unittest.TestCase):
    def test_official_codec_inputs(self):
        compiler = shutil.which('g++') or shutil.which('clang++')
        self.assertIsNotNone(compiler, 'A C++ compiler is required')
        source = SOURCE.read_text()
        function = re.search(r'static void dia_revert_delay\([\s\S]*?\n}\n', source)
        self.assertIsNotNone(function)
        fixture = json.loads((ROOT / 'tests/fixtures/dia-delay-reference.json').read_text())
        driver = '''#include <cstdint>
#include <cstddef>
#include <iostream>
#include <vector>
struct dia_model {
    uint32_t n_output_heads, max_delay, audio_vocab_size;
    std::vector<uint32_t> delay_pattern;
};
''' + function.group(0) + '''
int main() {
    dia_model m{}; size_t steps;
    std::cin >> steps >> m.n_output_heads >> m.max_delay >> m.audio_vocab_size;
    m.delay_pattern.resize(m.n_output_heads);
    for (auto &d : m.delay_pattern) std::cin >> d;
    std::vector<uint32_t> raw(steps * m.n_output_heads), actual;
    for (auto &x : raw) { int32_t token; std::cin >> token; x = (uint32_t)token; }
    dia_revert_delay(raw, actual, m);
    for (auto x : actual) std::cout << x << ' ';
}
'''
        with tempfile.TemporaryDirectory(prefix='dia-delay-') as folder:
            cpp, binary = Path(folder) / 'test.cpp', Path(folder) / 'test'
            cpp.write_text(driver)
            subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', str(cpp), '-o', str(binary)], check=True)
            for case in fixture['cases']:
                with self.subTest(case=case['name']):
                    raw = [x for row in case['raw'] for x in row]
                    params = [len(case['raw']), 9, 15, fixture['audio_vocab_size'], *fixture['delay_pattern'], *raw]
                    result = subprocess.run([str(binary)], input=' '.join(map(str, params)), text=True,
                                            capture_output=True, check=True)
                    actual = list(map(int, result.stdout.split()))
                    expected = [x for row in case['expected'] for x in row]
                    self.assertEqual(actual, expected, 'Codec frame count and every codebook must match the official source')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source-file', type=Path)
    args, remaining = parser.parse_known_args()
    if args.source_file:
        SOURCE = args.source_file
    unittest.main(argv=[__file__, *remaining])
