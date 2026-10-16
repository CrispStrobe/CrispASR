"""Compile actual Dia comparison/input guards against the actual GGUF reader.

Requires numpy, gguf, a C++ compiler and a built ggml tree. No model weights.
Use CRISPASR_DIA_DIFF_GGML_BUILD to select a standalone ggml build directory.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class DiaDiffCompare(unittest.TestCase):
    def test_actual_comparison_and_code_guards(self):
        build = Path(os.environ.get('CRISPASR_DIA_DIFF_GGML_BUILD', ROOT / 'build/ggml'))
        with tempfile.TemporaryDirectory(prefix='dia-diff-compare-') as temp:
            directory = Path(temp)
            archive = directory / 'ref.gguf'
            writer = gguf.GGUFWriter(str(archive), arch='crispasr.reference')
            writer.add_tensor('stage', np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype='f4'))
            for name, value in [('decoder_input_tokens', 1026), ('decoder_codes_rev', 1023),
                                ('fraction', 1.5), ('negative', -1), ('too_large', 1027), ('nonfinite', float('nan'))]:
                writer.add_tensor(name, np.full((2, 9), value, dtype='f4'))
            writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
            driver = directory / 'guard.cpp'
            driver.write_text(r'''
#include "crispasr_dia_diff.h"
#include <limits>
int main(int argc, char** argv) {
    if (argc != 3) return 1;
    crispasr_diff::Ref ref;
    if (!ref.load(argv[1])) return 2;
    std::vector<float> correct={1,2,3,4,2,3,4,5};
    if (!dia_diff::compare(ref, "stage", correct)) return 3;
    auto bad=correct; for (auto& x:bad) x*=2;
    if (dia_diff::compare(ref, "stage", bad)) return 4; // cosine 1, wrong scale
    bad=correct; bad[4]*=2;
    if (dia_diff::compare(ref, "stage", bad)) return 5;
    bad=correct; bad[0]=std::numeric_limits<float>::quiet_NaN();
    if (dia_diff::compare(ref, "stage", bad)) return 6;
    if (dia_diff::compare(ref, "missing", correct) || dia_diff::compare(ref, "stage", {})) return 7;
    bad=correct; bad.pop_back();
    if (dia_diff::compare(ref, "stage", bad)) return 8;
    if (!dia_diff::compare(ref, "stage", correct, true)) return 9;
    bad=correct; bad[0]+=0.00001f;
    if (dia_diff::compare(ref, "stage", bad, true)) return 10;
    const std::filesystem::path path=argv[2];
    if (!dia_diff::write_codes(ref, "decoder_input_tokens", path)) return 11;
    if (!dia_diff::write_codes(ref, "decoder_codes_rev", path)) return 12;
    for (const char* name : {"fraction", "negative", "too_large", "nonfinite", "missing", "stage"})
        if (dia_diff::write_codes(ref, name, path)) return 13;
    // Mutate a real archive's codec grid: EOS/BOS are invalid DAC indices.
    const auto codes=ref.get_f32("decoder_codes_rev");
    for (float value : {1024.f, 1025.f, 1026.f}) {
        const_cast<float*>(codes.first)[0]=value;
        if (dia_diff::write_codes(ref, "decoder_codes_rev", path)) return 14;
    }
    puts("Dia diff: 20 comparison/input acceptance and rejection controls passed");
    return 0;
}
''')
            # Link the actual current ggml libraries and production archive reader.
            binary = directory / 'guard'
            libraries = []
            for name in ['ggml', 'ggml-cpu', 'ggml-base']:
                matches = list(build.rglob('lib'+name+'.so')) or list(build.rglob('lib'+name+'.a'))
                if len(matches) != 1:
                    raise RuntimeError(f'Expected one built {name} library under {build}, got {matches}')
                libraries.append(matches[0])
            command = ['c++', '-std=c++17', '-O1', '-I', str(ROOT / 'examples/cli'),
                       '-I', str(ROOT / 'src'), '-I', str(ROOT / 'ggml/include'), '-I', str(ROOT / 'ggml/src'),
                       str(driver), str(ROOT / 'examples/cli/crispasr_diff.cpp'), str(ROOT / 'src/core/gguf_loader.cpp'),
                       *['-Wl,-rpath,'+str(p) for p in sorted({lib.parent for lib in libraries})],
                       '-Wl,--start-group', *map(str, libraries), '-Wl,--end-group',
                       '-fopenmp', '-pthread', '-ldl', '-o', str(binary)]
            subprocess.run(command, check=True)
            subprocess.run([str(binary), str(archive), str(directory / 'codes.i32')], check=True)


if __name__ == '__main__':
    unittest.main()
