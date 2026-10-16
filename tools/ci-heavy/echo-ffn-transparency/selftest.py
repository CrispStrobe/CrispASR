#!/usr/bin/env python3
"""Reject observer changes without loading weights or mocking positive output."""
import importlib.util
import ctypes
import os
import json
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / 'echo_ffn_transparency.py'
spec = importlib.util.spec_from_file_location('transparency', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ControlTest(unittest.TestCase):
    def setUp(self):
        self.before = {'logits': np.array([1, 2, 3], dtype='f4')}
        self.after = {k: v.copy() for k, v in self.before.items()}
        self.enabled = {k: v.copy() for k, v in self.before.items()}
        for suffix in module.SUFFIXES:
            self.enabled['llm_ffn_0_' + suffix] = np.arange(4 if suffix in ['mlp_in', 'down_out', 'mlp_out'] else 8, dtype='f4')

    def test_identity_and_complete_new_stages(self):
        self.assertEqual(module.compare_capture(self.before, self.enabled, self.after, 1),
                         dict(old_arrays=1, new_arrays=6, exact=True))

    def test_scale_change_rejected_despite_unit_cosine(self):
        self.enabled['logits'] *= 1.05
        with self.assertRaises(ValueError):
            module.compare_capture(self.before, self.enabled, self.after, 1)

    def test_each_missing_boundary_rejected(self):
        for suffix in module.SUFFIXES:
            bad = dict(self.enabled)
            del bad['llm_ffn_0_' + suffix]
            with self.assertRaises(ValueError):
                module.compare_capture(self.before, bad, self.after, 1)

    def test_after_hook_leak_rejected(self):
        self.after['llm_ffn_0_mlp_in'] = np.ones(4, dtype='f4')
        with self.assertRaises(ValueError):
            module.compare_capture(self.before, self.enabled, self.after, 1)

    def test_signed_zero_and_dtype_change_rejected(self):
        for a, b in [(np.array([0.], 'f4'), np.array([-0.], 'f4')),
                     (np.ones(2, 'f4'), np.ones(2, 'f8'))]:
            with self.assertRaises(ValueError):
                module.exact(a, b)

    def test_nonfinite_and_empty_rejected(self):
        for a in [np.array([np.nan], 'f4'), np.array([], 'f4')]:
            with self.assertRaises(ValueError):
                module.exact(a, a)

    def test_wrong_ffn_width_rejected(self):
        self.enabled['llm_ffn_0_up_out'] = np.ones(7, dtype='f4')
        with self.assertRaises(ValueError):
            module.compare_capture(self.before, self.enabled, self.after, 1)

    def test_capture_flag_is_literal_and_clean(self):
        parent = dict(INDEX_ECHO_REF_CAPTURE_FFN='1', CRISPASR_INDEX_ECHO_DUMP_FFN='1', HF_HOME='cache')
        off = module.observer_environment(parent, False, Path('proof'))
        self.assertNotIn('INDEX_ECHO_REF_CAPTURE_FFN', off)
        self.assertNotIn('CRISPASR_INDEX_ECHO_DUMP_FFN', off)
        self.assertEqual(off['HF_HOME'], 'cache')
        self.assertEqual(module.observer_environment(parent, True, Path('proof'))['CRISPASR_INDEX_ECHO_DUMP_FFN'], '1')

    def test_required_existing_inventory_and_model_widths(self):
        with self.assertRaises(ValueError):
            module.compare_capture(self.before, self.enabled, self.after, 1, required_old={'missing'})
        with self.assertRaises(ValueError):
            module.compare_capture(self.before, self.enabled, self.after, 1, widths=(2048, 6144))

    def test_cue_text_against_actual_cli_and_session_source(self):
        # Compile the actual formatting statements from both production paths.
        # Tiny carrier structs stand in only for their string/const-char fields.
        files = ['src/crispasr_c_api.cpp', 'examples/cli/crispasr_backend_index_echo.cpp']
        fixtures = [dict(transcript='Hello', translation=''),
                    dict(transcript='Guten Tag', translation='Good day')]
        pattern = r'segment\.text = cue\.transcript;\s*if \(\*cue\.translation\)\s*segment\.text \+= std::string\("\\n"\) \+ cue\.translation;'
        source = '#include <iostream>\n#include <string>\nstruct Cue{const char* transcript;const char* translation;};\nstruct Segment{std::string text;};\n'
        for i, name in enumerate(files):
            matches = re.findall(pattern, (module.ROOT / name).read_text())
            self.assertEqual(len(matches), 1, name)
            source += f'std::string render{i}(const Cue& cue){{Segment segment;{matches[0]} return segment.text;}}\n'
        source += 'int main(){\n'
        for i in range(len(files)):
            for cue in fixtures:
                source += 'std::cout << render' + str(i) + '({' + json.dumps(cue['transcript']) + ',' + json.dumps(cue['translation']) + "}); std::cout.put('\\0');\n"
        source += '}\n'
        with tempfile.TemporaryDirectory(dir=os.getenv('TMPDIR')) as temp:
            cpp, exe = Path(temp) / 'cue.cpp', Path(temp) / 'cue'
            cpp.write_text(source)
            subprocess.run([shutil.which('c++'), '-std=c++17', str(cpp), '-o', str(exe)], check=True)
            actual = subprocess.check_output([str(exe)]).split(b'\0')
        expected = [module.cue_text(c).encode() for _ in files for c in fixtures] + [b'']
        self.assertEqual(actual, expected)
        # The previous expectation fails the actual empty-translation contract.
        self.assertNotEqual(fixtures[0]['transcript'] + '\n' + fixtures[0]['translation'], actual[0].decode())

    def test_ctypes_layout_against_actual_native_header(self):
        # Compile the actual public declarations, never a duplicated C struct.
        compiler = shutil.which('c++')
        self.assertIsNotNone(compiler)
        types = [('Params', 'index_echo_context_params'), ('Cue', 'index_echo_cue'), ('Result', 'index_echo_result')]
        source = '#include "index_echo.h"\n#include <cstddef>\n#include <cstdio>\nint main(){\n'
        expected = []
        for pyname, cname in types:
            cls = getattr(module, pyname)
            source += f'printf("%zu\\n", sizeof({cname}));\n'
            expected.append(ctypes.sizeof(cls))
            for field, _ in cls._fields_:
                source += f'printf("%zu\\n", offsetof({cname}, {field}));\n'
                expected.append(getattr(cls, field).offset)
        source += '}\n'
        with tempfile.TemporaryDirectory(dir=os.getenv('TMPDIR')) as temp:
            cpp, exe = Path(temp) / 'layout.cpp', Path(temp) / 'layout'
            cpp.write_text(source)
            subprocess.run([compiler, '-std=c++17', '-I' + str(module.ROOT / 'src'), str(cpp), '-o', str(exe)], check=True)
            actual = list(map(int, subprocess.check_output([str(exe)], text=True).split()))
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
