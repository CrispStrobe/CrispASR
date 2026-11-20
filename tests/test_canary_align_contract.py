"""Compile our production subword tokenizer/Viterbi to test failure semantics.

No encoder/model mock claims timing quality: the tiny real-vocabulary fixtures
prove CTC path feasibility and ABI guards independently of model weights.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_session_utf8 import definition

ROOT = Path(__file__).resolve().parents[1]


class Word(ctypes.Structure):
    _fields_ = [('text', ctypes.c_char * 64), ('t0', ctypes.c_int64), ('t1', ctypes.c_int64)]


class SubwordAlignContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / 'src/canary_ctc.cpp').read_text()
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name)
        code = '''
#include "canary_ctc.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
'''
        code += definition(source, 'struct canary_ctc_vocab {') + ';\n'
        code += '''
// Only fields read by the literal postprocessor; no encoder or GPU context.
struct canary_ctc_context {
    struct { struct { int blank_id = 0; int frame_dur_cs = 8; } hparams; } model;
    canary_ctc_vocab vocab;
};
'''
        code += definition(source, 'static bool tokenise_word(') + '\n'
        for name in ('baseline', 'compact'):
            code += definition(source, f'static int canary_ctc_align_words_{name}(') + '\n'
        code += definition(source, 'extern "C" int canary_ctc_align_words(') + '\n'
        code += '''
extern "C" int run(const float* logits, int T, int V, const char** words, int n,
                    canary_ctc_word* out, int mode) {
    canary_ctc_context ctx;
    ctx.vocab.id_to_token = {"<blank>", "▁a", "b"};
    ctx.vocab.token_to_id = {{"<blank>", 0}, {"▁a", 1}, {"b", 2}};
    if (mode == 5) ctx.model.hparams.blank_id = V;
    if (mode == 6) ctx.vocab.token_to_id["▁a"] = V;
    return canary_ctc_align_words(mode == 1 ? nullptr : &ctx,
                                 mode == 2 ? nullptr : logits, T, V,
                                 mode == 3 ? nullptr : words, n,
                                 mode == 4 ? nullptr : out);
}
'''
        (path / 'probe.cpp').write_text(code)
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-shared', '-fPIC',
                        '-I', str(ROOT / 'src'), str(path / 'probe.cpp'), '-o', str(path / 'probe.so')], check=True)
        cls.lib = ctypes.CDLL(str(path / 'probe.so'))
        cls.lib.run.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.POINTER(Word), ctypes.c_int]
        cls.lib.run.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_case(self, scores, words, mode=0):
        scores = np.ascontiguousarray(scores, dtype=np.float32)
        encoded = (ctypes.c_char_p * len(words))(*(w.encode() for w in words))
        output = (Word * len(words))()
        results = []
        for gate in ('0', '1'):
            output = (Word * len(words))()
            with patch.dict(os.environ, {'CRISPASR_SUBWORD_CTC_ALIGN_OPT': gate}):
                rc = self.lib.run(scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  scores.shape[0], scores.shape[1], encoded, len(words), output, mode)
            results.append((rc, [(w.text.decode(), w.t0, w.t1) for w in output]))
        self.assertEqual(results[0], results[1], 'workspace gate changed failure/valid output')
        return results[0]

    def test_impossible_transcript_must_not_report_success(self):
        # Two distinct labels cannot fit in one frame.
        rc, stamps = self.run_case([[0, 9, 0]], ['ab'])
        self.assertNotEqual(rc, 0, ('impossible CTC path reported success', stamps))
        # Two equal labels require an intervening blank: two frames are insufficient.
        rc, stamps = self.run_case([[0, 9, 0], [0, 9, 0]], ['a', 'a'])
        self.assertNotEqual(rc, 0, ('repeated labels need three frames', stamps))

    def test_valid_known_paths_keep_native_centisecond_times(self):
        self.assertEqual(self.run_case([[0, 9, 0]], ['a']), (0, [('a', 0, 8)]))
        self.assertEqual(self.run_case([[0, 9, 0], [0, 0, 9]], ['ab']), (0, [('ab', 0, 16)]))
        self.assertEqual(self.run_case([[0, 9, 0], [9, 0, 0], [0, 9, 0]], ['a', 'a']),
                         (0, [('a', 0, 8), ('a', 16, 24)]))

    def test_invalid_abi_and_token_ids_reject_before_reading(self):
        for mode in range(1, 7):
            with self.subTest(mode=mode):
                rc, _ = self.run_case([[0, 9, 0]], ['a'], mode)
                self.assertNotEqual(rc, 0)
        self.assertNotEqual(self.run_case(np.zeros((0, 3)), ['a'])[0], 0)
        self.assertNotEqual(self.run_case([[0, 9, 0]], [])[0], 0)
        self.assertNotEqual(self.run_case([[0, 9, 0]], ['unknown'])[0], 0)

    def test_nonfinite_terminal_score_is_not_a_success(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            rc, _ = self.run_case(np.full((1, 3), value), ['a'])
            self.assertNotEqual(rc, 0)


if __name__ == '__main__':
    unittest.main()
