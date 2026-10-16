"""Freeze computation calls while adding diagnostic outputs to Dia.

The optional F32 cross-cache path is explicitly removed from this default-path
contract below; its output is checked separately. This is a source contract,
not runtime transparency proof: OFF/ON/OFF on the
full model is still required because marking outputs changes allocation.
The fixture was extracted from pristine main 9aa8df20, never from this port.
"""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'ggml_set_name', 'ggml_set_output', 'ggml_graph_get_tensor',
           'ggml_backend_tensor_get', 'ggml_nelements', 'ggml_free', 'ggml_is_contiguous'}


def computation_calls(source):
    source = source.replace('ctx0, cross_type, cross_kv_dim, T_enc, B',
                            'ctx0, GGML_TYPE_F16, cross_kv_dim, T_enc, B')
    source = re.sub(r'//[^\n]*|/\*.*?\*/', '', source, flags=re.S)
    calls = []
    for match in re.finditer(r'\b(ggml_\w+)\s*\(', source):
        if match[1] in IGNORED:
            continue
        end, depth = match.end(), 1
        while depth:
            depth += (source[end] == '(') - (source[end] == ')')
            end += 1
        call = re.sub(r'\s+', '', source[match.start():end])
        if call in {
            'ggml_backend_tensor_set(ggml_graph_get_tensor(gf,ckn.c_str()),cross_k_f32[l].data(),0,cross_k_f32[l].size()*sizeof(float))',
            'ggml_backend_tensor_set(ggml_graph_get_tensor(gf,cvn.c_str()),cross_v_f32[l].data(),0,cross_v_f32[l].size()*sizeof(float))',
        }:
            continue
        calls.append(call)
    return calls


class DiaCaptureNeutrality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((ROOT / 'tests/fixtures/dia-capture-pristine-main.json').read_text())
        cls.source = (ROOT / 'src/dia_tts.cpp').read_text()

    def test_ordered_computation_calls_equal_pristine_main(self):
        self.assertEqual(computation_calls(self.source), self.fixture['computation_calls'])

    def test_default_sampler_body_unchanged(self):
        source = (ROOT / 'src/dia_sampling.h').read_text()
        source = source.replace('#include \"dia_exponential_rng.h\"\n', '')
        source = source.replace('std::mt19937& rng, const float* exponential_noise = nullptr)', 'std::mt19937& rng)')
        source = source.replace('    if (exponential_noise) {\n        return dia_rng::exponential_race(probs.data(), exponential_noise, vocab_size);\n    }\n\n', '')
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                         self.fixture['sampler_sha256'])

    def test_guard_detects_graph_and_precision_mutations(self):
        for original, changed in [('ggml_mul_mat(', 'ggml_add('),
                                  ('ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F16', 'ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32')]:
            self.assertIn(original, self.source)
            self.assertNotEqual(computation_calls(self.source.replace(original, changed, 1)),
                                self.fixture['computation_calls'])


if __name__ == '__main__':
    unittest.main()
