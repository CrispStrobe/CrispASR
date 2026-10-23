"""Model-free rejection controls for the hosted Granite5 regression inputs."""
import copy
import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/ci-heavy'))
regression = importlib.import_module('granite5_regression')


class Granite5RegressionInputs(unittest.TestCase):
    def setUp(self):
        # Schema-only revision: no network or model is used in these controls.
        self.pins = dict(repo='cstr/granite-speech-5.0-2b-GGUF', revision='0' * 40,
            instruction=regression.INSTRUCTION, profiles=['f16', 'q8_0'],
            models={q: dict(filename=f'granite-speech-5.0-2b-{q}.gguf', sha256='0' * 64, bytes=1)
                    for q in ('f16', 'q8_0')},
            reference=dict(repo='cstr/crispasr-regression-fixtures', repo_type='dataset',
                revision='0426caae5f4f536d87058f5cb36d77525dd5bcde',
                path='granite-speech-5.0-2b/a02a0dd9299e0b759704f520fe3959f50a764f71/jfk-ref.gguf',
                sha256='ec91b6de806a1919d41edeb604e0aba8b636f3eadc7a4be487c8bd2ff57d5d84', bytes=19195808))

    def test_rejects_mutable_or_unqualified_inputs(self):
        regression.check_pins(self.pins)
        mutations = [
            lambda p: p.update(revision='main'),
            lambda p: p.update(instruction='can you transcribe the speech into a written format?'),
            lambda p: p['models'].pop('q8_0'),
            lambda p: p['models']['q8_0'].update(filename='granite-speech-5.0-2b-q4_k.gguf'),
            lambda p: p['reference'].update(sha256='0' * 64),
            lambda p: p['reference'].update(revision='main'),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                pins = copy.deepcopy(self.pins)
                mutate(pins)
                with self.assertRaises(ValueError):
                    regression.check_pins(pins)

    def test_same_size_corruption_and_truncation_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'
            content = b'original reference fixture'
            record = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            path.write_bytes(content)
            regression.check_content(path, record)
            for corrupt in (b'x' * len(content), content[:-1]):
                path.write_bytes(corrupt)
                with self.assertRaises(ValueError):
                    regression.check_content(path, record)

    def test_f16_only_requires_an_explicit_matching_profile_declaration(self):
        pins = copy.deepcopy(self.pins)
        pins['profiles'] = ['f16']
        with self.assertRaises(ValueError):
            regression.check_pins(pins)  # An undeclared Q8 file must not be tested/promoted.
        pins['models'].pop('q8_0')
        regression.check_pins(pins)


if __name__ == '__main__':
    unittest.main()
