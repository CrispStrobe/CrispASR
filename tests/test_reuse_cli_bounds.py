"""Exercise the actual standalone CLI bounds without model weights."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import gguf
    import numpy as np
except ImportError as error:
    print(f'optional RE-USE CLI fixture dependency unavailable: {error}', file=sys.stderr)
    raise SystemExit(77)

parser = argparse.ArgumentParser()
parser.add_argument('--normal-executable', required=True, type=Path)
parser.add_argument('--long-executable', required=True, type=Path)
arguments, unittest_arguments = parser.parse_known_args()


class CliBoundsTests(unittest.TestCase):
    def test_real_entrypoints_enforce_independent_sample_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_model = root / 'deliberately-missing-model.gguf'
            environment = os.environ.copy()
            environment['CRISPASR_REUSE_FRONTEND'] = 'native'
            for executable, samples, reaches_loader in (
                    (arguments.normal_executable, 80000, False),
                    (arguments.long_executable, 80000, True),
                    (arguments.long_executable, 80080, False)):
                with self.subTest(executable=executable.name, samples=samples):
                    reference = root / f'reference-{samples}.gguf'
                    writer = gguf.GGUFWriter(str(reference), arch='crispasr.reference')
                    writer.add_string('crispasr.ref.backend', 'reuse')
                    writer.add_tensor('raw_audio', np.zeros(samples, dtype=np.float32))
                    writer.add_tensor('frontend_phase', np.zeros((1 + samples // 80, 321), dtype=np.float32))
                    writer.write_header_to_file()
                    writer.write_kv_data_to_file()
                    writer.write_tensors_to_file()
                    writer.close()
                    waveform = root / f'{executable.name}-{samples}-output.f32'
                    result = subprocess.run([str(executable), str(missing_model), str(reference), str(waveform)],
                                            env=environment, text=True, capture_output=True, timeout=30)
                    diagnostic = result.stdout + result.stderr
                    self.assertEqual(result.returncode, 2, diagnostic)
                    self.assertFalse(waveform.exists(), diagnostic)
                    if reaches_loader:
                        self.assertIn(f"failed to open '{missing_model}' for metadata read", diagnostic)
                        self.assertNotIn('missing/invalid bounded raw_audio', diagnostic)
                    else:
                        self.assertIn('missing/invalid bounded raw_audio or frontend_phase reference', diagnostic)
                        self.assertNotIn(str(missing_model), diagnostic)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *unittest_arguments])
