"""Earlier Melo CUDA work must not falsely qualify an OpenVoice conversion."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('interfaces', ROOT/'tools/ci-heavy/openvoice_interfaces.py')
WORKER = importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(WORKER)


class Attribution(unittest.TestCase):
    def test_actual_converter_window_is_required(self):
        start = 'openvoice2: flow reverse (prior to target)\n'
        end = 'openvoice2: HiFi-GAN output 12 samples\n'
        cuda = '## SPLIT #1: CUDA0 # 1 inputs\n'
        cpu = '## SPLIT #2: CPU # 1 inputs\n'
        self.assertEqual(WORKER.cuda_vocoder_segments(cuda+start+cpu+end),0)
        self.assertEqual(WORKER.cuda_vocoder_segments(start+cuda),0)
        self.assertEqual(WORKER.cuda_vocoder_segments(start+cuda+end),1)
        self.assertEqual(WORKER.cuda_vocoder_segments((start+cuda+end)*6),6)


if __name__ == '__main__':
    unittest.main()
