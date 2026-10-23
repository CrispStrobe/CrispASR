"""Model-free controls for actual CUDA decoder evidence on all public paths."""
import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/ci-heavy'))
execution = importlib.import_module('granite5_execution')


class Granite5DeviceEvidence(unittest.TestCase):
    def test_dynamic_language_head_needs_no_bucket_profile_message(self):
        line = ('node #1443 (   MUL_MAT): node_1443 ( 392K) [CUDA0         ] use=1,c=1: '
                'output.weight ( 392M) [CUDA0         ] node_1442 ( 8K) [CUDA0         ]')
        self.assertEqual(execution.gpu_matmul_nodes(line), 1)
        self.assertEqual(execution.gpu_decoder_matmul_nodes(line), 1)

    def test_encoder_or_gpu_input_does_not_prove_cuda_decoder_work(self):
        cases = [
            '[granite dec-profile] backend=CUDA0 split=0',
            'node #0 (MUL_MAT): node_0 (2M) [CUDA0] use=1: enc.input.weight (640K) [CUDA0]',
            'node #1443 (MUL_MAT): node_1443 (392K) [CPU] use=1: output.weight (392M) [CUDA0]',
            'node #1443 (ADD): node_1443 (392K) [CUDA0] use=1: output.weight (392M) [CUDA0]',
            'node #1443 (MUL_MAT): output.weight (392K) [CPU] use=1: source [CUDA0]',
        ]
        for line in cases:
            with self.subTest(line=line):
                self.assertEqual(execution.gpu_decoder_matmul_nodes(line), 0)


if __name__ == '__main__':
    unittest.main()
