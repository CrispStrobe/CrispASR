"""Keep the hardware-free capability test attached to CUDA's actual dispatcher."""
from pathlib import Path
import re
import unittest

CUDA = Path(__file__).resolve().parents[1] / "ggml/src/ggml-cuda"


class CudaSsmDispatch(unittest.TestCase):
    def test_supports_op_uses_the_tested_predicate(self):
        source = (CUDA / "ggml-cuda.cu").read_text()
        branch = source.split("case GGML_OP_SSM_CONV: {", 1)[1].split("}", 1)[0]
        self.assertEqual(branch.strip(), "return ggml_cuda_ssm_conv_supported(op);")

    def test_advertised_widths_match_kernel_dispatch(self):
        dispatch = (CUDA / "ssm-conv.cu").read_text().split("switch (nc)", 1)[1].split("default:", 1)[0]
        kernels = {int(k) for k in re.findall(r"case\s+(\d+)\s*:", dispatch)}
        predicate = (CUDA / "ssm-conv-support.h").read_text()
        advertised = {int(k) for k in re.findall(r"k\s*==\s*(\d+)", predicate)}
        self.assertTrue(kernels)
        self.assertEqual(kernels, advertised)


if __name__ == "__main__":
    unittest.main()
