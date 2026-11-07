"""Small analytic controls for the non-fused CPU selective scan."""
import unittest
import torch
from torch.nn import functional as F
from mamba_ssm import selective_scan


class ScanTest(unittest.TestCase):
    def test_two_steps_and_skip(self):
        u = torch.tensor([[[2.0, 3.0]]])
        dt = torch.tensor([[[0.5, 0.25]]])
        A = torch.tensor([[-1.0]])
        B = torch.tensor([[[4.0], [5.0]]])
        C = torch.tensor([[[6.0], [7.0]]])
        D = torch.tensor([0.5])
        z = torch.tensor([[[1.0, 1.0]]])
        state1 = torch.tensor(0.5 * 2 * 4)
        state2 = torch.exp(torch.tensor(-0.25)) * state1 + 0.25 * 3 * 5
        expected = torch.tensor([[[state1 * 6 + 1, state2 * 7 + 1.5]]]) * F.silu(torch.tensor(1.0))
        torch.testing.assert_close(selective_scan(u, dt, A, B, C, D, z), expected)

    def test_zero_input_stays_zero(self):
        out = selective_scan(torch.zeros(2, 3, 4), torch.ones(2, 3, 4), -torch.ones(3, 2),
                             torch.ones(2, 4, 2), torch.ones(2, 4, 2), torch.ones(3), torch.ones(2, 3, 4))
        self.assertEqual(torch.count_nonzero(out).item(), 0)


if __name__ == '__main__':
    unittest.main()
