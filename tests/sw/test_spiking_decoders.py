"""
Test decoders
"""
import unittest
import torch

from hxtorch.spiking.transforms.decode import (
    MaxOverTime, SumOverTime, MeanOverTime)


class TestDecoder(unittest.TestCase):
    """ Test decoders """

    def test_max_over_time(self):
        """ Test max traces along time dimension """
        decoder = MaxOverTime()

        inputs = torch.zeros(100, 10, 100)
        for b in range(inputs.shape[1]):
            for n in range(inputs.shape[-1]):
                inputs[torch.randint(0, int(inputs.shape[2]), (1,)), b, n] = 1.

        # Forward
        scores = decoder(inputs)

        # Test shape
        self.assertTrue(torch.equal(
            torch.tensor(scores.shape),
            torch.tensor([inputs.shape[1], inputs.shape[-1]])))

        # Test
        self.assertTrue(torch.equal(scores, torch.ones(*(scores.shape))))

    def test_sum_over_time(self):
        """ Test sum traces along time dimension """
        decoder = SumOverTime()

        inputs = torch.zeros(100, 10, 100)
        for b in range(inputs.shape[1]):
            for n in range(inputs.shape[-1]):
                idx_1 = torch.randint(0, int(inputs.shape[2]) - 1, (1,))
                inputs[idx_1, b, n] = 1.
                inputs[idx_1 + 1, b, n] = 2.

        # Forward
        scores = decoder(inputs)

        # Test shape
        self.assertTrue(torch.equal(
            torch.tensor(scores.shape),
            torch.tensor([inputs.shape[1], inputs.shape[-1]])))

        # Test
        self.assertTrue(torch.equal(scores, 3. * torch.ones(*(scores.shape))))

    def test_mean_over_time(self):
        """ Test max traces along time dimension """
        decoder = MeanOverTime()

        inputs = torch.zeros(100, 10, 100)
        for b in range(inputs.shape[1]):
            for n in range(inputs.shape[-1]):
                inputs[torch.randint(
                    0, int(inputs.shape[2]), (1,)), b, n] = 100.

        # Forward
        scores = decoder(inputs)

        # Test shape
        self.assertTrue(torch.equal(
            torch.tensor(scores.shape),
            torch.tensor([inputs.shape[1], inputs.shape[-1]])))

        # Test
        self.assertTrue(torch.equal(scores, torch.ones(*(scores.shape))))


if __name__ == "__main__":
    unittest.main()
