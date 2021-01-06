from functools import partial
import unittest
import torch
import hxtorch

hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.INFO)

class TestHXMAC(unittest.TestCase):
    mac = hxtorch.mac

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware_minimal()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()

    def test_construction(self):
        data_in = torch.ones(5)
        weights_in = torch.empty(5, 3, 4)
        with self.assertRaises(RuntimeError):
            self.mac(data_in, weights_in)

        data_in = torch.full((5,), -2.)
        weights_in = torch.ones(5, 3)
        with self.assertRaises(OverflowError):
            self.mac(data_in, weights_in)

        data_in = torch.ones(5)
        weights_in = torch.full((5, 3), 65.)
        with self.assertRaises(OverflowError):
            self.mac(data_in, weights_in)

        weights_in = torch.ones(5, 3)
        data_in = torch.ones(5)
        result = self.mac(data_in, weights_in)
        self.assertEqual(result.size(), torch.Size([3]))

    def test_batch_input(self):
        weights_in = torch.ones(5, 3)
        weights_in.requires_grad = True
        data_in = torch.ones(10, 5)
        result = self.mac(data_in, weights_in)
        self.assertEqual(result.size(), torch.Size([10, 3]))
        loss = result.sum()
        loss.backward()


class TestHXMACmock(TestHXMAC):
    mac = partial(hxtorch.mac, mock=True)

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

if __name__ == '__main__':
    unittest.main()
