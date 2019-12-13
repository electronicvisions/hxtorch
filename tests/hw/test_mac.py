import unittest
import torch
import hxtorch
from dlens_vx import hxcomm, sta, logger
import pygrenade_vx as grenade

logger.default_config(level=logger.LogLevel.INFO)

class TestHXMAC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with hxcomm.ManagedConnection() as connection:
            sta.run(connection, sta.generate(sta.DigitalInit())[0].done())
            chip = grenade.ChipConfig()
            hxtorch.init(chip, connection)

    @classmethod
    def tearDownClass(cls):
        hxtorch.release()

    def test_construction(self):
        with self.assertRaises(RuntimeError):
            data_in = torch.ones(5)
            weights_in = torch.empty(5, 3, 4)
            hxtorch.mac(data_in, weights_in)

        weights_in = torch.ones(5, 3)
        data_in = torch.ones(5)
        result = hxtorch.mac(data_in, weights_in)
        self.assertEqual(result.size(), torch.Size([3]))


if __name__ == '__main__':
    unittest.main()