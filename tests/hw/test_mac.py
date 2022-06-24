from functools import partial
import unittest
import torch
import hxtorch
import time

hxtorch.logger.reset()
hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.INFO)
hxtorch.logger.set_loglevel(hxtorch.logger.get('hxcomm'), hxtorch.logger.LogLevel.WARN)
hxtorch.logger.set_loglevel(hxtorch.logger.get('grenade'), hxtorch.logger.LogLevel.WARN)

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


class TestHXMACPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()

    def test_sweep_batchsize(self):
        ops = None
        duration = None
        for batch in range(14):
            weights_in = (torch.rand(
                hxtorch.constants.hardware_matrix_height * 2,
                hxtorch.constants.hardware_matrix_width) - 0.5) \
                * hxtorch.constants.synaptic_weight_max
            data_in = torch.rand(
                2**batch, hxtorch.constants.hardware_matrix_height * 2) \
                * hxtorch.constants.input_activation_max
            # perform unmeasured pre-run to warm caches
            hxtorch.mac(data_in, weights_in)
            # measured run
            begin = time.time()
            hxtorch.mac(data_in, weights_in)
            duration = time.time() - begin
            ops = weights_in.shape[0] * weights_in.shape[1] * data_in.shape[0]
            hxtorch.logger.get(
                f"{self.__class__.__name__}.test_sweep_batchsize").INFO(
                "MAC performance {:.0f} op/s ".format(ops / duration) +
                "at batchsize: {} and matrix size: {}.".format(
                    data_in.shape[0], weights_in.shape))
        # for largest batchsize expect 300Mop/s with 20% allowed deviation
        self.assertGreater(ops / duration, 300e6 * 0.8)


    def test_sweep_matrixsize(self):
        ops = None
        duration = None
        for size in range(11):
            weights_in = (torch.rand(2**size, 2**size) - 0.5) \
                * hxtorch.constants.synaptic_weight_max
            data_in = torch.rand(2000, 2**size) \
                * hxtorch.constants.input_activation_max
            # perform unmeasured pre-run to warm caches
            hxtorch.mac(data_in, weights_in)
            # measured run
            begin = time.time()
            hxtorch.mac(data_in, weights_in)
            duration = time.time() - begin
            ops = 2000 * 2**size * 2**size
            hxtorch.logger.get(
                f"{self.__class__.__name__}.test_sweep_matrixsize").INFO(
                "MAC performance {:.0f} op/s ".format(ops / duration) +
                "at batch size: {} and matrix size: {}.".format(
                    2000, weights_in.shape))
        # for largest matrixsize expect 330Mop/s with 20% allowed deviation
        self.assertGreater(ops / duration, 330e6 * 0.8)


if __name__ == '__main__':
    unittest.main()
