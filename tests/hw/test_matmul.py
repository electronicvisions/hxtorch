from collections import namedtuple
from functools import partial
from typing import ClassVar
import unittest
import torch
import hxtorch
from hxtorch import logger

from hxtorch_shared_test_tools import rand_full

logger.default_config(level=logger.LogLevel.INFO)
logger.set_loglevel(logger.get("grenade"), logger.LogLevel.WARN)


class MatmulInput(namedtuple('MatmulInput', ["input", "other"])):
    """
    An input to a matmul operation.
    """
    def duplicate(self):
        return self._make(arg.data.clone().requires_grad_()
                          if hasattr(arg, "requires_grad") else arg
                          for arg in self)


class TestMatmulPyTorch(unittest.TestCase):
    """
    Tests the torch matmul operation.
    """

    matmul: ClassVar = torch.matmul
    noise_std: ClassVar[float] = 0.
    gain: ClassVar[float] = 1.

    def test_output_shape_gradient(self):
        """
        Compares the output shape and gradients of the matmul operation to the
        output of the torch implementation for different input dimensions.
        """

        test_inputs = {
            "1-d x 1-d":
            MatmulInput(rand_full((128,), 12.), rand_full((128,), 15.)),
            "1-d x 2-d":
            MatmulInput(rand_full((128,), 12.), rand_full((128, 5), 15.)),
            # TODO: implement > 2D weights
            # "1-d x 3-d":
            # MatmulInput(rand_full((128,), 12.), rand_full((2, 128, 5), 15.)),
            # "1-d x 4-d":
            # MatmulInput(rand_full((128,), 12.), rand_full((4, 2, 128, 5), 15.)),
            "2-d x 1-d":
            MatmulInput(rand_full((3, 128), 12.), rand_full((128,), 15.)),
            "2-d x 2-d":
            MatmulInput(rand_full((3, 128), 12.), rand_full((128, 5), 15.)),
            # "2-d x 3-d":
            # MatmulInput(rand_full((3, 128), 12.), rand_full((2, 128, 5), 15.)),
            # "2-d x 4-d":
            # MatmulInput(rand_full((3, 128), 12.), rand_full((4, 2, 128, 5), 15.)),
            "3-d x 1-d":
            MatmulInput(rand_full((2, 3, 128), 12.), rand_full((128,), 15.)),
            "3-d x 2-d":
            MatmulInput(rand_full((2, 3, 128), 12.), rand_full((128, 5), 15.)),
            # TODO: implement batched mode
            # "3-d x 3-d":
            # MatmulInput(rand_full((2, 3, 128), 12.), rand_full((2, 128, 5), 15.)),
            # "3-d x 4-d":
            # MatmulInput(rand_full((2, 3, 128), 12.), rand_full((4, 2, 128, 5), 15.)),
            "2-d x 2-d non-contiguous input":
            MatmulInput(
                rand_full((128, 3), 12.).data.t().requires_grad_(),
                rand_full((128, 5), 15.)
            ),
            "2-d x 2-d non-contiguous other":
            MatmulInput(
                rand_full((3, 128), 12.),
                rand_full((5, 128), 15.).data.t().requires_grad_()
            )
        }

        for mode, matmul_input in test_inputs.items():
            with self.subTest(mode=mode):
                result = self.matmul(**matmul_input._asdict())
                self.assertTrue(result.is_contiguous())

                matmul_input_torch = matmul_input.duplicate()
                result_torch = torch.matmul(**matmul_input_torch._asdict())
                self.assertEqual(result.size(), result_torch.size())

                # compute gradients
                result.backward(torch.ones_like(result))
                result_torch.backward(torch.ones_like(result_torch))

                for name, arg in matmul_input._asdict().items():
                    if hasattr(arg, "grad"):
                        grad = arg.grad
                        grad_torch = getattr(matmul_input_torch, name).grad \
                            * self.gain
                        self.assertTrue(
                            torch.allclose(grad, grad_torch, rtol=.001),
                            f"{name.capitalize()} gradient does not match:\n"
                            f"{grad}\n!=\n{grad_torch}")

    def test_noise_and_gain(self):
        log = hxtorch.logger.get(self.__class__.__name__)
        data_in = torch.full((100, 128), 12., dtype=torch.float)
        weights_in = torch.full((128, 256), 15., dtype=torch.float)

        result = self.matmul(data_in, weights_in)
        log.info(f"Mean output: {result.mean():.1f}")
        result_torch = torch.matmul(data_in, weights_in)

        # check gain
        gain = torch.median((result / result_torch).view(-1)).item()
        log.info(f"Gain: {gain:.5f} (median)")
        self.assertLess(abs(gain - self.gain), .25 * self.gain)

        # check noise
        noise = result - result_torch * gain
        noise_std = noise.std(dim=0).mean() # this removes fixed pattern noise
        noise_fixed_std = noise.mean(dim=0).std() # this removes stat. noise
        log.info(f"Noise: ±{noise_std:.4f} (stat.) "
                 f"/ {noise_fixed_std / result.mean() * 100:.2f}% (fixed)")
        self.assertLessEqual(noise_std, 2. * self.noise_std,
                             "Statistical noise on neurons is higher than expected.")


class TestMatmulHX(TestMatmulPyTorch):
    """
    Tests the hxtorch matmul operation.
    """
    matmul: ClassVar = partial(hxtorch.matmul, wait_between_events=10)

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()
        mock_parameter = hxtorch.measure_mock_parameter()
        hxtorch.set_mock_parameter(mock_parameter)
        cls.noise_std = mock_parameter.noise_std
        cls.gain = mock_parameter.gain

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()


class TestMatmulHXmock(TestMatmulPyTorch):
    """
    Tests the hxtorch matmul operation.
    """
    matmul: ClassVar = partial(hxtorch.matmul, mock=True)
    noise_std: ClassVar[float] = 2.
    gain: ClassVar[float] = 0.002

    @classmethod
    def setUpClass(cls):
        hxtorch.init(hxtorch.MockParameter(
            noise_std=cls.noise_std, gain=cls.gain))

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()
