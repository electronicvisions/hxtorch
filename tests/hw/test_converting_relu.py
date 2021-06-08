from collections import namedtuple
from functools import partial
from typing import ClassVar
import unittest
import torch
import hxtorch

from hxtorch_shared_test_tools import rand_full

hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.INFO)


class ConvertingReLUInput(namedtuple('ConvertingReLUInput', ["input"])):
    """
    An input to a converting_relu operation.
    """
    def duplicate(self):
        return self._make(arg.data.clone().requires_grad_()
                          if hasattr(arg, "requires_grad") else arg
                          for arg in self)


class TestConvertingReLUPyTorch(unittest.TestCase):
    """
    Tests the torch converting_relu operation.
    """

    @staticmethod
    def torch_converting_relu(*args, **kwargs):
        return torch.div(torch.relu(*args, **kwargs), 4)

    converting_relu: ClassVar = torch_converting_relu

    def test_output_shape_gradient(self):
        """
        Compares the output shape and gradients of the converting_relu operation to the
        output of the torch implementation for different input dimensions.
        """

        test_inputs = {
            "1-d signed":
            ConvertingReLUInput(torch.arange(-63., 65.).requires_grad_()),
            "1-d":
            ConvertingReLUInput(rand_full((128,), 20.)),
            "2-d":
            ConvertingReLUInput(rand_full((3, 128), 20.)),
            "3-d":
            ConvertingReLUInput(rand_full((2, 3, 128), 20.)),
            "2-d non-contiguous input":
            ConvertingReLUInput(rand_full((128, 3), 20.).data.t().requires_grad_()),
        }

        for mode, converting_relu_input in test_inputs.items():
            with self.subTest(mode=mode):
                result = self.converting_relu(**converting_relu_input._asdict())

                converting_relu_input_torch = converting_relu_input.duplicate()
                result_torch = torch.div(torch.relu(**converting_relu_input_torch._asdict()), 4)
                self.assertEqual(result.size(), result_torch.size())

                # compute gradients
                result.backward(torch.ones_like(result))
                result_torch.backward(torch.ones_like(result_torch))

                self.assertTrue(
                    torch.allclose(result, result_torch, atol=1.0),
                    f"Result does not match:\n"
                    f"{result}\n!=\n{result_torch}")

                self.assertTrue(
                    torch.all(result <= 31.),
                    f"Result not smaller equal 31:\n"
                    f"{result}")

                for name, arg in converting_relu_input._asdict().items():
                    if hasattr(arg, "grad"):
                        grad = arg.grad
                        grad_torch = getattr(converting_relu_input_torch, name).grad
                        self.assertTrue(
                            torch.allclose(grad, grad_torch, rtol=0.1),
                            f"{name.capitalize()} gradient does not match:\n"
                            f"{grad}\n!=\n{grad_torch}")


class TestConvertingReLUHX(TestConvertingReLUPyTorch):
    """
    Tests the hxtorch converting_relu operation.
    """
    converting_relu: ClassVar = hxtorch.converting_relu

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()


class TestConvertingReLUHXmock(TestConvertingReLUPyTorch):
    """
    Tests the hxtorch converting_relu operation.
    """
    converting_relu: ClassVar = partial(hxtorch.converting_relu, mock=True)


if __name__ == '__main__':
    unittest.main()
