from collections import namedtuple
from functools import partial
from typing import ClassVar
import unittest
import torch
import hxtorch

from hxtorch_shared_test_tools import rand_full

hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.INFO)


class ReLUInput(namedtuple('ReLUInput', ["input"])):
    """
    An input to a relu operation.
    """
    def duplicate(self):
        return self._make(arg.data.clone().requires_grad_()
                          if hasattr(arg, "requires_grad") else arg
                          for arg in self)


class TestReLUPyTorch(unittest.TestCase):
    """
    Tests the torch relu operation.
    """

    relu: ClassVar = torch.relu

    def test_output_shape_gradient(self):
        """
        Compares the output shape and gradients of the relu operation to the
        output of the torch implementation for different input dimensions.
        """

        test_inputs = {
            "1-d signed":
            ReLUInput(torch.arange(-63., 65.).requires_grad_()),
            "1-d":
            ReLUInput(rand_full((128,), 20.)),
            "2-d":
            ReLUInput(rand_full((3, 128), 20.)),
            "3-d":
            ReLUInput(rand_full((2, 3, 128), 20.)),
            "2-d non-contiguous input":
            ReLUInput(rand_full((128, 3), 20.).data.t().requires_grad_()),
        }

        for mode, relu_input in test_inputs.items():
            with self.subTest(mode=mode):
                result = self.relu(**relu_input._asdict())

                relu_input_torch = relu_input.duplicate()
                result_torch = torch.relu(**relu_input_torch._asdict())
                self.assertEqual(result.size(), result_torch.size())

                # compute gradients
                result.backward(torch.ones_like(result))
                result_torch.backward(torch.ones_like(result_torch))

                self.assertTrue(
                    torch.allclose(result, result_torch, atol=1.0),
                    f"Result does not match:\n"
                    f"{result}\n!=\n{result_torch}")

                for name, arg in relu_input._asdict().items():
                    if hasattr(arg, "grad"):
                        grad = arg.grad
                        grad_torch = getattr(relu_input_torch, name).grad
                        self.assertTrue(
                            torch.allclose(grad, grad_torch, rtol=0.1),
                            f"{name.capitalize()} gradient does not match:\n"
                            f"{grad}\n!=\n{grad_torch}")


class TestReLUHX(TestReLUPyTorch):
    """
    Tests the hxtorch relu operation.
    """
    relu: ClassVar = hxtorch.relu

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()


class TestReLUHXmock(TestReLUPyTorch):
    """
    Tests the hxtorch relu operation.
    """
    relu: ClassVar = partial(hxtorch.relu, mock=True)


if __name__ == '__main__':
    unittest.main()
