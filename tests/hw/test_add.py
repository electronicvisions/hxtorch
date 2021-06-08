from collections import namedtuple
from functools import partial
from typing import ClassVar
import unittest
import torch
import hxtorch

from hxtorch_shared_test_tools import rand_full

hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.INFO)


class AddInput(namedtuple('AddInput', ["input", "other"])):
    """
    An input to a add operation.
    """
    def duplicate(self):
        # clone necessary for independent gradient
        return self._make(arg.data.clone().requires_grad_()
                          if hasattr(arg, "requires_grad") else arg
                          for arg in self)


class TestAddPyTorch(unittest.TestCase):
    """
    Tests the torch add operation.
    """

    @staticmethod
    def torch_add(*args, **kwargs):
        return (kwargs["input"].clamp(-128., 127.) +
                kwargs["other"].clamp(-128., 127.)).clamp(-128., 127.)

    add: ClassVar = torch_add

    def test_output_shape_gradient(self):
        """
        Compares the output shape and gradients of the add operation to the
        output of the torch implementation for different input dimensions.
        """

        test_inputs = {
            "1-d broadcast":
            AddInput(rand_full((3, 128), 20.), rand_full((128,), 30.)),
            "1-d":
            AddInput(rand_full((128,), 20.), rand_full((128,), 30.)),
        }

        for mode, add_input in test_inputs.items():
            with self.subTest(mode=mode):
                result = self.add(**add_input._asdict())

                add_input_torch = add_input.duplicate()
                result_torch = self.torch_add(**add_input_torch._asdict())
                self.assertEqual(result.size(), result_torch.size())

                # compute gradients
                result.backward(torch.ones_like(result))
                result_torch.backward(torch.ones_like(result_torch))

                self.assertTrue(
                    torch.allclose(result, result_torch, atol=2.0),
                    f"Result does not match:\n"
                    f"{result}\n!=\n{result_torch}")

                for name, arg in add_input._asdict().items():
                    if hasattr(arg, "grad"):
                        grad = arg.grad
                        grad_torch = getattr(add_input_torch, name).grad
                        self.assertTrue(
                            torch.allclose(grad, grad_torch, rtol=0.1),
                            f"{name.capitalize()} gradient does not match:\n"
                            f"{grad}\n!=\n{grad_torch}")


class TestAddHX(TestAddPyTorch):
    """
    Tests the hxtorch add operation.
    """
    add: ClassVar = hxtorch.add

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()


class TestAddHXmock(TestAddPyTorch):
    """
    Tests the hxtorch add operation.
    """
    add: ClassVar = partial(hxtorch.add, mock=True)

    @classmethod
    def setUpClass(cls):
        hxtorch.init(hxtorch.MockParameter())


if __name__ == '__main__':
    unittest.main()
