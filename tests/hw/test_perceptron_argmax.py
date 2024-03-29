from collections import namedtuple
from functools import partial
from typing import ClassVar
import unittest
import torch
import hxtorch

from hxtorch_shared_test_tools import rand_full

hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.INFO)


class ArgMaxInput(namedtuple('ArgMaxInput', ["input", "dim", "keepdim"])):
    """
    An input to a argmax operation.
    """
    def duplicate(self):
        return self._make(arg for arg in self)


class TestArgMaxPyTorch(unittest.TestCase):
    """
    Tests the torch argmax operation.
    """

    argmax: ClassVar = torch.argmax

    def test_output_shape_value(self):
        """
        Compares the output shape and value of the argmax operation to the
        output of the torch implementation for different inputs.
        """

        test_inputs = {
            "1d dim=None":
            ArgMaxInput(rand_full((123), 20.).round(), dim=None, keepdim=False),
            "1d dim=0":
            ArgMaxInput(rand_full((123), 20.).round(), dim=0, keepdim=False),
            "1d dim=0 keepdim=True":
            ArgMaxInput(rand_full((123), 20.).round(), dim=0, keepdim=True),
            "2d dim=None":
            ArgMaxInput(rand_full((123, 456), 20.).round(), dim=None, keepdim=False),
            "2d dim=1":
            ArgMaxInput(rand_full((123, 456), 20.).round(), dim=1, keepdim=False),
            "2d dim=0":
            ArgMaxInput(rand_full((123, 456), 20.).round(), dim=0, keepdim=False),
            "2d dim=1 keepdim=True":
            ArgMaxInput(rand_full((123, 456), 20.).round(), dim=1, keepdim=True),
            "2d dim=0 keepdim=True":
            ArgMaxInput(rand_full((123, 456), 20.).round(), dim=0, keepdim=True),
            "3d dim=None":
            ArgMaxInput(rand_full((12, 45, 78), 20.).round(), dim=None, keepdim=False),
            "3d dim=2":
            ArgMaxInput(rand_full((12, 45, 78), 20.).round(), dim=2, keepdim=False),
            "3d dim=1":
            ArgMaxInput(rand_full((12, 45, 78), 20.).round(), dim=1, keepdim=False),
            "3d dim=0":
            ArgMaxInput(rand_full((12, 45, 78), 20.).round(), dim=0, keepdim=False),
            "3d dim=2 keepdim=True":
            ArgMaxInput(rand_full((12, 45, 78), 20.).round(), dim=2, keepdim=True),
            "3d dim=1 keepdim=True":
            ArgMaxInput(rand_full((12, 45, 78), 20.).round(), dim=1, keepdim=True),
            "3d dim=0 keepdim=True":
            ArgMaxInput(rand_full((12, 45, 78), 20.).round(), dim=0, keepdim=True),
        }

        for mode, argmax_input in test_inputs.items():
            with self.subTest(mode=mode):
                result = self.argmax(**argmax_input._asdict())

                argmax_input_torch = argmax_input.duplicate()
                result_torch = torch.argmax(**argmax_input_torch._asdict())
                self.assertTrue(
                    torch.equal(result, result_torch),
                    f"Result does not match:\n"
                    f"{result}\n!=\n{result_torch}")


class TestArgMaxHX(TestArgMaxPyTorch):
    """
    Tests the hxtorch argmax operation.
    """
    argmax: ClassVar = hxtorch.perceptron.argmax

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware(ann=True)

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()


class TestArgMaxHXmock(TestArgMaxPyTorch):
    """
    Tests the hxtorch argmax operation.
    """
    argmax: ClassVar = partial(hxtorch.perceptron.argmax, mock=True)


if __name__ == '__main__':
    unittest.main()
