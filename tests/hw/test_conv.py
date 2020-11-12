from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial
from typing import ClassVar, Dict
import unittest
import torch
import hxtorch

from hxtorch_shared_test_tools import rand_full

hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.INFO)


class ConvInput(namedtuple('ConvInput', ["input", "weight", "bias", "stride"],
                           defaults=[None, 1])):
    """
    An input to a convolution operation.
    """
    def duplicate(self):
        return self._make(arg.data.clone().requires_grad_()
                          if hasattr(arg, "requires_grad") else arg
                          for arg in self)


class TestConv(ABC, unittest.TestCase):
    """
    Tests a conv operation.
    """

    @abstractmethod
    def conv(**kwargs):
        raise NotImplementedError

    @abstractmethod
    def torch_conv(**kwargs):
        raise NotImplementedError

    test_inputs: ClassVar[Dict[str, ConvInput]]

    def test_output_shape_gradient(self):
        """
        Compares the output shape and gradients of the operation to the output
        of the torch implementation for different input arguments.
        """
        log = hxtorch.logger.get(self.__class__.__name__)

        for mode in self.test_inputs:
            with self.subTest(mode=mode):
                conv_input = self.test_inputs[mode].duplicate()
                result = self.conv(**conv_input._asdict())
                log.info(f"Mean output: {result.mean():.1f}")

                self.assertTrue(result.is_contiguous())

                conv_input_torch = conv_input.duplicate()
                result_torch = self.torch_conv(**conv_input_torch._asdict())
                self.assertEqual(result.size(), result_torch.size())

                # compute gradients
                gain = torch.mean((result / result_torch)).item()
                log.info(f"Gain: {gain:.5f} (mean)")
                result.backward(torch.ones_like(result))
                result_torch.backward(torch.ones_like(result_torch))

                for name, arg in conv_input._asdict().items():
                    if hasattr(arg, "grad"):
                        grad = arg.grad
                        grad_torch = getattr(conv_input_torch, name).grad
                        if name != "bias":
                            grad_torch *= gain
                        self.assertTrue(
                            torch.allclose(grad, grad_torch, rtol=.1),
                            f"{name.capitalize()} gradient does not match:\n"
                            f"{grad}\n!=\n{grad_torch}")


class TestConv1d(TestConv):
    """
    Tests the conv1d operation.
    """
    conv = torch.conv1d
    torch_conv = torch.conv1d

    test_inputs = {
        "batch1_outchannels1_inchannels1_kernel_larger_stride":
        ConvInput(rand_full((3, 1, 30), 20.), rand_full((1, 1, 5), 50.),
                  stride=7),
        "batch2_outchannels1_inchannels3_kernel_larger_stride":
        ConvInput(rand_full((2, 3, 30), 20.), rand_full((1, 3, 5), 50.),
                  stride=7),
        "batch2_outchannels4_inchannels3_kernel_larger_stride":
        ConvInput(rand_full((2, 3, 30), 20.), rand_full((4, 3, 5), 50.),
                  stride=7),
        "batch1_outchannels1_inchannels1_kernel_smaller_stride":
        ConvInput(rand_full((3, 1, 30), 20.), rand_full((1, 1, 5), 50.),
                  stride=4),
        "batch2_outchannels1_inchannels3_kernel_smaller_stride":
        ConvInput(rand_full((2, 3, 30), 20.), rand_full((1, 3, 5), 50.),
                  stride=4),
        "batch2_outchannels4_inchannels3_kernel_smaller_stride":
        ConvInput(rand_full((2, 3, 30), 20.), rand_full((4, 3, 5), 50.),
                  stride=4),
        "batch2_outchannels4_inchannels3_bias":
        ConvInput(rand_full((2, 3, 30), 20.), rand_full((4, 3, 5), 50.),
                  bias=torch.full((4,), 0.).requires_grad_(), stride=4),
    }


class TestConv1dHX(TestConv1d):
    """
    Tests the conv1d operation on HX.
    """
    conv = partial(hxtorch.conv1d, num_sends=5, wait_between_events=10)

    @classmethod
    def setUpClass(cls):
        hxtorch.init()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release()


class TestConv1dHXmock(TestConv1d):
    """
    Tests the mocked conv1d operation.
    """
    conv = partial(hxtorch.conv1d, num_sends=5, mock=True)


class TestConv2d(TestConv):
    """
    Tests the conv2d operation.
    """
    conv = torch.conv2d
    torch_conv = torch.conv2d

    test_inputs = {
        "batch1_outchannels1_inchannels1_kernel_larger_stride":
        ConvInput(rand_full((1, 1, 30, 60), 20), rand_full((1, 1, 5, 10), 20),
                  stride=(7, 14)),
        "batch2_outchannels1_inchannels3_kernel_larger_stride":
        ConvInput(rand_full((2, 3, 30, 60), 20), rand_full((1, 3, 5, 10), 20),
                  stride=(7, 14)),
        "batch2_outchannels4_inchannels3_kernel_larger_stride":
        ConvInput(rand_full((2, 3, 30, 60), 20), rand_full((4, 3, 5, 10), 20),
                  stride=(7, 14)),
        "batch1_outchannels1_inchannels1_kernel_smaller_stride":
        ConvInput(rand_full((1, 1, 30, 60), 20), rand_full((1, 1, 5, 10), 20),
                  stride=(4, 8)),
        "batch2_outchannels1_inchannels3_kernel_smaller_stride":
        ConvInput(rand_full((2, 3, 30, 60), 20), rand_full((1, 3, 5, 10), 20),
                  stride=(4, 8)),
        "batch2_outchannels4_inchannels3_kernel_smaller_stride":
        ConvInput(rand_full((2, 3, 30, 60), 20), rand_full((4, 3, 5, 10), 20),
                  stride=(4, 8)),
        "batch2_outchannels4_inchannels3_kernel_smaller_stride":
        ConvInput(rand_full((2, 3, 30, 60), 20), rand_full((4, 3, 5, 10), 20),
                  bias=torch.full((4,), 0.).requires_grad_(), stride=(4, 8))
    }


class TestConv2dHX(TestConv2d):
    """
    Tests the conv2d operation on HX.
    """
    conv = partial(hxtorch.conv2d, num_sends=1, wait_between_events=10)

    @classmethod
    def setUpClass(cls):
        hxtorch.init()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release()


class TestConv2dHXmock(TestConv2d):
    """
    Tests the mocked conv2d operation.
    """
    conv = partial(hxtorch.conv2d, num_sends=1, mock=True)


del TestConv  # remove abstract base class from tests

if __name__ == "__main__":
    unittest.main()
