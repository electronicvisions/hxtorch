from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial
from typing import ClassVar, Dict
import math
import unittest
import torch
import hxtorch
from hxtorch import logger

from hxtorch_shared_test_tools import rand_full

logger.default_config(level=logger.LogLevel.INFO)
logger.set_loglevel(logger.get("grenade"), logger.LogLevel.WARN)


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
    gain: ClassVar[float] = 1.

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
                result.backward(torch.ones_like(result))
                result_torch.backward(torch.ones_like(result_torch))

                for name, arg in conv_input._asdict().items():
                    if hasattr(arg, "grad"):
                        grad = arg.grad
                        grad_torch = getattr(conv_input_torch, name).grad
                        if name != "bias":
                            grad_torch *= self.gain
                        self.assertTrue(
                            torch.allclose(grad, grad_torch, rtol=.001),
                            f"{name.capitalize()} gradient does not match:\n"
                            f"{grad}\n!=\n{grad_torch}"
                            f"\ndiff:\n{grad - grad_torch}")


class TestConv1d(TestConv):
    """
    Tests the conv1d operation.
    """

    conv = torch.conv1d
    torch_conv = torch.conv1d

    test_inputs = {
        "batch1_outchannels1_inchannels1_kernel_larger_stride":
        ConvInput(rand_full((3, 1, 30), 25.), rand_full((1, 1, 5), 50.),
                  stride=7),
        "expanded_full_synram":
        ConvInput(rand_full((2, 1, 128), 10.), rand_full((14, 1, 43), 15.),
                  bias=torch.full((14,), 1.).requires_grad_(), stride=5),
        "expanded_overfull_synram":
        ConvInput(rand_full((2, 1, 138), 10.), rand_full((14, 1, 43), 15.),
                  bias=torch.full((14,), 1.).requires_grad_(), stride=5),
    }

    kernel_size = 5
    for n_batches in [2, 4]:
        for n_input_channels in [1, 3, 5]:
            for n_output_channels in [1, 4]:
                for stride in [7, 4, 2]:
                    test_inputs.update({
                        f"batch{n_batches}_outchannels{n_output_channels}_"
                        + f"inchannels{n_input_channels}_kernel{kernel_size}_"
                        + f"stride{stride}": ConvInput(
                            rand_full((n_batches, n_input_channels, 30), 10.),
                            rand_full((n_output_channels, n_input_channels,
                                       kernel_size), 50.),
                            stride=stride)})


class TestConv1dHX(TestConv1d):
    """
    Tests the conv1d operation on HX.
    """
    conv = hxtorch.conv1d

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()
        mock_parameter = hxtorch.measure_mock_parameter()
        hxtorch.set_mock_parameter(mock_parameter)
        cls.gain = mock_parameter.gain

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()


class TestConv1dHXmock(TestConv1d):
    """
    Tests the mocked conv1d operation.
    """
    conv = partial(hxtorch.conv1d, mock=True)

    @classmethod
    def setUpClass(cls):
        mock_parameter = hxtorch.MockParameter()
        hxtorch.set_mock_parameter(mock_parameter)
        cls.gain = mock_parameter.gain


class TestExpandedConv1d(TestConv1d):
    """
    Tests the conv1d operation.

    :cvar num_expansions: Number of expansions of the conv1d operation.
        Number of times the convolution kernel is placed side by side
        in the synapse matrix, shifted by the convolution's stride.
    """

    num_expansions = 18
    conv = partial(
        hxtorch.expanded_conv1d, num_expansions=num_expansions,
        num_sends=4, mock=True)

    @classmethod
    def setUpClass(cls):
        mock_parameter = hxtorch.MockParameter(gain=0.0015, noise_std=0)
        hxtorch.set_mock_parameter(mock_parameter)
        cls.gain = mock_parameter.gain * 4

    def test_compare_outputs(self):
        """
        Compares the outputs of the expanded conv1d operation to the outputs
        of the normal conv1d operation.
        """
        for mode in self.test_inputs:
            with self.subTest(mode=mode):
                conv_input = self.test_inputs[mode].duplicate()
                result_expanded = self.conv(**conv_input._asdict())
                result = hxtorch.conv1d(
                    num_sends=4, mock=True, **conv_input._asdict())

                # calculate how many synrams are filled with the given
                # operation, each will run as its own MAC operation and
                # will result in a rounding error up to 1.

                # size limitation in terms of height, caused by inputs:
                n_input_channels = self.test_inputs[mode].input.size()[1]
                kernel_size = self.test_inputs[mode].weight.size()[2]
                stride = self.test_inputs[mode].stride

                n_synapse_matrices = math.ceil(
                    ((n_input_channels * stride * self.num_expansions)
                     + kernel_size) / hxtorch.constants.hardware_matrix_height)

                # size limitation in terms of width, caused by outputs:
                n_output_channels = self.test_inputs[mode].weight.size()[0]
                n_synapse_matrices = max(
                    n_synapse_matrices,
                    math.ceil(n_output_channels
                              / hxtorch.constants.hardware_matrix_width))

                self.assertTrue(
                    torch.allclose(result_expanded, result,
                                   atol=n_synapse_matrices),
                    "Results do not match:\n"
                    f"{result_expanded}\n!=\n{result}")


class TestConv2d(TestConv):
    """
    Tests the conv2d operation.
    """
    conv = torch.conv2d
    torch_conv = torch.conv2d

    test_inputs = {
        "batch1_outchannels1_inchannels1_kernel_larger_stride":
        ConvInput(rand_full((1, 1, 30, 60), 25.), rand_full((1, 1, 5, 10), 20),
                  stride=(7, 14)),
        "batch2_outchannels1_inchannels3_kernel_larger_stride":
        ConvInput(rand_full((2, 3, 30, 60), 10.), rand_full((1, 3, 5, 10), 20),
                  stride=(7, 14)),
        "batch2_outchannels4_inchannels3_kernel_larger_stride":
        ConvInput(rand_full((2, 3, 30, 60), 10.), rand_full((4, 3, 5, 10), 20),
                  stride=(7, 14)),
        "batch1_outchannels1_inchannels1_kernel_smaller_stride":
        ConvInput(rand_full((1, 1, 30, 60), 25.), rand_full((1, 1, 5, 10), 20),
                  stride=(4, 8)),
        "batch2_outchannels1_inchannels3_kernel_smaller_stride":
        ConvInput(rand_full((2, 3, 30, 60), 10.), rand_full((1, 3, 5, 10), 20),
                  stride=(4, 8)),
        "batch2_outchannels4_inchannels3_kernel_smaller_stride":
        ConvInput(rand_full((2, 3, 30, 60), 10.), rand_full((4, 3, 5, 10), 20),
                  stride=(4, 8)),
        "batch2_outchannels4_inchannels3_kernel_smaller_stride":
        ConvInput(rand_full((2, 3, 30, 60), 10.), rand_full((4, 3, 5, 10), 20),
                  bias=torch.full((4,), 0.).requires_grad_(), stride=(4, 8))
    }


class TestConv2dHX(TestConv2d):
    """
    Tests the conv2d operation on HX.
    """
    conv = hxtorch.conv2d

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()
        mock_parameter = hxtorch.measure_mock_parameter()
        hxtorch.set_mock_parameter(mock_parameter)
        cls.gain = mock_parameter.gain

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()


class TestConv2dHXmock(TestConv2d):
    """
    Tests the mocked conv2d operation.
    """
    conv = partial(hxtorch.conv2d, mock=True)

    @classmethod
    def setUpClass(cls):
        mock_parameter = hxtorch.MockParameter()
        hxtorch.set_mock_parameter(mock_parameter)
        cls.gain = mock_parameter.gain


del TestConv  # remove abstract base class from tests

if __name__ == "__main__":
    unittest.main()
