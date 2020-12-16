from abc import abstractmethod
from typing import Callable, Tuple, Union, Optional
import torch
import _hxtorch
import pylogging as logger
import pyhaldls_vx_v2 as hal


def scale_input(x_in: torch.Tensor) -> torch.Tensor:
    """
    Scales the tensor to the maximal input range of the chip.
    """
    max_in = torch.max(x_in)
    factor = hal.PADIEvent.HagenActivation.max / max_in if max_in > 0 else 1
    return x_in * factor


def scale_weight(weight: torch.Tensor) -> torch.Tensor:
    """
    Scales the tensor to the maximal weight matrix range of HX.
    """
    max_in = torch.max(torch.abs(weight))
    factor = hal.SynapseQuad.Weight.max / max_in if max_in > 0 else 1
    return weight * factor


class Layer:
    """
    Base class of all layers in :mod:`hxtorch.nn`.
    """

    def __init__(self, num_sends: int = 1, wait_between_events: int = 5,
                 mock: bool = False, *,
                 input_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = None,
                 weight_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = scale_weight):
        """
        :param num_sends: Number of sends of the input. Values greater than 1
            result in higher output to the neurons and increases the s/n ratio.
            Defaults to ``1``.
        :param wait_between_events: Wait time between two successive vector
            inputs, in FPGA clock cycles. Defaults to ``5``.
            Shorter wait time can lead to saturation of the synaptic input.
        :param mock: Enable mock mode.
        :param input_transform: Function that receives the input and returns
            a tensor to be used as input to the chip.
        :param weight_transform: Function that receives the weight and returns
            a tensor to be used as weight matrix on the chip.
        """
        self.num_sends = num_sends
        self.wait_between_events = wait_between_events
        self.mock = mock
        self.input_transform = input_transform
        self.weight_transform = weight_transform

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}("
        if hasattr(self, "extra_repr"):
            repr_str += f"{self.extra_repr()}, "
        if self.input_transform:
            repr_str += f"input_transform={self.input_transform}, "
        if self.weight_transform:
            repr_str += f"weight_transform={self.weight_transform}, "
        repr_str += f"num_sends={self.num_sends}, "
        repr_str += f"wait_between_events={self.wait_between_events}, "
        repr_str += f"mock={self.mock})"
        return repr_str


class Linear(Layer, torch.nn.Linear):
    """
    Applies a linear transformation to the incoming data on Hicann-X.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 num_sends: int = 1, wait_between_events: int = 5,
                 mock: bool = False, *,
                 input_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = None,
                 weight_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = scale_weight):
        """
        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param bias: If set to `True`, the layer will learn an additive bias.
        :param num_sends: Number of sends of the input. Values greater than 1
            result in higher output to the neurons and increases the s/n ratio.
            Defaults to ``1``.
        :param wait_between_events: Wait time between two successive vector
            inputs, in FPGA clock cycles. Defaults to ``5``.
            Shorter wait time can lead to saturation of the synaptic input.
        :param mock: Enable mock mode.
        :param input_transform: Function that receives the input and returns
            a tensor to be used as input to the chip.
        :param weight_transform: Function that receives the weight and returns
            a tensor to be used as weight matrix on the chip.
        """
        if bias:
            logger.get(f"{__name__}.{self.__class__.__name__}").warn(
                "The bias will not be scaled along with the output, "
                "this may lead to unexpected results.")

        torch.nn.Linear.__init__(self, in_features, out_features, bias)
        Layer.__init__(
            self, num_sends, wait_between_events, mock,
            input_transform=input_transform, weight_transform=weight_transform)
        self._matmul = _hxtorch.matmul

    def forward(self, input):  # pylint: disable=redefined-builtin
        log = logger.get(__name__)
        log.debug(f"linear.forward:\tinput.shape {input.shape}\t"
                  f"weight.shape {self.weight.shape}")
        weight, bias = self.weight, self.bias
        if self.weight_transform is not None:
            weight = self.weight_transform(weight)
        if self.input_transform is not None:
            input = self.input_transform(input)
        output = self._matmul(input, weight.t(),
                              num_sends=self.num_sends,
                              wait_between_events=self.wait_between_events,
                              mock=self.mock)
        torch.set_printoptions(profile="full")
        log.debug(f"out:\n{output.to(int)}\n"
                  f"out.shape {output.shape}\tout.max {output.max()}")
        torch.set_printoptions(profile="default")
        if bias is not None:
            output = _hxtorch.add(output, bias, mock=self.mock)
        return output


class ConvNd(Layer, torch.nn.modules.conv._ConvNd):  # pylint: disable=protected-access
    """
    Base class for n-dimensional convolution.
    """

    @abstractmethod
    def _conv(self, x: torch.Tensor, weight: torch.Tensor,
              stride: Tuple[int, ...], num_sends: int) -> torch.Tensor:
        """
        Implementation of convolution function.
        """
        raise NotImplementedError

    def forward(self, input):  # pylint: disable=redefined-builtin,arguments-differ
        if self.dilation not in ((1,), (1, 1)):
            raise ValueError(
                f"Dilations greater than 1 are currently not supported.")
        if self.groups > 1:
            raise ValueError(f"More than 1 group is currently not supported.")

        if any(self.padding):
            expanded_padding = list()
            for padding_element in self.padding[::-1]:
                expanded_padding.append(padding_element)
                expanded_padding.append(padding_element)
            if self.padding_mode == "zeros":
                padding_mode = "constant"
            else:
                padding_mode = self.padding_mode
            input = torch.nn.functional.pad(
                input, tuple(expanded_padding), padding_mode)

        log = logger.get(__name__)
        log.debug(f"ConvNd.forward:\tinput.shape {input.shape}\t"
                  f"weight.shape {self.weight.shape}")
        weight, bias = self.weight, self.bias
        if self.weight_transform is not None:
            weight = self.weight_transform(weight)
        if self.input_transform is not None:
            input = self.input_transform(input)
        output = self._conv(input, weight, bias,
                            self.stride, num_sends=self.num_sends,
                            wait_between_events=self.wait_between_events,
                            mock=self.mock)
        torch.set_printoptions(profile="full")
        log.debug(f"out[0, 0]:\n{output[0, 0].to(int)}\n"
                  f"out.shape {output.shape}\tout.max {output.max()}")
        torch.set_printoptions(profile="default")
        return output


class Conv1d(ConvNd, torch.nn.Conv1d):
    """
    Applies a 1D convolution over an input signal composed of several input
    planes.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: int = 1, padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple] = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = 'zeros',
                 num_sends: int = 1, wait_between_events: int = 5,
                 mock: bool = False, *,
                 input_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = None,
                 weight_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = scale_weight):
        """
        :param in_channels: Number of channels in the input
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution
        :param padding: Zero-padding added to both sides of the input
        :param padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'
        :param dilation: Spacing between kernel elements
        :param groups: Number of blocked connections from input channels to
            output channels
        :param bias: If ``True``, adds a learnable bias to the output
        :param num_sends: Number of sends of the input. Values greater than 1
            result in higher output to the neurons and increases the s/n ratio.
            Defaults to ``1``.
        :param wait_between_events: Wait time between two successive vector
            inputs, in FPGA clock cycles. Defaults to ``5``.
            Shorter wait time can lead to saturation of the synaptic input.
        :param mock: Enable mock mode.
        :param input_transform: Function that receives the input and returns
            a tensor to be used as input to the chip.
        :param weight_transform: Function that receives the weight and returns
            a tensor to be used as weight matrix on the chip.
        """
        if bias:
            logger.get(f"{__name__}.{self.__class__.__name__}").warn(
                "The bias will not be scaled along with the output, "
                "this may lead to unexpected results.")

        torch.nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size,
                                 stride, padding, dilation, groups, bias,
                                 padding_mode)
        ConvNd.__init__(
            self, num_sends, wait_between_events, mock,
            input_transform=input_transform, weight_transform=weight_transform)
        self._conv = _hxtorch.conv1d


class Conv2d(ConvNd, torch.nn.Conv2d):
    """
    Applies a 2D convolution over an input image composed of several input
    planes.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1, padding: Union[int, Tuple[int, int]] = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', num_sends: int = 1,
                 wait_between_events: int = 5, mock: bool = False, *,
                 input_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = None,
                 weight_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = scale_weight):
        """
        :param in_channels: Number of channels in the input
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution
        :param padding: Zero-padding added to both sides of the input
        :param padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'
        :param dilation: Spacing between kernel elements
        :param groups: Number of blocked connections from input channels to
            output channels
        :param bias: If ``True``, adds a learnable bias to the output
        :param num_sends: Number of sends of the input. Values greater than 1
            result in higher output to the neurons and increases the s/n ratio.
            Defaults to ``1``.
        :param mock: Enable mock mode.
        :param wait_between_events: Wait time between two successive vector
            inputs, in FPGA clock cycles. Defaults to ``5``.
            Shorter wait time can lead to saturation of the synaptic input.
        :param input_transform: Function that receives the input and returns
            a tensor to be used as input to the chip.
        :param weight_transform: Function that receives the weight and returns
            a tensor to be used as weight matrix on the chip.
        """
        if bias:
            logger.get(f"{__name__}.{self.__class__.__name__}").warn(
                "The bias will not be scaled along with the output, "
                "this may lead to unexpected results.")

        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size,
                                 stride, padding, dilation, groups, bias,
                                 padding_mode)
        ConvNd.__init__(
            self, num_sends, wait_between_events, mock,
            input_transform=input_transform, weight_transform=weight_transform)
        self._conv = _hxtorch.conv2d


class ReLU(torch.nn.ReLU):
    """
    Applies a rectified linear unit to the input.
    """

    def __init__(self, mock: bool = False):
        """
        :param mock: Enable mock mode
        """
        super().__init__()
        self.mock = mock

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _hxtorch.relu(input, mock=self.mock)

    def extra_repr(self) -> str:
        return f"mock={self.mock}"


class ConvertingReLU(ReLU):
    """
    Applies a rectified linear unit to the input, shifts and clips to the
    input range of the chip.
    """

    def __init__(self, shift: int = 2, mock: bool = False):
        """
        :param shift: Number of bits the result is shifted by
        :param mock: Enable mock mode
        """
        super().__init__(mock)
        self.shift = shift

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _hxtorch.converting_relu(
            input, shift=self.shift, mock=self.mock)

    def extra_repr(self) -> str:
        return f"shift={self.shift}, {super().extra_repr()}"
