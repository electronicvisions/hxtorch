"""
This module contains layers that can be used in modules together
with the building blocks from py:mod:`torch.nn`. Unlike their counterparts,
their multiply-accumulate operations are performed with the
BrainScaleS-2 accelerator. Additional digital operations are performed
in the SIMD processors of BSS-2.
"""
from abc import abstractmethod
from inspect import signature
import math
from numbers import Integral, Real
from typing import Callable, Tuple, Union, Optional
import torch
import _hxtorch
from _hxtorch.constants import defaults  # pylint: disable=import-error


def scale_input(x_in: torch.Tensor) -> torch.Tensor:
    """
    Scales the tensor to the maximal input range of BrainScaleS-2.
    """
    max_in = torch.max(x_in)
    factor = \
        _hxtorch.constants.input_activation_max / max_in if max_in > 0 else 1
    return x_in * factor


def scale_weight(weight: torch.Tensor) -> torch.Tensor:
    """
    Scales the tensor to the maximal weight range of BrainScaleS-2.
    """
    max_in = torch.max(torch.abs(weight))
    factor = \
        _hxtorch.constants.synaptic_weight_max / max_in if max_in > 0 else 1
    return weight * factor


def clamp_weight_(weight: torch.Tensor) -> torch.Tensor:
    """
    Clamps all elements of the weight in-place into the maximal weight range
    of BrainScaleS-2.
    """
    max_weight = _hxtorch.constants.synaptic_weight_max
    with torch.no_grad():
        torch.clamp(weight, -max_weight, max_weight, out=weight)
    return weight


class Layer:
    """
    Base class of all layers in :mod:`hxtorch.nn`.
    """

    def __init__(self, mock: bool = False):
        """
        :param mock: Enable mock mode.
        """
        self.mock = mock

    def __repr__(self):
        repr_str = ""
        # get params from signature and include only non-default values:
        for name, param in signature(type(self)).parameters.items():
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                value = len(value) > 0
            if value is not param.default:
                repr_str += f"{name}={value}, "
        return f"{self.__class__.__name__}({repr_str[:-2]})"


class MACLayer(Layer):
    """
    Layer that performs a multiply accumulate operation.
    """

    def __init__(self, num_sends: Optional[Integral] = None,
                 wait_between_events: Integral = defaults.wait_between_events,
                 mock: bool = False, *,
                 input_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = None,
                 weight_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = clamp_weight_):
        """
        :param num_sends: Number of sends of the input. Values greater than 1
            result in higher output to the neurons and increases the s/n ratio.
            For ``None`` this is automatically adjusted during initialization.
        :param wait_between_events: Wait time between two successive vector
            inputs, in FPGA clock cycles.
            Shorter wait time can lead to saturation of the synaptic input.
        :param mock: Enable mock mode.
        :param input_transform: Function that receives the input and returns
            a tensor to be used as input to the chip.
        :param weight_transform: Function that receives the weight and returns
            a tensor to be used as weight matrix on the chip.
        """
        super().__init__(mock=mock)
        self.num_sends = num_sends
        self.wait_between_events = wait_between_events
        self.input_transform = input_transform
        self.weight_transform = weight_transform

    def reset_parameters(self, weight_mean: Real = 0.,
                         relu_shift: Integral = 1) -> None:
        """
        Reset parameters to reasonable initialization values. Method based on
        *Delving deep into rectifiers: Surpassing human-level performance on
        ImageNet classification* - He, K. et al. (2015)

        :param weight_mean: Mean value of the weight distribution
        :param relu_shift: Bit shift assumed in subsequent ConvertingReLU
        """
        fan_in = self.weight[0].numel()  # pylint: disable=no-member

        gain_relu = math.sqrt(2)
        gain_mac = _hxtorch.get_mock_parameter().gain
        gain_scale = pow(2, relu_shift)
        gain_tot = gain_scale * gain_relu / gain_mac
        std = gain_tot / math.sqrt(fan_in)

        if self.num_sends is None:
            self.num_sends = int(
                math.ceil(std / (_hxtorch.constants.synaptic_weight_max / 3)))
        std /= self.num_sends

        torch.nn.init.trunc_normal_(
            self.weight,  # pylint: disable=no-member
            mean=weight_mean,
            std=std,
            a=_hxtorch.constants.synaptic_weight_min,
            b=_hxtorch.constants.synaptic_weight_max
        )

        if self.bias is not None:  # pylint: disable=no-member
            # estimated standard deviation of the input
            std_in = _hxtorch.constants.input_activation_max / math.sqrt(3.)
            bound = gain_tot / math.sqrt(fan_in) / std_in
            with torch.no_grad():
                self.bias.uniform_(-bound, bound)  # pylint: disable=no-member


class Linear(MACLayer, torch.nn.Linear):
    """
    Applies a linear transformation to the incoming data on Hicann-X.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, in_features: Integral, out_features: Integral,
                 bias: bool = True, num_sends: Optional[Integral] = None,
                 wait_between_events: Integral = defaults.wait_between_events,
                 mock: bool = False, *,
                 avg: Integral = 1,
                 input_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = None,
                 weight_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = clamp_weight_):
        """
        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param bias: If set to `True`, the layer will learn an additive bias.
        :param num_sends: Number of sends of the input. Values greater than 1
            result in higher output to the neurons and increases the s/n ratio.
            For ``None`` this is automatically adjusted during initialization.
        :param wait_between_events: Wait time between two successive vector
            inputs, in FPGA clock cycles.
            Shorter wait time can lead to saturation of the synaptic input.
        :param mock: Enable mock mode.
        :param avg: Number of neurons to average over. This option is targeted
            at reducing statistical noise.
            Beware: We average over different fixed-pattern instances, but they
            are all configured at the same weight, so they are not trained
            individually. This could potentially have negative implications.
        :param input_transform: Function that receives the input and returns
            a tensor to be used as input to the chip.
        :param weight_transform: Function that receives the weight and returns
            a tensor to be used as weight matrix on the chip.
        """
        # super().__init__() would be nicer, but impossible due to different
        # parameters
        MACLayer.__init__(
            self, num_sends, wait_between_events, mock,
            input_transform=input_transform, weight_transform=weight_transform)
        torch.nn.Linear.__init__(self, in_features, out_features, bias)
        self._matmul = _hxtorch.matmul
        self.avg = avg

    # Allow redefinition of builtin in order to maintain PyTorch style
    def forward(self, input):  # pylint: disable=redefined-builtin
        weight, bias = self.weight, self.bias
        if self.weight_transform is not None:
            weight = self.weight_transform(weight)
        if self.input_transform is not None:
            input = self.input_transform(input)
        output = self._matmul(input,
                              weight.t().repeat_interleave(self.avg, -1),
                              num_sends=self.num_sends,
                              wait_between_events=self.wait_between_events,
                              mock=self.mock)
        if self.avg > 1:
            output = torch.nn.functional.avg_pool1d(
                output.unsqueeze(-2), self.avg).squeeze(-2)
        if bias is not None:
            output = _hxtorch.add(output, bias, mock=self.mock)
        return output


class ConvNd(MACLayer, torch.nn.modules.conv._ConvNd):  # pylint: disable=protected-access
    """
    Base class for n-dimensional convolution.
    """

    @abstractmethod
    def _conv(self, input: torch.Tensor, weight: torch.Tensor,  # pylint: disable=redefined-builtin
              bias: torch.Tensor, stride: Tuple[Integral, ...],
              **kwargs) -> torch.Tensor:
        """
        Implementation of convolution function.
        """
        raise NotImplementedError

    # Allow redefinition of builtin in order to maintain PyTorch style
    def forward(self, input):  # pylint: disable=redefined-builtin,arguments-differ
        if self.dilation not in ((1,), (1, 1)):
            raise ValueError(
                "Dilations greater than 1 are currently not supported.")
        if self.groups > 1:
            raise ValueError("More than 1 group is currently not supported.")

        if any(self.padding):
            expanded_padding = []
            for padding_element in self.padding[::-1]:
                expanded_padding.append(padding_element)
                expanded_padding.append(padding_element)
            if self.padding_mode == "zeros":
                padding_mode = "constant"
            else:
                padding_mode = self.padding_mode
            input = torch.nn.functional.pad(
                input, tuple(expanded_padding), padding_mode)

        weight, bias = self.weight, self.bias
        if self.weight_transform is not None:
            weight = self.weight_transform(weight)
        if self.input_transform is not None:
            input = self.input_transform(input)
        output = self._conv(input, weight, bias,
                            self.stride, num_sends=self.num_sends,
                            wait_between_events=self.wait_between_events,
                            mock=self.mock)
        return output


class Conv1d(ConvNd, torch.nn.Conv1d):  # pylint: disable=abstract-method # Issue 3983
    """
    Applies a 1D convolution over an input signal composed of several input
    planes.
    """

    # pylint: disable=too-many-arguments, super-init-not-called
    def __init__(self, in_channels: Integral, out_channels: Integral,
                 kernel_size: Union[Integral, Tuple[Integral]],
                 stride: Integral = 1,
                 padding: Union[Integral, Tuple[Integral, Integral]] = 0,
                 dilation: Union[Integral, Tuple] = 1, groups: Integral = 1,
                 bias: bool = True, padding_mode: str = 'zeros',
                 num_sends: Optional[Integral] = None,
                 wait_between_events: Integral = defaults.wait_between_events,
                 mock: bool = False, *,
                 input_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = None,
                 weight_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = clamp_weight_):
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
            For ``None`` this is automatically adjusted during initialization.
        :param wait_between_events: Wait time between two successive vector
            inputs, in FPGA clock cycles.
            Shorter wait time can lead to saturation of the synaptic input.
        :param mock: Enable mock mode.
        :param input_transform: Function that receives the input and returns
            a tensor to be used as input to the chip.
        :param weight_transform: Function that receives the weight and returns
            a tensor to be used as weight matrix on the chip.
        """
        # super().__init__() would be nicer, but impossible due to different
        # parameters
        MACLayer.__init__(  # pylint: disable=non-parent-init-called
            self, num_sends, wait_between_events, mock,
            input_transform=input_transform, weight_transform=weight_transform)
        torch.nn.Conv1d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode)
        self._conv = _hxtorch.conv1d


class ExpandedConv1d(Conv1d):
    """
    Unrolls the weight matrix for execution on hardware.
    This maximizes the use of the synapses array.

    Caveat:
    Fixed-pattern noise cannot be individually compensated for during
    training, because the same weights are used at different locations!
    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: int = 1, padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple] = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = 'zeros',
                 num_sends: Optional[int] = None,
                 wait_between_events: int = defaults.wait_between_events,
                 mock: bool = False, *,
                 input_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = None,
                 weight_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = clamp_weight_,
                 num_expansions: Optional[int] = None):
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
            For ``None`` this is automatically adjusted during initialization.
        :param wait_between_events: Wait time between two successive vector
            inputs, in FPGA clock cycles.
            Shorter wait time can lead to saturation of the synaptic input.
        :param mock: Enable mock mode.
        :param input_transform: Function that receives the input and returns
            a tensor to be used as input to the chip.
        :param weight_transform: Function that receives the weight and returns
            a tensor to be used as weight matrix on the chip.
        :param num_expansions: Number of enrolled kernels in a single operation
        """
        Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode,
                        num_sends, wait_between_events, mock,
                        input_transform=input_transform,
                        weight_transform=weight_transform)

        if num_expansions is not None:
            self.num_expansions = num_expansions
        else:
            max_out_channels = _hxtorch.constants.hardware_matrix_width
            max_kernel_size = _hxtorch.constants.hardware_matrix_height
            max_num_width = max_out_channels // self.out_channels
            max_num_height = (max_kernel_size - self.kernel_size[0]) \
                // self.stride[0] + 1
            self.num_expansions = min(max_num_width, max_num_height)

    def _conv(self, *args, **kwargs) -> torch.Tensor:
        return _hxtorch.expanded_conv1d(*args, **kwargs,
                                        num_expansions=self.num_expansions)

    def extra_repr(self) -> str:
        return f"num_expansions={self.num_expansions}, {super().extra_repr()}"


class Conv2d(ConvNd, torch.nn.Conv2d):  # pylint: disable=abstract-method # Issue 3983
    """
    Applies a 2D convolution over an input image composed of several input
    planes.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, in_channels: Integral, out_channels: Integral,
                 kernel_size: Union[Integral, Tuple[Integral, Integral]],
                 stride: Integral = 1,
                 padding: Union[Integral, Tuple[Integral, Integral]] = 0,
                 dilation: Integral = 1, groups: Integral = 1,
                 bias: bool = True, padding_mode: str = 'zeros',
                 num_sends: Optional[Integral] = None,
                 wait_between_events: Integral = defaults.wait_between_events,
                 mock: bool = False, *,
                 input_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = None,
                 weight_transform: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = clamp_weight_):
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
            For ``None`` this is automatically adjusted during initialization.
        :param mock: Enable mock mode.
        :param wait_between_events: Wait time between two successive vector
            inputs, in FPGA clock cycles.
            Shorter wait time can lead to saturation of the synaptic input.
        :param input_transform: Function that receives the input and returns
            a tensor to be used as input to the chip.
        :param weight_transform: Function that receives the weight and returns
            a tensor to be used as weight matrix on the chip.
        """
        # super().__init__() would be nicer, but impossible due to different
        # parameters
        ConvNd.__init__(
            self, num_sends, wait_between_events, mock,
            input_transform=input_transform, weight_transform=weight_transform)
        torch.nn.Conv2d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode)
        self._conv = _hxtorch.conv2d


class ReLU(Layer, torch.nn.ReLU):
    """
    Applies a rectified linear unit to the input.
    """

    def __init__(self, mock: bool = False):
        """
        :param mock: Enable mock mode
        """
        # super().__init__() would be nicer, but impossible due to different
        # parameters
        Layer.__init__(self, mock)
        torch.nn.ReLU.__init__(self)

    # Allow redefinition of builtin in order to maintain PyTorch style
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        output = _hxtorch.relu(input, mock=self.mock)
        return output


class ConvertingReLU(ReLU):
    """
    Applies a rectified linear unit to the input, shifts and clips to the
    input range of the chip.
    """

    def __init__(self, shift: Integral = 2, mock: bool = False):
        """
        :param shift: Number of bits the result is shifted by
        :param mock: Enable mock mode
        """
        super().__init__(mock)
        self.shift = shift

    # Allow redefinition of builtin in order to maintain PyTorch style
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        output = _hxtorch.converting_relu(
            input, shift=self.shift, mock=self.mock)
        return output
