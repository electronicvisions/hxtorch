"""
Define modules to decode SNN observables
"""
import torch


class MaxOverTime(torch.nn.Module):

    """ Simple max-over-time decoding """

    # pylint: disable=redefined-builtin
    # NOTE: - We redefine builtin as PyTorch does
    #       - We inherit for torch.nn.Module for consistency
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Translate an `input` tensor of shape (batch_size, time_length,
        population_size) into a tensor of shape (batch_size, population_size),
        where the time dimension is discarded by picking the maximum value
        along the time. Hence this module performs a 'max-over-time' operation.

        :param input: The input tensor to transform.
            expected shape: (batch_size, time_length, population_size)
        :return: Returns the tensor holding the max-over-time values.
        """
        return torch.amax(input, 0)


class SumOverTime(torch.nn.Module):

    """ Simple sum-over-time decoding """

    # pylint: disable=redefined-builtin
    # NOTE: - We redefine builtin as PyTorch does
    #       - We inherit for torch.nn.Module for consistency
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Translate an `input` tensor of shape (batch_size, time_length,
        population_size) into a tensor of shape (batch_size, population_size),
        where the time dimension is discarded by computing the sum along the
        time. Hence this module performs a 'sum-over-time' operation.

        :param input: The input tensor to transform.
            expected shape: (batch_size, time_length, population_size)
        :return: Returns the tensor holding the sum-over-time values.
        """
        return torch.sum(input, 0)


class MeanOverTime(torch.nn.Module):

    """ Simple mean-over-time decoding """

    # pylint: disable=redefined-builtin
    # NOTE: - We redefine builtin as PyTorch does
    #       - We inherit for torch.nn.Module for consistency
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Translate an `input` tensor of shape (batch_size, time_length,
        population_size) into a tensor of shape (batch_size, population_size),
        where the time dimension is discarded by computing the average value
        along the time. Hence this module performs a 'mean-over-time'
        operation.

        :param input: The input tensor to transform.
            expected shape: (batch_size, time_length, population_size)
        :return: Returns the tensor holding the mean-over-time values.
        """
        return torch.mean(input, 0)
