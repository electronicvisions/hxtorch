"""
Define modules to decode SNN observables
"""
import torch


class MaxOverTime(torch.nn.Module):

    """ Simple max-over-time decoding """

    # pylint: disable=redefined-builtin, no-self-use
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
        max_traces, _ = torch.max(input, 1)
        return max_traces


class SumOverTime(torch.nn.Module):

    """ Simple sum-over-time decoding """

    # pylint: disable=redefined-builtin, no-self-use
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
        sum_traces = torch.sum(input, 1)
        return sum_traces


class MeanOverTime(torch.nn.Module):

    """ Simple mean-over-time decoding """

    # pylint: disable=redefined-builtin, no-self-use
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
        return torch.mean(input, 1)


class ToSpikeTimes(torch.autograd.Function):

    """ Convert spike input to the corresponding spike times, and provide
    backward functionality. If no spike is present, spike time is set to inf.
    """

    # Allow dt as name for time step size
    # Allow for own argument namings
    # pylint: disable=invalid-name, arguments-differ
    @staticmethod
    def forward(ctx, spike_input: torch.Tensor, spike_count: torch.Tensor,
                dt: float) -> torch.Tensor:
        """
        Return times of first spike_count spikes (if spike_count <
        spike_input.shape[0], i.e. its time dim length). If less than
        spike_count spikes happened along a output trace, its remaining
        entries are set to inf.

        TODO: maybe check that spike_input only holds 0s and 1s

        :param spike_input: spike tensor with 0s and 1s (indicating spikes),
            shape '(time_steps, batch_size, output_size)'
        :param spike_count: number of elements (ascending) to return for each
            output channel.
        :param dt: time step size

        :returns: Returns a dense tensor of shape
            '(spike_count, batch_size, output_size)'
        """
        indexed_spike_input = spike_input \
            * torch.arange(1, spike_input.shape[1] + 1)[None, :, None] - 1.0
        indexed_spike_input[indexed_spike_input == -1.0] = float("inf")
        if spike_count < spike_input.shape[1]:
            spike_indices = torch.sort(indexed_spike_input,
                                       dim=1).values[:, :spike_count]
        else:
            spike_indices = torch.sort(indexed_spike_input,
                                       dim=1).values[:, :spike_input.shape[1]]
        ctx.save_for_backward(spike_indices, spike_input)
        ctx.shape = spike_input.shape
        return spike_indices * dt

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Local gradient is set -1 for spike whose time was returned by forward,
        and 0 for no spike or if a spike's index wasn't returned in forward.
        """
        (spike_indices, spk_rec) = ctx.saved_tensors
        batch_size, spikeidcs_size, out_size = spike_indices.shape
        noninf_spike_indices = spike_indices.flatten() != float("inf")

        grad_input = torch.zeros_like(spk_rec, dtype=torch.float)

        grad_input_indices = (
            torch.arange(batch_size)
            .repeat_interleave(out_size)
            .repeat(spikeidcs_size)[noninf_spike_indices],
            spike_indices.flatten()[noninf_spike_indices].type(torch.long),
            torch.arange(out_size)
            .repeat(batch_size)
            .repeat(spikeidcs_size)[noninf_spike_indices],
        )

        grad_input[grad_input_indices] = \
            - 1.0 * grad_output.flatten()[noninf_spike_indices]

        return grad_input, None, None


class SpikesToTimesDecoder(torch.nn.Module):

    """ Convert spike train into spike times """

    # Allow dt as name for time step size
    # pylint: disable=invalid-name
    def __init__(self, spike_count: torch.Tensor = torch.as_tensor(1),
                 dt: float = 1e-3):
        """
        Module for spike to spike-time decoder. Decodes input spikes into spike
        times tensor with shape '(batch_size, spike_count, output_size)' in
        ascending order.
        Important Note: This uses autograd function ToSpikeTimes, which
        converts spike train z into spike times t_z and backpropagates the
        gradient w.r.t. the spike times (instead of z). This is needed for
        the EventProp backward function to compute the correct adjoints.
        :param spike_count: number of elements (ascending) to return for each
            output channel.
        :param dt: time step of input's time axis
        """
        super().__init__()
        self.spike_count = spike_count
        self.dt = dt
        self.func = ToSpikeTimes.apply

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        """Call decoder custom autograd function
        Parameters:
            spike_input: spike tensor with 0s and 1s (indicating spikes)
        """
        spike_indices = self.func(spike_input, self.spike_count, self.dt)
        return spike_indices
