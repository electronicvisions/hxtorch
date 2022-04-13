"""
Define transformations for spiking input coding
"""
from typing import Optional
import torch


class PixelsToSpikeTimes(torch.nn.Module):

    """ Encode image by spike trains """

    def __init__(self, tau=20, threshold=0.2, t_max=1.0, epsilon=1e-7) -> None:
        """
        Initialize a pixel to spike time transformation.

        :param tau: Streching factor of pixel to spike-time transformation.
            Lager values correspond to later spike times.
        :param threshold: The threshold under which a pixel is considered
            non-spiking (resp. spike at time t_max).
        :param t_max: Maximum spike time.
        :param epsilon: Safty margin to prevent zero-division.
        """
        super().__init__()

        self._tau = tau
        self._threshold = threshold
        self._t_max = t_max
        self._epsilon = epsilon

    def forward(self, pic: torch.Tensor) -> torch.Tensor:
        """
        Turns an image into a spike representation. Image pixels >= threshold
        are assigned a spike time according to:

            t_spike = \tau * log(pixel / (pixel - threshold))

        If a pixel is < threshold it gets assigned the spike time t_max.

        :param pic: Image to transform into spike trains. Has to be of shape
            (color_channel, width, height).

        :return: Returns the image represented by spike trains, given as a
            tensor of shape (spike_time, color_channel, width, height).
        """
        non_spiking = pic < self._threshold
        pic = torch.clip(pic, self._threshold + self._epsilon, 1e9)

        times = self._tau * torch.log(pic / (pic - self._threshold))
        times[non_spiking] = self._t_max
        times = torch.clamp(times, 0, self._t_max)

        return times


class SpikeTimesToSparseTensor(torch.nn.Module):

    """ Convert spike times to a dense matrix of zeros and ones """

    def __init__(self, bin_width: float, size: int = 100) -> None:
        """
        Initialize the conversion of spike times to a dense matrix of zeros and
        ones.

        :param bin_width: Binning interval in seconds.
        :param size: number of bins along time axis.
        """
        super().__init__()

        self._time_step = bin_width
        self._size = size

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Convert spike times to dense matrix of zeros and ones.

        :param spikes: Spike times of shape '(color_channels, x0[, x1, ...])'.

        :returns: Returns a dense matirx of shape
            '(time_bins, color_channels, x0[, x1, ...])'.
        """
        dev = spikes.device

        bins = (spikes / self._time_step).long()
        mask = bins < self._size  # kill all events later than size
        mesh = torch.meshgrid([torch.arange(s) for s in spikes.shape])

        indices = torch.stack(
            (mesh[0].to(dev)[mask].reshape(-1),
             bins.to(dev)[mask].reshape(-1),
             *(mesh[i].to(dev)[mask].reshape(-1)
               for i in range(1, len(mesh)))))

        sparse_spikes = torch.sparse_coo_tensor(
            indices, torch.ones(indices.shape[1]).to(dev),
            (spikes.shape[0], self._size, *spikes.shape[1:]), dtype=int)

        return sparse_spikes


class SpikeTimesToDense(torch.nn.Module):

    """ Convert spike times to a dense matrix of zeros and ones """

    def __init__(self, bin_width: float, size: int = 100) -> None:
        """
        Initialize the conversion of spike times to a dense matrix of zeros and
        ones.

        :param bin_width: Binning interval in seconds.
        :param size: number of bins along time axis.
        """
        super().__init__()

        self._to_sparse = SpikeTimesToSparseTensor(bin_width, size)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Convert spike times to dense matrix of zeros and ones.

        :param spikes: Spike times of shape '(color_channels, x0[, x1, ...])'.

        :returns: Returns a dense matirx of shape
            '(time_bins, color_channels, x0[, x1, ...])'.
        """
        return self._to_sparse(spikes).to_dense()


class CoordinatesToSpikes(torch.nn.Module):

    """ Convert values between 0 and 1 to spikes in a given timeframe """

    # Allow dt for naming of time step width
    # pylint: disable=invalid-name, too-many-arguments
    def __init__(self, seq_length: int, t_early: float, t_late: float,
                 dt: float = 1.e-6, t_bias: Optional[float] = None,
                 device: torch.device = torch.device("cpu")) -> None:
        """
        Construct a coordinates-to-spikes converter. This converter takes
        coordinate values in [0, 1] and maps them to spike times in
        interval [t_early, t_late - t_early]. A spike is indicated by a
        1 on a dense time axis of length seq_length with temporal resolution
        dt. Further, it adds a bias spike at time t_bias, if t_bias is not
        None.

        :param seq_length: Number of time steps in the resulting time sequence.
            The effective time length is given by seq_length * dt.
        :param t_early: The earliest time a spike will occur.
        :param t_late: The latest time a spike will occur.
        :param dt: The temporal resolution.
        :param t_bias: The time a bias spike occurs.
        """
        super().__init__()
        self._seq_length = seq_length
        self._t_early = t_early
        self._t_late = t_late
        self._t_bias = t_bias
        self._dt = dt
        self._dev = device
        self.to(device)

    def forward(self, coordinate_values: torch.Tensor) -> torch.Tensor:
        """
        Convert coordinate values of to dense spike tensor of zeros and ones.

        :param coordinate_values: Tensor with values in [0, 1], shaped
            (batch_size, num_channels)

        :returns: Returns a dense tensor of shape (batch_size, seq_length,
            num_channels)
        """
        times = self._t_early + coordinate_values * (
            self._t_late - self._t_early)
        batch_size, num_channels = times.shape
        indices = torch.stack((
            torch.arange(times.shape[0]).repeat_interleave(times.shape[1]).to(self._dev),
            (times.flatten() / self._dt).round(),
            torch.arange(times.shape[1]).repeat(times.shape[0]).to(self._dev)
        )).type(torch.long)

        if not isinstance(times, torch.Tensor):
            ones = torch.ones_like(
                torch.tensor(times)).flatten().type(torch.float)
        else:
            ones = torch.ones_like(times).flatten().type(torch.float)

        if self._t_bias is None:
            spikes = torch.sparse_coo_tensor(
                indices, ones, (batch_size, self._seq_length, num_channels))
            spikes = spikes.to_dense()
        else:
            spikes = torch.sparse_coo_tensor(
                indices, ones, (
                    batch_size, self._seq_length, num_channels + 1))
            spikes = spikes.to_dense()
            bias_idx = int(self._t_bias / self._dt)
            spikes[:, bias_idx, -1] = 1.

        return spikes
