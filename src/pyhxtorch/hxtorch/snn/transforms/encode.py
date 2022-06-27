"""
Define transformations for spiking input coding
"""
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
