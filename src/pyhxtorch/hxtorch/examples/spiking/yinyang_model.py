"""
Model class for spiking HX torch yinyang example
"""
from typing import Optional, Tuple
from functools import partial
import torch

from dlens_vx_v3 import halco

import hxtorch.spiking as hxsnn
from hxtorch.spiking.transforms import weight_transforms
from hxtorch.spiking.utils import calib_helper
import hxtorch.spiking.functional as F


class SNN(torch.nn.Module):
    """
    SNN with one hidden LIF layer and one readout LI layer
    """
    # pylint: disable=too-many-arguments, invalid-name

    def __init__(self, n_in: int, n_hidden: int, n_out: int, mock: bool,
                 calib_path: str, dt: float = 1.0e-6, tau_mem: float = 8e-6,
                 tau_syn: float = 8e-6, alpha: float = 50,
                 trace_shift_hidden: int = 0, trace_shift_out: int = 0,
                 weight_init_hidden: Optional[Tuple[float, float]] = None,
                 weight_init_output: Optional[Tuple[float, float]] = None,
                 weight_scale: float = 1., trace_scale: float = 1.,
                 input_repetitions: int = 1,
                 device: torch.device = torch.device("cpu")) -> None:
        """
        Initialize the SNN.

        :param n_in: Number of input units.
        :param n_hidden: Number of hidden units.
        :param n_out: Number of output units.
        :param mock: Indicating whether to train in software or on hardware.
        :param calib_path: Path to hardware calibration file.
        :param dt: Time-binning width.
        :param tau_mem: Membrane time constant.
        :param tau_syn: Synaptic time constant.
        :param trace_shift_hidden: Indicates how many indices the membrane
            trace of hidden layer is shifted to left along time axis.
        :param trace_shift_out: Indicates how many indices the membrane
            trace of readout layer is shifted to left along time axis.
        :param weight_init_hidden: Hidden layer weight initialization mean
            and std value.
        :param weight_init_output: Output layer weight initialization mean
            and std value.
        :param weight_scale: The factor with which the software weights are
            scaled when mapped to hardware.
        :param input_repetitions: Number of times to repeat input channels.
        :param device: The used PyTorch device used for tensor operations in
            software.
        """
        super().__init__()

        # Neuron parameters
        lif_params = F.CUBALIFParams(
            tau_mem=tau_mem, tau_syn=tau_syn, alpha=alpha)
        li_params = F.CUBALIParams(tau_mem=tau_mem, tau_syn=tau_syn)

        # Experiment instance to work on
        self.exp = hxsnn.Experiment(
            mock=mock, dt=dt)
        if not mock:
            self.exp.default_execution_instance.load_calib(
                calib_path if calib_path
                else calib_helper.nightly_calix_native_path("spiking2"))

        # Repeat input
        self.input_repetitions = input_repetitions
        # Input projection
        self.linear_h = hxsnn.Synapse(
            n_in * input_repetitions, n_hidden, experiment=self.exp,
            transform=partial(
                weight_transforms.linear_saturating, scale=weight_scale))
        # Initialize weights
        if weight_init_hidden:
            w = torch.zeros(n_hidden, n_in)
            torch.nn.init.normal_(w, *weight_init_hidden)
            self.linear_h.weight.data = w.repeat(1, input_repetitions)

        # Hidden layer
        self.lif_h = hxsnn.Neuron(
            n_hidden, experiment=self.exp, func=F.cuba_lif_integration,
            params=lif_params, trace_scale=trace_scale,
            cadc_time_shift=trace_shift_hidden, shift_cadc_to_first=True)

        # Output projection
        self.linear_o = hxsnn.Synapse(
            n_hidden, n_out, experiment=self.exp,
            transform=partial(
                weight_transforms.linear_saturating, scale=weight_scale))

        # Readout layer
        self.li_readout = hxsnn.ReadoutNeuron(
            n_out, experiment=self.exp, func=F.cuba_li_integration,
            params=li_params, trace_scale=trace_scale,
            cadc_time_shift=trace_shift_out, shift_cadc_to_first=True,
            placement_constraint=list(
                halco.LogicalNeuronOnDLS(
                    hxsnn.morphology.SingleCompartmentNeuron(1).compartments,
                    halco.AtomicNeuronOnDLS(
                        halco.NeuronRowOnDLS(1), halco.NeuronColumnOnDLS(nrn)))
                for nrn in range(n_out)))
        # Initialize weights
        if weight_init_output:
            torch.nn.init.normal_(self.linear_o.weight, *weight_init_output)

        # Device
        self.device = device
        self.to(device)

        # placeholder for hidden spikes
        self.s_h = None

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward path.

        :param spikes: NeuronHandle holding spikes as input.

        :return: Returns the output of the network, i.e. membrane traces of the
            readout neurons.
        """
        # Increase synapse strength by repeating each input
        spikes = spikes.repeat(1, 1, self.input_repetitions)

        # Spike input handle
        spikes_handle = hxsnn.NeuronHandle(spikes)

        # Forward
        c_h = self.linear_h(spikes_handle)
        self.s_h = self.lif_h(c_h)  # Keep spikes for fire reg.
        c_o = self.linear_o(self.s_h)
        y_o = self.li_readout(c_o)

        # Execute on hardware
        hxsnn.run(self.exp, spikes.shape[0])

        return y_o.v_cadc


class Model(torch.nn.Module):

    """ Complete model with encoder, network (snn) and decoder """

    def __init__(self, encoder: torch.nn.Module,
                 network: torch.nn.Module,
                 decoder: torch.nn.Module,
                 readout_scale: float = 1.) -> None:
        """
        Initialize the model by assigning encoder, network and decoder

        :param encoder: Module to encode input data
        :param network: Network module containing layers and
            parameters / weights
        :param decoder: Module to decode network output
        """
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.decoder = decoder

        self.readout_scale = readout_scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through whole model, i.e.
        data -> encoder -> network -> decoder -> output

        :param inputs: tensor input data

        :returns: Returns tensor output
        """
        spikes = self.encoder(inputs)
        traces = self.network(spikes)
        self.scores = self.decoder(traces).clone()

        # scale outputs
        with torch.no_grad():
            self.scores *= self.readout_scale

        return self.scores

    def regularize(self, reg_readout: float, reg_bursts: float,
                   reg_w_hidden: float, reg_w_output: float) -> torch.Tensor:
        """
        Get regularization terms for bursts and weights like
        factor * (thing to be regularized) ** 2.

        :param reg_bursts: prefactor of burst / hidden spike regulaization
        :param reg_weights_hidden: prefactor of hidden weight regularization
        :param reg_weights_output: prefactor of output weight regularization

        :returns: Returns sum of regularization terms
        """
        reg = torch.tensor(0., device=self.scores.device)

        # Reg readout
        reg += reg_readout * torch.mean(self.scores ** 2)

        # bursts (hidden spikes) regularization
        reg += reg_bursts * torch.mean(
            torch.sum(self.network.s_h.spikes, dim=0) ** 2.)
        # weight regularization
        reg += reg_w_hidden * torch.mean(self.network.linear_h.weight ** 2.)
        reg += reg_w_output * torch.mean(self.network.linear_o.weight ** 2.)

        return reg
