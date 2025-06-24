from typing import Callable, Optional
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import unittest
import torch
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import hxtorch
from hxtorch.spiking import Experiment
from hxtorch.spiking.modules import ReadoutNeuronExp
from hxtorch.spiking.handle import LIFObservables
from hxtorch.spiking.transforms import weight_transforms
from hxtorch.spiking.parameter import (
    HXTransformedModelParameter, MixedHXModelParameter)
from dlens_vx_v3 import lola, halco
from dlens_vx_v3.hal import CapMemCell, NeuronConfig


@dataclass
class TestParameters:
    target_tau_mem: float
    target_tau_syn: float
    tau_mem_translation: Callable
    i_synin_gm: float
    synapse_dac_bias: float
    weight_scale: float
    trace_scale: float
    max_allowed_loss: float
    plot_path: Path
    use_conductance: bool
    start_tau_mem: Optional[float] = None
    start_tau_syn: Optional[float] = None


class LIConductanceNeuron(ReadoutNeuronExp):
    """
    A specialized LI neuron that scales inputs during simulation to reflect
    hardware conductance.
    This neuron model scales the incoming inputs in simulation according to the
    originally calibrated leak conductance for correct hardware modeling.
     """

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self, size: int,
                 experiment: Experiment,
                 leak=80,
                 tau_mem=10e-6,
                 tau_syn=10e-6,
                 i_synin_gm=500,
                 membrane_capacitance=63,
                 synapse_dac_bias=600,
                 execution_instance=None,
                 enable_cadc_recording=True,
                 enable_cadc_recording_placement_in_dram=False,
                 enable_madc_recording=False,
                 record_neuron_id=None,
                 placement_constraint=None,
                 trace_offset=0.,
                 trace_scale=1.,
                 cadc_time_shift=0, shift_cadc_to_first=False,
                 interpolation_mode="linear",
                 neuron_structure=None,
                 **extra_params) -> None:
        super().__init__(
            size,
            experiment,
            leak=leak,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            i_synin_gm=i_synin_gm,
            membrane_capacitance=membrane_capacitance,
            synapse_dac_bias=synapse_dac_bias,
            execution_instance=execution_instance,
            enable_cadc_recording=enable_cadc_recording,
            enable_cadc_recording_placement_in_dram=(
                enable_cadc_recording_placement_in_dram
            ),
            enable_madc_recording=enable_madc_recording,
            record_neuron_id=record_neuron_id,
            placement_constraint=placement_constraint,
            trace_offset=trace_offset, trace_scale=trace_scale,
            cadc_time_shift=cadc_time_shift,
            shift_cadc_to_first=shift_cadc_to_first,
            interpolation_mode=interpolation_mode,
            neuron_structure=neuron_structure,
            **extra_params)
        self.tau_mem_0 = self.tau_mem.hardware_value.clone().detach()

    def forward_func(self, *input, hw_data=None):
        for handle in input:
            handle.graded_spikes *= self.tau_mem.hardware_value / self.tau_mem_0
        return super().forward_func(*input, hw_data=hw_data)


class ConductanceModel(torch.nn.Module):
    """
    Model for training the membrane time constant with the leak conductance.
    """
    def __init__(self, test_parameters: TestParameters, mock: bool = False):
        super().__init__()
        self.dt = 1e-6
        self.experiment = Experiment(mock=mock, dt=self.dt)

        tau_mem = HXTransformedModelParameter(
            torch.exp(torch.ones(1) * -self.dt/test_parameters.target_tau_mem),
            lambda x: -self.dt/torch.log(x)
        )
        tau_syn = HXTransformedModelParameter(
            torch.exp(torch.ones(1) * -self.dt/test_parameters.target_tau_syn),
            lambda x: -self.dt / torch.log(x)
        )
        self.module1 = hxtorch.snn.Synapse(
            1, 1, self.experiment,
            transform=partial(
                weight_transforms.linear_saturating,
                scale=test_parameters.weight_scale)
        )
        self.module1.weight.requires_grad_(False)
        self.module2 = LIConductanceNeuron(
            1, self.experiment,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            leak=MixedHXModelParameter(0., 80),
            threshold=MixedHXModelParameter(0., 125),
            cadc_time_shift=-1,
            shift_cadc_to_first=True,
            trace_scale=test_parameters.trace_scale)

    def forward(self, input):
        ret = self.module2(self.module1(input))
        hxtorch.snn.run(self.experiment, input.spikes.shape[0])
        return ret

    def set_start(self, test_parameters):
        """ Initializes the model for the start of the training """

        if test_parameters.start_tau_mem:
            self.module2.tau_mem.model_value = torch.exp(
                torch.ones(1) * -self.dt/test_parameters.start_tau_mem
            )
            self.module2.tau_mem.make_trainable(
                test_parameters.tau_mem_translation
            )
        if test_parameters.start_tau_syn:
            self.module2.tau_syn.model_value = torch.exp(
                torch.ones(1) * -self.dt/test_parameters.start_tau_syn
            )
            self.module2.tau_syn.make_trainable(set_tau_syn)


class Model(torch.nn.Module):
    """
    Model for training the membrane time constant with the capacitance.
    """
    def __init__(self, test_parameters: TestParameters, mock: bool = False):
        super().__init__()
        self.dt = 1e-6
        self.experiment = Experiment(mock=mock, dt=self.dt)

        tau_mem = HXTransformedModelParameter(
            torch.exp(torch.ones(1) * -self.dt/test_parameters.target_tau_mem),
            lambda x: -self.dt/torch.log(x)
        )
        tau_syn = HXTransformedModelParameter(
            torch.exp(torch.ones(1) * -self.dt/test_parameters.target_tau_syn),
            lambda x: -self.dt / torch.log(x)
        )
        self.module1 = hxtorch.snn.Synapse(
            1, 1, self.experiment,
            transform=partial(
                weight_transforms.linear_saturating,
                scale=test_parameters.weight_scale
            )
        )
        self.module1.weight.requires_grad_(False)
        self.module2 = ReadoutNeuronExp(
            1, self.experiment,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            leak=MixedHXModelParameter(0., 80),
            threshold=MixedHXModelParameter(0., 125),
            i_synin_gm=test_parameters.i_synin_gm,
            synapse_dac_bias=test_parameters.synapse_dac_bias,
            membrane_capacitance=32,
            cadc_time_shift=-1,
            shift_cadc_to_first=True,
            trace_scale=test_parameters.trace_scale)

    def forward(self, input):
        ret = self.module2(self.module1(input))
        hxtorch.snn.run(self.experiment, input.spikes.shape[0])
        return ret

    def set_start(self, test_parameters):
        if test_parameters.start_tau_mem:
            tau_mem = HXTransformedModelParameter(
                torch.exp(torch.ones(1) * -self.dt/test_parameters.start_tau_mem),
                lambda x: -self.dt/torch.log(x)
            ).make_trainable(test_parameters.tau_mem_translation)
            self.module2.tau_mem = tau_mem
        if test_parameters.start_tau_syn:
            tau_syn = HXTransformedModelParameter(
                torch.exp(torch.ones(1) * -self.dt/test_parameters.start_tau_syn),
                lambda x: -self.dt / torch.log(x)
            ).make_trainable(set_tau_syn)
            self.module2.tau_syn = tau_syn


class TestTrainingParameters(unittest.TestCase):
    """
    Test class for training parameters. A target neuron trace is recorded using
    a calibration.
    Starting from values that are different from the calibration target values,
    the model is trained to replicate the target neuron trace.
    """

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_train_with_capacitance(self):
        """
        Test training of synaptic time constant and membrane time constant
        on a neuron trace.
        The membrane time constant is set with the capacitance on hardware.
        """
        test_parameters = TestParameters(
            target_tau_mem=25e-6,
            target_tau_syn=10e-6,
            start_tau_mem=8e-6,
            start_tau_syn=20e-6,
            tau_mem_translation=set_tau_mem_cap,
            i_synin_gm=60,
            synapse_dac_bias=1000,
            weight_scale=33,
            trace_scale=1/45,
            use_conductance=False,
            max_allowed_loss=0.002,
            plot_path=self.plot_path.joinpath("./train_neuron_cap.png"))
        self.run_li_training(test_parameters, epochs=80)

    def test_li_conductance(self):
        """
        Test training of synaptic time constant and membrane time constant
        on a neuron trace.
        The membrane time constant is set with the leak conductance
        on hardware.
        """
        test_parameters = TestParameters(
            target_tau_mem=10e-6,
            target_tau_syn=10e-6,
            start_tau_mem=20e-6,
            start_tau_syn=15e-6,
            tau_mem_translation=set_tau_mem_conductance,
            i_synin_gm=500,
            synapse_dac_bias=600,
            weight_scale=60,
            trace_scale=1/45,
            use_conductance=True,
            max_allowed_loss=0.004,
            plot_path=self.plot_path.joinpath("./train_neuron_conductance.png"))
        self.run_li_training(test_parameters, epochs=80)

    def run_li_training(self, test_parameters, epochs):
        """ Test if parameters of an LI neuron can be trained """
        hxtorch.init_hardware()

        # Forward
        input_spikes = torch.bernoulli(
            torch.ones((150, 150, 1)) * 0.04)
        input_handle = LIFObservables(spikes=input_spikes)

        loss_fn = torch.nn.MSELoss()

        # Model
        model = Model(test_parameters)
        if test_parameters.use_conductance:
            model = ConductanceModel(test_parameters)
        model.module1.weight.data = torch.ones_like(
            model.module1.weight.data) * 1

        with torch.no_grad():
            target = model(input_handle).membrane_cadc

        model.set_start(test_parameters)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        num_epochs = epochs
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=num_epochs//3, gamma=0.8)
        model.train()

        pbar = tqdm(total=num_epochs, unit="batch", leave=False)
        for i in range(num_epochs):
            model.zero_grad()

            scores = model(input_handle)
            loss_b = loss_fn(scores.membrane_cadc, target)

            if i == 0:
                plt.plot(
                    range(input_spikes.shape[0]),
                    torch.select(scores.membrane_cadc.detach(), 1, 0),
                    color="C3", alpha=0.4, label=str(i) + ' epochs')
            if i == num_epochs // 2:
                plt.plot(
                    range(input_spikes.shape[0]),
                    torch.select(scores.membrane_cadc.detach(), 1, 0),
                    color="C3", alpha=0.65, label=str(i) + ' epochs')

            loss_b.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(
                epoch=f"{i}", loss=f"{loss_b.item():.4f}")
            pbar.update()

        pbar.close()
        hxtorch.release_hardware()

        plt.plot(
            range(input_spikes.shape[0]),
            torch.select(scores.membrane_cadc.detach(), 1, 0),
            color="C3", label=str(num_epochs-1) + ' epochs')
        plt.plot(
            range(input_spikes.shape[0]),
            torch.select(target, 1, 0), '--',
            color="C0", label="target")
        plt.legend()
        plt.savefig(test_parameters.plot_path)
        plt.close()

        self.assertLess(loss_b.item(), test_parameters.max_allowed_loss)


@dataclass
class LeakBiasConfig:
    coefficent: float = 1 / 242
    coefficent_div: float = 1 / 851
    exponent: float = -1 / 0.72
    exponent_div: float = -1 / 0.59
    enable_div: float = 20e-6
    min_cap_mem = 10
    max_cap_mem = 1020


def set_tau_mem_conductance(tau_mem, chip, neuron_coordinates):
    """
    Sets leak conductance on the chip according to the membrane time constant
    using an ideal translation.
    """
    enable_division = tau_mem > LeakBiasConfig.enable_div
    leak_i_bias = torch.clamp(
        (tau_mem * 1e6 * LeakBiasConfig.coefficent)
        ** LeakBiasConfig.exponent,
        LeakBiasConfig.min_cap_mem,
        LeakBiasConfig.max_cap_mem
    )
    leak_i_bias[enable_division] = torch.clamp(
        (tau_mem * 1e6 * LeakBiasConfig.coefficent_div)
        ** LeakBiasConfig.exponent_div,
        LeakBiasConfig.min_cap_mem,
        LeakBiasConfig.max_cap_mem
    )
    for idx, coord in enumerate(neuron_coordinates):
        atomic_neuron_coord = coord.get_atomic_neurons()[0]
        config = chip.neuron_block.atomic_neurons[atomic_neuron_coord]
        if leak_i_bias.shape:
            config.leak.i_bias = CapMemCell.Value(
                int(leak_i_bias[idx].item())
            )
            config.leak.enable_division = enable_division[idx]
        else:
            config.leak.i_bias = CapMemCell.Value(
                int(leak_i_bias.item())
            )
            config.leak.enable_division = enable_division.item()


@dataclass
class CapConfigLong:
    offset: float = 4.64
    divisor: float = 0.734
    min_capacitance = 0
    max_capacitance = 63


def set_tau_mem_cap(tau_mem, chip, neuron_coordinates):
    """
    Sets capacitance on the chip according to the membrane time constant
    using an ideal translation.
    """
    capacitance = torch.clamp(
        (tau_mem * 1e6 - CapConfigLong.offset) / CapConfigLong.divisor,
        CapConfigLong.min_capacitance,
        CapConfigLong.max_capacitance
    ).int()
    for idx, coord in enumerate(neuron_coordinates):
        atomic_neuron_coord = coord.get_atomic_neurons()[0]
        config = chip.neuron_block.atomic_neurons[atomic_neuron_coord]
        if capacitance.shape:
            config.membrane_capacitance.capacitance = (
                NeuronConfig.MembraneCapacitorSize(capacitance[idx].item())
            )
        else:
            config.membrane_capacitance.capacitance = (
                NeuronConfig.MembraneCapacitorSize(capacitance.item())
            )


@dataclass
class SynConfig:
    coefficent_inh: float = 1 / 979
    coefficent_exc: float = 1 / 998
    exponent_inh: float = -1 / 0.968
    exponent_exc: float = -1 / 0.953
    min_cap_mem = 10
    max_cap_mem = 1020


def set_tau_syn(tau_syn, chip, neuron_coordinates):
    """
    Sets i_bias_tau on the chip according to the synpatic time constant
    using an ideal translation.
    """
    i_bias_tau_inh = torch.clamp(
        (tau_syn * 1e6 * SynConfig.coefficent_inh)
        ** SynConfig.exponent_inh,
        SynConfig.min_cap_mem,
        SynConfig.max_cap_mem
    )
    i_bias_tau_exc = torch.clamp(
        (tau_syn * 1e6 * SynConfig.coefficent_exc)
        ** SynConfig.exponent_exc,
        SynConfig.min_cap_mem,
        SynConfig.max_cap_mem
    )
    for idx, coord in enumerate(neuron_coordinates):
        atomic_neuron_coord = coord.get_atomic_neurons()[0]
        config = chip.neuron_block.atomic_neurons[atomic_neuron_coord]
        if i_bias_tau_inh.shape:
            cap_mem_inh = CapMemCell.Value(int(i_bias_tau_inh[idx].item()))
            cap_mem_exc = CapMemCell.Value(int(i_bias_tau_exc[idx].item()))
        else:
            cap_mem_inh = CapMemCell.Value(int(i_bias_tau_inh.item()))
            cap_mem_exc = CapMemCell.Value(int(i_bias_tau_exc[idx].item()))
        config.inhibitory_input.i_bias_tau = cap_mem_inh
        config.excitatory_input.i_bias_tau = cap_mem_exc


if __name__ == "__main__":
    unittest.main()
