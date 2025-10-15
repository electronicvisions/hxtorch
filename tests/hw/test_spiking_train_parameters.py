from typing import Optional
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import unittest
import torch
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import hxtorch
from hxtorch.spiking import Experiment, ModelParameter
from hxtorch.spiking.modules import LI
from hxtorch.spiking.handle import LIFObservables
from hxtorch.spiking.transforms import weight_transforms
from hxtorch.spiking.parameter import (
    TrainableHXTransformedModelParameter,
    MixedHXModelParameter
)
from hxtorch.core.utils import calib_helper

from dlens_vx_v3.hal import CapMemCell, NeuronConfig


@dataclass
class TestParameters:
    target_cap: float
    target_gl: float
    target_tau_syn: float
    weight_scale: float
    max_allowed_loss: float
    plot_path: Path
    use_conductance: bool
    start_cap: Optional[float] = None
    start_gl: Optional[float] = None
    start_tau_syn: Optional[float] = None


class Model(torch.nn.Module):
    """
    Model for training the time constants.
    """
    def __init__(self, test_parameters: TestParameters, mock: bool = False):
        super().__init__()
        self.dt = 1e-6
        self.experiment = Experiment(mock=mock, dt=self.dt)
        self.experiment.calibration = calib_helper.fixture_calibration_from_file(
            calib_helper.nightly_calib_path()
        )

        self.synapse = hxtorch.snn.Synapse(
            1,
            1,
            self.experiment,
            transform=partial(
                weight_transforms.linear_saturating,
                scale=test_parameters.weight_scale
            )
        )
        self.synapse.weight.requires_grad_(False)

        capacitance = MixedHXModelParameter(
            torch.tensor(test_parameters.target_cap),
            32 if test_parameters.start_cap else 63
        )

        g_l = ModelParameter(
            torch.tensor(test_parameters.target_gl)
        )

        tau_mem = ModelParameter(
            torch.tensor(
                test_parameters.target_cap / test_parameters.target_gl
            )
        )

        tau_syn = ModelParameter(
            torch.tensor(test_parameters.target_tau_syn),
        )
        self.neuron = LI(
            1, self.experiment,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            membrane_capacitance=capacitance,
            leak_conductance=g_l,
            leak=MixedHXModelParameter(0., 80),
            threshold=MixedHXModelParameter(0., 125),
            cadc_time_shift=-1,
            shift_cadc_to_first=True,
            trace_scale=1/45,
        )

    def forward(self, input):
        ret = self.neuron(self.synapse(input))
        hxtorch.snn.run(self.experiment, input.spikes.shape[0])
        return ret

    def set_start(self, test_parameters):

        if test_parameters.start_cap:
            def cap_transformer(cap):
                return torch.clamp(
                    (cap * 1e6 - CapConfig.offset) / CapConfig.divisor,
                    CapConfig.min_capacitance,
                    CapConfig.max_capacitance
                ).int()
            capacitance = TrainableHXTransformedModelParameter(
                test_parameters.start_cap,
                cap_transformer
            ).make_trainable(set_capacitance)
            self.neuron.membrane_capacitance = capacitance

        if test_parameters.start_tau_syn:
            tau_syn = TrainableHXTransformedModelParameter(
                test_parameters.start_tau_syn,
                SynTranslation.get_i_bias
            ).make_trainable(SynTranslation.set_tau_syn)
            self.neuron.tau_syn = tau_syn


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
            target_cap=5e-6,
            target_gl=1.,
            target_tau_syn=10e-6,
            start_cap=10e-6,
            start_tau_syn=6e-6,
            weight_scale=60.,
            use_conductance=False,
            max_allowed_loss=0.003,
            plot_path=self.plot_path.joinpath("./train_neuron_cap.png"))
        self.run_li_training(test_parameters, epochs=50, lr=1e-6)

    def run_li_training(self, test_parameters, epochs, lr):
        """ Test if parameters of an LI neuron can be trained """
        hxtorch.init_hardware()

        # Forward
        input_spikes = torch.bernoulli(
            torch.ones((150, 64, 1)) * 0.04)
        input_handle = LIFObservables(spikes=input_spikes)

        loss_fn = torch.nn.MSELoss()

        # Model
        model = Model(test_parameters)
        model.synapse.weight.data = torch.ones_like(
            model.synapse.weight.data) * 1

        with torch.no_grad():
            target = model(input_handle).membrane_cadc

        model.set_start(test_parameters)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        num_epochs = epochs
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=num_epochs//3, gamma=0.9)
        model.train()

        fig, (trace_plot, param_plot) = plt.subplots(2, 1)
        caps = [model.neuron.membrane_capacitance.model_value.clone().detach().numpy()]
        tau_syns = [model.neuron.tau_syn.model_value.clone().detach().numpy()]
        g_ls = [model.neuron.leak_conductance.model_value.clone().detach().numpy()]

        pbar = tqdm(total=num_epochs, unit="batch", leave=False)
        for i in range(num_epochs):
            model.zero_grad()
            model.apply(TimeConstantClipper())

            scores = model(input_handle)
            loss_b = loss_fn(scores.membrane_cadc, target)

            if i == 0:
                trace_plot.plot(
                    range(input_spikes.shape[0]),
                    torch.select(scores.membrane_cadc.detach(), 1, 0),
                    color="C3", alpha=0.4, label=str(i) + ' epochs')
            if i == num_epochs // 2:
                trace_plot.plot(
                    range(input_spikes.shape[0]),
                    torch.select(scores.membrane_cadc.detach(), 1, 0),
                    color="C3", alpha=0.65, label=str(i) + ' epochs')

            loss_b.backward()
            optimizer.step()
            scheduler.step()

            caps.append(model.neuron.membrane_capacitance.model_value.clone().detach().numpy())
            tau_syns.append(model.neuron.tau_syn.model_value.clone().detach().numpy())
            g_ls.append(model.neuron.leak_conductance.model_value.clone().detach().numpy())

            pbar.set_postfix(
                epoch=f"{i}", loss=f"{loss_b.item():.4f}")
            pbar.update()

        pbar.close()
        hxtorch.release_hardware()

        trace_plot.plot(
            range(input_spikes.shape[0]),
            torch.select(scores.membrane_cadc.detach(), 1, 0),
            color="C3", label=str(num_epochs-1) + ' epochs')
        trace_plot.plot(
            range(input_spikes.shape[0]),
            torch.select(target, 1, 0), '--',
            color="C0", label="target")
        trace_plot.legend()
        if test_parameters.start_cap:
            param_plot.axhline(test_parameters.target_cap, color="C0", ls='--')
            param_plot.plot(caps, color="C0", label=r'$C_m$')
        if test_parameters.start_tau_syn:
            param_plot.axhline(test_parameters.target_tau_syn, color="C1", ls='--')
            param_plot.plot(tau_syns, color="C1",label=r'$\tau_s$')
        if test_parameters.start_gl:
            conductance_plot = param_plot.twinx()
            conductance_plot.axhline(test_parameters.target_gl, color="C2", ls='--')
            conductance_plot.plot(g_ls, color="C2",label=r'$g_l$')
            conductance_plot.legend()
        param_plot.legend()
        plt.savefig(test_parameters.plot_path)
        plt.close()

        self.assertLess(loss_b.item(), test_parameters.max_allowed_loss)


class TimeConstantClipper(object):
    def __call__(self, module):
        if hasattr(module, 'tau_syn'):
            module.tau_syn.model_value.data.clamp_(3e-6, 100e-6)
        if hasattr(module, 'membrane_capacitance'):
            module.membrane_capacitance.model_value.data.clamp_(3e-6, 100e-6)


@dataclass
class CapConfig:
    offset: float = 0.80
    divisor: float = 0.13
    min_capacitance = 0
    max_capacitance = 63


def set_capacitance(capacitance, neuron_configs, neuron_coordinates):
    """
    Sets capacitance on the chip according to the membrane time constant
    using an ideal translation.
    """
    for idx, (configs, coords) in enumerate(
            zip(neuron_configs, neuron_coordinates)):
        for comp, an_configs in configs.items():
            for config in an_configs:
                if capacitance.ndim > 0:
                    config.membrane_capacitance.capacitance = (
                        NeuronConfig.MembraneCapacitorSize(
                            capacitance[idx].item(),
                        )
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


class SynTranslation:
    @staticmethod
    def get_i_bias(tau_syn):
        i_bias_exc = torch.clamp(
            (tau_syn * 1e6 * SynConfig.coefficent_exc)
            ** SynConfig.exponent_exc,
            SynConfig.min_cap_mem,
            SynConfig.max_cap_mem
        )
        i_bias_inh = torch.clamp(
            (tau_syn * 1e6 * SynConfig.coefficent_inh)
            ** SynConfig.exponent_inh,
            SynConfig.min_cap_mem,
            SynConfig.max_cap_mem
        )
        return (i_bias_exc, i_bias_inh)

    @staticmethod
    def set_tau_syn(tau_syn, neuron_configs, neuron_coordinates):
        i_bias_tau_exc, i_bias_tau_inh = tau_syn
        for idx, (configs, coords) in enumerate(
                zip(neuron_configs, neuron_coordinates)):
            for comp, an_configs in configs.items():
                for config in an_configs:
                    if i_bias_tau_inh.ndim > 0:
                        cap_mem_inh = CapMemCell.Value(
                            int(i_bias_tau_inh[idx].item())
                        )
                        cap_mem_exc = CapMemCell.Value(
                            int(i_bias_tau_exc[idx].item())
                        )
                    else:
                        cap_mem_inh = CapMemCell.Value(
                            int(i_bias_tau_inh.item())
                        )
                        cap_mem_exc = CapMemCell.Value(
                            int(i_bias_tau_exc.item())
                        )
                    config.inhibitory_input.i_bias_tau = cap_mem_inh
                    config.excitatory_input.i_bias_tau = cap_mem_exc


if __name__ == "__main__":
    unittest.main()
