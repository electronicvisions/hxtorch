import unittest
import torch

from pathlib import Path
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from dataclasses import dataclass

import hxtorch
from hxtorch.spiking import Experiment
from hxtorch.spiking.modules import LI
from hxtorch.spiking.handle import LIFObservables
from hxtorch.spiking.parameter import TrainableModelParameter


@dataclass
class ParameterSet:
    target_cap: float
    target_tau_syn: float
    plot_path: Path
    start_cap: float = None
    start_tau_syn: float = None


class Model(torch.nn.Module):
    def __init__(self, test_parameters, is_target=False):
        super().__init__()
        dt = 1e-6
        self.experiment = Experiment(mock=True, dt=dt)

        tau_mem = TrainableModelParameter(
            torch.tensor(test_parameters.target_cap)
        )
        tau_syn = TrainableModelParameter(
            torch.tensor(test_parameters.target_tau_syn)
        )

        if test_parameters.start_cap and not is_target:
            tau_mem = TrainableModelParameter(
                torch.tensor(test_parameters.start_cap)
            )
        if test_parameters.start_tau_syn and not is_target:
            tau_syn = TrainableModelParameter(
                torch.tensor(test_parameters.start_tau_syn)
            )

        self.synapse = hxtorch.snn.Synapse(
            1, 1, self.experiment)
        self.synapse.weight.requires_grad_(False)
        self.neuron = LI(
            1, self.experiment,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            membrane_capacitance=tau_mem,
            leak=TrainableModelParameter(torch.tensor(0.))
        )

    def forward(self, input):
        ret = self.neuron(self.synapse(input))
        hxtorch.snn.run(self.experiment, input.spikes.shape[0])
        return ret


class TestTranslationCapacitance(unittest.TestCase):

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_li_single(self):
        test_parameters = ParameterSet(
            target_cap=15e-6,
            target_tau_syn=10e-6,
            start_cap=4e-6,
            start_tau_syn=20e-6,
            plot_path=self.plot_path.joinpath("./train_neuron.png"))
        self.run_li_cap(test_parameters, epochs=80)

    def run_li_cap(self, test_parameters, epochs):
        """ Test leak can be trained """
        # Forward
        input_spikes = torch.bernoulli(torch.ones((150, 150, 1)) * 0.05)
        input_handle = LIFObservables(spikes=input_spikes)

        loss_fn = torch.nn.MSELoss()

        # Model
        model = Model(test_parameters)
        model.synapse.weight.data = torch.ones_like(
            model.synapse.weight.data) * 1
        model.train()

        with torch.no_grad():
            target_model = Model(test_parameters, is_target=True)
            target_model.synapse.weight.data = torch.ones_like(
                target_model.synapse.weight.data) * 1
            target = target_model(input_handle).membrane_cadc

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
        num_epochs = epochs
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=num_epochs//5, gamma=0.8)

        _, (trace_plot, param_plot) = plt.subplots(2, 1)
        caps = [model.neuron.membrane_capacitance.model_value
                .clone().detach().numpy()]
        tau_syns = [model.neuron.tau_syn.model_value.clone().detach().numpy()]
        pbar = tqdm(total=num_epochs, unit="batch", leave=False)
        for i in range(num_epochs):
            model.zero_grad()

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

            caps.append(model.neuron.membrane_capacitance.model_value
                        .clone().detach().numpy())
            tau_syns.append(model.neuron.tau_syn.model_value
                            .clone().detach().numpy())

            pbar.set_postfix(
                epoch=f"{i}", loss=f"{loss_b.item():.4f}")
            pbar.update()

        pbar.close()

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
            param_plot.axhline(test_parameters.target_tau_syn,
                               color="C1", ls='--')
            param_plot.plot(tau_syns, color="C1", label=r'$\tau_s$')
        param_plot.legend()

        plt.savefig(test_parameters.plot_path)
        plt.close()

        self.assertLess(loss_b.item(), 0.0004)


if __name__ == "__main__":
    unittest.main()
