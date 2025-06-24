from typing import Callable
import unittest
import torch

from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
from matplotlib import pyplot as plt
from dataclasses import dataclass

import hxtorch
from hxtorch.spiking import Experiment
from hxtorch.spiking.modules import ReadoutNeuronExp
from hxtorch.spiking.handle import LIFObservables
from hxtorch.spiking.transforms import weight_transforms
from hxtorch.spiking.parameter import ModelParameter


log = hxtorch.logger.get("tests.hw.translation")
hxtorch.logger.set_loglevel(log, hxtorch.logger.LogLevel.INFO)

@dataclass
class TestParameters:
    target_tau_mem: float
    target_tau_syn: float
    plot_path: Path
    start_tau_mem: float = None
    start_tau_syn: float = None


class Model(torch.nn.Module):
    def __init__(self, test_parameters: TestParameters, is_target: bool=False):
        super().__init__()
        dt = 1e-6
        self.experiment = Experiment(mock=True, dt=dt)

        tau_mem = ModelParameter(torch.exp(torch.ones(1) * -dt/test_parameters.target_tau_mem))
        tau_syn = ModelParameter(torch.exp(torch.ones(1) * -dt/test_parameters.target_tau_syn))

        if test_parameters.start_tau_mem and not is_target:
            tau_mem = ModelParameter(torch.exp(torch.ones(1) * -dt/test_parameters.start_tau_mem)).make_trainable()
        if test_parameters.start_tau_syn and not is_target:
            tau_syn = ModelParameter(torch.exp(torch.ones(1) * -dt/test_parameters.start_tau_syn)).make_trainable()

        self.module1 = hxtorch.snn.Synapse(
            1, 1, self.experiment, transform=partial(
                weight_transforms.linear_saturating))
        self.module1.weight.requires_grad_(False)
        self.module2 = ReadoutNeuronExp(
            1, self.experiment,
            leak=ModelParameter(0.),
            tau_mem=tau_mem,
            tau_syn=tau_syn)

    def forward(self, input):
        ret = self.module2(self.module1(input))
        hxtorch.snn.run(self.experiment, input.spikes.shape[0])
        return ret

class TestTranslation_Capacitance(unittest.TestCase):

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_li_single(self):
        test_parameters = TestParameters(
            target_tau_mem = 10e-6,
            target_tau_syn = 10e-6,
            start_tau_mem = 4e-6,
            start_tau_syn = 20e-6,
            plot_path=self.plot_path.joinpath(f"./train_neuron.png"))
        self.run_li_cap(test_parameters, epochs=80)

    def run_li_cap(self, test_parameters, epochs):
        """ Test leak can be trained """
        # Forward
        input_spikes = torch.bernoulli(
            torch.ones((150, 150, 1)) * 0.05)
        input_handle = LIFObservables(spikes=input_spikes)

        loss_fn = torch.nn.MSELoss()

        # Model
        model = Model(test_parameters)
        model.module1.weight.data = torch.ones_like(
            model.module1.weight.data) * 1

        model.train()

        with torch.no_grad():
            target_model = Model(test_parameters, is_target=True)
            target_model.module1.weight.data = torch.ones_like(
                target_model.module1.weight.data) * 1
            target = target_model(input_handle).membrane_cadc

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        num_epochs = epochs
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=num_epochs//5, gamma=0.8)

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

        self.assertLess(loss_b.item(), 0.0004)


if __name__ == "__main__":
    hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.INFO)
    for key in ["hxcomm", "grenade", "stadls", "calix"]:
        other_logger = hxtorch.logger.get(key)
        hxtorch.logger.set_loglevel(other_logger, hxtorch.logger.LogLevel.WARN)
    unittest.main()
