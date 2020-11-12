import functools
from pathlib import Path
import unittest
import matplotlib.pyplot as plt
import torch
import hxtorch

hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.INFO)

class TestHardwareDeviationsHX(unittest.TestCase):
    """
    Tests the deviations from linear behaviour.
    """
    matmul = hxtorch.matmul
    plot_path = Path(__file__).parent.joinpath("plots")

    @classmethod
    def setUpClass(cls):
        hxtorch.init()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release()

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_plot_linearity(self):
        data_in = torch.arange(0, 22, 3)
        data_in = torch.ones((len(data_in), 100, 128), dtype=torch.float) \
            * data_in.reshape(-1, 1, 1)

        weights = torch.empty(128, 256, dtype=torch.float).uniform_(-63, 63)
        weights[:, :126] = torch.arange(-63, 63, dtype=torch.float)

        plt.figure(figsize=(9, 4))
        plt.imshow(weights)
        plt.colorbar()
        plt.savefig(self.plot_path.joinpath(
            f"{self.__class__.__name__}_linearity_weight.png"), dpi=600)

        result = self.matmul(data_in, weights)
        self.assertEqual(result.size(), torch.Size([len(data_in), 100, 256]))

        plt.figure(figsize=(9, 5))
        plt.title(f"Mean and standard deviation of {data_in.shape[1]} batches")
        for i, single_result in enumerate(result):
            plt.errorbar(
                range(256), single_result.mean(dim=0),
                yerr=single_result.std(dim=0),
                fmt=".", ms=0.,
                linewidth=2,
                alpha=.7,
                label=f"{int(data_in[i, 0, 0])}",
            )
        plt.legend(ncol=len(result) // 3)
        plt.xlabel("neuron number")
        plt.ylabel("neuron activation")
        self.plot_path.mkdir(exist_ok=True)
        plt.savefig(self.plot_path.joinpath(
            f"{self.__class__.__name__}_linearity_result.png"), dpi=600)

    def test_plot_noise(self):
        log = hxtorch.logger.get(self.__class__.__name__)

        inputs = [(10, 15), (10, 25), (20, 15), (20, 25)]

        for (value_in, value_w) in inputs:
            log.info(f"Run with inputs: {(value_in, value_w)}")
            data_in = torch.full((100, 128), value_in, dtype=torch.float)
            weights_in = torch.full((128, 256), value_w, dtype=torch.float)

            result_torch = torch.matmul(data_in, weights_in)
            result = torch.zeros_like(result_torch)
            for vector_id, vector in enumerate(data_in):
                result[vector_id] = self.matmul(vector, weights_in, wait_between_events=6)
            log.info(f"Mean output: {result.mean():.1f}")

            # check gain
            gain = torch.mean((result / result_torch).view(-1)).item()
            log.info(f"Gain: {gain:.5f} (median)")

            # check noise
            noise = result - result_torch * gain
            noise_std = noise.std(dim=0).mean() # this removes fixed pattern noise
            noise_fixed_std = noise.mean(dim=0).std() # this removes stat. noise
            log.info(
                f"Noise: ±{noise_std:.4f} (stat.) "
                f"/ {noise_fixed_std / result.mean() * 100:.2f}% (fixed)"
            )

            plt.figure(figsize=(9, 5))
            plt.title(
                f"Mean output: {result.mean():.1f}   "
                f"Mean gain: {gain:.5f}   "
                f"Noise: ±{noise_std:.3f} (stat.) "
                f"/ {noise_fixed_std / result.mean() * 100:.2f}% (fixed)"
            )
            plt.errorbar(x=range(256), y=noise.mean(dim=0),
                         yerr=noise.std(dim=0), fmt=".", linewidth=1)
            plt.xlabel("neuron number")
            plt.ylabel("noise (abs.)")
            plt.savefig(self.plot_path.joinpath(
                f"{self.__class__.__name__}_noise_result"
                f"{value_in}{value_w}.png"), dpi=600)


class TestHardwareDeviationsMock(TestHardwareDeviationsHX):
    """
    Tests the deviations from linear behaviour.
    """
    matmul = functools.partial(hxtorch.matmul, mock=True)

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    unittest.main()
