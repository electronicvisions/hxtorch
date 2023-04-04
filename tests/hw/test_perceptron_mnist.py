"""
Tests linear and conv2d layers of hxtorch by classifying some
MNIST images. Hagen-mode calibration and MAC operation are used.
"""
from abc import ABCMeta, abstractmethod
import unittest
from pathlib import Path
import torch

import hxtorch
from hxtorch.perceptron.nn import scale_input, scale_weight

hxtorch.logger.reset()
hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.INFO)
hxtorch.logger.set_loglevel(hxtorch.logger.get('grenade'), hxtorch.logger.LogLevel.WARN)
hxtorch.logger.set_loglevel(hxtorch.logger.get('fisch.PlaybackProgramBuilder'), hxtorch.logger.LogLevel.ERROR)
torch.set_num_threads(1)


class HXTorchModel(torch.nn.Module):
    """
    Model used to classify MNIST images.

    Uses zero padding of 1 to produce images shaped (30, 30).
    Uses a convolutional layer with a kernel (10, 10) as first layer.
    Uses two dense layers afterwards.
    """

    def __init__(self, mock: bool):
        super().__init__()
        self.conv2d = hxtorch.perceptron.nn.Conv2d(
            1, out_channels=20, kernel_size=(10, 10), stride=(5, 5),
            bias=False, padding=1,
            input_transform=scale_input, weight_transform=scale_weight,
            num_sends=3, wait_between_events=2, mock=mock)
        self.fc1 = hxtorch.perceptron.nn.Linear(
            5 * 5 * 20, 128, bias=False,
            input_transform=scale_input, weight_transform=scale_weight,
            num_sends=6, wait_between_events=2, mock=mock)
        self.fc2 = hxtorch.perceptron.nn.Linear(
            128, 10, bias=False,
            input_transform=scale_input, weight_transform=scale_weight,
            num_sends=6, wait_between_events=2, mock=mock)

    def forward(self, *input):  # pylint: disable=redefined-builtin
        result = torch.nn.functional.relu(self.conv2d(input[0]))
        result = result.reshape(-1, 5 * 5 * 20)  # flatten
        result = torch.nn.functional.relu(self.fc1(result))
        result = self.fc2(result)
        return result


class PyTorchModel(torch.nn.Module):
    """
    The equivalent PyTorch model.
    """
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 20, (10, 10), 5, 1, bias=False)
        self.fc1 = torch.nn.Linear(5 * 5 * 20, 128, bias=False)
        self.fc2 = torch.nn.Linear(128, 10, bias=False)

    def forward(self, *input):  # pylint: disable=redefined-builtin
        result = torch.nn.functional.relu(self.conv2d(input[0]))
        result = result.reshape(-1, 5 * 5 * 20)  # flatten
        result = torch.nn.functional.relu(self.fc1(result))
        result = self.fc2(result)
        return result


class MNISTTest(unittest.TestCase, metaclass=ABCMeta):
    """
    Inference test on the MNIST dataset.

    :cvar model_args: Arguments for the model
    """
    model_args = {}

    @property
    @abstractmethod
    def model_class(self):
        raise NotImplementedError

    def test_mnist(self) -> None:
        """
        Run MNIST inference.
        """
        data_path = Path(__file__).parent.joinpath("test_perceptron_mnist")

        model = self.model_class(**self.model_args)
        model.load_state_dict(
            torch.load(data_path.joinpath("model_state.pkl")))
        model.eval()

        data = torch.load(data_path.joinpath("test_data.pt")).to(torch.float)
        output = model(data)
        predicted = output.argmax(dim=1)
        target = torch.load(data_path.joinpath("test_labels.pt"))
        accuracy = (100. * (predicted == target)).mean().item()

        hxtorch.logger.get(f"{self.__class__.__name__}.test_perceptron_mnist").INFO(
            f"Classified {len(data)} MNIST images, "
            f"accuracy: {accuracy:.1f}% ({self.__class__.__name__[9:]})")

        self.assertGreater(
            accuracy, 85, "MNIST success is lower than usual.")


class MNISTTestHX(MNISTTest):
    """
    Initializes and calibrates the chip.
    Uses a pre-trained model to classify some MNIST images.
    Asserts the success rate is as expected.
    """
    model_args = {"mock": False}
    model_class = HXTorchModel

    @classmethod
    def setUpClass(cls) -> None:
        hxtorch.init_hardware(ann=True)

    @classmethod
    def tearDownClass(cls) -> None:
        hxtorch.release_hardware()  # also disconnects executor


class MNISTTestMock(MNISTTest):
    """
    Tests the MNIST-example with the mock implementations.
    """
    model_args = {"mock": True}
    model_class = HXTorchModel

    @classmethod
    def setUpClass(cls) -> None:
        hxtorch.perceptron.set_mock_parameter(
            hxtorch.perceptron.MockParameter(noise_std=1.6, gain=0.0018))


class MNISTTestMockWithoutNoise(MNISTTest):
    """
    Tests the MNIST-example with the mock implementations without added noise.
    """
    model_args = {"mock": True}
    model_class = HXTorchModel

    @classmethod
    def setUpClass(cls) -> None:
        hxtorch.perceptron.set_mock_parameter(
            hxtorch.perceptron.MockParameter(noise_std=0, gain=0.0018))


class MNISTTestPyTorch(MNISTTest):
    """
    Tests the MNIST-example with pure pytorch layers.
    """
    model_class = PyTorchModel


# keeps testrunners from instantiating the abstract base class
del MNISTTest


if __name__ == "__main__":
    unittest.main()
