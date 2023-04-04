import unittest

from hxtorch.examples.perceptron import minimal, mnist
from dlens_vx_v3 import hxcomm


class MinimalExampleTest(unittest.TestCase):
    """
    Tests the minimal example.
    """

    def test_main(self) -> None:
        """
        Run the example.
        """
        minimal.main()


class MNISTExampleTest(unittest.TestCase):
    """
    Tests the MNIST example implementation.
    """

    def test_training(self, mock=False) -> None:
        """
        Run MNIST training and inference.
        """
        parser = mnist.get_parser()
        train_args = [
            "--epochs=1",
            "--dataset-fraction=0.05",
            "--batch-size=30",
            "--data-path=/loh/data/mnist",
        ]
        if mock:
            train_args.append("--mock")
        accuracy = mnist.main(parser.parse_args(train_args))
        self.assertGreater(accuracy, 0.5,
                           "Accuracy is much lower than expected.")

    def test_training_mock(self):
        self.test_training(mock=True)

    def test_custom_calib(self) -> None:
        """
        Initialize the experiment with custom calib.
        """
        # get the default calib path for used setup
        with hxcomm.ManagedConnection() as connection:
            calib_path = "/wang/data/calibration/hicann-dls-sr-hx/" \
                + connection.get_unique_identifier() \
                + "/stable/latest/hagen_cocolist.pbin"

        parser = mnist.get_parser()
        train_args = [
            "--epochs=0",
            "--dataset-fraction=0.05",
            "--batch-size=30",
            "--data-path=/loh/data/mnist",
            f"--calibration-path={calib_path}",
        ]
        mnist.main(parser.parse_args(train_args))


if __name__ == "__main__":
    unittest.main()
