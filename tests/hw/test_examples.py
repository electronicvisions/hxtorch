import unittest
import os

import hxtorch
from hxtorch.examples import minimal, mnist


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
    mock = False

    def test_training(self) -> None:
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
        if self.mock:
            train_args.append("--mock")
        accuracy = mnist.main(parser.parse_args(train_args))
        self.assertGreater(accuracy, 0.5,
                           "Accuracy is much lower than expected.")


class MNISTExampleTestMock(MNISTExampleTest):
    """
    Tests the MNIST example in mock mode.
    """
    mock = True


if __name__ == "__main__":
    unittest.main()
