"""
Test SNN examples
"""
import unittest

from hxtorch.examples.spiking import yinyang


class YinYangExampleTest(unittest.TestCase):
    """ Tests the YinYang example implementation """

    def test_training(self, mock=False) -> None:
        """
        Run YinYang training and inference.
        """
        parser = yinyang.get_parser()
        train_args = [
            "--alpha=150",
            "--batch-size=75",
            "--dt=2.0e-06",
            "--epochs=4",
            "--gamma=0.9",
            "--lr=0.002",
            "--n-hidden=120",
            "--plot-path=./test_yinyang_hw.png",
            "--readout-scaling=10.",
            "--reg-readout=0.0004",
            "--seed=0",
            "--step-size=10",
            "--tau-mem=6.0e-06",
            "--tau-syn=6.0e-06",
            "--testset-size=1000",
            "--trainset-size=5000",
            "--t-bias=1.8e-05",
            "--t-early=2.0e-06",
            "--t-late=4.0e-05",
            "--t-sim=6.0e-05",
            "--t-shift=-2.0e-06",
            "--trace-scale=0.0147",
            "--weight-scale=66.39",
            "--weight-init-hidden-std=0.25",
            "--weight-init-out-mean=0.0",
            "--weight-init-out-std=0.1",
        ]
        if mock:
            train_args.append("--mock")
            train_args.append("--readout-scaling=1.")
            train_args.append("--plot-path=./test_yinyang_mock.png")
            train_args.append("--weight-init-hidden-mean=1")
        else:
            train_args.append("--weight-init-hidden-mean=0.08")

        _, _, _, accuracy = yinyang.main(parser.parse_args(train_args))

        self.assertGreater(
            accuracy[-1], 0.85, "Accuracy is lower than expected.")

    def test_training_mock(self):
        self.test_training(mock=True)


if __name__ == "__main__":
    unittest.main()
