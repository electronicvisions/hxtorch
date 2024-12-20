"""
Test SNN examples
"""
import unittest

from hxtorch.examples.spiking import yinyang
from hxtorch.examples.spiking import calib_neuron


class YinYangExampleTest(unittest.TestCase):
    """ Tests the YinYang example implementation """

    def test_training(self, mock=False, surrogate_gradient=False) -> None:
        """
        Run YinYang training and inference.
        """
        parser = yinyang.get_parser()
        train_args = [
            "--alpha=150",
            "--batch-size=50",
            "--dt=1.0e-06",
            "--epochs=5",
            "--gamma=0.9",
            "--lr=0.002",
            "--n-hidden=120",
            "--readout-scaling=10.",
            "--reg-readout=0.0004",
            "--seed=0",
            "--step-size=10",
            "--tau-mem=6.0e-06",
            "--tau-syn=6.0e-06",
            "--testset-size=1000",
            "--trainset-size=5000",
            "--t-bias=2.0e-06",
            "--t-early=2.0e-06",
            "--t-late=2.6e-05",
            "--t-sim=3.8e-05",
            "--t-shift=0.0e-06",
            "--trace-scale=0.0147",
            "--weight-scale=70.00",
            "--weight-init-hidden-std=0.25",
            "--weight-init-out-mean=0.0",
            "--weight-init-out-std=0.1",
        ]
        if mock:
            train_args.append("--mock")
            train_args.append("--readout-scaling=1.")
            train_args.append("--weight-init-hidden-mean=0.8")
            if surrogate_gradient:
                train_args.append("--gradient-estimator=surrogate_gradient")
                train_args.append(
                    "--plot-path=./test_yinyang_mock_surrogate_gradient.png")
            else:
                train_args.append("--gradient-estimator=eventprop")
                train_args.append(
                    "--plot-path=./test_yinyang_mock_eventprop.png")
        else:
            train_args.append("--weight-init-hidden-mean=0.25")
            if surrogate_gradient:
                train_args.append("--gradient-estimator=surrogate_gradient")
                train_args.append(
                    "--plot-path=./test_yinyang_hw_surrogate_gradient.png")
            else:
                train_args.append("--gradient-estimator=eventprop")
                train_args.append("--epochs=10")
                train_args.append(
                    "--plot-path=./test_yinyang_hw_eventprop.png")

        _, _, _, accuracy = yinyang.main(parser.parse_args(train_args))

        if not mock and not surrogate_gradient:
            # EventProp is not as stable on HW. Issue: 4050
            self.assertGreater(
                accuracy[-1], 0.60, "Accuracy is lower than expected.")
        else:
            self.assertGreater(
                accuracy[-1], 0.85, "Accuracy is lower than expected.")

    def test_training_sg(self):
        self.test_training(mock=False, surrogate_gradient=True)

    def test_training_mock(self):
        self.test_training(mock=True, surrogate_gradient=False)

    def test_training_mock_sg(self):
        self.test_training(mock=True, surrogate_gradient=True)


class CalibNeuronExampleTest(unittest.TestCase):
    """ Test example calibration of neuron """

    def test_calib_neuron(self) -> None:
        spikes, _ = calib_neuron.main()

        self.assertGreater(spikes.to_sparse().shape[0], 0)



if __name__ == "__main__":
    import hxtorch
    hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.INFO)
    for key in ["hxcomm", "grenade", "stadls", "calix"]:
        other_logger = hxtorch.logger.get(key)
        hxtorch.logger.set_loglevel(other_logger, hxtorch.logger.LogLevel.ERROR)
    log = hxtorch.logger.get("Neuron")
    unittest.main()
