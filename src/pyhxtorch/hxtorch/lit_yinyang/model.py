from typing import Optional, Tuple, Callable
from functools import partial
import argparse
import pickle

import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

import hxtorch
import hxtorch.snn as snn
import hxtorch.snn.functional as F
import hxtorch.snn.transforms as hxtransforms

log = hxtorch.logger.get("hxtorch.examples.lit_yinyang.model")
hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)


class CustomSynapse(snn.Synapse):

    """ Custom class to use uniform in weight reset and scale weights """

    def __init__(self, *args, weight_scale=1., **kwargs) -> None:
        """
        :param weight_scale: Factor by which weights specified in
            code will be scaled when written to hardware
        """
        super().__init__(*args, **kwargs)
        self.weight_transform = partial(
            hxtransforms.linear_saturating,
            scale=weight_scale)


class LitSNN(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) \
            -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("LitModel")
        # topology
        parser.add_argument("--n-in", type=int, default=5)
        parser.add_argument("--multiplex-input", action="store_true", default=False)
        parser.add_argument("--input-repetitions", type=int, default=5)
        parser.add_argument("--n-hidden", type=int, default=120)
        parser.add_argument("--n-out", type=int, default=3)
        parser.add_argument("--mock", action="store_true", default=False)
        # timescales
        parser.add_argument("--dt", type=float, default=1e-6)
        parser.add_argument("--t-sim", type=float, default=60e-6)
        parser.add_argument("--tau-mem", type=float, default=8e-6)
        parser.add_argument("--tau-syn", type=float, default=8e-6)
        parser.add_argument("--alpha", type=float, default=150.)
        parser.add_argument("--t-shift", type=float, default=-2e-6)
        parser.add_argument("--t-early", type=float, default=2e-6)
        parser.add_argument("--t-late", type=float, default=40e-6)
        parser.add_argument("--t-bias", type=float, default=18e-6)
        # weights
        parser.add_argument("--weight-init-hidden", type=float, nargs=2,
                            default=[1.0, 0.4])
        parser.add_argument("--weight-init-output", type=float, nargs=2,
                            default=[0.0, 0.05])
        parser.add_argument("--weight-scale", type=float, default=50.)
        # hw operation point
        parser.add_argument("--trace-scale", type=float, default=0.03)
        parser.add_argument("--trace-offset", type=float, default=-50.)
        parser.add_argument("--hw-op", type=str, default=None)
        # synapse and neuron functions
        parser.add_argument("--funcs", type=str,
                            default="eventprop")
        # training
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--gamma", type=float, default=0.97)
        parser.add_argument("--scheduler-step-size", type=int, default=1)
        parser.add_argument("--reg-readout", type=float, default=0.0004)
        parser.add_argument("--readout-scaling", type=float, default=10.)

        return parent_parser

    def __init__(self,
                 n_in: int,
                 n_hidden: int,
                 n_out: int,
                 mock: bool,
                 dt: float = 1e-6,
                 t_sim: float = 60e-6,
                 t_early: float = 2e-6,
                 t_late: float = 40e-6,
                 t_bias: float = 18e-6,
                 tau_mem: float = 8e-6,
                 tau_syn: float = 8e-6,
                 alpha: float = 150.,
                 t_shift: float = -2e-6,
                 weight_init_hidden: Optional[Tuple[float, float]] = None,
                 weight_init_output: Optional[Tuple[float, float]] = None,
                 weight_scale: float = 1.,
                 lr: float = 0.001,
                 gamma: float = 0.97,
                 scheduler_step_size: int = 1,
                 reg_readout: float = 0.0004,
                 readout_scaling: float = 10.,
                 trace_scale: float = 1.,
                 trace_offset: float = 0.,
                 funcs: str = "eventprop",
                 hw_op: str = None,
                 multiplex_input: bool = False,
                 input_repetitions: int = 5,
                 **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters()

        # use eventprop or surrogate gradients
        if funcs == "eventprop":
            synapse_func_hidden = F.eventprop_synapse
            neuron_func_hidden = F.EventPropNeuron
        elif funcs == "eventprop_rounding":
            if isinstance(weight_scale, float):
                scale_temp = torch.tensor([weight_scale])
            elif isinstance(weight_scale, torch.Tensor):
                scale_temp = weight_scale.clone()
            else:
                log.info(f"Weight scale is neither float nor tensor, but "
                         + f"{type(weight_scale)}. Using 1. instead.")
            synapse_func_hidden = partial(F.eventprop_rounding_synapse,
                                          scale=scale_temp)
            neuron_func_hidden = F.EventPropNeuron
        elif funcs == "surrogate_gradient":
            synapse_func_hidden = torch.nn.functional.linear
            neuron_func_hidden = F.lif_integration
        elif funcs == "surrogate_gradient_exp":
            synapse_func_hidden = torch.nn.functional.linear
            neuron_func_hidden = F.lif_exp_integration
        else:
            log.info("argument 'funcs' should be either 'eventprop' or "
                     + f"'surrogate_gradient', but is {funcs}. Using the "
                     + "default 'eventprop'")
            synapse_func_hidden = F.eventprop_synapse
            neuron_func_hidden = F.EventPropNeuron

        # load from hw operation point
        if hw_op is not None:
            log.info("Overwriting `weight_scale`, `trace_scale`, "
                     + f"`trace_offset` with data from {hw_op}")
            # load hw operation point
            f = open(hw_op, "rb")
            dict_hw_op = pickle.load(f)
            f.close()
            del f

            trace_offset = {}
            trace_scale = {}
            for i, nrn in enumerate(dict_hw_op["neurons"]):
                trace_offset[nrn] = dict_hw_op["offset"][i]
                trace_scale[nrn] = \
                    1. / (dict_hw_op["threshold"][i] - dict_hw_op["offset"][i])
            weight_scale = dict_hw_op["mean_weight_scale"]
            self.hparams["weight_scale"] = weight_scale

        # Encoder
        self.encoder = snn.transforms.CoordinatesToSpikes(
            seq_length=int(t_sim / dt), t_early=t_early,
            t_late=t_late, dt=dt, t_bias=t_bias)

        self.multiplex_input = multiplex_input
        self.input_repetitions = input_repetitions

        # Instance to work on
        self.instance = snn.Instance(mock=mock, dt=dt)
        # Ajdust placement to use second hemisphere for output neurons
        self.instance.neuron_placement = snn.instance.NeuronPlacement(
            permutation = list(range(0, n_hidden)) + list(range(256, 259)))

        # Input projection
        self.linear_h = CustomSynapse(
            n_in * input_repetitions,
            n_hidden,
            func=synapse_func_hidden,
            instance=self.instance,
            weight_scale=weight_scale)
        if weight_init_hidden:
            w = torch.zeros(n_hidden, n_in)
            torch.nn.init.normal_(w, *weight_init_hidden)
            self.linear_h.weight.data = w.repeat(1, input_repetitions)
            # self.linear_h.reset_parameters(*weight_init_hidden)

        # Hidden layer
        lif_params = F.LIFParams(1. / tau_mem, 1. / tau_syn,
                                 dt=dt, alpha=alpha)
        self.lif_h = snn.Neuron(
            size=n_hidden, instance=self.instance, func=neuron_func_hidden,
            params=lif_params, trace_scale=trace_scale,
            cadc_time_shift=int(t_shift / dt), shift_cadc_to_first=True,
            enable_cadc_recording=False if funcs=="eventprop" else True)
        # Output projection
        self.linear_o = CustomSynapse(n_hidden, n_out, instance=self.instance,
                                      weight_scale=weight_scale)
        if weight_init_output:
            self.linear_o.reset_parameters(*weight_init_output)
        # self.linear_o.weight.requires_grad_(False)

        # Readout layer
        li_params = F.LIParams(1. / tau_mem, 1. / tau_syn, dt=dt)
        self.li_o = snn.ReadoutNeuron(
            size=n_out, instance=self.instance, func=F.li_integration,
            params=li_params, trace_scale=trace_scale,
            cadc_time_shift=int(t_shift / dt), shift_cadc_to_first=True)

        # Loss
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.reg_readout = reg_readout
        self.readout_scaling = readout_scaling

        # Time step
        self.dt = dt

        # learning rate
        self.lr = lr
        self.gamma = gamma
        self.scheduler_step_size = scheduler_step_size

        # placeholder for hidden spike
        self.s_h = None
        self.v_cadc = None

        # accuracies
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward path.

        :param spikes: NeuronHandle holding spikes as input.

        :return: Returns the output of the network, i.e. membrane traces of the
            readout neurons.
        """
        spikes = self.encoder(coordinates)

        spikes = spikes.repeat(1, 1, self.input_repetitions)

        spikes_handle = snn.NeuronHandle(spikes)
        c_h = self.linear_h(spikes_handle)
        self.s_h = self.lif_h(c_h)
        c_o = self.linear_o(self.s_h)
        v_o = self.li_o(c_o)
        hxtorch.snn.run(self.instance, spikes.shape[1])

        self.v_cadc = v_o.v_cadc
        max_traces, _ = torch.max(v_o.v_cadc, 1)

        return max_traces

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.gamma)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.forward(x)
        with torch.no_grad():
            z *= self.readout_scaling
        loss = self.loss_fn(z, y)
        loss += self.reg_readout * torch.mean(z ** 2)
        self.log("train_loss", loss)
        rate = float(self.s_h.spikes.sum() / x.shape[0])
        self.log("hidden rate", rate, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.forward(x)
        preds = z.argmax(dim=1)
        self.val_accuracy.update(preds, y)
        with torch.no_grad():
            z *= self.readout_scaling
        loss = self.loss_fn(z, y)
        loss += self.reg_readout * torch.mean(z ** 2)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        z = self.forward(x)
        preds = z.argmax(dim=1)
        with torch.no_grad():
            z *= self.readout_scaling
        loss = self.loss_fn(z, y)
        loss += self.reg_readout * torch.mean(z ** 2)
        self.test_accuracy.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
