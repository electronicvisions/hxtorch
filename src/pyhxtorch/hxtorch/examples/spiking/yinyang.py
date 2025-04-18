"""
Spiking HX torch yinyang example
"""
# pylint: disable=no-member
from typing import Tuple

import argparse
import os
import numpy as np

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import hxtorch
from hxtorch.spiking.modules import (
    Synapse, EventPropSynapse, Neuron, EventPropNeuron)
from hxtorch.examples.spiking.yinyang_model import SNN, Model
from hxtorch.spiking.datasets.yinyang import YinYangDataset
from hxtorch.spiking.transforms.decode import MaxOverTime
from hxtorch.spiking.transforms.encode import CoordinatesToSpikes

log = hxtorch.logger.get("hxtorch.examples.spiking.yinyang")
hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser with all the options.
    """
    parser = argparse.ArgumentParser(
        description="hxtorch spiking YinYang example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--mock", action="store_true", default=False,
        help="enable mock mode")

    # data
    parser.add_argument("--testset-size", type=int, default=1000)
    parser.add_argument("--trainset-size", type=int, default=5000)

    # encoding
    parser.add_argument("--t-shift", type=float, default=0e-6)
    parser.add_argument("--t-bias", type=float, default=2e-6)
    parser.add_argument("--t-early", type=float, default=2e-6)
    parser.add_argument("--t-late", type=float, default=26e-6)
    parser.add_argument("--t-sim", type=float, default=38e-6)

    # model
    parser.add_argument("--dt", type=float, default=1e-6)
    parser.add_argument("--n-hidden", type=int, default=120)
    parser.add_argument("--tau-mem", type=float, default=6e-6)
    parser.add_argument("--tau-syn", type=float, default=6e-6)
    parser.add_argument("--weight-init-hidden-mean", type=float, default=0.2)
    parser.add_argument("--weight-init-hidden-std", type=float, default=0.2)
    parser.add_argument("--weight-init-out-mean", type=float, default=0.0)
    parser.add_argument("--weight-init-out-std", type=float, default=0.1)

    # training
    parser.add_argument("--alpha", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size", type=int, default=50, metavar="<num samples>",
        help="input batch size for training")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--reg-bursts", type=float, default=0.0)
    parser.add_argument("--reg-weights-hidden", type=float, default=0.0)
    parser.add_argument("--reg-readout", type=float, default=0.0004)
    parser.add_argument("--reg-weights-output", type=float, default=0.0)
    parser.add_argument("--gradient-estimator", type=str, default="surrogate_gradient")

    # hw
    parser.add_argument("--readout-scaling", type=float, default=10.0)
    parser.add_argument("--weight-scale", type=float, default=64.)
    parser.add_argument("--trace-scale", type=float, default=1. / 50.)

    parser.add_argument("--plot-path", type=str)

    return parser


def plot(train_loss, train_acc, test_loss, test_acc, args):
    """
    Plot losses and accuracies.

    :param train_loss: List holding the average training loss for each epoch
    :param train_acc: List holding the average training accuracy for each epoch
    :param test_loss: List holding the average test loss each epoch
    :param test_acc: List holding the average accuracy for each epoch
    :param args: The arguments.
    """
    fig, axes = plt.subplots(2)
    epochs = np.arange(1, args.epochs + 1)

    # Losses
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_xlim(1, args.epochs)
    axes[0].plot(epochs, np.array(train_loss), label="Training")
    axes[0].plot(epochs, np.array(test_loss), label="Testing")
    axes[0].legend()
    axes[0].semilogy()

    # Accuracy
    axes[1].set_ylabel("Acc.")
    axes[1].set_xlabel("Epoch")
    axes[1].set_xlim(1, args.epochs)
    axes[1].plot(epochs, np.array(train_acc), label="Training")
    axes[1].plot(epochs, np.array(test_acc), label="Testing")

    axes[1].legend()
    axes[1].semilogy()

    fig.tight_layout()
    plt.savefig(args.plot_path)


def train(model: torch.nn.Module, loader: DataLoader,
          optimizer: torch.optim.Optimizer, args: argparse.Namespace,
          epoch: int) -> Tuple[float, float]:
    """
    Perform training for one epoch.

    :param model: The model to train.
    :param loader: Pytorch DataLoader instance providing training data.
    :param optimizer: The optimizer used or weight optimization.
    :param args: The argparse Namespace.
    :param epoch: Current epoch for logging.

    :returns: Tuple (training loss, training accuracy)
    """
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    model.train()
    dev = model.network.device

    loss, acc = 0., 0.
    n_total = len(loader)

    pbar = tqdm(total=len(loader), unit="batch", leave=False)
    for data, target in loader:
        model.zero_grad()
        # clip hidden and rescale output weights to hardware range
        if not args.mock:
            with torch.no_grad():
                # clamp to maximum possible weight (when scaled to HW)
                limit = 63. / args.weight_scale
                model.network.linear_h.weight.data.clamp_(-limit, limit)
                # scale output weights to fit HW range
                scale = min(
                    63. / np.abs(
                        model.network.linear_o.weight.cpu().detach()
                    ).max(),
                    args.weight_scale)
                # reset partial
                model.network.linear_o.weight_transform.keywords["scale"] \
                    = scale

        scores = model(data.to(dev))

        reg_loss = model.regularize(
            args.reg_readout, args.reg_bursts, args.reg_weights_hidden,
            args.reg_weights_output)
        loss_b = cross_entropy_loss(scores, target.to(dev)) + reg_loss

        loss_b.backward()
        optimizer.step()

        loss += loss_b.item() / n_total

        # Train accuracy
        pred = scores.detach().cpu().argmax(dim=1)
        acc_b = pred.eq(target.view_as(pred)).float().mean()
        acc += acc_b / n_total

        # Firing rates
        rate_b = model.network.s_h.spikes.detach().sum() / scores.shape[0]

        pbar.set_postfix(
            epoch=f"{epoch}", loss=f"{loss_b.item():.4f}", acc=f"{acc_b:.4f}",
            rate=f"{rate_b:.2f}", lr=f"{optimizer.param_groups[-1]['lr']}")
        pbar.update()

    pbar.close()

    return loss, acc


def test(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
         args: argparse.Namespace, epoch: int) -> Tuple[float, float]:
    """
    Test the model.

    :param model: The model to test
    :param loader: Data loader containing the test data set
    :param epoch: Current trainings epoch.

    :returns: Tuple of (test loss, test accuracy)
    """
    model.eval()
    dev = model.network.device

    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    loss, acc = 0., 0.
    n_total = len(loader)

    pbar = tqdm(total=len(loader), unit="batch", leave=False)
    for data, target in loader:
        scores = model(data.to(dev))

        reg_loss = model.regularize(
            args.reg_readout, args.reg_bursts, args.reg_weights_hidden,
            args.reg_weights_output)
        loss_b = cross_entropy_loss(scores, target.to(dev)) + reg_loss
        loss += loss_b.item() / n_total

        pred = scores.cpu().argmax(dim=1)
        acc += pred.eq(target.view_as(pred)).float().mean() / n_total

        pbar.update()

    pbar.close()

    log.info(
        f"Test epoch: {epoch}, average loss: {loss:.4f}, "
        + f"test acc={100 * acc:.2f}%")

    return loss, acc


def main(args: argparse.Namespace) -> float:
    """
    Entrypoint for SNN training on the YinYang dataset. Loads the dataset and
    executes training.

    :param args: Argparse Namespace providing necessary constants.

    :return: Returns the achieved accuracy after the last test epoch.
    """
    if not args.mock:
        hxtorch.init_hardware()

    torch.manual_seed(args.seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {dev}")

    log.info("Load train and test sets.")
    trainset = YinYangDataset(size=args.trainset_size, seed=42)
    testset = YinYangDataset(size=args.testset_size, seed=41)

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=True)
    log.info("Finished loading datasets and dataloaders.")

    hidden_cadc_recording = True
    log.INFO(f"{args.gradient_estimator}")
    if args.gradient_estimator == "surrogate_gradient":
        synapse_type = Synapse
        neuron_type = Neuron
    elif args.gradient_estimator == "eventprop":
        synapse_type = EventPropSynapse
        neuron_type = EventPropNeuron
        hidden_cadc_recording = False
    else:
        log.ERROR("Please specify one of the currently supported gradient "
                  "estimation algorithms, 'eventprop' and 'surrogate_gradient'."
                  f" Your input: '{args.gradient_estimator}'")

    # init model, optimizer and scheduler
    model = Model(
        CoordinatesToSpikes(
            seq_length=int(args.t_sim / args.dt),
            t_early=args.t_early,
            t_late=args.t_late,
            dt=args.dt,
            t_bias=args.t_bias),
        SNN(
            n_in=5,
            n_hidden=args.n_hidden,
            n_out=3,
            mock=args.mock,
            dt=args.dt,
            tau_mem=args.tau_mem,
            tau_syn=args.tau_syn,
            alpha=args.alpha,
            trace_shift_hidden=int(args.t_shift / args.dt),
            trace_shift_out=int(args.t_shift / args.dt),
            weight_init_hidden=(
                args.weight_init_hidden_mean, args.weight_init_hidden_std),
            weight_init_output=(
                args.weight_init_out_mean, args.weight_init_out_std),
            weight_scale=args.weight_scale,
            trace_scale=args.trace_scale,
            input_repetitions=1 if args.mock else 5,
            synapse_type=synapse_type,
            neuron_type=neuron_type,
            hidden_cadc_recording=hidden_cadc_recording,
            device=dev),
        MaxOverTime(),
        args.readout_scaling)
    log.info("mock = ", args.mock)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma)

    losses_train, accs_train = [], []
    losses_test, accs_test = [], []

    # Train and test
    for epoch in range(1, args.epochs + 1):
        loss_train, acc_train = train(
            model, train_loader, optimizer, args, epoch)
        loss_test, acc_test = test(model, test_loader, args, epoch)

        losses_train.append(loss_train)
        accs_train.append(acc_train)
        losses_test.append(loss_test)
        accs_test.append(acc_test)

        scheduler.step()

    if not args.mock:
        hxtorch.release_hardware()

    plot(losses_train, accs_train, losses_test, accs_test, args)

    return losses_train, accs_train, losses_test, accs_test


if __name__ == "__main__":
    main(get_parser().parse_args())
