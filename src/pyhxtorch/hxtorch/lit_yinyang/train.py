import argparse
import os
import multiprocessing

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor, Callback, ModelCheckpoint)

import numpy as np
import torch
from torch.utils.data import DataLoader

import hxtorch
from hxtorch.snn.datasets.yinyang import YinYangDataset

from model import LitSNN

log = hxtorch.logger.get("hxtorch.examples.lit_yinyang.train")
hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser with all the options.
    """
    parser = argparse.ArgumentParser(
        description="hxtorch spiking YinYang example using pytorch lightning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    # dataset
    parser.add_argument("--trainset-size", type=int, default=5000)
    parser.add_argument("--testset-size", type=int, default=1000)
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="<num samples>",
        help="input batch size for training")
    # hw or mock
    parser.add_argument(
        "--calibration-path", type=str, metavar="<path>",
        default=os.getenv("HXTORCH_CALIBRATION_PATH"),
        help="path to custom calibration to use instead of latest nightly")
    # experiment training
    parser.add_argument("--epochs", type=int, default=20)
    # model
    parser = LitSNN.add_model_specific_args(parser)
    # load from checkpoint
    parser.add_argument("--from-ckpt-file", type=str, default=None)
    # save dir
    parser.add_argument("--save-dir", type=str, default="lightning_logs/")

    return parser


class RescaleOutputWeights(Callback):
    """
    Rescale hidden to output layer weights to use full hw range
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx, dataloader_idx):
        try:
            pl.utilities.finite_checks.detect_nan_parameters(pl_module)
        except ValueError:
            log.info("pytorch lightning detected NaN parameter..")
        with torch.no_grad():
            # clamp to maximum possible weight (when scaled to HW)
            limit = 63. / pl_module.linear_h.weight_transform.keywords["scale"]
            pl_module.linear_h.weight.data.clamp_(-limit, limit)
            # scale output weights to fit HW range
            scale = min(
                63. / np.abs(pl_module.linear_o.weight.cpu().detach()).max(),
                pl_module.linear_h.weight_transform.keywords["scale"])
            pl_module.log("output_weight_scale", scale)
            # reset partial
            pl_module.linear_o.weight_transform.keywords["scale"] = scale


class NanToZero(Callback):
    """
    Check parameter gradients after backward and set to 0 if nan or +-inf
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def on_after_backward(self, trainer, pl_module):
        for p in pl_module.parameters():
            torch.nan_to_num(
                p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)


def main(args: argparse.Namespace) -> None:
    # load model and args from checkpoint
    if args.from_ckpt_file is not None:
        ckpt_file = args.from_ckpt_file
        epochs = args.epochs
        log.info(f"Loading model from checkpoint at {args.from_ckpt_file} ...")
        model = LitSNN.load_from_checkpoint(args.from_ckpt_file)
        log.info(f"Model loaded. Discarding parsed arguments and using "
                 + f"{os.path.dirname(args.from_ckpt_file) + 'hparams.yaml'}")
        args = argparse.Namespace(**model.hparams)
        args.from_ckpt_file = ckpt_file
        args.epochs = epochs
    else:
        model = LitSNN(**vars(args))

    # load calib
    if not args.mock:
        if args.calibration_path:
            hxtorch.init_hardware(
                hxtorch.CalibrationPath(args.calibration_path))
        else:
            log.info("Apply latest nightly calibration")
            hxtorch.init_hardware(spiking=True)  # defaults to a nightly calib

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load datasets
    log.info("Load train and test sets.")
    trainset = YinYangDataset(size=args.trainset_size, seed=42)
    valset = YinYangDataset(size=args.testset_size, seed=40)
    testset = YinYangDataset(size=args.testset_size, seed=41)

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=min(8, multiprocessing.cpu_count()))
    val_loader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(8, multiprocessing.cpu_count()))
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(8, multiprocessing.cpu_count()))
    log.info("Finished loading datasets and dataloaders.")

    # checkpointing callback
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="ckpt-yinyang-{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}")
    # lr momnitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # trainer
    trainer = pl.Trainer(max_epochs=args.epochs, log_every_n_steps=10,
                         logger=CSVLogger(save_dir=args.save_dir),
                         callbacks=[
                            checkpoint_callback,
                            lr_monitor,
                            RescaleOutputWeights(),
                            NanToZero()])
    # train
    trainer.fit(model, train_loader, val_loader)

    log.info(f"Load best model from '{checkpoint_callback.best_model_path}'")
    model = model.load_from_checkpoint(checkpoint_callback.best_model_path)

    # test
    test_losses, test_accs = [], []
    num_testruns = 10
    for _ in range(num_testruns):
        test_metrics = trainer.test(model, test_loader)[0]
        test_losses.append(test_metrics["test_loss"])
        test_accs.append(test_metrics["test_acc"])

    if len(test_losses) == num_testruns:
        log.info(
            "Saving test metrics to ",
            trainer.logger.log_dir + "/test_metrics.npz")
        np.savez(
            trainer.logger.log_dir + "/test_metrics.npz",
            num_testruns=num_testruns,
            test_losses=test_losses,
            test_accs=test_accs)

    # release
    hxtorch.release_hardware()


if __name__ == "__main__":
    main(get_parser().parse_args())
