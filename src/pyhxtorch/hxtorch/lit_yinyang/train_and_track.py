import argparse
import os
from pathlib import Path
import multiprocessing
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, Callback

import numpy as np
import torch
from torch.utils.data import DataLoader

import hxtorch
from hxtorch.snn.datasets.yinyang import YinYangDataset

from model import LitSNN

log = hxtorch.logger.get("hxtorch.examples.lit_yinyang.train")
hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)


class LogWeightsAndGrads(Callback):

    def __init__(self, *args, log_weights: bool = False,
                 log_grads: bool = False, **kwargs):
        super().__init__()
        self.samples = None
        self._log_weights = log_weights
        self._log_grads = log_grads

    def on_train_start(self, trainer, pl_module):
        # # plot traces
        # plot_dir = trainer.logger.log_dir + "/trace_plots"
        # Path(plot_dir).mkdir(parents=True, exist_ok=True)
        # # show hidden and output traces
        # fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(8, 15), sharex=True)
        # # forward
        # data, target = next(iter(trainer.train_dataloader))
        # self.samples = (data, target)
        # out = pl_module(data)
        # # plot traces
        # for i in range(5):
        #     ax[i][0].plot(pl_module.s_h.v_cadc[i].detach().numpy())
        #     for j in range(3):
        #         if j == target[i]:
        #             ax[i][1].plot(pl_module.v_cadc[i, :, j].detach().numpy(),
        #                           ls="solid", label=f"{j:.0f}")
        #         else:
        #             ax[i][1].plot(pl_module.v_cadc[i, :, j].detach().numpy(),
        #                           ls="dashed", label=f"{j:.0f}")
        # fig.tight_layout()
        # # show
        # plt.savefig(plot_dir + "/cadc_traces_before_training.png", dpi=300)
        # plt.close("all")

        # save weights
        if self._log_weights:
            weights_file = trainer.logger.log_dir + "/weight_log.txt"
            weights = pl_module.linear_h.weight.flatten().detach().numpy()
            indexed_weights = np.array([-1, *weights])
            with open(weights_file, "wb") as file:
                file.write(b"epoch weights\n")    
                file.close()
            with open(weights_file, "ab") as file:
                np.savetxt(file, indexed_weights.reshape(1, -1))
                file.close()

            with open(weights_file[:-4] + "_out.txt", "wb") as file:
                file.write(b"epoch weights\n")
                file.close()
            weights_o = pl_module.linear_o.weight.flatten().detach().numpy()
            indexed_weights_o = np.array([-1, *weights_o])
            with open(weights_file[:-4] + "_out.txt", "ab") as file:
                np.savetxt(file, indexed_weights_o.reshape(1, -1))
                file.close()
        if self._log_grads:
            grads_file = trainer.logger.log_dir + "/grad_log.txt"
            with open(grads_file, "wb") as grad_file:
                grad_file.write(b"epoch batch_index grad_mean grad_var"
                                b"output_grad_mean output_grad_var\n")
                grad_file.close()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        # if (epoch + 1) % 5 == 0 or (epoch + 1) == trainer.max_epochs:
        #     plot_dir = trainer.logger.log_dir + "/trace_plots"
        #     # show hidden and output traces
        #     fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(8, 15), sharex=True)
        #     # forward
        #     data, target = self.samples
        #     out = pl_module(data)
        #     # plot traces
        #     for i in range(5):
        #         ax[i][0].plot(pl_module.s_h.v_cadc[i].detach().numpy())
        #         for j in range(3):
        #             if j == target[i]:
        #                 ax[i][1].plot(pl_module.v_cadc[i, :, j].detach().numpy(),
        #                             ls="solid", label=f"{j:.0f}")
        #             else:
        #                 ax[i][1].plot(pl_module.v_cadc[i, :, j].detach().numpy(),
        #                             ls="dashed", label=f"{j:.0f}")
        #     fig.tight_layout()
        #     # save
        #     plt.savefig(plot_dir + "/cadc_traces_epoch_"
        #                 + f"{epoch:.0f}.png", dpi=300)
        #    plt.close("all")
        if self._log_weights:
            weights_file = trainer.logger.log_dir + "/weight_log.txt"
            weights = pl_module.linear_h.weight.flatten().detach().numpy()
            indexed_weights = np.array([epoch, *weights])
            with open(weights_file, "ab") as file:
                np.savetxt(file, indexed_weights.reshape(1, -1))
                file.close()
            # output weights
            weights_o = pl_module.linear_o.weight.flatten().detach().numpy()
            indexed_weights_o = np.array([epoch, *weights_o])
            with open(weights_file[:-4] + "_out.txt", "ab") as file:
                np.savetxt(file, indexed_weights_o.reshape(1, -1))
                file.close()

    def on_train_batch_end(self, trainer, pl_module, output, batch, batch_idx, dataloader_idx):
        epoch = pl_module.current_epoch
        if self._log_grads:
            grads_file = trainer.logger.log_dir + "/grad_log.txt"
            grads = pl_module.linear_h.weight.grad
            grads_o = pl_module.linear_o.weight.grad
            indexed_grads = np.array([
                epoch,
                batch_idx,
                grads.mean().item(),
                grads.var().item(),
                grads_o.mean().item(),
                grads_o.var().item()])
            with open(grads_file, "ab") as grad_file:
                np.savetxt(grad_file, indexed_grads.reshape(1, -1))
                grad_file.close()


class RescaleOutputWeights(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        with torch.no_grad():
            # clamp to maximum possible weight (when scaled to HW)
            limit = 63. / pl_module.linear_h.weight_transform.keywords["scale"]
            pl_module.linear_h.weight.data.clamp_(-limit, limit)
            # scale output weights to fit HW range
            scale = min(63. / np.abs(pl_module.linear_o.weight.cpu().detach()).max(),
                        pl_module.linear_h.weight_transform.keywords["scale"])
            pl_module.log("output_weight_scale", scale)
            # reset partial
            pl_module.linear_o.weight_transform.keywords["scale"] = scale


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser with all the options.
    """
    parser = argparse.ArgumentParser(
        description="hxtorch spiking YinYang example using pytorch lightning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    # dataset
    parser.add_argument("--trainset-size", type=int, default=4096)
    parser.add_argument("--testset-size", type=int, default=512)
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="<num samples>",
        help="input batch size for training")
    # hw or mock
    parser.add_argument(
        "--calibration-path", type=str, metavar="<path>",
        default=os.getenv("HXTORCH_CALIBRATION_PATH"),
        help="path to custom calibration to use instead of latest nightly")
    # experiment training
    parser.add_argument("--epochs", type=int, default=10)
    # model
    parser = LitSNN.add_model_specific_args(parser)
    # load from checkpoint
    parser.add_argument("--from-ckpt-file", type=str, default=None)
    # save dir
    parser.add_argument("--save-dir", type=str, default="lightning_logs/")

    # evaluate
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--weight-file", type=str, default=None)
    parser.add_argument("--grad-file", type=str, default=None)

    return parser


def main(args: argparse.Namespace) -> None:
    if not args.evaluate:
        # load model and args from checkpoint
        if args.from_ckpt_file is not None:
            ckpt_file = args.from_ckpt_file
            log.info(f"Discarding parsed arguments and using "
                    + f"{os.path.dirname(os.path.dirname(args.from_ckpt_file)) + '/hparams.yaml'}")
            epochs = args.epochs
            import yaml
            hparams_filename = os.path.dirname(os.path.dirname(args.from_ckpt_file)) + '/hparams.yaml'
            with open(hparams_filename, "r") as f:
                hparams_dict = yaml.load(f, Loader=yaml.SafeLoader)
                f.close()
            args = argparse.Namespace(**hparams_dict)
            args.from_ckpt_file = ckpt_file
            args.epochs = epochs

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
        testset = YinYangDataset(size=args.testset_size, seed=41)

        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True,
            num_workers=min(8, multiprocessing.cpu_count()))
        val_loader = DataLoader(
            testset, batch_size=args.batch_size, shuffle=False,
            num_workers=min(8, multiprocessing.cpu_count()))
        log.info("Finished loading datasets and dataloaders.")

        # model
        model = LitSNN(**vars(args))
        # lr momnitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        # trainer
        # TODO: if starting from ckpt, enable use of new learning rate
        trainer = pl.Trainer(max_epochs=args.epochs,
                             log_every_n_steps=20,
                             logger=CSVLogger(save_dir=args.save_dir),
                             callbacks=[lr_monitor,
                                        LogWeightsAndGrads(
                                            log_weights=True,
                                            log_grads=True),
                                        RescaleOutputWeights()],
                            resume_from_checkpoint=args.from_ckpt_file)

        # train
        trainer.fit(model, train_loader, val_loader)
        # test
        trainer.test(model, val_loader)

        # release
        hxtorch.release_hardware()

    # TODO: Make this a separate script
    else:
        data = np.loadtxt(args.weight_file, skiprows=1)
        epoch = data[:, 0]
        weights = data[:, 1:]
        # weights = torch.clamp(38. * torch.tensor(weights), -63., 63.).detach().numpy()

        fig, ax = plt.subplots(nrows=2, figsize=(8, 10))

        # # round and clip weights
        # scale = 50.
        # weights = np.round(np.clip(scale * weights, -63., 63.)) / scale

        ax[0].plot(epoch, weights[:, :100],
                   alpha=0.5)
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("weight")

        # if args.mock:
        ax[1].hist(weights[-1], bins=40, alpha=0.3,
                   label=f"epoch {epoch[-1]:.0f}", density=True)
        ax[1].hist(weights[0], bins=40, alpha=0.3,
                   label=f"epoch {epoch[0]:.0f}", density=True)
        # else:
        #     ax[1].hist(weights[-1], bins=40, alpha=0.3, range=(-63., 63.),
        #             label=f"after training", density=True)
        #     ax[1].hist(weights[0], bins=40, alpha=0.3, range=(-63., 63.),
        #             label=f"before training", density=True)
        ax[1].set_xlabel("weight")
        ax[1].set_ylabel("count")
        ax[1].legend()

        fig.tight_layout()
        plt.savefig(os.path.dirname(args.weight_file) + "/weights.png", dpi=300)


        data = np.loadtxt(args.weight_file[:-4] + "_out.txt", skiprows=1)
        epoch = data[:, 0]
        weights = data[:, 1:]
        # weights = torch.clamp(38. * torch.tensor(weights), -63., 63.).detach().numpy()

        fig, ax = plt.subplots(nrows=2, figsize=(8, 10))

        # # round and clip weights
        # scale = 50.
        # weights = np.round(np.clip(scale * weights, -63., 63.)) / scale

        ax[0].plot(epoch, weights[:, :100],
                   alpha=0.5)
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("weight")

        # if args.mock:
        ax[1].hist(weights[-1], bins=40, alpha=0.3,
                   label=f"epoch {epoch[-1]:.0f}", density=True)
        ax[1].hist(weights[0], bins=40, alpha=0.3,
                   label=f"epoch {epoch[0]:.0f}", density=True)
        # else:
        #     ax[1].hist(weights[-1], bins=40, alpha=0.3, range=(-63., 63.),
        #             label=f"after training", density=True)
        #     ax[1].hist(weights[0], bins=40, alpha=0.3, range=(-63., 63.),
        #             label=f"before training", density=True)
        ax[1].set_xlabel("weight")
        ax[1].set_ylabel("count")
        ax[1].legend()

        fig.tight_layout()
        plt.savefig(os.path.dirname(args.weight_file) + "/weights_o.png", dpi=300)


        if args.grad_file is not None:
            grad_data = np.loadtxt(args.grad_file, skiprows=1)
            epoch = grad_data[:, 0]
            batch_idx = grad_data[:, 1]
            grad_mean = grad_data[:, 2]
            grad_var = grad_data[:, 3]

            # mask where gradient is greater 0 (i.e. where a gradient
            # exists for at least one timestep)
            grad_mask = (grads > 0.).sum(axis=0) > 0.

            fig, ax = plt.subplots(figsize=(8, 6))
            num = 40
            changed_weights_idx = np.arange(num)[grad_mask[:num]]
            unchanged_weights_idx = np.arange(num)[~grad_mask[:num]]
            ax.plot(step * 16 / 4096, weights[:, unchanged_weights_idx],
                    alpha=0.5, color="black")
            ax.plot(step * 16 / 4096, weights[:, changed_weights_idx],
                    alpha=0.5)
            ax.set_title(f"Weights over epochs (#weights: {weights.shape[1]:.0f}, "
                         + f"#changed: {grad_mask.sum():.0f}, "
                         + f"#unchanged: {(~grad_mask).sum():.0f})")
            ax.set_xlabel("epoch")
            ax.set_ylabel("weight")

            fig.tight_layout()
            plt.savefig(os.path.dirname(args.weight_file) + "/weights_marked.png", dpi=300)


if __name__ == "__main__":
    main(get_parser().parse_args())
