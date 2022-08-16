import argparse
import os
import multiprocessing

from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import hxtorch
from hxtorch.snn.datasets.yinyang import YinYangDataset

from model import LitSNN

import matplotlib.pyplot as plt

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
    # model
    parser = LitSNN.add_model_specific_args(parser)
    # load from checkpoint
    parser.add_argument("--from-ckpt-file", type=str, default=None)

    return parser


def main(args: argparse.Namespace) -> None:

    # load from checkpoint
    if args.from_ckpt_file is not None:
        ckpt_file = args.from_ckpt_file
        log.info(f"Loading model from checkpoint at {args.from_ckpt_file} ...")
        model = LitSNN.load_from_checkpoint(args.from_ckpt_file)
        log.info(f"Model loaded. Discarding parsed arguments and using "
                 + f"{os.path.dirname(args.from_ckpt_file) + 'hparams.yaml'}")
        args = argparse.Namespace(**model.hparams)

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
    testset = YinYangDataset(size=args.testset_size, seed=41)

    val_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=min(8, multiprocessing.cpu_count()))
    log.info("Finished loading datasets and dataloaders.")

    # model
    # model = LitSNN(**vars(args))

    colors = np.array(["C1", "C2", "C3"])
    fig, ax = plt.subplots(figsize=(5, 5))
    n_correct = 0
    for data, target in tqdm(val_loader):
        # forward
        max_traces = model(data)
        pred = max_traces.argmax(dim=1)
        # store correctly classified data and targets
        mask = pred.eq(target.view_as(pred))
        correct_data = data[mask].detach().numpy()
        correct_target = target[mask].detach().numpy()
        # count correct classifications
        n_correct += mask.sum().item()
        # store wrongly classified data
        false_data = data[~mask].detach().numpy()

        # plot classifications
        ax.scatter(x=correct_data[:, 0], y=correct_data[:, 1],
                   c=colors[correct_target], linewidths=1.,
                   edgecolors="black", alpha=0.4)
        ax.scatter(x=false_data[:, 0], y=false_data[:, 1],
                   c="black", marker="x", alpha=0.7)
    # accuracy
    test_acc = n_correct / len(val_loader.dataset)
    ax.set_title(f"Test accuracy {1e2 * test_acc:.2f} %")
    # save
    plt.savefig(os.path.dirname(os.path.dirname(ckpt_file))
                + "/testset_calssifications.png", dpi=300)

    # release
    hxtorch.release_hardware()


if __name__ == "__main__":
    main(get_parser().parse_args())
