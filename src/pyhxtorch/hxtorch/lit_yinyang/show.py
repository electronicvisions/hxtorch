import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser with all the options.
    """
    parser = argparse.ArgumentParser(
        description="show trianing results of spiking hxtorch snn on YinYang",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--loglog", action="store_true", default=False)

    return parser


def main(args: argparse.Namespace) -> None:
    metrics = pd.read_csv(f"{args.data_dir}/metrics.csv")
    train_metrics = metrics.dropna(subset=["train_loss"])
    val_metrics = metrics.dropna(subset=["val_loss"])

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    # plot loss over epochs
    ax1.grid()
    ax1.plot(np.linspace(1. * train_metrics.epoch.tolist()[0],
                         1. * train_metrics.epoch.tolist()[-1],
                         train_metrics.train_loss.shape[0]),
            train_metrics["train_loss"], label="train_loss", alpha=0.4)
    ax1.plot(val_metrics["epoch"] + 1, val_metrics["val_loss"], label="val loss")
    ax1.set_ylabel("loss")
    # plot hidden spike rate
    ax1sec = ax1.twinx()
    rates = metrics.dropna(
        subset=["hidden rate"]).groupby("epoch", as_index=False).mean()
    ax1sec.plot(
        rates["epoch"],
        rates["hidden rate"],
        label="hidden rate",
        alpha=0.6,
        c="C2")
    ax1sec.set_ylabel
    ax1sec.xaxis.label.set_color("C2")
    ax1sec.spines["right"].set_edgecolor("C2")
    ax1sec.tick_params(axis="y", colors="C2")
    ax1sec.legend(loc="upper center")

    if args.loglog:
        ax1.loglog()
    ax1.legend(loc="upper left")
    # ax1.spines.right.set_visible(False)
    # ax1.spines.top.set_visible(False)

    # plot accuracy on validation set over epochs
    ax2.grid()
    ax2.plot(val_metrics["epoch"] + 1, val_metrics["val_acc"], label="val acc")
    final_acc = val_metrics["val_acc"].to_list()[-1]
    ax2.axhline(y=final_acc, ls="dashed", c="black", alpha=0.7,
                label=f"final acc. {100 * final_acc:.2f} %")
    ax2.set_ylabel("accuracy")
    ax2.set_xlabel("epochs")
    ax2.set_xlim(0.5, np.max(val_metrics["epoch"]) + 1.5)
    ax2.legend()
    if args.loglog:
        ax2.loglog()
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)

    fig.tight_layout()
    plt.savefig(f"{args.data_dir}/plot_val_data.png", dpi=150)


if __name__ == "__main__":
    main(get_parser().parse_args())
