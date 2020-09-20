"""
Training example for the MNIST handwritten-digits dataset using a host machine
with the BrainScaleS-2 ASIC in the loop.
"""
import argparse
import os
from pathlib import Path
import torch
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import hxtorch
import hxtorch.nn as hxnn

log = hxtorch.logger.get("hxtorch.examples.mnist")


class Model(torch.nn.Module):
    """
    Simple CNN model to classify written digits from the MNIST database.

    Model topology:
        - Conv2d with 10x10 kernel, stride 5
        - Linear layer with 128 hidden neurons
    """

    def __init__(self, mock: bool):
        """
        :param mock: Whether to use a software simulation instead of the ASIC.
        """
        super().__init__()
        self.features = torch.nn.Sequential(
            hxnn.Conv2d(
                # same parameters as in vanilla PyTorch
                in_channels=1, out_channels=20, kernel_size=10, stride=5,
                bias=False, padding=1,
                # hardware specific parameters:
                num_sends=3,           # scales the hardware-gain, will be
                                       # adjusted automatically if set to None.
                wait_between_events=2, # specifies wait time between two inputs
                                       # lower values may lead to saturation
                                       # effects in the drivers.
                mock=mock,             # enables simulation-mode.
            ),
            hxnn.ConvertingReLU(
                shift=1,  # shifts the output by 1 bit, i.e. divides it by 2.
                mock=mock,
            ),
        )
        self.classifier = torch.nn.Sequential(
            hxnn.Linear(
                in_features=5 * 5 * 20,
                out_features=128,
                bias=False,
                num_sends=2,
                wait_between_events=2,
                mock=mock,
            ),
            hxnn.ConvertingReLU(mock=mock, shift=1),
            hxnn.Linear(
                128, 10, bias=False,
                num_sends=3,
                wait_between_events=2,
                mock=mock
            )
        )

    def forward(self, *x):
        x = self.features(x[0])
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def init(calibration_path: str, mock: bool, mock_disable_noise: bool):
    """
    Initialize hxtorch connection and load calibration.

    :param calibration_path: Path of custom calibration
    :param mock: Whether to simulate the hardware
    :param mock_disable_noise: Disable noise in mock mode
    """
    if mock:
        mock_parameter = hxtorch.MockParameter(
            gain=0.002,
            noise_std=0. if mock_disable_noise else 2.
        )
        log.info(f"Initialize mock mode with {mock_parameter}")
    else:
        log.info("Initialize with BrainScaleS-2 ASIC")
        if calibration_path:
            log.info(f"Apply calibration from: '{args.calibration_path}'")
            hxtorch.init(hxtorch.CalibrationPath(args.calibration_path))
        else:
            log.info("Apply latest nightly calibration")
            hxtorch.init_hardware()  # defaults to a nightly default calib
        mock_parameter = hxtorch.measure_mock_parameter()

    # set mock parameter used in mock mode and backward pass
    hxtorch.set_mock_parameter(mock_parameter)


def train(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer):
    """
    Train the model.

    :param model: The model
    :param loader: Data loader containing the train data set
    :param optimizer: Optimizer that handles the weight updates
    """
    model.train()
    pbar = tqdm(total=len(loader), unit="batch", ncols=99, postfix=" " * 11)
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update()
    pbar.close()


def test(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> float:
    """
    Test the model.

    :param model: The model to test
    :param loader: Data loader containing the test data set
    :returns: Test accuracy
    """
    log.info("Evaluation on test set")
    model.eval()
    loss = 0
    n_correct = 0
    n_total = len(loader.dataset)
    pbar = tqdm(total=len(loader), unit="batch", leave=False, ncols=99)
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            loss += torch.nn.functional.cross_entropy(output, target,
                                                      reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            n_correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.update()
    pbar.close()
    loss /= n_total
    accuracy = n_correct / n_total

    log.info(
        f"Average loss: {loss:.4f}, "
        f"Accuracy: {n_correct}/{n_total} ({100. * accuracy:.1f}%)")
    return accuracy


def shrink_dataset(dataset: torch.utils.data.Dataset, fraction: float):
    """ Returns a fraction of the original dataset  """
    new_length = int(fraction * len(dataset))
    return torch.utils.data.Subset(
        dataset, torch.randperm(len(dataset)).tolist()[:new_length])


def main(args: argparse.Namespace):
    """
    The main experiment function.

    :param args: Command-line arguments
    """
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path \
        else None
    if checkpoint_path and not checkpoint_path.is_dir():
        raise OSError(
            f"Checkpoint directory does not exist: '{checkpoint_path}'")

    data_path = Path(args.data_path).resolve()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 31),  # scale to input range of BSS-2
    ])

    train_data = datasets.MNIST(data_path, train=True, transform=transform,
                                download=True)
    test_data = datasets.MNIST(data_path, train=False, transform=transform)

    if args.dataset_fraction < 1:
        train_data = shrink_dataset(train_data, args.dataset_fraction)
        test_data = shrink_dataset(test_data, args.dataset_fraction)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=args.test_batch_size)

    model = Model(mock=args.mock)
    log.info(f"Used model:\n{model}")
    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=args.gamma)

    init(mock=args.mock, mock_disable_noise=args.mock_disable_noise,
         calibration_path=args.calibration_path)

    accuracy = test(model, test_loader)
    for epoch in range(1, args.epochs + 1):
        log.info(f"Train epoch {epoch}")
        train(model, train_loader, optimizer)
        accuracy = test(model, test_loader)
        scheduler.step()

        if checkpoint_path:
            save_path = checkpoint_path.joinpath(f"state_{epoch}.pth")
            log.info(f"Save model state to '{save_path}'")
            torch.save(model.state_dict(), save_path)

    hxtorch.release_hardware()
    return accuracy


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser with all the options.
    """
    parser = argparse.ArgumentParser(
        description="hxtorch MNIST example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--batch-size", type=int, default=100, metavar="<num samples>",
        help="input batch size for training")
    parser.add_argument(
        "--test-batch-size", type=int, default=500, metavar="<num samples>",
        help="input batch size for testing")
    parser.add_argument(
        "--epochs", type=int, default=10, metavar="<num epochs>",
        help="number of epochs to train")
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="<learning rate>",
        help="learning rate")
    parser.add_argument(
        "--gamma", type=float, default=0.7, metavar="<gamma>",
        help="Learning rate decay")
    parser.add_argument(
        "--mock", action="store_true", default=False,
        help="enable mock mode")
    parser.add_argument(
        "--mock-disable-noise", action="store_true", default=False,
        help="disable artificial noise in mock mode")
    parser.add_argument(
        "--seed", type=int, default=0x5EEED, metavar="<seed>",
        help="seed used for initialization")
    parser.add_argument(
        "--data-path", type=str, metavar="<path>",
        default=os.getenv("HXTORCH_DATASETS_PATH", default="."),
        help="folder containing dataset in 'MNIST' subfolder, "
             "will be downloaded if empty")
    parser.add_argument(
        "--checkpoint-path", type=str, default="", metavar="<path>",
        help="if specified, the model state will be saved after every epoch")
    parser.add_argument(
        "--resume-from", type=str, default="", metavar="<path>",
        help="path of model state to resume from")
    parser.add_argument(
        "--calibration-path", type=str, metavar="<path>",
        default=os.getenv("HXTORCH_CALIBRATION_PATH"),
        help="path to custom calibration to use instead of latest nightly")
    parser.add_argument(
        "--dataset-fraction", type=float, default=1., metavar="<fraction>",
        help="The fraction of the dataset to use in the experiment")
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
