"""
YinYangDataset class from
https://github.com/lkriener/yin_yang_data_set/blob/master/dataset.py
with minor changes.
"""
from typing import Optional, Tuple
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


# pylint: disable=invalid-name, unused-variable
class YinYangDataset(Dataset):
    """ YinYang dataset """

    def __init__(self, r_small: float = 0.1, r_big: float = 0.5,
                 size: int = 1000, seed: int = 42,
                 transform: Optional[torch.nn.Module] = None) -> None:
        """
        Instantiate the YinYang dataset. This dataset provides datapoints on a
        2-dimensional plane within a yin-yang sign. Each data point is assigned
        to one of three classes: The eyes, the yin or the yang.

        :param r_small: The radius of the eyes in the yin-yang sign.
        :param r_big: The radius of the whole sign.
        :param size: The size of the dataset, i.e. number of data points within
            the sign.
        :param seed: Random seed.
        :param transform: An optional transformation applied to the returned
            samples.
        """
        super().__init__()

        # using a numpy RNG to allow compatibility to
        # other deep learning frameworks
        self.rng = np.random.RandomState(seed)

        # radii
        if 2 * r_small >= r_big:
            raise ValueError(
                "Argument 'r_small' must not be larger than 'r_big' / 2")
        self.r_small = r_small
        self.r_big = r_big

        # transformation
        self.transform = transform

        # values and corresponding classes
        self.__vals = []
        self.__cs = []
        self.class_names = ['yin', 'yang', 'dot']

        # create data points with classes
        for i in range(size):
            # keep num of class instances balanced by using rejection sampling
            # choose class for this sample
            goal_class = self.rng.randint(3)
            x, y, c = self.get_sample(goal=goal_class)

            # add mirrod axis values
            x_flipped = 1. - x
            y_flipped = 1. - y

            # append
            self.__vals.append(np.array([x, y, x_flipped, y_flipped]))
            self.__cs.append(c)

    def get_sample(self, goal: Optional[int] = None) -> Tuple[float, ...]:
        """
        Sample one data point from the yin-yang sign with goal class `goal`. If
        `goal` is None any sample of any class is returned.

        :param goal: The target class of the sample to return. If None is
            given, any sample regardless of its class is retuned.

        :returns: Returns a tuple (x coordiante, y coordinate, class), where
            the coordinates are on the xy-plane.
        """
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample (x, y) coordinates
            x, y = self.rng.rand(2) * 2. * self.r_big

            # check if within yin-yang circle
            if np.sqrt((x - self.r_big)**2 + (y - self.r_big)**2) > self.r_big:
                continue

            # check if they have the same class as the goal for this sample
            c = self.which_class(x, y)

            # check if class is accepted
            if goal is None or c == goal:
                found_sample_yet = True
                break

        return x, y, c

    def which_class(self, x: float, y: float) -> int:
        """
        Assign a sample on the xy-plane with coordinates (x, y) to its class.

        :param x: The x-coordinate.
        :param y: The y-coordinate.

        :returns: Am integer indicating the samples class.
        """
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big

        # check whether sample is in yin
        is_yin = criterion1 or criterion2 or criterion3

        # check whether sample is in eyes
        is_circles = d_right < self.r_small or d_left < self.r_small

        if is_circles:
            return 2

        return int(is_yin)

    def dist_to_right_dot(self, x: int, y: int) -> float:
        """
        Compute the distance to the right dot.

        :param x: The x-coordinate.
        :param y: The y-coordinate.

        :returns: Returns the distance to the right dot.
        """
        return np.sqrt((x - 1.5 * self.r_big)**2 + (y - self.r_big)**2)

    def dist_to_left_dot(self, x: int, y: int) -> float:
        """
        Compute the distance to the left dot.

        :param x: The x-coordinate.
        :param y: The y-coordinate.

        :returns: Returns the distance to the left dot.
        """
        return np.sqrt((x - 0.5 * self.r_big)**2 + (y - self.r_big)**2)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """
        Get an item from the dataset.

        :param index: Index of the item in the dataset.

        :returns: Returns a tuple (sample, target), where sample is an array
            with values (x, y, 1 - x, 1 - y).
        """
        sample, target = (self.__vals[index].copy(), self.__cs[index])

        # apply transformation
        if self.transform:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.__cs)
