from random import choices
from typing import Tuple, List

import numpy as np


def _func(x: float, y: float) -> float:
    """
    The given function
    :param x: input variable
    :param y: input variable
    :return: the calculated value
    """
    return np.sin(2 * np.sqrt(x ** 2 + y ** 2)) / (np.sqrt(x ** 2 + y ** 2) + 0.001)


def make_initial_data() -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Creates the set of the initial data
    :return: the sets of the initial data to train the NN
    """
    xs: List[float] = choices((np.cumsum(0.001 * np.ones(10001)) - 5.01) / 5, k=30)
    ys: List[float] = choices((np.cumsum(0.001 * np.ones(10001)) - 5.01) / 5, k=30)
    func = np.vectorize(_func)
    X: List[Tuple[float, float]] = list(zip(xs, ys))
    y: List[float] = func(xs, ys)
    return X, y