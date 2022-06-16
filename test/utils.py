import numpy as np
from random import choices


def _func(x: float, y: float):
    return np.sin(2 * np.sqrt(x ** 2 + y ** 2)) / (np.sqrt(x ** 2 + y ** 2) + 0.001)


def make_initial_data():
    xs = choices(np.cumsum(0.01 * np.ones(1001)) - 5.01, k=100)
    ys = choices(np.cumsum(0.01 * np.ones(1001)) - 5.01, k=100)
    func = np.vectorize(_func)
    X = list(zip(xs, ys))
    y = func(xs, ys)
    return X, y