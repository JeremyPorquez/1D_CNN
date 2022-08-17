import numpy as np
from typing import Callable


def generate_batch(func: Callable, size=10000):
    """
    Generates batch for training
    :param func: Must be a callable that returns a tuple with the first element as the train and second element as target value.
    :param size:
    :return:
    """
    n_points_X, n_points_y = map(lambda x: len(x), func())
    X = np.empty((size, n_points_X, 1))
    y = np.zeros((size, n_points_y))

    for i in range(size):
        X[i, :, 0], y[i, :] = func()
    return X, y
