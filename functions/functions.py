import numpy as np


def _extract_params(params, x):
    if len(x.shape) < 2:
        x = x[:, np.newaxis]
    x0 = params[:, 0]
    A = params[:, 1]
    sigma = params[:, 2]
    return A, sigma, x, x0


def gaussian(params, x: np.ndarray):
    """
    builds the normalized chi3 complex vector
    inputs:
        params: (n_lor, 3)
    outputs
        chi3: complex, (n_points, )
    """

    A, sigma, x, x0 = _extract_params(params, x)
    result = np.sum(A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)), axis=1)
    return result / np.max(result)


def laplacian(params, x: np.ndarray):
    """
    builds the normalized chi3 complex vector
    inputs:
        params: (n_lor, 3)
    outputs
        chi3: complex, (n_points, )
    """

    A, sigma, x, x0 = _extract_params(params, x)
    result = np.sum(A * np.exp(-abs(x - x0) / sigma), axis=1)
    return result / np.max(result)
