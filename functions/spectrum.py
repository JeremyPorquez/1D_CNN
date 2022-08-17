from scipy.interpolate import interp1d
from .params import generate_random_params
from . import functions
import numpy as np


class SpectralData:
    simulated = None
    clean = None
    parameters = None
    SNR = None
    noise = None


def generate_spectrum(kind="gaussian", n_points=256, max_features=5, min_features=0):
    """
    :param kind: "gaussian" or "laplacian"
    :type kind: str
    :param n_points: number of points to generate, default 256
    :type n_points: int
    :param max_features: number of maximum spectral features
    :type max_features: int
    :param min_features: minimum number of spectral features
    :type min_features: int
    :return: signal + noise, clean signal, params, SNR
    :rtype: SpectralData
    """
    params = generate_random_params(max_features, min_features)

    x = np.linspace(0, 1, np.random.randint(64, 256))
    if kind == "laplacian":
        f = functions.laplacian
    else:
        f = functions.gaussian
    _spec = f(params, x[:, np.newaxis])
    noise = np.random.randn(len(x)) * np.random.uniform(0.0, 0.1)

    f_spec = interp1d(x, _spec)
    f_noise = interp1d(x, noise)
    new_x = np.linspace(0, 1, n_points)
    spec = f_spec(new_x)
    noise = f_noise(new_x)

    signal = spec + noise
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    SNR = np.max(spec) / np.max(noise)
    result = SpectralData()
    result.simulated = signal
    result.clean = spec
    result.parameters = params
    result.SNR = SNR
    result.noise = noise
    return result
