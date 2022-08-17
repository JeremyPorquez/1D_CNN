import numpy as np


def generate_random_params(max_features=5, min_features=0):
    """
    generates a random spectrum, without NRB.
    output:
        params =  matrix of parameters. each row corresponds to the [amplitude, resonance, linewidth] of each generated feature (n_lor,3)
    """
    min_features = min_features if min_features > 0 else 0
    n_lor = np.random.randint(
        min_features, max_features + 1)  # take a random int between 1 and max_features

    def get_random_params():
        # take a random number between 0 and 1 with shape n_lor
        x0 = np.random.uniform(0.2, 0.8, n_lor)
        # take a random number between 0 and 1 with shape n_lor
        A = np.random.uniform(0.0, 1, n_lor)
        # take a random number between 0.001 and 0.008 with shape n_lor
        sigma = np.random.uniform(0.01, 0.1, n_lor)
        return x0, A, sigma

    x0, A, sigma = get_random_params()

    params = np.c_[x0, A, sigma]
    return params
