import numpy as np
from scipy.stats import norm

# Parameters of the ARCH(1) process
INITIAL_VALUE = 0.5


def mean_fn(x):
    return 1 / (1 + np.exp(-x))


def volatility_fn(x):
    return norm.pdf(x + 1.2) + 1.5 * norm.pdf(x - 1.2)
