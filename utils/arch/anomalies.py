import numpy as np


# Shock
def shock_mean_fn(x):
    return 5 * x


def shock_volatility_fn(x):
    return np.sqrt(x)


# Perturbed mean
def perturbed_mean_fn(x):
    return 1 / (2 + np.exp(-x))


# Increasing volatility
def increasing_volatility_fn(x):
    return 0.5 * np.sqrt(x**2 + 1)


# Constant mean
def constant_mean(x):
    return 2
