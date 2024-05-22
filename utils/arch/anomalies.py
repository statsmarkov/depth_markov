import numpy as np
from scipy.stats import norm


# Shock
def shock_mean_fn(x):
    return 5 * x


def shock_volatility_fn(x):
    return np.sqrt(np.abs(x))


# Perturbed mean
def perturbed_mean_fn(x):
    return 1 / (2 + np.exp(-x))


# Increasing volatility
def increasing_volatility_fn(x):
    return 0.5 * np.sqrt(x**2 + 1)


# Constant mean
def constant_mean(x):
    return 2


# Extra examples


# Perturbed mean
def extra_perturbed_mean_fn(x):
    return 1 / (4 + np.exp(-x))


def double_volatility_fn(x):
    return 2 * norm.pdf(x + 1.2) + 1.5 * norm.pdf(x - 1.2)


# Constant volatility
def deterministic_volatility(x):
    return 1
