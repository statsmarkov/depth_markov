import typing
import numpy as np
from scipy.stats import norm


def tukey_depth_depth_ar_1(
    current: float, previous: float, phi: float, sigma: float
) -> float:
    _cdf = norm.cdf(current, loc=phi * previous, scale=sigma)
    return np.min([_cdf, 1 - _cdf])


def tukey_depth_dimension_1(x: float, cdf: typing.Callable) -> float:
    """
    Given a univariante CDF, it returns the Tukey's depth for point x.
    """
    _y = cdf(x)
    if np.isnan(_y):
        # If the cdf(x) is NaN, it indicates that x is not admissible for the
        # CDF we have (the CDF is degenerated).
        return 0
    return np.min([_y, 1 - _y])


def simplicial_depth_dimension_1(x: float, cdf: typing.Callable) -> float:
    """
    Given a univariante CDF, it returns the simplicial's depth for point x.
    """
    _y = cdf(x)
    if np.isnan(_y):
        # If the cdf(x) is NaN, it indicates that x is not admissible for the
        # CDF we have (the CDF is degenerated).
        return 0
    return 2 * (1 - _y)
