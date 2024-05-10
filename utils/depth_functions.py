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
    return np.min([cdf(x), 1 - cdf(x)])


def simplicial_depth_dimension_1(x: float, cdf: typing.Callable) -> float:
    """
    Given a univariante CDF, it returns the simplicial's depth for point x.
    """
    return 2 * cdf(x)(1 - cdf(x))
