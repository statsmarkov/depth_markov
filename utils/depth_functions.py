import numpy as np
from scipy.stats import norm


def tukey_depth_depth_ar_1(
    current: float, previous: float, phi: float, sigma: float
) -> float:
    _cdf = norm.cdf(current, loc=phi * previous, scale=sigma)
    return np.min([_cdf, 1 - _cdf])
