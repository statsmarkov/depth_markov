from typing import Tuple
import numpy as np
from scipy.stats import norm, uniform, expon, norm
from .regular_process import EXPECTED_INTERARRIVAL_TIME, EXPECTED_SERVICE_TIME


# Could represent a broken machine.
# Tasks take, on average, 5 times more time to complete.
EXPECTED_SHOCK_SERVICE_TIME = 5 * EXPECTED_SERVICE_TIME


def shock_service_time(
    size: int | Tuple[int, int], random_state: int = None
) -> np.ndarray:
    return expon(scale=EXPECTED_SHOCK_SERVICE_TIME).rvs(size, random_state=random_state)


# Increase in the velocity of the arrivals
def reduced_interarrival_times(
    size: int | Tuple[int, int], random_state: int = None
) -> np.ndarray:
    return expon(scale=0.2 * EXPECTED_INTERARRIVAL_TIME).rvs(
        size, random_state=random_state
    )


# The average time to finish a task is now slightly higher than
# the interarrival times, making the queue non-recurrent
def increasing_service_times(
    size: int | Tuple[int, int], random_state: int = None
) -> np.ndarray:
    return (
        1.1
        * EXPECTED_INTERARRIVAL_TIME
        * uniform(loc=0, scale=2).rvs(size, random_state=random_state)
    )


# The interarrivals follow a geometric progression
def deterministic_geometric_arrivals(
    size: int | Tuple[int, int], random_state: int = None
) -> np.ndarray:
    p = 0.5
    if isinstance(size, int):
        return p ** (np.arange(1, size + 1))
    else:
        return np.array([p ** (np.arange(1, size[1] + 1)) for _ in range(size[0])])
