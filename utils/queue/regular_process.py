from typing import Tuple
import numpy as np
from scipy.stats import norm
from scipy.stats import norm, uniform, expon

# Parameters of the queue

# The queue starts empty
INITIAL_VALUE = 0

EXPECTED_INTERARRIVAL_TIME = 0.5

EXPECTED_SERVICE_TIME = 0.45


def interarrival_times(
    size: int | Tuple[int, int], random_state: int = None
) -> np.ndarray:
    return expon(scale=EXPECTED_INTERARRIVAL_TIME).rvs(size, random_state=random_state)


def service_times(size: int | Tuple[int, int], random_state: int = None) -> np.ndarray:
    return expon(scale=EXPECTED_SERVICE_TIME).rvs(size, random_state=random_state)
