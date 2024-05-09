import numpy as np
from typing import Callable


def is_anomalous(
    trajectory: np.array, depth_function: Callable, threshold: float, **kwargs
) -> bool:
    """
    Check if a trajectory is anomalous according to a given depth function.
    """
    return depth_function(trajectory, **kwargs)
