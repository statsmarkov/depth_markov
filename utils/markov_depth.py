from typing import Optional, Callable, List
from joblib import Parallel, delayed
import numpy as np
from .kernel_estimation import nadaraya_watson_marginal_cdf
from .depth_functions import tukey_depth_dimension_1


def calculate_markov_depth(
    trajectory: np.ndarray, depth_function: Callable, **kwargs
) -> float:
    """
    Calculates the Markov depth of a given trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory to calculate the depth of.
    depth_function : typing.Callable
        The depth function to use to calculate the depth.
    **kwargs : typing.Dict
        Extra parameters to pass to the depth function.

    Returns
    -------
    float
        The depth of the trajectory.
    """
    depths = []
    for i in range(1, len(trajectory)):
        depths.append(depth_function(trajectory[i], trajectory[i - 1], **kwargs))
    return np.exp(np.mean(np.log(depths)))


def calculate_markov_tukey_depth(
    trajectory: np.ndarray,
    long_trajectory: np.ndarray,
    inverse_bandwidth: Optional[float],
) -> float:
    """
    Calculates the Markov Tukey's depth of a given one dimensional trajectory.

    Uses the long_trajectory to estimate for kernel estimation.
    """
    return _base_calculate_markov_depth_dimension_1(
        trajectory=trajectory,
        long_trajectory=long_trajectory,
        depth_fn=tukey_depth_dimension_1,
        inverse_bandwidth=inverse_bandwidth,
    )


def _base_calculate_markov_depth_dimension_1(
    trajectory: np.ndarray,
    long_trajectory: np.ndarray,
    depth_fn: Callable[[float, Callable], float],
    inverse_bandwidth: Optional[float],
):
    depths = []
    for i in range(1, len(trajectory)):
        x = trajectory[i - 1]
        x_1 = trajectory[i]
        marginal_cdf = nadaraya_watson_marginal_cdf(
            x=x, data=long_trajectory, inverse_bandwidth=inverse_bandwidth
        )
        depths.append(depth_fn(x=x_1, cdf=marginal_cdf))
    # Early return, if one of the depths is 0, the markovian depth will also be 0
    if 0 in depths:
        return 0
    return np.exp(np.mean(np.log(depths)))


def calculate_markov_tukey_depth_for_trajectories(
    trajectories: np.ndarray,
    long_trajectory: np.ndarray,
    inverse_bandwidth: Optional[float],
) -> List[float]:
    """
    Calculates the Markov Tukey's depth of several given one dimensional trajectories.
    Uses the long_trajectory to estimate for kernel estimation.
    """
    depths = Parallel(n_jobs=-1)(
        delayed(calculate_markov_tukey_depth)(
            trajectory=process,
            long_trajectory=long_trajectory,
            inverse_bandwidth=inverse_bandwidth,
        )
        for process in trajectories
    )
    return np.array(depths)
