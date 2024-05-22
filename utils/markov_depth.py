from typing import Optional, Callable, List
from joblib import Parallel, delayed
import numpy as np
from .kernel_estimation import (
    nadaraya_watson_marginal_cdf,
    nadaraya_watson_average_marginal_cdf,
    queuing_model_marginal_cdf,
    queuing_model_average_marginal_cdf,
)
from .depth_functions import tukey_depth_dimension_1

NADARAYA_WATSON = "nadaraya_watson"
QUEUING_MODEL = "queuing_model"


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


def calculate_markov_tukey_depth_using_long_trajectory(
    trajectory: np.ndarray,
    long_trajectory: np.ndarray,
    inverse_bandwidth: Optional[float],
    marginal_cdf_estimator: Optional[str] = NADARAYA_WATSON,
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
        marginal_cdf_estimator=marginal_cdf_estimator,
    )


def calculate_markov_tukey_depth_using_sample_trajectories(
    trajectory: np.ndarray,
    sample_trajectories: List[np.ndarray],
    inverse_bandwidth: Optional[float] = None,
    marginal_cdf_estimator: Optional[str] = NADARAYA_WATSON,
) -> float:
    """
    Calculates the Markov Tukey's depth of a given one dimensional trajectory.

    Uses the long_trajectory to estimate for kernel estimation.
    """
    return _base_calculate_markov_depth_dimension_1(
        trajectory=trajectory,
        sample_trajectories=sample_trajectories,
        depth_fn=tukey_depth_dimension_1,
        inverse_bandwidth=inverse_bandwidth,
        marginal_cdf_estimator=marginal_cdf_estimator,
    )


def _base_calculate_markov_depth_dimension_1(
    trajectory: np.ndarray,
    depth_fn: Callable[[float, Callable], float],
    inverse_bandwidth: Optional[float] = None,
    long_trajectory: Optional[np.ndarray] = None,
    sample_trajectories: Optional[List[np.ndarray]] = None,
    marginal_cdf_estimator: Optional[str] = NADARAYA_WATSON,
):
    use_long_trajectory = long_trajectory is not None
    use_sample_trajectories = sample_trajectories is not None

    if not use_long_trajectory and not use_sample_trajectories:
        raise ValueError(
            "Either long_trajectory or sample_trajectories must be provided."
        )
    if use_long_trajectory and use_sample_trajectories:
        raise ValueError(
            "You can only provide one of long_trajectory or sample_trajectories."
        )

    if marginal_cdf_estimator == NADARAYA_WATSON:
        if use_long_trajectory:
            marginal_cdf_factory = nadaraya_watson_marginal_cdf
        else:
            marginal_cdf_factory = nadaraya_watson_average_marginal_cdf
    elif marginal_cdf_estimator == QUEUING_MODEL:
        if use_long_trajectory:
            marginal_cdf_factory = queuing_model_marginal_cdf
        else:
            marginal_cdf_factory = queuing_model_average_marginal_cdf
    else:
        raise ValueError(f"{marginal_cdf_estimator} is unknown.")
    if use_long_trajectory:
        kwargs = {"data": long_trajectory, "inverse_bandwidth": inverse_bandwidth}
    else:
        kwargs = {
            "samples": sample_trajectories,
            "inverse_bandwidth": inverse_bandwidth,
        }

    depths = []
    for x, x_1 in zip(trajectory[:-1], trajectory[1:]):
        marginal_cdf = marginal_cdf_factory(x=x, **kwargs)
        depths.append(depth_fn(x=x_1, cdf=marginal_cdf))
    # Early return, if one of the depths is 0, the markovian depth will also be 0
    if 0 in depths:
        return 0
    return np.exp(np.mean(np.log(depths)))


def calculate_markov_tukey_depth_for_trajectories_using_long_trajectory(
    trajectories: np.ndarray,
    long_trajectory: np.ndarray,
    inverse_bandwidth: Optional[float] = None,
    marginal_cdf_estimator: Optional[str] = NADARAYA_WATSON,
) -> List[float]:
    """
    Calculates the Markov Tukey's depth of several one dimensional trajectories.
    Uses the long_trajectory for kernel estimation.
    """
    depths = Parallel(n_jobs=-1)(
        delayed(calculate_markov_tukey_depth_using_long_trajectory)(
            trajectory=process,
            long_trajectory=long_trajectory,
            inverse_bandwidth=inverse_bandwidth,
            marginal_cdf_estimator=marginal_cdf_estimator,
        )
        for process in trajectories
    )
    return np.array(depths)


def calculate_markov_tukey_depth_for_trajectories_using_sample_trajectories(
    trajectories: np.ndarray | List[np.array],
    sample_trajectories: List[np.ndarray],
    inverse_bandwidth: Optional[float] = None,
    marginal_cdf_estimator: Optional[str] = NADARAYA_WATSON,
) -> List[float]:
    """
    Calculates the Markov Tukey's depth of several one dimensional trajectories.
    Uses the sample_trajectories for kernel estimation.
    """
    depths = Parallel(n_jobs=-1)(
        delayed(calculate_markov_tukey_depth_using_sample_trajectories)(
            trajectory=process,
            sample_trajectories=sample_trajectories,
            inverse_bandwidth=inverse_bandwidth,
            marginal_cdf_estimator=marginal_cdf_estimator,
        )
        for process in trajectories
    )
    return np.array(depths)
