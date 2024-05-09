import typing
import numpy as np


# TODO: Make the processing in parallel
def calculate_markov_depth(
    trajectory: np.ndarray, depth_function: typing.Callable, **kwargs
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
