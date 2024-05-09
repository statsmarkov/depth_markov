import numpy as np


def perturbate_trajectory(trajectory: np.array, num_perturbations: int) -> np.array:
    """
    Perturbate a trajectory by adding some noise to it.
    """
    if len(trajectory.shape) == 1:
        trajectory = trajectory.reshape(-1, 1)
    perturbed_trajectory = np.copy(trajectory)

    return trajectory + np.random.normal(0, 1, len(trajectory))
