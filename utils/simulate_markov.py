import numpy as np


# Define the simulation function
def simulate_ar1_process(
    n_steps: int,
    phi: float,
    sigma: float,
    initial_value: float,
    num_processes: int,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)  # Ensure reproducibility in parallel processing
    # Initialize all processes
    Y = np.zeros((num_processes, n_steps))
    Y[:, 0] = initial_value

    # Generate all noise terms at once
    noise = np.random.normal(0, sigma, size=(num_processes, n_steps - 1))

    # Vectorized computation of the AR(1) process
    for t in range(1, n_steps):
        Y[:, t] = phi * Y[:, t - 1] + noise[:, t - 1]

    return Y
