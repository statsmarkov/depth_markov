import numpy as np


# Define the simulation function
def simulate_ar1_process(
    n_steps: int,
    phi: float,
    sigma: float,
    initial_value: float,
    num_processes: int,
    seed: int = None,
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


def simulate_reflective_random_walk(n_steps: int, num_processes: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)  # Set the seed for reproducibility

    upper_bound = 1
    lower_bound = -1

    # Initialize the array to store the walks
    Y = np.zeros((num_processes, n_steps))  # Include the initial position

    # Generate all steps at once for all processes
    all_steps = np.random.uniform(
        low=lower_bound, high=upper_bound, size=(num_processes, n_steps - 1)
    )

    for i in range(num_processes):
        position = 0
        for j in range(n_steps - 1):
            new_position = position + all_steps[i, j]
            if new_position > upper_bound:
                excess = new_position - upper_bound
                position = upper_bound - excess  # Rebound at the upper boundary
            elif new_position < lower_bound:
                excess = lower_bound - new_position
                position = lower_bound + excess  # Rebound at the lower boundary
            else:
                position = new_position
            Y[i, j + 1] = position  # Store the position after the step

    return Y
