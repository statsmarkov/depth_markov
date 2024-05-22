import typing
import matplotlib.pyplot as plt
import numpy as np
from utils.markov_depth import (
    calculate_markov_tukey_depth_for_trajectories_using_sample_trajectories,
)


def obtain_dd_plot(
    x_trajectories: list[np.array],
    y_trajectories: list[np.array],
    marginal_cdf_estimator: str,
    index: str,
):
    total_trajectories = x_trajectories + y_trajectories
    depths_with_respect_x = (
        calculate_markov_tukey_depth_for_trajectories_using_sample_trajectories(
            trajectories=total_trajectories,
            sample_trajectories=x_trajectories,
            marginal_cdf_estimator=marginal_cdf_estimator,
        )
    )

    depths_with_respect_y = (
        calculate_markov_tukey_depth_for_trajectories_using_sample_trajectories(
            trajectories=total_trajectories,
            sample_trajectories=y_trajectories,
            marginal_cdf_estimator=marginal_cdf_estimator,
        )
    )

    M = len(x_trajectories)
    A = len(y_trajectories)

    # The first M trajectories in total_trajectories corresponds to X
    plt.scatter(
        depths_with_respect_x[:M],
        depths_with_respect_y[:M],
        color="blue",
        s=20,
        label=r"$\mathcal{X}$",
    )

    plt.scatter(
        depths_with_respect_x[M:],
        depths_with_respect_y[M:],
        color="orange",
        marker="*",
        s=20,
        label=r"$\mathcal{Y}_{%s}$" % index,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Dashed diagonal line
    total_depths = np.concatenate((depths_with_respect_x, depths_with_respect_y))
    lim_inf = np.min(total_depths) - 0.005
    lim_sup = np.max(total_depths) + 0.005
    plt.xlim(lim_inf, lim_sup)
    plt.ylim(lim_inf, lim_sup)
    plt.xlabel(r"$D_{\hat{\Pi}}$", fontsize=16)
    plt.ylabel(r"$D_{\hat{\Psi}}$", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()

    t = np.arange(1, M + A + 1)
    plt.scatter(t[:M], depths_with_respect_x[:M], color="blue")
    plt.scatter(t[M:], depths_with_respect_x[M:], color="orange", marker="*")
    plt.xlabel("Observation index", fontsize=16)
    plt.ylabel(r"$D_{\hat{\Pi}}$", fontsize=16)
    plt.tight_layout()
    plt.show()

    t = np.arange(1, M + A + 1)
    plt.scatter(t[:M], depths_with_respect_y[:M], color="blue")
    plt.scatter(t[M:], depths_with_respect_y[M:], color="orange", marker="*")
    plt.xlabel("Observation index", fontsize=16)
    plt.ylabel(r"$D_{\hat{\Psi}}$", fontsize=16)
    plt.tight_layout()
    plt.show()

    return depths_with_respect_x, depths_with_respect_y
