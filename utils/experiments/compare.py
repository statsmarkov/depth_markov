from typing import List, Optional
import logging
import numpy as np
import matplotlib.pyplot as plt
from utils.markov_depth import (
    calculate_markov_tukey_depth_for_trajectories_using_sample_trajectories,
    calculate_markov_tukey_depth_for_trajectories_using_long_trajectory
)
from utils.roc_curves import plot_roc_curve
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from depth.multivariate import mahalanobis



def compare_using_sample(
    normal_trajectories: List[np.array],
    anomalous_trajectories: List[np.array],
    model_name: str,
    random_seed: int,
    long_trajectory: Optional[np.array]  = None
):
    M = len(normal_trajectories)
    A = len(anomalous_trajectories)

    total_trajectories = np.concatenate([normal_trajectories, anomalous_trajectories])

    if long_trajectory is not None:
        print("Calculating markovian depths using the long trajectory.")
        long_trajectory_total_trajectories_depths = calculate_markov_tukey_depth_for_trajectories_using_long_trajectory(trajectories=total_trajectories, long_trajectory=long_trajectory)
        # Because we are labeling regular trajectories as False and anomalous as True,
        # and our depth function should be smaller for anomalies, we are using 1-depth 
        # for classification.
        labels = np.concatenate((np.zeros(M, dtype=bool), np.ones(A, dtype=bool)))
        plot_roc_curve(labels=labels, values=1-long_trajectory_total_trajectories_depths, title=f"Markovian depth (long trajectory) - ROC Curve for {model_name}")
        plt.show()
        print("Finished calculating markovian depths using the long trajectory.")

    print("Calculating markovian depths using averaging.")
    total_trajectories_depths = (
        calculate_markov_tukey_depth_for_trajectories_using_sample_trajectories(
            trajectories=total_trajectories,
            sample_trajectories=total_trajectories,
        )
    )
    print("Finished calculating markovian depths using averaging.")

    # Because we are labeling regular trajectories as False and anomalous as True,
    # and our depth function should be smaller for anomalies, we are using 1-depth 
    # for classification.
    labels = np.concatenate((np.zeros(M, dtype=bool), np.ones(A, dtype=bool)))
    plot_roc_curve(labels=labels, values=1-total_trajectories_depths, title=f"Markovian depth (averaging) - ROC Curve for {model_name}")
    plt.show()

    print("Applying Isolation Forest")
    random_seed +=1
    # Train Isolation Forest model
    model = IsolationForest(random_state=random_seed)
    model.fit(total_trajectories)
    scores = model.decision_function(total_trajectories)
    print("Finished Isolation Forest")

    labels = np.concatenate((np.ones(M, dtype=bool), np.zeros(A, dtype=bool)))
    plot_roc_curve(labels=labels, values=scores, title=f"Isolation Forest - ROC Curve for {model_name}")
    plt.show()

    print("Applying LOF")
    lof = LocalOutlierFactor()
    y_pred = lof.fit_predict(total_trajectories)
    print("Finished LOF")

    labels = np.concatenate((np.ones(M, dtype=bool), np.zeros(A, dtype=bool)))
    plot_roc_curve(labels=labels, values=y_pred, title=f"LOF - ROC Curve for {model_name}")
    plt.show()

    print("Applying mahalanobis depth")
    mahalanobis_depths = mahalanobis(x=total_trajectories, data=total_trajectories)
    print("Finished mahalanobis depth")

    labels = np.concatenate((np.zeros(M, dtype=bool), np.ones(A, dtype=bool)))
    plot_roc_curve(labels=labels, values=1-mahalanobis_depths, title=f"Mahalanobis depth - ROC Curve for {model_name}")
    plt.show()