from typing import List, Optional
import logging
import numpy as np
import matplotlib.pyplot as plt
from utils.markov_depth import (
    calculate_markov_tukey_depth_for_trajectories_using_sample_trajectories,
    calculate_markov_tukey_depth_for_trajectories_using_long_trajectory,
    QUEUING_MODEL,
    NADARAYA_WATSON,
)
from utils.roc_curves import plot_roc_curve
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from depth.multivariate import mahalanobis


def compare_without_training_data(
    normal_trajectories: List[np.array],
    anomalous_trajectories: List[np.array],
    model_name: str,
    random_seed: int,
    marginal_cdf_estimator: str = NADARAYA_WATSON,
    long_trajectory: Optional[np.array] = None,
):
    M = len(normal_trajectories)
    A = len(anomalous_trajectories)

    total_trajectories = np.concatenate([normal_trajectories, anomalous_trajectories])

    if long_trajectory is not None:
        print("Calculating markovian depths using the long trajectory.")
        long_trajectory_total_trajectories_depths = (
            calculate_markov_tukey_depth_for_trajectories_using_long_trajectory(
                trajectories=total_trajectories,
                long_trajectory=long_trajectory,
                marginal_cdf_estimator=marginal_cdf_estimator,
            )
        )
        # Because we are labeling regular trajectories as False and anomalous as True,
        # and our depth function should be smaller for anomalies, we are using 1-depth
        # for classification.
        labels = np.concatenate((np.zeros(M, dtype=bool), np.ones(A, dtype=bool)))
        plot_roc_curve(
            labels=labels,
            values=1 - long_trajectory_total_trajectories_depths,
            title=f"Markovian depth (long trajectory) - ROC Curve for {model_name}",
        )
        plt.show()
        print("Finished calculating markovian depths using the long trajectory.")

    print("Calculating markovian depths using averaging.")
    total_trajectories_depths = (
        calculate_markov_tukey_depth_for_trajectories_using_sample_trajectories(
            trajectories=total_trajectories,
            sample_trajectories=total_trajectories,
            marginal_cdf_estimator=marginal_cdf_estimator,
        )
    )
    print("Finished calculating markovian depths using averaging.")

    # Because we are labeling regular trajectories as False and anomalous as True,
    # and our depth function should be smaller for anomalies, we are using 1-depth
    # for classification.
    labels = np.concatenate((np.zeros(M, dtype=bool), np.ones(A, dtype=bool)))
    plot_roc_curve(
        labels=labels,
        values=1 - total_trajectories_depths,
        title=f"Markovian depth (averaging) - ROC Curve for {model_name}",
    )
    plt.show()

    print("Applying Isolation Forest")
    random_seed += 1
    # Train Isolation Forest model
    model = IsolationForest(random_state=random_seed)
    model.fit(total_trajectories)
    scores = model.decision_function(total_trajectories)
    print("Finished Isolation Forest")

    labels = np.concatenate((np.ones(M, dtype=bool), np.zeros(A, dtype=bool)))
    plot_roc_curve(
        labels=labels,
        values=scores,
        title=f"Isolation Forest - ROC Curve for {model_name}",
    )
    plt.show()

    print("Applying LOF")
    lof = LocalOutlierFactor()
    y_pred = lof.fit_predict(total_trajectories)
    print("Finished LOF")

    labels = np.concatenate((np.ones(M, dtype=bool), np.zeros(A, dtype=bool)))
    plot_roc_curve(
        labels=labels, values=y_pred, title=f"LOF - ROC Curve for {model_name}"
    )
    plt.show()

    print("Applying mahalanobis depth")
    mahalanobis_depths = mahalanobis(x=total_trajectories, data=total_trajectories)
    print("Finished mahalanobis depth")

    labels = np.concatenate((np.zeros(M, dtype=bool), np.ones(A, dtype=bool)))
    plot_roc_curve(
        labels=labels,
        values=1 - mahalanobis_depths,
        title=f"Mahalanobis depth - ROC Curve for {model_name}",
    )
    plt.show()


class MahalanobisClassifier:
    def __init__(self, x: np.array, data: np.array, **kwargs):
        self.x = x
        self.data = data

    def fit(self, x: np.array, y: np.array):
        self._regular_trajectory = None

    def predict_proba(self, x: np.array):
        return mahalanobis(x=x, data=self._regular_trajectory)


CLASSIFIERS = {
    "Isolation Forest": IsolationForest,
    "Local Outlier Factor": LocalOutlierFactor,
    "Mahalanobis": MahalanobisClassifier,
    "Support Vector Machine": SVC,
    "Logistic Regression": LogisticRegression,
    "K-Nearest Neighbors": KNeighborsClassifier,
}


def compare_using_training_sample(
    training_trajectories: List[np.array],
    normal_trajectories: List[np.array],
    anomalous_trajectories: List[np.array],
    model_name: str,
    random_seed: int,
    marginal_cdf_estimator: str = NADARAYA_WATSON,
    long_trajectory: Optional[np.array] = None,
):
    M = len(normal_trajectories)
    A = len(anomalous_trajectories)

    total_trajectories = np.concatenate([normal_trajectories, anomalous_trajectories])

    print("Applying Isolation Forest")
    random_seed += 1
    # Train Isolation Forest model
    model = IsolationForest(random_state=random_seed)
    model.fit(training_trajectories)
    scores = model.decision_function(total_trajectories)
    print("Finished Isolation Forest")

    labels = np.concatenate((np.ones(M, dtype=bool), np.zeros(A, dtype=bool)))
    isolation_forest_roc_data = plot_roc_curve(
        labels=labels,
        values=scores,
        title=f"Isolation Forest - ROC Curve for {model_name}",
    )

    plt.show()

    print("Applying LOF")
    # Train LOF model
    lof = LocalOutlierFactor(novelty=True)
    lof = lof.fit(training_trajectories)
    y_pred = lof.decision_function(total_trajectories)
    print("Finished LOF")

    labels = np.concatenate((np.ones(M, dtype=bool), np.zeros(A, dtype=bool)))
    lof_roc_data = plot_roc_curve(
        labels=labels, values=y_pred, title=f"LOF - ROC Curve for {model_name}"
    )
    plt.show()

    if long_trajectory is not None:
        print("Calculating markovian depths using the long trajectory.")
        long_trajectory_total_trajectories_depths = (
            calculate_markov_tukey_depth_for_trajectories_using_long_trajectory(
                trajectories=total_trajectories,
                long_trajectory=long_trajectory,
                marginal_cdf_estimator=marginal_cdf_estimator,
            )
        )
        # Because we are labeling regular trajectories as False and anomalous as True,
        # and our depth function should be smaller for anomalies, we are using 1-depth
        # for classification.
        labels = np.concatenate((np.zeros(M, dtype=bool), np.ones(A, dtype=bool)))
        long_trajectory_roc_data = plot_roc_curve(
            labels=labels,
            values=1 - long_trajectory_total_trajectories_depths,
            title=f"Markovian depth (long trajectory) - ROC Curve for {model_name}",
        )
        plt.show()
        print("Finished calculating markovian depths using the long trajectory.")

    print("Calculating markovian depths using averaging.")
    total_trajectories_depths = (
        calculate_markov_tukey_depth_for_trajectories_using_sample_trajectories(
            trajectories=total_trajectories,
            sample_trajectories=training_trajectories,
            marginal_cdf_estimator=marginal_cdf_estimator,
        )
    )
    print("Finished calculating markovian depths using averaging.")

    # Because we are labeling regular trajectories as False and anomalous as True,
    # and our depth function should be smaller for anomalies, we are using 1-depth
    # for classification.
    labels = np.concatenate((np.zeros(M, dtype=bool), np.ones(A, dtype=bool)))
    averaging_roc_data = plot_roc_curve(
        labels=labels,
        values=1 - total_trajectories_depths,
        title=f"Markovian depth (averaging) - ROC Curve for {model_name}",
    )

    print("Applying mahalanobis depth")
    mahalanobis_depths = mahalanobis(x=total_trajectories, data=total_trajectories)
    print("Finished mahalanobis depth")

    labels = np.concatenate((np.zeros(M, dtype=bool), np.ones(A, dtype=bool)))
    mahalanobis_roc_data = plot_roc_curve(
        labels=labels,
        values=1 - mahalanobis_depths,
        title=f"Mahalanobis depth - ROC Curve for {model_name}",
    )
    plt.show()

    # Plot ROC curves together
    plt.figure()
    plt.plot(
        isolation_forest_roc_data[0],
        isolation_forest_roc_data[1],
        color="Orange",
        lw=2,
        label="Isolation Forest (area = %0.2f)" % isolation_forest_roc_data[3],
    )
    plt.plot(
        lof_roc_data[0],
        lof_roc_data[1],
        color="Red",
        lw=2,
        label="LOF (area = %0.2f)" % lof_roc_data[3],
    )
    plt.plot(
        mahalanobis_roc_data[0],
        mahalanobis_roc_data[1],
        color="Blue",
        lw=2,
        label="Mahalanobis depth (area = %0.2f)" % mahalanobis_roc_data[3],
    )
    plt.plot(
        averaging_roc_data[0],
        averaging_roc_data[1],
        color="Green",
        lw=2,
        label="Markovian depth (area = %0.2f)" % averaging_roc_data[3],
    )

    # Plot diagonal line for reference
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    # plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
