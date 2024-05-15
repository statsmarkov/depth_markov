from typing import Tuple
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


def plot_roc_curve(
    labels: np.ndarray, values: np.ndarray, title: str = "ROC curve"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot the ROC curve and calculate the area under the curve.

    Args:
        labels (np.ndarray): True binary labels.
        values (np.ndarray): Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
        title (str, optional): Title for the plot. Defaults to "ROC curve".

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: False positive rates, true positive rates, and thresholds.
    """
    fpr, tpr, thresholds = roc_curve(labels, values, pos_label=1)

    # Calculate are under the curve
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    return fpr, tpr, thresholds
