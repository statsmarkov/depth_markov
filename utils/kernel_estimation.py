from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform, bernoulli, norm
from scipy.optimize import minimize
import seaborn as sns
from typing import Callable
from scipy.stats import norm


def nadaraya_watson_pdf(data: np.array, inverse_bandwidth: float):
    """
    Nadaraya-Watson estimator with Gaussian kernel.

    :param data: list of observations.
    :param bandwidth: the bandwidth parameter for the kernel density estimate.
    :return: a function representing the estimated conditional expectation.
    """
    data = np.array(data)
    X = data[:-1]  # states at time t
    Y = data[1:]  # states at time t+1

    def estimator(x, y):
        """Nadaraya-Watson estimator function."""
        u = inverse_bandwidth * (x - X)
        K = norm.pdf(u)
        uL = inverse_bandwidth * (y - Y)
        L = norm.pdf(uL)
        return np.sum(K * L) / np.sum(K)

    return estimator


def nadaraya_watson_marginal_pdf(x: float, data: np.array, inverse_bandwidth=None):
    """
    Nadaraya-Watson estimator with Gaussian kernel.

    :param data: list of observations.
    :param bandwidth: the bandwidth parameter for the kernel density estimate.
    :return: a function representing the estimated conditional expectation.
    """
    data = np.array(data)
    X = data[:-1]  # states at time t
    Y = data[1:]  # states at time t+1

    if not inverse_bandwidth:
        # If the bandwidth is not passed in, we will choose
        # an optimal one
        inverse_bandwidth = np.power(len(data), 1 / 5)

    u = inverse_bandwidth * (x - X)
    K = norm.pdf(u)
    denominator = np.sum(K)

    def estimator(y):
        """Nadaraya-Watson estimator function."""
        uL = inverse_bandwidth * (y - Y)
        L = norm.pdf(uL)
        return inverse_bandwidth * np.sum(K * L) / denominator

    return estimator


def nadaraya_watson_marginal_cdf(x: float, data: np.array, inverse_bandwidth=None):
    """
    Nadaraya-Watson estimator of the CDF with Gaussian kernel using the integration approach.

    :param data: list of observations.
    :param bandwidth: the bandwidth parameter for the kernel density estimate.
    :return: a function representing the estimated conditional expectation.
    """
    data = np.array(data)
    X = data[:-1]  # states at time t
    Y = data[1:]  # states at time t+1

    if not inverse_bandwidth:
        # If the bandwidth is not passed in, we will choose
        # an optimal one
        inverse_bandwidth = np.power(len(data), 1 / 5)

    u = inverse_bandwidth * (x - X)
    K = norm.pdf(u)
    denominator = np.sum(K)

    def estimator(y):
        """Nadaraya-Watson estimator function."""
        uL = inverse_bandwidth * (y - Y)
        L = norm.cdf(uL)
        return np.sum(K * L) / denominator

    return estimator


def nadaraya_watson_marginal_cdf_direct(
    x: float, data: np.array, inverse_bandwidth=None
):
    """
    Nadaraya-Watson estimator of the CDF with Gaussian kernel using the direct approach

    :param data: list of observations.
    :param bandwidth: the bandwidth parameter for the kernel density estimate.
    :return: a function representing the estimated conditional expectation.
    """
    data = np.array(data)
    X = data[:-1]  # states at time t
    Y = data[1:]  # states at time t+1

    if not inverse_bandwidth:
        # If the bandwidth is not passed in, we will choose
        # an optimal one
        inverse_bandwidth = np.power(len(data), 1 / 5)

    u = inverse_bandwidth * (x - X)
    K = norm.pdf(u)
    denominator = np.sum(K)

    def estimator(y):
        """Nadaraya-Watson estimator function."""
        L = Y < y
        return np.sum(K * L) / denominator

    return estimator


def nadaraya_watson_average_marginal_cdf_direct(
    x: float, samples: np.ndarray, inverse_bandwidth=None
):
    """
    Nadaraya-Watson estimator of the CDF with Gaussian kernel based in several trajectories
    using the direct approach.

    The length of the trajectories are not necessarily unique.

    For each trajectory, we estimate the CDF of X_1|X_0=x and we return the average function.
    """
    estimators = list()
    # TODO: Parallelize this.
    for sample in samples:
        estimators.append(
            nadaraya_watson_marginal_cdf_direct(
                x=x, data=sample, inverse_bandwidth=inverse_bandwidth
            )
        )

    def estimator(y):
        return np.mean([est(y) for est in estimators])

    return estimator


def nadaraya_watson_average_marginal_pdf(
    x: float, samples: List[np.ndarray], inverse_bandwidth=None
):
    """
    Nadaraya-Watson estimator of the PDF with Gaussian kernel based in several trajectories
    using the integration approach.

    The length of the trajectories are not necessarily unique.

    For each trajectory, we estimate the CDF of X_1|X_0=x and we return the average function.
    """
    estimators = list()
    # TODO: Parallelize this.
    for sample in samples:
        estimators.append(
            nadaraya_watson_marginal_pdf(
                x=x, data=sample, inverse_bandwidth=inverse_bandwidth
            )
        )

    def estimator(y):
        return np.mean([est(y) for est in estimators])

    return estimator


def nadaraya_watson_average_marginal_cdf(
    x: float, samples: List[np.ndarray], inverse_bandwidth=None
):
    """
    Nadaraya-Watson estimator of the CDF with Gaussian kernel based in several trajectories
    using the integration approach.

    The length of the trajectories are not necessarily unique.

    For each trajectory, we estimate the CDF of X_1|X_0=x and we return the average function.
    """
    estimators = list()
    # TODO: Parallelize this.
    for sample in samples:
        estimators.append(
            nadaraya_watson_marginal_cdf(
                x=x, data=sample, inverse_bandwidth=inverse_bandwidth
            )
        )

    def estimator(y):
        return np.mean([est(y) for est in estimators])

    return estimator


def queuing_model_marginal_cdf(x: float, data: np.array, inverse_bandwidth=None):
    def estimation_zero(y):
        counter = np.sum((data[1:] > 0) & (data[1:] - data[:-1] <= y)) + np.sum(
            data[1:] <= 0
        )
        # To avoid returning 1.
        if counter == len(data) - 1:
            counter -= 1
        return counter / (len(data) - 1)

    # For x=0, we have a specific estimator
    if x == 0:

        return estimation_zero

    def _marginal_cdf_queue(y):
        if y >= x:
            return estimation_zero(y - x)
        else:
            return nadaraya_watson_marginal_cdf(
                x=x, data=data, inverse_bandwidth=inverse_bandwidth
            )(y)

    return _marginal_cdf_queue


def queuing_model_average_marginal_cdf(
    x: float, samples: List[np.ndarray], inverse_bandwidth=None
):
    estimators = list()
    # TODO: Parallelize this.
    for sample in samples:
        estimators.append(
            queuing_model_marginal_cdf(
                x=x, data=sample, inverse_bandwidth=inverse_bandwidth
            )
        )

    def estimator(y):
        return np.mean([est(y) for est in estimators])

    return estimator
