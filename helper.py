import numpy as np
from numpy import ndarray


def mse_forward(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Cost function. Implementation of the mean-squared-error.

        Args:
            :param y_hat: (np.array) prediction expressed in probability
            :param y: (np.array) actual value, i.e. true label

        Returns:
            :return: mean squared error loss
        """
    m = y.shape[1]
    return 1 / m * np.sum(np.square(np.subtract(y, y_hat)))


def mse_backward(y_hat: np.ndarray, y: np.ndarray) -> ndarray:
    """Derivative of the mean squared error loss function for backpropagation.

        Args:
            :param y_hat: (np.array) prediction expressed in probability
            :param y: (np.array) actual value, i.e. true label

        Returns:
            :return: derivative of the mean squared error loss
    """
    return - 2 * (y - y_hat)


def sigmoid_forward(z: np.ndarray) -> ndarray:
    """Implementation of the sigmoid activation function.

    Args:
        :param z: output of the linear layer and input to the activation function

    Retuns:
        :return: activation
    """
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))


def sigmoid_backprop(da: np.ndarray, z: np.ndarray) -> ndarray:
    """Dervivatice of the sigmoid activation function for backpropagation.

    Args:
        :param da: gradient of the activation
        :param z: linear transformation

    Retuns:
        :return: backpropagation through sigmoid
    """
    sigmoid = sigmoid_forward(z)
    return np.multiply(da, np.multiply(sigmoid, np.subtract(1, sigmoid)))
