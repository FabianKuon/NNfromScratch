import numpy as np
from numpy import ndarray


def mse_forward(Y_hat: np.ndarray, Y: np.ndarray) -> float:
    """Cost function. Implementation of the mean-squared-error.

        Args:
            :param Y_hat: (np.array) prediction expressed in probability
            :param Y: (np.array) actual value, i.e. true label

        Returns:
            :return: mean squared error loss
        """
    m = Y.shape[1]
    return 1 / m * np.sum(np.square(np.subtract(Y, Y_hat)))


def mse_backward(Y_hat: np.ndarray, Y: np.ndarray) -> ndarray:
    """Derivative of the mean squared error loss function for backpropagation.

        Args:
            :param Y_hat: (np.array) prediction expressed in probability
            :param Y: (np.array) actual value, i.e. true label

        Returns:
            :return: derivative of the mean squared error loss
    """
    return - 2 * (Y - Y_hat)


def sigmoid_forward(Z: np.ndarray) -> ndarray:
    """Implementation of the sigmoid activation function.

    Args:
        :param Z: output of the linear layer and input to the activation function

    Retuns:
        :return: activation
    """
    return 1 / (1 + np.exp(-np.clip(Z, -250, 250)))


def sigmoid_backprop(dA: np.ndarray, Z: np.ndarray) -> ndarray:
    """Dervivatice of the sigmoid activation function for backpropagation.

    Args:
        :param dA: gradient of the activation
        :param Z: linear transformation

    Retuns:
        :return: backpropagation through sigmoid
    """
    sigmoid = sigmoid_forward(Z)
    return np.multiply(dA, np.multiply(sigmoid, np.subtract(1, sigmoid)))
