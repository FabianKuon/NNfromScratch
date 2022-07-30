from typing import Tuple, Any, List, Union, Dict

import numpy as np
from numpy import ndarray
import helper as hp


class NeuralNet:
    """
    Simple implementation of a feedforward neural network with linear layers and non-linear activation functions.
    """

    def __init__(self, nn_architecture: List[dict], learning_rate: float = 0.01, cost_func: str = 'mse'):
        self.params = {}
        self.grads = {}
        self.architecture = nn_architecture
        self.learning_rate = learning_rate
        self.cost_func = cost_func
        self.no_layers = len(self.architecture)
        self.init_params()

    def init_params(self, seed: int = 100):
        """Initialize weights and biases for all neural network layers (no of layers depending on dims vector).

        Args:
            :param seed (bool, optional) -- if a seed should be injected (for replication and testing purposes)
        """
        if seed:
            np.random.seed(seed)

        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            layer_input_dim = layer['input_dim']
            layer_output_dim = layer['output_dim']
            self.params['W' + str(layer_idx)] = \
                np.random.randn(layer_output_dim, layer_input_dim) * 0.001
            self.params['b' + str(layer_idx)] = np.zeros((layer_output_dim, 1))
            self.grads['dW' + str(layer_idx)] = np.zeros((layer_output_dim, layer_input_dim))
            self.grads['db' + str(layer_idx)] = np.zeros((layer_output_dim, 1))

    def layerwise_forward(self, A_prev: np.ndarray, W_curr: np.ndarray, b_curr: np.ndarray, activation: str) \
            -> Tuple[ndarray, ndarray]:
        """Linear network layer Z_t = W_t*A_{t-1}+b_t.

        Args:
            :param A_prev: activation from previous layer (i.e. equal to X on the first layer)
            :param W_curr: weights matrix from current layer
            :param b_curr: bias from current layer
            :param activation: activation function for the current layer of form activation(Z)

        Returns:
            :return Z_curr: input to the activation function of the current layer
            :return A_curr: value of the current activation
        """
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation == 'sigmoid':
            A_curr = hp.sigmoid_forward(Z_curr)
        else:
            raise RuntimeError('Undefined activation function used. Please define the {}_forward '
                               'and {}_backward in the helper class'.format(activation, activation))

        return A_curr, Z_curr

    def full_forward(self, X: ndarray) -> Tuple[ndarray, dict]:
        """Forward propagation of the input X through the whole network.

        Args:
            :param X: Input vector X

        Returns:
            :return A_curr: value of last layer activation output
            :return memory: dictionary containing input to last activation Z and output of previous activation
        """
        memory = {}
        A_curr = X

        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            A_prev = A_curr
            activation_curr = layer['activation']
            W_curr = self.params['W' + str(layer_idx)]
            b_curr = self.params['b' + str(layer_idx)]
            A_curr, Z_curr = self.layerwise_forward(A_prev, W_curr, b_curr, activation_curr)
            memory['A' + str(idx)] = A_prev
            memory['Z' + str(layer_idx)] = Z_curr

        return A_curr, memory

    def layerwise_backward(self, dA_curr: ndarray, W_curr: ndarray, Z_curr: ndarray, A_prev: ndarray,
                           activation: str) -> Tuple[ndarray, ndarray, ndarray]:
        """Implement the linear portion of backward propagation for a single layer (layer l)

        Args:
            :param dA_curr: derivative of current activation
            :param W_curr: current weights
            :param Z_curr: W_curr*A_prev+b_curr
            :param A_prev: activation of previous layer
            :param activation: activation function name

        Returns:
            :return dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape
                               as A_prev
            :return dW_curr -- Gradient of the cost with respect to W (current layer l), same shape as W
            :return db_curr -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        m = A_prev.shape[1]
        if activation == 'sigmoid':
            dZ_curr = hp.sigmoid_backprop(dA_curr, Z_curr)
        else:
            raise RuntimeError('Undefined activation function used. Please define the {}_forward '
                               'and {}_backward in the helper class'.format(activation, activation))

        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_backward(self, Y: ndarray, Y_hat: ndarray, memory: dict):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
            :param Y: true labels
            :param Y_hat: predicted labels
            :param memory: current values of Z and A (pre- and post-activation)
        """
        Y = Y.reshape(Y_hat.shape)

        if self.cost_func == 'mse':
            dA_prev = hp.mse_backward(Y_hat, Y)
        else:
            raise RuntimeError('Undefined cost function used. Please define the {}_forward '
                               'and {}_backward in the helper class'.format(self.cost_func, self.cost_func))

        for idx, layer in reversed(list(enumerate(self.architecture))):
            layer_idx = idx + 1
            activation_curr = layer['activation']
            dA_curr = dA_prev
            A_prev = memory['A' + str(idx)]
            Z_curr = memory['Z' + str(layer_idx)]
            W_curr = self.params['W' + str(layer_idx)]

            dA_prev, dW_curr, db_curr = self.layerwise_backward(dA_curr, W_curr, Z_curr, A_prev, activation_curr)

            self.grads['dW' + str(layer_idx)] = dW_curr
            self.grads['db' + str(layer_idx)] = db_curr

    def update_parameters(self):
        """Update parameters using gradient descent.
        """
        for layer_idx in range(1, self.no_layers + 1):
            self.params['W' + str(layer_idx)] -= self.learning_rate * self.grads['dW' + str(layer_idx)]
            self.params['b' + str(layer_idx)] -= self.learning_rate * self.grads['db' + str(layer_idx)]

    def predict(self, X: ndarray) -> ndarray:
        """Predict model output for given input sample.

        Args:
            :param X: input sample X

        Returns:
            :return: prediction Y_hat
        """
        Y_hat, _ = self.full_forward(X)
        return Y_hat

    def compute_mse_loss(self, Y: ndarray, Y_hat: ndarray) -> float:
        """Compute the means squared error loss.
        Args:
            :param Y: true labels
            :param Y_hat: predicted labels

        Returns:
            :return: mean squared error loss
        """
        return hp.mse_forward(Y_hat, Y)

    def train_sgd(self, X: ndarray, Y: ndarray, batch_size: int = 20, cost_func: str = 'mse',
                  no_iter: int = 10000, print_cost: bool = False, seed: int = 10) -> List[float]:
        """Train the defined network architecture in a stochastic gradient descent optimization fashion.

        Args:
            :param X: training samples
            :param Y: ground truth labels
            :param batch_size: size of the randomly selected samples
            :param cost_func: cost function which will be used as performance measure
            :param no_iter: no of training iterations
            :param print_cost: should the value of the cost function be printed (each 50 iterations)
            :param seed: set the seed for reproducibility possibility

        Returns:
            :return Weights and biases of the trained model
            :return cost_history: vector containing the values of the cost function after each training iteration
        """
        cost_history = []
        no_samples = X.shape[0]
        rng = np.random.default_rng(seed=seed)
        data = np.c_[X, Y]

        for i in range(no_iter):
            rng.shuffle(data)
            cost_batch = []
            for start in range(0, no_samples, batch_size):
                stop = start + batch_size
                X_batch = data[start:stop, :-1].T
                Y_batch = data[start:stop, -1:].T
                Y_hat, memory = self.full_forward(X_batch)
                if cost_func == 'mse':
                    cost_curr = self.compute_mse_loss(Y_batch, Y_hat)
                else:
                    raise RuntimeError('Undefined cost function used. Please define the {}_forward '
                                       'and {}_backward in the helper class'.format(self.cost_func, self.cost_func))
                cost_batch.append(cost_curr)
                self.full_backward(Y_batch, Y_hat, memory)
                self.update_parameters()

            if i % 50 == 0 and print_cost:
                print("Iteration: {} - cost: {:.5f}".format(i, cost_curr))
            cost_history.append(sum(cost_batch)/len(cost_batch))

        return cost_history
