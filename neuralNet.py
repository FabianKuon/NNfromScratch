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
            self.grads['dW' + str(layer_idx)
                       ] = np.zeros((layer_output_dim, layer_input_dim))
            self.grads['db' + str(layer_idx)] = np.zeros((layer_output_dim, 1))

    def layerwise_forward(self, a_prev: np.ndarray, w_curr: np.ndarray, b_curr: np.ndarray, activation: str) \
            -> Tuple[ndarray, ndarray]:
        """Linear network layer Z_t = W_t*A_{t-1}+b_t.

        Args:
            :param a_prev: activation from previous layer (i.e. equal to x on the first layer)
            :param w_curr: weights matrix from current layer
            :param b_curr: bias from current layer
            :param activation: activation function for the current layer of form activation(Z)

        Returns:
            :return z_curr: input to the activation function of the current layer
            :return a_curr: value of the current activation
        """
        z_curr = np.dot(w_curr, a_prev) + b_curr

        if activation == 'sigmoid':
            a_curr = hp.sigmoid_forward(z_curr)
        else:
            raise RuntimeError('Undefined activation function used. Please define the {}_forward '
                               'and {}_backward in the helper class'.format(activation, activation))

        return a_curr, z_curr

    def full_forward(self, x: ndarray) -> Tuple[ndarray, dict]:
        """Forward propagation of the input x through the whole network.

        Args:
            :param x: Input vector x

        Returns:
            :return a_curr: value of last layer activation output
            :return memory: dictionary containing input to last activation Z and output of previous activation
        """
        memory = {}
        a_curr = x

        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            a_prev = a_curr
            activation_curr = layer['activation']
            w_curr = self.params['W' + str(layer_idx)]
            b_curr = self.params['b' + str(layer_idx)]
            a_curr, z_curr = self.layerwise_forward(
                a_prev, w_curr, b_curr, activation_curr)
            memory['A' + str(idx)] = a_prev
            memory['Z' + str(layer_idx)] = z_curr

        return a_curr, memory

    def layerwise_backward(self, da_curr: ndarray, w_curr: ndarray, z_curr: ndarray, a_prev: ndarray,
                           activation: str) -> Tuple[ndarray, ndarray, ndarray]:
        """Implement the linear portion of backward propagation for a single layer (layer l)

        Args:
            :param da_curr: derivative of current activation
            :param w_curr: current weights
            :param z_curr: w_curr*a_prev+b_curr
            :param a_prev: activation of previous layer
            :param activation: activation function name

        Returns:
            :return da_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape
                               as a_prev
            :return dw_curr -- Gradient of the cost with respect to W (current layer l), same shape as W
            :return db_curr -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        m = a_prev.shape[1]
        if activation == 'sigmoid':
            dz_curr = hp.sigmoid_backprop(da_curr, z_curr)
        else:
            raise RuntimeError('Undefined activation function used. Please define the {}_forward '
                               'and {}_backward in the helper class'.format(activation, activation))

        dw_curr = np.dot(dz_curr, a_prev.T) / m
        db_curr = np.sum(dz_curr, axis=1, keepdims=True) / m
        da_prev = np.dot(w_curr.T, dz_curr)

        return da_prev, dw_curr, db_curr

    def full_backward(self, y: ndarray, y_hat: ndarray, memory: dict):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
            :param y: true labels
            :param y_hat: predicted labels
            :param memory: current values of Z and A (pre- and post-activation)
        """
        y = y.reshape(y_hat.shape)

        if self.cost_func == 'mse':
            da_prev = hp.mse_backward(y_hat, y)
        else:
            raise RuntimeError('Undefined cost function used. Please define the {}_forward '
                               'and {}_backward in the helper class'.format(self.cost_func, self.cost_func))

        for idx, layer in reversed(list(enumerate(self.architecture))):
            layer_idx = idx + 1
            activation_curr = layer['activation']
            da_curr = da_prev
            a_prev = memory['A' + str(idx)]
            z_curr = memory['Z' + str(layer_idx)]
            w_curr = self.params['W' + str(layer_idx)]

            da_prev, dw_curr, db_curr = self.layerwise_backward(
                da_curr, w_curr, z_curr, a_prev, activation_curr)

            self.grads['dW' + str(layer_idx)] = dw_curr
            self.grads['db' + str(layer_idx)] = db_curr

    def update_parameters(self):
        """Update parameters using gradient descent.
        """
        for layer_idx in range(1, self.no_layers + 1):
            self.params['W' + str(layer_idx)] -= self.learning_rate * \
                self.grads['dW' + str(layer_idx)]
            self.params['b' + str(layer_idx)] -= self.learning_rate * \
                self.grads['db' + str(layer_idx)]

    def predict(self, x: ndarray) -> ndarray:
        """Predict model output for given input sample.

        Args:
            :param x: input sample x

        Returns:
            :return: prediction y_hat
        """
        y_hat, _ = self.full_forward(x)
        return y_hat

    def compute_mse_loss(self, y: ndarray, y_hat: ndarray) -> float:
        """Compute the means squared error loss.
        Args:
            :param y: true labels
            :param y_hat: predicted labels

        Returns:
            :return: mean squared error loss
        """
        return hp.mse_forward(y_hat, y)

    def train_sgd(self, x: ndarray, y: ndarray, batch_size: int = 20, cost_func: str = 'mse',
                  no_iter: int = 10000, print_cost: bool = False, seed: int = 10) -> List[float]:
        """Train the defined network architecture in a stochastic gradient descent optimization fashion.

        Args:
            :param x: training samples
            :param y: ground truth labels
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
        no_samples = x.shape[0]
        rng = np.random.default_rng(seed=seed)
        data = np.c_[x, y]

        for i in range(no_iter):
            rng.shuffle(data)
            cost_batch = []
            for start in range(0, no_samples, batch_size):
                stop = start + batch_size
                x_batch = data[start:stop, :-1].T
                y_batch = data[start:stop, -1:].T
                y_hat, memory = self.full_forward(x_batch)
                if cost_func == 'mse':
                    cost_curr = self.compute_mse_loss(y_batch, y_hat)
                else:
                    raise RuntimeError(
                        f'Undefined cost function used. Please define the {self.cost_func}_forward ' +
                        f'and {self.cost_func}_backward in the helper class')
                cost_batch.append(cost_curr)
                self.full_backward(y_batch, y_hat, memory)
                self.update_parameters()

            if i % 50 == 0 and print_cost:
                print(
                    f"Iteration: {i} - cost: {cost_curr:.5f}")
            cost_history.append(sum(cost_batch)/len(cost_batch))

        return cost_history
