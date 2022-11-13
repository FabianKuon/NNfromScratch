"""Multi-Layer-Perceptron implementation"""
from Model.layer import Layer
from Model.value_repr import Value
import numpy as np


class MLP:
    """
    Multi-Layer-Perceptron
    """

    def __init__(self, dim_in: int, dim_outs: list[int]):
        """
        param: dim_in   dimension of input features
        param: dim_outs list of neurons at each layer
        """
        size = [dim_in] + dim_outs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(dim_outs))]
        self.cost_history = []
        self.predictions = np.zeros(dim_in)

    def __call__(self, features: list) -> list:
        for layer in self.layers:
            activation = layer(features)
        return activation

    def parameters(self):
        """
        List of all parameters belonging to the MLP.
        """
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def train(self, data_train: list[Value], ground_truth: list[float], no_epoches: int, verbose: bool = True):
        """
        Implementation of the training procedure of the MLP.
        Iterate over forward and backward pass as long as a certain 
        convergence level is reached.

        Args:
            - data_train (list[Value]): training data set
            - ground_truth (list[float]): labels for training data
            - no_epoches (int): number of training epoches
            - verbose (bool): should the cost value be printed?
        """
        for k in range(no_epoches):
            # forward pass
            ypred = [self(x) for x in data_train]
            loss = sum((yout - ygt)**2 for ygt,
                       yout in zip(ground_truth, ypred))

            # backward pass
            for param in self.parameters():
                param.grad = 0.0
            loss.backward()

            # update
            for param in self.parameters():
                param.data += -0.1 * param.grad

            if verbose and k % 10 == 0:
                print(f"Iteration: {k} - cost: {loss.data:.5f}")

            self.cost_history.append(loss.data)
            self.predictions = [prediction.data for prediction in ypred]
