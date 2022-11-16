"""Multi-Layer-Perceptron implementation"""
import matplotlib.pyplot as plt
import numpy as np

from Model.activation_functions_enum import ValidActivationFunctions as vaf
from Model.layer import Layer
from Model.value_repr import Value


class MLP:
    """
    Multi-Layer-Perceptron
    """

    def __init__(self, dim_in: int, dim_outs: list[int], activation_func: vaf = vaf.RELU):
        """
        param: dim_in   dimension of input features
        param: dim_outs list of neurons at each layer
        """
        size = [dim_in] + dim_outs
        self.layers = [Layer(size[i], size[i+1], activation_func)
                       for i in range(len(dim_outs))]
        self.cost_history = []
        self.predictions = np.zeros(dim_in)
        self.final_loss = None

    def __call__(self, features: list) -> list:
        for layer in self.layers:
            activation = layer(features)
        return activation

    def parameters(self):
        """
        List of all parameters belonging to the MLP.
        """
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def train(self, data_train: list[Value], ground_truth: list[float], no_epoches: int, learning_rate: float = 0.01, verbose: bool = True):
        """
        Implementation of the training procedure of the MLP.
        Iterate over forward and backward pass as long as a certain 
        convergence level is reached.

        Args:
            - data_train (list[Value]): training data set
            - ground_truth (list[float]): labels for training data
            - no_epoches (int): number of training epoches
            - learning_rate (float): learning rate
            - verbose (bool): should the cost value be printed?
        """
        for k in range(no_epoches):
            # forward pass
            ypred = [self(x) for x in data_train]
            loss = sum((yout - ygt)**2 for ygt,
                       yout in zip(ground_truth, ypred))

            # backward pass - reset gradients before backward pass
            for param in self.parameters():
                param.grad = 0.0
            loss.backward()

            # update parameters
            for param in self.parameters():
                param.data += -learning_rate * param.grad

            if verbose and k % 10 == 0:
                print(f"Iteration: {k} - cost: {loss.data:.5f}")

            self.cost_history.append(loss.data)
        self.predictions = [pred.data for pred in ypred]
        self.final_loss = loss.data

    def visualize(self, labels: list[float]):
        """
        Plot the cost history graph as well as true vs. predicted labels.

        Args:
            - labels (list[float]): ground truth labels of training data
        """
        _, axis = plt.subplots(1, 2)
        axis[0].set_title('Cost history of model training')
        axis[0].plot(self.cost_history)
        axis[1].set_title('Predicted vs ground truth lables')
        axis[1].plot(labels, label='ground truth')
        axis[1].plot(self.predictions, linestyle='dashed', label='predictions')
        axis[1].legend(loc='upper right', ncol=1, shadow=True, fancybox=True)
        plt.show()
