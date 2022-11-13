"""MLP Prediction of requirements task"""
import matplotlib.pyplot as plt

from Model.multi_layer_perceptron import MLP
from Model.activation_functions_enum import ValidActivationFunctions as vaf


if __name__ == '__main__':

    X = []
    Y = []
    with open('data.txt', 'r', encoding='utf8') as file:
        for line in file:
            x, y = line.split(' ')
            X.append(float(x))
            Y.append(float(y))

    model = MLP(1, [4, 4, 1], vaf.TANH)
    model.train(X, Y, 10000)
    model.visualize(Y)
    plt.show()
