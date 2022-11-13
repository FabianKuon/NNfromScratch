"""MLP Prediction of requirements task"""
import matplotlib.pyplot as plt
import numpy as np

from Model.multi_layer_perceptron import MLP
from Model.activation_functions_enum import ValidActivationFunctions as vaf
from Model.value_repr import draw_dot

if __name__ == '__main__':

    X = []
    Y = []
    with open('data.txt', 'r', encoding='utf8') as file:
        for line in file:
            x, y = line.split(' ')
            X.append(float(x))
            Y.append(float(y))

    model = MLP(1, [4, 4, 1], vaf.RELU)
    model.train(X, Y, 100)
    # draw_dot(model.final_loss).render(directory='graph-output')

    pred_data = np.column_stack((X, model.predictions))
    pred_data_sort = pred_data[np.argsort(pred_data[:, 0])]

    data_set = np.column_stack((X, Y))
    labels_sorted = data_set[np.argsort(data_set[:, 0])]

    plt.title('Ground truth labels')
    plt.plot(labels_sorted[:, 1], linestyle='dashed')
    plt.show()
    plt.title('Predicted labels')
    plt.plot(pred_data_sort[:, 1])
    plt.show()
