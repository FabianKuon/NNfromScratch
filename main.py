"""Main method for neural net training"""
import numpy as np
import matplotlib.pyplot as plt
from neuralNet import NeuralNet


if __name__ == '__main__':

    data = np.loadtxt('data.txt')
    X, Y = np.array_split(data, [-1], axis=1)

    NN_ARCHITECTURE = [
        {"input_dim": 1, "output_dim": 4, "activation": "sigmoid"},
        {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
    ]

    neural_net = NeuralNet(nn_architecture=NN_ARCHITECTURE, learning_rate=0.01)
    cost_hist = neural_net.train_sgd(X, Y, print_cost=True)
    plt.plot(np.squeeze(cost_hist))
    plt.title('Loss function')
    plt.show()

    Y_hat = neural_net.predict(X.T)

    pred_data = np.column_stack((X, Y_hat.T))
    pred_data_sort = pred_data[np.argsort(pred_data[:, 0])]

    sorted_data = data[np.argsort(data[:, 0])]
    plt.title('Predicted and true output')
    plt.plot(sorted_data[:, 1])
    plt.plot(pred_data_sort[:, 1])
    plt.show()
