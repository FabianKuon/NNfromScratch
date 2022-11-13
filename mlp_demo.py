"""Training of a mlp with a simple data example"""
import matplotlib.pyplot as plt
from Model.multiLayerPerceptron import MLP


if __name__ == '__main__':
    # a multi-layer-perceptron with input dimension 3,
    # first and second layer dimension 4 and outpit dimension 1
    model = MLP(3, [4, 4, 1])
    features = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]

    labels = [1.0, -1.0, -1.0, 1.0]

    model.train(features, labels, 1000, True)

    figure, axis = plt.subplots(1, 2)
    axis[0].set_title('Cost history of model training')
    axis[0].plot(model.cost_history)
    axis[1].set_title('Predicted vs ground truth lables')
    axis[1].plot(labels, label='ground truth')
    axis[1].plot(model.predictions, linestyle='dashed', label='predictions')
    axis[1].legend(loc='upper right', ncol=1, shadow=True, fancybox=True)
    plt.show()
