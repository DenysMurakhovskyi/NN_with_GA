import random

import numpy as np

from app.ga import GeneticAlgorithm
from app.neural_network import NeuralNetwork
from test.utils import make_initial_data

if __name__ == '__main__':
    nn = NeuralNetwork(layers_config=np.array([4, 8, 10, 8, 8]),
                       variables_number=2)

    ga = GeneticAlgorithm(len_of_citizen=nn.len_weights + nn.len_biases,
                          fitness_func=nn.calculate_rmse,
                          values_range=np.array([-1, 1]))
    random.seed(34)
    X, y = make_initial_data()
    nn.fit(X, y, verbose=True)