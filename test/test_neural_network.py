from unittest import TestCase

import numpy as np

from app.neural_network import NeuralNetwork


class TestNeuralNetwork(TestCase):

    def test_nn_creation(self):
        nn = NeuralNetwork(layers_config=np.array([4, 8, 10, 8, 8]),
                           variables_number=2)
        pass