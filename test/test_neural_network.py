from unittest import TestCase

import numpy as np

from app.neural_network import NeuralNetwork


class TestNeuralNetwork(TestCase):

    def setUp(self) -> None:
        self.nn = NeuralNetwork(layers_config=np.array([4, 8, 10, 8, 8]),
                                variables_number=2)

    def test_nn_weights_biases(self):
        actual = self.nn.weights
        self.assertEqual(272, len(actual))
        actual = self.nn.biases
        self.assertEqual(39, len(actual))