from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from app.neural_network import Layer


class TestLayer(TestCase):

    def test_layer_creation(self):
        layer = Layer(number_of_neurons=2, number_of_inputs=2, input_layer=False)
        self.assertIsNotNone(layer)
        self.assertEqual(2, layer.number_of_neurons)
        self.assertEqual('Neuron', layer.neurons_type)

        input_layer = Layer(number_of_neurons=2, input_layer=True)
        self.assertEqual(2, input_layer.number_of_neurons)
        self.assertEqual('InputNeuron', input_layer.neurons_type)

    def test_layer_calculation(self):
        layer = Layer(number_of_neurons=2, number_of_inputs=2, input_layer=False)
        inputs = np.array([1, 1])
        actual = layer.feed_forward(inputs)
        assert_array_almost_equal(np.array([0.88079708, 0.88079708]), actual, decimal=5)
