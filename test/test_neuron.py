from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal

from app.neuron import InputNeuron, Neuron


class TestNeuron(TestCase):

    def test_input_neuron(self):
        inu = InputNeuron()
        self.assertEqual(1, inu.feedforward(inputs=1))

    def test_neuron(self):
        nu = Neuron(number_of_inputs=2)
        actual = nu.feedforward(np.array([2, 3]))
        assert_almost_equal(actual, 0.993307, decimal=6)

        nu.weights, nu.bias = np.array([0, 1]), 4
        actual = nu.feedforward(np.array([2, 3]))
        assert_almost_equal(actual, 0.999088, decimal=6)
