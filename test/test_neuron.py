from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal

from app.neuron import InputNeuron, Neuron


class TestNeuron(TestCase):

    def test_input_neuron(self):
        inu = InputNeuron()
        self.assertEqual(1, inu.feedforward(inputs=1))

    def test_neuron(self):
        nu = Neuron(weights=np.array([0, 1]), bias=4)
        actual = nu.feedforward(np.array([2, 3]))
        assert_almost_equal(actual, 0.999088, decimal=6)
