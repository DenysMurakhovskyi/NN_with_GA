from random import choices
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from app.neural_network import NeuralNetwork


class TestNeuralNetwork(TestCase):

    def setUp(self) -> None:
        self.nn = NeuralNetwork(layers_config=np.array([4, 8, 10, 8, 8]),
                                variables_number=2)

    @staticmethod
    def _func(x: float, y: float):
        return np.sin(2 * np.sqrt(x**2 + y**2)) / (np.sqrt(x**2 + y**2) + 0.001)

    def test_nn_weights_biases(self):
        actual = self.nn.weights
        self.assertEqual(272, len(actual))
        actual = self.nn.biases
        self.assertEqual(39, len(actual))
        self.assertEqual(272, self.nn.len_weights)

    def test_calculate(self):
        actual = self.nn._calculate(inputs=np.array([1, 1]))
        pass

    def test_fit(self):
        xs = choices(np.cumsum(0.01 * np.ones(1001)) - 5.01, k=100)
        ys = choices(np.cumsum(0.01 * np.ones(1001)) - 5.01, k=100)
        func = np.vectorize(self._func)
        y = func(xs, ys)
        r2 = self.nn.r2_
        self.assertEqual(-np.inf, r2)
        self.nn.fit(list(zip(xs, ys)), y)
        r2 = self.nn.r2_
        self.assertLess(-1000, r2)

    def test_calculate_vectorized(self):
        xs = choices(np.cumsum(0.01 * np.ones(1001)) - 5.01, k=20)
        ys = choices(np.cumsum(0.01 * np.ones(1001)) - 5.01, k=20)
        X = list(zip(xs, ys))
        result = self.nn._calculate_vectorized(X)
        pass

    def test_set_weights(self):
        values = 2 * np.ones(self.nn.len_weights)
        self.nn.weights = values
        assert_array_equal(values, self.nn.weights)

    def test_set_biases(self):
        values = 2 * np.ones(len(self.nn.biases))
        self.nn.biases = values
        assert_array_equal(values, self.nn.biases)

    def test_set_values(self):
        values = 2 * np.ones(self.nn.len_weights + self.nn.len_biases)
        self.nn._set_values(values)
        assert_array_equal(2 * np.ones(self.nn.len_weights), self.nn.weights)
        assert_array_equal(2 * np.ones(self.nn.len_biases), self.nn.biases)




