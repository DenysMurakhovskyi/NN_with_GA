from random import choices
from unittest import TestCase

import numpy as np

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




