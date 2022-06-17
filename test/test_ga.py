import random
from unittest import TestCase

import numpy as np

from app.ga import GeneticAlgorithm
from app.neural_network import NeuralNetwork
from test.utils import make_initial_data


class TestGA(TestCase):

    @staticmethod
    def _func(x: float, y: float):
        return np.sin(2 * np.sqrt(x**2 + y**2)) / (np.sqrt(x**2 + y**2) + 0.001)

    def setUp(self) -> None:
        self.nn = NeuralNetwork(layers_config=np.array([4, 8, 10, 8, 8]),
                                variables_number=2)
        self.citizen_len = self.nn.len_weights + self.nn.len_biases
        self.ga = GeneticAlgorithm(len_of_citizen=self.citizen_len,
                                   fitness_func=self.nn.calculate_rmse,
                                   values_range=np.array([-1, 1]))

    def test_generate(self):
        self.ga._generate()
        self.assertEqual(self.citizen_len, self.ga.population.shape[1])
        self.assertEqual(200, self.ga.population.shape[0])

    def test_evaluate_fitness(self):
        X, y = make_initial_data()
        self.nn._prepare_fit(X, y)
        self.ga._generate()
        evaluations = []
        for citizen in self.ga.population:
            evaluations.append(self.ga.fitness(citizen))
        self.assertLess(10, len(set(evaluations)))

        actual = self.ga._evaluate_fitness()
        self.assertEqual(max(evaluations), max(actual))

    def test_ga_solve(self):
        random.seed(34)
        X, y = make_initial_data()
        self.nn.fit(X, y, verbose=True)

