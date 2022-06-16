from unittest import TestCase

from app.ga import GeneticAlgorithm

import numpy as np


class TestGA(TestCase):

    def setUp(self) -> None:
        self.ga = GeneticAlgorithm(len_of_citizen=40,
                                   values_range=np.array([-500, 500]))

    def test_generate(self):
        self.ga._generate()
        self.assertEqual(40, self.ga.population.shape[1])
        self.assertEqual(200, self.ga.population.shape[0])