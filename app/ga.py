import math
import random
from dataclasses import dataclass
from random import uniform
from typing import List, Union

import numpy as np


@dataclass
class GeneticParams:
    population_size: int = 40
    max_generations: int = 150
    n_best_share: float = 0.2
    n_rand_share: float = 0.25
    mutation_share: float = 0.3
    mutation_rate: float = 0.8


class GeneticAlgorithm:
    """
    Class implements genetic algorithm model
    """

    @property
    def population(self) -> Union[List[List], None]:
        return self._population

    def __init__(self, len_of_citizen: int,
                 values_range: np.ndarray,
                 params: GeneticParams = None,
                 fitness_func=None,
                 verbose=False) -> None:
        if len(values_range) != 2:
            raise ValueError('Incorrect boundaries range')
        self.len_of_citizen: int = len_of_citizen
        self.low_bound: Union[float, int] = values_range[0]
        self.high_bound: Union[float, int] = values_range[1]
        self.parameters = params if params else GeneticParams()
        self.fitness = fitness_func if fitness_func else self._default_fitness
        self._population: Union[List[List], None] = None
        self._verbose = verbose

    def solve(self) -> List:
        """
        Main method implementing genetic algorith
        :return: found solution
        """

        # generate population
        self._generate()

        n_best_members: int = math.floor(self.parameters.n_best_share * self.parameters.population_size)
        random_members_number: int = int(self.parameters.population_size *
                                         self.parameters.n_rand_share *
                                         (1 - self.parameters.n_best_share))

        # main loop
        for _ in range(self.parameters.max_generations):

            # evaluate population with fitness function
            population_evaluation = self._evaluate_fitness()
            min_fitness_value = min(population_evaluation)
            if self._verbose:
                print(f'Epoch {_}, min fitness func value {min_fitness_value},'
                      f'max fitness func value {max(population_evaluation)}')

            # choose N best members
            sorted_population_evaluation = sorted(population_evaluation)
            high_score_bound = sorted_population_evaluation[n_best_members]
            best_population = [citizen for citizen, score in zip(self.population, population_evaluation)
                               if score < high_score_bound]

            # define left members and choose random N members
            left_members = [(citizen, score) for citizen, score in zip(self.population, population_evaluation)
                            if score >= high_score_bound]

            randomly_chosen_members = random.sample([item[0] for item in left_members], random_members_number)

            # create new population from best and randomly chosen members
            new_population = randomly_chosen_members

            # apply crossover
            children = self._multi_crossover(high_score_bound)
            if children is not None:
                new_population += children

            # fill the population to its size with the best ones from the left members
            left_members = sorted(left_members, key=lambda x: x[1], reverse=True)
            left_members_to_add = left_members[:self.parameters.population_size -
                                                len(new_population) -
                                                len(best_population)]
            new_population += [item[0] for item in left_members_to_add]

            # mutation in new population
            mutate_members_number: int = int(round(len(new_population) *
                                                   self.parameters.mutation_share))

            members_to_mutate = random.sample(list(range(len(new_population))),
                                              mutate_members_number)
            for member in members_to_mutate:
                citizen_to_mutate = new_population[member]
                new_population[member] = self._mutate(citizen_to_mutate)

            self._population = best_population + new_population

        # return results
        min_score = min(final_scores := self._evaluate_fitness())
        best_citizen_index = final_scores.index(min_score)
        return self.population[best_citizen_index]

    @staticmethod
    def _default_fitness(y_pred) -> float:
        """
        Dummy function. Usually an external one is used
        :param y_pred: a citizen to evaluate
        :return: evaluation value
        """
        return 1

    def _multi_crossover(self, min_fitness_value) -> List:
        """
        Applying single crossover on randomly chosen members from the population
        :param min_fitness_value: the previously calculated border value of the fitness function
        :return:
        """
        children: Union[List, None] = None
        for _ in range(int((1 - self.parameters.n_best_share - self.parameters.n_rand_share)
                           * self.parameters.population_size)):
            # choose parents
            parents_numbers = random.sample(list(range(len(self.population))), 2)
            parents = self.population[parents_numbers[0]], self._population[parents_numbers[1]]

            # make child
            child = self._single_crossover(parents[0], parents[1])

            # check child
            if self.fitness(child) < min_fitness_value:
                if children is None:
                    children = [child.copy()]
                else:
                    children.append(child)
        return children

    def _evaluate_fitness(self) -> List[float]:
        """
        Evaluates the fitness function for the sequence of given members
        :return: the list of calculated values
        """
        evaluations = []
        for citizen in self.population:
            evaluations.append(self.fitness(citizen))
        return evaluations

    def _generate(self) -> None:
        """
        Initial population generator
        :return: None
        """
        initial_population = []
        for _ in range(self.parameters.population_size):
            initial_population.append([uniform(self.low_bound, self.high_bound) for _ in range(self.len_of_citizen)])
        self._population = initial_population

    def _single_crossover(self, chromosome_1: List, chromosome_2: List,
                          start_position=-1, length=-1) -> List:
        """
        Crossover for two chosen members
        :param chromosome_1, chromosome_2: the chosen citizens from the population
        :param start_position: start position of gens' exchange. Using for tests.
        :param length: the length of the gens' interchange. Using for tests.
        :return: new chromosome
        """
        exchange_start_position: int = random.randint(10, self.len_of_citizen - 10) if start_position == -1 \
            else start_position
        try:
            exchange_length: int = random.randint(1, self.len_of_citizen - exchange_start_position) if length == -1 \
                else length
        except Exception as ex:
            exchange_length: int = int(0.1 * self.len_of_citizen)

        new_chromosome: List = chromosome_1[0:exchange_start_position] + \
                               chromosome_2[exchange_start_position: exchange_start_position + exchange_length] + \
                               chromosome_1[exchange_start_position + exchange_length:]
        return new_chromosome

    def _mutate(self, chromosome: List) -> List:
        """
        Mutate the chromosome
        :param chromosome: a chromosome to mutate
        :return: mutated chromosome
        """
        gens_to_mutate: List = random.sample(list(range(self.len_of_citizen)),
                                             k=math.ceil(self.parameters.mutation_rate * self.len_of_citizen))
        removed_values: List = [chromosome[n] for n in gens_to_mutate]
        random.shuffle(removed_values)
        for i, n in enumerate(gens_to_mutate):
            chromosome[n] = removed_values[i]

        return chromosome
