import logging
from typing import Tuple, List, Union, NoReturn
import numpy as np
from numpy.typing import NDArray
import random
from dataclasses import dataclass
from collections import Counter
import math
from random import uniform


logger = logging.getLogger('main')
logger.setLevel(logging.INFO)


@dataclass
class GeneticParams:
    population_size: int = 200
    max_generations: int = 200
    n_best_share: float = 0.4
    n_rand_share: float = 0.1
    mutation_rate: float = 0.1
    steps_for_stop_criteria = 10
    stop_decrease_ratio = 0.01


class GeneticAlgorithm:

    @property
    def population(self):
        return self._population

    def __init__(self, len_of_citizen: int,
                 values_range: np.ndarray,
                 params: GeneticParams = None,
                 fitness_func=None) -> None:
        if len(values_range) != 2:
            raise ValueError('Incorrect boundaries range')
        self.len_of_citizen: int = len_of_citizen
        self.low_bound: Union[float, int] = values_range[0]
        self.high_bound: Union[float, int] = values_range[1]
        self.parameters = params if params else GeneticParams()
        self.fitness = fitness_func if fitness_func else self._default_fitness
        self._population: Union[NDArray[NDArray], None] = None

    def solve(self) -> Tuple[NDArray, float]:
        """
        Main method implementing genetic algorith, for TSP problem
        :return: path and its length
        """

        # generate population
        self._generate()

        # main loop
        for _ in range(self.parameters.max_generations):

            # evaluate population with fitness function
            population_evaluation = self._evaluate_fitness(self.population)
            max_fitness_value = max(list(population_evaluation.values()))
            logger.info(f'Iteration {_}, min fitness func value {min(list(population_evaluation.values()))},'
                        f'max fitness func value {max_fitness_value}')

            # choose N best members
            n_best_members = math.floor(self.parameters.n_best_share * self.parameters.population_size)
            best_population = list(population_evaluation.keys())[:n_best_members]

            # define left members and choose random N members
            left_members = list(self.members_set - set(best_population))
            random_members_number = int(self.parameters.population_size *
                                        self.parameters.n_rand_share *
                                        (1 - self.parameters.n_best_share))
            randomly_chosen_members = random.sample(left_members, random_members_number)

            # create new population from best and randomly chosen members
            new_population = np.vstack([self.population[best_population, ],
                                        self.population[randomly_chosen_members]])

            # apply crossover
            children = self._multi_crossover(max_fitness_value)
            if children is not None:
                new_population = np.vstack((new_population, children))

            # fill the population to its size with the best ones from the left members
            left_members = sorted(left_members, key=lambda x: population_evaluation[x])
            left_members_to_add = left_members[:self.parameters.population_size - new_population.shape[0]]
            new_population = np.vstack((new_population, self.population[left_members_to_add, ]))

            # mutation in new population
            members_to_mutate = random.sample(self.members_list, int(round(self.parameters.population_size *
                                                                           self.parameters.mutation_rate)))
            for member in members_to_mutate:
                citizen_to_mutate = new_population[member, ]
                new_population[member, ] = self._mutate(citizen_to_mutate)

            self.population = new_population

            evaluated_population = self._evaluate_fitness(self.population)
            best_result = np.min(np.array(list(evaluated_population.values())))
            if self._define_stop_criteria(best_result):
                print(f'Loop was broken on {_} iteration')
                break

        # return results
        evaluated_final_population = self._evaluate_fitness(self.population)
        return self.population[0, ], list(evaluated_final_population.values())[0]

    @staticmethod
    def _default_fitness(y_pred):
        return 0

    def _define_stop_criteria(self, current_best_result: float) -> bool:
        """
        Defines heuristic criteria of loop stop
        :return: bool
        """
        if current_best_result < np.max(self.current_best_results):
            self.current_best_results[np.argmax(self.current_best_results)] = current_best_result

        if self.current_best_results.sum() == np.inf:
            return False

        return np.std(self.current_best_results) < self.parameters.stop_decrease_ratio *\
               np.mean(self.current_best_results)

    def _multi_crossover(self, max_fitness_value):
        """
        Applying single crossover on randomly chosen members from the population
        :param max_fitness_value: the previously calculated maximum of the fitness function
        :return:
        """
        children: Union[NDArray[NDArray], None] = None
        for _ in range(int((1 - self.parameters.n_best_share - self.parameters.n_rand_share)
                           * self.parameters.population_size)):
            # choose parents
            parents = random.sample(self.members_list, 2)

            # make child
            child = self._single_crossover(self.population[parents[0], ], self.population[parents[1], ])

            # check child
            if self.fitness(child, self.distances) < max_fitness_value:
                if children is None:
                    children = child.copy()
                else:
                    children = np.vstack((children, child))
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
        self._population = np.array(initial_population)


    def _single_crossover(self, chromosome_1: NDArray, chromosome_2: NDArray,
                          start_position=-1, length=-1) -> NDArray:
        """
        Crossover for two chosen members
        :param chromosome_1, chromosome_2: the chosen citizens from the population
        :param start_position: start position of gens' exchange. Using for tests.
        :param length: the length of the gens' interchange. Using for tests.
        :return: new chromosome
        """
        exchange_start_position: int = random.randint(0, self.n_cities - 1) if start_position == -1\
            else start_position
        exchange_length: int = random.randint(1, self.n_cities - exchange_start_position) if length == -1\
            else length
        new_chromosome: NDArray = np.hstack((chromosome_1[0:exchange_start_position, ],
                                             chromosome_2[exchange_start_position: exchange_start_position
                                                          + exchange_length, ],
                                             chromosome_1[exchange_start_position + exchange_length:, ]))
        not_included_gens = [gen for gen in chromosome_1 if gen not in new_chromosome]
        duplicated = list(dict(filter(lambda x: x[1] > 1, Counter(new_chromosome).items())).keys())
        for i in range(self.n_cities - 1, 0, -1):
            if new_chromosome[i] in duplicated:
                duplicated.pop(duplicated.index(new_chromosome[i]))
                new_chromosome[i] = random.choice(not_included_gens)
                not_included_gens.pop(not_included_gens.index(new_chromosome[i]))
                if len(not_included_gens) == 0:
                    break

        return new_chromosome

    def _mutate(self, chromosome):
        """
        Mutate the chromosome
        :param chromosome: a chromosome to mutate
        :return: mutated chromosome
        """
        gens_to_mutate = random.sample(list(range(self.n_cities)),
                                       k=math.ceil(self.parameters.mutation_rate * self.n_cities))
        removed_values = [chromosome[n] for n in gens_to_mutate]
        random.shuffle(removed_values)
        for i, n in enumerate(gens_to_mutate):
            chromosome[n] = removed_values[i]

        return chromosome
