import numpy as np
from typing import Callable, NoReturn, Any, List
from abc import ABC, abstractmethod


class AbsNeuron(ABC):

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = value

    @property
    def func(self):
        return self._func

    @property
    def inputs(self) -> int:
        return len(self._weights)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        if len(value) == len(self._weights):
            self._weights = value
        else:
            raise ValueError('Improper length of the value')

    def __init__(self, number_of_inputs=1, func=None) -> NoReturn:
        self._weights: np.array = np.ones(number_of_inputs)
        self._bias: float = 0
        self._func: Callable = func

    def __repr__(self):
        return f"{self.__class__.__name__}: inputs={self.inputs}"

    @abstractmethod
    def feed_forward(self, inputs: np.array) -> Any:
        pass


class InputNeuron(AbsNeuron):

    def __init__(self) -> NoReturn:
        super().__init__(number_of_inputs=1, func=lambda x: x)

    def feed_forward(self, inputs: float) -> float:
        if isinstance(inputs, float) | isinstance(inputs, int):
            return inputs
        else:
            raise ValueError('Improper input for InputNeuron')


class Neuron(AbsNeuron):

    def __init__(self, number_of_inputs=1, func=None) -> NoReturn:
        if not func:
            func = self._sigmoid
        super().__init__(number_of_inputs=number_of_inputs, func=func)

    @classmethod
    def _sigmoid(cls, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def feed_forward(self, inputs: np.array) -> float:
        total: float = np.dot(self.weights, inputs) + self.bias
        return self._func(total)