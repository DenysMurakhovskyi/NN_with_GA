import numpy as np
from typing import Callable, NoReturn, Any, List
from abc import ABC, abstractmethod


class AbsNeuron(ABC):

    @property
    def inputs(self) -> int:
        return len(self.weights)

    def __init__(self, weights, bias, func=None) -> NoReturn:
        self.weights: np.array = weights
        self.bias: float = bias
        self.func: Callable = func

    @abstractmethod
    def feedforward(self, inputs: np.array) -> Any:
        pass


class InputNeuron(AbsNeuron):

    def __init__(self, weights=1, bias=0, func=lambda x: x) -> NoReturn:
        super().__init__(weights, bias, func)

    def feedforward(self, inputs: float) -> float:
        if isinstance(inputs, float) | isinstance(inputs, int):
            return inputs
        else:
            raise ValueError('Improper input for InputNeuron')


class Neuron(AbsNeuron):

    def __init__(self, weights, bias, func=None) -> NoReturn:
        super().__init__(weights, bias, func if func else self._sigmoid)

    @classmethod
    def _sigmoid(cls, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def feedforward(self, inputs: np.array) -> float:
        total: float = np.dot(self.weights, inputs) + self.bias
        return self.func(total)