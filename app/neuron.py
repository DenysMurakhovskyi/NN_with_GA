import numpy as np
from typing import Callable, NoReturn, Any
from abc import ABC, abstractmethod


EXP_COEF = 4
BIAS_MULTIPLIER = 3


class AbsNeuron(ABC):
    """
    The abstract method for the neuron
    """

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: inputs={self.inputs}"

    @abstractmethod
    def feed_forward(self, inputs: np.array) -> Any:
        pass


class InputNeuron(AbsNeuron):

    """
    The neuron of the input layer
    """

    def __init__(self) -> NoReturn:
        super().__init__(number_of_inputs=1, func=lambda x: x)

    def feed_forward(self, inputs: Any) -> float:
        try:
            inputs = float(inputs)
            return inputs
        except TypeError | ValueError:
            raise ValueError('Improper input for InputNeuron')


class Neuron(AbsNeuron):

    """
    The neuron of hidden layers and the output layer
    """

    def __init__(self, number_of_inputs=1, func=None) -> NoReturn:
        if not func:
            func = self._sigmoid
        super().__init__(number_of_inputs=number_of_inputs, func=func)

    @classmethod
    def _sigmoid(cls, x: float) -> float:
        """
        The activation function. The sigmoid is used
        :param x: the argument value
        :return: the calculated value
        """
        result: float = 2 / (1 + np.exp(-4 * x)) - 1
        return result

    def feed_forward(self, inputs: np.array) -> float:
        """
        The feed forward function
        :param inputs: the arguments
        :return: the function value
        """
        total: float = np.dot(self.weights, inputs)
        return self._func(total)