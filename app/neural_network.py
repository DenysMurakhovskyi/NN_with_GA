import numpy as np
from typing import NoReturn

from .neuron import Neuron, InputNeuron


class Layer:

    @property
    def neurons_type(self):
        if self.number_of_neurons > 0:
            return self._neurons[0].__class__.__name__
        else:
            return "Not defined"

    @property
    def number_of_neurons(self):
        if self._neurons:
            return len(self._neurons)
        else:
            return 0

    def __init__(self, number_of_neurons=1, number_of_inputs=1, input_layer=False):
        self._neurons = self._define_layer(number_of_neurons=number_of_neurons,
                                           number_of_inputs=number_of_inputs,
                                           input_layer=input_layer)

    def __repr__(self):
        return f"Layer. Consists of {self.number_of_neurons}. Neurons type: {self.neurons_type}"

    @classmethod
    def _define_layer(cls, number_of_neurons=1, number_of_inputs=1, input_layer=False):
        if input_layer:
            return [InputNeuron(weights=np.ones(number_of_inputs), bias=0)] * number_of_neurons
        else:
            return [Neuron(weights=np.ones(number_of_inputs), bias=0)] * number_of_neurons


class NeuralNetwork:

    @property
    def weights(self):
        pass

    def __init__(self, layers_config: np.array, variables_number: int = 1) -> NoReturn:
        self._layers = [self._define_layer(number_of_neurons=variables_number, input_layer=True)]
        for num in layers_config:
            self._layers.append(self._define_layer(number_of_neurons=num, input_layer=False))
        self._layers.append([Neuron])



