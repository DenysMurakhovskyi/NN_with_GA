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

    def __iter__(self):
        for neuron in self._neurons:
            yield neuron

    @staticmethod
    def create_layer(number_of_neurons=2,
                     number_of_inputs=2,
                     input_layer=False) -> "Layer":
        if input_layer:
            number_of_inputs = 1
        return Layer(number_of_neurons=number_of_neurons,
                     number_of_inputs=number_of_inputs,
                     input_layer=input_layer)

    def feed_forward(self, inputs: np.array) -> np.array:
        return np.array([neuron.feed_forward(inputs) for neuron in self._neurons])

    @classmethod
    def _define_layer(cls, number_of_neurons=1, number_of_inputs=1, input_layer=False):
        if input_layer:
            return [InputNeuron()] * number_of_neurons
        else:
            return [Neuron(number_of_inputs=number_of_inputs)] * number_of_neurons


class NeuralNetwork:

    @property
    def biases(self):
        result = []
        for layer in self._layers:
            if layer.neurons_type == 'InputNeuron':
                continue
            for neuron in layer:
                result.append(neuron.bias)
        return result

    @property
    def layers(self):
        return self._layers

    @property
    def num_of_layers(self):
        return len(self._layers)

    @property
    def weights(self):
        result = []
        for layer in self._layers:
            if layer.neurons_type == 'InputNeuron':
                continue
            for neuron in layer:
                result.extend(neuron.weights)
        return result

    def __init__(self, layers_config: np.array, variables_number: int = 1) -> NoReturn:
        self._layers = [Layer.create_layer(number_of_neurons=variables_number,
                                           input_layer=True),
                        Layer.create_layer(number_of_neurons=layers_config[0],
                                           number_of_inputs=variables_number)]

        for number_of_inputs, number_of_neurons in zip(layers_config[:-1], layers_config[1:]):
            self._layers.append(Layer.create_layer(number_of_neurons=number_of_neurons,
                                                   number_of_inputs=number_of_inputs))

        self._layers.append(Layer.create_layer(number_of_neurons=1, number_of_inputs=layers_config[-1]))



