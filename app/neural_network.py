import numpy as np
from typing import NoReturn, List

from .neuron import Neuron, InputNeuron
from random import shuffle


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
        if self.neurons_type == 'InputNeuron':
            return np.array([neuron.feed_forward(input_value) for neuron, input_value in zip(self._neurons, inputs)])
        else:
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
    def variables_number(self):
        return self._variables_number

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
        self._variables_number = variables_number
        self._create_layers(layers_config, variables_number)
        self._X_train, self._y_train, self._X_test, self._y_test = [], [], [], []
        self._r2 = -np.inf

    def _create_layers(self, layers_config, variables_number):
        self._layers = [Layer.create_layer(number_of_neurons=variables_number,
                                           input_layer=True),
                        Layer.create_layer(number_of_neurons=layers_config[0],
                                           number_of_inputs=variables_number)]
        for number_of_inputs, number_of_neurons in zip(layers_config[:-1], layers_config[1:]):
            self._layers.append(Layer.create_layer(number_of_neurons=number_of_neurons,
                                                   number_of_inputs=number_of_inputs))
        self._layers.append(Layer.create_layer(number_of_neurons=1, number_of_inputs=layers_config[-1]))

    def fit(self, X: List[np.array], y: List, verbose=False) -> NoReturn:
        if len(X) != len(y):
            raise ValueError('Different length of arguments and func value')
        values_numbers = list(range(len(X)))
        shuffle(values_numbers)
        train_values = values_numbers[:int(len(X) * 0.8)]
        test_values = [value for value in values_numbers if value not in train_values]
        self._X_train = np.take(X, train_values, axis=0)
        self._y_train = np.take(y, train_values, axis=0)
        self._X_test = np.take(X, test_values, axis=0)
        self._y_test = np.take(y, test_values, axis=0)
        pass

    def _calculate_r2(self):
        pass

    def predict(self, inputs: np.array) -> float:
        if len(inputs) != self.variables_number:
            raise ValueError('Improper number of variables')
        result = self._layers[0].feed_forward(inputs)
        for layer in self._layers[1]:
            result = layer.feed_forward(result)
        return result[0]



