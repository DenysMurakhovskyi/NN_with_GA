from random import shuffle
from typing import NoReturn, List, Union, Tuple

import numpy as np
from numpy.typing import NDArray

from app.ga import GeneticAlgorithm
from .neuron import Neuron, InputNeuron, BIAS_MULTIPLIER


class Layer:

    @property
    def neurons(self) -> Union[List[Neuron], List[InputNeuron]]:
        return self._neurons

    @property
    def neurons_type(self) -> str:
        if self.number_of_neurons > 0:
            return self._neurons[0].__class__.__name__
        else:
            return "Not defined"

    @property
    def number_of_neurons(self) -> int:
        if self._neurons:
            return len(self._neurons)
        else:
            return 0

    def __init__(self, number_of_neurons=1, number_of_inputs=1, input_layer=False) -> None:
        self._neurons = self._define_layer(number_of_neurons=number_of_neurons,
                                           number_of_inputs=number_of_inputs,
                                           input_layer=input_layer)

    def __repr__(self) -> str:
        return f"Layer. Consists of {self.number_of_neurons}. Neurons type: {self.neurons_type}"

    def __iter__(self) -> Union[Neuron, InputNeuron]:
        for neuron in self._neurons:
            yield neuron

    @staticmethod
    def create_layer(number_of_neurons=2,
                     number_of_inputs=2,
                     input_layer=False) -> "Layer":
        if input_layer:
            number_of_inputs: int = 1
        return Layer(number_of_neurons=number_of_neurons,
                     number_of_inputs=number_of_inputs,
                     input_layer=input_layer)

    def feed_forward(self, inputs: NDArray) -> NDArray:
        if self.neurons_type == 'InputNeuron':
            return np.array([neuron.feed_forward(input_value) for neuron, input_value in zip(self._neurons, inputs)])
        else:
            return np.array([neuron.feed_forward(inputs) for neuron in self._neurons])

    @classmethod
    def _define_layer(cls, number_of_neurons=1, number_of_inputs=1, input_layer=False) -> Union[List[Neuron],
                                                                                                List[InputNeuron]]:
        if input_layer:
            return [InputNeuron() for _ in range(number_of_neurons)]
        else:
            return [Neuron(number_of_inputs=number_of_inputs) for _ in range(number_of_neurons)]


class NeuralNetwork:

    @property
    def biases(self) -> List[float]:
        """
        Property returns the list of biases in the NN's instance
        """
        result = []
        for layer in self._layers:
            if layer.neurons_type == 'InputNeuron':
                continue
            for neuron in layer:
                result.append(neuron.bias)
        return result

    @biases.setter
    def biases(self, value) -> NoReturn:
        counter = 0
        if len(value) != len(self.biases):
            raise ValueError('Improper len of values array')
        for layer in self.layers:
            if layer.neurons_type == 'InputNeuron':
                continue
            for neuron in layer.neurons:
                neuron.bias = value[counter]
                counter += 1

    @property
    def layers(self) -> List[Layer]:
        """
        Property returns the list of layers in the NN's instance
        """
        return self._layers

    @property
    def len_biases(self) -> int:
        """
        Property returns the length of biases list
        """
        result = 0
        for layer in self._layers:
            if layer.neurons_type == 'InputNeuron':
                continue
            result += layer.number_of_neurons
        return result

    @property
    def len_weights(self) -> int:
        """
        Property returns the length of weights list
        """
        result = 0
        for layer in self._layers:
            if layer.neurons_type == 'InputNeuron':
                continue
            for neuron in layer:
                result += len(neuron.weights)
        return result

    @property
    def num_of_layers(self) -> int:
        """
        Property returns the number of layers in the NN
        """
        return len(self._layers)

    @property
    def r2_(self) -> float:
        """
        Property returns the R^2 score
        """
        if len(self._y_test) > 0:
            y_pred = self._calculate_vectorized(self._X_test)
            return self._r2_score(y_pred, self._y_test)
        else:
            return -np.inf

    @property
    def variables_number(self) -> int:
        """
        Property returns the number of the input variables
        """
        return self._variables_number

    @property
    def weights(self) -> List[float]:
        """
        Property returns the list of weights in the NN
        """
        result = []
        for layer in self._layers:
            if layer.neurons_type == 'InputNeuron':
                continue
            for neuron in layer:
                result.extend(neuron.weights)
        return result

    @weights.setter
    def weights(self, value) -> NoReturn:
        counter = 0
        if len(value) != self.len_weights:
            raise ValueError('Improper len of values array')
        for layer in self.layers:
            if layer.neurons_type == 'InputNeuron':
                continue
            for neuron in layer.neurons:
                neuron.weights = value[counter:counter + len(neuron.weights)]
                counter += len(neuron.weights)

    def __init__(self, layers_config: np.array, variables_number: int = 1) -> NoReturn:
        self._variables_number = variables_number
        self._create_layers(layers_config, variables_number)
        self._X_train, self._y_train, self._X_test, self._y_test, self.scale = [], [], [], [], 1
        self.optimized_values = None

    def _create_layers(self, layers_config: NDArray, variables_number: int) -> NoReturn:
        """
        The method creates the NN's lauers due to the given template
        :param layers_config: the NDArray which represents the number of neurons in the layers
        :param variables_number: the number of input's variables
        :return: None
        """
        self._layers: List = [Layer.create_layer(number_of_neurons=variables_number,
                                                 input_layer=True),
                              Layer.create_layer(number_of_neurons=layers_config[0],
                                                 number_of_inputs=variables_number)]
        for number_of_inputs, number_of_neurons in zip(layers_config[:-1], layers_config[1:]):
            self._layers.append(Layer.create_layer(number_of_neurons=number_of_neurons,
                                                   number_of_inputs=number_of_inputs))
        self._layers.append(Layer.create_layer(number_of_neurons=1, number_of_inputs=layers_config[-1]))

    def fit(self, X: List[np.array], y: List, verbose=False) -> NoReturn:
        """
        The fit method. Trains the NN on given data
        :param X: The list of arguments values grouped in tuples
        :param y: The list of true function's values
        :param verbose: if True the training retails will be printed in StdOut
        :return: None
        """
        self._prepare_fit(X, y)
        ga = GeneticAlgorithm(len_of_citizen=self.len_weights + self.len_biases,
                              fitness_func=self.calculate_rmse,
                              values_range=np.array([-1, 1]),
                              verbose=verbose)
        self.optimized_values = ga.solve()
        self._set_values(self.optimized_values)
        print(f'R^2 score={self.r2_}')

    def _prepare_fit(self, X, y) -> None:
        """
        Prepares data for training loop
        :param X: The list of arguments values grouped in tuples
        :param y: The list of true function's values
        :return: None
        """
        if len(X) != len(y):
            raise ValueError('Different length of arguments and func value')
        self._X_train, self._X_test, self._y_train, self._y_test = self._train_test_split(X, y)
        self.scale = np.abs(max(y) - min(y)) / 2
        self.shift = np.abs(max(y) + min(y)) / 2

    def _set_values(self, values) -> None:
        """
        Set values given as a list of values w/o split into the weights and biases
        Used for training purposes
        :param values: list of the values
        :return: None
        """
        if len(values) != self.len_biases + self.len_weights:
            raise ValueError('Improper values length')
        self.weights = np.array(values[:self.len_weights])
        self.biases = BIAS_MULTIPLIER * np.array(values[self.len_weights:])

    @staticmethod
    def _train_test_split(X: List[List], y: List) -> Tuple[List[List], List[List], List, List]:
        """
        Splits the training data into the training and test sets
        :param X: The list of arguments values grouped in tuples
        :param y: The list of true function's values
        :return: training and test sets of the data (split data)
        """
        values_numbers = list(range(len(X)))
        shuffle(values_numbers)
        train_values = values_numbers[:int(len(X) * 0.8)]
        test_values = [value for value in values_numbers if value not in train_values]
        X_train = np.take(X, train_values, axis=0)
        y_train = np.take(y, train_values, axis=0)
        X_test = np.take(X, test_values, axis=0)
        y_test = np.take(y, test_values, axis=0)
        return X_train, X_test, y_train, y_test

    def _calculate_vectorized(self, inputs_array: Union[List[Tuple], List]) -> List[float]:
        """
        Calculates the function value on the list of input data tuples
        :param inputs_array:
        :return:
        """
        result = [self._calculate(inputs) for inputs in inputs_array]
        return result

    @staticmethod
    def _r2_score(y_pred: List[float], y_true: List[float]) -> float:
        """
        Calculates R^2 score
        :param y_pred: predicted values
        :param y_true: tru values
        :return: R^2 score value
        """
        corr_matrix: NDArray = np.corrcoef(y_pred, y_true)
        corr: float = corr_matrix[0, 1]
        return corr ** 2

    @staticmethod
    def _rmse(y_pred, y_true):
        """
        Calculates RMSE (Root Mean Square Error)
        :param y_pred: predicted values
        :param y_true: tru values
        :return:the calculated RMSE value
        """
        return np.sqrt(np.mean((np.array(y_pred) - np.array(y_true)) ** 2))

    def calculate_rmse(self, values: List[float]) -> float:
        """
        Calculates RMSE for the given set of values
        :param values: set of the input variables values
        :return: the RMSE value
        """
        self._set_values(values)
        y_pred = self._calculate_vectorized(self._X_train)
        return self._rmse(y_pred, self._y_train)

    def _calculate(self, inputs: np.array) -> float:
        """
        Calculates the function value.
        :param inputs: The values of the inputs variables
        :return: the function value
        """
        if len(inputs) != self.variables_number:
            raise ValueError('Improper number of variables')
        result: Union[List[float], float] = self._layers[0].feed_forward(inputs)
        for layer in self._layers[1:]:
            result = layer.feed_forward(result)
        return self.scale * result[0] + self.shift
