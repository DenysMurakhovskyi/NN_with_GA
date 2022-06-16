from unittest import TestCase

from app.neural_network import Layer


class TestLayer(TestCase):

    def test_layer_creation(self):
        layer = Layer(number_of_neurons=2, number_of_inputs=2, input_layer=False)
        pass

        input_layer = Layer(number_of_neurons=2, input_layer=True)
        pass