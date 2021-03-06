from matrix import Matrix
import math
from random import random


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dsigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes=False, output_nodes=False):
        if not hidden_nodes:
            a = input_nodes
            self.input_nodes = a.input_nodes
            self.hidden_nodes = a.hidden_nodes
            self.output_nodes = a.output_nodes
            self.weights_ih = a.weights_ih.copy()
            self.weights_ho = a.weights_ho.copy()
            self.weights_ho_t = a.weights_ho_t.copy()
            self.bias_h = a.bias_h.copy()
            self.bias_o = a.bias_o.copy()
        else:
            self.input_nodes = input_nodes
            self.hidden_nodes = hidden_nodes
            self.output_nodes = output_nodes
            self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
            self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
            self.weights_ho_t = Matrix.transpose(self.weights_ho)
            self.bias_h = Matrix(self.hidden_nodes, 1)
            self.bias_o = Matrix(self.output_nodes, 1)
        self.weights_ih.randomize()
        self.weights_ho.randomize()
        self.bias_h.randomize()
        self.bias_o.randomize()
        self.learning_rate = 0.1

    def predict(self, input_array):
        # Computing Hidden Outputs
        inputs = Matrix.fromArray(input_array)
        hidden = Matrix.multiply1(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        hidden.map1(sigmoid)  # Activation Function

        # Computing Output Layer's Output!
        outputs = Matrix.multiply1(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map1(sigmoid)

        return outputs.toArray()

    def train(self, input_array, target_array):
        # Computing Hidden Outputs
        inputs = Matrix.fromArray(input_array)
        hidden = Matrix.multiply1(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        hidden.map1(sigmoid)  # Activation function

        # Computing Output Layer's Output!
        outputs = Matrix.multiply1(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map1(sigmoid)  # Neural Net's Guess

        # Converting target array to matrix object
        targets = Matrix.fromArray(target_array)

        # Calculate Error
        output_errors = Matrix.subtract(targets, outputs)

        # Calculate Hidden Errors
        hidden_errors = Matrix.multiply1(self.weights_ho_t, output_errors)

        # Calculate gradients
        gradients = Matrix.map2(outputs, dsigmoid)
        gradients.multiply2(output_errors)
        gradients.multiply2(self.learning_rate)

        # Calculate Deltas
        hidden_t = Matrix.transpose(hidden)
        weight_ho_deltas = Matrix.multiply1(gradients, hidden_t)

        # Adjust Hidden -> Output weights and output layer's biases
        self.weights_ho.add(weight_ho_deltas)
        self.bias_o.add(gradients)

        # Calculate the hidden gradients
        hidden_gradients = Matrix.map2(hidden, dsigmoid)
        hidden_gradients.multiply2(hidden_errors)
        hidden_gradients.multiply2(self.learning_rate)

        # Calculate Deltas
        input_t = Matrix.transpose(inputs)
        weight_ih_deltas = Matrix.multiply1(hidden_gradients, input_t)

        # Adjust Input -> Hidden weights and hidden biases
        self.weights_ih.add(weight_ih_deltas)
        self.bias_h.add(hidden_gradients)

    def mutate(self, rate):
        def mutate(x): return x + randomGaussian(0, 0.1) if random() < rate else x
        self.weights_ih.map1(mutate)
        self.weights_ho.map1(mutate)
        self.bias_h.map1(mutate)
        self.bias_o.map1(mutate)

    def copy(self):
        return NeuralNetwork(self)


# Random Gaussian
y2 = None
previous = False


def randomGaussian(mean=0, sd=1):
    y1, x1, x2, w = None, None, None, None
    global previous, y2
    if previous:
        y1 = y2
        previous = False
    else:
        while True:
            x1 = random() * 2 - 1
            x2 = random() * 2 - 1
            w = x1 * x1 + x2 * x2
            if w < 1:
                break
        w = math.sqrt((-2 * math.log(w)) / w)
        y1 = x1 * w
        y2 = x2 * w
        previous = True

    return y1 * sd + mean
