from math import exp
from random import random
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, n_inputs, n_outputs, l_rate, epochs,n_hidden):
        self.n_epoch = epochs
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.l_rate = l_rate
        self.network = self.initialize_network()

    # Initialize a network
    def initialize_network(self):
        network = list()
        hidden_layer = [{'weights': [random() for i in range(self.n_inputs + 1)]} for i in range(self.n_hidden)]
        network.append(hidden_layer)
        output_layer = [{'weights': [random() for i in range(self.n_hidden + 1)]} for i in range(self.n_outputs)]
        network.append(output_layer)
        return network

    # Calculate neuron activation for an input
    @staticmethod
    def activate(weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    @staticmethod
    def transfer(activation):
        return 1.0 / (1.0 + exp(-activation / 10))

    # Forward propagate input to a network output
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = NeuralNetwork.activate(neuron['weights'], inputs)
                neuron['output'] = NeuralNetwork.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Calculate the derivative of an neuron output
    @staticmethod
    def transfer_derivative(output):
        return output * (1.0 - output)

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * NeuralNetwork.transfer_derivative(neuron['output'])

    # Update network weights with error
    def update_weights(self, row):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.l_rate * neuron['delta']

    # Train a network for a fixed number of epochs
    def train_network(self, train):
        error_list = []
        for epoch in range(self.n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(row)
                expected = row
                # expected[row[-1]] = 1
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row)
            print('>epoch=%d, l_rate=%.3f, error=%.3f' % (epoch, self.l_rate, sum_error))
            error_list.append(sum_error)

        return error_list

    # Make a prediction with a network
    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs

    def test_predict(self, dataset):
        predict_list = []
        error_list = []
        for row in dataset:
            outputs = self.predict(row)
            delta_error = NeuralNetwork.delta_error(row, outputs)
            error_list.append(delta_error)
            predict_list.append(outputs)
        return predict_list, error_list

    @staticmethod
    def delta_error(row, expected):
        sum_error = 0
        for i in range(len(row)):
            sum_error += row[i] - expected[i]
        return sum_error

    @staticmethod
    def draw_error_graph(error_list, y_label, x_label, graph_name):
        # x axis values
        x = list(range(1, len(error_list)+1))
        # corresponding y axis values
        y = error_list

        # plotting the points
        plt.plot(x, y)

        # naming the x axis
        plt.xlabel(x_label)
        # naming the y axis
        plt.ylabel(y_label)

        # giving a title to my graph
        plt.title(graph_name)

        # function to show the plot
        plt.show()


