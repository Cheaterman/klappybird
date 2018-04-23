from sklearn.neural_network import MLPRegressor
import json
import numpy as np
import random


class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs, weights=None, bias=None):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self._mlp = MLPRegressor(
            activation='logistic',
            solver='lbfgs',
            hidden_layer_sizes=hidden,
        ).fit(
            [[0 for _ in range(inputs)]],
            [0] if outputs == 1 else [[0 for _ in range(outputs)]]
        )
        if weights is not None:
            self._mlp.coefs_ = weights
        if bias is not None:
            self._mlp.intercepts_ = bias

    def copy(self):
        return NeuralNetwork(
            self.inputs,
            self.hidden,
            self.outputs,
            [matrix.copy() for matrix in self._mlp.coefs_],
            [matrix.copy() for matrix in self._mlp.intercepts_],
        )

    def predict(self, inputs):
        return self._mlp.predict([inputs])

    def mutate(self, rate):
        for matrix in self._mlp.coefs_:
            for line in matrix:
                for index, value in enumerate(line[:]):
                    line[index] = self.do_mutate(line[index], rate)

        for line in self._mlp.intercepts_:
            for index, value in enumerate(line[:]):
                line[index] = self.do_mutate(line[index], rate)

    def do_mutate(self, value, rate):
        if random.random() > rate:
            return value
        else:
            return value + random.gauss(0, .05)

    def crossover(self, network):
        weights = []
        for m_index, matrix in enumerate(self._mlp.coefs_):
            new_matrix = []

            for l_index, line in enumerate(matrix):
                new_line = []
                new_matrix.append(new_line)

                for v_index, value in enumerate(line):
                    new_line.append(self.do_crossover(
                        value,
                        network._mlp.coefs_[m_index][l_index][v_index]
                    ))

            weights.append(np.array(new_matrix))

        bias = []
        for l_index, line in enumerate(self._mlp.intercepts_):
            new_line = []

            for v_index, value in enumerate(line):
                new_line.append(self.do_crossover(
                    value,
                    network._mlp.intercepts_[l_index][v_index]
                ))

            bias.append(np.array(new_line))

        return NeuralNetwork(self.inputs, self.hidden, self.outputs, weights, bias)

    def do_crossover(self, value_a, value_b):
        return value_a if random.random() < .5 else value_b

    def serialize(self):
        return json.dumps(dict(
            inputs=self.inputs,
            hidden=self.hidden,
            outputs=self.outputs,
            weights=[
                matrix.tolist()
                for matrix in self._mlp.coefs_
            ],
            bias=[
                line.tolist()
                for line in self._mlp.intercepts_
            ],
        ))

    @staticmethod
    def deserialize(json):
        return NeuralNetwork(
            json['inputs'],
            json['hidden'],
            json['outputs'],
            [np.array(matrix) for matrix in json['weights']],
            [np.array(matrix) for matrix in json['bias']],
        )
