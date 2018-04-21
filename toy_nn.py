from sklearn.neural_network import MLPRegressor
import numpy as np
import random


class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs, coefs=None):
        self._mlp = MLPRegressor(hidden_layer_sizes=(hidden,)).fit(
            [[0 for _ in range(inputs)]], [0 for _ in range(outputs)]
        )
        if coefs is not None:
            self._mlp.coefs_ = coefs

    def copy(self):
        return NeuralNetwork(
            self.inputs, self.hidden, self.outputs, self._mlp.coefs_
        )

    def predict(self, inputs):
        return self._mlp.predict([inputs])

    def mutate(self, rate):
        for matrix in self._mlp.coefs_:
            for line in matrix:
                for index, value in enumerate(line[:]):
                    if random.random() > rate:
                        continue
                    line[index] = random.random() * 2 - 1

    def crossover(self, network):
        coefs = []
        for m_index, matrix in enumerate(self._mlp.coefs_):
            new_matrix = []

            for l_index, line in enumerate(matrix):
                new_line = []
                new_matrix.append(new_line)

                for v_index, value in enumerate(line):
                    new_line.append(
                        value if random.random() < .5 else
                        network.coefs_[m_index][l_index][v_index]
                    )

            coefs.append(np.array(new_matrix))

        return NeuralNetwork(self.inputs, self.hidden, self.outputs, coefs)
