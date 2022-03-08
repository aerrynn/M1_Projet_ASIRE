import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x):                                           # Sigmoide derivative
    return sigmoid(x) * (1.0 - sigmoid(x))


def to_one_hot(y, k):
    one_hot = np.zeros(k)
    one_hot[y] = 1
    return one_hot


class Layer:
    '''
    Une seule couche de neurones.
    '''

    def __init__(self, size, input_size):
        self.size = size
        self.input_size = input_size
        self.weights = np.random.randn(size, input_size)    #
        self.biases = np.random.randn(size)                 # Neuron Bias

    def forward(self, data):
        aggregation = self.aggregation(data)
        activation = self.activation(aggregation)
        return activation

    def aggregation(self, data):
        return np.dot(self.weights, data) + self.biases

    def activation(self, x):
        return sigmoid(x)

    def activation_prime(self, x):
        return sigmoid_prime(x)

    # gradient update
    def update_weights(self, gradient, learning_rate):
        self.weights -= learning_rate * gradient

    # Bias update
    def update_biases(self, gradient, learning_rate):
        self.biases -= learning_rate * gradient


class NeuralNetwork:
    '''
    Actual set of neurons
    Constructor
        :param input_dim: input dimension
        :param nb_layers: amount of layers of the network (1 = perceptron)
        :param size_layer: number of neurons for each layer
    '''

    def __init__(self, input_dim, nb_layers=1, size_layer=8):
        self.input_dim = input_dim
        self.layers = []
        for _ in range(nb_layers):
            self.add_layer(size_layer)

    def add_layer(self, size):
        '''
        add_layer
            :param size: amount of neurons in the new layer
        '''
        if len(self.layers) > 0:
            input_dim = self.layers[-1].size
        else:
            input_dim = self.input_dim

        self.layers.append(Layer(size, input_dim))

    def feedforward(self, input_data):
        '''
        feedforward: Layer to layer propagation
            :@input input_data: data detected by the sensors
        '''
        activation = input_data
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    # Class selected by the NN
    def to_output(self, input_data):
        nn = self.feedforward(input_data)
        m = len(input_data)//2
        return np.sum(nn[:m]), np.sum(nn[m:])

    # Fonction d'entraînement du modèle.
    # Comme décrit dans le billet, nous allons faire tourner la
    # rétropropagation sur un certain nombre d'exemples (batch_size) avant
    # de calculer un gradient moyen, et de mettre à jour les poids.

    def train(self, X, Y, steps=30, learning_rate=0.3, batch_size=10):
        n = Y.size
        for i in range(steps):
            X, Y = shuffle(X, Y)
            for batch_start in range(0, n, batch_size):
                X_batch, Y_batch = X[batch_start:batch_start +
                                     batch_size], Y[batch_start:batch_start + batch_size]
                self.train_batch(X_batch, Y_batch, learning_rate)

    # Cette fonction combine les algos du retropropagation du gradient +
    # gradient descendant.
    def train_batch(self, X, Y, learning_rate):
        # Initialise les gradients pour les poids et les biais.
        weight_gradient = [np.zeros(layer.weights.shape)
                           for layer in self.layers]
        bias_gradient = [np.zeros(layer.biases.shape) for layer in self.layers]

        # On fait tourner l'algo de rétropropagation pour calculer les
        # gradients un certain nombre de fois. On fera la moyenne ensuite.
        for (x, y) in zip(X, Y):
            new_weight_gradient, new_bias_gradient = self.backprop(x, y)
            weight_gradient = [wg + nwg for wg,
                               nwg in zip(weight_gradient, new_weight_gradient)]
            bias_gradient = [bg + nbg for bg,
                             nbg in zip(bias_gradient, new_bias_gradient)]

        # C'est ici qu'on calcule les moyennes des gradients calculés
        avg_weight_gradient = [wg / Y.size for wg in weight_gradient]
        avg_bias_gradient = [bg / Y.size for bg in bias_gradient]

        # Il ne reste plus qu'à mettre à jour les poids et biais en
        # utilisant l'algo du gradient descendant.
        for layer, weight_gradient, bias_gradient in zip(self.layers,
                                                         avg_weight_gradient,
                                                         avg_bias_gradient):
            layer.update_weights(weight_gradient, learning_rate)
            layer.update_biases(bias_gradient, learning_rate)

    def backprop(self, x, y):
        aggregations = []
        activation = x
        activations = [activation]
        for layer in self.layers:
            aggregation = layer.aggregation(activation)
            aggregations.append(aggregation)
            activation = layer.activation(aggregation)
            activations.append(activation)
        target = y
        delta = self.get_output_delta(aggregation, activation, target)
        deltas = [delta]
        nb_layers = len(self.layers)
        for l in reversed(range(nb_layers - 1)):
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            activation_prime = layer.activation_prime(aggregations[l])
            delta = activation_prime * \
                np.dot(next_layer.weights.transpose(), delta)
            deltas.append(delta)
        deltas = list(reversed(deltas))
        weight_gradient = []
        bias_gradient = []
        for l in range(len(self.layers)):
            prev_activation = activations[l]
            weight_gradient.append(np.outer(deltas[l], prev_activation))
            bias_gradient.append(deltas[l])

        return weight_gradient, bias_gradient

    def get_output_delta(self, z, a, target):
        return a - target
